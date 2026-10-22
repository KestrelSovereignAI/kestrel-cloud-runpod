"""Deterministic doubles for generic catalog Pod capacity tests."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpcore
import httpx
import pytest
from pydantic import SecretStr

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    ComputeProduct,
    GPUProfile,
    PlacementDecision,
    RunPodManagerError,
)
from kestrel_cloud_runpod.pod_capacity_contracts import (
    CatalogAttemptCapability,
    CatalogPodCapacityRequest,
    CatalogPodWorkloadState,
    PodBillingReceipt,
    PodCapacityConstraints,
    PodCapacityQuote,
    PodRealizedPlacement,
    pod_cost_usd,
)
from kestrel_cloud_runpod.pod_capacity_provider import (
    CreatedPodCapacity,
    PodCapacityObservation,
    RunpodPodCapacityProvider,
)
from kestrel_cloud_runpod.pod_capacity_repository import (
    SQLitePodCapacityRepository,
)
from kestrel_cloud_runpod.pod_capacity_service import PodCapacityLeaseService
from kestrel_cloud_runpod.pod_transport import CatalogPodWorkloadObservation
from kestrel_cloud_runpod.providers import DirectRunPodProvider

TOKEN = "catalog-pod-token-" + "x" * 40
REQUEST_SHA = "b" * 64
RESULT_SHA = "d" * 64
RECEIPT_SHA = "e" * 64
ACK_SHA = "f" * 64
PARAMETERS_SHA = "c" * 64
IMAGE = "ghcr.io/kestrel/catalog-worker@sha256:" + "a" * 64


def native_httpx_connect_failure() -> httpx.TransportError:
    """Capture the real HTTPX -> HTTPCore -> socket chain without external I/O."""

    raised: httpx.TransportError | None = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        with httpx.Client(timeout=0.2, trust_env=False) as client:
            try:
                client.get(f"http://127.0.0.1:{port}/")
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raised = exc
    if raised is None:
        raise AssertionError(
            "owned closed loopback port unexpectedly accepted a request"
        )
    public_pair = {
        httpx.ConnectError: httpcore.ConnectError,
        httpx.ConnectTimeout: httpcore.ConnectTimeout,
    }
    if type(raised.__cause__) is not public_pair[type(raised)]:
        raise AssertionError("native HTTPX failure omitted its exact HTTPCore cause")
    if not isinstance(raised.__cause__.__context__, (OSError, TimeoutError)):
        pytest.fail("native HTTPCore failure omitted its socket context")
    return raised


def complete_pod_payload(*, pod_id: str, status: str) -> dict[str, Any]:
    return {
        "id": pod_id,
        "name": "catalog-pod-test",
        "status": status,
        "gpu": {"id": "NVIDIA RTX PRO 4500", "count": 1},
        "cost": 0.4,
        "cloud": "SECURE",
        "dataCenterId": "US-TX-3",
        "cudaVersion": "12.8",
        "image": IMAGE,
        "env": {"CATALOG_WORKER_MODE": "pod"},
        "createdAt": "2026-08-02T12:00:00Z",
    }


@contextmanager
def native_pod_action_harness():
    """Serve deterministic v2 Pod action/readback responses on loopback only."""

    state: dict[str, Any] = {
        "pod_id": "pod-catalog-1",
        "action_status": 204,
        "action_payload": None,
        "readback": "running",
        "calls": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def _json_response(self, status: int, payload: object) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            state["calls"].append((self.command, path, body))
            if path != f"/v2/pods/{state['pod_id']}/action":
                self._json_response(404, {"title": "not found"})
                return
            if state["action_status"] == 204:
                self.send_response(204)
                self.end_headers()
                return
            self._json_response(state["action_status"], state["action_payload"])

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            state["calls"].append((self.command, path, None))
            if path != f"/v2/pods/{state['pod_id']}":
                self._json_response(404, {"title": "not found"})
                return
            outcome = state["readback"]
            if outcome == "running":
                self._json_response(
                    200,
                    complete_pod_payload(pod_id=state["pod_id"], status="RUNNING"),
                )
            elif outcome == "terminal":
                self._json_response(
                    200,
                    complete_pod_payload(pod_id=state["pod_id"], status="TERMINATED"),
                )
            elif outcome == "absent":
                self._json_response(404, {"title": "not found"})
            elif outcome == "foreign":
                self._json_response(
                    200,
                    complete_pod_payload(pod_id="pod-foreign", status="TERMINATED"),
                )
            elif outcome == "authorization":
                self._json_response(401, {"title": "unauthorized"})
            elif outcome == "structural":
                body = b"{"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif outcome == "transport":
                body = b'{"id":'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body) + 64))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
            else:
                raise AssertionError(f"unsupported readback outcome: {outcome}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="pod-action-loopback")
    thread.start()
    client = RunpodControlPlaneClient(
        api_key="local-loopback-only",
        base_url=f"http://127.0.0.1:{server.server_port}/v2",
        max_safe_retries=0,
    )
    direct = DirectRunPodProvider("local-loopback-only", client=client)
    adapter = RunpodPodCapacityProvider(direct)
    try:
        yield state, direct, adapter
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise AssertionError("Pod action loopback server did not terminate")


def acknowledgement() -> dict[str, Any]:
    return {
        "attempt_id": "attempt:catalog-run-0001",
        "request_sha256": REQUEST_SHA,
        "result_sha256": RESULT_SHA,
        "receipt_sha256": RECEIPT_SHA,
        "acknowledgement_sha256": ACK_SHA,
        "opaque_private_signature": "not-public-or-persisted-by-sdk",
    }


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 2, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def constraints(gpu_count: int = 1) -> PodCapacityConstraints:
    return PodCapacityConstraints(
        min_vram_gb=24,
        gpu_count=gpu_count,
        cloud=CloudType.SECURE,
        min_cuda_version="12.8",
        allowed_gpu_ids=("NVIDIA RTX PRO 4500",),
        allowed_data_center_ids=("US-TX-3",),
        max_hourly_rate_usd=Decimal("0.50"),
        benchmark_id="catalog-lora-v1",
        allowed_products=(ComputeProduct.POD,),
    )


def quote(clock: MutableClock, gpu_count: int = 1) -> PodCapacityQuote:
    requirements = constraints(gpu_count).placement_requirements()
    placement = PlacementDecision(
        gpu_id="NVIDIA RTX PRO 4500",
        gpu_pool=None,
        gpu_name="RTX PRO 4500",
        memory_gb=32,
        cloud=CloudType.SECURE,
        gpu_count=gpu_count,
        offered_cost_per_hr=0.4,
        availability=Availability.HIGH,
        catalog_observed_at=clock(),
        requirements=requirements,
    )
    return PodCapacityQuote(
        schema_version=3,
        capability_version="runpod-pod-capacity-v1",
        provider_quote_id="runpod-pod:" + "d" * 64,
        workload_kind="catalog-lora",
        parameters_sha256=PARAMETERS_SHA,
        constraints=constraints(gpu_count),
        gpu_type_id=placement.gpu_id,
        gpu_display_name=placement.gpu_name,
        hourly_cost_usd=Decimal("0.4"),
        # Derived the same way the quote's own check derives them, so the
        # fixture stays consistent for any GPU count.
        estimated_cost_usd=pod_cost_usd(Decimal("0.4"), 360, gpu_count),
        cost_ceiling_usd=pod_cost_usd(Decimal("0.4"), 600, gpu_count),
        estimated_startup_seconds=60,
        estimated_execution_seconds=300,
        maximum_runtime_seconds=600,
        observed_at=clock(),
        expires_at=clock() + timedelta(seconds=60),
        placement=placement,
    )


def request(
    clock: MutableClock, gpu_count: int = 1, **changes: Any
) -> CatalogPodCapacityRequest:
    values: dict[str, Any] = {
        "capacity_id": "capacity:catalog-attempt-0001",
        "cleanup_family_id": "capacity:catalog-attempt-0001",
        "owner_id": "owner:catalog-user-0001",
        "workload_id": "workload:lora-job-0001",
        "attempt_id": "attempt:catalog-run-0001",
        "idempotency_key": "idempotency:lora-run-0001",
        "request_sha256": REQUEST_SHA,
        "workload_kind": "catalog-lora",
        "parameters_sha256": PARAMETERS_SHA,
        "image_reference": IMAGE,
        "profile_id": "catalog-lora",
        "quote": quote(clock, gpu_count),
        # Must equal the quote's own ceiling, which scales with the GPU count.
        "accepted_max_cost_usd": pod_cost_usd(Decimal("0.4"), 600, gpu_count),
        "created_at": clock(),
        "readiness_deadline": clock() + timedelta(seconds=30),
        "hard_deadline": clock() + timedelta(seconds=600),
        "bearer_expires_at": clock() + timedelta(seconds=660),
        "idle_timeout_seconds": 30,
        "cleanup_grace_seconds": 60,
        "attempt_environment": {"MODEL_REPOSITORY": "private/model"},
    }
    values.update(changes)
    return CatalogPodCapacityRequest(**values)


def profile() -> GPUProfile:
    return GPUProfile(
        id="catalog-lora",
        name="Catalog LoRA",
        task_type="catalog-lora",
        image_name=IMAGE,
        container_disk_gb=40,
        volume_gb=0,
        ports=["8080/http"],
        inference_port=8080,
        default_model="flux-lora",
        min_vram_gb=24,
        gpu_count=1,
    )


class FakeCapabilityStore:
    def __init__(self, clock: MutableClock) -> None:
        self.clock = clock
        self.capabilities: dict[str, CatalogAttemptCapability] = {}
        self.revoked: list[str] = []
        self.revoke_error: RunPodManagerError | None = None

    async def load_or_create(
        self, attempt_id: str, expires_at: datetime
    ) -> CatalogAttemptCapability:
        existing = self.capabilities.get(attempt_id)
        if existing is not None:
            return existing
        capability = CatalogAttemptCapability(
            secret_id="secret:catalog-capability-0001",
            bearer_token=SecretStr(TOKEN),
            token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest(),
            expires_at=expires_at,
        )
        self.capabilities[attempt_id] = capability
        return capability

    async def load(self, attempt_id: str) -> CatalogAttemptCapability | None:
        return self.capabilities.get(attempt_id)

    async def revoke(self, attempt_id: str) -> None:
        if self.revoke_error is not None:
            raise self.revoke_error
        self.revoked.append(attempt_id)


class FakeCapacityProvider:
    def __init__(self, clock: MutableClock) -> None:
        self.clock = clock
        self.status_value = "RUNNING"
        self.route = "https://pod-catalog-1-8080.proxy.runpod.net"
        self.create_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[str] = []
        self.find_result: str | None = None
        self.create_error: Exception | None = None
        self.realized: PodRealizedPlacement | None = realized_placement(clock)
        self.billing_receipt: PodBillingReceipt | None = None
        self.billing_calls: list[str] = []
        self.billing_rates: list[Decimal] = []

    async def quote(self, request):
        return quote(self.clock)

    async def observe(self, pod_id: str, *, profile: GPUProfile):
        return PodCapacityObservation(
            provider_pod_id=pod_id,
            status=self.status_value,
            backend_base_url=self.route,
            raw={"id": pod_id, "status": self.status_value},
        )

    async def start(self, pod_id: str, *, gpu_count: int) -> None:
        raise AssertionError("catalog capacity never resumes a shared Pod")

    async def create(
        self,
        *,
        profile: GPUProfile,
        resource_name: str,
        companion_id: str,
        environment: Mapping[str, str] | None = None,
        capacity_spec=None,
    ):
        self.create_calls.append(
            {
                "profile": profile,
                "resource_name": resource_name,
                "companion_id": companion_id,
                "environment": dict(environment or {}),
                "capacity_spec": capacity_spec,
            }
        )
        if self.create_error is not None:
            raise self.create_error
        return CreatedPodCapacity(
            provider_pod_id="pod-catalog-1",
            placement=quote(self.clock).placement,
            realized_placement=self.realized,
            raw={"id": "pod-catalog-1"},
        )

    async def find_by_name(self, resource_name: str) -> str | None:
        raise AssertionError("catalog recovery must validate immutable metadata")

    async def find_exact(
        self, resource_name: str, capacity_spec
    ) -> CreatedPodCapacity | None:
        if self.find_result is None:
            return None
        return CreatedPodCapacity(
            provider_pod_id=self.find_result,
            placement=None,
            realized_placement=realized_placement(
                self.clock, provider_pod_id=self.find_result
            ),
            raw=None,
        )

    async def stop(self, pod_id: str) -> bool:
        raise AssertionError("catalog capacity must be terminated, not stopped")

    async def terminate(self, pod_id: str) -> bool:
        self.terminate_calls.append(pod_id)
        self.status_value = "TERMINATED"
        return True

    async def final_billing(
        self,
        pod_id: str,
        *,
        capacity_spec,
        created_at: datetime,
        terminated_at: datetime,
        realized_hourly_rate_usd: Decimal,
    ) -> PodBillingReceipt | None:
        self.billing_calls.append(pod_id)
        self.billing_rates.append(realized_hourly_rate_usd)
        if self.billing_receipt is None:
            return None
        # Mirror the real provider: the receipt records the REALIZED rate.
        return replace(self.billing_receipt, hourly_price_usd=realized_hourly_rate_usd)


class FakeWorkloadTransport:
    def __init__(self) -> None:
        self.health_calls: list[str] = []
        self.submit_calls: list[Mapping[str, Any]] = []
        self.status_value = CatalogPodWorkloadState.RUNNING
        self.result_payload: Mapping[str, Any] = {
            "protocol_version": 1,
            "receipt": {
                "attempt_id": "attempt:catalog-run-0001",
                "request_sha256": REQUEST_SHA,
                "result_sha256": RESULT_SHA,
                "receipt_sha256": RECEIPT_SHA,
            },
            "result": {
                "dispatch_attempt_id": "attempt:catalog-run-0001",
                "request_sha256": REQUEST_SHA,
                "artifact": {"disposition": "stored"},
            },
        }
        self.cancel_calls = 0
        self.acknowledgement_calls: list[Mapping[str, Any]] = []

    async def health(self, base_url: str) -> None:
        self.health_calls.append(base_url)

    async def submit(self, **values):
        self.submit_calls.append(values["payload"])
        return CatalogPodWorkloadObservation(
            attempt_id=values["attempt_id"],
            request_sha256=values["request_sha256"],
            state=CatalogPodWorkloadState.RUNNING,
            error_type=None,
            result_available=False,
        )

    async def status(self, **values):
        handoff = self.status_value in {
            CatalogPodWorkloadState.AWAITING_ACK,
            CatalogPodWorkloadState.ACKNOWLEDGED,
        }
        return CatalogPodWorkloadObservation(
            attempt_id=values["attempt_id"],
            request_sha256=values["request_sha256"],
            state=self.status_value,
            error_type=(
                "CatalogExecutionError" if self.status_value.value == "failed" else None
            ),
            result_available=handoff
            or self.status_value is CatalogPodWorkloadState.SUCCEEDED,
            result_sha256=RESULT_SHA if handoff else None,
            receipt_sha256=RECEIPT_SHA if handoff else None,
            acknowledgement_sha256=ACK_SHA
            if self.status_value is CatalogPodWorkloadState.ACKNOWLEDGED
            else None,
        )

    async def result(self, **values):
        return dict(self.result_payload)

    async def acknowledge(self, **values):
        self.acknowledgement_calls.append(values["payload"])
        self.status_value = CatalogPodWorkloadState.ACKNOWLEDGED
        return await self.status(**values)

    async def cancel(self, **values):
        self.cancel_calls += 1
        return CatalogPodWorkloadObservation(
            attempt_id=values["attempt_id"],
            request_sha256=values["request_sha256"],
            state=CatalogPodWorkloadState.CANCEL_REQUESTED,
            error_type=None,
            result_available=False,
        )


def service(
    tmp_path: Path,
    clock: MutableClock,
    provider: FakeCapacityProvider,
    store: FakeCapabilityStore,
    transport: FakeWorkloadTransport,
) -> PodCapacityLeaseService:
    return PodCapacityLeaseService(
        repository=SQLitePodCapacityRepository(tmp_path / "capacity.sqlite3"),
        provider=provider,
        profiles={"catalog-lora": profile()},
        poll_interval_seconds=5,
        orphan_timeout_seconds=30,
        capability_store=store,
        workload_transport=transport,
        clock=clock,
        sleep=clock.sleep,
    )


def final_receipt(clock: MutableClock) -> PodBillingReceipt:
    return PodBillingReceipt(
        provider_billing_id="runpod-billing:" + "e" * 64,
        provider_pod_id="pod-catalog-1",
        billed_from=clock() - timedelta(minutes=10),
        billed_until=clock(),
        billed_seconds=600,
        hourly_price_usd=Decimal("0.4"),
        actual_cost_usd=Decimal("0.061"),
        reconciled_at=clock(),
    )


def realized_placement(
    clock: MutableClock, *, provider_pod_id: str = "pod-catalog-1"
) -> PodRealizedPlacement:
    return PodRealizedPlacement(
        provider_pod_id=provider_pod_id,
        gpu_type_id="NVIDIA RTX PRO 4500",
        gpu_display_name="RTX PRO 4500",
        gpu_count=1,
        cloud=CloudType.SECURE,
        data_center_id="US-TX-3",
        hourly_rate_usd=Decimal("0.4"),
        observed_at=clock(),
    )
