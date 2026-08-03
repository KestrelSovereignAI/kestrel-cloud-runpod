"""Deterministic doubles for generic catalog Pod capacity tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import SecretStr

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
)
from kestrel_cloud_runpod.pod_capacity_provider import (
    CreatedPodCapacity,
    PodCapacityObservation,
)
from kestrel_cloud_runpod.pod_capacity_repository import (
    SQLitePodCapacityRepository,
)
from kestrel_cloud_runpod.pod_capacity_service import PodCapacityLeaseService
from kestrel_cloud_runpod.pod_transport import CatalogPodWorkloadObservation

TOKEN = "catalog-pod-token-" + "x" * 40
REQUEST_SHA = "b" * 64
PARAMETERS_SHA = "c" * 64
IMAGE = "ghcr.io/kestrel/catalog-worker@sha256:" + "a" * 64


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 2, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def constraints() -> PodCapacityConstraints:
    return PodCapacityConstraints(
        min_vram_gb=24,
        gpu_count=1,
        cloud=CloudType.SECURE,
        min_cuda_version="12.8",
        allowed_gpu_ids=("NVIDIA RTX PRO 4500",),
        allowed_data_center_ids=("US-TX-3",),
        max_hourly_rate_usd=Decimal("0.50"),
        benchmark_id="catalog-lora-v1",
        allowed_products=(ComputeProduct.POD,),
    )


def quote(clock: MutableClock) -> PodCapacityQuote:
    requirements = constraints().placement_requirements()
    placement = PlacementDecision(
        gpu_id="NVIDIA RTX PRO 4500",
        gpu_pool=None,
        gpu_name="RTX PRO 4500",
        memory_gb=32,
        cloud=CloudType.SECURE,
        gpu_count=1,
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
        constraints=constraints(),
        gpu_type_id=placement.gpu_id,
        gpu_display_name=placement.gpu_name,
        hourly_cost_usd=Decimal("0.4"),
        estimated_cost_usd=Decimal("0.040000"),
        cost_ceiling_usd=Decimal("0.066667"),
        estimated_startup_seconds=60,
        estimated_execution_seconds=300,
        maximum_runtime_seconds=600,
        observed_at=clock(),
        expires_at=clock() + timedelta(seconds=60),
        placement=placement,
    )


def request(clock: MutableClock, **changes: Any) -> CatalogPodCapacityRequest:
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
        "quote": quote(clock),
        "accepted_max_cost_usd": Decimal("0.066667"),
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
        self.billing_receipt: PodBillingReceipt | None = None
        self.billing_calls: list[str] = []

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
            raw={"id": "pod-catalog-1"},
        )

    async def find_by_name(self, resource_name: str) -> str | None:
        raise AssertionError("catalog recovery must validate immutable metadata")

    async def find_exact(self, resource_name: str, capacity_spec) -> str | None:
        return self.find_result

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
    ) -> PodBillingReceipt | None:
        self.billing_calls.append(pod_id)
        return self.billing_receipt


class FakeWorkloadTransport:
    def __init__(self) -> None:
        self.health_calls: list[str] = []
        self.submit_calls: list[Mapping[str, Any]] = []
        self.status_value = CatalogPodWorkloadState.RUNNING
        self.result_payload: Mapping[str, Any] = {
            "dispatch_attempt_id": "attempt:catalog-run-0001",
            "request_sha256": REQUEST_SHA,
            "artifact": {"disposition": "stored"},
        }
        self.cancel_calls = 0

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
        return CatalogPodWorkloadObservation(
            attempt_id=values["attempt_id"],
            request_sha256=values["request_sha256"],
            state=self.status_value,
            error_type=(
                "CatalogExecutionError" if self.status_value.value == "failed" else None
            ),
            result_available=self.status_value is CatalogPodWorkloadState.SUCCEEDED,
        )

    async def result(self, **values):
        return dict(self.result_payload)

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
