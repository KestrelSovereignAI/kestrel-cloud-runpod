"""Native acceptance for Ollama lifecycle billing and telemetry custody."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    pod_plan,
    runtime_telemetry_payload,
    serverless_plan,
)
from test_ollama_provider import _ControlClient, _provider

from kestrel_cloud_runpod import read_exact_serverless_billing
from kestrel_cloud_runpod.inference_provider import _lease_metadata
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaActualCostCapStatus,
    OllamaCleanupBasis,
    OllamaCleanupIntent,
    OllamaCleanupReceipt,
    OllamaLeaseMode,
    OllamaLeaseState,
    OllamaLifecycleBillingObservation,
    OllamaLifecycleBillingState,
    OllamaResourceType,
    ProvisionedOllamaResource,
)
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService
from kestrel_cloud_runpod.serverless_capacity_contracts import (
    serverless_endpoint_hour_cost_totals,
)

pytest_plugins = ("test_retired_endpoint_billing", "test_exact_pod_billing")


class _LifecycleBillingProvider(FakeOllamaProvider):
    """Expose the accepted generic billing boundary to the lease service."""

    def __init__(self, plan, *, control_client, billing_clock, endpoint_id):
        super().__init__(plan)
        self.resource = replace(
            self.resource,
            resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
            provider_resource_id=endpoint_id,
        )
        self.control_client = control_client
        self.billing_clock = billing_clock
        self.lifecycle_billing_calls = 0

    async def observe_lifecycle_billing(self, intent, provisioning, cleanup):
        """Use the existing exact reader; do not invent catalog authority."""

        self.lifecycle_billing_calls += 1
        hours = await read_exact_serverless_billing(
            self.control_client,
            endpoint_id=provisioning.provider_resource_id,
            created_at=provisioning.provider_created_at,
            verified_absent_at=cleanup.verified_absent_at,
            clock=self.billing_clock,
            batch_hours=24,
        )
        if hours is None:
            return None
        return OllamaLifecycleBillingObservation(
            intent_sha256=intent.intent_sha256,
            provisioning_receipt_id=provisioning.receipt_id,
            cleanup_receipt_id=cleanup.cleanup_receipt_id,
            lease_id=intent.lease_id,
            provision_attempt_id=intent.provision_attempt_id,
            resource_type=intent.resource_type,
            provider_resource_id=provisioning.provider_resource_id,
            cost_scope="serverless_endpoint_lifecycle",
            provider_capacity_actual_cost_usd=serverless_endpoint_hour_cost_totals(
                hours
            )["actual_cost_usd"],
            first_observed_at=self.billing_clock(),
            serverless_endpoint_hour_costs=hours,
        )


@pytest.mark.asyncio
async def test_runpod_provider_uses_exact_generic_serverless_billing_binding(
    billing_server,
):
    state, exact_provider, billing_clock, _request, _quote, activation = billing_server
    clock = MutableClock()
    fake = FakeOllamaProvider(serverless_plan())
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    prepared = fake.validate_runtime_request(
        request, serverless_plan(), provision_requested_at=clock()
    )
    resource = ProvisionedOllamaResource(
        OllamaResourceType.SERVERLESS_ENDPOINT,
        activation.endpoint_id,
        prepared.intent.resource_name,
    )
    provisioning = await fake.observe_provisioning(prepared.intent, resource)
    provisioning = replace(
        provisioning,
        provider_created_at=activation.create_started_at,
        readback_at=activation.readback_at,
    )
    cleanup_intent = OllamaCleanupIntent(
        intent_sha256=prepared.intent.intent_sha256,
        provisioning_receipt_id=provisioning.receipt_id,
        lease_id=prepared.intent.lease_id,
        request_fingerprint=prepared.intent.request_fingerprint,
        provision_attempt_id=prepared.intent.provision_attempt_id,
        resource_type=prepared.intent.resource_type,
        provider_resource_id=provisioning.provider_resource_id,
        resource_name=prepared.intent.resource_name,
        termination_reason="test_cleanup",
        delete_started_at=activation.readback_at + timedelta(minutes=1),
    )
    cleanup = OllamaCleanupReceipt(
        cleanup_intent_id=cleanup_intent.cleanup_intent_id,
        verified_absent_at=activation.readback_at + timedelta(minutes=2),
        basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
    )
    billing_clock.value = cleanup.verified_absent_at + timedelta(hours=2)
    provider = _provider(client=exact_provider.control_client)
    provider._clock = billing_clock

    observed = await provider.observe_lifecycle_billing(
        prepared.intent, provisioning, cleanup
    )

    assert observed is not None
    assert observed.provider_resource_id == activation.endpoint_id
    assert observed.cleanup_receipt_id == cleanup.cleanup_receipt_id
    assert observed.provider_capacity_actual_cost_usd == Decimal("0.26")
    assert observed.provider_finality_confirmed is False
    assert OllamaCleanupIntent.from_dict(cleanup_intent.to_dict()) == cleanup_intent
    assert OllamaCleanupReceipt.from_dict(cleanup.to_dict()) == cleanup
    assert OllamaLifecycleBillingObservation.from_dict(observed.to_dict()) == observed
    assert state["calls"]
    assert all(
        method == "GET" and path == "/v2/billing/serverless"
        for method, path, _query in state["calls"]
    )


@pytest.mark.asyncio
async def test_receiptless_cleanup_only_accepts_receipt_for_exact_resource():
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    prepared = provider.validate_runtime_request(
        request, serverless_plan(), provision_requested_at=clock()
    )
    resource_a = ProvisionedOllamaResource(
        OllamaResourceType.SERVERLESS_ENDPOINT,
        "endpoint-a",
        prepared.intent.resource_name,
    )
    resource_b = replace(resource_a, provider_resource_id="endpoint-b")
    receipt_a = await provider.observe_provisioning(prepared.intent, resource_a)
    receipt_b = await provider.observe_provisioning(prepared.intent, resource_b)
    cleanup = OllamaCleanupIntent(
        intent_sha256=prepared.intent.intent_sha256,
        provisioning_receipt_id=None,
        lease_id=prepared.intent.lease_id,
        request_fingerprint=prepared.intent.request_fingerprint,
        provision_attempt_id=prepared.intent.provision_attempt_id,
        resource_type=prepared.intent.resource_type,
        provider_resource_id=resource_a.provider_resource_id,
        resource_name=prepared.intent.resource_name,
        termination_reason="test cleanup",
        delete_started_at=clock(),
    )

    cleanup.validate_provisioning(prepared.intent, None)
    cleanup.validate_provisioning(prepared.intent, receipt_a)
    with pytest.raises(ValueError, match="does not bind its receipt"):
        cleanup.validate_provisioning(prepared.intent, receipt_b)

    bound = replace(cleanup, provisioning_receipt_id=receipt_a.receipt_id)
    bound.validate_provisioning(prepared.intent, receipt_a)
    with pytest.raises(ValueError, match="does not bind its receipt"):
        bound.validate_provisioning(prepared.intent, receipt_b)


@pytest.mark.asyncio
async def test_runpod_provider_uses_exact_generic_pod_billing_binding(
    native_billing,
):
    state, _read, billing_clock = native_billing
    history = state["history"]["records"]
    first_hour = datetime.fromisoformat(history[0]["startTime"])
    provider_pod_id = history[0]["podId"]
    clock = MutableClock()
    fake = FakeOllamaProvider(pod_plan(rate=0.5))
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=5.0,
    )
    prepared = fake.validate_runtime_request(
        request, pod_plan(rate=0.5), provision_requested_at=clock()
    )
    resource = ProvisionedOllamaResource(
        OllamaResourceType.POD,
        provider_pod_id,
        prepared.intent.resource_name,
    )
    provisioning = await fake.observe_provisioning(prepared.intent, resource)
    provisioning = replace(
        provisioning,
        provider_created_at=first_hour + timedelta(minutes=10),
        readback_at=first_hour + timedelta(minutes=11),
    )
    cleanup_intent = OllamaCleanupIntent(
        intent_sha256=prepared.intent.intent_sha256,
        provisioning_receipt_id=provisioning.receipt_id,
        lease_id=prepared.intent.lease_id,
        request_fingerprint=prepared.intent.request_fingerprint,
        provision_attempt_id=prepared.intent.provision_attempt_id,
        resource_type=prepared.intent.resource_type,
        provider_resource_id=provisioning.provider_resource_id,
        resource_name=prepared.intent.resource_name,
        termination_reason="test_cleanup",
        delete_started_at=first_hour + timedelta(hours=2, minutes=40),
    )
    cleanup = OllamaCleanupReceipt(
        cleanup_intent_id=cleanup_intent.cleanup_intent_id,
        verified_absent_at=first_hour + timedelta(hours=2, minutes=50),
        basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
    )
    provider = _provider(client=state["client"])
    provider._clock = billing_clock

    observed = await provider.observe_lifecycle_billing(
        prepared.intent, provisioning, cleanup
    )

    assert observed is not None
    assert observed.provider_resource_id == provider_pod_id
    assert observed.provider_capacity_actual_cost_usd == Decimal("1.2")
    assert observed.pod_billing_receipt is not None
    assert observed.pod_billing_receipt.provider_pod_id == provider_pod_id
    assert observed.provider_finality_confirmed is False
    assert len(state["calls"]) == 1


@pytest.mark.asyncio
async def test_terminated_lease_remains_due_for_exact_lifecycle_billing(
    tmp_path, billing_server
):
    """Cleanup completion must not make the original actual cost disappear."""

    state, exact_provider, billing_clock, _, _, activation = billing_server
    clock = MutableClock()
    provider = _LifecycleBillingProvider(
        serverless_plan(),
        control_client=exact_provider.control_client,
        billing_clock=billing_clock,
        endpoint_id=activation.endpoint_id,
    )
    repository = SQLiteOllamaLeaseRepository(tmp_path / "ollama-lifecycle.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    ready = await service.acquire(request, plan=serverless_plan())
    released = await service.release(
        ready.lease_id,
        owner_id=ready.owner_id,
        workload_id=ready.workload_id,
    )
    assert released.state is OllamaLeaseState.TERMINATED
    assert repository.get(request.lease_id).state is OllamaLeaseState.TERMINATED

    state["calls"].clear()
    billing_clock.value = activation.readback_at + timedelta(hours=4)
    restarted = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(repository.path),
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    reconciled = (await restarted.reconcile())[0]

    assert provider.lifecycle_billing_calls == 1
    assert any(
        method == "GET" and path == "/v2/billing/serverless"
        for method, path, _query in state["calls"]
    )
    assert reconciled.lifecycle_billing_state is OllamaLifecycleBillingState.SELECTED
    assert reconciled.lifecycle_billing_selection is not None
    assert len(reconciled.lifecycle_billing_observations) == 1
    selected = reconciled.lifecycle_billing_observations[0]
    assert selected.observation_id == (
        reconciled.lifecycle_billing_selection.observation_id
    )
    assert selected.provider_finality_confirmed is False
    assert selected.provider_capacity_actual_cost_usd == Decimal("0.26")
    assert reconciled.actual_cost_cap_status is OllamaActualCostCapStatus.UNKNOWN
    selected_metadata = _lease_metadata(reconciled)
    assert selected_metadata["provider_capacity_selected_actual_cost_usd"] == "0.26"
    assert selected_metadata["provider_capacity_latest_actual_cost_usd"] == "0.26"
    assert selected_metadata["provider_finality_confirmed"] is False
    assert selected_metadata["actual_cost_cap_status"] == "unknown"

    # A later complete provider revision is retained as correction evidence;
    # it never replaces the frozen selection or authorizes another charge.
    state["gpu_amount"] = 0.1
    clock.advance(3601)
    billing_clock.value += timedelta(hours=1)
    corrected = (await restarted.reconcile())[0]
    assert corrected.lifecycle_billing_state is (
        OllamaLifecycleBillingState.CORRECTION_PENDING
    )
    assert corrected.lifecycle_billing_selection == (
        reconciled.lifecycle_billing_selection
    )
    assert len(corrected.lifecycle_billing_observations) == 2
    assert len(corrected.lifecycle_billing_corrections) == 1
    correction = corrected.lifecycle_billing_corrections[0]
    assert correction.selected_observation_id == selected.observation_id
    assert correction.revised_observation_id == (
        corrected.lifecycle_billing_observations[-1].observation_id
    )
    assert correction.actual_cost_delta_usd == Decimal("-0.1")
    corrected_metadata = _lease_metadata(corrected)
    assert corrected_metadata["provider_capacity_selected_actual_cost_usd"] == "0.26"
    assert corrected_metadata["provider_capacity_latest_actual_cost_usd"] == "0.16"
    assert corrected_metadata["lifecycle_billing_correction_pending"] is True
    assert corrected_metadata["lifecycle_billing_correction_count"] == 1

    # Retention is bounded without truncating or silently replacing evidence.
    restarted.maximum_billing_observations = 2
    state["gpu_amount"] = 0.15
    clock.advance(3601)
    billing_clock.value += timedelta(hours=1)
    bounded = (await restarted.reconcile())[0]
    assert bounded.lifecycle_billing_selection == corrected.lifecycle_billing_selection
    assert bounded.lifecycle_billing_observations == (
        corrected.lifecycle_billing_observations
    )
    assert bounded.lifecycle_billing_corrections == (
        corrected.lifecycle_billing_corrections
    )
    assert bounded.lifecycle_billing_state is (
        OllamaLifecycleBillingState.CORRECTION_PENDING
    )
    assert bounded.billing_last_provider_error == (
        "billing evidence retention requires manual reconciliation"
    )


@pytest.mark.asyncio
async def test_ready_observation_reads_authenticated_bound_startup_telemetry(
    monkeypatch,
):
    """Provider readiness must consume the runtime's bound startup receipt."""

    paths: list[tuple[str, str | None]] = []
    telemetry_payload: dict[str, object] = {}

    def runtime_handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.url.path, request.headers.get("Authorization")))
        if request.url.path == "/api/tags" and "Authorization" not in request.headers:
            return httpx.Response(401)
        if request.url.path == "/ping":
            return httpx.Response(200)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        if request.url.path == "/kestrel/telemetry":
            return httpx.Response(200, json=telemetry_payload)
        return httpx.Response(404)

    monkeypatch.setenv(
        "TEST_OLLAMA_IMAGE",
        "ghcr.io/kestrelsovereignai/kestrel-cloud-runpod-ollama-runtime@sha256:"
        + "a" * 64,
    )
    control_client = _ControlClient()
    provider = _provider(
        client=control_client,
        transport=httpx.MockTransport(runtime_handler),
    )
    clock = MutableClock()
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    plan = pod_plan()
    prepared = provider.validate_runtime_request(
        request, plan, provision_requested_at=clock()
    )
    timestamp = clock().isoformat()
    telemetry_payload.update(
        runtime_telemetry_payload(prepared.runtime_telemetry_binding_sha256, timestamp)
    )
    resource = await provider.provision(prepared)
    control_client.pod = replace(
        control_client.pod,
        status="RUNNING",
        raw={
            **control_client.pod.raw,
            "runtime": {"ports": [{"private": 11434, "type": "http"}]},
        },
    )

    receipt = await provider.observe_provisioning(prepared.intent, resource)
    observation = await provider.observe(resource, prepared.intent, receipt)

    assert observation.provider_ready is True
    assert observation.runtime_telemetry is not None
    assert any(path == "/kestrel/telemetry" for path, _authorization in paths)


@pytest.mark.asyncio
async def test_runtime_telemetry_uses_real_loopback_bearer_and_exact_binding():
    """Exercise the provider's native HTTP client, not only MockTransport."""

    clock = MutableClock()
    fake = FakeOllamaProvider(pod_plan())
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    prepared = fake.validate_runtime_request(
        request, pod_plan(), provision_requested_at=clock()
    )
    resource = ProvisionedOllamaResource(
        OllamaResourceType.POD,
        "pod-loopback-telemetry",
        prepared.intent.resource_name,
    )
    provisioning = await fake.observe_provisioning(prepared.intent, resource)
    payload = runtime_telemetry_payload(
        prepared.runtime_telemetry_binding_sha256,
        clock().isoformat(),
    )
    requests: list[tuple[str, str | None]] = []
    provider = _provider()
    provider._clock = clock
    bearer = provider.bearer_token_for(OllamaResourceType.POD)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    valid_payload = dict(payload)
    try:
        observed = await provider._runtime_telemetry(
            f"http://127.0.0.1:{server.server_port}",
            bearer_token=bearer,
            intent=prepared.intent,
            provisioning=provisioning,
        )
        payload["placement_and_image_pull_wall_ms"] = 1
        with pytest.raises(RunPodManagerError, match="telemetry is invalid"):
            await provider._runtime_telemetry(
                f"http://127.0.0.1:{server.server_port}",
                bearer_token=bearer,
                intent=prepared.intent,
                provisioning=provisioning,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert requests == [
        ("/kestrel/telemetry", f"Bearer {bearer}"),
        ("/kestrel/telemetry", f"Bearer {bearer}"),
    ]
    assert observed.snapshot.to_dict() == valid_payload
    assert observed.snapshot.telemetry_binding_sha256 == (
        prepared.runtime_telemetry_binding_sha256
    )
    assert type(observed).from_dict(observed.to_dict()) == observed
    encoded = json.dumps(observed.to_dict(), sort_keys=True)
    assert bearer not in encoded
    assert request.model not in encoded


@pytest.mark.asyncio
async def test_python_accepts_go_nanosecond_timestamps_with_zero_millisecond_delta():
    clock = MutableClock()
    fake = FakeOllamaProvider(pod_plan())
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    prepared = fake.validate_runtime_request(
        request, pod_plan(), provision_requested_at=clock()
    )
    resource = ProvisionedOllamaResource(
        OllamaResourceType.POD,
        "pod-go-nanosecond-telemetry",
        prepared.intent.resource_name,
    )
    provisioning = await fake.observe_provisioning(prepared.intent, resource)
    payload = runtime_telemetry_payload(
        prepared.runtime_telemetry_binding_sha256,
        clock().isoformat(),
    )
    payload.update(
        ollama_process_started_at="2026-08-01T12:00:00.000000900Z",
        ollama_healthy_at="2026-08-01T12:00:00.001000100Z",
        ollama_boot_ms=0,
        model_load_started_at="2026-08-01T12:00:00.001000100Z",
        model_ready_at="2026-08-01T12:00:00.001000100Z",
        ready_at="2026-08-01T12:00:00.001000100Z",
        model_load_ms=0,
        container_to_ready_ms=1,
    )
    provider = _provider(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=payload)
        )
    )
    provider._clock = clock

    observed = await provider._runtime_telemetry(
        "https://mock.invalid",
        bearer_token="test-" + "s" * 32,
        intent=prepared.intent,
        provisioning=provisioning,
    )

    assert observed.snapshot.ollama_boot_ms == 0
    assert observed.snapshot.model_load_ms == 0
    assert observed.snapshot.container_to_ready_ms == 1
