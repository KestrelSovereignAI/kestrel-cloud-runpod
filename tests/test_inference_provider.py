"""SDK provider boundary tests for durable private Runpod inference."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from importlib import metadata as importlib_metadata

import pytest
from kestrel_sdk.llm import (
    INFERENCE_LEASE_PROVIDER_ENTRY_POINT_GROUP,
    InferenceLeaseConstraintError,
    InferenceLeaseOwnershipError,
    InferenceLeaseProvider,
    InferenceLeaseProviderUnavailableError,
    InferenceLeaseProvisioningError,
    InferenceLeaseRequest,
    InferenceLeaseState,
    InferencePrivacy,
)
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    serverless_plan,
)

from kestrel_cloud_runpod.inference_provider import RunpodInferenceLeaseProvider
from kestrel_cloud_runpod.models import GPUProfile, RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseState,
    OllamaResourceType,
)
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService

_DIGEST = "a" * 64
_ROUTE_KEY = "route-only-key-which-is-longer-than-32-characters"


def _profile() -> GPUProfile:
    return GPUProfile(
        id="ollama",
        name="Private Ollama",
        task_type="ollama",
        image_name=(
            "ghcr.io/kestrelsovereignai/kestrel-cloud-runpod-ollama-runtime"
            f"@sha256:{'b' * 64}"
        ),
        container_disk_gb=40,
        volume_gb=0,
        ports=["11434/http"],
        inference_port=11434,
        min_vram_gb=24,
        allowed_data_center_ids=("US-TX-3",),
        max_context_window=32768,
        env={"KESTREL_OLLAMA_ALLOWED_MODELS": f"qwen3:8b@sha256:{_DIGEST}"},
    )


def _settings() -> dict[str, int]:
    return {
        "quote_ttl_seconds": 60,
        "serverless_estimated_ready_seconds": 30,
        "pod_estimated_ready_seconds": 90,
    }


def _request(clock: MutableClock, **changes) -> InferenceLeaseRequest:
    request = InferenceLeaseRequest(
        request_id="request-001",
        owner_id="owner-001",
        model="qwen3:8b",
        runtime="ollama",
        max_hourly_cost_usd=Decimal("1.00"),
        max_total_cost_usd=Decimal("1.00"),
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        capabilities=("chat", "streaming"),
        allowed_regions=("us-tx-3",),
        expected_concurrency=1,
        expected_session_seconds=300,
        idle_ttl_seconds=60,
        ready_deadline_seconds=120,
        requested_at=clock(),
    )
    return replace(request, **changes)


def _adapter(tmp_path, clock, capacity=None):
    capacity = capacity or FakeOllamaProvider(serverless_plan())
    service = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=capacity,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    adapter = RunpodInferenceLeaseProvider(
        service=service,
        profile=_profile(),
        settings=_settings(),
        route_key=lambda resource_type: (
            _ROUTE_KEY
            if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
            else "pod-route-key-which-is-longer-than-32-characters"
        ),
        clock=clock,
    )
    return adapter, service, capacity


def test_dedicated_entry_point_loads_sdk_provider_contract():
    entries = importlib_metadata.entry_points().select(
        group=INFERENCE_LEASE_PROVIDER_ENTRY_POINT_GROUP,
        name="runpod",
    )

    assert len(entries) == 1
    provider = next(iter(entries)).load()()
    assert provider.provider_name == "runpod"
    assert isinstance(provider, InferenceLeaseProvider)


def test_capabilities_are_deterministic_and_policy_scoped(tmp_path):
    clock = MutableClock()
    adapter, _service, _capacity = _adapter(tmp_path, clock)

    first = adapter.capabilities()
    second = adapter.capabilities()

    assert first == second
    assert first[0].runtime == "ollama"
    assert first[0].privacy == (InferencePrivacy.AUTHENTICATED_ENDPOINT,)
    assert first[0].regions == ("us-tx-3",)
    assert first[0].max_concurrency == 1


@pytest.mark.asyncio
async def test_quote_is_read_only_and_acquire_returns_pending(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)

    quote = await adapter.quote(request)

    assert quote.provider_name == "runpod"
    assert quote.region == "us-tx-3"
    assert quote.estimated_ready_seconds == 30
    assert quote.metadata["mode"] == "serverless_load_balancer"
    assert capacity.provision_calls == 0

    lease = await adapter.acquire(request, quote)

    assert lease.state is InferenceLeaseState.PENDING
    assert lease.route is None
    assert capacity.provision_calls == 1
    assert service.repository.get(lease.lease_id).route_url is None


@pytest.mark.asyncio
async def test_quote_expires_before_estimated_cold_start_window_closes(tmp_path):
    clock = MutableClock()
    adapter, _service, capacity = _adapter(tmp_path, clock)
    request = _request(clock, ready_deadline_seconds=70)

    quote = await adapter.quote(request)

    assert quote.expires_at == request.requested_at + timedelta(seconds=40)
    clock.advance(41)
    with pytest.raises(InferenceLeaseConstraintError, match="expired"):
        await adapter.acquire(request, quote)
    assert capacity.provision_calls == 0


@pytest.mark.asyncio
async def test_catalog_refresh_cannot_consume_remaining_cold_start_window(tmp_path):
    clock = MutableClock()
    capacity = FakeOllamaProvider(serverless_plan())
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)
    request = _request(clock)
    quote = await adapter.quote(request)
    original_plan = capacity.plan

    async def slow_refresh(request):
        plan = await original_plan(request)
        clock.advance(91)
        return plan

    capacity.plan = slow_refresh

    with pytest.raises(InferenceLeaseConstraintError, match="cold start"):
        await adapter.acquire(request, quote)

    assert capacity.provision_calls == 0


@pytest.mark.asyncio
async def test_status_returns_only_exact_authenticated_ready_route(tmp_path):
    clock = MutableClock()
    adapter, service, _capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)

    ready = await adapter.status(request.owner_id, pending.lease_id)

    assert ready.state is InferenceLeaseState.READY
    assert ready.route is not None
    assert ready.route.endpoint.get_secret_value() == "https://private.example/v1"
    assert ready.route.api_key.get_secret_value() == _ROUTE_KEY
    assert "private.example" not in repr(ready)
    assert _ROUTE_KEY not in repr(ready.route)
    public = json.dumps(ready.to_public_dict(), sort_keys=True)
    assert "private.example" not in public
    assert _ROUTE_KEY not in public
    assert service.repository.get(ready.lease_id).route_url is None
    host_lease = await service.get(
        ready.lease_id,
        owner_id=request.owner_id,
        workload_id=request.request_id,
    )
    assert host_lease.public_route_url == "https://private.example"
    assert "route_url" not in host_lease.to_dict()
    assert "private.example" not in json.dumps(host_lease.to_dict())


@pytest.mark.asyncio
async def test_restart_reconciles_same_lease_without_duplicate_capacity(tmp_path):
    clock = MutableClock()
    first, _service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await first.quote(request)
    pending = await first.acquire(request, quote)

    restarted, _restarted_service, _capacity = _adapter(
        tmp_path, clock, capacity=capacity
    )
    ready = await restarted.status(request.owner_id, pending.lease_id)
    duplicate = await restarted.acquire(request, quote)

    assert ready.lease_id == pending.lease_id == duplicate.lease_id
    assert ready.quote_id == quote.quote_id == duplicate.quote_id
    assert ready.state is InferenceLeaseState.READY
    assert duplicate.state is InferenceLeaseState.READY
    assert capacity.provision_calls == 1


@pytest.mark.asyncio
async def test_owner_isolation_precedes_status_or_release_mutation(tmp_path):
    clock = MutableClock()
    adapter, _service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)

    with pytest.raises(InferenceLeaseOwnershipError):
        await adapter.status("owner-other", pending.lease_id)
    with pytest.raises(InferenceLeaseOwnershipError):
        await adapter.release("owner-other", pending.lease_id)

    assert capacity.teardown_calls == 0


@pytest.mark.asyncio
async def test_release_is_idempotent_and_route_is_absent_before_teardown(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    await adapter.status(request.owner_id, pending.lease_id)

    released = await adapter.release(request.owner_id, pending.lease_id)
    clock.advance(request.ready_deadline_seconds + 1)
    duplicate = await adapter.release(request.owner_id, pending.lease_id)

    assert released.state is InferenceLeaseState.RELEASED
    assert duplicate.state is InferenceLeaseState.RELEASED
    assert released.route is None
    assert service.repository.get(pending.lease_id).route_url is None
    assert capacity.teardown_calls == 1


@pytest.mark.asyncio
async def test_idle_expiry_tears_down_capacity_and_returns_expired(tmp_path):
    clock = MutableClock()
    adapter, _service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    assert ready.state is InferenceLeaseState.READY
    clock.advance(61)

    expired = await adapter.status(request.owner_id, pending.lease_id)

    assert expired.state is InferenceLeaseState.EXPIRED
    assert expired.route is None
    assert capacity.teardown_calls == 1


@pytest.mark.asyncio
async def test_transient_ready_route_loss_fails_closed_without_cold_restart(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    assert ready.state is InferenceLeaseState.READY
    capacity.models = ()

    with pytest.raises(InferenceLeaseProvisioningError):
        await adapter.status(request.owner_id, pending.lease_id)

    durable = service.repository.get(pending.lease_id)
    assert durable.state is OllamaLeaseState.READY
    assert durable.route_url is None
    assert capacity.teardown_calls == 0


class _FailingProvisionProvider(FakeOllamaProvider):
    async def provision(self, **kwargs):
        del kwargs
        self.provision_calls += 1
        raise RunPodManagerError("provider URL=https://secret.invalid token=hidden")


@pytest.mark.asyncio
async def test_failed_acquire_is_sanitized_and_durable(tmp_path):
    clock = MutableClock()
    capacity = _FailingProvisionProvider(serverless_plan())
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)
    request = _request(clock)
    quote = await adapter.quote(request)

    failed = await adapter.acquire(request, quote)

    assert failed.state is InferenceLeaseState.FAILED
    assert failed.failure is not None
    assert "secret.invalid" not in failed.failure.message
    assert "hidden" not in failed.failure.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"privacy": InferencePrivacy.PRIVATE_NETWORK},
        {"model": "unapproved:latest"},
        {"expected_concurrency": 2},
        {"capabilities": ("vision",)},
        {"allowed_regions": ("eu-ro-1",)},
        {"max_hourly_cost_usd": Decimal(0)},
    ],
)
async def test_static_refusals_happen_before_provider_access(tmp_path, changes):
    clock = MutableClock()
    capacity = FakeOllamaProvider(serverless_plan())
    plan_calls = 0
    original_plan = capacity.plan

    async def counted_plan(request):
        nonlocal plan_calls
        plan_calls += 1
        return await original_plan(request)

    capacity.plan = counted_plan
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)

    with pytest.raises(InferenceLeaseConstraintError):
        await adapter.quote(_request(clock, **changes))

    assert plan_calls == 0
    assert capacity.provision_calls == 0


@pytest.mark.asyncio
async def test_elapsed_readiness_deadline_refuses_before_provider_access(tmp_path):
    clock = MutableClock()
    request = _request(clock, ready_deadline_seconds=10)
    clock.advance(11)
    adapter, _service, capacity = _adapter(tmp_path, clock)

    with pytest.raises(InferenceLeaseConstraintError, match="deadline"):
        await adapter.quote(request)

    assert capacity.provision_calls == 0


@pytest.mark.asyncio
async def test_positive_cost_refusal_never_provisions(tmp_path):
    clock = MutableClock()
    adapter, _service, capacity = _adapter(tmp_path, clock)
    request = _request(
        clock,
        max_hourly_cost_usd=Decimal("0.01"),
        max_total_cost_usd=Decimal("0.01"),
    )

    with pytest.raises(InferenceLeaseProviderUnavailableError):
        await adapter.quote(request)

    assert capacity.provision_calls == 0
