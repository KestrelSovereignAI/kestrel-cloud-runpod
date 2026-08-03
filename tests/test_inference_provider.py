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
        capabilities=("chat", "streaming", "tools"),
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
    assert first[0].capabilities == (
        "chat",
        "completions",
        "embeddings",
        "streaming",
        "tools",
    )
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
    assert quote.metadata["maximum_serverless_cold_starts"] == 1
    assert capacity.provision_calls == 0

    lease = await adapter.acquire(request, quote)

    assert lease.state is InferenceLeaseState.PENDING
    assert lease.route is None
    assert capacity.provision_calls == 1
    assert service.repository.get(lease.lease_id).route_url is None


@pytest.mark.asyncio
async def test_sdk_quote_and_acquire_expose_and_persist_all_in_cost_authority(
    tmp_path,
):
    clock = MutableClock()
    capacity = FakeOllamaProvider(
        serverless_plan(
            estimated_cost=0.1,
            maximum_compute_cost=0.6,
            estimated_non_compute_cost=0.05,
            maximum_non_compute_cost=0.3,
        )
    )
    adapter, service, _capacity = _adapter(tmp_path, clock, capacity=capacity)
    request = _request(clock, max_total_cost_usd=Decimal("0.90"))

    quote = await adapter.quote(request)

    assert quote.estimated_total_cost_usd == Decimal("0.15")
    assert quote.metadata["estimated_compute_cost_usd"] == "0.1"
    assert quote.metadata["estimated_non_compute_cost_usd"] == "0.05"
    assert quote.metadata["maximum_compute_cost_usd"] == "0.6"
    assert quote.metadata["maximum_non_compute_cost_usd"] == "0.3"
    assert quote.metadata["all_in_cost_ceiling_usd"] == "0.9"
    assert quote.metadata["cost_kind"] == (
        "conservative_authorization_not_observed_billing"
    )

    pending = await adapter.acquire(request, quote)
    durable = service.repository.get(pending.lease_id)

    assert durable is not None
    assert durable.estimated_compute_cost == 0.1
    assert durable.estimated_non_compute_cost == 0.05
    assert durable.maximum_compute_cost == 0.6
    assert durable.maximum_non_compute_cost == 0.3
    assert durable.cost_ceiling == pytest.approx(0.9)
    assert durable.cost_policy_components
    assert capacity.provision_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("overhead, available", [(0.5, True), (0.500001, False)])
async def test_sdk_quote_enforces_all_in_ceiling_boundary(
    tmp_path, overhead, available
):
    clock = MutableClock()
    capacity = FakeOllamaProvider(
        serverless_plan(
            estimated_cost=0.1,
            maximum_compute_cost=0.5,
            maximum_non_compute_cost=overhead,
        )
    )
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)
    request = _request(clock, max_total_cost_usd=Decimal("1.0"))

    if available:
        quote = await adapter.quote(request)
        assert quote.metadata["all_in_cost_ceiling_usd"] == "1.0"
    else:
        with pytest.raises(InferenceLeaseProviderUnavailableError):
            await adapter.quote(request)

    assert capacity.provision_calls == 0


@pytest.mark.asyncio
async def test_sdk_acquire_revalidates_exact_non_compute_policy(tmp_path):
    clock = MutableClock()
    capacity = FakeOllamaProvider(
        serverless_plan(
            estimated_cost=0.1,
            maximum_compute_cost=0.5,
            estimated_non_compute_cost=0.05,
            maximum_non_compute_cost=0.2,
        )
    )
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)
    request = _request(clock)
    quote = await adapter.quote(request)
    capacity.selected_plan = serverless_plan(
        estimated_cost=0.1,
        maximum_compute_cost=0.5,
        estimated_non_compute_cost=0.05,
        maximum_non_compute_cost=0.21,
    )

    with pytest.raises(
        InferenceLeaseConstraintError, match="non-compute policy differs"
    ):
        await adapter.acquire(request, quote)

    assert capacity.provision_calls == 0


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
async def test_touch_renews_exact_lease_and_returns_authoritative_sdk_route(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    before = service.repository.get(ready.lease_id)
    assert before is not None
    clock.advance(30)
    capacity.route_url = "https://rotated-private.example"

    touched = await adapter.touch(request.owner_id, ready.lease_id)

    durable = service.repository.get(ready.lease_id)
    assert touched.lease_id == ready.lease_id
    assert touched.owner_id == request.owner_id
    assert touched.request_id == request.request_id
    assert touched.quote_id == ready.quote_id
    assert touched.state is InferenceLeaseState.READY
    assert touched.route is not None
    assert touched.route.endpoint.get_secret_value() == (
        "https://rotated-private.example/v1"
    )
    assert touched.route.api_key.get_secret_value() == _ROUTE_KEY
    assert touched.updated_at >= ready.updated_at
    assert touched.expires_at == ready.expires_at
    assert durable is not None
    assert durable.last_used_at == clock()
    assert durable.idle_deadline > before.idle_deadline
    assert durable.route_url is None
    assert capacity.provision_calls == 1


@pytest.mark.asyncio
async def test_touch_fails_closed_when_ready_route_is_no_longer_exact(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    before = service.repository.get(ready.lease_id)
    assert before is not None
    clock.advance(30)
    capacity.models = ()

    with pytest.raises(InferenceLeaseProvisioningError, match="could not renew"):
        await adapter.touch(request.owner_id, ready.lease_id)

    durable = service.repository.get(ready.lease_id)
    assert durable is not None
    assert durable.last_used_at == before.last_used_at
    assert durable.idle_deadline == before.idle_deadline
    assert durable.route_url is None
    assert capacity.provision_calls == 1
    assert capacity.teardown_calls == 0


@pytest.mark.asyncio
async def test_touch_cannot_resurrect_an_idle_expired_lease(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    clock.advance(request.idle_ttl_seconds + 1)

    expired = await adapter.touch(request.owner_id, ready.lease_id)

    durable = service.repository.get(ready.lease_id)
    assert expired.lease_id == ready.lease_id
    assert expired.owner_id == request.owner_id
    assert expired.request_id == request.request_id
    assert expired.state is InferenceLeaseState.EXPIRED
    assert expired.route is None
    assert durable is not None
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.termination_reason == "deadline_or_cost_cap"
    assert capacity.provision_calls == 1
    assert capacity.teardown_calls == 1


@pytest.mark.asyncio
async def test_touch_retry_returns_the_same_expired_lease_without_teardown(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    clock.advance(request.idle_ttl_seconds + 1)

    first = await adapter.touch(request.owner_id, ready.lease_id)
    durable_before_retry = service.repository.get(ready.lease_id)
    duplicate = await adapter.touch(request.owner_id, ready.lease_id)

    assert first.state is InferenceLeaseState.EXPIRED
    assert duplicate.state is InferenceLeaseState.EXPIRED
    assert duplicate.route is None
    assert duplicate.expires_at == first.expires_at
    assert service.repository.get(ready.lease_id) == durable_before_retry
    assert capacity.provision_calls == 1
    assert capacity.teardown_calls == 1


@pytest.mark.asyncio
async def test_touch_observes_expiry_after_external_reconciler_wins(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    clock.advance(request.idle_ttl_seconds + 1)

    reconciled = await service.reconcile()
    durable_before_touch = service.repository.get(ready.lease_id)
    observed = await adapter.touch(request.owner_id, ready.lease_id)

    assert reconciled[0].state is OllamaLeaseState.TERMINATED
    assert observed.state is InferenceLeaseState.EXPIRED
    assert observed.route is None
    assert service.repository.get(ready.lease_id) == durable_before_touch
    assert capacity.provision_calls == 1
    assert capacity.teardown_calls == 1


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
async def test_owner_isolation_precedes_status_touch_or_release_mutation(tmp_path):
    clock = MutableClock()
    adapter, service, capacity = _adapter(tmp_path, clock)
    request = _request(clock)
    quote = await adapter.quote(request)
    pending = await adapter.acquire(request, quote)
    ready = await adapter.status(request.owner_id, pending.lease_id)
    before = service.repository.get(ready.lease_id)
    assert before is not None
    clock.advance(30)

    with pytest.raises(InferenceLeaseOwnershipError):
        await adapter.status("owner-other", pending.lease_id)
    with pytest.raises(InferenceLeaseOwnershipError):
        await adapter.touch("owner-other", pending.lease_id)
    with pytest.raises(InferenceLeaseOwnershipError):
        await adapter.release("owner-other", pending.lease_id)

    durable = service.repository.get(ready.lease_id)
    assert durable is not None
    assert durable.last_used_at == before.last_used_at
    assert durable.idle_deadline == before.idle_deadline
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


@pytest.mark.asyncio
async def test_sdk_quote_hourly_cost_covers_every_gpu_in_the_placement(tmp_path):
    """``hourly_cost_usd`` is what the SDK enforces ``max_hourly_cost_usd`` on.

    /catalog/gpus prices per GPU, so reporting the unit price let a multi-GPU
    placement pass an hourly cap it exceeds outright — the endpoint really is
    created with ``gpu_count`` GPUs. The total-cost dimension was already
    corrected; this is the hourly dimension the SDK guards independently.
    """
    clock = MutableClock()
    capacity = FakeOllamaProvider(
        serverless_plan(rate=0.8, estimated_cost=0.1, gpu_count=2)
    )
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)

    quote = await adapter.quote(_request(clock))

    # 0.8/GPU/hr across 2 GPUs is 1.60/hr, not 0.80.
    assert quote.hourly_cost_usd == Decimal("1.6")


@pytest.mark.asyncio
async def test_sdk_quote_rejects_a_multi_gpu_placement_over_the_hourly_cap(
    tmp_path,
):
    """The cap must bind on the whole lease, not on one GPU of it."""
    clock = MutableClock()
    capacity = FakeOllamaProvider(
        serverless_plan(rate=0.8, estimated_cost=0.1, gpu_count=2)
    )
    adapter, _service, _capacity = _adapter(tmp_path, clock, capacity=capacity)

    quote = await adapter.quote(_request(clock, max_hourly_cost_usd=Decimal("1.00")))

    with pytest.raises(InferenceLeaseConstraintError, match="hourly"):
        quote.validate_for(_request(clock, max_hourly_cost_usd=Decimal("1.00")))
