"""Lifecycle, crash recovery, readiness, cost, and teardown tests."""

import asyncio
from datetime import timedelta

import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    serverless_plan,
)

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseReadinessError,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaResourceType,
    OllamaTeardownState,
    ProvisionedOllamaResource,
    provision_attempt_id,
    resource_name,
)
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService


def _service(tmp_path, clock, provider):
    return OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )


def _persist_uncertain_creation(repository, request, clock):
    lease, _ = repository.insert_request(request, now=clock())
    lease = repository.compare_and_set(
        lease,
        changes={
            "state": OllamaLeaseState.PROVISIONING,
            "mode": serverless_plan().mode,
            "resource_type": OllamaResourceType.SERVERLESS_ENDPOINT,
            "resource_name": resource_name(request.lease_id),
            "creation_uncertain": True,
            "provision_attempt_id": provision_attempt_id(request),
            "provision_attempts": 1,
            "provisioning_started_at": clock(),
            "offered_rate_per_hr": 0.9,
            "estimated_cost": 0.1,
        },
    )
    return repository.compare_and_set(
        lease, changes={"state": OllamaLeaseState.RECONCILE_REQUIRED}
    )


@pytest.mark.asyncio
async def test_acquire_pulls_missing_model_and_only_returns_ready_route(tmp_path):
    clock = MutableClock()
    provider = _InspectingPullProvider(serverless_plan())
    provider.models = ()
    service = _service(tmp_path, clock, provider)
    provider.repository = service.repository

    lease = await service.acquire(make_request(clock))

    assert provider.pull_calls == 1
    assert lease.state is OllamaLeaseState.READY
    assert lease.public_route_url == "https://private.example"
    assert lease.model_pull_attempts == 1
    assert lease.model_ready_at is not None
    assert lease.cold_start_seconds == 1
    assert lease.provision_attempt_id
    assert provider.route_was_hidden_during_pull is True


@pytest.mark.asyncio
async def test_duplicate_acquire_does_not_create_second_resource(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    request = make_request(clock)

    first = await service.acquire(request)
    second = await service.acquire(request)

    assert second.lease_id == first.lease_id
    assert provider.provision_calls == 1


@pytest.mark.asyncio
async def test_touch_renews_only_after_reobserving_the_exact_ready_route(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    ready = await service.acquire(make_request(clock, idle_timeout_seconds=60))
    original_idle_deadline = ready.idle_deadline
    clock.advance(30)
    provider.route_url = "https://rotated-private.example"

    touched = await service.touch(
        ready.lease_id,
        owner_id=ready.owner_id,
        workload_id=ready.workload_id,
    )

    durable = service.repository.get(ready.lease_id)
    assert touched.lease_id == ready.lease_id
    assert touched.owner_id == ready.owner_id
    assert touched.workload_id == ready.workload_id
    assert touched.last_used_at == clock()
    assert touched.idle_deadline == clock() + timedelta(seconds=60)
    assert touched.idle_deadline > original_idle_deadline
    assert touched.public_route_url == "https://rotated-private.example"
    assert durable is not None
    assert durable.last_used_at == touched.last_used_at
    assert durable.idle_deadline == touched.idle_deadline
    assert durable.route_url is None
    assert provider.provision_calls == 1


@pytest.mark.asyncio
async def test_touch_route_failure_does_not_renew_or_replace_capacity(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    ready = await service.acquire(make_request(clock, idle_timeout_seconds=60))
    original_idle_deadline = ready.idle_deadline
    original_last_used_at = ready.last_used_at
    clock.advance(30)
    provider.models = ()

    with pytest.raises(OllamaLeaseReadinessError, match="temporarily unavailable"):
        await service.touch(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )

    durable = service.repository.get(ready.lease_id)
    assert durable is not None
    assert durable.last_used_at == original_last_used_at
    assert durable.idle_deadline == original_idle_deadline
    assert durable.route_url is None
    assert provider.provision_calls == 1
    assert provider.teardown_calls == 0


@pytest.mark.asyncio
async def test_reconciler_recovers_crash_after_requested_insert(tmp_path):
    clock = MutableClock()
    request = make_request(clock)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, inserted = repository.insert_request(request, now=clock())
    assert inserted is True
    assert lease.state is OllamaLeaseState.REQUESTED
    provider = FakeOllamaProvider(serverless_plan())

    restarted = _service(tmp_path, clock, provider)
    results = await restarted.reconcile()

    assert results[0].state is OllamaLeaseState.READY
    assert results[0].provider_resource_id == "provider-001"
    assert provider.provision_calls == 1


class _BlockingProvisionProvider(FakeOllamaProvider):
    def __init__(self, plan):
        super().__init__(plan)
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()

    async def provision(self, **kwargs):
        self.provision_calls += 1
        self.resource = ProvisionedOllamaResource(
            resource_type=kwargs["plan"].resource_type,
            provider_resource_id="provider-001",
            resource_name=kwargs["resource_name"],
        )
        self.entered.set()
        await self.resume.wait()
        return self.resource


@pytest.mark.asyncio
async def test_concurrent_duplicate_acquire_cannot_double_provision(tmp_path):
    clock = MutableClock()
    provider = _BlockingProvisionProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    request = make_request(clock)

    winner = asyncio.create_task(service.acquire(request))
    await provider.entered.wait()
    duplicate = await service.acquire(request)

    assert duplicate.state is OllamaLeaseState.PROVISIONING
    assert provider.provision_calls == 1
    provider.resume.set()
    assert (await winner).state is OllamaLeaseState.READY


@pytest.mark.asyncio
async def test_release_is_idempotent(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    await service.acquire(make_request(clock))

    released = await service.release(
        "lease-001", owner_id="owner-001", workload_id="workload-001"
    )
    duplicate = await service.release(
        "lease-001", owner_id="owner-001", workload_id="workload-001"
    )

    assert released.state is OllamaLeaseState.TERMINATED
    assert duplicate.state is OllamaLeaseState.TERMINATED
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_teardown_failure_keeps_provider_id_and_reconciler_retries(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    await service.acquire(make_request(clock))
    provider.teardown_failures = 1

    with pytest.raises(OllamaLeaseTeardownError, match="retryable"):
        await service.release(
            "lease-001", owner_id="owner-001", workload_id="workload-001"
        )
    failed = service.repository.get("lease-001")
    assert failed.provider_resource_id == "provider-001"
    assert failed.state is OllamaLeaseState.RELEASING
    assert failed.teardown_state is OllamaTeardownState.RETRYABLE_FAILURE

    reconciled = await service.reconcile()

    assert reconciled[0].state is OllamaLeaseState.TERMINATED
    assert reconciled[0].provider_resource_id == "provider-001"
    assert reconciled[0].teardown_attempts == 2


@pytest.mark.asyncio
async def test_restart_reconciles_resource_created_before_id_was_persisted(tmp_path):
    clock = MutableClock()
    request = make_request(clock)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, _ = repository.insert_request(request, now=clock())
    lease = repository.compare_and_set(
        lease,
        changes={
            "state": OllamaLeaseState.PROVISIONING,
            "mode": serverless_plan().mode,
            "resource_type": OllamaResourceType.SERVERLESS_ENDPOINT,
            "resource_name": resource_name(request.lease_id),
            "provision_attempt_id": provision_attempt_id(request),
            "provision_attempts": 1,
            "provisioning_started_at": clock(),
            "offered_rate_per_hr": 0.9,
            "estimated_cost": 0.1,
        },
    )
    provider = FakeOllamaProvider(serverless_plan())
    provider.find_result = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="recovered-provider",
        resource_name=lease.resource_name,
    )
    restarted = _service(tmp_path, clock, provider)

    results = await restarted.reconcile()

    assert results[0].state is OllamaLeaseState.READY
    assert results[0].provider_resource_id == "recovered-provider"
    assert provider.provision_calls == 0


@pytest.mark.asyncio
async def test_release_does_not_orphan_late_ambiguous_creation(tmp_path):
    clock = MutableClock()
    request = make_request(clock)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease = _persist_uncertain_creation(repository, request, clock)
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)

    with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
        await service.release(
            lease.lease_id, owner_id=lease.owner_id, workload_id=lease.workload_id
        )
    pending = repository.get(lease.lease_id)
    assert pending.state is OllamaLeaseState.RELEASING
    assert pending.creation_uncertain is True
    assert pending.provider_resource_id is None

    provider.find_result = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="late-provider",
        resource_name=lease.resource_name,
    )
    results = await service.reconcile()

    assert results[0].state is OllamaLeaseState.TERMINATED
    assert results[0].provider_resource_id == "late-provider"
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_uncertain_creation_is_not_declared_absent_before_reconcile_window(
    tmp_path,
):
    clock = MutableClock()
    request = make_request(clock, readiness_timeout_seconds=10)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease = _persist_uncertain_creation(repository, request, clock)
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)

    with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
        await service.release(
            lease.lease_id, owner_id=lease.owner_id, workload_id=lease.workload_id
        )
    assert repository.get(lease.lease_id).state is OllamaLeaseState.RELEASING

    clock.advance(11)
    results = await service.reconcile()

    assert results[0].state is OllamaLeaseState.TERMINATED
    assert provider.teardown_calls == 0


@pytest.mark.asyncio
async def test_idle_deadline_does_not_reap_a_still_provisioning_lease(tmp_path):
    clock = MutableClock()
    request = make_request(clock, idle_timeout_seconds=5, readiness_timeout_seconds=20)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease = _persist_uncertain_creation(repository, request, clock)
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    clock.advance(6)

    results = await service.reconcile()

    assert results[0].state is OllamaLeaseState.RECONCILE_REQUIRED
    assert results[0].lease_id == lease.lease_id
    assert provider.teardown_calls == 0


@pytest.mark.asyncio
async def test_reconciler_terminates_idle_lease_after_process_restart(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    first_process = _service(tmp_path, clock, provider)
    await first_process.acquire(make_request(clock, idle_timeout_seconds=10))
    clock.advance(11)

    restarted = _service(tmp_path, clock, provider)
    results = await restarted.reconcile()

    assert results[0].state is OllamaLeaseState.TERMINATED
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_one_reconcile_conflict_does_not_starve_later_expired_lease(
    tmp_path, monkeypatch
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    await service.acquire(make_request(clock, lease_id="lease-first"))
    clock.advance(1)
    await service.acquire(make_request(clock, lease_id="lease-second"))
    clock.advance(301)
    original_compare_and_set = service.repository.compare_and_set
    raised = False

    def conflict_once(lease, *, changes):
        nonlocal raised
        if (
            not raised
            and lease.lease_id == "lease-first"
            and changes.get("state") is OllamaLeaseState.RELEASING
        ):
            raised = True
            raise OllamaLeaseConflictError("simulated concurrent update")
        return original_compare_and_set(lease, changes=changes)

    monkeypatch.setattr(service.repository, "compare_and_set", conflict_once)

    results = await service.reconcile()

    assert {lease.lease_id for lease in results} == {"lease-first", "lease-second"}
    assert service.repository.get("lease-first").state is OllamaLeaseState.READY
    assert (
        "simulated concurrent update"
        in service.repository.get("lease-first").last_provider_error
    )
    assert service.repository.get("lease-second").state is OllamaLeaseState.TERMINATED
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_readiness_timeout_tears_down_without_publishing_route(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    provider.provider_ready = False
    service = _service(tmp_path, clock, provider)

    with pytest.raises(OllamaLeaseReadinessError, match="model-ready"):
        await service.acquire(make_request(clock, readiness_timeout_seconds=2))

    lease = service.repository.get("lease-001")
    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.public_route_url is None
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_cost_cap_stops_slow_cold_start(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan(rate=3.6, estimated_cost=0.0001))
    provider.provider_ready = False
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(
        make_request(
            clock,
            readiness_timeout_seconds=10,
            max_authorized_cost=0.0005,
        )
    )

    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.accrued_estimated_cost >= 0.0005
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_cost_cap_stops_cold_start_when_observation_keeps_failing(tmp_path):
    clock = MutableClock()
    provider = _AlwaysFailingObserveProvider(
        serverless_plan(rate=3.6, estimated_cost=0.0001)
    )
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(
        make_request(
            clock,
            readiness_timeout_seconds=10,
            max_authorized_cost=0.0005,
        )
    )

    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.accrued_estimated_cost >= 0.0005
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_owner_and_workload_are_both_authorized(tmp_path):
    clock = MutableClock()
    service = _service(tmp_path, clock, FakeOllamaProvider(serverless_plan()))
    await service.acquire(make_request(clock))

    with pytest.raises(OllamaLeaseAuthorizationError, match="ownership"):
        await service.get(
            "lease-001", owner_id="other-owner", workload_id="workload-001"
        )


class _RetryingPullProvider(FakeOllamaProvider):
    async def pull_model(self, resource, route_url, model):
        self.pull_calls += 1
        if self.pull_calls == 1:
            raise RunPodManagerError("pull temporarily unavailable")
        self.models = (model,)


class _InspectingPullProvider(FakeOllamaProvider):
    repository = None
    route_was_hidden_during_pull = False

    async def pull_model(self, resource, route_url, model):
        waiting = self.repository.get("lease-001")
        self.route_was_hidden_during_pull = (
            waiting.state is OllamaLeaseState.WAITING_FOR_MODEL
            and waiting.public_route_url is None
        )
        await super().pull_model(resource, route_url, model)


class _FlakyObserveProvider(FakeOllamaProvider):
    observe_calls = 0

    async def observe(self, resource):
        self.observe_calls += 1
        if self.observe_calls == 1:
            raise RunPodManagerError("temporary control-plane read failure")
        return await super().observe(resource)


class _AlwaysFailingObserveProvider(FakeOllamaProvider):
    async def observe(self, resource):
        raise RunPodManagerError("control-plane observation failed")


@pytest.mark.asyncio
async def test_transient_readiness_error_is_persisted_then_cleared_on_success(tmp_path):
    clock = MutableClock()
    provider = _FlakyObserveProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(make_request(clock))

    assert lease.state is OllamaLeaseState.READY
    assert lease.last_provider_error is None
    assert provider.observe_calls == 2


@pytest.mark.asyncio
async def test_service_rejects_provider_plan_above_cost_cap_before_creation(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan(estimated_cost=5.0))
    service = _service(tmp_path, clock, provider)

    with pytest.raises(RunPodManagerError, match="maximum authorized cost"):
        await service.acquire(make_request(clock, max_authorized_cost=1.0))

    lease = service.repository.get("lease-001")
    assert lease.state is OllamaLeaseState.FAILED
    assert provider.provision_calls == 0


@pytest.mark.asyncio
async def test_failed_model_pull_is_visible_and_retried(tmp_path):
    clock = MutableClock()
    provider = _RetryingPullProvider(serverless_plan())
    provider.models = ()
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(make_request(clock))

    assert lease.state is OllamaLeaseState.READY
    assert lease.model_pull_attempts == 2
    assert provider.pull_calls == 2
    assert lease.last_provider_error is None
