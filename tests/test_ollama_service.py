"""Lifecycle, crash recovery, readiness, cost, and teardown tests."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext

import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    serverless_plan,
)

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaCleanupBasis,
    OllamaCleanupReceipt,
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseReadinessError,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaLifecycleBillingState,
    OllamaProvisioningEvidenceState,
    OllamaResourceType,
    OllamaRuntimeTelemetryState,
    OllamaTeardownState,
    ProvisionedOllamaResource,
    accrued_cost,
    provision_attempt_id,
    resource_name,
)
from kestrel_cloud_runpod.ollama_repository import (
    SQLiteOllamaLeaseRepository,
    request_from_lease,
)
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
    plan = serverless_plan()
    lease, _ = repository.insert_request(request, now=clock())
    prepared = FakeOllamaProvider(plan).validate_runtime_request(
        request, plan, provision_requested_at=clock()
    )
    lease = repository.compare_and_set(
        lease,
        changes={
            "state": OllamaLeaseState.PROVISIONING,
            "mode": plan.mode,
            "resource_type": OllamaResourceType.SERVERLESS_ENDPOINT,
            "resource_name": resource_name(request.lease_id),
            "creation_uncertain": True,
            "provision_attempt_id": provision_attempt_id(request),
            "provision_attempts": 1,
            "provisioning_started_at": clock(),
            "provisioning_evidence_state": (
                OllamaProvisioningEvidenceState.INTENT_RECORDED
            ),
            "provisioning_intent_json": prepared.intent,
            "runtime_telemetry_binding_sha256": (
                prepared.runtime_telemetry_binding_sha256
            ),
            **_persisted_plan_changes(plan),
        },
    )
    return repository.compare_and_set(
        lease, changes={"state": OllamaLeaseState.RECONCILE_REQUIRED}
    )


def _persisted_plan_changes(plan):
    return {
        "offered_rate_per_hr": plan.placement.offered_cost_per_hr,
        "placement_gpu_count": plan.placement.gpu_count,
        "estimated_cost": plan.estimated_cost,
        "estimated_compute_cost": plan.estimated_compute_cost,
        "maximum_compute_cost": plan.maximum_compute_cost,
        "estimated_non_compute_cost": plan.estimated_non_compute_cost,
        "maximum_non_compute_cost": plan.maximum_non_compute_cost,
        "cost_ceiling": plan.cost_ceiling,
        "cost_policy_components_json": json.dumps(
            [item.value for item in plan.non_compute_components],
            separators=(",", ":"),
        ),
        "maximum_concurrent_workers": plan.maximum_concurrent_workers,
        "estimated_billable_seconds": plan.estimated_billable_seconds,
        "maximum_billable_seconds": plan.maximum_billable_seconds,
        "selected_gpu_id": plan.placement.gpu_id,
        "selected_gpu_pool": plan.placement.gpu_pool,
        "selected_gpu_name": plan.placement.gpu_name,
        "catalog_observed_at": plan.placement.catalog_observed_at,
    }


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
async def test_integer_and_float_money_spellings_share_first_acceptance(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    base_request = make_request(clock)
    integer_request = replace(
        base_request,
        max_authorized_cost=2,
        constraints=replace(base_request.constraints, max_hourly_rate=1),
    )
    float_request = replace(
        integer_request,
        max_authorized_cost=2.0,
        constraints=replace(integer_request.constraints, max_hourly_rate=1.0),
    )

    first = await service.acquire(integer_request)
    repeated = await service.acquire(integer_request)
    equivalent = await service.acquire(float_request)

    assert first.request_fingerprint == repeated.request_fingerprint
    assert first.request_fingerprint == equivalent.request_fingerprint
    assert provider.provision_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", ["max_authorized_cost", "max_hourly_rate"])
async def test_unrepresentable_public_integer_cost_fails_before_admission(
    tmp_path, field_name
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)
    request = make_request(clock)
    if field_name == "max_authorized_cost":
        request = replace(request, max_authorized_cost=10**400)
    else:
        request = replace(
            request,
            constraints=replace(request.constraints, max_hourly_rate=10**400),
        )

    with pytest.raises(ValueError, match="represented as a finite float"):
        await service.acquire(request)

    assert provider.provision_calls == 0
    assert service.repository.get(request.lease_id) is None


@pytest.mark.asyncio
async def test_requested_restart_recovers_exact_original_fingerprint_inputs(tmp_path):
    clock = MutableClock()
    request = make_request(
        clock,
        requested_at=clock(),
        readiness_timeout_seconds=73,
    )
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    original, inserted = repository.insert_request(request, now=clock())
    assert inserted is True
    assert original.request_fingerprint == request.fingerprint
    provider = FakeOllamaProvider(serverless_plan())

    recovered_request = request_from_lease(
        SQLiteOllamaLeaseRepository(database).get(request.lease_id)
    )
    assert recovered_request == request
    restarted = _service(tmp_path, clock, provider)
    reconciled = await restarted.reconcile()

    assert provider.provision_calls == 1
    assert reconciled[0].state in {
        OllamaLeaseState.WAITING_FOR_MODEL,
        OllamaLeaseState.READY,
    }
    assert reconciled[0].provisioning_intent is not None
    assert reconciled[0].provisioning_intent.request_fingerprint == request.fingerprint


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
    provider = FakeOllamaProvider(
        serverless_plan(
            estimated_cost=0.1,
            maximum_compute_cost=0.6,
            estimated_non_compute_cost=0.05,
            maximum_non_compute_cost=0.2,
        )
    )

    restarted = _service(tmp_path, clock, provider)
    results = await restarted.reconcile()

    assert results[0].state is OllamaLeaseState.READY
    assert results[0].provider_resource_id == "provider-001"
    assert results[0].estimated_compute_cost == 0.1
    assert results[0].estimated_non_compute_cost == 0.05
    assert results[0].maximum_non_compute_cost == 0.2
    assert results[0].estimated_cost == pytest.approx(0.15)
    assert results[0].cost_ceiling == pytest.approx(0.8)
    assert provider.provision_calls == 1


class _BlockingProvisionProvider(FakeOllamaProvider):
    def __init__(self, plan):
        super().__init__(plan)
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()

    async def provision(self, prepared):
        self.provision_calls += 1
        self.resource = ProvisionedOllamaResource(
            resource_type=prepared.intent.resource_type,
            provider_resource_id="provider-001",
            resource_name=prepared.intent.resource_name,
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

    assert duplicate.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert "no attributable resource" in duplicate.last_provider_error
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


class _BlockingTeardownProvider(FakeOllamaProvider):
    def __init__(self):
        super().__init__(serverless_plan())
        self.teardown_entered = asyncio.Event()
        self.teardown_finish = asyncio.Event()

    async def teardown(self, resource, cleanup_intent):
        del resource
        self.teardown_calls += 1
        self.teardown_entered.set()
        await self.teardown_finish.wait()
        return OllamaCleanupReceipt(
            cleanup_intent_id=cleanup_intent.cleanup_intent_id,
            verified_absent_at=cleanup_intent.delete_started_at,
            basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
        )


class _ConcurrentFinalTelemetryFailureProvider(FakeOllamaProvider):
    def __init__(self):
        super().__init__(serverless_plan())
        self.repository = None
        self.fail_final_observation = False

    async def observe(self, resource, intent, provisioning):
        if not self.fail_final_observation:
            return await super().observe(resource, intent, provisioning)
        current = self.repository.get(intent.lease_id)
        assert current is not None
        self.repository.compare_and_set(
            current,
            changes={"last_provider_error": "concurrent reconciler observation"},
        )
        raise RunPodManagerError("final telemetry temporarily unavailable")


class _ControlFailureFinalTelemetryProvider(FakeOllamaProvider):
    def __init__(self):
        super().__init__(serverless_plan())
        self.fail_final_observation = False
        self.fail_teardown = False

    async def observe(self, resource, intent, provisioning):
        if self.fail_final_observation:
            from kestrel_sovereign.execution_custody import (
                ExecutionCommitOutcomeError,
            )

            raise ExecutionCommitOutcomeError("unknown")
        return await super().observe(resource, intent, provisioning)

    async def teardown(self, resource, cleanup_intent):
        if self.fail_teardown:
            raise RunPodManagerError("provider delete remains retryable")
        return await super().teardown(resource, cleanup_intent)


@pytest.mark.asyncio
async def test_concurrent_release_joins_one_same_id_teardown(tmp_path):
    clock = MutableClock()
    provider = _BlockingTeardownProvider()
    service = _service(tmp_path, clock, provider)
    await service.acquire(make_request(clock))

    first = asyncio.create_task(
        service.release("lease-001", owner_id="owner-001", workload_id="workload-001")
    )
    await provider.teardown_entered.wait()
    second = asyncio.create_task(
        service.release("lease-001", owner_id="owner-001", workload_id="workload-001")
    )
    try:
        await asyncio.sleep(0.05)
        calls_while_first_teardown_is_live = provider.teardown_calls
    finally:
        provider.teardown_finish.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.state is OllamaLeaseState.TERMINATED
    assert second_result.state is OllamaLeaseState.TERMINATED
    assert calls_while_first_teardown_is_live == 1
    assert provider.teardown_calls == 1
    assert first_result.teardown_attempts == 1
    assert second_result.teardown_attempts == 1


@pytest.mark.asyncio
async def test_concurrent_final_telemetry_failure_cannot_block_cleanup(tmp_path):
    clock = MutableClock()
    provider = _ConcurrentFinalTelemetryFailureProvider()
    service = _service(tmp_path, clock, provider)
    provider.repository = service.repository
    ready = await service.acquire(make_request(clock))
    provider.fail_final_observation = True

    released = await service.release(
        ready.lease_id,
        owner_id=ready.owner_id,
        workload_id=ready.workload_id,
    )

    assert released.state is OllamaLeaseState.TERMINATED
    assert released.runtime_telemetry_state is OllamaRuntimeTelemetryState.PARTIAL
    assert released.cleanup_receipt is not None
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_final_telemetry_control_failure_is_rethrown_after_cleanup(tmp_path):
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    clock = MutableClock()
    provider = _ControlFailureFinalTelemetryProvider()
    service = _service(tmp_path, clock, provider)
    ready = await service.acquire(make_request(clock))
    provider.fail_final_observation = True

    with pytest.raises(ExecutionCommitOutcomeError) as raised:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )

    assert is_execution_control_error(raised.value)
    released = service.repository.get(ready.lease_id)
    assert released is not None
    assert released.state is OllamaLeaseState.TERMINATED
    assert released.cleanup_receipt is not None
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_final_telemetry_control_and_cleanup_failure_are_both_retained(tmp_path):
    from kestrel_sovereign.execution_custody import is_execution_control_error

    clock = MutableClock()
    provider = _ControlFailureFinalTelemetryProvider()
    service = _service(tmp_path, clock, provider)
    ready = await service.acquire(make_request(clock))
    provider.fail_final_observation = True
    provider.fail_teardown = True

    with pytest.raises(BaseExceptionGroup) as raised:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )

    assert is_execution_control_error(raised.value)
    assert any(
        isinstance(error, OllamaLeaseTeardownError) for error in raised.value.exceptions
    )
    retained = service.repository.get(ready.lease_id)
    assert retained is not None
    assert retained.state is OllamaLeaseState.RELEASING
    assert retained.cleanup_receipt is None


@pytest.mark.asyncio
async def test_accrued_multi_gpu_cost_is_independent_of_decimal_context(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(
        serverless_plan(
            rate=0.47,
            estimated_cost=1.41,
            maximum_compute_cost=2.82,
            gpu_count=3,
        )
    )
    service = _service(tmp_path, clock, provider)
    request = make_request(
        clock,
        constraints=replace(make_request(clock).constraints, gpu_count=3),
        max_authorized_cost=5.0,
    )
    lease = await service.acquire(request, wait_until_ready=False)
    clock.advance(3600)

    with localcontext() as context:
        context.prec = 2
        exposure = accrued_cost(lease, clock())

    assert Decimal(str(exposure)) >= Decimal("1.41")


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
    assert failed.cleanup_intent is not None
    original_cleanup_intent = failed.cleanup_intent
    clock.advance(60)

    restarted = _service(tmp_path, clock, provider)
    reconciled = await restarted.reconcile()

    assert reconciled[0].state is OllamaLeaseState.TERMINATED
    assert reconciled[0].provider_resource_id == "provider-001"
    assert reconciled[0].teardown_attempts == 2
    assert reconciled[0].cleanup_intent == original_cleanup_intent
    assert reconciled[0].cleanup_receipt is not None
    assert reconciled[0].cleanup_receipt.cleanup_intent_id == (
        original_cleanup_intent.cleanup_intent_id
    )


@pytest.mark.asyncio
async def test_restart_reconciles_resource_created_before_id_was_persisted(tmp_path):
    clock = MutableClock()
    request = make_request(clock)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, _ = repository.insert_request(request, now=clock())
    plan = serverless_plan()
    prepared = FakeOllamaProvider(plan).validate_runtime_request(
        request, plan, provision_requested_at=clock()
    )
    lease = repository.compare_and_set(
        lease,
        changes={
            "state": OllamaLeaseState.PROVISIONING,
            "mode": plan.mode,
            "resource_type": OllamaResourceType.SERVERLESS_ENDPOINT,
            "resource_name": resource_name(request.lease_id),
            "provision_attempt_id": provision_attempt_id(request),
            "provision_attempts": 1,
            "provisioning_started_at": clock(),
            "provisioning_evidence_state": (
                OllamaProvisioningEvidenceState.INTENT_RECORDED
            ),
            "provisioning_intent_json": prepared.intent,
            "runtime_telemetry_binding_sha256": (
                prepared.runtime_telemetry_binding_sha256
            ),
            **_persisted_plan_changes(plan),
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
@pytest.mark.parametrize("ack", ["provider_id", "provisioning_receipt"])
@pytest.mark.parametrize("committed_before_lost_ack", [False, True])
async def test_restart_recovers_each_create_ack_crash_without_second_post(
    tmp_path, monkeypatch, ack, committed_before_lost_ack
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    owner = _service(tmp_path, clock, provider)
    request = make_request(clock)
    original_compare_and_set = owner.repository.compare_and_set
    original_attach = owner.repository.attach_provider_acknowledgment
    original_outcome = owner.repository.record_provisioning_outcome
    injected = False

    def lose_receipt_ack(lease, *, changes):
        nonlocal injected
        is_target = ack == "provisioning_receipt" and (
            "provisioning_receipt_json" in changes
        )
        if not injected and is_target:
            injected = True
            if committed_before_lost_ack:
                original_compare_and_set(lease, changes=changes)
            raise RunPodManagerError(f"simulated lost {ack} ACK")
        return original_compare_and_set(lease, changes=changes)

    def lose_provider_ack(lease, resource, *, now):
        nonlocal injected
        if not injected and ack == "provider_id":
            injected = True
            if committed_before_lost_ack:
                original_attach(lease, resource, now=now)
            raise RunPodManagerError(f"simulated lost {ack} ACK")
        return original_attach(lease, resource, now=now)

    def lose_outcome(lease, **outcome):
        nonlocal injected
        is_target = ack == "provisioning_receipt" and outcome.get("receipt") is not None
        if not injected and is_target:
            injected = True
            if committed_before_lost_ack:
                original_outcome(lease, **outcome)
            raise RunPodManagerError(f"simulated lost {ack} ACK")
        return original_outcome(lease, **outcome)

    monkeypatch.setattr(owner.repository, "compare_and_set", lose_receipt_ack)
    monkeypatch.setattr(
        owner.repository, "attach_provider_acknowledgment", lose_provider_ack
    )
    monkeypatch.setattr(owner.repository, "record_provisioning_outcome", lose_outcome)
    with pytest.raises(RunPodManagerError, match=f"lost {ack} ACK"):
        await owner.acquire(request, wait_until_ready=False)

    interrupted = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3").get(
        request.lease_id
    )
    assert provider.provision_calls == 1
    assert interrupted.provisioning_intent is not None
    if interrupted.provider_resource_id is None:
        provider.find_result = provider.resource

    restarted = _service(tmp_path, clock, provider)
    reconciled = await restarted.reconcile()

    assert reconciled[0].state is OllamaLeaseState.READY
    assert reconciled[0].provisioning_receipt is not None
    assert reconciled[0].provider_resource_id == provider.resource.provider_resource_id
    assert provider.provision_calls == 1


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
async def test_uncertain_creation_is_not_declared_absent_after_reconcile_window(
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

    assert results[0].state is OllamaLeaseState.RELEASING
    assert results[0].creation_uncertain is True
    assert results[0].teardown_state is OllamaTeardownState.RETRYABLE_FAILURE
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
async def test_planning_that_crosses_original_readiness_deadline_never_posts(tmp_path):
    clock = MutableClock()

    class SlowPlanningProvider(FakeOllamaProvider):
        async def plan(self, request):
            clock.advance(11)
            return await super().plan(request)

    provider = SlowPlanningProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(
        make_request(clock, readiness_timeout_seconds=10),
        wait_until_ready=False,
    )

    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.termination_reason == "expired"
    assert lease.public_route_url is None
    assert provider.provision_calls == 0
    assert provider.teardown_calls == 0
    assert lease.lifecycle_billing_state is OllamaLifecycleBillingState.NOT_APPLICABLE
    assert lease.lifecycle_billing_observations == ()


@pytest.mark.asyncio
async def test_readiness_observation_crossing_deadline_cannot_publish_route(tmp_path):
    clock = MutableClock()

    class SlowReadinessProvider(FakeOllamaProvider):
        async def observe(self, resource, intent, provisioning):
            observation = await super().observe(resource, intent, provisioning)
            clock.advance(11)
            return observation

    provider = SlowReadinessProvider(serverless_plan())
    service = _service(tmp_path, clock, provider)

    with pytest.raises(OllamaLeaseReadinessError, match="model-ready"):
        await service.acquire(make_request(clock, readiness_timeout_seconds=10))

    lease = service.repository.get("lease-001")
    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.termination_reason == "readiness_timeout"
    assert lease.ready_at is None
    assert lease.public_route_url is None
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_cost_cap_stops_slow_cold_start(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan(rate=3.6, estimated_cost=0.001))
    provider.provider_ready = False
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(
        make_request(
            clock,
            readiness_timeout_seconds=10,
            max_authorized_cost=0.0015,
        )
    )

    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.accrued_estimated_cost >= 0.0015
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_cost_cap_stops_cold_start_when_observation_keeps_failing(tmp_path):
    clock = MutableClock()
    provider = _AlwaysFailingObserveProvider(
        serverless_plan(rate=3.6, estimated_cost=0.001)
    )
    service = _service(tmp_path, clock, provider)

    lease = await service.acquire(
        make_request(
            clock,
            readiness_timeout_seconds=10,
            max_authorized_cost=0.0015,
        )
    )

    assert lease.state is OllamaLeaseState.TERMINATED
    assert lease.accrued_estimated_cost >= 0.0015
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

    async def observe(self, resource, intent, provisioning):
        self.observe_calls += 1
        if self.observe_calls == 1:
            raise RunPodManagerError("temporary control-plane read failure")
        return await super().observe(resource, intent, provisioning)


class _AlwaysFailingObserveProvider(FakeOllamaProvider):
    async def observe(self, resource, intent, provisioning):
        del intent, provisioning
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
async def test_unretained_startup_telemetry_cannot_publish_a_ready_route(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    service = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=provider,
        poll_interval_seconds=1,
        maximum_evidence_bytes=1,
        clock=clock,
        sleep=clock.sleep,
    )
    waiting = await service.acquire(make_request(clock), wait_until_ready=False)

    reconciled = await service.reconcile_lease(
        waiting.lease_id,
        owner_id=waiting.owner_id,
        workload_id=waiting.workload_id,
    )

    assert reconciled.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert reconciled.public_route_url is None
    assert reconciled.runtime_telemetry_state is OllamaRuntimeTelemetryState.PARTIAL
    assert reconciled.runtime_telemetry_observations == ()


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
@pytest.mark.parametrize("overhead, provisions", [(0.5, 1), (0.500001, 0)])
async def test_direct_acquire_enforces_all_in_ceiling_boundary(
    tmp_path, overhead, provisions
):
    clock = MutableClock()
    provider = FakeOllamaProvider(
        serverless_plan(
            estimated_cost=0.1,
            maximum_compute_cost=0.5,
            maximum_non_compute_cost=overhead,
        )
    )
    service = _service(tmp_path, clock, provider)

    if provisions:
        lease = await service.acquire(
            make_request(clock, max_authorized_cost=1.0),
            wait_until_ready=False,
        )
        assert lease.cost_ceiling == 1.0
        assert lease.maximum_non_compute_cost == 0.5
    else:
        with pytest.raises(RunPodManagerError, match="all-in cost ceiling"):
            await service.acquire(make_request(clock, max_authorized_cost=1.0))

    assert provider.provision_calls == provisions


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


@pytest.mark.asyncio
async def test_service_persists_the_placement_gpu_count_it_provisioned(tmp_path):
    """The accrual fix is only real if the count is actually written.

    accrued_cost multiplies by lease.placement_gpu_count, and nothing
    downstream cross-checks it against the constraints, so a missing write
    fails OPEN: every lease persists the default 1 and the runtime cost gate
    under-counts by gpu_count.
    """
    clock = MutableClock()
    plan = serverless_plan(rate=0.5, estimated_cost=0.1, gpu_count=4)
    capacity = FakeOllamaProvider(plan)
    service = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=capacity,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, max_authorized_cost=100.0)

    lease = await service.acquire(request)

    assert lease.placement_gpu_count == 4
    assert service.repository.get(lease.lease_id).placement_gpu_count == 4


@pytest.mark.asyncio
async def test_touch_releases_a_ready_lease_that_passed_its_cost_cap(tmp_path):
    """The cost clause of the release gate, on the path the SDK actually uses.

    RunpodInferenceLeaseProvider.acquire calls service.acquire with
    wait_until_ready=False, so the only previously-pinned gate (in
    _wait_until_ready) never runs on the SDK path. touch/status/release do not
    reach it either. Here the deadlines are still in the future and only the
    cost cap has been exceeded.
    """
    clock = MutableClock()
    capacity = FakeOllamaProvider(serverless_plan(rate=1.0, estimated_cost=0.1))
    service = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=capacity,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    # Idle and hard deadlines both stay far in the future, so the ONLY clause
    # that can fire is the cost cap. Without this the release happens through
    # the idle clause and the test proves nothing about cost.
    request = make_request(
        clock,
        max_authorized_cost=0.5,
        idle_timeout_seconds=7200,
        hard_deadline=clock() + timedelta(hours=6),
    )
    lease = await service.acquire(request)
    assert lease.state is OllamaLeaseState.READY

    clock.advance(3600)
    assert clock() < lease.hard_deadline
    assert clock() < lease.idle_deadline
    assert accrued_cost(lease, clock()) > request.max_authorized_cost

    touched = await service.touch(
        lease.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )

    assert touched.state in {
        OllamaLeaseState.RELEASING,
        OllamaLeaseState.TERMINATED,
    }
    assert touched.termination_reason == "deadline_or_cost_cap"
    assert capacity.teardown_calls == 1


@pytest.mark.asyncio
async def test_get_releases_a_ready_lease_that_passed_its_cost_cap(tmp_path):
    """The cost gate inside ``_ready_with_route``, isolated on its own surface.

    ``get`` is a live public surface (``RunPodManagerMixin.get_ollama_lease``)
    and, unlike ``touch``, it has no second cost gate after
    ``_ready_with_route`` returns. So this is the only test that fails if that
    gate's cost clause is deleted. Without it, a READY lease polled through
    ``get`` keeps being handed back its route after passing
    ``max_authorized_cost``, billing past its authorization until a deadline
    finally fires.
    """
    clock = MutableClock()
    capacity = FakeOllamaProvider(serverless_plan(rate=1.0, estimated_cost=0.1))
    service = _service(tmp_path, clock, capacity)
    # Both deadlines stay far in the future so the cost clause is the only one
    # that can fire.
    request = make_request(
        clock,
        max_authorized_cost=0.5,
        idle_timeout_seconds=7200,
        hard_deadline=clock() + timedelta(hours=6),
    )
    lease = await service.acquire(request)
    assert lease.state is OllamaLeaseState.READY

    clock.advance(3600)
    assert clock() < lease.hard_deadline
    assert clock() < lease.idle_deadline
    assert accrued_cost(lease, clock()) > request.max_authorized_cost

    fetched = await service.get(
        lease.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )

    assert fetched.state in {
        OllamaLeaseState.RELEASING,
        OllamaLeaseState.TERMINATED,
    }
    assert fetched.termination_reason == "deadline_or_cost_cap"
    assert fetched.route_url is None
    assert capacity.teardown_calls == 1


class _ObserveAdvancingProvider(FakeOllamaProvider):
    """Burn authorized budget during the provider observation round-trip."""

    def __init__(self, plan, *, clock, advance_seconds):
        super().__init__(plan)
        self._clock = clock
        self._advance_seconds = advance_seconds
        self.advance_enabled = False

    async def observe(self, resource, intent, provisioning):
        observation = await super().observe(resource, intent, provisioning)
        if self.advance_enabled:
            self._clock.advance(self._advance_seconds)
        return observation


class _BillingAdvancingProvider(FakeOllamaProvider):
    """Make a missing provider-history read outlast one retry interval."""

    def __init__(self, plan, *, clock, advance_seconds):
        super().__init__(plan)
        self._clock = clock
        self._advance_seconds = advance_seconds

    async def observe_lifecycle_billing(self, intent, provisioning, cleanup):
        del intent, provisioning, cleanup
        self._clock.advance(self._advance_seconds)


@pytest.mark.asyncio
async def test_slow_billing_read_schedules_retry_from_completed_observation(tmp_path):
    clock = MutableClock()
    capacity = _BillingAdvancingProvider(
        serverless_plan(), clock=clock, advance_seconds=7200
    )
    service = _service(tmp_path, clock, capacity)
    ready = await service.acquire(make_request(clock))
    terminated = await service.release(
        ready.lease_id,
        owner_id=ready.owner_id,
        workload_id=ready.workload_id,
    )

    reconciled = await service.reconcile_lease(
        terminated.lease_id,
        owner_id=terminated.owner_id,
        workload_id=terminated.workload_id,
    )

    assert reconciled.lifecycle_billing_state is OllamaLifecycleBillingState.PENDING
    assert reconciled.billing_last_checked_at == clock()
    assert reconciled.billing_next_check_at == clock() + timedelta(hours=1)
    assert reconciled.billing_last_provider_error == (
        "provider lifecycle billing history is incomplete"
    )


@pytest.mark.asyncio
async def test_touch_releases_a_lease_that_passes_its_cost_cap_while_observing(
    tmp_path,
):
    """``touch``'s own cost gate, isolated from the one in ``_ready_with_route``.

    ``touch`` re-reads the clock *after* the awaited ``provider.observe``, so
    its gate is the only thing that catches a lease whose accrual crosses the
    authorization during that round-trip. Here ``_ready_with_route``'s gate is
    evaluated before ``observe`` and legitimately passes; only ``touch``'s
    post-observation gate can fire. This is the test that fails if ``touch``'s
    gate is deleted while ``_ready_with_route``'s is left in place.
    """
    clock = MutableClock()
    capacity = _ObserveAdvancingProvider(
        serverless_plan(rate=1.0, estimated_cost=0.1),
        clock=clock,
        advance_seconds=1800,
    )
    service = _service(tmp_path, clock, capacity)
    request = make_request(
        clock,
        max_authorized_cost=0.5,
        idle_timeout_seconds=7200,
        hard_deadline=clock() + timedelta(hours=6),
    )
    lease = await service.acquire(request)
    assert lease.state is OllamaLeaseState.READY

    # Sit just under the authorization, so the pre-observation gate passes...
    clock.advance(1200)
    assert accrued_cost(lease, clock()) < request.max_authorized_cost
    # ...and cross it during the observation itself.
    capacity.advance_enabled = True

    touched = await service.touch(
        lease.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )

    assert clock() < lease.hard_deadline
    assert clock() < lease.idle_deadline
    assert touched.state in {
        OllamaLeaseState.RELEASING,
        OllamaLeaseState.TERMINATED,
    }
    assert touched.termination_reason == "deadline_or_cost_cap"
    assert capacity.teardown_calls == 1
