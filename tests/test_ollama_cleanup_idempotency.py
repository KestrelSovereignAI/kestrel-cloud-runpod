"""Native regressions for monotonic multi-resource Ollama cleanup."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    serverless_plan,
)
from test_ollama_provider import (
    _claimed_two_target_cleanup,
    _contains_exception_identity,
    _ready_two_target_cleanup,
    _TwoTargetCleanupProvider,
)

from kestrel_cloud_runpod.models import EndpointResource
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaCleanupBasis,
    OllamaCleanupReceipt,
    OllamaLeaseConflictError,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaProvisioningMismatchError,
    OllamaProvisioningMismatchReason,
    OllamaResourceType,
    OllamaTeardownState,
    ProvisionedOllamaResource,
)
from kestrel_cloud_runpod.ollama_provider import _raise_mismatch
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService


def _secondary_resource(lease, resource_id="late-original-ack"):
    intent = lease.provisioning_intent
    assert intent is not None
    return ProvisionedOllamaResource(
        resource_type=intent.resource_type,
        provider_resource_id=resource_id,
        resource_name=intent.resource_name,
    )


def _absence_receipt(
    cleanup_intent,
    *,
    microseconds=1,
    basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
):
    return OllamaCleanupReceipt(
        cleanup_intent_id=cleanup_intent.cleanup_intent_id,
        verified_absent_at=(
            cleanup_intent.delete_started_at + timedelta(microseconds=microseconds)
        ),
        basis=basis,
    )


def _secondary_mismatch(lease, *, reason):
    intent = lease.provisioning_intent
    assert intent is not None
    resource = _secondary_resource(lease)
    observed = EndpointResource(
        id=resource.provider_resource_id,
        name=resource.resource_name,
        endpoint_type=None,
        request_urls={},
        raw={
            "id": resource.provider_resource_id,
            "name": resource.resource_name,
        },
    )
    with pytest.raises(OllamaProvisioningMismatchError) as caught:
        _raise_mismatch(
            observed,
            intent,
            provider_resource_id=resource.provider_resource_id,
            observed_at=lease.updated_at,
            reason=reason,
            detail="test delayed mismatch evidence",
        )
    return caught.value.evidence


async def _completed_two_target_lease(tmp_path, *, database_name):
    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        database_name=database_name,
    )
    completed = await service._teardown_all_effects(
        claimed,
        reason="test completed two-target cleanup",
    )
    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.teardown_state is OllamaTeardownState.COMPLETE
    assert completed.cleanup_receipt is not None
    assert completed.additional_cleanup_targets[0].cleanup_receipt is not None
    return service, provider, completed


async def _completed_secondary_with_open_primary(
    tmp_path,
    *,
    database_name,
    retryable,
):
    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        database_name=database_name,
    )
    repository = service.repository
    with_intent = repository.record_additional_cleanup_intent(
        claimed,
        provider_resource_id="late-original-ack",
        now=service._now(),
        termination_reason="test completed secondary cleanup",
    )
    target = with_intent.additional_cleanup_targets[0]
    assert target.cleanup_intent is not None
    current = repository.record_additional_cleanup_outcome(
        with_intent,
        provider_resource_id=target.provider_resource_id,
        now=service._now(),
        accrued_estimated_cost=with_intent.accrued_estimated_cost,
        termination_reason="test completed secondary cleanup",
        cleanup_receipt=_absence_receipt(target.cleanup_intent),
    )
    if retryable:
        current = repository.record_teardown_outcome(
            current,
            provider_resource_id=current.provider_resource_id,
            now=service._now(),
            accrued_estimated_cost=current.accrued_estimated_cost,
            termination_reason="test primary remains retryable",
            provider_error="test primary retryable failure",
            retry_at=service._now() + timedelta(seconds=10),
        )
    assert current.state is OllamaLeaseState.RELEASING
    assert current.cleanup_receipt is None
    assert current.additional_cleanup_targets[0].cleanup_receipt is not None
    return service, provider, current


@pytest.mark.asyncio
async def test_completed_secondary_ack_replay_preserves_terminal_aggregate(tmp_path):
    service, provider, completed = await _completed_two_target_lease(
        tmp_path,
        database_name="completed-secondary-replay.sqlite3",
    )
    calls_before = tuple(provider.teardown_resource_ids)

    replayed = service.repository.attach_provider_acknowledgment(
        completed,
        _secondary_resource(completed),
        now=service._now(),
    )

    assert replayed == completed
    assert tuple(provider.teardown_resource_ids) == calls_before


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [False, True], ids=["pending", "retryable"])
async def test_completed_secondary_ack_replay_preserves_open_primary_disposition(
    tmp_path,
    retryable,
):
    service, provider, current = await _completed_secondary_with_open_primary(
        tmp_path,
        database_name=f"completed-secondary-{retryable}.sqlite3",
        retryable=retryable,
    )
    calls_before = tuple(provider.teardown_resource_ids)

    replayed = service.repository.attach_provider_acknowledgment(
        current,
        _secondary_resource(current),
        now=service._now(),
    )

    assert replayed == current
    assert tuple(provider.teardown_resource_ids) == calls_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disposition",
    ["pending", "retryable", "completed"],
    ids=["pending", "retryable", "completed"],
)
async def test_completed_secondary_delayed_mismatch_preserves_disposition(
    tmp_path,
    disposition,
):
    if disposition == "completed":
        service, _, current = await _completed_two_target_lease(
            tmp_path,
            database_name="secondary-mismatch-completed.sqlite3",
        )
    else:
        service, _, current = await _completed_secondary_with_open_primary(
            tmp_path,
            database_name=f"secondary-mismatch-{disposition}.sqlite3",
            retryable=disposition == "retryable",
        )
    repository = service.repository
    mismatch = _secondary_mismatch(
        current,
        reason=OllamaProvisioningMismatchReason.RUNTIME,
    )

    retained = repository.record_provisioning_outcome(
        current,
        now=service._now(),
        acknowledged_resource=_secondary_resource(current),
        mismatch=mismatch,
    )

    assert retained.additional_cleanup_targets[0].provisioning_mismatch == mismatch
    assert (
        replace(
            retained,
            additional_cleanup_targets=current.additional_cleanup_targets,
            updated_at=current.updated_at,
            revision=current.revision,
            writer_epoch=current.writer_epoch,
        )
        == current
    )
    replayed = repository.record_provisioning_outcome(
        retained,
        now=service._now(),
        acknowledged_resource=_secondary_resource(retained),
        mismatch=mismatch,
    )
    assert replayed == retained

    conflicting = _secondary_mismatch(
        retained,
        reason=OllamaProvisioningMismatchReason.IMAGE_OR_ENVIRONMENT,
    )
    with pytest.raises(OllamaLeaseConflictError, match="mismatch is immutable"):
        repository.record_provisioning_outcome(
            replayed,
            now=service._now(),
            acknowledged_resource=_secondary_resource(replayed),
            mismatch=conflicting,
        )
    assert repository.get(current.lease_id) == retained


@pytest.mark.asyncio
async def test_genuinely_new_secondary_ack_reopens_completed_primary_cleanup(tmp_path):
    provider = FakeOllamaProvider(serverless_plan())
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name="new-secondary-positive.sqlite3",
    )
    # The fixture adds a secondary target; remove that test-only setup by using
    # the same canonical service with a fresh lease database.
    clock = service._clock
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "new-secondary-primary-only.sqlite3"
    )
    primary_service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = replace(
        make_request(clock),
        lease_id="new-secondary-lease",
        workload_id="new-secondary-workload",
    )
    primary_ready = await primary_service.acquire(request, plan=serverless_plan())
    terminated = await primary_service.release(
        primary_ready.lease_id,
        owner_id=primary_ready.owner_id,
        workload_id=primary_ready.workload_id,
    )
    assert terminated.state is OllamaLeaseState.TERMINATED
    assert terminated.cleanup_receipt is not None
    assert terminated.additional_cleanup_targets == ()

    reopened = repository.attach_provider_acknowledgment(
        terminated,
        _secondary_resource(terminated, "genuine-late-secondary"),
        now=clock(),
    )

    assert reopened.state is OllamaLeaseState.RELEASING
    assert reopened.teardown_state is OllamaTeardownState.PENDING
    assert reopened.cleanup_receipt == terminated.cleanup_receipt
    assert [
        target.provider_resource_id for target in reopened.additional_cleanup_targets
    ] == ["genuine-late-secondary"]


class _PrimaryRaceCoordinator:
    def __init__(self, *, second_primary_fails):
        self.second_primary_fails = second_primary_fails
        self.first_primary_entered = asyncio.Event()
        self.second_primary_entered = asyncio.Event()
        self.allow_first_primary = asyncio.Event()
        self.allow_second_primary = asyncio.Event()
        self.first_secondary_error = OllamaLeaseTeardownError(
            "first service retained secondary cleanup failure"
        )
        self.second_primary_error = OllamaLeaseTeardownError(
            "second service retained late primary cleanup failure"
        )
        self.calls = []
        self.secondary_calls = 0


class _PrimaryRaceProvider(FakeOllamaProvider):
    def __init__(self, coordinator, *, role):
        super().__init__(serverless_plan())
        self.coordinator = coordinator
        self.role = role

    async def teardown(self, resource, cleanup_intent):
        coordinator = self.coordinator
        coordinator.calls.append((self.role, resource.provider_resource_id))
        if resource.provider_resource_id == "provider-001":
            if self.role == "first":
                coordinator.first_primary_entered.set()
                await coordinator.allow_first_primary.wait()
                return _absence_receipt(cleanup_intent)
            coordinator.second_primary_entered.set()
            await coordinator.allow_second_primary.wait()
            if coordinator.second_primary_fails:
                raise coordinator.second_primary_error
            return _absence_receipt(
                cleanup_intent,
                microseconds=2,
                basis=OllamaCleanupBasis.AMBIGUOUS_DELETE_VERIFIED_ABSENT,
            )
        coordinator.secondary_calls += 1
        if coordinator.secondary_calls == 1:
            raise coordinator.first_secondary_error
        return _absence_receipt(cleanup_intent, microseconds=3)


async def _two_service_primary_race(tmp_path, *, second_primary_fails, database_name):
    coordinator = _PrimaryRaceCoordinator(second_primary_fails=second_primary_fails)
    clock, admission_service, with_late_ack = await _ready_two_target_cleanup(
        tmp_path,
        FakeOllamaProvider(serverless_plan()),
        database_name=database_name,
    )
    repository = admission_service.repository
    releasing = repository.compare_and_set(
        with_late_ack,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test two-service cleanup race",
        },
    )
    admitted = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test two-service cleanup race",
    )
    claimed = repository.claim_cleanup_attempt(admitted, now=clock())
    first = OllamaLeaseService(
        repository=repository,
        provider=_PrimaryRaceProvider(coordinator, role="first"),
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    second = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(repository.path),
        provider=_PrimaryRaceProvider(coordinator, role="second"),
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    owned_tasks = []
    try:
        first_task = asyncio.create_task(
            first._teardown_all_effects(claimed, reason="test first service")
        )
        owned_tasks.append(first_task)
        await asyncio.wait_for(coordinator.first_primary_entered.wait(), timeout=5)
        second_task = asyncio.create_task(
            second._teardown_all_effects(claimed, reason="test second service")
        )
        owned_tasks.append(second_task)
        await asyncio.wait_for(coordinator.second_primary_entered.wait(), timeout=5)
        coordinator.allow_first_primary.set()
        first_result = (await asyncio.gather(first_task, return_exceptions=True))[0]
        after_first = repository.get(claimed.lease_id)
        assert after_first is not None
        assert after_first.cleanup_receipt is not None
        assert after_first.additional_cleanup_targets[0].cleanup_receipt is None
        retry_after_first = after_first.reconcile_after
        clock.advance(10)
        coordinator.allow_second_primary.set()
        second_result = (await asyncio.gather(second_task, return_exceptions=True))[0]
        durable = SQLiteOllamaLeaseRepository(repository.path).get(claimed.lease_id)
        assert durable is not None
        return (
            coordinator,
            first_result,
            second_result,
            after_first,
            retry_after_first,
            durable,
        )
    finally:
        coordinator.allow_first_primary.set()
        coordinator.allow_second_primary.set()
        for task in owned_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owned_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_services_retain_first_primary_success_and_finish_secondary(tmp_path):
    (
        coordinator,
        first_result,
        second_result,
        after_first,
        _,
        durable,
    ) = await _two_service_primary_race(
        tmp_path,
        second_primary_fails=False,
        database_name="two-service-success-success.sqlite3",
    )

    assert _contains_exception_identity(first_result, coordinator.first_secondary_error)
    assert not isinstance(second_result, BaseException), repr(second_result)
    assert durable.cleanup_receipt == after_first.cleanup_receipt
    assert durable.additional_cleanup_targets[0].cleanup_receipt is not None
    assert durable.state is OllamaLeaseState.TERMINATED


@pytest.mark.asyncio
async def test_two_services_late_primary_error_preserves_receipt_and_progress(tmp_path):
    (
        coordinator,
        first_result,
        second_result,
        after_first,
        retry_after_first,
        durable,
    ) = await _two_service_primary_race(
        tmp_path,
        second_primary_fails=True,
        database_name="two-service-success-error.sqlite3",
    )

    assert _contains_exception_identity(first_result, coordinator.first_secondary_error)
    assert _contains_exception_identity(second_result, coordinator.second_primary_error)
    assert durable.cleanup_receipt == after_first.cleanup_receipt
    assert durable.reconcile_after == retry_after_first
    assert durable.additional_cleanup_targets[0].cleanup_receipt is not None
    assert durable.state is OllamaLeaseState.TERMINATED


async def _primary_completed_with_secondary_pending(tmp_path, *, database_name):
    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        database_name=database_name,
    )
    assert claimed.cleanup_intent is not None
    receipt = _absence_receipt(claimed.cleanup_intent)
    current = service.repository.record_teardown_outcome(
        claimed,
        provider_resource_id=claimed.provider_resource_id,
        now=service._now(),
        accrued_estimated_cost=claimed.accrued_estimated_cost,
        termination_reason="test primary complete",
        cleanup_receipt=receipt,
        billing_next_check_at=service._now(),
    )
    assert current.state is OllamaLeaseState.RELEASING
    assert current.cleanup_receipt == receipt
    return service, current


@pytest.mark.asyncio
async def test_completed_primary_refuses_foreign_cleanup_lineage(tmp_path):
    service, current = await _primary_completed_with_secondary_pending(
        tmp_path,
        database_name="foreign-cleanup-lineage.sqlite3",
    )
    assert current.cleanup_intent is not None
    incoming = _absence_receipt(current.cleanup_intent, microseconds=2)
    foreign_snapshots = (
        replace(current, lease_id="foreign-lease"),
        replace(
            current,
            cleanup_intent=replace(
                current.cleanup_intent,
                provider_resource_id="foreign-resource",
            ),
        ),
    )
    for foreign in foreign_snapshots:
        with pytest.raises(OllamaLeaseConflictError):
            service.repository.record_teardown_outcome(
                foreign,
                provider_resource_id=current.provider_resource_id,
                now=service._now(),
                accrued_estimated_cost=current.accrued_estimated_cost,
                termination_reason="test foreign cleanup lineage",
                cleanup_receipt=incoming,
                billing_next_check_at=service._now(),
            )
    with pytest.raises(OllamaLeaseConflictError):
        service.repository.record_teardown_outcome(
            current,
            provider_resource_id="foreign-resource",
            now=service._now(),
            accrued_estimated_cost=current.accrued_estimated_cost,
            termination_reason="test foreign cleanup resource",
            cleanup_receipt=incoming,
            billing_next_check_at=service._now(),
        )


@pytest.mark.asyncio
async def test_completed_primary_refuses_receipt_before_original_delete(tmp_path):
    service, current = await _primary_completed_with_secondary_pending(
        tmp_path,
        database_name="invalid-cleanup-time.sqlite3",
    )
    assert current.cleanup_intent is not None
    invalid = OllamaCleanupReceipt(
        cleanup_intent_id=current.cleanup_intent.cleanup_intent_id,
        verified_absent_at=(
            current.cleanup_intent.delete_started_at - timedelta(microseconds=1)
        ),
        basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
    )

    with pytest.raises(OllamaLeaseConflictError, match="unbound cleanup"):
        service.repository.record_teardown_outcome(
            current,
            provider_resource_id=current.provider_resource_id,
            now=service._now(),
            accrued_estimated_cost=current.accrued_estimated_cost,
            termination_reason="test invalid cleanup time",
            cleanup_receipt=invalid,
            billing_next_check_at=service._now(),
        )


@pytest.mark.asyncio
async def test_raw_sql_cannot_replace_first_primary_cleanup_receipt(tmp_path):
    service, current = await _primary_completed_with_secondary_pending(
        tmp_path,
        database_name="raw-primary-receipt-rewrite.sqlite3",
    )
    assert current.cleanup_intent is not None
    replacement = _absence_receipt(current.cleanup_intent, microseconds=2)

    with closing(sqlite3.connect(service.repository.path)) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """UPDATE ollama_leases
                   SET cleanup_receipt_json = ?, revision = revision + 1,
                       writer_epoch = writer_epoch + 1
                   WHERE lease_id = ?""",
                (
                    json.dumps(
                        replacement.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    current.lease_id,
                ),
            )
    retained = service.repository.get(current.lease_id)
    assert retained is not None
    assert retained.cleanup_receipt == current.cleanup_receipt


@pytest.mark.asyncio
async def test_reconciler_continues_after_concurrent_primary_completion(tmp_path):
    coordinator = _PrimaryRaceCoordinator(second_primary_fails=False)
    clock, admission_service, with_late_ack = await _ready_two_target_cleanup(
        tmp_path,
        FakeOllamaProvider(serverless_plan()),
        database_name="reconciler-primary-race.sqlite3",
    )
    repository = admission_service.repository
    releasing = repository.compare_and_set(
        with_late_ack,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test reconciler cleanup race",
        },
    )
    admitted = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test reconciler cleanup race",
    )
    external_claimed = repository.claim_cleanup_attempt(admitted, now=clock())
    unrelated_request = replace(
        make_request(clock),
        lease_id="lease-unrelated",
        workload_id="workload-unrelated",
    )
    repository.insert_request(unrelated_request, now=clock())
    external_service = OllamaLeaseService(
        repository=repository,
        provider=_PrimaryRaceProvider(coordinator, role="first"),
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    reconciler_provider = _PrimaryRaceProvider(coordinator, role="second")
    reconciler = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(repository.path),
        provider=reconciler_provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )

    owned_tasks = []
    try:
        reconcile_task = asyncio.create_task(reconciler.reconcile())
        owned_tasks.append(reconcile_task)
        await asyncio.wait_for(coordinator.second_primary_entered.wait(), timeout=5)
        external_task = asyncio.create_task(
            external_service._teardown_all_effects(
                external_claimed,
                reason="test external cleanup",
            )
        )
        owned_tasks.append(external_task)
        await asyncio.wait_for(coordinator.first_primary_entered.wait(), timeout=5)
        coordinator.allow_first_primary.set()
        external_result = (await asyncio.gather(external_task, return_exceptions=True))[
            0
        ]
        assert _contains_exception_identity(
            external_result, coordinator.first_secondary_error
        )
        coordinator.allow_second_primary.set()
        reconcile_result = (
            await asyncio.gather(reconcile_task, return_exceptions=True)
        )[0]

        assert not isinstance(reconcile_result, BaseException), repr(reconcile_result)
        unrelated = SQLiteOllamaLeaseRepository(repository.path).get(
            unrelated_request.lease_id
        )
        assert unrelated is not None
        assert unrelated.state is not OllamaLeaseState.REQUESTED
        assert reconciler_provider.provision_calls == 1
        completed = SQLiteOllamaLeaseRepository(repository.path).get(admitted.lease_id)
        assert completed is not None
        assert completed.state is OllamaLeaseState.TERMINATED
        assert completed.cleanup_receipt is not None
        assert completed.additional_cleanup_targets[0].cleanup_receipt is not None
    finally:
        coordinator.allow_first_primary.set()
        coordinator.allow_second_primary.set()
        for task in owned_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owned_tasks, return_exceptions=True)


class _SecondaryRaceCoordinator:
    def __init__(self, *, second_fails):
        self.second_fails = second_fails
        self.first_entered = asyncio.Event()
        self.second_entered = asyncio.Event()
        self.allow_first = asyncio.Event()
        self.allow_second = asyncio.Event()
        self.second_error = OllamaLeaseTeardownError(
            "second service retained late secondary cleanup failure"
        )
        self.calls = []


class _SecondaryRaceProvider(FakeOllamaProvider):
    def __init__(self, coordinator, *, role):
        super().__init__(serverless_plan())
        self.coordinator = coordinator
        self.role = role

    async def teardown(self, resource, cleanup_intent):
        coordinator = self.coordinator
        assert resource.provider_resource_id == "late-original-ack"
        coordinator.calls.append((self.role, resource.provider_resource_id))
        if self.role == "first":
            coordinator.first_entered.set()
            await coordinator.allow_first.wait()
            return _absence_receipt(cleanup_intent)
        coordinator.second_entered.set()
        await coordinator.allow_second.wait()
        if coordinator.second_fails:
            raise coordinator.second_error
        return _absence_receipt(
            cleanup_intent,
            microseconds=2,
            basis=OllamaCleanupBasis.AMBIGUOUS_DELETE_VERIFIED_ABSENT,
        )


async def _two_service_secondary_race(
    tmp_path,
    *,
    second_fails,
    database_name,
):
    coordinator = _SecondaryRaceCoordinator(second_fails=second_fails)
    service, current = await _primary_completed_with_secondary_pending(
        tmp_path,
        database_name=database_name,
    )
    repository = service.repository
    admitted = repository.record_additional_cleanup_intent(
        current,
        provider_resource_id="late-original-ack",
        now=service._now(),
        termination_reason="test two-service secondary cleanup race",
    )
    first = OllamaLeaseService(
        repository=repository,
        provider=_SecondaryRaceProvider(coordinator, role="first"),
        poll_interval_seconds=1,
        clock=service._clock,
        sleep=service._sleep,
    )
    second = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(repository.path),
        provider=_SecondaryRaceProvider(coordinator, role="second"),
        poll_interval_seconds=1,
        clock=service._clock,
        sleep=service._sleep,
    )
    owned_tasks = []
    try:
        first_task = asyncio.create_task(
            first._teardown_all_effects(admitted, reason="test first service")
        )
        owned_tasks.append(first_task)
        await asyncio.wait_for(coordinator.first_entered.wait(), timeout=5)
        second_task = asyncio.create_task(
            second._teardown_all_effects(admitted, reason="test second service")
        )
        owned_tasks.append(second_task)
        await asyncio.wait_for(coordinator.second_entered.wait(), timeout=5)
        coordinator.allow_first.set()
        first_result = (await asyncio.gather(first_task, return_exceptions=True))[0]
        after_first = repository.get(admitted.lease_id)
        assert after_first is not None
        assert after_first.state is OllamaLeaseState.TERMINATED
        first_receipt = after_first.additional_cleanup_targets[0].cleanup_receipt
        assert first_receipt is not None
        coordinator.allow_second.set()
        second_result = (await asyncio.gather(second_task, return_exceptions=True))[0]
        durable = SQLiteOllamaLeaseRepository(repository.path).get(admitted.lease_id)
        assert durable is not None
        return (
            coordinator,
            first_result,
            second_result,
            first_receipt,
            after_first,
            durable,
        )
    finally:
        coordinator.allow_first.set()
        coordinator.allow_second.set()
        for task in owned_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owned_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_services_retain_first_secondary_success(tmp_path):
    (
        coordinator,
        first_result,
        second_result,
        first_receipt,
        after_first,
        durable,
    ) = await _two_service_secondary_race(
        tmp_path,
        second_fails=False,
        database_name="two-service-secondary-success-success.sqlite3",
    )

    assert not isinstance(first_result, BaseException), repr(first_result)
    assert not isinstance(second_result, BaseException), repr(second_result)
    assert coordinator.calls == [
        ("first", "late-original-ack"),
        ("second", "late-original-ack"),
    ]
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.additional_cleanup_targets[0].cleanup_receipt == first_receipt
    assert durable == after_first


@pytest.mark.asyncio
async def test_two_services_late_secondary_error_preserves_first_receipt(tmp_path):
    (
        coordinator,
        first_result,
        second_result,
        first_receipt,
        after_first,
        durable,
    ) = await _two_service_secondary_race(
        tmp_path,
        second_fails=True,
        database_name="two-service-secondary-success-error.sqlite3",
    )

    assert not isinstance(first_result, BaseException), repr(first_result)
    assert second_result is coordinator.second_error
    assert coordinator.calls == [
        ("first", "late-original-ack"),
        ("second", "late-original-ack"),
    ]
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.additional_cleanup_targets[0].cleanup_receipt == first_receipt
    assert durable == after_first


async def _releasing_two_target_prefix(tmp_path, *, database_name):
    clock, service, with_target = await _ready_two_target_cleanup(
        tmp_path,
        FakeOllamaProvider(serverless_plan()),
        database_name=database_name,
    )
    releasing = service.repository.compare_and_set(
        with_target,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test cleanup prefix",
        },
    )
    return clock, service, releasing


def _changed_cleanup_prefix(lease, mutation):
    intent = lease.provisioning_intent
    assert intent is not None
    if mutation == "fingerprint":
        return replace(lease, request_fingerprint="f" * 64)
    if mutation == "attempt":
        return replace(lease, provision_attempt_id="foreign-attempt")
    if mutation == "intent":
        changed_intent = replace(
            intent,
            provision_requested_at=(
                intent.provision_requested_at + timedelta(seconds=1)
            ),
        )
        assert changed_intent.provider_request_json == intent.provider_request_json
        assert changed_intent.provider_request_sha256 == intent.provider_request_sha256
        assert changed_intent.intent_sha256 != intent.intent_sha256
        return replace(
            lease,
            provisioning_intent=changed_intent,
        )
    if mutation == "resource-id":
        return replace(lease, provider_resource_id="foreign-primary-resource")
    if mutation == "resource-type":
        return replace(lease, resource_type=OllamaResourceType.POD)
    if mutation == "resource-name":
        return replace(lease, resource_name="foreign-resource-name")
    if mutation == "writer-version":
        return replace(
            lease,
            writer_contract_version=lease.writer_contract_version + 1,
        )
    if mutation == "future-epoch":
        return replace(lease, writer_epoch=lease.writer_epoch + 10)
    if mutation == "future-revision":
        return replace(lease, revision=lease.revision + 10)
    if mutation == "target-prefix":
        target = lease.additional_cleanup_targets[0]
        return replace(
            lease,
            additional_cleanup_targets=(
                replace(target, resource_name="foreign-target-name"),
            ),
        )
    raise AssertionError(f"unknown cleanup-prefix mutation: {mutation}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    ["primary-intent", "additional-intent", "additional-outcome"],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "fingerprint",
        "attempt",
        "intent",
        "resource-id",
        "resource-type",
        "resource-name",
        "writer-version",
        "future-epoch",
        "future-revision",
        "target-prefix",
    ],
)
async def test_cleanup_entry_refuses_changed_supplied_prefix(
    tmp_path,
    door,
    mutation,
):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name=f"changed-prefix-{door}-{mutation}.sqlite3",
    )
    repository = service.repository
    supplied = releasing
    if door == "additional-outcome":
        supplied = repository.record_additional_cleanup_intent(
            releasing,
            provider_resource_id="late-original-ack",
            now=clock(),
            termination_reason="test cleanup prefix",
        )
    changed = _changed_cleanup_prefix(supplied, mutation)
    durable_before = repository.get(supplied.lease_id)
    assert durable_before is not None

    with pytest.raises(OllamaLeaseConflictError):
        if door == "primary-intent":
            repository.record_cleanup_intent(
                changed,
                now=clock(),
                termination_reason="test cleanup prefix",
            )
        elif door == "additional-intent":
            repository.record_additional_cleanup_intent(
                changed,
                provider_resource_id="late-original-ack",
                now=clock(),
                termination_reason="test cleanup prefix",
            )
        else:
            target = supplied.additional_cleanup_targets[0]
            assert target.cleanup_intent is not None
            repository.record_additional_cleanup_outcome(
                changed,
                provider_resource_id="late-original-ack",
                now=clock(),
                accrued_estimated_cost=supplied.accrued_estimated_cost,
                termination_reason="test cleanup prefix",
                cleanup_receipt=_absence_receipt(target.cleanup_intent),
            )
    assert repository.get(supplied.lease_id) == durable_before


@pytest.mark.asyncio
async def test_primary_cleanup_intent_birth_accepts_compatible_old_prefix(tmp_path):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name="primary-intent-old-prefix.sqlite3",
    )
    first = service.repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test primary intent birth",
    )
    replayed = SQLiteOllamaLeaseRepository(
        service.repository.path
    ).record_cleanup_intent(
        releasing,
        now=clock() + timedelta(seconds=1),
        termination_reason="changed replay reason",
    )

    assert replayed == first
    assert replayed.cleanup_intent is not None
    assert replayed.cleanup_intent.delete_started_at == clock()


@pytest.mark.asyncio
async def test_additional_cleanup_intent_birth_accepts_supplied_target_prefix(tmp_path):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name="additional-intent-old-prefix.sqlite3",
    )
    first = service.repository.record_additional_cleanup_intent(
        releasing,
        provider_resource_id="late-original-ack",
        now=clock(),
        termination_reason="test additional intent birth",
    )
    replayed = SQLiteOllamaLeaseRepository(
        service.repository.path
    ).record_additional_cleanup_intent(
        releasing,
        provider_resource_id="late-original-ack",
        now=clock() + timedelta(seconds=1),
        termination_reason="changed replay reason",
    )

    assert replayed == first
    target = replayed.additional_cleanup_targets[0]
    assert target.cleanup_intent is not None
    assert target.cleanup_intent.delete_started_at == clock()


@pytest.mark.asyncio
async def test_additional_intent_accepts_old_prefix_after_primary_intent_birth(
    tmp_path,
):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name="additional-intent-after-primary-birth.sqlite3",
    )
    repository = service.repository
    with_primary_intent = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test primary intent birth",
    )
    admitted = repository.record_additional_cleanup_intent(
        releasing,
        provider_resource_id="late-original-ack",
        now=clock(),
        termination_reason="test supplied secondary target",
    )

    assert admitted.cleanup_intent == with_primary_intent.cleanup_intent
    assert admitted.cleanup_receipt is None
    assert admitted.state is OllamaLeaseState.RELEASING
    assert admitted.additional_cleanup_targets[0].cleanup_intent is not None
    assert admitted.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
async def test_additional_outcome_accepts_old_prefix_after_primary_intent_birth(
    tmp_path,
):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name="additional-outcome-after-primary-birth.sqlite3",
    )
    repository = service.repository
    with_secondary_intent = repository.record_additional_cleanup_intent(
        releasing,
        provider_resource_id="late-original-ack",
        now=clock(),
        termination_reason="test supplied secondary target",
    )
    secondary_intent = with_secondary_intent.additional_cleanup_targets[
        0
    ].cleanup_intent
    assert secondary_intent is not None
    with_primary_intent = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test primary intent birth",
    )
    completed_secondary = repository.record_additional_cleanup_outcome(
        with_secondary_intent,
        provider_resource_id="late-original-ack",
        now=clock(),
        accrued_estimated_cost=with_secondary_intent.accrued_estimated_cost,
        termination_reason="test supplied secondary receipt",
        cleanup_receipt=_absence_receipt(secondary_intent),
    )

    assert completed_secondary.cleanup_intent == with_primary_intent.cleanup_intent
    assert completed_secondary.cleanup_receipt is None
    assert completed_secondary.state is OllamaLeaseState.RELEASING
    assert completed_secondary.additional_cleanup_targets[0].cleanup_intent == (
        secondary_intent
    )
    assert completed_secondary.additional_cleanup_targets[0].cleanup_receipt == (
        _absence_receipt(secondary_intent)
    )


@pytest.mark.asyncio
async def test_additional_outcome_cannot_borrow_current_target_intent(tmp_path):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name="additional-outcome-unseen-target-intent.sqlite3",
    )
    repository = service.repository
    with_secondary_intent = repository.record_additional_cleanup_intent(
        releasing,
        provider_resource_id="late-original-ack",
        now=clock(),
        termination_reason="test unseen secondary intent",
    )
    secondary_intent = with_secondary_intent.additional_cleanup_targets[
        0
    ].cleanup_intent
    assert secondary_intent is not None
    durable_before = repository.get(releasing.lease_id)
    assert durable_before == with_secondary_intent

    with pytest.raises(OllamaLeaseConflictError, match="cleanup target|lineage"):
        repository.record_additional_cleanup_outcome(
            releasing,
            provider_resource_id="late-original-ack",
            now=clock(),
            accrued_estimated_cost=releasing.accrued_estimated_cost,
            termination_reason="test unseen secondary intent",
            cleanup_receipt=_absence_receipt(secondary_intent),
        )
    assert repository.get(releasing.lease_id) == durable_before


async def _cleanup_prefix_before_late_target(tmp_path, *, database_name):
    provider = FakeOllamaProvider(serverless_plan())
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / database_name)
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(make_request(clock), plan=serverless_plan())
    releasing = repository.compare_and_set(
        ready,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test unseen cleanup target",
        },
    )
    supplied = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test unseen cleanup target",
    )
    current = repository.attach_provider_acknowledgment(
        supplied,
        _secondary_resource(supplied),
        now=clock(),
    )
    return clock, service, supplied, current


@pytest.mark.asyncio
async def test_additional_intent_refuses_target_absent_from_supplied_prefix(tmp_path):
    clock, service, supplied, current = await _cleanup_prefix_before_late_target(
        tmp_path,
        database_name="unseen-additional-intent.sqlite3",
    )
    latest = service.repository.latest_cleanup_lineage(supplied)
    assert latest == current
    durable_before = service.repository.get(supplied.lease_id)
    assert durable_before == current

    with pytest.raises(OllamaLeaseConflictError, match="cleanup target|lineage"):
        service.repository.record_additional_cleanup_intent(
            supplied,
            provider_resource_id="late-original-ack",
            now=clock(),
            termination_reason="test unseen cleanup target",
        )
    assert service.repository.get(supplied.lease_id) == durable_before


@pytest.mark.asyncio
async def test_additional_outcome_refuses_target_absent_from_supplied_prefix(tmp_path):
    clock, service, supplied, current = await _cleanup_prefix_before_late_target(
        tmp_path,
        database_name="unseen-additional-outcome.sqlite3",
    )
    with_intent = service.repository.record_additional_cleanup_intent(
        current,
        provider_resource_id="late-original-ack",
        now=clock(),
        termination_reason="test unseen cleanup target",
    )
    target = with_intent.additional_cleanup_targets[0]
    assert target.cleanup_intent is not None
    durable_before = service.repository.get(supplied.lease_id)
    assert durable_before == with_intent

    with pytest.raises(OllamaLeaseConflictError, match="cleanup target|lineage"):
        service.repository.record_additional_cleanup_outcome(
            supplied,
            provider_resource_id="late-original-ack",
            now=clock(),
            accrued_estimated_cost=with_intent.accrued_estimated_cost,
            termination_reason="test unseen cleanup target",
            cleanup_receipt=_absence_receipt(target.cleanup_intent),
        )
    assert service.repository.get(supplied.lease_id) == durable_before


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["intent", "time"])
async def test_completed_secondary_validates_incoming_receipt_before_replay(
    tmp_path,
    invalid_kind,
):
    clock, service, releasing = await _releasing_two_target_prefix(
        tmp_path,
        database_name=f"invalid-secondary-replay-{invalid_kind}.sqlite3",
    )
    with_intent = service.repository.record_additional_cleanup_intent(
        releasing,
        provider_resource_id="late-original-ack",
        now=clock(),
        termination_reason="test invalid secondary replay",
    )
    target = with_intent.additional_cleanup_targets[0]
    assert target.cleanup_intent is not None
    completed_secondary = service.repository.record_additional_cleanup_outcome(
        with_intent,
        provider_resource_id="late-original-ack",
        now=clock(),
        accrued_estimated_cost=with_intent.accrued_estimated_cost,
        termination_reason="test invalid secondary replay",
        cleanup_receipt=_absence_receipt(target.cleanup_intent),
    )
    assert completed_secondary.state is OllamaLeaseState.RELEASING
    assert completed_secondary.cleanup_receipt is None
    assert completed_secondary.additional_cleanup_targets[0].cleanup_receipt is not None
    if invalid_kind == "intent":
        incoming = OllamaCleanupReceipt(
            cleanup_intent_id="f" * 64,
            verified_absent_at=target.cleanup_intent.delete_started_at,
            basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
        )
    else:
        incoming = OllamaCleanupReceipt(
            cleanup_intent_id=target.cleanup_intent.cleanup_intent_id,
            verified_absent_at=(
                target.cleanup_intent.delete_started_at - timedelta(microseconds=1)
            ),
            basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
        )

    durable_before = service.repository.get(completed_secondary.lease_id)
    assert durable_before == completed_secondary
    with pytest.raises(OllamaLeaseConflictError, match="unbound additional cleanup"):
        service.repository.record_additional_cleanup_outcome(
            completed_secondary,
            provider_resource_id=target.provider_resource_id,
            now=service._now(),
            accrued_estimated_cost=completed_secondary.accrued_estimated_cost,
            termination_reason="test invalid secondary replay",
            cleanup_receipt=incoming,
        )
    assert service.repository.get(completed_secondary.lease_id) == durable_before


@pytest.mark.asyncio
async def test_raw_sql_cannot_replace_first_secondary_cleanup_receipt(tmp_path):
    service, _, completed = await _completed_two_target_lease(
        tmp_path,
        database_name="raw-secondary-receipt-rewrite.sqlite3",
    )
    target = completed.additional_cleanup_targets[0]
    assert target.cleanup_intent is not None
    replacement = replace(
        target,
        cleanup_receipt=_absence_receipt(target.cleanup_intent, microseconds=2),
    )

    with (
        closing(sqlite3.connect(service.repository.path)) as connection,
        pytest.raises(sqlite3.IntegrityError),
    ):
        connection.execute(
            """UPDATE ollama_leases
               SET additional_cleanup_targets_json = ?, revision = revision + 1,
                   writer_epoch = writer_epoch + 1
               WHERE lease_id = ?""",
            (
                json.dumps(
                    [replacement.to_dict()],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                completed.lease_id,
            ),
        )
    retained = service.repository.get(completed.lease_id)
    assert retained is not None
    assert retained.additional_cleanup_targets == completed.additional_cleanup_targets


@pytest.mark.asyncio
async def test_completed_secondary_intent_phase_never_repeats_delete(tmp_path):
    service, provider, completed = await _completed_two_target_lease(
        tmp_path,
        database_name="completed-secondary-intent-no-delete.sqlite3",
    )
    calls_before = tuple(provider.teardown_resource_ids)

    with pytest.raises(OllamaLeaseConflictError):
        await service._teardown_additional_effect(
            completed,
            provider_resource_id="late-original-ack",
            reason="test completed intent phase",
        )

    assert tuple(provider.teardown_resource_ids) == calls_before


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [False, True], ids=["pending", "retryable"])
async def test_completed_secondary_open_primary_never_repeats_delete(
    tmp_path,
    retryable,
):
    service, provider, current = await _completed_secondary_with_open_primary(
        tmp_path,
        database_name=f"completed-secondary-open-primary-{retryable}.sqlite3",
        retryable=retryable,
    )
    calls_before = tuple(provider.teardown_resource_ids)
    durable_before = service.repository.get(current.lease_id)
    assert durable_before == current
    conflict = None

    try:
        await service._teardown_additional_effect(
            current,
            provider_resource_id="late-original-ack",
            reason="test completed secondary with open primary",
        )
    except OllamaLeaseConflictError as exc:
        conflict = exc

    assert tuple(provider.teardown_resource_ids) == calls_before
    assert service.repository.get(current.lease_id) == durable_before
    assert isinstance(conflict, OllamaLeaseConflictError)


@pytest.mark.asyncio
async def test_new_secondary_ack_progresses_through_exact_cleanup(tmp_path):
    provider = FakeOllamaProvider(serverless_plan())
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "new-secondary-progress.sqlite3"
    )
    fresh_service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = replace(
        make_request(clock),
        lease_id="new-secondary-progress-lease",
        workload_id="new-secondary-progress-workload",
    )
    primary = await fresh_service.acquire(request, plan=serverless_plan())
    terminated = await fresh_service.release(
        primary.lease_id,
        owner_id=primary.owner_id,
        workload_id=primary.workload_id,
    )
    calls_before = provider.teardown_calls
    reopened = repository.attach_provider_acknowledgment(
        terminated,
        _secondary_resource(terminated, "genuine-late-progress-target"),
        now=clock(),
    )
    completed = await fresh_service.release(
        reopened.lease_id,
        owner_id=reopened.owner_id,
        workload_id=reopened.workload_id,
    )

    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.cleanup_receipt == terminated.cleanup_receipt
    assert completed.additional_cleanup_targets[0].cleanup_receipt is not None
    assert provider.teardown_calls == calls_before + 1
