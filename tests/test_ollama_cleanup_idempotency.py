"""Native regressions for monotonic multi-resource Ollama cleanup."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest
from ollama_test_support import FakeOllamaProvider, make_request, serverless_plan
from test_ollama_provider import (
    _TwoTargetCleanupProvider,
    _claimed_two_target_cleanup,
    _contains_exception_identity,
    _ready_two_target_cleanup,
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
