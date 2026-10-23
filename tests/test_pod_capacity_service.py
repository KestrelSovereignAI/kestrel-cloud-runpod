"""End-to-end durable catalog Pod lifecycle tests without live resources."""

import asyncio
import inspect
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from pod_capacity_test_support import (
    ACK_SHA,
    REQUEST_SHA,
    TOKEN,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    acknowledgement,
    execution_control_carrier,
    final_receipt,
    record_control_injection,
    request,
    service,
)

from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    GPUOffer,
    PodResource,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)
from kestrel_cloud_runpod.pod_capacity_contracts import (
    CatalogPodWorkloadState,
    PodCapacityBillingState,
    PodCapacityCleanupError,
    PodCapacityCleanupState,
    PodCapacityConflictError,
    PodCapacityLifecycleError,
    PodCapacitySpec,
    PodCapacityState,
    TrainingPodRequest,
    TrainingPodSource,
    attempt_environment_sha256,
)
from kestrel_cloud_runpod.pod_capacity_provider import (
    PodCapacityCreatedMismatchError,
    RunpodPodCapacityProvider,
)
from kestrel_cloud_runpod.pod_transport import (
    CatalogPodTransportError,
    CatalogPodWorkloadObservation,
)
from kestrel_cloud_runpod.providers import DirectRunPodProvider
from kestrel_cloud_runpod.training_contracts import TrainingPodCleanupError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["acquire", "readiness", "release", "reconcile"],
)
async def test_capacity_service_preserves_exact_execution_control(
    tmp_path, boundary, record_property
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, transport)
    value = request(clock)
    carrier, control, explicit = execution_control_carrier(
        RunPodManagerError(f"capacity {boundary} remains unknown"),
        grouped=boundary == "reconcile",
    )
    durable_at_injection = None
    revoked_at_injection = None
    injection_entries = 0

    def capture_injection() -> None:
        nonlocal durable_at_injection, revoked_at_injection, injection_entries
        durable_at_injection = runtime.repository.get(value.capacity_id)
        revoked_at_injection = list(store.revoked)
        injection_entries += 1
        record_control_injection(record_property, boundary)

    if boundary == "acquire":
        provider.create_error = carrier
        provider.error_callback = lambda _phase: capture_injection()
        operation = lambda: runtime.provision_catalog(value)
        durable_before = None
    elif boundary == "readiness":
        admitted = await runtime.provision_catalog(value)
        durable_before = runtime.repository.get(admitted.cleanup_token)

        async def controlled_observe(*_args, **_kwargs):
            capture_injection()
            assert durable_at_injection is not None
            raise carrier

        provider.observe = controlled_observe
        operation = lambda: runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
    elif boundary == "release":
        admitted = await runtime.provision_catalog(value)
        durable_before = runtime.repository.get(admitted.cleanup_token)

        async def controlled_terminate(_pod_id):
            capture_injection()
            assert durable_at_injection is not None
            raise carrier

        provider.terminate = controlled_terminate
        operation = lambda: runtime.release(value.capacity_id, reason="test release")
    else:
        ambiguous = RunPodAmbiguousResultError(
            title="unknown",
            detail="create response was lost",
            method="POST",
            resource="/pods",
        )
        provider.create_error = ambiguous
        with pytest.raises(PodCapacityLifecycleError):
            await runtime.provision_catalog(value)
        durable_before = runtime.repository.get(value.capacity_id)
        assert durable_before is not None
        assert durable_before.capacity_spec is not None
        clock.value = min(
            durable_before.last_heartbeat_at
            + timedelta(
                seconds=durable_before.capacity_spec.request.idle_timeout_seconds
            ),
            durable_before.hard_deadline,
        )

        async def controlled_find(*_args, **_kwargs):
            capture_injection()
            raise carrier

        provider.find_exact = controlled_find
        operation = runtime.reconcile

    terminate_calls = list(provider.terminate_calls)
    with pytest.raises(type(carrier)) as raised:
        await operation()

    assert raised.value is carrier
    assert carrier.__cause__ is explicit
    if isinstance(explicit, BaseExceptionGroup):
        assert explicit.exceptions[0] is control
    else:
        assert explicit is control
    assert provider.terminate_calls == terminate_calls
    assert injection_entries == 1
    durable_after = runtime.repository.get(value.capacity_id)
    if durable_at_injection is not None:
        assert durable_after == durable_at_injection
        assert revoked_at_injection is not None
        assert store.revoked == revoked_at_injection
        if boundary == "readiness":
            assert durable_before is not None
            assert durable_at_injection.revision == durable_before.revision + 1
            assert durable_at_injection.state is durable_before.state
            assert durable_at_injection.cleanup_state is durable_before.cleanup_state
            assert durable_at_injection.stop_attempts == durable_before.stop_attempts
        elif boundary == "release":
            assert durable_before is not None
            assert durable_at_injection.family_release_requested
            assert durable_at_injection.state is PodCapacityState.RELEASING
            assert durable_at_injection.cleanup_state is PodCapacityCleanupState.PENDING
            assert (
                durable_at_injection.stop_attempts == durable_before.stop_attempts + 1
            )
    elif durable_before is not None:
        assert durable_after == durable_before
    else:
        assert durable_after is not None
        assert durable_after.last_provider_error is None
        assert durable_after.cleanup_state is PodCapacityCleanupState.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["acquire", "readiness", "release", "reconcile"])
async def test_capacity_service_retains_ordinary_normalization_contract(
    tmp_path, monkeypatch: pytest.MonkeyPatch, boundary
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    runtime = service(
        tmp_path,
        clock,
        provider,
        FakeCapabilityStore(clock),
        FakeWorkloadTransport(),
    )
    value = request(clock)
    ordinary = RunPodManagerError(f"ordinary capacity {boundary} failure")

    if boundary == "acquire":
        provider.create_error = ordinary
        with pytest.raises(PodCapacityLifecycleError) as raised:
            await runtime.provision_catalog(value)
        assert raised.value.__cause__ is ordinary
        durable = runtime.repository.get(value.capacity_id)
        assert durable is not None
        assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
        assert provider.terminate_calls == []
        return

    if boundary == "readiness":
        admitted = await runtime.provision_catalog(value)

        async def failed_observe(*_args, **_kwargs):
            raise ordinary

        provider.observe = failed_observe
        with pytest.raises(PodCapacityLifecycleError) as raised:
            await runtime.poll_catalog_readiness(
                capacity_id=value.capacity_id,
                owner_id=value.owner_id,
                workload_id=value.workload_id,
            )
        assert raised.value.__cause__ is ordinary
        durable = runtime.repository.get(admitted.cleanup_token)
        assert durable is not None
        assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
        assert provider.terminate_calls == ["pod-catalog-1"]
        return

    if boundary == "release":
        admitted = await runtime.provision_catalog(value)

        async def failed_terminate(_pod_id):
            raise ordinary

        provider.terminate = failed_terminate
        with pytest.raises(PodCapacityCleanupError) as raised:
            await runtime.release(admitted.cleanup_token, reason="ordinary release")
        member_error = raised.value.__cause__
        assert type(member_error) is TrainingPodCleanupError
        assert member_error.__cause__ is ordinary
        durable = runtime.repository.get(admitted.cleanup_token)
        assert durable is not None
        assert durable.cleanup_state is PodCapacityCleanupState.RETRYABLE_FAILURE
        return

    provider.create_error = RunPodAmbiguousResultError(
        title="unknown",
        detail="ordinary create response loss",
        method="POST",
        resource="/pods",
    )
    with pytest.raises(PodCapacityLifecycleError):
        await runtime.provision_catalog(value)
    durable_before = runtime.repository.get(value.capacity_id)
    assert durable_before is not None
    assert durable_before.capacity_spec is not None
    clock.value = min(
        durable_before.last_heartbeat_at
        + timedelta(seconds=durable_before.capacity_spec.request.idle_timeout_seconds),
        durable_before.hard_deadline,
    )
    find_entries = 0
    seen_ordinary = None

    async def failed_find(*_args, **_kwargs):
        nonlocal find_entries, seen_ordinary
        find_entries += 1
        seen_ordinary = ordinary
        raise ordinary

    monkeypatch.setattr(provider, "find_exact", failed_find)
    results = await runtime.reconcile()
    durable = runtime.repository.get(value.capacity_id)
    assert durable is not None
    assert find_entries == 1
    assert seen_ordinary is ordinary
    assert results == (durable,)
    assert durable.cleanup_state is PodCapacityCleanupState.RETRYABLE_FAILURE
    assert durable.last_provider_error == type(ordinary).__name__
    assert provider.terminate_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "created-mismatch",
        "created-mismatch-cleanup",
        "ambiguous-create",
        "readiness-health",
        "readiness-cleanup",
        "cancel-workload",
        "acknowledgement-conflict",
        "billing-conflict",
        "evidence-conflict",
        "observation-conflict",
    ],
)
async def test_catalog_capacity_boundary_preserves_exact_execution_control(
    tmp_path, monkeypatch: pytest.MonkeyPatch, boundary: str, record_property
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    value = request(clock)
    if boundary.endswith("-cleanup"):
        error = PodCapacityCleanupError(
            boundary,
            cleanup_token=value.capacity_id,
            pod_id="pod-catalog-1",
            cleanup_state=PodCapacityCleanupState.RETRYABLE_FAILURE,
            billing_risk=True,
        )
    elif boundary.endswith("-conflict"):
        error = PodCapacityConflictError(f"{boundary} remains unknown")
    elif boundary == "created-mismatch":
        error = PodCapacityCreatedMismatchError("pod-catalog-1")
    elif boundary == "ambiguous-create":
        error = RunPodAmbiguousResultError(
            title="unknown",
            detail="create response remains unknown",
            method="POST",
            resource="/pods",
        )
    elif boundary == "readiness-health":
        error = CatalogPodTransportError("readiness remains unknown")
    else:
        error = RunPodManagerError(f"{boundary} remains unknown")
    carrier, control, explicit = execution_control_carrier(
        error,
        grouped=boundary in {"acknowledgement-conflict", "observation-conflict"},
    )
    durable_at_injection = None
    revoked_at_injection = None
    injection_entries = 0

    def capture_injection() -> None:
        nonlocal durable_at_injection, revoked_at_injection, injection_entries
        durable_at_injection = runtime.repository.get(value.capacity_id)
        revoked_at_injection = list(store.revoked)
        injection_entries += 1
        record_control_injection(record_property, boundary)

    if boundary in {"created-mismatch", "created-mismatch-cleanup"}:
        provider.create_error = (
            PodCapacityCreatedMismatchError("pod-catalog-1")
            if boundary.endswith("-cleanup")
            else carrier
        )
        if boundary.endswith("-cleanup"):

            async def controlled_release(*_args, **_kwargs):
                capture_injection()
                raise carrier

            monkeypatch.setattr(runtime, "_release_one", controlled_release)
        operation = lambda: runtime.provision_catalog(value)
        if not boundary.endswith("-cleanup"):
            provider.error_callback = lambda _phase: capture_injection()
    elif boundary == "ambiguous-create":
        provider.create_error = carrier
        provider.error_callback = lambda _phase: capture_injection()
        operation = lambda: runtime.provision_catalog(value)
    elif boundary in {"readiness-health", "readiness-cleanup"}:
        admitted = await runtime.provision_catalog(value)
        if boundary == "readiness-health":

            async def controlled_health(_base_url):
                transport.health_calls.append(_base_url)
                capture_injection()
                raise carrier

            monkeypatch.setattr(transport, "health", controlled_health)
        else:

            async def ordinary_observe(*_args, **_kwargs):
                raise RunPodManagerError("ordinary readiness failure")

            async def controlled_release(*_args, **_kwargs):
                capture_injection()
                raise carrier

            monkeypatch.setattr(provider, "observe", ordinary_observe)
            monkeypatch.setattr(runtime, "_release_one", controlled_release)
        operation = lambda: runtime.poll_catalog_readiness(
            capacity_id=admitted.cleanup_token,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
    elif boundary == "cancel-workload":
        await runtime.acquire_catalog(value)

        async def controlled_cancel(**_kwargs):
            capture_injection()
            raise carrier

        monkeypatch.setattr(transport, "cancel", controlled_cancel)
        operation = lambda: runtime.cancel_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
    elif boundary == "acknowledgement-conflict":
        await runtime.acquire_catalog(value)
        await runtime.submit_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
            payload={
                "dispatch_attempt_id": value.attempt_id,
                "request_sha256": value.request_sha256,
            },
        )
        transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
        await runtime.observe_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        compare_and_set = runtime.repository.compare_and_set
        injected = False

        def controlled_compare_and_set(current, *, changes):
            nonlocal injected
            if not injected and "artifact_handoff_json" in changes:
                injected = True
                capture_injection()
                raise carrier
            return compare_and_set(current, changes=changes)

        monkeypatch.setattr(
            runtime.repository, "compare_and_set", controlled_compare_and_set
        )
        operation = lambda: runtime.acknowledge_catalog_result(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
            acknowledgement=acknowledgement(),
        )
    elif boundary == "billing-conflict":
        admitted = await runtime.provision_catalog(value)
        current = runtime.repository.compare_and_set(
            admitted,
            changes={
                "state": PodCapacityState.RELEASING,
                "cleanup_state": PodCapacityCleanupState.COMPLETE,
                "terminated_at": clock(),
            },
        )
        provider.billing_receipt = final_receipt(clock)
        transition_with_evidence = runtime._transition_with_evidence

        def controlled_transition(*args, **kwargs):
            transition_with_evidence(*args, **kwargs)
            capture_injection()
            raise carrier

        monkeypatch.setattr(runtime, "_transition_with_evidence", controlled_transition)
        operation = lambda: runtime._reconcile_billing(current)
    elif boundary == "evidence-conflict":
        admitted = await runtime.provision_catalog(value)
        compare_and_set = runtime.repository.compare_and_set
        injected = False

        def controlled_compare_and_set(current, *, changes):
            nonlocal injected
            if not injected and "evidence_json" in changes:
                injected = True
                capture_injection()
                raise carrier
            return compare_and_set(current, changes=changes)

        monkeypatch.setattr(
            runtime.repository, "compare_and_set", controlled_compare_and_set
        )
        operation = lambda: runtime._transition_with_evidence(
            admitted,
            changes={"last_provider_error": "controlled"},
            lifecycle={"worker_ready_at": clock()},
        )
    else:
        admitted = await runtime.provision_catalog(value)
        ready = await runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        assert ready.state is PodCapacityState.READY
        observation = CatalogPodWorkloadObservation(
            attempt_id=value.attempt_id,
            request_sha256=value.request_sha256,
            state=CatalogPodWorkloadState.RUNNING,
            error_type=None,
            result_available=False,
        )
        transition_with_evidence = runtime._transition_with_evidence
        injected = False

        def controlled_transition(*args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                capture_injection()
                raise carrier
            return transition_with_evidence(*args, **kwargs)

        monkeypatch.setattr(runtime, "_transition_with_evidence", controlled_transition)
        operation = lambda: runtime._record_catalog_observation(
            admitted.cleanup_token, observation
        )

    provider_call_counts = (
        len(provider.create_calls),
        len(provider.terminate_calls),
        len(provider.billing_calls),
    )
    workload_calls = (
        len(transport.health_calls),
        len(transport.submit_calls),
        len(transport.acknowledgement_calls),
        transport.cancel_calls,
    )
    with pytest.raises(type(carrier)) as raised:
        outcome = operation()
        if inspect.isawaitable(outcome):
            await outcome

    assert raised.value is carrier
    assert carrier.__cause__ is explicit
    if isinstance(explicit, BaseExceptionGroup):
        assert explicit.exceptions[0] is control
    else:
        assert explicit is control
    assert injection_entries == 1
    assert (
        len(provider.create_calls),
        len(provider.terminate_calls),
        len(provider.billing_calls),
    ) == (
        provider_call_counts[0] + int(boundary.startswith(("created-", "ambiguous-"))),
        provider_call_counts[1],
        provider_call_counts[2] + int(boundary == "billing-conflict"),
    )
    assert (
        len(transport.health_calls),
        len(transport.submit_calls),
        len(transport.acknowledgement_calls),
        transport.cancel_calls,
    ) == (
        workload_calls[0] + int(boundary == "readiness-health"),
        workload_calls[1],
        workload_calls[2],
        workload_calls[3],
    )
    if durable_at_injection is not None:
        assert durable_at_injection is not None
        assert revoked_at_injection is not None
        assert runtime.repository.get(value.capacity_id) == durable_at_injection
        assert store.revoked == revoked_at_injection
    if boundary == "billing-conflict":
        assert (
            durable_at_injection.billing_state is PodCapacityBillingState.AUTHORITATIVE
        )
        assert durable_at_injection.state is PodCapacityState.RELEASING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "created-mismatch",
        "created-mismatch-cleanup",
        "ambiguous-create",
        "readiness-health",
        "readiness-cleanup",
        "cancel-workload",
        "acknowledgement-conflict",
        "billing-conflict",
        "evidence-conflict",
        "observation-conflict",
    ],
)
async def test_catalog_capacity_boundary_retains_ordinary_normalization_contract(
    tmp_path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)

    if boundary in {"created-mismatch", "created-mismatch-cleanup"}:
        mismatch = PodCapacityCreatedMismatchError("pod-catalog-1")
        provider.create_error = mismatch
        if boundary.endswith("-cleanup"):
            cleanup = PodCapacityCleanupError(
                "ordinary created-mismatch cleanup",
                cleanup_token=value.capacity_id,
                pod_id="pod-catalog-1",
                cleanup_state=PodCapacityCleanupState.RETRYABLE_FAILURE,
                billing_risk=True,
            )

            async def failed_release(*_args, **_kwargs):
                raise cleanup

            monkeypatch.setattr(runtime, "_release_one", failed_release)
            with pytest.raises(PodCapacityCleanupError) as raised:
                await runtime.provision_catalog(value)
            assert raised.value is cleanup
            assert raised.value.__cause__ is mismatch
            assert provider.terminate_calls == []
        else:
            with pytest.raises(PodCapacityLifecycleError) as raised:
                await runtime.provision_catalog(value)
            assert raised.value.__cause__ is mismatch
            assert provider.terminate_calls == ["pod-catalog-1"]
        return

    if boundary == "ambiguous-create":
        ambiguous = RunPodAmbiguousResultError(
            title="unknown",
            detail="ordinary create response loss",
            method="POST",
            resource="/pods",
        )
        provider.create_error = ambiguous
        with pytest.raises(PodCapacityLifecycleError) as raised:
            await runtime.provision_catalog(value)
        assert raised.value.__cause__ is ambiguous
        assert provider.terminate_calls == []
        return

    admitted = await runtime.provision_catalog(value)
    if boundary == "readiness-health":
        ordinary = CatalogPodTransportError("ordinary unready worker")

        async def failed_health(base_url):
            transport.health_calls.append(base_url)
            raise ordinary

        monkeypatch.setattr(transport, "health", failed_health)
        result = await runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        assert result.state is PodCapacityState.STARTING
        assert result.last_provider_error == "CatalogPodTransportError"
        assert transport.health_calls == [provider.route]
        return

    if boundary == "readiness-cleanup":
        ordinary = RunPodManagerError("ordinary readiness failure")
        cleanup = PodCapacityCleanupError(
            "ordinary readiness cleanup",
            cleanup_token=value.capacity_id,
            pod_id="pod-catalog-1",
            cleanup_state=PodCapacityCleanupState.RETRYABLE_FAILURE,
            billing_risk=True,
        )

        async def failed_observe(*_args, **_kwargs):
            raise ordinary

        async def failed_release(*_args, **_kwargs):
            raise cleanup

        monkeypatch.setattr(provider, "observe", failed_observe)
        monkeypatch.setattr(runtime, "_release_one", failed_release)
        with pytest.raises(PodCapacityCleanupError) as raised:
            await runtime.poll_catalog_readiness(
                capacity_id=value.capacity_id,
                owner_id=value.owner_id,
                workload_id=value.workload_id,
            )
        assert raised.value is cleanup
        assert raised.value.__cause__ is ordinary
        assert provider.terminate_calls == []
        return

    if boundary == "cancel-workload":
        ready = await runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        assert ready.state is PodCapacityState.READY
        ordinary = CatalogPodTransportError("ordinary cancellation failure")

        async def failed_cancel(**_kwargs):
            transport.cancel_calls += 1
            raise ordinary

        monkeypatch.setattr(transport, "cancel", failed_cancel)
        released = await runtime.cancel_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        assert released.cleanup_state is PodCapacityCleanupState.COMPLETE
        assert transport.cancel_calls == 1
        assert provider.terminate_calls == ["pod-catalog-1"]
        return

    if boundary == "acknowledgement-conflict":
        await runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        await runtime.submit_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
            payload={
                "dispatch_attempt_id": value.attempt_id,
                "request_sha256": value.request_sha256,
            },
        )
        transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
        await runtime.observe_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        compare_and_set = runtime.repository.compare_and_set
        injected = False

        def conflict_once(current, *, changes):
            nonlocal injected
            if not injected and "artifact_handoff_json" in changes:
                injected = True
                raise PodCapacityConflictError("ordinary concurrent acknowledgement")
            return compare_and_set(current, changes=changes)

        monkeypatch.setattr(runtime.repository, "compare_and_set", conflict_once)
        result = await runtime.acknowledge_catalog_result(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
            acknowledgement=acknowledgement(),
        )
        assert result.workload_state is CatalogPodWorkloadState.ACKNOWLEDGED
        assert len(transport.acknowledgement_calls) == 1
        return

    if boundary == "billing-conflict":
        current = runtime.repository.compare_and_set(
            admitted,
            changes={
                "state": PodCapacityState.RELEASING,
                "cleanup_state": PodCapacityCleanupState.COMPLETE,
                "terminated_at": clock(),
            },
        )
        provider.billing_receipt = final_receipt(clock)
        transition_with_evidence = runtime._transition_with_evidence

        def persist_then_conflict(*args, **kwargs):
            transition_with_evidence(*args, **kwargs)
            raise PodCapacityConflictError("ordinary concurrent billing")

        monkeypatch.setattr(runtime, "_transition_with_evidence", persist_then_conflict)
        result = await runtime._reconcile_billing(current)
        assert result.state is PodCapacityState.RELEASED
        assert result.billing_state is PodCapacityBillingState.AUTHORITATIVE
        assert provider.billing_calls == ["pod-catalog-1"]
        return

    compare_and_set = runtime.repository.compare_and_set
    injected = False

    if boundary == "observation-conflict":
        ready = await runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
        assert ready.state is PodCapacityState.READY

    def conflict_once(current, *, changes):
        nonlocal injected
        if not injected:
            injected = True
            raise PodCapacityConflictError(f"ordinary {boundary}")
        return compare_and_set(current, changes=changes)

    monkeypatch.setattr(runtime.repository, "compare_and_set", conflict_once)
    if boundary == "evidence-conflict":
        result = runtime._transition_with_evidence(
            admitted,
            changes={"last_provider_error": "ordinary"},
            lifecycle={"worker_ready_at": clock()},
        )
        assert result.last_provider_error == "ordinary"
    else:
        observation = CatalogPodWorkloadObservation(
            attempt_id=value.attempt_id,
            request_sha256=value.request_sha256,
            state=CatalogPodWorkloadState.RUNNING,
            error_type=None,
            result_available=False,
        )
        result = runtime._record_catalog_observation(
            admitted.cleanup_token, observation
        )
        assert injected
        assert result.workload_state is CatalogPodWorkloadState.RUNNING


@pytest.mark.asyncio
async def test_provision_and_poll_share_lifecycle_without_waiting_or_recreating(
    tmp_path,
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.status_value = "PROVISIONING"
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)
    scope = {
        "capacity_id": value.capacity_id,
        "owner_id": value.owner_id,
        "workload_id": value.workload_id,
    }
    lease = await runtime.provision_catalog(value)
    assert lease.state is PodCapacityState.STARTING
    assert lease.provider_pod_id == "pod-catalog-1"
    assert lease.evidence.lifecycle.provider_create_accepted_at == clock()
    assert not transport.health_calls
    assert (await runtime.provision_catalog(value)).capacity_id == lease.capacity_id
    assert (
        await runtime.poll_catalog_readiness(**scope)
    ).state is PodCapacityState.STARTING
    provider.status_value = "RUNNING"
    ready = await runtime.poll_catalog_readiness(**scope)
    assert ready.state is PodCapacityState.READY
    assert ready.evidence.lifecycle.worker_ready_at == clock()
    assert len(provider.create_calls) == 1
    assert transport.health_calls == [provider.route]


@pytest.mark.asyncio
async def test_readiness_poll_validates_scope_and_never_revives_stop(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)
    scope = {
        "capacity_id": value.capacity_id,
        "owner_id": value.owner_id,
        "workload_id": value.workload_id,
    }
    await runtime.provision_catalog(value)
    for key in ("owner_id", "workload_id"):
        with pytest.raises(PodCapacityConflictError):
            await runtime.poll_catalog_readiness(**{**scope, key: "other"})
    runtime.request_cancellation(value.capacity_id)
    stopped = await runtime.poll_catalog_readiness(**scope)
    assert stopped.state is PodCapacityState.CANCEL_REQUESTED
    with pytest.raises(RunPodManagerError, match="not ready"):
        await runtime.submit_catalog_workload(**scope, payload={})
    assert not transport.submit_calls
    assert not transport.health_calls
    assert len(provider.create_calls) == 1
    await runtime.reconcile()
    assert provider.terminate_calls == ["pod-catalog-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [False, True])
async def test_cancellation_or_stop_during_health_cannot_promote_or_submit(
    tmp_path, stop
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    entered, finish = asyncio.Event(), asyncio.Event()

    class BlockingHealth(FakeWorkloadTransport):
        async def health(self, base_url):
            entered.set()
            await finish.wait()

    transport = BlockingHealth()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)
    await runtime.provision_catalog(value)
    task = asyncio.create_task(
        runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    if stop:
        runtime.request_cancellation(value.capacity_id)
        finish.set()
        assert (await task).state is PodCapacityState.CANCEL_REQUESTED
        await runtime.reconcile()
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert provider.terminate_calls == ["pod-catalog-1"]
    assert runtime.repository.get(value.capacity_id).state is not PodCapacityState.READY
    assert not transport.submit_calls


@pytest.mark.asyncio
async def test_readiness_timeout_terminates_without_replacement(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    runtime = service(
        tmp_path, clock, provider, FakeCapabilityStore(clock), FakeWorkloadTransport()
    )
    value = request(clock)
    await runtime.provision_catalog(value)
    clock.value += timedelta(seconds=31)
    with pytest.raises(PodCapacityLifecycleError, match="readiness timeout"):
        await runtime.poll_catalog_readiness(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
        )
    assert provider.terminate_calls == ["pod-catalog-1"]
    assert len(provider.create_calls) == 1


@pytest.mark.asyncio
async def test_stop_while_loading_capability_prevents_submission(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    entered, finish = asyncio.Event(), asyncio.Event()

    class BlockingCapabilityStore(FakeCapabilityStore):
        async def load(self, attempt_id):
            entered.set()
            await finish.wait()
            return await super().load(attempt_id)

    runtime = service(
        tmp_path, clock, provider, BlockingCapabilityStore(clock), transport
    )
    value = request(clock)
    await runtime.acquire_catalog(value)
    task = asyncio.create_task(
        runtime.submit_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
            payload={},
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    runtime.request_cancellation(value.capacity_id)
    finish.set()
    with pytest.raises(RunPodManagerError, match="not ready"):
        await task
    assert not transport.submit_calls


@pytest.mark.asyncio
async def test_readiness_cas_retry_does_not_overwrite_concurrent_stop(
    tmp_path, monkeypatch
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)
    await runtime.provision_catalog(value)
    original = runtime.repository.compare_and_set

    def racing_cas(lease, *, changes):
        if changes.get("state") is PodCapacityState.READY:
            runtime.request_cancellation(value.capacity_id)
        return original(lease, changes=changes)

    monkeypatch.setattr(runtime.repository, "compare_and_set", racing_cas)
    observed = await runtime.poll_catalog_readiness(
        capacity_id=value.capacity_id,
        owner_id=value.owner_id,
        workload_id=value.workload_id,
    )
    assert observed.state is PodCapacityState.CANCEL_REQUESTED
    assert observed.evidence.lifecycle.worker_ready_at is None


@pytest.mark.asyncio
async def test_unhealthy_poll_is_one_shot_and_health_cannot_cross_deadline(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)

    class ColdTransport(FakeWorkloadTransport):
        async def health(self, base_url):
            if not self.health_calls:
                self.health_calls.append(base_url)
                raise CatalogPodTransportError("worker is cold")
            clock.value += timedelta(seconds=31)

    transport = ColdTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)
    await runtime.provision_catalog(value)
    scope = {
        "capacity_id": value.capacity_id,
        "owner_id": value.owner_id,
        "workload_id": value.workload_id,
    }
    before = clock()
    assert (
        await runtime.poll_catalog_readiness(**scope)
    ).state is PodCapacityState.STARTING
    assert clock() == before  # The one-shot poll never sleeps/retries health.
    with pytest.raises(PodCapacityLifecycleError, match="readiness timeout"):
        await runtime.poll_catalog_readiness(**scope)
    assert provider.terminate_calls == ["pod-catalog-1"]


@pytest.mark.asyncio
async def test_capability_io_cannot_extend_creation_authorization(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)

    class SlowCapabilityStore(FakeCapabilityStore):
        async def load_or_create(self, attempt_id, expires_at):
            result = await super().load_or_create(attempt_id, expires_at)
            clock.value += timedelta(seconds=61)
            return result

    runtime = service(
        tmp_path, clock, provider, SlowCapabilityStore(clock), FakeWorkloadTransport()
    )
    with pytest.raises(RunPodManagerError, match="expired before creation"):
        await runtime.provision_catalog(request(clock))
    assert not provider.create_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("during_cas", [False, True])
async def test_late_workload_acceptance_cannot_revive_durable_stop(
    tmp_path, monkeypatch, during_cas
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    value = request(clock)

    class StoppingTransport(FakeWorkloadTransport):
        async def submit(self, **values):
            if not during_cas:
                runtime.request_cancellation(value.capacity_id)
            return await super().submit(**values)

    transport = StoppingTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    await runtime.acquire_catalog(value)
    if during_cas:
        original = runtime.repository.compare_and_set

        def racing_cas(lease, *, changes):
            if changes.get("workload_state") is CatalogPodWorkloadState.RUNNING:
                runtime.request_cancellation(value.capacity_id)
            return original(lease, changes=changes)

        monkeypatch.setattr(runtime.repository, "compare_and_set", racing_cas)
    await runtime.submit_catalog_workload(
        capacity_id=value.capacity_id,
        owner_id=value.owner_id,
        workload_id=value.workload_id,
        payload={},
    )
    current = runtime.repository.get(value.capacity_id)
    assert current.state is PodCapacityState.CANCEL_REQUESTED
    assert current.workload_state is CatalogPodWorkloadState.CANCEL_REQUESTED
    await runtime.reconcile()
    assert provider.terminate_calls == ["pod-catalog-1"]


@pytest.mark.asyncio
async def test_submission_hard_deadline_is_not_extended_by_cleanup_capability(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    value = request(clock)
    await runtime.acquire_catalog(value)
    clock.value = value.hard_deadline
    assert value.bearer_expires_at > clock()
    with pytest.raises(RunPodManagerError, match="hard deadline"):
        await runtime.submit_catalog_workload(
            capacity_id=value.capacity_id,
            owner_id=value.owner_id,
            workload_id=value.workload_id,
            payload={},
        )
    assert not transport.submit_calls


@pytest.mark.asyncio
async def test_acquire_injects_attempt_env_and_persists_no_secret(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)

    lease = await runtime.acquire_catalog(catalog_request)

    assert lease.state is PodCapacityState.READY
    assert transport.health_calls == [provider.route]
    environment = provider.create_calls[0]["environment"]
    assert environment == {
        "MODEL_REPOSITORY": "private/model",
        "CATALOG_WORKER_MODE": "pod",
        "CATALOG_POD_ATTEMPT_ID": catalog_request.attempt_id,
        "CATALOG_POD_BEARER_TOKEN": TOKEN,
        "CATALOG_POD_BEARER_EXPIRES_AT": (
            catalog_request.bearer_expires_at.isoformat()
        ),
        "CONTAINER_DIGEST": "sha256:" + "a" * 64,
        "CATALOG_POD_IMAGE_DIGEST": "sha256:" + "a" * 64,
        "CATALOG_POD_REQUEST_SHA256": catalog_request.request_sha256,
    }
    public = runtime.repository.get(catalog_request.capacity_id).to_public_dict()
    assert TOKEN not in str(public)
    assert "private/model" not in str(public)
    assert public["capacity"]["request_sha256"] == REQUEST_SHA


@pytest.mark.asyncio
async def test_legacy_protocol_cannot_authorize_new_paid_capacity(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    legacy = request(clock, artifact_handoff_protocol_version=None)
    with pytest.raises(
        RunPodManagerError, match="requires artifact handoff protocol 1"
    ):
        await runtime.provision_catalog(legacy)
    assert not provider.create_calls
    assert runtime.repository.get(legacy.capacity_id) is None


@pytest.mark.asyncio
async def test_historical_stored_capacity_cannot_authorize_a_new_submission(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, FakeCapabilityStore(clock), transport)
    legacy = request(clock, artifact_handoff_protocol_version=None)
    spec = PodCapacitySpec(
        request=legacy,
        capability_secret_id="secret:catalog-capability-0001",
        capability_token_sha256="a" * 64,
        capability_expires_at=legacy.bearer_expires_at,
        attempt_environment_sha256=attempt_environment_sha256(
            legacy.attempt_environment
        ),
    )
    old, _created = runtime.repository.reserve(
        TrainingPodRequest(
            cleanup_token=legacy.capacity_id,
            root_cleanup_token=legacy.cleanup_family_id,
            companion_id=legacy.owner_id,
            profile_id=legacy.profile_id,
            source=TrainingPodSource.CREATED,
            resource_name=legacy.resource_name,
            provider_pod_id=None,
            created_at=legacy.created_at,
            readiness_deadline=legacy.readiness_deadline,
            hard_deadline=legacy.hard_deadline,
            capacity_spec=spec,
        )
    )
    with pytest.raises(
        RunPodManagerError, match="Historical catalog capacity cannot accept"
    ):
        await runtime.submit_catalog_workload(
            capacity_id=legacy.capacity_id,
            owner_id=legacy.owner_id,
            workload_id=legacy.workload_id,
            payload={
                "dispatch_attempt_id": legacy.attempt_id,
                "request_sha256": legacy.request_sha256,
            },
        )
    assert runtime.repository.get(legacy.capacity_id) == old
    assert not transport.submit_calls
    assert not provider.create_calls


@pytest.mark.asyncio
async def test_identical_replay_adopts_ready_lease_but_changed_env_conflicts(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    original = request(clock)
    first = await runtime.acquire_catalog(original)

    replay = await runtime.acquire_catalog(original)
    assert replay.capacity_id == first.capacity_id
    assert len(provider.create_calls) == 1

    changed = replace(
        original,
        attempt_environment={"MODEL_REPOSITORY": "different/model"},
    )
    with pytest.raises(PodCapacityConflictError, match="different request"):
        await runtime.acquire_catalog(changed)
    assert len(provider.create_calls) == 1


@pytest.mark.asyncio
async def test_ambiguous_create_never_retries_and_delayed_visibility_is_adopted(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.create_error = RunPodAmbiguousResultError(
        title="timeout",
        detail="response lost",
        method="POST",
        resource="/pods",
    )
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)

    with pytest.raises(PodCapacityLifecycleError) as raised:
        await runtime.acquire_catalog(catalog_request)
    assert raised.value.billing_risk is True
    assert len(provider.create_calls) == 1
    assert provider.terminate_calls == []

    clock.value = clock.value.replace(second=31)
    await runtime.reconcile()
    unresolved = runtime.repository.get(catalog_request.capacity_id)
    assert unresolved is not None
    assert unresolved.creation_uncertain is True
    assert unresolved.billing_state is PodCapacityBillingState.UNRESOLVED
    assert len(provider.create_calls) == 1

    provider.find_result = "pod-catalog-1"
    await runtime.reconcile()
    terminated = runtime.repository.get(catalog_request.capacity_id)
    assert terminated is not None
    assert terminated.state is PodCapacityState.RELEASING
    assert terminated.billing_state is PodCapacityBillingState.PENDING
    assert provider.terminate_calls == ["pod-catalog-1"]
    persisted_rate = terminated.evidence.realized_placement.hourly_rate_usd
    provider.realized = replace(provider.realized, hourly_rate_usd=Decimal("0.39"))

    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    billed = runtime.repository.get(catalog_request.capacity_id)
    assert billed is not None
    assert billed.state is PodCapacityState.RELEASED
    assert billed.settlement_ready is True
    assert billed.billing_receipt.actual_cost_usd == Decimal("0.061")
    assert provider.billing_rates
    assert all(rate == persisted_rate for rate in provider.billing_rates)
    assert store.revoked == [catalog_request.attempt_id]


@pytest.mark.asyncio
async def test_known_created_mismatch_is_cleaned_up_but_billing_stays_unresolved(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.create_error = PodCapacityCreatedMismatchError("pod-mismatched-1")
    provider.billing_receipt = replace(
        final_receipt(clock), provider_pod_id="pod-mismatched-1"
    )
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)

    with pytest.raises(PodCapacityLifecycleError) as raised:
        await runtime.acquire_catalog(catalog_request)

    assert provider.billing_calls == []
    assert provider.billing_rates == []
    assert raised.value.billing_risk is False
    assert provider.terminate_calls == ["pod-mismatched-1"]
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable.provider_pod_id == "pod-mismatched-1"
    assert durable.state is PodCapacityState.RELEASING
    assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert durable.terminated_at is not None
    assert durable.billing_state is PodCapacityBillingState.UNRESOLVED
    assert durable.billing_receipt is None
    assert durable.settlement_ready is False
    assert durable.evidence.billing is None
    assert durable.evidence.lifecycle.billing_reconciled_at is None
    assert store.revoked == []
    restarted = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    await restarted.reconcile()
    durable = restarted.repository.get(catalog_request.capacity_id)
    assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert durable.billing_state is PodCapacityBillingState.UNRESOLVED
    assert durable.billing_receipt is None and not durable.settlement_ready
    assert provider.terminate_calls == ["pod-mismatched-1"]
    assert provider.billing_calls == provider.billing_rates == []


@pytest.mark.asyncio
async def test_missing_realized_placement_is_cleaned_up_but_billing_stays_unresolved(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.realized = None
    provider.billing_receipt = final_receipt(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)

    with pytest.raises(PodCapacityLifecycleError) as raised:
        await runtime.acquire_catalog(catalog_request)

    assert provider.billing_calls == []
    assert provider.billing_rates == []
    assert raised.value.billing_risk is False
    assert provider.terminate_calls == ["pod-catalog-1"]
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable is not None
    assert durable.provider_pod_id == "pod-catalog-1"
    assert durable.state is PodCapacityState.RELEASING
    assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert durable.terminated_at is not None
    assert durable.billing_state is PodCapacityBillingState.UNRESOLVED
    assert durable.billing_receipt is None
    assert durable.settlement_ready is False
    assert durable.evidence is not None
    assert durable.evidence.realized_placement is None
    assert durable.evidence.billing is None
    assert durable.evidence.lifecycle.billing_reconciled_at is None
    assert store.revoked == []
    restarted = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    await restarted.reconcile()
    durable = restarted.repository.get(catalog_request.capacity_id)
    assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert durable.billing_state is PodCapacityBillingState.UNRESOLVED
    assert durable.billing_receipt is None and not durable.settlement_ready
    assert provider.terminate_calls == ["pod-catalog-1"]
    assert provider.billing_calls == provider.billing_rates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("realized_rate", [Decimal(0), Decimal("0.38"), Decimal("0.4")])
async def test_billing_uses_exact_original_durable_realized_rate_after_restart(
    tmp_path, realized_rate
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.realized = replace(provider.realized, hourly_rate_usd=realized_rate)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)
    acquired = await runtime.acquire_catalog(catalog_request)
    assert acquired.evidence.realized_placement.hourly_rate_usd == realized_rate
    provider.realized = replace(provider.realized, hourly_rate_usd=Decimal("0.39"))
    provider.billing_receipt = final_receipt(clock)

    restarted = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    released = await restarted.release(catalog_request.capacity_id, reason="test")

    assert released.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert released.billing_state is PodCapacityBillingState.AUTHORITATIVE
    assert released.settlement_ready
    assert released.billing_receipt.hourly_price_usd == realized_rate
    assert provider.billing_rates == [realized_rate]
    assert released.billing_receipt.actual_cost_usd == Decimal("0.061")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_evidence", [False, True])
async def test_missing_or_foreign_durable_placement_cannot_authorize_billing(
    tmp_path, missing_evidence
):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)
    acquired = await runtime.acquire_catalog(catalog_request)
    evidence = (
        None
        if missing_evidence
        else replace(
            acquired.evidence,
            realized_placement=replace(
                acquired.evidence.realized_placement, provider_pod_id="other-pod"
            ),
        )
    )
    runtime.repository.compare_and_set(acquired, changes={"evidence_json": evidence})
    provider.billing_receipt = final_receipt(clock)
    restarted = service(tmp_path, clock, provider, store, FakeWorkloadTransport())

    unresolved = await restarted.release(catalog_request.capacity_id, reason="test")

    assert unresolved.provider_pod_id == "pod-catalog-1"
    assert unresolved.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert unresolved.terminated_at is not None
    assert unresolved.billing_state is PodCapacityBillingState.UNRESOLVED
    assert unresolved.billing_receipt is None and not unresolved.settlement_ready
    assert provider.terminate_calls == ["pod-catalog-1"]
    assert provider.billing_calls == provider.billing_rates == []


@pytest.mark.asyncio
async def test_submit_success_result_replays_until_ack_then_billing_reconciles(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    opaque = {
        "schema_version": 3,
        "dispatch_attempt_id": catalog_request.attempt_id,
        "request_sha256": catalog_request.request_sha256,
        "training": {"trigger_word": "private-value"},
    }

    submitted = await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload=opaque,
    )
    assert submitted.state is CatalogPodWorkloadState.RUNNING
    assert transport.submit_calls == [opaque]

    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    observed = await runtime.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert observed.result_available is True
    result = await runtime.retrieve_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert result == transport.result_payload
    replay = await runtime.retrieve_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert replay == result
    completed = runtime.repository.get(catalog_request.capacity_id)
    assert completed.state is PodCapacityState.JOB_COMPLETED
    assert provider.terminate_calls == []

    await runtime.acknowledge_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        acknowledgement=acknowledgement(),
    )
    assert not provider.terminate_calls
    await runtime.release_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        host_confirmed_acknowledgement_sha256=ACK_SHA,
    )
    pending = runtime.repository.get(catalog_request.capacity_id)
    assert pending.state is PodCapacityState.RELEASING
    assert pending.billing_state is PodCapacityBillingState.PENDING
    assert provider.terminate_calls == ["pod-catalog-1"]

    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    final = runtime.repository.get(catalog_request.capacity_id)
    assert final.settlement_ready is True
    assert final.billing_receipt.actual_cost_usd == Decimal("0.061")


@pytest.mark.asyncio
async def test_result_replays_after_host_restart_until_durable_ack(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    first_process = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await first_process.acquire_catalog(catalog_request)
    await first_process.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload={
            "dispatch_attempt_id": catalog_request.attempt_id,
            "request_sha256": catalog_request.request_sha256,
        },
    )
    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    await first_process.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    expected = await first_process.retrieve_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )

    restarted = service(tmp_path, clock, provider, store, transport)
    replay = await restarted.retrieve_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert replay == expected
    assert provider.terminate_calls == []

    await restarted.acknowledge_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        acknowledgement=acknowledgement(),
    )
    assert provider.terminate_calls == []
    await restarted.release_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        host_confirmed_acknowledgement_sha256=ACK_SHA,
    )
    assert provider.terminate_calls == ["pod-catalog-1"]


@pytest.mark.asyncio
async def test_restart_finishes_teardown_after_ack_transition_crash(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    first_process = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await first_process.acquire_catalog(catalog_request)
    await first_process.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload={
            "dispatch_attempt_id": catalog_request.attempt_id,
            "request_sha256": catalog_request.request_sha256,
        },
    )
    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    await first_process.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )

    async def interrupted_release(*_args, **_kwargs):
        raise asyncio.CancelledError

    await first_process.acknowledge_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        acknowledgement=acknowledgement(),
    )
    first_process.release = interrupted_release
    with pytest.raises(asyncio.CancelledError):
        await first_process.release_catalog_result(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
            host_confirmed_acknowledgement_sha256=ACK_SHA,
        )
    acknowledged = first_process.repository.get(catalog_request.capacity_id)
    assert acknowledged.state is PodCapacityState.RESULT_RETRIEVED
    assert provider.terminate_calls == []

    restarted = service(tmp_path, clock, provider, store, transport)
    await restarted.reconcile()
    assert provider.terminate_calls == ["pod-catalog-1"]


@pytest.mark.asyncio
async def test_cancellation_is_worker_authenticated_then_capacity_terminated(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload={
            "dispatch_attempt_id": catalog_request.attempt_id,
            "request_sha256": catalog_request.request_sha256,
        },
    )

    cancelled = await runtime.cancel_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert transport.cancel_calls == 1
    assert provider.terminate_calls == ["pod-catalog-1"]
    assert cancelled.workload_state is CatalogPodWorkloadState.CANCEL_REQUESTED
    assert cancelled.state is PodCapacityState.RELEASING


@pytest.mark.asyncio
async def test_owner_mismatch_fails_before_transport_or_provider_mutation(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)

    with pytest.raises(PodCapacityConflictError, match="owner/workload"):
        await runtime.submit_catalog_workload(
            capacity_id=catalog_request.capacity_id,
            owner_id="owner:different-user-0002",
            workload_id=catalog_request.workload_id,
            payload={},
        )
    assert transport.submit_calls == []


@pytest.mark.asyncio
async def test_get_catalog_capacity_is_owner_bound_and_reports_settlement(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)

    assert (
        runtime.find_catalog_capacity(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
        )
        is None
    )
    assert provider.create_calls == []
    with pytest.raises(RunPodManagerError, match="not found"):
        runtime.get_catalog_capacity(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
        )

    await runtime.acquire_catalog(catalog_request)

    active = runtime.get_catalog_capacity(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert active.settlement_ready is False
    assert active.billing_receipt is None
    assert active.backend_base_url not in str(active.to_public_dict())
    assert (
        runtime.find_catalog_capacity(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
        )
        == active
    )

    with pytest.raises(PodCapacityConflictError, match="owner/workload"):
        runtime.get_catalog_capacity(
            capacity_id=catalog_request.capacity_id,
            owner_id="owner:different-user-0002",
            workload_id=catalog_request.workload_id,
        )
    with pytest.raises(PodCapacityConflictError, match="owner/workload"):
        runtime.find_catalog_capacity(
            capacity_id=catalog_request.capacity_id,
            owner_id="owner:different-user-0002",
            workload_id=catalog_request.workload_id,
        )

    provider.billing_receipt = final_receipt(clock)
    await runtime.release(catalog_request.capacity_id, reason="test settlement")
    settled = runtime.get_catalog_capacity(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert settled.settlement_ready is True
    assert settled.billing_receipt.actual_cost_usd == Decimal("0.061")


@pytest.mark.asyncio
async def test_definitive_pre_provider_rejection_is_authoritative_zero(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.create_error = RunPodManagerError("capacity unavailable")
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)

    with pytest.raises(PodCapacityLifecycleError) as raised:
        await runtime.acquire_catalog(catalog_request)
    assert raised.value.billing_risk is False
    lease = runtime.repository.get(catalog_request.capacity_id)
    assert lease.settlement_ready is True
    assert lease.billing_receipt.actual_cost_usd == 0
    assert lease.billing_receipt.provider_pod_id is None
    assert store.revoked == [catalog_request.attempt_id]


@pytest.mark.asyncio
async def test_acknowledged_invalid_create_retains_exact_pod_custody_and_cost_unknown(
    tmp_path,
) -> None:
    """A malformed optional sibling cannot erase a successful POST identity."""

    clock = MutableClock()

    class AcknowledgingControlClient:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str]] = []
            self.reads: list[str] = []

        def list_gpus(self, **_kwargs):
            return (
                GPUOffer(
                    id="NVIDIA RTX PRO 4500",
                    name="RTX PRO 4500",
                    pool=None,
                    manufacturer="NVIDIA",
                    memory_gb=32,
                    secure=True,
                    community=False,
                    secure_price_per_hr=0.4,
                    community_price_per_hr=0.0,
                    secure_max_count=8,
                    community_max_count=0,
                    availability=Availability.HIGH,
                    data_centers=({"id": "US-TX-3", "availability": "HIGH"},),
                    availability_min_cuda_version="12.8",
                ),
            )

        def create_pod(self, create_request):
            return PodResource.from_dict(
                {
                    "id": "pod-acknowledged-invalid-cost",
                    "name": create_request.name,
                    "status": "PROVISIONING",
                    "gpu": {
                        "id": create_request.gpu_id,
                        "count": create_request.gpu_count,
                    },
                    "cost": True,
                    "cloud": CloudType.SECURE.value,
                    "dataCenterId": "US-TX-3",
                    "image": create_request.image,
                    "env": dict(create_request.env),
                }
            )

        def pod_action(self, pod_id, action):
            self.actions.append((pod_id, action))

        def get_pod(self, pod_id):
            self.reads.append(pod_id)
            raise RunPodAPIError(
                title="Not found",
                detail="Exact acknowledged Pod is absent",
                status_code=404,
                method="GET",
                resource=f"/pods/{pod_id}",
            )

    client = AcknowledgingControlClient()
    provider = RunpodPodCapacityProvider(
        DirectRunPodProvider(api_key="", client=client), clock=clock
    )
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)

    with pytest.raises(PodCapacityLifecycleError):
        await runtime.acquire_catalog(catalog_request)

    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable is not None
    assert durable.provider_pod_id == "pod-acknowledged-invalid-cost"
    assert durable.creation_uncertain is False
    assert durable.cleanup_state is PodCapacityCleanupState.COMPLETE
    assert durable.billing_state is PodCapacityBillingState.UNRESOLVED
    assert durable.settlement_ready is False
    assert durable.billing_receipt is None
    assert client.actions == [("pod-acknowledged-invalid-cost", "terminate")]
    assert client.reads == ["pod-acknowledged-invalid-cost"]


@pytest.mark.asyncio
async def test_result_transport_interruption_preserves_retryable_success(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)

    class InterruptedResultTransport(FakeWorkloadTransport):
        async def result(self, **values):
            raise CatalogPodTransportError("result transport failed")

    transport = InterruptedResultTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload={
            "dispatch_attempt_id": catalog_request.attempt_id,
            "request_sha256": catalog_request.request_sha256,
        },
    )
    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    await runtime.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )

    with pytest.raises(CatalogPodTransportError):
        await runtime.retrieve_catalog_result(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
        )
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable.workload_state is CatalogPodWorkloadState.AWAITING_ACK
    assert durable.state is PodCapacityState.JOB_COMPLETED
    assert provider.terminate_calls == []


@pytest.mark.asyncio
async def test_failed_workload_terminates_and_retains_actual_cost(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload={
            "dispatch_attempt_id": catalog_request.attempt_id,
            "request_sha256": catalog_request.request_sha256,
        },
    )
    provider.billing_receipt = final_receipt(clock)
    transport.status_value = CatalogPodWorkloadState.FAILED

    observed = await runtime.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    final = runtime.repository.get(catalog_request.capacity_id)
    assert observed.error_type == "CatalogExecutionError"
    assert final.state is PodCapacityState.RELEASED
    assert final.workload_state is CatalogPodWorkloadState.FAILED
    assert final.billing_receipt.actual_cost_usd == Decimal("0.061")


@pytest.mark.asyncio
async def test_restart_requires_same_durable_capability_without_reprovisioning(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)

    missing_store = FakeCapabilityStore(clock)
    restarted = service(tmp_path, clock, provider, missing_store, transport)
    with pytest.raises(RunPodManagerError, match="capability is unavailable"):
        await restarted.submit_catalog_workload(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
            payload={
                "dispatch_attempt_id": catalog_request.attempt_id,
                "request_sha256": catalog_request.request_sha256,
            },
        )
    assert len(provider.create_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_release_and_reconcile_converge_without_reviving_capacity(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)

    await asyncio.gather(
        runtime.release(catalog_request.capacity_id, reason="caller release"),
        runtime.reconcile(),
    )
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable.state is PodCapacityState.RELEASING
    assert durable.billing_state is PodCapacityBillingState.PENDING
    assert durable.cleanup_state.value == "complete"
    assert set(provider.terminate_calls) == {"pod-catalog-1"}

    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    assert runtime.repository.get(catalog_request.capacity_id).settlement_ready is True


@pytest.mark.asyncio
async def test_capability_revoke_failure_keeps_billed_lease_nonterminal(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    provider.billing_receipt = final_receipt(clock)
    store.revoke_error = RunPodManagerError("secret store unavailable")

    with pytest.raises(RunPodManagerError, match="Reconcile cleanup token"):
        await runtime.release(catalog_request.capacity_id, reason="test")
    pending = runtime.repository.get(catalog_request.capacity_id)
    assert pending.state is PodCapacityState.RELEASING
    assert pending.billing_state is PodCapacityBillingState.AUTHORITATIVE
    assert pending.billing_receipt.actual_cost_usd == Decimal("0.061")

    store.revoke_error = None
    await runtime.reconcile()
    final = runtime.repository.get(catalog_request.capacity_id)
    assert final.settlement_ready is True
    assert store.revoked == [catalog_request.attempt_id]


@pytest.mark.asyncio
async def test_mismatched_provider_billing_identity_never_settles(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, FakeWorkloadTransport())
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    provider.billing_receipt = replace(
        final_receipt(clock), provider_pod_id="pod-unrelated-1"
    )

    with pytest.raises(PodCapacityLifecycleError, match="Reconcile cleanup token"):
        await runtime.release(catalog_request.capacity_id, reason="test")

    pending = runtime.repository.get(catalog_request.capacity_id)
    assert pending.state is PodCapacityState.RELEASING
    assert pending.billing_state is PodCapacityBillingState.PENDING
    assert pending.billing_receipt is None

    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    assert runtime.repository.get(catalog_request.capacity_id).settlement_ready is True
