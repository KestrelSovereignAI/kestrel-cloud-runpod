"""Restart/cancellation/Stop safety of the public opaque artifact handoff."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import pytest
from pod_capacity_test_support import (
    ACK_SHA,
    RECEIPT_SHA,
    REQUEST_SHA,
    RESULT_SHA,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    acknowledgement,
    final_receipt,
    request,
    service,
)

from kestrel_cloud_runpod import CatalogArtifactHandoffEvidence
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.pod_capacity_contracts import (
    CatalogPodWorkloadState,
    PodCapacitySpec,
    TrainingPodConflictError,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
    attempt_environment_sha256,
)
from kestrel_cloud_runpod.pod_transport import (
    CatalogPodTransportError,
    CatalogPodWorkloadObservation,
)


async def pending(tmp_path, transport=None, store=None):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    transport = transport or FakeWorkloadTransport()
    store = store or FakeCapabilityStore(clock)
    runtime = service(tmp_path, clock, provider, store, transport)
    value = request(clock)
    scope = {
        "capacity_id": value.capacity_id,
        "owner_id": value.owner_id,
        "workload_id": value.workload_id,
    }
    await runtime.acquire_catalog(value)
    await runtime.submit_catalog_workload(
        **scope,
        payload={
            "dispatch_attempt_id": value.attempt_id,
            "request_sha256": value.request_sha256,
        },
    )
    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    await runtime.observe_catalog_workload(**scope)
    return runtime, scope, clock, provider, store, transport


def test_evidence_is_content_free_strict_and_canonical():
    clock = MutableClock()
    evidence = CatalogArtifactHandoffEvidence(
        "attempt:catalog-run-0001", REQUEST_SHA, RESULT_SHA, RECEIPT_SHA
    )
    intent = evidence.bind_acknowledgement(acknowledgement())
    assert (
        intent.bind_acknowledgement(dict(reversed(list(acknowledgement().items()))))
        == intent
    )
    assert CatalogArtifactHandoffEvidence.from_dict(intent.to_dict()) == intent
    assert "not-public-or-persisted-by-sdk" not in json.dumps(intent.to_dict())
    for changes in (
        {"acknowledgement_payload_sha256": "a" * 64},
        {"remote_ack_observed_at": clock()},
        {"host_confirmation_declared_at": clock()},
        {"result_sha256": True},
    ):
        with pytest.raises(ValueError):
            replace(evidence, **changes)
    with pytest.raises(RunPodManagerError):
        CatalogArtifactHandoffEvidence.from_dict({**intent.to_dict(), "private": "no"})
    with pytest.raises(ValueError):
        evidence.bind_acknowledgement({**acknowledgement(), "opaque": float("nan")})


@pytest.mark.asyncio
async def test_exact_ack_intent_and_observed_proof_survive_restart_and_revocation(
    tmp_path,
):
    runtime, scope, clock, provider, store, transport = await pending(tmp_path)
    first = await runtime.acknowledge_catalog_result(
        **scope, acknowledgement=acknowledgement()
    )
    assert first.state is TrainingPodState.JOB_COMPLETED
    assert first.artifact_handoff.remote_ack_observed_at == clock()
    assert not provider.terminate_calls
    await store.revoke(first.capacity_spec.request.attempt_id)
    clock.value += timedelta(days=1)
    restarted = service(tmp_path, clock, provider, store, transport)
    assert (
        await restarted.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
        == first
    )
    assert len(transport.acknowledgement_calls) == 1
    for changed in (
        {
            **acknowledgement(),
            "opaque_private_signature": "changed-but-same-claimed-digest",
        },
        {**acknowledgement(), "receipt_sha256": "a" * 64},
    ):
        with pytest.raises(TrainingPodConflictError):
            await restarted.acknowledge_catalog_result(**scope, acknowledgement=changed)
    for changed_scope in (
        {**scope, "owner_id": "other"},
        {**scope, "workload_id": "other"},
    ):
        with pytest.raises(TrainingPodConflictError):
            await restarted.acknowledge_catalog_result(
                **changed_scope, acknowledgement=acknowledgement()
            )
    with pytest.raises(TrainingPodConflictError):
        await restarted.release_catalog_result(
            **scope, host_confirmed_acknowledgement_sha256="a" * 64
        )
    assert not provider.terminate_calls
    provider.billing_receipt = final_receipt(clock)
    released = await restarted.release_catalog_result(
        **scope, host_confirmed_acknowledgement_sha256=ACK_SHA
    )
    assert released.is_terminal
    assert released.artifact_handoff.host_confirmation_declared_at == clock()
    assert (
        await restarted.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
        == released
    )
    with pytest.raises(TrainingPodConflictError):
        await restarted.release_catalog_result(
            **scope, host_confirmed_acknowledgement_sha256="a" * 64
        )
    assert len(provider.terminate_calls) == 1


@pytest.mark.asyncio
async def test_ack_timeout_preserves_job_and_exact_intent_until_authenticated_recovery(
    tmp_path,
):
    class LostAckResponse(FakeWorkloadTransport):
        async def acknowledge(self, **values):
            await super().acknowledge(**values)
            raise CatalogPodTransportError("response lost after remote acceptance")

    runtime, scope, clock, provider, store, transport = await pending(
        tmp_path, LostAckResponse()
    )
    with pytest.raises(CatalogPodTransportError):
        await runtime.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
    original = runtime.repository.get(scope["capacity_id"])
    assert original.state is TrainingPodState.JOB_COMPLETED
    assert original.artifact_handoff.acknowledgement_sha256 == ACK_SHA
    assert original.artifact_handoff.remote_ack_observed_at is None
    await runtime.reconcile()
    assert not provider.terminate_calls
    assert (
        runtime.repository.get(
            scope["capacity_id"]
        ).artifact_handoff.remote_ack_observed_at
        is None
    )
    restarted = service(tmp_path, clock, provider, store, transport)
    await restarted.observe_catalog_workload(**scope)
    observed = restarted.repository.get(scope["capacity_id"])
    assert observed.artifact_handoff.remote_ack_observed_at == clock()
    assert (
        await restarted.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
        == observed
    )
    assert len(transport.acknowledgement_calls) == 1
    assert not provider.terminate_calls


@pytest.mark.asyncio
async def test_crash_after_remote_ack_before_local_receipt_is_recoverable(tmp_path):
    runtime, scope, clock, provider, store, transport = await pending(tmp_path)
    original_record = runtime._record_catalog_observation

    def interrupted_record(*args, **kwargs):
        raise OSError("simulated durable-state write failure")

    runtime._record_catalog_observation = interrupted_record
    with pytest.raises(OSError):
        await runtime.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
    assert (
        runtime.repository.get(
            scope["capacity_id"]
        ).artifact_handoff.remote_ack_observed_at
        is None
    )
    assert not provider.terminate_calls
    runtime._record_catalog_observation = original_record
    restarted = service(tmp_path, clock, provider, store, transport)
    await restarted.observe_catalog_workload(**scope)
    assert (
        restarted.repository.get(
            scope["capacity_id"]
        ).artifact_handoff.remote_ack_observed_at
        == clock()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [False, True])
async def test_ack_tail_remains_owned_through_repeated_cancellation_and_stop(
    tmp_path, stop
):
    entered, finish = asyncio.Event(), asyncio.Event()

    class BlockingAck(FakeWorkloadTransport):
        async def acknowledge(self, **values):
            entered.set()
            await finish.wait()
            return await super().acknowledge(**values)

    runtime, scope, clock, provider, _store, transport = await pending(
        tmp_path, BlockingAck()
    )
    task = asyncio.create_task(
        runtime.acknowledge_catalog_result(**scope, acknowledgement=acknowledgement())
    )
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    if stop:
        runtime.request_cancellation(scope["capacity_id"])
        await runtime.reconcile()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.artifact_handoff.remote_ack_observed_at == clock()
    assert len(transport.acknowledgement_calls) == 1
    assert saved.state is (
        TrainingPodState.RELEASING if stop else TrainingPodState.JOB_COMPLETED
    )
    assert saved.workload_state is (
        CatalogPodWorkloadState.CANCEL_REQUESTED
        if stop
        else CatalogPodWorkloadState.ACKNOWLEDGED
    )
    assert saved.artifact_handoff.host_confirmation_declared_at is None
    assert len(provider.terminate_calls) == int(stop)


@pytest.mark.asyncio
async def test_stop_during_capability_load_prevents_ack_dispatch(tmp_path):
    runtime, scope, _clock, provider, store, transport = await pending(tmp_path)
    load = store.load

    async def stop_then_load(attempt_id):
        runtime.request_cancellation(scope["capacity_id"])
        return await load(attempt_id)

    store.load = stop_then_load
    with pytest.raises(RunPodManagerError, match="stopped before dispatch"):
        await runtime.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
    assert not transport.acknowledgement_calls
    assert not provider.terminate_calls


@pytest.mark.asyncio
async def test_status_recovery_preserves_ack_fact_when_stop_wins_in_flight(tmp_path):
    entered, finish = asyncio.Event(), asyncio.Event()

    class RecoveringAck(FakeWorkloadTransport):
        block_status = False

        async def acknowledge(self, **values):
            await super().acknowledge(**values)
            raise CatalogPodTransportError("lost remote ACK response")

        async def status(self, **values):
            observation = await super().status(**values)
            if self.block_status:
                entered.set()
                await finish.wait()
            return observation

    runtime, scope, clock, provider, store, transport = await pending(
        tmp_path, RecoveringAck()
    )
    with pytest.raises(CatalogPodTransportError):
        await runtime.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
    transport.block_status = True
    task = asyncio.create_task(runtime.observe_catalog_workload(**scope))
    await asyncio.wait_for(entered.wait(), 5)
    provider.billing_receipt = final_receipt(clock)
    runtime.request_cancellation(scope["capacity_id"])
    finish.set()
    outward = await task
    assert outward.state is CatalogPodWorkloadState.CANCEL_REQUESTED
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.is_terminal
    assert saved.workload_state is CatalogPodWorkloadState.CANCEL_REQUESTED
    assert saved.artifact_handoff.remote_ack_observed_at == clock()
    assert saved.artifact_handoff.host_confirmation_declared_at is None
    assert saved.terminal_success_evidence_complete is False
    assert store.revoked == [saved.capacity_spec.request.attempt_id]
    restarted = service(tmp_path, clock, provider, store, transport)
    assert (
        await restarted.acknowledge_catalog_result(
            **scope, acknowledgement=acknowledgement()
        )
        == saved
    )
    assert len(transport.acknowledgement_calls) == 1
    assert len(provider.terminate_calls) == 1


@pytest.mark.asyncio
async def test_unbound_or_changed_remote_hashes_cannot_authorize_cleanup(tmp_path):
    runtime, scope, _clock, provider, _store, transport = await pending(tmp_path)
    transport.status_value = CatalogPodWorkloadState.ACKNOWLEDGED
    with pytest.raises(TrainingPodConflictError, match="persisted host intent"):
        await runtime.observe_catalog_workload(**scope)
    with pytest.raises(TrainingPodConflictError):
        await runtime.release_catalog_result(
            **scope, host_confirmed_acknowledgement_sha256=ACK_SHA
        )
    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    transport.result_payload = {
        **transport.result_payload,
        "receipt": {**transport.result_payload["receipt"], "receipt_sha256": "a" * 64},
    }
    with pytest.raises(TrainingPodConflictError, match="differs"):
        await runtime.retrieve_catalog_result(**scope)
    assert not provider.terminate_calls


@pytest.mark.asyncio
async def test_stale_phase_and_cas_retry_cannot_regress_acknowledgement(tmp_path):
    runtime, scope, _clock, _provider, _store, transport = await pending(tmp_path)
    before = runtime.repository.get(scope["capacity_id"])
    stale = CatalogPodWorkloadObservation(
        before.capacity_spec.request.attempt_id,
        REQUEST_SHA,
        CatalogPodWorkloadState.RUNNING,
        None,
        False,
    )
    cas = runtime.repository.compare_and_set
    cas(
        before,
        changes={
            "workload_state": CatalogPodWorkloadState.RUNNING,
            "artifact_handoff_json": None,
            "state": TrainingPodState.JOB_SUBMITTED,
        },
    )
    awaiting = await transport.status(
        attempt_id=before.capacity_spec.request.attempt_id, request_sha256=REQUEST_SHA
    )
    raced = False

    def racing_cas(lease, *, changes):
        nonlocal raced
        if (
            not raced
            and changes.get("workload_state") is CatalogPodWorkloadState.AWAITING_ACK
        ):
            raced = True
            intent = before.artifact_handoff.bind_acknowledgement(acknowledgement())
            cas(
                lease,
                changes={
                    "workload_state": CatalogPodWorkloadState.ACKNOWLEDGED,
                    "artifact_handoff_json": replace(
                        intent, remote_ack_observed_at=before.updated_at
                    ),
                },
            )
        return cas(lease, changes=changes)

    runtime.repository.compare_and_set = racing_cas
    saved = runtime._record_catalog_observation(scope["capacity_id"], awaiting)
    assert saved.workload_state is CatalogPodWorkloadState.ACKNOWLEDGED
    assert raced
    runtime.repository.compare_and_set = cas
    await runtime.acknowledge_catalog_result(**scope, acknowledgement=acknowledgement())
    transport.status_value = CatalogPodWorkloadState.AWAITING_ACK
    await runtime.observe_catalog_workload(**scope)
    saved = runtime._record_catalog_observation(scope["capacity_id"], stale)
    assert saved.workload_state is CatalogPodWorkloadState.ACKNOWLEDGED


@pytest.mark.asyncio
async def test_late_ack_failure_wins_caller_cancellation_without_losing_intent(
    tmp_path,
):
    entered, finish = asyncio.Event(), asyncio.Event()

    class LateFailure(FakeWorkloadTransport):
        async def acknowledge(self, **values):
            entered.set()
            await finish.wait()
            raise TypeError("implementation failure remains visible")

    runtime, scope, _clock, provider, _store, _transport = await pending(
        tmp_path, LateFailure()
    )
    task = asyncio.create_task(
        runtime.acknowledge_catalog_result(**scope, acknowledgement=acknowledgement())
    )
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    finish.set()
    with pytest.raises(TypeError, match="implementation failure"):
        await task
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.state is TrainingPodState.JOB_COMPLETED
    assert saved.artifact_handoff.acknowledgement_sha256 == ACK_SHA
    assert saved.artifact_handoff.remote_ack_observed_at is None
    assert not provider.terminate_calls


@pytest.mark.asyncio
async def test_hard_deadline_cleanup_never_infers_ack_or_host_confirmation(tmp_path):
    runtime, scope, clock, provider, _store, _transport = await pending(tmp_path)
    value = runtime.repository.get(scope["capacity_id"])
    clock.value = value.hard_deadline
    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    result = runtime.repository.get(scope["capacity_id"])
    assert result.is_terminal
    assert result.artifact_handoff.remote_ack_observed_at is None
    assert result.artifact_handoff.host_confirmation_declared_at is None
    assert result.terminal_success_evidence_complete is False
    with pytest.raises(TrainingPodConflictError):
        await runtime.release_catalog_result(
            **scope, host_confirmed_acknowledgement_sha256=ACK_SHA
        )


@pytest.mark.asyncio
async def test_generic_catalog_acquisition_cannot_bypass_new_work_protocol(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    value = request(clock, artifact_handoff_protocol_version=None)
    spec = PodCapacitySpec(
        value,
        "secret:catalog-capability-0001",
        "a" * 64,
        value.bearer_expires_at,
        attempt_environment_sha256(value.attempt_environment),
    )
    training = TrainingPodRequest(
        value.capacity_id,
        value.owner_id,
        value.profile_id,
        TrainingPodSource.CREATED,
        value.resource_name,
        None,
        value.created_at,
        value.readiness_deadline,
        value.hard_deadline,
        capacity_spec=spec,
    )
    with pytest.raises(RunPodManagerError, match="Historical catalog capacity"):
        await runtime.acquire(training)
    assert runtime.repository.get(value.capacity_id) is None
    assert not provider.create_calls
    assert not store.capabilities


@pytest.mark.asyncio
async def test_historical_owned_lease_remains_readable_and_cleanup_compatible(tmp_path):
    runtime, scope, _clock, provider, _store, transport = await pending(tmp_path)
    current = runtime.repository.get(scope["capacity_id"])
    # Pre-upgrade SQLite representation: no handoff protocol field or evidence.
    spec = current.capacity_spec.to_dict()
    spec["request"].pop("artifact_handoff_protocol_version")
    with runtime.repository._connect() as connection:
        connection.execute(
            "UPDATE pod_capacity_leases SET capacity_spec_json=?, artifact_handoff_json=NULL, workload_state='succeeded' WHERE cleanup_token=?",
            (json.dumps(spec), current.cleanup_token),
        )
    transport.result_payload = {
        "dispatch_attempt_id": current.capacity_spec.request.attempt_id,
        "request_sha256": REQUEST_SHA,
        "private": "opaque-legacy-result",
    }
    assert await runtime.retrieve_catalog_result(**scope) == transport.result_payload
    await runtime.acknowledge_catalog_result(**scope)
    assert provider.terminate_calls == ["pod-catalog-1"]
    assert not transport.acknowledgement_calls


@pytest.mark.asyncio
async def test_generic_training_mutators_cannot_bypass_catalog_handoff(tmp_path):
    runtime, scope, _clock, provider, _store, _transport = await pending(tmp_path)
    for mutation, args in (
        (runtime.record_job, ("other-job",)),
        (runtime.record_status, ("completed",)),
        (runtime.record_result_retrieved, ()),
    ):
        with pytest.raises(RunPodManagerError, match="authenticated catalog lifecycle"):
            mutation(scope["capacity_id"], *args)
    await runtime.reconcile()
    assert not provider.terminate_calls
