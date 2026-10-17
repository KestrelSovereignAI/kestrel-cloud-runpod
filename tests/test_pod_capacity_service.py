"""End-to-end durable catalog Pod lifecycle tests without live resources."""

import asyncio
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from pod_capacity_test_support import (
    REQUEST_SHA,
    TOKEN,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    final_receipt,
    request,
    service,
)

from kestrel_cloud_runpod.models import (
    RunPodAmbiguousResultError,
    RunPodManagerError,
)
from kestrel_cloud_runpod.pod_capacity_contracts import (
    CatalogPodWorkloadState,
    PodCapacityBillingState,
    PodCapacityConflictError,
    PodCapacityLifecycleError,
    PodCapacityState,
)
from kestrel_cloud_runpod.pod_capacity_provider import (
    PodCapacityCreatedMismatchError,
)
from kestrel_cloud_runpod.pod_transport import CatalogPodTransportError


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
    scope = dict(
        capacity_id=value.capacity_id,
        owner_id=value.owner_id,
        workload_id=value.workload_id,
    )
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
    scope = dict(
        capacity_id=value.capacity_id,
        owner_id=value.owner_id,
        workload_id=value.workload_id,
    )
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
    scope = dict(
        capacity_id=value.capacity_id,
        owner_id=value.owner_id,
        workload_id=value.workload_id,
    )
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
    }
    public = runtime.repository.get(catalog_request.capacity_id).to_public_dict()
    assert TOKEN not in str(public)
    assert "private/model" not in str(public)
    assert public["capacity"]["request_sha256"] == REQUEST_SHA


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

    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    billed = runtime.repository.get(catalog_request.capacity_id)
    assert billed is not None
    assert billed.state is PodCapacityState.RELEASED
    assert billed.settlement_ready is True
    assert billed.billing_receipt.actual_cost_usd == Decimal("0.061")
    assert store.revoked == [catalog_request.attempt_id]


@pytest.mark.asyncio
async def test_known_created_mismatch_is_durably_terminated_and_billed(
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

    assert raised.value.billing_risk is False
    assert provider.terminate_calls == ["pod-mismatched-1"]
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable.provider_pod_id == "pod-mismatched-1"
    assert durable.settlement_ready is True
    assert durable.billing_receipt.provider_pod_id == "pod-mismatched-1"
    assert store.revoked == [catalog_request.attempt_id]


@pytest.mark.asyncio
async def test_missing_realized_placement_is_durably_terminated_and_billed(
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

    assert raised.value.billing_risk is False
    assert provider.terminate_calls == ["pod-catalog-1"]
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable is not None
    assert durable.provider_pod_id == "pod-catalog-1"
    assert durable.settlement_ready is True
    assert durable.evidence is not None
    assert durable.evidence.realized_placement is None
    assert store.revoked == [catalog_request.attempt_id]


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

    transport.status_value = CatalogPodWorkloadState.SUCCEEDED
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
    transport.status_value = CatalogPodWorkloadState.SUCCEEDED
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
    transport.status_value = CatalogPodWorkloadState.SUCCEEDED
    await first_process.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )

    async def interrupted_release(*_args, **_kwargs):
        raise asyncio.CancelledError

    first_process.release = interrupted_release
    with pytest.raises(asyncio.CancelledError):
        await first_process.acknowledge_catalog_result(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
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
    transport.status_value = CatalogPodWorkloadState.SUCCEEDED
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
    assert durable.workload_state is CatalogPodWorkloadState.SUCCEEDED
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
