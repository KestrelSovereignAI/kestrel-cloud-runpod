"""Original Ollama effects must physically finish before cancellation escapes."""

import asyncio
import sqlite3
import threading
from contextlib import closing
from dataclasses import replace

import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    release_gates_and_boundedly_join_owned_tasks,
    serverless_plan,
)

from kestrel_cloud_runpod.models import (
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaProvisioningEvidenceError,
    OllamaProvisioningMismatchError,
    OllamaProvisioningMismatchEvidence,
    OllamaProvisioningMismatchReason,
    OllamaTeardownState,
)
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService


class ThreadedProvider(FakeOllamaProvider):
    def __init__(self, *, block):
        super().__init__(serverless_plan())
        self.block = block
        self.entered = threading.Event()
        self.finish = threading.Event()
        self.completed = threading.Event()
        self.confirmed_mismatch = False
        self.mismatched_resource_ids: set[str] = set()
        self.unavailable_resource_ids: set[str] = set()
        self.observe_calls = 0

    def physical_effect(self):
        self.entered.set()
        assert self.finish.wait(5), "test did not release the original provider thread"
        self.completed.set()

    async def provision(self, prepared):
        self.resource = replace(
            self.resource, resource_name=prepared.intent.resource_name
        )
        self.provision_calls += 1
        if self.block == "create":
            await asyncio.to_thread(self.physical_effect)
        return self.resource

    async def observe_provisioning(self, intent, resource, **kwargs):
        self.observe_calls += 1
        if self.block == "readback":
            await asyncio.to_thread(self.physical_effect)
        if (
            self.confirmed_mismatch
            or resource.provider_resource_id in self.mismatched_resource_ids
        ):
            raise OllamaProvisioningMismatchError(
                "Ollama provider confirmed an incompatible runtime",
                OllamaProvisioningMismatchEvidence(
                    intent_sha256=intent.intent_sha256,
                    lease_id=intent.lease_id,
                    provision_attempt_id=intent.provision_attempt_id,
                    resource_type=intent.resource_type,
                    provider_resource_id=resource.provider_resource_id,
                    observed_resource_name=resource.resource_name,
                    observed_at=intent.provision_requested_at,
                    readback_sha256="c" * 64,
                    reason=OllamaProvisioningMismatchReason.RUNTIME,
                ),
            )
        if resource.provider_resource_id in self.unavailable_resource_ids:
            raise OllamaProvisioningEvidenceError(
                "candidate has incomplete original provisioning evidence"
            )
        return await super().observe_provisioning(intent, resource, **kwargs)

    async def teardown(self, resource, cleanup_intent):
        assert resource == self.resource
        if self.block == "delete":
            await asyncio.to_thread(self.physical_effect)
        return await super().teardown(resource, cleanup_intent)


def service(tmp_path, provider, clock):
    return OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("effect", ["create", "readback", "delete"])
async def test_cancellation_joins_original_thread_and_durable_receipt(tmp_path, effect):
    clock = MutableClock()
    provider = ThreadedProvider(block=effect)
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    if effect in {"create", "readback"}:
        task = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    else:
        ready = await owner.acquire(request)
        task = asyncio.create_task(
            owner.release(
                ready.lease_id,
                owner_id=ready.owner_id,
                workload_id=ready.workload_id,
            )
        )
    try:
        assert await asyncio.to_thread(provider.entered.wait, 5)
        for _ in range(4):
            task.cancel("original caller stopped")
            await asyncio.sleep(0)
            assert not task.done(), (
                "caller escaped while original provider thread is live"
            )
    finally:
        await release_gates_and_boundedly_join_owned_tasks(task, provider.finish)
        # Always collect the actual task, including when the RED assertion fails.
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(provider.completed.wait, 5)

    durable = owner.repository.get(request.lease_id)
    assert durable.provider_resource_id == provider.resource.provider_resource_id
    assert durable.creation_uncertain is False
    if effect in {"create", "readback"}:
        assert durable.state is OllamaLeaseState.WAITING_FOR_MODEL
    else:
        assert durable.state is OllamaLeaseState.TERMINATED
        assert durable.teardown_state is OllamaTeardownState.COMPLETE
        assert durable.cleanup_intent is not None
        assert durable.cleanup_receipt is not None
        assert durable.cleanup_receipt.cleanup_intent_id == (
            durable.cleanup_intent.cleanup_intent_id
        )
        assert durable.cleanup_receipt.verified_absent_at >= (
            durable.cleanup_intent.delete_started_at
        )


@pytest.mark.asyncio
async def test_release_during_post_cannot_discard_late_exact_acknowledgment(tmp_path):
    clock = MutableClock()
    provider = ThreadedProvider(block="create")
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    acquire = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        assert await asyncio.to_thread(provider.entered.wait, 5)
        with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
            await owner.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
    finally:
        await release_gates_and_boundedly_join_owned_tasks(acquire, provider.finish)

    completed = await acquire
    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.provider_resource_id == provider.resource.provider_resource_id
    assert completed.creation_uncertain is False
    assert completed.teardown_state is OllamaTeardownState.COMPLETE
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_late_exact_ack_does_not_replace_a_different_untrusted_locator(tmp_path):
    clock = MutableClock()
    provider = ThreadedProvider(block="create")
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    acquire = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        assert await asyncio.to_thread(provider.entered.wait, 5)
        candidate = replace(
            provider.resource, provider_resource_id="name-only-candidate"
        )
        provider.find_result = candidate
        provider.mismatched_resource_ids.add(candidate.provider_resource_id)
        with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
            await owner.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
    finally:
        await release_gates_and_boundedly_join_owned_tasks(acquire, provider.finish)

    completed = await acquire

    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.ambiguous_create_candidate_id == "name-only-candidate"
    assert completed.ambiguous_candidate_mismatch is not None
    assert (
        completed.ambiguous_candidate_mismatch.provider_resource_id
        == "name-only-candidate"
    )
    assert completed.provisioning_mismatch is None
    assert completed.provider_resource_id == provider.resource.provider_resource_id
    assert completed.provisioning_receipt is not None
    assert completed.provisioning_receipt.reconciled_after_ambiguous_create is False
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_original_ack_folds_after_same_id_discovery_reaches_ready(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    original_provision = provider.provision

    async def blocked(prepared):
        resource = await original_provision(prepared)
        provider.find_result = resource
        entered.set()
        await finish.wait()
        return resource

    provider.provision = blocked
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        reconciled = await owner.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        assert reconciled.state is OllamaLeaseState.READY
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)
    completed = await original

    assert completed.state is OllamaLeaseState.READY
    assert completed.provider_resource_id == provider.resource.provider_resource_id
    assert completed.provisioning_receipt == reconciled.provisioning_receipt
    assert completed.route_url == reconciled.route_url
    assert completed.provider_health_url == reconciled.provider_health_url
    assert provider.provision_calls == 1


@pytest.mark.asyncio
async def test_original_ack_test_barrier_is_released_after_early_assertion(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    original_provision = provider.provision

    async def blocked(prepared):
        resource = await original_provision(prepared)
        entered.set()
        await finish.wait()
        return resource

    provider.provision = blocked
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    observed_failure = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        raise AssertionError("synthetic pre-release test failure")
    except AssertionError as exc:
        observed_failure = exc
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)

    completed = await original
    assert str(observed_failure) == "synthetic pre-release test failure"
    assert completed.provider_resource_id == provider.resource.provider_resource_id
    assert original.done()


@pytest.mark.asyncio
async def test_shared_owned_task_cleanup_retains_failure_before_entry_observation():
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def blocked():
        entered.set()
        await finish.wait()
        return "settled"

    owned = asyncio.create_task(blocked())
    original_failure = AssertionError("synthetic failure before entry observation")
    observed_failure = None
    try:
        raise original_failure
    except AssertionError as exc:
        observed_failure = exc
    finally:
        await release_gates_and_boundedly_join_owned_tasks(owned, finish)

    assert observed_failure is original_failure
    assert entered.is_set()
    assert owned.result() == "settled"


@pytest.mark.asyncio
async def test_shared_owned_task_cleanup_retrieves_every_terminal_exception():
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    first_failure = RuntimeError("first owned task failed")
    second_failure = ValueError("second owned task failed")

    async def fail_after(gate, failure):
        await gate.wait()
        raise failure

    owned = []
    first = asyncio.create_task(fail_after(first_gate, first_failure))
    owned.append(first)
    try:
        second = asyncio.create_task(fail_after(second_gate, second_failure))
        owned.append(second)
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            owned, first_gate, second_gate
        )

    assert all(task.done() for task in owned)
    assert owned[0].exception() is first_failure
    assert owned[1].exception() is second_failure


@pytest.mark.asyncio
async def test_different_valid_candidate_does_not_discard_original_post_custody(
    tmp_path,
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    deleted: list[str] = []
    original_provision = provider.provision
    original_teardown = provider.teardown

    async def blocked(prepared):
        resource = await original_provision(prepared)
        entered.set()
        await finish.wait()
        return resource

    async def tracked_teardown(resource, cleanup_intent):
        deleted.append(resource.provider_resource_id)
        return await original_teardown(resource, cleanup_intent)

    provider.provision = blocked
    provider.teardown = tracked_teardown
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        provider.find_result = replace(
            provider.resource, provider_resource_id="different-complete-candidate"
        )

        discovered = await owner.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        assert discovered.state is OllamaLeaseState.READY
        assert discovered.provider_resource_id == "different-complete-candidate"
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)
    completed = await original
    assert completed.state in {
        OllamaLeaseState.RELEASING,
        OllamaLeaseState.TERMINATED,
    }
    released = await owner.release(
        request.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )

    assert released.state is OllamaLeaseState.TERMINATED
    assert set(deleted) == {"different-complete-candidate", "provider-001"}
    assert deleted.count("different-complete-candidate") == 1
    assert deleted.count("provider-001") == 1


@pytest.mark.asyncio
async def test_original_post_ack_reopens_only_cleanup_after_candidate_is_terminal(
    tmp_path,
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    deleted: list[str] = []
    original_provision = provider.provision
    original_teardown = provider.teardown

    async def blocked(prepared):
        resource = await original_provision(prepared)
        entered.set()
        await finish.wait()
        return resource

    async def tracked_teardown(resource, cleanup_intent):
        deleted.append(resource.provider_resource_id)
        return await original_teardown(resource, cleanup_intent)

    provider.provision = blocked
    provider.teardown = tracked_teardown
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        provider.find_result = replace(
            provider.resource, provider_resource_id="different-complete-candidate"
        )
        discovered = await owner.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        terminated_candidate = await owner.release(
            discovered.lease_id,
            owner_id=discovered.owner_id,
            workload_id=discovered.workload_id,
        )
        assert terminated_candidate.state is OllamaLeaseState.TERMINATED
        assert deleted == ["different-complete-candidate"]
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)
    completed = await original

    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.public_route_url is None
    assert set(deleted) == {"different-complete-candidate", "provider-001"}
    assert completed.actual_cost_cap_status.value == "unknown"
    assert len(completed.additional_cleanup_targets) == 1
    assert completed.additional_cleanup_targets[0].cleanup_receipt is not None
    database = tmp_path / "leases.sqlite3"
    mutations = (
        "'[]'",
        (
            "json_set(additional_cleanup_targets_json, "
            "'$[0].provider_resource_id', 'foreign-resource')"
        ),
        "json_remove(additional_cleanup_targets_json, '$[0].cleanup_receipt')",
    )
    for mutation in mutations:
        with (
            closing(sqlite3.connect(database)) as connection,
            pytest.raises(
                sqlite3.IntegrityError,
                match="invalid Ollama lifecycle evidence mutation",
            ),
        ):
            connection.execute(
                "UPDATE ollama_leases "
                f"SET additional_cleanup_targets_json = {mutation}, "
                "revision = revision + 1, writer_epoch = writer_epoch + 1 "
                "WHERE lease_id = ?",
                (completed.lease_id,),
            )

    retained = owner.repository.get(completed.lease_id)
    assert retained is not None
    assert retained.additional_cleanup_targets == completed.additional_cleanup_targets


@pytest.mark.asyncio
async def test_second_exact_resource_failure_retains_primary_receipt_and_retry_state(
    tmp_path,
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    fail_original = True
    original_provision = provider.provision
    original_teardown = provider.teardown

    async def blocked(prepared):
        resource = await original_provision(prepared)
        entered.set()
        await finish.wait()
        return resource

    async def selective_teardown(resource, cleanup_intent):
        if resource.provider_resource_id == "provider-001" and fail_original:
            raise OllamaLeaseTeardownError(
                "additional teardown remains retryable for original acknowledgment"
            )
        return await original_teardown(resource, cleanup_intent)

    provider.provision = blocked
    provider.teardown = selective_teardown
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        provider.find_result = replace(
            provider.resource, provider_resource_id="different-complete-candidate"
        )
        discovered = await owner.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        terminated_candidate = await owner.release(
            discovered.lease_id,
            owner_id=discovered.owner_id,
            workload_id=discovered.workload_id,
        )
        assert terminated_candidate.cleanup_receipt is not None
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)

    with pytest.raises(OllamaLeaseTeardownError, match="additional teardown"):
        await original

    retained = owner.repository.get(request.lease_id)
    assert retained is not None
    assert retained.state is OllamaLeaseState.RELEASING
    assert retained.teardown_state is OllamaTeardownState.RETRYABLE_FAILURE
    assert retained.cleanup_receipt == terminated_candidate.cleanup_receipt
    assert len(retained.additional_cleanup_targets) == 1
    assert retained.additional_cleanup_targets[0].cleanup_receipt is None

    fail_original = False
    completed = await owner.release(
        request.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )
    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.additional_cleanup_targets[0].cleanup_receipt is not None


@pytest.mark.asyncio
async def test_same_id_concurrent_readback_keeps_first_committed_receipt(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = 0
    original_observe = provider.observe_provisioning

    async def blocked_first_observation(intent, resource, **kwargs):
        nonlocal calls
        calls += 1
        receipt = await original_observe(intent, resource, **kwargs)
        receipt = replace(receipt, readback_at=clock())
        if calls == 1:
            entered.set()
            await finish.wait()
        return receipt

    provider.observe_provisioning = blocked_first_observation
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        clock.advance(1)

        reconciled = await owner.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        assert reconciled.state is OllamaLeaseState.READY
        assert reconciled.provisioning_receipt is not None
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)
    completed = await original

    assert completed.state is OllamaLeaseState.READY
    assert completed.provisioning_receipt == reconciled.provisioning_receipt
    assert completed.provider_resource_id == reconciled.provider_resource_id
    assert provider.provision_calls == 1


@pytest.mark.asyncio
async def test_late_mismatching_ack_preserves_prior_receipt_and_enters_cleanup(
    tmp_path,
):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    entered = asyncio.Event()
    finish = asyncio.Event()
    original_provision = provider.provision
    original_observe = provider.observe
    observe_calls = 0

    async def counted_observe(resource, intent, provisioning):
        nonlocal observe_calls
        observe_calls += 1
        return await original_observe(resource, intent, provisioning)

    async def blocked_mismatch(prepared):
        resource = await original_provision(prepared)
        provider.find_result = resource
        entered.set()
        await finish.wait()
        raise OllamaProvisioningMismatchError(
            "original POST acknowledgment returned incompatible identity",
            OllamaProvisioningMismatchEvidence(
                intent_sha256=prepared.intent.intent_sha256,
                lease_id=prepared.intent.lease_id,
                provision_attempt_id=prepared.intent.provision_attempt_id,
                resource_type=prepared.intent.resource_type,
                provider_resource_id=resource.provider_resource_id,
                observed_resource_name="wrong-response-name",
                observed_at=clock(),
                readback_sha256="f" * 64,
                reason=OllamaProvisioningMismatchReason.RESOURCE_IDENTITY,
            ),
        )

    provider.provision = blocked_mismatch
    provider.observe = counted_observe
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        reconciled = await owner.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        assert reconciled.state is OllamaLeaseState.READY
        assert reconciled.provisioning_receipt is not None
        readiness_observe_calls = observe_calls
    finally:
        await release_gates_and_boundedly_join_owned_tasks(original, finish)
    with pytest.raises(OllamaProvisioningMismatchError):
        await original

    durable = owner.repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.route_url is None
    assert durable.provisioning_receipt == reconciled.provisioning_receipt
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.observed_resource_name == (
        "wrong-response-name"
    )
    assert observe_calls == readiness_observe_calls
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_unresolved_same_id_releases_join_before_provider_readback(tmp_path):
    clock = MutableClock()
    provider = FakeOllamaProvider(serverless_plan())
    original_read = provider.observe_provisioning

    async def unavailable(*_args, **_kwargs):
        raise OllamaProvisioningEvidenceError("original readback incomplete")

    provider.observe_provisioning = unavailable
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    with pytest.raises(OllamaProvisioningEvidenceError):
        await owner.acquire(request, wait_until_ready=False)

    entered = asyncio.Event()
    finish = asyncio.Event()
    read_calls = 0

    async def blocked_read(intent, resource, **kwargs):
        nonlocal read_calls
        read_calls += 1
        receipt = await original_read(intent, resource, **kwargs)
        entered.set()
        await finish.wait()
        return receipt

    provider.observe_provisioning = blocked_read
    first = asyncio.create_task(
        owner.release(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
    )
    owned = [first]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = asyncio.create_task(
            owner.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
        )
        owned.append(second)
        await asyncio.sleep(0)
    finally:
        await release_gates_and_boundedly_join_owned_tasks(owned, finish)
    first_result = first.result()
    second_result = second.result()

    assert first_result.state is second_result.state is OllamaLeaseState.TERMINATED
    assert read_calls == 1
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "concurrent_state", [OllamaLeaseState.FAILED, OllamaLeaseState.TERMINATED]
)
async def test_late_ack_reopens_only_cleanup_after_concurrent_no_work_state(
    tmp_path, concurrent_state
):
    clock = MutableClock()
    provider = ThreadedProvider(block="create")
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    acquire = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        assert await asyncio.to_thread(provider.entered.wait, 5)
        pending = owner.repository.get(request.lease_id)
        owner.repository.compare_and_set(
            pending,
            changes={
                "state": concurrent_state,
                "teardown_state": (
                    OllamaTeardownState.COMPLETE
                    if concurrent_state is OllamaLeaseState.TERMINATED
                    else OllamaTeardownState.NOT_REQUESTED
                ),
                "creation_uncertain": False,
                "termination_reason": "concurrent_cancel",
            },
        )
    finally:
        await release_gates_and_boundedly_join_owned_tasks(acquire, provider.finish)

    completed = await acquire

    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.provider_resource_id == provider.resource.provider_resource_id
    assert completed.termination_reason == "concurrent_cancel"
    assert completed.route_url is None
    assert completed.provider_health_url is None
    assert provider.observe_calls == 0
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_mismatch_cancellation_joins_same_known_id_teardown_and_ack(tmp_path):
    clock = MutableClock()
    provider = ThreadedProvider(block="delete")
    provider.confirmed_mismatch = True
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    task = asyncio.create_task(owner.acquire(request, wait_until_ready=False))
    try:
        assert await asyncio.to_thread(provider.entered.wait, 5)
        pending = owner.repository.get(request.lease_id)
        assert pending.state is OllamaLeaseState.RELEASING
        assert pending.provider_resource_id == provider.resource.provider_resource_id
        assert pending.provisioning_mismatch is not None
        assert pending.route_url is None
        for _ in range(4):
            task.cancel("original caller stopped")
            await asyncio.sleep(0)
            assert not task.done(), "caller escaped while mismatch teardown was live"
    finally:
        await release_gates_and_boundedly_join_owned_tasks(task, provider.finish)
        with pytest.raises(OllamaProvisioningMismatchError):
            await task
        assert await asyncio.to_thread(provider.completed.wait, 5)

    durable = owner.repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.teardown_state is OllamaTeardownState.COMPLETE
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_receipt is None
    assert durable.route_url is None
    assert provider.pull_calls == 0
    assert provider.teardown_calls == 1
    with (
        closing(sqlite3.connect(owner.repository.path)) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="invalid Ollama provisioning evidence mutation",
        ),
    ):
        connection.execute(
            """UPDATE ollama_leases
               SET provisioning_mismatch_json = NULL,
                   revision = revision + 1,
                   writer_epoch = writer_epoch + 1
               WHERE lease_id = ?""",
            (request.lease_id,),
        )


@pytest.mark.asyncio
async def test_lost_mismatch_ack_restarts_from_durable_evidence_without_readback(
    tmp_path, monkeypatch
):
    clock = MutableClock()
    provider = ThreadedProvider(block="none")
    provider.confirmed_mismatch = True
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    original_outcome = owner.repository.record_provisioning_outcome
    lost_ack = False

    def commit_then_lose_ack(lease, **outcome):
        nonlocal lost_ack
        updated = original_outcome(lease, **outcome)
        if not lost_ack and outcome.get("mismatch") is not None:
            lost_ack = True
            raise RunPodManagerError("simulated lost mismatch ACK")
        return updated

    monkeypatch.setattr(
        owner.repository, "record_provisioning_outcome", commit_then_lose_ack
    )
    with pytest.raises(RunPodManagerError, match="lost mismatch ACK"):
        await owner.acquire(request, wait_until_ready=False)

    interrupted = SQLiteOllamaLeaseRepository(owner.repository.path).get(
        request.lease_id
    )
    assert interrupted.state is OllamaLeaseState.RELEASING
    assert interrupted.provisioning_mismatch is not None
    assert interrupted.provider_resource_id == provider.resource.provider_resource_id

    restarted = service(tmp_path, provider, clock)
    final = (await restarted.reconcile())[0]

    assert final.state is OllamaLeaseState.TERMINATED
    assert final.provisioning_mismatch == interrupted.provisioning_mismatch
    assert provider.observe_calls == 1
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
async def test_absent_ambiguous_create_remains_unknown_past_all_deadlines(tmp_path):
    class AmbiguousProvider(FakeOllamaProvider):
        async def provision(self, prepared):
            self.provision_calls += 1
            self.resource = replace(
                self.resource, resource_name=prepared.intent.resource_name
            )
            raise RunPodAmbiguousResultError(
                title="Create outcome unknown",
                detail="Original response was lost",
                method="POST",
                resource="/serverless",
            )

    clock = MutableClock()
    provider = AmbiguousProvider(serverless_plan())
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    with pytest.raises(RunPodAmbiguousResultError):
        await owner.acquire(request, wait_until_ready=False)
    clock.advance(3 * 3600)
    restarted = service(tmp_path, provider, clock)
    for _ in range(3):
        results = await restarted.reconcile()
        assert len(results) == 1, "unknown original create must remain discoverable"
        assert results[0].state is OllamaLeaseState.RELEASING
        assert results[0].creation_uncertain is True
        assert results[0].teardown_state is OllamaTeardownState.RETRYABLE_FAILURE
        clock.advance(3600)
    with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
        await restarted.release(
            request.lease_id, owner_id=request.owner_id, workload_id=request.workload_id
        )
    provider.find_result = provider.resource
    results = await restarted.reconcile()
    assert results[0].state is OllamaLeaseState.TERMINATED
    assert results[0].provider_resource_id == provider.resource.provider_resource_id
    assert provider.provision_calls == 1
    assert provider.teardown_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RunPodManagerError("Successful create response omitted resource identity"),
        RunPodAPIError(
            title="Invalid response", detail="Non-JSON create response", status_code=201
        ),
    ],
)
async def test_incomplete_create_response_never_proves_no_resource(tmp_path, error):
    class IncompleteProvider(FakeOllamaProvider):
        async def provision(self, prepared):
            self.provision_calls += 1
            self.resource = replace(
                self.resource, resource_name=prepared.intent.resource_name
            )
            raise error

    clock = MutableClock()
    provider = IncompleteProvider(serverless_plan())
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    with pytest.raises(type(error)):
        await owner.acquire(request, wait_until_ready=False)
    pending = owner.repository.get(request.lease_id)
    assert pending.creation_uncertain is True
    assert pending.state is OllamaLeaseState.RECONCILE_REQUIRED
    clock.advance(3 * 3600)
    provider.find_result = provider.resource
    assert (await owner.reconcile())[0].state is OllamaLeaseState.TERMINATED
    assert provider.provision_calls == 1
    assert provider.teardown_calls == 1
