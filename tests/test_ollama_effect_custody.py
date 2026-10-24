"""Original Ollama effects must physically finish before cancellation escapes."""

import asyncio
import inspect
import sqlite3
import sys
import threading
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest
from kestrel_sovereign.async_ownership import (
    create_owned_task,
    owned_task_error,
    pending_owned_retirement_error,
    wait_owned_tasks_until,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_terminal_error,
)
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    pod_plan,
    release_gates_and_boundedly_join_owned_tasks,
    serverless_plan,
)
from test_effect_custody import _original_errors
from test_ollama_repository import _terminated_pod_billing_evidence

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
    def __init__(self, *, block, plan=None):
        super().__init__(serverless_plan() if plan is None else plan)
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
        primary = sys.exception()
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                task, provider.finish, primary=primary
            )
        except (
            BaseException
        ) as terminal:  # assert exact expected original only without a body failure
            if primary is not None:
                raise
            expected = owned_task_error(task)
            assert isinstance(expected, asyncio.CancelledError)
            assert terminal is expected
        else:
            pytest.fail("original caller cancellation was discarded")
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
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, provider.finish, primary=sys.exception()
        )

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
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, provider.finish, primary=sys.exception()
        )

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
        await release_gates_and_boundedly_join_owned_tasks(
            original, finish, primary=sys.exception()
        )
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
        await release_gates_and_boundedly_join_owned_tasks(
            original, finish, primary=sys.exception()
        )

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
        await release_gates_and_boundedly_join_owned_tasks(
            owned, finish, primary=sys.exception()
        )

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
        primary = sys.exception()
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                owned, first_gate, second_gate, primary=primary
            )
        except BaseException as terminal:  # both exact original failures are the oracle
            if primary is not None:
                raise
            originals = tuple(_original_errors(terminal))
            assert any(error is first_failure for error in originals)
            assert any(error is second_failure for error in originals)
            assert not any(
                isinstance(error, asyncio.CancelledError) for error in originals
            )
        else:
            pytest.fail("both original owned failures were discarded")

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
        await release_gates_and_boundedly_join_owned_tasks(
            original, finish, primary=sys.exception()
        )
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
        await release_gates_and_boundedly_join_owned_tasks(
            original, finish, primary=sys.exception()
        )
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
        primary = sys.exception()
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                original, finish, primary=primary
            )
        except (
            BaseException
        ) as terminal:  # expected original teardown error, never a body failure
            if primary is not None:
                raise
            expected = owned_task_error(original)
            assert isinstance(expected, OllamaLeaseTeardownError)
            assert "additional teardown" in str(expected)
            assert terminal is expected
        else:
            pytest.fail("original additional teardown failure was discarded")

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
        await release_gates_and_boundedly_join_owned_tasks(
            original, finish, primary=sys.exception()
        )
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
        primary = sys.exception()
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                original, finish, primary=primary
            )
        except (
            BaseException
        ) as terminal:  # exact acknowledged mismatch remains the oracle
            if primary is not None:
                raise
            expected = owned_task_error(original)
            assert isinstance(expected, OllamaProvisioningMismatchError)
            assert terminal is expected
        else:
            pytest.fail("original acknowledgment mismatch was discarded")
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
        await release_gates_and_boundedly_join_owned_tasks(
            owned, finish, primary=sys.exception()
        )
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
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, provider.finish, primary=sys.exception()
        )

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
        primary = sys.exception()
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                task, provider.finish, primary=primary
            )
        except (
            BaseException
        ) as terminal:  # mismatch and observer withdrawal must both remain
            if primary is not None:
                raise
            expected = owned_task_error(task)
            assert isinstance(expected, OllamaProvisioningMismatchError)
            assert terminal is expected
            assert any(
                isinstance(error, asyncio.CancelledError)
                for error in _original_errors(terminal)
            )
        else:
            pytest.fail("original mismatch/cancellation evidence was discarded")
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


async def _r27_cached_case(tmp_path, kind):
    """Existing public lifecycle, existing provider body, original SQLite facts."""
    if kind == "cleanup":
        clock = MutableClock()
        provider = ThreadedProvider(block="delete")
        owner = service(tmp_path, provider, clock)
        lease = await owner.acquire(make_request(clock))
        invoke = lambda: owner.release(
            lease.lease_id, owner_id=lease.owner_id, workload_id=lease.workload_id
        )
        key = (lease.lease_id, lease.provisioning_intent.intent_sha256)
        cache_attr, operation_attr = "_cleanup_tasks", "_cleanup_effect"
    else:
        (
            repository,
            lease,
            observed_at,
            observation,
        ) = await _terminated_pod_billing_evidence(tmp_path)
        clock = MutableClock()
        clock.value = observed_at
        provider = ThreadedProvider(block="billing", plan=pod_plan())
        owner = OllamaLeaseService(
            repository=repository,
            provider=provider,
            poll_interval_seconds=1,
            clock=clock,
            sleep=clock.sleep,
        )

        async def gated_billing(intent, provisioning, cleanup):
            assert intent == lease.provisioning_intent
            assert provisioning == lease.provisioning_receipt
            assert cleanup == lease.cleanup_receipt
            await asyncio.to_thread(provider.physical_effect)
            return observation

        provider.observe_lifecycle_billing = gated_billing
        invoke = lambda: owner.reconcile_lease(
            lease.lease_id, owner_id=lease.owner_id, workload_id=lease.workload_id
        )
        key = (lease.lease_id, lease.cleanup_receipt.cleanup_receipt_id)
        cache_attr, operation_attr = "_billing_tasks", "_billing_effect"
    return SimpleNamespace(
        owner=owner,
        provider=provider,
        lease=lease,
        invoke=invoke,
        key=key,
        cache_attr=cache_attr,
        operation_attr=operation_attr,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cleanup", "billing"])
@pytest.mark.parametrize(
    "fault", ["forget-before", "forget-after", "cache-before", "cache-after"]
)
async def test_r27_existing_cached_publication_door_joins_original(
    tmp_path, kind, fault, record_property
):
    case = await _r27_cached_case(tmp_path, kind)
    opening = RuntimeError("original cache publication: " + fault)
    entries, accepted = [], []
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    class NativePublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if (
                fault.startswith("forget-")
                and getattr(callback, "__name__", "") == "forget"
            ):
                entries.append(self)
                record_property("injection_entry", fault)
                if fault == "forget-after":
                    super().add_done_callback(callback, context=context)
                raise opening
            return super().add_done_callback(callback, context=context)

    def factory(current_loop, coroutine, **kwargs):
        task = NativePublicationTask(coroutine, loop=current_loop, **kwargs)
        accepted.append(task)
        return task

    class NativeCache(dict):
        def __setitem__(self, key, task):
            entries.append(task)
            record_property("injection_entry", fault)
            if fault == "cache-after":
                super().__setitem__(key, task)
            raise opening

    if fault.startswith("cache-"):
        setattr(case.owner, case.cache_attr, NativeCache())
    caller = None
    try:
        loop.set_task_factory(factory)
        caller = asyncio.create_task(case.invoke(), name="r27-public-caller")
        assert await asyncio.to_thread(case.provider.entered.wait, 5), (
            "actual original body did not enter"
        )
        if fault.startswith("forget-"):
            # T2: the redundant callback was REMOVED, not successfully retried.
            assert entries == []
            record_property("removed_forget_callback_absent", True)
        else:
            assert entries, "the same cache publication boundary was not reached"
        assert not caller.done(), (
            "public caller escaped before the same physical tail completed"
        )
    finally:
        primary = sys.exception()
        loop.set_task_factory(original_factory)
        effects = [
            task for task in accepted if task.get_name().startswith("ollama-" + kind)
        ]
        # T1 deliberately uses the EXISTING helper call shape. No primary kwarg.
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                [task for task in [caller, *effects] if task is not None],
                case.provider.finish,
                primary=primary,
            )
        except BaseException as cleanup:  # noqa: BLE001 - retain actual primary even if the current helper evolves
            if primary is not None:
                raise execution_terminal_error(
                    primary,
                    cleanup,
                    retain_originals=True,
                    isolate_control_originals=True,
                )
            assert fault.startswith("cache-")
            assert cleanup is opening
    assert case.provider.completed.is_set()
    assert caller.done() and not caller.cancelled()
    if fault.startswith("forget-"):
        assert caller.exception() is None
    else:
        assert owned_task_error(caller) is opening
    assert len(effects) == 1 and effects[0].done()
    assert case.key not in getattr(case.owner, case.cache_attr)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cleanup", "billing"])
async def test_r27_existing_cached_allocation_refusal_closes_original(
    tmp_path, monkeypatch, kind, record_property
):
    case = await _r27_cached_case(tmp_path, kind)
    original_operation = getattr(case.owner, case.operation_attr)
    originals, admissions = [], []
    opening = RuntimeError("original cache Task allocation refused")
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    def operation(*args, **kwargs):
        coroutine = original_operation(*args, **kwargs)
        originals.append(coroutine)
        return coroutine

    def refuse(_loop, _coroutine, **_kwargs):
        admissions.append("refused-before-Task")
        record_property("injection_entry", "actual-loop-allocation-refusal")
        raise opening

    monkeypatch.setattr(case.owner, case.operation_attr, operation)
    try:
        loop.set_task_factory(refuse)
        with pytest.raises(RuntimeError) as caught:
            await case.invoke()
        assert caught.value is opening
        assert admissions == ["refused-before-Task"] and len(originals) == 1
        assert inspect.getcoroutinestate(originals[0]) == inspect.CORO_CLOSED
    finally:
        loop.set_task_factory(original_factory)
        # Baseline safety only: no native Task was accepted. This original
        # unadmitted coroutine is never retried or treated as a physical ACK.
        for coroutine in originals:
            if inspect.getcoroutinestate(coroutine) != inspect.CORO_CLOSED:
                record_property("baseline_outer_closed_unadmitted_original", True)
                coroutine.close()
    assert not case.provider.entered.is_set()
    assert case.key not in getattr(case.owner, case.cache_attr)


@pytest.mark.asyncio
async def test_r27_existing_drain_preserves_two_actual_child_failures(record_property):
    first_gate, second_gate = asyncio.Event(), asyncio.Event()
    first_error = RuntimeError("first exact real child")
    second_error = ExecutionCommitOutcomeError("unknown")

    async def fail(gate, error):
        await gate.wait()
        record_property("injection_entry", type(error).__name__)
        raise error

    first = asyncio.create_task(fail(first_gate, first_error))
    second = asyncio.create_task(fail(second_gate, second_error))
    observed = None
    try:
        # No new primary keyword: this is a real lost-error behavioral door.
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                [first, second], first_gate, second_gate
            )
        except BaseException as error:  # noqa: BLE001 - original finite terminal evidence is the test result
            observed = error
        assert first.done() and second.done()
        assert first.exception() is first_error and second.exception() is second_error
        assert observed is not None, (
            "current drain discarded both actual child failures"
        )
        originals = tuple(_original_errors(observed))
        assert any(error is first_error for error in originals)
        assert any(error is second_error for error in originals)
    finally:
        primary = sys.exception()
        try:
            await release_gates_and_boundedly_join_owned_tasks(
                [first, second], first_gate, second_gate, primary=primary
            )
        except BaseException as cleanup:  # noqa: BLE001 - exact original pending-test primary must survive
            if primary is not None:
                raise execution_terminal_error(
                    primary,
                    cleanup,
                    retain_originals=True,
                    isolate_control_originals=True,
                )
            originals = tuple(_original_errors(cleanup))
            assert any(error is first_error for error in originals)
            assert any(error is second_error for error in originals)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cleanup", "billing"])
@pytest.mark.parametrize("publication", ["before", "after"])
async def test_r27_core_postaccept_publication_keeps_exact_original(
    tmp_path, monkeypatch, kind, publication, record_property
):
    """T2 ONLY: the canonical creator is a newly adopted service path."""
    case = await _r27_cached_case(tmp_path, kind)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    opening = RuntimeError("actual Core callback publication: " + publication)
    accepted, entries, originals = [], [], []
    real_operation = getattr(case.owner, case.operation_attr)

    def operation(*args, **kwargs):
        coroutine = real_operation(*args, **kwargs)
        originals.append(coroutine)
        return coroutine

    class NativePublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                entries.append(self)
                record_property(
                    "injection_entry", "Core-close_unstarted-" + publication
                )
                if publication == "after":
                    super().add_done_callback(callback, context=context)
                raise opening
            return super().add_done_callback(callback, context=context)

    def factory(current_loop, coroutine, **kwargs):
        task = NativePublicationTask(coroutine, loop=current_loop, **kwargs)
        accepted.append(task)
        return task

    monkeypatch.setattr(case.owner, case.operation_attr, operation)
    try:
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            await case.invoke()
        assert caught.value is opening
        assert len(entries) == len(accepted) == len(originals) == 1
        assert accepted[0].done(), "the service lost the actual accepted gated Task"
        assert owned_task_error(accepted[0]) is opening
        assert inspect.getcoroutinestate(originals[0]) == inspect.CORO_CLOSED
        assert not case.provider.entered.is_set()
        assert case.key not in getattr(case.owner, case.cache_attr)
    finally:
        primary = sys.exception()
        loop.set_task_factory(previous_factory)
        case.provider.finish.set()
        observation = await wait_owned_tasks_until(accepted, deadline=loop.time() + 5)
        debt = pending_owned_retirement_error(
            case, accepted, message="actual accepted cache Task still pending"
        )
        failures = [primary, observation.error, observation.cancellation, debt]
        for task in accepted:
            if task.done():
                original = owned_task_error(task)
                if primary is not None or original is not opening:
                    failures.append(original)
        # With primary=None only the exact opening already asserted above is
        # handled as expected. Every independent/late original still raises.
        terminal = execution_terminal_error(
            *failures, retain_originals=True, isolate_control_originals=True
        )
        if terminal is not None:
            raise terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cleanup", "billing"])
@pytest.mark.parametrize("late_kind", ["ordinary", "unknown", "control"])
async def test_r27_cache_publication_retains_late_original_and_repeated_cancel(
    tmp_path, kind, late_kind, record_property
):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        execution_commit_outcome,
        is_execution_control_error,
    )

    case = await _r27_cached_case(tmp_path, kind)
    opening = RuntimeError("actual postaccept cache insertion failure")
    late = (
        ValueError("actual late body failure")
        if late_kind == "ordinary"
        else ExecutionCommitOutcomeError("unknown")
        if late_kind == "unknown"
        else ExecutionAuthorityError("original execution scope denied")
    )
    actual_physical = case.provider.physical_effect

    def physical():
        actual_physical()
        record_property("injection_entry", "late-physical-" + late_kind)
        raise late

    case.provider.physical_effect = physical
    entries = []

    class NativeCache(dict):
        def __setitem__(self, key, task):
            super().__setitem__(key, task)
            entries.append(task)
            record_property("injection_entry", "accepted-cache-after-store")
            raise opening

    setattr(case.owner, case.cache_attr, NativeCache())
    caller = create_owned_task(case.invoke(), retain_originals=True)
    try:
        assert await asyncio.to_thread(case.provider.entered.wait, 5)
        assert len(entries) == 1
        for index in range(4):
            caller.cancel("actual observer withdrawal-" + str(index))
            await asyncio.sleep(0)
            assert not caller.done()
    finally:
        primary = sys.exception()
        case.provider.finish.set()
        observation = await wait_owned_tasks_until(
            [caller, *entries], deadline=asyncio.get_running_loop().time() + 5
        )
        debt = pending_owned_retirement_error(
            case, [caller, *entries], message="original cache owners still pending"
        )
        failures = [primary, observation.error, observation.cancellation, debt]
        if any(error is not None for error in failures):
            failures.extend(
                owned_task_error(task) for task in [caller, *entries] if task.done()
            )
        # Only a successful outer observation leaves the expected exact
        # publication/body/C graph for the assertions immediately below.
        terminal = execution_terminal_error(
            *failures, retain_originals=True, isolate_control_originals=True
        )
        if terminal is not None:
            raise terminal
    failure = owned_task_error(caller)
    originals = tuple(_original_errors(failure))
    assert any(error is opening for error in originals)
    assert any(error is late for error in originals)
    assert any(
        isinstance(error, asyncio.CancelledError)
        and error.args == ("actual observer withdrawal-0",)
        for error in originals
    )
    assert case.provider.completed.is_set() and entries[0].done()
    assert case.key not in getattr(case.owner, case.cache_attr)
    if late_kind != "ordinary":
        assert is_execution_control_error(failure) and late.__cause__ is None
    if late_kind == "unknown":
        assert execution_commit_outcome(failure) == "unknown"
    # Unit/native-tail UNKNOWN above is NOT an actual SQL COMMIT outcome.


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cleanup", "billing"])
async def test_r27_cache_borrowers_join_same_original_then_identity_evict(
    tmp_path, kind
):
    case = await _r27_cached_case(tmp_path, kind)
    calls = []
    real_physical = case.provider.physical_effect

    def physical():
        calls.append("same original body")
        return real_physical()

    case.provider.physical_effect = physical
    first = create_owned_task(case.invoke(), retain_originals=True)
    owned = [first]
    try:
        assert await asyncio.to_thread(case.provider.entered.wait, 5)
        original = getattr(case.owner, case.cache_attr)[case.key]
        second = create_owned_task(case.invoke(), retain_originals=True)
        owned.append(second)
        await asyncio.sleep(0)
        assert getattr(case.owner, case.cache_attr)[case.key] is original
        assert not first.done() and not second.done()
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            owned, case.provider.finish, primary=sys.exception()
        )
    assert first.result() == second.result()
    assert original.done() and calls == ["same original body"]
    assert case.key not in getattr(case.owner, case.cache_attr)


@pytest.mark.asyncio
async def test_r27_new_primary_carry_retains_body_and_unknown_after_drain():
    """T2 only: new primary wiring is not an old-interface behavioral RED."""
    gate = asyncio.Event()
    primary = AssertionError("actual active test body")
    unknown = ExecutionCommitOutcomeError("unknown")

    async def body():
        await gate.wait()
        raise unknown

    task = create_owned_task(body(), retain_originals=True)

    async def exercise_actual_finally():
        try:
            raise primary
        finally:
            # Never enter an expected-child catch with an active primary.
            await release_gates_and_boundedly_join_owned_tasks(
                task, gate, primary=sys.exception()
            )

    with pytest.raises(BaseException) as caught:
        await exercise_actual_finally()
    originals = tuple(_original_errors(caught.value))
    assert any(error is primary for error in originals)
    assert any(error is unknown for error in originals)
    assert task.done() and owned_task_error(task) is unknown


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["first", "second"])
async def test_r27_drain_observer_cancellation_retains_original_waits(
    monkeypatch, phase, record_property
):
    import ollama_test_support as support
    from kestrel_sovereign.async_ownership import await_owned_task

    entered, exit_gate, wait_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real_wait = support.wait_owned_tasks_until
    cancellations = []

    async def child_body():
        entered.set()
        try:
            await exit_gate.wait()
        except asyncio.CancelledError as error:
            cancellations.append(error)
            await exit_gate.wait()
        return "same child retired"

    child = create_owned_task(child_body(), retain_originals=True)

    async def observed_wait(originals, *, deadline):
        observed_phase = "second" if child.cancelling() else "first"
        if observed_phase == phase:
            record_property(
                "injection_entry", "actual-" + phase + "-bounded-observation"
            )
            wait_entered.set()
        return await real_wait(originals, deadline=deadline)

    monkeypatch.setattr(support, "wait_owned_tasks_until", observed_wait)
    gate = asyncio.Event()
    observer = create_owned_task(
        release_gates_and_boundedly_join_owned_tasks(child, gate), retain_originals=True
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(wait_entered.wait(), 10)
        for index in range(4):
            observer.cancel("actual drain observer withdrawal-" + str(index))
            await asyncio.sleep(0)
            assert not observer.done(), (
                "actual original still held during bounded observation"
            )
    finally:
        primary = sys.exception()
        exit_gate.set()
        observation = await real_wait(
            [observer, child], deadline=asyncio.get_running_loop().time() + 5
        )
        debt = pending_owned_retirement_error(
            child, [observer, child], message="drain acceptance originals still pending"
        )
        failures = [primary, observation.error, observation.cancellation, debt]
        if any(error is not None for error in failures):
            failures.extend(
                owned_task_error(task) for task in [observer, child] if task.done()
            )
        terminal = execution_terminal_error(
            *failures, retain_originals=True, isolate_control_originals=True
        )
        if terminal is not None:
            raise terminal
    outcome = await await_owned_task(observer)
    assert any(
        isinstance(error, asyncio.CancelledError)
        and error.args == ("actual drain observer withdrawal-0",)
        for error in _original_errors(outcome.error)
    )
    assert child.result() == "same child retired"
    assert len(cancellations) == (1 if phase == "second" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["normal", "cancel-error", "gate-error"])
async def test_r27_pending_original_is_retained_until_outer_physical_join(
    delivery, record_property
):
    from kestrel_sovereign.execution_custody import execution_cleanup_debts

    start, exit_gate = asyncio.Event(), asyncio.Event()
    forwarding = RuntimeError("actual native cancel delivery returned error")
    gate_error = ValueError("actual release gate returned error")
    cancel_calls = []

    async def child_body():
        await start.wait()
        try:
            await exit_gate.wait()
        except asyncio.CancelledError:
            await exit_gate.wait()
        return "exact original retired"

    class NativeTask(asyncio.Task):
        def cancel(self, message=None):
            cancel_calls.append(self)
            result = super().cancel(message)
            if delivery == "cancel-error":
                record_property("injection_entry", "actual-native-cancel-error")
                raise forwarding
            return result

    class NativeGate(asyncio.Event):
        def set(self):
            super().set()
            start.set()
            if delivery == "gate-error":
                record_property("injection_entry", "actual-native-gate-error")
                raise gate_error

    task = NativeTask(child_body())
    cohort = (task,)
    try:
        with pytest.raises(BaseException) as caught:
            await release_gates_and_boundedly_join_owned_tasks(
                cohort, NativeGate(), timeout=0.01
            )
        assert cancel_calls == [task]
        assert not task.done()
        assert any(
            resource is cohort and owner is task
            for resource, owner in execution_cleanup_debts(caught.value)
        )
        if delivery != "normal":
            original = forwarding if delivery == "cancel-error" else gate_error
            assert any(error is original for error in _original_errors(caught.value))
    finally:
        primary = sys.exception()
        start.set()
        exit_gate.set()
        observation = await wait_owned_tasks_until(
            cohort, deadline=asyncio.get_running_loop().time() + 5
        )
        debt = pending_owned_retirement_error(
            cohort, cohort, message="outer original Task not physically joined"
        )
        terminal = execution_terminal_error(
            primary,
            observation.error,
            observation.cancellation,
            debt,
            owned_task_error(task) if task.done() else None,
            retain_originals=True,
            isolate_control_originals=True,
        )
        if terminal is not None:
            raise terminal
    assert task.result() == "exact original retired"


@pytest.mark.asyncio
@pytest.mark.parametrize("already_done", [False, True])
async def test_r27_drain_ordinary_success_keeps_exact_original(already_done):
    gate = asyncio.Event()

    async def body():
        await gate.wait()
        return "same original receipt"

    task = create_owned_task(body(), retain_originals=True)
    try:
        if already_done:
            gate.set()
            observation = await wait_owned_tasks_until(
                [task], deadline=asyncio.get_running_loop().time() + 5
            )
            assert observation.result == ({task}, set())
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            task, gate, primary=sys.exception()
        )
    assert task.result() == "same original receipt"


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
