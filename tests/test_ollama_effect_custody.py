"""Original Ollama effects must physically finish before cancellation escapes."""

import asyncio
import threading
from dataclasses import replace

import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
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

    def physical_effect(self):
        self.entered.set()
        assert self.finish.wait(5), "test did not release the original provider thread"
        self.completed.set()

    async def provision(self, **kwargs):
        self.resource = replace(self.resource, resource_name=kwargs["resource_name"])
        self.provision_calls += 1
        if self.block == "create":
            await asyncio.to_thread(self.physical_effect)
        return self.resource

    async def teardown(self, resource):
        assert resource == self.resource
        self.teardown_calls += 1
        if self.block == "delete":
            await asyncio.to_thread(self.physical_effect)


def service(tmp_path, provider, clock):
    return OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3"),
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("effect", ["create", "delete"])
async def test_cancellation_joins_original_thread_and_durable_receipt(tmp_path, effect):
    clock = MutableClock()
    provider = ThreadedProvider(block=effect)
    owner = service(tmp_path, provider, clock)
    request = make_request(clock)
    if effect == "create":
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
        provider.finish.set()
        # Always collect the actual task, including when the RED assertion fails.
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(provider.completed.wait, 5)

    durable = owner.repository.get(request.lease_id)
    assert durable.provider_resource_id == provider.resource.provider_resource_id
    assert durable.creation_uncertain is False
    if effect == "create":
        assert durable.state is OllamaLeaseState.WAITING_FOR_MODEL
    else:
        assert durable.state is OllamaLeaseState.TERMINATED
        assert durable.teardown_state is OllamaTeardownState.COMPLETE


@pytest.mark.asyncio
async def test_absent_ambiguous_create_remains_unknown_past_all_deadlines(tmp_path):
    class AmbiguousProvider(FakeOllamaProvider):
        async def provision(self, **kwargs):
            self.provision_calls += 1
            self.resource = replace(
                self.resource, resource_name=kwargs["resource_name"]
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
        async def provision(self, **kwargs):
            self.provision_calls += 1
            self.resource = replace(
                self.resource, resource_name=kwargs["resource_name"]
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
