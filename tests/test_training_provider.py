"""Runpod REST v2 training capacity adapter contracts."""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from training_test_support import training_profile

from kestrel_cloud_runpod.models import RunPodAmbiguousResultError, RunPodManagerError
from kestrel_cloud_runpod.providers import DirectRunPodProvider
from kestrel_cloud_runpod.training_provider import RunpodTrainingPodProvider


@pytest.mark.asyncio
async def test_observe_resolves_exact_http_proxy_route() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.get_status.return_value = {
        "id": "pod-1",
        "status": "RUNNING",
        "runtime": {"ports": [{"private": 8888, "type": "http"}]},
    }
    adapter = RunpodTrainingPodProvider(provider)

    observed = await adapter.observe("pod-1", profile=training_profile())

    assert observed.is_running is True
    assert observed.backend_base_url == "https://pod-1-8888.proxy.runpod.net"


@pytest.mark.asyncio
async def test_create_uses_provider_v2_boundary_and_validates_id() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.start_pod.return_value = {"id": "pod-created"}
    adapter = RunpodTrainingPodProvider(provider)

    created = await adapter.create(
        profile=training_profile(),
        resource_name="kestrel-lora-durable-name",
        companion_id="companion-1",
    )

    assert created.provider_pod_id == "pod-created"
    assert provider.start_pod.call_args.args[1] == {
        "name": "kestrel-lora-durable-name",
        "companion_id": "companion-1",
        "purpose": "lora_training",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"status": "RUNNING"},
        {"id": "pod-created", "_kestrel_placement": "invalid"},
    ],
)
async def test_successful_create_with_unusable_identity_remains_reconcilable(
    response,
) -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.start_pod.return_value = response
    adapter = RunpodTrainingPodProvider(provider)

    with pytest.raises(RunPodAmbiguousResultError) as raised:
        await adapter.create(
            profile=training_profile(),
            resource_name="kestrel-lora-durable-name",
            companion_id="companion-1",
        )

    assert raised.value.reconcile_required is True


@pytest.mark.asyncio
async def test_find_by_name_rejects_duplicate_provider_resources() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.list_pods.return_value = [
        {"id": "pod-1", "name": "same"},
        {"id": "pod-2", "name": "same"},
    ]
    adapter = RunpodTrainingPodProvider(provider)

    with pytest.raises(RunPodManagerError, match="Multiple"):
        await adapter.find_by_name("same")


@pytest.mark.asyncio
async def test_stop_requires_v2_confirmation() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.stop_pod.return_value = {"id": "pod-1", "status": "STOPPING"}
    provider.get_status.return_value = {"id": "pod-1", "status": "RUNNING"}
    adapter = RunpodTrainingPodProvider(provider)

    assert await adapter.stop("pod-1") is False


@pytest.mark.asyncio
async def test_cancelled_start_waits_for_inflight_v2_mutation_to_resolve() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    entered = threading.Event()
    finish = threading.Event()

    def blocking_start(*_):
        entered.set()
        finish.wait(timeout=5)
        return {"id": "pod-1", "status": "RUNNING"}

    provider.resume_pod.side_effect = blocking_start
    adapter = RunpodTrainingPodProvider(provider)
    task = asyncio.create_task(adapter.start("pod-1", gpu_count=1))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
