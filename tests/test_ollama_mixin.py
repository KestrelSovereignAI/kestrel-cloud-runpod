"""Manager integration tests for the durable Ollama lease surface."""

from types import SimpleNamespace

import pytest
from ollama_test_support import MutableClock, make_request

from kestrel_cloud_runpod.models import GPUProfile, RunPodManagerError
from kestrel_cloud_runpod.ollama import RunPodOllamaMixin
from kestrel_cloud_runpod.providers import DirectRunPodProvider


class _Service:
    def __init__(self):
        self.calls = []

    async def acquire(self, request):
        self.calls.append(("acquire", request.lease_id))
        return "acquired"

    async def get(self, lease_id, **identity):
        self.calls.append(("get", lease_id, identity))
        return "found"

    async def touch(self, lease_id, **identity):
        self.calls.append(("touch", lease_id, identity))
        return "touched"

    async def release(self, lease_id, **identity):
        self.calls.append(("release", lease_id, identity))
        return "released"

    async def reconcile(self):
        self.calls.append(("reconcile",))
        return ("reconciled",)


class _Harness(RunPodOllamaMixin):
    def __init__(self):
        self.provider = SimpleNamespace()
        self.config = {}


@pytest.mark.asyncio
async def test_mixin_delegates_only_to_injected_durable_service():
    harness = _Harness()
    service = _Service()
    harness.set_ollama_lease_service(service)
    identity = {"owner_id": "owner-001", "workload_id": "workload-001"}

    assert (
        await harness.acquire_ollama_lease(make_request(MutableClock())) == "acquired"
    )
    assert await harness.get_ollama_lease("lease-001", **identity) == "found"
    assert await harness.touch_ollama_lease("lease-001", **identity) == "touched"
    assert await harness.release_ollama_lease("lease-001", **identity) == "released"
    assert await harness.reconcile_ollama_leases() == ("reconciled",)
    assert [call[0] for call in service.calls] == [
        "acquire",
        "get",
        "touch",
        "release",
        "reconcile",
    ]


def test_mixin_has_no_in_memory_or_managed_provider_fallback():
    harness = _Harness()

    with pytest.raises(RunPodManagerError, match="direct Runpod v2"):
        harness._get_ollama_lease_service()


def test_mixin_builds_one_configured_durable_service(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_SERVERLESS_API_KEY", "serverless-key")
    monkeypatch.setenv("RUNPOD_OLLAMA_BEARER_TOKEN", "pod-token")
    harness = _Harness()
    harness.provider = DirectRunPodProvider(api_key="", client=SimpleNamespace())
    harness.config = {
        "ollama_leases": {
            "database_path": str(tmp_path / "leases.sqlite3"),
            "poll_interval_seconds": 5,
            "serverless_workers_max": 1,
            "serverless_request_count": 1,
            "serverless_execution_timeout_ms": 300_000,
            "serverless_flashboot": "FLASHBOOT",
            "http_timeout_seconds": 300,
            "serverless_non_compute_cost": {
                "estimated_cost_usd": 0.01,
                "maximum_cost_usd": 0.02,
                "covered_components": [
                    "container_disk",
                    "model_transfer",
                    "retry_allowance",
                ],
            },
            "pod_non_compute_cost": {
                "estimated_cost_usd": 0.01,
                "maximum_cost_usd": 0.02,
                "covered_components": [
                    "container_disk",
                    "model_transfer",
                    "retry_allowance",
                ],
            },
        }
    }
    profile = GPUProfile(
        id="ollama",
        name="Ollama",
        task_type="ollama",
        image_name="registry.example/ollama:sha",
        container_disk_gb=40,
        volume_gb=0,
        ports=["11434/http"],
        inference_port=11434,
        min_vram_gb=24,
    )
    harness._select_profile = lambda name: profile

    first = harness._get_ollama_lease_service()
    second = harness._get_ollama_lease_service()

    assert first is second
    assert first.repository.path == tmp_path / "leases.sqlite3"
    assert first.provider.deployment.serverless_non_compute_cost.maximum_cost_usd == (
        0.02
    )
    assert first.provider.deployment.pod_non_compute_cost.maximum_cost_usd == 0.02
