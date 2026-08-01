"""Contracts for RunPod profile-owned model defaults."""

from types import SimpleNamespace

import pytest

from kestrel_cloud_runpod.core import RunPodManagerCore
from kestrel_cloud_runpod.models import RunPodManagerError


class _ResumeHarness:
    resume_stopped_pod = RunPodManagerCore.resume_stopped_pod

    def __init__(self, profile):
        self.provider = SimpleNamespace(resume_pod=lambda pod_id, gpu_count: None)
        self._lock = None
        self._session = None
        self.wait_called = False
        self.profile = profile

    async def _wait_until_ready(self):
        self.wait_called = True


class _AsyncNullLock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_resume_stopped_pod_requires_profile_default_model():
    profile = SimpleNamespace(
        id="training",
        task_type="training",
        default_model=None,
        pod_type="a100",
    )
    harness = _ResumeHarness(profile)
    harness._lock = _AsyncNullLock()

    with pytest.raises(RunPodManagerError, match="has no default_model configured"):
        await harness.resume_stopped_pod(
            {"id": "pod-123", "gpuCount": 1}, profile, 3600
        )
