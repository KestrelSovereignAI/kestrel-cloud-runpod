"""Manager integration for durable training capacity and workload recovery."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from training_test_support import FakeTrainingProvider, MutableClock, training_service

from kestrel_cloud_runpod.manager import RunPodManager
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.providers import DirectRunPodProvider
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupError,
    TrainingPodState,
)


class FakeWorkloadClient:
    def __init__(self, *, fail_train: bool = False, fail_status: bool = False) -> None:
        self.fail_train = fail_train
        self.fail_status = fail_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def get(self, url: str):
        if url.endswith("/ready"):
            return _response(url, 200, {"gpu": "test", "gpu_memory_gb": 24})
        if url.endswith("/current-job"):
            return _response(url, 200, {"current_job": None})
        if "/status/" in url:
            if self.fail_status:
                raise httpx.ConnectError("status unavailable")
            return _response(url, 200, {"status": "completed", "progress": 1.0})
        if "/download/" in url:
            return httpx.Response(
                200, content=b"weights", request=httpx.Request("GET", url)
            )
        raise AssertionError(f"unexpected GET {url}")

    async def post(self, url: str, **_):
        if url.endswith("/train"):
            if self.fail_train:
                raise httpx.ConnectError("submission unavailable")
            return _response(url, 200, {"job_id": "job-1"})
        if "/cancel/" in url:
            return _response(url, 200, {"status": "cancelled"})
        raise AssertionError(f"unexpected POST {url}")


def _response(url: str, status: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("GET", url),
    )


def _manager(tmp_path: Path, provider: FakeTrainingProvider) -> RunPodManager:
    direct = MagicMock(spec=DirectRunPodProvider)
    config = {
        "manager": {"cloud_type": "SECURE", "max_ttl_seconds": 3600},
        "profiles": {
            "training": {
                "name": "Training",
                "image_name": "registry/training@sha256:" + "a" * 64,
                "min_vram_gb": 24,
                "default_model": "flux-lora-trainer",
                "persistent_pod_id": "pod-training-1",
                "ports": ["8888/http"],
                "inference_port": 8888,
            }
        },
    }
    with patch.object(RunPodManager, "_build_provider", return_value=direct):
        manager = RunPodManager(config=config)
    clock = MutableClock()
    manager.set_training_pod_lease_service(training_service(tmp_path, clock, provider))
    return manager


@pytest.mark.asyncio
async def test_manager_session_tracks_job_result_and_confirmed_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    client = FakeWorkloadClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)

    session = await manager.start_training_pod(
        "companion-1", cleanup_token="training:manager-token-0001"
    )
    job_id = await manager.submit_training_job(
        session, b"\x89PNG\r\n\x1a\nimage", "companion-1"
    )
    assert await manager.poll_training_status(session, job_id) == {
        "status": "completed",
        "progress": 1.0,
    }
    assert await manager.download_lora(session, job_id) == b"weights"
    stopped = await manager.stop_session()

    assert session.training_cleanup_token == "training:manager-token-0001"
    assert stopped["training_cleanup_state"] == "complete"
    lease = manager.get_training_pod_lease(session.training_cleanup_token)
    assert lease.state is TrainingPodState.RELEASED
    assert provider.stop_calls == ["pod-training-1"]


@pytest.mark.asyncio
async def test_submission_failure_stops_capacity_and_preserves_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_: FakeWorkloadClient(fail_train=True)
    )
    session = await manager.start_training_pod(
        "companion-1", cleanup_token="training:submit-fail-0001"
    )

    with pytest.raises(RunPodManagerError, match="capacity was stopped"):
        await manager.submit_training_job(session, b"avatar", "companion-1")

    lease = manager.get_training_pod_lease("training:submit-fail-0001")
    assert lease.state is TrainingPodState.RELEASED
    assert provider.stop_calls == ["pod-training-1"]


@pytest.mark.asyncio
async def test_status_failure_keeps_job_and_cleanup_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    client = FakeWorkloadClient(fail_status=True)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    session = await manager.start_training_pod(
        "companion-1", cleanup_token="training:status-fail-0001"
    )
    job_id = await manager.submit_training_job(session, b"avatar", "companion-1")

    with pytest.raises(RunPodManagerError, match="reconcile cleanup token"):
        await manager.poll_training_status(session, job_id)

    lease = manager.get_training_pod_lease("training:status-fail-0001")
    assert lease.provider_pod_id == "pod-training-1"
    assert lease.provider_job_id == "job-1"
    assert lease.state is TrainingPodState.JOB_SUBMITTED
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_cancel_stops_pod_and_stop_failure_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    provider.stop_error = RunPodManagerError("stop failed")
    manager = _manager(tmp_path, provider)
    client = FakeWorkloadClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    session = await manager.start_training_pod(
        "companion-1", cleanup_token="training:cancel-fail-0001"
    )
    job_id = await manager.submit_training_job(session, b"avatar", "companion-1")

    with pytest.raises(TrainingPodCleanupError) as raised:
        await manager.cancel_training_job(session, job_id)

    assert raised.value.cleanup_token == "training:cancel-fail-0001"
    lease = manager.get_training_pod_lease(raised.value.cleanup_token)
    assert lease.provider_pod_id == "pod-training-1"
    assert lease.state is TrainingPodState.RECONCILE_REQUIRED
