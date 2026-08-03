"""Compatibility-provider migration tests for the v2 client boundary."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from kestrel_cloud_runpod.manager import RunPodManager
from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    GPUOffer,
    GPUProfile,
    PodResource,
    PodStatus,
    RunPodAmbiguousResultError,
    RunPodManagerError,
    RunPodSession,
)
from kestrel_cloud_runpod.providers import DirectRunPodProvider
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupError,
    TrainingPodCleanupState,
)

MINIMAL_RUNPOD_CONFIG = {
    "manager": {"cloud_type": "SECURE"},
    "profiles": {
        "llm": {
            "name": "Test LLM",
            "image_name": "runpod/pytorch:latest",
            "min_vram_gb": 24,
            "max_cost_per_hr": 1.0,
        }
    },
}


def _profile() -> GPUProfile:
    return GPUProfile(
        id="image",
        name="Image",
        task_type="image",
        image_name="registry/image:sha",
        container_disk_gb=50,
        volume_gb=0,
        ports=["9000/http"],
        inference_port=9000,
        min_vram_gb=24,
        min_cuda_version="12.8",
        max_cost_per_hr=1.0,
        cloud=CloudType.SECURE,
        default_model="flux",
        env={"MODEL": "flux", "EMPTY": None},
    )


def _offer() -> GPUOffer:
    return GPUOffer(
        id="live-gpu-id",
        name="Live GPU",
        pool="BLACKWELL_24",
        manufacturer="NVIDIA",
        memory_gb=24,
        secure=True,
        community=False,
        secure_price_per_hr=0.69,
        community_price_per_hr=0.0,
        secure_max_count=1,
        community_max_count=0,
        availability=Availability.HIGH,
        availability_min_cuda_version="12.8",
    )


class _FakeControlClient:
    def __init__(self):
        self.catalog_kwargs = None
        self.create_request = None
        self.actions = []

    def list_gpus(self, **kwargs):
        self.catalog_kwargs = kwargs
        return (_offer(),)

    def create_pod(self, request):
        self.create_request = request
        return PodResource.from_dict(
            {
                "id": "pod-123",
                "name": request.name,
                "status": "PROVISIONING",
                "gpu": {"id": request.gpu_id, "count": request.gpu_count},
                "cost": 0.69,
            }
        )

    def pod_action(self, pod_id, action):
        self.actions.append((pod_id, action))
        if action == "terminate":
            return None
        return PodResource.from_dict(
            {
                "id": pod_id,
                "name": "pod",
                "status": "EXITED" if action == "stop" else "STARTING",
                "gpu": {"id": "live-gpu-id", "count": 1},
                "cost": 0.0,
            }
        )

    def iter_pod_logs(self, pod_id, *, tail, stream_window_seconds):
        assert pod_id == "pod-123"
        assert tail == 2
        assert stream_window_seconds == 2.0
        yield {"source": "container", "line": "first"}
        yield {"source": "system", "line": "second"}


def test_direct_provider_selects_from_live_catalog_and_records_placement():
    client = _FakeControlClient()
    provider = DirectRunPodProvider(api_key="", client=client)

    result = provider.start_pod(
        _profile(),
        {
            "name": "kestrel-image",
            "env_overrides": {"JOB_KIND": "selfie"},
        },
    )

    assert client.catalog_kwargs["min_cuda_version"] == "12.8"
    assert client.create_request.gpu_id == "live-gpu-id"
    assert client.create_request.env == {"MODEL": "flux", "JOB_KIND": "selfie"}
    assert result["_kestrel_placement"].offered_cost_per_hr == 0.69
    assert result["gpu"]["id"] == "live-gpu-id"


def test_direct_provider_fails_before_catalog_when_required_env_is_unset(
    monkeypatch,
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = _FakeControlClient()
    provider = DirectRunPodProvider(api_key="", client=client)
    profile = _profile()
    profile.env["HF_TOKEN"] = "${HF_TOKEN}"

    with pytest.raises(
        RunPodManagerError, match="unset environment variable 'HF_TOKEN'"
    ):
        provider.start_pod(profile, {"name": "kestrel-image"})

    assert client.catalog_kwargs is None
    assert client.create_request is None


def test_profile_rejects_persistent_volume_below_runpod_floor():
    with pytest.raises(ValueError, match="at least 10"):
        replace(_profile(), volume_gb=9)


def test_direct_provider_lifecycle_and_v2_logs():
    client = _FakeControlClient()
    provider = DirectRunPodProvider(api_key="", client=client)

    assert provider.stop_pod("pod-123")["status"] == "EXITED"
    assert provider.resume_pod("pod-123")["status"] == "STARTING"
    assert provider.terminate_pod("pod-123") == {
        "id": "pod-123",
        "status": "TERMINATED",
    }
    assert provider.get_logs("pod-123", tail=2) == "first\nsecond"
    assert client.actions == [
        ("pod-123", "stop"),
        ("pod-123", "start"),
        ("pod-123", "terminate"),
    ]


def test_private_cli_ssh_execution_has_clear_migration_error():
    provider = DirectRunPodProvider(api_key="", client=_FakeControlClient())
    with pytest.raises(
        RunPodManagerError, match="not available through Runpod REST v2"
    ):
        provider.exec_command("pod-123", "uname -a")


@pytest.mark.asyncio
async def test_manager_get_logs_uses_provider_v2_log_stream():
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    mock_provider.get_logs.return_value = "log line 1\nlog line 2"
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=MINIMAL_RUNPOD_CONFIG)
    manager._session = RunPodSession(
        pod_id="pod-123",
        task_profile="llm",
        model_name="test-model",
        status=PodStatus.READY,
        started_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        profile=manager.profiles["llm"],
        ttl_seconds=3600,
        pod_type="dedicated",
    )

    logs = await manager.get_logs(tail=50)

    mock_provider.get_logs.assert_called_with("pod-123", 50)
    assert logs == "log line 1\nlog line 2"


def test_legacy_profile_fields_fail_with_migration_guidance():
    legacy = {
        "profiles": {
            "llm": {
                "name": "Legacy",
                "image_name": "image",
                "gpu_type_id": "NVIDIA_H100",
                "cost_per_hr": 4.75,
            }
        }
    }
    with (
        patch.object(RunPodManager, "_build_provider"),
        pytest.raises(RunPodManagerError, match="live v2 catalog"),
    ):
        RunPodManager(config=legacy)


def test_manager_maps_v2_http_runtime_port_to_pod_proxy_url():
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=MINIMAL_RUNPOD_CONFIG)
    profile = manager.profiles["llm"]
    session = RunPodSession(
        pod_id="pod-123",
        profile=profile,
        task_profile="llm",
        model_name="qwen",
        pod_type="dedicated",
        status=PodStatus.PROVISIONING,
        ttl_seconds=3600,
        started_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    manager._update_session_from_runtime(
        session,
        {
            "id": "pod-123",
            "status": "RUNNING",
            "runtime": {
                "ports": [{"private": 8888, "public": None, "type": "http", "ip": None}]
            },
        },
    )

    assert session.status is PodStatus.READY
    assert session.backend_base_url == "https://pod-123-8888.proxy.runpod.net"
    assert session.inference_url == "https://pod-123-8888.proxy.runpod.net/v1"


def test_manager_redacts_echoed_environment_from_public_status():
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=MINIMAL_RUNPOD_CONFIG)
    session = RunPodSession(
        pod_id="pod-123",
        profile=manager.profiles["llm"],
        task_profile="llm",
        model_name="qwen",
        pod_type="dedicated",
        status=PodStatus.PROVISIONING,
        ttl_seconds=3600,
        started_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    manager._update_session_from_runtime(
        session,
        {
            "id": "pod-123",
            "status": "RUNNING",
            "env": {"HF_TOKEN": "hf-secret", "MODEL": "qwen"},
            "runtime": {"ports": []},
        },
    )

    assert "hf-secret" not in repr(session.to_dict())
    assert session.runtime["env"] == {
        "HF_TOKEN": "[REDACTED]",
        "MODEL": "[REDACTED]",
    }


@pytest.mark.parametrize(
    ("runpod_status", "expected"),
    [
        ("PROVISIONING", PodStatus.PROVISIONING),
        ("STARTING", PodStatus.PROVISIONING),
        ("RUNNING", PodStatus.READY),
        ("EXITED", PodStatus.OFFLINE),
        ("ERROR", PodStatus.ERROR),
        ("TERMINATED", PodStatus.OFFLINE),
    ],
)
def test_manager_maps_every_v2_pod_status(runpod_status, expected):
    assert RunPodManager._map_status(runpod_status) is expected


@pytest.mark.asyncio
async def test_manager_preserves_ambiguous_create_for_reconciliation():
    ambiguous = RunPodAmbiguousResultError(
        title="Runpod transport failed",
        detail="ReadTimeout",
        method="POST",
        resource="/pods",
    )
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    mock_provider.start_pod.side_effect = ambiguous
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=MINIMAL_RUNPOD_CONFIG)

    with pytest.raises(RunPodAmbiguousResultError) as raised:
        await manager.start_session("llm", model_name="qwen")

    assert raised.value is ambiguous
    assert raised.value.reconcile_required is True
    assert manager._session is None


@pytest.mark.asyncio
async def test_training_does_not_try_fallback_profile_after_ambiguous_create(tmp_path):
    ambiguous = RunPodAmbiguousResultError(
        title="Runpod transport failed",
        detail="ReadTimeout",
        method="POST",
        resource="/pods",
    )
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    mock_provider.start_pod.side_effect = ambiguous
    mock_provider.list_pods.return_value = []
    training_profile = {
        "name": "Training",
        "image_name": "registry/training:sha",
        "min_vram_gb": 48,
        "default_model": "flux-lora-trainer",
    }
    config = {
        "manager": {"cloud_type": "SECURE", "max_ttl_seconds": 3600},
        "training_pods": {
            "database_path": str(tmp_path / "training.sqlite3"),
            "poll_interval_seconds": 1,
            "orphan_timeout_seconds": 30,
        },
        "profiles": {
            "training": training_profile,
            "training-h100": {**training_profile, "name": "Training fallback"},
        },
    }
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=config)

    with pytest.raises(TrainingPodCleanupError) as raised:
        await manager.start_training_pod("companion-123")

    assert raised.value.pod_id is None
    assert raised.value.billing_risk is True
    mock_provider.start_pod.assert_called_once()


@pytest.mark.asyncio
async def test_training_uses_distinct_cleanup_token_for_safe_profile_fallback(tmp_path):
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    mock_provider.start_pod.side_effect = [
        RunPodManagerError("profile unavailable"),
        {"id": "pod-fallback"},
    ]
    mock_provider.get_status.return_value = {
        "id": "pod-fallback",
        "status": "RUNNING",
        "runtime": {"ports": [{"private": 8888, "type": "http"}]},
    }
    training_profile = {
        "name": "Training",
        "image_name": "registry/training:sha",
        "min_vram_gb": 48,
        "default_model": "flux-lora-trainer",
        "ports": ["8888/http"],
        "inference_port": 8888,
    }
    config = {
        "manager": {"cloud_type": "SECURE", "max_ttl_seconds": 3600},
        "training_pods": {
            "database_path": str(tmp_path / "training.sqlite3"),
            "poll_interval_seconds": 1,
            "orphan_timeout_seconds": 30,
        },
        "profiles": {
            "training": training_profile,
            "training-h100": {**training_profile, "name": "Training fallback"},
        },
    }
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=config)

    session = await manager.start_training_pod(
        "companion-123", cleanup_token="training:stable-request-0001"
    )

    assert session.training_cleanup_token != "training:stable-request-0001"
    assert session.training_cleanup_token.startswith("training:")
    first = manager.get_training_pod_lease("training:stable-request-0001")
    assert first.state.value == "released"
    assert first.cleanup_state is TrainingPodCleanupState.COMPLETE
    fallback = manager.get_training_pod_lease(session.training_cleanup_token)
    assert fallback.profile_id == "training-h100"
    assert fallback.state.value == "ready"
    public_fallback = fallback.to_public_dict()
    assert public_fallback["root_cleanup_token"] == "training:stable-request-0001"
    assert "backend_base_url" not in public_fallback
    assert "proxy.runpod.net" not in repr(public_fallback)
    assert mock_provider.start_pod.call_count == 2

    # Model death before the returned session reaches its caller: a fresh
    # manager has only the original token and the durable SQLite state.
    mock_provider.stop_pod.return_value = {
        "id": "pod-fallback",
        "status": "EXITED",
    }
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        restarted = RunPodManager(config=config)
    recovered_session = await restarted.start_training_pod(
        "companion-123", cleanup_token="training:stable-request-0001"
    )
    assert recovered_session.training_cleanup_token == session.training_cleanup_token
    assert mock_provider.start_pod.call_count == 2
    released_root = await restarted.release_training_pod(
        "training:stable-request-0001", reason="restart recovery"
    )
    released_fallback = restarted.get_training_pod_lease(session.training_cleanup_token)
    assert released_root.family_release_complete is True
    assert released_fallback.state.value == "released"
    assert restarted._session is None
    mock_provider.stop_pod.assert_called_once_with("pod-fallback")


@pytest.mark.asyncio
async def test_manager_termination_clears_session_only_after_v2_confirmation():
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=MINIMAL_RUNPOD_CONFIG)
    session = RunPodSession(
        pod_id="pod-123",
        profile=manager.profiles["llm"],
        task_profile="llm",
        model_name="qwen",
        pod_type="dedicated",
        status=PodStatus.READY,
        ttl_seconds=3600,
        started_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    manager._session = session

    await manager.terminate_session(session)

    mock_provider.terminate_pod.assert_called_once_with("pod-123")
    assert manager._session is None
    assert session.status is PodStatus.OFFLINE


@pytest.mark.asyncio
async def test_manager_termination_failure_preserves_reconciliation_handle():
    mock_provider = MagicMock(spec=DirectRunPodProvider)
    mock_provider.terminate_pod.side_effect = RunPodManagerError("provider unavailable")
    with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
        manager = RunPodManager(config=MINIMAL_RUNPOD_CONFIG)
    session = RunPodSession(
        pod_id="pod-123",
        profile=manager.profiles["llm"],
        task_profile="llm",
        model_name="qwen",
        pod_type="dedicated",
        status=PodStatus.READY,
        ttl_seconds=3600,
        started_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    manager._session = session

    with pytest.raises(RunPodManagerError, match="provider unavailable"):
        await manager.terminate_session(session)

    assert manager._session is session
    assert session.status is PodStatus.READY
