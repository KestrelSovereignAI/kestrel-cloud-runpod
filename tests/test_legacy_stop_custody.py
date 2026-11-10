"""Legacy stop must not manufacture ownership, cleanup, or metering evidence."""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from scripts.check_runpod_openapi import load_openapi

from kestrel_cloud_runpod.feature import RunPodFeature
from kestrel_cloud_runpod.manager import RunPodManager
from kestrel_cloud_runpod.models import PodStatus, RunPodManagerError, RunPodSession
from kestrel_cloud_runpod.training_contracts import TrainingPodCleanupState, TrainingPodLifecycleError
from pod_capacity_test_support import complete_pod_payload, native_pod_action_harness
from training_test_support import (
    MutableClock,
    original_training_session,
    training_service,
)


def _native_manager(tmp_path, monkeypatch, direct, provider, pod_id):
    # Dependency injection selects the actual owned-loopback HTTP transport.
    # No public admission override, real cloud create, or model evidence.
    monkeypatch.setattr(RunPodManager, "_build_provider", lambda self: direct)
    manager = RunPodManager(config={
        "manager": {"cloud_type": "SECURE"},
        "profiles": {"training": {
            "name": "Historical fixture", "image_name": "registry/trainer:fixture",
            "min_vram_gb": 24, "persistent_pod_id": pod_id,
            "default_model": "fixture", "ports": ["8888/http"],
            "inference_port": 8888,
        }},
    })
    clock = MutableClock()
    clock.value = datetime.now(UTC)
    manager.set_training_pod_lease_service(
        training_service(tmp_path, clock, provider, profiles=manager.profiles)
    )
    return manager


def _feature(manager):
    feature = RunPodFeature(SimpleNamespace())
    feature.manager = manager
    feature.disabled = False
    feature._start_lock = asyncio.Lock()
    return feature


def _ready_payload(pod_id):
    document = load_openapi(
        (Path(__file__).resolve().parents[1] / "vendor/runpod-v2-openapi.yaml")
        .read_text(encoding="utf-8")
    )
    response = document["paths"]["/v2/pods/{id}"]["get"]["responses"]["200"]
    ready = deepcopy(response["content"]["application/json"]["examples"]["pod"]["value"])
    ready["id"] = pod_id
    ready["runtime"] = {"uptime": 1, "ports": [{
        "private": 8888, "public": 18888, "type": "http", "ip": "127.0.0.1",
    }]}
    Draft202012Validator({
        **document, "$ref": "#/components/schemas/Pod",
    }).validate(ready)
    return ready


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [
    PodStatus.READY, PodStatus.TERMINATING, PodStatus.ERROR, PodStatus.OFFLINE,
])
async def test_ownerless_stop_refuses_and_preserves_known_identity(
    tmp_path, monkeypatch, status,
):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        now = datetime.now(UTC)
        session = RunPodSession(
            pod_id=state["pod_id"], profile=manager.profiles["training"],
            task_profile="training", model_name="fixture", pod_type="dedicated",
            status=status, ttl_seconds=600,
            started_at=now, expires_at=now + timedelta(seconds=600),
            companion_id="companion-1", user_id="user-1",
        )
        manager._session = session
        with pytest.raises(RunPodManagerError, match="Ownerless Pod termination"):
            await manager.stop_session()

        result = await _feature(manager)._stop()
        assert result.status == "error"
        assert result.data["session_before_stop"]["pod_id"] == state["pod_id"]
        assert "may still be billing" in result.data["warning"]
        assert manager._session is session and session.status is status
        assert state["calls"] == []


@pytest.mark.asyncio
async def test_wall_clock_metering_registration_refuses_without_callback_or_io(
    tmp_path, monkeypatch,
):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        calls = []

        async def callback(**values):
            calls.append(values)

        with pytest.raises(RunPodManagerError, match="Wall-clock.*retired"):
            manager.set_metering_callback(callback)
        assert calls == state["calls"] == []


@pytest.mark.asyncio
async def test_borrowed_release_never_claims_provider_stopped(tmp_path, monkeypatch):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        ready = _ready_payload(state["pod_id"])
        # Both acquire's observation and readiness read use the same exact
        # declared v2 response. The route is never contacted or called a GPU.
        state["readback_payloads"] = [ready, ready, ready]
        session = await original_training_session(
            manager, "companion-1", cleanup_token="training:borrowed-stop-0001",
        )
        result = await _feature(manager)._stop()

        lease = manager.get_training_pod_lease(session.training_cleanup_token)
        assert lease.cleanup_state is TrainingPodCleanupState.NOT_OWNED
        assert lease.provider_pod_id == state["pod_id"]
        assert not any(call[0] == "POST" for call in state["calls"])
        assert result.status == "ok"
        assert result.data["session"]["training_cleanup_state"] == "not_owned"
        assert "not stopped" in result.confirmation.lower()
        assert "no-op" not in result.confirmation.lower()


async def _owned_session(manager, state):
    ready = _ready_payload(state["pod_id"])
    stopped = deepcopy(ready)
    stopped.update(status="EXITED", runtime=None, actions=["start", "terminate"])
    state["readback_payloads"] = [stopped, ready]
    return await original_training_session(
        manager, "companion-1", cleanup_token="training:owned-native-stop-0001",
    )


@pytest.mark.asyncio
async def test_original_confirmed_stop_is_not_mislabeled_noop(tmp_path, monkeypatch):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        session = await _owned_session(manager, state)
        terminal = _ready_payload(state["pod_id"])
        terminal.update(status="TERMINATED", runtime=None, actions=[])
        state["readback_payloads"] = [terminal]
        state["calls"].clear()
        result = await _feature(manager)._stop()

        assert result.status == "ok"
        assert "cleanup confirmed" in result.confirmation
        assert "no-op" not in result.confirmation
        assert result.data["session"]["training_cleanup_state"] == "complete"
        assert manager._session is None
        assert manager.get_training_pod_lease(session.training_cleanup_token).family_release_complete
        assert state["calls"] == [
            ("POST", f"/v2/pods/{state['pod_id']}/action", {"action": "stop"}),
            ("GET", f"/v2/pods/{state['pod_id']}", None),
        ]


@pytest.mark.asyncio
async def test_no_session_stop_has_no_native_io(tmp_path, monkeypatch):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        result = await _feature(manager)._stop()
        assert result.status == "ok" and "no-op" in result.confirmation
        assert state["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["stop", "terminate", "tool"])
@pytest.mark.parametrize("borrowed", [False, True])
async def test_completed_original_release_is_idempotent_through_retained_session(
    tmp_path, monkeypatch, entrypoint, borrowed,
):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        if borrowed:
            ready = _ready_payload(state["pod_id"])
            state["readback_payloads"] = [ready, ready]
            session = await original_training_session(
                manager, "companion-1", cleanup_token="training:completed-borrow-0001",
            )
        else:
            session = await _owned_session(manager, state)
            terminal = _ready_payload(state["pod_id"])
            terminal.update(status="TERMINATED", runtime=None, actions=[])
            state["readback_payloads"] = [terminal]

        # A separate canonical controller completes the original obligation.
        # It does not update this manager's process-local projection.
        service = manager._get_training_pod_lease_service()
        original = await service.release(session.training_cleanup_token, reason="external original release")
        assert original.is_terminal and manager._session is session
        state["calls"].clear()
        # Contain even a deliberately broken workload-fence request to this
        # owned peer; no test may contact the vendor's proxy hostname.
        session.backend_base_url = direct.client.transport.base_url
        assert manager._session_lease(session) == original
        with pytest.raises(TrainingPodLifecycleError, match="heartbeat"):
            await manager.get_current_job(session)
        assert state["calls"] == []

        if entrypoint == "stop":
            result = await manager.stop_session()
            assert result["training_cleanup_state"] == original.cleanup_state.value
        elif entrypoint == "terminate":
            assert await manager.terminate_session(session) == original
        else:
            result = await _feature(manager)._stop()
            assert result.status == "ok"
            assert result.data["session"]["training_cleanup_state"] == original.cleanup_state.value
        assert manager._session is None
        assert manager.get_training_pod_lease(original.cleanup_token) == original
        assert state["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["pod", "owner", "kind"])
async def test_terminal_cleanup_still_requires_original_immutable_binding(
    tmp_path, monkeypatch, mismatch,
):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        session = await _owned_session(manager, state)
        terminal = _ready_payload(state["pod_id"])
        terminal.update(status="TERMINATED", runtime=None, actions=[])
        state["readback_payloads"] = [terminal]
        service = manager._get_training_pod_lease_service()
        original = await service.release(session.training_cleanup_token, reason="external original release")
        state["calls"].clear()
        if mismatch == "pod":
            session.pod_id = "foreign-pod"
        elif mismatch == "owner":
            session.companion_id = "foreign-owner"
        else:
            session.task_profile = "llm"
        with pytest.raises(RunPodManagerError, match="no longer matches"):
            await manager.stop_session()
        assert manager._session is session
        assert manager.get_training_pod_lease(original.cleanup_token) == original
        assert state["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("readback", [
    "running", "foreign", "authorization", "structural", "transport",
])
async def test_original_stop_failure_retains_same_sqlite_lease_for_restart(
    tmp_path, monkeypatch, readback,
):
    with native_pod_action_harness() as (state, direct, provider):
        manager = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        session = await _owned_session(manager, state)
        token = session.training_cleanup_token
        state["readback"] = readback
        state["calls"].clear()
        result = await _feature(manager)._stop()

        assert result.status == "error"
        assert manager._session is session
        assert "may still be billing" in result.data["warning"]
        assert result.data["session_before_stop"]["pod_id"] == state["pod_id"]
        lease = manager.get_training_pod_lease(token)
        assert lease.provider_pod_id == state["pod_id"]
        assert lease.cleanup_state in {
            TrainingPodCleanupState.PENDING, TrainingPodCleanupState.RETRYABLE_FAILURE,
        }
        assert lease.family_release_requested and not lease.family_release_complete
        assert lease.terminated_at is None

        restarted = _native_manager(tmp_path, monkeypatch, direct, provider, state["pod_id"])
        assert restarted.get_training_pod_lease(token) == lease
        state["readback"] = "absent"
        released = await restarted.release_training_pod(token, reason="explicit original-token recovery")
        assert released.cleanup_state is TrainingPodCleanupState.COMPLETE
        assert released.family_release_complete
        assert not any(call[1] == "/v2/pods" for call in state["calls"])
