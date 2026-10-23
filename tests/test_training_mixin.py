"""Manager integration for durable training capacity and workload recovery."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pod_capacity_test_support import (
    execution_control_carrier,
    record_control_injection,
)
from training_test_support import FakeTrainingProvider, MutableClock, training_service

from kestrel_cloud_runpod.manager import RunPodManager
from kestrel_cloud_runpod.models import (
    PodStatus,
    RunPodLegacyAdmissionRetiredError,
    RunPodManagerError,
    RunPodSession,
)
from kestrel_cloud_runpod.providers import DirectRunPodProvider
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodLifecycleError,
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
async def test_manager_stop_preserves_metering_execution_control(
    tmp_path: Path, record_property
):
    provider = FakeTrainingProvider()
    manager = _manager(tmp_path, provider)
    carrier, control, explicit = execution_control_carrier(
        RunPodManagerError("metering outcome remains unknown")
    )
    profile = manager.profiles["training"]
    started_at = datetime.now(UTC)
    manager._session = RunPodSession(
        pod_id="pod-metered-1",
        profile=profile,
        task_profile="training",
        model_name="trainer",
        pod_type=None,
        status=PodStatus.READY,
        ttl_seconds=600,
        started_at=started_at,
        expires_at=started_at + timedelta(seconds=600),
        companion_id="companion-1",
        user_id="user-1",
    )

    async def controlled_metering(**_kwargs):
        record_control_injection(record_property, "manager-stop-metering")
        raise carrier

    manager.provider.stop_pod.return_value = {"id": "pod-metered-1"}
    manager.set_metering_callback(controlled_metering)

    with pytest.raises(type(carrier)) as raised:
        await manager.stop_session()

    assert raised.value is carrier
    assert carrier.__cause__ is explicit is control
    manager.provider.stop_pod.assert_called_once_with("pod-metered-1")
    assert manager._session is None


@pytest.mark.asyncio
async def test_manager_stop_retains_ordinary_metering_nonfatal_contract(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    manager = _manager(tmp_path, provider)
    profile = manager.profiles["training"]
    started_at = datetime.now(UTC)
    manager._session = RunPodSession(
        pod_id="pod-metered-ordinary-1",
        profile=profile,
        task_profile="training",
        model_name="trainer",
        pod_type=None,
        status=PodStatus.READY,
        ttl_seconds=600,
        started_at=started_at,
        expires_at=started_at + timedelta(seconds=600),
        companion_id="companion-1",
        user_id="user-1",
    )
    ordinary = RuntimeError("ordinary metering failure")

    async def failed_metering(**_kwargs):
        raise ordinary

    manager.provider.stop_pod.return_value = {"id": "pod-metered-ordinary-1"}
    manager.set_metering_callback(failed_metering)

    result = await manager.stop_session()

    assert result["active"] is False
    assert result["pod_id"] == "pod-metered-ordinary-1"
    manager.provider.stop_pod.assert_called_once_with("pod-metered-ordinary-1")
    assert manager._session is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "reconcile-status",
        "readiness-connect",
        "readiness-timeout",
        "profile-fallback",
        "inference-start",
        "model-readiness",
        "model-readiness-cleanup",
        "submit-connect",
        "submit-http-status",
        "submit-timeout",
        "submit-invalid",
        "submit-invalid-cleanup",
        "missing-job-id-cleanup",
        "submission-recovery-readback",
        "submission-recovery-cleanup",
        "current-job",
        "cancel",
        "cancel-cleanup",
        "clear",
        "status",
        "download",
    ],
)
async def test_training_workload_preserves_exact_execution_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    record_property,
):
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    if boundary == "profile-fallback":
        error = TrainingPodLifecycleError(
            "controlled profile acquisition",
            cleanup_token="training:profile-fallback:0001",
            pod_id=None,
            cleanup_state=TrainingPodCleanupState.NOT_OWNED,
            billing_risk=False,
        )
    elif boundary.endswith("-cleanup"):
        error = TrainingPodCleanupError(
            "controlled cleanup",
            cleanup_token=f"training:{boundary}:0001",
            pod_id="pod-training-1",
            cleanup_state=TrainingPodCleanupState.RETRYABLE_FAILURE,
            billing_risk=True,
        )
    elif boundary in {"model-readiness", "model-readiness-cleanup", "inference-start"}:
        error = RunPodManagerError("model readiness remains unknown")
    elif boundary == "submit-http-status":
        request_value = httpx.Request("POST", "https://training.invalid/train")
        error = httpx.HTTPStatusError(
            "training submission remains unknown",
            request=request_value,
            response=httpx.Response(503, request=request_value),
        )
    elif boundary in {"readiness-timeout", "submit-timeout"}:
        error = httpx.ReadTimeout(f"training {boundary} remains unknown")
    elif boundary == "submit-invalid":
        error = ValueError("training submission response remains unknown")
    else:
        error = httpx.ConnectError(f"training {boundary} remains unknown")
    carrier, control, explicit = execution_control_carrier(
        error,
        grouped=boundary in {"submission-recovery-readback", "profile-fallback"},
    )
    calls: list[tuple[str, str]] = []
    durable_at_injection = None
    family_at_injection = None
    calls_at_injection = None

    class ControlledClient(FakeWorkloadClient):
        async def get(self, url: str):
            nonlocal calls_at_injection, durable_at_injection
            calls.append(("GET", url))
            if (
                boundary == "reconcile-status"
                or (boundary.startswith("readiness-") and url.endswith("/ready"))
                or boundary == "submission-recovery-readback"
                and url.endswith("/current-job")
                or boundary == "current-job"
                and url.endswith("/current-job")
                or boundary == "status"
                and "/status/" in url
                or boundary == "download"
                and "/download/" in url
            ):
                durable_at_injection = runtime.repository.get(
                    session.training_cleanup_token
                )
                assert durable_at_injection is not None
                calls_at_injection = tuple(calls)
                record_control_injection(record_property, boundary)
                raise carrier
            return _response(url, 200, {"current_job": None})

        async def post(self, url: str, **_kwargs):
            nonlocal calls_at_injection, durable_at_injection
            calls.append(("POST", url))
            if boundary in {
                "submit-connect",
                "submit-http-status",
                "submit-timeout",
                "submit-invalid",
            }:
                durable_at_injection = runtime.repository.get(
                    session.training_cleanup_token
                )
                assert durable_at_injection is not None
                calls_at_injection = tuple(calls)
                record_control_injection(record_property, boundary)
                raise carrier
            if boundary in {
                "submission-recovery-readback",
                "submission-recovery-cleanup",
            }:
                raise httpx.ConnectError("ordinary ambiguous training submission")
            if boundary == "submit-invalid-cleanup":
                raise ValueError("ordinary invalid submission")
            if boundary == "cancel" and "/cancel/" in url:
                durable_at_injection = runtime.repository.get(
                    session.training_cleanup_token
                )
                assert durable_at_injection is not None
                calls_at_injection = tuple(calls)
                record_control_injection(record_property, boundary)
                raise carrier
            if boundary == "cancel-cleanup" and "/cancel/" in url:
                raise httpx.ConnectError("ordinary failed cancellation")
            if boundary == "clear" and url.endswith("/clear-current-job"):
                durable_at_injection = runtime.repository.get(
                    session.training_cleanup_token
                )
                assert durable_at_injection is not None
                calls_at_injection = tuple(calls)
                record_control_injection(record_property, boundary)
                raise carrier
            if boundary == "missing-job-id-cleanup":
                return _response(url, 200, {})
            return await super().post(url)

    client = ControlledClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    if boundary == "profile-fallback":
        manager.profiles["training"] = replace(
            manager.profiles["training"], persistent_pod_id=None
        )
        manager.profiles["training-h100"] = replace(
            manager.profiles["training"], id="training-h100"
        )
        manager.find_stopped_pod = AsyncMock(return_value=None)
        attempts = []

        async def controlled_acquire(**kwargs):
            attempts.append(kwargs["profile"].id)
            record_control_injection(record_property, boundary)
            raise carrier

        manager._acquire_training_session = controlled_acquire
        operation = lambda: manager.start_training_pod(
            "companion-1", cleanup_token="training:profile-fallback:0001"
        )
        durable_before = None
        stop_calls = list(provider.stop_calls)
    elif boundary == "inference-start":

        async def controlled_start_session(**_kwargs):
            record_control_injection(record_property, boundary)
            raise carrier

        manager.start_session = controlled_start_session
        operation = lambda: manager.start_inference_pod("companion-1")
        durable_before = None
        stop_calls = list(provider.stop_calls)
    else:
        session = await manager.start_training_pod(
            "companion-1", cleanup_token=f"training:{boundary}:0001"
        )
        runtime = manager._get_training_pod_lease_service()
        if boundary == "reconcile-status":
            runtime.record_job(session.training_cleanup_token, "job-control")
            lease = runtime.repository.get(session.training_cleanup_token)
            assert lease is not None
            operation = lambda: manager._reconcile_training_workload_status(lease)
        elif boundary.startswith("readiness-"):
            base = datetime(2026, 8, 1, tzinfo=UTC)
            instants = iter((base, base, base + timedelta(seconds=2)))

            class AdvancingDateTime:
                @staticmethod
                def now(_timezone):
                    return next(instants)

            monkeypatch.setattr(
                "kestrel_cloud_runpod.training.datetime", AdvancingDateTime
            )
            operation = lambda: manager._wait_for_training_ready(
                session, timeout=1, poll_interval=0
            )
        elif boundary in {"model-readiness", "model-readiness-cleanup"}:

            async def controlled_readiness(*_args, **_kwargs):
                if boundary == "model-readiness-cleanup":
                    raise RunPodManagerError("ordinary model readiness failure")
                record_control_injection(record_property, boundary)
                raise carrier

            manager._wait_for_training_ready = controlled_readiness
            operation = lambda: manager.submit_training_job(
                session, b"avatar", "companion-1"
            )
        elif (
            boundary.startswith("submit-")
            or boundary.startswith("submission-")
            or boundary == "missing-job-id-cleanup"
        ):
            operation = lambda: manager.submit_training_job(
                session,
                b"avatar",
                "companion-1",
                wait_for_model_ready=False,
            )
        elif boundary == "current-job":
            operation = lambda: manager.get_current_job(session)
        elif boundary in {"cancel", "cancel-cleanup"}:
            operation = lambda: manager.cancel_training_job(session, "job-control")
        elif boundary == "clear":
            operation = lambda: manager.clear_current_job(session)
        elif boundary == "status":
            operation = lambda: manager.poll_training_status(session, "job-control")
        else:
            operation = lambda: manager.download_lora(session, "job-control")

        if boundary in {
            "model-readiness-cleanup",
            "submit-invalid-cleanup",
            "missing-job-id-cleanup",
            "submission-recovery-cleanup",
            "cancel-cleanup",
        }:

            async def controlled_release(*_args, **_kwargs):
                nonlocal calls_at_injection, durable_at_injection, family_at_injection
                durable_at_injection = runtime.repository.get(
                    session.training_cleanup_token
                )
                assert durable_at_injection is not None
                family_at_injection = runtime.repository.list_cleanup_family(
                    durable_at_injection.root_cleanup_token
                )
                calls_at_injection = tuple(calls)
                record_control_injection(record_property, boundary)
                raise carrier

            runtime.release = controlled_release

        durable_before = runtime.repository.get(session.training_cleanup_token)
        assert durable_before is not None
        stop_calls = list(provider.stop_calls)
    with pytest.raises(type(carrier)) as raised:
        await operation()

    assert raised.value is carrier
    assert carrier.__cause__ is explicit
    if isinstance(explicit, BaseExceptionGroup):
        assert explicit.exceptions[0] is control
    else:
        assert explicit is control
    assert provider.stop_calls == stop_calls
    if boundary == "profile-fallback":
        assert attempts == ["training"]
    elif durable_at_injection is not None:
        durable_after = runtime.repository.get(durable_at_injection.cleanup_token)
        assert durable_after == durable_at_injection
        assert durable_before is not None
        if boundary.endswith("-cleanup"):
            assert family_at_injection is not None
            assert (
                runtime.repository.list_cleanup_family(
                    durable_at_injection.root_cleanup_token
                )
                == family_at_injection
            )
            if boundary == "cancel-cleanup":
                # Cancellation intent and the ordinary failed HTTP observation
                # are both durably recorded before cleanup is attempted.
                assert durable_at_injection.revision == durable_before.revision + 2
                assert durable_at_injection.last_provider_error == "ConnectError"
                assert durable_at_injection.state is TrainingPodState.CANCEL_REQUESTED
                assert (
                    durable_at_injection.cleanup_state is durable_before.cleanup_state
                )
                assert (
                    durable_at_injection.stop_attempts == durable_before.stop_attempts
                )
            else:
                assert durable_at_injection.state is TrainingPodState.READY
                assert (
                    durable_at_injection.cleanup_state
                    is TrainingPodCleanupState.NOT_REQUESTED
                )
        elif boundary in {
            "readiness-connect",
            "readiness-timeout",
            "current-job",
            "clear",
            "status",
            "download",
        }:
            assert durable_at_injection.revision == durable_before.revision + 1
            assert durable_at_injection.state is durable_before.state
            assert durable_at_injection.cleanup_state is durable_before.cleanup_state
            assert durable_at_injection.stop_attempts == durable_before.stop_attempts
        elif boundary == "cancel":
            assert durable_at_injection.revision == durable_before.revision + 1
            assert durable_at_injection.state is TrainingPodState.CANCEL_REQUESTED
            assert durable_at_injection.cleanup_state is durable_before.cleanup_state
            assert durable_at_injection.stop_attempts == durable_before.stop_attempts
        else:
            assert durable_at_injection == durable_before
    elif durable_before is not None and not boundary.endswith("-cleanup"):
        assert runtime.repository.get(durable_before.cleanup_token) == durable_before
    if boundary == "submission-recovery-readback":
        assert [method for method, _url in calls] == ["POST", "GET"]
    if calls_at_injection is not None:
        assert tuple(calls) == calls_at_injection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "reconcile-status",
        "readiness-connect",
        "readiness-timeout",
        "inference-start",
        "model-readiness",
        "model-readiness-cleanup",
        "submit-http-status",
        "submit-timeout",
        "submit-invalid",
        "submit-invalid-cleanup",
        "missing-job-id-cleanup",
        "submission-recovery-readback",
        "submission-recovery-cleanup",
        "current-job",
        "cancel",
        "cancel-cleanup",
        "clear",
        "download",
    ],
)
async def test_training_workload_retains_ordinary_normalization_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    calls: list[tuple[str, str]] = []
    if boundary == "submit-http-status":
        request_value = httpx.Request("POST", "https://training.invalid/train")
        ordinary_error: BaseException = httpx.HTTPStatusError(
            "ordinary training submission failure",
            request=request_value,
            response=httpx.Response(503, request=request_value),
        )
    elif boundary in {"readiness-timeout", "submit-timeout"}:
        ordinary_error = httpx.ReadTimeout(f"ordinary {boundary} failure")
    elif boundary in {"submit-invalid", "submit-invalid-cleanup"}:
        ordinary_error = ValueError("ordinary invalid submission")
    elif boundary in {
        "model-readiness",
        "model-readiness-cleanup",
        "inference-start",
    }:
        ordinary_error = RunPodManagerError(f"ordinary {boundary} failure")
    else:
        ordinary_error = httpx.ConnectError(f"ordinary {boundary} failure")
    cleanup_error = TrainingPodCleanupError(
        "ordinary cleanup failure",
        cleanup_token=f"training:{boundary}:0001",
        pod_id="pod-training-1",
        cleanup_state=TrainingPodCleanupState.RETRYABLE_FAILURE,
        billing_risk=True,
    )

    class OrdinaryClient(FakeWorkloadClient):
        async def get(self, url: str):
            calls.append(("GET", url))
            if (
                boundary == "reconcile-status"
                or boundary.startswith("readiness-")
                or boundary == "submission-recovery-readback"
                and url.endswith("/current-job")
                or boundary == "current-job"
                and url.endswith("/current-job")
                or boundary == "download"
                and "/download/" in url
            ):
                raise ordinary_error
            return _response(url, 200, {"current_job": None})

        async def post(self, url: str, **_kwargs):
            calls.append(("POST", url))
            if boundary in {
                "submit-http-status",
                "submit-timeout",
                "submit-invalid",
                "submit-invalid-cleanup",
                "submission-recovery-readback",
                "submission-recovery-cleanup",
            }:
                raise ordinary_error
            if boundary in {"cancel", "cancel-cleanup"} and "/cancel/" in url:
                raise ordinary_error
            if boundary == "clear" and url.endswith("/clear-current-job"):
                raise ordinary_error
            if boundary == "missing-job-id-cleanup":
                return _response(url, 200, {})
            return await super().post(url)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: OrdinaryClient())

    if boundary == "inference-start":
        manager.start_session = AsyncMock(side_effect=ordinary_error)
        assert await manager.start_inference_pod("companion-1") is None
        assert provider.stop_calls == []
        return

    session = await manager.start_training_pod(
        "companion-1", cleanup_token=f"training:{boundary}:0001"
    )
    runtime = manager._get_training_pod_lease_service()
    durable_at_release = None
    family_at_release = None
    if boundary == "reconcile-status":
        runtime.record_job(session.training_cleanup_token, "job-ordinary")
        lease = runtime.repository.get(session.training_cleanup_token)
        assert lease is not None
        operation = lambda: manager._reconcile_training_workload_status(lease)
    elif boundary.startswith("readiness-"):
        base = datetime(2026, 8, 1, tzinfo=UTC)
        instants = iter((base, base, base + timedelta(seconds=2)))

        class AdvancingDateTime:
            @staticmethod
            def now(_timezone):
                return next(instants)

        monkeypatch.setattr("kestrel_cloud_runpod.training.datetime", AdvancingDateTime)
        operation = lambda: manager._wait_for_training_ready(
            session, timeout=1, poll_interval=0
        )
    elif boundary in {"model-readiness", "model-readiness-cleanup"}:

        async def ordinary_readiness(*_args, **_kwargs):
            raise ordinary_error

        manager._wait_for_training_ready = ordinary_readiness
        operation = lambda: manager.submit_training_job(
            session, b"avatar", "companion-1"
        )
    elif (
        boundary.startswith("submit-")
        or boundary.startswith("submission-")
        or boundary == "missing-job-id-cleanup"
    ):
        operation = lambda: manager.submit_training_job(
            session,
            b"avatar",
            "companion-1",
            wait_for_model_ready=False,
        )
    elif boundary == "current-job":
        operation = lambda: manager.get_current_job(session)
    elif boundary in {"cancel", "cancel-cleanup"}:
        operation = lambda: manager.cancel_training_job(session, "job-ordinary")
    elif boundary == "clear":
        operation = lambda: manager.clear_current_job(session)
    else:
        operation = lambda: manager.download_lora(session, "job-ordinary")

    if boundary.endswith("-cleanup"):

        async def ordinary_release(*_args, **_kwargs):
            nonlocal durable_at_release, family_at_release
            durable_at_release = runtime.repository.get(session.training_cleanup_token)
            assert durable_at_release is not None
            family_at_release = runtime.repository.list_cleanup_family(
                durable_at_release.root_cleanup_token
            )
            raise cleanup_error

        runtime.release = ordinary_release

    if boundary == "model-readiness":
        expected_error = ordinary_error
    elif boundary.endswith("-cleanup"):
        expected_error = cleanup_error
    else:
        expected_error = None

    if expected_error is not None:
        with pytest.raises(type(expected_error)) as raised:
            await operation()
        assert raised.value is expected_error
        if boundary == "missing-job-id-cleanup":
            assert type(raised.value.__cause__) is RunPodManagerError
            assert "omitted its job ID" in str(raised.value.__cause__)
        elif boundary.endswith("-cleanup"):
            assert raised.value.__cause__ is ordinary_error
    else:
        with pytest.raises(RunPodManagerError) as raised:
            await operation()
        assert raised.value is not ordinary_error
        if boundary.startswith("readiness-"):
            assert raised.value.__cause__ is None
        else:
            assert raised.value.__cause__ is ordinary_error

    durable = runtime.repository.get(session.training_cleanup_token)
    assert durable is not None
    if boundary.startswith("readiness-"):
        assert durable.state is TrainingPodState.READY
        assert provider.stop_calls == []
    elif boundary == "reconcile-status":
        assert durable.state is TrainingPodState.JOB_SUBMITTED
        assert provider.stop_calls == []
    elif boundary in {
        "model-readiness",
        "submit-http-status",
        "submit-timeout",
        "submit-invalid",
        "submission-recovery-readback",
        "cancel",
    }:
        assert durable.state is TrainingPodState.RELEASED
        assert provider.stop_calls == ["pod-training-1"]
    elif boundary.endswith("-cleanup"):
        assert durable_at_release is not None
        assert family_at_release is not None
        assert durable == durable_at_release
        assert (
            runtime.repository.list_cleanup_family(
                durable_at_release.root_cleanup_token
            )
            == family_at_release
        )
        assert provider.stop_calls == []
    else:
        assert provider.stop_calls == []
    if boundary == "submission-recovery-readback":
        assert [method for method, _url in calls] == ["POST", "GET"]


@pytest.mark.asyncio
async def test_training_profile_fallback_retains_ordinary_rejection_contract(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    manager = _manager(tmp_path, provider)
    manager.profiles["training"] = replace(
        manager.profiles["training"], persistent_pod_id=None
    )
    manager.profiles["training-h100"] = replace(
        manager.profiles["training"], id="training-h100"
    )
    manager.find_stopped_pod = AsyncMock(return_value=None)
    ordinary = TrainingPodLifecycleError(
        "ordinary first-profile rejection",
        cleanup_token="training:ordinary-profile-fallback:0001",
        pod_id=None,
        cleanup_state=TrainingPodCleanupState.NOT_OWNED,
        billing_risk=False,
    )
    started_at = datetime.now(UTC)
    fallback = RunPodSession(
        pod_id="pod-training-h100-1",
        profile=manager.profiles["training-h100"],
        task_profile="training",
        model_name="flux-lora-trainer",
        pod_type=None,
        status=PodStatus.READY,
        ttl_seconds=3600,
        started_at=started_at,
        expires_at=started_at + timedelta(seconds=3600),
        backend_base_url="https://pod-training-h100-1-8888.proxy.runpod.net",
        companion_id="companion-1",
        training_cleanup_token="training:ordinary-profile-fallback:h100",
    )
    attempts: list[str] = []

    async def ordinary_acquire(**kwargs):
        attempts.append(kwargs["profile"].id)
        if len(attempts) == 1:
            raise ordinary
        return fallback

    manager._acquire_training_session = ordinary_acquire

    result = await manager.start_training_pod(
        "companion-1", cleanup_token="training:ordinary-profile-fallback:0001"
    )

    assert result is fallback
    assert attempts == ["training", "training-h100"]
    manager.find_stopped_pod.assert_awaited_once_with("lora_training", "training")
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_legacy_lora_inference_admission_fails_before_provider_post(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    manager = _manager(tmp_path, provider)
    manager.profiles["image"] = manager.profiles["training"]
    manager.find_stopped_pod = AsyncMock(return_value=None)

    with pytest.raises(RunPodLegacyAdmissionRetiredError, match="durable"):
        await manager.start_inference_pod("companion-1")

    manager.provider.start_pod.assert_not_called()
    manager.find_stopped_pod.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_lora_inference_does_not_resume_name_matched_foreign_pod(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    manager = _manager(tmp_path, provider)
    manager.profiles["image"] = manager.profiles["training"]
    manager.find_stopped_pod = AsyncMock(
        return_value={
            "id": "foreign-pod-1",
            "name": "kestrel-selfie-foreign",
            "status": "EXITED",
        }
    )
    manager.resume_stopped_pod = AsyncMock()

    with pytest.raises(RunPodLegacyAdmissionRetiredError, match="durable"):
        await manager.start_inference_pod("companion-1")

    manager.find_stopped_pod.assert_not_awaited()
    manager.resume_stopped_pod.assert_not_awaited()
    manager.provider.resume_pod.assert_not_called()


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
