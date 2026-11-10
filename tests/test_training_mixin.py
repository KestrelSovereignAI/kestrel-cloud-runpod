"""Manager integration for durable training capacity and workload recovery."""

import asyncio
import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from kestrel_sovereign.async_ownership import await_owned_operation
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)
from pod_capacity_test_support import (
    LateCloseAsyncClient,
    assert_late_http_close_retained,
    complete_pod_payload,
    execution_control_carrier,
    record_control_injection,
)
from test_effect_custody import _original_errors
from training_test_support import FakeTrainingProvider, MutableClock, original_training_session, training_service

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
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
from kestrel_cloud_runpod.training_provider import RunpodTrainingPodProvider


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


async def _training_operation(
    manager: RunPodManager,
    session: RunPodSession,
    boundary: str,
):
    runtime = manager._get_training_pod_lease_service()
    if boundary == "reconcile-status":
        runtime.record_job(session.training_cleanup_token, "job-r27")
        lease = runtime.repository.get(session.training_cleanup_token)
        assert lease is not None
        return await manager._reconcile_training_workload_status(lease)
    if boundary == "readiness":
        return await manager._wait_for_training_ready(
            session, timeout=1, poll_interval=0
        )
    if boundary == "submission":
        return await manager.submit_training_job(
            session, b"avatar", "companion-1", wait_for_model_ready=False
        )
    if boundary == "current-job":
        return await manager.get_current_job(session)
    if boundary == "job-cancellation":
        return await manager.cancel_training_job(session, "job-r27")
    if boundary == "clear-current-job":
        return await manager.clear_current_job(session)
    if boundary == "status":
        return await manager.poll_training_status(session, "job-r27")
    if boundary == "download":
        return await manager.download_lora(session, "job-r27")
    if boundary == "generate":
        return await manager.generate_with_lora(session, "prompt", "lora.safetensors")
    raise AssertionError(boundary)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["job-cancellation"]
)
async def test_training_cancellation_classifies_control_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    record_property,
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token=f"training:r27-cancel:{boundary}"
    )
    runtime = manager._get_training_pod_lease_service()
    cancellation, control, explicit = execution_control_carrier(
        asyncio.CancelledError(f"{boundary} execution control")
    )
    entry_rows = []
    stop_calls = tuple(provider.stop_calls)

    def inject() -> None:
        row = runtime.repository.get(session.training_cleanup_token)
        assert row is not None
        entry_rows.append(row)
        record_control_injection(record_property, "training-cancel-" + boundary)
        raise cancellation

    if boundary == "model-readiness":

        async def cancelled_readiness(*_args, **_kwargs):
            inject()

        manager._wait_for_training_ready = cancelled_readiness

        def operation():
            return manager.submit_training_job(session, b"avatar", "companion-1")
    else:

        class CancelledClient(FakeWorkloadClient):
            async def post(self, url: str, **_kwargs):
                if boundary == "submission" and url.endswith("/train"):
                    inject()
                if boundary == "job-cancellation" and "/cancel/" in url:
                    inject()
                return await super().post(url)

        monkeypatch.setattr(httpx, "AsyncClient", lambda **_: CancelledClient())
        if boundary == "submission":

            def operation():
                return manager.submit_training_job(
                    session, b"avatar", "companion-1", wait_for_model_ready=False
                )
        else:

            def operation():
                return manager.cancel_training_job(session, "job-r27")

    with pytest.raises(asyncio.CancelledError) as raised:
        await operation()

    assert raised.value is cancellation
    assert cancellation.__cause__ is explicit is control
    assert len(entry_rows) == 1
    assert runtime.repository.get(session.training_cleanup_token) == entry_rows[0]
    assert tuple(provider.stop_calls) == stop_calls


@pytest.mark.asyncio
async def test_training_cancellation_retains_original_and_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:r27-cancel-cleanup"
    )
    runtime = manager._get_training_pod_lease_service()
    cancellation = asyncio.CancelledError("ordinary caller withdrawal")
    cleanup = TrainingPodCleanupError(
        "cleanup failed",
        cleanup_token=session.training_cleanup_token,
        pod_id=session.pod_id,
        cleanup_state=TrainingPodCleanupState.RETRYABLE_FAILURE,
        billing_risk=True,
    )
    entry_rows = []

    class CancelledClient(FakeWorkloadClient):
        async def post(self, url: str, **_kwargs):
            if "/cancel/" in url:
                raise cancellation
            return await super().post(url)

    async def failed_release(*_args, **_kwargs):
        row = runtime.repository.get(session.training_cleanup_token)
        assert row is not None
        entry_rows.append(row)
        record_control_injection(record_property, "training-cancel-cleanup")
        raise cleanup

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: CancelledClient())
    runtime.release = failed_release

    with pytest.raises(asyncio.CancelledError) as raised:
        await manager.cancel_training_job(session, "job-r27")

    assert raised.value is cancellation
    assert cancellation.__cause__ is cleanup
    assert len(entry_rows) == 1
    assert runtime.repository.get(session.training_cleanup_token) == entry_rows[0]
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_original_cancellation_intent_sql_failure_stops_post_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property,
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    token = "training:original-cancel-intent-sql"
    session = await original_training_session(manager, "companion-1", cleanup_token=token)
    runtime = manager._get_training_pod_lease_service()
    before = runtime.repository.get(token)
    original_request = runtime.request_cancellation
    errors, releases = [], []
    with closing(sqlite3.connect(runtime.repository.path)) as connection:
        connection.execute("""
            CREATE TRIGGER fail_original_cancellation_intent
            BEFORE UPDATE OF state ON pod_capacity_leases
            WHEN NEW.state = 'cancel_requested'
            BEGIN SELECT RAISE(ABORT, 'injected original cancellation intent failure'); END
        """)
        connection.commit()

    def request_cancellation(cleanup_token):
        record_property("persistence_injection", "original-cancel-intent-sql")
        try:
            return original_request(cleanup_token)
        except sqlite3.IntegrityError as error:
            errors.append(error)
            raise

    def unexpected_http(**_kwargs):
        raise AssertionError("failed original intent reached HTTP")

    async def unexpected_release(*args, **kwargs):
        releases.append((args, kwargs))
        raise AssertionError("failed original intent reached capacity release")

    monkeypatch.setattr(runtime, "request_cancellation", request_cancellation)
    monkeypatch.setattr(runtime, "release", unexpected_release)
    monkeypatch.setattr(httpx, "AsyncClient", unexpected_http)
    with pytest.raises(sqlite3.IntegrityError) as observed:
        await manager.cancel_training_job(session, "job-r27")
    assert len(errors) == 1 and observed.value is errors[0]
    assert runtime.repository.get(token) == before
    assert not releases and not provider.stop_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["job-cancellation"]
)
async def test_training_ordinary_cancellation_still_releases_owned_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token=f"training:r27-ordinary-cancel:{boundary}"
    )
    runtime = manager._get_training_pod_lease_service()
    cancellation = asyncio.CancelledError("ordinary caller withdrawal")

    if boundary == "model-readiness":

        async def cancelled_readiness(*_args, **_kwargs):
            raise cancellation

        manager._wait_for_training_ready = cancelled_readiness

        def operation():
            return manager.submit_training_job(session, b"avatar", "companion-1")
    else:

        class CancelledClient(FakeWorkloadClient):
            async def post(self, url: str, **_kwargs):
                if boundary == "submission" and url.endswith("/train"):
                    raise cancellation
                if boundary == "job-cancellation" and "/cancel/" in url:
                    raise cancellation
                return await super().post(url)

        monkeypatch.setattr(httpx, "AsyncClient", lambda **_: CancelledClient())
        if boundary == "submission":

            def operation():
                return manager.submit_training_job(
                    session, b"avatar", "companion-1", wait_for_model_ready=False
                )
        else:

            def operation():
                return manager.cancel_training_job(session, "job-r27")

    with pytest.raises(asyncio.CancelledError) as raised:
        await operation()

    assert raised.value is cancellation
    durable = runtime.repository.get(session.training_cleanup_token)
    assert durable is not None
    assert durable.state is TrainingPodState.RELEASED
    assert durable.cleanup_state is TrainingPodCleanupState.COMPLETE
    assert provider.stop_calls == [session.pod_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "reconcile-status",
        "readiness",
        "current-job",
        "job-cancellation",
        "clear-current-job",
        "status",
        "download",
    ],
)
async def test_training_request_control_survives_late_native_client_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    record_property,
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token=f"training:r27-http:{boundary}"
    )
    runtime = manager._get_training_pod_lease_service()
    carrier, control, explicit = execution_control_carrier(
        httpx.ConnectError(f"{boundary} request remains unknown")
    )
    close_error = OSError(f"{boundary} client close failed")
    clients: list[LateCloseAsyncClient] = []
    requests: list[httpx.Request] = []
    entry_rows = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        clients[-1].fail_close = True
        row = runtime.repository.get(session.training_cleanup_token)
        assert row is not None
        entry_rows.append(row)
        record_control_injection(record_property, "training-http-close-" + boundary)
        raise carrier

    def client_factory(**kwargs):
        client = LateCloseAsyncClient(
            close_error=close_error,
            transport=httpx.MockTransport(handler),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    with pytest.raises(type(carrier)) as raised:
        await _training_operation(manager, session, boundary)

    assert raised.value is carrier
    assert carrier.__cause__ is explicit is control
    assert len(requests) == len(clients) == len(entry_rows) == 1
    assert runtime.repository.get(session.training_cleanup_token) == entry_rows[0]
    assert_late_http_close_retained(
        raised.value, client=clients[0], close_error=close_error
    )


@pytest.mark.asyncio
async def test_retired_metering_refuses_before_control_callback_registration(
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

    metering_calls = []

    async def controlled_metering(**_kwargs):
        metering_calls.append(_kwargs)
        record_control_injection(record_property, "manager-stop-metering")
        raise carrier

    manager.provider.stop_pod.return_value = {"id": "pod-metered-1"}
    original_session = manager._session
    with pytest.raises(RunPodManagerError, match="Wall-clock.*retired"):
        manager.set_metering_callback(controlled_metering)

    assert carrier.__cause__ is explicit is control
    assert metering_calls == []
    manager.provider.stop_pod.assert_not_called()
    assert manager._session is original_session
    assert original_session.status is PodStatus.READY


@pytest.mark.asyncio
async def test_retired_metering_refuses_before_ordinary_callback_registration(
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

    metering_calls = []

    async def failed_metering(**_kwargs):
        metering_calls.append(_kwargs)
        raise ordinary

    manager.provider.stop_pod.return_value = {"id": "pod-metered-ordinary-1"}
    original_session = manager._session
    with pytest.raises(RunPodManagerError, match="Wall-clock.*retired"):
        manager.set_metering_callback(failed_metering)

    assert metering_calls == []
    manager.provider.stop_pod.assert_not_called()
    assert manager._session is original_session
    assert original_session.status is PodStatus.READY


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["direct", "group", "implicit-context"])
async def test_readiness_detail_json_conversion_control_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    record_property,
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token=f"training:readiness-json:{mode}"
    )
    runtime = manager._get_training_pod_lease_service()
    durable_before = runtime.repository.get(session.training_cleanup_token)
    assert durable_before is not None
    carrier, control, explicit = execution_control_carrier(
        ValueError("training readiness detail JSON remains unknown"),
        grouped=mode == "group",
    )
    if mode == "implicit-context":
        carrier.__cause__ = None
        carrier.__context__ = control
        carrier.__suppress_context__ = False
    calls: list[str] = []
    injections: list[str] = []
    durable_at_injection = None
    durable_at_second_get = None

    class ControlledResponse(httpx.Response):
        def json(self, **_kwargs):
            nonlocal durable_at_injection
            durable_at_injection = runtime.repository.get(
                session.training_cleanup_token
            )
            assert durable_at_injection is not None
            injections.append("json")
            record_control_injection(
                record_property, "training-readiness-detail-json-" + mode
            )
            raise carrier

    class ControlledClient(FakeWorkloadClient):
        async def get(self, url: str):
            nonlocal durable_at_second_get
            calls.append(url)
            if len(calls) == 1:
                return ControlledResponse(
                    503,
                    text="loading",
                    request=httpx.Request("GET", url),
                )
            durable_at_second_get = runtime.repository.get(
                session.training_cleanup_token
            )
            assert durable_at_second_get is not None
            return _response(url, 200, {"gpu": "test", "gpu_memory_gb": 24})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: ControlledClient())
    caught = None
    try:
        await manager._wait_for_training_ready(
            session,
            timeout=1,
            poll_interval=0,
        )
    except (ValueError, RunPodManagerError) as error:
        caught = error

    assert injections == ["json"]
    assert durable_at_injection is not None
    assert durable_at_injection.revision == durable_before.revision + 1
    assert (
        replace(
            durable_at_injection,
            updated_at=durable_before.updated_at,
            revision=durable_before.revision,
        )
        == durable_before
    )
    durable_after = runtime.repository.get(session.training_cleanup_token)
    if mode == "implicit-context":
        assert caught is None
        assert len(calls) == 2
        assert carrier.__cause__ is None
        assert carrier.__context__ is control
        assert durable_at_second_get is not None
        assert durable_at_second_get.revision == durable_at_injection.revision + 1
        assert (
            replace(
                durable_at_second_get,
                updated_at=durable_at_injection.updated_at,
                revision=durable_at_injection.revision,
            )
            == durable_at_injection
        )
        assert durable_after == durable_at_second_get
    else:
        assert calls == [session.backend_base_url + "/ready"]
        assert caught is carrier
        assert carrier.__cause__ is explicit
        if mode == "group":
            assert explicit.exceptions[0] is control
        else:
            assert explicit is control
        assert durable_after == durable_at_injection
    assert provider.stop_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "reconcile-status",
        "readiness-connect",
        "readiness-timeout",
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
    cleanup_boundary = boundary.endswith("-cleanup")
    recorded_operation_errors: list[BaseException] = []
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
                or (
                    boundary == "submission-recovery-readback"
                    and url.endswith("/current-job")
                )
                or (boundary == "current-job" and url.endswith("/current-job"))
                or (boundary == "status" and "/status/" in url)
                or (boundary == "download" and "/download/" in url)
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

        def operation():
            return manager.start_training_pod(
                "companion-1", cleanup_token="training:profile-fallback:0001"
            )

        durable_before = None
        stop_calls = list(provider.stop_calls)
    elif boundary == "inference-start":

        async def controlled_start_session(**_kwargs):
            record_control_injection(record_property, boundary)
            raise carrier

        manager.start_session = controlled_start_session

        def operation():
            return manager.start_inference_pod("companion-1")

        durable_before = None
        stop_calls = list(provider.stop_calls)
    else:
        session = await original_training_session(manager,
            "companion-1", cleanup_token=f"training:{boundary}:0001"
        )
        runtime = manager._get_training_pod_lease_service()
        if boundary == "reconcile-status":
            runtime.record_job(session.training_cleanup_token, "job-control")
            lease = runtime.repository.get(session.training_cleanup_token)
            assert lease is not None

            def operation():
                return manager._reconcile_training_workload_status(lease)
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

            def operation():
                return manager._wait_for_training_ready(
                    session, timeout=1, poll_interval=0
                )
        elif boundary in {"model-readiness", "model-readiness-cleanup"}:

            async def controlled_readiness(*_args, **_kwargs):
                if boundary == "model-readiness-cleanup":
                    raise RunPodManagerError("ordinary model readiness failure")
                record_control_injection(record_property, boundary)
                raise carrier

            manager._wait_for_training_ready = controlled_readiness

            def operation():
                return manager.submit_training_job(session, b"avatar", "companion-1")
        elif (
            boundary.startswith(("submit-", "submission-"))
            or boundary == "missing-job-id-cleanup"
        ):

            def operation():
                return manager.submit_training_job(
                    session,
                    b"avatar",
                    "companion-1",
                    wait_for_model_ready=False,
                )
        elif boundary == "current-job":

            def operation():
                return manager.get_current_job(session)
        elif boundary in {"cancel", "cancel-cleanup"}:

            def operation():
                return manager.cancel_training_job(session, "job-control")
        elif boundary == "clear":

            def operation():
                return manager.clear_current_job(session)
        elif boundary == "status":

            def operation():
                return manager.poll_training_status(session, "job-control")
        else:

            def operation():
                return manager.download_lora(session, "job-control")

        if cleanup_boundary:
            record_operation_error = runtime.record_operation_error

            def capture_operation_error(cleanup_token: str, error: BaseException):
                assert cleanup_token == session.training_cleanup_token
                recorded_operation_errors.append(error)
                return record_operation_error(cleanup_token, error)

            monkeypatch.setattr(
                runtime, "record_operation_error", capture_operation_error
            )

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
    if cleanup_boundary:
        # Accept the old carrier only to expose its missing-primary assertion
        # on the immutable pre-repair implementation, not as valid behavior.
        with pytest.raises((ExceptionGroup, type(carrier))) as raised:
            await operation()
    else:
        with pytest.raises(type(carrier)) as raised:
            await operation()

    if not cleanup_boundary:
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

    if cleanup_boundary:
        assert len(recorded_operation_errors) == 1
        primary = recorded_operation_errors[0]
        assert primary is not carrier
        assert not is_execution_control_error(primary)
        assert type(raised.value) is ExceptionGroup
        assert len(raised.value.exceptions) == 2
        assert raised.value.exceptions[0] is carrier
        assert raised.value.exceptions[1] is primary
        assert is_execution_control_error(raised.value)
        assert carrier.__cause__ is explicit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "reconcile-status",
        "readiness-connect",
        "readiness-timeout",
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
                or (
                    boundary == "submission-recovery-readback"
                    and url.endswith("/current-job")
                )
                or (boundary == "current-job" and url.endswith("/current-job"))
                or (boundary == "download" and "/download/" in url)
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

    session = await original_training_session(manager,
        "companion-1", cleanup_token=f"training:{boundary}:0001"
    )
    runtime = manager._get_training_pod_lease_service()
    durable_at_release = None
    family_at_release = None
    if boundary == "reconcile-status":
        runtime.record_job(session.training_cleanup_token, "job-ordinary")
        lease = runtime.repository.get(session.training_cleanup_token)
        assert lease is not None

        def operation():
            return manager._reconcile_training_workload_status(lease)
    elif boundary.startswith("readiness-"):
        base = datetime(2026, 8, 1, tzinfo=UTC)
        instants = iter((base, base, base + timedelta(seconds=2)))

        class AdvancingDateTime:
            @staticmethod
            def now(_timezone):
                return next(instants)

        monkeypatch.setattr("kestrel_cloud_runpod.training.datetime", AdvancingDateTime)

        def operation():
            return manager._wait_for_training_ready(session, timeout=1, poll_interval=0)
    elif boundary in {"model-readiness", "model-readiness-cleanup"}:

        async def ordinary_readiness(*_args, **_kwargs):
            raise ordinary_error

        manager._wait_for_training_ready = ordinary_readiness

        def operation():
            return manager.submit_training_job(session, b"avatar", "companion-1")
    elif (
        boundary.startswith(("submit-", "submission-"))
        or boundary == "missing-job-id-cleanup"
    ):

        def operation():
            return manager.submit_training_job(
                session,
                b"avatar",
                "companion-1",
                wait_for_model_ready=False,
            )
    elif boundary == "current-job":

        def operation():
            return manager.get_current_job(session)
    elif boundary in {"cancel", "cancel-cleanup"}:

        def operation():
            return manager.cancel_training_job(session, "job-ordinary")
    elif boundary == "clear":

        def operation():
            return manager.clear_current_job(session)
    else:

        def operation():
            return manager.download_lora(session, "job-ordinary")

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
async def test_retired_training_does_not_fall_back_to_another_gpu_profile(tmp_path):
    provider = FakeTrainingProvider()
    manager = _manager(tmp_path, provider)
    manager.profiles["training"] = replace(
        manager.profiles["training"], persistent_pod_id=None
    )
    manager.profiles["training-h100"] = replace(
        manager.profiles["training"], id="training-h100"
    )
    manager.find_stopped_pod = AsyncMock(side_effect=AssertionError("new profile discovery"))
    with pytest.raises(RunPodLegacyAdmissionRetiredError, match="retired"):
        await manager.start_training_pod(
            "companion-1", cleanup_token="training:retired-profile-fallback"
        )
    manager.find_stopped_pod.assert_not_awaited()
    assert not provider.create_calls and not provider.start_calls and not provider.stop_calls
    with closing(sqlite3.connect(manager._get_training_pod_lease_service().repository.path)) as connection:
        assert connection.execute("SELECT count(*) FROM pod_capacity_leases").fetchone()[0] == 0


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

    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:manager-token-0001"
    )
    job_id = "job-1"
    manager._get_training_pod_lease_service().record_job(session.training_cleanup_token, job_id)
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
@pytest.mark.parametrize("entrypoint", ["session", "pod-id"])
async def test_legacy_termination_routes_current_training_session_through_durable_release(
    tmp_path: Path, entrypoint: str
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token=f"training:r27-terminate:{entrypoint}"
    )

    if entrypoint == "session":
        await manager.terminate_session(session)
    else:
        await manager.terminate_pod(session.pod_id)

    durable = manager.get_training_pod_lease(session.training_cleanup_token)
    assert durable.state is TrainingPodState.RELEASED
    assert durable.cleanup_state is TrainingPodCleanupState.COMPLETE
    assert manager._session is None
    assert session.status is PodStatus.OFFLINE
    assert provider.stop_calls == [session.pod_id]
    manager.provider.terminate_pod.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_termination_refuses_mutated_session_resource_binding(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:r27-terminate-mutable"
    )
    durable = manager.get_training_pod_lease(session.training_cleanup_token)
    session.pod_id = "foreign-pod"

    with pytest.raises(RunPodManagerError, match="no longer matches"):
        await manager.terminate_session(session)

    assert manager.get_training_pod_lease(durable.cleanup_token) == durable
    assert manager._session is session
    assert provider.stop_calls == []
    manager.provider.terminate_pod.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_termination_does_not_stop_preexisting_configured_capacity(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    provider.status = "RUNNING"
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:r27-terminate-preexisting"
    )

    await manager.terminate_session(session)

    durable = manager.get_training_pod_lease(session.training_cleanup_token)
    assert durable.state is TrainingPodState.RELEASED
    assert durable.cleanup_state is TrainingPodCleanupState.NOT_OWNED
    assert manager._session is None
    assert session.status is PodStatus.OFFLINE
    assert provider.stop_calls == []
    manager.provider.terminate_pod.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_termination_retains_session_when_durable_release_is_retryable(
    tmp_path: Path,
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:r27-terminate-retry"
    )
    provider.stop_error = RunPodManagerError("provider unavailable")

    with pytest.raises(TrainingPodCleanupError) as raised:
        await manager.terminate_session(session)

    durable = manager.get_training_pod_lease(session.training_cleanup_token)
    assert durable.state is TrainingPodState.RECONCILE_REQUIRED
    assert durable.cleanup_state is TrainingPodCleanupState.RETRYABLE_FAILURE
    assert raised.value.cleanup_token == session.training_cleanup_token
    assert manager._session is session
    assert session.status is PodStatus.READY
    assert provider.stop_calls == [session.pod_id]


@pytest.mark.asyncio
async def test_retired_submission_does_not_stop_original_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_: FakeWorkloadClient(fail_train=True)
    )
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:submit-fail-0001"
    )

    before = manager.get_training_pod_lease(session.training_cleanup_token)
    with pytest.raises(RunPodLegacyAdmissionRetiredError, match="retired"):
        await manager.submit_training_job(session, b"avatar", "companion-1")

    lease = manager.get_training_pod_lease("training:submit-fail-0001")
    assert lease == before and lease.state is TrainingPodState.READY
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_status_failure_keeps_job_and_cleanup_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    client = FakeWorkloadClient(fail_status=True)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:status-fail-0001"
    )
    job_id = "job-1"
    manager._get_training_pod_lease_service().record_job(session.training_cleanup_token, job_id)

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
    session = await original_training_session(manager,
        "companion-1", cleanup_token="training:cancel-fail-0001"
    )
    job_id = "job-1"
    manager._get_training_pod_lease_service().record_job(session.training_cleanup_token, job_id)

    with pytest.raises(TrainingPodCleanupError) as raised:
        await manager.cancel_training_job(session, job_id)

    assert raised.value.cleanup_token == "training:cancel-fail-0001"
    lease = manager.get_training_pod_lease(raised.value.cleanup_token)
    assert lease.provider_pod_id == "pod-training-1"
    assert lease.state is TrainingPodState.RECONCILE_REQUIRED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary,cleanup_mode",
    [
        pytest.param("C05-cancel", "healthy", id="C05-healthy"),
        pytest.param("C05-cancel", "plain-control", id="C05-plain-control"),
        pytest.param("C05-cancel", "grouped-control", id="C05-grouped-control"),
    ],
)
async def test_r29_f3_training_preserves_operation_and_owned_cleanup_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    cleanup_mode: str,
    record_property,
) -> None:
    """Actual existing manager/SQLite/HTTPX paths; no replacement release API."""
    token = f"training:r29-f3:{boundary}:{cleanup_mode}"
    pod_id = "pod-training-1"
    events = {"status": "EXITED", "provider_calls": [], "workload_calls": []}
    stop_entered = threading.Event()
    stop_exited = threading.Event()
    clients = []
    recorded_errors = []
    driver = {}
    original_async_client = httpx.AsyncClient
    cleanup = control = explicit = None
    if cleanup_mode != "healthy":
        cleanup, control, explicit = execution_control_carrier(
            TrainingPodCleanupError(
                "native transport cleanup remains unknown",
                cleanup_token=token,
                pod_id=pod_id,
                cleanup_state=TrainingPodCleanupState.PENDING,
                billing_risk=True,
            ),
            grouped=cleanup_mode == "grouped-control",
        )

    def provider_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        events["provider_calls"].append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == f"/v2/pods/{pod_id}":
            payload = complete_pod_payload(pod_id=pod_id, status=events["status"])
            payload["runtime"] = {"ports": [{"private": 8888, "type": "http"}]}
            return httpx.Response(200, json=payload)
        assert request.method == "POST"
        assert request.url.path == f"/v2/pods/{pod_id}/action"
        if body == {"action": "start"}:
            events["status"] = "RUNNING"
            return httpx.Response(204)
        assert body == {"action": "stop"}
        stop_entered.set()
        events["cleanup_row"] = runtime.repository.get(token)
        record_property(
            "injection_entry",
            f"{boundary}:{cleanup_mode}:actual-provider-stop-before-outcome",
        )
        try:
            if cleanup is not None:
                raise cleanup
            events["status"] = "EXITED"
            return httpx.Response(204)
        finally:
            stop_exited.set()

    def workload_transport(request: httpx.Request) -> httpx.Response:
        events["workload_calls"].append((request.method, request.url.path))
        assert request.url.host == f"{pod_id}-8888.proxy.runpod.net"
        if boundary == "C01-readiness":
            assert (request.method, request.url.path) == ("GET", "/ready")
            return httpx.Response(503, json={"detail": "Training in progress: old-job"})
        if boundary == "C05-cancel":
            assert (request.method, request.url.path) == ("POST", "/cancel/job-f3")
            raise httpx.ConnectError(
                "ordinary cancellation request failure", request=request
            )
        if request.url.path == "/train":
            assert request.method == "POST"
            if boundary == "C02-invalid-object":
                return httpx.Response(200, json=[])
            if boundary == "C03-missing-id":
                return httpx.Response(200, json={})
            assert boundary == "C04-recovery"
            raise httpx.ConnectError(
                "ordinary training request failure", request=request
            )
        assert boundary == "C04-recovery"
        assert (request.method, request.url.path) == ("GET", "/current-job")
        return httpx.Response(200, json={"current_job": None})

    class RecordedAsyncClient(original_async_client):
        async def aclose(self) -> None:
            self.f3_close_calls += 1
            self.f3_close_task = asyncio.current_task()
            await super().aclose()

    def workload_client(**kwargs):
        client = RecordedAsyncClient(
            transport=httpx.MockTransport(workload_transport), trust_env=False, **kwargs
        )
        client.f3_close_calls = 0
        client.f3_close_task = None
        clients.append(client)
        return client

    native = RunpodControlPlaneClient(
        api_key="test-not-a-live-credential",
        http_transport=httpx.MockTransport(provider_transport),
        max_safe_retries=0,
    )
    setup_error = outcome = None
    try:
        direct = DirectRunPodProvider(api_key="", client=native)
        manager = _manager(tmp_path, RunpodTrainingPodProvider(direct))
        runtime = manager._get_training_pod_lease_service()
        session = await original_training_session(manager, "companion-1", cleanup_token=token)
        if boundary == "C05-cancel":
            # Existing public metadata fixture API, not a new submission grant.
            runtime.record_job(token, "job-f3")
        before = runtime.repository.get(token)
        assert before is not None
        acquire_calls = tuple(events["provider_calls"])
        original_record_error = runtime.record_operation_error

        def record_error(cleanup_token, error):
            recorded_errors.append(error)
            events["before_error_record"] = runtime.repository.get(cleanup_token)
            result = original_record_error(cleanup_token, error)
            events["after_error_record"] = runtime.repository.get(cleanup_token)
            return result

        monkeypatch.setattr(runtime, "record_operation_error", record_error)
        monkeypatch.setattr(httpx, "AsyncClient", workload_client)

        async def operation():
            driver["task"] = asyncio.current_task()
            try:
                if boundary == "C05-cancel":
                    return await manager.cancel_training_job(session, "job-f3")
                return await manager.submit_training_job(
                    session,
                    b"avatar",
                    "companion-1",
                    wait_for_model_ready=boundary == "C01-readiness",
                )
            finally:
                driver["body_exited"] = True

        outcome = await await_owned_operation(operation(), retain_originals=True)
        after = runtime.repository.get(token)
    except BaseException as error:
        setup_error = error
        raise
    finally:
        # Physically close the SAME real sync client once, even on fixture failure.

        async def retire_native_client():
            events["native_close_task"] = asyncio.current_task()
            await asyncio.to_thread(native.close)
            events["native_close_returned"] = True

        close = await await_owned_operation(
            retire_native_client(),
            retirement_resource=native,
            retain_originals=True,
        )
        retain_outcome = (
            setup_error is not None
            or close.error is not None
            or close.cancellation is not None
        )
        terminal = execution_terminal_error(
            setup_error,
            outcome.error if retain_outcome and outcome is not None else None,
            outcome.cancellation if retain_outcome and outcome is not None else None,
            close.error,
            close.cancellation,
            retain_originals=True,
            isolate_control_originals=True,
        )
        if terminal is not None:
            raise terminal

    primary = recorded_errors[0] if recorded_errors else None
    originals = tuple(_original_errors(outcome.error))
    facts = {
        "boundary": boundary,
        "cleanup_mode": cleanup_mode,
        "before": before.to_public_dict(),
        "before_error_record": events["before_error_record"].to_public_dict(),
        "after_error_record": events["after_error_record"].to_public_dict(),
        "cleanup_entry": events["cleanup_row"].to_public_dict(),
        "after": after.to_public_dict(),
        "provider_acquire_calls": acquire_calls,
        "provider_cleanup_calls": events["provider_calls"][len(acquire_calls) :],
        "workload_calls": events["workload_calls"],
        "primary_type": type(primary).__name__,
        "primary_message": str(primary),
        "public_error_type": type(outcome.error).__name__,
        "primary_is_ordinary": primary is not None
        and not is_execution_control_error(primary),
        "same_primary_explicitly_retained": any(
            error is primary for error in originals
        ),
        "same_cleanup_explicitly_retained": cleanup is not None
        and any(error is cleanup for error in originals),
        "same_control_explicitly_retained": control is not None
        and any(error is control for error in originals),
        "same_explicit_group_or_cause_retained": explicit is not None
        and any(error is explicit for error in originals),
        "original_cleanup_cause_unmodified": cleanup is None
        or cleanup.__cause__ is explicit,
        "driver_body_exited": driver.get("body_exited", False),
        "same_driver_task_done": driver.get("task") is not None
        and driver["task"].done(),
        "native_stop_call_entered_and_exited": stop_entered.is_set()
        and stop_exited.is_set(),
        "native_control_client_closed": native.transport._client.is_closed,
        "native_close_returned": events.get("native_close_returned", False),
        "same_native_close_task_done": events["native_close_task"].done(),
        "async_clients": [
            {
                "closed": client.is_closed,
                "close_calls": client.f3_close_calls,
                "same_close_task_done": client.f3_close_task is not None
                and client.f3_close_task.done(),
            }
            for client in clients
        ],
        "scope": "invented HTTP fixtures/real SQLite and native HTTPX; no cloud or physical training proof",
    }
    record_property("r29_f3", json.dumps(facts, sort_keys=True))
    assert outcome.cancellation is None
    assert len(recorded_errors) == len(clients) == 1
    assert primary is not None and not is_execution_control_error(primary)
    assert driver["body_exited"] and driver["task"].done()
    assert stop_entered.is_set() and stop_exited.is_set()
    assert native.transport._client.is_closed
    assert events["native_close_returned"] and events["native_close_task"].done()
    assert all(
        client.is_closed and client.f3_close_calls == 1 and client.f3_close_task.done()
        for client in clients
    )
    assert after.cleanup_token == before.cleanup_token == token
    assert after.provider_pod_id == before.provider_pod_id == pod_id
    assert after.hard_deadline == before.hard_deadline
    assert after.stop_attempts == before.stop_attempts + 1
    assert (
        events["workload_calls"]
        == {
            "C01-readiness": [("GET", "/ready")],
            "C02-invalid-object": [("POST", "/train")],
            "C03-missing-id": [("POST", "/train")],
            "C04-recovery": [("POST", "/train"), ("GET", "/current-job")],
            "C05-cancel": [("POST", "/cancel/job-f3")],
        }[boundary]
    )
    if boundary in {"C01-readiness", "C02-invalid-object", "C03-missing-id"}:
        assert type(primary) is RunPodManagerError
    else:
        assert type(primary) is httpx.ConnectError
    if boundary == "C02-invalid-object":
        assert (
            str(primary)
            == "Training workload returned a non-object submission response"
        )
    if boundary == "C03-missing-id":
        assert str(primary) == "Training workload omitted its job ID"
    assert events["provider_calls"][len(acquire_calls)] == (
        "POST",
        f"/v2/pods/{pod_id}/action",
        {"action": "stop"},
    )
    if cleanup is not None:
        assert len(events["provider_calls"]) == len(acquire_calls) + 1
        assert after.state is TrainingPodState.RELEASING
        assert after.cleanup_state is TrainingPodCleanupState.PENDING
        assert cleanup.__cause__ is explicit
        assert all(
            any(error is original for error in originals)
            for original in (cleanup, control, explicit)
        )
        assert is_execution_control_error(outcome.error)
    else:
        assert len(events["provider_calls"]) == len(acquire_calls) + 2
        assert after.state is TrainingPodState.RELEASED
        assert after.cleanup_state is TrainingPodCleanupState.COMPLETE
    # The expected pre-repair semantic failure is only this missing ORIGINAL.
    assert any(error is primary for error in originals), (
        "ordinary operation primary was lost when original owned cleanup failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary,persistence_mode",
    [
        pytest.param("C05-cancel", "ordinary", id="C05-ordinary-dml"),
        pytest.param("R01-readiness-connect", "ordinary", id="R01-ordinary-dml"),
        pytest.param("R02-readiness-timeout", "ordinary", id="R02-ordinary-dml"),
        pytest.param("C05-cancel", "control", id="C05-control-dml"),
        pytest.param("R01-readiness-connect", "control", id="R01-control-dml"),
        pytest.param("R02-readiness-timeout", "control", id="R02-control-dml"),
    ],
)
async def test_training_error_recording_dml_retains_original_and_stops_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    persistence_mode: str,
    record_property,
) -> None:
    """A real failed error-recording UPDATE retains both exact originals."""
    token = f"training:c06:{boundary}:{persistence_mode}"
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    runtime = manager._get_training_pod_lease_service()
    session = await original_training_session(manager, "companion-1", cleanup_token=token)
    if boundary == "C05-cancel":
        runtime.record_job(token, "job-c06")
    before = runtime.repository.get(token)
    assert before is not None
    stop_calls_before = tuple(provider.stop_calls)
    workload_calls = []
    injected_transport_errors = []

    def workload_transport(request: httpx.Request) -> httpx.Response:
        workload_calls.append((request.method, request.url.path))
        if request.url.path == "/ready":
            assert request.method == "GET"
            if boundary == "R01-readiness-connect":
                error = httpx.ConnectError(
                    "ordinary readiness connection failure", request=request
                )
                injected_transport_errors.append(error)
                raise error
            if boundary == "R02-readiness-timeout":
                error = httpx.ReadTimeout("ordinary readiness timeout", request=request)
                injected_transport_errors.append(error)
                raise error
            assert boundary == "C01-readiness"
            return httpx.Response(503, json={"detail": "Training in progress: old-job"})
        if request.url.path == "/train":
            assert request.method == "POST"
            if boundary == "C02-invalid-object":
                return httpx.Response(200, json=[])
            if boundary == "C03-missing-id":
                return httpx.Response(200, json={})
            assert boundary == "C04-recovery"
            error = httpx.ConnectError(
                "ordinary training request failure", request=request
            )
            injected_transport_errors.append(error)
            raise error
        if request.url.path == "/current-job":
            assert boundary == "C04-recovery"
            assert request.method == "GET"
            return httpx.Response(200, json={"current_job": None})
        assert boundary == "C05-cancel"
        assert (request.method, request.url.path) == ("POST", "/cancel/job-c06")
        error = httpx.ConnectError(
            "ordinary cancellation request failure", request=request
        )
        injected_transport_errors.append(error)
        raise error

    clients = []
    original_async_client = httpx.AsyncClient

    class RecordedAsyncClient(original_async_client):
        async def aclose(self) -> None:
            self.c06_close_calls += 1
            self.c06_close_task = asyncio.current_task()
            await super().aclose()

    def workload_client(**kwargs):
        client = RecordedAsyncClient(
            transport=httpx.MockTransport(workload_transport), trust_env=False, **kwargs
        )
        client.c06_close_calls = 0
        client.c06_close_task = None
        clients.append(client)
        return client

    with closing(sqlite3.connect(runtime.repository.path)) as connection:
        connection.execute(
            """CREATE TRIGGER fail_c06_error_recording
               BEFORE UPDATE OF last_provider_error ON pod_capacity_leases
               WHEN NEW.last_provider_error IS NOT OLD.last_provider_error
               BEGIN
                   SELECT RAISE(ABORT, 'injected operation error recording failure');
               END"""
        )
        connection.commit()

    primaries = []
    recording_entries = []
    persistence_errors = []
    control = explicit = None
    original_record_error = runtime.record_operation_error
    original_compare_and_set = runtime.repository.compare_and_set

    def record_error(cleanup_token: str, error: BaseException):
        entry = runtime.repository.get(cleanup_token)
        assert entry is not None
        primaries.append(error)
        recording_entries.append(entry)
        return original_record_error(cleanup_token, error)

    def compare_and_set(current, *, changes):
        nonlocal control, explicit
        try:
            return original_compare_and_set(current, changes=changes)
        except sqlite3.IntegrityError as persistence_error:
            persistence_errors.append(persistence_error)
            if persistence_mode == "control":
                _, control, explicit = execution_control_carrier(persistence_error)
                record_control_injection(
                    record_property, f"c06-{boundary}-error-recording-dml"
                )
            record_property(
                "c06_error_recording",
                json.dumps(
                    {
                        "boundary": boundary,
                        "entry": current.to_public_dict(),
                        "mode": persistence_mode,
                        "primary_type": type(primaries[0]).__name__,
                        "sqlite_errorcode": persistence_error.sqlite_errorcode,
                    },
                    sort_keys=True,
                ),
            )
            raise

    monkeypatch.setattr(runtime, "record_operation_error", record_error)
    monkeypatch.setattr(runtime.repository, "compare_and_set", compare_and_set)
    monkeypatch.setattr(httpx, "AsyncClient", workload_client)
    driver = {}

    async def operation():
        driver["task"] = asyncio.current_task()
        try:
            if boundary == "C05-cancel":
                return await manager.cancel_training_job(session, "job-c06")
            if boundary in {"R01-readiness-connect", "R02-readiness-timeout"}:
                return await manager._wait_for_training_ready(session, timeout=1, poll_interval=0)
            return await manager.submit_training_job(
                session,
                b"avatar",
                "companion-1",
                wait_for_model_ready=boundary
                in {
                    "C01-readiness",
                    "R01-readiness-connect",
                    "R02-readiness-timeout",
                },
            )
        finally:
            driver["body_exited"] = True

    outcome = await await_owned_operation(operation(), retain_originals=True)
    after = runtime.repository.get(token)
    originals = tuple(_original_errors(outcome.error))
    primary = primaries[0] if len(primaries) == 1 else None
    entry = recording_entries[0] if len(recording_entries) == 1 else None
    persistence_error = persistence_errors[0] if len(persistence_errors) == 1 else None
    driver_task = driver.get("task")
    result_facts = {
        "after": after.to_public_dict() if after is not None else None,
        "before": before.to_public_dict(),
        "boundary": boundary,
        "clients": [
            {
                "close_calls": client.c06_close_calls,
                "close_task_done": client.c06_close_task is not None
                and client.c06_close_task.done(),
                "closed": client.is_closed,
            }
            for client in clients
        ],
        "driver_body_exited": driver.get("body_exited", False),
        "driver_task_done": driver_task is not None and driver_task.done(),
        "entry": entry.to_public_dict() if entry is not None else None,
        "mode": persistence_mode,
        "outcome_cancellation": type(outcome.cancellation).__name__
        if outcome.cancellation is not None
        else None,
        "outcome_is_control": is_execution_control_error(outcome.error),
        "outcome_type": type(outcome.error).__name__,
        "persistence_error_count": len(persistence_errors),
        "persistence_error_type": type(persistence_error).__name__,
        "primary_count": len(primaries),
        "primary_is_caught_transport": not injected_transport_errors
        or primary is injected_transport_errors[0],
        "primary_type": type(primary).__name__,
        "provider_stop_calls_after": list(provider.stop_calls),
        "provider_stop_calls_before": list(stop_calls_before),
        "recording_entry_count": len(recording_entries),
        "same_control_retained": control is not None
        and any(error is control for error in originals),
        "same_dml_retained": persistence_error is not None
        and any(error is persistence_error for error in originals),
        "same_explicit_retained": explicit is not None
        and any(error is explicit for error in originals),
        "same_primary_retained": primary is not None
        and any(error is primary for error in originals),
        "workload_calls": workload_calls,
    }
    record_property(
        "c06_error_recording_result", json.dumps(result_facts, sort_keys=True)
    )

    assert after is not None
    assert outcome.cancellation is None
    assert len(primaries) == len(recording_entries) == len(persistence_errors) == 1
    assert primary is not None and entry is not None and persistence_error is not None
    assert persistence_error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER
    assert any(error is persistence_error for error in originals)
    assert after == entry
    assert tuple(provider.stop_calls) == stop_calls_before
    assert driver["body_exited"] and driver["task"].done()
    assert all(
        client.is_closed
        and client.c06_close_calls == 1
        and client.c06_close_task is not None
        and client.c06_close_task.done()
        for client in clients
    )
    if boundary in {
        "C01-readiness",
        "R01-readiness-connect",
        "R02-readiness-timeout",
    }:
        assert entry.revision == before.revision + 1
    elif boundary == "C05-cancel":
        assert before.state is TrainingPodState.JOB_SUBMITTED
        assert entry.state is TrainingPodState.CANCEL_REQUESTED
        assert entry.revision == before.revision + 1
    else:
        assert entry == before
    assert (
        workload_calls
        == {
            "C01-readiness": [("GET", "/ready")],
            "C02-invalid-object": [("POST", "/train")],
            "C03-missing-id": [("POST", "/train")],
            "C04-recovery": [("POST", "/train"), ("GET", "/current-job")],
            "C05-cancel": [("POST", "/cancel/job-c06")],
            "R01-readiness-connect": [("GET", "/ready")],
            "R02-readiness-timeout": [("GET", "/ready")],
        }[boundary]
    )
    if persistence_mode == "control":
        assert control is not None and explicit is not None
        assert persistence_error.__cause__ is explicit
        assert all(
            any(error is original for error in originals)
            for original in (control, explicit)
        )
        assert is_execution_control_error(outcome.error)
    else:
        assert not is_execution_control_error(persistence_error)
        assert not is_execution_control_error(outcome.error)
    # The pre-repair semantic failures are confined to explicit-primary custody
    # and the two readiness sites' synthetic replacement of the caught object.
    assert not injected_transport_errors or primary is injected_transport_errors[0]
    assert any(error is primary for error in originals)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        pytest.param(httpx.ConnectError, id="R01-connect-recording-success"),
        pytest.param(httpx.ReadTimeout, id="R02-timeout-recording-success"),
    ],
)
async def test_training_readiness_records_same_caught_error_and_sanitizes_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[httpx.HTTPError],
    record_property,
) -> None:
    """Successful error recording retains the caught object and type-only detail."""
    expected_type = error_type.__name__
    token = f"training:c06-readiness-success:{expected_type}"
    marker = f"private-body-marker:https://private.invalid/{expected_type}"
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    manager = _manager(tmp_path, provider)
    session = await original_training_session(manager, "companion-1", cleanup_token=token)
    runtime = manager._get_training_pod_lease_service()
    before = runtime.repository.get(token)
    assert before is not None
    stop_calls_before = tuple(provider.stop_calls)
    workload_calls = []
    injected_errors = []
    recorded_errors = []
    recording_entries = []
    persisted_rows = []
    clients = []

    def workload_transport(request: httpx.Request) -> httpx.Response:
        workload_calls.append((request.method, request.url.path))
        assert (request.method, request.url.path) == ("GET", "/ready")
        if len(workload_calls) == 1:
            error = error_type(marker, request=request)
            injected_errors.append(error)
            raise error
        return httpx.Response(
            200,
            json={"gpu": "test", "gpu_memory_gb": 24},
            request=request,
        )

    original_async_client = httpx.AsyncClient

    class RecordedAsyncClient(original_async_client):
        async def aclose(self) -> None:
            self.c06_success_close_calls += 1
            self.c06_success_close_task = asyncio.current_task()
            await super().aclose()

    def workload_client(**kwargs):
        client = RecordedAsyncClient(
            transport=httpx.MockTransport(workload_transport), trust_env=False, **kwargs
        )
        client.c06_success_close_calls = 0
        client.c06_success_close_task = None
        clients.append(client)
        return client

    original_record_error = runtime.record_operation_error

    def record_error(cleanup_token: str, error: BaseException):
        entry = runtime.repository.get(cleanup_token)
        assert entry is not None
        recording_entries.append(entry)
        recorded_errors.append(error)
        persisted = original_record_error(cleanup_token, error)
        persisted_rows.append(persisted)
        return persisted

    monkeypatch.setattr(runtime, "record_operation_error", record_error)
    monkeypatch.setattr(httpx, "AsyncClient", workload_client)
    operation_task = asyncio.create_task(
        manager._wait_for_training_ready(session, timeout=60, poll_interval=0),
        name=f"c06-readiness-success-{expected_type}",
    )
    await operation_task
    after = runtime.repository.get(token)
    assert after is not None
    injected = injected_errors[0] if len(injected_errors) == 1 else None
    recorded = recorded_errors[0] if len(recorded_errors) == 1 else None
    recording_entry = recording_entries[0] if len(recording_entries) == 1 else None
    persisted = persisted_rows[0] if len(persisted_rows) == 1 else None
    facts = {
        "after": after.to_public_dict(),
        "before": before.to_public_dict(),
        "client_close_calls": [client.c06_success_close_calls for client in clients],
        "client_close_tasks_done": [
            client.c06_success_close_task is not None
            and client.c06_success_close_task.done()
            for client in clients
        ],
        "client_close_tasks_are_operation_task": [
            client.c06_success_close_task is operation_task for client in clients
        ],
        "client_close_tasks_present": [
            client.c06_success_close_task is not None for client in clients
        ],
        "clients_closed": [client.is_closed for client in clients],
        "expected_type": expected_type,
        "injected_count": len(injected_errors),
        "injected_request_url": str(injected.request.url)
        if injected is not None and injected.request is not None
        else None,
        "operation_task_done": operation_task.done(),
        "persisted": persisted.to_public_dict() if persisted is not None else None,
        "provider_stop_calls_after": list(provider.stop_calls),
        "provider_stop_calls_before": list(stop_calls_before),
        "recorded_count": len(recorded_errors),
        "recorded_is_injected": recorded is injected and injected is not None,
        "recorded_type": type(recorded).__name__,
        "recording_entry": recording_entry.to_public_dict()
        if recording_entry is not None
        else None,
        "workload_calls": workload_calls,
    }
    record_property(
        "c06_readiness_recording_success", json.dumps(facts, sort_keys=True)
    )

    assert operation_task.done()
    assert workload_calls == [("GET", "/ready"), ("GET", "/ready")]
    assert len(injected_errors) == len(recorded_errors) == 1
    assert len(recording_entries) == len(persisted_rows) == len(clients) == 1
    client = clients[0]
    assert client.is_closed
    assert client.c06_success_close_calls == 1
    assert client.c06_success_close_task is not None
    assert client.c06_success_close_task is not operation_task
    assert client.c06_success_close_task.done()
    assert tuple(provider.stop_calls) == stop_calls_before
    assert before.state is after.state is TrainingPodState.READY
    assert before.cleanup_state is after.cleanup_state
    assert before.stop_attempts == after.stop_attempts
    assert before.cleanup_token == after.cleanup_token == token
    assert before.provider_pod_id == after.provider_pod_id == session.pod_id
    assert recording_entry is not None and persisted is not None
    assert recording_entry.revision == before.revision + 1
    assert persisted.revision == before.revision + 2
    assert after.revision == before.revision + 3
    assert after.last_provider_error == persisted.last_provider_error
    assert after.last_provider_error is not None
    assert marker not in after.last_provider_error
    assert "private-body-marker" not in after.last_provider_error
    assert str(injected.request.url) not in after.last_provider_error
    assert recorded is injected
    assert after.last_provider_error == expected_type
