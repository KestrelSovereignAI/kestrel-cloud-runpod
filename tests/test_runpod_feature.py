from types import SimpleNamespace

import pytest

from kestrel_cloud_runpod.feature import RunPodFeature
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_sdk.llm import BackendType
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus


class FakeRunPodManager:
    def __init__(self):
        profile = SimpleNamespace(max_context_window=32768)
        image_profile = SimpleNamespace(max_context_window=4096)
        self.profiles = {"llm": profile, "image": image_profile}
        self.default_ttl_seconds = 1800
        self.started = False
        self.start_calls = []
        self.stop_calls = 0

    async def start_session(self, **kwargs):
        self.started = True
        self.start_calls.append(kwargs)
        profile = kwargs["task_profile"]
        base_response = {
            "pod_id": "pod-123",
            "task_profile": profile,
            "model_name": kwargs.get("model_name") or "llama-3",
            "remaining_ttl_seconds": kwargs.get("ttl_seconds", 1800),
            "status": "ready",
        }
        if profile == "image":
            base_response["image_endpoint"] = "http://gpu:9000/invoke"
        else:
            base_response["inference_url"] = "http://gpu:8000/v1"
        return base_response

    async def get_status(self, **_):
        return {
            "pod_id": "pod-123",
            "task_profile": "llm",
            "model_name": "llama-3",
            "status": "ready",
            "remaining_ttl_seconds": 1700,
        }

    async def stop_session(self):
        self.stop_calls += 1
        self.started = False
        return {
            "pod_id": "pod-123",
            "task_profile": "llm",
            "status": "terminating",
            "remaining_ttl_seconds": 0,
        }


class DummyLLMService:
    """Fake LLMService that tracks backend switching calls."""

    def __init__(self):
        self.last_backend = BackendType.CLOUD
        self.switch_calls = []
        self.deactivate_reasons = []

    def switch_backend(self, backend, *, config):
        self.last_backend = backend
        self.switch_calls.append((backend, config))

    def _deactivate_remote_backend(self, reason=None):
        self.last_backend = BackendType.CLOUD
        self.deactivate_reasons.append(reason)

    def get_backend_status(self):
        return {
            "current_backend": self.last_backend.value,
            "remote_active": self.last_backend == BackendType.REMOTE_GPU,
        }


@pytest.fixture
async def runpod_feature(monkeypatch):
    fake_manager = FakeRunPodManager()
    monkeypatch.setattr("kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager)

    llm_service = DummyLLMService()
    agent = SimpleNamespace(llm_service=llm_service)

    feature = RunPodFeature(agent)
    await feature.initialize()
    feature._post_json = lambda url, payload: {"url": url, "payload": payload}

    return feature, fake_manager, llm_service


@pytest.mark.asyncio
async def test_manage_gpu_start_and_stop(runpod_feature):
    feature, manager, llm_service = runpod_feature

    start_result = await feature.manage_gpu(
        action="on",
        model_name="llama-3-70b",
        task_profile="llm",
        ttl_seconds="120",
        pod_type="h100-single",
    )

    assert isinstance(start_result, ToolResult)
    assert start_result.status is ToolResultStatus.OK
    assert manager.started is True
    assert llm_service.switch_calls
    assert start_result.data["session"]["status"] == "ready"

    stop_result = await feature.manage_gpu(action="off")
    assert isinstance(stop_result, ToolResult)
    assert stop_result.status is ToolResultStatus.OK
    assert stop_result.data["session"]["status"] == "terminating"
    assert manager.started is False
    assert llm_service.deactivate_reasons[-1] == "Requested via !gpu off"


@pytest.mark.asyncio
async def test_image_generation_tears_down_session(runpod_feature):
    """Test that image generation automatically tears down the GPU session.

    Note: Uses generate_image_on_runpod method (dream_image command was removed).
    """
    feature, manager, llm_service = runpod_feature

    # Use the internal method (dream_image was removed, see feature.py lines 65-68)
    image_result = await feature.generate_image_on_runpod(prompt="sunset beach in watercolor")

    assert isinstance(image_result, ToolResult)
    assert image_result.status is ToolResultStatus.OK
    assert "result" in image_result.data
    assert manager.started is False
    assert llm_service.deactivate_reasons[-1] == "image generation completed"
    assert llm_service.last_backend == BackendType.CLOUD


# ---------------------------------------------------------------------------
# Pre-emptive #1042 honesty checklist (failure paths land in ToolResult)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manage_gpu_unknown_action_returns_failed(runpod_feature):
    """Unknown action lands in ToolResult.failed, NOT a raised
    ValueError that escapes the envelope."""
    feature, _manager, _ = runpod_feature

    result = await feature.manage_gpu(action="dance")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Unsupported GPU action" in result.error
    assert result.data["available_actions"] == ["status", "on", "off", "logs"]


@pytest.mark.asyncio
async def test_start_unknown_profile_returns_failed(runpod_feature):
    """Unknown profile lands in ToolResult.failed (pre-flight check),
    not a raised RunPodManagerError that escapes the envelope."""
    feature, manager, _ = runpod_feature

    result = await feature._start(
        model_name="",
        task_profile="not-a-real-profile",
        ttl_seconds="",
        pod_type="",
    )

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "not-a-real-profile" in result.error
    # Manager.start_session should NOT have been called.
    assert manager.started is False


@pytest.mark.asyncio
async def test_start_invalid_ttl_returns_failed(runpod_feature):
    """Non-numeric ttl_seconds lands in ToolResult.failed, not a
    raised ValueError out of _coerce_optional_int."""
    feature, manager, _ = runpod_feature

    result = await feature._start(
        model_name="",
        task_profile="llm",
        ttl_seconds="abc",
        pod_type="",
    )

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Invalid ttl_seconds" in result.error
    assert result.data["argument"] == "ttl_seconds"
    assert manager.started is False


@pytest.mark.asyncio
async def test_logs_invalid_lines_returns_failed(runpod_feature):
    """Non-numeric ``lines`` argument lands in ToolResult.failed,
    not a raised ValueError out of _coerce_optional_int."""
    feature, manager, _ = runpod_feature

    result = await feature.manage_gpu(action="logs", lines="abc")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Invalid lines" in result.error
    assert result.data["argument"] == "lines"


@pytest.mark.asyncio
async def test_disabled_feature_returns_failed(monkeypatch):
    """A feature whose manager couldn't be constructed (e.g. missing
    RUNPOD_API_KEY) lands in ToolResult.failed, not a legacy dict."""

    def _raise(*_, **__):
        raise RunPodManagerError("RUNPOD_API_KEY not set")

    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", _raise
    )

    feature = RunPodFeature(SimpleNamespace())
    await feature.initialize()
    assert feature.disabled is True

    result = await feature.manage_gpu(action="status")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "disabled" in result.error.lower()
    assert "RUNPOD_API_KEY" in result.data["reason"]


@pytest.mark.asyncio
async def test_stop_with_no_active_session_returns_no_op_confirmation(monkeypatch):
    """When ``!gpu off`` runs with no active session and the manager
    returns ``{active: False, status: "offline"}``, the confirmation
    must say "no-op" — saying "Stopped RunPod session" would be the
    #1042 confident-lie failure mode."""

    class _NoSessionManager(FakeRunPodManager):
        async def stop_session(self):
            return {"active": False, "status": "offline"}

    fake_manager = _NoSessionManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="off")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "no-op" in result.confirmation.lower()
    assert "Stopped RunPod session" not in result.confirmation, (
        "regression of #1042 honesty fix: confirmation claims an "
        "action happened when there was nothing to stop"
    )


@pytest.mark.parametrize(
    "raise_message",
    [
        "A RunPod session is already active",
        "ttl_seconds 99999 exceeds profile max 3600",
        "image profile has no default_model configured",
        "Some other validation we haven't seen yet",
    ],
    ids=[
        "already-active",
        "ttl-too-high",
        "no-default-model",
        "unknown-pre-creation-validation",
    ],
)
@pytest.mark.asyncio
async def test_image_generation_does_not_stop_unrelated_active_session(
    monkeypatch, raise_message,
):
    """When ``generate_image_on_runpod`` is called while another
    session (e.g., an LLM pod) is already active, ANY pre-creation
    raise from ``manager.start_session`` (already-active conflict,
    TTL validation, no-default-model, …) MUST NOT trigger
    ``stop_session()`` — that would tear down the unrelated,
    perfectly-fine LLM session.

    Codex iterations on this PR escalated through three layers of
    this same root issue: round 1 missed cleanup on inference
    failure, round 2 missed cleanup on post-creation start failure,
    round 3 introduced over-aggressive cleanup that tore down
    pre-existing sessions on the "already active" sentinel. Round 4
    surfaced that other pre-creation validations (TTL,
    default_model) had the same regression. The fix uses a
    pre/post session-active check rather than message matching —
    parameterising this test across multiple pre-creation messages
    pins the contract that ALL of them get the same correct
    treatment, not just the one we happen to know about today.
    """

    class _AlreadyActiveManager(FakeRunPodManager):
        async def get_status(self, **_):
            # An unrelated LLM session was already running before
            # generate_image_on_runpod was called.
            return {
                "active": True,
                "status": "ready",
                "task_profile": "llm",
            }

        async def start_session(self, **_):
            raise RunPodManagerError(raise_message)

    fake_manager = _AlreadyActiveManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert raise_message in result.error
    # The unrelated active session MUST NOT have been touched —
    # regardless of which pre-creation validation raised.
    assert fake_manager.stop_calls == 0, (
        f"regression: pre-creation raise '{raise_message}' "
        "triggered stop_session on an unrelated active session "
        "(should only stop sessions we actually created)"
    )


@pytest.mark.asyncio
async def test_image_generation_preflight_status_failure_does_not_escape_envelope(
    monkeypatch,
):
    """When the preflight ``get_status`` call itself raises a
    provider-level exception (HTTP 5xx, timeout, etc), the failure
    MUST land in ToolResult.failed via the start_session path —
    NOT escape the @tool method.

    Codex round 5 catch: ``get_status`` can perform a provider
    refresh that raises ``requests.exceptions.HTTPError`` / etc.
    Catching only ``RunPodManagerError`` in the preflight let those
    escape. The fix catches broadly and fails safe (treat as
    "session was active" → skip teardown).
    """

    class _FlakyStatusManager(FakeRunPodManager):
        async def get_status(self, **_):
            # Generic provider exception, not a RunPodManagerError.
            raise RuntimeError("HTTPError 503 Service Unavailable")

        async def start_session(self, **_):
            # If we get here at all, return a normal-looking session
            # so we can verify it WASN'T torn down.
            return {
                "active": True,
                "status": "ready",
                "image_endpoint": "http://gpu/invoke",
                "model_name": "stable-diffusion",
            }

    fake_manager = _FlakyStatusManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()
    feature._post_json = lambda url, payload: {"image": "base64..."}

    # The call should NOT raise — it should land in a ToolResult.
    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    # The actual generate_image flow continued normally and succeeded
    # because get_status only gates the teardown decision, not the
    # whole call.
    assert result.status is ToolResultStatus.OK


@pytest.mark.asyncio
async def test_image_generation_pre_creation_validation_no_existing_session(monkeypatch):
    """Pre-creation validation failure (e.g., TTL too high) when NO
    pre-existing session is active. The teardown decision logic
    will call ``stop_session()`` (which is safely a no-op since no
    pod was created), and the user gets the ToolResult.failed."""

    class _PreValidationManager(FakeRunPodManager):
        async def get_status(self, **_):
            return {"active": False, "status": "offline"}

        async def start_session(self, **_):
            raise RunPodManagerError(
                "ttl_seconds 99999 exceeds profile max 3600"
            )

        async def stop_session(self):
            # Tracks calls so the test can verify behaviour, but
            # returns a no-op success shape consistent with what the
            # real manager does when nothing is running.
            self.stop_calls += 1
            return {"active": False, "status": "offline"}

    fake_manager = _PreValidationManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "exceeds profile max" in result.error
    # stop_session WAS called (because nothing was active before, so
    # cleanup is safe) but it's a no-op at the manager level.
    # This is acceptable: a small wasted call vs a complex
    # introspect-what-the-manager-actually-created path.
    assert fake_manager.stop_calls == 1


@pytest.mark.asyncio
async def test_image_generation_startup_failure_still_stops_pod(monkeypatch):
    """When ``manager.start_session`` raises (e.g., ``_wait_until_ready``
    times out after ``provider.start_pod`` already created the
    pod), the helper MUST attempt teardown before returning
    ``ToolResult.failed``. Otherwise the pod billing leaks.

    Modeled after a real RunPod failure mode codex flagged: the pod
    is created, billing starts, then readiness polling times out.
    """

    class _StartupTimeoutManager(FakeRunPodManager):
        def __init__(self):
            super().__init__()
            self.start_session_calls = 0

        async def get_status(self, **_):
            # Realistic pre-call state for this scenario: no session
            # was active before generate_image_on_runpod was invoked.
            # The teardown decision logic uses this to know that
            # anything we leave behind during start_session is OUR
            # doing (not an unrelated LLM session).
            return {"active": False, "status": "offline"}

        async def start_session(self, **_):
            self.start_session_calls += 1
            # Pod was created (start_pod ran), then readiness wait
            # raised — the manager's ``_session`` is already set
            # and the pod is billing.
            raise RunPodManagerError("timed out waiting for pod readiness")

    fake_manager = _StartupTimeoutManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "timed out waiting for pod readiness" in result.error
    # Best-effort teardown must have been attempted regardless of
    # the start-session failure — pod billing must not leak.
    assert fake_manager.stop_calls == 1, (
        "regression: start_session raised after pod creation but "
        "stop_session was never called → pod billing leak"
    )


@pytest.mark.asyncio
async def test_image_generation_inference_failure_still_stops_pod(runpod_feature):
    """When the image-endpoint POST raises (HTTP 500, timeout,
    connection error, …), the pod MUST still be torn down — the
    user's exception path can't bill them indefinitely. The error
    must also land in ToolResult.failed so the audit hook can see
    it (instead of escaping the envelope as a raised exception)."""
    feature, manager, llm_service = runpod_feature

    def _failing_post(_url, _payload):
        raise RuntimeError("HTTP 500: model OOM")

    feature._post_json = _failing_post

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "HTTP 500: model OOM" in result.error
    # Pod was torn down despite the inference failure.
    assert manager.stop_calls == 1
    assert manager.started is False
    # Backend was detached with the failure-specific reason (honest
    # narration — not "completed").
    assert llm_service.deactivate_reasons[-1] == "image generation failed"
    # Teardown info still surfaces in data so the caller can confirm
    # the cleanup happened.
    assert "teardown" in result.data


@pytest.mark.asyncio
async def test_helper_wraps_manager_error_in_tool_result(monkeypatch):
    """Every helper that calls into RunPodManager catches
    RunPodManagerError and converts to ToolResult.failed."""

    class _ErroringManager(FakeRunPodManager):
        async def get_status(self, **_):
            raise RunPodManagerError("boom-status")

        async def get_logs(self, **_):
            raise RunPodManagerError("boom-logs")

        async def stop_session(self):
            raise RunPodManagerError("boom-stop")

        async def start_session(self, **_):
            raise RunPodManagerError("boom-start")

    fake_manager = _ErroringManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    for action, kwargs, expected in [
        ("status", {}, "boom-status"),
        ("logs", {"lines": "10"}, "boom-logs"),
        ("off", {}, "boom-stop"),
        ("on", {"task_profile": "llm"}, "boom-start"),
    ]:
        result = await feature.manage_gpu(action=action, **kwargs)
        assert isinstance(result, ToolResult), action
        assert result.status is ToolResultStatus.ERROR, action
        assert expected in result.error, f"action {action}: {result.error!r}"
