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
    assert "unavailable" in result.error.lower()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised",
    [
        RuntimeError("HTTPError 503: bad gateway"),
        TimeoutError("provider timed out"),
    ],
)
async def test_helper_wraps_raw_provider_exception_in_tool_result(
    monkeypatch, raised
):
    """Codex round 6: every helper that calls into RunPodManager must
    catch broader than RunPodManagerError. The underlying provider/SDK
    can raise raw HTTPError/Timeout/etc. which would otherwise escape
    the @tool envelope. This test pins the broad-catch contract for
    every action surface."""

    class _RawErroringManager(FakeRunPodManager):
        async def get_status(self, **_):
            raise raised

        async def get_logs(self, **_):
            raise raised

        async def stop_session(self):
            raise raised

        async def start_session(self, **_):
            raise raised

    fake_manager = _RawErroringManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    for action, kwargs in [
        ("status", {}),
        ("logs", {"lines": "10"}),
        ("off", {}),
        ("on", {"task_profile": "llm"}),
    ]:
        result = await feature.manage_gpu(action=action, **kwargs)
        assert isinstance(result, ToolResult), action
        assert result.status is ToolResultStatus.ERROR, action
        assert str(raised) in result.error, (
            f"action {action}: expected {raised!r} in error, got {result.error!r}"
        )


@pytest.mark.asyncio
async def test_logs_passes_tail_kwarg_to_manager(monkeypatch):
    """Codex round 6: ``RunPodManager.get_logs`` takes ``tail=``, not
    ``lines=``. A previous version of _logs sent ``lines=`` which
    would TypeError against the real signature; the FakeRunPodManager's
    ``**_`` swallowed it silently. Pin the call kwarg explicitly."""
    captured = {}

    class _LogsCapturingManager(FakeRunPodManager):
        async def get_logs(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "log line 1\nlog line 2\n"

    fake_manager = _LogsCapturingManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="logs", lines="42")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    # The contract: get_logs is called with ``tail=`` (real signature),
    # NOT ``lines=`` (which would fail against the production manager).
    assert captured["args"] == ()
    assert captured["kwargs"] == {"tail": 42}, (
        f"feature called manager.get_logs with {captured['kwargs']!r}; "
        "must use tail= to match RunPodManager.get_logs signature"
    )


@pytest.mark.asyncio
async def test_start_raw_provider_exception_attempts_teardown(monkeypatch):
    """Codex round 6: when ``_start`` calls ``start_session`` and a raw
    (non-RunPodManagerError) provider exception escapes after the pod
    is created, ``_start`` must (a) return ToolResult.failed instead of
    letting the exception escape the envelope, and (b) attempt
    best-effort teardown so the orphan pod doesn't leak (provided no
    pre-existing session was active)."""
    stop_called = []

    class _RawRaisingManager(FakeRunPodManager):
        async def start_session(self, **kwargs):
            raise RuntimeError("HTTPError 502: pod created, then readiness wait failed")

        async def stop_session(self):
            stop_called.append(True)
            return await FakeRunPodManager.stop_session(self)

        async def get_status(self, **_):
            # No session active before _start ran.
            return {"active": False, "status": "offline"}

    fake_manager = _RawRaisingManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="on", task_profile="llm")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "HTTPError 502" in result.error
    # No pre-existing session → best-effort teardown of the orphan pod.
    assert stop_called == [True], (
        "raw exception after start_session must trigger best-effort teardown "
        "when no session was pre-active"
    )


@pytest.mark.asyncio
async def test_start_raw_exception_with_preexisting_session_skips_teardown(
    monkeypatch,
):
    """Codex round 6 + round 4 honesty contract: when start_session
    raises AND a session was already active before, we must NOT call
    stop_session — that would tear down an unrelated user session."""
    stop_called = []

    class _PreActiveRaisingManager(FakeRunPodManager):
        async def start_session(self, **kwargs):
            raise RuntimeError("HTTPError 503: provider degraded")

        async def stop_session(self):
            stop_called.append(True)
            return await FakeRunPodManager.stop_session(self)

        async def get_status(self, **_):
            # Pre-existing active session — NOT ours to tear down.
            return {
                "active": True,
                "status": "ready",
                "pod_id": "pod-someone-elses",
            }

    fake_manager = _PreActiveRaisingManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="on", task_profile="llm")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "HTTPError 503" in result.error
    assert stop_called == [], (
        "session was pre-active; tearing it down on a failed start "
        "would destroy unrelated user state"
    )


@pytest.mark.asyncio
async def test_image_generation_raw_provider_exception_does_not_escape(monkeypatch):
    """Codex round 6: generate_image_on_runpod must catch broader than
    RunPodManagerError around start_session, since the provider can
    raise raw exceptions after creating the pod. Pin the broad-catch
    contract."""
    stop_called = []

    class _RawRaisingImageManager(FakeRunPodManager):
        async def start_session(self, **kwargs):
            raise RuntimeError("HTTPError 504: image pod readiness timeout")

        async def stop_session(self):
            stop_called.append(True)
            return await FakeRunPodManager.stop_session(self)

        async def get_status(self, **_):
            return {"active": False, "status": "offline"}

    fake_manager = _RawRaisingImageManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.generate_image_on_runpod(prompt="a sunset")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "HTTPError 504" in result.error
    assert stop_called == [True], (
        "image-gen acquire-use-release: orphan pod must be torn down "
        "when start_session raises after creation"
    )


@pytest.mark.parametrize(
    "pre_status,expected",
    [
        # Manager-canonical 'active' key — trust it directly.
        ({"active": True, "status": "ready"}, True),
        ({"active": False, "status": "offline"}, False),
        # Codex round 7 catch: terminating/error are NOT active in
        # production (RunPodSession.is_active gates them out), even
        # though their status string isn't 'offline'.
        ({"active": False, "status": "terminating"}, False),
        ({"active": False, "status": "error"}, False),
        # Conflicting state — trust the explicit 'active' key over
        # the status string. Production sets active from is_active.
        ({"active": False, "status": "ready"}, False),
        # Test-fake / legacy shapes that omit 'active' → fall back
        # to status-string allowlist that matches is_active semantics.
        ({"status": "ready"}, True),
        ({"status": "provisioning"}, True),
        ({"status": "loading"}, True),
        ({"status": "offline"}, False),
        ({"status": "terminated"}, False),
        ({"status": "terminating"}, False),
        ({"status": "error"}, False),
        # Empty / missing.
        ({}, False),
        ({"status": None}, False),
    ],
)
def test_status_was_active_aligns_with_manager_is_active(pre_status, expected):
    """Codex round 7 catch: ``pre_was_active`` must agree with
    ``RunPodSession.is_active``. If we mis-classify a TERMINATING or
    ERROR session as active, a failed start_session leaks our orphan
    pod (we skip teardown). If we mis-classify a real active session
    as inactive, we tear it down on a failed start. Pin the
    semantics."""
    assert RunPodFeature._status_was_active(pre_status) is expected


@pytest.mark.asyncio
async def test_start_with_terminating_pre_state_tears_down_orphan(monkeypatch):
    """Codex round 7 regression: when preflight returns
    ``{"active": False, "status": "terminating"}``, that's NOT a
    pre-existing active session. If start_session then raises after
    creating a new pod, we MUST tear down our orphan, not skip."""
    stop_called = []

    class _TerminatingThenRaiseManager(FakeRunPodManager):
        async def get_status(self, **_):
            # Old session is still in TERMINATING wind-down — manager
            # treats this as inactive and is willing to start fresh.
            return {"active": False, "status": "terminating"}

        async def start_session(self, **_):
            raise RuntimeError("readiness timeout post-creation")

        async def stop_session(self):
            stop_called.append(True)
            return await FakeRunPodManager.stop_session(self)

    fake_manager = _TerminatingThenRaiseManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="on", task_profile="llm")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert stop_called == [True], (
        "TERMINATING is not pre-active per RunPodSession.is_active; "
        "failed start MUST tear down our orphan pod"
    )


@pytest.mark.asyncio
async def test_logs_confirmation_phrases_tail_not_count(runpod_feature):
    """Codex round 7 catch: ``Retrieved N log line(s)`` lies when
    fewer than N lines actually came back. The confirmation must
    phrase the request, not claim a count."""
    feature, manager, _ = runpod_feature

    async def _short_logs(*, tail):
        # Pod has 2 lines; user requested 100.
        return "line A\nline B\n"

    manager.get_logs = _short_logs

    result = await feature.manage_gpu(action="logs", lines="100")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "100" in result.confirmation
    assert "tail" in result.confirmation.lower(), (
        "confirmation must phrase ``lines`` as the tail REQUEST, not a "
        "count of lines actually retrieved (#1042 honesty contract)"
    )
    # Should NOT claim "Retrieved 100 log line(s)" — that overstates.
    assert "Retrieved 100 log line" not in result.confirmation


@pytest.mark.asyncio
async def test_concurrent_start_does_not_tear_down_winners_pod(monkeypatch):
    """Codex round 8 catch: two concurrent ``!gpu on`` calls each run
    preflight, see inactive, race into start_session. Without
    feature-level serialization, the loser observes the winner's
    session, raises ``"already active"`` from the manager, computes
    ``pre_was_active=False`` from its stale snapshot, and tears down
    the winner's pod.

    This test pins the contract that the feature-level lock
    serializes preflight + start_session, so the loser's preflight
    runs AFTER the winner's session is established and observes
    active=True (correctly skipping teardown)."""
    import asyncio as _asyncio

    class _RaceManager(FakeRunPodManager):
        """Mimics production manager: start_session is serialized by
        an internal lock, but get_status is NOT (it's a snapshot
        read). This is the exact shape that creates the TOCTOU race
        the feature-level lock fixes."""

        def __init__(self):
            super().__init__()
            self._active = False
            self._inner_lock = _asyncio.Lock()
            self._start_calls = 0
            self.stop_calls = 0

        async def get_status(self, **_):
            # No lock — fast snapshot read, like production.
            return {
                "active": self._active,
                "status": "ready" if self._active else "offline",
            }

        async def start_session(self, **kwargs):
            # Serialized by an internal lock, like the production
            # manager's ``async with self._lock`` in core.py.
            async with self._inner_lock:
                if self._active:
                    raise RunPodManagerError(
                        "A RunPod session is already active"
                    )
                # Provider call delay — long enough for the loser's
                # preflight to interleave when the feature lock is
                # absent.
                await _asyncio.sleep(0.02)
                self._start_calls += 1
                self._active = True
                return {
                    "active": True,
                    "status": "ready",
                    "pod_id": f"pod-{self._start_calls}",
                    "task_profile": kwargs["task_profile"],
                    "model_name": "llama-3",
                    "remaining_ttl_seconds": 1800,
                    "inference_url": "http://gpu:8000/v1",
                }

        async def stop_session(self):
            async with self._inner_lock:
                self.stop_calls += 1
                self._active = False
                return {"active": False, "status": "terminating"}

    fake_manager = _RaceManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    # Two concurrent !gpu on calls. With the feature-level lock, the
    # loser's preflight observes the winner's active session and
    # skips teardown. Without the lock, the loser tears down the
    # winner's pod.
    results = await _asyncio.gather(
        feature.manage_gpu(action="on", task_profile="llm"),
        feature.manage_gpu(action="on", task_profile="llm"),
        return_exceptions=True,
    )

    statuses = [r.status for r in results if isinstance(r, ToolResult)]
    # Exactly one OK, one ERROR (the loser saw "already active").
    assert statuses.count(ToolResultStatus.OK) == 1, statuses
    assert statuses.count(ToolResultStatus.ERROR) == 1, statuses
    # CRITICAL: the loser must NOT have torn down the winner's pod.
    assert fake_manager.stop_calls == 0, (
        f"loser tore down winner's pod (stop_calls={fake_manager.stop_calls}); "
        "feature-level lock must serialize preflight + start_session"
    )
    # Winner's session is still active.
    assert fake_manager._active is True


@pytest.mark.asyncio
async def test_start_router_attach_failure_returns_partial(monkeypatch):
    """Codex round 9: ``llm_service.switch_backend`` and
    ``get_backend_status`` were called outside the ToolResult
    envelope. If they raise, the pod is up but the @tool escapes
    with an exception. Pod-up + router-failed should be
    ToolResult.partial — that's exactly the partial-success case."""

    class _BrokenSwitchService(DummyLLMService):
        def switch_backend(self, backend, *, config):
            raise RuntimeError("router adapter broken")

    fake_manager = FakeRunPodManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=_BrokenSwitchService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="on", task_profile="llm")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.PARTIAL, (
        f"expected PARTIAL (pod up, router failed); got {result.status}"
    )
    assert "Started RunPod session" in result.confirmation
    assert "router attach failed" in result.confirmation.lower()
    assert "router adapter broken" in result.error
    assert result.data["session"]["pod_id"] == "pod-123"
    # Pod is still up — confirm the manager actually started it.
    assert fake_manager.started is True


@pytest.mark.asyncio
async def test_stop_router_detach_failure_returns_partial(monkeypatch):
    """Codex round 9: ``_detach_gpu_backend`` was called outside the
    ToolResult envelope in _stop. A broken detach should produce
    ToolResult.partial, not escape as exception. The pod IS stopped
    either way; partial surfaces the router caveat."""

    class _BrokenDetachService(DummyLLMService):
        def _deactivate_remote_backend(self, reason=None):
            raise RuntimeError("router detach broken")

    fake_manager = FakeRunPodManager()
    fake_manager.started = True  # Pretend something is running.
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=_BrokenDetachService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="off")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.PARTIAL
    assert "Stopped RunPod session" in result.confirmation
    assert "router detach failed" in result.confirmation.lower()
    assert "router detach broken" in result.error
    # Pod was actually stopped.
    assert fake_manager.stop_calls == 1


@pytest.mark.asyncio
async def test_status_router_failure_degrades_to_warning(monkeypatch):
    """Codex round 9: for inspection-only commands (status, logs), a
    broken router should NOT fail the whole tool. The user wants the
    manager session info even if the router can't be inspected.
    Surface the router error inside the data payload."""

    class _BrokenRouterService(DummyLLMService):
        def get_backend_status(self):
            raise RuntimeError("router status broken")

    fake_manager = FakeRunPodManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=_BrokenRouterService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="status")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK, (
        "router failure on a read-only status command should NOT fail "
        "the whole tool — degrade gracefully with a warning payload"
    )
    assert "router status broken" in str(result.data["router"])


@pytest.mark.asyncio
async def test_failed_start_surfaces_teardown_error_in_data(monkeypatch):
    """Codex round 9: when start_session raises AND the best-effort
    teardown also fails, the previous code only logged the teardown
    error. Production stop_session clears _session BEFORE
    provider.stop_pod, so a failed stop_pod leaves the manager with
    no handle for retry. The orphan is invisible to the user.
    Surface teardown_error + warning in the ToolResult.failed data."""
    import asyncio as _asyncio

    class _DoubleFailManager(FakeRunPodManager):
        async def get_status(self, **_):
            return {"active": False, "status": "offline"}

        async def start_session(self, **_):
            raise RuntimeError("readiness timeout post-creation")

        async def stop_session(self):
            raise RuntimeError("provider 503 on stop_pod")

    fake_manager = _DoubleFailManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="on", task_profile="llm")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "readiness timeout" in result.error
    # CRITICAL: teardown failure must be surfaced so the user knows
    # there's an orphan they need to clean up out of band.
    assert "teardown_error" in result.data, (
        "failed teardown after failed start was silently logged; the "
        "orphan pod is invisible to the user"
    )
    assert "provider 503 on stop_pod" in result.data["teardown_error"]
    assert "warning" in result.data
    assert "orphan" in result.data["warning"].lower()


@pytest.mark.asyncio
async def test_stop_serializes_with_in_flight_start(monkeypatch):
    """Codex round 9: _stop must take _start_lock to prevent
    stop-during-start races. Production start_session sets _session,
    releases the manager's lock, then runs readiness wait. A
    concurrent !gpu off can clear _session mid-readiness, producing
    a spurious 'start failed' message after the user already chose
    to stop."""
    import asyncio as _asyncio

    start_release = _asyncio.Event()
    stop_observed_lock_held = _asyncio.Event()

    class _SlowStartManager(FakeRunPodManager):
        async def start_session(self, **kwargs):
            # Block inside start_session so we know stop arrives
            # while start is in flight.
            await start_release.wait()
            self.started = True
            return {
                "active": True,
                "status": "ready",
                "pod_id": "pod-1",
                "task_profile": kwargs["task_profile"],
                "model_name": "llama-3",
                "remaining_ttl_seconds": 1800,
                "inference_url": "http://gpu:8000/v1",
            }

    fake_manager = _SlowStartManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    # Kick off a start that blocks inside start_session.
    start_task = _asyncio.create_task(
        feature.manage_gpu(action="on", task_profile="llm")
    )
    # Give start time to acquire _start_lock and enter start_session.
    await _asyncio.sleep(0.05)

    # Kick off a stop. With the round-9 fix, this must wait for
    # start to finish (lock held). Without the fix, stop would
    # proceed immediately.
    stop_task = _asyncio.create_task(feature.manage_gpu(action="off"))
    # Give stop a chance to attempt the lock.
    await _asyncio.sleep(0.05)

    assert not stop_task.done(), (
        "_stop must block on _start_lock while a start is in flight"
    )

    # Release start; stop should now proceed.
    start_release.set()

    start_result, stop_result = await _asyncio.gather(start_task, stop_task)

    assert isinstance(start_result, ToolResult)
    assert isinstance(stop_result, ToolResult)
    assert start_result.status is ToolResultStatus.OK
    assert stop_result.status is ToolResultStatus.OK
    # The serialization order is: start completed first (lock held),
    # then stop saw the started session and stopped it.
    assert fake_manager.stop_calls == 1


@pytest.mark.asyncio
async def test_image_gen_failed_start_with_failed_teardown_surfaces_orphan_warning(
    monkeypatch,
):
    """Codex round 10: generate_image_on_runpod must surface
    teardown_error in the same way as _start. If start_session
    raises AND teardown also fails, the user has an invisible orphan
    pod that's still billing."""
    fake_manager = FakeRunPodManager()

    async def _raise_start(**_):
        raise RuntimeError("readiness post-creation timeout")

    async def _raise_stop():
        raise RuntimeError("provider 503 during teardown")

    async def _inactive_status(**_):
        return {"active": False, "status": "offline"}

    fake_manager.start_session = _raise_start
    fake_manager.stop_session = _raise_stop
    fake_manager.get_status = _inactive_status
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()
    feature._post_json = lambda url, payload: {"unused": True}

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "readiness post-creation timeout" in result.error
    assert result.data["teardown_error"] == "provider 503 during teardown"
    assert "orphan" in result.data["warning"].lower()


@pytest.mark.asyncio
async def test_image_gen_missing_endpoint_with_failed_teardown_surfaces_warning(
    monkeypatch,
):
    """Codex round 10: missing-endpoint path must also surface the
    teardown_error. Without it, a pod that produced no endpoint AND
    failed to stop is invisible to the user — GPU billing
    continues, no warning."""

    class _NoEndpointManager(FakeRunPodManager):
        async def start_session(self, **kwargs):
            self.started = True
            # No image_endpoint, no inference_url.
            return {
                "pod_id": "pod-broken",
                "task_profile": kwargs["task_profile"],
                "model_name": "llama-3",
                "status": "ready",
                "remaining_ttl_seconds": 1800,
            }

        async def stop_session(self):
            raise RuntimeError("provider 504 on stop_pod")

    fake_manager = _NoEndpointManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Image endpoint not provided" in result.error
    assert result.data["teardown_error"] == "provider 504 on stop_pod"
    assert "still be billing" in result.data["warning"].lower()


@pytest.mark.asyncio
async def test_image_gen_success_with_failed_teardown_returns_partial(monkeypatch):
    """Codex round 10: image gen produced an image, but stop_session
    failed. The GPU may still be billing — saying 'pod stopped' in
    a clean OK is the #1042 confident-lie failure mode. Result must
    be PARTIAL with the teardown caveat in confirmation + error."""
    fake_manager = FakeRunPodManager()

    async def _raise_stop():
        raise RuntimeError("provider 502 on stop_pod")

    fake_manager.stop_session = _raise_stop
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()
    feature._post_json = lambda url, payload: {"image_b64": "fake"}

    result = await feature.generate_image_on_runpod(prompt="a sunset")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.PARTIAL, (
        "image succeeded + teardown failed = PARTIAL, not OK; saying "
        "'pod stopped' when it's still billing is a confident lie"
    )
    assert "Generated image" in result.confirmation
    assert "teardown failed" in result.confirmation.lower()
    assert "still be billing" in result.confirmation.lower()
    assert "provider 502" in result.error
    # Image result still surfaces in data.
    assert result.data["result"] == {"image_b64": "fake"}


@pytest.mark.asyncio
async def test_image_gen_failure_with_failed_teardown_warns_about_billing(
    monkeypatch,
):
    """Codex round 10: when image generation FAILS and teardown also
    fails, surface the warning. Two problems: image didn't render,
    AND there's an orphan pod billing."""
    fake_manager = FakeRunPodManager()

    async def _raise_stop():
        raise RuntimeError("provider 502 on stop_pod")

    fake_manager.stop_session = _raise_stop
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )

    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    def _raise_post(url, payload):
        raise RuntimeError("HTTP 500 from image endpoint")

    feature._post_json = _raise_post

    result = await feature.generate_image_on_runpod(prompt="a sunset")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Image generation request failed" in result.error
    assert "still be billing" in result.data["warning"].lower()


@pytest.mark.asyncio
async def test_failed_stop_surfaces_session_before_stop(monkeypatch):
    """Codex round 11: when manager.stop_session raises, production
    has already cleared _session before the provider.stop_pod call
    that failed. The pod may still be billing and the manager has
    lost the handle. _stop must pre-capture session info and surface
    it so the user can stop the pod manually."""

    class _LostHandleManager(FakeRunPodManager):
        def __init__(self):
            super().__init__()
            self._stop_attempted = False

        async def get_status(self, **_):
            # Pre-stop snapshot: an active session.
            return {
                "active": True,
                "status": "ready",
                "pod_id": "pod-orphan-42",
                "task_profile": "llm",
                "model_name": "llama-3",
                "remaining_ttl_seconds": 1500,
            }

        async def stop_session(self):
            # Production behavior: clears _session BEFORE provider
            # call, then raises. Manager has lost the handle.
            raise RuntimeError("provider 504 on stop_pod")

    fake_manager = _LostHandleManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="off")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "provider 504" in result.error
    # CRITICAL: pod_id and warning must be in the data so the user
    # can clean up out-of-band.
    assert "session_before_stop" in result.data, (
        "stop failure didn't surface the captured session; the "
        "manager lost the pod handle and the user has no way to "
        "find the orphan"
    )
    assert result.data["session_before_stop"]["pod_id"] == "pod-orphan-42"
    assert "warning" in result.data
    assert "still be billing" in result.data["warning"].lower()
    assert "manually" in result.data["warning"].lower()


@pytest.mark.asyncio
async def test_failed_stop_with_no_active_session_no_orphan_warning(monkeypatch):
    """Codex round 11 negative case: when stop_session fails AND no
    session was active pre-stop, there's no orphan to warn about.
    Don't add the warning in that case (would confuse the user)."""

    class _NoSessionRaiseManager(FakeRunPodManager):
        async def get_status(self, **_):
            return {"active": False, "status": "offline"}

        async def stop_session(self):
            raise RuntimeError("transient HTTP 503")

    fake_manager = _NoSessionRaiseManager()
    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", lambda: fake_manager
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()

    result = await feature.manage_gpu(action="off")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "transient HTTP 503" in result.error
    # No active pre-stop session → no orphan possible → no warning.
    assert "session_before_stop" not in result.data
    assert "warning" not in result.data


@pytest.mark.asyncio
async def test_disabled_feature_image_gen_returns_failed_not_attribute_error(
    monkeypatch,
):
    """Codex round 11: generate_image_on_runpod must check the
    disabled flag before dereferencing self.manager.default_ttl_seconds.
    Previously raised AttributeError (escaping the @tool envelope)
    on a disabled feature."""

    def _raise_init():
        raise RunPodManagerError("RUNPOD_API_KEY not set")

    monkeypatch.setattr(
        "kestrel_cloud_runpod.feature.RunPodManager", _raise_init
    )
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    await feature.initialize()
    assert feature.disabled is True
    assert feature.manager is None

    result = await feature.generate_image_on_runpod(prompt="anything")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "unavailable" in result.error.lower()
    assert result.data["reason"] == "RUNPOD_API_KEY not set"


@pytest.mark.asyncio
async def test_pre_initialize_returns_failed_not_attribute_error():
    """Codex round 12: if manage_gpu / generate_image_on_runpod is
    invoked before initialize() runs (e.g. test misconfig, framework
    bug), the previous code AttributeError-ed on the manager
    dereference. Now both paths must return ToolResult.failed instead
    of escaping the @tool envelope as the legacy {success: False}
    shape."""
    feature = RunPodFeature(SimpleNamespace(llm_service=DummyLLMService()))
    # Deliberately do NOT call await feature.initialize().
    assert not hasattr(feature, "manager") or feature.manager is None

    for action in ("status", "on", "off", "logs"):
        result = await feature.manage_gpu(action=action)
        assert isinstance(result, ToolResult), action
        assert result.status is ToolResultStatus.ERROR, action
        assert "unavailable" in result.error.lower(), action
        assert "not initialized" in result.data["reason"], action

    image_result = await feature.generate_image_on_runpod(prompt="anything")
    assert isinstance(image_result, ToolResult)
    assert image_result.status is ToolResultStatus.ERROR
    assert "not initialized" in image_result.data["reason"]
