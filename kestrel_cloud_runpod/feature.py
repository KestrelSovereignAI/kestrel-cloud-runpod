import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from kestrel_sdk.features.base import Feature, tool
from kestrel_cloud_runpod.manager import RunPodManager
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_sdk.llm.types import BackendType
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class RunPodFeature(Feature):
    """Feature layer that exposes GPU orchestration via the tool system."""

    @property
    def tool_description(self) -> str:
        return (
            "Manage RunPod GPU sessions - start and stop on-demand GPU pods for "
            "heavy inference, check status, view logs, and route LLM traffic to GPU"
        )

    async def initialize(self):
        try:
            self.manager = RunPodManager()
        except (RunPodManagerError, ImportError) as e:
            logger.warning(f"RunPodFeature disabled: {e}")
            self.manager = None
            self.disabled = True
            self.disabled_reason = str(e)
            return
        self.disabled = False
        self.llm_service = getattr(self.agent, "llm_service", None)
        if not self.llm_service:
            logger.warning("LLMService not available; GPU routing disabled")

    @tool(
        name="manage_gpu",
        description="Start, stop, or inspect RunPod GPU sessions (usage: !gpu <action> [...]).",
        category=ToolCategory.SYSTEM,
        command_prefix="!gpu",
    )
    async def manage_gpu(
        self,
        action: str = "status",
        model_name: str = "",
        task_profile: str = "llm",
        ttl_seconds: str = "",
        pod_type: str = "",
        lines: str = "100",
    ) -> ToolResult:
        if getattr(self, 'disabled', False):
            return ToolResult.failed(
                "RunPod feature is disabled",
                data={
                    "action": action,
                    "reason": getattr(
                        self, "disabled_reason", "RUNPOD_API_KEY not set"
                    ),
                },
            )
        action_normalized = (action or "status").lower()

        if action_normalized in {"status"}:
            return await self._status()
        if action_normalized in {"logs", "log"}:
            # Pre-flight coerce ``lines`` so a non-numeric value lands
            # in the ToolResult envelope, not as a raised ValueError
            # (#1042 layer 4b honesty contract).
            try:
                num_lines = self._coerce_optional_int(lines) or 100
            except ValueError:
                return ToolResult.failed(
                    f"Invalid lines '{lines}'. Expected a non-negative integer.",
                    data={"argument": "lines", "received": lines},
                )
            return await self._logs(lines=num_lines)
        if action_normalized in {"off", "stop"}:
            return await self._stop()
        if action_normalized in {"on", "start"}:
            return await self._start(
                model_name=model_name,
                task_profile=task_profile,
                ttl_seconds=ttl_seconds,
                pod_type=pod_type,
            )
        return ToolResult.failed(
            f"Unsupported GPU action: {action}. Use on, off, status, or logs.",
            data={
                "available_actions": ["status", "on", "off", "logs"],
            },
        )

    # NOTE: !dream command removed
    # generate_image_on_runpod() exists below but NOTHING CALLS IT
    # visual_identity feature exists but is also NOT USED
    

    async def generate_image_on_runpod(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> ToolResult:
        """
        Generate an image on a RunPod GPU.

        Currently has no @tool decorator (no command surface) but is
        exercised by the integration test suite, so it's migrated to
        the ToolResult envelope alongside the rest of the feature.
        Automatically starts a pod, generates the image, and stops
        the pod.

        Args:
            prompt: Image generation prompt
            model_name: Optional model to use
            ttl_seconds: Optional TTL (default: min of default_ttl or 900)

        Returns:
            ``ToolResult.ok(confirmation, data={result, session, teardown, ...})``
            on success; ``ToolResult.failed(error, data=...)`` on
            validation or manager error. Pod is always stopped before
            returning, even on the failure paths, so cost runaway
            cannot persist.
        """
        prompt = prompt.strip()
        if not prompt:
            return ToolResult.failed(
                "Prompt is required for image generation",
                data={"argument": "prompt"},
            )

        ttl = ttl_seconds or min(self.manager.default_ttl_seconds, 900)

        # Capture whether a session was already active before we
        # touched the manager. This is the load-bearing signal for
        # whether a teardown after a failed ``start_session`` would
        # tear down OUR orphaned pod (safe) or someone else's
        # active session (catastrophic).
        #
        # If a session was active before we started:
        #   - The manager will refuse to overwrite ("already active"
        #     and other pre-creation validations).
        #   - That session is NOT ours — never call stop_session.
        #
        # If no session was active before:
        #   - Any session state we leave behind on a raise is OUR
        #     doing (readiness-timeout post-creation, etc.).
        #   - stop_session is safe to call: either it cleans up our
        #     orphaned pod, or it's a no-op when nothing was created
        #     (TTL-too-high / missing-default-model and other pre-
        #     creation validation failures).
        # Catch BROADLY here, not just RunPodManagerError. The
        # ``get_status`` call can perform a provider refresh that
        # raises requests.exceptions.HTTPError, TimeoutError, etc.
        # when the RunPod/managed API is degraded. Those raw
        # exceptions would otherwise escape generate_image_on_runpod
        # before it can return ToolResult.failed. Failing safe by
        # assuming a session was active (skip teardown) is the
        # right call when status is unreadable: if we can't tell
        # whether we'd be tearing down our own pod or someone
        # else's, we err on the side of touching nothing.
        try:
            pre_status = await self.manager.get_status()
        except Exception as e:
            logger.warning(
                f"Pre-flight get_status failed; "
                f"assuming a session may be active and skipping any "
                f"teardown on subsequent start_session failure: {e}"
            )
            pre_was_active = True
        else:
            pre_was_active = bool(pre_status.get("active")) or (
                pre_status.get("status") not in {"offline", "terminated", None}
            )

        # Catch broadly: start_session can raise raw provider/SDK
        # exceptions AFTER creating a pod (during readiness wait or
        # status refresh). Catching only RunPodManagerError would let
        # those escape and leak a billing-active pod (codex round 6).
        try:
            image_status = await self.manager.start_session(
                task_profile="image",
                model_name=model_name,
                ttl_seconds=ttl,
            )
        except Exception as e:
            # Only attempt teardown if there was nothing active
            # before we tried — anything we created in the failed
            # start_session is fair game; anything that was already
            # there isn't ours to stop.
            if not pre_was_active:
                try:
                    await self.manager.stop_session()
                except Exception as stop_err:
                    logger.warning(
                        f"stop_session after failed start also failed: {stop_err}"
                    )
            return ToolResult.failed(str(e))

        endpoint = image_status.get("image_endpoint") or image_status.get("inference_url")
        if not endpoint:
            # Always stop the pod before returning to prevent cost
            # runaway. We swallow stop errors here — the primary
            # failure (missing endpoint) is what the caller needs.
            # Catch broadly: provider exceptions in stop_session can
            # also be non-RunPodManagerError (codex round 6).
            try:
                await self.manager.stop_session()
            except Exception as stop_err:
                logger.warning(f"stop_session also failed: {stop_err}")
            return ToolResult.failed(
                "Image endpoint not provided by pod",
                data={"session": image_status},
            )

        payload = {"prompt": prompt, "model": model_name or image_status.get("model_name")}

        # ``_post_json`` can raise from requests (HTTPError, Timeout,
        # ConnectionError) when the pod is up but the inference call
        # fails. We MUST tear down the pod on every exit path,
        # successful or not — otherwise an HTTP 500 from the image
        # endpoint leaves an idle GPU billing until TTL expiry. The
        # exception is also captured into the ToolResult.failed
        # envelope (#1042 layer 4b) instead of escaping the @tool.
        image_result: Optional[Dict[str, Any]] = None
        image_error: Optional[Exception] = None
        try:
            image_result = await asyncio.to_thread(self._post_json, endpoint, payload)
        except Exception as e:
            image_error = e
            logger.error(f"Image generation request failed: {e}")

        # CRITICAL: stop the pod whether image generation succeeded
        # or the request raised. ``stop_session`` errors during
        # teardown are logged but don't override the primary result.
        # Catch broadly: provider exceptions in stop_session can also
        # be non-RunPodManagerError (codex round 6).
        try:
            teardown = await self.manager.stop_session()
        except Exception as e:
            logger.warning(f"stop_session after image gen failed: {e}")
            teardown = {"warning": str(e)}
        self._detach_gpu_backend(
            "image generation failed" if image_error is not None
            else "image generation completed"
        )

        if image_error is not None:
            return ToolResult.failed(
                f"Image generation request failed: {image_error}",
                data={
                    "action": "generate_image",
                    "prompt": prompt,
                    "session": image_status,
                    "teardown": teardown,
                },
            )

        logger.info("✅ RunPod image generation complete, pod stopped")

        return ToolResult.ok(
            confirmation=f"Generated image for prompt: {prompt[:60]}",
            data={
                "action": "generate_image",
                "prompt": prompt,
                "result": image_result,
                "session": image_status,
                "teardown": teardown,
            },
        )

    async def _start(
        self,
        *,
        model_name: str,
        task_profile: str,
        ttl_seconds: str,
        pod_type: str,
    ) -> ToolResult:
        profile_key = (task_profile or "llm").lower()
        if profile_key not in self.manager.profiles:
            # Validation failure → ToolResult.failed (NOT an exception).
            # See #1042 layer 4b honesty contract.
            available = list(self.manager.profiles.keys())
            return ToolResult.failed(
                f"Unknown task_profile '{task_profile}'. "
                f"Available: {', '.join(available)}",
                data={"available_profiles": available},
            )

        try:
            ttl = self._coerce_optional_int(ttl_seconds)
        except ValueError:
            return ToolResult.failed(
                f"Invalid ttl_seconds '{ttl_seconds}'. Expected a non-negative integer.",
                data={"argument": "ttl_seconds", "received": ttl_seconds},
            )
        target_model = model_name or None
        env_overrides = {
            "KESTREL_PROFILE": profile_key,
        }
        if target_model:
            env_overrides["TARGET_MODEL"] = target_model

        metadata = {
            "name": f"kestrel-{profile_key}-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            "env_overrides": env_overrides,
            "pod_type": pod_type or None,
        }

        # Pre-flight: was a session already active before we touched
        # the manager? Same load-bearing signal as
        # generate_image_on_runpod — discriminates orphaned-pod cleanup
        # (safe) from tearing down an unrelated active session
        # (catastrophic). Catch broadly: get_status itself can raise
        # raw provider exceptions; fail safe by assuming a session was
        # active so we never auto-teardown on unreadable state.
        try:
            pre_status = await self.manager.get_status()
        except Exception as e:
            logger.warning(
                f"Pre-flight get_status before start failed; "
                f"assuming a session may be active and skipping any "
                f"teardown on subsequent start_session failure: {e}"
            )
            pre_was_active = True
        else:
            pre_was_active = bool(pre_status.get("active")) or (
                pre_status.get("status") not in {"offline", "terminated", None}
            )

        # Catch broadly: start_session can raise raw provider/SDK
        # exceptions (HTTPError, TimeoutError, etc.) AFTER creating a
        # pod, during the readiness wait or status refresh. Those
        # would otherwise escape the @tool envelope and leave a
        # billing-active pod the user can't see (codex round 6).
        try:
            status = await self.manager.start_session(
                task_profile=profile_key,
                model_name=target_model,
                ttl_seconds=ttl,
                pod_type=pod_type or None,
                metadata=metadata,
            )
        except Exception as e:
            # Best-effort teardown only when there was nothing active
            # before — any session state we leave behind is OUR doing
            # (post-creation readiness raise, etc.). If a session was
            # already active, it's not ours to stop.
            if not pre_was_active:
                try:
                    await self.manager.stop_session()
                except Exception as stop_err:
                    logger.warning(
                        f"stop_session after failed start also failed: {stop_err}"
                    )
            return ToolResult.failed(str(e))

        if profile_key == "llm":
            self._attach_gpu_backend(status)

        return ToolResult.ok(
            confirmation=f"Started RunPod session (profile: {profile_key})",
            data={
                "action": "start",
                "session": status,
                "router": self._router_status(),
            },
        )

    async def _stop(self) -> ToolResult:
        """Stop the current pod.

        Branch the confirmation between actual-stop and no-op so the
        LLM doesn't narrate "Stopped" when there was nothing to
        stop (#1042 honesty contract). The manager returns a status
        like ``terminating`` or ``terminated`` when there was a pod
        and ``offline`` when the call was a no-op.
        """
        # Catch broadly: provider/SDK calls inside stop_session can
        # raise raw HTTPError/Timeout that aren't RunPodManagerError;
        # those would otherwise escape the @tool envelope (codex round
        # 6 catch).
        try:
            status = await self.manager.stop_session()
        except Exception as e:
            return ToolResult.failed(str(e))
        self._detach_gpu_backend("Requested via !gpu off")
        was_no_op = status.get("status") == "offline" and not status.get("active")
        confirmation = (
            "No active RunPod session to stop (no-op)"
            if was_no_op
            else "Stopped RunPod session"
        )
        return ToolResult.ok(
            confirmation=confirmation,
            data={
                "action": "stop",
                "session": status,
                "router": self._router_status(),
            },
        )

    async def _status(self) -> ToolResult:
        # Catch broadly: provider/SDK calls inside get_status can
        # raise raw HTTPError/Timeout that aren't RunPodManagerError;
        # those would otherwise escape the @tool envelope (codex round
        # 6 catch).
        try:
            status = await self.manager.get_status()
        except Exception as e:
            return ToolResult.failed(str(e))
        return ToolResult.ok(
            confirmation=f"RunPod session status: {status.get('status', 'unknown')}",
            data={
                "action": "status",
                "session": status,
                "router": self._router_status(),
            },
        )

    async def _logs(self, lines: int) -> ToolResult:
        # ``RunPodManager.get_logs`` takes ``tail=``, not ``lines=``.
        # Also catch broadly: provider/SDK calls inside get_logs can
        # raise raw HTTPError/Timeout that aren't RunPodManagerError;
        # those would otherwise escape the @tool envelope (codex round
        # 6 catch).
        try:
            logs = await self.manager.get_logs(tail=lines)
        except Exception as e:
            return ToolResult.failed(str(e))
        return ToolResult.ok(
            confirmation=f"Retrieved {lines} log line(s) from RunPod",
            data={
                "action": "logs",
                "lines": lines,
                "logs": logs,
                "router": self._router_status(),
            },
        )

    def _attach_gpu_backend(self, session_status: Dict[str, Any]) -> None:
        if not self.llm_service:
            logger.warning("Cannot attach GPU backend without LLMService")
            return

        base_url = session_status.get("inference_url")
        if not base_url:
            logger.warning("RunPod session missing inference URL; skipping activation")
            return

        remaining = session_status.get("remaining_ttl_seconds")
        profile_id = session_status.get("task_profile")
        profile = self.manager.profiles.get(profile_id)
        context_window = profile.max_context_window if profile else None

        config = {
            "base_url": base_url,
            "model": session_status.get("model_name"),
            "ttl_seconds": remaining,
            "context_window": context_window,
            "metadata": {
                "pod_id": session_status.get("pod_id"),
                "profile": profile_id,
            },
        }
        self.llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

    def _detach_gpu_backend(self, reason: str) -> None:
        if self.llm_service:
            self.llm_service._deactivate_remote_backend(reason=reason)

    def _router_status(self) -> Optional[Dict[str, Any]]:
        if not self.llm_service:
            return None
        return self.llm_service.get_backend_status()

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        if not text.isdigit():
            raise ValueError("TTL must be an integer number of seconds")
        return int(text)

    @staticmethod
    def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
