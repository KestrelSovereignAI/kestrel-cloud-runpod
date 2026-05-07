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
        # Feature-level serialization for "preflight get_status +
        # start_session + best-effort teardown on failure". The
        # manager's own lock only protects each method internally; it
        # does NOT cover the read-then-create-then-cleanup decision
        # the feature makes across two manager calls. Without this
        # lock, two concurrent !gpu on calls can both observe inactive
        # state (TOCTOU), one creates a pod, the other gets "already
        # active", computes ``pre_was_active=False`` from its stale
        # snapshot, and tears down the first call's pod (codex
        # round 8 catch).
        self._start_lock = asyncio.Lock()
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
        # Single guard for both disabled AND pre-initialized states
        # (codex round 12). Without this, calling manage_gpu before
        # initialize() runs raises AttributeError for ``manager`` /
        # ``_start_lock``, escaping the @tool envelope through the
        # legacy ``{success: False}`` shape that the audit hook
        # can't read.
        not_ready_reason = self._not_ready_reason()
        if not_ready_reason is not None:
            return ToolResult.failed(
                f"RunPod feature is unavailable: {not_ready_reason}",
                data={"action": action, "reason": not_ready_reason},
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
        # Disabled / pre-initialized guard (codex rounds 11 + 12).
        # ``self.manager = None`` after a failed init; missing
        # ``manager`` attribute entirely if initialize() never ran.
        # Either way, we MUST return ToolResult.failed instead of
        # AttributeError-ing on the next manager dereference.
        not_ready_reason = self._not_ready_reason()
        if not_ready_reason is not None:
            return ToolResult.failed(
                f"RunPod feature is unavailable: {not_ready_reason}",
                data={"action": "generate_image", "reason": not_ready_reason},
            )

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
        # Serialize the preflight + start + teardown sequence against
        # other concurrent acquire-style callers — same shared lock
        # used by ``_start`` (codex round 8). Two concurrent !gpu on
        # / image-gen calls observed inactive in their preflight, both
        # raced into start_session, the loser tore down the winner's
        # pod. The manager's own lock doesn't span this multi-call
        # sequence.
        async with self._start_lock:
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
                pre_was_active = self._status_was_active(pre_status)

            # Catch broadly: start_session can raise raw provider/SDK
            # exceptions AFTER creating a pod (during readiness wait
            # or status refresh). Catching only RunPodManagerError
            # would let those escape and leak a billing-active pod
            # (codex round 6).
            try:
                image_status = await self.manager.start_session(
                    task_profile="image",
                    model_name=model_name,
                    ttl_seconds=ttl,
                )
            except Exception as e:
                # Only attempt teardown if there was nothing active
                # before we tried — anything we created in the failed
                # start_session is fair game; anything that was
                # already there isn't ours to stop.
                # Surface teardown errors (codex round 10): same
                # invisibility issue as _start. The manager's
                # stop_session clears _session before provider.stop_pod,
                # so a failed stop leaves the orphan invisible.
                teardown_error: Optional[str] = None
                if not pre_was_active:
                    try:
                        await self.manager.stop_session()
                    except Exception as stop_err:
                        teardown_error = str(stop_err)
                        logger.warning(
                            f"stop_session after failed start also failed: {stop_err}"
                        )
                fail_data: Dict[str, Any] = {
                    "action": "generate_image",
                    "prompt": prompt,
                }
                if teardown_error is not None:
                    fail_data["teardown_error"] = teardown_error
                    fail_data["warning"] = (
                        "best-effort teardown after failed image-pod start "
                        "also failed; the manager has lost the pod handle. "
                        "Inspect the RunPod console for orphan pods."
                    )
                return ToolResult.failed(str(e), data=fail_data)

        endpoint = image_status.get("image_endpoint") or image_status.get("inference_url")
        if not endpoint:
            # Always stop the pod before returning to prevent cost
            # runaway. Surface teardown failures (codex round 10):
            # if no endpoint AND stop_session also fails, the user
            # has TWO problems (no endpoint, possible orphan), not
            # one.
            no_endpoint_teardown_err: Optional[str] = None
            try:
                await self.manager.stop_session()
            except Exception as stop_err:
                no_endpoint_teardown_err = str(stop_err)
                logger.warning(f"stop_session also failed: {stop_err}")
            no_endpoint_data: Dict[str, Any] = {
                "action": "generate_image",
                "prompt": prompt,
                "session": image_status,
            }
            if no_endpoint_teardown_err is not None:
                no_endpoint_data["teardown_error"] = no_endpoint_teardown_err
                no_endpoint_data["warning"] = (
                    "image pod produced no endpoint AND teardown failed; "
                    "GPU may still be billing. Check provider directly."
                )
            return ToolResult.failed(
                "Image endpoint not provided by pod",
                data=no_endpoint_data,
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
        # teardown are now BOTH logged AND surfaced into the
        # ToolResult — saying "Generated image, pod stopped" when
        # the pod is still billing is the #1042 confident-lie
        # failure mode (codex round 10).
        teardown_failed = False
        try:
            teardown = await self.manager.stop_session()
        except Exception as e:
            teardown_failed = True
            logger.warning(f"stop_session after image gen failed: {e}")
            teardown = {"warning": str(e), "teardown_failed": True}
        # Wrap detach in try/except (codex round 9): the LLM service
        # could raise, and image-gen has already done its work — we
        # don't want a router cleanup failure to hide the result.
        try:
            self._detach_gpu_backend(
                "image generation failed" if image_error is not None
                else "image generation completed"
            )
        except Exception as detach_err:
            logger.warning(f"router detach after image gen failed: {detach_err}")

        if image_error is not None:
            err_data = {
                "action": "generate_image",
                "prompt": prompt,
                "session": image_status,
                "teardown": teardown,
            }
            if teardown_failed:
                err_data["warning"] = (
                    "image generation failed AND pod teardown failed; "
                    "GPU may still be billing. Check provider directly."
                )
            return ToolResult.failed(
                f"Image generation request failed: {image_error}",
                data=err_data,
            )

        # Image generation succeeded. If teardown ALSO succeeded,
        # this is a clean OK. If teardown failed, it's PARTIAL —
        # the user got their image, but there's a billing-active
        # orphan they need to know about.
        ok_data = {
            "action": "generate_image",
            "prompt": prompt,
            "result": image_result,
            "session": image_status,
            "teardown": teardown,
        }
        if teardown_failed:
            return ToolResult.partial(
                confirmation=(
                    f"Generated image for prompt: {prompt[:60]}; "
                    "pod teardown failed (GPU may still be billing)"
                ),
                error=f"teardown error: {teardown.get('warning')}",
                data=ok_data,
            )

        logger.info("✅ RunPod image generation complete, pod stopped")
        return ToolResult.ok(
            confirmation=f"Generated image for prompt: {prompt[:60]}",
            data=ok_data,
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

        # Serialize the preflight + start + cleanup decision against
        # other concurrent acquire-style callers (codex round 8). The
        # manager's lock only covers each method body; it doesn't span
        # our read-then-attempt-then-discriminate sequence.
        async with self._start_lock:
            # Pre-flight: was a session already active before we
            # touched the manager? Discriminates orphaned-pod cleanup
            # (safe) from tearing down an unrelated active session
            # (catastrophic). Catch broadly: get_status itself can
            # raise raw provider exceptions; fail safe by assuming a
            # session was active so we never auto-teardown on
            # unreadable state.
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
                pre_was_active = self._status_was_active(pre_status)

            # Catch broadly: start_session can raise raw provider/SDK
            # exceptions (HTTPError, TimeoutError, etc.) AFTER
            # creating a pod, during the readiness wait or status
            # refresh. Those would otherwise escape the @tool envelope
            # and leave a billing-active pod the user can't see
            # (codex round 6).
            try:
                status = await self.manager.start_session(
                    task_profile=profile_key,
                    model_name=target_model,
                    ttl_seconds=ttl,
                    pod_type=pod_type or None,
                    metadata=metadata,
                )
            except Exception as e:
                # Best-effort teardown only when there was nothing
                # active before — any session state we leave behind is
                # OUR doing (post-creation readiness raise, etc.). If
                # a session was already active, it's not ours to stop.
                # Surface teardown errors in the ToolResult data
                # (codex round 9): production stop_session clears
                # ``_session`` BEFORE provider.stop_pod, so a failed
                # stop_pod leaves the manager with no handle for a
                # later retry. The orphan is invisible to the user
                # unless we tell them.
                teardown_error: Optional[str] = None
                if not pre_was_active:
                    try:
                        await self.manager.stop_session()
                    except Exception as stop_err:
                        teardown_error = str(stop_err)
                        logger.warning(
                            f"stop_session after failed start also failed: {stop_err}"
                        )
                fail_data: Dict[str, Any] = {"action": "start"}
                if teardown_error is not None:
                    fail_data["teardown_error"] = teardown_error
                    fail_data["warning"] = (
                        "best-effort teardown after failed start_session "
                        "also failed; manager may have lost the pod handle "
                        "(stop_session clears _session before provider.stop_pod). "
                        "Inspect the RunPod console directly for orphan pods."
                    )
                return ToolResult.failed(str(e), data=fail_data)

        # Wrap router/attach in try/except (codex round 9): if
        # ``llm_service.switch_backend`` or ``get_backend_status``
        # raises (broken adapter, unexpected backend state, etc.),
        # the pod is up but the @tool would still escape with an
        # exception. Pod started + router failed = ToolResult.partial,
        # which is exactly the partial-success case the envelope was
        # designed for.
        try:
            if profile_key == "llm":
                self._attach_gpu_backend(status)
            router_payload = self._router_status()
        except Exception as router_err:
            logger.error(
                f"Router/attach failed after successful pod start: {router_err}"
            )
            return ToolResult.partial(
                confirmation=(
                    f"Started RunPod session (profile: {profile_key}); "
                    "LLM router attach failed"
                ),
                error=f"router attach error: {router_err}",
                data={
                    "action": "start",
                    "session": status,
                    "router_error": str(router_err),
                },
            )

        return ToolResult.ok(
            confirmation=f"Started RunPod session (profile: {profile_key})",
            data={
                "action": "start",
                "session": status,
                "router": router_payload,
            },
        )

    async def _stop(self) -> ToolResult:
        """Stop the current pod.

        Branch the confirmation between actual-stop and no-op so the
        LLM doesn't narrate "Stopped" when there was nothing to
        stop (#1042 honesty contract). The manager returns a status
        like ``terminating`` or ``terminated`` when there was a pod
        and ``offline`` when the call was a no-op.

        Acquires the same ``_start_lock`` used by ``_start`` to
        prevent stop-vs-start races (codex round 9): if a !gpu on
        is in its readiness wait, the manager has already returned
        from start_session but the feature is still inside its
        guarded section. Without this lock, a concurrent !gpu off
        could clear ``_session`` while readiness polls, producing
        a spurious "start failed" message after the user already
        chose to stop.
        """
        async with self._start_lock:
            # Pre-capture session info BEFORE stop_session
            # (codex round 11). Production stop_session clears
            # ``_session`` before calling provider.stop_pod, so if
            # provider.stop_pod fails the manager has lost the pod
            # handle for any retry. The user must be told both
            # "stop failed" AND "pod may still be billing — here's
            # the pod_id we tried to stop" so they can clean up
            # out-of-band. Catch broadly here too — pre-capture
            # itself can fail.
            session_before_stop: Optional[Dict[str, Any]] = None
            try:
                session_before_stop = await self.manager.get_status()
            except Exception as pre_err:
                logger.warning(
                    f"Pre-stop get_status failed; cannot capture pod "
                    f"handle for orphan-warning: {pre_err}"
                )
            # Catch broadly: provider/SDK calls inside stop_session
            # can raise raw HTTPError/Timeout that aren't
            # RunPodManagerError; those would otherwise escape the
            # @tool envelope (codex round 6 catch).
            try:
                status = await self.manager.stop_session()
            except Exception as e:
                fail_data: Dict[str, Any] = {"action": "stop"}
                # If a session was active pre-stop, the manager has
                # already cleared its handle by the time
                # provider.stop_pod raised. Surface the captured
                # session so the user can stop the pod manually.
                if (
                    session_before_stop is not None
                    and self._status_was_active(session_before_stop)
                ):
                    fail_data["session_before_stop"] = session_before_stop
                    fail_data["warning"] = (
                        "stop_session failed AFTER the manager cleared "
                        "its session handle; the pod may still be "
                        "billing. session_before_stop preserves the "
                        "pod_id so you can stop it manually via the "
                        "RunPod console."
                    )
                return ToolResult.failed(str(e), data=fail_data)
            # Wrap router detach (codex round 9). If the LLM service
            # raises during detach (broken adapter, etc.), the pod IS
            # stopped — return partial with the teardown caveat.
            try:
                self._detach_gpu_backend("Requested via !gpu off")
                router_payload = self._router_status()
            except Exception as router_err:
                logger.error(
                    f"Router detach failed after successful pod stop: {router_err}"
                )
                was_no_op = (
                    status.get("status") == "offline" and not status.get("active")
                )
                base_confirmation = (
                    "No active RunPod session to stop (no-op)"
                    if was_no_op
                    else "Stopped RunPod session"
                )
                return ToolResult.partial(
                    confirmation=f"{base_confirmation}; LLM router detach failed",
                    error=f"router detach error: {router_err}",
                    data={
                        "action": "stop",
                        "session": status,
                        "router_error": str(router_err),
                    },
                )
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
                "router": router_payload,
            },
        )

    async def _status(self) -> ToolResult:
        # Catch broadly: provider/SDK calls inside get_status can
        # raise raw HTTPError/Timeout that aren't RunPodManagerError;
        # those would otherwise escape the @tool envelope (codex round
        # 6 catch).
        #
        # Known limitation (codex round 10): _status does NOT take
        # ``_start_lock``. Manager.stop_session clears ``_session``
        # before calling provider.stop_pod, so a concurrent !gpu
        # status during a slow stop will return ``offline`` while
        # the provider tear-down is still in flight. Acquiring the
        # lock would block frequent UI status polls for the full
        # duration of provider.stop_pod (often 10-30s). Accepted as
        # a transient honesty gap; the deeper fix is in the manager
        # (refresh from provider before returning offline).
        try:
            status = await self.manager.get_status()
        except Exception as e:
            return ToolResult.failed(str(e))
        return ToolResult.ok(
            confirmation=f"RunPod session status: {status.get('status', 'unknown')}",
            data={
                "action": "status",
                "session": status,
                "router": self._safe_router_status(),
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
        # ``lines`` is the tail REQUEST, not the count returned. The
        # pod may have fewer lines, or the manager may truncate.
        # Saying "Retrieved N log line(s)" overstates the result if
        # fewer were available (#1042 honesty contract). Phrase the
        # confirmation as the request itself.
        return ToolResult.ok(
            confirmation=f"Retrieved RunPod logs (tail: {lines})",
            data={
                "action": "logs",
                "lines": lines,
                "logs": logs,
                "router": self._safe_router_status(),
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

    def _safe_router_status(self) -> Optional[Dict[str, Any]]:
        """Read-only router-status fetch that degrades to a warning
        payload if the LLM service raises. Used by ``_status`` and
        ``_logs`` (codex round 9): for inspection-only commands, a
        broken router shouldn't fail the whole tool — surface the
        error in the data payload instead so the user sees both the
        manager session info and the router caveat.
        """
        try:
            return self._router_status()
        except Exception as e:
            logger.warning(f"router_status read failed: {e}")
            return {"warning": f"router_status unavailable: {e}"}

    def _not_ready_reason(self) -> Optional[str]:
        """Return a human-readable reason if the feature isn't ready
        to serve commands, or None if it is. Covers both:

        - Disabled (initialize ran but the manager wouldn't construct
          — e.g. missing RUNPOD_API_KEY → ``self.disabled = True``).
        - Pre-initialized (initialize never ran, so ``manager`` /
          ``_start_lock`` attributes don't exist yet).

        Without this guard, the second case would AttributeError on
        the first manager dereference and escape the @tool envelope
        as the legacy ``{success: False}`` shape (codex round 12).
        """
        if getattr(self, "disabled", False):
            return getattr(self, "disabled_reason", "RUNPOD_API_KEY not set")
        if not hasattr(self, "manager") or self.manager is None:
            return "feature not initialized (initialize() has not been called)"
        if not hasattr(self, "_start_lock"):
            return "feature not initialized (initialize() has not been called)"
        return None

    @staticmethod
    def _status_was_active(pre_status: Dict[str, Any]) -> bool:
        """Was a manager session active before we touched it?

        This is the load-bearing predicate for orphan-pod cleanup. If
        we get it wrong:
        - True when the manager would actually accept a new
          start_session: we skip teardown on a failed start, leaking
          OUR orphan pod (the one we just created).
        - False when there really IS an unrelated active session: we
          tear it down on a failed start, destroying user state.

        Production manager (RunPodManager.get_status) returns the
        ``active`` key explicitly, computed from
        ``RunPodSession.is_active`` — which is False for OFFLINE,
        TERMINATING, and ERROR. Trust ``active`` when present.

        For test fakes / older code paths that don't set ``active``,
        fall back to the inactive-states allowlist that matches
        ``is_active``: include TERMINATING and ERROR (not just
        OFFLINE/TERMINATED) so a manager in a non-active terminal
        state isn't treated as "session in progress".
        """
        active_value = pre_status.get("active")
        if active_value is not None:
            return bool(active_value)
        return pre_status.get("status") not in {
            "offline",
            "terminating",
            "terminated",
            "error",
            None,
        }

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
