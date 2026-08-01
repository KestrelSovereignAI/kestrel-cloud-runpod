"""
RunPod Core Manager Operations.

Contains the core RunPodManagerCore class with SDK operations,
profile loading, and session management.
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.config.constants import RUNPOD_URL_POLL_INTERVAL
from kestrel_sovereign.config import load_config

from .models import (
    CloudType,
    GPUProfile,
    PlacementDecision,
    PodStatus,
    RunPodAmbiguousResultError,
    RunPodManagerError,
    RunPodSession,
    sanitize_resource_payload,
)
from .providers import DirectRunPodProvider, GPUProvider, ManagedRunPodProvider

logger = logging.getLogger(__name__)


class RunPodManagerCore:
    """
    Core RunPod operations.

    Handles SDK initialization, profile loading, session management,
    and basic pod lifecycle.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, mode: Optional[str] = None
    ):
        self.config = config or load_config("runpod_config.toml")
        self.manager_config = self.config.get("manager", {})
        self.mode = (
            mode
            or os.getenv("RUNPOD_MODE")
            or self.manager_config.get("mode", "direct")
        )
        self.default_ttl_seconds = int(
            os.getenv(
                "GPU_DEFAULT_TTL_SECONDS",
                self.manager_config.get("default_ttl_seconds", 1800),
            )
        )
        self.max_ttl_seconds = int(
            self.manager_config.get("max_ttl_seconds", self.default_ttl_seconds)
        )
        self.poll_interval = int(self.manager_config.get("poll_interval_seconds", 10))
        self.readiness_timeout = int(
            self.manager_config.get("readiness_timeout_seconds", 600)
        )
        self.profiles = self._load_profiles(self.config.get("profiles", {}))
        if not self.profiles:
            raise RunPodManagerError(
                "No GPU profiles configured. Create runpod_config.toml."
            )
        self.provider = self._build_provider()
        self._session: Optional[RunPodSession] = None
        self._lock = asyncio.Lock()

        # Metering callback for usage billing (Vending Machine)
        # Set via set_metering_callback() after initialization
        self._metering_callback = None

    def set_metering_callback(self, callback) -> None:
        """Set the metering callback for GPU usage billing (Vending Machine).

        The callback will be called when a session ends with:
            await callback(
                companion_id=str,
                user_id=str,
                provider=str,  # 'runpod'
                resource_type=str,  # GPU type
                duration_seconds=float,
                operation_id=str,  # pod_id
            )

        Args:
            callback: Async function to call when session ends
        """
        self._metering_callback = callback
        logger.info("RunPod metering enabled")

    def _load_profiles(self, raw_profiles: Dict[str, Any]) -> Dict[str, GPUProfile]:
        profiles: Dict[str, GPUProfile] = {}
        for key, data in raw_profiles.items():
            try:
                legacy_fields = {
                    "gpu_type_id",
                    "cost_per_hr",
                    "vram_gb",
                    "template_id",
                }.intersection(data)
                if legacy_fields:
                    names = ", ".join(sorted(legacy_fields))
                    raise RunPodManagerError(
                        f"Profile '{key}' uses legacy Runpod fields ({names}). "
                        "Migrate to min_vram_gb/min_cuda_version/max_cost_per_hr "
                        "and registry_id; live v2 catalog data now selects and "
                        "prices GPUs."
                    )
                raw_env = data.get("env", {})
                if not isinstance(raw_env, dict):
                    raise RunPodManagerError(
                        f"Profile '{key}' env must be a string mapping"
                    )

                profiles[key] = GPUProfile(
                    id=data.get("id", key),
                    name=data["name"],
                    task_type=data.get("task_type", key),
                    image_name=data["image_name"],
                    container_disk_gb=int(data.get("container_disk_gb", 50)),
                    volume_gb=int(data.get("volume_gb", 0)),
                    ports=data.get("ports", ["8888/http"]),
                    inference_port=int(data.get("inference_port", 8888)),
                    inference_protocol=data.get("inference_protocol", "http"),
                    inference_base_path=data.get("inference_base_path", "/v1"),
                    image_invoke_path=data.get("image_invoke_path"),
                    default_model=data.get("default_model"),
                    pod_type=data.get("pod_type"),
                    min_vram_gb=int(data["min_vram_gb"]),
                    min_cuda_version=data.get("min_cuda_version"),
                    max_cost_per_hr=data.get("max_cost_per_hr"),
                    gpu_count=int(data.get("gpu_count", 1)),
                    cloud=CloudType(
                        str(
                            data.get(
                                "cloud",
                                self.manager_config.get("cloud_type", "SECURE"),
                            )
                        ).upper()
                    ),
                    allowed_gpu_ids=tuple(data.get("allowed_gpu_ids", ())),
                    allowed_data_center_ids=tuple(
                        data.get("allowed_data_center_ids", ())
                    ),
                    max_context_window=data.get("max_context_window"),
                    readiness_timeout_seconds=data.get("readiness_timeout_seconds"),
                    registry_id=data.get("registry_id"),
                    network_volume_id=data.get(
                        "network_volume_id"
                    ),  # Persistent network storage
                    volume_mount_path=data.get(
                        "volume_mount_path"
                    ),  # Mount path (e.g., /workspace)
                    persistent_pod_id=data.get(
                        "persistent_pod_id"
                    ),  # Raw value - expanded at runtime
                    env=dict(raw_env),
                )
            except KeyError as exc:
                raise RunPodManagerError(
                    f"Incomplete profile '{key}': missing {exc}"
                ) from exc
            except ValueError as exc:
                raise RunPodManagerError(f"Invalid profile '{key}': {exc}") from exc
        return profiles

    @staticmethod
    def _expand_single_env_var(value: Optional[str]) -> Optional[str]:
        """Expand ${VAR} syntax in a single string value."""
        if not value or not isinstance(value, str) or "${" not in value:
            return value

        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")  # Empty string if not set

        expanded = re.sub(r"\$\{([^}]+)\}", replace_var, value)
        return expanded if expanded else None  # Return None if result is empty

    def _build_provider(self) -> GPUProvider:
        if self.mode == "managed":
            api_base = os.getenv("KESTREL_API_BASE") or self.manager_config.get(
                "managed_api_base"
            )
            api_key = os.getenv("KESTREL_API_KEY") or self.manager_config.get(
                "managed_api_key"
            )
            return ManagedRunPodProvider(api_base=api_base, api_key=api_key)
        api_key = os.getenv("RUNPOD_API_KEY")
        cloud_type = os.getenv(
            "RUNPOD_CLOUD_TYPE", self.manager_config.get("cloud_type", "SECURE")
        )
        control_plane_base_url = os.getenv(
            "RUNPOD_CONTROL_PLANE_BASE_URL"
        ) or self.manager_config.get("control_plane_base_url")
        user_agent = os.getenv("RUNPOD_USER_AGENT") or self.manager_config.get(
            "user_agent"
        )
        return DirectRunPodProvider(
            api_key=api_key,
            cloud_type=cloud_type,
            control_plane_base_url=control_plane_base_url,
            user_agent=user_agent,
        )

    async def start_session(
        self,
        task_profile: str,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        pod_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = self._select_profile(task_profile)
        ttl = self._validate_ttl(ttl_seconds)
        chosen_model = model_name or profile.default_model
        if not chosen_model:
            raise RunPodManagerError(
                "Model name is required when profile has no default_model configured"
            )
        metadata = metadata or {}
        async with self._lock:
            if self._session and self._session.is_active:
                raise RunPodManagerError("A RunPod session is already active")
            try:
                response = await asyncio.to_thread(
                    self.provider.start_pod, profile, metadata
                )
            except RunPodAmbiguousResultError:
                # The POST may have succeeded. Preserve the typed reconciliation
                # signal so no caller can create a replacement resource first.
                raise
            except RunPodManagerError as exc:
                raise RunPodManagerError(f"Failed to create Pod: {exc}") from exc
            pod_id = response.get("id") or response.get("podId")
            if not pod_id:
                raise RunPodManagerError("RunPod did not return a pod id")
            placement = response.pop("_kestrel_placement", None)
            if placement is not None and not isinstance(placement, PlacementDecision):
                raise RunPodManagerError(
                    "Runpod provider returned invalid placement metadata"
                )
            gpu = response.get("gpu") or {}
            started_at = datetime.now(timezone.utc)
            self._session = RunPodSession(
                pod_id=pod_id,
                profile=profile,
                task_profile=task_profile,
                model_name=chosen_model,
                pod_type=pod_type or profile.pod_type,
                status=PodStatus.PROVISIONING,
                ttl_seconds=ttl,
                started_at=started_at,
                expires_at=started_at + timedelta(seconds=ttl),
                gpu_type_id=(placement.gpu_id if placement else gpu.get("id")),
                cost_per_hr=(
                    placement.offered_cost_per_hr if placement else response.get("cost")
                ),
                placement=placement,
            )
        await self._wait_until_ready()
        return await self.get_status()

    async def get_status(self, refresh: bool = True) -> Dict[str, Any]:
        async with self._lock:
            session = self._session
        if not session:
            return {"active": False, "status": PodStatus.OFFLINE.value}
        if refresh:
            pod_info = await asyncio.to_thread(self.provider.get_status, session.pod_id)
            self._update_session_from_runtime(session, pod_info)
        payload = session.to_dict()
        payload["active"] = session.is_active
        return payload

    async def stop_session(self) -> Dict[str, Any]:
        async with self._lock:
            session = self._session
            if not session:
                return {"active": False, "status": PodStatus.OFFLINE.value}
        await asyncio.to_thread(self.provider.stop_pod, session.pod_id)
        session.status = PodStatus.TERMINATING
        async with self._lock:
            if self._session is session:
                self._session = None

        # Record GPU usage for billing if metering is enabled
        if self._metering_callback and session.companion_id and session.user_id:
            try:
                duration_seconds = (
                    datetime.now(timezone.utc) - session.started_at
                ).total_seconds()
                await self._metering_callback(
                    companion_id=session.companion_id,
                    user_id=session.user_id,
                    provider="runpod",
                    resource_type=session.gpu_type_id or "unknown",
                    duration_seconds=duration_seconds,
                    operation_id=session.pod_id,
                )
                logger.info(
                    f"Recorded GPU usage: {duration_seconds:.1f}s on "
                    f"{session.gpu_type_id or 'unknown'} "
                    f"for companion {session.companion_id}"
                )
            except Exception as e:
                logger.error(f"Failed to record GPU metering: {e}")

        payload = session.to_dict()
        payload["active"] = False
        return payload

    async def find_stopped_pod(
        self, purpose: str, profile_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find a stopped pod that can be resumed instead of creating a new one.

        Current availability and readiness are determined from v2 state.

        Args:
            purpose: e.g. "lora_training" or "lora_inference"
            profile_name: e.g. "training" or "image"

        Returns:
            Pod dict if found, None otherwise
        """
        if not isinstance(self.provider, DirectRunPodProvider):
            return None  # Only works with direct RunPod API

        all_pods = await asyncio.to_thread(self.provider.list_pods)
        names_by_purpose = {
            "lora_training": "kestrel-lora",
            "lora_inference": "kestrel-selfie",
            "ollama_server": "kestrel-ollama",
        }
        name_fragment = names_by_purpose.get(purpose)
        if name_fragment is None:
            raise RunPodManagerError(f"Unknown stopped-Pod purpose: {purpose}")
        for pod in all_pods:
            if (pod.get("status") or pod.get("desiredStatus")) != "EXITED":
                continue
            if name_fragment in pod.get("name", ""):
                logger.info("Found stopped %s Pod %s", purpose, pod["id"])
                return pod
        return None

    async def resume_stopped_pod(
        self, pod: Dict[str, Any], profile: GPUProfile, ttl_seconds: int
    ) -> RunPodSession:
        """
        Resume a stopped pod instead of creating new.

        The v2 start action may need to place compute again.
        """
        pod_id = pod["id"]
        gpu_count = (pod.get("gpu") or {}).get("count", pod.get("gpuCount", 1))
        if not profile.default_model:
            raise RunPodManagerError(
                f"Profile '{profile.id}' has no default_model configured; "
                "cannot resume Pod"
            )

        logger.info(f"Resuming stopped pod {pod_id} (faster than creating new)")
        resumed = await asyncio.to_thread(self.provider.resume_pod, pod_id, gpu_count)
        resumed_gpu = resumed.get("gpu") or pod.get("gpu") or {}

        started_at = datetime.now(timezone.utc)
        session = RunPodSession(
            pod_id=pod_id,
            profile=profile,
            task_profile=profile.task_type,
            model_name=profile.default_model,
            pod_type=profile.pod_type,
            status=PodStatus.PROVISIONING,
            ttl_seconds=ttl_seconds,
            started_at=started_at,
            expires_at=started_at + timedelta(seconds=ttl_seconds),
            gpu_type_id=resumed_gpu.get("id"),
            cost_per_hr=resumed.get("cost", pod.get("cost")),
        )

        async with self._lock:
            self._session = session

        await self._wait_until_ready()
        return session

    async def get_logs(self, tail: int = 100) -> str:
        """Retrieves the last N lines of logs from the active pod."""
        async with self._lock:
            session = self._session

        if not session or not session.is_active:
            raise RunPodManagerError("No active session to get logs from")

        if isinstance(self.provider, DirectRunPodProvider):
            return await asyncio.to_thread(self.provider.get_logs, session.pod_id, tail)

        elif isinstance(self.provider, ManagedRunPodProvider):
            raise NotImplementedError(
                "Log retrieval not yet implemented for managed provider"
            )

        return ""

    async def _wait_until_ready(self) -> None:
        async with self._lock:
            session = self._session
        if not session:
            return
        timeout = session.profile.readiness_timeout_seconds or self.readiness_timeout
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        # Phase 1: Wait for pod status to be READY
        while datetime.now(timezone.utc) < deadline:
            status = await self.get_status(refresh=True)
            if status.get("status") == PodStatus.READY.value:
                logger.info("RunPod session %s status is READY", session.pod_id)
                break
            await asyncio.sleep(self.poll_interval)
        else:
            raise RunPodManagerError("RunPod pod did not become ready before timeout")

        # Phase 2: Wait for backend URL to be populated (ports may lag behind status)
        # This is critical - RunPod sometimes reports RUNNING before ports are assigned
        # Increased from 60s to 120s as cold-start pods can take longer for ports
        url_deadline = datetime.now(timezone.utc) + timedelta(
            seconds=120
        )  # 120s extra for URL
        while datetime.now(timezone.utc) < url_deadline:
            async with self._lock:
                if session.backend_base_url:
                    logger.info(
                        "RunPod session %s backend URL ready: %s",
                        session.pod_id,
                        session.backend_base_url,
                    )
                    return
            # Refresh to get updated port info
            await self.get_status(refresh=True)
            await asyncio.sleep(
                RUNPOD_URL_POLL_INTERVAL
            )  # Shorter interval for URL polling

        logger.warning(
            "RunPod session %s ready but no backend URL after 120s", session.pod_id
        )
        # Don't raise error - let caller handle missing URL if needed

    async def wait_for_ready(
        self, session: Optional[RunPodSession] = None, timeout: Optional[int] = None
    ) -> bool:
        """
        Wait for a RunPod session to be ready.

        This is the public API for waiting after start_session() returns.
        Note: start_session() already calls _wait_until_ready() internally,
        so this is primarily useful when resuming a stopped pod or checking
        readiness after an external event.

        Args:
            session: Session to wait for. Defaults to current session.
            timeout: Timeout in seconds. Defaults to the profile readiness timeout.

        Returns:
            True if ready within timeout, False otherwise.
        """
        target_session = session
        if target_session is None:
            async with self._lock:
                target_session = self._session

        if not target_session:
            logger.warning("wait_for_ready called with no session")
            return False

        effective_timeout = (
            timeout
            or target_session.profile.readiness_timeout_seconds
            or self.readiness_timeout
        )
        deadline = datetime.now(timezone.utc) + timedelta(seconds=effective_timeout)

        while datetime.now(timezone.utc) < deadline:
            status = await self.get_status(refresh=True)
            if status.get("status") == PodStatus.READY.value:
                logger.info("RunPod session %s is ready", target_session.pod_id)
                return True
            if status.get("status") in {
                PodStatus.ERROR.value,
                PodStatus.TERMINATING.value,
            }:
                logger.error(
                    "RunPod session %s entered error/terminating state",
                    target_session.pod_id,
                )
                return False
            await asyncio.sleep(self.poll_interval)

        logger.warning(
            "RunPod session %s did not become ready before timeout (%ds)",
            target_session.pod_id,
            effective_timeout,
        )
        return False

    def _select_profile(self, task_profile: str) -> GPUProfile:
        profile = self.profiles.get(task_profile)
        if not profile:
            raise RunPodManagerError(
                f"Unknown task_profile '{task_profile}'. "
                f"Available: {list(self.profiles.keys())}"
            )
        return profile

    def _validate_ttl(self, ttl_seconds: Optional[int]) -> int:
        ttl = ttl_seconds or self.default_ttl_seconds
        if ttl > self.max_ttl_seconds:
            raise RunPodManagerError(
                f"TTL {ttl}s exceeds max allowed {self.max_ttl_seconds}s"
            )
        return ttl

    def _update_session_from_runtime(
        self, session: RunPodSession, pod_info: Dict[str, Any]
    ) -> None:
        if not pod_info:
            logger.warning("pod_info is None for session %s", session.pod_id)
            return
        safe_pod_info = sanitize_resource_payload(pod_info)
        raw_status = safe_pod_info.get("status") or safe_pod_info.get("desiredStatus")
        session.status = self._map_status(raw_status)
        session.runtime = safe_pod_info

        # Extract port information from runtime
        runtime = safe_pod_info.get("runtime") or {}
        ports = runtime.get("ports", [])

        # Log port info for debugging
        if not ports and raw_status in ("RUNNING", "running"):
            logger.debug(
                "Pod %s is RUNNING but runtime.ports is empty. Full runtime: %s",
                session.pod_id,
                runtime,
            )

        for port in ports:
            private_port = port.get("private") or port.get("privatePort")
            if private_port and int(private_port) == session.profile.inference_port:
                ip = port.get("ip")
                public_port = port.get("public") or port.get("publicPort")
                is_public = port.get("isIpPublic", bool(ip and public_port))
                port_type = str(port.get("type", "")).lower()

                # V2 HTTP ports are exposed by the stable Pod proxy even when
                # the runtime correctly reports no direct public IP/port.
                if port_type == "http" or (ip and public_port and not is_public):
                    base_url = (
                        f"https://{session.pod_id}-{private_port}.proxy.runpod.net"
                    )
                    logger.debug("Using Runpod HTTP proxy URL: %s", base_url)
                elif ip and public_port:
                    base_url = (
                        f"{session.profile.inference_protocol}://{ip}:{public_port}"
                    )
                    logger.debug("Using direct URL: %s (public IP)", base_url)
                else:
                    continue

                session.backend_base_url = base_url
                session.inference_url = (
                    f"{base_url}{session.profile.inference_base_path}"
                ).rstrip("/")
                if session.profile.image_invoke_path:
                    session.image_endpoint = (
                        f"{base_url}{session.profile.image_invoke_path}"
                    )
                logger.debug("Backend URL set: %s", base_url)

        if session.remaining_ttl_seconds == 0:
            session.status = PodStatus.TERMINATING

    @staticmethod
    def _map_status(raw_status: Optional[str]) -> PodStatus:
        normalized = (raw_status or "").lower()
        if normalized in {"running", "ready"}:
            return PodStatus.READY
        if normalized in {"starting", "provisioning"}:
            return PodStatus.PROVISIONING
        if normalized == "loading":
            return PodStatus.LOADING
        if normalized in {"stopping", "terminating"}:
            return PodStatus.TERMINATING
        if normalized in {"exited", "terminated", "stopped"}:
            return PodStatus.OFFLINE
        if normalized in {"failed", "error"}:
            return PodStatus.ERROR
        return PodStatus.OFFLINE

    async def terminate_session(self, session: RunPodSession) -> None:
        """Terminate a specific session's pod."""
        if session and session.pod_id:
            terminate = getattr(self.provider, "terminate_pod", None)
            if terminate is None:
                raise RunPodManagerError(
                    "Configured provider does not support Pod termination"
                )
            await asyncio.to_thread(terminate, session.pod_id)
            session.status = PodStatus.OFFLINE
            async with self._lock:
                if self._session is session:
                    self._session = None
            logger.info("Terminated Pod %s", session.pod_id)

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod by ID."""
        terminate = getattr(self.provider, "terminate_pod", None)
        if terminate is None:
            raise RunPodManagerError(
                "Configured provider does not support Pod termination"
            )
        await asyncio.to_thread(terminate, pod_id)
        async with self._lock:
            if self._session and self._session.pod_id == pod_id:
                self._session.status = PodStatus.OFFLINE
                self._session = None
        logger.info("Terminated Pod %s", pod_id)
