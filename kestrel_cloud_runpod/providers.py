"""Provider adapters backed by the Runpod v2 REST control plane."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from kestrel_sdk.config.constants import HTTP_TIMEOUT_DEFAULT, HTTP_TIMEOUT_QUICK

from .clients import RunpodControlPlaneClient
from .models import (
    CloudType,
    ComputeProduct,
    GPUProfile,
    PlacementRequirements,
    PodCreateRequest,
    RunPodManagerError,
)
from .placement import select_gpu

logger = logging.getLogger(__name__)
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(env_vars: Dict[str, Any]) -> Dict[str, str]:
    """Resolve required environment references immediately before provisioning."""

    resolved: Dict[str, str] = {}
    for key, raw_value in env_vars.items():
        if raw_value is None:
            continue
        value = str(raw_value)

        def replace_var(match: re.Match[str], env_key: str = key) -> str:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise RunPodManagerError(
                    f"Pod env '{env_key}' references unset environment variable "
                    f"'{var_name}'"
                )
            return os.environ[var_name]

        resolved[key] = _ENV_VAR_RE.sub(replace_var, value)
    return resolved


class GPUProvider(ABC):
    """Abstract provider that knows how to manage Pods."""

    @abstractmethod
    def start_pod(
        self, profile: GPUProfile, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start a new GPU Pod with the given workload profile."""

    @abstractmethod
    def get_status(self, pod_id: str) -> Dict[str, Any]:
        """Get the current status of a Pod."""

    @abstractmethod
    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        """Stop a running Pod without deleting it."""


class DirectRunPodProvider(GPUProvider):
    """Compatibility provider implemented exclusively on Runpod REST v2."""

    def __init__(
        self,
        api_key: str,
        cloud_type: str = "SECURE",
        *,
        client: Optional[RunpodControlPlaneClient] = None,
        control_plane_base_url: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        if not api_key and client is None:
            raise RunPodManagerError("RUNPOD_API_KEY is required for direct mode")
        try:
            self.cloud = CloudType(cloud_type.upper())
        except ValueError as exc:
            raise RunPodManagerError(
                "RUNPOD_CLOUD_TYPE must be SECURE or COMMUNITY"
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if control_plane_base_url:
            client_kwargs["base_url"] = control_plane_base_url
        if user_agent:
            client_kwargs["user_agent"] = user_agent
        self.client = client or RunpodControlPlaneClient(**client_kwargs)

    def start_pod(
        self, profile: GPUProfile, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        pod_env = _resolve_env_vars(
            {**profile.env, **metadata.get("env_overrides", {})}
        )
        cloud_value = metadata.get("cloud_type", profile.cloud.value)
        try:
            cloud = CloudType(str(cloud_value).upper())
        except ValueError as exc:
            raise RunPodManagerError(
                "Pod cloud_type must be SECURE or COMMUNITY"
            ) from exc
        requirements = PlacementRequirements(
            product=ComputeProduct.POD,
            min_vram_gb=profile.min_vram_gb,
            gpu_count=profile.gpu_count,
            cloud=cloud,
            min_cuda_version=profile.min_cuda_version,
            max_cost_per_hr=profile.max_cost_per_hr,
            allowed_gpu_ids=profile.allowed_gpu_ids,
            allowed_data_center_ids=profile.allowed_data_center_ids,
            benchmark_id=profile.task_type,
        )
        offers = self.client.list_gpus(
            products=(ComputeProduct.POD,),
            count=requirements.gpu_count,
            cloud=requirements.cloud,
            min_cuda_version=requirements.min_cuda_version,
        )
        placement = select_gpu(offers, requirements)

        mounts: Optional[Dict[str, Any]] = None
        if profile.network_volume_id:
            mounts = {
                "network": [
                    {
                        "volumeId": profile.network_volume_id,
                        "path": profile.volume_mount_path or "/runpod-volume",
                    }
                ]
            }
        elif profile.volume_gb:
            mounts = {
                "persistent": {
                    "size": profile.volume_gb,
                    "path": profile.volume_mount_path or "/workspace",
                }
            }

        pod = self.client.create_pod(
            PodCreateRequest(
                name=metadata.get("name", f"kestrel-{profile.id}"),
                image=profile.image_name,
                gpu_id=placement.gpu_id,
                gpu_count=placement.gpu_count,
                cloud=placement.cloud,
                disk_gb=profile.container_disk_gb,
                ports=tuple(profile.ports),
                env=pod_env,
                args=metadata.get("docker_args"),
                registry_id=profile.registry_id,
                data_center_ids=profile.allowed_data_center_ids,
                mounts=mounts,
            )
        )
        result = dict(pod.raw)
        result["_kestrel_placement"] = placement
        return result

    def get_status(self, pod_id: str) -> Dict[str, Any]:
        return dict(self.client.get_pod(pod_id).raw)

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        pod = self.client.pod_action(pod_id, "stop")
        return dict(pod.raw) if pod is not None else {"id": pod_id, "status": "EXITED"}

    def resume_pod(self, pod_id: str, gpu_count: int = 1) -> Dict[str, Any]:
        """Start a stopped Pod; v2 chooses currently available compute."""

        if gpu_count != 1:
            logger.debug("Runpod v2 start action uses the Pod's configured GPU count")
        pod = self.client.pod_action(pod_id, "start")
        return (
            dict(pod.raw) if pod is not None else {"id": pod_id, "status": "STARTING"}
        )

    def terminate_pod(self, pod_id: str) -> Dict[str, Any]:
        """Permanently terminate a Pod through the v2 action contract."""

        pod = self.client.pod_action(pod_id, "terminate")
        return (
            dict(pod.raw) if pod is not None else {"id": pod_id, "status": "TERMINATED"}
        )

    def list_pods(self) -> List[Dict[str, Any]]:
        return [dict(pod.raw) for pod in self.client.list_pods()]

    def get_logs(self, pod_id: str, tail: int = 100) -> str:
        """Collect the requested Pod log backfill from the v2 SSE endpoint."""

        if tail < 1 or tail > 5000:
            raise ValueError("tail must be between 1 and 5000")
        lines: list[str] = []
        for event in self.client.iter_pod_logs(
            pod_id, tail=tail, stream_window_seconds=2.0
        ):
            line = event.get("line")
            if isinstance(line, str):
                lines.append(line)
            if len(lines) >= tail:
                break
        return "\n".join(lines)

    def exec_command(self, pod_id: str, command: str) -> str:
        """Reject the removed private-CLI SSH path with migration guidance."""

        raise RunPodManagerError(
            "Arbitrary SSH execution is not available through Runpod REST v2. "
            "Expose a scoped workload HTTP operation or use the v2 Pod log stream."
        )


class ManagedRunPodProvider(GPUProvider):
    """Provider proxying through a managed Kestrel platform API."""

    def __init__(self, api_base: str, api_key: str):
        if not api_base or not api_key:
            raise RunPodManagerError(
                "Managed provider requires KESTREL_API_BASE and KESTREL_API_KEY"
            )
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def start_pod(
        self, profile: GPUProfile, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = {"profile": profile.id, "metadata": metadata}
        response = self.session.post(
            f"{self.api_base}/runpod/pods",
            json=payload,
            timeout=HTTP_TIMEOUT_DEFAULT,
        )
        response.raise_for_status()
        return response.json()

    def get_status(self, pod_id: str) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.api_base}/runpod/pods/{pod_id}", timeout=HTTP_TIMEOUT_QUICK
        )
        response.raise_for_status()
        return response.json()

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.api_base}/runpod/pods/{pod_id}/stop", timeout=HTTP_TIMEOUT_QUICK
        )
        response.raise_for_status()
        return response.json()
