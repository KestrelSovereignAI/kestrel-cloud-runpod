"""Runpod REST v2 capacity adapter for durable training Pod leases."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .models import (
    GPUProfile,
    PlacementDecision,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)
from .providers import DirectRunPodProvider

_T = TypeVar("_T")


@dataclass(frozen=True)
class TrainingPodObservation:
    """Content-free capacity state needed by the lifecycle service."""

    provider_pod_id: str
    status: str
    backend_base_url: str | None
    raw: Mapping[str, Any]

    @property
    def is_running(self) -> bool:
        return self.status.upper() == "RUNNING"

    @property
    def is_stopped(self) -> bool:
        return self.status.upper() in {"EXITED", "STOPPED", "TERMINATED"}

    @property
    def is_failed(self) -> bool:
        return self.status.upper() in {"FAILED", "ERROR"}


@dataclass(frozen=True)
class CreatedTrainingPod:
    """Provider identity and live placement returned by a v2 Pod create."""

    provider_pod_id: str
    placement: PlacementDecision | None


class TrainingPodCapacityProvider(Protocol):
    """Capacity operations required by the durable training lifecycle."""

    async def observe(
        self, pod_id: str, *, profile: GPUProfile
    ) -> TrainingPodObservation: ...

    async def start(self, pod_id: str, *, gpu_count: int) -> None: ...

    async def create(
        self,
        *,
        profile: GPUProfile,
        resource_name: str,
        companion_id: str,
    ) -> CreatedTrainingPod: ...

    async def find_by_name(self, resource_name: str) -> str | None: ...

    async def stop(self, pod_id: str) -> bool: ...


class RunpodTrainingPodProvider:
    """Expose only the v2 Pod operations the durable lifecycle requires."""

    def __init__(self, provider: DirectRunPodProvider) -> None:
        self.provider = provider

    async def observe(
        self, pod_id: str, *, profile: GPUProfile
    ) -> TrainingPodObservation:
        raw = await asyncio.to_thread(self.provider.get_status, pod_id)
        status = raw.get("status") or raw.get("desiredStatus")
        if not isinstance(status, str) or not status:
            raise RunPodManagerError("Runpod v2 Pod status response omitted status")
        return TrainingPodObservation(
            provider_pod_id=pod_id,
            status=status,
            backend_base_url=_pod_base_url(pod_id, raw, profile),
            raw=raw,
        )

    async def start(self, pod_id: str, *, gpu_count: int) -> None:
        await _mutation_to_thread(self.provider.resume_pod, pod_id, gpu_count)

    async def create(
        self,
        *,
        profile: GPUProfile,
        resource_name: str,
        companion_id: str,
    ) -> CreatedTrainingPod:
        result = await _mutation_to_thread(
            self.provider.start_pod,
            profile,
            {
                "name": resource_name,
                "companion_id": companion_id,
                "purpose": "lora_training",
            },
        )
        pod_id = result.get("id") or result.get("podId")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodAmbiguousResultError(
                title="Runpod v2 create response was incomplete",
                detail="A successful Pod create response omitted the Pod ID",
                method="POST",
                resource="/pods",
            )
        placement = result.get("_kestrel_placement")
        if placement is not None and not isinstance(placement, PlacementDecision):
            raise RunPodAmbiguousResultError(
                title="Runpod v2 create response was incomplete",
                detail="A successful Pod create response had invalid placement metadata",
                method="POST",
                resource="/pods",
            )
        return CreatedTrainingPod(provider_pod_id=pod_id, placement=placement)

    async def find_by_name(self, resource_name: str) -> str | None:
        pods = await asyncio.to_thread(self.provider.list_pods)
        matches = [pod for pod in pods if pod.get("name") == resource_name]
        if len(matches) > 1:
            raise RunPodManagerError(
                f"Multiple Runpod Pods match durable name '{resource_name}'"
            )
        if not matches:
            return None
        pod_id = matches[0].get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodManagerError("Runpod v2 Pod list item omitted its ID")
        return pod_id

    async def stop(self, pod_id: str) -> bool:
        """Request idempotent stop and report whether v2 confirms non-billing state."""

        try:
            result = await _mutation_to_thread(self.provider.stop_pod, pod_id)
        except RunPodAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        status = result.get("status") or result.get("desiredStatus")
        if isinstance(status, str) and status.upper() in {
            "EXITED",
            "STOPPED",
            "TERMINATED",
        }:
            return True
        try:
            raw = await asyncio.to_thread(self.provider.get_status, pod_id)
        except RunPodAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        observed_status = raw.get("status") or raw.get("desiredStatus")
        if not isinstance(observed_status, str) or not observed_status:
            raise RunPodManagerError("Runpod v2 Pod status response omitted status")
        return observed_status.upper() in {"EXITED", "STOPPED", "TERMINATED"}


def _pod_base_url(
    pod_id: str, payload: Mapping[str, Any], profile: GPUProfile
) -> str | None:
    """Resolve the workload route from the exact v2 runtime port response."""

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    ports = runtime.get("ports")
    if not isinstance(ports, Sequence) or isinstance(ports, (str, bytes)):
        return None
    for item in ports:
        if not isinstance(item, Mapping):
            continue
        private = item.get("private") or item.get("privatePort")
        try:
            matches = (
                isinstance(private, (str, int))
                and not isinstance(private, bool)
                and int(private) == profile.inference_port
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            continue
        port_type = str(item.get("type", "")).lower()
        if port_type == "http":
            return f"https://{pod_id}-{profile.inference_port}.proxy.runpod.net"
        ip = item.get("ip")
        public = item.get("public") or item.get("publicPort")
        if (
            isinstance(ip, str)
            and isinstance(public, (str, int))
            and not isinstance(public, bool)
        ):
            protocol = profile.inference_protocol.strip().lower()
            if protocol not in {"http", "https"}:
                raise RunPodManagerError(
                    "Training profile inference_protocol must be http or https"
                )
            return f"{protocol}://{ip}:{int(public)}"
    return None


async def _mutation_to_thread(operation, /, *args: Any) -> _T:
    """Do not propagate cancellation until an in-flight v2 mutation resolves."""

    task: asyncio.Task[_T] = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Threads cannot be cancelled. Waiting here closes the race where a
        # cleanup stop could run before a delayed create/start completed.
        await task
        raise
