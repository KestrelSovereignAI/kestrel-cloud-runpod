"""Runpod private-Ollama lease integration for :class:`RunPodManager`."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any, Protocol, cast

from .models import FlashBoot, GPUProfile, RunPodManagerError
from .ollama_contracts import (
    OllamaLease,
    OllamaLeaseRequest,
    OllamaNonComputeCostComponent,
    OllamaNonComputeCostPolicy,
)
from .ollama_provider import (
    RunpodOllamaCapacityProvider,
    RunpodOllamaDeployment,
)
from .ollama_repository import (
    SQLiteOllamaLeaseRepository,
    lease_database_path,
)
from .ollama_service import OllamaLeaseService
from .providers import DirectRunPodProvider, GPUProvider, _resolve_env_vars


class _OllamaManagerHost(Protocol):
    provider: GPUProvider
    config: Mapping[str, Any]

    def _select_profile(self, task_profile: str) -> GPUProfile: ...


class RunPodOllamaMixin:
    """Expose durable, ownership-scoped Ollama leases through the manager.

    The previous singleton session helpers were intentionally removed: they lost
    billing ownership and TTL state on process exit.  Callers now supply a stable
    lease/owner/workload identity and an explicit deadline/cost policy.
    """

    _ollama_lease_service: OllamaLeaseService | None = None

    def set_ollama_lease_service(self, service: OllamaLeaseService) -> None:
        """Inject the durable service (primarily for managed hosting and tests)."""

        self._ollama_lease_service = service

    async def acquire_ollama_lease(self, request: OllamaLeaseRequest) -> OllamaLease:
        return await self._get_ollama_lease_service().acquire(request)

    async def get_ollama_lease(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        return await self._get_ollama_lease_service().get(
            lease_id, owner_id=owner_id, workload_id=workload_id
        )

    async def touch_ollama_lease(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        return await self._get_ollama_lease_service().touch(
            lease_id, owner_id=owner_id, workload_id=workload_id
        )

    async def release_ollama_lease(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        return await self._get_ollama_lease_service().release(
            lease_id, owner_id=owner_id, workload_id=workload_id
        )

    async def reconcile_ollama_leases(self) -> tuple[OllamaLease, ...]:
        """Run one restart-safe reconciliation pass from an external scheduler."""

        return await self._get_ollama_lease_service().reconcile()

    def _get_ollama_lease_service(self) -> OllamaLeaseService:
        existing = self._ollama_lease_service
        if existing is not None:
            return existing
        host = cast(_OllamaManagerHost, self)
        if not isinstance(host.provider, DirectRunPodProvider):
            raise RunPodManagerError(
                "Durable Ollama leases require the direct Runpod v2 provider or an "
                "explicitly injected OllamaLeaseService"
            )
        raw_settings = host.config.get("ollama_leases")
        if not isinstance(raw_settings, Mapping):
            raise RunPodManagerError(
                "Configure the ollama_leases section before acquiring capacity"
            )
        serverless_api_key = os.getenv("RUNPOD_SERVERLESS_API_KEY")
        pod_bearer_token = os.getenv("RUNPOD_OLLAMA_BEARER_TOKEN")
        if not serverless_api_key and not pod_bearer_token:
            raise RunPodManagerError(
                "Configure RUNPOD_SERVERLESS_API_KEY, "
                "RUNPOD_OLLAMA_BEARER_TOKEN, or both for Runpod Ollama"
            )
        try:
            flashboot = FlashBoot(
                _required_string(raw_settings, "serverless_flashboot")
            )
        except ValueError as exc:
            raise RunPodManagerError(
                "ollama_leases.serverless_flashboot is invalid"
            ) from exc
        deployment = RunpodOllamaDeployment(
            profile=host._select_profile("ollama"),
            serverless_workers_max=_required_int(
                raw_settings, "serverless_workers_max"
            ),
            serverless_request_count=_required_int(
                raw_settings, "serverless_request_count"
            ),
            serverless_execution_timeout_ms=_required_int(
                raw_settings, "serverless_execution_timeout_ms"
            ),
            serverless_flashboot=flashboot,
            http_timeout_seconds=_required_float(raw_settings, "http_timeout_seconds"),
            serverless_non_compute_cost=_required_non_compute_cost_policy(
                raw_settings, "serverless_non_compute_cost"
            ),
            pod_non_compute_cost=_required_non_compute_cost_policy(
                raw_settings, "pod_non_compute_cost"
            ),
        )
        service = OllamaLeaseService(
            repository=SQLiteOllamaLeaseRepository(lease_database_path(raw_settings)),
            provider=RunpodOllamaCapacityProvider(
                client=host.provider.client,
                deployment=deployment,
                serverless_api_key=serverless_api_key,
                pod_bearer_token=pod_bearer_token,
                control_plane_api_key=os.getenv("RUNPOD_API_KEY"),
            ),
            poll_interval_seconds=_required_float(
                raw_settings, "poll_interval_seconds"
            ),
        )
        self._ollama_lease_service = service
        return service


def _required_string(settings: Mapping[str, Any], name: str) -> str:
    value = settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RunPodManagerError(f"ollama_leases.{name} must be configured")
    return value


def _required_int(settings: Mapping[str, Any], name: str) -> int:
    value = settings.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RunPodManagerError(f"ollama_leases.{name} must be a positive integer")
    return value


def _required_float(settings: Mapping[str, Any], name: str) -> float:
    value = settings.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RunPodManagerError(f"ollama_leases.{name} must be a positive number")
    return float(value)


def _required_non_compute_cost_policy(
    settings: Mapping[str, Any], name: str
) -> OllamaNonComputeCostPolicy:
    raw = settings.get(name)
    if not isinstance(raw, Mapping):
        raise RunPodManagerError(
            f"ollama_leases.{name} must explicitly configure non-compute cost"
        )
    raw_components = raw.get("covered_components")
    if (
        not isinstance(raw_components, list)
        or not raw_components
        or any(not isinstance(item, str) for item in raw_components)
    ):
        raise RunPodManagerError(
            f"ollama_leases.{name}.covered_components must be a nonempty list"
        )
    try:
        components = tuple(
            sorted(
                (OllamaNonComputeCostComponent(item) for item in raw_components),
                key=lambda item: item.value,
            )
        )
        return OllamaNonComputeCostPolicy(
            estimated_cost_usd=_required_nonnegative_cost(
                raw, "estimated_cost_usd", section=name
            ),
            maximum_cost_usd=_required_nonnegative_cost(
                raw, "maximum_cost_usd", section=name
            ),
            covered_components=components,
        )
    except ValueError as exc:
        raise RunPodManagerError(
            f"ollama_leases.{name} contains an invalid cost policy"
        ) from exc


def _required_nonnegative_cost(
    settings: Mapping[str, Any], name: str, *, section: str
) -> float:
    raw = settings.get(name)
    if isinstance(raw, str):
        raw = _resolve_env_vars({name: raw})[name]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError(
            f"ollama_leases.{section}.{name} must be configured"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise RunPodManagerError(
            f"ollama_leases.{section}.{name} must be finite and nonnegative"
        )
    return value
