"""Restart-safe one-shot reaper for explicitly registered Ollama capacity."""

from __future__ import annotations

import asyncio
import os
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, TextIO

from ._one_shot_reconciler import run_one_shot_reconciler
from .clients import RunpodControlPlaneClient
from .models import EndpointCreateRequest, FlashBoot, RunPodManagerError
from .ollama_contracts import iso_datetime
from .ollama_repository import SQLiteOllamaLeaseRepository, lease_database_path
from .ollama_runtime import OLLAMA_RUNTIME_PORT, require_cache_only_ollama_image
from .registered_ollama import RegisteredOllamaEndpoint, RegisteredOllamaLifecycle
from .registered_ollama_journal import RegisteredActivation, RegisteredOllamaJournal


def load_registered_endpoint(data: Mapping[str, object]) -> RegisteredOllamaEndpoint:
    """Read explicit operator-owned registry settings; never accept a route key."""
    string_fields = {
        "endpoint_id", "endpoint_name", "created_at", "image", "gpu_pool", "region",
        "cache_id", "registry_id", "min_cuda",
    }
    integer_fields = {"disk_gb", "worker_idle_seconds", "execution_timeout_ms"}
    if set(data) != string_fields | integer_fields:
        raise RunPodManagerError("Registered Ollama config has missing or unsupported fields")
    if any(not isinstance(data[field], str) or not data[field] for field in string_fields):
        raise RunPodManagerError("Invalid registered Ollama identity")
    if any(type(data[field]) is not int or data[field] < 1 for field in integer_fields):
        raise RunPodManagerError("Invalid registered Ollama worker limits")
    if not data["region"].startswith("US-"):
        raise RunPodManagerError("Registered Ollama requires an explicitly configured US data center")
    try:
        created = datetime.fromisoformat(data["created_at"])
        iso_datetime(created)
        runtime = str(OLLAMA_RUNTIME_PORT)
        return RegisteredOllamaEndpoint(
            endpoint_id=data["endpoint_id"],
            original_created_at=created,
            baseline=EndpointCreateRequest(
                name=data["endpoint_name"], image=require_cache_only_ollama_image(data["image"]),
                endpoint_type="LOAD_BALANCER", gpu_pools=(data["gpu_pool"],),
                scaling={"type": "REQUEST_COUNT", "requestCount": 1},
                workers_min=0, workers_max=0, idle_timeout_seconds=data["worker_idle_seconds"],
                disk_gb=data["disk_gb"], ports=(runtime + "/http",),
                env={"PORT": runtime, "PORT_HEALTH": runtime},
                registry_id=data["registry_id"], data_center_ids=(data["region"],),
                network_volume_ids=(data["cache_id"],), execution_timeout_ms=data["execution_timeout_ms"],
                flashboot=FlashBoot.OFF, min_cuda_version=data["min_cuda"],
            ),
        )
    except (TypeError, ValueError):
        raise RunPodManagerError("Invalid registered endpoint definition") from None


async def reap_once() -> tuple[RegisteredActivation, ...]:
    registration_path = os.getenv("RUNPOD_REGISTERED_OLLAMA_CONFIG")
    if not registration_path or not Path(registration_path).is_absolute():
        raise RunPodManagerError("Configure an absolute RUNPOD_REGISTERED_OLLAMA_CONFIG")
    control_key = os.getenv("RUNPOD_API_KEY")
    if not control_key:
        raise RunPodManagerError("Registered cleanup requires the Runpod management credential")
    config = tomllib.loads(Path(registration_path).read_text(encoding="utf-8"))
    if set(config) != {"endpoint", "inference"}:
        raise RunPodManagerError("Invalid registered inference config shape")
    registration = load_registered_endpoint(config["endpoint"])
    database = lease_database_path({})

    def worker() -> tuple[RegisteredActivation, ...]:
        clock = lambda: datetime.now(UTC)
        repository = SQLiteOllamaLeaseRepository(database)
        journal = RegisteredOllamaJournal(repository.path, clock=clock)
        client = RunpodControlPlaneClient(api_key=control_key)
        try:
            return RegisteredOllamaLifecycle(
                client=client, endpoint=registration, journal=journal, clock=clock,
            ).reap_due()
        finally:
            client.close()

    return await asyncio.to_thread(worker)


def run_cli(argv: Sequence[str] | None = None, *, stderr: TextIO | None = None) -> int:
    return run_one_shot_reconciler(
        argv, program="kestrel-runpod-reconcile-registered-ollama",
        description="Park expired registered Ollama leases without inference calls.",
        operation_factory=reap_once, stderr=stderr,
    )


def main() -> NoReturn:
    raise SystemExit(run_cli())
