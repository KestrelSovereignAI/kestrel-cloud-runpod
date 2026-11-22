"""Content-free readiness evidence for one registered Runpod worker."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .clients import endpoint_worker_population_is_empty
from .ollama_contracts import OllamaProvisioningEvidenceError, OllamaRuntimeTelemetrySnapshot
from .ollama_runtime import OLLAMA_RUNTIME_VERSION

_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,254}")


@dataclass(frozen=True, slots=True)
class RegisteredWorkerReadback:
    worker_id: str
    endpoint_version: int
    started_at: datetime

    @classmethod
    def verified(
        cls, payload: Mapping[str, Any], *, data_center: str,
        gpu_id: str, image: str, activated_not_before: datetime,
    ) -> RegisteredWorkerReadback:
        """Do not infer worker location/GPU/preload from the endpoint defaults."""
        if endpoint_worker_population_is_empty(payload) or len(payload["workers"]) != 1:
            raise OllamaProvisioningEvidenceError("Registered endpoint does not have one worker")
        worker = payload["workers"][0]
        version = payload.get("endpointVersion")
        if (
            not isinstance(worker, Mapping) or type(version) is not int or version < 1
            or worker.get("isStale") is not False
            or worker.get("status") not in ("IDLE", "RUNNING")
            or payload["summary"][worker["status"].lower()] != 1
        ):
            raise OllamaProvisioningEvidenceError("Registered worker readiness is not proved")
        realized_version = worker.get("version")
        if not (
            (type(realized_version) is int and realized_version == version)
            or (version == 1 and realized_version is None)
        ):
            raise OllamaProvisioningEvidenceError("Registered worker is not on the current deployment")
        identifier = worker.get("id")
        if (
            not isinstance(identifier, str) or _WORKER_ID.fullmatch(identifier) is None
            or worker.get("dataCenterId") != data_center or worker.get("gpuTypeId") != gpu_id
            or type(worker.get("gpuCount")) is not int or worker["gpuCount"] != 1
            or worker.get("image") != image
        ):
            raise OllamaProvisioningEvidenceError("Registered worker placement or image changed")
        try:
            started = datetime.fromisoformat(worker.get("startedAt"))
        except (ValueError, TypeError):
            raise OllamaProvisioningEvidenceError("Registered worker start time is unknown") from None
        if started.utcoffset() is None or started < activated_not_before:
            raise OllamaProvisioningEvidenceError("Registered worker predates its activation")
        return cls(identifier, version, started.astimezone(UTC))


def verify_registered_runtime(
    telemetry: Mapping[str, Any], tags: Mapping[str, Any], *,
    telemetry_binding_sha256: str, expected_model: str, model_digest: str,
    first_worker: RegisteredWorkerReadback, last_worker: RegisteredWorkerReadback,
) -> OllamaRuntimeTelemetrySnapshot:
    """Match exact worker lineage before/after reading cache-only telemetry."""
    if first_worker != last_worker:
        raise OllamaProvisioningEvidenceError("Registered worker changed during evidence collection")
    try:
        snapshot = OllamaRuntimeTelemetrySnapshot.from_dict(telemetry)
    except (TypeError, ValueError):
        raise OllamaProvisioningEvidenceError("Registered workload telemetry is invalid") from None
    if (
        snapshot.runtime_version != OLLAMA_RUNTIME_VERSION
        or snapshot.telemetry_binding_sha256 != telemetry_binding_sha256
        or snapshot.ready is not True or snapshot.ollama_process_running is not True
        or snapshot.ready_at is None or snapshot.terminal_error is not None
        or not snapshot.proves_model_cache_hit
    ):
        raise OllamaProvisioningEvidenceError("Registered runtime did not prove a pinned cache hit")
    models = tags.get("models")
    if not isinstance(models, list):
        raise OllamaProvisioningEvidenceError("Registered runtime model inventory is unavailable")
    matches = [
        model for model in models if isinstance(model, Mapping) and model.get("name") == expected_model
    ]
    if (
        len(matches) != 1 or not isinstance(matches[0].get("digest"), str)
        or matches[0]["digest"].removeprefix("sha256:") != model_digest.removeprefix("sha256:")
    ):
        raise OllamaProvisioningEvidenceError("Registered runtime model digest does not match the cache")
    return snapshot
