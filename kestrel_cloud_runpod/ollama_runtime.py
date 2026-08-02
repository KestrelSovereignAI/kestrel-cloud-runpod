"""Canonical configuration contract for the authenticated Ollama workload."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath

from .models import RunPodManagerError
from .ollama_contracts import OllamaLeaseMode

OLLAMA_RUNTIME_CONTRACT = "ollama-runtime/v1"
OLLAMA_RUNTIME_VERSION = "1.0.0"
OLLAMA_RUNTIME_IMAGE_REPOSITORY = (
    "ghcr.io/kestrelsovereignai/kestrel-cloud-runpod-ollama-runtime"
)
OLLAMA_RUNTIME_PORT = 11434

_IMAGE_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[a-f0-9]{64}$")
_MODEL_PIN_RE = re.compile(
    r"^([a-z0-9][a-z0-9._/-]*:[a-z0-9][a-z0-9._-]*)@sha256:([a-f0-9]{64})$"
)
_RESERVED_ENV = frozenset(
    {
        "HEALTH_CHECK_PATH",
        "KESTREL_OLLAMA_BEARER_TOKEN",
        "KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT",
        "KESTREL_OLLAMA_MODE",
        "KESTREL_OLLAMA_MODEL_STORAGE_PATH",
        "KESTREL_OLLAMA_PROVISION_REQUESTED_AT",
        "KESTREL_OLLAMA_REQUIRED_MODEL",
        "OLLAMA_HOST",
        "OLLAMA_MODELS",
        "OLLAMA_MODELS_PULL",
        "PORT",
        "PORT_HEALTH",
    }
)


def require_immutable_ollama_image(image: str) -> str:
    """Reject mutable tags before a billable Runpod create operation."""

    if not _IMAGE_DIGEST_RE.fullmatch(image):
        raise RunPodManagerError(
            "RUNPOD_OLLAMA_IMAGE must be an immutable "
            + f"{OLLAMA_RUNTIME_IMAGE_REPOSITORY}@sha256:<digest> reference"
        )
    repository = image.split("@", 1)[0]
    if repository != OLLAMA_RUNTIME_IMAGE_REPOSITORY:
        raise RunPodManagerError(
            "RUNPOD_OLLAMA_IMAGE must use the reviewed Kestrel Ollama runtime "
            + f"repository {OLLAMA_RUNTIME_IMAGE_REPOSITORY}"
        )
    return image


def build_ollama_runtime_environment(
    profile_environment: Mapping[str, str],
    *,
    requested_model: str,
    bearer_token: str,
    bearer_token_expires_at: datetime,
    mode: OllamaLeaseMode,
    model_storage_path: str,
    provision_requested_at: datetime,
) -> dict[str, str]:
    """Create the sole Pod/LB runtime environment from trusted policy inputs."""

    conflicts = sorted(_RESERVED_ENV.intersection(profile_environment))
    if conflicts:
        raise RunPodManagerError(
            "Ollama profile env cannot override runtime-owned settings: "
            + ", ".join(conflicts)
        )
    if mode is OllamaLeaseMode.AUTO:
        raise RunPodManagerError("Ollama runtime requires a resolved Pod or LB mode")
    storage_path = _model_storage_path(model_storage_path, mode)
    if (
        len(bearer_token) < 32
        or bearer_token.strip() != bearer_token
        or any(character in bearer_token for character in "\r\n")
    ):
        raise RunPodManagerError(
            "The scoped Ollama inference bearer must contain at least 32 "
            + "non-whitespace characters"
        )
    expires_at = _aware_utc(bearer_token_expires_at, "bearer expiry")
    requested_at = _aware_utc(provision_requested_at, "provision timestamp")
    if expires_at <= requested_at:
        raise RunPodManagerError("Ollama workload bearer expires before provisioning")
    allowed_raw = profile_environment.get("KESTREL_OLLAMA_ALLOWED_MODELS")
    if not isinstance(allowed_raw, str) or not allowed_raw.strip():
        raise RunPodManagerError(
            "profiles.ollama.env.KESTREL_OLLAMA_ALLOWED_MODELS must configure "
            + "digest-pinned operator policy"
        )
    allowed = parse_ollama_model_allowlist(allowed_raw)
    if requested_model not in allowed:
        raise RunPodManagerError(
            f"Ollama model '{requested_model}' is not in the operator allowlist"
        )
    environment = dict(profile_environment)
    environment.update(
        {
            "HEALTH_CHECK_PATH": "/ping",
            "KESTREL_OLLAMA_BEARER_TOKEN": bearer_token,
            "KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT": _rfc3339(expires_at),
            "KESTREL_OLLAMA_MODE": mode.value,
            "KESTREL_OLLAMA_MODEL_STORAGE_PATH": storage_path,
            "KESTREL_OLLAMA_PROVISION_REQUESTED_AT": _rfc3339(requested_at),
            "KESTREL_OLLAMA_REQUIRED_MODEL": requested_model,
            "PORT": str(OLLAMA_RUNTIME_PORT),
            "PORT_HEALTH": str(OLLAMA_RUNTIME_PORT),
        }
    )
    return environment


def parse_ollama_model_allowlist(raw: str) -> dict[str, str]:
    """Parse the operator's digest-pinned model policy without provisioning."""

    allowed: dict[str, str] = {}
    for item in raw.split(","):
        spec = item.strip()
        if not spec:
            continue
        match = _MODEL_PIN_RE.fullmatch(spec)
        if match is None:
            raise RunPodManagerError(
                "Ollama model allowlist entries must use "
                + "name:tag@sha256:<64 lowercase hex>"
            )
        name = match.group(1)
        if name in allowed:
            raise RunPodManagerError(
                f"Ollama model allowlist contains duplicate '{name}'"
            )
        allowed[name] = f"sha256:{match.group(2)}"
    if not allowed:
        raise RunPodManagerError("Ollama model allowlist cannot be empty")
    return allowed


def _model_storage_path(path: str, mode: OllamaLeaseMode) -> str:
    parsed = PurePosixPath(path)
    normalized = str(parsed)
    if not parsed.is_absolute() or normalized != path or ".." in parsed.parts:
        raise RunPodManagerError(
            "Ollama model storage path must be normalized and absolute"
        )
    allowed_roots = (
        ("/models", "/runpod-volume")
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
        else ("/models", "/workspace")
    )
    if not any(path == root or path.startswith(f"{root}/") for root in allowed_roots):
        raise RunPodManagerError(
            f"Ollama {mode.value} storage must live under " + " or ".join(allowed_roots)
        )
    return normalized


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise RunPodManagerError(f"Ollama runtime {name} must be timezone-aware")
    return value.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
