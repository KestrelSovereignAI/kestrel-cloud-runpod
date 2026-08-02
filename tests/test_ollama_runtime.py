"""Authenticated Ollama runtime and Runpod template contract tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import OllamaLeaseMode
from kestrel_cloud_runpod.ollama_runtime import (
    OLLAMA_RUNTIME_CONTRACT,
    OLLAMA_RUNTIME_IMAGE_REPOSITORY,
    OLLAMA_RUNTIME_PORT,
    OLLAMA_RUNTIME_VERSION,
    build_ollama_runtime_environment,
    require_immutable_ollama_image,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime" / "ollama-runtime"
MODEL_DIGEST = "sha256:" + "a" * 64
IMAGE = f"{OLLAMA_RUNTIME_IMAGE_REPOSITORY}@{MODEL_DIGEST}"
TOKEN = "workload-" + "s" * 32


def _environment(**changes: object) -> dict[str, str]:
    requested_at = datetime(2026, 8, 1, 12, tzinfo=UTC)
    values: dict[str, object] = {
        "profile_environment": {
            "KESTREL_OLLAMA_ALLOWED_MODELS": f"qwen3:8b@{MODEL_DIGEST}"
        },
        "requested_model": "qwen3:8b",
        "bearer_token": TOKEN,
        "bearer_token_expires_at": requested_at + timedelta(hours=2),
        "mode": OllamaLeaseMode.DEDICATED_POD,
        "model_storage_path": "/models",
        "provision_requested_at": requested_at,
    }
    values.update(changes)
    return build_ollama_runtime_environment(**values)  # type: ignore[arg-type]


def test_runtime_image_must_use_reviewed_repository_and_digest() -> None:
    assert require_immutable_ollama_image(IMAGE) == IMAGE
    for invalid in (
        f"{OLLAMA_RUNTIME_IMAGE_REPOSITORY}:latest",
        f"registry.example/other@{MODEL_DIGEST}",
        f"{OLLAMA_RUNTIME_IMAGE_REPOSITORY}@sha256:not-a-digest",
    ):
        with pytest.raises(RunPodManagerError, match="RUNPOD_OLLAMA_IMAGE"):
            require_immutable_ollama_image(invalid)


def test_runtime_environment_is_owned_and_lease_bounded() -> None:
    environment = _environment()

    assert environment["KESTREL_OLLAMA_REQUIRED_MODEL"] == "qwen3:8b"
    assert environment["KESTREL_OLLAMA_BEARER_TOKEN"] == TOKEN
    assert environment["KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT"].endswith("Z")
    assert environment["KESTREL_OLLAMA_MODE"] == "dedicated_pod"
    assert environment["KESTREL_OLLAMA_MODEL_STORAGE_PATH"] == "/models"
    assert environment["HEALTH_CHECK_PATH"] == "/ping"
    assert environment["PORT"] == str(OLLAMA_RUNTIME_PORT)
    assert environment["PORT_HEALTH"] == str(OLLAMA_RUNTIME_PORT)
    assert "RUNPOD_API_KEY" not in environment


def test_runtime_environment_rejects_unapproved_model_and_overrides() -> None:
    with pytest.raises(RunPodManagerError, match="not in the operator allowlist"):
        _environment(requested_model="arbitrary:latest")
    with pytest.raises(RunPodManagerError, match="cannot override"):
        _environment(
            profile_environment={
                "KESTREL_OLLAMA_ALLOWED_MODELS": f"qwen3:8b@{MODEL_DIGEST}",
                "PORT": "9999",
            }
        )
    with pytest.raises(RunPodManagerError, match="name:tag@sha256"):
        _environment(
            profile_environment={"KESTREL_OLLAMA_ALLOWED_MODELS": "qwen3:latest"}
        )


def test_runtime_environment_rejects_weak_or_expired_capability() -> None:
    requested_at = datetime(2026, 8, 1, 12, tzinfo=UTC)
    with pytest.raises(RunPodManagerError, match="at least 32"):
        _environment(bearer_token="short")
    with pytest.raises(RunPodManagerError, match="expires before"):
        _environment(
            provision_requested_at=requested_at,
            bearer_token_expires_at=requested_at,
        )


def test_runtime_environment_restricts_mode_specific_storage_roots() -> None:
    assert (
        _environment(model_storage_path="/workspace/ollama")[
            "KESTREL_OLLAMA_MODEL_STORAGE_PATH"
        ]
        == "/workspace/ollama"
    )
    serverless = _environment(
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        model_storage_path="/runpod-volume/ollama",
    )
    assert serverless["KESTREL_OLLAMA_MODEL_STORAGE_PATH"] == ("/runpod-volume/ollama")
    for mode, path in (
        (OllamaLeaseMode.DEDICATED_POD, "/runpod-volume/ollama"),
        (OllamaLeaseMode.SERVERLESS_LOAD_BALANCER, "/workspace/ollama"),
        (OllamaLeaseMode.DEDICATED_POD, "/models/../private"),
    ):
        with pytest.raises(RunPodManagerError, match="storage"):
            _environment(mode=mode, model_storage_path=path)


def test_runpod_template_covers_pod_and_native_load_balancer() -> None:
    contract = json.loads((RUNTIME / "runpod-template.json").read_text())

    assert contract["contract"] == OLLAMA_RUNTIME_CONTRACT
    assert contract["runtime_version"] == OLLAMA_RUNTIME_VERSION
    assert contract["image"] == {
        "repository": OLLAMA_RUNTIME_IMAGE_REPOSITORY,
        "require_digest": True,
        "platform": "linux/amd64",
    }
    assert contract["health"]["readiness_path"] == "/ping"
    assert contract["health"]["initializing_status"] == 204
    assert contract["modes"]["dedicated_pod"]["port"] == "11434/http"
    load_balancer = contract["modes"]["serverless_load_balancer"]
    assert load_balancer["endpoint_type"] == "LOAD_BALANCER"
    assert load_balancer["health_path"] == "/ping"
    assert contract["models"]["container_cache_path"] == "/models"
    assert contract["models"]["pod_volume_cache_path"] == "/workspace/ollama"
    assert contract["models"]["serverless_network_cache_path"] == (
        "/runpod-volume/ollama"
    )
    assert contract["models"]["arbitrary_pull"] is False


def test_container_contract_is_pinned_nonroot_and_secret_free() -> None:
    dockerfile = (RUNTIME / "Dockerfile").read_text()

    from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) == 3
    assert all("@sha256:" in line for line in from_lines)
    assert "FROM ollama/ollama:0.32.5@sha256:" in dockerfile
    assert "OLLAMA_COMMIT=eec8e0b9458b8a01be0c216a9cc53eefde24ef50" in dockerfile
    for fixed_module in (
        "github.com/buger/jsonparser@v1.1.2",
        "golang.org/x/crypto@v0.52.0",
        "golang.org/x/image@v0.43.0",
        "golang.org/x/net@v0.55.0",
        "golang.org/x/text@v0.39.0",
    ):
        assert fixed_module in dockerfile
    assert "USER 0:0" in dockerfile
    assert "prepareRuntimeFilesystem(config.ModelStoragePath)" in (
        RUNTIME / "main.go"
    ).read_text()
    assert 'VOLUME ["/models"]' in dockerfile
    assert "EXPOSE 11434" in dockerfile
    assert "RUNPOD_API_KEY=" not in dockerfile
    assert "KESTREL_OLLAMA_BEARER_TOKEN=" not in dockerfile


def test_tracked_configs_share_the_runtime_policy_surface() -> None:
    for name in ("runpod_config.toml", "runpod_config.toml.example"):
        content = (ROOT / name).read_text()
        assert "${RUNPOD_OLLAMA_IMAGE}" in content
        assert "${RUNPOD_OLLAMA_ALLOWED_MODELS}" in content
        assert "OLLAMA_HOST" not in content


def test_runtime_workflow_scans_and_publishes_one_exact_image() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ollama-runtime.yml").read_text()

    assert workflow.count("docker/build-push-action@") == 1
    assert "Publish the exact scanned image" in workflow
    assert 'docker push "$RUNTIME_IMAGE_REF"' in workflow
    assert 'docker pull "${IMAGE_REPOSITORY}@${VERSION_DIGEST}"' in workflow
    assert workflow.count("subject-digest: ${{ steps.publish.outputs.digest }}") == 2
    assert "sbom-path: ollama-runtime.spdx.json" in workflow
    assert "KESTREL_OLLAMA_BEARER_TOKEN" in workflow
