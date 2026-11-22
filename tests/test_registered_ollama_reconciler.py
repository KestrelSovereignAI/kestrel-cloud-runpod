"""Operator config and non-inference entry point contract."""

import io
import os

import pytest

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_runtime import OLLAMA_RUNTIME_IMAGE_REPOSITORY
from kestrel_cloud_runpod.registered_ollama_reconciler import load_registered_endpoint, run_cli


@pytest.fixture
def registration():
    return {
        "endpoint_id": "registered-exact", "endpoint_name": "registered-parked",
        "created_at": "2026-11-22T12:00:30.001Z",
        "image": OLLAMA_RUNTIME_IMAGE_REPOSITORY
        + "@sha256:c316c4e8a71cc801844b0651b4b174d79c75a7ca6725eb75b00138862c8d5cd3",
        "gpu_pool": "ADA_24", "region": "US-NC-1",
        "cache_id": "operator-cache", "registry_id": "operator-registry",
        "min_cuda": "12.8", "disk_gb": 40, "worker_idle_seconds": 5,
        "execution_timeout_ms": 60_000,
    }


def test_registered_reaper_config_describes_only_the_initial_parked_endpoint(registration):
    endpoint = load_registered_endpoint(registration)
    assert endpoint.endpoint_id == registration["endpoint_id"]
    assert endpoint.baseline.to_payload()["workers"] == {"min": 0, "max": 0, "idleTimeout": 5}
    assert endpoint.baseline.network_volume_ids == ("operator-cache",)
    assert endpoint.baseline.data_center_ids == ("US-NC-1",)
    assert endpoint.baseline.env == {"PORT": "11434", "PORT_HEALTH": "11434"}


@pytest.mark.parametrize("overrides", [
    {"region": "CA-MTL-1"}, {"worker_idle_seconds": True}, {"execution_timeout_ms": 0},
    {"created_at": "2026-11-22T19:00:30"}, {"route_key": "not-allowed"},
    {"image": "unreviewed/image:latest"},
])
def test_registered_reaper_refuses_out_of_scope_or_unpinned_config(registration, overrides):
    with pytest.raises(RunPodManagerError):
        load_registered_endpoint({**registration, **overrides})


def test_registered_reaper_rejects_arguments_without_echoing_them():
    stderr = io.StringIO()
    with pytest.raises(SystemExit) as exit:
        run_cli(["--possibly-sensitive-value"], stderr=stderr)
    assert exit.value.code == 2
    assert "invalid arguments" in stderr.getvalue()
    assert "sensitive" not in stderr.getvalue()


def test_registered_reaper_requires_explicit_config_before_network(monkeypatch):
    monkeypatch.delenv("RUNPOD_REGISTERED_OLLAMA_CONFIG", raising=False)
    assert os.getenv("RUNPOD_REGISTERED_OLLAMA_CONFIG") is None
    with pytest.raises(RunPodManagerError, match="absolute RUNPOD_REGISTERED_OLLAMA_CONFIG"):
        run_cli([])
