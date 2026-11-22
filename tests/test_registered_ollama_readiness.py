"""Strict non-secret acceptance evidence for a registered worker."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from ollama_test_support import runtime_telemetry_payload

from kestrel_cloud_runpod.ollama_contracts import OllamaProvisioningEvidenceError
from kestrel_cloud_runpod.ollama_runtime import OLLAMA_RUNTIME_VERSION
from kestrel_cloud_runpod.registered_ollama_readiness import RegisteredWorkerReadback, verify_registered_runtime

START = datetime(2026, 11, 22, tzinfo=UTC)
BINDING = "b" * 64
MODEL = "gpt-oss:20b"
DIGEST = "a" * 64
IMAGE = "registry.example/owner/runtime@sha256:" + "a" * 64
WORKER = {
    "id": "example-worker", "status": "IDLE", "isStale": False, "version": 2,
    "gpuCount": 1, "gpuTypeId": "NVIDIA GeForce RTX 4090", "dataCenterId": "US-NC-1",
    "image": IMAGE, "startedAt": START.isoformat(),
}


def verified_worker(**overrides):
    return RegisteredWorkerReadback.verified(
        {"workers": [{**WORKER, **overrides}], "endpointVersion": 2,
         "summary": {"running": 0, "idle": 1, "initializing": 0, "throttled": 0, "unhealthy": 0, "total": 1}},
        data_center="US-NC-1", gpu_id="NVIDIA GeForce RTX 4090", image=IMAGE,
        activated_not_before=START,
    )


@pytest.mark.parametrize("overrides", [
    {"dataCenterId": "CA-MTL-1"}, {"gpuTypeId": "RTX 4000"}, {"gpuCount": True},
    {"isStale": True}, {"version": None}, {"version": 1}, {"image": "different"},
    {"status": "INITIALIZING"}, {"id": "../other"},
    {"startedAt": (START - timedelta(seconds=1)).isoformat()},
])
def test_worker_requires_exact_provider_id_placement_and_deployment(overrides):
    with pytest.raises(OllamaProvisioningEvidenceError):
        verified_worker(**overrides)


def evidence(snapshot, models, *, change_worker=False):
    worker = verified_worker()
    return verify_registered_runtime(
        snapshot, {"models": models}, telemetry_binding_sha256=BINDING,
        expected_model=MODEL, model_digest="sha256:" + DIGEST, first_worker=worker,
        last_worker=replace(worker, worker_id="different-worker") if change_worker else worker,
    )


def telemetry():
    return {**runtime_telemetry_payload(BINDING, START.isoformat()), "runtime_version": OLLAMA_RUNTIME_VERSION}


def test_route_evidence_binds_cache_only_snapshot_and_exact_model_without_persisting_secrets():
    model = {"name": MODEL, "digest": DIGEST}
    assert evidence(telemetry(), [model]).proves_model_cache_hit
    for candidates in ([{**model, "digest": "b" * 64}], [model, model], []):
        with pytest.raises(OllamaProvisioningEvidenceError, match="model digest"):
            evidence(telemetry(), candidates)
    with pytest.raises(OllamaProvisioningEvidenceError, match="worker changed"):
        evidence(telemetry(), [model], change_worker=True)
    with pytest.raises(OllamaProvisioningEvidenceError, match="pinned cache"):
        evidence({**telemetry(), "telemetry_binding_sha256": "d" * 64}, [model])
