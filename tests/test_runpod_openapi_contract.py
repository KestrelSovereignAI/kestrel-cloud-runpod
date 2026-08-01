"""Parity tests between Kestrel's typed calls and the pinned beta schema."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from kestrel_cloud_runpod.models import (
    CONTROL_PLANE_BASE_URL,
    EndpointCreateRequest,
    EndpointUpdateRequest,
    FlashBoot,
    PodCreateRequest,
)
from scripts.check_runpod_openapi import load_openapi, semantic_diff

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "vendor" / "runpod-v2-openapi.yaml"
LOCK_PATH = ROOT / "vendor" / "runpod-v2-openapi.lock.json"

REQUIRED_OPERATIONS = {
    ("/v2/catalog/gpus", "get"): "listGpuTypes",
    ("/v2/catalog/gpus/{id}", "get"): "getGpuType",
    ("/v2/pods", "get"): "listPods",
    ("/v2/pods", "post"): "createPod",
    ("/v2/pods/{id}", "get"): "getPod",
    ("/v2/pods/{id}", "delete"): "deletePod",
    ("/v2/pods/{id}/action", "post"): "podAction",
    ("/v2/pods/{id}/logs", "get"): "getPodLogs",
    ("/v2/serverless", "get"): "listEndpoints",
    ("/v2/serverless", "post"): "createEndpoint",
    ("/v2/serverless/{id}", "get"): "getEndpoint",
    ("/v2/serverless/{id}", "patch"): "updateEndpoint",
    ("/v2/serverless/{id}", "delete"): "deleteEndpoint",
    ("/v2/serverless/{id}/workers", "get"): "listEndpointWorkers",
    ("/v2/serverless/{id}/workers/{workerId}/logs", "get"): "getWorkerLogs",
    ("/v2/billing/pods", "get"): "listPodBilling",
    ("/v2/billing/serverless", "get"): "listServerlessBilling",
}


def _schema():
    return load_openapi(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_pin_checksum_and_validator_script():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    assert digest == lock["sha256"]
    result = subprocess.run(
        [sys.executable, "scripts/check_runpod_openapi.py", "--check-pin"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "checksum and document are valid" in result.stdout


def test_all_consumed_control_plane_operations_match_pin():
    paths = _schema()["paths"]
    for (path, method), operation_id in REQUIRED_OPERATIONS.items():
        assert paths[path][method]["operationId"] == operation_id


def test_consumed_request_and_response_shapes_match_pin():
    schemas = _schema()["components"]["schemas"]
    assert set(schemas["GpuType"]["required"]) >= {
        "id",
        "pool",
        "memory",
        "price",
        "maxCount",
    }
    assert schemas["ListGpuTypesResponse"]["required"] == ["gpus"]
    assert schemas["ListPodsResponse"]["required"] == ["pods"]
    assert schemas["ListEndpointsResponse"]["required"] == ["endpoints"]
    assert schemas["ListEndpointWorkersResponse"]["required"] == [
        "workers",
        "summary",
    ]
    assert schemas["ListPodBillingResponse"]["required"] == ["records", "metadata"]
    assert schemas["ListServerlessBillingResponse"]["required"] == [
        "records",
        "metadata",
    ]
    assert set(schemas["ErrorResponse"]["required"]) == {"title", "status", "detail"}
    assert set(schemas["PodAction"]["enum"]) == {
        "start",
        "stop",
        "restart",
        "terminate",
    }


def test_typed_create_and_update_payloads_validate_against_pin():
    document = _schema()

    def validate(component, payload):
        schema = {**document, "$ref": f"#/components/schemas/{component}"}
        Draft202012Validator(schema).validate(payload)

    validate(
        "CreatePodRequest",
        PodCreateRequest(
            name="kestrel-pod",
            image="registry.example/kestrel:sha",
            gpu_id="NVIDIA RTX PRO 4000 Blackwell",
            env={"MODEL": "flux"},
        ).to_payload(),
    )
    validate(
        "CreateEndpointRequest",
        EndpointCreateRequest(
            name="kestrel-catalog",
            image="registry.example/catalog:sha",
            gpu_pools=("BLACKWELL_24",),
            endpoint_type="QUEUE",
            scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
            flashboot=FlashBoot.FLASHBOOT,
        ).to_payload(),
    )
    validate(
        "UpdateEndpointRequest",
        EndpointUpdateRequest(
            gpu_pools=("BLACKWELL_24",),
            gpu_count=1,
            flashboot=FlashBoot.OFF,
        ).to_payload(),
    )


def test_runtime_base_url_is_explicit_despite_beta_schema_server_discrepancy():
    document = _schema()
    assert document["servers"][0]["url"] == "https://api.runpod.io"
    assert CONTROL_PLANE_BASE_URL == "https://v2-rest.runpod.io/v2"


def test_semantic_diff_reports_breaking_and_additive_changes():
    pinned = {
        "paths": {"/v2/pods": {"get": {}}},
        "components": {
            "schemas": {"Pod": {"required": ["id"], "enum": ["RUNNING", "EXITED"]}}
        },
    }
    live = {
        "paths": {"/v2/serverless": {"post": {}}},
        "components": {
            "schemas": {
                "Pod": {
                    "required": ["id", "status"],
                    "enum": ["RUNNING", "ERROR"],
                },
                "Endpoint": {},
            }
        },
    }
    changes = semantic_diff(pinned, live)
    assert "REMOVED operation GET /v2/pods" in changes
    assert "ADDED operation POST /v2/serverless" in changes
    assert "NEW required field Pod.status" in changes
    assert "REMOVED enum value Pod.EXITED" in changes
    assert "ADDED schema Endpoint" in changes


def test_production_package_contains_no_v1_graphql_or_legacy_sdk_calls():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "kestrel_cloud_runpod").glob("*.py")
    )
    assert "rest.runpod.io/v1" not in source
    assert "runpod.create_pod" not in source
    assert "runpod.get_pod" not in source
    assert "runpod.api_key" not in source
    assert "runpod.cli" not in source
    assert "graphql.runpod.io" not in source.lower()
