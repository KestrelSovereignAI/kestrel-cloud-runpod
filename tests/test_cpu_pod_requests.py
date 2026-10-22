"""CPU/GPU wire variants through the existing v2 transport; no cloud calls."""

import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import httpx
import pytest
from jsonschema import Draft202012Validator

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import (
    CpuPodCreateRequest,
    PodCreateRequest,
    PodResource,
    ResourceFieldState,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)
from scripts.check_runpod_openapi import load_openapi


def cpu_request(**overrides):
    return CpuPodCreateRequest(
        **{
            "name": "cpu-browser",
            "image": "registry.example/browser@sha256:" + "a" * 64,
            "cpu_id": "cpu5c",
            "vcpu_count": 4,
            **overrides,
        }
    )


def pod(**overrides):
    return {
        "id": "cpu-pod",
        "name": "cpu-browser",
        "status": "PROVISIONING",
        "gpu": None,
        "cpu": {"id": "cpu5c", "vcpuCount": 4, "memory": 8},
        "cost": 0.2,
        **overrides,
    }


def test_cpu_and_gpu_requests_use_the_same_pinned_schema():
    root = Path(__file__).resolve().parents[1]
    document = load_openapi((root / "vendor/runpod-v2-openapi.yaml").read_text())
    validator = Draft202012Validator(
        {
            "$ref": "#/components/schemas/CreatePodRequest",
            "components": document["components"],
        }
    )
    cpu = cpu_request(
        ports=("8000/http",),
        env={"PRIVATE": "not-in-repr"},
        args="python worker.py",
        registry_id="registry",
        data_center_ids=("US-TX-3",),
        disk_gb=20,
        mounts={"network": [{"volumeId": "owned-volume", "path": "/workspace"}]},
    )
    gpu = PodCreateRequest(
        name="gpu", image="image", gpu_id="gpu-id", min_cuda_version="12"
    )
    for request, kind in ((cpu, "cpu"), (gpu, "gpu")):
        payload = request.to_payload()
        validator.validate(payload)
        assert {"cpu", "gpu"} & payload.keys() == {kind}
    wire = cpu.to_payload()
    assert wire["cpu"] == {"id": "cpu5c", "vcpuCount": 4}
    assert wire["mounts"] == {
        "network": [{"volumeId": "owned-volume", "path": "/workspace"}]
    }
    assert wire["env"] == {"PRIVATE": "not-in-repr"}
    assert wire["args"] == "python worker.py" and wire["registry"] == "registry"
    assert wire["dataCenterIds"] == ["US-TX-3"] and wire["ports"] == ["8000/http"]
    assert wire["disk"] == 20 and wire["globalNetworking"] is False
    assert "not-in-repr" not in repr(cpu)
    assert "python worker.py" not in repr(cpu)
    assert gpu.to_payload()["gpu"] == {
        "id": "gpu-id",
        "count": 1,
        "minCudaVersion": "12.0",
    }


@pytest.mark.parametrize("count", [0, 1, 3, 6, -2, True, False, 4.0, "4", None])
def test_invalid_cpu_counts_are_refused_before_any_transport(count):
    with pytest.raises(ValueError, match="vcpu_count"):
        cpu_request(vcpu_count=count)


@pytest.mark.parametrize("count", [2, 4, 8, 64])
def test_cpu_count_is_explicit_without_requesting_ram(count):
    assert cpu_request(vcpu_count=count).to_payload()["cpu"] == {
        "id": "cpu5c",
        "vcpuCount": count,
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"gpu_id": "gpu"},
        {"gpu_count": 1},
        {"min_cuda_version": "12"},
        {"memory": 8},
    ],
)
def test_gpu_and_ram_constraints_cannot_be_added_to_cpu_variant(fields):
    with pytest.raises(TypeError):
        cpu_request(**fields)


@pytest.mark.parametrize("cpu_id", [None, "", 5, False])
def test_cpu_flavor_is_required(cpu_id):
    with pytest.raises(ValueError, match="cpu_id"):
        cpu_request(cpu_id=cpu_id)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"global_networking": True}, id="fields0"),
        pytest.param({"global_networking": 0}, id="fields1"),
        pytest.param(
            {"mounts": {"persistent": {"size": 10, "path": "/workspace"}}},
            id="fields2",
        ),
        pytest.param({"mounts": {"persistent": None}}, id="fields3"),
        pytest.param({"mounts": {"global": []}}, id="fields4"),
        pytest.param({"mounts": {"unknown": []}}, id="fields5"),
        pytest.param({"mounts": []}, id="fields6"),
        pytest.param({"mounts": {"network": None}}, id="fields7"),
        pytest.param(
            {"mounts": {"network": [{"volumeId": "v", "path": "/v"}] * 2}},
            id="fields9",
        ),
        pytest.param(
            {"mounts": {"network": [{"volumeId": "v"}]}},
            id="fields10",
        ),
        pytest.param(
            {"mounts": {"network": [{"volumeId": "", "path": "/v"}]}},
            id="fields11",
        ),
        pytest.param(
            {"mounts": {"network": [{"volumeId": "v", "path": "/v", "extra": True}]}},
            id="fields12",
        ),
    ],
)
def test_cpu_rejects_unsupported_networking_and_storage(fields):
    with pytest.raises(ValueError, match="CPU Pod"):
        cpu_request(**fields)


def test_cpu_request_deeply_owns_environment_and_mount_payload():
    mounts = {"network": [{"volumeId": "v", "path": "/workspace"}]}
    environment = {"PRIVATE": "original"}
    request = cpu_request(mounts=mounts, env=environment)
    mounts["persistent"] = {"size": 1, "path": "/other"}
    mounts["network"][0]["volumeId"] = "foreign"
    environment["PRIVATE"] = "mutated"
    calls = []

    def handler(raw_request):
        calls.append(raw_request)
        payload = json.loads(raw_request.content)
        assert payload["env"] == {"PRIVATE": "original"}
        assert payload["mounts"] == {
            "network": [{"volumeId": "v", "path": "/workspace"}]
        }
        return httpx.Response(201, json=pod())

    with closing(
        RunpodControlPlaneClient(
            api_key="unit-test", http_transport=httpx.MockTransport(handler)
        )
    ) as client:
        client.create_pod(request)
    assert len(calls) == 1


def test_frozen_cpu_request_supports_ordinary_dataclass_replacement():
    original = cpu_request(
        mounts={"network": [{"volumeId": "owned-volume", "path": "/workspace"}]},
        env={"PRIVATE": "original"},
    )
    replaced = replace(original, name="cpu-browser-next")

    assert replaced.name == "cpu-browser-next"
    assert replaced.to_payload() == {
        **original.to_payload(),
        "name": "cpu-browser-next",
    }


@pytest.mark.parametrize("outer_kind", ["dict", "mapping-proxy"])
@pytest.mark.parametrize("network_kind", ["list", "tuple"])
@pytest.mark.parametrize("populated", [False, True])
def test_cpu_mount_sequences_have_one_immutable_value_policy(
    outer_kind, network_kind, populated
):
    mount = {"volumeId": "owned-volume", "path": "/workspace"}
    items = [mount] if populated else []
    network = items if network_kind == "list" else tuple(items)
    mutable_outer = {"network": network}
    mounts = mutable_outer if outer_kind == "dict" else MappingProxyType(mutable_outer)
    original = cpu_request(mounts=mounts)
    first = replace(original, name="cpu-browser-next")
    second = replace(first, name="cpu-browser-final")
    expected = [{"volumeId": "owned-volume", "path": "/workspace"}] if populated else []
    assert second.to_payload()["mounts"] == {"network": expected}

    mount["volumeId"] = "foreign"
    mutable_outer["network"] = []
    returned = second.to_payload()
    if populated:
        returned["mounts"]["network"][0]["volumeId"] = "mutated"
    assert second.to_payload()["mounts"] == {"network": expected}


def test_cpu_create_inspect_and_delete_use_existing_v2_transport():
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            assert json.loads(request.content)["cpu"] == {"id": "cpu5c", "vcpuCount": 4}
            assert "gpu" not in json.loads(request.content)
            return httpx.Response(201, json=pod(env={"PRIVATE": "must-not-leak"}))
        if request.method == "GET":
            return httpx.Response(200, json=pod(status="RUNNING"))
        assert request.method == "DELETE"
        return httpx.Response(204)

    with closing(
        RunpodControlPlaneClient(
            api_key="unit-test", http_transport=httpx.MockTransport(handler)
        )
    ) as client:
        created = client.create_pod(cpu_request())
        assert created.gpu_id is None and created.gpu_count == 0
        assert created.raw["cpu"] == {"id": "cpu5c", "vcpuCount": 4, "memory": 8}
        assert "must-not-leak" not in repr(created)
        assert client.get_pod(created.id).status == "RUNNING"
        client.delete_pod(created.id)
    assert [request.url.path for request in calls] == [
        "/v2/pods",
        "/v2/pods/cpu-pod",
        "/v2/pods/cpu-pod",
    ]
    assert all(request.url.host == "api.runpod.io" for request in calls)


def test_lost_cpu_create_response_remains_unknown_without_replay():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    with (
        closing(
            RunpodControlPlaneClient(
                api_key="unit-test", http_transport=httpx.MockTransport(handler)
            )
        ) as client,
        pytest.raises(RunPodAmbiguousResultError),
    ):
        client.create_pod(cpu_request())
    assert len(calls) == 1


def test_known_create_id_retains_invalid_optional_success_evidence():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            201,
            json=pod(cost=-1, env={"PRIVATE": "must-not-leak"}),
            headers={"Retry-After": "3"},
        )

    with closing(
        RunpodControlPlaneClient(
            api_key="unit-test", http_transport=httpx.MockTransport(handler)
        )
    ) as client:
        created = client.create_pod(cpu_request())
        assert client.transport.last_rate_limit.retry_after_seconds == 3
    assert len(calls) == 1
    assert created.id == "cpu-pod"
    assert created.cost_per_hr is None
    assert created.field_evidence.state("cost") is ResourceFieldState.INVALID
    assert created.raw["env"] == {"PRIVATE": "[REDACTED]"}
    with pytest.raises(RunPodManagerError, match="cost") as result:
        created.require_valid_fields(context="v2 Pod create")
    assert "must-not-leak" not in str(result.value)


@pytest.mark.parametrize("body", [b"", b"[]", b"not json", b"{}"])
def test_malformed_success_keeps_create_custody(body):
    with (
        closing(
            RunpodControlPlaneClient(
                api_key="unit-test",
                http_transport=httpx.MockTransport(
                    lambda req: httpx.Response(201, content=body)
                ),
            )
        ) as client,
        pytest.raises(RunPodAmbiguousResultError),
    ):
        client.create_pod(cpu_request())


def test_known_provider_refusal_stays_a_known_refusal():
    with (
        closing(
            RunpodControlPlaneClient(
                api_key="unit-test",
                http_transport=httpx.MockTransport(
                    lambda req: httpx.Response(
                        422,
                        json={
                            "title": "Invalid request",
                            "detail": "CPU unavailable",
                            "status": 422,
                        },
                    )
                ),
            )
        ) as client,
        pytest.raises(RunPodAPIError) as result,
    ):
        client.create_pod(cpu_request())
    assert not isinstance(result.value, RunPodAmbiguousResultError)


@pytest.mark.parametrize("count", [0, -1, True, False, 1.5, "1", None])
def test_gpu_variant_keeps_positive_integer_counts(count):
    with pytest.raises(ValueError, match="gpu_count"):
        PodCreateRequest(name="gpu", image="image", gpu_id="gpu", gpu_count=count)


@pytest.mark.parametrize("size", [0, -1, True, 1.0, "1", None])
def test_shared_disk_size_is_a_positive_integer(size):
    with pytest.raises(ValueError, match="disk_gb"):
        cpu_request(disk_gb=size)


@pytest.mark.parametrize("gpu", [None, {"id": "gpu", "count": 1}])
def test_missing_rate_remains_unknown_for_every_pod(gpu):
    response = pod(gpu=gpu)
    response.pop("cost")
    if gpu is not None:
        response.pop("cpu")
    assert PodResource.from_dict(response).cost_per_hr is None


@pytest.mark.parametrize("rate", [-1, float("inf"), float("nan"), 10**400, True, "0.2"])
def test_invalid_pod_rate_is_not_a_usable_price(rate):
    resource = PodResource.from_dict(pod(cost=rate))
    assert resource.cost_per_hr is None
    assert resource.field_evidence.state("cost") is ResourceFieldState.INVALID
    with pytest.raises(RunPodManagerError, match="cost"):
        resource.require_valid_fields(context="v2 Pod readback")


def test_explicit_zero_rate_remains_zero():
    assert PodResource.from_dict(pod(cost=0)).cost_per_hr == 0


@pytest.mark.parametrize("depth", [2000, 40000])
def test_deep_success_json_keeps_create_custody(depth):
    body = (
        b'{"id":"p","name":"cpu","status":"RUNNING","extra":'
        + b'{"x":' * depth
        + b"0"
        + b"}" * depth
        + b"}"
    )
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(201, content=body)

    with (
        closing(
            RunpodControlPlaneClient(
                api_key="synthetic", http_transport=httpx.MockTransport(handler)
            )
        ) as client,
        pytest.raises(RunPodAmbiguousResultError),
    ):
        client.create_pod(cpu_request())
    assert len(calls) == 1


def test_parsed_receipt_recursion_keeps_create_custody(monkeypatch):
    # Exercise the post-decode sanitizer on every supported Python version;
    # JSON decoder recursion thresholds differ between 3.11 and 3.14.
    raw = pod()
    cursor = raw
    for _ in range(2000):
        child = {}
        cursor["extra"] = child
        cursor = child
    calls = []

    def response(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    with (
        closing(
            RunpodControlPlaneClient(
                api_key="synthetic", http_transport=httpx.MockTransport(lambda _: None)
            )
        ) as client,
        pytest.raises(RunPodAmbiguousResultError),
    ):
        monkeypatch.setattr(client.transport, "request_json", response)
        client.create_pod(cpu_request())
    assert len(calls) == 1


@pytest.mark.parametrize("status", [200, 202, 204, 206])
def test_unexpected_success_status_preserves_mutation_custody(status):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(status, json=pod())

    with (
        closing(
            RunpodControlPlaneClient(
                api_key="synthetic", http_transport=httpx.MockTransport(handler)
            )
        ) as client,
        pytest.raises(RunPodAmbiguousResultError),
    ):
        client.create_pod(cpu_request())
    assert len(calls) == 1


@pytest.mark.parametrize("status,ambiguous", [(201, True), (500, True), (422, False)])
def test_decoder_recursion_preserves_http_outcome_classification(status, ambiguous):
    body = b"[" * 200000 + b"0" + b"]" * 200000
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(status, content=body)

    with (
        closing(
            RunpodControlPlaneClient(
                api_key="synthetic", http_transport=httpx.MockTransport(handler)
            )
        ) as client,
        pytest.raises(RunPodAPIError) as result,
    ):
        client.create_pod(cpu_request())
    assert isinstance(result.value, RunPodAmbiguousResultError) is ambiguous
    assert len(calls) == 1
