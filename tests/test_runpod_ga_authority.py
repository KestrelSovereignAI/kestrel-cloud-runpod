"""GA creation and storage readback must enforce accepted workload constraints."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import (
    EndpointCreateRequest,
    EndpointResourceVersion,
    PodCreateRequest,
    RunPodManagerError,
    normalize_cuda_floor,
    validate_endpoint_mounts,
)
from kestrel_cloud_runpod.serverless_capacity_contracts import ServerlessEndpointSpec


@pytest.mark.parametrize(
    "floor,expected",
    [(None, None), ("12", "12.0"), ("12.8", "12.8"), ("12.11", "12.11")],
)
def test_cuda_floor_is_sent_in_both_create_requests(floor, expected):
    requests = (
        PodCreateRequest(
            name="pod", image="image", gpu_id="gpu", min_cuda_version=floor
        ),
        EndpointCreateRequest(
            name="endpoint",
            image="image",
            gpu_pools=("pool",),
            endpoint_type="QUEUE",
            scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
            min_cuda_version=floor,
        ),
    )
    for request in requests:
        gpu = request.to_payload()["gpu"]
        if expected is None:
            assert "minCudaVersion" not in gpu
        else:
            assert gpu["minCudaVersion"] == expected
    assert normalize_cuda_floor(floor) == expected


@pytest.mark.parametrize("floor", ["", "12.x", " 12.8", "12.8.1", 12.8, True])
def test_invalid_cuda_constraints_fail_before_creation(floor):
    with pytest.raises(ValueError, match="min_cuda_version"):
        PodCreateRequest(
            name="pod", image="image", gpu_id="gpu", min_cuda_version=floor
        )
    with pytest.raises(ValueError, match="min_cuda_version"):
        EndpointCreateRequest(
            name="endpoint",
            image="image",
            gpu_pools=("pool",),
            endpoint_type="QUEUE",
            scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
            min_cuda_version=floor,
        )


def test_endpoint_create_explicitly_binds_both_storage_representations():
    payload = EndpointCreateRequest(
        name="endpoint",
        image="image",
        gpu_pools=("pool",),
        endpoint_type="QUEUE",
        scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
        network_volume_ids=("accepted-volume",),
    ).to_payload()
    assert payload["networkVolumes"] == ["accepted-volume"]
    assert payload["mounts"] == {
        "network": [{"volumeId": "accepted-volume"}],
        "global": [],
    }
    validate_endpoint_mounts(payload, ("accepted-volume",))


def test_historical_spec_round_trips_without_rewriting_its_authority():
    value = json.loads(
        (
            Path(__file__).parent / "fixtures/serverless-0.10.1-authority.json"
        ).read_text()
    )["endpoint"]["spec"]
    assert ServerlessEndpointSpec.from_dict(value).to_dict() == value


def test_legacy_wire_shape_is_readable_but_the_transport_cannot_create_from_it():
    value = json.loads(
        (
            Path(__file__).parent / "fixtures/serverless-0.10.1-authority.json"
        ).read_text()
    )["endpoint"]["spec"]
    spec = ServerlessEndpointSpec.from_dict(value)
    request = spec.create_request(
        "legacy-endpoint",
        gpu_pool=spec.constraints.allowed_gpu_pools[0],
        data_center_id=spec.constraints.allowed_data_center_ids[0],
    )
    assert "mounts" not in request.to_payload()
    assert "minCudaVersion" not in request.to_payload()["gpu"]
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(201, json={})

    client = RunpodControlPlaneClient(
        api_key="unit-test",
        http_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RunPodManagerError, match="upgrade"):
        client.create_endpoint(request)
    assert sent == []


@pytest.mark.parametrize("version", [True, 1, 3, "2", None])
def test_unknown_or_noncanonical_resource_version_cannot_be_deserialized(version):
    value = json.loads(
        (
            Path(__file__).parent / "fixtures/serverless-0.10.1-authority.json"
        ).read_text()
    )["endpoint"]["spec"]
    value["resource_schema_version"] = version
    with pytest.raises(RunPodManagerError):
        ServerlessEndpointSpec.from_dict(value)


def test_removing_a_ga_version_marker_cannot_downgrade_its_saved_authority():
    historical = json.loads(
        (
            Path(__file__).parent / "fixtures/serverless-0.10.1-authority.json"
        ).read_text()
    )["endpoint"]["spec"]
    ga = replace(
        ServerlessEndpointSpec.from_dict(historical),
        resource_schema_version=EndpointResourceVersion.GA,
    ).to_dict()
    del ga["resource_schema_version"]
    with pytest.raises(RunPodManagerError, match="digest mismatches"):
        ServerlessEndpointSpec.from_dict(ga)


@pytest.mark.parametrize(
    "mounts",
    [
        None,
        {},
        {"network": []},
        {"network": [], "global": [{"volumeId": "unquoted-global-volume"}]},
        {"network": [{"volumeId": "unquoted-network-volume"}], "global": []},
        {"network": "not-an-array", "global": []},
        {"network": [], "global": None},
        {"network": [None], "global": []},
        {"network": [{"volumeId": ""}], "global": []},
        {"network": [], "global": [], "persistent": {"size": 100}},
    ],
)
def test_unknown_or_ambiguous_storage_cannot_be_treated_as_mountless(mounts):
    with pytest.raises(RunPodManagerError):
        validate_endpoint_mounts(
            {"networkVolumes": [], "mounts": mounts},
            (),
        )
