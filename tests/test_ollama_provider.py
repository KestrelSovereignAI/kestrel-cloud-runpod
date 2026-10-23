"""Runpod v2 adapter tests with in-memory HTTP/control-plane doubles."""

import asyncio
import json
import sqlite3
import sys
import threading
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpcore
import httpx
import pytest
from jsonschema import Draft202012Validator
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    OwnedLoopbackHTTPServer,
    OwnedSpawnProcesses,
    make_request,
    non_compute_cost_policy,
    pod_plan,
    release_gates_and_boundedly_join_owned_tasks,
    runtime_telemetry_payload,
    serverless_plan,
)
from pod_capacity_test_support import (
    execution_control_carrier,
    native_httpx_connect_failure,
    native_pod_action_harness,
    record_control_injection,
)

import kestrel_cloud_runpod.ollama_provider as ollama_provider_module
from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import (
    Availability,
    ComputeProduct,
    EndpointResource,
    FlashBoot,
    GPUOffer,
    GPUProfile,
    PodResource,
    RunPodAPIError,
    RunPodManagerError,
    assess_endpoint_mounts,
    assess_pod_mounts,
    environment_values_sha256,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaCleanupBasis,
    OllamaCleanupIntent,
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaLifecycleBillingState,
    OllamaProvisioningEvidenceError,
    OllamaProvisioningMismatchError,
    OllamaProvisioningMismatchReason,
    OllamaResourceType,
    OllamaTeardownState,
    ProvisionedOllamaResource,
)
from kestrel_cloud_runpod.ollama_provider import (
    RunpodOllamaCapacityProvider,
    RunpodOllamaDeployment,
    _pod_base_url,
)
from kestrel_cloud_runpod.ollama_repository import (
    SQLiteOllamaLeaseRepository,
    _same_provisioning_receipt_authority,
)
from kestrel_cloud_runpod.ollama_runtime import OLLAMA_RUNTIME_IMAGE_REPOSITORY
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService
from scripts.check_runpod_openapi import load_openapi

_TEST_DIGEST = "sha256:" + "a" * 64
_TEST_IMAGE = f"{OLLAMA_RUNTIME_IMAGE_REPOSITORY}@{_TEST_DIGEST}"
_SERVERLESS_TOKEN = "serverless-" + "s" * 32
_POD_TOKEN = "pod-" + "p" * 32


def _cleanup_intent(resource: ProvisionedOllamaResource) -> OllamaCleanupIntent:
    return OllamaCleanupIntent(
        intent_sha256="a" * 64,
        provisioning_receipt_id=None,
        lease_id="lease-test",
        request_fingerprint="b" * 64,
        provision_attempt_id="attempt-test",
        resource_type=resource.resource_type,
        provider_resource_id=resource.provider_resource_id,
        resource_name=resource.resource_name,
        termination_reason="test",
        delete_started_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def _profile() -> GPUProfile:
    return GPUProfile(
        id="ollama",
        name="Ollama",
        task_type="ollama",
        image_name="${TEST_OLLAMA_IMAGE}",
        container_disk_gb=40,
        volume_gb=0,
        ports=["11434/http"],
        inference_port=11434,
        min_vram_gb=24,
        gpu_count=1,
        env={"KESTREL_OLLAMA_ALLOWED_MODELS": f"qwen3:8b@{_TEST_DIGEST}"},
    )


def _deployment() -> RunpodOllamaDeployment:
    return RunpodOllamaDeployment(
        profile=_profile(),
        serverless_workers_max=1,
        serverless_request_count=1,
        serverless_execution_timeout_ms=300_000,
        serverless_flashboot=FlashBoot.FLASHBOOT,
        http_timeout_seconds=10,
        serverless_non_compute_cost=non_compute_cost_policy(),
        pod_non_compute_cost=non_compute_cost_policy(),
    )


def _offer(product: ComputeProduct) -> GPUOffer:
    return GPUOffer(
        id=f"gpu-{product.value.lower()}",
        name="GPU",
        pool="pool-24" if product is ComputeProduct.SERVERLESS else None,
        manufacturer="NVIDIA",
        memory_gb=24,
        secure=True,
        community=False,
        secure_price_per_hr=0.5,
        serverless_price_per_hr=1.0 if product is ComputeProduct.SERVERLESS else None,
        community_price_per_hr=0,
        secure_max_count=0 if product is ComputeProduct.SERVERLESS else 1,
        community_max_count=0,
        availability=Availability.HIGH,
    )


class _ControlClient:
    def __init__(self) -> None:
        self.endpoint_request = None
        self.pod_request = None
        self.actions = []
        self.delete_error = None
        self.error_callback = None
        self.absent = set()
        self.pod = None
        self.endpoint = EndpointResource(
            id="endpoint-1",
            name="resource",
            endpoint_type="LOAD_BALANCER",
            request_urls={
                "base": "https://endpoint.example",
                "health": "https://endpoint.example/ping",
            },
            raw={"mounts": {"network": [], "global": []}},
        )

    def list_gpus(self, *, products, **kwargs):
        return (
            replace(
                _offer(products[0]),
                availability_min_cuda_version=kwargs.get("min_cuda_version"),
            ),
        )

    def create_endpoint(self, request):
        self.endpoint_request = request
        self.endpoint = EndpointResource(
            id="endpoint-1",
            name=request.name,
            endpoint_type=request.endpoint_type,
            request_urls={
                "base": "https://endpoint.example",
                "health": "https://endpoint.example/ping",
            },
            raw=request.to_payload(),
            image=request.image,
            environment_sha256=environment_values_sha256(request.env),
            created_at=datetime(2026, 8, 1, 12, tzinfo=UTC),
        )
        return self.endpoint

    def create_pod(self, request):
        self.pod_request = request
        response = request.to_payload()
        response["args"] = response.get("args", "")
        response["mounts"] = response.get("mounts", {})
        response["registry"] = response.get("registry")
        response["globalNetworking"] = {"enabled": response["globalNetworking"]}
        self.pod = PodResource(
            id="pod-evidence-1",
            name=request.name,
            status="PROVISIONING",
            gpu_id=request.gpu_id,
            gpu_count=request.gpu_count,
            cost_per_hr=Decimal("0.5"),
            raw=response,
            cloud=request.cloud,
            data_center_id=(request.data_center_ids or ("US-TX-3",))[0],
            cuda_version=request.min_cuda_version or "12.0",
            image=request.image,
            environment_sha256=environment_values_sha256(request.env),
            created_at=datetime(2026, 8, 1, 12, tzinfo=UTC),
        )
        return self.pod

    def get_endpoint(self, endpoint_id):
        assert endpoint_id == "endpoint-1"
        self._require_present(endpoint_id)
        return self.endpoint

    def get_pod(self, pod_id):
        self._require_present(pod_id)
        if self.pod is not None:
            assert pod_id == self.pod.id
            return self.pod
        return PodResource(
            id=pod_id,
            name="resource",
            status="RUNNING",
            gpu_id="gpu-pod",
            gpu_count=1,
            cost_per_hr=0.5,
            raw={"runtime": {"ports": [{"private": 11434, "type": "http"}]}},
        )

    def list_endpoints(self):
        return (self.endpoint,)

    def list_pods(self):
        return ()

    def delete_endpoint(self, endpoint_id):
        if self.delete_error:
            if self.error_callback is not None:
                self.error_callback("delete")
            if self.delete_error.status_code == 404:
                self.absent.add(endpoint_id)
            raise self.delete_error
        self.actions.append((endpoint_id, "delete"))
        self.absent.add(endpoint_id)

    def pod_action(self, pod_id, action):
        self.actions.append((pod_id, action))
        if action == "terminate":
            self.absent.add(pod_id)

    def _require_present(self, resource_id):
        if resource_id in self.absent:
            raise RunPodAPIError(
                title="Not found",
                detail="Exact resource is absent",
                status_code=404,
            )


def _provider(
    *,
    client=None,
    transport=None,
    deployment=None,
    serverless_api_key=_SERVERLESS_TOKEN,
    pod_bearer_token=_POD_TOKEN,
    control_plane_api_key="control-" + "c" * 32,
):
    return RunpodOllamaCapacityProvider(
        client=client or _ControlClient(),
        deployment=deployment or _deployment(),
        serverless_api_key=serverless_api_key,
        pod_bearer_token=pod_bearer_token,
        control_plane_api_key=control_plane_api_key,
        http_transport=transport,
        clock=MutableClock(),
    )


async def _readiness_authority(provider, mode):
    request = make_request(
        provider._clock,
        mode=mode,
        max_authorized_cost=5.0,
    )
    plan = (
        serverless_plan()
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
        else pod_plan(rate=0.5)
    )
    prepared = provider.validate_runtime_request(
        request,
        plan,
        provision_requested_at=provider._clock(),
    )
    resource = await provider.provision(prepared)
    if mode is OllamaLeaseMode.DEDICATED_POD:
        assert isinstance(provider.client, _ControlClient)
        pod = provider.client.pod
        assert pod is not None
        provider.client.pod = replace(
            pod,
            status="RUNNING",
            raw={
                **pod.raw,
                "runtime": {
                    "ports": [{"private": 11434, "type": "http"}],
                },
            },
        )
    receipt = await provider.observe_provisioning(prepared.intent, resource)
    return resource, prepared.intent, receipt, prepared


def _execution_control_carrier(error):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    control = ExecutionCommitOutcomeError("unknown")
    error.__cause__ = control
    return error, control


@pytest.mark.asyncio
async def test_plan_does_not_aggregate_a_manager_carried_execution_control():
    client = _ControlClient()
    carrier, control = _execution_control_carrier(
        RunPodAPIError(
            title="Control",
            detail="catalog result remains unknown",
            status_code=503,
        )
    )

    def controlled_catalog(**_kwargs):
        raise carrier

    client.list_gpus = controlled_catalog
    provider = _provider(client=client)

    with pytest.raises(RunPodAPIError) as raised:
        await provider.plan(make_request(MutableClock()))

    assert raised.value is carrier
    assert raised.value.__cause__ is control


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "carrier_type",
    [RunPodAPIError, OllamaProvisioningEvidenceError],
    ids=("ordinary-manager-wrapper", "typed-evidence-preserved"),
)
async def test_provisioning_readback_adapter_preserves_execution_control_graph(
    monkeypatch, carrier_type
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider = _provider()
    resource, intent, _receipt, _prepared = await _readiness_authority(
        provider, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    if carrier_type is RunPodAPIError:
        error = carrier_type(
            title="Control",
            detail="readback outcome remains unknown",
            status_code=503,
        )
    else:
        error = carrier_type("typed readback evidence remains unknown")
    carrier, control = _execution_control_carrier(error)

    def controlled_validation(*_args, **_kwargs):
        raise carrier

    monkeypatch.setattr(
        ollama_provider_module,
        "_validate_endpoint_provisioning",
        controlled_validation,
    )

    with pytest.raises(carrier_type) as raised:
        await provider.observe_provisioning(intent, resource)

    assert raised.value is carrier
    assert raised.value.__cause__ is control


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "endpoint-runtime",
        "endpoint-gpu",
        "endpoint-list",
        "pod-runtime",
        "pod-networking",
    ],
)
async def test_typed_readback_field_adapter_does_not_erase_execution_control(
    monkeypatch, boundary
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    mode = (
        OllamaLeaseMode.DEDICATED_POD
        if boundary.startswith("pod-")
        else OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    provider = _provider()
    resource, intent, _receipt, _prepared = await _readiness_authority(provider, mode)
    carrier, control = _execution_control_carrier(
        OllamaProvisioningEvidenceError(f"{boundary} typed evidence remains unknown")
    )
    if boundary in {"endpoint-runtime", "pod-runtime"}:
        original = ollama_provider_module._typed_runtime_value

        def controlled_runtime_value(key, value, *, context):
            if context == ("Pod" if boundary.startswith("pod-") else "Endpoint"):
                raise carrier
            return original(key, value, context=context)

        monkeypatch.setattr(
            ollama_provider_module, "_typed_runtime_value", controlled_runtime_value
        )
    elif boundary == "endpoint-gpu":
        original = ollama_provider_module._required_int

        def controlled_required_int(value, context):
            if context == "Endpoint.gpu.count":
                raise carrier
            return original(value, context)

        monkeypatch.setattr(
            ollama_provider_module, "_required_int", controlled_required_int
        )
    elif boundary == "endpoint-list":
        original = ollama_provider_module._string_tuple

        def controlled_string_tuple(value, context):
            if context == "Endpoint.dataCenterIds":
                raise carrier
            return original(value, context)

        monkeypatch.setattr(
            ollama_provider_module, "_string_tuple", controlled_string_tuple
        )
    else:

        def controlled_global_networking(_value):
            raise carrier

        monkeypatch.setattr(
            ollama_provider_module,
            "_pod_global_networking_enabled",
            controlled_global_networking,
        )

    observed = (
        provider.client.pod
        if mode is OllamaLeaseMode.DEDICATED_POD
        else provider.client.endpoint
    )
    validator = (
        ollama_provider_module._validate_pod_provisioning
        if mode is OllamaLeaseMode.DEDICATED_POD
        else ollama_provider_module._validate_endpoint_provisioning
    )
    assert observed is not None
    with pytest.raises(OllamaProvisioningEvidenceError) as raised:
        validator(
            observed,
            intent,
            provider_resource_id=resource.provider_resource_id,
            observed_at=intent.provision_requested_at,
        )

    assert raised.value is carrier
    assert raised.value.__cause__ is control


def _runtime_telemetry_response(prepared):
    return httpx.Response(
        200,
        json=runtime_telemetry_payload(
            prepared.runtime_telemetry_binding_sha256,
            prepared.intent.provision_requested_at.isoformat(),
        ),
    )


def test_workload_credentials_must_not_reuse_control_or_cross_product_keys():
    with pytest.raises(RunPodManagerError, match="differ from RUNPOD_API_KEY"):
        _provider(
            serverless_api_key=_SERVERLESS_TOKEN,
            control_plane_api_key=_SERVERLESS_TOKEN,
        )
    with pytest.raises(RunPodManagerError, match="must be distinct"):
        _provider(
            serverless_api_key=_SERVERLESS_TOKEN,
            pod_bearer_token=_SERVERLESS_TOKEN,
        )


@pytest.mark.asyncio
async def test_network_volume_cache_rejects_concurrent_serverless_writers(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    deployment = replace(
        _deployment(),
        profile=replace(_profile(), network_volume_id="volume-1"),
        serverless_workers_max=2,
        serverless_non_compute_cost=non_compute_cost_policy(
            include_network_volume=True
        ),
        pod_non_compute_cost=non_compute_cost_policy(include_network_volume=True),
    )
    provider = _provider(deployment=deployment)
    request = make_request(
        MutableClock(),
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    plan = await provider.plan(request)

    with pytest.raises(RunPodManagerError, match="exactly one Serverless worker"):
        prepared = provider.validate_runtime_request(
            request, plan, provision_requested_at=provider._clock()
        )
        await provider.provision(prepared)


@pytest.mark.asyncio
async def test_private_ollama_rejects_shared_serverless_gpu_pool(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    offer = _offer(ComputeProduct.SERVERLESS)
    client.list_gpus = lambda **_: (offer, replace(offer, id="other-gpu"))
    provider = _provider(client=client)
    request = make_request(
        MutableClock(), mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )

    with pytest.raises(RunPodManagerError, match="one exact catalog GPU"):
        await provider.plan(request)
    assert client.endpoint_request is None
    assert client.pod_request is None


@pytest.mark.asyncio
async def test_plan_uses_product_specific_live_catalog_prices(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    provider = _provider(client=client)

    plan = await provider.plan(make_request(MutableClock()))

    assert plan.mode is OllamaLeaseMode.DEDICATED_POD
    assert plan.placement.gpu_id == "gpu-pod"
    assert plan.estimated_cost == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_poolless_serverless_availability_fails_clearly_without_create(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()

    def poolless_mig(*, products, **kwargs):
        del kwargs
        assert products == (ComputeProduct.SERVERLESS,)
        return (
            replace(
                _offer(ComputeProduct.SERVERLESS),
                id=("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"),
                pool=None,
                secure=False,
                community=False,
                secure_max_count=0,
                community_max_count=0,
                availability=Availability.HIGH,
            ),
        )

    client.list_gpus = poolless_mig
    provider = _provider(client=client)
    request = make_request(
        MutableClock(), mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )

    with pytest.raises(
        RunPodManagerError,
        match="without the canonical pool ID.*refusing to guess",
    ):
        await provider.plan(request)

    assert client.endpoint_request is None
    assert client.pod_request is None


@pytest.mark.asyncio
async def test_pooled_serverless_offer_survives_unrelated_poolless_offer(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()

    def mixed_serverless(*, products, **kwargs):
        del kwargs
        assert products == (ComputeProduct.SERVERLESS,)
        pooled = _offer(ComputeProduct.SERVERLESS)
        return (
            replace(
                pooled,
                id=("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"),
                pool=None,
                availability=Availability.HIGH,
            ),
            pooled,
        )

    client.list_gpus = mixed_serverless
    provider = _provider(client=client)

    plan = await provider.plan(
        make_request(
            MutableClock(),
            mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            max_authorized_cost=5.0,
        )
    )

    assert plan.placement.gpu_id == "gpu-serverless"
    assert plan.placement.gpu_pool == "pool-24"
    assert client.endpoint_request is None
    assert client.pod_request is None


@pytest.mark.asyncio
async def test_serverless_provision_is_load_balanced_and_configuration_owned(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    monkeypatch.setenv("HOST_SECRET_THAT_MUST_NOT_EXPAND", "leaked-secret")
    client = _ControlClient()
    provider = _provider(client=client)
    request = make_request(
        MutableClock(),
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    plan = await provider.plan(request)

    prepared = provider.validate_runtime_request(
        request, plan, provision_requested_at=provider._clock()
    )
    resource = await provider.provision(prepared)

    assert resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
    assert client.endpoint_request.endpoint_type == "LOAD_BALANCER"
    assert client.endpoint_request.image == _TEST_IMAGE
    assert client.endpoint_request.env["KESTREL_OLLAMA_REQUIRED_MODEL"] == "qwen3:8b"
    assert (
        client.endpoint_request.env["KESTREL_OLLAMA_BEARER_TOKEN"] == _SERVERLESS_TOKEN
    )
    assert client.endpoint_request.env["HEALTH_CHECK_PATH"] == "/ping"
    assert client.endpoint_request.env["PORT"] == "11434"
    assert client.endpoint_request.env["PORT_HEALTH"] == "11434"
    assert client.endpoint_request.env["KESTREL_OLLAMA_MODEL_STORAGE_PATH"] == (
        "/models"
    )
    assert "leaked-secret" not in client.endpoint_request.env.values()
    assert client.endpoint_request.scaling["type"] == "REQUEST_COUNT"


@pytest.mark.asyncio
async def test_prepared_request_is_single_use_safe_and_mutation_detected(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    provider = _provider(client=client)
    clock = MutableClock()
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    plan = await provider.plan(request)
    prepared = provider.validate_runtime_request(
        request, plan, provision_requested_at=clock()
    )

    assert _SERVERLESS_TOKEN not in repr(prepared)
    assert _SERVERLESS_TOKEN not in repr(prepared.provider_request)
    assert _SERVERLESS_TOKEN not in prepared.intent.provider_request_json
    assert prepared.intent.provision_requested_at == clock()
    entered = threading.Event()
    resume = threading.Event()
    original_create = client.create_endpoint

    def delayed_create(provider_request):
        entered.set()
        assert resume.wait(timeout=5)
        return original_create(provider_request)

    client.create_endpoint = delayed_create
    provision = asyncio.create_task(provider.provision(prepared))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(TypeError):
            prepared.provider_request.env["KESTREL_OLLAMA_REQUIRED_MODEL"] = (
                "wrong:latest"
            )
        with pytest.raises(TypeError):
            prepared.provider_request.scaling["requestCount"] = 2
    finally:
        await release_gates_and_boundedly_join_owned_tasks(provision, resume)
    resource = await provision

    assert resource.provider_resource_id == "endpoint-1"
    assert client.endpoint_request is prepared.provider_request
    assert client.endpoint_request.env["KESTREL_OLLAMA_REQUIRED_MODEL"] == "qwen3:8b"
    assert client.endpoint_request.scaling["requestCount"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [OllamaLeaseMode.SERVERLESS_LOAD_BALANCER, OllamaLeaseMode.DEDICATED_POD]
)
async def test_ollama_creation_carries_the_accepted_cuda_floor(mode, monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    provider = _provider(client=client)
    value = make_request(MutableClock(), mode=mode, max_authorized_cost=5.0)
    value = replace(
        value, constraints=replace(value.constraints, min_cuda_version="12")
    )
    plan = await provider.plan(value)
    prepared = provider.validate_runtime_request(
        value, plan, provision_requested_at=provider._clock()
    )
    await provider.provision(prepared)
    created = client.endpoint_request or client.pod_request
    assert created.to_payload()["gpu"]["minCudaVersion"] == "12.0"


@pytest.mark.asyncio
async def test_ollama_refuses_unaccepted_global_storage_before_workload_requests(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    provider = _provider(client=client, transport=httpx.MockTransport(handler))
    resource, intent, receipt, _prepared = await _readiness_authority(
        provider, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    client.endpoint = replace(
        client.endpoint,
        raw={
            **client.endpoint.raw,
            "mounts": {
                "network": [],
                "global": [{"volumeId": "unquoted-volume"}],
            },
        },
    )
    with pytest.raises(OllamaProvisioningMismatchError) as mismatch:
        await provider.observe(resource, intent, receipt)
    assert (
        mismatch.value.evidence.reason
        is OllamaProvisioningMismatchReason.ENDPOINT_POLICY
    )
    assert mismatch.value.evidence.provider_resource_id == resource.provider_resource_id
    mismatch.value.evidence.validate_intent(intent)
    assert seen == []


@pytest.mark.asyncio
async def test_observe_without_authority_refuses_before_workload_probe(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, _intent, _receipt, _prepared = await _readiness_authority(
        provider, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )

    with pytest.raises(OllamaProvisioningEvidenceError):
        await provider.observe(resource, None, None)

    assert seen == []


@pytest.mark.asyncio
async def test_observe_requires_provider_health_and_reads_exact_model(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    requests = []
    authority = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {_SERVERLESS_TOKEN}"
        if request.url.path == "/ping":
            return httpx.Response(200)
        if request.url.path == "/kestrel/telemetry":
            return _runtime_telemetry_response(authority["prepared"])
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, prepared = await _readiness_authority(
        provider, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    authority["prepared"] = prepared

    observation = await provider.observe(resource, intent, receipt)

    assert observation.provider_ready is True
    assert observation.has_model("qwen3:8b")
    assert observation.runtime_telemetry is not None
    assert [request.url.path for request in requests] == [
        "/ping",
        "/api/tags",
        "/kestrel/telemetry",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "serverless-health",
        "serverless-models",
        "pod-anonymous",
        "pod-health",
        "telemetry-http",
        "telemetry-json",
        "telemetry-contract",
    ],
)
async def test_workload_probe_preserves_exact_execution_control(
    monkeypatch, boundary, record_property
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    mode = (
        OllamaLeaseMode.DEDICATED_POD
        if boundary.startswith("pod-")
        else OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    caught_type = (
        httpx.ConnectError
        if boundary
        in {
            "serverless-health",
            "serverless-models",
            "pod-anonymous",
            "pod-health",
            "telemetry-http",
        }
        else ValueError
    )
    carrier, control = _execution_control_carrier(
        caught_type(f"{boundary} outcome remains unknown")
    )
    requests: list[str] = []
    authority = {}
    telemetry_requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal telemetry_requested
        path = request.url.path
        requests.append(path)
        if path == "/kestrel/telemetry":
            telemetry_requested = True
        target = {
            "serverless-health": "/ping",
            "serverless-models": "/api/tags",
            "pod-anonymous": "/api/tags",
            "pod-health": "/ping",
            "telemetry-http": "/kestrel/telemetry",
        }.get(boundary)
        if target == path and (
            boundary != "pod-anonymous" or "Authorization" not in request.headers
        ):
            record_control_injection(record_property, boundary)
            raise carrier
        if (
            mode is OllamaLeaseMode.DEDICATED_POD
            and "Authorization" not in request.headers
        ):
            return httpx.Response(401)
        if path == "/ping":
            return httpx.Response(200)
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        if path == "/kestrel/telemetry":
            return _runtime_telemetry_response(authority["prepared"])
        raise AssertionError(path)

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, prepared = await _readiness_authority(provider, mode)
    authority["prepared"] = prepared
    if boundary == "telemetry-json":
        original_json = httpx.Response.json

        def controlled_json(response, **kwargs):
            if response.request.url.path == "/kestrel/telemetry":
                record_control_injection(record_property, boundary)
                raise carrier
            return original_json(response, **kwargs)

        monkeypatch.setattr(httpx.Response, "json", controlled_json)
    elif boundary == "telemetry-contract":
        original_clock = provider._clock

        def controlled_clock():
            if telemetry_requested:
                record_control_injection(record_property, boundary)
                raise carrier
            return original_clock()

        provider._clock = controlled_clock

    with pytest.raises(type(carrier)) as raised:
        await provider.observe(resource, intent, receipt)

    assert raised.value is carrier
    assert carrier.__cause__ is control
    assert (
        requests
        == {
            "serverless-health": ["/ping"],
            "serverless-models": ["/ping", "/api/tags"],
            "pod-anonymous": ["/api/tags"],
            "pod-health": ["/api/tags", "/ping"],
            "telemetry-http": ["/ping", "/api/tags", "/kestrel/telemetry"],
            "telemetry-json": ["/ping", "/api/tags", "/kestrel/telemetry"],
            "telemetry-contract": ["/ping", "/api/tags", "/kestrel/telemetry"],
        }[boundary]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "serverless-health",
        "serverless-models",
        "pod-anonymous",
        "pod-health",
        "telemetry-http",
        "telemetry-json",
        "telemetry-contract",
    ],
)
async def test_workload_probe_retains_ordinary_normalization_contract(
    monkeypatch, boundary
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    mode = (
        OllamaLeaseMode.DEDICATED_POD
        if boundary.startswith("pod-")
        else OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    requests = []
    authority = {}
    telemetry_requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal telemetry_requested
        path = request.url.path
        requests.append(path)
        if path == "/kestrel/telemetry":
            telemetry_requested = True
        target = {
            "serverless-health": "/ping",
            "serverless-models": "/api/tags",
            "pod-anonymous": "/api/tags",
            "pod-health": "/ping",
            "telemetry-http": "/kestrel/telemetry",
        }.get(boundary)
        if target == path and (
            boundary != "pod-anonymous" or "Authorization" not in request.headers
        ):
            raise httpx.ConnectError(f"ordinary {boundary} failure")
        if (
            mode is OllamaLeaseMode.DEDICATED_POD
            and "Authorization" not in request.headers
        ):
            return httpx.Response(401)
        if path == "/ping":
            return httpx.Response(200)
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        if path == "/kestrel/telemetry":
            return _runtime_telemetry_response(authority["prepared"])
        raise AssertionError(path)

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, prepared = await _readiness_authority(provider, mode)
    authority["prepared"] = prepared
    if boundary == "telemetry-json":
        original_json = httpx.Response.json

        def invalid_json(response, **kwargs):
            if response.request.url.path == "/kestrel/telemetry":
                raise ValueError("ordinary invalid telemetry JSON")
            return original_json(response, **kwargs)

        monkeypatch.setattr(httpx.Response, "json", invalid_json)
    elif boundary == "telemetry-contract":
        original_clock = provider._clock

        def invalid_clock():
            if telemetry_requested:
                raise ValueError("ordinary telemetry observation clock failure")
            return original_clock()

        provider._clock = invalid_clock

    if boundary.startswith("telemetry-"):
        with pytest.raises(RunPodManagerError) as raised:
            await provider.observe(resource, intent, receipt)
        assert type(raised.value.__cause__) in {
            httpx.ConnectError,
            ValueError,
        }
    else:
        observation = await provider.observe(resource, intent, receipt)
        if boundary == "serverless-health":
            assert observation.provider_ready is False
            assert observation.model_names == ("qwen3:8b",)
        elif boundary == "serverless-models":
            assert observation.provider_ready is True
            assert observation.model_names == ()
        else:
            assert observation.provider_ready is False
            assert observation.model_names == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["delete", "readback"])
async def test_teardown_auth_translation_preserves_exact_execution_control(
    phase, record_property
):
    client = _ControlClient()
    carrier, control = _execution_control_carrier(
        RunPodAPIError(
            title="authorization",
            detail="cleanup authorization remains unknown",
            status_code=401,
        )
    )
    if phase == "delete":
        client.error_callback = lambda _phase: record_control_injection(
            record_property, f"teardown-{phase}"
        )
        client.delete_error = carrier
    else:

        def controlled_readback(_endpoint_id):
            record_control_injection(record_property, f"teardown-{phase}")
            raise carrier

        client.get_endpoint = controlled_readback
    provider = _provider(client=client)
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="resource",
    )

    with pytest.raises(RunPodAPIError) as raised:
        await provider.teardown(resource, _cleanup_intent(resource))

    assert raised.value is carrier
    assert carrier.__cause__ is control
    assert client.actions == ([] if phase == "delete" else [("endpoint-1", "delete")])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "status_code"),
    [
        pytest.param("delete", 401, id="delete-401"),
        pytest.param("delete", 403, id="delete-403"),
        pytest.param("readback", 401, id="readback-401"),
        pytest.param("readback", 403, id="readback-403"),
    ],
)
async def test_teardown_auth_translation_retains_ordinary_contract(phase, status_code):
    client = _ControlClient()
    error = RunPodAPIError(
        title="authorization",
        detail="cleanup authorization was rejected",
        status_code=status_code,
    )
    if phase == "delete":
        client.delete_error = error
    else:

        def rejected_readback(_endpoint_id):
            raise error

        client.get_endpoint = rejected_readback
    provider = _provider(client=client)
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="resource",
    )

    with pytest.raises(OllamaLeaseAuthorizationError) as raised:
        await provider.teardown(resource, _cleanup_intent(resource))

    assert raised.value.__cause__ is error
    assert client.actions == ([] if phase == "delete" else [("endpoint-1", "delete")])


@pytest.mark.asyncio
async def test_teardown_retryable_control_graph_is_not_reclassified():
    client = _ControlClient()
    carrier, control = _execution_control_carrier(
        RunPodAPIError(
            title="unavailable",
            detail="cleanup outcome remains unknown",
            status_code=503,
        )
    )
    client.delete_error = carrier
    provider = _provider(client=client)
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="resource",
    )

    with pytest.raises(RunPodAPIError) as raised:
        await provider.teardown(resource, _cleanup_intent(resource))

    assert raised.value is carrier
    assert carrier.__cause__ is control
    assert client.actions == []


@pytest.mark.asyncio
async def test_pod_observation_uses_v2_status_and_runtime_route(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    authority = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "Authorization" not in request.headers:
            return httpx.Response(401)
        assert request.headers["Authorization"] == f"Bearer {_POD_TOKEN}"
        if request.url.path == "/ping":
            return httpx.Response(200)
        if request.url.path == "/kestrel/telemetry":
            return _runtime_telemetry_response(authority["prepared"])
        return httpx.Response(200, json={"models": []})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, prepared = await _readiness_authority(
        provider, OllamaLeaseMode.DEDICATED_POD
    )
    authority["prepared"] = prepared

    observation = await provider.observe(resource, intent, receipt)

    assert observation.provider_ready is True
    assert observation.route_url == "https://pod-evidence-1-11434.proxy.runpod.net"
    assert observation.runtime_telemetry is not None


@pytest.mark.asyncio
async def test_pod_initializing_health_is_not_ready(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)

    def handler(request: httpx.Request) -> httpx.Response:
        if "Authorization" not in request.headers:
            return httpx.Response(401)
        if request.url.path == "/ping":
            return httpx.Response(204)
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, _prepared = await _readiness_authority(
        provider, OllamaLeaseMode.DEDICATED_POD
    )
    observation = await provider.observe(resource, intent, receipt)

    assert observation.provider_ready is False
    assert observation.model_names == ()


@pytest.mark.asyncio
async def test_pod_observation_fails_closed_when_route_allows_anonymous_access(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, _prepared = await _readiness_authority(
        provider, OllamaLeaseMode.DEDICATED_POD
    )

    observation = await provider.observe(resource, intent, receipt)

    assert observation.provider_ready is False
    assert observation.model_names == ()


def test_tcp_pod_route_requires_tls_before_bearer_token_can_be_sent():
    runtime = {
        "runtime": {
            "ports": [
                {"private": 11434, "type": "tcp", "ip": "203.0.113.8", "public": 1}
            ]
        }
    }

    assert _pod_base_url("pod-1", runtime, 11434, "http") is None
    assert _pod_base_url("pod-1", runtime, 11434, "https") == "https://203.0.113.8:1"


@pytest.mark.asyncio
async def test_model_pull_is_owned_by_the_digest_verifying_runtime():
    provider = _provider()
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.POD,
        provider_resource_id="pod-1",
        resource_name="resource",
    )

    with pytest.raises(RunPodManagerError, match="runtime owns model pulls"):
        await provider.pull_model(resource, "https://pod.example", "qwen3:8b")


@pytest.mark.asyncio
async def test_serverless_initializing_health_is_not_ready(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ping":
            return httpx.Response(204)
        return httpx.Response(200, json={"models": []})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource, intent, receipt, _prepared = await _readiness_authority(
        provider, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    )
    observation = await provider.observe(resource, intent, receipt)

    assert observation.provider_ready is False


@pytest.mark.asyncio
async def test_teardown_uses_delete_for_endpoint_and_terminate_for_pod():
    client = _ControlClient()
    provider = _provider(client=client)

    endpoint = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="endpoint",
    )
    await provider.teardown(endpoint, _cleanup_intent(endpoint))
    pod = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.POD,
        provider_resource_id="pod-1",
        resource_name="pod",
    )
    await provider.teardown(pod, _cleanup_intent(pod))

    assert client.actions == [("endpoint-1", "delete"), ("pod-1", "terminate")]


@pytest.mark.asyncio
async def test_teardown_treats_already_absent_resource_as_success():
    client = _ControlClient()
    client.delete_error = RunPodAPIError(
        title="Not found", detail="Endpoint is absent", status_code=404
    )
    provider = _provider(client=client)

    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="endpoint",
    )
    await provider.teardown(resource, _cleanup_intent(resource))


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", list(OllamaResourceType))
@pytest.mark.parametrize("failure_stage", ["delete", "readback"])
@pytest.mark.parametrize("unknown_subclass", [False, True])
@pytest.mark.parametrize("structural_cause", [False, True])
async def test_only_canonical_404_can_prove_exact_resource_absence(
    monkeypatch,
    resource_type,
    failure_stage,
    unknown_subclass,
    structural_cause,
):
    class UnknownNotFoundError(RunPodAPIError):
        pass

    client = _ControlClient()
    resource = ProvisionedOllamaResource(
        resource_type=resource_type,
        provider_resource_id="endpoint-1"
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else "pod-1",
        resource_name="resource",
    )
    error_type = UnknownNotFoundError if unknown_subclass else RunPodAPIError
    error = error_type(
        title="Not found",
        detail="exact resource is absent",
        status_code=404,
    )
    if structural_cause:
        error.__cause__ = ValueError("structural parser failure")
    readback_ids = []
    original_get_endpoint = client.get_endpoint
    original_get_pod = client.get_pod

    def clean_readback(resource_id):
        readback_ids.append(resource_id)
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            return original_get_endpoint(resource_id)
        return original_get_pod(resource_id)

    def fail_delete(resource_id):
        client.absent.add(resource_id)
        raise error

    def fail_action(resource_id, action):
        assert action == "terminate"
        client.absent.add(resource_id)
        raise error

    def fail_readback(resource_id):
        readback_ids.append(resource_id)
        raise error

    if failure_stage == "delete":
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            monkeypatch.setattr(client, "delete_endpoint", fail_delete)
            monkeypatch.setattr(client, "get_endpoint", clean_readback)
        else:
            monkeypatch.setattr(client, "pod_action", fail_action)
            monkeypatch.setattr(client, "get_pod", clean_readback)
    elif resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
        monkeypatch.setattr(client, "get_endpoint", fail_readback)
    else:
        monkeypatch.setattr(client, "get_pod", fail_readback)

    provider = _provider(client=client)
    if unknown_subclass or structural_cause:
        with pytest.raises(error_type) as caught:
            await provider.teardown(resource, _cleanup_intent(resource))
        assert caught.value is error
        if failure_stage == "delete":
            assert readback_ids == []
    else:
        receipt = await provider.teardown(resource, _cleanup_intent(resource))
        assert receipt.verified_absent_at is not None
        assert readback_ids == [resource.provider_resource_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource_type", "error_type"),
    [
        pytest.param(
            OllamaResourceType.SERVERLESS_ENDPOINT,
            ValueError,
            id="endpoint-value",
        ),
        pytest.param(OllamaResourceType.POD, RecursionError, id="pod-recursion"),
    ],
)
async def test_native_absence_problem_decoder_control_is_not_404(
    monkeypatch,
    native_teardown,
    resource_type,
    error_type,
):
    provider, state = native_teardown
    carrier, control, explicit = execution_control_carrier(
        error_type("problem decoder execution control"),
        grouped=True,
    )
    structural = explicit.exceptions[1]
    original_json = httpx.Response.json

    def controlled_json(response, **kwargs):
        if response.status_code == 404:
            raise carrier
        return original_json(response, **kwargs)

    monkeypatch.setattr(httpx.Response, "json", controlled_json)
    resource = ProvisionedOllamaResource(resource_type, "original-1", "original")

    with pytest.raises(error_type) as caught:
        await provider.teardown(resource, _cleanup_intent(resource))

    assert caught.value is carrier
    assert carrier.__cause__ is explicit
    assert explicit.exceptions == (control, structural)
    path = (
        "/v2/serverless/original-1"
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else "/v2/pods/original-1"
    )
    action = (
        ("DELETE", path)
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else ("POST", path + "/action")
    )
    assert state["calls"] == [action, ("GET", path)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource_type",
    [
        pytest.param(
            OllamaResourceType.SERVERLESS_ENDPOINT,
            id="serverless-endpoint",
        ),
        pytest.param(OllamaResourceType.POD, id="pod"),
    ],
)
async def test_native_absence_ordinary_malformed_problem_remains_canonical_404(
    monkeypatch,
    native_teardown,
    resource_type,
):
    provider, state = native_teardown
    original_json = httpx.Response.json

    def malformed_json(response, **kwargs):
        if response.status_code == 404:
            raise ValueError("ordinary malformed problem JSON")
        return original_json(response, **kwargs)

    monkeypatch.setattr(httpx.Response, "json", malformed_json)
    resource = ProvisionedOllamaResource(resource_type, "original-1", "original")
    intent = _cleanup_intent(resource)

    receipt = await provider.teardown(resource, intent)

    assert receipt.cleanup_intent_id == intent.cleanup_intent_id
    receipt.validate_intent(intent)
    assert receipt.verified_absent_at is not None
    assert receipt.basis is OllamaCleanupBasis.DELETE_ACKNOWLEDGED
    path = (
        "/v2/serverless/original-1"
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else "/v2/pods/original-1"
    )
    action = (
        ("DELETE", path)
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else ("POST", path + "/action")
    )
    assert state["calls"] == [action, ("GET", path)]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["typed_transport", "statusless", "malformed"])
async def test_teardown_retry_disposition_requires_canonical_transport_evidence(case):
    client = _ControlClient()
    api_error = RunPodAPIError(
        title="Runpod transport failed" if case != "malformed" else "Invalid response",
        detail="bounded test",
        status_code=200 if case == "malformed" else None,
    )
    if case == "typed_transport":
        api_error.__cause__ = native_httpx_connect_failure()
    elif case == "malformed":
        api_error.__cause__ = ValueError("invalid JSON")
    client.delete_error = api_error
    provider = _provider(client=client)
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="endpoint",
    )

    if case == "typed_transport":
        with pytest.raises(OllamaLeaseTeardownError) as caught:
            await provider.teardown(resource, _cleanup_intent(resource))
        assert caught.value.__cause__ is api_error
    else:
        with pytest.raises(RunPodAPIError) as caught:
            await provider.teardown(resource, _cleanup_intent(resource))
        assert caught.value is api_error


@pytest.fixture
def native_teardown():
    state = {"calls": [], "delete_status": 204, "read_status": 404}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, *, action):
            assert self.headers["Authorization"] == "Bearer local-loopback-only"
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["calls"].append((self.command, self.path))
            status = state["delete_status"] if action else state["read_status"]
            body = (
                b""
                if status == 204
                else json.dumps(
                    {
                        "id": "original-1",
                        "name": "original",
                        "status": "RUNNING",
                        "type": "LOAD_BALANCER",
                        "gpu": {"id": "gpu-1", "count": 1},
                        "cost": 0.5,
                        "cloud": "SECURE",
                        "dataCenterId": "US-TX-3",
                    }
                ).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            self.respond(action=True)

        def do_POST(self):
            self.respond(action=True)

        def do_GET(self):
            self.respond(action=False)

    owner = OwnedLoopbackHTTPServer(
        server_factory=lambda: ThreadingHTTPServer(("127.0.0.1", 0), Handler),
        thread_factory=lambda server: threading.Thread(target=server.serve_forever),
        client_factory=lambda server: RunpodControlPlaneClient(
            api_key="local-loopback-only",
            base_url=f"http://127.0.0.1:{server.server_port}/v2",
            max_safe_retries=0,
        ),
    )
    with owner as client:
        yield _provider(client=client), state


@pytest.fixture
def native_provisioning():
    state = {
        "calls": [],
        "billing_calls": [],
        "request": None,
        "readback": None,
        "drop_on_get": (),
        "replace_on_get": {},
        "replace_on_post": {},
        "present": False,
        "pod_cost": 0.5,
        "read_status": None,
        "ambiguous_post": False,
        "inventory_visible": True,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, payload):
            body = b"" if status == 204 else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer local-loopback-only"
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            state["calls"].append((self.command, self.path))
            if self.path == "/v2/pods/pod-evidence-1/action":
                assert payload == {"action": "terminate"}
                state["present"] = False
                self.respond(204, {})
                return
            assert self.path in {"/v2/serverless", "/v2/pods"}
            state["request"] = payload
            state["present"] = True
            if self.path == "/v2/serverless":
                state["readback"] = {
                    **payload,
                    "args": payload.get("args", ""),
                    "id": "endpoint-evidence-1",
                    "createdAt": "2026-08-01T12:00:00Z",
                    "requestUrls": {
                        "base": "https://endpoint.example",
                        "health": "https://endpoint.example/ping",
                    },
                }
                state["readback"]["gpu"].update(
                    {
                        "allowedCudaVersions": [],
                        "excludedTypes": [],
                        "minCudaVersion": payload["gpu"].get("minCudaVersion"),
                    }
                )
            else:
                state["readback"] = {
                    **payload,
                    "args": payload.get("args", ""),
                    "mounts": payload.get("mounts", {}),
                    "globalNetworking": {
                        "enabled": payload.get("globalNetworking", False)
                    },
                    "id": "pod-evidence-1",
                    "status": "RUNNING",
                    "actions": ["stop", "restart", "terminate"],
                    "cost": state["pod_cost"],
                    "dataCenterId": "US-TX-3",
                    "cudaVersion": "12.0",
                    "registry": payload.get("registry"),
                    "ssh": {"proxy": None, "direct": None},
                    "template": None,
                    "locked": False,
                    "runtime": None,
                    "createdAt": "2026-08-01T12:00:00Z",
                    "startedAt": "2026-08-01T12:00:00Z",
                }
                state["readback"]["gpu"].update({"vcpuCount": 4, "memory": 24})
            state["readback"].update(state["replace_on_post"])
            if state["ambiguous_post"]:
                self.respond(503, {"title": "create response unavailable"})
            else:
                self.respond(201, state["readback"])

        def do_GET(self):
            path = urlsplit(self.path)
            if path.path == "/v2/billing/serverless":
                query = {key: values[0] for key, values in parse_qs(path.query).items()}
                state["calls"].append((self.command, path.path))
                state["billing_calls"].append(query)
                start = datetime.fromisoformat(query["startTime"])
                end = datetime.fromisoformat(query["endTime"])
                records = []
                while start < end:
                    records.append(
                        {
                            "startTime": start.isoformat().replace("+00:00", "Z"),
                            "endTime": (start + timedelta(hours=1))
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "serverlessId": "endpoint-evidence-1",
                            "gpuAmount": 0.04,
                            "cpuAmount": 0.01,
                            "diskAmount": 0.02,
                            "feeAmount": 0.03,
                            "totalAmount": 0.10,
                        }
                    )
                    start += timedelta(hours=1)
                totals = {
                    key: float(
                        sum(
                            (Decimal(str(record[key])) for record in records),
                            Decimal(0),
                        )
                    )
                    for key in (
                        "gpuAmount",
                        "cpuAmount",
                        "diskAmount",
                        "feeAmount",
                        "totalAmount",
                    )
                }
                self.respond(
                    200,
                    {
                        "records": records,
                        "metadata": {
                            "query": query,
                            "recordCount": len(records),
                            "uniqueServerlessCount": 1 if records else 0,
                            "totals": totals,
                        },
                    },
                )
                return
            if self.path == "/v2/serverless":
                state["calls"].append((self.command, self.path))
                self.respond(
                    200,
                    {
                        "endpoints": (
                            [state["readback"]]
                            if state["present"] and state["inventory_visible"]
                            else []
                        ),
                        "pagination": {"hasNextPage": False, "nextCursor": None},
                    },
                )
                return
            assert self.path in {
                "/v2/serverless/endpoint-evidence-1",
                "/v2/pods/pod-evidence-1",
            }
            state["calls"].append((self.command, self.path))
            if state["read_status"] is not None:
                self.respond(state["read_status"], {"title": "readback unavailable"})
                return
            if not state["present"]:
                self.respond(404, {"title": "not found"})
                return
            payload = dict(state["readback"])
            for key in state["drop_on_get"]:
                payload.pop(key, None)
            payload.update(state["replace_on_get"])
            self.respond(200, payload)

        def do_DELETE(self):
            assert self.path == "/v2/serverless/endpoint-evidence-1"
            state["calls"].append((self.command, self.path))
            state["present"] = False
            self.respond(204, {})

    owner = OwnedLoopbackHTTPServer(
        server_factory=lambda: ThreadingHTTPServer(("127.0.0.1", 0), Handler),
        thread_factory=lambda server: threading.Thread(target=server.serve_forever),
        client_factory=lambda server: RunpodControlPlaneClient(
            api_key="local-loopback-only",
            base_url=f"http://127.0.0.1:{server.server_port}/v2",
            max_safe_retries=0,
        ),
    )
    with owner as client:
        yield _provider(client=client), state


def _native_loopback_fixture_ownership_child(
    fixture_name,
    failure_stage,
    result_queue,
):
    """Exercise an actual fixture, then finitely rescue only captured resources."""

    fixture_namespace = (
        native_pod_action_harness.__wrapped__.__globals__
        if fixture_name == "pod-action"
        else globals()
    )
    real_threading = fixture_namespace["threading"]
    real_server_type = fixture_namespace["ThreadingHTTPServer"]
    real_client_type = fixture_namespace["RunpodControlPlaneClient"]
    captured = {
        "server": None,
        "server_shutdown": None,
        "server_close": None,
        "thread": None,
        "thread_join": None,
        "client": None,
        "client_close": None,
    }
    explicit_cause = ValueError("native fixture injected explicit cause")
    injected = RuntimeError(
        f"native fixture {fixture_name} failed during {failure_stage}"
    )
    injected.__cause__ = explicit_cause
    monkeypatch = pytest.MonkeyPatch()

    def server_factory(*args, **kwargs):
        server = real_server_type(*args, **kwargs)
        captured["server"] = server
        captured["server_shutdown"] = server.shutdown
        captured["server_close"] = server.server_close
        if failure_stage == "server-shutdown":

            def fail_shutdown():
                captured["server_shutdown"]()
                raise injected

            server.shutdown = fail_shutdown
        return server

    def thread_factory(*args, **kwargs):
        thread = real_threading.Thread(*args, **kwargs)
        captured["thread"] = thread
        captured["thread_join"] = thread.join
        if failure_stage == "thread-start":
            original_start = thread.start

            def fail_start():
                original_start()
                raise injected

            thread.start = fail_start
        elif failure_stage == "thread-join":

            def fail_join(*join_args, **join_kwargs):
                captured["thread_join"](*join_args, **join_kwargs)
                raise injected

            thread.join = fail_join
        return thread

    def client_factory(*args, **kwargs):
        if failure_stage == "client-constructor":
            raise injected
        client = real_client_type(*args, **kwargs)
        captured["client"] = client
        captured["client_close"] = client.close
        if failure_stage == "client-close":

            def fail_close():
                captured["client_close"]()
                raise injected

            client.close = fail_close
        return client

    def resource_facts():
        server = captured["server"]
        thread = captured["thread"]
        client = captured["client"]
        server_open = server is not None and server.fileno() != -1
        thread_alive = thread is not None and thread.is_alive()
        client_open = client is not None and not client.transport._client.is_closed
        return {
            "server_open": server_open,
            "thread_alive": thread_alive,
            "client_present": client is not None,
            "client_open": client_open,
            "leaked": server_open or thread_alive or client_open,
        }

    def invoke_actual_fixture():
        if fixture_name == "pod-action":
            with native_pod_action_harness():
                pass
            return
        fixture = (
            native_teardown
            if fixture_name == "native-teardown"
            else native_provisioning
        )
        generator = fixture.__wrapped__()
        try:
            next(generator)
            try:
                next(generator)
            except StopIteration:
                pass
        finally:
            generator.close()

    monkeypatch.setitem(fixture_namespace, "ThreadingHTTPServer", server_factory)
    monkeypatch.setitem(
        fixture_namespace,
        "threading",
        SimpleNamespace(Thread=thread_factory),
    )
    monkeypatch.setitem(fixture_namespace, "RunpodControlPlaneClient", client_factory)
    if failure_stage == "provider-constructor":
        if fixture_name == "pod-action":

            def fail_provider(*_args, **_kwargs):
                raise injected

            monkeypatch.setitem(
                fixture_namespace,
                "DirectRunPodProvider",
                fail_provider,
            )
        else:

            def fail_provider(**_kwargs):
                raise injected

            monkeypatch.setattr(sys.modules[__name__], "_provider", fail_provider)

    caught = None
    pre_rescue = None
    try:
        try:
            invoke_actual_fixture()
        except BaseException as error:  # noqa: BLE001 - assert exact injected graph
            caught = error
        pre_rescue = resource_facts()
        assert caught is injected
        assert caught.__cause__ is explicit_cause
        assert caught.__traceback__ is not None
    finally:
        try:
            client = captured["client"]
            if client is not None and not client.transport._client.is_closed:
                captured["client_close"]()
        finally:
            try:
                thread = captured["thread"]
                if thread is not None and thread.is_alive():
                    captured["server_shutdown"]()
            finally:
                try:
                    server = captured["server"]
                    if server is not None and server.fileno() != -1:
                        captured["server_close"]()
                finally:
                    thread = captured["thread"]
                    if thread is not None:
                        captured["thread_join"](
                            timeout=5,
                        )
        monkeypatch.undo()

    post_rescue = resource_facts()
    assert pre_rescue is not None
    assert not post_rescue["server_open"]
    assert not post_rescue["thread_alive"]
    assert not post_rescue["client_open"]
    result_queue.put(
        {
            "fixture": fixture_name,
            "failure_stage": failure_stage,
            "pre_rescue": pre_rescue,
            "post_rescue": post_rescue,
            "exact_error": True,
        }
    )


def test_owned_loopback_server_cleans_early_client_constructor_failure():
    events = []
    opening, control, explicit = execution_control_carrier(
        RuntimeError("client constructor failed"),
        grouped=True,
    )

    class Server:
        server_port = 1

        def serve_forever(self):
            raise AssertionError("fake server target is not executed")

        def shutdown(self):
            events.append("server-shutdown")

        def server_close(self):
            events.append("server-close")

    class Thread:
        ident = None

        def __init__(self):
            self.alive = False

        def start(self):
            self.ident = 1
            self.alive = True
            events.append("thread-start")

        def is_alive(self):
            return self.alive

        def join(self, *, timeout):
            assert timeout == 5
            events.append("thread-join")
            self.alive = False

    def client_factory(_server):
        raise opening

    owner = OwnedLoopbackHTTPServer(
        server_factory=Server,
        thread_factory=lambda _server: Thread(),
        client_factory=client_factory,
    )

    with pytest.raises(RuntimeError) as caught, owner:
        raise AssertionError("client construction unexpectedly succeeded")

    assert caught.value is opening
    assert opening.__cause__ is explicit
    assert explicit.exceptions[0] is control
    assert events == [
        "thread-start",
        "server-shutdown",
        "thread-join",
        "server-close",
    ]
    assert owner.client is None
    assert owner.cleanup_errors == ()


@pytest.mark.parametrize(
    "body_type",
    [RuntimeError, GeneratorExit],
    ids=["ordinary-control", "generator-close-control"],
)
def test_owned_loopback_server_retains_body_and_every_cleanup_failure(body_type):
    from kestrel_sovereign.execution_custody import is_execution_control_error

    events = []
    body, control, explicit = execution_control_carrier(
        body_type("body failed"),
        grouped=True,
    )
    client_error = ValueError("client close failed")
    shutdown_error = RuntimeError("server shutdown failed")
    join_error = OSError("thread join failed")
    close_error = RuntimeError("server close failed")

    class Server:
        def serve_forever(self):
            raise AssertionError("fake server target is not executed")

        def shutdown(self):
            events.append("server-shutdown")
            raise shutdown_error

        def server_close(self):
            events.append("server-close")
            raise close_error

    class Thread:
        ident = 1

        def __init__(self):
            self.alive = False

        def start(self):
            self.alive = True
            events.append("thread-start")

        def is_alive(self):
            return self.alive

        def join(self, *, timeout):
            assert timeout == 5
            events.append("thread-join")
            self.alive = False
            raise join_error

    class Client:
        def close(self):
            events.append("client-close")
            raise client_error

    owner = OwnedLoopbackHTTPServer(
        server_factory=Server,
        thread_factory=lambda _server: Thread(),
        client_factory=lambda _server: Client(),
    )

    with pytest.raises(BaseExceptionGroup) as caught, owner:
        raise body

    assert is_execution_control_error(caught.value)
    assert caught.value.exceptions == (
        body,
        client_error,
        shutdown_error,
        join_error,
        close_error,
    )
    assert body.__cause__ is explicit
    assert explicit.exceptions[0] is control
    assert owner.cleanup_errors == (
        client_error,
        shutdown_error,
        join_error,
        close_error,
    )
    assert events == [
        "thread-start",
        "client-close",
        "server-shutdown",
        "thread-join",
        "server-close",
    ]


def test_owned_loopback_server_reports_an_unsettled_thread_after_all_cleanup():
    events = []

    class Server:
        def serve_forever(self):
            raise AssertionError("fake server target is not executed")

        def shutdown(self):
            events.append("server-shutdown")

        def server_close(self):
            events.append("server-close")

    class Thread:
        ident = 1

        def start(self):
            events.append("thread-start")

        def is_alive(self):
            return True

        def join(self, *, timeout):
            assert timeout == 5
            events.append("thread-join")

    class Client:
        def close(self):
            events.append("client-close")

    owner = OwnedLoopbackHTTPServer(
        server_factory=Server,
        thread_factory=lambda _server: Thread(),
        client_factory=lambda _server: Client(),
    )

    with (
        pytest.raises(
            AssertionError,
            match="owned loopback server thread did not terminate",
        ),
        owner,
    ):
        pass

    assert events == [
        "thread-start",
        "client-close",
        "server-shutdown",
        "thread-join",
        "server-close",
    ]
    assert len(owner.cleanup_errors) == 1


@pytest.mark.parametrize(
    ("fixture_name", "failure_stage"),
    [
        pytest.param("pod-action", "thread-start", id="pod-action-thread-start"),
        pytest.param(
            "pod-action", "client-constructor", id="pod-action-client-constructor"
        ),
        pytest.param(
            "pod-action", "provider-constructor", id="pod-action-provider-constructor"
        ),
        pytest.param("pod-action", "client-close", id="pod-action-client-close"),
        pytest.param(
            "native-teardown",
            "client-constructor",
            id="native-teardown-client-constructor",
        ),
        pytest.param(
            "native-teardown",
            "provider-constructor",
            id="native-teardown-provider-constructor-control",
        ),
        pytest.param(
            "native-teardown", "client-close", id="native-teardown-client-close"
        ),
        pytest.param(
            "native-teardown",
            "server-shutdown",
            id="native-teardown-server-shutdown",
        ),
        pytest.param(
            "native-provisioning",
            "client-constructor",
            id="native-provisioning-client-constructor",
        ),
        pytest.param(
            "native-provisioning",
            "provider-constructor",
            id="native-provisioning-provider-constructor-control",
        ),
        pytest.param(
            "native-provisioning",
            "client-close",
            id="native-provisioning-client-close",
        ),
        pytest.param(
            "native-provisioning",
            "thread-join",
            id="native-provisioning-thread-join",
        ),
    ],
)
def test_native_loopback_fixture_failure_owns_every_started_resource(
    fixture_name,
    failure_stage,
):
    context = get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_native_loopback_fixture_ownership_child,
        args=(fixture_name, failure_stage, result_queue),
    )
    ownership = OwnedSpawnProcesses(
        (process,),
        gates=(),
        queues=(result_queue,),
    )
    with ownership:
        result = result_queue.get(timeout=15)
        ownership.join()

    assert ownership.final_exitcodes == [0]
    assert ownership.forced_cleanup == []
    assert ownership.cleanup_failures == []
    assert ownership.closed_processes == 1
    assert result["fixture"] == fixture_name
    assert result["failure_stage"] == failure_stage
    assert result["exact_error"] is True
    assert result["post_rescue"]["leaked"] is False
    if failure_stage == "client-constructor":
        assert result["pre_rescue"]["client_present"] is False
    assert result["pre_rescue"]["leaked"] is False


def _native_readiness_provider(fixture_provider, state, clock):
    state["workload_calls"] = []

    def handler(request: httpx.Request) -> httpx.Response:
        state["workload_calls"].append(
            (request.url.path, "Authorization" in request.headers)
        )
        if "Authorization" not in request.headers:
            return httpx.Response(401)
        if request.url.path == "/ping":
            return httpx.Response(200)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        if request.url.path == "/kestrel/telemetry":
            environment = state["request"]["env"]
            return httpx.Response(
                200,
                json=runtime_telemetry_payload(
                    environment["KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256"],
                    environment["KESTREL_OLLAMA_PROVISION_REQUESTED_AT"],
                ),
            )
        return httpx.Response(404)

    provider = _provider(
        client=fixture_provider.client,
        deployment=fixture_provider.deployment,
        transport=httpx.MockTransport(handler),
    )
    provider._clock = clock
    return provider


def _apply_readiness_policy_drift(state, mode):
    if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER:
        workers = dict(state["readback"]["workers"])
        workers.update({"min": 4, "max": 4})
        state["replace_on_get"] = {"workers": workers}
        return
    gpu = dict(state["readback"]["gpu"])
    gpu["count"] = 2
    state["replace_on_get"] = {"gpu": gpu}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(OllamaLeaseMode.SERVERLESS_LOAD_BALANCER, id="endpoint"),
        pytest.param(OllamaLeaseMode.DEDICATED_POD, id="pod"),
    ],
)
async def test_readiness_revalidates_original_policy_before_workload_probe(
    native_provisioning, tmp_path, monkeypatch, mode
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    fixture_provider, state = native_provisioning
    clock = MutableClock()
    provider = _native_readiness_provider(fixture_provider, state, clock)
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / f"readiness-policy-{mode.value}.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=mode, max_authorized_cost=5.0)
    plan = (
        serverless_plan()
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
        else pod_plan(rate=0.5)
    )
    waiting = await service.acquire(request, plan=plan, wait_until_ready=False)
    intent = waiting.provisioning_intent
    receipt = waiting.provisioning_receipt
    assert intent is not None and receipt is not None
    resource = ProvisionedOllamaResource(
        intent.resource_type,
        receipt.provider_resource_id,
        intent.resource_name,
    )
    _apply_readiness_policy_drift(state, mode)

    with pytest.raises(OllamaProvisioningMismatchError):
        await provider.observe(resource, intent, receipt)

    assert state["workload_calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "invalid_evidence"),
    [
        pytest.param(
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            {"workers": {"min": 0, "max": 1}},
            id="endpoint",
        ),
        pytest.param(
            OllamaLeaseMode.DEDICATED_POD,
            {"gpu": None},
            id="pod",
        ),
    ],
)
async def test_readiness_invalid_policy_evidence_stays_unavailable_not_mismatch(
    native_provisioning,
    tmp_path,
    monkeypatch,
    mode,
    invalid_evidence,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    fixture_provider, state = native_provisioning
    clock = MutableClock()
    provider = _native_readiness_provider(fixture_provider, state, clock)
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / f"readiness-unavailable-{mode.value}.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=mode, max_authorized_cost=5.0)
    plan = (
        serverless_plan()
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
        else pod_plan(rate=0.5)
    )
    waiting = await service.acquire(request, plan=plan, wait_until_ready=False)
    intent = waiting.provisioning_intent
    receipt = waiting.provisioning_receipt
    assert intent is not None and receipt is not None
    state["replace_on_get"] = invalid_evidence

    with pytest.raises(OllamaProvisioningEvidenceError):
        await provider.observe(
            ProvisionedOllamaResource(
                intent.resource_type,
                receipt.provider_resource_id,
                intent.resource_name,
            ),
            intent,
            receipt,
        )

    assert state["workload_calls"] == []
    assert repository.get(waiting.lease_id).provisioning_mismatch is None


@pytest.mark.asyncio
async def test_readiness_confirmed_drift_precedes_invalid_optional_route_evidence(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    fixture_provider, state = native_provisioning
    clock = MutableClock()
    provider = _native_readiness_provider(fixture_provider, state, clock)
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "readiness-mismatch-precedence.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    waiting = await service.acquire(
        request, plan=serverless_plan(), wait_until_ready=False
    )
    intent = waiting.provisioning_intent
    receipt = waiting.provisioning_receipt
    assert intent is not None and receipt is not None
    workers = dict(state["readback"]["workers"])
    workers.update({"min": 4, "max": 4})
    state["replace_on_get"] = {"workers": workers, "requestUrls": 17}

    with pytest.raises(OllamaProvisioningMismatchError):
        await provider.observe(
            ProvisionedOllamaResource(
                intent.resource_type,
                receipt.provider_resource_id,
                intent.resource_name,
            ),
            intent,
            receipt,
        )

    assert state["workload_calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("service_path", ["initial", "ready", "final_cleanup"])
async def test_service_persists_readiness_drift_and_joins_exact_cleanup(
    native_provisioning, tmp_path, monkeypatch, service_path
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    fixture_provider, state = native_provisioning
    clock = MutableClock()
    provider = _native_readiness_provider(fixture_provider, state, clock)
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / f"readiness-service-{service_path}.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    waiting = await service.acquire(
        request, plan=serverless_plan(), wait_until_ready=False
    )
    assert waiting.state is OllamaLeaseState.WAITING_FOR_MODEL
    if service_path != "initial":
        ready = await service.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        assert ready.state is OllamaLeaseState.READY
    calls_before_drift = tuple(state["workload_calls"])
    _apply_readiness_policy_drift(state, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER)

    if service_path == "initial":
        closed = await service.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
    elif service_path == "ready":
        closed = await service.get(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
    else:
        closed = await service.release(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )

    durable = repository.get(request.lease_id)
    assert closed.state is OllamaLeaseState.TERMINATED
    assert durable is not None
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value == "endpoint_policy"
    assert tuple(state["workload_calls"]) == calls_before_drift
    assert state["calls"][-2:] == [
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]


@pytest.mark.asyncio
async def test_service_readiness_foreign_exact_get_never_gains_route_or_cleanup_authority(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    fixture_provider, state = native_provisioning
    clock = MutableClock()
    provider = _native_readiness_provider(fixture_provider, state, clock)
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "readiness-foreign-exact-get.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    waiting = await service.acquire(
        request, plan=serverless_plan(), wait_until_ready=False
    )
    assert waiting.provider_resource_id == "endpoint-evidence-1"
    state["calls"].clear()
    state["replace_on_get"] = {"id": "foreign-endpoint"}

    closed = await service.reconcile_lease(
        request.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )

    durable = repository.get(request.lease_id)
    assert closed.state is OllamaLeaseState.TERMINATED
    assert durable is not None
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.route_url is None
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value == "resource_identity"
    assert state["workload_calls"] == []
    assert state["calls"] == [
        ("GET", "/v2/serverless/endpoint-evidence-1"),
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]
    assert all("foreign-endpoint" not in path for _method, path in state["calls"])


@pytest.mark.asyncio
async def test_native_provisioning_persists_safe_exact_readback_before_waiting(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "provisioning.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    lease = await service.acquire(
        request, plan=serverless_plan(), wait_until_ready=False
    )

    assert lease.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert lease.provider_resource_id == "endpoint-evidence-1"
    assert lease.provisioning_receipt is not None
    assert lease.provisioning_receipt.provider_resource_id == "endpoint-evidence-1"
    assert state["calls"] == [
        ("POST", "/v2/serverless"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]
    assert state["request"]["env"]["KESTREL_OLLAMA_BEARER_TOKEN"] == (_SERVERLESS_TOKEN)
    assert state["request"]["env"]["KESTREL_OLLAMA_PROVISION_REQUESTED_AT"] == (
        "2026-08-01T12:00:00.000000Z"
    )
    assert _SERVERLESS_TOKEN not in lease.provisioning_intent.provider_request_json
    assert _SERVERLESS_TOKEN.encode() not in repository.path.read_bytes()


@pytest.mark.asyncio
async def test_native_late_valid_receipt_survives_receiptless_completed_cleanup(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    clock = MutableClock()
    original_observe = provider.observe_provisioning
    first_readback_entered = asyncio.Event()
    first_readback_finished = asyncio.Event()
    readbacks = 0

    async def delayed_first_readback(*args, **kwargs):
        nonlocal readbacks
        readbacks += 1
        if readbacks == 1:
            receipt = await original_observe(*args, **kwargs)
            first_readback_entered.set()
            await first_readback_finished.wait()
            return receipt
        state["read_status"] = 503
        try:
            return await original_observe(*args, **kwargs)
        finally:
            state["read_status"] = None

    provider.observe_provisioning = delayed_first_readback
    repository = SQLiteOllamaLeaseRepository(tmp_path / "late-valid-receipt.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    acquire = asyncio.create_task(
        service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    )
    try:
        await asyncio.wait_for(first_readback_entered.wait(), timeout=5)
        closed = await service.release(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
        assert closed.state is OllamaLeaseState.TERMINATED
        assert closed.cleanup_intent is not None
        assert closed.cleanup_intent.provisioning_receipt_id is None
        assert closed.cleanup_receipt is not None
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, first_readback_finished
        )

    completed = await acquire
    durable = repository.get(request.lease_id)
    assert durable is not None
    assert completed.state is OllamaLeaseState.TERMINATED
    assert durable.provisioning_receipt is not None
    assert durable.cleanup_intent == closed.cleanup_intent
    assert durable.cleanup_receipt == closed.cleanup_receipt
    assert durable.route_url is None
    assert state["calls"].count(("DELETE", "/v2/serverless/endpoint-evidence-1")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reconcile_path", "enrollment_exclusion"),
    [
        ("direct", None),
        ("timer", None),
        ("direct", "cleanup_incomplete"),
    ],
)
async def test_native_late_first_receipt_enrolls_completed_cleanup_for_billing(
    native_provisioning,
    tmp_path,
    monkeypatch,
    reconcile_path,
    enrollment_exclusion,
):
    """Only an exactly closed single-resource lifecycle enrolls billing."""

    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    clock = MutableClock()
    provider._clock = clock
    original_observe = provider.observe_provisioning
    first_readback_entered = asyncio.Event()
    first_readback_finished = asyncio.Event()
    readbacks = 0

    async def delayed_first_readback(*args, **kwargs):
        nonlocal readbacks
        readbacks += 1
        if readbacks == 1:
            receipt = await original_observe(*args, **kwargs)
            first_readback_entered.set()
            await first_readback_finished.wait()
            return receipt
        state["read_status"] = 503
        try:
            return await original_observe(*args, **kwargs)
        finally:
            state["read_status"] = None

    provider.observe_provisioning = delayed_first_readback
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / f"late-first-billing-{reconcile_path}.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    acquire = asyncio.create_task(
        service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    )
    try:
        await asyncio.wait_for(first_readback_entered.wait(), timeout=5)
        clock.advance(3700)
        if enrollment_exclusion == "cleanup_incomplete":

            async def fail_cleanup(*_args, **_kwargs):
                raise OllamaLeaseTeardownError(
                    "injected exact cleanup remains retryable"
                )

            monkeypatch.setattr(provider, "teardown", fail_cleanup)
            with pytest.raises(OllamaLeaseTeardownError):
                await service.release(
                    request.lease_id,
                    owner_id=request.owner_id,
                    workload_id=request.workload_id,
                )
            closed = repository.get(request.lease_id)
            assert closed is not None
            assert closed.state is OllamaLeaseState.RELEASING
            assert closed.cleanup_intent is not None
            assert closed.cleanup_receipt is None
        else:
            closed = await service.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
            assert closed.state is OllamaLeaseState.TERMINATED
            assert closed.cleanup_intent is not None
            assert closed.cleanup_intent.provisioning_receipt_id is None
            assert closed.cleanup_receipt is not None
        assert closed.lifecycle_billing_state is OllamaLifecycleBillingState.UNKNOWN
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, first_readback_finished
        )

    if enrollment_exclusion == "cleanup_incomplete":
        with pytest.raises(OllamaLeaseTeardownError):
            await acquire
        completed = repository.get(request.lease_id)
        assert completed is not None
    else:
        completed = await acquire
    assert completed.provisioning_receipt is not None
    assert completed.cleanup_intent == closed.cleanup_intent
    assert completed.cleanup_receipt == closed.cleanup_receipt
    assert completed.max_authorized_cost == request.max_authorized_cost
    assert state["billing_calls"] == []

    if enrollment_exclusion is not None:
        assert completed.lifecycle_billing_state is OllamaLifecycleBillingState.UNKNOWN
        assert completed.billing_next_check_at is None
        if enrollment_exclusion == "cleanup_incomplete":
            assert completed.state is OllamaLeaseState.RELEASING
            assert completed.cleanup_receipt is None
        return

    assert completed.state is OllamaLeaseState.TERMINATED

    clock.advance(7200)
    if reconcile_path == "direct":
        reconciled = await service.reconcile_lease(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
    else:
        matches = [
            item
            for item in await service.reconcile()
            if item.lease_id == request.lease_id
        ]
        reconciled = matches[0] if matches else repository.get(request.lease_id)

    assert len(state["billing_calls"]) == 1
    if reconcile_path == "timer":
        assert len(matches) == 1
    assert reconciled is not None
    assert reconciled.lifecycle_billing_state is OllamaLifecycleBillingState.SELECTED
    assert reconciled.max_authorized_cost == request.max_authorized_cost
    assert reconciled.cleanup_intent == closed.cleanup_intent
    assert reconciled.cleanup_receipt == closed.cleanup_receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "response_defaults"),
    [
        (OllamaLeaseMode.SERVERLESS_LOAD_BALANCER, {"args": ""}),
        (
            OllamaLeaseMode.DEDICATED_POD,
            {
                "args": "",
                "mounts": {},
                "globalNetworking": {"enabled": False},
            },
        ),
    ],
)
async def test_native_openapi_response_defaults_match_omitted_create_defaults(
    native_provisioning,
    tmp_path,
    monkeypatch,
    mode,
    response_defaults,
):
    """Pinned response-only representations must preserve admitted intent."""

    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = response_defaults
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "response-shape.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=mode, max_authorized_cost=5.0)
    plan = (
        serverless_plan()
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
        else pod_plan(rate=0.5)
    )

    lease = await service.acquire(request, plan=plan, wait_until_ready=False)

    document = load_openapi(
        (
            Path(__file__).resolve().parent.parent / "vendor/runpod-v2-openapi.yaml"
        ).read_text(encoding="utf-8")
    )
    component = (
        "Endpoint" if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER else "Pod"
    )
    Draft202012Validator(
        {**document, "$ref": f"#/components/schemas/{component}"}
    ).validate(state["readback"])
    assert lease.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert lease.provisioning_receipt is not None
    assert lease.provisioning_mismatch is None
    assert not any(method == "DELETE" for method, _path in state["calls"])
    assert not any(path.endswith("/action") for _method, path in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid_response", "message"),
    [
        ({"globalNetworking": False}, "global networking evidence"),
        ({"mounts": []}, "mount evidence"),
    ],
)
async def test_native_pod_response_normalization_rejects_request_shapes(
    native_provisioning,
    tmp_path,
    monkeypatch,
    invalid_response,
    message,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = invalid_response
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "invalid-response.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)

    with pytest.raises(RunPodManagerError, match=message):
        await service.acquire(request, plan=pod_plan(rate=0.5), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provider_resource_id == "pod-evidence-1"
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is None
    assert not any(path.endswith("/action") for _method, path in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drop_on_get",
    [
        ("image",),
        ("env",),
        ("createdAt",),
    ],
)
async def test_native_missing_provisioning_evidence_retains_known_id_for_retry(
    native_provisioning,
    tmp_path,
    monkeypatch,
    drop_on_get,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["drop_on_get"] = drop_on_get
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "mismatch.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.creation_uncertain is True
    assert durable.provisioning_intent is not None
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is None
    assert _SERVERLESS_TOKEN.encode() not in repository.path.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["type", "disk", "ports"])
async def test_native_optional_endpoint_policy_omission_is_unavailable_not_mismatch(
    native_provisioning,
    tmp_path,
    monkeypatch,
    missing,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["drop_on_get"] = (missing,)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / f"missing-{missing}.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="omitted.*evidence"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    document = load_openapi(
        (
            Path(__file__).resolve().parent.parent / "vendor/runpod-v2-openapi.yaml"
        ).read_text(encoding="utf-8")
    )
    response = dict(state["readback"])
    response.pop(missing)
    Draft202012Validator(
        {**document, "$ref": "#/components/schemas/Endpoint"}
    ).validate(response)
    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_accepted_endpoint_registry_omission_is_unavailable(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    provider.deployment = replace(
        provider.deployment,
        profile=replace(_profile(), registry_id="registry-original"),
    )
    state["drop_on_get"] = ("registry",)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "missing-registry.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="omitted.*registry"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_endpoint_idle_timeout_omission_is_unavailable(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = {"workers": {"min": 0, "max": 1}}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "missing-idle-timeout.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="omitted.*idleTimeout"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provisioning_mismatch is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mounts", [pytest.param(None, id="null"), pytest.param("missing")]
)
async def test_native_pod_requires_explicit_mount_evidence(
    native_provisioning,
    tmp_path,
    monkeypatch,
    mounts,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    if mounts == "missing":
        state["drop_on_get"] = ("mounts",)
    else:
        state["replace_on_get"] = {"mounts": mounts}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "missing-pod-mounts.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)

    with pytest.raises(RunPodManagerError, match="mount evidence"):
        await service.acquire(request, plan=pod_plan(rate=0.5), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provider_resource_id == "pod-evidence-1"
    assert durable.provisioning_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_numeric",
    [
        {"workers": {"min": 0, "max": True, "idleTimeout": 300}},
        {"scaling": {"type": "REQUEST_COUNT", "requestCount": True}},
    ],
)
async def test_native_endpoint_boolean_numbers_never_complete_evidence(
    native_provisioning,
    tmp_path,
    monkeypatch,
    invalid_numeric,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = invalid_numeric
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "boolean-policy.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is not OllamaLeaseState.WAITING_FOR_MODEL
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is None
    assert not any(path.endswith("/action") for _method, path in state["calls"])


@pytest.mark.asyncio
async def test_native_confirmed_mismatch_wins_over_missing_optional_evidence(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["drop_on_get"] = ("env",)
    state["replace_on_get"] = {"image": "wrong.invalid/runtime@sha256:" + "f" * 64}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "masked-mismatch.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="does not match"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value == "image_or_environment"
    assert state["calls"][-2:] == [
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]


@pytest.mark.asyncio
async def test_native_invalid_endpoint_request_urls_cannot_complete_ack_evidence(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = {"requestUrls": 17}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "invalid-ack-urls.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="requestUrls"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable is not None
    assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.creation_uncertain is True
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is None
    assert durable.route_url is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_invalid_endpoint_request_urls_keep_discovery_untrusted(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["ambiguous_post"] = True
    state["inventory_visible"] = False
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "invalid-discovery-urls.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    state["inventory_visible"] = True
    state["replace_on_get"] = {"requestUrls": 17}
    current = (await service.reconcile())[0]

    assert current.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert current.provider_resource_id is None
    assert current.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert current.creation_uncertain is True
    assert current.provisioning_receipt is None
    assert current.provisioning_mismatch is None
    assert current.route_url is None
    assert not any(method == "DELETE" for method, _path in state["calls"])

    clock.advance(int((current.hard_deadline - clock()).total_seconds()) + 1)
    await service.reconcile()
    expired = repository.get(request.lease_id)
    assert expired is not None
    assert expired.provider_resource_id is None
    assert expired.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert expired.provisioning_receipt is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_valid_endpoint_request_urls_complete_ack_evidence(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "valid-ack-urls.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    completed = await service.acquire(
        request,
        plan=serverless_plan(),
        wait_until_ready=False,
    )

    assert completed.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert completed.provider_resource_id == "endpoint-evidence-1"
    assert completed.provisioning_receipt is not None
    assert completed.provisioning_mismatch is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_missing_endpoint_request_urls_remain_optional_until_readiness(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["drop_on_get"] = ("requestUrls",)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "missing-ack-urls.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    completed = await service.acquire(
        request,
        plan=serverless_plan(),
        wait_until_ready=False,
    )

    assert completed.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert completed.provisioning_receipt is not None
    assert completed.route_url is None
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_request_url_invalidity_does_not_mask_owned_ack_mismatch(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = {
        "requestUrls": 17,
        "image": "wrong.invalid/runtime@sha256:" + "f" * 64,
    }
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "invalid-urls-owned-mismatch.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="does not match"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable is not None
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value == "image_or_environment"
    assert state["calls"][-2:] == [
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]


@pytest.mark.asyncio
async def test_native_request_url_invalidity_does_not_authorize_mismatched_discovery(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["ambiguous_post"] = True
    state["inventory_visible"] = False
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / "invalid-urls-discovery-mismatch.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    state["inventory_visible"] = True
    state["replace_on_get"] = {
        "requestUrls": 17,
        "image": "wrong.invalid/runtime@sha256:" + "f" * 64,
    }
    current = (await service.reconcile())[0]

    assert current.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert current.provider_resource_id is None
    assert current.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert current.provisioning_receipt is None
    assert current.provisioning_mismatch is None
    assert current.ambiguous_candidate_mismatch is not None
    assert current.ambiguous_candidate_mismatch.reason.value == "image_or_environment"
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missing_update", "wrong_update"),
    [
        (
            {"dataCenterId": None},
            {
                "gpu": {
                    "id": "foreign-gpu",
                    "count": 1,
                    "vcpuCount": 4,
                    "memory": 24,
                }
            },
        ),
        (
            {"cudaVersion": None},
            {
                "mounts": {
                    "network": [],
                    "global": [{"volumeId": "unauthorized-global-volume"}],
                }
            },
        ),
    ],
)
async def test_native_confirmed_pod_violation_wins_over_independent_missing_evidence(
    native_provisioning,
    tmp_path,
    monkeypatch,
    missing_update,
    wrong_update,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = {**missing_update, **wrong_update}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "masked-pod-mismatch.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=5.0,
    )
    if "cudaVersion" in missing_update:
        request = replace(
            request,
            constraints=replace(request.constraints, min_cuda_version="12.0"),
        )

    with pytest.raises(RunPodManagerError, match="does not match|differs"):
        await service.acquire(request, plan=pod_plan(rate=0.5), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value in {
        "pod_placement_or_rate",
        "runtime",
    }
    assert state["calls"][-2:] == [
        ("POST", "/v2/pods/pod-evidence-1/action"),
        ("GET", "/v2/pods/pod-evidence-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mounts", "confirmed_mismatch"),
    [
        ({"network": [17]}, False),
        (
            {"network": [{"volumeId": "foreign-volume", "path": "/workspace"}]},
            True,
        ),
        (
            {
                "network": [
                    17,
                    {"volumeId": "foreign-volume", "path": "/workspace"},
                ]
            },
            True,
        ),
    ],
)
async def test_native_pod_mounts_separate_invalid_from_confirmed_foreign_storage(
    native_provisioning,
    tmp_path,
    monkeypatch,
    mounts,
    confirmed_mismatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    fixture_provider, state = native_provisioning
    state["replace_on_get"] = {"mounts": mounts}
    deployment = replace(
        fixture_provider.deployment,
        profile=replace(
            fixture_provider.deployment.profile,
            network_volume_id="approved-volume",
        ),
        serverless_non_compute_cost=non_compute_cost_policy(
            include_network_volume=True
        ),
        pod_non_compute_cost=non_compute_cost_policy(include_network_volume=True),
    )
    provider = _provider(client=fixture_provider.client, deployment=deployment)
    clock = MutableClock()
    provider._clock = clock
    repository = SQLiteOllamaLeaseRepository(
        tmp_path / f"pod-mount-{confirmed_mismatch}.sqlite3"
    )
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    plan = replace(
        pod_plan(rate=0.5),
        non_compute_components=(deployment.pod_non_compute_cost.covered_components),
    )

    expected_error = (
        OllamaProvisioningMismatchError if confirmed_mismatch else RunPodManagerError
    )
    with pytest.raises(expected_error):
        await service.acquire(request, plan=plan, wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable is not None
    if confirmed_mismatch:
        assert durable.state is OllamaLeaseState.TERMINATED
        assert durable.provisioning_mismatch is not None
        assert durable.provisioning_mismatch.reason.value == "runtime"
        assert state["calls"][-2:] == [
            ("POST", "/v2/pods/pod-evidence-1/action"),
            ("GET", "/v2/pods/pod-evidence-1"),
        ]
    else:
        assert durable.state is OllamaLeaseState.RECONCILE_REQUIRED
        assert durable.provisioning_mismatch is None
        assert durable.cleanup_receipt is None
        assert not any(path.endswith("/action") for _method, path in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,replace_on_get",
    [
        (
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            {
                "gpu": None,
                "mounts": {
                    "network": [],
                    "global": [{"volumeId": "unauthorized-global-volume"}],
                },
            },
        ),
        (
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            {
                "mounts": {
                    "global": [{"volumeId": "unauthorized-global-volume"}],
                },
            },
        ),
        (
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            {
                "mounts": {
                    "network": [{"volumeId": "confirmed-foreign"}, 17],
                    "global": [],
                },
            },
        ),
        (
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            {
                "gpu": {
                    "pools": ["pool-24"],
                    "allowedCudaVersions": [],
                    "excludedTypes": [],
                    "minCudaVersion": None,
                },
                "workers": {"min": 0, "max": 9, "idleTimeout": 300},
            },
        ),
        (
            OllamaLeaseMode.DEDICATED_POD,
            {
                "mounts": None,
                "globalNetworking": {"enabled": True},
            },
        ),
    ],
)
async def test_native_every_confirmed_violation_wins_over_missing_evidence(
    native_provisioning,
    tmp_path,
    monkeypatch,
    mode,
    replace_on_get,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = replace_on_get
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "complete-assessment.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=mode, max_authorized_cost=5.0)
    plan = (
        serverless_plan()
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
        else pod_plan(rate=0.5)
    )

    with pytest.raises(RunPodManagerError, match="does not match"):
        await service.acquire(request, plan=plan, wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_receipt is None
    if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER:
        assert state["calls"][-2:] == [
            ("DELETE", "/v2/serverless/endpoint-evidence-1"),
            ("GET", "/v2/serverless/endpoint-evidence-1"),
        ]
    else:
        assert state["calls"][-2:] == [
            ("POST", "/v2/pods/pod-evidence-1/action"),
            ("GET", "/v2/pods/pod-evidence-1"),
        ]


@pytest.mark.asyncio
async def test_native_accepted_rate_is_independent_of_ambient_decimal_precision(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["pod_cost"] = 0.99
    clock = MutableClock()
    database = tmp_path / "decimal-context.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=5.0,
    )

    with localcontext() as context:
        context.prec = 1
        await service.acquire(
            request,
            plan=pod_plan(rate=0.99),
            wait_until_ready=False,
        )

    durable = SQLiteOllamaLeaseRepository(database).get(request.lease_id)
    assert durable.provisioning_intent is not None
    assert durable.provisioning_intent.accepted_whole_hourly_rate_usd == Decimal("0.99")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rate", "gpu_count", "non_compute_maximum", "total_cap"),
    [
        (0.47, 3, 0.0, 1.40),
        (0.99, 1, 0.02, 1.00),
    ],
)
async def test_native_total_cost_plan_never_rounds_below_exact_exposure(
    tmp_path,
    monkeypatch,
    rate,
    gpu_count,
    non_compute_maximum,
    total_cap,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    offer = replace(
        _offer(ComputeProduct.POD),
        secure_price_per_hr=rate,
        secure_max_count=gpu_count,
    )
    client.list_gpus = lambda **_: (offer,)
    deployment = replace(
        _deployment(),
        profile=replace(_profile(), gpu_count=gpu_count),
        pod_non_compute_cost=non_compute_cost_policy(maximum=non_compute_maximum),
    )
    provider = _provider(client=client, deployment=deployment)
    clock = MutableClock()
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        constraints=replace(
            make_request(clock).constraints,
            gpu_count=gpu_count,
            max_hourly_rate=rate,
        ),
        hard_deadline=clock() + timedelta(hours=1),
        max_authorized_cost=total_cap,
    )
    service = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(tmp_path / "exact-total.sqlite3"),
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )

    with localcontext() as context:
        context.prec = 2
        with pytest.raises(RunPodManagerError, match="authorized cost"):
            plan = await provider.plan(request)
            await service.acquire(request, plan=plan, wait_until_ready=False)

    assert client.endpoint_request is None


@pytest.mark.asyncio
async def test_native_ambient_precision_cannot_admit_rate_above_accepted_ceiling(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["pod_cost"] = 1
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "decimal-ceiling.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=5.0,
    )

    with localcontext() as context:
        context.prec = 1
        with pytest.raises(RunPodManagerError, match="exceeds or differs"):
            await service.acquire(
                request,
                plan=pod_plan(rate=0.99),
                wait_until_ready=False,
            )

    durable = SQLiteOllamaLeaseRepository(repository.path).get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value == "pod_placement_or_rate"
    assert state["calls"][-2:] == [
        ("POST", "/v2/pods/pod-evidence-1/action"),
        ("GET", "/v2/pods/pod-evidence-1"),
    ]


@pytest.mark.asyncio
async def test_native_acknowledged_id_get_404_still_reaches_exact_cleanup(
    native_provisioning,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["read_status"] = 404
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "acknowledged-404.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    acknowledged = repository.get(request.lease_id)
    assert acknowledged.provider_resource_id == "endpoint-evidence-1"
    assert acknowledged.creation_uncertain is True

    released = await service.release(
        request.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )
    repeated = await service.release(
        request.lease_id,
        owner_id=request.owner_id,
        workload_id=request.workload_id,
    )

    assert released.state is OllamaLeaseState.TERMINATED
    assert repeated == released
    assert released.creation_uncertain is False
    assert released.teardown_state is OllamaTeardownState.COMPLETE
    assert state["calls"][-2:] == [
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]
    assert state["calls"].count(("DELETE", "/v2/serverless/endpoint-evidence-1")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replace_on_get,reason",
    [
        (
            {"image": "wrong.invalid/runtime@sha256:" + "f" * 64},
            "image_or_environment",
        ),
        ({"env": {"KESTREL_OLLAMA_BEARER_TOKEN": "wrong"}}, "image_or_environment"),
        ({"id": "foreign-endpoint"}, "resource_identity"),
        ({"args": "--serve-another-runtime"}, "runtime"),
        (
            {
                "gpu": {
                    "pools": ["pool-24"],
                    "count": 1,
                    "excludedTypes": ["gpu-serverless"],
                }
            },
            "endpoint_policy",
        ),
        (
            {
                "mounts": {
                    "network": [],
                    "global": [{"volumeId": "unauthorized-global-volume"}],
                }
            },
            "endpoint_policy",
        ),
    ],
)
async def test_native_confirmed_mismatch_is_recorded_and_teardown_is_joined(
    native_provisioning,
    tmp_path,
    monkeypatch,
    replace_on_get,
    reason,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = replace_on_get
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "mismatch.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="does not match|another resource"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.teardown_state is OllamaTeardownState.COMPLETE
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.creation_uncertain is False
    assert durable.route_url is None
    assert durable.provider_health_url is None
    assert durable.provisioning_receipt is None
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.reason.value == reason
    assert durable.termination_reason == "provisioning_evidence_mismatch"
    assert state["calls"] == [
        ("POST", "/v2/serverless"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]
    assert _SERVERLESS_TOKEN.encode() not in repository.path.read_bytes()


@pytest.mark.asyncio
async def test_native_confirmed_post_identity_mismatch_acks_id_before_teardown(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_post"] = {"name": "foreign-endpoint"}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "post-mismatch.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError, match="identity is inconsistent"):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provider_resource_id == "endpoint-evidence-1"
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.observed_resource_name == "foreign-endpoint"
    assert durable.provisioning_mismatch.reason.value == "resource_identity"
    assert state["calls"] == [
        ("POST", "/v2/serverless"),
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]


class _BlockedWrongNameCreateClient(_ControlClient):
    def __init__(self):
        super().__init__()
        self.create_entered = threading.Event()
        self.create_finish = threading.Event()

    def create_endpoint(self, request):
        self.create_entered.set()
        assert self.create_finish.wait(5), "test did not release original POST"
        acknowledged = super().create_endpoint(request)
        self.endpoint = replace(
            acknowledged,
            name="wrong-acknowledged-name",
            raw={**acknowledged.raw, "name": "wrong-acknowledged-name"},
        )
        return self.endpoint


@pytest.mark.asyncio
async def test_native_wrong_name_post_ack_survives_concurrent_release(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _BlockedWrongNameCreateClient()
    provider = _provider(client=client)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "late-wrong-name.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER)
    acquire = asyncio.create_task(
        service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    )
    try:
        assert await asyncio.to_thread(client.create_entered.wait, 5)
        with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
            await service.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, client.create_finish
        )
    with pytest.raises(OllamaProvisioningMismatchError):
        await acquire

    durable = repository.get(request.lease_id)
    assert durable.state is OllamaLeaseState.TERMINATED
    assert durable.provider_resource_id == "endpoint-1"
    assert durable.provisioning_mismatch is not None
    assert durable.provisioning_mismatch.observed_resource_name == (
        "wrong-acknowledged-name"
    )
    assert client.actions == [("endpoint-1", "delete")]


class _BlockedCandidateThenOriginalClient(_ControlClient):
    def __init__(self):
        super().__init__()
        self.create_entered = threading.Event()
        self.create_finish = threading.Event()

    def create_endpoint(self, request):
        self.endpoint_request = request
        self.endpoint = EndpointResource(
            id="candidate-endpoint",
            name=request.name,
            endpoint_type=request.endpoint_type,
            request_urls={},
            raw=request.to_payload(),
            image="wrong.invalid/runtime@sha256:" + "f" * 64,
            environment_sha256=environment_values_sha256(request.env),
            created_at=datetime(2026, 8, 1, 12, tzinfo=UTC),
        )
        self.create_entered.set()
        assert self.create_finish.wait(5), "test did not release original POST"
        self.endpoint = EndpointResource(
            id="original-endpoint",
            name=request.name,
            endpoint_type=request.endpoint_type,
            request_urls={},
            raw=request.to_payload(),
            image=request.image,
            environment_sha256=environment_values_sha256(request.env),
            created_at=datetime(2026, 8, 1, 12, tzinfo=UTC),
        )
        return self.endpoint

    def get_endpoint(self, endpoint_id):
        self._require_present(endpoint_id)
        assert endpoint_id == self.endpoint.id
        return self.endpoint


@pytest.mark.asyncio
async def test_native_original_ack_survives_different_candidate_mismatch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _BlockedCandidateThenOriginalClient()
    provider = _provider(client=client)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "candidate-and-ack.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER)
    acquire = asyncio.create_task(
        service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    )
    try:
        assert await asyncio.to_thread(client.create_entered.wait, 5)
        with pytest.raises(OllamaLeaseTeardownError, match="still reconciling"):
            await service.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            acquire, client.create_finish
        )

    completed = await acquire
    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.provider_resource_id == "original-endpoint"
    assert completed.ambiguous_create_candidate_id == "candidate-endpoint"
    assert completed.ambiguous_candidate_mismatch is not None
    assert completed.ambiguous_candidate_mismatch.provider_resource_id == (
        "candidate-endpoint"
    )
    assert client.actions == [("original-endpoint", "delete")]


async def _run_native_late_original_ack_with_cleanup_failure(
    tmp_path,
    monkeypatch,
    *,
    fail_primary_cleanup: bool,
    fail_additional_cleanup: bool = False,
):
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    clock = MutableClock()
    post_entered = threading.Event()
    post_finished = threading.Event()
    final_telemetry_entered = asyncio.Event()
    final_telemetry_finished = asyncio.Event()
    state = {
        "resources": {},
        "calls": [],
        "request_body": None,
        "final_telemetry": False,
    }

    def control_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        state["calls"].append((request.method, path))
        if request.method == "POST" and path == "/v2/serverless":
            body = json.loads(request.content)
            state["request_body"] = body
            gpu = dict(body["gpu"])
            gpu.update(
                allowedCudaVersions=[],
                excludedTypes=[],
                minCudaVersion=gpu.get("minCudaVersion"),
            )
            base = {
                **body,
                "gpu": gpu,
                "args": body.get("args", ""),
                "createdAt": clock().isoformat(),
                "requestUrls": {
                    "base": "https://runtime.invalid",
                    "health": "https://runtime.invalid/ping",
                },
            }
            candidate = {**base, "id": "candidate-valid"}
            original = {**base, "id": "original-post-ack"}
            state["resources"] = {
                "candidate-valid": candidate,
                "original-post-ack": original,
            }
            post_entered.set()
            assert post_finished.wait(5), "test did not release original POST"
            return httpx.Response(201, json=original)
        if request.method == "GET" and path == "/v2/serverless":
            return httpx.Response(
                200,
                json={
                    "endpoints": [state["resources"]["candidate-valid"]],
                    "pagination": {"hasNextPage": False, "nextCursor": None},
                },
            )
        if request.method == "GET":
            resource_id = path.rsplit("/", 1)[-1]
            resource = state["resources"].get(resource_id)
            if resource is None:
                return httpx.Response(404, json={"title": "absent"})
            return httpx.Response(200, json=resource)
        if request.method == "DELETE":
            resource_id = path.rsplit("/", 1)[-1]
            if fail_primary_cleanup and resource_id == "candidate-valid":
                return httpx.Response(503, json={"title": "primary delete unavailable"})
            if fail_additional_cleanup and resource_id == "original-post-ack":
                return httpx.Response(
                    503, json={"title": "additional delete unavailable"}
                )
            state["resources"].pop(resource_id, None)
            return httpx.Response(204)
        raise AssertionError((request.method, path))

    async def runtime_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ping":
            return httpx.Response(200)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        if request.url.path == "/kestrel/telemetry":
            if state["final_telemetry"]:
                final_telemetry_entered.set()
                await final_telemetry_finished.wait()
                raise ExecutionCommitOutcomeError("unknown")
            body = state["request_body"]
            assert isinstance(body, dict)
            environment = body["env"]
            return httpx.Response(
                200,
                json=runtime_telemetry_payload(
                    environment["KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256"],
                    environment["KESTREL_OLLAMA_PROVISION_REQUESTED_AT"],
                ),
            )
        return httpx.Response(404)

    client = RunpodControlPlaneClient(
        api_key="synthetic-control-only",
        http_transport=httpx.MockTransport(control_handler),
        max_safe_retries=0,
    )
    owned_tasks = []
    try:
        deployment = replace(
            _deployment(), profile=replace(_profile(), image_name=_TEST_IMAGE)
        )
        provider = _provider(
            client=client,
            deployment=deployment,
            transport=httpx.MockTransport(runtime_handler),
        )
        provider._clock = clock
        repository = SQLiteOllamaLeaseRepository(tmp_path / "late-ack-control.sqlite3")
        service = OllamaLeaseService(
            repository=repository,
            provider=provider,
            poll_interval_seconds=1,
            clock=clock,
            sleep=clock.sleep,
        )
        request = make_request(
            clock,
            mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            max_authorized_cost=5.0,
        )
        acquire = asyncio.create_task(
            service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
        )
        owned_tasks.append(acquire)
        try:
            assert await asyncio.to_thread(post_entered.wait, 5)
            adopted = await service.reconcile_lease(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
            assert adopted.state is OllamaLeaseState.READY
            assert adopted.provider_resource_id == "candidate-valid"

            state["final_telemetry"] = True
            release = asyncio.create_task(
                service.release(
                    request.lease_id,
                    owner_id=request.owner_id,
                    workload_id=request.workload_id,
                )
            )
            owned_tasks.append(release)
            await asyncio.wait_for(final_telemetry_entered.wait(), timeout=5)
            post_finished.set()
            for _ in range(1_000):
                await asyncio.sleep(0)
                durable = repository.get(request.lease_id)
                assert durable is not None
                if durable.additional_cleanup_targets:
                    break
            else:
                pytest.fail("late original acknowledgment was not durably retained")
            final_telemetry_finished.set()
        finally:
            await release_gates_and_boundedly_join_owned_tasks(
                owned_tasks, post_finished, final_telemetry_finished
            )

        results = await asyncio.gather(acquire, release, return_exceptions=True)
        result_evidence = [
            (type(result).__name__, str(result), is_execution_control_error(result))
            for result in results
        ]
        assert all(item[2] for item in result_evidence), result_evidence
        delete_ids = [
            path.rsplit("/", 1)[-1]
            for method, path in state["calls"]
            if method == "DELETE"
        ]
        completed = repository.get(request.lease_id)
        assert completed is not None
        return state, result_evidence, delete_ids, completed
    finally:
        client.close()


@pytest.mark.asyncio
async def test_native_late_original_ack_joins_final_control_failure_cleanup(
    tmp_path, monkeypatch
):
    (
        state,
        result_evidence,
        delete_ids,
        completed,
    ) = await _run_native_late_original_ack_with_cleanup_failure(
        tmp_path, monkeypatch, fail_primary_cleanup=False
    )

    assert state["resources"] == {}
    assert all(item[2] for item in result_evidence), result_evidence
    assert sorted(delete_ids) == ["candidate-valid", "original-post-ack"]
    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.cleanup_receipt is not None
    assert len(completed.additional_cleanup_targets) == 1
    assert completed.additional_cleanup_targets[0].cleanup_receipt is not None


@pytest.mark.asyncio
async def test_native_additional_cleanup_failure_retains_primary_receipt(
    tmp_path, monkeypatch
):
    (
        state,
        result_evidence,
        delete_ids,
        completed,
    ) = await _run_native_late_original_ack_with_cleanup_failure(
        tmp_path,
        monkeypatch,
        fail_primary_cleanup=False,
        fail_additional_cleanup=True,
    )

    assert all(item[2] for item in result_evidence), result_evidence
    assert sorted(delete_ids) == ["candidate-valid", "original-post-ack"]
    assert "candidate-valid" not in state["resources"]
    assert "original-post-ack" in state["resources"]
    assert completed.state is OllamaLeaseState.RELEASING
    assert completed.cleanup_receipt is not None
    assert len(completed.additional_cleanup_targets) == 1
    assert completed.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
async def test_native_every_cleanup_failure_is_retained_after_join(
    tmp_path, monkeypatch
):
    (
        state,
        result_evidence,
        delete_ids,
        completed,
    ) = await _run_native_late_original_ack_with_cleanup_failure(
        tmp_path,
        monkeypatch,
        fail_primary_cleanup=True,
        fail_additional_cleanup=True,
    )

    assert all(item[2] for item in result_evidence), result_evidence
    assert sorted(delete_ids) == ["candidate-valid", "original-post-ack"]
    assert set(state["resources"]) == {"candidate-valid", "original-post-ack"}
    assert completed.state is OllamaLeaseState.RELEASING
    assert completed.cleanup_receipt is None
    assert len(completed.additional_cleanup_targets) == 1
    assert completed.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
async def test_native_primary_cleanup_failure_does_not_starve_original_ack(
    tmp_path, monkeypatch
):
    (
        state,
        result_evidence,
        delete_ids,
        completed,
    ) = await _run_native_late_original_ack_with_cleanup_failure(
        tmp_path, monkeypatch, fail_primary_cleanup=True
    )

    assert all(item[2] for item in result_evidence), result_evidence
    assert "candidate-valid" in delete_ids
    assert "original-post-ack" in delete_ids
    assert "candidate-valid" in state["resources"]
    assert "original-post-ack" not in state["resources"]
    assert completed.state is OllamaLeaseState.RELEASING
    assert completed.cleanup_receipt is None
    assert len(completed.additional_cleanup_targets) == 1
    assert completed.additional_cleanup_targets[0].cleanup_receipt is not None
    assert completed.teardown_state is OllamaTeardownState.RETRYABLE_FAILURE
    assert completed.last_provider_error is not None
    assert "primary delete unavailable" in completed.last_provider_error
    assert completed.reconcile_after is not None


class _TwoTargetCleanupProvider(FakeOllamaProvider):
    def __init__(self, *, failed_resource_ids=()):
        super().__init__(serverless_plan())
        self.failed_resource_ids = set(failed_resource_ids)
        self.teardown_resource_ids = []

    async def teardown(self, resource, cleanup_intent):
        self.teardown_resource_ids.append(resource.provider_resource_id)
        if resource.provider_resource_id in self.failed_resource_ids:
            scope = (
                "teardown"
                if resource.provider_resource_id == "provider-001"
                else "additional teardown"
            )
            raise OllamaLeaseTeardownError(
                f"{scope} remains retryable for {resource.provider_resource_id}"
            )
        return await super().teardown(resource, cleanup_intent)


class _UnknownCleanupContractError(RunPodManagerError):
    """A structural provider-contract failure unknown to the service."""


class _UnknownTeardownContractError(OllamaLeaseTeardownError):
    """A teardown subclass cannot manufacture the exact retry disposition."""


class _StructuralCleanupProvider(FakeOllamaProvider):
    def __init__(self, error_type):
        super().__init__(serverless_plan())
        self.error_type = error_type
        self.teardown_resource_ids = []

    async def teardown(self, resource, cleanup_intent):
        self.teardown_resource_ids.append(resource.provider_resource_id)
        if resource.provider_resource_id == "provider-001":
            raise self.error_type("structural cleanup failure")
        return await super().teardown(resource, cleanup_intent)


async def _ready_two_target_cleanup(tmp_path, provider, *, database_name):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / database_name)
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(
        make_request(clock, mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER),
        plan=serverless_plan(),
    )
    assert ready.provisioning_intent is not None
    with_late_ack = repository.attach_provider_acknowledgment(
        ready,
        ProvisionedOllamaResource(
            resource_type=ready.provisioning_intent.resource_type,
            provider_resource_id="late-original-ack",
            resource_name=ready.provisioning_intent.resource_name,
        ),
        now=clock(),
    )
    assert with_late_ack.provider_resource_id == "provider-001"
    assert [
        target.provider_resource_id
        for target in with_late_ack.additional_cleanup_targets
    ] == ["late-original-ack"]
    return clock, service, with_late_ack


async def _claimed_two_target_cleanup(
    tmp_path,
    provider,
    *,
    admission_provider=None,
    database_name="two-target-cleanup.sqlite3",
):
    clock, service, with_late_ack = await _ready_two_target_cleanup(
        tmp_path,
        admission_provider or provider,
        database_name=database_name,
    )
    if admission_provider is not None:
        service = OllamaLeaseService(
            repository=service.repository,
            provider=provider,
            poll_interval_seconds=1,
            clock=clock,
            sleep=clock.sleep,
        )
    repository = service.repository
    releasing = repository.compare_and_set(
        with_late_ack,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test two-target cleanup",
        },
    )
    admitted = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test two-target cleanup",
    )
    return service, repository.claim_cleanup_attempt(admitted, now=clock())


def _all_nested_errors(error):
    if isinstance(error, BaseExceptionGroup):
        return [
            nested for child in error.exceptions for nested in _all_nested_errors(child)
        ]
    return [error]


def _explicit_error_graph(error):
    pending = [error]
    visited = set()
    errors = []
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        errors.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return errors


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_native_cleanup_auth_failure_stops_before_second_owned_target(
    tmp_path,
    native_teardown,
    status_code,
):
    """A provider authentication failure must stop later billable cleanup I/O."""

    provider, state = native_teardown
    state["delete_status"] = status_code
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        admission_provider=FakeOllamaProvider(serverless_plan()),
        database_name=f"cleanup-auth-{status_code}.sqlite3",
    )

    try:
        await service._teardown_all_effects(claimed, reason="test auth blocker")
    except BaseException as error:  # noqa: BLE001 - exact graph is asserted
        caught = error
    else:
        pytest.fail("provider authentication failure was not retained")

    graph = _explicit_error_graph(caught)
    api_errors = [error for error in graph if isinstance(error, RunPodAPIError)]
    assert len(api_errors) == 1
    assert api_errors[0].status_code == status_code
    assert state["calls"] == [("DELETE", "/v2/serverless/provider-001")]
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.state is OllamaLeaseState.RELEASING
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
async def test_statusless_http_authorization_response_stops_later_owned_cleanup(
    tmp_path,
):
    request = httpx.Request(
        "DELETE", "https://api.runpod.io/v2/serverless/provider-001"
    )
    response = httpx.Response(401, request=request)
    response_error = httpx.HTTPStatusError(
        "authorization denied", request=request, response=response
    )
    api_error = RunPodAPIError(
        title="Runpod request failed",
        detail="HTTPStatusError",
    )
    api_error.__cause__ = response_error

    class StatuslessAuthorizationClient(_ControlClient):
        def __init__(self):
            super().__init__()
            self.delete_ids = []

        def delete_endpoint(self, endpoint_id):
            self.delete_ids.append(endpoint_id)
            if endpoint_id == "provider-001":
                raise api_error
            return super().delete_endpoint(endpoint_id)

        def get_endpoint(self, endpoint_id):
            self._require_present(endpoint_id)
            return replace(self.endpoint, id=endpoint_id)

    client = StatuslessAuthorizationClient()
    provider = _provider(client=client)
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        admission_provider=FakeOllamaProvider(serverless_plan()),
        database_name="statusless-http-auth.sqlite3",
    )

    try:
        await service._teardown_all_effects(
            claimed,
            reason="test statusless response blocker",
        )
    except BaseException as error:  # noqa: BLE001 - exact graph is asserted
        caught = error
    else:
        pytest.fail("provider authentication failure was not retained")

    assert any(error is api_error for error in _explicit_error_graph(caught))
    assert client.delete_ids == ["provider-001"]
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker_kind", ["structural", "control", "group", "cycle"])
async def test_transport_graph_blocker_stops_later_owned_cleanup(
    tmp_path,
    blocker_kind,
):
    """An HTTPX carrier cannot hide a structural or execution-control blocker."""

    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    request = httpx.Request(
        "DELETE", "https://api.runpod.io/v2/serverless/provider-001"
    )
    transport = httpx.ConnectError("connection failed", request=request)
    core = httpcore.ConnectError("connection failed")
    transport.__cause__ = core
    if blocker_kind == "structural":
        core.__cause__ = ValueError("invalid transport state")
    elif blocker_kind == "control":
        core.__cause__ = ExecutionAuthorityError("original authority retired")
    elif blocker_kind == "group":
        core.__cause__ = ExceptionGroup(
            "mixed transport graph",
            [
                TimeoutError("native timeout"),
                ExecutionAuthorityError("original authority retired"),
            ],
        )
    else:
        structural = ValueError("cyclic structural failure")
        core.__cause__ = structural
        structural.__cause__ = core
    api_error = RunPodAPIError(
        title="Runpod transport failed",
        detail="ConnectError",
    )
    api_error.__cause__ = transport

    class BlockedTransportClient(_ControlClient):
        def __init__(self):
            super().__init__()
            self.delete_ids = []

        def delete_endpoint(self, endpoint_id):
            self.delete_ids.append(endpoint_id)
            if endpoint_id == "provider-001":
                raise api_error
            return super().delete_endpoint(endpoint_id)

        def get_endpoint(self, endpoint_id):
            self._require_present(endpoint_id)
            return replace(self.endpoint, id=endpoint_id)

    client = BlockedTransportClient()
    provider = _provider(client=client)
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        admission_provider=FakeOllamaProvider(serverless_plan()),
        database_name=f"transport-graph-{blocker_kind}.sqlite3",
    )

    try:
        await service._teardown_all_effects(
            claimed,
            reason="test blocked transport graph",
        )
    except BaseException as error:  # noqa: BLE001 - exact graph is asserted
        caught = error
    else:
        pytest.fail("blocked transport graph was not retained")

    assert any(error is api_error for error in _explicit_error_graph(caught))
    assert client.delete_ids == ["provider-001"]
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("readback", "allows_later_owned_target"),
    [("exact", True), ("foreign-id", False), ("wrong-type", False)],
)
async def test_cleanup_retry_requires_exact_typed_resource_readback(
    tmp_path,
    readback,
    allows_later_owned_target,
):
    class CleanupReadbackClient(_ControlClient):
        def __init__(self):
            super().__init__()
            self.delete_ids = []

        def delete_endpoint(self, endpoint_id):
            self.delete_ids.append(endpoint_id)
            if endpoint_id != "provider-001":
                return super().delete_endpoint(endpoint_id)

        def get_endpoint(self, endpoint_id):
            self._require_present(endpoint_id)
            if endpoint_id != "provider-001":
                return replace(self.endpoint, id=endpoint_id)
            if readback == "exact":
                return replace(self.endpoint, id=endpoint_id)
            if readback == "foreign-id":
                return replace(self.endpoint, id="foreign-endpoint")
            return PodResource(
                id=endpoint_id,
                name=self.endpoint.name,
                status="RUNNING",
                gpu_id="gpu-pod",
                gpu_count=1,
                cost_per_hr=Decimal("0.5"),
                raw={"runtime": {"ports": []}},
            )

    client = CleanupReadbackClient()
    provider = _provider(client=client)
    service, claimed = await _claimed_two_target_cleanup(
        tmp_path,
        provider,
        admission_provider=FakeOllamaProvider(serverless_plan()),
        database_name=f"cleanup-readback-{readback}.sqlite3",
    )

    try:
        await service._teardown_all_effects(
            claimed,
            reason="test exact cleanup readback identity",
        )
    except BaseException as error:  # noqa: BLE001 - custody is asserted below
        caught = error
    else:
        pytest.fail("still-present cleanup outcome was not retained")
    assert _explicit_error_graph(caught)

    expected_ids = (
        ["provider-001", "late-original-ack"]
        if allows_later_owned_target
        else ["provider-001"]
    )
    assert client.delete_ids == expected_ids
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is None
    if allows_later_owned_target:
        assert durable.additional_cleanup_targets[0].cleanup_receipt is not None
    else:
        assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        OllamaLeaseConflictError,
        _UnknownCleanupContractError,
        _UnknownTeardownContractError,
    ],
)
async def test_native_structural_cleanup_failure_stops_before_second_owned_target(
    tmp_path,
    error_type,
):
    """Unknown manager subclasses are blockers, not implicit provider retries."""

    provider = _StructuralCleanupProvider(error_type)
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)

    try:
        await service._teardown_all_effects(claimed, reason="test structural blocker")
    except BaseException as error:  # noqa: BLE001 - exact graph is asserted
        caught = error
    else:
        pytest.fail("structural cleanup failure was not retained")

    assert any(isinstance(error, error_type) for error in _explicit_error_graph(caught))
    assert provider.teardown_resource_ids == ["provider-001"]
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.state is OllamaLeaseState.RELEASING
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


def _fail_repository_stage_once(
    monkeypatch, repository, method_name, message, *, error=None, witness=None
):
    original = getattr(repository, method_name)
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            if witness is not None:
                witness(*args, **kwargs)
            raise error if error is not None else sqlite3.OperationalError(message)
        return original(*args, **kwargs)

    monkeypatch.setattr(repository, method_name, fail_once)


def _conflict_repository_stage_once(monkeypatch, repository, method_name, message):
    original = getattr(repository, method_name)
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OllamaLeaseConflictError(message)
        return original(*args, **kwargs)

    monkeypatch.setattr(repository, method_name, fail_once)


def _native_sqlite_operational_error(tmp_path, condition):
    """Produce a real file-SQLite BUSY or shared-cache LOCKED failure."""

    database = tmp_path / f"native-{condition}.sqlite3"
    uri = f"file:{database}?cache=shared"
    with (
        closing(
            sqlite3.connect(uri, uri=True, timeout=0, isolation_level=None)
        ) as first,
        closing(
            sqlite3.connect(uri, uri=True, timeout=0, isolation_level=None)
        ) as second,
    ):
        first.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, value TEXT)")
        first.execute("INSERT INTO evidence VALUES (1, 'original')")
        first.execute("BEGIN IMMEDIATE")
        first.execute("UPDATE evidence SET value = 'held' WHERE id = 1")
        try:
            if condition == "busy":
                with closing(
                    sqlite3.connect(database, timeout=0, isolation_level=None)
                ) as independent:
                    independent.execute("BEGIN IMMEDIATE")
            else:
                second.execute("UPDATE evidence SET value = 'contended' WHERE id = 1")
        except sqlite3.OperationalError as error:
            return error
        finally:
            first.rollback()
    raise AssertionError(f"SQLite did not produce the expected {condition} failure")


def _native_sqlite_nonallowlisted_error(tmp_path):
    """Produce SQLite's genuine generic SQL error, not an availability error."""

    database = tmp_path / "native-nonallowlisted.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        try:
            connection.execute("SELECT * FROM table_that_does_not_exist")
        except sqlite3.OperationalError as error:
            assert error.sqlite_errorcode == sqlite3.SQLITE_ERROR
            return error
    raise AssertionError("SQLite did not produce SQLITE_ERROR")


def _native_sqlite_programming_error(tmp_path):
    """Produce a genuine closed-connection programming failure."""

    connection = sqlite3.connect(tmp_path / "native-programming.sqlite3")
    connection.close()
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError as error:
        return error
    raise AssertionError("SQLite did not produce ProgrammingError")


def _native_sqlite_trigger_integrity_error(tmp_path, *, database_name):
    """Produce SQLite's genuine constraint-trigger integrity failure."""

    database = tmp_path / database_name
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(
            """
            CREATE TABLE evidence (id INTEGER PRIMARY KEY);
            CREATE TRIGGER reject_evidence
            BEFORE INSERT ON evidence
            BEGIN
                SELECT RAISE(ABORT, 'injected native integrity blocker');
            END;
            """
        )
        try:
            connection.execute("INSERT INTO evidence VALUES (1)")
        except sqlite3.IntegrityError as error:
            assert error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER
            return error
    raise AssertionError("SQLite did not produce SQLITE_CONSTRAINT_TRIGGER")


def _install_cleanup_receipt_integrity_trigger(repository):
    """Reject the first primary cleanup receipt in the real lease ledger."""

    with closing(sqlite3.connect(repository.path)) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_primary_cleanup_receipt
            BEFORE UPDATE OF cleanup_receipt_json ON ollama_leases
            WHEN NEW.cleanup_receipt_json IS NOT OLD.cleanup_receipt_json
            BEGIN
                SELECT RAISE(ABORT, 'injected primary cleanup receipt blocker');
            END
            """
        )
        connection.commit()


def _cleanup_blocker(tmp_path, blocker_kind):
    """Build exact blocker leaves/groups for cleanup classification tests."""

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    if blocker_kind == "integrity":
        blocker = sqlite3.IntegrityError("injected cleanup integrity blocker")
        return blocker, sqlite3.IntegrityError
    if blocker_kind == "nested_control_and_integrity":
        blocker = BaseExceptionGroup(
            "outer mixed cleanup blocker",
            [
                ExecutionCommitOutcomeError("unknown"),
                BaseExceptionGroup(
                    "inner integrity blocker",
                    [sqlite3.IntegrityError("injected nested integrity blocker")],
                ),
            ],
        )
        return blocker, sqlite3.IntegrityError
    if blocker_kind == "control_and_conflict":
        blocker = BaseExceptionGroup(
            "outer mixed cleanup blocker",
            [
                ExecutionCommitOutcomeError("unknown"),
                BaseExceptionGroup(
                    "inner authority blocker",
                    [OllamaLeaseConflictError("injected nested authority conflict")],
                ),
            ],
        )
        return blocker, OllamaLeaseConflictError
    if blocker_kind == "control_and_unexpected":
        blocker = BaseExceptionGroup(
            "outer mixed cleanup blocker",
            [
                ExecutionCommitOutcomeError("unknown"),
                BaseExceptionGroup(
                    "inner unexpected blocker",
                    [ValueError("injected nested unexpected failure")],
                ),
            ],
        )
        return blocker, ValueError
    if blocker_kind == "programming":
        return _native_sqlite_programming_error(tmp_path), sqlite3.ProgrammingError
    if blocker_kind == "uncoded_operational":
        blocker = sqlite3.OperationalError("synthetic uncoded operational failure")
        assert not hasattr(blocker, "sqlite_errorcode")
        return blocker, sqlite3.OperationalError
    return _native_sqlite_nonallowlisted_error(tmp_path), sqlite3.OperationalError


def _contains_exception_identity(error, expected):
    if error is expected:
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(
            _contains_exception_identity(child, expected) for child in error.exceptions
        )
    return False


def _cleanup_explicit_cause_graph(tmp_path, *, carrier_kind, cause_cycle):
    """Build one real integrity cause behind an otherwise permitted carrier."""

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    control = ExecutionCommitOutcomeError("unknown")
    if carrier_kind == "control":
        carrier = control
    else:
        carrier = BaseExceptionGroup(
            "cleanup carrier with one control member",
            [control],
        )
    integrity = _native_sqlite_trigger_integrity_error(
        tmp_path,
        database_name=f"explicit-cause-{carrier_kind}-{cause_cycle}.sqlite3",
    )
    carrier.__cause__ = integrity
    if cause_cycle:
        integrity.__cause__ = carrier
    return carrier, control, integrity


def _inject_cleanup_stage_once(
    monkeypatch,
    repository,
    provider,
    *,
    stage,
    error,
):
    """Inject one cleanup setup/refresh failure at the requested native stage."""

    if stage == "claim":
        original = repository.claim_cleanup_attempt

        def fail_claim_once(*args, **kwargs):
            monkeypatch.setattr(repository, "claim_cleanup_attempt", original)
            raise error

        monkeypatch.setattr(repository, "claim_cleanup_attempt", fail_claim_once)
        return

    original = repository.latest_cleanup_lineage
    failed = False

    def fail_lineage_once(*args, **kwargs):
        nonlocal failed
        should_fail = (
            stage == "initial_refresh" and not provider.teardown_resource_ids
        ) or (
            stage == "post_first_refresh"
            and provider.teardown_resource_ids == ["provider-001"]
        )
        if should_fail and not failed:
            failed = True
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(repository, "latest_cleanup_lineage", fail_lineage_once)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    ["claim", "initial_refresh", "post_first_refresh"],
)
@pytest.mark.parametrize(
    "blocker_kind",
    [
        "integrity",
        "nested_control_and_integrity",
        "control_and_conflict",
        "control_and_unexpected",
        "programming",
        "uncoded_operational",
        "native_nonallowlisted",
    ],
)
async def test_native_cleanup_blocker_leaf_stops_later_provider_io(
    tmp_path,
    monkeypatch,
    stage,
    blocker_kind,
):
    """Any nested authority/integrity blocker must outrank control leaves."""

    provider = _TwoTargetCleanupProvider()
    blocker, expected_type = _cleanup_blocker(tmp_path, blocker_kind)

    if stage == "claim":
        _, service, lease = await _ready_two_target_cleanup(
            tmp_path,
            provider,
            database_name=f"blocker-{stage}-{blocker_kind}.sqlite3",
        )
    else:
        service, lease = await _claimed_two_target_cleanup(tmp_path, provider)
    _inject_cleanup_stage_once(
        monkeypatch,
        service.repository,
        provider,
        stage=stage,
        error=blocker,
    )

    try:
        if stage == "claim":
            await service.release(
                lease.lease_id,
                owner_id=lease.owner_id,
                workload_id=lease.workload_id,
            )
        else:
            await service._teardown_all_effects(lease, reason="test cleanup blocker")
    except BaseException as error:  # noqa: BLE001 - exact leaf algebra is asserted
        caught = error
        failures = _all_nested_errors(error)
    else:
        pytest.fail("cleanup blocker was not retained")

    assert any(isinstance(error, expected_type) for error in failures)
    assert _contains_exception_identity(caught, blocker)
    assert provider.teardown_resource_ids == (
        ["provider-001"] if stage == "post_first_refresh" else []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    ["claim", "initial_refresh", "post_first_refresh"],
)
@pytest.mark.parametrize("sqlite_condition", ["busy", "locked"])
async def test_native_cleanup_recoverable_leaf_group_joins_known_targets(
    tmp_path,
    monkeypatch,
    stage,
    sqlite_condition,
):
    """Coded transient SQLite plus native control retains all proven cleanup."""

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    provider = _TwoTargetCleanupProvider()
    if stage == "claim":
        _, service, lease = await _ready_two_target_cleanup(
            tmp_path,
            provider,
            database_name=f"recoverable-{stage}.sqlite3",
        )
    else:
        service, lease = await _claimed_two_target_cleanup(tmp_path, provider)
    recoverable = BaseExceptionGroup(
        "recoverable cleanup setup",
        [
            ExecutionCommitOutcomeError("unknown"),
            _native_sqlite_operational_error(tmp_path, sqlite_condition),
        ],
    )
    _inject_cleanup_stage_once(
        monkeypatch,
        service.repository,
        provider,
        stage=stage,
        error=recoverable,
    )

    try:
        if stage == "claim":
            await service.release(
                lease.lease_id,
                owner_id=lease.owner_id,
                workload_id=lease.workload_id,
            )
        else:
            await service._teardown_all_effects(lease, reason="test recoverable setup")
    except BaseException as error:  # noqa: BLE001 - exact leaf algebra is asserted
        caught = error
        failures = _all_nested_errors(error)
    else:
        pytest.fail("recoverable setup failures were not retained")

    assert any(is_execution_control_error(error) for error in failures)
    sqlite_failures = [
        error for error in failures if isinstance(error, sqlite3.OperationalError)
    ]
    assert len(sqlite_failures) == 1
    assert sqlite_failures[0].sqlite_errorcode & 0xFF == (
        sqlite3.SQLITE_BUSY if sqlite_condition == "busy" else sqlite3.SQLITE_LOCKED
    )
    assert _contains_exception_identity(caught, recoverable)
    assert provider.teardown_resource_ids == ["provider-001", "late-original-ack"]


@pytest.mark.asyncio
async def test_native_cleanup_outcome_integrity_blocks_later_provider_io(tmp_path):
    """An uncommitted absence receipt cannot authorize another DELETE."""

    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    _install_cleanup_receipt_integrity_trigger(service.repository)

    try:
        await service._teardown_all_effects(
            claimed,
            reason="test native cleanup receipt blocker",
        )
    except BaseException as error:  # noqa: BLE001 - exact native error is evidence
        caught = error
        failures = _all_nested_errors(error)
    else:
        pytest.fail("native cleanup receipt blocker was not retained")

    integrity = [
        failure for failure in failures if isinstance(failure, sqlite3.IntegrityError)
    ]
    assert len(integrity) == 1
    assert integrity[0].sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER
    assert _contains_exception_identity(caught, integrity[0])
    assert provider.teardown_resource_ids == ["provider-001"]
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocker_kind", "expected_type"),
    [
        ("nested_control_and_integrity", sqlite3.IntegrityError),
        ("control_and_conflict", OllamaLeaseConflictError),
        ("control_and_unexpected", ValueError),
    ],
)
async def test_nested_cleanup_outcome_blocker_stops_later_provider_io(
    tmp_path,
    monkeypatch,
    blocker_kind,
    expected_type,
):
    """Every effect/outcome graph is classified before the next target."""

    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    blocker, _ = _cleanup_blocker(tmp_path, blocker_kind)
    _fail_repository_stage_once(
        monkeypatch,
        service.repository,
        "record_teardown_outcome",
        "unused exact blocker carrier",
        error=blocker,
    )

    try:
        await service._teardown_all_effects(
            claimed,
            reason="test nested cleanup outcome blocker",
        )
    except BaseException as error:  # noqa: BLE001 - exact graph is asserted
        caught = error
        failures = _all_nested_errors(error)
    else:
        pytest.fail("nested cleanup outcome blocker was not retained")

    assert _contains_exception_identity(caught, blocker)
    assert any(isinstance(failure, expected_type) for failure in failures)
    assert provider.teardown_resource_ids == ["provider-001"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cause_cycle", [False, True])
async def test_explicit_control_cause_cannot_make_integrity_carrier_recoverable(
    tmp_path,
    monkeypatch,
    cause_cycle,
):
    """A direct structural carrier outranks a nested native-control cause."""

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    provider = _TwoTargetCleanupProvider()
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name=f"explicit-control-cause-{cause_cycle}.sqlite3",
    )
    blocker = _native_sqlite_trigger_integrity_error(
        tmp_path,
        database_name=f"native-trigger-carrier-{cause_cycle}.sqlite3",
    )
    control = ExecutionCommitOutcomeError("unknown")
    blocker.__cause__ = control
    if cause_cycle:
        control.__cause__ = blocker
    _inject_cleanup_stage_once(
        monkeypatch,
        service.repository,
        provider,
        stage="claim",
        error=blocker,
    )

    try:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )
    except BaseException as error:  # noqa: BLE001 - exact graph is asserted
        caught = error
    else:
        pytest.fail("integrity carrier with explicit control cause was not retained")

    assert _contains_exception_identity(caught, blocker)
    assert blocker.__cause__ is control
    if cause_cycle:
        assert control.__cause__ is blocker
    assert provider.teardown_resource_ids == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlite_condition", ["busy", "locked"])
async def test_native_cleanup_outcome_failure_stops_before_another_provider_effect(
    tmp_path,
    monkeypatch,
    sqlite_condition,
):
    """A provider result without its durable receipt cannot permit another DELETE."""

    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    transient = _native_sqlite_operational_error(tmp_path, sqlite_condition)
    primary_receipt_witness = []

    def retain_primary_receipt_witness(lease, **_kwargs):
        assert lease.cleanup_receipt is not None
        primary_receipt_witness.append(lease.cleanup_receipt)

    record_method = (
        "record_teardown_outcome"
        if sqlite_condition == "busy"
        else "record_additional_cleanup_outcome"
    )
    _fail_repository_stage_once(
        monkeypatch,
        service.repository,
        record_method,
        "unused transient outcome error",
        error=transient,
        witness=(
            retain_primary_receipt_witness if sqlite_condition == "locked" else None
        ),
    )

    try:
        await service._teardown_all_effects(
            claimed,
            reason="test transient cleanup outcome",
        )
    except BaseException as error:  # noqa: BLE001 - identity is asserted
        caught = error
    else:
        pytest.fail("transient cleanup outcome failure was not retained")

    assert _contains_exception_identity(caught, transient)
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    additional = durable.additional_cleanup_targets[0]
    if sqlite_condition == "busy":
        assert provider.teardown_resource_ids == ["provider-001"]
        assert durable.cleanup_receipt is None
        assert additional.cleanup_receipt is None
    else:
        assert len(primary_receipt_witness) == 1
        assert provider.teardown_resource_ids == [
            "provider-001",
            "late-original-ack",
        ]
        assert durable.cleanup_intent == claimed.cleanup_intent
        assert durable.cleanup_receipt is not None
        assert durable.cleanup_receipt == primary_receipt_witness[0]
        assert durable.cleanup_receipt.to_dict() == (
            primary_receipt_witness[0].to_dict()
        )
        durable.cleanup_receipt.validate_intent(durable.cleanup_intent)
        assert additional.provider_resource_id == "late-original-ack"
        assert additional.cleanup_intent is not None
        assert additional.cleanup_intent.provider_resource_id == (
            additional.provider_resource_id
        )
        assert additional.cleanup_receipt is None
        assert durable.state is OllamaLeaseState.RELEASING
        assert durable.teardown_state is OllamaTeardownState.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "carrier_kind", "cause_cycle"),
    [
        ("claim", "group", False),
        ("claim", "control", True),
        ("outcome", "group", True),
        ("outcome", "control", False),
    ],
)
async def test_cleanup_integrity_explicit_cause_blocks_later_provider_io(
    tmp_path,
    monkeypatch,
    stage,
    carrier_kind,
    cause_cycle,
):
    """A permitted carrier cannot hide its structural explicit cause."""

    provider = _TwoTargetCleanupProvider()
    carrier, control, integrity = _cleanup_explicit_cause_graph(
        tmp_path,
        carrier_kind=carrier_kind,
        cause_cycle=cause_cycle,
    )
    if stage == "claim":
        _, service, lease = await _ready_two_target_cleanup(
            tmp_path,
            provider,
            database_name=f"explicit-cause-{stage}-{carrier_kind}.sqlite3",
        )
        _inject_cleanup_stage_once(
            monkeypatch,
            service.repository,
            provider,
            stage="claim",
            error=carrier,
        )
    else:
        service, lease = await _claimed_two_target_cleanup(tmp_path, provider)
        _fail_repository_stage_once(
            monkeypatch,
            service.repository,
            "record_teardown_outcome",
            "unused explicit-cause carrier",
            error=carrier,
        )

    try:
        if stage == "claim":
            await service.release(
                lease.lease_id,
                owner_id=lease.owner_id,
                workload_id=lease.workload_id,
            )
        else:
            await service._teardown_all_effects(
                lease,
                reason="test explicit-cause cleanup blocker",
            )
    except BaseException as error:  # noqa: BLE001 - identity graph is asserted
        caught = error
    else:
        pytest.fail("explicit-cause cleanup blocker was not retained")

    assert _contains_exception_identity(caught, carrier)
    assert carrier.__cause__ is integrity
    assert integrity.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER
    if carrier_kind == "group":
        assert carrier.exceptions == (control,)
    else:
        assert carrier is control
    if cause_cycle:
        assert integrity.__cause__ is carrier
    assert provider.teardown_resource_ids == (
        [] if stage == "claim" else ["provider-001"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "method_name"),
    [
        ("final telemetry persistence", "record_runtime_telemetry"),
        ("cleanup claim", "claim_cleanup_attempt"),
        ("initial lineage reload", "latest_cleanup_lineage"),
    ],
)
async def test_native_file_sqlite_setup_failure_cannot_starve_proven_cleanup_targets(
    tmp_path, monkeypatch, stage, method_name
):
    provider = _TwoTargetCleanupProvider()
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name=f"setup-failure-{method_name}.sqlite3",
    )
    _fail_repository_stage_once(
        monkeypatch,
        service.repository,
        method_name,
        f"injected {stage}",
        error=_native_sqlite_operational_error(tmp_path, "busy"),
    )

    try:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )
    except BaseException as error:  # noqa: BLE001 - native control groups are evidence
        failures = _all_nested_errors(error)
    else:
        pytest.fail(f"{stage} failure was not retained")

    assert any(
        isinstance(error, sqlite3.OperationalError)
        and error.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
        for error in failures
    )
    assert provider.teardown_resource_ids == ["provider-001", "late-original-ack"]
    durable = service.repository.get(ready.lease_id)
    assert durable is not None
    assert durable.cleanup_intent is not None
    assert durable.cleanup_receipt is not None
    assert len(durable.additional_cleanup_targets) == 1
    assert durable.additional_cleanup_targets[0].cleanup_receipt is not None
    assert durable.state is OllamaLeaseState.TERMINATED


@pytest.mark.asyncio
async def test_native_control_and_cleanup_claim_failure_do_not_starve_known_targets(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    provider = _TwoTargetCleanupProvider(failed_resource_ids={"provider-001"})
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name="control-and-claim-failure.sqlite3",
    )

    async def final_telemetry_control_failure(*_args, **_kwargs):
        raise ExecutionCommitOutcomeError("unknown")

    monkeypatch.setattr(provider, "observe", final_telemetry_control_failure)
    _fail_repository_stage_once(
        monkeypatch,
        service.repository,
        "claim_cleanup_attempt",
        "injected cleanup claim",
        error=_native_sqlite_operational_error(tmp_path, "busy"),
    )

    try:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )
    except BaseException as error:  # noqa: BLE001 - exact failure algebra is asserted
        failures = _all_nested_errors(error)
    else:
        pytest.fail("native control and cleanup failures were not retained")

    assert any(is_execution_control_error(error) for error in failures)
    assert any(
        isinstance(error, sqlite3.OperationalError)
        and error.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
        for error in failures
    )
    assert any("teardown remains retryable" in str(error) for error in failures)
    assert provider.teardown_resource_ids == ["provider-001", "late-original-ack"]
    durable = service.repository.get(ready.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "method_name"),
    [
        ("cleanup claim", "claim_cleanup_attempt"),
        ("stored authority validation", "validate_stored_cleanup_authority"),
        ("initial lineage refresh", "latest_cleanup_lineage"),
    ],
)
async def test_native_control_and_authority_conflict_retain_both_without_io(
    tmp_path, monkeypatch, stage, method_name
):
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    provider = _TwoTargetCleanupProvider()
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name=f"control-conflict-{method_name}.sqlite3",
    )

    async def final_telemetry_control_failure(*_args, **_kwargs):
        raise ExecutionCommitOutcomeError("unknown")

    monkeypatch.setattr(provider, "observe", final_telemetry_control_failure)
    _conflict_repository_stage_once(
        monkeypatch,
        service.repository,
        method_name,
        f"injected {stage}",
    )

    try:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )
    except BaseException as error:  # noqa: BLE001 - exact failure algebra is asserted
        failures = _all_nested_errors(error)
    else:
        pytest.fail("native control and authority conflict were not retained")

    assert any(is_execution_control_error(error) for error in failures)
    assert any(
        isinstance(error, OllamaLeaseConflictError) and stage in str(error)
        for error in failures
    )
    assert provider.teardown_resource_ids == []


@pytest.mark.asyncio
async def test_native_control_and_unclassified_claim_failure_retain_both_without_io(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    provider = _TwoTargetCleanupProvider()
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name="control-unclassified-claim.sqlite3",
    )

    async def final_telemetry_control_failure(*_args, **_kwargs):
        raise ExecutionCommitOutcomeError("unknown")

    def unexpected_claim_failure(*_args, **_kwargs):
        raise ValueError("unexpected cleanup claim failure")

    monkeypatch.setattr(provider, "observe", final_telemetry_control_failure)
    monkeypatch.setattr(
        service.repository,
        "claim_cleanup_attempt",
        unexpected_claim_failure,
    )

    try:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )
    except BaseException as error:  # noqa: BLE001 - exact failure algebra is asserted
        failures = _all_nested_errors(error)
    else:
        pytest.fail("native control and unexpected failure were not retained")

    assert any(is_execution_control_error(error) for error in failures)
    assert any(
        isinstance(error, ValueError) and "unexpected cleanup claim" in str(error)
        for error in failures
    )
    assert provider.teardown_resource_ids == []


@pytest.mark.asyncio
async def test_uncoded_outcome_persistence_blocker_stops_other_cleanup(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider(failed_resource_ids={"provider-001"})
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    persistence = sqlite3.OperationalError(
        "injected primary uncoded outcome persistence"
    )
    assert not hasattr(persistence, "sqlite_errorcode")
    _fail_repository_stage_once(
        monkeypatch,
        service.repository,
        "record_teardown_outcome",
        "unused uncoded persistence blocker",
        error=persistence,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        await service._teardown_all_effects(claimed, reason="test cleanup")

    failures = _all_nested_errors(caught.value)
    assert any("teardown remains retryable" in str(error) for error in failures)
    assert _contains_exception_identity(caught.value, persistence)
    assert provider.teardown_resource_ids == ["provider-001"]
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
async def test_failed_reload_never_authorizes_an_unseen_cleanup_target(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider()
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "unseen-target.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(make_request(clock), plan=serverless_plan())
    releasing = repository.compare_and_set(
        ready,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test unseen target",
        },
    )
    stale = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test unseen target",
    )
    assert stale.provisioning_intent is not None
    current = repository.attach_provider_acknowledgment(
        stale,
        ProvisionedOllamaResource(
            resource_type=stale.provisioning_intent.resource_type,
            provider_resource_id="unseen-late-ack",
            resource_name=stale.provisioning_intent.resource_name,
        ),
        now=clock(),
    )
    assert len(current.additional_cleanup_targets) == 1
    _fail_repository_stage_once(
        monkeypatch,
        repository,
        "latest_cleanup_lineage",
        "injected unreadable newer suffix",
        error=_native_sqlite_operational_error(tmp_path, "busy"),
    )

    with pytest.raises(sqlite3.OperationalError) as caught:
        await service._teardown_all_effects(stale, reason="test unseen target")

    assert caught.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY

    assert provider.teardown_resource_ids == ["provider-001"]
    durable = repository.get(stale.lease_id)
    assert durable is not None
    assert durable.cleanup_receipt is not None
    assert durable.additional_cleanup_targets[0].cleanup_receipt is None


@pytest.mark.asyncio
async def test_changed_or_missing_cleanup_authority_never_reaches_provider(
    tmp_path,
):
    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    assert claimed.cleanup_intent is not None
    invalid = (
        replace(claimed, provider_resource_id="foreign-resource"),
        replace(claimed, cleanup_intent=None),
        replace(
            claimed,
            cleanup_intent=replace(
                claimed.cleanup_intent,
                provider_resource_id="foreign-resource",
            ),
        ),
        replace(claimed, writer_contract_version=99),
        replace(claimed, writer_epoch=claimed.writer_epoch + 1),
    )

    for lease in invalid:
        with pytest.raises(OllamaLeaseConflictError, match="cleanup lineage"):
            await service._teardown_all_effects(lease, reason="invalid authority")

    assert provider.teardown_resource_ids == []


@pytest.mark.asyncio
async def test_cleanup_intent_persistence_failure_admits_no_provider_io(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider()
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name="cleanup-intent-admission.sqlite3",
    )
    _fail_repository_stage_once(
        monkeypatch,
        service.repository,
        "record_cleanup_intent",
        "injected cleanup intent admission",
    )

    with pytest.raises(sqlite3.OperationalError, match="cleanup intent admission"):
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )

    assert provider.teardown_resource_ids == []


def _group_messages(error):
    if isinstance(error, BaseExceptionGroup):
        return [
            message
            for nested in error.exceptions
            for message in _group_messages(nested)
        ]
    return [str(error)]


@pytest.mark.asyncio
async def test_cleanup_lineage_conflict_preserves_prior_failure_and_stops_other_target(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider(failed_resource_ids={"provider-001"})
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    original_latest = service.repository.latest_cleanup_lineage
    failed_reload = False

    def fail_once_after_first_provider_attempt(lease):
        nonlocal failed_reload
        if provider.teardown_resource_ids and not failed_reload:
            failed_reload = True
            raise OllamaLeaseConflictError("cleanup lineage reload failed")
        return original_latest(lease)

    monkeypatch.setattr(
        service.repository,
        "latest_cleanup_lineage",
        fail_once_after_first_provider_attempt,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        await service._teardown_all_effects(claimed, reason="test cleanup")

    assert provider.teardown_resource_ids == ["provider-001"]
    messages = _group_messages(caught.value)
    assert any("teardown remains retryable" in message for message in messages)
    assert any("cleanup lineage reload failed" in message for message in messages)


@pytest.mark.asyncio
async def test_unclassified_post_attempt_refresh_retains_prior_failure_and_stops(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider(failed_resource_ids={"provider-001"})
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)
    original_latest = service.repository.latest_cleanup_lineage
    failed_reload = False

    def fail_once_after_first_provider_attempt(lease):
        nonlocal failed_reload
        if provider.teardown_resource_ids and not failed_reload:
            failed_reload = True
            raise ValueError("unexpected post-attempt lineage failure")
        return original_latest(lease)

    monkeypatch.setattr(
        service.repository,
        "latest_cleanup_lineage",
        fail_once_after_first_provider_attempt,
    )

    try:
        await service._teardown_all_effects(claimed, reason="test cleanup")
    except BaseException as error:  # noqa: BLE001 - exact failure algebra is asserted
        failures = _all_nested_errors(error)
    else:
        pytest.fail("provider and post-attempt refresh failures were not retained")

    assert any("teardown remains retryable" in str(error) for error in failures)
    assert any(
        isinstance(error, ValueError) and "post-attempt lineage" in str(error)
        for error in failures
    )
    assert provider.teardown_resource_ids == ["provider-001"]


@pytest.mark.asyncio
async def test_unclassified_final_empty_refresh_retains_prior_control_failure(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )

    provider = _TwoTargetCleanupProvider()
    _, service, ready = await _ready_two_target_cleanup(
        tmp_path,
        provider,
        database_name="final-empty-refresh.sqlite3",
    )

    async def final_telemetry_control_failure(*_args, **_kwargs):
        raise ExecutionCommitOutcomeError("unknown")

    original_latest = service.repository.latest_cleanup_lineage
    all_targets_attempted_refreshes = 0

    def fail_final_empty_refresh(lease):
        nonlocal all_targets_attempted_refreshes
        if provider.teardown_resource_ids == ["provider-001", "late-original-ack"]:
            all_targets_attempted_refreshes += 1
            if all_targets_attempted_refreshes == 2:
                raise ValueError("unexpected final-empty lineage failure")
        return original_latest(lease)

    monkeypatch.setattr(provider, "observe", final_telemetry_control_failure)
    monkeypatch.setattr(
        service.repository,
        "latest_cleanup_lineage",
        fail_final_empty_refresh,
    )

    try:
        await service.release(
            ready.lease_id,
            owner_id=ready.owner_id,
            workload_id=ready.workload_id,
        )
    except BaseException as error:  # noqa: BLE001 - exact failure algebra is asserted
        failures = _all_nested_errors(error)
    else:
        pytest.fail("control and final-empty refresh failures were not retained")

    assert any(is_execution_control_error(error) for error in failures)
    assert any(
        isinstance(error, ValueError) and "final-empty lineage" in str(error)
        for error in failures
    )
    assert provider.teardown_resource_ids == ["provider-001", "late-original-ack"]


@pytest.mark.asyncio
async def test_cleanup_initial_reload_failure_sends_no_unverified_delete(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider()
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)

    def fail_lineage_reload(_lease):
        raise OllamaLeaseConflictError("cleanup authority is unreadable")

    monkeypatch.setattr(
        service.repository, "latest_cleanup_lineage", fail_lineage_reload
    )

    with pytest.raises(OllamaLeaseConflictError, match="authority is unreadable"):
        await service._teardown_all_effects(claimed, reason="test cleanup")

    assert provider.teardown_resource_ids == []


@pytest.mark.asyncio
async def test_cleanup_claim_failure_composes_with_prior_provider_failure(
    tmp_path, monkeypatch
):
    provider = _TwoTargetCleanupProvider(failed_resource_ids={"provider-001"})
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)

    def fail_additional_claim(*_args, **_kwargs):
        raise OllamaLeaseConflictError("additional cleanup claim failed")

    monkeypatch.setattr(
        service.repository,
        "record_additional_cleanup_intent",
        fail_additional_claim,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        await service._teardown_all_effects(claimed, reason="test cleanup")

    assert provider.teardown_resource_ids == ["provider-001"]
    messages = _group_messages(caught.value)
    assert any("teardown remains retryable" in message for message in messages)
    assert any("additional cleanup claim failed" in message for message in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_resource_id", "record_method", "expected_resource_ids"),
    [
        ("provider-001", "record_teardown_outcome", ["provider-001"]),
        (
            "late-original-ack",
            "record_additional_cleanup_outcome",
            ["provider-001", "late-original-ack"],
        ),
    ],
)
async def test_cleanup_provider_and_outcome_cas_failures_are_both_retained(
    tmp_path,
    monkeypatch,
    failed_resource_id,
    record_method,
    expected_resource_ids,
):
    provider = _TwoTargetCleanupProvider(failed_resource_ids={failed_resource_id})
    service, claimed = await _claimed_two_target_cleanup(tmp_path, provider)

    def fail_outcome_cas(*_args, **_kwargs):
        raise OllamaLeaseConflictError(
            f"cleanup outcome CAS failed for {failed_resource_id}"
        )

    monkeypatch.setattr(service.repository, record_method, fail_outcome_cas)

    with pytest.raises(BaseExceptionGroup) as caught:
        await service._teardown_all_effects(claimed, reason="test cleanup")

    assert provider.teardown_resource_ids == expected_resource_ids
    messages = _group_messages(caught.value)
    assert any("teardown remains retryable" in message for message in messages)
    assert any("outcome CAS failed" in message for message in messages)
    durable = service.repository.get(claimed.lease_id)
    assert durable is not None
    if failed_resource_id == "provider-001":
        assert durable.cleanup_receipt is None
        assert durable.additional_cleanup_targets[0].cleanup_receipt is None
    else:
        assert durable.cleanup_receipt is not None
        assert durable.additional_cleanup_targets[0].cleanup_receipt is None


class _BlockingDeleteClient(_ControlClient):
    def __init__(self):
        super().__init__()
        self.delete_entered = threading.Event()
        self.delete_finish = threading.Event()
        self.delete_calls = 0

    def delete_endpoint(self, endpoint_id):
        self.delete_calls += 1
        self.delete_entered.set()
        assert self.delete_finish.wait(5), "test did not release teardown"
        super().delete_endpoint(endpoint_id)


@pytest.mark.asyncio
async def test_native_concurrent_release_joins_one_exact_id_delete(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _BlockingDeleteClient()
    provider = _provider(client=client)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "joined-delete.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER)
    await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)

    first = asyncio.create_task(
        service.release(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
    )
    owned_tasks = [first]
    try:
        assert await asyncio.to_thread(client.delete_entered.wait, 5)
        second = asyncio.create_task(
            service.release(
                request.lease_id,
                owner_id=request.owner_id,
                workload_id=request.workload_id,
            )
        )
        owned_tasks.append(second)
        await asyncio.sleep(0.05)
        calls_while_first_delete_is_live = client.delete_calls
    finally:
        await release_gates_and_boundedly_join_owned_tasks(
            owned_tasks, client.delete_finish
        )
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.state is OllamaLeaseState.TERMINATED
    assert second_result.state is OllamaLeaseState.TERMINATED
    assert calls_while_first_delete_is_live == 1
    assert client.delete_calls == 1


@pytest.mark.asyncio
async def test_native_ambiguous_name_collision_never_gains_cleanup_authority(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["ambiguous_post"] = True
    state["inventory_visible"] = False
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "ambiguous.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    first = (await service.reconcile())[0]
    assert first.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert first.creation_uncertain is True
    assert first.provider_resource_id is None
    assert first.provisioning_receipt is None

    state["inventory_visible"] = True
    state["replace_on_get"] = {"image": "wrong.invalid/runtime@sha256:" + "f" * 64}
    clock.advance(1)
    final = (await service.reconcile())[0]

    assert final.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert final.provider_resource_id is None
    assert final.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert final.provisioning_mismatch is None
    assert final.ambiguous_candidate_mismatch is not None
    assert final.provisioning_receipt is None
    assert final.route_url is None
    assert state["calls"] == [
        ("POST", "/v2/serverless"),
        ("GET", "/v2/serverless"),
        ("GET", "/v2/serverless"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]

    clock.advance(int((final.hard_deadline - clock()).total_seconds()) + 1)
    still_untrusted = (await service.reconcile())[0]
    assert still_untrusted.provider_resource_id is None
    assert still_untrusted.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_ambiguous_discovery_rejects_preexisting_matching_resource(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["ambiguous_post"] = True
    state["inventory_visible"] = False
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "preexisting-candidate.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    state["readback"]["createdAt"] = "2026-07-01T12:00:00Z"
    state["inventory_visible"] = True

    final = (await service.reconcile())[0]

    assert final.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert final.provider_resource_id is None
    assert final.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert final.provisioning_mismatch is None
    assert final.ambiguous_candidate_mismatch is not None
    assert final.ambiguous_candidate_mismatch.reason.value == "resource_identity"
    assert not any(method == "DELETE" for method, _path in state["calls"])


@pytest.mark.asyncio
async def test_native_ambiguous_create_persists_untrusted_locator_and_provenance(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["ambiguous_post"] = True
    state["inventory_visible"] = False
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "ambiguous-locator.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    state["inventory_visible"] = True
    state["drop_on_get"] = ("env",)

    incomplete = (await service.reconcile())[0]

    assert incomplete.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert incomplete.provider_resource_id is None
    assert incomplete.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert incomplete.creation_uncertain is True
    assert incomplete.provisioning_receipt is None
    with (
        closing(sqlite3.connect(repository.path)) as connection,
        connection,
        pytest.raises(sqlite3.IntegrityError, match="invalid Ollama provisioning"),
    ):
        connection.execute(
            """UPDATE ollama_leases
               SET ambiguous_create_candidate_id = 'foreign-endpoint',
                   revision = revision + 1,
                   writer_epoch = writer_epoch + 1
               WHERE lease_id = ?""",
            (request.lease_id,),
        )
    state["drop_on_get"] = ()
    clock.advance(1)

    restarted = OllamaLeaseService(
        repository=SQLiteOllamaLeaseRepository(repository.path),
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    completed = (await restarted.reconcile())[0]

    assert completed.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert completed.provider_resource_id == "endpoint-evidence-1"
    assert completed.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert completed.creation_uncertain is False
    assert completed.provisioning_receipt is not None
    assert completed.provisioning_receipt.reconciled_after_ambiguous_create is True


@pytest.mark.asyncio
async def test_native_discovered_name_never_authorizes_cleanup_without_bound_get(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["ambiguous_post"] = True
    state["inventory_visible"] = True
    state["read_status"] = 503
    clock = MutableClock()
    provider._clock = clock
    repository = SQLiteOllamaLeaseRepository(tmp_path / "ambiguous-read.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )

    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    discovered = (await service.reconcile())[0]
    assert discovered.provider_resource_id is None
    assert discovered.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert discovered.creation_uncertain is True

    clock.advance(int((discovered.hard_deadline - clock()).total_seconds()) + 1)
    pending = (await service.reconcile())[0]

    assert pending.state is OllamaLeaseState.RELEASING
    assert pending.creation_uncertain is True
    assert not any(method == "DELETE" for method, _path in state["calls"])

    state["read_status"] = None
    resolved = (await service.reconcile())[0]

    assert resolved.state is OllamaLeaseState.TERMINATED
    assert resolved.provider_resource_id == "endpoint-evidence-1"
    assert resolved.ambiguous_create_candidate_id == "endpoint-evidence-1"
    assert resolved.provisioning_receipt is not None
    assert resolved.provisioning_receipt.reconciled_after_ambiguous_create is True
    assert state["calls"][-2:] == [
        ("DELETE", "/v2/serverless/endpoint-evidence-1"),
        ("GET", "/v2/serverless/endpoint-evidence-1"),
    ]


@pytest.mark.asyncio
async def test_live_pod_zero_rate_is_exact_evidence_not_missing(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    provider = _provider(client=client)
    clock = MutableClock()
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    plan = pod_plan(rate=0.5)
    prepared = provider.validate_runtime_request(
        request, plan, provision_requested_at=clock()
    )
    resource = await provider.provision(prepared)
    client.pod = replace(client.pod, status="RUNNING", cost_per_hr=Decimal(0))

    receipt = await provider.observe_provisioning(prepared.intent, resource)

    assert receipt.pod_realized_placement is not None
    assert receipt.pod_realized_placement.hourly_rate_usd == Decimal(0)
    assert receipt.provider_rate_precision.value == "parsed_json_number"


@pytest.mark.asyncio
async def test_compatible_pod_readbacks_ignore_poll_local_placement_time(
    monkeypatch,
    tmp_path,
):
    """Repeated exact-ID GETs retain one immutable provisioning authority."""

    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    clock = MutableClock()
    provider = _provider(client=client)
    provider._clock = clock
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    plan = pod_plan(rate=0.5)
    repository = SQLiteOllamaLeaseRepository(tmp_path / "stable-readback.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    admitted = await service.acquire(request, plan=plan, wait_until_ready=False)
    first = admitted.provisioning_receipt
    intent = admitted.provisioning_intent
    assert first is not None
    assert intent is not None
    resource = ProvisionedOllamaResource(
        resource_type=intent.resource_type,
        provider_resource_id=first.provider_resource_id,
        resource_name=first.resource_name,
    )
    clock.advance(1)
    second = await provider.observe_provisioning(intent, resource)

    assert first.readback_at != second.readback_at
    assert first.pod_realized_placement is not None
    assert second.pod_realized_placement is not None
    assert (
        first.pod_realized_placement.observed_at
        != second.pod_realized_placement.observed_at
    )
    assert first.readback_sha256 == second.readback_sha256
    assert _same_provisioning_receipt_authority(first, second)
    retained = repository.record_provisioning_outcome(
        admitted, now=clock(), receipt=second
    )
    assert retained.provisioning_receipt == first


def test_invalid_endpoint_mount_member_does_not_invent_expected_volume_absence():
    assessment = assess_endpoint_mounts(
        {"mounts": {"network": [17], "global": []}},
        ("approved-volume",),
    )

    assert assessment.invalid_kinds == ("network",)
    assert assessment.missing_kinds == ()
    assert assessment.mismatched_kinds == ()


def test_invalid_endpoint_mount_member_preserves_independent_foreign_volume():
    assessment = assess_endpoint_mounts(
        {
            "mounts": {
                "network": [17, {"volumeId": "foreign-volume"}],
                "global": [],
            }
        },
        ("approved-volume",),
    )

    assert assessment.invalid_kinds == ("network",)
    assert assessment.missing_kinds == ()
    assert assessment.mismatched_kinds == ("network",)


@pytest.mark.parametrize(
    ("assessment", "expected_invalid", "expected_mismatch"),
    [
        (
            lambda: assess_endpoint_mounts(
                {
                    "mounts": {
                        "network": [{"volumeId": "accepted-a"}, 17],
                        "global": [],
                    }
                },
                ("accepted-a", "accepted-b"),
            ),
            ("network",),
            (),
        ),
        (
            lambda: assess_pod_mounts(
                {
                    "mounts": {
                        "network": [
                            {"volumeId": "accepted-a", "path": "/a"},
                            17,
                        ]
                    }
                },
                {
                    "network": [
                        {"volumeId": "accepted-a", "path": "/a"},
                        {"volumeId": "accepted-b", "path": "/b"},
                    ]
                },
            ),
            ("network",),
            (),
        ),
        (
            lambda: assess_endpoint_mounts(
                {
                    "mounts": {
                        "network": [],
                        "global": [17],
                    }
                },
                (),
            ),
            ("global",),
            (),
        ),
    ],
)
def test_partially_typed_allowed_mounts_do_not_invent_missing_accepted_members(
    assessment, expected_invalid, expected_mismatch
):
    result = assessment()

    assert result.invalid_kinds == expected_invalid
    assert result.mismatched_kinds == expected_mismatch


@pytest.mark.parametrize(
    "assessment",
    [
        lambda: assess_endpoint_mounts(
            {
                "mounts": {
                    "network": [
                        {"volumeId": "accepted-a"},
                        {"volumeId": "accepted-b"},
                    ],
                    "global": [],
                }
            },
            ("accepted-a", "accepted-b"),
        ),
        lambda: assess_pod_mounts(
            {
                "mounts": {
                    "network": [
                        {"volumeId": "accepted-a", "path": "/a"},
                        {"volumeId": "accepted-b", "path": "/b"},
                    ]
                }
            },
            {
                "network": [
                    {"volumeId": "accepted-a", "path": "/a"},
                    {"volumeId": "accepted-b", "path": "/b"},
                ]
            },
        ),
    ],
)
def test_fully_typed_allowed_mounts_remain_comparable(assessment):
    result = assessment()

    assert result.invalid_kinds == ()
    assert result.mismatched_kinds == ()


@pytest.mark.parametrize(
    "assessment",
    [
        lambda: assess_endpoint_mounts(
            {
                "mounts": {
                    "network": [
                        {"volumeId": "accepted-a"},
                        {"volumeId": "confirmed-foreign"},
                    ],
                    "global": [],
                }
            },
            ("accepted-a", "accepted-b"),
        ),
        lambda: assess_pod_mounts(
            {
                "mounts": {
                    "network": [
                        {"volumeId": "accepted-a", "path": "/a"},
                        {"volumeId": "confirmed-foreign", "path": "/foreign"},
                    ]
                }
            },
            {
                "network": [
                    {"volumeId": "accepted-a", "path": "/a"},
                    {"volumeId": "accepted-b", "path": "/b"},
                ]
            },
        ),
    ],
)
def test_fully_typed_foreign_mounts_remain_confirmed_mismatches(assessment):
    result = assessment()

    assert result.invalid_kinds == ()
    assert result.mismatched_kinds == ("network",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actual_rate", "expected_state"),
    [
        (0, OllamaLeaseState.WAITING_FOR_MODEL),
        (None, OllamaLeaseState.RECONCILE_REQUIRED),
        (0.6, OllamaLeaseState.TERMINATED),
    ],
)
async def test_native_pod_rate_distinguishes_zero_missing_and_over_ceiling(
    native_provisioning,
    tmp_path,
    monkeypatch,
    actual_rate,
    expected_state,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["pod_cost"] = actual_rate
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "pod-rate.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=5.0,
    )

    if expected_state is not OllamaLeaseState.WAITING_FOR_MODEL:
        expected_error = (
            "exceeds or differs"
            if expected_state is OllamaLeaseState.TERMINATED
            else "omitted realized placement or rate"
        )
        with pytest.raises(RunPodManagerError, match=expected_error):
            await service.acquire(
                request, plan=pod_plan(rate=0.5), wait_until_ready=False
            )
    else:
        await service.acquire(request, plan=pod_plan(rate=0.5), wait_until_ready=False)

    durable = repository.get(request.lease_id)
    assert durable.state is expected_state
    assert durable.route_url is None
    if expected_state is OllamaLeaseState.TERMINATED:
        assert durable.provisioning_mismatch is not None
        assert durable.provisioning_mismatch.reason.value == "pod_placement_or_rate"
        assert durable.provisioning_receipt is None
        assert state["calls"] == [
            ("POST", "/v2/pods"),
            ("GET", "/v2/pods/pod-evidence-1"),
            ("POST", "/v2/pods/pod-evidence-1/action"),
            ("GET", "/v2/pods/pod-evidence-1"),
        ]
    elif expected_state is OllamaLeaseState.RECONCILE_REQUIRED:
        assert durable.provisioning_mismatch is None
        assert durable.provisioning_receipt is None
        assert state["calls"] == [
            ("POST", "/v2/pods"),
            ("GET", "/v2/pods/pod-evidence-1"),
        ]
    else:
        assert durable.provisioning_receipt is not None
        placement = durable.provisioning_receipt.pod_realized_placement
        assert placement is not None
        assert placement.hourly_rate_usd == Decimal(0)


@pytest.mark.asyncio
async def test_native_missing_cuda_is_unknown_but_reported_below_floor_is_mismatch(
    native_provisioning, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    provider, state = native_provisioning
    state["replace_on_get"] = {"cudaVersion": None}
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "cuda-evidence.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    request = replace(
        request,
        constraints=replace(request.constraints, min_cuda_version="12.0"),
    )

    with pytest.raises(RunPodManagerError, match="omitted CUDA"):
        await service.acquire(request, plan=pod_plan(rate=0.5), wait_until_ready=False)

    unknown = repository.get(request.lease_id)
    assert unknown.state is OllamaLeaseState.RECONCILE_REQUIRED
    assert unknown.provider_resource_id == "pod-evidence-1"
    assert unknown.provisioning_mismatch is None
    assert not any(path.endswith("/action") for _method, path in state["calls"])

    state["replace_on_get"] = {"cudaVersion": "11.8"}
    mismatch = (await service.reconcile())[0]

    assert mismatch.state is OllamaLeaseState.TERMINATED
    assert mismatch.provisioning_mismatch is not None
    assert mismatch.provisioning_mismatch.reason.value == "runtime"
    assert state["calls"][-2:] == [
        ("POST", "/v2/pods/pod-evidence-1/action"),
        ("GET", "/v2/pods/pod-evidence-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,error",
    [
        ({"cost_per_hr": None}, "omitted realized placement or rate"),
        ({"status": "EXITED", "cost_per_hr": Decimal(0)}, "not live"),
        ({"status": "TERMINATED", "cost_per_hr": Decimal(0)}, "not live"),
        ({"data_center_id": None}, "omitted realized placement or rate"),
        ({"cost_per_hr": Decimal("0.5000000000000000001")}, "exceeds"),
    ],
)
async def test_pod_readback_never_invents_missing_or_stopped_realized_rate(
    monkeypatch, changes, error
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    provider = _provider(client=client)
    clock = MutableClock()
    request = make_request(clock, mode=OllamaLeaseMode.DEDICATED_POD)
    plan = pod_plan(rate=0.5)
    prepared = provider.validate_runtime_request(
        request, plan, provision_requested_at=clock()
    )
    resource = await provider.provision(prepared)
    client.pod = replace(client.pod, **changes)

    with pytest.raises(RunPodManagerError, match=error):
        await provider.observe_provisioning(prepared.intent, resource)


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", list(OllamaResourceType))
@pytest.mark.parametrize(
    ("read_status", "expected_error"),
    [
        (200, OllamaLeaseTeardownError),
        (401, OllamaLeaseAuthorizationError),
        (403, OllamaLeaseAuthorizationError),
        (404, None),
        (503, OllamaLeaseTeardownError),
    ],
)
async def test_native_teardown_requires_exact_authenticated_absence(
    native_teardown,
    resource_type,
    read_status,
    expected_error,
):
    provider, state = native_teardown
    state["read_status"] = read_status
    resource = ProvisionedOllamaResource(resource_type, "original-1", "original")
    if expected_error is None:
        await provider.teardown(resource, _cleanup_intent(resource))
    else:
        with pytest.raises(expected_error) as caught:
            await provider.teardown(resource, _cleanup_intent(resource))
        if read_status in {401, 403, 503}:
            assert isinstance(caught.value.__cause__, RunPodAPIError)
            assert caught.value.__cause__.status_code == read_status
    path = (
        "/v2/serverless/original-1"
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else "/v2/pods/original-1"
    )
    action = (
        ("DELETE", path)
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else ("POST", path + "/action")
    )
    assert state["calls"] == [action, ("GET", path)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_type", OllamaResourceType.POD),
        ("provider_resource_id", "different-resource"),
        ("resource_name", "different-name"),
    ],
)
def test_cleanup_intent_validates_one_exact_resource(field, value):
    resource = ProvisionedOllamaResource(
        OllamaResourceType.SERVERLESS_ENDPOINT,
        "original-1",
        "original",
    )
    cleanup_intent = _cleanup_intent(resource)

    cleanup_intent.validate_resource(resource)
    with pytest.raises(ValueError, match="does not bind its resource"):
        cleanup_intent.validate_resource(replace(resource, **{field: value}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_type", OllamaResourceType.POD),
        ("provider_resource_id", "different-resource"),
        ("resource_name", "different-name"),
    ],
)
async def test_native_teardown_rejects_unbound_resource_before_provider_io(
    native_teardown,
    field,
    value,
):
    provider, state = native_teardown
    resource = ProvisionedOllamaResource(
        OllamaResourceType.SERVERLESS_ENDPOINT,
        "original-1",
        "original",
    )
    cleanup_intent = _cleanup_intent(resource)

    with pytest.raises(OllamaLeaseConflictError, match="resource is unbound"):
        await provider.teardown(
            replace(resource, **{field: value}),
            cleanup_intent,
        )

    assert state["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [OllamaLeaseMode.AUTO, OllamaLeaseMode.SERVERLESS_LOAD_BALANCER]
)
@pytest.mark.parametrize("invalid", ["image", "allowlist", "credential"])
async def test_native_precreate_validation_never_admits_uncertain_capacity(
    native_teardown, tmp_path, monkeypatch, mode, invalid
):
    provider, state = native_teardown
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    if invalid == "image":
        monkeypatch.setenv("TEST_OLLAMA_IMAGE", "unreviewed:latest")
    elif invalid == "allowlist":
        provider.deployment.profile.env = {
            "KESTREL_OLLAMA_ALLOWED_MODELS": f"different:8b@{_TEST_DIGEST}"
        }
    else:
        provider = _provider(client=provider.client, serverless_api_key=None)
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "precreate.sqlite3")
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock, mode=mode)
    with pytest.raises(RunPodManagerError):
        await service.acquire(request, plan=serverless_plan(), wait_until_ready=False)
    lease = repository.get(request.lease_id)
    assert lease.state is OllamaLeaseState.FAILED
    assert lease.provision_attempts == 0
    assert lease.provisioning_started_at is None
    assert lease.provision_attempt_id is None
    assert lease.creation_uncertain is False
    assert lease.provider_resource_id is None
    assert state["calls"] == []
    clock.advance(10_000)
    for _ in range(3):
        await service.reconcile()
    assert repository.get(request.lease_id).creation_uncertain is False
    assert state["calls"] == []


@pytest.mark.asyncio
async def test_native_delete_404_is_not_absence_if_exact_get_still_finds_resource(
    native_teardown,
):
    provider, state = native_teardown
    state.update(delete_status=404, read_status=200)
    with pytest.raises(RunPodManagerError, match="still present"):
        resource = ProvisionedOllamaResource(
            OllamaResourceType.SERVERLESS_ENDPOINT,
            "original-1",
            "original",
        )
        await provider.teardown(resource, _cleanup_intent(resource))
    assert state["calls"] == [
        ("DELETE", "/v2/serverless/original-1"),
        ("GET", "/v2/serverless/original-1"),
    ]


@pytest.mark.asyncio
async def test_serverless_idle_limit_falls_back_only_when_auto():
    provider = _provider()
    request = make_request(MutableClock(), idle_timeout_seconds=3601)

    plan = await provider.plan(request)

    assert plan.mode is OllamaLeaseMode.DEDICATED_POD


@pytest.mark.asyncio
async def test_plan_excludes_product_without_scoped_credential():
    provider = _provider(serverless_api_key=None)

    plan = await provider.plan(make_request(MutableClock()))

    assert plan.mode is OllamaLeaseMode.DEDICATED_POD


@pytest.mark.asyncio
async def test_pod_provision_injects_only_workload_scoped_token(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    client = _ControlClient()
    provider = _provider(client=client)
    request = make_request(MutableClock(), mode=OllamaLeaseMode.DEDICATED_POD)
    plan = await provider.plan(request)

    prepared = provider.validate_runtime_request(
        request, plan, provision_requested_at=provider._clock()
    )
    await provider.provision(prepared)

    assert client.pod_request.env["KESTREL_OLLAMA_BEARER_TOKEN"] == _POD_TOKEN
    assert client.pod_request.env["KESTREL_OLLAMA_MODEL_STORAGE_PATH"] == "/models"
    assert _SERVERLESS_TOKEN not in client.pod_request.env.values()


@pytest.mark.asyncio
async def test_volume_cache_paths_follow_runpod_product_conventions(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", _TEST_IMAGE)
    profile = replace(_profile(), network_volume_id="volume-1")
    deployment = replace(
        _deployment(),
        profile=profile,
        serverless_non_compute_cost=non_compute_cost_policy(
            include_network_volume=True
        ),
        pod_non_compute_cost=non_compute_cost_policy(include_network_volume=True),
    )

    serverless_client = _ControlClient()
    serverless = _provider(client=serverless_client, deployment=deployment)
    serverless_request = make_request(
        MutableClock(),
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        max_authorized_cost=5.0,
    )
    serverless_plan = await serverless.plan(serverless_request)
    serverless_prepared = serverless.validate_runtime_request(
        serverless_request,
        serverless_plan,
        provision_requested_at=serverless._clock(),
    )
    await serverless.provision(serverless_prepared)
    assert serverless_client.endpoint_request.network_volume_ids == ("volume-1",)
    assert (
        serverless_client.endpoint_request.env["KESTREL_OLLAMA_MODEL_STORAGE_PATH"]
        == "/runpod-volume/ollama"
    )

    pod_client = _ControlClient()
    pod = _provider(client=pod_client, deployment=deployment)
    pod_request = make_request(MutableClock(), mode=OllamaLeaseMode.DEDICATED_POD)
    pod_plan = await pod.plan(pod_request)
    pod_prepared = pod.validate_runtime_request(
        pod_request,
        pod_plan,
        provision_requested_at=pod._clock(),
    )
    await pod.provision(pod_prepared)
    assert pod_client.pod_request.to_payload()["mounts"] == {
        "network": [{"volumeId": "volume-1", "path": "/workspace"}]
    }
    assert pod_client.pod_request.env["KESTREL_OLLAMA_MODEL_STORAGE_PATH"] == (
        "/workspace/ollama"
    )


def test_persistent_pod_volume_is_not_billed_as_a_network_volume():
    """``volume_gb`` is container-persistent storage, not a network volume.

    A Pod's ``volume_gb`` maps to ``mounts.persistent`` and is priced under
    CONTAINER_DISK, which every policy already covers. NETWORK_VOLUME must
    mean exactly one thing — an explicitly attached ``network_volume_id`` —
    otherwise operators are forced to declare a component their deployment
    never attaches, and the only signal that a shared network volume is in
    play stops being trustworthy.
    """
    persistent_only = replace(_profile(), volume_gb=50, network_volume_id=None)

    deployment = replace(
        _deployment(),
        profile=persistent_only,
        serverless_non_compute_cost=non_compute_cost_policy(),
        pod_non_compute_cost=non_compute_cost_policy(),
    )
    assert deployment.profile.volume_gb == 50
    assert deployment.profile.network_volume_id is None

    # Declaring NETWORK_VOLUME without one attached is now the error, because
    # the policy must cover the *exact* configured components.
    with pytest.raises(ValueError, match="does not cover the exact"):
        replace(
            _deployment(),
            profile=persistent_only,
            serverless_non_compute_cost=non_compute_cost_policy(),
            pod_non_compute_cost=non_compute_cost_policy(include_network_volume=True),
        )


def test_attached_network_volume_still_requires_the_component():
    """The real signal keeps working: an attached volume must be declared."""
    attached = replace(_profile(), volume_gb=0, network_volume_id="volume-1")

    with pytest.raises(ValueError, match="does not cover the exact"):
        replace(
            _deployment(),
            profile=attached,
            serverless_non_compute_cost=non_compute_cost_policy(),
            pod_non_compute_cost=non_compute_cost_policy(),
        )

    deployment = replace(
        _deployment(),
        profile=attached,
        serverless_non_compute_cost=non_compute_cost_policy(
            include_network_volume=True
        ),
        pod_non_compute_cost=non_compute_cost_policy(include_network_volume=True),
    )
    assert deployment.profile.network_volume_id == "volume-1"
