"""Runpod v2 adapter tests with in-memory HTTP/control-plane doubles."""

import asyncio
import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from ollama_test_support import (
    MutableClock,
    make_request,
    non_compute_cost_policy,
    pod_plan,
    serverless_plan,
)

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
    environment_values_sha256,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaCleanupIntent,
    OllamaLeaseMode,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaProvisioningMismatchError,
    OllamaResourceType,
    OllamaTeardownState,
    ProvisionedOllamaResource,
)
from kestrel_cloud_runpod.ollama_provider import (
    RunpodOllamaCapacityProvider,
    RunpodOllamaDeployment,
    _pod_base_url,
)
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
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
    assert await asyncio.to_thread(entered.wait, 5)
    with pytest.raises(TypeError):
        prepared.provider_request.env["KESTREL_OLLAMA_REQUIRED_MODEL"] = "wrong:latest"
    with pytest.raises(TypeError):
        prepared.provider_request.scaling["requestCount"] = 2
    resume.set()
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
async def test_ollama_refuses_unaccepted_global_storage_before_workload_requests():
    client = _ControlClient()
    client.endpoint = replace(
        client.endpoint,
        raw={
            "mounts": {"network": [], "global": [{"volumeId": "unquoted-volume"}]},
        },
    )
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    provider = _provider(client=client, transport=httpx.MockTransport(handler))
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="resource",
    )
    with pytest.raises(RunPodManagerError, match="global mounts"):
        await provider.observe(resource)
    assert seen == []


@pytest.mark.asyncio
async def test_observe_requires_provider_health_and_reads_exact_model():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {_SERVERLESS_TOKEN}"
        if request.url.path == "/ping":
            return httpx.Response(200)
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        provider_resource_id="endpoint-1",
        resource_name="resource",
    )

    observation = await provider.observe(resource)

    assert observation.provider_ready is True
    assert observation.has_model("qwen3:8b")
    assert [request.url.path for request in requests] == ["/ping", "/api/tags"]


@pytest.mark.asyncio
async def test_pod_observation_uses_v2_status_and_runtime_route():
    def handler(request: httpx.Request) -> httpx.Response:
        if "Authorization" not in request.headers:
            return httpx.Response(401)
        assert request.headers["Authorization"] == f"Bearer {_POD_TOKEN}"
        if request.url.path == "/ping":
            return httpx.Response(200)
        return httpx.Response(200, json={"models": []})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.POD,
        provider_resource_id="pod-1",
        resource_name="resource",
    )

    observation = await provider.observe(resource)

    assert observation.provider_ready is True
    assert observation.route_url == "https://pod-1-11434.proxy.runpod.net"


@pytest.mark.asyncio
async def test_pod_initializing_health_is_not_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        if "Authorization" not in request.headers:
            return httpx.Response(401)
        if request.url.path == "/ping":
            return httpx.Response(204)
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    observation = await _provider(transport=httpx.MockTransport(handler)).observe(
        ProvisionedOllamaResource(
            resource_type=OllamaResourceType.POD,
            provider_resource_id="pod-1",
            resource_name="resource",
        )
    )

    assert observation.provider_ready is False
    assert observation.model_names == ()


@pytest.mark.asyncio
async def test_pod_observation_fails_closed_when_route_allows_anonymous_access():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.POD,
        provider_resource_id="pod-1",
        resource_name="resource",
    )

    observation = await provider.observe(resource)

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
async def test_serverless_initializing_health_is_not_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ping":
            return httpx.Response(204)
        return httpx.Response(200, json={"models": []})

    provider = _provider(transport=httpx.MockTransport(handler))
    observation = await provider.observe(
        ProvisionedOllamaResource(
            resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
            provider_resource_id="endpoint-1",
            resource_name="resource",
        )
    )

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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = RunpodControlPlaneClient(
        api_key="local-loopback-only",
        base_url=f"http://127.0.0.1:{server.server_port}/v2",
        max_safe_retries=0,
    )
    try:
        yield _provider(client=client), state
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


@pytest.fixture
def native_provisioning():
    state = {
        "calls": [],
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = RunpodControlPlaneClient(
        api_key="local-loopback-only",
        base_url=f"http://127.0.0.1:{server.server_port}/v2",
        max_safe_retries=0,
    )
    try:
        yield _provider(client=client), state
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


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
        client.create_finish.set()
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
        client.create_finish.set()

    completed = await acquire
    assert completed.state is OllamaLeaseState.TERMINATED
    assert completed.provider_resource_id == "original-endpoint"
    assert completed.ambiguous_create_candidate_id == "candidate-endpoint"
    assert completed.ambiguous_candidate_mismatch is not None
    assert completed.ambiguous_candidate_mismatch.provider_resource_id == (
        "candidate-endpoint"
    )
    assert client.actions == [("original-endpoint", "delete")]


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
    await asyncio.to_thread(client.delete_entered.wait, 5)
    second = asyncio.create_task(
        service.release(
            request.lease_id,
            owner_id=request.owner_id,
            workload_id=request.workload_id,
        )
    )
    try:
        await asyncio.sleep(0.05)
        calls_while_first_delete_is_live = client.delete_calls
    finally:
        client.delete_finish.set()
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
@pytest.mark.parametrize("read_status", [200, 401, 404, 503])
async def test_native_teardown_requires_exact_authenticated_absence(
    native_teardown,
    resource_type,
    read_status,
):
    provider, state = native_teardown
    state["read_status"] = read_status
    resource = ProvisionedOllamaResource(resource_type, "original-1", "original")
    if read_status == 404:
        await provider.teardown(resource, _cleanup_intent(resource))
    else:
        with pytest.raises(RunPodManagerError):
            await provider.teardown(resource, _cleanup_intent(resource))
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
