"""Runpod v2 adapter tests with in-memory HTTP/control-plane doubles."""

import httpx
import pytest
from ollama_test_support import MutableClock, make_request

from kestrel_cloud_runpod.models import (
    Availability,
    ComputeProduct,
    EndpointResource,
    FlashBoot,
    GPUOffer,
    GPUProfile,
    PodResource,
    RunPodAPIError,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseMode,
    OllamaResourceType,
    ProvisionedOllamaResource,
)
from kestrel_cloud_runpod.ollama_provider import (
    RunpodOllamaCapacityProvider,
    RunpodOllamaDeployment,
    _pod_base_url,
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
        env={"OLLAMA_HOST": "0.0.0.0:11434"},
    )


def _deployment() -> RunpodOllamaDeployment:
    return RunpodOllamaDeployment(
        profile=_profile(),
        serverless_workers_max=1,
        serverless_request_count=1,
        serverless_execution_timeout_ms=300_000,
        serverless_flashboot=FlashBoot.FLASHBOOT,
        http_timeout_seconds=10,
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
        secure_price_per_hr=1.0 if product is ComputeProduct.SERVERLESS else 0.5,
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
        self.endpoint = EndpointResource(
            id="endpoint-1",
            name="resource",
            endpoint_type="LOAD_BALANCER",
            request_urls={
                "base": "https://endpoint.example",
                "health": "https://endpoint.example/ping",
            },
            raw={},
        )

    def list_gpus(self, *, products, **kwargs):
        del kwargs
        return (_offer(products[0]),)

    def create_endpoint(self, request):
        self.endpoint_request = request
        return self.endpoint

    def create_pod(self, request):
        self.pod_request = request
        return PodResource(
            id="pod-1",
            name="resource",
            status="CREATED",
            gpu_id=request.gpu_id,
            gpu_count=request.gpu_count,
            cost_per_hr=0.5,
            raw={},
        )

    def get_endpoint(self, endpoint_id):
        assert endpoint_id == "endpoint-1"
        return self.endpoint

    def get_pod(self, pod_id):
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
            raise self.delete_error
        self.actions.append((endpoint_id, "delete"))

    def pod_action(self, pod_id, action):
        self.actions.append((pod_id, action))


def _provider(
    *,
    client=None,
    transport=None,
    serverless_api_key="serverless-secret",
    pod_bearer_token="pod-secret",
):
    return RunpodOllamaCapacityProvider(
        client=client or _ControlClient(),
        deployment=_deployment(),
        serverless_api_key=serverless_api_key,
        pod_bearer_token=pod_bearer_token,
        http_transport=transport,
    )


@pytest.mark.asyncio
async def test_plan_uses_product_specific_live_catalog_prices(monkeypatch):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", "registry.example/ollama:sha")
    client = _ControlClient()
    provider = _provider(client=client)

    plan = await provider.plan(make_request(MutableClock()))

    assert plan.mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    assert plan.placement.gpu_id == "gpu-serverless"


@pytest.mark.asyncio
async def test_serverless_provision_is_load_balanced_and_configuration_owned(
    monkeypatch,
):
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", "registry.example/ollama:sha")
    monkeypatch.setenv("HOST_SECRET_THAT_MUST_NOT_EXPAND", "leaked-secret")
    client = _ControlClient()
    provider = _provider(client=client)
    request = make_request(
        MutableClock(), model="qwen-${HOST_SECRET_THAT_MUST_NOT_EXPAND}:8b"
    )
    plan = await provider.plan(request)

    resource = await provider.provision(
        request=request, plan=plan, resource_name="kestrel-ollama-test"
    )

    assert resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
    assert client.endpoint_request.endpoint_type == "LOAD_BALANCER"
    assert client.endpoint_request.image == "registry.example/ollama:sha"
    assert (
        client.endpoint_request.env["OLLAMA_MODELS_PULL"]
        == "qwen-${HOST_SECRET_THAT_MUST_NOT_EXPAND}:8b"
    )
    assert "leaked-secret" not in client.endpoint_request.env.values()
    assert client.endpoint_request.scaling["type"] == "REQUEST_COUNT"


@pytest.mark.asyncio
async def test_observe_requires_provider_health_and_reads_exact_model():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer serverless-secret"
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
        assert request.headers["Authorization"] == "Bearer pod-secret"
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
async def test_model_pull_is_bounded_nonstreaming_request():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"status": "success"})

    provider = _provider(transport=httpx.MockTransport(handler))
    resource = ProvisionedOllamaResource(
        resource_type=OllamaResourceType.POD,
        provider_resource_id="pod-1",
        resource_name="resource",
    )

    await provider.pull_model(resource, "https://pod.example", "qwen3:8b")

    assert captured[0].url.path == "/api/pull"
    assert captured[0].headers["Authorization"] == "Bearer pod-secret"
    assert captured[0].content == b'{"name":"qwen3:8b","stream":false}'


@pytest.mark.asyncio
async def test_teardown_uses_delete_for_endpoint_and_terminate_for_pod():
    client = _ControlClient()
    provider = _provider(client=client)

    await provider.teardown(
        ProvisionedOllamaResource(
            resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
            provider_resource_id="endpoint-1",
            resource_name="endpoint",
        )
    )
    await provider.teardown(
        ProvisionedOllamaResource(
            resource_type=OllamaResourceType.POD,
            provider_resource_id="pod-1",
            resource_name="pod",
        )
    )

    assert client.actions == [("endpoint-1", "delete"), ("pod-1", "terminate")]


@pytest.mark.asyncio
async def test_teardown_treats_already_absent_resource_as_success():
    client = _ControlClient()
    client.delete_error = RunPodAPIError(
        title="Not found", detail="Endpoint is absent", status_code=404
    )
    provider = _provider(client=client)

    await provider.teardown(
        ProvisionedOllamaResource(
            resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
            provider_resource_id="endpoint-1",
            resource_name="endpoint",
        )
    )


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
    monkeypatch.setenv("TEST_OLLAMA_IMAGE", "registry.example/ollama:sha")
    client = _ControlClient()
    provider = _provider(client=client)
    request = make_request(MutableClock(), mode=OllamaLeaseMode.DEDICATED_POD)
    plan = await provider.plan(request)

    await provider.provision(
        request=request, plan=plan, resource_name="kestrel-ollama-test"
    )

    assert client.pod_request.env["KESTREL_OLLAMA_BEARER_TOKEN"] == "pod-secret"
    assert "serverless-secret" not in client.pod_request.env.values()
