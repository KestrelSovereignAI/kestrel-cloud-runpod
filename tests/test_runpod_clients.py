"""HTTP and request-shape contracts for both Runpod v2 services."""

from __future__ import annotations

import json

import httpx
import pytest

from kestrel_cloud_runpod.clients import (
    DEFAULT_USER_AGENT,
    RunpodControlPlaneClient,
    RunpodServerlessClient,
)
from kestrel_cloud_runpod.models import (
    CloudType,
    ComputeProduct,
    EndpointCreateRequest,
    EndpointUpdateRequest,
    PodCreateRequest,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)


def _gpu(**overrides):
    value = {
        "id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "name": "RTX PRO 6000",
        "pool": "BLACKWELL_48",
        "manufacturer": "NVIDIA",
        "memory": 48,
        "secure": True,
        "community": False,
        "price": {"secure": 1.75, "community": 0.0},
        "maxCount": {"secure": 2, "community": 0},
        "availability": "HIGH",
        "dataCenters": [{"id": "US-TX-3", "name": "Texas", "availability": "HIGH"}],
    }
    value.update(overrides)
    return value


def _pod(**overrides):
    value = {
        "id": "pod-123",
        "name": "kestrel-image",
        "status": "PROVISIONING",
        "gpu": {"id": "gpu-id", "count": 1},
        "cost": 0.9,
    }
    value.update(overrides)
    return value


def _endpoint(**overrides):
    value = {
        "id": "ep-123",
        "name": "catalog-worker",
        "type": "QUEUE",
        "requestUrls": {"run": "https://api.runpod.ai/v2/ep-123/run"},
    }
    value.update(overrides)
    return value


def test_control_client_auth_user_agent_catalog_query_and_timeouts():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"gpus": [_gpu()]})

    client = RunpodControlPlaneClient(
        api_key="secret",
        user_agent="kite-agent/runpod",
        connect_timeout=3.0,
        read_timeout=17.0,
        http_transport=httpx.MockTransport(handler),
    )
    offers = client.list_gpus(
        products=(ComputeProduct.SERVERLESS,),
        cloud=CloudType.SECURE,
        min_cuda_version="12.8",
    )

    assert offers[0].pool == "BLACKWELL_48"
    assert offers[0].availability_min_cuda_version == "12.8"
    request = seen[0]
    assert request.headers["Authorization"] == "Bearer secret"
    assert request.headers["User-Agent"] == "kite-agent/runpod"
    assert request.url.path == "/v2/catalog/gpus"
    assert dict(request.url.params) == {
        "include": "AVAILABILITY",
        "product": "SERVERLESS",
        "count": "1",
        "cloud": "SECURE",
        "minCudaVersion": "12.8",
    }
    assert client.transport._client.timeout.connect == 3.0
    assert client.transport._client.timeout.read == 17.0


def test_default_user_agent_is_explicit_and_non_generic():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"pods": []})

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    client.list_pods()

    assert seen[0].headers["User-Agent"] == DEFAULT_USER_AGENT
    assert DEFAULT_USER_AGENT != "Python-urllib/3"


def test_catalog_availability_requires_one_product_context():
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"gpus": []})
        ),
    )
    with pytest.raises(ValueError, match="exactly one product"):
        client.list_gpus(products=(ComputeProduct.POD, ComputeProduct.SERVERLESS))


def test_rfc9457_problem_is_typed_and_does_not_expose_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            headers={
                "Content-Type": "application/problem+json",
                "RateLimit": "limit=100, remaining=99, reset=30",
                "RateLimit-Policy": "100;w=60",
            },
            json={
                "title": "Unprocessable Entity",
                "status": 422,
                "detail": "invalid GPU pool at https://storage.test/file?signature=secret",
                "errors": ["$.gpu.pools: unknown pool; api_key=must-not-leak"],
                "secret": "must-not-leak",
            },
        )

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RunPodAPIError) as raised:
        client.create_endpoint(
            EndpointCreateRequest(
                name="test",
                image="example/image:sha",
                gpu_pools=("bad",),
                endpoint_type="QUEUE",
                scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
            )
        )

    error = raised.value
    assert error.status_code == 422
    assert error.detail == "invalid GPU pool at [REDACTED_URL]"
    assert error.errors == ("$.gpu.pools: unknown pool; api_key=[REDACTED]",)
    assert error.rate_limit.raw == "limit=100, remaining=99, reset=30"
    assert error.rate_limit.policy == "100;w=60"
    assert "must-not-leak" not in str(error)


def test_safe_get_retries_using_retry_after_and_records_rate_limit():
    calls = 0
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={
                    "Retry-After": "2",
                    "RateLimit": "limit=10, remaining=0, reset=2",
                    "RateLimit-Policy": "10;w=60",
                },
                json={
                    "title": "Too Many Requests",
                    "status": 429,
                    "detail": "slow down",
                },
            )
        return httpx.Response(
            200,
            headers={"RateLimit": "limit=10, remaining=9, reset=60"},
            json={"pods": []},
        )

    client = RunpodControlPlaneClient(
        api_key="secret",
        sleep=delays.append,
        http_transport=httpx.MockTransport(handler),
    )
    assert client.list_pods() == ()
    assert calls == 2
    assert delays == [2.0]
    assert client.transport.last_rate_limit.raw == "limit=10, remaining=9, reset=60"


def test_safe_get_uses_rate_limit_reset_when_retry_after_is_absent():
    calls = 0
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"RateLimit": "limit=10, remaining=0, reset=3.5"},
                json={"title": "Rate limited", "status": 429, "detail": "wait"},
            )
        return httpx.Response(200, json={"pods": []})

    client = RunpodControlPlaneClient(
        api_key="secret",
        sleep=delays.append,
        http_transport=httpx.MockTransport(handler),
    )
    client.list_pods()

    assert calls == 2
    assert delays == [3.5]


def test_create_is_not_retried_after_ambiguous_server_error():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            503,
            json={"title": "Unavailable", "status": 503, "detail": "unknown result"},
        )

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RunPodAmbiguousResultError) as raised:
        client.create_pod(
            PodCreateRequest(name="test", image="example/image:sha", gpu_id="gpu-id")
        )

    assert calls == 1
    assert raised.value.reconcile_required is True


def test_create_timeout_is_ambiguous_and_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RunPodAmbiguousResultError) as raised:
        client.create_pod(
            PodCreateRequest(name="test", image="image:sha", gpu_id="gpu-id")
        )

    assert calls == 1
    assert raised.value.detail == "ReadTimeout"


def test_pod_action_timeout_is_ambiguous_and_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RunPodAmbiguousResultError) as raised:
        client.pod_action("pod-1", "stop")

    assert calls == 1
    assert raised.value.resource == "/pods/pod-1/action"


def test_invalid_list_envelope_fails_clearly():
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"items": []})
        ),
    )
    with pytest.raises(RunPodManagerError, match="expected 'pods'"):
        client.list_pods()


def test_pod_lifecycle_request_shapes():
    seen = []
    responses = [
        httpx.Response(
            201,
            json=_pod(
                env={"HF_TOKEN": "hf-secret", "MODEL": "qwen"},
                authorization="must-not-leak",
            ),
        ),
        httpx.Response(200, json=_pod(status="RUNNING")),
        httpx.Response(200, json={"pods": [_pod()]}),
        httpx.Response(200, json=_pod(status="EXITED")),
        httpx.Response(204),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    created = client.create_pod(
        PodCreateRequest(
            name="kestrel",
            image="registry/image:sha",
            gpu_id="gpu-id",
            cloud=CloudType.SECURE,
            ports=("8000/http",),
            env={"MODEL": "qwen"},
        )
    )
    assert created.id == "pod-123"
    assert created.raw["env"] == {
        "HF_TOKEN": "[REDACTED]",
        "MODEL": "[REDACTED]",
    }
    assert created.raw["authorization"] == "[REDACTED]"
    assert client.get_pod("pod-123").status == "RUNNING"
    assert len(client.list_pods()) == 1
    assert client.pod_action("pod-123", "stop").status == "EXITED"
    client.delete_pod("pod-123")

    assert [request.method for request in seen] == [
        "POST",
        "GET",
        "GET",
        "POST",
        "DELETE",
    ]
    assert [request.url.path for request in seen] == [
        "/v2/pods",
        "/v2/pods/pod-123",
        "/v2/pods",
        "/v2/pods/pod-123/action",
        "/v2/pods/pod-123",
    ]
    assert json.loads(seen[0].content)["gpu"] == {"id": "gpu-id", "count": 1}
    assert json.loads(seen[3].content) == {"action": "stop"}


def test_endpoint_worker_and_billing_request_shapes():
    seen = []
    responses = [
        httpx.Response(201, json=_endpoint()),
        httpx.Response(200, json=_endpoint()),
        httpx.Response(200, json={"endpoints": [_endpoint()]}),
        httpx.Response(200, json=_endpoint(name="updated")),
        httpx.Response(200, json={"workers": [], "summary": {"total": 0}}),
        httpx.Response(200, json={"records": [], "metadata": {"recordCount": 0}}),
        httpx.Response(200, json={"records": [], "metadata": {"recordCount": 0}}),
        httpx.Response(204),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    request = EndpointCreateRequest(
        name="catalog",
        image="registry/catalog:sha",
        gpu_pools=("BLACKWELL_48",),
        endpoint_type="QUEUE",
        scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
        workers_min=0,
        workers_max=2,
    )
    assert client.create_endpoint(request).id == "ep-123"
    assert client.get_endpoint("ep-123").id == "ep-123"
    assert len(client.list_endpoints()) == 1
    assert (
        client.update_endpoint("ep-123", EndpointUpdateRequest(workers_max=1)).name
        == "updated"
    )
    assert client.list_endpoint_workers("ep-123")["summary"]["total"] == 0
    assert client.pod_billing(last_n=7, pod_id="pod-123").records == ()
    assert client.serverless_billing(last_n=7, endpoint_id="ep-123").records == ()
    client.delete_endpoint("ep-123")

    create_body = json.loads(seen[0].content)
    assert create_body["gpu"] == {"pools": ["BLACKWELL_48"], "count": 1}
    assert create_body["type"] == "QUEUE"
    assert create_body["flashboot"] == "FLASHBOOT"
    assert json.loads(seen[3].content) == {"workers": {"max": 1}}
    assert seen[4].url.path == "/v2/serverless/ep-123/workers"
    assert dict(seen[5].url.params) == {"lastN": "7", "podId": "pod-123"}
    assert dict(seen[6].url.params) == {"lastN": "7", "serverlessId": "ep-123"}


def test_pod_and_worker_log_streams_parse_sse_and_send_resume_shape():
    seen = []
    body = (
        "id: 2026-08-01T00:00:00Z/1\n"
        'data: {"source":"container","line":"model ready",'
        '"ts":"2026-08-01T00:00:00Z"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, text=body
        )

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    pod_events = list(
        client.iter_pod_logs(
            "pod-123",
            tail=20,
            source="container",
            since="2026-08-01T00:00:00Z",
            last_event_id="cursor-1",
            stream_window_seconds=1,
        )
    )
    worker_events = list(
        client.iter_worker_logs("ep-123", "worker-456", tail=5, stream_window_seconds=1)
    )

    assert pod_events[0]["line"] == "model ready"
    assert worker_events[0]["source"] == "container"
    assert client.transport.last_sse_event_id == "2026-08-01T00:00:00Z/1"
    assert seen[0].url.path == "/v2/pods/pod-123/logs"
    assert dict(seen[0].url.params) == {
        "tail": "20",
        "source": "container",
        "since": "2026-08-01T00:00:00Z",
    }
    assert seen[0].headers["Last-Event-ID"] == "cursor-1"
    assert seen[1].url.path == "/v2/serverless/ep-123/workers/worker-456/logs"


def test_serverless_job_operations_use_data_plane_auth_and_shapes():
    seen = []
    responses = [
        httpx.Response(200, json={"id": "job-1", "status": "IN_QUEUE"}),
        httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "COMPLETED",
                "output": {"artifact": "gs://bucket/result"},
                "delayTime": 1200,
                "executionTime": 3400,
            },
        ),
        httpx.Response(200, json={"id": "job-1", "status": "CANCELLED"}),
        httpx.Response(200, json={"id": "job-1", "status": "IN_QUEUE"}),
        httpx.Response(200, json={"jobs": {}, "workers": {}}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    client = RunpodServerlessClient(
        api_key="serverless-secret", http_transport=httpx.MockTransport(handler)
    )
    job = client.run(
        "ep-123",
        {"catalogJobId": 42},
        webhook_url="https://example.test/callback",
        execution_timeout_ms=900_000,
        ttl_ms=3_600_000,
    )
    assert job.status == "IN_QUEUE"
    completed = client.status("ep-123", "job-1", ttl_ms=60_000)
    assert completed.output == {"artifact": "gs://bucket/result"}
    assert completed.delay_time_ms == 1200
    assert client.cancel("ep-123", "job-1").status == "CANCELLED"
    assert client.retry("ep-123", "job-1").status == "IN_QUEUE"
    assert client.health("ep-123") == {"jobs": {}, "workers": {}}

    assert seen[0].headers["Authorization"] == "Bearer serverless-secret"
    assert seen[0].url.path == "/v2/ep-123/run"
    assert json.loads(seen[0].content) == {
        "input": {"catalogJobId": 42},
        "webhook": "https://example.test/callback",
        "policy": {"executionTimeout": 900_000, "ttl": 3_600_000},
    }
    assert dict(seen[1].url.params) == {"ttl": "60000"}
    assert [request.method for request in seen[2:4]] == ["POST", "POST"]
    assert seen[4].url.path == "/v2/ep-123/health"


def test_v1_and_non_versioned_base_urls_are_rejected():
    for url in ("https://rest.runpod.io/v1", "https://example.test", "not-a-url"):
        with pytest.raises(ValueError, match="/v2|absolute"):
            RunpodControlPlaneClient(api_key="secret", base_url=url)
