"""HTTP and request-shape contracts for both Runpod v2 services."""

from __future__ import annotations

import json
import socket
import threading

import httpcore
import httpx
import pytest
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from pod_capacity_test_support import (
    execution_control_carrier,
    native_httpx_connect_failure,
    record_control_injection,
)

from kestrel_cloud_runpod.clients import (
    DEFAULT_USER_AGENT,
    RunpodControlPlaneClient,
    RunpodServerlessClient,
    RunpodTransport,
    is_retryable_runpod_api_error,
)
from kestrel_cloud_runpod.models import (
    CloudType,
    ComputeProduct,
    EndpointCreateRequest,
    EndpointResource,
    EndpointUpdateRequest,
    PodCreateRequest,
    PodResource,
    ResourceFieldState,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)


def _page(key, items, next_cursor=None):
    return {
        key: items,
        "pagination": {
            "nextCursor": next_cursor,
            "hasNextPage": next_cursor is not None,
        },
    }


def _gpu(**overrides):
    value = {
        "id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "name": "RTX PRO 6000",
        "pool": "BLACKWELL_48",
        "manufacturer": "NVIDIA",
        "memory": 48,
        "secure": True,
        "community": False,
        "price": {"secure": 1.75, "community": 0.0, "serverless": 2.50},
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


def _endpoint_request() -> EndpointCreateRequest:
    return EndpointCreateRequest(
        name="catalog",
        image="registry/catalog:sha",
        gpu_pools=("BLACKWELL_48",),
        endpoint_type="QUEUE",
        scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
        workers_min=0,
        workers_max=2,
    )


def _network_volume(**overrides):
    value = {
        "id": "volume-123",
        "name": "model-cache",
        "size": 100,
        "dataCenter": "US-TX-3",
        "type": "HIGH_PERFORMANCE",
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
    assert offers[0].serverless_price_per_hr == 2.50
    assert seen[0].url.host == "api.runpod.io"
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


@pytest.mark.parametrize(
    "method,key,path,factory",
    [
        ("list_pods", "pods", "/v2/pods", _pod),
        ("list_endpoints", "endpoints", "/v2/serverless", _endpoint),
        (
            "list_network_volumes",
            "networkVolumes",
            "/v2/network-volumes",
            _network_volume,
        ),
    ],
)
def test_inventory_walks_all_pages_including_an_empty_intermediate_page(
    method, key, path, factory
):
    seen = []
    cursors = ("opaque+/cursor?one", "second-cursor", None)
    pages = [
        _page(key, [factory(id="first")], cursors[0]),
        _page(key, [], cursors[1]),
        _page(key, [factory(id="last")]),
    ]

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=pages.pop(0))

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    resources = getattr(client, method)()
    assert [resource.id for resource in resources] == ["first", "last"]
    assert [request.url.params.get("cursor") for request in seen] == [
        None,
        cursors[0],
        cursors[1],
    ]
    assert all(request.url.host == "api.runpod.io" for request in seen)
    assert all(request.url.path == path for request in seen)
    if method == "list_pods":
        assert all(
            request.url.params["includeClusterPods"] == "true" for request in seen
        )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"hasNextPage": "false", "nextCursor": None},
        {"hasNextPage": 0, "nextCursor": None},
        {"hasNextPage": False},
        {"hasNextPage": True, "nextCursor": None},
        {"hasNextPage": True, "nextCursor": ""},
        {"hasNextPage": True, "nextCursor": 123},
        {"hasNextPage": False, "nextCursor": "unexpected"},
    ],
)
def test_inventory_rejects_missing_or_ambiguous_pagination(metadata):
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"pods": [], "pagination": metadata}
            )
        ),
    )
    with pytest.raises(RunPodManagerError, match="pagination"):
        client.list_pods()


def test_inventory_rejects_cursor_cycles_instead_of_hanging():
    cursors = iter(("first", "second", "first"))
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_page("pods", [], next(cursors)))
        ),
    )
    with pytest.raises(RunPodManagerError, match="repeated.*cursor"):
        client.list_pods()


def test_inventory_bounds_a_provider_that_keeps_issuing_unique_cursors():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_page("pods", [], f"cursor-{len(seen)}"))

    client = RunpodControlPlaneClient(
        api_key="secret",
        max_inventory_pages=2,
        http_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RunPodManagerError, match="completeness is unknown"):
        client.list_pods()
    assert len(seen) == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_inventory_page_bound_is_strictly_positive_and_integral(limit):
    with pytest.raises(ValueError, match="max_inventory_pages"):
        RunpodControlPlaneClient(api_key="secret", max_inventory_pages=limit)


def test_inventory_rejects_duplicate_resources_across_pages():
    pages = iter((_page("pods", [_pod()], "next"), _page("pods", [_pod()])))
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=next(pages))
        ),
    )
    with pytest.raises(RunPodManagerError, match="Duplicate.*identity"):
        client.list_pods()


def test_inventory_does_not_return_partial_success_after_a_later_page_fails():
    responses = iter(
        (
            httpx.Response(200, json=_page("pods", [_pod()], "next")),
            httpx.Response(503, json={"title": "Unavailable", "detail": "retry later"}),
        )
    )
    client = RunpodControlPlaneClient(
        api_key="secret",
        max_safe_retries=0,
        http_transport=httpx.MockTransport(lambda request: next(responses)),
    )
    with pytest.raises(RunPodAPIError) as raised:
        client.list_pods()
    assert raised.value.status_code == 503


def test_default_user_agent_is_explicit_and_non_generic():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_page("pods", []))

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
        client.create_endpoint(_endpoint_request())

    error = raised.value
    assert error.status_code == 422
    assert error.detail == "invalid GPU pool at [REDACTED_URL]"
    assert error.errors == ("$.gpu.pools: unknown pool; api_key=[REDACTED]",)
    assert error.rate_limit.raw == "limit=100, remaining=99, reset=30"
    assert error.rate_limit.policy == "100;w=60"
    assert "must-not-leak" not in str(error)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (429, True),
        (502, True),
        (503, True),
        (504, True),
        (None, False),
        (200, False),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (500, False),
    ],
)
def test_api_error_retry_disposition_uses_the_transport_policy(status_code, expected):
    error = RunPodAPIError(
        title="Runpod error",
        detail="bounded test",
        status_code=status_code,
    )
    assert is_retryable_runpod_api_error(error) is expected


_PUBLIC_TRANSPORT_PAIRS = (
    (httpx.ConnectTimeout, httpcore.ConnectTimeout),
    (httpx.ReadTimeout, httpcore.ReadTimeout),
    (httpx.WriteTimeout, httpcore.WriteTimeout),
    (httpx.PoolTimeout, httpcore.PoolTimeout),
    (httpx.ConnectError, httpcore.ConnectError),
    (httpx.ReadError, httpcore.ReadError),
    (httpx.WriteError, httpcore.WriteError),
    (httpx.RemoteProtocolError, httpcore.RemoteProtocolError),
)


def _public_transport_pair(
    outer_type: type[httpx.TransportError],
    core_type: type[Exception],
) -> httpx.TransportError:
    request = httpx.Request("GET", "https://api.runpod.io/v2/pods/test")
    outer = outer_type("transport failed", request=request)
    outer.__cause__ = core_type("transport failed")
    return outer


def test_statusless_api_error_requires_the_original_httpx_failure_cause():
    transport = _public_transport_pair(httpx.ConnectError, httpcore.ConnectError)
    try:
        raise RunPodAPIError(
            title="Runpod transport failed",
            detail="ConnectError",
        ) from transport
    except RunPodAPIError as error:
        retained = error

    assert is_retryable_runpod_api_error(retained)

    malformed = RunPodAPIError(
        title="Invalid Runpod response",
        detail="Expected JSON",
    )
    malformed.__cause__ = ValueError("invalid JSON")
    assert not is_retryable_runpod_api_error(malformed)

    ambiguous = RunPodAmbiguousResultError(
        title="Ambiguous mutation",
        detail="provider outcome is unknown",
        status_code=503,
    )
    assert is_retryable_runpod_api_error(ambiguous)

    ambiguous_without_transport = RunPodAmbiguousResultError(
        title="Ambiguous mutation",
        detail="provider outcome is unknown",
    )
    assert not is_retryable_runpod_api_error(ambiguous_without_transport)

    try:
        raise RunPodAmbiguousResultError(
            title="Ambiguous mutation",
            detail="provider transport failed",
        ) from transport
    except RunPodAmbiguousResultError as ambiguous_transport:
        retained_ambiguous = ambiguous_transport
    assert is_retryable_runpod_api_error(retained_ambiguous)

    class UnknownAPIError(RunPodAPIError):
        pass

    unknown = UnknownAPIError(
        title="Unknown subclass",
        detail="provider outcome is unknown",
        status_code=503,
    )
    assert not is_retryable_runpod_api_error(unknown)


def test_statusless_http_response_error_is_not_transport_retry_evidence():
    request = httpx.Request("DELETE", "https://api.runpod.io/v2/serverless/exact")
    response = httpx.Response(401, request=request)
    response_error = httpx.HTTPStatusError(
        "authorization denied", request=request, response=response
    )
    error = RunPodAPIError(
        title="Runpod request failed",
        detail="HTTPStatusError",
    )
    error.__cause__ = response_error

    assert not is_retryable_runpod_api_error(error)


@pytest.mark.parametrize(("outer_type", "core_type"), _PUBLIC_TRANSPORT_PAIRS)
def test_exact_public_transport_pairs_are_retryable(outer_type, core_type):
    transport = _public_transport_pair(outer_type, core_type)
    error = RunPodAPIError(
        title="Runpod transport failed",
        detail=outer_type.__name__,
    )
    error.__cause__ = transport

    assert is_retryable_runpod_api_error(error)


@pytest.mark.parametrize(("outer_type", "core_type"), _PUBLIC_TRANSPORT_PAIRS)
def test_bare_public_httpx_transport_error_does_not_grant_retry_authority(
    outer_type, core_type
):
    del core_type
    request = httpx.Request("DELETE", "https://api.runpod.io/v2/serverless/exact")
    transport = outer_type("transport failed", request=request)
    error = RunPodAPIError(
        title="Runpod transport failed",
        detail=outer_type.__name__,
    )
    error.__cause__ = transport

    assert not is_retryable_runpod_api_error(error)


def test_retryable_status_cannot_hide_an_explicit_structural_cause():
    error = RunPodAPIError(
        title="Runpod request failed",
        detail="structural response failure",
        status_code=503,
    )
    structural = ValueError("invalid response body")
    error.__cause__ = structural

    assert not is_retryable_runpod_api_error(error)


def test_retryable_status_without_explicit_cause_remains_retryable():
    canonical = RunPodAPIError(
        title="Runpod unavailable",
        detail="retry later",
        status_code=503,
    )
    assert is_retryable_runpod_api_error(canonical)


def _native_httpx_stream_failure(kind: str) -> httpx.TransportError:
    """Capture real synchronous stream failures against an owned loopback peer."""

    release = threading.Event()
    server_done = threading.Event()
    server_errors: list[BaseException] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2.0)
        port = listener.getsockname()[1]

        def serve() -> None:
            try:
                with listener.accept()[0] as connection:
                    if kind == "write-timeout":
                        release.wait(1.0)
                        return
                    request = b""
                    while b"\r\n\r\n" not in request:
                        part = connection.recv(65_536)
                        if not part:
                            raise RuntimeError("loopback client closed before headers")
                        request += part
                    if kind == "read-timeout":
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n"
                        )
                        release.wait(1.0)
                    elif kind == "malformed-protocol":
                        connection.sendall(b"NOT HTTP\r\n\r\n")
                    elif kind == "truncated-protocol":
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc"
                        )
                    else:
                        raise AssertionError(f"unsupported native failure kind: {kind}")
            # Every original background failure is retained and asserted below.
            except BaseException as exc:  # noqa: BLE001
                server_errors.append(exc)
            finally:
                server_done.set()

        server = threading.Thread(target=serve, name=f"runpod-{kind}-peer")
        server.start()
        try:
            socket_options = None
            content = None
            if kind == "write-timeout":
                socket_options = [(socket.SOL_SOCKET, socket.SO_SNDBUF, 1_024)]
                chunk = b"x" * (1024 * 1024)
                content = (chunk for _ in range(1_024))
            transport = httpx.HTTPTransport(socket_options=socket_options)
            timeout = httpx.Timeout(0.5, read=0.05, write=0.05, pool=0.5)
            with httpx.Client(
                transport=transport,
                timeout=timeout,
                trust_env=False,
            ) as client:
                method = client.post if kind == "write-timeout" else client.get
                with pytest.raises(httpx.TransportError) as raised:
                    if content is None:
                        method(f"http://127.0.0.1:{port}/")
                    else:
                        method(f"http://127.0.0.1:{port}/", content=content)
        finally:
            release.set()
            server_done.wait(2.0)
            server.join(2.0)

    assert not server.is_alive()
    assert server_errors == []
    return raised.value


def _blocked_httpx_error(kind: str) -> httpx.HTTPError:
    """Build finite public exception graphs while retaining every original node."""

    request = httpx.Request("GET", "https://api.runpod.io/v2/pods")
    if kind == "http-status":
        return httpx.HTTPStatusError(
            "authorization denied",
            request=request,
            response=httpx.Response(401, request=request),
        )
    public_pairs: dict[str, tuple[type[httpx.HTTPError], type[Exception]]] = {
        "structural": (httpx.ConnectError, httpcore.ConnectError),
        "control": (httpx.ConnectError, httpcore.ConnectError),
        "mixed-group": (httpx.ConnectError, httpcore.ConnectError),
        "cycle": (httpx.ConnectError, httpcore.ConnectError),
        "local-protocol": (httpx.LocalProtocolError, httpcore.LocalProtocolError),
        "unsupported-protocol": (
            httpx.UnsupportedProtocol,
            httpcore.UnsupportedProtocol,
        ),
        "proxy": (httpx.ProxyError, httpcore.ProxyError),
    }
    if kind == "unknown-transport":

        class UnknownTransportError(httpx.TransportError):
            pass

        outer: httpx.HTTPError = UnknownTransportError(
            "unknown transport", request=request
        )
        outer.__cause__ = httpcore.ConnectError("connection failed")
        return outer
    outer_type, core_type = public_pairs[kind]
    outer = outer_type("transport failed", request=request)
    core = core_type("transport failed")
    outer.__cause__ = core
    if kind == "structural":
        core.__cause__ = ValueError("invalid transport state")
    elif kind == "control":
        core.__cause__ = ExecutionAuthorityError("original authority retired")
    elif kind == "mixed-group":
        core.__cause__ = ExceptionGroup(
            "mixed transport graph",
            [
                TimeoutError("native timeout"),
                ExecutionAuthorityError("original authority retired"),
            ],
        )
    elif kind == "cycle":
        structural = ValueError("cyclic structural failure")
        core.__cause__ = structural
        structural.__cause__ = core
    return outer


def _explicit_graph_ids(error: BaseException) -> tuple[int, ...]:
    seen: set[int] = set()
    pending = [error]
    ordered = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        ordered.append(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
    return tuple(ordered)


def test_native_httpx_httpcore_socket_chain_is_retryable_at_safe_ingress():
    transport_error = native_httpx_connect_failure()
    api_error = RunPodAPIError(
        title="Runpod transport failed",
        detail=type(transport_error).__name__,
    )
    api_error.__cause__ = transport_error
    assert is_retryable_runpod_api_error(api_error)

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise transport_error
        return httpx.Response(200, json=_page("pods", []))

    client = RunpodControlPlaneClient(
        api_key="secret",
        max_safe_retries=1,
        sleep=lambda _: None,
        http_transport=httpx.MockTransport(handler),
    )
    assert client.list_pods() == ()
    assert len(calls) == 2
    assert api_error.__cause__ is transport_error


@pytest.mark.parametrize(
    ("kind", "outer_type", "core_type"),
    [
        ("read-timeout", httpx.ReadTimeout, httpcore.ReadTimeout),
        (
            "malformed-protocol",
            httpx.RemoteProtocolError,
            httpcore.RemoteProtocolError,
        ),
        (
            "truncated-protocol",
            httpx.RemoteProtocolError,
            httpcore.RemoteProtocolError,
        ),
        ("write-timeout", httpx.WriteTimeout, httpcore.WriteTimeout),
    ],
)
def test_native_stream_failure_retains_exact_public_transport_pair(
    kind, outer_type, core_type
):
    transport_error = _native_httpx_stream_failure(kind)
    assert type(transport_error) is outer_type
    assert type(transport_error.__cause__) is core_type
    assert transport_error.__cause__.__cause__ is None
    assert transport_error.__cause__.__context__ is not None

    api_error = RunPodAPIError(
        title="Runpod transport failed",
        detail=outer_type.__name__,
    )
    api_error.__cause__ = transport_error
    assert is_retryable_runpod_api_error(api_error)


@pytest.mark.parametrize(
    "kind",
    [
        "structural",
        "control",
        "mixed-group",
        "cycle",
        "unknown-transport",
        "local-protocol",
        "unsupported-protocol",
        "proxy",
        "http-status",
    ],
)
def test_transport_graph_blockers_never_grant_retry_authority(kind):
    transport_error = _blocked_httpx_error(kind)
    original_graph = _explicit_graph_ids(transport_error)
    error = RunPodAPIError(
        title="Runpod transport failed",
        detail=type(transport_error).__name__,
    )
    error.__cause__ = transport_error

    assert not is_retryable_runpod_api_error(error)
    assert error.__cause__ is transport_error
    assert _explicit_graph_ids(transport_error) == original_graph


@pytest.mark.parametrize(
    "kind",
    [
        "structural",
        "control",
        "mixed-group",
        "cycle",
        "unknown-transport",
        "local-protocol",
        "unsupported-protocol",
        "proxy",
        "http-status",
    ],
)
def test_safe_method_ingress_never_retries_a_blocked_transport_graph(kind):
    transport_error = _blocked_httpx_error(kind)
    original_graph = _explicit_graph_ids(transport_error)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise transport_error
        return httpx.Response(200, json=_page("pods", []))

    client = RunpodControlPlaneClient(
        api_key="secret",
        max_safe_retries=1,
        sleep=lambda _: None,
        http_transport=httpx.MockTransport(handler),
    )
    if kind in {"control", "mixed-group"}:
        with pytest.raises(type(transport_error)) as caught:
            client.list_pods()
        assert caught.value is transport_error
    else:
        with pytest.raises(RunPodAPIError) as caught:
            client.list_pods()
        assert type(caught.value) is RunPodAPIError
        assert caught.value.__cause__ is transport_error
    assert len(calls) == 1
    assert _explicit_graph_ids(transport_error) == original_graph


@pytest.mark.parametrize(
    ("kind", "method"),
    [
        pytest.param("control", "GET", id="safe-get"),
        pytest.param("mixed-group", "GET", id="safe-get-explicit-group"),
        pytest.param("control", "POST", id="mutating-post"),
    ],
)
def test_request_transport_preserves_exact_execution_control(
    kind, method, record_property
):
    transport_error = _blocked_httpx_error(kind)
    original_graph = _explicit_graph_ids(transport_error)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        record_control_injection(record_property, f"request-{kind}-{method.lower()}")
        raise transport_error

    transport = RunpodTransport(
        api_key="secret",
        base_url="https://api.runpod.io/v2",
        bearer_auth=False,
        max_safe_retries=1,
        sleep=lambda _: None,
        http_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(type(transport_error)) as caught:
        transport.request_json(method, "/pods", json_body={"name": "x"})

    assert caught.value is transport_error
    assert len(calls) == 1
    assert _explicit_graph_ids(transport_error) == original_graph


@pytest.mark.parametrize(
    "boundary",
    ["open-stream", "active", "deadline"],
)
def test_sse_preserves_exact_execution_control(monkeypatch, boundary, record_property):
    transport_error = _blocked_httpx_error("control")
    original_graph = _explicit_graph_ids(transport_error)
    closed = []

    class Response:
        status_code = 200
        headers = {}

        def read(self):
            return b""

        def iter_lines(self):
            record_control_injection(record_property, f"sse-{boundary}")
            raise transport_error

        def close(self):
            closed.append(True)

    class Stream:
        def __enter__(self):
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def stream(self, *_args, **_kwargs):
            if boundary == "open-stream":
                record_control_injection(record_property, f"sse-{boundary}")
                raise transport_error
            return Stream()

    class ImmediateTimer:
        daemon = False

        def __init__(self, _seconds, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

    transport = RunpodTransport(
        api_key="secret",
        base_url="https://api.runpod.io/v2",
        bearer_auth=False,
    )
    transport._client = Client()
    if boundary == "deadline":
        monkeypatch.setattr(
            "kestrel_cloud_runpod.clients.threading.Timer", ImmediateTimer
        )

    with pytest.raises(type(transport_error)) as caught:
        tuple(
            transport.iter_sse(
                "/logs",
                stream_window_seconds=1 if boundary == "deadline" else None,
            )
        )

    assert caught.value is transport_error
    assert _explicit_graph_ids(transport_error) == original_graph
    assert closed == ([True] if boundary == "deadline" else [])


@pytest.mark.parametrize("boundary", ["open-stream", "active", "deadline"])
def test_sse_retains_ordinary_normalization_contract(monkeypatch, boundary):
    transport_error = httpx.ConnectError("ordinary stream failure")
    closed = []

    class Response:
        status_code = 200
        headers = {}

        def read(self):
            return b""

        def iter_lines(self):
            raise transport_error

        def close(self):
            closed.append(True)

    class Stream:
        def __enter__(self):
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def stream(self, *_args, **_kwargs):
            if boundary == "open-stream":
                raise transport_error
            return Stream()

    class ImmediateTimer:
        daemon = False

        def __init__(self, _seconds, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

    transport = RunpodTransport(
        api_key="secret",
        base_url="https://api.runpod.io/v2",
        bearer_auth=False,
    )
    transport._client = Client()
    if boundary == "deadline":
        monkeypatch.setattr(
            "kestrel_cloud_runpod.clients.threading.Timer", ImmediateTimer
        )

    if boundary == "deadline":
        assert tuple(transport.iter_sse("/logs", stream_window_seconds=1)) == ()
        assert closed == [True]
    else:
        with pytest.raises(RunPodAPIError) as raised:
            tuple(transport.iter_sse("/logs"))
        assert raised.value.__cause__ is transport_error
        assert closed == []


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
            json=_page("pods", []),
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
        return httpx.Response(200, json=_page("pods", []))

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


@pytest.mark.parametrize("resource_kind", ["pod", "endpoint"])
def test_paid_create_decoder_control_carrier_preserves_exact_graph(
    monkeypatch,
    resource_kind,
):
    carrier, control, explicit = execution_control_carrier(
        RunPodManagerError("decoder execution control"),
        grouped=True,
    )
    structural = explicit.exceptions[1]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = _pod() if resource_kind == "pod" else _endpoint()
        return httpx.Response(201, json=payload)

    def fail_decode(_value):
        raise carrier

    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(handler),
    )
    try:
        if resource_kind == "pod":
            monkeypatch.setattr(PodResource, "from_dict", staticmethod(fail_decode))
            create = client.create_pod
            request = PodCreateRequest(name="test", image="image:sha", gpu_id="gpu-id")
        else:
            monkeypatch.setattr(
                EndpointResource, "from_dict", staticmethod(fail_decode)
            )
            create = client.create_endpoint
            request = _endpoint_request()

        with pytest.raises(RunPodManagerError) as caught:
            create(request)

        assert caught.value is carrier
        assert carrier.__cause__ is explicit
        assert explicit.exceptions == (control, structural)
        assert len(calls) == 1
        assert calls[0].method == "POST"
    finally:
        client.close()


@pytest.mark.parametrize("body", [{}, {"id": "endpoint-ack", "name": None}])
def test_endpoint_create_without_a_usable_identity_remains_ambiguous(body):
    calls = []
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(201, json=body)
        ),
    )

    with pytest.raises(RunPodAmbiguousResultError, match="usable identity"):
        client.create_endpoint(_endpoint_request())

    assert len(calls) == 1


def test_endpoint_create_retains_identity_and_invalid_optional_evidence():
    payload = _endpoint(
        requestUrls=17,
        env={"PRIVATE": "must-not-leak"},
    )
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(201, json=payload)
        ),
    )

    resource = client.create_endpoint(_endpoint_request())

    assert resource.id == "ep-123"
    assert resource.request_urls == {}
    assert resource.field_evidence.state("requestUrls") is ResourceFieldState.INVALID
    assert resource.raw["env"] == {"PRIVATE": "[REDACTED]"}
    with pytest.raises(RunPodManagerError, match="requestUrls") as raised:
        resource.require_valid_fields(context="v2 Endpoint create")
    assert "must-not-leak" not in str(raised.value)


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


def test_endpoint_delete_timeout_is_ambiguous_and_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RunPodAmbiguousResultError) as raised:
        client.delete_endpoint("endpoint-1")

    assert calls == 1
    assert raised.value.resource == "/serverless/endpoint-1"


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
        httpx.Response(200, json=_page("pods", [_pod()])),
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


def test_network_volume_inventory_is_typed_and_read_only():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_page("networkVolumes", [_network_volume()]))

    client = RunpodControlPlaneClient(
        api_key="secret", http_transport=httpx.MockTransport(handler)
    )

    volumes = client.list_network_volumes()

    assert len(volumes) == 1
    assert volumes[0].id == "volume-123"
    assert volumes[0].size_gb == 100
    assert volumes[0].volume_type.value == "HIGH_PERFORMANCE"
    assert [(request.method, request.url.path) for request in seen] == [
        ("GET", "/v2/network-volumes")
    ]


@pytest.mark.parametrize(
    "volume",
    [
        _network_volume(type="unknown"),
        _network_volume(size=1),
        _network_volume(size=4097),
        _network_volume(dataCenter=None),
    ],
)
def test_network_volume_inventory_rejects_invalid_provider_shapes(volume):
    client = RunpodControlPlaneClient(
        api_key="secret",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_page("networkVolumes", [volume]))
        ),
    )

    with pytest.raises(RunPodManagerError, match="(?i)network volume"):
        client.list_network_volumes()


def test_endpoint_worker_and_billing_request_shapes():
    seen = []
    responses = [
        httpx.Response(201, json=_endpoint()),
        httpx.Response(200, json=_endpoint()),
        httpx.Response(200, json=_page("endpoints", [_endpoint()])),
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
    request = _endpoint_request()
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
