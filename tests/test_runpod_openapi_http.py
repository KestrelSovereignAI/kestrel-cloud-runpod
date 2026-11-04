"""Current v2 compatibility over owned loopback HTTP, never a Runpod account."""

from __future__ import annotations

import json
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import (
    CpuPodCreateRequest,
    EndpointCreateRequest,
    EndpointUpdateRequest,
    PodCreateRequest,
    RunPodAPIError,
)


@pytest.fixture
def native_v2_http():
    state = {"responses": [], "calls": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self):
            self.connection.settimeout(5)
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            assert self.headers["Authorization"] == "Bearer owned-loopback-only"
            state["calls"].append(
                (self.command, self.path, json.loads(raw) if raw else None)
            )
            assert state["responses"], "unexpected HTTP request"
            status, payload = state["responses"].pop(0)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/problem+json" if status == 413 else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("RateLimit", '"minute";r=179;t=30')
            self.send_header("RateLimit-Policy", '"minute";q=180;w=60')
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PATCH = respond

    with ExitStack() as stack:
        server = stack.enter_context(ThreadingHTTPServer(("127.0.0.1", 0), Handler))
        server.daemon_threads = False
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()

        def join_original_server():
            thread.join(timeout=5)
            assert not thread.is_alive(), "owned HTTP server did not join"

        stack.callback(join_original_server)
        stack.callback(server.shutdown)
        client = RunpodControlPlaneClient(
            api_key="owned-loopback-only",
            base_url=f"http://127.0.0.1:{server.server_port}/v2",
        )
        stack.callback(client.close)
        yield state, client


@pytest.mark.parametrize(
    "operation",
    [
        "gpu-create",
        "cpu-create",
        "pod-action",
        "endpoint-create",
        "endpoint-update",
        "inventory",
    ],
)
def test_413_is_definitive_and_never_replayed(native_v2_http, operation):
    state, client = native_v2_http
    state["responses"].append(
        (
            413,
            {
                "title": "Request Entity Too Large",
                "status": 413,
                "detail": "Request or response exceeds the operation limit.",
            },
        )
    )
    with pytest.raises(RunPodAPIError) as caught:
        if operation == "gpu-create":
            client.create_pod(
                PodCreateRequest(
                    name="local-test", image="image:test", gpu_id="gpu-test"
                )
            )
        elif operation == "cpu-create":
            client.create_pod(
                CpuPodCreateRequest(
                    name="local-test",
                    image="image:test",
                    cpu_id="cpu-test",
                    vcpu_count=4,
                )
            )
        elif operation == "pod-action":
            client.pod_action("pod-test", "stop")
        elif operation == "endpoint-create":
            client.create_endpoint(
                EndpointCreateRequest(
                    name="local-test",
                    image="image:test",
                    gpu_pools=("BLACKWELL_24",),
                    endpoint_type="QUEUE",
                    scaling={"type": "QUEUE_DELAY", "queueDelay": 4},
                )
            )
        elif operation == "endpoint-update":
            client.update_endpoint(
                "endpoint-test", EndpointUpdateRequest(workers_max=1)
            )
        else:
            client.list_pods()
    error = caught.value
    assert type(error) is RunPodAPIError
    assert error.status_code == 413
    assert error.detail == "Request or response exceeds the operation limit."
    assert error.rate_limit.raw == '"minute";r=179;t=30'
    assert error.rate_limit.policy == '"minute";q=180;w=60'
    assert len(state["calls"]) == 1
    assert state["responses"] == []


def test_late_413_never_returns_partial_or_empty_inventory(native_v2_http):
    state, client = native_v2_http
    state["responses"].extend(
        [
            (
                200,
                {
                    "pods": [
                        {
                            "id": "pod-test",
                            "name": "test",
                            "status": "RUNNING",
                            "cost": 0.5,
                        }
                    ],
                    "pagination": {"nextCursor": "page-two", "hasNextPage": True},
                },
            ),
            (
                413,
                {
                    "title": "Request Entity Too Large",
                    "status": 413,
                    "detail": "Response too large for one request.",
                },
            ),
        ]
    )
    with pytest.raises(RunPodAPIError) as caught:
        client.list_pods()
    assert caught.value.status_code == 413
    assert len(state["calls"]) == 2
    assert "cursor=page-two" in state["calls"][1][1]


def test_nullable_inventory_requires_separate_exact_get_and_keeps_stopped_rate(
    native_v2_http,
):
    state, client = native_v2_http
    listed = {
        "id": "pod-test",
        "name": "test",
        "status": "RUNNING",
        "cost": 0.5,
        "runtime": None,
        "ssh": {"direct": None},
    }
    direct = {
        "host": "203.0.113.1",
        "port": 22000,
        "username": "root",
        "command": "ssh test",
    }
    stopped = {**listed, "status": "EXITED", "ssh": {"direct": direct}}
    state["responses"].extend(
        [
            (
                200,
                {
                    "pods": [listed],
                    "pagination": {"nextCursor": None, "hasNextPage": False},
                },
            ),
            (200, stopped),
        ]
    )
    inventory = client.list_pods()
    assert inventory[0].raw["runtime"] is None
    assert inventory[0].raw["ssh"]["direct"] is None
    exact = client.get_pod(inventory[0].id)
    assert exact.id == inventory[0].id
    assert exact.status == "EXITED"
    assert exact.cost_per_hr == 0.5
    assert exact.raw["ssh"]["direct"] == direct
    assert inventory[0].raw["ssh"]["direct"] is None
    assert [(method, urlsplit(path).path) for method, path, _ in state["calls"]] == [
        ("GET", "/v2/pods"),
        ("GET", "/v2/pods/pod-test"),
    ]
