"""Authenticated single-attempt workload transport tests."""

import httpx
import pytest
from pod_capacity_test_support import REQUEST_SHA, TOKEN
from pydantic import SecretStr

from kestrel_cloud_runpod.pod_capacity_contracts import CatalogPodWorkloadState
from kestrel_cloud_runpod.pod_transport import (
    CatalogPodTransportConflictError,
    CatalogPodTransportError,
    CatalogPodWorkloadTransport,
)

ATTEMPT = "attempt:catalog-run-0001"
BASE = "https://pod-1-8080.proxy.runpod.net"


def _payload() -> dict[str, object]:
    return {
        "schema_version": 3,
        "dispatch_attempt_id": ATTEMPT,
        "request_sha256": REQUEST_SHA,
        "private": {"signed_url": "must-remain-opaque"},
    }


@pytest.mark.asyncio
async def test_transport_uses_anonymous_health_and_bearer_for_job_routes() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(
            202,
            json={
                "attempt_id": ATTEMPT,
                "request_sha256": REQUEST_SHA,
                "state": "running",
                "error_type": None,
                "result_available": False,
            },
        )

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    await transport.health(BASE)
    observation = await transport.submit(
        base_url=BASE,
        attempt_id=ATTEMPT,
        request_sha256=REQUEST_SHA,
        bearer_token=SecretStr(TOKEN),
        payload=_payload(),
    )

    assert "Authorization" not in seen[0].headers
    assert seen[1].headers["Authorization"] == f"Bearer {TOKEN}"
    assert observation.state is CatalogPodWorkloadState.RUNNING
    assert b"must-remain-opaque" in seen[1].content


@pytest.mark.asyncio
async def test_transport_rejects_local_hash_mismatch_before_network() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    with pytest.raises(CatalogPodTransportConflictError, match="hash"):
        await transport.submit(
            base_url=BASE,
            attempt_id=ATTEMPT,
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
            payload={**_payload(), "request_sha256": "f" * 64},
        )
    assert calls == 0


@pytest.mark.asyncio
async def test_transport_maps_remote_conflict_without_echoing_body() -> None:
    private = "signed-private-conflict-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": private})

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    with pytest.raises(CatalogPodTransportConflictError) as raised:
        await transport.submit(
            base_url=BASE,
            attempt_id=ATTEMPT,
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
            payload=_payload(),
        )
    assert private not in str(raised.value)


@pytest.mark.asyncio
async def test_transport_returns_result_unchanged_but_checks_attempt_binding() -> None:
    result = {
        "dispatch_attempt_id": ATTEMPT,
        "request_sha256": REQUEST_SHA,
        "artifact": {"storage_key": "private-key"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=result)

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    received = await transport.result(
        base_url=BASE,
        attempt_id=ATTEMPT,
        request_sha256=REQUEST_SHA,
        bearer_token=SecretStr(TOKEN),
    )
    assert received == result


@pytest.mark.asyncio
async def test_transport_requires_tls_and_refuses_redirects() -> None:
    transport = CatalogPodWorkloadTransport(
        http_transport=httpx.MockTransport(lambda _: httpx.Response(302))
    )
    with pytest.raises(CatalogPodTransportError, match="HTTPS"):
        await transport.health("http://pod.invalid")
    with pytest.raises(CatalogPodTransportError, match="HTTP 302"):
        await transport.health(BASE)
