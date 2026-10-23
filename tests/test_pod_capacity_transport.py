"""Authenticated single-attempt workload transport tests."""

import json

import httpx
import pytest
from pod_capacity_test_support import (
    ACK_SHA,
    RECEIPT_SHA,
    REQUEST_SHA,
    RESULT_SHA,
    TOKEN,
    acknowledgement,
    execution_control_carrier,
    record_control_injection,
)
from pydantic import SecretStr

from kestrel_cloud_runpod.pod_capacity_contracts import CatalogPodWorkloadState
from kestrel_cloud_runpod.pod_transport import (
    CatalogPodTransportConflictError,
    CatalogPodTransportError,
    CatalogPodWorkloadTransport,
)

ATTEMPT = "attempt:catalog-run-0001"
BASE = "https://pod-1-8080.proxy.runpod.net"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "transport"])
async def test_catalog_transport_preserves_exact_execution_control(
    kind, record_property
) -> None:
    error_type = httpx.ReadTimeout if kind == "timeout" else httpx.ConnectError
    carrier, control, explicit = execution_control_carrier(
        error_type(f"catalog {kind} outcome remains unknown")
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        record_control_injection(record_property, f"catalog-transport-{kind}")
        raise carrier

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))

    with pytest.raises(type(carrier)) as raised:
        await transport.health(BASE)

    assert raised.value is carrier
    assert carrier.__cause__ is explicit is control
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "transport"])
async def test_catalog_transport_retains_ordinary_error_contract(kind) -> None:
    error_type = httpx.ReadTimeout if kind == "timeout" else httpx.ConnectError
    ordinary = error_type(f"ordinary catalog {kind} failure")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise ordinary

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))

    with pytest.raises(CatalogPodTransportError) as raised:
        await transport.health(BASE)

    assert raised.value.__cause__ is ordinary
    assert len(calls) == 1


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
        "protocol_version": 1,
        "receipt": {
            "attempt_id": ATTEMPT,
            "request_sha256": REQUEST_SHA,
            "opaque_signature": "private",
        },
        "result": {
            "dispatch_attempt_id": ATTEMPT,
            "request_sha256": REQUEST_SHA,
            "artifact": {"storage_key": "private-key"},
        },
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


@pytest.mark.asyncio
async def test_transport_forwards_opaque_ack_and_requires_exact_authenticated_proof():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "attempt_id": ATTEMPT,
                "request_sha256": REQUEST_SHA,
                "state": "acknowledged",
                "error_type": None,
                "result_available": True,
                "result_sha256": RESULT_SHA,
                "receipt_sha256": RECEIPT_SHA,
                "acknowledgement_sha256": ACK_SHA,
            },
        )

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    observed = await transport.acknowledge(
        base_url=BASE,
        attempt_id=ATTEMPT,
        request_sha256=REQUEST_SHA,
        bearer_token=SecretStr(TOKEN),
        payload=acknowledgement(),
    )
    assert observed.state is CatalogPodWorkloadState.ACKNOWLEDGED
    assert (
        seen[0].url.path == "/v1/catalog/jobs/attempt:catalog-run-0001/acknowledgement"
    )
    assert seen[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(seen[0].content) == acknowledgement()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"attempt_id": "attempt:other"},
        {"request_sha256": "a" * 64},
        {"request_sha256": None},
        {"result_sha256": "a" * 64},
        {"receipt_sha256": "a" * 64},
        {"acknowledgement_sha256": "a" * 64},
        {"acknowledgement_sha256": None},
        {"result_sha256": True},
        {"state": "running"},
        {"result_available": False},
    ],
)
async def test_ack_cannot_accept_missing_changed_or_earlier_phase_proof(changed):
    payload = {
        "attempt_id": ATTEMPT,
        "request_sha256": REQUEST_SHA,
        "state": "acknowledged",
        "error_type": None,
        "result_available": True,
        "result_sha256": RESULT_SHA,
        "receipt_sha256": RECEIPT_SHA,
        "acknowledgement_sha256": ACK_SHA,
    }
    transport = CatalogPodWorkloadTransport(
        http_transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={**payload, **changed})
        )
    )
    with pytest.raises(CatalogPodTransportError):
        await transport.acknowledge(
            base_url=BASE,
            attempt_id=ATTEMPT,
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
            payload=acknowledgement(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 409, 500])
async def test_ack_http_error_never_echoes_private_receipt(status):
    transport = CatalogPodWorkloadTransport(
        http_transport=httpx.MockTransport(
            lambda _: httpx.Response(status, json={"secret": TOKEN})
        )
    )
    with pytest.raises(CatalogPodTransportError) as raised:
        await transport.acknowledge(
            base_url=BASE,
            attempt_id=ATTEMPT,
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
            payload=acknowledgement(),
        )
    assert TOKEN not in str(raised.value)


@pytest.mark.asyncio
async def test_ack_cannot_accept_omitted_request_hash():
    payload = {
        "attempt_id": ATTEMPT,
        "state": "acknowledged",
        "error_type": None,
        "result_available": True,
        "result_sha256": RESULT_SHA,
        "receipt_sha256": RECEIPT_SHA,
        "acknowledgement_sha256": ACK_SHA,
    }
    transport = CatalogPodWorkloadTransport(
        http_transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    with pytest.raises(CatalogPodTransportError):
        await transport.acknowledge(
            base_url=BASE,
            attempt_id=ATTEMPT,
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
            payload=acknowledgement(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"protocol_version": True},
        {"protocol_version": 2},
        {"receipt": {}},
        {"result": []},
        {
            "result": {
                "dispatch_attempt_id": "attempt:other",
                "request_sha256": REQUEST_SHA,
            }
        },
    ],
)
async def test_result_protocol_and_nested_bindings_fail_closed(changed):
    payload = {
        "protocol_version": 1,
        "receipt": {"attempt_id": ATTEMPT, "request_sha256": REQUEST_SHA},
        "result": {"dispatch_attempt_id": ATTEMPT, "request_sha256": REQUEST_SHA},
    }
    transport = CatalogPodWorkloadTransport(
        http_transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={**payload, **changed})
        )
    )
    with pytest.raises(CatalogPodTransportError):
        await transport.result(
            base_url=BASE,
            attempt_id=ATTEMPT,
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
        )


@pytest.mark.asyncio
async def test_legacy_flat_result_requires_explicit_persisted_legacy_protocol():
    payload = {"dispatch_attempt_id": ATTEMPT, "request_sha256": REQUEST_SHA}
    transport = CatalogPodWorkloadTransport(
        http_transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    args = {
        "base_url": BASE,
        "attempt_id": ATTEMPT,
        "request_sha256": REQUEST_SHA,
        "bearer_token": SecretStr(TOKEN),
    }
    with pytest.raises(CatalogPodTransportError):
        await transport.result(**args)
    assert (
        await transport.result(**args, artifact_handoff_protocol_version=None)
        == payload
    )
