"""Historical billing through the real HTTP client; never a paid provider call."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest
from serverless_capacity_test_support import MutableClock, legacy_planned_authority

from kestrel_cloud_runpod import (
    RunpodControlPlaneClient,
    RunpodServerlessCapacityProvider,
)
from kestrel_cloud_runpod.models import RunPodAPIError, RunPodManagerError
from kestrel_cloud_runpod.serverless_capacity_contracts import (
    ServerlessEndpointCleanupReceipt,
    ServerlessRetiredEndpointBillingReceipt,
    iso_datetime,
    json_sha256,
    parse_datetime,
    serverless_billing_hour_bounds,
    serverless_billing_hour_starts,
)


@pytest.fixture
def billing_server():
    request, quote, activation, raw = legacy_planned_authority()
    state = {
        "present": True,
        "calls": [],
        "mode": "complete",
        "gpu_amount": 0.2,
        "entered": threading.Event(),
        "release": threading.Event(),
        "mutation": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer local-test-not-a-runpod-key"
            path = urlsplit(self.path)
            state["calls"].append(("GET", path.path, parse_qs(path.query)))
            if path.path == "/v2/billing/serverless":
                query = {key: values[0] for key, values in parse_qs(path.query).items()}
                start = parse_datetime(query["startTime"])
                end = parse_datetime(query["endTime"])
                records = []
                while start < end:
                    records.append(
                        {
                            "startTime": iso_datetime(start),
                            "endTime": iso_datetime(start + timedelta(hours=1)),
                            "serverlessId": activation.endpoint_id,
                            "gpuAmount": state["gpu_amount"],
                            "cpuAmount": 0.01,
                            "diskAmount": 0.02,
                            "feeAmount": 0.03,
                            "totalAmount": float(
                                Decimal(str(state["gpu_amount"])) + Decimal("0.06")
                            ),
                        }
                    )
                    start += timedelta(hours=1)
                if state["mode"] == "empty":
                    records.clear()
                totals = {
                    key: float(
                        sum((Decimal(str(item[key])) for item in records), Decimal(0))
                    )
                    for key in (
                        "gpuAmount",
                        "cpuAmount",
                        "diskAmount",
                        "feeAmount",
                        "totalAmount",
                    )
                }
                body = {
                    "records": records,
                    "metadata": {
                        "query": query,
                        "recordCount": len(records),
                        "uniqueServerlessCount": 1 if records else 0,
                        "totals": totals,
                    },
                }
                if state["mutation"] is not None:
                    state["mutation"](body)
                if state["mode"] == "unauthorized":
                    self.reply({"error": "restricted credential"}, 401)
                    return
                if state["mode"] == "blocked":
                    state["entered"].set()
                    assert state["release"].wait(5), (
                        "test must release billing response"
                    )
                self.reply(body)
            elif (
                path.path == f"/v2/serverless/{activation.endpoint_id}"
                and state["present"]
            ):
                self.reply(raw)
            elif (
                path.path == f"/v2/serverless/{activation.endpoint_id}/workers"
                and state["present"]
            ):
                self.reply(
                    {
                        "workers": [],
                        "summary": dict.fromkeys(
                            (
                                "running",
                                "idle",
                                "initializing",
                                "throttled",
                                "unhealthy",
                                "total",
                            ),
                            0,
                        ),
                    }
                )
            elif path.path == "/v2/serverless":
                self.reply(
                    {
                        "endpoints": [raw] if state["present"] else [],
                        "pagination": {"hasNextPage": False, "nextCursor": None},
                    }
                )
            else:
                self.reply({"error": "endpoint deleted; status is unavailable"}, 404)

        def do_DELETE(self):
            state["calls"].append(("DELETE", urlsplit(self.path).path, {}))
            assert self.path == f"/v2/serverless/{activation.endpoint_id}"
            state["present"] = False
            self.send_response(204)
            self.end_headers()

        def reply(self, body, status=200):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    client = RunpodControlPlaneClient(
        api_key="local-test-not-a-runpod-key",
        base_url=f"http://127.0.0.1:{server.server_port}/v2",
        max_safe_retries=0,
    )
    clock = MutableClock()
    clock.value = activation.readback_at + timedelta(minutes=5)
    provider = RunpodServerlessCapacityProvider(control_client=client, clock=clock)
    try:
        yield state, provider, clock, request, quote, activation
    finally:
        state["release"].set()
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_retired_endpoint_billing_needs_no_job_or_live_endpoint(billing_server):
    state, provider, clock, request, quote, activation = billing_server
    cleanup = await provider.cleanup_planned_endpoint(request, quote, activation)
    assert not state["present"]
    state["calls"].clear()
    clock.value = cleanup.verified_absent_at + timedelta(hours=2)

    receipt = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )

    assert receipt is not None
    assert receipt.actual_cost_usd == Decimal("0.26")
    assert receipt.activation == activation
    assert receipt.cleanup == cleanup
    assert receipt.provider_finality_confirmed is False
    assert all(
        method == "GET" and path == "/v2/billing/serverless"
        for method, path, _query in state["calls"]
    )
    assert state["calls"]
    assert "job_id" not in receipt.to_dict()


async def _retire(billing_server, *, after=timedelta(minutes=5)):
    state, provider, clock, request, quote, activation = billing_server
    clock.value = activation.readback_at + after
    cleanup = await provider.cleanup_planned_endpoint(request, quote, activation)
    state["calls"].clear()
    clock.value = cleanup.verified_absent_at + timedelta(hours=2)
    return state, provider, clock, request, quote, activation, cleanup


@pytest.mark.asyncio
async def test_closed_history_round_trips_and_late_cost_revision_changes_identity(
    billing_server,
):
    state, provider, clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    first = await provider.retired_endpoint_billing(request, quote, activation, cleanup)
    clock.value += timedelta(days=1)
    replay = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert first is not None and replay is not None
    assert replay.provider_billing_id == first.provider_billing_id
    assert replay.reconciled_at > first.reconciled_at
    assert ServerlessRetiredEndpointBillingReceipt.from_dict(first.to_dict()) == first
    assert (
        ServerlessRetiredEndpointBillingReceipt.from_dict(
            json.loads(json.dumps(first.to_dict()))
        )
        == first
    )

    state["gpu_amount"] = 0.4
    revised = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert revised is not None
    assert revised.actual_cost_usd == Decimal("0.46")
    assert revised.provider_billing_id != first.provider_billing_id
    assert (
        revised.endpoint_hour_costs[0].provider_observation_id
        != first.endpoint_hour_costs[0].provider_observation_id
    )
    assert (
        revised.endpoint_hour_costs[0].utc_hour_start
        == first.endpoint_hour_costs[0].utc_hour_start
    )
    assert not revised.provider_finality_confirmed


@pytest.mark.asyncio
async def test_pending_empty_or_open_billing_never_becomes_zero(billing_server):
    state, provider, clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    clock.value = cleanup.verified_absent_at
    assert (
        await provider.retired_endpoint_billing(request, quote, activation, cleanup)
        is None
    )
    assert state["calls"] == []
    clock.value += timedelta(hours=2)
    state["mode"] = "empty"
    assert (
        await provider.retired_endpoint_billing(request, quote, activation, cleanup)
        is None
    )
    assert len(state["calls"]) == 1


@pytest.mark.asyncio
async def test_reported_zero_is_distinct_from_missing_history(billing_server):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )

    def zero(body):
        for key in body["metadata"]["totals"]:
            body["metadata"]["totals"][key] = 0
            for record in body["records"]:
                record[key] = 0

    state["mutation"] = zero
    receipt = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert receipt is not None and receipt.actual_cost_usd == 0
    assert len(receipt.endpoint_hour_costs) == 1
    assert receipt.provider_finality_confirmed is False


def _billing_fault(body, fault):
    records = body["records"]
    meta = body["metadata"]
    if fault == "partial_hour":
        records[-1]["endTime"] = iso_datetime(
            parse_datetime(records[-1]["endTime"]) - timedelta(minutes=5)
        )
    elif fault == "missing_last_hour":
        records.pop()
        meta["recordCount"] -= 1
    elif fault == "wrong_endpoint":
        records[0]["serverlessId"] = "someone-elses-endpoint"
    elif fault == "wrong_query":
        meta["query"]["serverlessId"] = "someone-elses-endpoint"
    elif fault == "wrong_total":
        meta["totals"]["gpuAmount"] += 1
        meta["totals"]["totalAmount"] += 1
    elif fault == "wrong_count":
        meta["recordCount"] += 1
    elif fault == "duplicate_hour":
        records.append(deepcopy(records[0]))
        meta["recordCount"] += 1
    elif fault == "private_field":
        records[0]["output"] = "private generated payload must not reach receipts"
    elif fault == "negative":
        records[0]["gpuAmount"] = -1
    elif fault == "nan":
        records[0]["gpuAmount"] = float("nan")
    elif fault == "escaped_window":
        records[0]["startTime"] = iso_datetime(
            parse_datetime(records[0]["startTime"]) - timedelta(hours=1)
        )
    else:
        raise AssertionError(f"Unhandled test fault {fault}")


@pytest.mark.parametrize("fault", ["partial_hour", "missing_last_hour"])
@pytest.mark.asyncio
async def test_incomplete_history_stays_pending(billing_server, fault):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server, after=timedelta(hours=1, minutes=5)
    )
    state["mutation"] = lambda body: _billing_fault(body, fault)
    assert (
        await provider.retired_endpoint_billing(request, quote, activation, cleanup)
        is None
    )


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_endpoint",
        "wrong_query",
        "wrong_total",
        "wrong_count",
        "duplicate_hour",
        "private_field",
        "negative",
        "nan",
        "escaped_window",
    ],
)
@pytest.mark.asyncio
async def test_invalid_history_fails_closed_through_actual_http(billing_server, fault):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    state["mutation"] = lambda body: _billing_fault(body, fault)
    with pytest.raises(RunPodManagerError):
        await provider.retired_endpoint_billing(request, quote, activation, cleanup)


@pytest.mark.asyncio
async def test_actual_provider_error_does_not_become_pending_or_free(billing_server):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    state["mode"] = "unauthorized"
    with pytest.raises(RunPodAPIError):
        await provider.retired_endpoint_billing(request, quote, activation, cleanup)


@pytest.mark.asyncio
async def test_original_authority_required_before_any_billing_read(billing_server):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    with pytest.raises(RunPodManagerError, match="does not match"):
        await provider.retired_endpoint_billing(
            replace(request, parameters_sha256="f" * 64), quote, activation, cleanup
        )
    with pytest.raises(RunPodManagerError, match="accepted planned quote"):
        await provider.retired_endpoint_billing(
            request,
            replace(quote, provider_quote_id="other-quote"),
            activation,
            cleanup,
        )
    mismatched = cleanup.to_dict()
    mismatched["provider_activation_id"] = "other-activation"
    identity = {
        key: value
        for key, value in mismatched.items()
        if key not in {"schema_version", "provider_cleanup_id"}
    }
    mismatched["provider_cleanup_id"] = "runpod-serverless-cleanup:" + json_sha256(
        identity
    )
    with pytest.raises(RunPodManagerError, match="does not match its endpoint"):
        await provider.retired_endpoint_billing(
            request,
            quote,
            activation,
            ServerlessEndpointCleanupReceipt.from_dict(mismatched),
        )
    forged = cleanup.to_dict()
    for field in (
        "workers_zero_observed_at",
        "delete_started_at",
        "verified_absent_at",
    ):
        forged[field] = iso_datetime(activation.readback_at - timedelta(hours=1))
    identity = {
        key: value
        for key, value in forged.items()
        if key not in {"schema_version", "provider_cleanup_id"}
    }
    forged["provider_cleanup_id"] = "runpod-serverless-cleanup:" + json_sha256(identity)
    premature = ServerlessEndpointCleanupReceipt.from_dict(forged)
    with pytest.raises(RunPodManagerError, match="precedes activation"):
        await provider.retired_endpoint_billing(request, quote, activation, premature)
    assert state["calls"] == []


@pytest.mark.parametrize("batch_hours", [1, 2, 10**30])
@pytest.mark.asyncio
async def test_configured_history_batches_preserve_same_cost_and_identity(
    billing_server, batch_hours
):
    state, provider, clock, request, quote, activation, cleanup = await _retire(
        billing_server, after=timedelta(hours=4, minutes=5)
    )
    original = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert original is not None
    configured = RunpodServerlessCapacityProvider(
        control_client=provider.control_client,
        clock=clock,
        billing_history_batch_hours=batch_hours,
    )
    state["calls"].clear()
    observed = await configured.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert observed == original
    assert len(state["calls"]) == (5 + batch_hours - 1) // batch_hours
    for method, path, query in state["calls"]:
        assert method == "GET" and path == "/v2/billing/serverless"
        assert query["serverlessId"] == [activation.endpoint_id]
        assert query["bucketSize"] == ["hour"]
        assert "lastN" not in query


@pytest.mark.asyncio
async def test_missing_late_batch_cannot_publish_partial_receipt(billing_server):
    state, provider, clock, request, quote, activation, cleanup = await _retire(
        billing_server, after=timedelta(hours=4, minutes=5)
    )
    configured = RunpodServerlessCapacityProvider(
        control_client=provider.control_client,
        clock=clock,
        billing_history_batch_hours=2,
    )

    def missing_late_batch(body):
        if len(state["calls"]) == 2:
            body["records"] = []
            body["metadata"]["recordCount"] = 0
            body["metadata"]["uniqueServerlessCount"] = 0
            body["metadata"]["totals"] = dict.fromkeys(body["metadata"]["totals"], 0)

    state["mutation"] = missing_late_batch
    assert (
        await configured.retired_endpoint_billing(request, quote, activation, cleanup)
        is None
    )
    assert len(state["calls"]) == 2


@pytest.mark.asyncio
async def test_historical_cleanup_cost_is_not_truncated_to_job_authorization(
    billing_server,
):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server, after=timedelta(days=8, hours=1, minutes=5)
    )
    assert quote.expires_at < cleanup.verified_absent_at
    receipt = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert receipt is not None
    assert len(receipt.endpoint_hour_costs) == 194
    assert receipt.actual_cost_usd == Decimal("0.26") * 194
    assert len(state["calls"]) == 9
    for method, path, query in state["calls"]:
        assert method == "GET" and path == "/v2/billing/serverless"
        start = parse_datetime(query["startTime"][0])
        end = parse_datetime(query["endTime"][0])
        assert end - start <= timedelta(hours=24)
    with pytest.raises(ValueError, match="policy bounds"):
        serverless_billing_hour_starts(
            activation.create_started_at, cleanup.verified_absent_at
        )
    state["calls"].clear()
    with pytest.raises(RunPodManagerError, match="upgrade|expired"):
        await provider.activate_planned_endpoint(
            request, quote, accepted_cost_ceiling_usd=quote.cost_ceiling_usd
        )
    assert state["calls"] == []


@pytest.mark.parametrize(
    "change", ["digest", "totals", "hours", "scope", "finality", "extra"]
)
@pytest.mark.asyncio
async def test_persisted_receipt_cannot_forge_evidence_or_finality(
    billing_server, change
):
    _state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    receipt = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert receipt is not None
    raw = receipt.to_dict()
    if change == "digest":
        raw["provider_billing_id"] = "forged-id"
    elif change == "totals":
        raw["cost_totals"]["actual_cost_usd"] = "0"
    elif change == "hours":
        raw["endpoint_hour_costs"] = []
    elif change == "scope":
        raw["cost_scope"] = "per_job"
    elif change == "finality":
        raw["provider_finality_confirmed"] = True
    elif change == "extra":
        raw["private_prompt"] = "not allowed"
    with pytest.raises(RunPodManagerError):
        ServerlessRetiredEndpointBillingReceipt.from_dict(raw)


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_real_http_read_before_client_can_close(
    billing_server,
):
    state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server
    )
    state["mode"] = "blocked"
    task = asyncio.create_task(
        provider.retired_endpoint_billing(request, quote, activation, cleanup)
    )
    try:
        assert await asyncio.to_thread(state["entered"].wait, 5)
        for _ in range(4):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
    finally:
        state["release"].set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("hours", [0, -1, True, "24", None])
def test_invalid_history_batch_configuration_fails_fast(hours):
    with pytest.raises(ValueError, match="positive integer"):
        RunpodServerlessCapacityProvider(
            control_client=None, billing_history_batch_hours=hours
        )


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "boolean_schema",
        "contract",
        "activation",
        "cleanup",
        "list",
        "empty",
        "wrong_type",
        "wrong_endpoint",
        "duplicate",
        "missing",
        "open",
    ],
)
@pytest.mark.asyncio
async def test_receipt_constructor_requires_exact_complete_lifecycle(
    billing_server, change
):
    _state, provider, _clock, request, quote, activation, cleanup = await _retire(
        billing_server, after=timedelta(hours=1, minutes=5)
    )
    receipt = await provider.retired_endpoint_billing(
        request, quote, activation, cleanup
    )
    assert receipt is not None
    changes = {
        "schema": {"schema_version": 2},
        "boolean_schema": {"schema_version": True},
        "contract": {"contract_version": "other-contract"},
        "activation": {"activation": None},
        "cleanup": {"cleanup": None},
        "list": {"endpoint_hour_costs": list(receipt.endpoint_hour_costs)},
        "empty": {"endpoint_hour_costs": ()},
        "wrong_type": {"endpoint_hour_costs": (None,)},
        "wrong_endpoint": {
            "endpoint_hour_costs": (
                replace(receipt.endpoint_hour_costs[0], endpoint_id="wrong-endpoint"),
                receipt.endpoint_hour_costs[1],
            )
        },
        "duplicate": {"endpoint_hour_costs": receipt.endpoint_hour_costs * 2},
        "missing": {"endpoint_hour_costs": receipt.endpoint_hour_costs[:-1]},
        "open": {"reconciled_at": cleanup.verified_absent_at},
    }
    with pytest.raises((ValueError, TypeError)):
        replace(receipt, **changes[change])


def test_hour_bounds_use_utc_without_weakening_finite_job_policy():
    local = ZoneInfo("America/Chicago")
    start = datetime(2026, 10, 31, 22, 30, tzinfo=local)
    end = datetime(2026, 11, 1, 2, 0, tzinfo=local)
    assert serverless_billing_hour_bounds(start, end) == (
        datetime(2026, 11, 1, 3, 0, tzinfo=UTC),
        datetime(2026, 11, 1, 8, 0, tzinfo=UTC),
    )
    for invalid_end in (start, start - timedelta(seconds=1)):
        with pytest.raises(ValueError, match="interval is invalid"):
            serverless_billing_hour_bounds(start, invalid_end)
    with pytest.raises(ValueError, match="timezone-aware"):
        serverless_billing_hour_bounds(start.replace(tzinfo=None), end)
    # UTC elapsed time, not local wall time, enforces the pre-existing 7d+1h cap.
    end = datetime(2026, 11, 8, 1, 0, tzinfo=local)
    start = end - timedelta(days=7, hours=1)
    with pytest.raises(ValueError, match="policy bounds"):
        serverless_billing_hour_starts(start, end)
