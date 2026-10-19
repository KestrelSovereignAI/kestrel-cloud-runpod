"""Historical billing through the real HTTP client; never a paid provider call."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import threading
from contextlib import closing
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
    ServerlessEndpointCleanupError,
)
from kestrel_cloud_runpod.models import RunPodAPIError, RunPodManagerError
from kestrel_cloud_runpod.serverless_capacity_contracts import (
    ServerlessEndpointCleanupIntent,
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
        "cleanup_block_request": None,
        "cleanup_get_status": None,
        "cleanup_get_fault": None,
        "cleanup_delete_keeps_present": False,
        "cleanup_workers_present": False,
        "cleanup_inventory": None,
        "cleanup_inventory_pagination": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def request_barrier(self, method, path):
            if state["cleanup_block_request"] == (method, path):
                state["entered"].set()
                assert state["release"].wait(5), (
                    "test must release its owned HTTP response"
                )

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer local-test-not-a-runpod-key"
            path = urlsplit(self.path)
            state["calls"].append(("GET", path.path, parse_qs(path.query)))
            self.request_barrier("GET", path.path)
            if (
                path.path == f"/v2/serverless/{activation.endpoint_id}"
                and state["cleanup_get_fault"] is not None
            ):
                if state["cleanup_get_fault"] == "disconnect":
                    self.connection.close()
                    return
                self.reply({"id": activation.endpoint_id})
                return
            if (
                path.path == f"/v2/serverless/{activation.endpoint_id}"
                and state["cleanup_get_status"] is not None
            ):
                self.reply(
                    {"error": "explicit cleanup read failure"},
                    state["cleanup_get_status"],
                )
                return
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
                        "workers": [{"id": "renewed-worker"}]
                        if state["cleanup_workers_present"]
                        else [],
                        "summary": {
                            **dict.fromkeys(
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
                            "running": int(state["cleanup_workers_present"]),
                            "total": int(state["cleanup_workers_present"]),
                        },
                    }
                )
            elif path.path == "/v2/serverless":
                self.reply(
                    {
                        "endpoints": state["cleanup_inventory"]
                        if state["cleanup_inventory"] is not None
                        else ([raw] if state["present"] else []),
                        "pagination": state["cleanup_inventory_pagination"]
                        if state["cleanup_inventory_pagination"] is not None
                        else {"hasNextPage": False, "nextCursor": None},
                    }
                )
            else:
                self.reply({"error": "endpoint deleted; status is unavailable"}, 404)

        def do_DELETE(self):
            state["calls"].append(("DELETE", urlsplit(self.path).path, {}))
            self.request_barrier("DELETE", urlsplit(self.path).path)
            assert self.path == f"/v2/serverless/{activation.endpoint_id}"
            if not state["cleanup_delete_keeps_present"]:
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


class _CleanupIntentJournal:
    """Real test-host persistence, not a second production cleanup manager."""

    def __init__(self, path):
        self.path = str(path)

    def persist(self, intent):
        payload = json.dumps(intent.to_dict(), sort_keys=True, separators=(",", ":"))
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cleanup_intent (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
            )
            existing = connection.execute(
                "SELECT payload FROM cleanup_intent WHERE id=1"
            ).fetchone()
            if existing is not None and existing[0] != payload:
                raise ValueError("test host refuses to replace original cleanup intent")
            connection.execute(
                "INSERT INTO cleanup_intent VALUES (1,?) ON CONFLICT(id) DO NOTHING",
                (payload,),
            )
        return self.load()

    def load(self):
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT payload FROM cleanup_intent WHERE id=1"
            ).fetchone()
        assert row is not None
        return ServerlessEndpointCleanupIntent.from_dict(json.loads(row[0]))

    async def acknowledge(self, intent):
        return await asyncio.to_thread(self.persist, intent)


def test_real_cleanup_journal_closes_connections_after_success_and_conflict(
    tmp_path, monkeypatch
):
    connections = []
    real_connect = sqlite3.connect

    def observed_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    activation = legacy_planned_authority()[2]
    intent = ServerlessEndpointCleanupIntent(
        schema_version=1,
        contract_version=activation.contract_version,
        activation=activation,
        workers_zero_observed_at=activation.readback_at,
        delete_started_at=activation.readback_at,
    )
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    assert journal.persist(intent) == intent
    assert journal.load() == intent
    with pytest.raises(ValueError, match="replace original cleanup intent"):
        journal.persist(
            replace(
                intent,
                delete_started_at=intent.delete_started_at + timedelta(seconds=1),
            )
        )
    assert len(connections) == 4  # Actual write/read/read and rejected-write sessions.
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


_CLEANUP_CRASH_PROCESS = r"""
import asyncio
import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tests"))
from serverless_capacity_test_support import legacy_planned_authority
from test_retired_endpoint_billing import _CleanupIntentJournal
from kestrel_cloud_runpod import RunpodControlPlaneClient, RunpodServerlessCapacityProvider

request, quote, activation, _raw = legacy_planned_authority()
journal = _CleanupIntentJournal(sys.argv[2])
client = RunpodControlPlaneClient(api_key="local-test-not-a-runpod-key", base_url=sys.argv[1], max_safe_retries=0)
provider = RunpodServerlessCapacityProvider(control_client=client, clock=lambda: activation.readback_at + timedelta(minutes=5))
phase = sys.argv[3]

async def acknowledge(intent):
    result = await journal.acknowledge(intent)
    if phase == "intent_ack_lost":
        os._exit(73)
    return result

if phase == "before_delete":
    # Fatal fault at the real client's mutation boundary, before any wire send.
    def fatal_before_wire(_endpoint_id):
        os._exit(73)
    client.delete_endpoint = fatal_before_wire

async def run():
    try:
        await provider.cleanup_planned_endpoint(request, quote, activation, persist_intent=acknowledge)
        # Full provider cleanup succeeded; the host has not committed its receipt.
        os._exit(74)
    finally:
        client.close()

asyncio.run(run())
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", ["intent_ack_lost", "before_delete", "receipt_commit_lost"]
)
async def test_cleanup_converges_after_actual_process_death_without_fabricated_delivery(
    billing_server, tmp_path, phase
):
    state, provider, clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CLEANUP_CRASH_PROCESS,
        provider.control_client.transport.base_url,
        journal.path,
        phase,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHON_DOTENV_DISABLED": "1", "KESTREL_SKIP_DOTENV": "1"},
    )
    try:
        _stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (74 if phase == "receipt_commit_lost" else 73), (
        stderr.decode()
    )
    intent = journal.load()  # New connection after abrupt process death.
    assert state["present"] is (phase != "receipt_commit_lost")
    prior_deletes = sum(method == "DELETE" for method, _path, _query in state["calls"])
    assert prior_deletes == int(phase == "receipt_commit_lost")
    clock.value += timedelta(hours=2)
    # Entirely new service instance, no in-memory mutation/result state to adopt.
    restarted = RunpodServerlessCapacityProvider(
        control_client=provider.control_client, clock=clock
    )
    receipt = await restarted.cleanup_planned_endpoint(
        request,
        quote,
        activation,
        prior_intent=intent,
        persist_intent=journal.acknowledge,
    )
    assert receipt.cleanup_evidence_version == 2
    assert receipt.cleanup_basis == (
        "interrupted_intent_verified_absent"
        if phase == "receipt_commit_lost"
        else "delete_acknowledged"
    )
    assert receipt.reconciled_after_delete_error is False
    assert receipt.workers_zero_observed_at == intent.workers_zero_observed_at
    assert receipt.delete_started_at == intent.delete_started_at
    assert receipt.verified_absent_at == clock.value
    assert sum(method == "DELETE" for method, _path, _query in state["calls"]) == 1
    assert not state["present"]


async def _save_intent_without_delete(provider, request, quote, activation, journal):
    async def lose_ack(intent):
        await journal.acknowledge(intent)
        raise OSError("test host lost intent COMMIT acknowledgement")

    with pytest.raises(OSError, match="lost intent COMMIT acknowledgement"):
        await provider.cleanup_planned_endpoint(
            request, quote, activation, persist_intent=lose_ack
        )
    return journal.load()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["exception", "none", "foreign_intent", "self_cancel"]
)
async def test_cleanup_intent_ack_failure_never_sends_delete(billing_server, failure):
    state, provider, _clock, request, quote, activation = billing_server

    async def refuse(intent):
        if failure == "exception":
            raise OSError("host persistence failed")
        if failure == "none":
            return None
        if failure == "self_cancel":
            raise asyncio.CancelledError
        return replace(
            intent, delete_started_at=intent.delete_started_at + timedelta(seconds=1)
        )

    with pytest.raises(
        (OSError, ServerlessEndpointCleanupError, asyncio.CancelledError)
    ):
        await provider.cleanup_planned_endpoint(
            request, quote, activation, persist_intent=refuse
        )
    assert state["present"]
    assert not any(method == "DELETE" for method, _path, _query in state["calls"])


@pytest.mark.asyncio
async def test_recovery_does_not_trust_empty_inventory_while_exact_endpoint_exists(
    billing_server, tmp_path
):
    state, provider, _clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    intent = await _save_intent_without_delete(
        provider, request, quote, activation, journal
    )
    state["calls"].clear()
    state["cleanup_inventory"] = []
    state["cleanup_workers_present"] = True
    with pytest.raises(ServerlessEndpointCleanupError, match="active workers"):
        await provider.cleanup_planned_endpoint(
            request,
            quote,
            activation,
            prior_intent=intent,
            persist_intent=journal.acknowledge,
        )
    assert state["present"]
    assert not any(method == "DELETE" for method, _path, _query in state["calls"])
    assert any(path.endswith("/workers") for _method, path, _query in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
async def test_prior_cleanup_intent_does_not_turn_read_errors_into_absence(
    billing_server, tmp_path, status
):
    state, provider, _clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    intent = await _save_intent_without_delete(
        provider, request, quote, activation, journal
    )
    state["calls"].clear()
    state["cleanup_get_status"] = status
    with pytest.raises(RunPodAPIError) as caught:
        await provider.cleanup_planned_endpoint(
            request, quote, activation, prior_intent=intent
        )
    assert caught.value.status_code == status
    assert not any(
        method == "DELETE" or path == "/v2/serverless"
        for method, path, _query in state["calls"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["disconnect", "invalid_resource"])
async def test_cleanup_read_transport_or_parse_failure_does_not_become_absence(
    billing_server, tmp_path, fault
):
    state, provider, _clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    intent = await _save_intent_without_delete(
        provider, request, quote, activation, journal
    )
    state["calls"].clear()
    state["cleanup_get_fault"] = fault
    with pytest.raises(RunPodManagerError):
        await provider.cleanup_planned_endpoint(
            request, quote, activation, prior_intent=intent
        )
    assert not any(
        method == "DELETE" or path == "/v2/serverless"
        for method, path, _query in state["calls"]
    )


@pytest.mark.asyncio
async def test_acknowledged_delete_with_live_exact_endpoint_is_not_cleanup_proof(
    billing_server,
):
    state, provider, _clock, request, quote, activation = billing_server
    state["cleanup_inventory"] = []
    state["cleanup_delete_keeps_present"] = True
    with pytest.raises(ServerlessEndpointCleanupError, match="remains present"):
        await provider.cleanup_planned_endpoint(request, quote, activation)
    assert state["present"]
    assert sum(method == "DELETE" for method, _path, _query in state["calls"]) == 1


@pytest.mark.asyncio
async def test_future_or_receipt_mismatched_intent_fails_before_http(
    billing_server, tmp_path
):
    state, provider, clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    intent = await _save_intent_without_delete(
        provider, request, quote, activation, journal
    )
    state["calls"].clear()
    future = replace(intent, delete_started_at=clock.value + timedelta(seconds=1))
    with pytest.raises(ServerlessEndpointCleanupError, match="future"):
        await provider.cleanup_planned_endpoint(
            request, quote, activation, prior_intent=future
        )
    assert state["calls"] == []
    receipt = await provider.cleanup_planned_endpoint(
        request,
        quote,
        activation,
        prior_intent=intent,
        persist_intent=journal.acknowledge,
    )
    state["calls"].clear()
    clock.value += timedelta(seconds=2)
    mismatch = replace(
        intent, delete_started_at=intent.delete_started_at + timedelta(seconds=1)
    )
    with pytest.raises(
        ServerlessEndpointCleanupError, match="differs from its original intent"
    ):
        await provider.cleanup_planned_endpoint(
            request, quote, activation, prior_intent=mismatch, prior_receipt=receipt
        )
    assert state["calls"] == []


@pytest.mark.asyncio
async def test_absent_endpoint_without_committed_intent_is_not_cleanup_proof(
    billing_server,
):
    state, provider, _clock, request, quote, activation = billing_server
    state["present"] = False
    with pytest.raises(
        ServerlessEndpointCleanupError,
        match="requires original durable cleanup evidence",
    ):
        await provider.cleanup_planned_endpoint(request, quote, activation)
    assert not any(method == "DELETE" for method, _path, _query in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["name_reuse", "id_reappears", "truncated_inventory", "duplicate_ids"]
)
async def test_prior_intent_requires_complete_unambiguous_account_absence(
    billing_server, tmp_path, fault
):
    state, provider, _clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    intent = await _save_intent_without_delete(
        provider, request, quote, activation, journal
    )
    raw = deepcopy(legacy_planned_authority()[3])
    raw["gpu"].update({"minCudaVersion": None, "allowedCudaVersions": []})
    state["present"] = False
    state["calls"].clear()
    if fault == "name_reuse":
        state["cleanup_inventory"] = [{**raw, "id": "foreign-id-same-name"}]
    elif fault == "id_reappears":
        state["cleanup_inventory"] = [raw]
    elif fault == "duplicate_ids":
        state["cleanup_inventory"] = [raw, raw]
    else:
        state["cleanup_inventory_pagination"] = {
            "hasNextPage": True,
            "nextCursor": None,
        }
    with pytest.raises(RunPodManagerError):
        await provider.cleanup_planned_endpoint(
            request, quote, activation, prior_intent=intent
        )
    assert not any(method == "DELETE" for method, _path, _query in state["calls"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["readback", "workers", "intent_ack", "delete", "absence", "absence_error"],
)
async def test_repeated_cancellation_retains_actual_cleanup_and_durable_intent(
    billing_server, tmp_path, boundary
):
    state, provider, _clock, request, quote, activation = billing_server
    journal = _CleanupIntentJournal(tmp_path / "intent.sqlite")
    ack_entered, ack_finish = asyncio.Event(), asyncio.Event()
    path = f"/v2/serverless/{activation.endpoint_id}"
    requests = {
        "readback": ("GET", path),
        "workers": ("GET", path + "/workers"),
        "delete": ("DELETE", path),
        "absence": ("GET", "/v2/serverless"),
        "absence_error": ("GET", "/v2/serverless"),
    }
    if boundary != "intent_ack":
        state["cleanup_block_request"] = requests[boundary]
    if boundary == "absence_error":
        state["cleanup_inventory_pagination"] = {
            "hasNextPage": True,
            "nextCursor": None,
        }

    async def acknowledge(intent):
        result = await journal.acknowledge(intent)
        if boundary == "intent_ack":
            ack_entered.set()
            await ack_finish.wait()
        return result

    owner = asyncio.create_task(
        provider.cleanup_planned_endpoint(
            request, quote, activation, persist_intent=acknowledge
        )
    )
    try:
        if boundary == "intent_ack":
            await asyncio.wait_for(ack_entered.wait(), 5)
            assert not any(
                method == "DELETE" for method, _path, _query in state["calls"]
            )
        else:
            assert await asyncio.to_thread(state["entered"].wait, 5)
        for _ in range(4):
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done()
        state["release"].set()
        ack_finish.set()
        with pytest.raises(
            RunPodManagerError
            if boundary == "absence_error"
            else asyncio.CancelledError
        ):
            await asyncio.wait_for(owner, 5)
        # These observations must belong to the cancelled ORIGINAL cleanup,
        # not be supplied by the recovery invocation below.
        assert any(
            method == "GET" and observed == "/v2/serverless"
            for method, observed, _query in state["calls"]
        )
        assert (
            sum(
                method == "GET" and observed == path
                for method, observed, _query in state["calls"]
            )
            >= 2
        )
        assert not state["present"]
        intent = journal.load()
        state["cleanup_block_request"] = None
        state["cleanup_inventory_pagination"] = None
        recovered = await provider.cleanup_planned_endpoint(
            request, quote, activation, prior_intent=intent
        )
        assert recovered.cleanup_basis == "interrupted_intent_verified_absent"
        assert sum(method == "DELETE" for method, _path, _query in state["calls"]) == 1
    finally:
        state["release"].set()
        ack_finish.set()
        await asyncio.gather(owner, return_exceptions=True)
