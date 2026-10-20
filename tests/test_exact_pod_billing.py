"""Exact Pod billing through the native HTTP client; no Runpod account access."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from pod_capacity_test_support import MutableClock, request

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import RunPodAPIError, RunPodManagerError
from kestrel_cloud_runpod.pod_capacity_contracts import (
    PodCapacitySpec,
    attempt_environment_sha256,
)
from kestrel_cloud_runpod.pod_capacity_provider import RunpodPodCapacityProvider
from kestrel_cloud_runpod.providers import DirectRunPodProvider

DAY = datetime(2026, 10, 20, tzinfo=UTC)
POD = "owned-pod-billing-test"


def _record(hour: int) -> dict:
    return {
        "podId": POD,
        "startTime": (DAY + timedelta(hours=hour)).isoformat(),
        "endTime": (DAY + timedelta(hours=hour + 1)).isoformat(),
        "gpuAmount": 0.2,
        "cpuAmount": 0.1,
        "diskAmount": 0.1,
        "totalAmount": 0.4,
    }


def _history(records: list[dict]) -> dict:
    return {
        "records": records,
        "metadata": {
            "recordCount": len(records),
            "uniquePodCount": 1 if records else 0,
            "totals": {
                name: float(
                    sum((Decimal(str(item[name])) for item in records), Decimal(0))
                )
                for name in ("gpuAmount", "cpuAmount", "diskAmount", "totalAmount")
            },
        },
    }


@pytest.fixture
def native_billing():
    state = {
        "history": _history([_record(hour) for hour in range(3)]),
        "calls": [],
        "blocked": False,
        "entered": threading.Event(),
        "release": threading.Event(),
        "response_status": 200,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer local-loopback-only"
            path = urlsplit(self.path)
            assert path.path == "/v2/billing/pods"
            query = {name: values[0] for name, values in parse_qs(path.query).items()}
            state["calls"].append(query)
            if state["blocked"]:
                state["entered"].set()
                assert state["release"].wait(5), "test must release its HTTP response"
            payload = deepcopy(state["history"])
            payload["metadata"].setdefault("query", query)
            if state["response_status"] != 200:
                payload = {"error": "owned loopback billing observation failed"}
            body = json.dumps(payload).encode()
            self.send_response(state["response_status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RunpodControlPlaneClient(
        api_key="local-loopback-only",
        base_url=f"http://127.0.0.1:{server.server_port}/v2",
        max_safe_retries=0,
    )
    state["client"] = client
    clock = MutableClock()
    clock.value = DAY + timedelta(hours=4)
    item = request(clock)
    spec = PodCapacitySpec(
        request=item,
        capability_secret_id="secret:catalog-capability-0001",
        capability_token_sha256="a" * 64,
        capability_expires_at=item.bearer_expires_at,
        attempt_environment_sha256=attempt_environment_sha256(item.attempt_environment),
    )
    provider = RunpodPodCapacityProvider(
        DirectRunPodProvider("local-loopback-only", client=client), clock=clock
    )

    async def read():
        return await provider.final_billing(
            POD,
            capacity_spec=spec,
            created_at=DAY + timedelta(minutes=10),
            terminated_at=DAY + timedelta(hours=2, minutes=50),
            realized_hourly_rate_usd=Decimal("0.5"),
        )

    try:
        yield state, read, clock
    finally:
        state["release"].set()
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_native_complete_exact_pod_history(native_billing):
    state, read, _ = native_billing
    receipt = await read()
    assert receipt.provider_pod_id == POD
    assert receipt.actual_cost_usd == Decimal("1.2")
    assert len(state["calls"]) == 1


@pytest.mark.asyncio
async def test_native_missing_middle_hour_is_pending_not_min_max_coverage(
    native_billing,
):
    state, read, _ = native_billing
    state["history"] = _history([_record(0), _record(2)])
    assert await read() is None


@pytest.mark.asyncio
async def test_native_rejects_component_total_disagreement(native_billing):
    state, read, _ = native_billing
    state["history"]["records"][0]["gpuAmount"] = 0.5
    with pytest.raises(RunPodManagerError, match="component|amount|total"):
        await read()


@pytest.mark.asyncio
async def test_native_rejects_metadata_total_not_equal_to_records(native_billing):
    state, read, _ = native_billing
    state["history"]["metadata"]["totals"]["totalAmount"] = 9.0
    with pytest.raises(RunPodManagerError, match="component|amount|total"):
        await read()


@pytest.mark.asyncio
async def test_native_component_revision_changes_receipt_without_extra_charge(
    native_billing,
):
    state, read, _ = native_billing
    original = await read()
    changed = state["history"]["records"]
    changed[0]["gpuAmount"] = 0.25
    changed[0]["diskAmount"] = 0.05
    state["history"] = _history(changed)
    revised = await read()
    assert original.actual_cost_usd == revised.actual_cost_usd == Decimal("1.2")
    assert original.provider_billing_id != revised.provider_billing_id


@pytest.mark.asyncio
async def test_native_rejects_foreign_pod_even_with_matching_query(native_billing):
    state, read, _ = native_billing
    state["history"]["records"][0]["podId"] = "unowned-other-pod"
    with pytest.raises(RunPodManagerError, match="identity|Pod|pod"):
        await read()


@pytest.mark.asyncio
async def test_native_rejects_duplicate_overlapping_hour(native_billing):
    state, read, _ = native_billing
    state["history"] = _history([_record(0), _record(0), _record(1), _record(2)])
    with pytest.raises(RunPodManagerError, match="overlap|duplicate|hour"):
        await read()


@pytest.mark.asyncio
async def test_native_rejects_unsupported_pagination_metadata(native_billing):
    state, read, _ = native_billing
    state["history"]["metadata"]["pagination"] = {
        "hasNextPage": True,
        "nextCursor": "not-exhausted",
    }
    with pytest.raises(RunPodManagerError, match="metadata|pagination|field"):
        await read()


@pytest.mark.asyncio
async def test_native_rejects_inconsistent_record_count(native_billing):
    state, read, _ = native_billing
    state["history"]["metadata"]["recordCount"] = 99
    with pytest.raises(RunPodManagerError, match="count|metadata"):
        await read()


@pytest.mark.parametrize("missing", [0, 2])
@pytest.mark.asyncio
async def test_native_missing_boundary_hour_is_pending(native_billing, missing):
    state, read, _ = native_billing
    state["history"] = _history([_record(hour) for hour in range(3) if hour != missing])
    assert await read() is None


@pytest.mark.asyncio
async def test_native_empty_history_is_unknown_not_zero(native_billing):
    state, read, _ = native_billing
    state["history"] = _history([])
    assert await read() is None


@pytest.mark.asyncio
async def test_native_closed_zero_cost_history_is_a_real_observation(native_billing):
    state, read, _ = native_billing
    rows = [_record(hour) for hour in range(3)]
    for row in rows:
        row.update(gpuAmount=0, cpuAmount=0, diskAmount=0, totalAmount=0)
    state["history"] = _history(rows)
    receipt = await read()
    assert receipt is not None and receipt.actual_cost_usd == 0
    assert receipt.billed_seconds == 3 * 3600


@pytest.mark.asyncio
async def test_native_equivalent_numeric_spellings_have_one_identity(native_billing):
    state, read, _ = native_billing
    rows = [_record(hour) for hour in range(3)]
    for row in rows:
        row.update(gpuAmount=1, cpuAmount=0, diskAmount=0, totalAmount=1)
    state["history"] = _history(rows)
    original = await read()
    for row in rows:
        row.update(gpuAmount=1.0, cpuAmount=-0.0, diskAmount=0.0, totalAmount=1.0)
    state["history"] = _history(rows)
    equivalent = await read()
    assert original.provider_billing_id == equivalent.provider_billing_id


@pytest.mark.asyncio
async def test_native_open_hour_never_queries_or_settles(native_billing):
    state, read, clock = native_billing
    clock.value = DAY + timedelta(hours=2, minutes=59)
    assert await read() is None
    assert state["calls"] == []


@pytest.mark.asyncio
async def test_native_retry_preserves_source_identity_not_poll_time(native_billing):
    _, read, clock = native_billing
    original = await read()
    clock.value += timedelta(minutes=5)
    retry = await read()
    assert retry.provider_billing_id == original.provider_billing_id
    assert retry.reconciled_at > original.reconciled_at


@pytest.mark.asyncio
async def test_native_retired_resource_read_needs_no_new_work_authority(native_billing):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        ExecutionCustody,
        ProcessRuntimeExecutionFence,
        bind_execution_cleanup,
        current_execution_custody,
        require_execution_work,
    )

    state, read, _ = native_billing
    original = ExecutionCustody(ProcessRuntimeExecutionFence("original-lifecycle"))
    original.revoke("original Pod lifecycle retired")
    with pytest.raises(ExecutionAuthorityError, match="retired"):
        original.require_work()
    with bind_execution_cleanup(None, (original,)):
        with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
            require_execution_work()
        receipt = await read()
        assert receipt.actual_cost_usd == Decimal("1.2")
        assert current_execution_custody() == (original,)
        with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
            require_execution_work()
    assert len(state["calls"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_native_cancelled_billing_read_joins_physical_http_and_original_outcome(
    native_billing, status
):
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody,
        ProcessRuntimeExecutionFence,
        bind_execution_cleanup,
    )

    state, read, _ = native_billing
    state.update(blocked=True, response_status=status)
    original = ExecutionCustody(ProcessRuntimeExecutionFence("original-lifecycle"))
    original.revoke("original Pod lifecycle retired")
    with bind_execution_cleanup(None, (original,)):
        owner = asyncio.create_task(read())
    try:
        assert await asyncio.to_thread(state["entered"].wait, 5)
        for _ in range(4):
            owner.cancel("original billing observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done(), "physical HTTP response is still owned"
    finally:
        state["release"].set()
    expected = asyncio.CancelledError if status == 200 else RunPodAPIError
    with pytest.raises(expected) as caught:
        await owner
    pending, seen, originals = [caught.value], set(), []
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        originals.append(error)
        pending.append(error.__cause__)
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    assert any(
        isinstance(error, asyncio.CancelledError)
        and error.args == ("original billing observer withdrew",)
        for error in originals
    )
    assert len(state["calls"]) == 1


@pytest.mark.asyncio
async def test_native_row_order_and_equivalent_timezones_have_one_identity(
    native_billing,
):
    state, read, _ = native_billing
    original = await read()
    rows = list(reversed(state["history"]["records"]))
    from datetime import timezone

    offset = timezone(timedelta(hours=-5))
    for row in rows:
        for name in ("startTime", "endTime"):
            row[name] = datetime.fromisoformat(row[name]).astimezone(offset).isoformat()
    state["history"] = _history(rows)
    assert (await read()).provider_billing_id == original.provider_billing_id


@pytest.mark.asyncio
async def test_native_hour_revision_is_distinct_even_when_total_cost_is_equal(
    native_billing,
):
    state, read, _ = native_billing
    original = await read()
    rows = state["history"]["records"]
    rows[0].update(gpuAmount=0.3, totalAmount=0.5)
    rows[1].update(gpuAmount=0.1, totalAmount=0.3)
    state["history"] = _history(rows)
    revised = await read()
    assert original.actual_cost_usd == revised.actual_cost_usd
    assert original.provider_billing_id != revised.provider_billing_id


@pytest.mark.asyncio
async def test_native_generic_reader_needs_no_training_capacity_spec(native_billing):
    from kestrel_cloud_runpod import read_exact_pod_billing

    state, read, clock = native_billing
    direct = await read_exact_pod_billing(
        state["client"],
        pod_id=POD,
        created_at=DAY + timedelta(minutes=10),
        terminated_at=DAY + timedelta(hours=2, minutes=50),
        realized_hourly_rate_usd=Decimal("0.5"),
        clock=clock,
    )
    assert direct == await read()


@pytest.mark.parametrize(
    "change",
    [
        "root_pagination",
        "wrong_bucket",
        "wrong_query_pod",
        "wrong_query_time",
        "unique_count",
        "negative_amount",
        "boolean_amount",
        "extra_record_field",
        "extra_totals_field",
        "oversize_hour",
        "outside_window",
        "naive_timestamp",
    ],
)
@pytest.mark.asyncio
async def test_native_rejects_unsafe_history_dimensions(native_billing, change):
    state, read, _ = native_billing
    history = state["history"]
    query = {
        "podId": POD,
        "bucketSize": "hour",
        "startTime": DAY.isoformat(),
        "endTime": (DAY + timedelta(hours=3)).isoformat(),
    }
    if change == "root_pagination":
        history["pagination"] = {"hasNextPage": True, "nextCursor": "unread"}
    elif change == "wrong_bucket":
        history["metadata"]["query"] = {**query, "bucketSize": "day"}
    elif change == "wrong_query_pod":
        history["metadata"]["query"] = {**query, "podId": "other-pod"}
    elif change == "wrong_query_time":
        history["metadata"]["query"] = {**query, "endTime": DAY.isoformat()}
    elif change == "unique_count":
        history["metadata"]["uniquePodCount"] = 2
    elif change == "negative_amount":
        history["records"][0]["gpuAmount"] = -1
    elif change == "boolean_amount":
        history["records"][0]["gpuAmount"] = True
    elif change == "extra_record_field":
        history["records"][0]["private_input"] = "must-not-be-accepted"
    elif change == "extra_totals_field":
        history["metadata"]["totals"]["unexpectedAmount"] = 1
    elif change == "oversize_hour":
        history["records"][0]["endTime"] = (DAY + timedelta(hours=2)).isoformat()
    elif change == "outside_window":
        history["records"][-1]["endTime"] = (DAY + timedelta(hours=4)).isoformat()
    elif change == "naive_timestamp":
        history["records"][0]["startTime"] = DAY.replace(tzinfo=None).isoformat()
    with pytest.raises(RunPodManagerError):
        await read()
