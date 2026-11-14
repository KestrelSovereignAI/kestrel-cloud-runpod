"""Cold-start observations over owned native HTTP; never a Runpod account."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from ollama_test_support import OwnedLoopbackHTTPServer

import kestrel_cloud_runpod.endpoint_metrics as metrics
from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import RunPodAPIError, RunPodManagerError


def bucket(hour=13, **changes):
    value = {
        "time": f"2026-11-14T{hour:02d}:00:00Z",
        "count": 2,
        "durationMs": {
            "min": 8200,
            "max": 11900,
            "total": 20100.25,
            "p70": 11900,
            "p90": 11900,
            "p95": 11900,
            "p98": 11900,
        },
    }
    value.update(changes)
    return value


def page(*records, cursor=None, **changes):
    value = {
        "startTime": "2026-11-14T00:00:00Z",
        "endTime": "2026-11-15T00:00:00Z",
        "interval": "1h",
        "records": list(records),
        "pagination": {"hasNextPage": cursor is not None, "nextCursor": cursor},
    }
    value.update(changes)
    return value


@pytest.fixture
def cold_start_http():
    state = {"responses": [], "calls": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.connection.settimeout(5)
            state["calls"].append((self.command, self.path, dict(self.headers)))
            assert self.headers["Authorization"] == "Bearer owned-metrics-only"
            assert self.headers["User-Agent"] == "kestrel-cloud-runpod/v2"
            assert state["responses"], "unexpected HTTP request"
            status, payload = state["responses"].pop(0)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("RateLimit", '"minute";r=179;t=30')
            self.send_header("RateLimit-Policy", '"minute";q=180;w=60')
            self.end_headers()
            self.wfile.write(body)

    def server_factory():
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = False
        return server

    owner = OwnedLoopbackHTTPServer(
        server_factory=server_factory,
        thread_factory=lambda server: threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        ),
        client_factory=lambda server: RunpodControlPlaneClient(
            api_key="owned-metrics-only",
            max_safe_retries=0,
            base_url=f"http://127.0.0.1:{server.server_port}/v2",
        ),
    )
    with owner as client:
        yield state, client
    assert owner.cleanup_errors == ()
    assert not owner.thread.is_alive()


def test_native_sparse_pages_preserve_cursor_query_and_vendor_statistics(
    cold_start_http,
):
    state, client = cold_start_http
    cursor = " opaque+/= &? cursor "
    state["responses"] = [
        (200, page(bucket(10), cursor=cursor)),
        (200, page(bucket(12))),
    ]
    before = datetime.now(UTC)
    observed = client.endpoint_cold_start_metrics("ep/a?b", limit=1)
    assert before <= observed.observed_at <= datetime.now(UTC)
    assert observed.endpoint_id == "ep/a?b"
    assert observed.interval is metrics.EndpointMetricsInterval.ONE_HOUR
    assert [item.time.hour for item in observed.records] == [10, 12]
    assert len(observed.pages) == 2
    assert observed.pages[0].pagination.next_cursor == cursor
    assert observed.pages[-1].pagination.has_next_page is False
    assert observed.pages[-1].pagination.next_cursor is None
    assert observed.start_time == datetime(2026, 11, 14, tzinfo=UTC)
    assert observed.end_time == datetime(2026, 11, 15, tzinfo=UTC)
    for item in observed.records:
        assert item.count == 2
        assert item.duration_ms.total_ms == Decimal("20100.25")
        assert item.duration_ms.minimum_ms == 8200
        assert item.duration_ms.maximum_ms == 11900
        assert (
            item.duration_ms.p70_ms,
            item.duration_ms.p90_ms,
            item.duration_ms.p95_ms,
            item.duration_ms.p98_ms,
        ) == (11900,) * 4
    first, second = (urlsplit(call[1]) for call in state["calls"])
    assert first.path == second.path == "/v2/serverless/ep%2Fa%3Fb/metrics/cold-starts"
    assert parse_qs(first.query) == {"limit": ["1"]}
    assert parse_qs(second.query) == {"limit": ["1"], "cursor": [cursor]}
    assert state["responses"] == []
    assert client.transport.last_rate_limit is not None


def test_native_empty_is_empty_not_synthetic_zero_buckets(cold_start_http):
    state, client = cold_start_http
    state["responses"] = [(200, page())]
    observed = client.endpoint_cold_start_metrics("ep-empty")
    assert observed.records == () and len(observed.pages) == 1
    assert urlsplit(state["calls"][0][1]).query == ""


def test_native_unaligned_start_and_future_end_keep_original_query(cold_start_http):
    state, client = cold_start_http
    start, future = "2026-11-14T13:15:00Z", "2099-01-01T00:00:00Z"
    first = page(bucket(13), cursor="next", startTime=start)
    second = page(bucket(15), startTime=start)
    state["responses"] = [(200, first), (200, second)]
    observed = client.endpoint_cold_start_metrics(
        "ep", start_time=start, end_time=future, interval="1h"
    )
    assert observed.records[0].time < observed.start_time
    for _, path, _ in state["calls"]:
        query = parse_qs(urlsplit(path).query)
        assert query["startTime"] == [start] and query["endTime"] == [future]
        assert query["interval"] == ["1h"]


@pytest.mark.parametrize("interval", list(metrics.EndpointMetricsInterval))
@pytest.mark.parametrize("limit", [1, 1000])
def test_native_documented_interval_and_limit_bounds(cold_start_http, interval, limit):
    state, client = cold_start_http
    state["responses"] = [(200, page(interval=interval.value))]
    observed = client.endpoint_cold_start_metrics("ep", interval=interval, limit=limit)
    assert observed.interval is interval
    assert parse_qs(urlsplit(state["calls"][0][1]).query) == {
        "interval": [interval.value],
        "limit": [str(limit)],
    }


@pytest.mark.parametrize(
    "identity,kwargs",
    [
        (None, {}),
        (3, {}),
        ("", {}),
        (" ", {}),
        (".", {}),
        ("..", {}),
        ("ep\n", {}),
        *[("ep", {"limit": value}) for value in (0, 1001, -1, True, 1.5, "1")],
        *[("ep", {"max_pages": value}) for value in (0, -1, True, 1.5, "1")],
        *[("ep", {"interval": value}) for value in ("2h", "", 1, True)],
        *[
            ("ep", {key: value})
            for key in ("start_time", "end_time")
            for value in (
                3,
                "",
                "2026-11-14",
                "2026-11-14 00:00:00Z",
                "2026-11-14T00:00:00",
                "2026-99-14T00:00:00Z",
            )
        ],
        (
            "ep",
            {"start_time": "2026-11-15T00:00:00Z", "end_time": "2026-11-14T00:00:00Z"},
        ),
    ],
)
def test_bad_input_fails_before_any_native_http(cold_start_http, identity, kwargs):
    state, client = cold_start_http
    with pytest.raises(ValueError):
        client.endpoint_cold_start_metrics(identity, **kwargs)
    assert state["calls"] == []


@pytest.mark.parametrize(
    "kind",
    [
        "missing-start",
        "missing-end",
        "missing-interval",
        "missing-records",
        "missing-pagination",
        "naive-start",
        "invalid-end",
        "unknown-interval",
        "reversed-window",
        "records-object",
        "record-null",
        "missing-count",
        "bool-count",
        "negative-count",
        "duration-null",
        "missing-total",
        "bool-total",
        "nan-total",
        "infinite-total",
        "negative-total",
        "huge-total",
        "float-percentile",
        "negative-percentile",
        "bucket-naive",
        "bucket-outside",
        "descending",
        "duplicate",
        "pagination-missing-flag",
        "pagination-int-flag",
        "pagination-missing-cursor",
        "pagination-no-next",
        "pagination-final-cursor",
        "pagination-blank-cursor",
    ],
)
def test_malformed_response_never_becomes_success_or_zero(cold_start_http, kind):
    state, client = cold_start_http
    value = page(bucket())
    if kind.startswith("missing-") and kind.split("-", 1)[1] in {
        "start",
        "end",
        "interval",
        "records",
        "pagination",
    }:
        value.pop({"start": "startTime", "end": "endTime"}.get(kind[8:], kind[8:]))
    else:
        changes = {
            "naive-start": lambda: value.update(startTime="2026-11-14T00:00:00"),
            "invalid-end": lambda: value.update(endTime="not-a-time"),
            "unknown-interval": lambda: value.update(interval="2h"),
            "reversed-window": lambda: value.update(endTime=value["startTime"]),
            "records-object": lambda: value.update(records={}),
            "record-null": lambda: value.update(records=[None]),
            "missing-count": lambda: value["records"][0].pop("count"),
            "bool-count": lambda: value["records"][0].update(count=True),
            "negative-count": lambda: value["records"][0].update(count=-1),
            "duration-null": lambda: value["records"][0].update(durationMs=None),
            "missing-total": lambda: value["records"][0]["durationMs"].pop("total"),
            "bool-total": lambda: value["records"][0]["durationMs"].update(total=True),
            "nan-total": lambda: value["records"][0]["durationMs"].update(
                total=float("nan")
            ),
            "infinite-total": lambda: value["records"][0]["durationMs"].update(
                total=float("inf")
            ),
            "negative-total": lambda: value["records"][0]["durationMs"].update(
                total=-1
            ),
            "huge-total": lambda: value["records"][0]["durationMs"].update(
                total=10**400
            ),
            "float-percentile": lambda: value["records"][0]["durationMs"].update(
                p90=11900.5
            ),
            "negative-percentile": lambda: value["records"][0]["durationMs"].update(
                p90=-1
            ),
            "bucket-naive": lambda: value["records"][0].update(
                time="2026-11-14T13:00:00"
            ),
            "bucket-outside": lambda: value["records"][0].update(time=value["endTime"]),
            "descending": lambda: value.update(records=[bucket(13), bucket(12)]),
            "duplicate": lambda: value.update(records=[bucket(), bucket()]),
            "pagination-missing-flag": lambda: value["pagination"].pop("hasNextPage"),
            "pagination-int-flag": lambda: value["pagination"].update(hasNextPage=0),
            "pagination-missing-cursor": lambda: value["pagination"].pop("nextCursor"),
            "pagination-no-next": lambda: value["pagination"].update(hasNextPage=True),
            "pagination-final-cursor": lambda: value["pagination"].update(
                nextCursor="secret-cursor"
            ),
            "pagination-blank-cursor": lambda: value["pagination"].update(
                nextCursor=" ", hasNextPage=True
            ),
        }
        changes[kind]()
    state["responses"] = [(200, value)]
    with pytest.raises(RunPodManagerError) as caught:
        client.endpoint_cold_start_metrics("ep")
    assert "secret-cursor" not in str(caught.value)
    assert len(state["calls"]) == 1


@pytest.mark.parametrize(
    "kind",
    ["start", "end", "interval", "duplicate", "descending", "cycle", "bound", "limit"],
)
def test_conflicting_or_incomplete_pages_fail_atomically(cold_start_http, kind):
    state, client = cold_start_http
    first, second = page(bucket(12), cursor="next"), page(bucket(13))
    kwargs = {}
    if kind in {"start", "end"}:
        second[kind + "Time"] = (
            "2026-11-14T01:00:00Z" if kind == "start" else "2026-11-15T01:00:00Z"
        )
    elif kind == "interval":
        second["interval"] = "30m"
    elif kind in {"duplicate", "descending"}:
        second["records"] = [bucket(12 if kind == "duplicate" else 11)]
    elif kind == "cycle":
        second["pagination"] = deepcopy(first["pagination"])
    elif kind == "bound":
        kwargs["max_pages"] = 1
    elif kind == "limit":
        kwargs["limit"] = 1
        second["records"] = [bucket(13), bucket(14)]
    state["responses"] = [(200, first), (200, second)]
    with pytest.raises(RunPodManagerError):
        client.endpoint_cold_start_metrics("ep", **kwargs)
    assert len(state["calls"]) == (1 if kind == "bound" else 2)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 429, 503])
def test_native_provider_failure_is_not_empty_success_and_is_sanitized(
    cold_start_http, status
):
    state, client = cold_start_http
    state["responses"] = [
        (200, page(cursor="next")),
        (
            status,
            {
                "title": "Metrics unavailable",
                "status": status,
                "detail": "https://secret.example/?token=private-value Bearer private-value",
            },
        ),
    ]
    with pytest.raises(RunPodAPIError) as caught:
        client.endpoint_cold_start_metrics("ep")
    assert caught.value.status_code == status
    assert "private-value" not in str(caught.value)
    assert "secret.example" not in str(caught.value)
    assert len(state["calls"]) == 2


def test_native_control_from_response_boundary_is_same_original(
    cold_start_http, monkeypatch
):
    state, client = cold_start_http
    state["responses"] = [(200, page(bucket()))]
    original = ExecutionAuthorityError("original native control")

    def fail_integer(*_args):
        raise original

    monkeypatch.setattr(metrics, "_integer", fail_integer)
    with pytest.raises(ExecutionAuthorityError) as caught:
        client.endpoint_cold_start_metrics("ep")
    assert caught.value is original
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("kind", ["start", "end", "default-interval"])
def test_first_response_must_match_explicit_query_and_documented_defaults(
    cold_start_http, kind
):
    state, client = cold_start_http
    value, kwargs = page(), {}
    if kind == "start":
        kwargs["start_time"] = "2026-11-14T01:00:00Z"
    elif kind == "end":
        kwargs["end_time"] = "2026-11-14T23:00:00Z"
    else:
        value["interval"] = "5m"
    state["responses"] = [(200, value)]
    with pytest.raises(RunPodManagerError):
        client.endpoint_cold_start_metrics("ep", **kwargs)


@pytest.mark.parametrize(
    "interval,time",
    [
        ("1m", "13:00:30"),
        ("5m", "13:01:00"),
        ("15m", "13:05:00"),
        ("30m", "13:15:00"),
        ("1h", "13:30:00"),
        ("6h", "13:00:00"),
        ("12h", "06:00:00"),
        ("1d", "12:00:00"),
        ("1h", "13:00:00.000001"),
        ("1h", "13:00:00.0000001"),
    ],
)
def test_native_rejects_bucket_not_on_utc_interval_grid(
    cold_start_http, interval, time
):
    state, client = cold_start_http
    state["responses"] = [
        (200, page(bucket(time=f"2026-11-14T{time}Z"), interval=interval))
    ]
    with pytest.raises(RunPodManagerError, match="align"):
        client.endpoint_cold_start_metrics("ep", interval=interval)
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("across_pages", [False, True])
@pytest.mark.parametrize("time", ["13:30:00", "13:00:00.000001", "13:00:00.0000001"])
def test_native_rejects_overlapping_hour_buckets(cold_start_http, across_pages, time):
    state, client = cold_start_http
    second = bucket(time=f"2026-11-14T{time}Z")
    if across_pages:
        state["responses"] = [(200, page(bucket(), cursor="next")), (200, page(second))]
    else:
        state["responses"] = [(200, page(bucket(), second))]
    with pytest.raises(RunPodManagerError):
        client.endpoint_cold_start_metrics("ep")
    assert len(state["calls"]) == (2 if across_pages else 1)


@pytest.mark.parametrize("offset", ["+00:60", "+01:99", "-00:60"])
@pytest.mark.parametrize("key", ["start_time", "end_time"])
def test_bad_rfc3339_offset_fails_before_native_http(cold_start_http, offset, key):
    state, client = cold_start_http
    value = f"2026-11-14T13:00:00{offset}"
    wire_key = "startTime" if key == "start_time" else "endTime"
    state["responses"] = [(200, page(**{wire_key: value}))]
    with pytest.raises(ValueError, match="RFC3339"):
        client.endpoint_cold_start_metrics("ep", **{key: value})
    assert state["calls"] == []


@pytest.mark.parametrize("offset", ["+00:60", "+01:99", "-00:60"])
@pytest.mark.parametrize("field", ["startTime", "endTime", "bucket"])
def test_native_rejects_malformed_response_offset(cold_start_http, offset, field):
    state, client = cold_start_http
    value = page(bucket())
    day = "15" if field == "endTime" else "14"
    timestamp = f"2026-11-{day}T13:00:00{offset}"
    if field == "bucket":
        value["records"][0]["time"] = timestamp
    else:
        value[field] = timestamp
    state["responses"] = [(200, value)]
    with pytest.raises(RunPodManagerError):
        client.endpoint_cold_start_metrics("ep")
    assert len(state["calls"]) == 1


def test_native_aligned_instants_allow_equivalent_offsets_and_partial_query_window(
    cold_start_http,
):
    state, client = cold_start_http
    start = "2026-11-14T13:15:00Z"
    state["responses"] = [
        (
            200,
            page(
                bucket(time="2026-11-14T13:30:00+01:30"), startTime=start, interval="6h"
            ),
        )
    ]
    observed = client.endpoint_cold_start_metrics("ep", start_time=start, interval="6h")
    assert observed.records[0].time == datetime(2026, 11, 14, 12, tzinfo=UTC)
    assert observed.records[0].time < observed.start_time


def test_native_valid_offset_components_preserve_the_original_query(cold_start_http):
    state, client = cold_start_http
    start = "2026-11-14T13:00:00+00:59"
    state["responses"] = [(200, page(startTime="2026-11-14T12:01:00Z"))]
    observed = client.endpoint_cold_start_metrics("ep", start_time=start)
    assert observed.start_time == datetime(2026, 11, 14, 12, 1, tzinfo=UTC)
    assert parse_qs(urlsplit(state["calls"][0][1]).query)["startTime"] == [start]


def test_native_last_datetime_bucket_does_not_require_overflowing_its_end(
    cold_start_http,
):
    state, client = cold_start_http
    start = "9999-12-31T00:00:00Z"
    state["responses"] = [
        (
            200,
            page(
                bucket(time=start),
                startTime=start,
                endTime="9999-12-31T23:59:59.999999Z",
                interval="1d",
            ),
        )
    ]
    observed = client.endpoint_cold_start_metrics("ep", interval="1d")
    assert observed.records[0].time == datetime(9999, 12, 31, tzinfo=UTC)


@pytest.mark.parametrize(
    "field,first,second",
    [
        ("startTime", "2026-11-14T00:00:00Z", "2026-11-14T00:00:00+00:00"),
        ("startTime", "2026-11-14T00:00:00Z", "2026-11-14T01:00:00+01:00"),
        ("endTime", "2026-11-15T00:00:00Z", "2026-11-15T00:00:00.000Z"),
        (
            "startTime",
            "2026-11-14T00:00:00.123456789Z",
            "2026-11-14T01:00:00.1234567890+01:00",
        ),
    ],
)
def test_native_pages_accept_equivalent_exact_window_instants(
    cold_start_http, field, first, second
):
    state, client = cold_start_http
    state["responses"] = [
        (200, page(cursor="next", **{field: first})),
        (200, page(**{field: second})),
    ]
    observed = client.endpoint_cold_start_metrics("ep")
    assert len(observed.pages) == 2
    actual = "start_time_text" if field == "startTime" else "end_time_text"
    assert getattr(observed, actual) == first
    assert getattr(observed.pages[0], actual) == first
    assert getattr(observed.pages[1], actual) == second
    view = "start_time" if field == "startTime" else "end_time"
    assert getattr(observed, view) == datetime.fromisoformat(first).astimezone(UTC)


@pytest.mark.parametrize(
    "key,requested,response",
    [
        ("start_time", "2026-11-14T00:00:00.0000001Z", "2026-11-14T00:00:00.0000009Z"),
        ("end_time", "2026-11-15T00:00:00.0000001Z", "2026-11-15T00:00:00.0000009Z"),
        (
            "end_time",
            "2026-11-15T00:00:00.1Z",
            "2026-11-15T00:00:00.100000000000000000001Z",
        ),
    ],
)
def test_native_response_must_honor_full_precision_query_boundaries(
    cold_start_http, key, requested, response
):
    state, client = cold_start_http
    field = "startTime" if key == "start_time" else "endTime"
    state["responses"] = [(200, page(**{field: response}))]
    with pytest.raises(RunPodManagerError):
        client.endpoint_cold_start_metrics("ep", **{key: requested})
    assert len(state["calls"]) == 1


@pytest.mark.parametrize(
    "start,end,records",
    [
        ("2026-11-14T13:00:00.0000001Z", "2026-11-14T13:00:00.0000002Z", (13,)),
        ("2026-11-14T13:59:59.9999999Z", "2026-11-14T14:00:00.0000001Z", (13, 14)),
    ],
)
def test_native_submicrosecond_windows_and_bucket_overlap_are_not_rounded_away(
    cold_start_http, start, end, records
):
    state, client = cold_start_http
    state["responses"] = [
        (200, page(*(bucket(hour) for hour in records), startTime=start, endTime=end))
    ]
    observed = client.endpoint_cold_start_metrics("ep", start_time=start, end_time=end)
    assert [item.time.hour for item in observed.records] == list(records)
    assert observed.pages[0].start_time_text == start
    assert observed.pages[0].end_time_text == end


@pytest.mark.parametrize("field", ["startTime", "endTime"])
def test_native_cross_page_submicrosecond_drift_still_fails(cold_start_http, field):
    state, client = cold_start_http
    day = "14" if field == "startTime" else "15"
    first = f"2026-11-{day}T00:00:00.0000001Z"
    second = f"2026-11-{day}T00:00:00.0000009Z"
    state["responses"] = [
        (200, page(cursor="next", **{field: first})),
        (200, page(**{field: second})),
    ]
    with pytest.raises(RunPodManagerError):
        client.endpoint_cold_start_metrics("ep")
    assert len(state["calls"]) == 2
