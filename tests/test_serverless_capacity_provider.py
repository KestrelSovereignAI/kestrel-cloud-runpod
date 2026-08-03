"""REST v2 adapter tests for Serverless quote and billing reconciliation."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from serverless_capacity_test_support import (
    MutableClock,
    ambiguous_window,
    attempt,
    endpoint,
    offer,
    quote,
    request,
)

from kestrel_cloud_runpod.clients import (
    RunpodControlPlaneClient,
    RunpodServerlessClient,
)
from kestrel_cloud_runpod.models import (
    Availability,
    BillingPage,
    RunPodManagerError,
    ServerlessJob,
)
from kestrel_cloud_runpod.serverless_capacity_provider import (
    RunpodServerlessCapacityProvider,
)


def _gpu_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "PRO-6000-MIG-1g-24gb",
        "name": "PRO 6000 MIG 1g.24gb",
        "pool": "BLACKWELL_24",
        "manufacturer": "NVIDIA",
        "memory": 24,
        "secure": True,
        "community": False,
        "price": {"secure": 0.69, "community": 0.0},
        "maxCount": {"secure": 0, "community": 0},
        "availability": "HIGH",
        "dataCenters": [{"id": "US-TX-3", "availability": "HIGH"}],
    }
    payload.update(changes)
    return payload


def _endpoint_payload() -> dict[str, object]:
    return dict(endpoint().raw)


@pytest.mark.asyncio
async def test_quote_is_exact_read_only_v2_catalog_and_endpoint_observation() -> None:
    seen: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        if http_request.url.path == "/v2/catalog/gpus":
            return httpx.Response(200, json={"gpus": [_gpu_payload()]})
        if http_request.url.path == "/v2/serverless/endpoint-selfie-01":
            return httpx.Response(200, json=_endpoint_payload())
        raise AssertionError(
            f"unexpected request: {http_request.method} {http_request.url}"
        )

    control = RunpodControlPlaneClient(
        api_key="control-secret",
        http_transport=httpx.MockTransport(handler),
    )
    provider = RunpodServerlessCapacityProvider(
        control_client=control,
        clock=MutableClock(),
    )

    result = await provider.quote(request())

    assert [item.method for item in seen] == ["GET", "GET"]
    assert dict(seen[0].url.params) == {
        "include": "AVAILABILITY",
        "product": "SERVERLESS",
        "count": "1",
        "cloud": "SECURE",
        "minCudaVersion": "12.8",
    }
    assert result.gpu_id == "PRO-6000-MIG-1g-24gb"
    assert result.gpu_pool == "BLACKWELL_24"
    assert result.vram_gb == 24
    assert result.data_center_id == "US-TX-3"
    assert len(result.catalog_observation_sha256) == 64
    assert result.hourly_worker_rate_usd == Decimal("0.69")
    assert result.estimated_cost_usd == Decimal("0.019667")
    assert result.cost_ceiling_usd == Decimal("0.035500")
    assert "requestUrls" not in result.to_dict()
    assert "private_runtime_value" not in json.dumps(result.to_dict()).lower()


@pytest.mark.asyncio
async def test_billing_requires_a_separate_restricted_job_status_client() -> None:
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(),
        clock=clock,
    )
    with pytest.raises(RunPodManagerError, match="restricted job-status client"):
        await provider.final_billing(attempt(), quote())


@pytest.mark.asyncio
async def test_quote_uses_existing_selector_for_mig_cuda_region_and_availability() -> (
    None
):
    selected = offer()
    low = offer(
        id="LOW-MIG",
        name="Low MIG",
        availability=Availability.LOW,
        pool="OTHER_POOL",
    )
    control = SimpleNamespace(
        list_gpus=lambda **_: (low, selected),
        get_endpoint=lambda _: endpoint(),
    )
    provider = RunpodServerlessCapacityProvider(
        control_client=control,
        job_client=SimpleNamespace(),
        clock=MutableClock(),
    )

    result = await provider.quote(request())

    assert result.gpu_id == selected.id
    assert selected.secure_max_count == 0
    assert result.min_cuda_version == "12.8"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_change", "message"),
    [
        ({"type": "LOAD_BALANCER"}, "identity or type"),
        ({"image": "wrong@sha256:" + "f" * 64}, "runtime or billing"),
        ({"gpu": {"pools": ["OTHER_POOL"], "count": 1}}, "quoted pool"),
        ({"dataCenterIds": ["US-KS-2"]}, "quoted data center"),
        ({"workers": {"min": 0, "max": 2, "idleTimeout": 10}}, "worker policy"),
        ({"timeout": 60_000}, "runtime or billing"),
    ],
)
async def test_quote_rejects_endpoint_profile_mismatch(
    raw_change: dict[str, object], message: str
) -> None:
    actual = endpoint()
    mismatched = replace(actual, raw={**actual.raw, **raw_change})
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            list_gpus=lambda **_: (offer(),),
            get_endpoint=lambda _: mismatched,
        ),
        job_client=SimpleNamespace(),
        clock=MutableClock(),
    )

    with pytest.raises(RunPodManagerError, match=message):
        await provider.quote(request())


@pytest.mark.asyncio
async def test_quote_rejects_shared_pool_and_unsafe_catalog_numbers() -> None:
    shared = offer(id="OTHER-GPU", name="Other GPU")
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            list_gpus=lambda **_: (offer(), shared),
            get_endpoint=lambda _: endpoint(),
        ),
        job_client=SimpleNamespace(),
        clock=MutableClock(),
    )
    with pytest.raises(RunPodManagerError, match="one exact catalog GPU"):
        await provider.quote(request())

    provider.control_client.list_gpus = lambda **_: (
        offer(secure_price_per_hr=float("nan")),
    )
    with pytest.raises(RunPodManagerError, match="unsafe GPU rate"):
        await provider.quote(request())


@pytest.mark.asyncio
async def test_submission_validation_rejects_rate_pool_workload_and_expiry_drift() -> (
    None
):
    clock = MutableClock()
    offers = [offer()]
    control = SimpleNamespace(
        list_gpus=lambda **_: tuple(offers),
        get_endpoint=lambda _: endpoint(),
    )
    provider = RunpodServerlessCapacityProvider(
        control_client=control,
        job_client=SimpleNamespace(),
        clock=clock,
    )
    original_request = request()
    accepted = await provider.quote(original_request)
    clock.value += timedelta(seconds=1)

    current = await provider.validate_quote_for_submission(
        original_request,
        accepted,
        accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
    )
    assert current is accepted
    assert current.hourly_worker_rate_usd == accepted.hourly_worker_rate_usd

    offers[:] = [offer(secure_price_per_hr=0.68)]
    assert (
        await provider.validate_quote_for_submission(
            original_request,
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )
        is accepted
    )

    offers[:] = [offer(secure_price_per_hr=0.70)]
    with pytest.raises(RunPodManagerError, match="rate increased"):
        await provider.validate_quote_for_submission(
            original_request,
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )

    with pytest.raises(RunPodManagerError, match="workload or configured profile"):
        await provider.validate_quote_for_submission(
            request(workload_kind="other-selfie"),
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )

    with pytest.raises(RunPodManagerError, match="workload or configured profile"):
        await provider.validate_quote_for_submission(
            request(estimated_execution_seconds=31),
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )

    with pytest.raises(RunPodManagerError, match="workload or configured profile"):
        await provider.validate_quote_for_submission(
            request(job_execution_timeout_ms=119_000),
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )

    with pytest.raises(RunPodManagerError, match="workload or configured profile"):
        await provider.validate_quote_for_submission(
            request(job_ttl_ms=301_000),
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )

    clock.value = accepted.expires_at
    with pytest.raises(RunPodManagerError, match="expired"):
        await provider.validate_quote_for_submission(
            original_request,
            accepted,
            accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        )


def _billing_page(
    *,
    endpoint_id: str = "endpoint-selfie-01",
    record_start: str = "2026-08-03T10:00:00+00:00",
    record_end: str = "2026-08-03T11:00:00+00:00",
    records: bool = True,
    extra_record: dict[str, object] | None = None,
) -> BillingPage:
    record = {
        "startTime": record_start,
        "endTime": record_end,
        "serverlessId": endpoint_id,
        "totalAmount": 0.023,
        "gpuAmount": 0.020,
        "cpuAmount": 0.0,
        "diskAmount": 0.001,
        "feeAmount": 0.002,
    }
    if extra_record:
        record.update(extra_record)
    items = (record,) if records else ()
    totals = {
        "totalAmount": 0.023 if records else 0.0,
        "gpuAmount": 0.020 if records else 0.0,
        "cpuAmount": 0.0,
        "diskAmount": 0.001 if records else 0.0,
        "feeAmount": 0.002 if records else 0.0,
    }
    return BillingPage(
        records=items,
        metadata={
            "query": {
                "startTime": "2026-08-03T10:00:00+00:00",
                "endTime": "2026-08-03T11:00:00+00:00",
                "bucketSize": "hour",
                "serverlessId": endpoint_id,
            },
            "recordCount": len(items),
            "uniqueServerlessCount": 1 if records else 0,
            "totals": totals,
        },
    )


def _terminal_job(**changes: object) -> ServerlessJob:
    values: dict[str, object] = {
        "id": "job-selfie-0001",
        "status": "COMPLETED",
        "output": {"artifact": "must-not-serialize"},
        "delay_time_ms": 2_000,
        "execution_time_ms": 30_000,
        "raw": {"prompt": "must-not-serialize"},
    }
    values.update(changes)
    return ServerlessJob(**values)  # type: ignore[arg-type]


def _two_hour_billing_page() -> BillingPage:
    records = (
        {
            "startTime": "2026-08-03T10:00:00+00:00",
            "endTime": "2026-08-03T11:00:00+00:00",
            "serverlessId": "endpoint-selfie-01",
            "totalAmount": 0.0115,
            "gpuAmount": 0.010,
            "cpuAmount": 0.0,
            "diskAmount": 0.0005,
            "feeAmount": 0.001,
        },
        {
            "startTime": "2026-08-03T11:00:00+00:00",
            "endTime": "2026-08-03T12:00:00+00:00",
            "serverlessId": "endpoint-selfie-01",
            "totalAmount": 0.022,
            "gpuAmount": 0.018,
            "cpuAmount": 0.0,
            "diskAmount": 0.001,
            "feeAmount": 0.003,
        },
    )
    return BillingPage(
        records=records,
        metadata={
            "query": {
                "startTime": "2026-08-03T10:00:00+00:00",
                "endTime": "2026-08-03T12:00:00+00:00",
                "bucketSize": "hour",
                "serverlessId": "endpoint-selfie-01",
            },
            "recordCount": 2,
            "uniqueServerlessCount": 1,
            "totals": {
                "totalAmount": 0.0335,
                "gpuAmount": 0.028,
                "cpuAmount": 0.0,
                "diskAmount": 0.0015,
                "feeAmount": 0.004,
            },
        },
    )


@pytest.mark.asyncio
async def test_final_billing_binds_terminal_job_and_complete_endpoint_window() -> None:
    clock = MutableClock()
    clock.value = clock.value + timedelta(hours=1)
    billing_calls: list[dict[str, object]] = []
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **kwargs: (
                billing_calls.append(kwargs) or _billing_page()
            )
        ),
        job_client=SimpleNamespace(status=lambda *_: _terminal_job()),
        clock=clock,
    )
    accepted = quote()
    billing_attempt = attempt(accepted)

    receipt = await provider.final_billing(billing_attempt, accepted)
    replay = await provider.final_billing(billing_attempt, accepted)

    assert receipt is not None and replay is not None
    assert receipt.provider_billing_id == replay.provider_billing_id
    assert receipt.endpoint_id == billing_attempt.endpoint_id
    assert receipt.job_id == billing_attempt.job_id
    assert receipt.attempt_id == billing_attempt.attempt_id
    assert receipt.actual_cost_usd == Decimal("0.023")
    assert receipt.pre_execution_delay_ms == 2_000
    assert receipt.worker_startup_ms is None
    assert receipt.execution_ms == 30_000
    assert receipt.accepted_idle_tail_ms == 10_000
    assert receipt.idle_tail_ms is None
    assert receipt.billable_coverage_until == billing_attempt.completed_at + timedelta(
        seconds=accepted.idle_tail_seconds
    )
    assert receipt.exclusive_billing_hour_starts == (
        datetime(2026, 8, 3, 10, tzinfo=UTC),
    )
    assert billing_calls == [
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T11:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        },
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T11:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        },
    ]
    serialized = json.dumps(receipt.to_dict()).lower()
    assert "must-not-serialize" not in serialized
    assert "prompt" not in serialized


@pytest.mark.asyncio
async def test_ambiguous_window_billing_needs_no_job_id_or_job_client() -> None:
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    billing_calls: list[dict[str, object]] = []
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **kwargs: (
                billing_calls.append(kwargs) or _billing_page()
            )
        ),
        clock=clock,
    )
    accepted = quote()
    allocation = ambiguous_window(accepted)

    receipt = await provider.final_ambiguous_window_billing(allocation, accepted)
    replay = await provider.final_ambiguous_window_billing(allocation, accepted)

    assert receipt is not None and replay is not None
    assert receipt.provider_billing_id == replay.provider_billing_id
    assert receipt.attempt_id == allocation.attempt_id
    assert receipt.actual_cost_usd == Decimal("0.023")
    assert receipt.capped_cost_usd == Decimal("0.023")
    assert receipt.operator_loss_usd == Decimal(0)
    assert receipt.accepted_cost_ceiling_usd == accepted.cost_ceiling_usd
    assert billing_calls == [
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T11:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        },
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T11:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        },
    ]


@pytest.mark.asyncio
async def test_ambiguous_window_billing_waits_for_closed_complete_buckets() -> None:
    clock = MutableClock()
    calls = 0

    def billing(**_: object) -> BillingPage:
        nonlocal calls
        calls += 1
        return _billing_page(records=False)

    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(serverless_billing=billing),
        clock=clock,
    )
    accepted = quote()
    allocation = ambiguous_window(accepted)

    assert await provider.final_ambiguous_window_billing(allocation, accepted) is None
    assert calls == 0
    clock.value += timedelta(hours=1)
    assert await provider.final_ambiguous_window_billing(allocation, accepted) is None
    assert calls == 1


@pytest.mark.asyncio
async def test_ambiguous_window_billing_caps_consumer_cost_and_records_loss() -> None:
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    page = _billing_page(
        extra_record={
            "totalAmount": 0.050,
            "gpuAmount": 0.047,
        }
    )
    page = replace(
        page,
        metadata={
            **page.metadata,
            "totals": {
                "totalAmount": 0.050,
                "gpuAmount": 0.047,
                "cpuAmount": 0.0,
                "diskAmount": 0.001,
                "feeAmount": 0.002,
            },
        },
    )
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(serverless_billing=lambda **_: page),
        clock=clock,
    )
    accepted = quote()

    receipt = await provider.final_ambiguous_window_billing(
        ambiguous_window(accepted), accepted
    )

    assert receipt is not None
    assert receipt.actual_cost_usd == Decimal("0.05")
    assert receipt.capped_cost_usd == accepted.cost_ceiling_usd
    assert receipt.operator_loss_usd == Decimal("0.0145")


@pytest.mark.asyncio
async def test_ambiguous_billing_preserves_ordered_unequal_endpoint_hours() -> None:
    quote_clock = MutableClock()
    quote_clock.value = datetime(2026, 8, 3, 10, 57, 30, tzinfo=UTC)
    accepted = quote(quote_clock)
    allocation = ambiguous_window(
        accepted,
        attempted_at=datetime(2026, 8, 3, 10, 58, tzinfo=UTC),
    )
    clock = MutableClock()
    clock.value = datetime(2026, 8, 3, 12, tzinfo=UTC)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **_: _two_hour_billing_page()
        ),
        clock=clock,
    )

    receipt = await provider.final_ambiguous_window_billing(allocation, accepted)

    assert receipt is not None
    assert tuple(item.utc_hour_start for item in receipt.endpoint_hour_costs) == (
        datetime(2026, 8, 3, 10, tzinfo=UTC),
        datetime(2026, 8, 3, 11, tzinfo=UTC),
    )
    assert tuple(item.actual_cost_usd for item in receipt.endpoint_hour_costs) == (
        Decimal("0.0115"),
        Decimal("0.022"),
    )
    assert (
        receipt.endpoint_hour_costs[0].provider_observation_id
        != receipt.endpoint_hour_costs[1].provider_observation_id
    )
    assert all(
        item.provider_observation_id.startswith("runpod-serverless-hour:")
        for item in receipt.endpoint_hour_costs
    )
    assert receipt.actual_cost_usd == Decimal("0.0335")


@pytest.mark.asyncio
async def test_ambiguous_billing_queries_quote_lifetime_hour_superset() -> None:
    quote_clock = MutableClock()
    quote_clock.value = datetime(2026, 8, 3, 10, 59, 30, tzinfo=UTC)
    accepted = quote(quote_clock)
    reserved_hours = (
        datetime(2026, 8, 3, 10, tzinfo=UTC),
        datetime(2026, 8, 3, 11, tzinfo=UTC),
    )
    allocation = ambiguous_window(
        accepted,
        attempted_at=datetime(2026, 8, 3, 11, 0, 10, tzinfo=UTC),
        exclusive_billing_hour_starts=reserved_hours,
    )
    billing_calls: list[dict[str, object]] = []
    clock = MutableClock()
    clock.value = datetime(2026, 8, 3, 12, tzinfo=UTC)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **kwargs: (
                billing_calls.append(kwargs) or _two_hour_billing_page()
            )
        ),
        clock=clock,
    )

    receipt = await provider.final_ambiguous_window_billing(allocation, accepted)

    assert receipt is not None
    assert receipt.exclusive_billing_hour_starts == reserved_hours
    assert (
        tuple(item.utc_hour_start for item in receipt.endpoint_hour_costs)
        == reserved_hours
    )
    assert billing_calls == [
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T12:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        }
    ]


@pytest.mark.asyncio
async def test_ambiguous_window_rejects_incomplete_allocation_or_ceiling() -> None:
    accepted = quote()
    allocation = ambiguous_window(accepted)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(),
        clock=MutableClock(),
    )

    with pytest.raises(ValueError, match="does not cover every"):
        replace(
            allocation,
            exclusive_billing_hour_starts=(datetime(2026, 8, 3, 9, tzinfo=UTC),),
        )
    with pytest.raises(RunPodManagerError, match="ceiling does not match"):
        await provider.final_ambiguous_window_billing(
            replace(
                allocation,
                accepted_cost_ceiling_usd=accepted.cost_ceiling_usd + Decimal("0.01"),
            ),
            accepted,
        )


@pytest.mark.asyncio
async def test_final_billing_accepts_cold_start_in_pre_execution_delay() -> None:
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    accepted = quote()
    billing_attempt = attempt(
        accepted,
        completed_at=datetime(2026, 8, 3, 10, 4, 30, tzinfo=UTC),
    )
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(serverless_billing=lambda **_: _billing_page()),
        job_client=SimpleNamespace(
            status=lambda *_: _terminal_job(delay_time_ms=200_000)
        ),
        clock=clock,
    )

    receipt = await provider.final_billing(billing_attempt, accepted)

    assert receipt is not None
    assert receipt.pre_execution_delay_ms == 200_000
    assert receipt.worker_startup_ms is None
    assert "queue_delay_ms" not in receipt.to_dict()


@pytest.mark.asyncio
async def test_final_billing_reserves_idle_tail_across_hour_boundary() -> None:
    clock = MutableClock()
    quote_clock = MutableClock()
    quote_clock.value = datetime(2026, 8, 3, 10, 57, 30, tzinfo=UTC)
    accepted = quote(quote_clock)
    billing_attempt = attempt(
        accepted,
        submitted_at=datetime(2026, 8, 3, 10, 58, tzinfo=UTC),
        completed_at=datetime(2026, 8, 3, 10, 59, 55, tzinfo=UTC),
    )
    billing_calls: list[dict[str, object]] = []
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **kwargs: (
                billing_calls.append(kwargs) or _two_hour_billing_page()
            )
        ),
        job_client=SimpleNamespace(status=lambda *_: _terminal_job()),
        clock=clock,
    )

    clock.value = datetime(2026, 8, 3, 11, 59, 59, tzinfo=UTC)
    assert await provider.final_billing(billing_attempt, accepted) is None
    assert billing_calls == []

    clock.value = datetime(2026, 8, 3, 12, tzinfo=UTC)
    receipt = await provider.final_billing(billing_attempt, accepted)

    assert receipt is not None
    assert receipt.billable_coverage_until == datetime(2026, 8, 3, 11, 0, 5, tzinfo=UTC)
    assert receipt.accepted_idle_tail_ms == 10_000
    assert receipt.idle_tail_ms is None
    assert receipt.exclusive_billing_hour_starts == (
        datetime(2026, 8, 3, 10, tzinfo=UTC),
        datetime(2026, 8, 3, 11, tzinfo=UTC),
    )
    assert billing_calls == [
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T12:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        }
    ]


@pytest.mark.asyncio
async def test_final_billing_queries_quote_lifetime_hour_superset() -> None:
    quote_clock = MutableClock()
    quote_clock.value = datetime(2026, 8, 3, 10, 59, 30, tzinfo=UTC)
    accepted = quote(quote_clock)
    reserved_hours = (
        datetime(2026, 8, 3, 10, tzinfo=UTC),
        datetime(2026, 8, 3, 11, tzinfo=UTC),
    )
    billing_attempt = attempt(
        accepted,
        submitted_at=datetime(2026, 8, 3, 11, 0, 10, tzinfo=UTC),
        completed_at=datetime(2026, 8, 3, 11, 4, tzinfo=UTC),
        exclusive_billing_hour_starts=reserved_hours,
    )
    billing_calls: list[dict[str, object]] = []
    clock = MutableClock()
    clock.value = datetime(2026, 8, 3, 12, tzinfo=UTC)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **kwargs: (
                billing_calls.append(kwargs) or _two_hour_billing_page()
            )
        ),
        job_client=SimpleNamespace(status=lambda *_: _terminal_job()),
        clock=clock,
    )

    receipt = await provider.final_billing(billing_attempt, accepted)

    assert receipt is not None
    assert receipt.exclusive_billing_hour_starts == reserved_hours
    assert receipt.billing_window_from == reserved_hours[0]
    assert receipt.billing_window_until == reserved_hours[-1] + timedelta(hours=1)
    assert receipt.actual_cost_usd == Decimal("0.0335")
    assert billing_calls == [
        {
            "start_time": "2026-08-03T10:00:00+00:00",
            "end_time": "2026-08-03T12:00:00+00:00",
            "bucket_size": "hour",
            "endpoint_id": "endpoint-selfie-01",
        }
    ]


@pytest.mark.asyncio
async def test_final_billing_rejects_exclusivity_missing_idle_tail_hour() -> None:
    clock = MutableClock()
    clock.value = datetime(2026, 8, 3, 12, tzinfo=UTC)
    quote_clock = MutableClock()
    quote_clock.value = datetime(2026, 8, 3, 10, 57, 30, tzinfo=UTC)
    accepted = quote(quote_clock)
    incomplete = attempt(
        accepted,
        submitted_at=datetime(2026, 8, 3, 10, 58, tzinfo=UTC),
        completed_at=datetime(2026, 8, 3, 10, 59, 55, tzinfo=UTC),
        exclusive_billing_hour_starts=(datetime(2026, 8, 3, 10, tzinfo=UTC),),
    )
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(),
        job_client=SimpleNamespace(),
        clock=clock,
    )

    with pytest.raises(RunPodManagerError, match="does not cover every"):
        await provider.final_billing(incomplete, accepted)


@pytest.mark.asyncio
async def test_final_billing_waits_for_closed_bucket_delayed_or_partial_records() -> (
    None
):
    clock = MutableClock()
    accepted = quote()
    billing_attempt = attempt(accepted)
    calls = 0

    def billing(**_: object) -> BillingPage:
        nonlocal calls
        calls += 1
        return _billing_page(records=False)

    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(serverless_billing=billing),
        job_client=SimpleNamespace(status=lambda *_: _terminal_job()),
        clock=clock,
    )
    clock.value = billing_attempt.completed_at + timedelta(minutes=1)
    assert await provider.final_billing(billing_attempt, accepted) is None
    assert calls == 0

    clock.value = clock.value.replace(hour=11, minute=0, second=0)
    assert await provider.final_billing(billing_attempt, accepted) is None
    assert calls == 1

    provider.control_client.serverless_billing = lambda **_: _billing_page(
        record_end="2026-08-03T10:30:00+00:00"
    )
    assert await provider.final_billing(billing_attempt, accepted) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job", "message"),
    [
        (_terminal_job(id="job-other"), "mismatched"),
        (_terminal_job(status="IN_PROGRESS"), "not terminal"),
        (_terminal_job(delay_time_ms=-1), "unsafe"),
        (_terminal_job(delay_time_ms=240_001), "pre-execution delay exceeds"),
        (_terminal_job(execution_time_ms=180_001), "execution exceeds"),
    ],
)
async def test_final_billing_rejects_unsafe_job_identity_or_state(
    job: ServerlessJob, message: str
) -> None:
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(serverless_billing=lambda **_: _billing_page()),
        job_client=SimpleNamespace(status=lambda *_: job),
        clock=clock,
    )
    with pytest.raises(RunPodManagerError, match=message):
        await provider.final_billing(attempt(), quote())


@pytest.mark.asyncio
async def test_final_billing_rejects_mismatch_extra_content_and_unsafe_amounts() -> (
    None
):
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    pages = [
        _billing_page(endpoint_id="endpoint-other"),
        _billing_page(extra_record={"prompt": "private"}),
        _billing_page(extra_record={"totalAmount": float("nan")}),
    ]
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(serverless_billing=lambda **_: pages.pop(0)),
        job_client=SimpleNamespace(status=lambda *_: _terminal_job()),
        clock=clock,
    )

    with pytest.raises(RunPodManagerError, match="query echo"):
        await provider.final_billing(attempt(), quote())
    with pytest.raises(RunPodManagerError, match="unsupported fields"):
        await provider.final_billing(attempt(), quote())
    with pytest.raises(RunPodManagerError, match="Invalid Runpod"):
        await provider.final_billing(attempt(), quote())


@pytest.mark.asyncio
async def test_final_billing_rejects_zero_length_billing_record() -> None:
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    provider = RunpodServerlessCapacityProvider(
        control_client=SimpleNamespace(
            serverless_billing=lambda **_: _billing_page(
                record_end="2026-08-03T10:00:00+00:00"
            )
        ),
        job_client=SimpleNamespace(status=lambda *_: _terminal_job()),
        clock=clock,
    )
    with pytest.raises(RunPodManagerError, match="interval is invalid"):
        await provider.final_billing(attempt(), quote())


@pytest.mark.asyncio
async def test_final_billing_uses_only_get_requests_and_never_serializes_job_output() -> (
    None
):
    clock = MutableClock()
    clock.value += timedelta(hours=1)
    control_seen: list[httpx.Request] = []
    job_seen: list[httpx.Request] = []

    def control_handler(http_request: httpx.Request) -> httpx.Response:
        control_seen.append(http_request)
        page = _billing_page()
        return httpx.Response(
            200,
            json={"records": list(page.records), "metadata": dict(page.metadata)},
        )

    def job_handler(http_request: httpx.Request) -> httpx.Response:
        job_seen.append(http_request)
        return httpx.Response(
            200,
            json={
                "id": "job-selfie-0001",
                "status": "COMPLETED",
                "delayTime": 2_000,
                "executionTime": 30_000,
                "output": {"image": "private-generated-content"},
            },
        )

    provider = RunpodServerlessCapacityProvider(
        control_client=RunpodControlPlaneClient(
            api_key="control-secret",
            http_transport=httpx.MockTransport(control_handler),
        ),
        job_client=RunpodServerlessClient(
            api_key="job-secret",
            http_transport=httpx.MockTransport(job_handler),
        ),
        clock=clock,
    )
    receipt = await provider.final_billing(attempt(), quote())

    assert receipt is not None
    assert [item.method for item in control_seen + job_seen] == ["GET", "GET"]
    assert control_seen[0].url.path == "/v2/billing/serverless"
    assert job_seen[0].url.path == ("/v2/endpoint-selfie-01/status/job-selfie-0001")
    encoded = json.dumps(receipt.to_dict()).lower()
    assert "private-generated-content" not in encoded
    assert "image" not in encoded
