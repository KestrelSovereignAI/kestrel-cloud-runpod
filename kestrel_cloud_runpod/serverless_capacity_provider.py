"""Runpod REST v2 adapter for finite Serverless quotes and billing."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from .clients import RunpodControlPlaneClient, RunpodServerlessClient
from .models import (
    Availability,
    BillingPage,
    ComputeProduct,
    EndpointResource,
    GPUOffer,
    RunPodManagerError,
    ServerlessJob,
)
from .placement import select_gpu
from .serverless_capacity_contracts import (
    SERVERLESS_CAPACITY_CONTRACT_VERSION,
    SERVERLESS_CAPACITY_SCHEMA_VERSION,
    ServerlessBillingAttempt,
    ServerlessBillingReceipt,
    ServerlessCapacityQuote,
    ServerlessCapacityQuoteRequest,
    ServerlessEndpointProfile,
    decimal_text,
    iso_datetime,
    json_sha256,
    parse_datetime,
    serverless_worker_cost_usd,
)

_TERMINAL_JOB_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})
_BILLING_RECORD_KEYS = frozenset(
    {
        "startTime",
        "endTime",
        "serverlessId",
        "totalAmount",
        "gpuAmount",
        "cpuAmount",
        "diskAmount",
        "feeAmount",
    }
)
_BILLING_METADATA_KEYS = frozenset(
    {"query", "recordCount", "uniqueServerlessCount", "totals"}
)
_BILLING_QUERY_KEYS = frozenset({"startTime", "endTime", "bucketSize", "serverlessId"})
_BILLING_TOTAL_KEYS = frozenset(
    {"totalAmount", "gpuAmount", "cpuAmount", "diskAmount", "feeAmount"}
)
_AVAILABILITY_RANK = {
    Availability.NONE.value: 0,
    Availability.LOW.value: 1,
    Availability.MEDIUM.value: 2,
    Availability.HIGH.value: 3,
}


class RunpodServerlessCapacityProvider:
    """Read-only v2 catalog/endpoint quote and billing reconciliation surface."""

    def __init__(
        self,
        *,
        control_client: RunpodControlPlaneClient,
        job_client: RunpodServerlessClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.control_client = control_client
        self.job_client = job_client
        self._clock = clock

    async def quote(
        self, request: ServerlessCapacityQuoteRequest
    ) -> ServerlessCapacityQuote:
        """Observe catalog capacity and endpoint configuration using GET only."""

        return await self._observe_quote(request)

    async def validate_quote_for_submission(
        self,
        request: ServerlessCapacityQuoteRequest,
        quote: ServerlessCapacityQuote,
        *,
        accepted_cost_ceiling_usd: Decimal,
    ) -> ServerlessCapacityQuote:
        """Re-read capacity immediately before submission and reject drift."""

        now = self._now()
        quote.assert_fresh(now=now, accepted_cost_ceiling_usd=accepted_cost_ceiling_usd)
        _validate_request_binding(request, quote)
        current = await self._observe_quote(request)
        stable_dimensions = (
            "profile_id",
            "endpoint_profile_sha256",
            "endpoint_id",
            "gpu_id",
            "gpu_pool",
            "vram_gb",
            "data_center_id",
            "cloud",
            "gpu_count",
            "min_cuda_version",
            "benchmark_id",
        )
        if any(
            getattr(current, name) != getattr(quote, name) for name in stable_dimensions
        ):
            raise RunPodManagerError(
                "Serverless endpoint or live placement drifted after quote acceptance"
            )
        if current.hourly_worker_rate_usd > quote.hourly_worker_rate_usd:
            raise RunPodManagerError(
                "Serverless worker rate increased after quote acceptance"
            )
        if current.cost_ceiling_usd > accepted_cost_ceiling_usd:
            raise RunPodManagerError(
                "Current Serverless cost exceeds the accepted ceiling"
            )
        return quote

    async def final_billing(
        self,
        attempt: ServerlessBillingAttempt,
        quote: ServerlessCapacityQuote,
    ) -> ServerlessBillingReceipt | None:
        """Return complete endpoint-window billing or ``None`` while it settles.

        REST v2 currently aggregates Serverless billing by endpoint and hour.
        ``exclusive_window_sha256`` is therefore an external durable proof that
        no other job shares the resolved endpoint window.  This provider binds
        that proof to the exact terminal job but does not own queue/job state.
        """

        attempt.validate_quote(quote)
        if self.job_client is None:
            raise RunPodManagerError(
                "Serverless billing requires a restricted job-status client"
            )
        now = self._now()
        if now < attempt.completed_at:
            raise RunPodManagerError(
                "Serverless billing cannot be reconciled before attempt completion"
            )
        job = await asyncio.to_thread(
            self.job_client.status, attempt.endpoint_id, attempt.job_id
        )
        _validate_terminal_job(job, attempt)
        if (
            job.delay_time_ms is not None
            and job.delay_time_ms > quote.maximum_queue_delay_seconds * 1_000
        ):
            raise RunPodManagerError(
                "Serverless job queue delay exceeds the accepted maximum"
            )
        if (
            job.execution_time_ms is not None
            and job.execution_time_ms > quote.maximum_execution_seconds * 1_000
        ):
            raise RunPodManagerError(
                "Serverless job execution exceeds the accepted maximum"
            )
        window_from, window_until = _hour_window(
            attempt.submitted_at, attempt.completed_at
        )
        if now < window_until:
            return None
        page = await asyncio.to_thread(
            self.control_client.serverless_billing,
            start_time=iso_datetime(window_from),
            end_time=iso_datetime(window_until),
            bucket_size="hour",
            endpoint_id=attempt.endpoint_id,
        )
        amounts = _authoritative_amounts(
            page,
            endpoint_id=attempt.endpoint_id,
            window_from=window_from,
            window_until=window_until,
        )
        if amounts is None:
            return None
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "provider_quote_id": quote.provider_quote_id,
            "endpoint_profile_sha256": quote.endpoint_profile_sha256,
            "endpoint_id": attempt.endpoint_id,
            "job_id": attempt.job_id,
            "attempt_id": attempt.attempt_id,
            "exclusive_window_sha256": attempt.exclusive_window_sha256,
            "attempt_started_at": iso_datetime(attempt.submitted_at),
            "attempt_completed_at": iso_datetime(attempt.completed_at),
            "billing_window_from": iso_datetime(window_from),
            "billing_window_until": iso_datetime(window_until),
            "hourly_worker_rate_usd": decimal_text(quote.hourly_worker_rate_usd),
            "queue_delay_ms": job.delay_time_ms,
            "worker_startup_ms": None,
            "execution_ms": job.execution_time_ms,
            "idle_tail_ms": None,
            **{name: decimal_text(value) for name, value in amounts.items()},
        }
        return ServerlessBillingReceipt(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_billing_id="runpod-serverless-billing:" + json_sha256(identity),
            provider_quote_id=quote.provider_quote_id,
            endpoint_profile_sha256=quote.endpoint_profile_sha256,
            endpoint_id=attempt.endpoint_id,
            job_id=attempt.job_id,
            attempt_id=attempt.attempt_id,
            exclusive_window_sha256=attempt.exclusive_window_sha256,
            attempt_started_at=attempt.submitted_at,
            attempt_completed_at=attempt.completed_at,
            billing_window_from=window_from,
            billing_window_until=window_until,
            hourly_worker_rate_usd=quote.hourly_worker_rate_usd,
            queue_delay_ms=job.delay_time_ms,
            worker_startup_ms=None,
            execution_ms=job.execution_time_ms,
            idle_tail_ms=None,
            gpu_cost_usd=amounts["gpu_cost_usd"],
            cpu_cost_usd=amounts["cpu_cost_usd"],
            disk_cost_usd=amounts["disk_cost_usd"],
            fee_cost_usd=amounts["fee_cost_usd"],
            actual_cost_usd=amounts["actual_cost_usd"],
            reconciled_at=now,
        )

    async def _observe_quote(
        self, request: ServerlessCapacityQuoteRequest
    ) -> ServerlessCapacityQuote:
        profile = request.profile
        requirements = profile.constraints.placement_requirements()
        offers = await asyncio.to_thread(
            self.control_client.list_gpus,
            include_availability=True,
            products=(ComputeProduct.SERVERLESS,),
            count=requirements.gpu_count,
            cloud=requirements.cloud,
            min_cuda_version=requirements.min_cuda_version,
        )
        _validate_catalog_numbers(offers)
        observed_at = self._now()
        placement = select_gpu(offers, requirements, observed_at=observed_at)
        if placement.gpu_pool is None or placement.availability is None:
            raise RunPodManagerError(
                "Serverless catalog selection omitted pool or availability"
            )
        matching_pool = tuple(
            offer for offer in offers if offer.pool == placement.gpu_pool
        )
        if len(matching_pool) != 1 or matching_pool[0].id != placement.gpu_id:
            raise RunPodManagerError(
                "Serverless endpoint pool does not identify one exact catalog GPU"
            )
        data_center_id = _select_exact_data_center(matching_pool[0], profile)
        catalog_observation_sha256 = _catalog_observation_sha256(
            matching_pool[0],
            profile,
            data_center_id=data_center_id,
        )
        endpoint = await asyncio.to_thread(
            self.control_client.get_endpoint, profile.endpoint_id
        )
        _validate_endpoint_binding(
            endpoint,
            profile,
            gpu_pool=placement.gpu_pool,
            data_center_id=data_center_id,
        )
        hourly = Decimal(str(placement.offered_cost_per_hr))
        estimated_billable = (
            request.estimated_worker_start_seconds
            + request.estimated_execution_seconds
            + profile.idle_tail_seconds
        )
        estimated_worker = serverless_worker_cost_usd(
            hourly, requirements.gpu_count, estimated_billable
        )
        maximum_worker = serverless_worker_cost_usd(
            hourly, requirements.gpu_count, request.maximum_billable_seconds
        )
        estimated = estimated_worker + request.estimated_non_worker_cost_usd
        ceiling = maximum_worker + request.maximum_non_worker_cost_usd
        expires_at = observed_at + timedelta(seconds=request.quote_ttl_seconds)
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "workload_kind": request.workload_kind,
            "parameters_sha256": request.parameters_sha256,
            "profile_id": profile.profile_id,
            "endpoint_profile_sha256": profile.profile_sha256,
            "endpoint_id": profile.endpoint_id,
            "catalog_observation_sha256": catalog_observation_sha256,
            "gpu_id": placement.gpu_id,
            "gpu_pool": placement.gpu_pool,
            "vram_gb": placement.memory_gb,
            "data_center_id": data_center_id,
            "cloud": requirements.cloud.value,
            "gpu_count": requirements.gpu_count,
            "min_cuda_version": requirements.min_cuda_version,
            "availability": placement.availability.value,
            "benchmark_id": profile.constraints.benchmark_id,
            "hourly_worker_rate_usd": decimal_text(hourly),
            "estimated_queue_delay_seconds": request.estimated_queue_delay_seconds,
            "estimated_worker_start_seconds": request.estimated_worker_start_seconds,
            "estimated_execution_seconds": request.estimated_execution_seconds,
            "idle_tail_seconds": profile.idle_tail_seconds,
            "maximum_queue_delay_seconds": request.maximum_queue_delay_seconds,
            "maximum_worker_start_seconds": request.maximum_worker_start_seconds,
            "maximum_execution_seconds": request.maximum_execution_seconds,
            "maximum_billable_seconds": request.maximum_billable_seconds,
            "estimated_non_worker_cost_usd": decimal_text(
                request.estimated_non_worker_cost_usd
            ),
            "maximum_non_worker_cost_usd": decimal_text(
                request.maximum_non_worker_cost_usd
            ),
            "catalog_observed_at": iso_datetime(observed_at),
            "expires_at": iso_datetime(expires_at),
        }
        return ServerlessCapacityQuote(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_quote_id="runpod-serverless:" + json_sha256(identity),
            workload_kind=request.workload_kind,
            parameters_sha256=request.parameters_sha256,
            profile_id=profile.profile_id,
            endpoint_profile_sha256=profile.profile_sha256,
            endpoint_id=profile.endpoint_id,
            catalog_observation_sha256=catalog_observation_sha256,
            gpu_id=placement.gpu_id,
            gpu_pool=placement.gpu_pool,
            gpu_name=placement.gpu_name,
            vram_gb=placement.memory_gb,
            data_center_id=data_center_id,
            cloud=requirements.cloud,
            gpu_count=requirements.gpu_count,
            min_cuda_version=requirements.min_cuda_version,
            availability=placement.availability,
            benchmark_id=profile.constraints.benchmark_id,
            hourly_worker_rate_usd=hourly,
            estimated_queue_delay_seconds=request.estimated_queue_delay_seconds,
            estimated_worker_start_seconds=request.estimated_worker_start_seconds,
            estimated_execution_seconds=request.estimated_execution_seconds,
            idle_tail_seconds=profile.idle_tail_seconds,
            maximum_queue_delay_seconds=request.maximum_queue_delay_seconds,
            maximum_worker_start_seconds=request.maximum_worker_start_seconds,
            maximum_execution_seconds=request.maximum_execution_seconds,
            estimated_billable_seconds=estimated_billable,
            maximum_billable_seconds=request.maximum_billable_seconds,
            estimated_worker_cost_usd=estimated_worker,
            maximum_worker_cost_usd=maximum_worker,
            estimated_non_worker_cost_usd=request.estimated_non_worker_cost_usd,
            maximum_non_worker_cost_usd=request.maximum_non_worker_cost_usd,
            estimated_cost_usd=estimated,
            cost_ceiling_usd=ceiling,
            catalog_observed_at=observed_at,
            expires_at=expires_at,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Serverless capacity clock must be timezone-aware")
        return value.astimezone(UTC)


def _validate_request_binding(
    request: ServerlessCapacityQuoteRequest, quote: ServerlessCapacityQuote
) -> None:
    profile = request.profile
    if (
        request.workload_kind != quote.workload_kind
        or request.parameters_sha256 != quote.parameters_sha256
        or profile.profile_id != quote.profile_id
        or profile.profile_sha256 != quote.endpoint_profile_sha256
        or profile.endpoint_id != quote.endpoint_id
        or profile.constraints.gpu_count != quote.gpu_count
        or profile.constraints.cloud != quote.cloud
        or profile.constraints.min_cuda_version != quote.min_cuda_version
        or profile.constraints.allowed_gpu_pools != (quote.gpu_pool,)
        or profile.constraints.allowed_data_center_ids != (quote.data_center_id,)
        or profile.constraints.benchmark_id != quote.benchmark_id
        or request.estimated_queue_delay_seconds != quote.estimated_queue_delay_seconds
        or request.estimated_worker_start_seconds
        != quote.estimated_worker_start_seconds
        or request.estimated_execution_seconds != quote.estimated_execution_seconds
        or profile.idle_tail_seconds != quote.idle_tail_seconds
        or request.maximum_queue_delay_seconds != quote.maximum_queue_delay_seconds
        or request.maximum_worker_start_seconds != quote.maximum_worker_start_seconds
        or request.maximum_execution_seconds != quote.maximum_execution_seconds
        or request.maximum_billable_seconds != quote.maximum_billable_seconds
        or request.estimated_non_worker_cost_usd != quote.estimated_non_worker_cost_usd
        or request.maximum_non_worker_cost_usd != quote.maximum_non_worker_cost_usd
    ):
        raise RunPodManagerError(
            "Serverless quote does not match the workload or configured profile"
        )


def _validate_catalog_numbers(offers: Sequence[GPUOffer]) -> None:
    for offer in offers:
        rates = (offer.secure_price_per_hr, offer.community_price_per_hr)
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise RunPodManagerError("Serverless catalog contained an unsafe GPU rate")
        if (
            offer.memory_gb < 0
            or offer.secure_max_count < 0
            or offer.community_max_count < 0
        ):
            raise RunPodManagerError("Serverless catalog contained an unsafe GPU value")


def _select_exact_data_center(
    offer: GPUOffer, profile: ServerlessEndpointProfile
) -> str:
    expected = profile.constraints.allowed_data_center_ids[0]
    matches = [item for item in offer.data_centers if item.get("id") == expected]
    if len(matches) != 1:
        raise RunPodManagerError(
            "Serverless catalog did not return the exact configured data center"
        )
    availability = matches[0].get("availability")
    if not isinstance(availability, str) or availability not in _AVAILABILITY_RANK:
        raise RunPodManagerError(
            "Serverless catalog data-center availability is invalid"
        )
    if (
        _AVAILABILITY_RANK[availability]
        < _AVAILABILITY_RANK[profile.constraints.minimum_availability.value]
    ):
        raise RunPodManagerError(
            "Serverless catalog data-center availability is below the profile minimum"
        )
    return expected


def _catalog_observation_sha256(
    offer: GPUOffer,
    profile: ServerlessEndpointProfile,
    *,
    data_center_id: str,
) -> str:
    selected_data_center = next(
        item for item in offer.data_centers if item.get("id") == data_center_id
    )
    return json_sha256(
        {
            "product": ComputeProduct.SERVERLESS.value,
            "gpu_id": offer.id,
            "gpu_pool": offer.pool,
            "gpu_name": offer.name,
            "manufacturer": offer.manufacturer,
            "vram_gb": offer.memory_gb,
            "cloud": profile.constraints.cloud.value,
            "gpu_count": profile.constraints.gpu_count,
            "hourly_worker_rate_usd": decimal_text(
                Decimal(str(offer.price_for(profile.constraints.cloud)))
            ),
            "availability": offer.availability.value if offer.availability else None,
            "data_center_id": data_center_id,
            "data_center_availability": selected_data_center.get("availability"),
            "min_cuda_version": offer.availability_min_cuda_version,
        }
    )


def _validate_endpoint_binding(
    endpoint: EndpointResource,
    profile: ServerlessEndpointProfile,
    *,
    gpu_pool: str,
    data_center_id: str,
) -> None:
    if (
        endpoint.id != profile.endpoint_id
        or endpoint.name != profile.endpoint_name
        or endpoint.endpoint_type != "QUEUE"
    ):
        raise RunPodManagerError(
            "Configured Serverless endpoint identity or type does not match profile"
        )
    raw = endpoint.raw
    if not isinstance(raw, Mapping):
        raise RunPodManagerError("Serverless endpoint omitted its configuration")
    if raw.get("type") != "QUEUE":
        raise RunPodManagerError(
            "Configured Serverless endpoint identity or type does not match profile"
        )
    gpu = _mapping(raw.get("gpu"), "endpoint.gpu")
    workers = _mapping(raw.get("workers"), "endpoint.workers")
    scaling = _mapping(raw.get("scaling"), "endpoint.scaling")
    if _string_tuple(gpu.get("pools"), "endpoint.gpu.pools") != (gpu_pool,):
        raise RunPodManagerError(
            "Serverless endpoint does not constrain execution to the quoted pool"
        )
    if (
        _integer(gpu.get("count"), "endpoint.gpu.count")
        != profile.constraints.gpu_count
    ):
        raise RunPodManagerError("Serverless endpoint GPU count does not match profile")
    if (
        _integer(workers.get("min"), "endpoint.workers.min") != profile.workers_min
        or _integer(workers.get("max"), "endpoint.workers.max") != profile.workers_max
        or _integer(workers.get("idleTimeout"), "endpoint.workers.idleTimeout")
        != profile.idle_tail_seconds
    ):
        raise RunPodManagerError(
            "Serverless endpoint worker policy does not match profile"
        )
    scaling_type = scaling.get("type")
    scaling_key = "queueDelay"
    if (
        scaling_type != profile.scaling_type
        or _decimal(scaling.get(scaling_key), f"endpoint.scaling.{scaling_key}")
        != profile.scaling_value
    ):
        raise RunPodManagerError(
            "Serverless endpoint scaling policy does not match profile"
        )
    if _string_tuple(raw.get("dataCenterIds"), "endpoint.dataCenterIds") != (
        data_center_id,
    ):
        raise RunPodManagerError(
            "Serverless endpoint does not constrain execution to the quoted data center"
        )
    if _string_tuple(raw.get("networkVolumes"), "endpoint.networkVolumes") != (
        profile.network_volume_ids
    ):
        raise RunPodManagerError(
            "Serverless endpoint network volumes do not match profile"
        )
    expected_scalars = (
        ("image", profile.worker_reference),
        ("timeout", profile.execution_timeout_ms),
        ("flashboot", profile.flashboot.value),
        ("disk", profile.disk_gb),
    )
    if any(raw.get(name) != expected for name, expected in expected_scalars):
        raise RunPodManagerError(
            "Serverless endpoint runtime or billing profile does not match configuration"
        )


def _validate_terminal_job(
    job: ServerlessJob, attempt: ServerlessBillingAttempt
) -> None:
    if not isinstance(job.id, str) or job.id != attempt.job_id:
        raise RunPodManagerError("Runpod returned a mismatched Serverless job")
    if (
        not isinstance(job.status, str)
        or job.status.upper() not in _TERMINAL_JOB_STATES
    ):
        raise RunPodManagerError("Serverless job is not terminal for billing")
    for name, value in (
        ("delayTime", job.delay_time_ms),
        ("executionTime", job.execution_time_ms),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise RunPodManagerError(f"Serverless job {name} is unsafe")


def _hour_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    window_from = start_utc.replace(minute=0, second=0, microsecond=0)
    window_until = end_utc.replace(minute=0, second=0, microsecond=0)
    if end_utc != window_until:
        window_until += timedelta(hours=1)
    return window_from, window_until


def _authoritative_amounts(
    page: BillingPage,
    *,
    endpoint_id: str,
    window_from: datetime,
    window_until: datetime,
) -> dict[str, Decimal] | None:
    metadata = page.metadata
    _exact_keys(metadata, _BILLING_METADATA_KEYS, "billing metadata")
    query = _mapping(metadata.get("query"), "billing metadata.query")
    totals = _mapping(metadata.get("totals"), "billing metadata.totals")
    _exact_keys(query, _BILLING_QUERY_KEYS, "billing query")
    _exact_keys(totals, _BILLING_TOTAL_KEYS, "billing totals")
    if (
        query.get("serverlessId") != endpoint_id
        or query.get("bucketSize") != "hour"
        or parse_datetime(_string(query.get("startTime"), "billing query start"))
        != window_from
        or parse_datetime(_string(query.get("endTime"), "billing query end"))
        != window_until
    ):
        raise RunPodManagerError("Runpod billing query echo does not match attempt")
    record_count = _integer(metadata.get("recordCount"), "billing recordCount")
    unique_count = _integer(
        metadata.get("uniqueServerlessCount"), "billing uniqueServerlessCount"
    )
    if record_count != len(page.records):
        raise RunPodManagerError("Runpod billing record count is inconsistent")
    if not page.records:
        if unique_count not in {0, 1}:
            raise RunPodManagerError("Runpod empty billing endpoint count is invalid")
        return None
    if unique_count != 1:
        raise RunPodManagerError("Runpod billing spans multiple Serverless endpoints")
    parsed: list[tuple[datetime, datetime, dict[str, Decimal]]] = []
    for record in page.records:
        _exact_keys(record, _BILLING_RECORD_KEYS, "billing record")
        if record.get("serverlessId") != endpoint_id:
            raise RunPodManagerError("Runpod billing record endpoint is mismatched")
        record_from = parse_datetime(
            _string(record.get("startTime"), "billing record start")
        )
        record_until = parse_datetime(
            _string(record.get("endTime"), "billing record end")
        )
        amounts = _amounts_from_mapping(record, "billing record")
        parsed.append((record_from, record_until, amounts))
    parsed.sort(key=lambda item: item[0])
    cursor = window_from
    summed = {name: Decimal(0) for name in _AMOUNT_NAMES}
    for record_from, record_until, amounts in parsed:
        if record_until <= record_from:
            raise RunPodManagerError("Runpod billing record interval is invalid")
        if record_from > cursor and record_from < window_until:
            return None
        if record_from != cursor:
            raise RunPodManagerError(
                "Runpod billing records overlap or escape the window"
            )
        if record_until > window_until:
            raise RunPodManagerError(
                "Runpod billing record exceeds the requested window"
            )
        for name in _AMOUNT_NAMES:
            summed[name] += amounts[name]
        cursor = record_until
    if cursor < window_until:
        return None
    expected_totals = _amounts_from_mapping(totals, "billing totals")
    if summed != expected_totals:
        raise RunPodManagerError("Runpod billing totals do not equal its records")
    return summed


_AMOUNT_NAMES = (
    "actual_cost_usd",
    "gpu_cost_usd",
    "cpu_cost_usd",
    "disk_cost_usd",
    "fee_cost_usd",
)
_AMOUNT_KEYS = {
    "actual_cost_usd": "totalAmount",
    "gpu_cost_usd": "gpuAmount",
    "cpu_cost_usd": "cpuAmount",
    "disk_cost_usd": "diskAmount",
    "fee_cost_usd": "feeAmount",
}


def _amounts_from_mapping(value: Mapping[str, Any], context: str) -> dict[str, Decimal]:
    amounts = {
        name: _decimal(value.get(key), f"{context}.{key}")
        for name, key in _AMOUNT_KEYS.items()
    }
    if amounts["actual_cost_usd"] != (
        amounts["gpu_cost_usd"]
        + amounts["cpu_cost_usd"]
        + amounts["disk_cost_usd"]
        + amounts["fee_cost_usd"]
    ):
        raise RunPodManagerError(f"{context} cost components do not equal total")
    return amounts


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return value


def _integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return value


def _decimal(value: Any, context: str) -> Decimal:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RunPodManagerError(f"Invalid Runpod {context}")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise RunPodManagerError(f"Invalid Runpod {context}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return parsed


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return tuple(value)


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    if set(value) != expected:
        raise RunPodManagerError(
            f"Runpod {context} contains missing or unsupported fields"
        )


__all__ = ["RunpodServerlessCapacityProvider"]
