"""Shared fixtures for finite Serverless capacity tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    EndpointResource,
    FlashBoot,
    GPUOffer,
)
from kestrel_cloud_runpod.serverless_capacity_contracts import (
    SERVERLESS_CAPACITY_CONTRACT_VERSION,
    SERVERLESS_CAPACITY_SCHEMA_VERSION,
    PlannedServerlessCapacityQuote,
    PlannedServerlessCapacityQuoteRequest,
    PlannedServerlessEndpoint,
    ServerlessActivatedSubmission,
    ServerlessAmbiguousBillingWindow,
    ServerlessBillingAttempt,
    ServerlessCapacityConstraints,
    ServerlessCapacityQuote,
    ServerlessCapacityQuoteRequest,
    ServerlessEndpointActivationReceipt,
    ServerlessEndpointProfile,
    ServerlessEndpointSpec,
    serverless_billing_hour_starts,
    serverless_worker_cost_usd,
)

PARAMETERS_SHA256 = "a" * 64
PROFILE_SHA256 = "b" * 64
WORKER_REFERENCE = "ghcr.io/kestrelsovereignai/catalog-worker@sha256:" + "c" * 64


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 3, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def constraints(**changes: object) -> ServerlessCapacityConstraints:
    values: dict[str, object] = {
        "min_vram_gb": 24,
        "gpu_count": 1,
        "cloud": CloudType.SECURE,
        "min_cuda_version": "12.8",
        "allowed_gpu_pools": ("BLACKWELL_24",),
        "allowed_data_center_ids": ("US-TX-3",),
        "max_hourly_worker_rate_usd": Decimal("0.75"),
        "minimum_availability": Availability.HIGH,
        "benchmark_id": "catalog-selfie-v1",
    }
    values.update(changes)
    return ServerlessCapacityConstraints(**values)  # type: ignore[arg-type]


def endpoint_spec(**changes: object) -> ServerlessEndpointSpec:
    values: dict[str, object] = {
        "worker_reference": WORKER_REFERENCE,
        "constraints": constraints(),
        "workers_min": 0,
        "workers_max": 1,
        "idle_tail_seconds": 10,
        "scaling_type": "QUEUE_DELAY",
        "scaling_value": Decimal(4),
        "execution_timeout_ms": 120_000,
        "flashboot": FlashBoot.FLASHBOOT,
        "disk_gb": 20,
        "registry_id": None,
        "network_volume_ids": (),
    }
    values.update(changes)
    return ServerlessEndpointSpec(**values)  # type: ignore[arg-type]


def profile(**changes: object) -> ServerlessEndpointProfile:
    values: dict[str, object] = {
        "profile_id": changes.pop("profile_id", "selfie-blackwell-01"),
        "endpoint_id": changes.pop("endpoint_id", "endpoint-selfie-01"),
        "endpoint_name": changes.pop("endpoint_name", "selfie-blackwell-01"),
    }
    supplied_spec = changes.pop("spec", None)
    values["spec"] = supplied_spec or endpoint_spec(**changes)
    return ServerlessEndpointProfile(**values)  # type: ignore[arg-type]


def planned_endpoint(**changes: object) -> PlannedServerlessEndpoint:
    values: dict[str, object] = {
        "plan_id": "selfie-run-0001",
        "endpoint_name": "kestrel-selfie-run-0001",
        "spec": endpoint_spec(),
    }
    values.update(changes)
    return PlannedServerlessEndpoint(**values)  # type: ignore[arg-type]


def request(**changes: object) -> ServerlessCapacityQuoteRequest:
    values: dict[str, object] = {
        "profile": profile(),
        "workload_kind": "catalog-selfie",
        "parameters_sha256": PARAMETERS_SHA256,
        "estimated_queue_delay_seconds": 15,
        "estimated_worker_start_seconds": 60,
        "estimated_execution_seconds": 30,
        "maximum_queue_delay_seconds": 120,
        "maximum_worker_start_seconds": 120,
        "maximum_execution_seconds": 50,
        "maximum_billable_seconds": 180,
        "job_execution_timeout_ms": 120_000,
        "job_ttl_ms": 300_000,
        "estimated_non_worker_cost_usd": Decimal("0.000500"),
        "maximum_non_worker_cost_usd": Decimal("0.001000"),
        "quote_ttl_seconds": 60,
    }
    values.update(changes)
    return ServerlessCapacityQuoteRequest(**values)  # type: ignore[arg-type]


def planned_request(**changes: object) -> PlannedServerlessCapacityQuoteRequest:
    values: dict[str, object] = {
        "endpoint": planned_endpoint(),
        "workload_kind": "catalog-selfie",
        "parameters_sha256": PARAMETERS_SHA256,
        "estimated_queue_delay_seconds": 15,
        "estimated_worker_start_seconds": 60,
        "estimated_execution_seconds": 30,
        "maximum_queue_delay_seconds": 120,
        "maximum_worker_start_seconds": 120,
        "maximum_execution_seconds": 50,
        "maximum_billable_seconds": 180,
        "job_execution_timeout_ms": 120_000,
        "job_ttl_ms": 300_000,
        "estimated_non_worker_cost_usd": Decimal("0.000500"),
        "maximum_non_worker_cost_usd": Decimal("0.001000"),
        "quote_ttl_seconds": 60,
    }
    values.update(changes)
    return PlannedServerlessCapacityQuoteRequest(**values)  # type: ignore[arg-type]


def offer(**changes: object) -> GPUOffer:
    values: dict[str, object] = {
        "id": "PRO-6000-MIG-1g-24gb",
        "name": "PRO 6000 MIG 1g.24gb",
        "pool": "BLACKWELL_24",
        "manufacturer": "NVIDIA",
        "memory_gb": 24,
        "secure": True,
        "community": False,
        "secure_price_per_hr": 0.69,
        "community_price_per_hr": 0.0,
        "secure_max_count": 0,
        "community_max_count": 0,
        "availability": Availability.HIGH,
        "data_centers": ({"id": "US-TX-3", "availability": "HIGH"},),
        "availability_min_cuda_version": "12.8",
    }
    values.update(changes)
    return GPUOffer(**values)  # type: ignore[arg-type]


def endpoint(
    endpoint_profile: ServerlessEndpointProfile | None = None,
) -> EndpointResource:
    selected = endpoint_profile or profile()
    pool = selected.constraints.allowed_gpu_pools[0]
    data_center = selected.constraints.allowed_data_center_ids[0]
    return EndpointResource.from_dict(
        {
            "id": selected.endpoint_id,
            "name": selected.endpoint_name,
            "type": "QUEUE",
            "requestUrls": {
                "run": f"https://api.runpod.ai/v2/{selected.endpoint_id}/run"
            },
            "image": selected.worker_reference,
            "gpu": {"pools": [pool], "count": selected.constraints.gpu_count},
            "workers": {
                "min": selected.workers_min,
                "max": selected.workers_max,
                "idleTimeout": selected.idle_tail_seconds,
            },
            "scaling": {
                "type": selected.scaling_type,
                "queueDelay": float(selected.scaling_value),
            },
            "dataCenterIds": [data_center],
            "networkVolumes": list(selected.network_volume_ids),
            "timeout": selected.execution_timeout_ms,
            "flashboot": selected.flashboot.value,
            "disk": selected.disk_gb,
            "ports": [],
            "env": {},
            "args": None,
            "registry": selected.registry_id,
        }
    )


def planned_endpoint_resource(
    planned: PlannedServerlessEndpoint | None = None,
    *,
    endpoint_id: str = "endpoint-planned-01",
    **raw_changes: object,
) -> EndpointResource:
    selected = planned or planned_endpoint()
    spec = selected.spec
    raw: dict[str, object] = {
        "id": endpoint_id,
        "name": selected.endpoint_name,
        "type": "QUEUE",
        "requestUrls": {"run": f"https://api.runpod.ai/v2/{endpoint_id}/run"},
        "image": spec.worker_reference,
        "gpu": {
            "pools": [spec.constraints.allowed_gpu_pools[0]],
            "count": spec.constraints.gpu_count,
        },
        "workers": {
            "min": spec.workers_min,
            "max": spec.workers_max,
            "idleTimeout": spec.idle_tail_seconds,
        },
        "scaling": {
            "type": spec.scaling_type,
            "queueDelay": float(spec.scaling_value),
        },
        "dataCenterIds": [spec.constraints.allowed_data_center_ids[0]],
        "networkVolumes": [],
        "timeout": spec.execution_timeout_ms,
        "flashboot": spec.flashboot.value,
        "disk": spec.disk_gb,
        "ports": [],
        "env": {},
        "args": None,
        "registry": spec.registry_id,
    }
    raw.update(raw_changes)
    return EndpointResource.from_dict(raw)


def quote(clock: MutableClock | None = None) -> ServerlessCapacityQuote:
    selected_clock = clock or MutableClock()
    selected_profile = profile()
    hourly = Decimal("0.69")
    estimated_worker = serverless_worker_cost_usd(hourly, 1, 100)
    maximum_worker = serverless_worker_cost_usd(hourly, 1, 180)
    return ServerlessCapacityQuote(
        schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
        contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
        provider_quote_id="runpod-serverless:" + "d" * 64,
        workload_kind="catalog-selfie",
        parameters_sha256=PARAMETERS_SHA256,
        profile_id=selected_profile.profile_id,
        endpoint_profile_sha256=selected_profile.profile_sha256,
        endpoint_id=selected_profile.endpoint_id,
        catalog_observation_sha256="9" * 64,
        gpu_id="PRO-6000-MIG-1g-24gb",
        gpu_pool="BLACKWELL_24",
        gpu_name="PRO 6000 MIG 1g.24gb",
        vram_gb=24,
        data_center_id="US-TX-3",
        cloud=CloudType.SECURE,
        gpu_count=1,
        min_cuda_version="12.8",
        availability=Availability.HIGH,
        benchmark_id="catalog-selfie-v1",
        hourly_worker_rate_usd=hourly,
        estimated_queue_delay_seconds=15,
        estimated_worker_start_seconds=60,
        estimated_execution_seconds=30,
        idle_tail_seconds=10,
        maximum_queue_delay_seconds=120,
        maximum_worker_start_seconds=120,
        maximum_execution_seconds=50,
        job_execution_timeout_ms=120_000,
        job_ttl_ms=300_000,
        estimated_billable_seconds=100,
        maximum_billable_seconds=180,
        estimated_worker_cost_usd=estimated_worker,
        maximum_worker_cost_usd=maximum_worker,
        estimated_non_worker_cost_usd=Decimal("0.000500"),
        maximum_non_worker_cost_usd=Decimal("0.001000"),
        estimated_cost_usd=estimated_worker + Decimal("0.000500"),
        cost_ceiling_usd=maximum_worker + Decimal("0.001000"),
        catalog_observed_at=selected_clock(),
        expires_at=selected_clock() + timedelta(seconds=60),
    )


def attempt(
    item: ServerlessCapacityQuote | None = None, **changes: object
) -> ServerlessBillingAttempt:
    selected = item or quote()
    values: dict[str, object] = {
        "attempt_id": "attempt-selfie-0001",
        "job_id": "job-selfie-0001",
        "endpoint_id": selected.endpoint_id,
        "provider_quote_id": selected.provider_quote_id,
        "exclusive_window_sha256": "e" * 64,
        "submitted_at": datetime(2026, 8, 3, 10, 0, 30, tzinfo=UTC),
        "completed_at": datetime(2026, 8, 3, 10, 4, tzinfo=UTC),
    }
    values.update(changes)
    if "exclusive_billing_hour_starts" not in changes:
        completed_at = values["completed_at"]
        submitted_at = values["submitted_at"]
        assert isinstance(completed_at, datetime)
        assert isinstance(submitted_at, datetime)
        values["exclusive_billing_hour_starts"] = serverless_billing_hour_starts(
            submitted_at,
            completed_at + timedelta(seconds=selected.idle_tail_seconds),
        )
    return ServerlessBillingAttempt(**values)  # type: ignore[arg-type]


def ambiguous_window(
    item: ServerlessCapacityQuote | None = None, **changes: object
) -> ServerlessAmbiguousBillingWindow:
    selected = item or quote()
    attempted_at = datetime(2026, 8, 3, 10, 0, 30, tzinfo=UTC)
    values: dict[str, object] = {
        "attempt_id": "attempt-selfie-ambiguous-0001",
        "endpoint_id": selected.endpoint_id,
        "provider_quote_id": selected.provider_quote_id,
        "exclusive_window_sha256": "7" * 64,
        "attempted_at": attempted_at,
        "accepted_cost_ceiling_usd": selected.cost_ceiling_usd,
    }
    values.update(changes)
    if "billable_coverage_until" not in changes:
        selected_attempted_at = values["attempted_at"]
        assert isinstance(selected_attempted_at, datetime)
        values["billable_coverage_until"] = selected_attempted_at + timedelta(
            seconds=(
                selected.maximum_queue_delay_seconds
                + selected.maximum_worker_start_seconds
                + selected.maximum_execution_seconds
                + selected.idle_tail_seconds
            )
        )
    if "exclusive_billing_hour_starts" not in changes:
        start = values["attempted_at"]
        coverage_until = values["billable_coverage_until"]
        assert isinstance(start, datetime)
        assert isinstance(coverage_until, datetime)
        values["exclusive_billing_hour_starts"] = serverless_billing_hour_starts(
            start, coverage_until
        )
    return ServerlessAmbiguousBillingWindow(**values)  # type: ignore[arg-type]


def activated_ambiguous_window(
    item: PlannedServerlessCapacityQuote,
    activation: ServerlessEndpointActivationReceipt,
    submission: ServerlessActivatedSubmission,
    **changes: object,
) -> ServerlessAmbiguousBillingWindow:
    attempted_at = submission.authorized_at + timedelta(seconds=30)
    values: dict[str, object] = {
        "attempt_id": "attempt-selfie-activated-ambiguous-0001",
        "endpoint_id": activation.endpoint_id,
        "provider_quote_id": item.provider_quote_id,
        "exclusive_window_sha256": submission.exclusive_window_sha256,
        "attempted_at": attempted_at,
        "accepted_cost_ceiling_usd": item.cost_ceiling_usd,
    }
    values.update(changes)
    if "billable_coverage_until" not in changes:
        selected_attempted_at = values["attempted_at"]
        assert isinstance(selected_attempted_at, datetime)
        values["billable_coverage_until"] = selected_attempted_at + timedelta(
            seconds=(
                item.maximum_queue_delay_seconds
                + item.maximum_worker_start_seconds
                + item.maximum_execution_seconds
                + item.idle_tail_seconds
            )
        )
    if "exclusive_billing_hour_starts" not in changes:
        start = values["attempted_at"]
        coverage_until = values["billable_coverage_until"]
        assert isinstance(start, datetime)
        assert isinstance(coverage_until, datetime)
        values["exclusive_billing_hour_starts"] = serverless_billing_hour_starts(
            start, coverage_until
        )
    return ServerlessAmbiguousBillingWindow(**values)  # type: ignore[arg-type]
