"""Provider-neutral contracts for finite Serverless capacity and billing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

from .models import (
    Availability,
    CloudType,
    ComputeProduct,
    FlashBoot,
    PlacementRequirements,
    RunPodManagerError,
)

SERVERLESS_CAPACITY_SCHEMA_VERSION = 1
SERVERLESS_CAPACITY_CONTRACT_VERSION = "serverless-capacity-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}")
_IMMUTABLE_REFERENCE_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class ServerlessCapacityConstraints:
    """Hard placement constraints for one reusable queue endpoint."""

    min_vram_gb: int
    gpu_count: int = 1
    cloud: CloudType = CloudType.SECURE
    min_cuda_version: str | None = None
    allowed_gpu_pools: tuple[str, ...] = ()
    allowed_data_center_ids: tuple[str, ...] = ()
    max_hourly_worker_rate_usd: Decimal | None = None
    minimum_availability: Availability = Availability.LOW
    benchmark_id: str = "catalog-selfie"

    def __post_init__(self) -> None:
        _positive_int(self.min_vram_gb, "min_vram_gb")
        _positive_int(self.gpu_count, "gpu_count")
        if not isinstance(self.cloud, CloudType):
            raise TypeError("Serverless capacity cloud must be a CloudType")
        if not isinstance(self.minimum_availability, Availability):
            raise TypeError(
                "Serverless capacity minimum_availability must be an Availability"
            )
        if self.min_cuda_version is not None and not re.fullmatch(
            r"\d+(?:\.\d+)?", self.min_cuda_version
        ):
            raise ValueError("Serverless capacity CUDA version is invalid")
        _one_identifier(
            self.allowed_gpu_pools,
            "allowed_gpu_pools",
            reason="an exact endpoint pool is required",
        )
        _one_identifier(
            self.allowed_data_center_ids,
            "allowed_data_center_ids",
            reason="an exact endpoint data center is required",
        )
        _safe_identifier(self.benchmark_id, "benchmark_id")
        if self.max_hourly_worker_rate_usd is not None:
            _positive_decimal(
                self.max_hourly_worker_rate_usd, "max_hourly_worker_rate_usd"
            )

    def placement_requirements(self) -> PlacementRequirements:
        maximum_rate = (
            float(self.max_hourly_worker_rate_usd)
            if self.max_hourly_worker_rate_usd is not None
            else None
        )
        if maximum_rate is not None and not math.isfinite(maximum_rate):
            raise ValueError("Serverless capacity hourly-rate limit is too large")
        return PlacementRequirements(
            product=ComputeProduct.SERVERLESS,
            min_vram_gb=self.min_vram_gb,
            gpu_count=self.gpu_count,
            cloud=self.cloud,
            min_cuda_version=self.min_cuda_version,
            max_cost_per_hr=maximum_rate,
            allowed_gpu_pools=self.allowed_gpu_pools,
            allowed_data_center_ids=self.allowed_data_center_ids,
            minimum_availability=self.minimum_availability,
            benchmark_id=self.benchmark_id,
        )

    def fingerprint_dict(self) -> dict[str, Any]:
        return {
            "min_vram_gb": self.min_vram_gb,
            "gpu_count": self.gpu_count,
            "cloud": self.cloud.value,
            "min_cuda_version": self.min_cuda_version,
            "allowed_gpu_pools": list(self.allowed_gpu_pools),
            "allowed_data_center_ids": list(self.allowed_data_center_ids),
            "max_hourly_worker_rate_usd": decimal_text(self.max_hourly_worker_rate_usd),
            "minimum_availability": self.minimum_availability.value,
            "benchmark_id": self.benchmark_id,
        }


@dataclass(frozen=True)
class ServerlessEndpointProfile:
    """Expected immutable and billable settings of a reusable queue endpoint."""

    profile_id: str
    endpoint_id: str
    endpoint_name: str
    worker_reference: str
    constraints: ServerlessCapacityConstraints
    workers_min: int
    workers_max: int
    idle_tail_seconds: int
    scaling_type: str
    scaling_value: Decimal
    execution_timeout_ms: int
    flashboot: FlashBoot
    disk_gb: int
    network_volume_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("profile_id", self.profile_id),
            ("endpoint_id", self.endpoint_id),
            ("endpoint_name", self.endpoint_name),
        ):
            _safe_identifier(value, name)
        if not _IMMUTABLE_REFERENCE_RE.fullmatch(self.worker_reference):
            raise ValueError(
                "Serverless worker reference must be pinned by a SHA-256 digest"
            )
        if "://" in self.worker_reference:
            raise ValueError("Serverless worker reference cannot be a URL")
        if not isinstance(self.constraints, ServerlessCapacityConstraints):
            raise TypeError("Serverless endpoint constraints have an invalid type")
        _nonnegative_int(self.workers_min, "workers_min")
        _positive_int(self.workers_max, "workers_max")
        if self.workers_min != 0 or self.workers_max != 1:
            raise ValueError(
                "Authoritative per-attempt billing requires a scale-to-zero, "
                "single-worker endpoint"
            )
        _positive_int(self.idle_tail_seconds, "idle_tail_seconds")
        if self.idle_tail_seconds > 3_600:
            raise ValueError("Serverless endpoint idle_tail_seconds exceeds v2 maximum")
        if self.scaling_type != "QUEUE_DELAY":
            raise ValueError(
                "Finite Serverless billing requires QUEUE_DELAY scaling with an "
                "observable idle tail"
            )
        _positive_decimal(self.scaling_value, "scaling_value")
        if self.scaling_value < Decimal("0.5"):
            raise ValueError("Serverless QUEUE_DELAY target must be at least 0.5")
        _positive_int(self.execution_timeout_ms, "execution_timeout_ms")
        if not isinstance(self.flashboot, FlashBoot):
            raise TypeError("Serverless endpoint flashboot must be a FlashBoot")
        _positive_int(self.disk_gb, "disk_gb")
        if self.network_volume_ids:
            raise ValueError(
                "Finite Serverless billing cannot attribute persistent network "
                "volume cost to one attempt"
            )

    @property
    def profile_sha256(self) -> str:
        return json_sha256(
            {
                "profile_id": self.profile_id,
                "endpoint_id": self.endpoint_id,
                "endpoint_name": self.endpoint_name,
                "worker_reference": self.worker_reference,
                "constraints": self.constraints.fingerprint_dict(),
                "workers": {"min": self.workers_min, "max": self.workers_max},
                "idle_tail_seconds": self.idle_tail_seconds,
                "scaling_type": self.scaling_type,
                "scaling_value": decimal_text(self.scaling_value),
                "execution_timeout_ms": self.execution_timeout_ms,
                "flashboot": self.flashboot.value,
                "disk_gb": self.disk_gb,
                "network_volume_ids": list(self.network_volume_ids),
            }
        )


@dataclass(frozen=True)
class ServerlessCapacityQuoteRequest:
    """Content-free inputs to one read-only Serverless capacity observation."""

    profile: ServerlessEndpointProfile
    workload_kind: str
    parameters_sha256: str
    estimated_queue_delay_seconds: int | None
    estimated_worker_start_seconds: int
    estimated_execution_seconds: int
    maximum_queue_delay_seconds: int
    maximum_worker_start_seconds: int
    maximum_execution_seconds: int
    maximum_billable_seconds: int
    estimated_non_worker_cost_usd: Decimal
    maximum_non_worker_cost_usd: Decimal
    quote_ttl_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ServerlessEndpointProfile):
            raise TypeError("Serverless quote profile has an invalid type")
        _safe_identifier(self.workload_kind, "workload_kind")
        _sha256(self.parameters_sha256, "parameters_sha256")
        if self.estimated_queue_delay_seconds is not None:
            _nonnegative_int(
                self.estimated_queue_delay_seconds,
                "estimated_queue_delay_seconds",
            )
        _nonnegative_int(
            self.estimated_worker_start_seconds,
            "estimated_worker_start_seconds",
        )
        _positive_int(self.estimated_execution_seconds, "estimated_execution_seconds")
        _positive_int(self.maximum_queue_delay_seconds, "maximum_queue_delay_seconds")
        _positive_int(self.maximum_worker_start_seconds, "maximum_worker_start_seconds")
        _positive_int(self.maximum_execution_seconds, "maximum_execution_seconds")
        _positive_int(self.maximum_billable_seconds, "maximum_billable_seconds")
        _nonnegative_decimal(
            self.estimated_non_worker_cost_usd,
            "estimated_non_worker_cost_usd",
        )
        _positive_decimal(
            self.maximum_non_worker_cost_usd,
            "maximum_non_worker_cost_usd",
        )
        _positive_int(self.quote_ttl_seconds, "quote_ttl_seconds")
        if self.quote_ttl_seconds > 300:
            raise ValueError("Serverless quote TTL cannot exceed five minutes")
        if (
            self.estimated_queue_delay_seconds is not None
            and self.estimated_queue_delay_seconds > self.maximum_queue_delay_seconds
        ):
            raise ValueError("Serverless estimated queue delay exceeds its maximum")
        if self.estimated_worker_start_seconds > self.maximum_worker_start_seconds:
            raise ValueError("Serverless estimated worker start exceeds its maximum")
        if self.estimated_execution_seconds > self.maximum_execution_seconds:
            raise ValueError("Serverless estimated execution exceeds its maximum")
        estimated_billable = (
            self.estimated_worker_start_seconds
            + self.estimated_execution_seconds
            + self.profile.idle_tail_seconds
        )
        maximum_billable = (
            self.maximum_worker_start_seconds
            + self.maximum_execution_seconds
            + self.profile.idle_tail_seconds
        )
        if self.maximum_billable_seconds != maximum_billable:
            raise ValueError("Serverless maximum billable time is inconsistent")
        if estimated_billable > maximum_billable:
            raise ValueError("Serverless estimated billable time exceeds its maximum")
        if self.maximum_execution_seconds * 1_000 > self.profile.execution_timeout_ms:
            raise ValueError(
                "Serverless maximum execution exceeds the endpoint timeout"
            )
        if self.estimated_non_worker_cost_usd > self.maximum_non_worker_cost_usd:
            raise ValueError("Serverless estimated non-worker cost exceeds its maximum")


@dataclass(frozen=True)
class ServerlessCapacityQuote:
    """Immutable content-free capacity, endpoint, time, and cost evidence."""

    schema_version: int
    contract_version: str
    provider_quote_id: str
    workload_kind: str
    parameters_sha256: str
    profile_id: str
    endpoint_profile_sha256: str
    endpoint_id: str
    catalog_observation_sha256: str
    gpu_id: str
    gpu_pool: str
    gpu_name: str
    vram_gb: int
    data_center_id: str
    cloud: CloudType
    gpu_count: int
    min_cuda_version: str | None
    availability: Availability
    benchmark_id: str
    hourly_worker_rate_usd: Decimal
    estimated_queue_delay_seconds: int | None
    estimated_worker_start_seconds: int
    estimated_execution_seconds: int
    idle_tail_seconds: int
    maximum_queue_delay_seconds: int
    maximum_worker_start_seconds: int
    maximum_execution_seconds: int
    estimated_billable_seconds: int
    maximum_billable_seconds: int
    estimated_worker_cost_usd: Decimal
    maximum_worker_cost_usd: Decimal
    estimated_non_worker_cost_usd: Decimal
    maximum_non_worker_cost_usd: Decimal
    estimated_cost_usd: Decimal
    cost_ceiling_usd: Decimal
    catalog_observed_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != SERVERLESS_CAPACITY_SCHEMA_VERSION:
            raise ValueError("Serverless capacity quote schema is unsupported")
        if self.contract_version != SERVERLESS_CAPACITY_CONTRACT_VERSION:
            raise ValueError("Serverless capacity quote contract is unsupported")
        for name, value in (
            ("provider_quote_id", self.provider_quote_id),
            ("workload_kind", self.workload_kind),
            ("profile_id", self.profile_id),
            ("endpoint_id", self.endpoint_id),
            ("gpu_id", self.gpu_id),
            ("gpu_pool", self.gpu_pool),
            ("data_center_id", self.data_center_id),
            ("benchmark_id", self.benchmark_id),
        ):
            _safe_identifier(value, name)
        _safe_display_name(self.gpu_name, "gpu_name")
        _sha256(self.parameters_sha256, "parameters_sha256")
        _sha256(self.endpoint_profile_sha256, "endpoint_profile_sha256")
        _sha256(self.catalog_observation_sha256, "catalog_observation_sha256")
        if not isinstance(self.cloud, CloudType):
            raise TypeError("Serverless capacity quote cloud is invalid")
        if not isinstance(self.availability, Availability):
            raise TypeError("Serverless capacity quote availability is invalid")
        if self.availability is Availability.NONE:
            raise ValueError(
                "Serverless capacity quote cannot bind unavailable capacity"
            )
        _positive_int(self.gpu_count, "gpu_count")
        _positive_int(self.vram_gb, "vram_gb")
        if self.min_cuda_version is not None and not re.fullmatch(
            r"\d+(?:\.\d+)?", self.min_cuda_version
        ):
            raise ValueError("Serverless capacity quote CUDA version is invalid")
        for name, value in (
            ("hourly_worker_rate_usd", self.hourly_worker_rate_usd),
            ("estimated_worker_cost_usd", self.estimated_worker_cost_usd),
            ("maximum_worker_cost_usd", self.maximum_worker_cost_usd),
            ("estimated_cost_usd", self.estimated_cost_usd),
            ("cost_ceiling_usd", self.cost_ceiling_usd),
        ):
            _positive_decimal(value, name)
        _nonnegative_decimal(
            self.estimated_non_worker_cost_usd, "estimated_non_worker_cost_usd"
        )
        _positive_decimal(
            self.maximum_non_worker_cost_usd, "maximum_non_worker_cost_usd"
        )
        if self.estimated_queue_delay_seconds is not None:
            _nonnegative_int(
                self.estimated_queue_delay_seconds,
                "estimated_queue_delay_seconds",
            )
        _nonnegative_int(
            self.estimated_worker_start_seconds,
            "estimated_worker_start_seconds",
        )
        for name, value in (
            ("estimated_execution_seconds", self.estimated_execution_seconds),
            ("idle_tail_seconds", self.idle_tail_seconds),
            ("maximum_queue_delay_seconds", self.maximum_queue_delay_seconds),
            ("maximum_worker_start_seconds", self.maximum_worker_start_seconds),
            ("maximum_execution_seconds", self.maximum_execution_seconds),
            ("estimated_billable_seconds", self.estimated_billable_seconds),
            ("maximum_billable_seconds", self.maximum_billable_seconds),
        ):
            _positive_int(value, name)
        expected_billable = (
            self.estimated_worker_start_seconds
            + self.estimated_execution_seconds
            + self.idle_tail_seconds
        )
        if self.estimated_billable_seconds != expected_billable:
            raise ValueError("Serverless estimated billable time is inconsistent")
        if self.estimated_billable_seconds > self.maximum_billable_seconds:
            raise ValueError("Serverless estimated billable time exceeds its maximum")
        if self.estimated_worker_start_seconds > self.maximum_worker_start_seconds:
            raise ValueError("Serverless estimated worker start exceeds its maximum")
        if self.estimated_execution_seconds > self.maximum_execution_seconds:
            raise ValueError("Serverless estimated execution exceeds its maximum")
        if self.maximum_billable_seconds != (
            self.maximum_worker_start_seconds
            + self.maximum_execution_seconds
            + self.idle_tail_seconds
        ):
            raise ValueError("Serverless maximum billable time is inconsistent")
        if (
            self.estimated_queue_delay_seconds is not None
            and self.estimated_queue_delay_seconds > self.maximum_queue_delay_seconds
        ):
            raise ValueError("Serverless estimated queue delay exceeds its maximum")
        expected_worker = serverless_worker_cost_usd(
            self.hourly_worker_rate_usd,
            self.gpu_count,
            self.estimated_billable_seconds,
        )
        maximum_worker = serverless_worker_cost_usd(
            self.hourly_worker_rate_usd,
            self.gpu_count,
            self.maximum_billable_seconds,
        )
        if (
            self.estimated_worker_cost_usd != expected_worker
            or self.maximum_worker_cost_usd != maximum_worker
            or self.estimated_cost_usd
            != expected_worker + self.estimated_non_worker_cost_usd
            or self.cost_ceiling_usd
            != maximum_worker + self.maximum_non_worker_cost_usd
        ):
            raise ValueError("Serverless capacity quote cost is inconsistent")
        if self.estimated_cost_usd > self.cost_ceiling_usd:
            raise ValueError("Serverless estimated cost exceeds its ceiling")
        require_aware(self.catalog_observed_at, "catalog_observed_at")
        require_aware(self.expires_at, "expires_at")
        if self.expires_at <= self.catalog_observed_at:
            raise ValueError("Serverless capacity quote expiry is invalid")

    def assert_fresh(
        self, *, now: datetime, accepted_cost_ceiling_usd: Decimal
    ) -> None:
        require_aware(now, "quote validation time")
        _positive_decimal(accepted_cost_ceiling_usd, "accepted_cost_ceiling_usd")
        if not self.catalog_observed_at <= now < self.expires_at:
            raise RunPodManagerError("Serverless capacity quote has expired")
        if accepted_cost_ceiling_usd != self.cost_ceiling_usd:
            raise RunPodManagerError(
                "Accepted Serverless cost ceiling does not match the quote"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "provider_quote_id": self.provider_quote_id,
            "workload_kind": self.workload_kind,
            "parameters_sha256": self.parameters_sha256,
            "profile_id": self.profile_id,
            "endpoint_profile_sha256": self.endpoint_profile_sha256,
            "endpoint_id": self.endpoint_id,
            "catalog_observation_sha256": self.catalog_observation_sha256,
            "gpu_id": self.gpu_id,
            "gpu_pool": self.gpu_pool,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "data_center_id": self.data_center_id,
            "cloud": self.cloud.value,
            "gpu_count": self.gpu_count,
            "min_cuda_version": self.min_cuda_version,
            "availability": self.availability.value,
            "benchmark_id": self.benchmark_id,
            "hourly_worker_rate_usd": decimal_text(self.hourly_worker_rate_usd),
            "estimated_queue_delay_seconds": self.estimated_queue_delay_seconds,
            "estimated_worker_start_seconds": self.estimated_worker_start_seconds,
            "estimated_execution_seconds": self.estimated_execution_seconds,
            "idle_tail_seconds": self.idle_tail_seconds,
            "maximum_queue_delay_seconds": self.maximum_queue_delay_seconds,
            "maximum_worker_start_seconds": self.maximum_worker_start_seconds,
            "maximum_execution_seconds": self.maximum_execution_seconds,
            "estimated_billable_seconds": self.estimated_billable_seconds,
            "maximum_billable_seconds": self.maximum_billable_seconds,
            "estimated_worker_cost_usd": decimal_text(self.estimated_worker_cost_usd),
            "maximum_worker_cost_usd": decimal_text(self.maximum_worker_cost_usd),
            "estimated_non_worker_cost_usd": decimal_text(
                self.estimated_non_worker_cost_usd
            ),
            "maximum_non_worker_cost_usd": decimal_text(
                self.maximum_non_worker_cost_usd
            ),
            "estimated_cost_usd": decimal_text(self.estimated_cost_usd),
            "cost_ceiling_usd": decimal_text(self.cost_ceiling_usd),
            "catalog_observed_at": iso_datetime(self.catalog_observed_at),
            "expires_at": iso_datetime(self.expires_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessCapacityQuote:
        _exact_keys(value, _QUOTE_FIELDS, "Serverless capacity quote")
        return cls(
            schema_version=_required_int(value, "schema_version"),
            contract_version=_required_string(value, "contract_version"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            workload_kind=_required_string(value, "workload_kind"),
            parameters_sha256=_required_string(value, "parameters_sha256"),
            profile_id=_required_string(value, "profile_id"),
            endpoint_profile_sha256=_required_string(value, "endpoint_profile_sha256"),
            endpoint_id=_required_string(value, "endpoint_id"),
            catalog_observation_sha256=_required_string(
                value, "catalog_observation_sha256"
            ),
            gpu_id=_required_string(value, "gpu_id"),
            gpu_pool=_required_string(value, "gpu_pool"),
            gpu_name=_required_string(value, "gpu_name"),
            vram_gb=_required_int(value, "vram_gb"),
            data_center_id=_required_string(value, "data_center_id"),
            cloud=CloudType(_required_string(value, "cloud")),
            gpu_count=_required_int(value, "gpu_count"),
            min_cuda_version=_optional_string(value.get("min_cuda_version")),
            availability=Availability(_required_string(value, "availability")),
            benchmark_id=_required_string(value, "benchmark_id"),
            hourly_worker_rate_usd=_required_decimal(value, "hourly_worker_rate_usd"),
            estimated_queue_delay_seconds=_optional_int(
                value.get("estimated_queue_delay_seconds"),
                "estimated_queue_delay_seconds",
            ),
            estimated_worker_start_seconds=_required_int(
                value, "estimated_worker_start_seconds"
            ),
            estimated_execution_seconds=_required_int(
                value, "estimated_execution_seconds"
            ),
            idle_tail_seconds=_required_int(value, "idle_tail_seconds"),
            maximum_queue_delay_seconds=_required_int(
                value, "maximum_queue_delay_seconds"
            ),
            maximum_worker_start_seconds=_required_int(
                value, "maximum_worker_start_seconds"
            ),
            maximum_execution_seconds=_required_int(value, "maximum_execution_seconds"),
            estimated_billable_seconds=_required_int(
                value, "estimated_billable_seconds"
            ),
            maximum_billable_seconds=_required_int(value, "maximum_billable_seconds"),
            estimated_worker_cost_usd=_required_decimal(
                value, "estimated_worker_cost_usd"
            ),
            maximum_worker_cost_usd=_required_decimal(value, "maximum_worker_cost_usd"),
            estimated_non_worker_cost_usd=_required_decimal(
                value, "estimated_non_worker_cost_usd"
            ),
            maximum_non_worker_cost_usd=_required_decimal(
                value, "maximum_non_worker_cost_usd"
            ),
            estimated_cost_usd=_required_decimal(value, "estimated_cost_usd"),
            cost_ceiling_usd=_required_decimal(value, "cost_ceiling_usd"),
            catalog_observed_at=parse_datetime(
                _required_string(value, "catalog_observed_at")
            ),
            expires_at=parse_datetime(_required_string(value, "expires_at")),
        )


@dataclass(frozen=True)
class ServerlessBillingAttempt:
    """Exact externally-owned attempt interval presented for settlement."""

    attempt_id: str
    job_id: str
    endpoint_id: str
    provider_quote_id: str
    exclusive_window_sha256: str
    submitted_at: datetime
    completed_at: datetime

    def __post_init__(self) -> None:
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("job_id", self.job_id),
            ("endpoint_id", self.endpoint_id),
            ("provider_quote_id", self.provider_quote_id),
        ):
            _safe_identifier(value, name)
        _sha256(self.exclusive_window_sha256, "exclusive_window_sha256")
        require_aware(self.submitted_at, "submitted_at")
        require_aware(self.completed_at, "completed_at")
        if self.completed_at <= self.submitted_at:
            raise ValueError("Serverless billing attempt interval is invalid")

    def validate_quote(self, quote: ServerlessCapacityQuote) -> None:
        if self.endpoint_id != quote.endpoint_id:
            raise RunPodManagerError("Serverless billing endpoint does not match quote")
        if self.provider_quote_id != quote.provider_quote_id:
            raise RunPodManagerError("Serverless billing quote identity does not match")
        if not quote.catalog_observed_at <= self.submitted_at < quote.expires_at:
            raise RunPodManagerError(
                "Serverless job submission falls outside the accepted quote interval"
            )
        maximum_elapsed = (
            quote.maximum_queue_delay_seconds + quote.maximum_billable_seconds
        )
        if (self.completed_at - self.submitted_at).total_seconds() > maximum_elapsed:
            raise RunPodManagerError(
                "Serverless attempt exceeds the accepted maximum interval"
            )


@dataclass(frozen=True)
class ServerlessBillingReceipt:
    """Final endpoint billing evidence bound to one exclusive job attempt."""

    schema_version: int
    contract_version: str
    provider_billing_id: str
    provider_quote_id: str
    endpoint_profile_sha256: str
    endpoint_id: str
    job_id: str
    attempt_id: str
    exclusive_window_sha256: str
    attempt_started_at: datetime
    attempt_completed_at: datetime
    billing_window_from: datetime
    billing_window_until: datetime
    hourly_worker_rate_usd: Decimal
    queue_delay_ms: int | None
    worker_startup_ms: int | None
    execution_ms: int | None
    idle_tail_ms: int | None
    gpu_cost_usd: Decimal
    cpu_cost_usd: Decimal
    disk_cost_usd: Decimal
    fee_cost_usd: Decimal
    actual_cost_usd: Decimal
    reconciled_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != SERVERLESS_CAPACITY_SCHEMA_VERSION:
            raise ValueError("Serverless billing receipt schema is unsupported")
        if self.contract_version != SERVERLESS_CAPACITY_CONTRACT_VERSION:
            raise ValueError("Serverless billing receipt contract is unsupported")
        for name, value in (
            ("provider_billing_id", self.provider_billing_id),
            ("provider_quote_id", self.provider_quote_id),
            ("endpoint_id", self.endpoint_id),
            ("job_id", self.job_id),
            ("attempt_id", self.attempt_id),
        ):
            _safe_identifier(value, name)
        _sha256(self.endpoint_profile_sha256, "endpoint_profile_sha256")
        _sha256(self.exclusive_window_sha256, "exclusive_window_sha256")
        for name, value in (
            ("attempt_started_at", self.attempt_started_at),
            ("attempt_completed_at", self.attempt_completed_at),
            ("billing_window_from", self.billing_window_from),
            ("billing_window_until", self.billing_window_until),
            ("reconciled_at", self.reconciled_at),
        ):
            require_aware(value, name)
        if not (
            self.billing_window_from
            <= self.attempt_started_at
            < self.attempt_completed_at
            <= self.billing_window_until
            <= self.reconciled_at
        ):
            raise ValueError("Serverless billing receipt intervals are inconsistent")
        _positive_decimal(self.hourly_worker_rate_usd, "hourly_worker_rate_usd")
        for name, value in (
            ("queue_delay_ms", self.queue_delay_ms),
            ("worker_startup_ms", self.worker_startup_ms),
            ("execution_ms", self.execution_ms),
            ("idle_tail_ms", self.idle_tail_ms),
        ):
            if value is not None:
                _nonnegative_int(value, name)
        for name, value in (
            ("gpu_cost_usd", self.gpu_cost_usd),
            ("cpu_cost_usd", self.cpu_cost_usd),
            ("disk_cost_usd", self.disk_cost_usd),
            ("fee_cost_usd", self.fee_cost_usd),
            ("actual_cost_usd", self.actual_cost_usd),
        ):
            _nonnegative_decimal(value, name)
        if self.actual_cost_usd != (
            self.gpu_cost_usd
            + self.cpu_cost_usd
            + self.disk_cost_usd
            + self.fee_cost_usd
        ):
            raise ValueError("Serverless billing components do not equal total cost")
        observed_job_ms = sum(
            value for value in (self.queue_delay_ms, self.execution_ms) if value
        )
        attempt_ms = int(
            (self.attempt_completed_at - self.attempt_started_at).total_seconds()
            * 1_000
        )
        if observed_job_ms > attempt_ms:
            raise ValueError(
                "Serverless job timing exceeds the recorded attempt interval"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "provider_billing_id": self.provider_billing_id,
            "provider_quote_id": self.provider_quote_id,
            "endpoint_profile_sha256": self.endpoint_profile_sha256,
            "endpoint_id": self.endpoint_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "exclusive_window_sha256": self.exclusive_window_sha256,
            "attempt_started_at": iso_datetime(self.attempt_started_at),
            "attempt_completed_at": iso_datetime(self.attempt_completed_at),
            "billing_window_from": iso_datetime(self.billing_window_from),
            "billing_window_until": iso_datetime(self.billing_window_until),
            "hourly_worker_rate_usd": decimal_text(self.hourly_worker_rate_usd),
            "queue_delay_ms": self.queue_delay_ms,
            "worker_startup_ms": self.worker_startup_ms,
            "execution_ms": self.execution_ms,
            "idle_tail_ms": self.idle_tail_ms,
            "gpu_cost_usd": decimal_text(self.gpu_cost_usd),
            "cpu_cost_usd": decimal_text(self.cpu_cost_usd),
            "disk_cost_usd": decimal_text(self.disk_cost_usd),
            "fee_cost_usd": decimal_text(self.fee_cost_usd),
            "actual_cost_usd": decimal_text(self.actual_cost_usd),
            "reconciled_at": iso_datetime(self.reconciled_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessBillingReceipt:
        _exact_keys(value, _RECEIPT_FIELDS, "Serverless billing receipt")
        return cls(
            schema_version=_required_int(value, "schema_version"),
            contract_version=_required_string(value, "contract_version"),
            provider_billing_id=_required_string(value, "provider_billing_id"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            endpoint_profile_sha256=_required_string(value, "endpoint_profile_sha256"),
            endpoint_id=_required_string(value, "endpoint_id"),
            job_id=_required_string(value, "job_id"),
            attempt_id=_required_string(value, "attempt_id"),
            exclusive_window_sha256=_required_string(value, "exclusive_window_sha256"),
            attempt_started_at=parse_datetime(
                _required_string(value, "attempt_started_at")
            ),
            attempt_completed_at=parse_datetime(
                _required_string(value, "attempt_completed_at")
            ),
            billing_window_from=parse_datetime(
                _required_string(value, "billing_window_from")
            ),
            billing_window_until=parse_datetime(
                _required_string(value, "billing_window_until")
            ),
            hourly_worker_rate_usd=_required_decimal(value, "hourly_worker_rate_usd"),
            queue_delay_ms=_optional_int(value.get("queue_delay_ms"), "queue_delay_ms"),
            worker_startup_ms=_optional_int(
                value.get("worker_startup_ms"), "worker_startup_ms"
            ),
            execution_ms=_optional_int(value.get("execution_ms"), "execution_ms"),
            idle_tail_ms=_optional_int(value.get("idle_tail_ms"), "idle_tail_ms"),
            gpu_cost_usd=_required_decimal(value, "gpu_cost_usd"),
            cpu_cost_usd=_required_decimal(value, "cpu_cost_usd"),
            disk_cost_usd=_required_decimal(value, "disk_cost_usd"),
            fee_cost_usd=_required_decimal(value, "fee_cost_usd"),
            actual_cost_usd=_required_decimal(value, "actual_cost_usd"),
            reconciled_at=parse_datetime(_required_string(value, "reconciled_at")),
        )


def serverless_worker_cost_usd(
    hourly_rate_usd: Decimal, gpu_count: int, seconds: int
) -> Decimal:
    """Return a six-decimal upper bound for per-second worker compute."""

    _positive_decimal(hourly_rate_usd, "hourly_rate_usd")
    _positive_int(gpu_count, "gpu_count")
    _positive_int(seconds, "seconds")
    return (
        hourly_rate_usd * Decimal(gpu_count) * Decimal(seconds) / Decimal(3600)
    ).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)


def decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if not value.is_finite():
        raise ValueError("Decimal value must be finite")
    return format(value, "f")


def json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def iso_datetime(value: datetime) -> str:
    require_aware(value, "timestamp")
    return value.astimezone(UTC).isoformat()


def parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RunPodManagerError("Stored Serverless timestamp is invalid") from exc
    require_aware(parsed, "stored timestamp")
    return parsed.astimezone(UTC)


def require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"Serverless capacity {name} must be timezone-aware")


def _safe_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Serverless capacity {name} is not a safe identifier")


def _safe_display_name(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 255
        or any(ord(char) < 32 for char in value)
        or "http://" in value.lower()
        or "https://" in value.lower()
    ):
        raise ValueError(f"Serverless capacity {name} is invalid")


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"Serverless capacity {name} must be a SHA-256 digest")


def _one_identifier(values: tuple[str, ...], name: str, *, reason: str) -> None:
    if not isinstance(values, tuple) or len(values) != 1:
        raise ValueError(f"Serverless capacity {name}: {reason}")
    _safe_identifier(values[0], name)


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Serverless capacity {name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Serverless capacity {name} must be a nonnegative integer")


def _positive_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"Serverless capacity {name} must be finite and positive")


def _nonnegative_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"Serverless capacity {name} must be finite and nonnegative")


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RunPodManagerError(f"{name} contains missing or unsupported fields")


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RunPodManagerError(f"Stored Serverless {key} must be a string")
    return item


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RunPodManagerError("Stored Serverless optional string is invalid")
    return value


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise RunPodManagerError(f"Stored Serverless {key} must be an integer")
    return item


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunPodManagerError(f"Stored Serverless {name} must be an integer")
    return value


def _required_decimal(value: Mapping[str, Any], key: str) -> Decimal:
    item = value.get(key)
    if not isinstance(item, str):
        raise RunPodManagerError(f"Stored Serverless {key} must be a decimal string")
    try:
        return Decimal(item)
    except (InvalidOperation, ValueError) as exc:
        raise RunPodManagerError(f"Stored Serverless {key} is invalid") from exc


_QUOTE_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "provider_quote_id",
        "workload_kind",
        "parameters_sha256",
        "profile_id",
        "endpoint_profile_sha256",
        "endpoint_id",
        "catalog_observation_sha256",
        "gpu_id",
        "gpu_pool",
        "gpu_name",
        "vram_gb",
        "data_center_id",
        "cloud",
        "gpu_count",
        "min_cuda_version",
        "availability",
        "benchmark_id",
        "hourly_worker_rate_usd",
        "estimated_queue_delay_seconds",
        "estimated_worker_start_seconds",
        "estimated_execution_seconds",
        "idle_tail_seconds",
        "maximum_queue_delay_seconds",
        "maximum_worker_start_seconds",
        "maximum_execution_seconds",
        "estimated_billable_seconds",
        "maximum_billable_seconds",
        "estimated_worker_cost_usd",
        "maximum_worker_cost_usd",
        "estimated_non_worker_cost_usd",
        "maximum_non_worker_cost_usd",
        "estimated_cost_usd",
        "cost_ceiling_usd",
        "catalog_observed_at",
        "expires_at",
    }
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "provider_billing_id",
        "provider_quote_id",
        "endpoint_profile_sha256",
        "endpoint_id",
        "job_id",
        "attempt_id",
        "exclusive_window_sha256",
        "attempt_started_at",
        "attempt_completed_at",
        "billing_window_from",
        "billing_window_until",
        "hourly_worker_rate_usd",
        "queue_delay_ms",
        "worker_startup_ms",
        "execution_ms",
        "idle_tail_ms",
        "gpu_cost_usd",
        "cpu_cost_usd",
        "disk_cost_usd",
        "fee_cost_usd",
        "actual_cost_usd",
        "reconciled_at",
    }
)


__all__ = [
    "SERVERLESS_CAPACITY_CONTRACT_VERSION",
    "SERVERLESS_CAPACITY_SCHEMA_VERSION",
    "ServerlessBillingAttempt",
    "ServerlessBillingReceipt",
    "ServerlessCapacityConstraints",
    "ServerlessCapacityQuote",
    "ServerlessCapacityQuoteRequest",
    "ServerlessEndpointProfile",
    "decimal_text",
    "iso_datetime",
    "json_sha256",
    "parse_datetime",
    "require_aware",
    "serverless_worker_cost_usd",
]
