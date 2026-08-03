"""Provider-neutral contracts for finite Serverless capacity and billing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

from .models import (
    Availability,
    CloudType,
    ComputeProduct,
    EndpointCreateRequest,
    FlashBoot,
    PlacementRequirements,
    RunPodManagerError,
)

SERVERLESS_CAPACITY_SCHEMA_VERSION = 1
SERVERLESS_CAPACITY_CONTRACT_VERSION = "serverless-capacity-v1"
RUNPOD_V2_OPENAPI_SOURCE_URL = "https://api.runpod.io/v2/openapi.json"
RUNPOD_V2_OPENAPI_SHA256 = (
    "0bbdd828569233765e310e773e34586b33a6e38f55afde989ebd670152ed5c13"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}")
_IMMUTABLE_REFERENCE_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_MAX_SERVERLESS_POLICY_MS = 7 * 24 * 60 * 60 * 1_000
_MAX_BILLABLE_COVERAGE_SECONDS = 7 * 24 * 60 * 60 + 3_600
_MAX_EXCLUSIVE_BILLING_HOURS = (
    math.ceil((_MAX_BILLABLE_COVERAGE_SECONDS + 300) / 3_600) + 1
)
_MIN_EXECUTION_TIMEOUT_MS = 5_000
_MIN_JOB_TTL_MS = 10_000


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
        _identifiers(
            self.allowed_gpu_pools,
            "allowed_gpu_pools",
            reason="at least one allowed endpoint pool is required",
        )
        _identifiers(
            self.allowed_data_center_ids,
            "allowed_data_center_ids",
            reason="at least one allowed endpoint data center is required",
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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessCapacityConstraints:
        _exact_keys(value, _CONSTRAINT_FIELDS, "Serverless capacity constraints")
        pools = value.get("allowed_gpu_pools")
        data_centers = value.get("allowed_data_center_ids")
        if not isinstance(pools, list) or not all(
            isinstance(item, str) for item in pools
        ):
            raise RunPodManagerError("Stored Serverless GPU pools are invalid")
        if not isinstance(data_centers, list) or not all(
            isinstance(item, str) for item in data_centers
        ):
            raise RunPodManagerError("Stored Serverless data centers are invalid")
        maximum_rate = value.get("max_hourly_worker_rate_usd")
        if maximum_rate is not None and not isinstance(maximum_rate, str):
            raise RunPodManagerError(
                "Stored Serverless maximum hourly rate must be a decimal string"
            )
        return cls(
            min_vram_gb=_required_int(value, "min_vram_gb"),
            gpu_count=_required_int(value, "gpu_count"),
            cloud=CloudType(_required_string(value, "cloud")),
            min_cuda_version=_optional_string(value.get("min_cuda_version")),
            allowed_gpu_pools=tuple(pools),
            allowed_data_center_ids=tuple(data_centers),
            max_hourly_worker_rate_usd=(
                _required_decimal(value, "max_hourly_worker_rate_usd")
                if maximum_rate is not None
                else None
            ),
            minimum_availability=Availability(
                _required_string(value, "minimum_availability")
            ),
            benchmark_id=_required_string(value, "benchmark_id"),
        )


@dataclass(frozen=True)
class ServerlessEndpointSpec:
    """Canonical immutable configuration for one finite queue endpoint."""

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
    registry_id: str | None = None
    network_volume_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
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
        if not (
            _MIN_EXECUTION_TIMEOUT_MS
            <= self.execution_timeout_ms
            <= _MAX_SERVERLESS_POLICY_MS
        ):
            raise ValueError(
                "Serverless endpoint execution_timeout_ms must be between "
                "5 seconds and 7 days"
            )
        if not isinstance(self.flashboot, FlashBoot):
            raise TypeError("Serverless endpoint flashboot must be a FlashBoot")
        _positive_int(self.disk_gb, "disk_gb")
        if self.registry_id is not None:
            _safe_identifier(self.registry_id, "registry_id")
        if self.network_volume_ids:
            raise ValueError(
                "Finite Serverless billing cannot attribute persistent network "
                "volume cost to one attempt"
            )

    @property
    def spec_sha256(self) -> str:
        return json_sha256(
            {
                "worker_reference": self.worker_reference,
                "constraints": self.constraints.fingerprint_dict(),
                "workers": {"min": self.workers_min, "max": self.workers_max},
                "idle_tail_seconds": self.idle_tail_seconds,
                "scaling_type": self.scaling_type,
                "scaling_value": decimal_text(self.scaling_value),
                "execution_timeout_ms": self.execution_timeout_ms,
                "flashboot": self.flashboot.value,
                "disk_gb": self.disk_gb,
                "registry_id": self.registry_id,
                "network_volume_ids": list(self.network_volume_ids),
            }
        )

    def create_request(
        self,
        endpoint_name: str,
        *,
        gpu_pool: str,
        data_center_id: str,
    ) -> EndpointCreateRequest:
        """Build the one canonical REST v2 create request for this spec."""

        _safe_identifier(endpoint_name, "endpoint_name")
        if gpu_pool not in self.constraints.allowed_gpu_pools:
            raise ValueError(
                "Selected Serverless GPU pool is outside the endpoint spec"
            )
        if data_center_id not in self.constraints.allowed_data_center_ids:
            raise ValueError(
                "Selected Serverless data center is outside the endpoint spec"
            )
        return EndpointCreateRequest(
            name=endpoint_name,
            image=self.worker_reference,
            gpu_pools=(gpu_pool,),
            endpoint_type="QUEUE",
            scaling={
                "type": self.scaling_type,
                "queueDelay": float(self.scaling_value),
            },
            gpu_count=self.constraints.gpu_count,
            workers_min=self.workers_min,
            workers_max=self.workers_max,
            idle_timeout_seconds=self.idle_tail_seconds,
            disk_gb=self.disk_gb,
            ports=(),
            env={},
            args=None,
            registry_id=self.registry_id,
            data_center_ids=(data_center_id,),
            network_volume_ids=self.network_volume_ids,
            execution_timeout_ms=self.execution_timeout_ms,
            flashboot=self.flashboot,
        )

    def operation_digest(
        self,
        endpoint_name: str,
        *,
        gpu_pool: str,
        data_center_id: str,
    ) -> str:
        """Digest the exact JSON body that REST v2 will receive."""

        return json_sha256(
            self.create_request(
                endpoint_name,
                gpu_pool=gpu_pool,
                data_center_id=data_center_id,
            ).to_payload()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_reference": self.worker_reference,
            "constraints": self.constraints.fingerprint_dict(),
            "workers_min": self.workers_min,
            "workers_max": self.workers_max,
            "idle_tail_seconds": self.idle_tail_seconds,
            "scaling_type": self.scaling_type,
            "scaling_value": decimal_text(self.scaling_value),
            "execution_timeout_ms": self.execution_timeout_ms,
            "flashboot": self.flashboot.value,
            "disk_gb": self.disk_gb,
            "registry_id": self.registry_id,
            "network_volume_ids": list(self.network_volume_ids),
            "spec_sha256": self.spec_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessEndpointSpec:
        _exact_keys(value, _ENDPOINT_SPEC_FIELDS, "Serverless endpoint spec")
        raw_constraints = value.get("constraints")
        if not isinstance(raw_constraints, Mapping):
            raise RunPodManagerError("Stored Serverless constraints are invalid")
        volumes = value.get("network_volume_ids")
        if not isinstance(volumes, list) or not all(
            isinstance(item, str) for item in volumes
        ):
            raise RunPodManagerError("Stored Serverless network volumes are invalid")
        registry = value.get("registry_id")
        if registry is not None and not isinstance(registry, str):
            raise RunPodManagerError("Stored Serverless registry ID is invalid")
        parsed = cls(
            worker_reference=_required_string(value, "worker_reference"),
            constraints=ServerlessCapacityConstraints.from_dict(raw_constraints),
            workers_min=_required_int(value, "workers_min"),
            workers_max=_required_int(value, "workers_max"),
            idle_tail_seconds=_required_int(value, "idle_tail_seconds"),
            scaling_type=_required_string(value, "scaling_type"),
            scaling_value=_required_decimal(value, "scaling_value"),
            execution_timeout_ms=_required_int(value, "execution_timeout_ms"),
            flashboot=FlashBoot(_required_string(value, "flashboot")),
            disk_gb=_required_int(value, "disk_gb"),
            registry_id=registry,
            network_volume_ids=tuple(volumes),
        )
        stored_digest = _required_string(value, "spec_sha256")
        _sha256(stored_digest, "spec_sha256")
        if stored_digest != parsed.spec_sha256:
            raise RunPodManagerError(
                "Stored Serverless endpoint spec digest mismatches"
            )
        return parsed


@dataclass(frozen=True)
class ServerlessEndpointProfile:
    """Identity of one reusable endpoint plus its canonical immutable spec."""

    profile_id: str
    endpoint_id: str
    endpoint_name: str
    spec: ServerlessEndpointSpec

    def __post_init__(self) -> None:
        for name, value in (
            ("profile_id", self.profile_id),
            ("endpoint_id", self.endpoint_id),
            ("endpoint_name", self.endpoint_name),
        ):
            _safe_identifier(value, name)
        if not isinstance(self.spec, ServerlessEndpointSpec):
            raise TypeError("Serverless endpoint profile spec has an invalid type")

    @property
    def profile_sha256(self) -> str:
        return json_sha256(
            {
                "profile_id": self.profile_id,
                "endpoint_id": self.endpoint_id,
                "endpoint_name": self.endpoint_name,
                "endpoint_spec_sha256": self.spec.spec_sha256,
            }
        )

    @property
    def worker_reference(self) -> str:
        return self.spec.worker_reference

    @property
    def constraints(self) -> ServerlessCapacityConstraints:
        return self.spec.constraints

    @property
    def workers_min(self) -> int:
        return self.spec.workers_min

    @property
    def workers_max(self) -> int:
        return self.spec.workers_max

    @property
    def idle_tail_seconds(self) -> int:
        return self.spec.idle_tail_seconds

    @property
    def scaling_type(self) -> str:
        return self.spec.scaling_type

    @property
    def scaling_value(self) -> Decimal:
        return self.spec.scaling_value

    @property
    def execution_timeout_ms(self) -> int:
        return self.spec.execution_timeout_ms

    @property
    def flashboot(self) -> FlashBoot:
        return self.spec.flashboot

    @property
    def disk_gb(self) -> int:
        return self.spec.disk_gb

    @property
    def registry_id(self) -> str | None:
        return self.spec.registry_id

    @property
    def network_volume_ids(self) -> tuple[str, ...]:
        return self.spec.network_volume_ids


@dataclass(frozen=True)
class PlannedServerlessEndpoint:
    """Pre-registered run-scoped endpoint identity with no provider resource ID."""

    plan_id: str
    endpoint_name: str
    spec: ServerlessEndpointSpec

    def __post_init__(self) -> None:
        _safe_identifier(self.plan_id, "plan_id")
        _safe_identifier(self.endpoint_name, "endpoint_name")
        if not isinstance(self.spec, ServerlessEndpointSpec):
            raise TypeError("Planned Serverless endpoint spec has an invalid type")

    @property
    def plan_sha256(self) -> str:
        return json_sha256(
            {
                "plan_id": self.plan_id,
                "endpoint_name": self.endpoint_name,
                "endpoint_spec_sha256": self.spec.spec_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "endpoint_name": self.endpoint_name,
            "spec": self.spec.to_dict(),
            "plan_sha256": self.plan_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlannedServerlessEndpoint:
        _exact_keys(value, _PLANNED_ENDPOINT_FIELDS, "Planned Serverless endpoint")
        raw_spec = value.get("spec")
        if not isinstance(raw_spec, Mapping):
            raise RunPodManagerError("Stored planned Serverless spec is invalid")
        parsed = cls(
            plan_id=_required_string(value, "plan_id"),
            endpoint_name=_required_string(value, "endpoint_name"),
            spec=ServerlessEndpointSpec.from_dict(raw_spec),
        )
        stored_digest = _required_string(value, "plan_sha256")
        _sha256(stored_digest, "plan_sha256")
        if stored_digest != parsed.plan_sha256:
            raise RunPodManagerError(
                "Stored planned Serverless endpoint digest mismatches"
            )
        return parsed


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
    job_execution_timeout_ms: int
    job_ttl_ms: int
    estimated_non_worker_cost_usd: Decimal
    maximum_non_worker_cost_usd: Decimal
    quote_ttl_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ServerlessEndpointProfile):
            raise TypeError("Serverless quote profile has an invalid type")
        _validate_capacity_request(self, self.profile.spec)


@dataclass(frozen=True)
class PlannedServerlessCapacityQuoteRequest:
    """Content-free inputs for a not-yet-created run-scoped endpoint."""

    endpoint: PlannedServerlessEndpoint
    workload_kind: str
    parameters_sha256: str
    estimated_queue_delay_seconds: int | None
    estimated_worker_start_seconds: int
    estimated_execution_seconds: int
    maximum_queue_delay_seconds: int
    maximum_worker_start_seconds: int
    maximum_execution_seconds: int
    maximum_billable_seconds: int
    job_execution_timeout_ms: int
    job_ttl_ms: int
    estimated_non_worker_cost_usd: Decimal
    maximum_non_worker_cost_usd: Decimal
    quote_ttl_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, PlannedServerlessEndpoint):
            raise TypeError("Planned Serverless quote endpoint has an invalid type")
        _validate_capacity_request(self, self.endpoint.spec)


def _validate_capacity_request(request: Any, spec: ServerlessEndpointSpec) -> None:
    """Validate the shared finite-job envelope against one endpoint spec."""

    _safe_identifier(request.workload_kind, "workload_kind")
    _sha256(request.parameters_sha256, "parameters_sha256")
    if request.estimated_queue_delay_seconds is not None:
        _nonnegative_int(
            request.estimated_queue_delay_seconds,
            "estimated_queue_delay_seconds",
        )
    _nonnegative_int(
        request.estimated_worker_start_seconds,
        "estimated_worker_start_seconds",
    )
    _positive_int(request.estimated_execution_seconds, "estimated_execution_seconds")
    _positive_int(request.maximum_queue_delay_seconds, "maximum_queue_delay_seconds")
    _positive_int(request.maximum_worker_start_seconds, "maximum_worker_start_seconds")
    _positive_int(request.maximum_execution_seconds, "maximum_execution_seconds")
    _positive_int(request.maximum_billable_seconds, "maximum_billable_seconds")
    _positive_int(request.job_execution_timeout_ms, "job_execution_timeout_ms")
    _positive_int(request.job_ttl_ms, "job_ttl_ms")
    if not (
        _MIN_EXECUTION_TIMEOUT_MS
        <= request.job_execution_timeout_ms
        <= _MAX_SERVERLESS_POLICY_MS
    ):
        raise ValueError(
            "Serverless job_execution_timeout_ms must be between 5 seconds and 7 days"
        )
    if not _MIN_JOB_TTL_MS <= request.job_ttl_ms <= _MAX_SERVERLESS_POLICY_MS:
        raise ValueError("Serverless job_ttl_ms must be between 10 seconds and 7 days")
    _nonnegative_decimal(
        request.estimated_non_worker_cost_usd,
        "estimated_non_worker_cost_usd",
    )
    _positive_decimal(
        request.maximum_non_worker_cost_usd,
        "maximum_non_worker_cost_usd",
    )
    _positive_int(request.quote_ttl_seconds, "quote_ttl_seconds")
    if request.quote_ttl_seconds > 300:
        raise ValueError("Serverless quote TTL cannot exceed five minutes")
    if (
        request.estimated_queue_delay_seconds is not None
        and request.estimated_queue_delay_seconds > request.maximum_queue_delay_seconds
    ):
        raise ValueError("Serverless estimated queue delay exceeds its maximum")
    if request.estimated_worker_start_seconds > request.maximum_worker_start_seconds:
        raise ValueError("Serverless estimated worker start exceeds its maximum")
    if request.estimated_execution_seconds > request.maximum_execution_seconds:
        raise ValueError("Serverless estimated execution exceeds its maximum")
    estimated_billable = (
        request.estimated_worker_start_seconds
        + request.estimated_execution_seconds
        + spec.idle_tail_seconds
    )
    maximum_billable = (
        request.maximum_worker_start_seconds
        + request.maximum_execution_seconds
        + spec.idle_tail_seconds
    )
    if request.maximum_billable_seconds != maximum_billable:
        raise ValueError("Serverless maximum billable time is inconsistent")
    if estimated_billable > maximum_billable:
        raise ValueError("Serverless estimated billable time exceeds its maximum")
    if request.job_execution_timeout_ms > spec.execution_timeout_ms:
        raise ValueError(
            "Serverless job execution timeout exceeds the endpoint timeout"
        )
    if request.maximum_execution_seconds * 1_000 > request.job_execution_timeout_ms:
        raise ValueError(
            "Serverless maximum execution exceeds the job execution timeout"
        )
    maximum_job_lifespan_ms = (
        request.maximum_queue_delay_seconds
        + request.maximum_worker_start_seconds
        + request.maximum_execution_seconds
    ) * 1_000
    if maximum_job_lifespan_ms > request.job_ttl_ms:
        raise ValueError("Serverless maximum job lifespan exceeds the job TTL")
    if request.estimated_non_worker_cost_usd > request.maximum_non_worker_cost_usd:
        raise ValueError("Serverless estimated non-worker cost exceeds its maximum")


@dataclass(frozen=True)
class PlannedServerlessCapacityQuote:
    """GET-only capacity evidence for one pre-registered endpoint plan."""

    schema_version: int
    contract_version: str
    provider_quote_id: str
    workload_kind: str
    parameters_sha256: str
    plan_id: str
    plan_sha256: str
    endpoint_name: str
    endpoint_spec_sha256: str
    operation_digest: str
    worker_reference: str
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
    job_execution_timeout_ms: int
    job_ttl_ms: int
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
            raise ValueError("Planned Serverless quote schema is unsupported")
        if self.contract_version != SERVERLESS_CAPACITY_CONTRACT_VERSION:
            raise ValueError("Planned Serverless quote contract is unsupported")
        for name, value in (
            ("provider_quote_id", self.provider_quote_id),
            ("workload_kind", self.workload_kind),
            ("plan_id", self.plan_id),
            ("endpoint_name", self.endpoint_name),
            ("gpu_id", self.gpu_id),
            ("gpu_pool", self.gpu_pool),
            ("data_center_id", self.data_center_id),
            ("benchmark_id", self.benchmark_id),
        ):
            _safe_identifier(value, name)
        _safe_display_name(self.gpu_name, "gpu_name")
        for name, value in (
            ("parameters_sha256", self.parameters_sha256),
            ("plan_sha256", self.plan_sha256),
            ("endpoint_spec_sha256", self.endpoint_spec_sha256),
            ("operation_digest", self.operation_digest),
            ("catalog_observation_sha256", self.catalog_observation_sha256),
        ):
            _sha256(value, name)
        if not _IMMUTABLE_REFERENCE_RE.fullmatch(self.worker_reference):
            raise ValueError(
                "Planned Serverless worker reference must be pinned by SHA-256"
            )
        if "://" in self.worker_reference:
            raise ValueError("Planned Serverless worker reference cannot be a URL")
        if not isinstance(self.cloud, CloudType):
            raise TypeError("Planned Serverless quote cloud is invalid")
        if not isinstance(self.availability, Availability):
            raise TypeError("Planned Serverless quote availability is invalid")
        if self.availability is Availability.NONE:
            raise ValueError(
                "Planned Serverless quote cannot bind unavailable capacity"
            )
        _positive_int(self.gpu_count, "gpu_count")
        _positive_int(self.vram_gb, "vram_gb")
        if self.min_cuda_version is not None and not re.fullmatch(
            r"\d+(?:\.\d+)?", self.min_cuda_version
        ):
            raise ValueError("Planned Serverless quote CUDA version is invalid")
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
            ("job_execution_timeout_ms", self.job_execution_timeout_ms),
            ("job_ttl_ms", self.job_ttl_ms),
            ("estimated_billable_seconds", self.estimated_billable_seconds),
            ("maximum_billable_seconds", self.maximum_billable_seconds),
        ):
            _positive_int(value, name)
        if self.estimated_billable_seconds != (
            self.estimated_worker_start_seconds
            + self.estimated_execution_seconds
            + self.idle_tail_seconds
        ):
            raise ValueError(
                "Planned Serverless estimated billable time is inconsistent"
            )
        if self.maximum_billable_seconds != (
            self.maximum_worker_start_seconds
            + self.maximum_execution_seconds
            + self.idle_tail_seconds
        ):
            raise ValueError("Planned Serverless maximum billable time is inconsistent")
        if self.estimated_billable_seconds > self.maximum_billable_seconds:
            raise ValueError(
                "Planned Serverless estimated billable time exceeds its maximum"
            )
        if self.estimated_worker_start_seconds > self.maximum_worker_start_seconds:
            raise ValueError(
                "Planned Serverless estimated worker start exceeds its maximum"
            )
        if self.estimated_execution_seconds > self.maximum_execution_seconds:
            raise ValueError(
                "Planned Serverless estimated execution exceeds its maximum"
            )
        if (
            self.estimated_queue_delay_seconds is not None
            and self.estimated_queue_delay_seconds > self.maximum_queue_delay_seconds
        ):
            raise ValueError(
                "Planned Serverless estimated queue delay exceeds its maximum"
            )
        if not (
            _MIN_EXECUTION_TIMEOUT_MS
            <= self.job_execution_timeout_ms
            <= _MAX_SERVERLESS_POLICY_MS
        ):
            raise ValueError("Planned Serverless job execution timeout is invalid")
        if not _MIN_JOB_TTL_MS <= self.job_ttl_ms <= _MAX_SERVERLESS_POLICY_MS:
            raise ValueError("Planned Serverless job TTL is invalid")
        if self.maximum_execution_seconds * 1_000 > self.job_execution_timeout_ms:
            raise ValueError(
                "Planned Serverless maximum execution exceeds the job timeout"
            )
        if (
            self.maximum_queue_delay_seconds
            + self.maximum_worker_start_seconds
            + self.maximum_execution_seconds
        ) * 1_000 > self.job_ttl_ms:
            raise ValueError("Planned Serverless maximum job lifespan exceeds its TTL")
        if self.estimated_non_worker_cost_usd > self.maximum_non_worker_cost_usd:
            raise ValueError(
                "Planned Serverless estimated non-worker cost exceeds its maximum"
            )
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
            raise ValueError("Planned Serverless quote cost is inconsistent")
        if self.estimated_cost_usd > self.cost_ceiling_usd:
            raise ValueError("Planned Serverless estimated cost exceeds its ceiling")
        require_aware(self.catalog_observed_at, "catalog_observed_at")
        require_aware(self.expires_at, "expires_at")
        if self.expires_at <= self.catalog_observed_at:
            raise ValueError("Planned Serverless quote expiry is invalid")
        if self.expires_at - self.catalog_observed_at > timedelta(minutes=5):
            raise ValueError("Planned Serverless quote TTL exceeds five minutes")

    @property
    def quote_sha256(self) -> str:
        return json_sha256(self._evidence_dict())

    def assert_fresh(
        self, *, now: datetime, accepted_cost_ceiling_usd: Decimal
    ) -> None:
        require_aware(now, "planned quote validation time")
        _positive_decimal(accepted_cost_ceiling_usd, "accepted_cost_ceiling_usd")
        if not self.catalog_observed_at <= now < self.expires_at:
            raise RunPodManagerError("Planned Serverless capacity quote has expired")
        if accepted_cost_ceiling_usd != self.cost_ceiling_usd:
            raise RunPodManagerError(
                "Accepted Serverless cost ceiling does not match the planned quote"
            )

    def _evidence_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "provider_quote_id": self.provider_quote_id,
            "workload_kind": self.workload_kind,
            "parameters_sha256": self.parameters_sha256,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "endpoint_name": self.endpoint_name,
            "endpoint_spec_sha256": self.endpoint_spec_sha256,
            "operation_digest": self.operation_digest,
            "worker_reference": self.worker_reference,
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
            "job_execution_timeout_ms": self.job_execution_timeout_ms,
            "job_ttl_ms": self.job_ttl_ms,
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

    def to_dict(self) -> dict[str, Any]:
        return {**self._evidence_dict(), "quote_sha256": self.quote_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlannedServerlessCapacityQuote:
        _exact_keys(value, _PLANNED_QUOTE_FIELDS, "Planned Serverless quote")
        stored_digest = _required_string(value, "quote_sha256")
        parsed = cls(
            schema_version=_required_int(value, "schema_version"),
            contract_version=_required_string(value, "contract_version"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            workload_kind=_required_string(value, "workload_kind"),
            parameters_sha256=_required_string(value, "parameters_sha256"),
            plan_id=_required_string(value, "plan_id"),
            plan_sha256=_required_string(value, "plan_sha256"),
            endpoint_name=_required_string(value, "endpoint_name"),
            endpoint_spec_sha256=_required_string(value, "endpoint_spec_sha256"),
            operation_digest=_required_string(value, "operation_digest"),
            worker_reference=_required_string(value, "worker_reference"),
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
            job_execution_timeout_ms=_required_int(value, "job_execution_timeout_ms"),
            job_ttl_ms=_required_int(value, "job_ttl_ms"),
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
        _sha256(stored_digest, "quote_sha256")
        if stored_digest != parsed.quote_sha256:
            raise RunPodManagerError(
                "Stored planned Serverless quote digest mismatches"
            )
        return parsed


@dataclass(frozen=True)
class ServerlessEndpointActivationReceipt:
    """Immutable identity returned only after create and exact REST v2 readback."""

    schema_version: int
    contract_version: str
    provider_activation_id: str
    provider_quote_id: str
    quote_sha256: str
    plan_id: str
    plan_sha256: str
    endpoint_name: str
    endpoint_id: str
    endpoint_spec_sha256: str
    operation_digest: str
    endpoint_readback_sha256: str
    create_started_at: datetime
    activated_at: datetime
    readback_at: datetime
    reconciled_after_ambiguous_create: bool

    def __post_init__(self) -> None:
        if self.schema_version != SERVERLESS_CAPACITY_SCHEMA_VERSION:
            raise ValueError("Serverless activation schema is unsupported")
        if self.contract_version != SERVERLESS_CAPACITY_CONTRACT_VERSION:
            raise ValueError("Serverless activation contract is unsupported")
        for name, value in (
            ("provider_activation_id", self.provider_activation_id),
            ("provider_quote_id", self.provider_quote_id),
            ("plan_id", self.plan_id),
            ("endpoint_name", self.endpoint_name),
            ("endpoint_id", self.endpoint_id),
        ):
            _safe_identifier(value, name)
        for name, value in (
            ("quote_sha256", self.quote_sha256),
            ("plan_sha256", self.plan_sha256),
            ("endpoint_spec_sha256", self.endpoint_spec_sha256),
            ("operation_digest", self.operation_digest),
            ("endpoint_readback_sha256", self.endpoint_readback_sha256),
        ):
            _sha256(value, name)
        require_aware(self.create_started_at, "create_started_at")
        require_aware(self.activated_at, "activated_at")
        require_aware(self.readback_at, "readback_at")
        if not self.create_started_at <= self.activated_at <= self.readback_at:
            raise ValueError("Serverless activation timestamps are inconsistent")
        if not isinstance(self.reconciled_after_ambiguous_create, bool):
            raise TypeError("Serverless activation reconciliation flag must be boolean")
        if self.provider_activation_id != (
            "runpod-serverless-activation:" + json_sha256(self._activation_identity())
        ):
            raise ValueError("Serverless activation identity digest is inconsistent")

    def _activation_identity(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "provider_quote_id": self.provider_quote_id,
            "quote_sha256": self.quote_sha256,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "endpoint_name": self.endpoint_name,
            "endpoint_id": self.endpoint_id,
            "endpoint_spec_sha256": self.endpoint_spec_sha256,
            "operation_digest": self.operation_digest,
            "endpoint_readback_sha256": self.endpoint_readback_sha256,
            "create_started_at": iso_datetime(self.create_started_at),
            "activated_at": iso_datetime(self.activated_at),
            "readback_at": iso_datetime(self.readback_at),
            "reconciled_after_ambiguous_create": (
                self.reconciled_after_ambiguous_create
            ),
        }

    def validate_quote(self, quote: PlannedServerlessCapacityQuote) -> None:
        if (
            self.provider_quote_id != quote.provider_quote_id
            or self.quote_sha256 != quote.quote_sha256
            or self.plan_id != quote.plan_id
            or self.plan_sha256 != quote.plan_sha256
            or self.endpoint_name != quote.endpoint_name
            or self.endpoint_spec_sha256 != quote.endpoint_spec_sha256
            or self.operation_digest != quote.operation_digest
        ):
            raise RunPodManagerError(
                "Serverless activation does not match the accepted planned quote"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "provider_activation_id": self.provider_activation_id,
            "provider_quote_id": self.provider_quote_id,
            "quote_sha256": self.quote_sha256,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "endpoint_name": self.endpoint_name,
            "endpoint_id": self.endpoint_id,
            "endpoint_spec_sha256": self.endpoint_spec_sha256,
            "operation_digest": self.operation_digest,
            "endpoint_readback_sha256": self.endpoint_readback_sha256,
            "create_started_at": iso_datetime(self.create_started_at),
            "activated_at": iso_datetime(self.activated_at),
            "readback_at": iso_datetime(self.readback_at),
            "reconciled_after_ambiguous_create": (
                self.reconciled_after_ambiguous_create
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessEndpointActivationReceipt:
        _exact_keys(value, _ACTIVATION_RECEIPT_FIELDS, "Serverless activation receipt")
        flag = value.get("reconciled_after_ambiguous_create")
        if not isinstance(flag, bool):
            raise RunPodManagerError(
                "Stored Serverless activation reconciliation flag is invalid"
            )
        return cls(
            schema_version=_required_int(value, "schema_version"),
            contract_version=_required_string(value, "contract_version"),
            provider_activation_id=_required_string(value, "provider_activation_id"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            quote_sha256=_required_string(value, "quote_sha256"),
            plan_id=_required_string(value, "plan_id"),
            plan_sha256=_required_string(value, "plan_sha256"),
            endpoint_name=_required_string(value, "endpoint_name"),
            endpoint_id=_required_string(value, "endpoint_id"),
            endpoint_spec_sha256=_required_string(value, "endpoint_spec_sha256"),
            operation_digest=_required_string(value, "operation_digest"),
            endpoint_readback_sha256=_required_string(
                value, "endpoint_readback_sha256"
            ),
            create_started_at=parse_datetime(
                _required_string(value, "create_started_at")
            ),
            activated_at=parse_datetime(_required_string(value, "activated_at")),
            readback_at=parse_datetime(_required_string(value, "readback_at")),
            reconciled_after_ambiguous_create=flag,
        )


@dataclass(frozen=True)
class ServerlessEndpointCleanupReceipt:
    """Proof that the exact activated endpoint was idle, deleted, and absent."""

    schema_version: int
    contract_version: str
    provider_cleanup_id: str
    provider_quote_id: str
    quote_sha256: str
    provider_activation_id: str
    endpoint_name: str
    endpoint_id: str
    endpoint_spec_sha256: str
    operation_digest: str
    endpoint_readback_sha256: str
    workers_zero_observed_at: datetime
    delete_started_at: datetime
    verified_absent_at: datetime
    reconciled_after_delete_error: bool

    def __post_init__(self) -> None:
        if self.schema_version != SERVERLESS_CAPACITY_SCHEMA_VERSION:
            raise ValueError("Serverless cleanup schema is unsupported")
        if self.contract_version != SERVERLESS_CAPACITY_CONTRACT_VERSION:
            raise ValueError("Serverless cleanup contract is unsupported")
        for name, value in (
            ("provider_cleanup_id", self.provider_cleanup_id),
            ("provider_quote_id", self.provider_quote_id),
            ("provider_activation_id", self.provider_activation_id),
            ("endpoint_name", self.endpoint_name),
            ("endpoint_id", self.endpoint_id),
        ):
            _safe_identifier(value, name)
        for name, value in (
            ("quote_sha256", self.quote_sha256),
            ("endpoint_spec_sha256", self.endpoint_spec_sha256),
            ("operation_digest", self.operation_digest),
            ("endpoint_readback_sha256", self.endpoint_readback_sha256),
        ):
            _sha256(value, name)
        require_aware(self.workers_zero_observed_at, "workers_zero_observed_at")
        require_aware(self.delete_started_at, "delete_started_at")
        require_aware(self.verified_absent_at, "verified_absent_at")
        if not (
            self.workers_zero_observed_at
            <= self.delete_started_at
            <= self.verified_absent_at
        ):
            raise ValueError("Serverless cleanup timestamps are inconsistent")
        if not isinstance(self.reconciled_after_delete_error, bool):
            raise TypeError("Serverless cleanup reconciliation flag must be boolean")
        if self.provider_cleanup_id != (
            "runpod-serverless-cleanup:" + json_sha256(self._cleanup_identity())
        ):
            raise ValueError("Serverless cleanup identity digest is inconsistent")

    def _cleanup_identity(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "provider_quote_id": self.provider_quote_id,
            "quote_sha256": self.quote_sha256,
            "provider_activation_id": self.provider_activation_id,
            "endpoint_name": self.endpoint_name,
            "endpoint_id": self.endpoint_id,
            "endpoint_spec_sha256": self.endpoint_spec_sha256,
            "operation_digest": self.operation_digest,
            "endpoint_readback_sha256": self.endpoint_readback_sha256,
            "workers_zero_observed_at": iso_datetime(self.workers_zero_observed_at),
            "delete_started_at": iso_datetime(self.delete_started_at),
            "verified_absent_at": iso_datetime(self.verified_absent_at),
            "reconciled_after_delete_error": self.reconciled_after_delete_error,
        }

    def validate_activation(
        self, activation: ServerlessEndpointActivationReceipt
    ) -> None:
        if (
            self.provider_quote_id != activation.provider_quote_id
            or self.quote_sha256 != activation.quote_sha256
            or self.provider_activation_id != activation.provider_activation_id
            or self.endpoint_name != activation.endpoint_name
            or self.endpoint_id != activation.endpoint_id
            or self.endpoint_spec_sha256 != activation.endpoint_spec_sha256
            or self.operation_digest != activation.operation_digest
            or self.endpoint_readback_sha256 != activation.endpoint_readback_sha256
        ):
            raise RunPodManagerError(
                "Serverless cleanup does not match its endpoint activation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "provider_cleanup_id": self.provider_cleanup_id,
            "provider_quote_id": self.provider_quote_id,
            "quote_sha256": self.quote_sha256,
            "provider_activation_id": self.provider_activation_id,
            "endpoint_name": self.endpoint_name,
            "endpoint_id": self.endpoint_id,
            "endpoint_spec_sha256": self.endpoint_spec_sha256,
            "operation_digest": self.operation_digest,
            "endpoint_readback_sha256": self.endpoint_readback_sha256,
            "workers_zero_observed_at": iso_datetime(self.workers_zero_observed_at),
            "delete_started_at": iso_datetime(self.delete_started_at),
            "verified_absent_at": iso_datetime(self.verified_absent_at),
            "reconciled_after_delete_error": self.reconciled_after_delete_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessEndpointCleanupReceipt:
        _exact_keys(value, _CLEANUP_RECEIPT_FIELDS, "Serverless cleanup receipt")
        flag = value.get("reconciled_after_delete_error")
        if not isinstance(flag, bool):
            raise RunPodManagerError(
                "Stored Serverless cleanup reconciliation flag is invalid"
            )
        return cls(
            schema_version=_required_int(value, "schema_version"),
            contract_version=_required_string(value, "contract_version"),
            provider_cleanup_id=_required_string(value, "provider_cleanup_id"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            quote_sha256=_required_string(value, "quote_sha256"),
            provider_activation_id=_required_string(value, "provider_activation_id"),
            endpoint_name=_required_string(value, "endpoint_name"),
            endpoint_id=_required_string(value, "endpoint_id"),
            endpoint_spec_sha256=_required_string(value, "endpoint_spec_sha256"),
            operation_digest=_required_string(value, "operation_digest"),
            endpoint_readback_sha256=_required_string(
                value, "endpoint_readback_sha256"
            ),
            workers_zero_observed_at=parse_datetime(
                _required_string(value, "workers_zero_observed_at")
            ),
            delete_started_at=parse_datetime(
                _required_string(value, "delete_started_at")
            ),
            verified_absent_at=parse_datetime(
                _required_string(value, "verified_absent_at")
            ),
            reconciled_after_delete_error=flag,
        )


@dataclass(frozen=True)
class ServerlessActivatedSubmission:
    """Provider eligibility token for one exclusive activated-endpoint window."""

    provider_submission_id: str
    provider_quote_id: str
    quote_sha256: str
    provider_activation_id: str
    endpoint_name: str
    endpoint_id: str
    endpoint_spec_sha256: str
    operation_digest: str
    exclusive_window_sha256: str
    authorized_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        for name, value in (
            ("provider_submission_id", self.provider_submission_id),
            ("provider_quote_id", self.provider_quote_id),
            ("provider_activation_id", self.provider_activation_id),
            ("endpoint_name", self.endpoint_name),
            ("endpoint_id", self.endpoint_id),
        ):
            _safe_identifier(value, name)
        for name, value in (
            ("quote_sha256", self.quote_sha256),
            ("endpoint_spec_sha256", self.endpoint_spec_sha256),
            ("operation_digest", self.operation_digest),
            ("exclusive_window_sha256", self.exclusive_window_sha256),
        ):
            _sha256(value, name)
        require_aware(self.authorized_at, "authorized_at")
        require_aware(self.expires_at, "expires_at")
        if self.authorized_at >= self.expires_at:
            raise ValueError("Serverless submission authorization is already expired")
        expected_id = "runpod-serverless-submission:" + json_sha256(
            {
                "provider_quote_id": self.provider_quote_id,
                "quote_sha256": self.quote_sha256,
                "provider_activation_id": self.provider_activation_id,
                "endpoint_name": self.endpoint_name,
                "endpoint_id": self.endpoint_id,
                "endpoint_spec_sha256": self.endpoint_spec_sha256,
                "operation_digest": self.operation_digest,
                "exclusive_window_sha256": self.exclusive_window_sha256,
                "authorized_at": iso_datetime(self.authorized_at),
                "expires_at": iso_datetime(self.expires_at),
            }
        )
        if self.provider_submission_id != expected_id:
            raise ValueError("Serverless submission identity digest is inconsistent")

    def validate(
        self,
        quote: PlannedServerlessCapacityQuote,
        activation: ServerlessEndpointActivationReceipt,
    ) -> None:
        activation.validate_quote(quote)
        if (
            self.provider_quote_id != quote.provider_quote_id
            or self.quote_sha256 != quote.quote_sha256
            or self.provider_activation_id != activation.provider_activation_id
            or self.endpoint_name != activation.endpoint_name
            or self.endpoint_id != activation.endpoint_id
            or self.endpoint_spec_sha256 != activation.endpoint_spec_sha256
            or self.operation_digest != activation.operation_digest
            or not quote.catalog_observed_at <= self.authorized_at < quote.expires_at
            or self.expires_at != quote.expires_at
        ):
            raise RunPodManagerError(
                "Serverless submission is not the activated child of its quote"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_submission_id": self.provider_submission_id,
            "provider_quote_id": self.provider_quote_id,
            "quote_sha256": self.quote_sha256,
            "provider_activation_id": self.provider_activation_id,
            "endpoint_name": self.endpoint_name,
            "endpoint_id": self.endpoint_id,
            "endpoint_spec_sha256": self.endpoint_spec_sha256,
            "operation_digest": self.operation_digest,
            "exclusive_window_sha256": self.exclusive_window_sha256,
            "authorized_at": iso_datetime(self.authorized_at),
            "expires_at": iso_datetime(self.expires_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessActivatedSubmission:
        _exact_keys(value, _ACTIVATED_SUBMISSION_FIELDS, "Serverless submission")
        return cls(
            provider_submission_id=_required_string(value, "provider_submission_id"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            quote_sha256=_required_string(value, "quote_sha256"),
            provider_activation_id=_required_string(value, "provider_activation_id"),
            endpoint_name=_required_string(value, "endpoint_name"),
            endpoint_id=_required_string(value, "endpoint_id"),
            endpoint_spec_sha256=_required_string(value, "endpoint_spec_sha256"),
            operation_digest=_required_string(value, "operation_digest"),
            exclusive_window_sha256=_required_string(value, "exclusive_window_sha256"),
            authorized_at=parse_datetime(_required_string(value, "authorized_at")),
            expires_at=parse_datetime(_required_string(value, "expires_at")),
        )


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
    job_execution_timeout_ms: int
    job_ttl_ms: int
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
            ("job_execution_timeout_ms", self.job_execution_timeout_ms),
            ("job_ttl_ms", self.job_ttl_ms),
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
        if not (
            _MIN_EXECUTION_TIMEOUT_MS
            <= self.job_execution_timeout_ms
            <= _MAX_SERVERLESS_POLICY_MS
        ):
            raise ValueError("Serverless job execution timeout is invalid")
        if not _MIN_JOB_TTL_MS <= self.job_ttl_ms <= _MAX_SERVERLESS_POLICY_MS:
            raise ValueError("Serverless job TTL is invalid")
        if self.maximum_execution_seconds * 1_000 > self.job_execution_timeout_ms:
            raise ValueError(
                "Serverless maximum execution exceeds the job execution timeout"
            )
        maximum_job_lifespan_ms = (
            self.maximum_queue_delay_seconds
            + self.maximum_worker_start_seconds
            + self.maximum_execution_seconds
        ) * 1_000
        if maximum_job_lifespan_ms > self.job_ttl_ms:
            raise ValueError("Serverless maximum job lifespan exceeds the job TTL")
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
            "job_execution_timeout_ms": self.job_execution_timeout_ms,
            "job_ttl_ms": self.job_ttl_ms,
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
            job_execution_timeout_ms=_required_int(value, "job_execution_timeout_ms"),
            job_ttl_ms=_required_int(value, "job_ttl_ms"),
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


def serverless_billing_hour_starts(
    attempt_started_at: datetime, billable_coverage_until: datetime
) -> tuple[datetime, ...]:
    """Return every UTC endpoint-hour bucket touched by a billable interval."""

    require_aware(attempt_started_at, "attempt_started_at")
    require_aware(billable_coverage_until, "billable_coverage_until")
    start = attempt_started_at.astimezone(UTC)
    coverage_until = billable_coverage_until.astimezone(UTC)
    if coverage_until <= start:
        raise ValueError("Serverless billable coverage interval is invalid")
    if (coverage_until - start).total_seconds() > _MAX_BILLABLE_COVERAGE_SECONDS:
        raise ValueError("Serverless billable coverage interval exceeds policy bounds")
    first = start.replace(minute=0, second=0, microsecond=0)
    window_until = coverage_until.replace(minute=0, second=0, microsecond=0)
    if coverage_until != window_until:
        window_until += timedelta(hours=1)
    hours: list[datetime] = []
    cursor = first
    while cursor < window_until:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    return tuple(hours)


@dataclass(frozen=True)
class ServerlessBillingAttempt:
    """Exact externally-owned attempt interval presented for settlement."""

    attempt_id: str
    job_id: str
    endpoint_id: str
    provider_quote_id: str
    exclusive_window_sha256: str
    exclusive_billing_hour_starts: tuple[datetime, ...]
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
        _normalized_exclusive_hours(self.exclusive_billing_hour_starts)
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
            quote.maximum_queue_delay_seconds
            + quote.maximum_worker_start_seconds
            + quote.maximum_execution_seconds
        )
        if (self.completed_at - self.submitted_at).total_seconds() > maximum_elapsed:
            raise RunPodManagerError(
                "Serverless attempt exceeds the accepted maximum interval"
            )
        coverage_until = self.completed_at + timedelta(seconds=quote.idle_tail_seconds)
        expected_hours = serverless_billing_hour_starts(
            self.submitted_at, coverage_until
        )
        actual_hours = tuple(
            value.astimezone(UTC) for value in self.exclusive_billing_hour_starts
        )
        if not _ordered_hours_cover(actual_hours, expected_hours):
            raise RunPodManagerError(
                "Serverless exclusive billing allocation does not cover every "
                "accepted endpoint-hour bucket"
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
    exclusive_billing_hour_starts: tuple[datetime, ...]
    attempt_started_at: datetime
    attempt_completed_at: datetime
    billable_coverage_until: datetime
    billing_window_from: datetime
    billing_window_until: datetime
    hourly_worker_rate_usd: Decimal
    pre_execution_delay_ms: int | None
    worker_startup_ms: int | None
    execution_ms: int | None
    accepted_idle_tail_ms: int
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
        if not isinstance(self.exclusive_billing_hour_starts, tuple):
            raise TypeError(
                "Serverless exclusive billing hour starts must be an immutable tuple"
            )
        for name, value in (
            ("attempt_started_at", self.attempt_started_at),
            ("attempt_completed_at", self.attempt_completed_at),
            ("billable_coverage_until", self.billable_coverage_until),
            ("billing_window_from", self.billing_window_from),
            ("billing_window_until", self.billing_window_until),
            ("reconciled_at", self.reconciled_at),
        ):
            require_aware(value, name)
        if not (
            self.billing_window_from
            <= self.attempt_started_at
            < self.attempt_completed_at
            < self.billable_coverage_until
            <= self.billing_window_until
            <= self.reconciled_at
        ):
            raise ValueError("Serverless billing receipt intervals are inconsistent")
        _positive_int(self.accepted_idle_tail_ms, "accepted_idle_tail_ms")
        expected_coverage_until = self.attempt_completed_at + timedelta(
            milliseconds=self.accepted_idle_tail_ms
        )
        if self.billable_coverage_until != expected_coverage_until:
            raise ValueError(
                "Serverless billing coverage does not equal the accepted idle tail"
            )
        required_hours = serverless_billing_hour_starts(
            self.attempt_started_at, self.billable_coverage_until
        )
        actual_hours = _normalized_exclusive_hours(self.exclusive_billing_hour_starts)
        if not _ordered_hours_cover(actual_hours, required_hours):
            raise ValueError(
                "Serverless receipt allocation does not cover every attempted "
                "endpoint-hour bucket"
            )
        if self.billing_window_from != actual_hours[
            0
        ] or self.billing_window_until != actual_hours[-1] + timedelta(hours=1):
            raise ValueError(
                "Serverless billing receipt window does not match its hour allocation"
            )
        _positive_decimal(self.hourly_worker_rate_usd, "hourly_worker_rate_usd")
        for name, value in (
            ("pre_execution_delay_ms", self.pre_execution_delay_ms),
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
            value for value in (self.pre_execution_delay_ms, self.execution_ms) if value
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
            "exclusive_billing_hour_starts": [
                iso_datetime(value) for value in self.exclusive_billing_hour_starts
            ],
            "attempt_started_at": iso_datetime(self.attempt_started_at),
            "attempt_completed_at": iso_datetime(self.attempt_completed_at),
            "billable_coverage_until": iso_datetime(self.billable_coverage_until),
            "billing_window_from": iso_datetime(self.billing_window_from),
            "billing_window_until": iso_datetime(self.billing_window_until),
            "hourly_worker_rate_usd": decimal_text(self.hourly_worker_rate_usd),
            "pre_execution_delay_ms": self.pre_execution_delay_ms,
            "worker_startup_ms": self.worker_startup_ms,
            "execution_ms": self.execution_ms,
            "accepted_idle_tail_ms": self.accepted_idle_tail_ms,
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
            exclusive_billing_hour_starts=_required_datetime_tuple(
                value, "exclusive_billing_hour_starts"
            ),
            attempt_started_at=parse_datetime(
                _required_string(value, "attempt_started_at")
            ),
            attempt_completed_at=parse_datetime(
                _required_string(value, "attempt_completed_at")
            ),
            billable_coverage_until=parse_datetime(
                _required_string(value, "billable_coverage_until")
            ),
            billing_window_from=parse_datetime(
                _required_string(value, "billing_window_from")
            ),
            billing_window_until=parse_datetime(
                _required_string(value, "billing_window_until")
            ),
            hourly_worker_rate_usd=_required_decimal(value, "hourly_worker_rate_usd"),
            pre_execution_delay_ms=_optional_int(
                value.get("pre_execution_delay_ms"), "pre_execution_delay_ms"
            ),
            worker_startup_ms=_optional_int(
                value.get("worker_startup_ms"), "worker_startup_ms"
            ),
            execution_ms=_optional_int(value.get("execution_ms"), "execution_ms"),
            accepted_idle_tail_ms=_required_int(value, "accepted_idle_tail_ms"),
            idle_tail_ms=_optional_int(value.get("idle_tail_ms"), "idle_tail_ms"),
            gpu_cost_usd=_required_decimal(value, "gpu_cost_usd"),
            cpu_cost_usd=_required_decimal(value, "cpu_cost_usd"),
            disk_cost_usd=_required_decimal(value, "disk_cost_usd"),
            fee_cost_usd=_required_decimal(value, "fee_cost_usd"),
            actual_cost_usd=_required_decimal(value, "actual_cost_usd"),
            reconciled_at=parse_datetime(_required_string(value, "reconciled_at")),
        )


@dataclass(frozen=True)
class ServerlessAmbiguousBillingWindow:
    """Exclusive worst-case endpoint window for an ambiguous submission."""

    attempt_id: str
    endpoint_id: str
    provider_quote_id: str
    exclusive_window_sha256: str
    exclusive_billing_hour_starts: tuple[datetime, ...]
    attempted_at: datetime
    billable_coverage_until: datetime
    accepted_cost_ceiling_usd: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("endpoint_id", self.endpoint_id),
            ("provider_quote_id", self.provider_quote_id),
        ):
            _safe_identifier(value, name)
        _sha256(self.exclusive_window_sha256, "exclusive_window_sha256")
        require_aware(self.attempted_at, "attempted_at")
        require_aware(self.billable_coverage_until, "billable_coverage_until")
        required_hours = serverless_billing_hour_starts(
            self.attempted_at, self.billable_coverage_until
        )
        actual_hours = _normalized_exclusive_hours(self.exclusive_billing_hour_starts)
        if not _ordered_hours_cover(actual_hours, required_hours):
            raise ValueError(
                "Serverless ambiguous allocation does not cover every attempted "
                "endpoint-hour bucket"
            )
        _positive_decimal(self.accepted_cost_ceiling_usd, "accepted_cost_ceiling_usd")

    def validate_quote(self, quote: ServerlessCapacityQuote) -> None:
        """Bind the ambiguous mutation window to its exact accepted quote."""

        if self.endpoint_id != quote.endpoint_id:
            raise RunPodManagerError(
                "Serverless ambiguous billing endpoint does not match quote"
            )
        if self.provider_quote_id != quote.provider_quote_id:
            raise RunPodManagerError(
                "Serverless ambiguous billing quote identity does not match"
            )
        if self.accepted_cost_ceiling_usd != quote.cost_ceiling_usd:
            raise RunPodManagerError(
                "Serverless ambiguous billing ceiling does not match quote"
            )
        if not quote.catalog_observed_at <= self.attempted_at < quote.expires_at:
            raise RunPodManagerError(
                "Serverless ambiguous submission falls outside the accepted quote"
            )
        expected_coverage_until = self.attempted_at + timedelta(
            seconds=(
                quote.maximum_queue_delay_seconds
                + quote.maximum_worker_start_seconds
                + quote.maximum_execution_seconds
                + quote.idle_tail_seconds
            )
        )
        if self.billable_coverage_until != expected_coverage_until:
            raise RunPodManagerError(
                "Serverless ambiguous allocation does not cover the accepted "
                "worst-case billable interval"
            )

    def validate_activated_submission(
        self,
        quote: PlannedServerlessCapacityQuote,
        activation: ServerlessEndpointActivationReceipt,
        submission: ServerlessActivatedSubmission,
    ) -> None:
        """Bind this window to one exact planned endpoint submission."""

        submission.validate(quote, activation)
        if (
            self.endpoint_id != activation.endpoint_id
            or self.provider_quote_id != quote.provider_quote_id
            or self.exclusive_window_sha256 != submission.exclusive_window_sha256
        ):
            raise RunPodManagerError(
                "Serverless ambiguous billing window is not bound to the "
                "activated submission"
            )
        if self.accepted_cost_ceiling_usd != quote.cost_ceiling_usd:
            raise RunPodManagerError(
                "Serverless ambiguous billing ceiling does not match planned quote"
            )
        if not submission.authorized_at <= self.attempted_at < submission.expires_at:
            raise RunPodManagerError(
                "Serverless ambiguous submission falls outside its authorization"
            )
        expected_coverage_until = self.attempted_at + timedelta(
            seconds=(
                quote.maximum_queue_delay_seconds
                + quote.maximum_worker_start_seconds
                + quote.maximum_execution_seconds
                + quote.idle_tail_seconds
            )
        )
        if self.billable_coverage_until != expected_coverage_until:
            raise RunPodManagerError(
                "Serverless ambiguous allocation does not cover the accepted "
                "planned worst-case billable interval"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "endpoint_id": self.endpoint_id,
            "provider_quote_id": self.provider_quote_id,
            "exclusive_window_sha256": self.exclusive_window_sha256,
            "exclusive_billing_hour_starts": [
                iso_datetime(value) for value in self.exclusive_billing_hour_starts
            ],
            "attempted_at": iso_datetime(self.attempted_at),
            "billable_coverage_until": iso_datetime(self.billable_coverage_until),
            "accepted_cost_ceiling_usd": decimal_text(self.accepted_cost_ceiling_usd),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessAmbiguousBillingWindow:
        _exact_keys(
            value,
            _AMBIGUOUS_WINDOW_FIELDS,
            "Serverless ambiguous billing window",
        )
        return cls(
            attempt_id=_required_string(value, "attempt_id"),
            endpoint_id=_required_string(value, "endpoint_id"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            exclusive_window_sha256=_required_string(value, "exclusive_window_sha256"),
            exclusive_billing_hour_starts=_required_datetime_tuple(
                value, "exclusive_billing_hour_starts"
            ),
            attempted_at=parse_datetime(_required_string(value, "attempted_at")),
            billable_coverage_until=parse_datetime(
                _required_string(value, "billable_coverage_until")
            ),
            accepted_cost_ceiling_usd=_required_decimal(
                value, "accepted_cost_ceiling_usd"
            ),
        )


@dataclass(frozen=True)
class ServerlessEndpointHourCost:
    """One canonical content-free Runpod v2 endpoint-hour observation."""

    provider_observation_id: str
    endpoint_id: str
    utc_hour_start: datetime
    utc_hour_end: datetime
    gpu_cost_usd: Decimal
    cpu_cost_usd: Decimal
    disk_cost_usd: Decimal
    fee_cost_usd: Decimal
    actual_cost_usd: Decimal

    def __post_init__(self) -> None:
        _safe_identifier(self.provider_observation_id, "provider_observation_id")
        _safe_identifier(self.endpoint_id, "endpoint_id")
        require_aware(self.utc_hour_start, "utc_hour_start")
        require_aware(self.utc_hour_end, "utc_hour_end")
        start = self.utc_hour_start.astimezone(UTC)
        end = self.utc_hour_end.astimezone(UTC)
        if (
            start.minute != 0
            or start.second != 0
            or start.microsecond != 0
            or end != start + timedelta(hours=1)
        ):
            raise ValueError(
                "Serverless endpoint-hour cost must cover exactly one UTC hour"
            )
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
            raise ValueError(
                "Serverless endpoint-hour cost components do not equal total"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_observation_id": self.provider_observation_id,
            "endpoint_id": self.endpoint_id,
            "utc_hour_start": iso_datetime(self.utc_hour_start),
            "utc_hour_end": iso_datetime(self.utc_hour_end),
            "gpu_cost_usd": decimal_text(self.gpu_cost_usd),
            "cpu_cost_usd": decimal_text(self.cpu_cost_usd),
            "disk_cost_usd": decimal_text(self.disk_cost_usd),
            "fee_cost_usd": decimal_text(self.fee_cost_usd),
            "actual_cost_usd": decimal_text(self.actual_cost_usd),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServerlessEndpointHourCost:
        _exact_keys(
            value,
            _ENDPOINT_HOUR_COST_FIELDS,
            "Serverless endpoint-hour cost",
        )
        return cls(
            provider_observation_id=_required_string(value, "provider_observation_id"),
            endpoint_id=_required_string(value, "endpoint_id"),
            utc_hour_start=parse_datetime(_required_string(value, "utc_hour_start")),
            utc_hour_end=parse_datetime(_required_string(value, "utc_hour_end")),
            gpu_cost_usd=_required_decimal(value, "gpu_cost_usd"),
            cpu_cost_usd=_required_decimal(value, "cpu_cost_usd"),
            disk_cost_usd=_required_decimal(value, "disk_cost_usd"),
            fee_cost_usd=_required_decimal(value, "fee_cost_usd"),
            actual_cost_usd=_required_decimal(value, "actual_cost_usd"),
        )


@dataclass(frozen=True)
class ServerlessAmbiguousWindowBillingReceipt:
    """Authoritative endpoint-hour cost for an ambiguous submission window."""

    schema_version: int
    contract_version: str
    provider_billing_id: str
    provider_quote_id: str
    endpoint_profile_sha256: str
    endpoint_id: str
    attempt_id: str
    exclusive_window_sha256: str
    exclusive_billing_hour_starts: tuple[datetime, ...]
    attempted_at: datetime
    billable_coverage_until: datetime
    billing_window_from: datetime
    billing_window_until: datetime
    accepted_cost_ceiling_usd: Decimal
    endpoint_hour_costs: tuple[ServerlessEndpointHourCost, ...]
    gpu_cost_usd: Decimal
    cpu_cost_usd: Decimal
    disk_cost_usd: Decimal
    fee_cost_usd: Decimal
    actual_cost_usd: Decimal
    capped_cost_usd: Decimal
    operator_loss_usd: Decimal
    reconciled_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != SERVERLESS_CAPACITY_SCHEMA_VERSION:
            raise ValueError(
                "Serverless ambiguous billing receipt schema is unsupported"
            )
        if self.contract_version != SERVERLESS_CAPACITY_CONTRACT_VERSION:
            raise ValueError(
                "Serverless ambiguous billing receipt contract is unsupported"
            )
        for name, value in (
            ("provider_billing_id", self.provider_billing_id),
            ("provider_quote_id", self.provider_quote_id),
            ("endpoint_id", self.endpoint_id),
            ("attempt_id", self.attempt_id),
        ):
            _safe_identifier(value, name)
        _sha256(self.endpoint_profile_sha256, "endpoint_profile_sha256")
        _sha256(self.exclusive_window_sha256, "exclusive_window_sha256")
        for name, value in (
            ("attempted_at", self.attempted_at),
            ("billable_coverage_until", self.billable_coverage_until),
            ("billing_window_from", self.billing_window_from),
            ("billing_window_until", self.billing_window_until),
            ("reconciled_at", self.reconciled_at),
        ):
            require_aware(value, name)
        if not (
            self.billing_window_from
            <= self.attempted_at
            < self.billable_coverage_until
            <= self.billing_window_until
            <= self.reconciled_at
        ):
            raise ValueError(
                "Serverless ambiguous billing receipt intervals are inconsistent"
            )
        required_hours = serverless_billing_hour_starts(
            self.attempted_at, self.billable_coverage_until
        )
        actual_hours = _normalized_exclusive_hours(self.exclusive_billing_hour_starts)
        if not _ordered_hours_cover(actual_hours, required_hours):
            raise ValueError(
                "Serverless ambiguous receipt allocation does not cover every "
                "attempted endpoint-hour bucket"
            )
        if self.billing_window_from != actual_hours[
            0
        ] or self.billing_window_until != actual_hours[-1] + timedelta(hours=1):
            raise ValueError(
                "Serverless ambiguous receipt window does not match its allocation"
            )
        if (
            not isinstance(self.endpoint_hour_costs, tuple)
            or not self.endpoint_hour_costs
        ):
            raise ValueError(
                "Serverless ambiguous receipt must contain endpoint-hour costs"
            )
        observed_hours: list[datetime] = []
        observed_ids: set[str] = set()
        for item in self.endpoint_hour_costs:
            if not isinstance(item, ServerlessEndpointHourCost):
                raise TypeError("Serverless endpoint-hour cost has an invalid type")
            if item.endpoint_id != self.endpoint_id:
                raise ValueError(
                    "Serverless endpoint-hour cost does not match receipt endpoint"
                )
            if item.provider_observation_id in observed_ids:
                raise ValueError(
                    "Serverless endpoint-hour provider observation ID is duplicated"
                )
            observed_hours.append(item.utc_hour_start.astimezone(UTC))
            observed_ids.add(item.provider_observation_id)
        if tuple(observed_hours) != actual_hours:
            raise ValueError(
                "Serverless endpoint-hour costs do not match the ordered allocation"
            )
        _positive_decimal(self.accepted_cost_ceiling_usd, "accepted_cost_ceiling_usd")
        for name, value in (
            ("gpu_cost_usd", self.gpu_cost_usd),
            ("cpu_cost_usd", self.cpu_cost_usd),
            ("disk_cost_usd", self.disk_cost_usd),
            ("fee_cost_usd", self.fee_cost_usd),
            ("actual_cost_usd", self.actual_cost_usd),
            ("capped_cost_usd", self.capped_cost_usd),
            ("operator_loss_usd", self.operator_loss_usd),
        ):
            _nonnegative_decimal(value, name)
        if self.actual_cost_usd != (
            self.gpu_cost_usd
            + self.cpu_cost_usd
            + self.disk_cost_usd
            + self.fee_cost_usd
        ):
            raise ValueError(
                "Serverless ambiguous billing components do not equal total cost"
            )
        derived_amounts = _endpoint_hour_cost_totals(self.endpoint_hour_costs)
        if any(
            getattr(self, name) != derived_amounts[name]
            for name in _SERVERLESS_COST_FIELD_NAMES
        ):
            raise ValueError(
                "Serverless ambiguous aggregate costs do not equal endpoint-hour costs"
            )
        if self.capped_cost_usd != min(
            self.actual_cost_usd, self.accepted_cost_ceiling_usd
        ):
            raise ValueError("Serverless ambiguous capped cost is inconsistent")
        if self.operator_loss_usd != self.actual_cost_usd - self.capped_cost_usd:
            raise ValueError("Serverless ambiguous operator loss is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "provider_billing_id": self.provider_billing_id,
            "provider_quote_id": self.provider_quote_id,
            "endpoint_profile_sha256": self.endpoint_profile_sha256,
            "endpoint_id": self.endpoint_id,
            "attempt_id": self.attempt_id,
            "exclusive_window_sha256": self.exclusive_window_sha256,
            "exclusive_billing_hour_starts": [
                iso_datetime(value) for value in self.exclusive_billing_hour_starts
            ],
            "attempted_at": iso_datetime(self.attempted_at),
            "billable_coverage_until": iso_datetime(self.billable_coverage_until),
            "billing_window_from": iso_datetime(self.billing_window_from),
            "billing_window_until": iso_datetime(self.billing_window_until),
            "accepted_cost_ceiling_usd": decimal_text(self.accepted_cost_ceiling_usd),
            "endpoint_hour_costs": [
                item.to_dict() for item in self.endpoint_hour_costs
            ],
            "gpu_cost_usd": decimal_text(self.gpu_cost_usd),
            "cpu_cost_usd": decimal_text(self.cpu_cost_usd),
            "disk_cost_usd": decimal_text(self.disk_cost_usd),
            "fee_cost_usd": decimal_text(self.fee_cost_usd),
            "actual_cost_usd": decimal_text(self.actual_cost_usd),
            "capped_cost_usd": decimal_text(self.capped_cost_usd),
            "operator_loss_usd": decimal_text(self.operator_loss_usd),
            "reconciled_at": iso_datetime(self.reconciled_at),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> ServerlessAmbiguousWindowBillingReceipt:
        _exact_keys(
            value,
            _AMBIGUOUS_RECEIPT_FIELDS,
            "Serverless ambiguous billing receipt",
        )
        return cls(
            schema_version=_required_int(value, "schema_version"),
            contract_version=_required_string(value, "contract_version"),
            provider_billing_id=_required_string(value, "provider_billing_id"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            endpoint_profile_sha256=_required_string(value, "endpoint_profile_sha256"),
            endpoint_id=_required_string(value, "endpoint_id"),
            attempt_id=_required_string(value, "attempt_id"),
            exclusive_window_sha256=_required_string(value, "exclusive_window_sha256"),
            exclusive_billing_hour_starts=_required_datetime_tuple(
                value, "exclusive_billing_hour_starts"
            ),
            attempted_at=parse_datetime(_required_string(value, "attempted_at")),
            billable_coverage_until=parse_datetime(
                _required_string(value, "billable_coverage_until")
            ),
            billing_window_from=parse_datetime(
                _required_string(value, "billing_window_from")
            ),
            billing_window_until=parse_datetime(
                _required_string(value, "billing_window_until")
            ),
            accepted_cost_ceiling_usd=_required_decimal(
                value, "accepted_cost_ceiling_usd"
            ),
            endpoint_hour_costs=_required_endpoint_hour_costs(
                value, "endpoint_hour_costs"
            ),
            gpu_cost_usd=_required_decimal(value, "gpu_cost_usd"),
            cpu_cost_usd=_required_decimal(value, "cpu_cost_usd"),
            disk_cost_usd=_required_decimal(value, "disk_cost_usd"),
            fee_cost_usd=_required_decimal(value, "fee_cost_usd"),
            actual_cost_usd=_required_decimal(value, "actual_cost_usd"),
            capped_cost_usd=_required_decimal(value, "capped_cost_usd"),
            operator_loss_usd=_required_decimal(value, "operator_loss_usd"),
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


def _identifiers(values: tuple[str, ...], name: str, *, reason: str) -> None:
    if not isinstance(values, tuple) or not values or len(values) != len(set(values)):
        raise ValueError(f"Serverless capacity {name}: {reason}")
    for value in values:
        _safe_identifier(value, name)


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


def _required_datetime_tuple(
    value: Mapping[str, Any], key: str
) -> tuple[datetime, ...]:
    items = value.get(key)
    if not isinstance(items, list) or not items:
        raise RunPodManagerError(
            f"Stored Serverless {key} must be a nonempty timestamp list"
        )
    parsed: list[datetime] = []
    for item in items:
        if not isinstance(item, str):
            raise RunPodManagerError(
                f"Stored Serverless {key} must contain timestamp strings"
            )
        parsed.append(parse_datetime(item))
    return tuple(parsed)


def _normalized_exclusive_hours(values: tuple[datetime, ...]) -> tuple[datetime, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(
            "Serverless exclusive billing allocation must contain hour buckets"
        )
    normalized: list[datetime] = []
    if len(values) > _MAX_EXCLUSIVE_BILLING_HOURS:
        raise ValueError(
            "Serverless exclusive billing allocation exceeds policy bounds"
        )
    previous: datetime | None = None
    for value in values:
        require_aware(value, "exclusive_billing_hour_starts")
        item = value.astimezone(UTC)
        if (
            item.minute != 0
            or item.second != 0
            or item.microsecond != 0
            or (previous is not None and item != previous + timedelta(hours=1))
        ):
            raise ValueError(
                "Serverless exclusive billing allocation must contain contiguous "
                "UTC hour starts"
            )
        normalized.append(item)
        previous = item
    return tuple(normalized)


def _ordered_hours_cover(
    actual_hours: tuple[datetime, ...], required_hours: tuple[datetime, ...]
) -> bool:
    return (
        bool(actual_hours)
        and bool(required_hours)
        and actual_hours[0] <= required_hours[0]
        and actual_hours[-1] >= required_hours[-1]
    )


def _endpoint_hour_cost_totals(
    values: tuple[ServerlessEndpointHourCost, ...],
) -> dict[str, Decimal]:
    return {
        name: sum((getattr(item, name) for item in values), start=Decimal(0))
        for name in _SERVERLESS_COST_FIELD_NAMES
    }


def _required_endpoint_hour_costs(
    value: Mapping[str, Any], key: str
) -> tuple[ServerlessEndpointHourCost, ...]:
    items = value.get(key)
    if not isinstance(items, list) or not items:
        raise RunPodManagerError(
            f"Stored Serverless {key} must be a nonempty object list"
        )
    parsed: list[ServerlessEndpointHourCost] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise RunPodManagerError(f"Stored Serverless {key} must contain objects")
        parsed.append(ServerlessEndpointHourCost.from_dict(item))
    return tuple(parsed)


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
        "job_execution_timeout_ms",
        "job_ttl_ms",
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

_CONSTRAINT_FIELDS = frozenset(
    {
        "min_vram_gb",
        "gpu_count",
        "cloud",
        "min_cuda_version",
        "allowed_gpu_pools",
        "allowed_data_center_ids",
        "max_hourly_worker_rate_usd",
        "minimum_availability",
        "benchmark_id",
    }
)

_ENDPOINT_SPEC_FIELDS = frozenset(
    {
        "worker_reference",
        "constraints",
        "workers_min",
        "workers_max",
        "idle_tail_seconds",
        "scaling_type",
        "scaling_value",
        "execution_timeout_ms",
        "flashboot",
        "disk_gb",
        "registry_id",
        "network_volume_ids",
        "spec_sha256",
    }
)

_PLANNED_ENDPOINT_FIELDS = frozenset(
    {"plan_id", "endpoint_name", "spec", "plan_sha256"}
)

_PLANNED_QUOTE_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "provider_quote_id",
        "quote_sha256",
        "workload_kind",
        "parameters_sha256",
        "plan_id",
        "plan_sha256",
        "endpoint_name",
        "endpoint_spec_sha256",
        "operation_digest",
        "worker_reference",
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
        "job_execution_timeout_ms",
        "job_ttl_ms",
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

_ACTIVATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "provider_activation_id",
        "provider_quote_id",
        "quote_sha256",
        "plan_id",
        "plan_sha256",
        "endpoint_name",
        "endpoint_id",
        "endpoint_spec_sha256",
        "operation_digest",
        "endpoint_readback_sha256",
        "create_started_at",
        "activated_at",
        "readback_at",
        "reconciled_after_ambiguous_create",
    }
)

_CLEANUP_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "provider_cleanup_id",
        "provider_quote_id",
        "quote_sha256",
        "provider_activation_id",
        "endpoint_name",
        "endpoint_id",
        "endpoint_spec_sha256",
        "operation_digest",
        "endpoint_readback_sha256",
        "workers_zero_observed_at",
        "delete_started_at",
        "verified_absent_at",
        "reconciled_after_delete_error",
    }
)

_ACTIVATED_SUBMISSION_FIELDS = frozenset(
    {
        "provider_submission_id",
        "provider_quote_id",
        "quote_sha256",
        "provider_activation_id",
        "endpoint_name",
        "endpoint_id",
        "endpoint_spec_sha256",
        "operation_digest",
        "exclusive_window_sha256",
        "authorized_at",
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
        "exclusive_billing_hour_starts",
        "attempt_started_at",
        "attempt_completed_at",
        "billable_coverage_until",
        "billing_window_from",
        "billing_window_until",
        "hourly_worker_rate_usd",
        "pre_execution_delay_ms",
        "worker_startup_ms",
        "execution_ms",
        "accepted_idle_tail_ms",
        "idle_tail_ms",
        "gpu_cost_usd",
        "cpu_cost_usd",
        "disk_cost_usd",
        "fee_cost_usd",
        "actual_cost_usd",
        "reconciled_at",
    }
)

_AMBIGUOUS_WINDOW_FIELDS = frozenset(
    {
        "attempt_id",
        "endpoint_id",
        "provider_quote_id",
        "exclusive_window_sha256",
        "exclusive_billing_hour_starts",
        "attempted_at",
        "billable_coverage_until",
        "accepted_cost_ceiling_usd",
    }
)

_ENDPOINT_HOUR_COST_FIELDS = frozenset(
    {
        "provider_observation_id",
        "endpoint_id",
        "utc_hour_start",
        "utc_hour_end",
        "gpu_cost_usd",
        "cpu_cost_usd",
        "disk_cost_usd",
        "fee_cost_usd",
        "actual_cost_usd",
    }
)

_SERVERLESS_COST_FIELD_NAMES = (
    "gpu_cost_usd",
    "cpu_cost_usd",
    "disk_cost_usd",
    "fee_cost_usd",
    "actual_cost_usd",
)

_AMBIGUOUS_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "provider_billing_id",
        "provider_quote_id",
        "endpoint_profile_sha256",
        "endpoint_id",
        "attempt_id",
        "exclusive_window_sha256",
        "exclusive_billing_hour_starts",
        "attempted_at",
        "billable_coverage_until",
        "billing_window_from",
        "billing_window_until",
        "accepted_cost_ceiling_usd",
        "endpoint_hour_costs",
        "gpu_cost_usd",
        "cpu_cost_usd",
        "disk_cost_usd",
        "fee_cost_usd",
        "actual_cost_usd",
        "capped_cost_usd",
        "operator_loss_usd",
        "reconciled_at",
    }
)


__all__ = [
    "RUNPOD_V2_OPENAPI_SHA256",
    "RUNPOD_V2_OPENAPI_SOURCE_URL",
    "SERVERLESS_CAPACITY_CONTRACT_VERSION",
    "SERVERLESS_CAPACITY_SCHEMA_VERSION",
    "PlannedServerlessCapacityQuote",
    "PlannedServerlessCapacityQuoteRequest",
    "PlannedServerlessEndpoint",
    "ServerlessActivatedSubmission",
    "ServerlessAmbiguousBillingWindow",
    "ServerlessAmbiguousWindowBillingReceipt",
    "ServerlessBillingAttempt",
    "ServerlessBillingReceipt",
    "ServerlessCapacityConstraints",
    "ServerlessCapacityQuote",
    "ServerlessCapacityQuoteRequest",
    "ServerlessEndpointActivationReceipt",
    "ServerlessEndpointCleanupReceipt",
    "ServerlessEndpointHourCost",
    "ServerlessEndpointProfile",
    "ServerlessEndpointSpec",
    "decimal_text",
    "iso_datetime",
    "json_sha256",
    "parse_datetime",
    "require_aware",
    "serverless_billing_hour_starts",
    "serverless_worker_cost_usd",
]
