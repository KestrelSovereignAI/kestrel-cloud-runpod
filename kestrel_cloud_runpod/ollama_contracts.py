"""Typed contracts and cost policy for durable private-Ollama leases."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from typing import Any, Protocol

from .billing import (
    billing_exposure_ratio_float,
    billing_exposure_sum_float,
    ceil_timedelta_seconds,
    conservative_decimal_ratio_float,
)
from .models import (
    CloudType,
    ComputeProduct,
    EndpointCreateRequest,
    PlacementDecision,
    PlacementRequirements,
    PodCreateRequest,
    RunPodManagerError,
)
from .pod_capacity_contracts import (
    PodBillingReceipt,
    PodRealizedPlacement,
    decimal_text,
)
from .serverless_capacity_contracts import (
    ServerlessEndpointHourCost,
    serverless_endpoint_hour_cost_totals,
)

_URL_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization)(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


def _positive_public_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and (not isinstance(value, float) or math.isfinite(value))
    )


_INTENT_KEYS = {
    "evidence_version",
    "lease_id",
    "request_fingerprint",
    "provision_attempt_id",
    "mode",
    "resource_type",
    "resource_name",
    "provision_requested_at",
    "environment_sha256",
    "environment_keys",
    "provider_request",
    "provider_request_sha256",
    "selected_gpu_id",
    "selected_gpu_pool",
    "selected_gpu_name",
    "selected_gpu_count",
    "selected_cloud",
    "allowed_data_center_ids",
    "accepted_whole_hourly_rate_usd",
}
_RECEIPT_KEYS = {
    "evidence_version",
    "receipt_id",
    "intent_sha256",
    "lease_id",
    "request_fingerprint",
    "provision_attempt_id",
    "resource_type",
    "provider_resource_id",
    "resource_name",
    "provider_created_at",
    "readback_at",
    "readback_sha256",
    "environment_sha256",
    "reconciled_after_ambiguous_create",
    "pod_realized_placement",
    "provider_rate_precision",
}
_MISMATCH_KEYS = {
    "evidence_version",
    "evidence_id",
    "intent_sha256",
    "lease_id",
    "provision_attempt_id",
    "resource_type",
    "provider_resource_id",
    "observed_resource_name",
    "observed_at",
    "readback_sha256",
    "reason",
}
_CLEANUP_INTENT_KEYS = {
    "evidence_version",
    "cleanup_intent_id",
    "intent_sha256",
    "provisioning_receipt_id",
    "lease_id",
    "request_fingerprint",
    "provision_attempt_id",
    "resource_type",
    "provider_resource_id",
    "resource_name",
    "termination_reason",
    "delete_started_at",
}
_CLEANUP_RECEIPT_KEYS = {
    "evidence_version",
    "cleanup_receipt_id",
    "cleanup_intent_id",
    "verified_absent_at",
    "basis",
}
_ADDITIONAL_CLEANUP_TARGET_KEYS = {
    "evidence_version",
    "target_id",
    "intent_sha256",
    "lease_id",
    "provision_attempt_id",
    "resource_type",
    "resource_name",
    "provider_resource_id",
    "source",
    "provisioning_mismatch",
    "cleanup_intent",
    "cleanup_receipt",
}
_TELEMETRY_SNAPSHOT_KEYS = {
    "schema_version",
    "runtime_version",
    "ollama_version",
    "telemetry_binding_sha256",
    "runtime_instance_id",
    "telemetry_revision",
    "provision_requested_at",
    "container_started_at",
    "ollama_process_started_at",
    "ollama_healthy_at",
    "model_pull_started_at",
    "model_pull_completed_at",
    "model_load_started_at",
    "model_ready_at",
    "ready_at",
    "placement_and_image_pull_wall_ms",
    "ollama_boot_ms",
    "model_pull_ms",
    "model_load_ms",
    "container_to_ready_ms",
    "cache_hit",
    "startup_strategy",
    "ready",
    "ollama_process_running",
    "terminal_error",
    "inference_started",
    "inference_completed",
    "inference_failed",
    "inference_cancelled",
    "inference_in_flight",
    "first_inference_started_at",
    "last_inference_started_at",
    "last_inference_completed_at",
    "inference_busy_ms",
    "inference_idle_ms",
}
_TELEMETRY_OBSERVATION_KEYS = {
    "evidence_version",
    "observation_id",
    "intent_sha256",
    "provisioning_receipt_id",
    "lease_id",
    "provision_attempt_id",
    "resource_type",
    "provider_resource_id",
    "snapshot",
    "snapshot_sha256",
    "fetch_started_at",
    "observed_at",
    "authentication_basis",
}
_BILLING_OBSERVATION_KEYS = {
    "evidence_version",
    "observation_id",
    "intent_sha256",
    "provisioning_receipt_id",
    "cleanup_receipt_id",
    "lease_id",
    "provision_attempt_id",
    "resource_type",
    "provider_resource_id",
    "cost_scope",
    "provider_finality_confirmed",
    "provider_capacity_actual_cost_usd",
    "first_observed_at",
    "pod_billing_receipt",
    "serverless_endpoint_hour_costs",
}


class OllamaLeaseMode(str, Enum):
    """Interactive execution modes; queue Serverless is deliberately absent."""

    AUTO = "auto"
    SERVERLESS_LOAD_BALANCER = "serverless_load_balancer"
    DEDICATED_POD = "dedicated_pod"


class OllamaResourceType(str, Enum):
    SERVERLESS_ENDPOINT = "serverless_endpoint"
    POD = "pod"


class OllamaLeaseState(str, Enum):
    REQUESTED = "requested"
    PROVISIONING = "provisioning"
    RECONCILE_REQUIRED = "reconcile_required"
    WAITING_FOR_MODEL = "waiting_for_model"
    READY = "ready"
    RELEASING = "releasing"
    TERMINATED = "terminated"
    FAILED = "failed"


class OllamaTeardownState(str, Enum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    RETRYABLE_FAILURE = "retryable_failure"
    COMPLETE = "complete"


class OllamaProvisioningEvidenceState(str, Enum):
    """Durable authority available for the original provider create."""

    REQUIRED = "required"
    INTENT_RECORDED = "intent_recorded"
    COMPLETE = "complete"
    HISTORICAL_UNKNOWN = "historical_unknown"


class OllamaProviderRatePrecision(str, Enum):
    """Be explicit that the SDK does not retain the provider JSON lexeme."""

    PARSED_JSON_NUMBER = "parsed_json_number"


class OllamaCleanupBasis(str, Enum):
    DELETE_ACKNOWLEDGED = "delete_acknowledged"
    AMBIGUOUS_DELETE_VERIFIED_ABSENT = "ambiguous_delete_verified_absent"
    INTERRUPTED_INTENT_VERIFIED_ABSENT = "interrupted_intent_verified_absent"


class OllamaRuntimeTelemetryState(str, Enum):
    UNKNOWN = "unknown"
    STARTUP_COMPLETE = "startup_complete"
    PARTIAL = "partial"
    COMPLETE = "complete"


class OllamaLifecycleBillingState(str, Enum):
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    SELECTED = "selected"
    CORRECTION_PENDING = "correction_pending"


class OllamaActualCostCapStatus(str, Enum):
    UNKNOWN = "unknown"
    EXCEEDED = "exceeded"


class OllamaProvisioningMismatchReason(str, Enum):
    """Confirmed incompatibilities that require immediate fenced cleanup."""

    RESOURCE_IDENTITY = "resource_identity"
    IMAGE_OR_ENVIRONMENT = "image_or_environment"
    ENDPOINT_POLICY = "endpoint_policy"
    POD_NOT_LIVE = "pod_not_live"
    POD_PLACEMENT_OR_RATE = "pod_placement_or_rate"
    RUNTIME = "runtime"


class OllamaNonComputeCostComponent(str, Enum):
    """Operator-priced billable exposure outside GPU worker execution."""

    CONTAINER_DISK = "container_disk"
    NETWORK_VOLUME = "network_volume"
    MODEL_TRANSFER = "model_transfer"
    RETRY_ALLOWANCE = "retry_allowance"


class OllamaLeaseConflictError(RunPodManagerError):
    """A stable lease ID was reused for a different request."""


class OllamaLeaseAuthorizationError(RunPodManagerError):
    """The supplied owner/workload pair does not own the lease."""


class OllamaLeaseReadinessError(RunPodManagerError):
    """Capacity did not become model-ready within its bounded lease policy."""


class OllamaLeaseTeardownError(RunPodManagerError):
    """Provider teardown failed and remains durable/retryable."""


class OllamaProvisioningEvidenceError(RunPodManagerError):
    """The exact created resource cannot yet prove its original request."""


class OllamaProvisioningMismatchError(OllamaProvisioningEvidenceError):
    """A known resource is confirmed incompatible with its original intent."""

    def __init__(
        self,
        detail: str,
        evidence: OllamaProvisioningMismatchEvidence,
    ) -> None:
        super().__init__(detail)
        self.evidence = evidence


@dataclass(frozen=True)
class OllamaNonComputeCostPolicy:
    """Conservative deployment-owned non-compute authorization policy.

    Values are explicit per-session amounts, not live provider rates and not
    observed billing.  Operators derive them from the deployment's container
    disk, optional network volume, model transfer/egress, and retry exposure.
    """

    estimated_cost_usd: float
    maximum_cost_usd: float
    covered_components: tuple[OllamaNonComputeCostComponent, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("estimated_cost_usd", self.estimated_cost_usd),
            ("maximum_cost_usd", self.maximum_cost_usd),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Ollama {name} must be finite and nonnegative")
        if self.maximum_cost_usd < self.estimated_cost_usd:
            raise ValueError("Ollama maximum non-compute cost must cover its estimate")
        if (
            not isinstance(self.covered_components, tuple)
            or not self.covered_components
            or any(
                not isinstance(item, OllamaNonComputeCostComponent)
                for item in self.covered_components
            )
            or len(set(self.covered_components)) != len(self.covered_components)
        ):
            raise ValueError(
                "Ollama non-compute cost components must be a unique nonempty tuple"
            )
        if tuple(sorted(self.covered_components, key=lambda item: item.value)) != (
            self.covered_components
        ):
            raise ValueError("Ollama non-compute cost components must be sorted")


@dataclass(frozen=True)
class OllamaResourceConstraints:
    """Provider-neutral placement inputs used by both Runpod products."""

    min_vram_gb: int
    gpu_count: int = 1
    cloud: CloudType = CloudType.SECURE
    min_cuda_version: str | None = None
    allowed_gpu_ids: tuple[str, ...] = ()
    allowed_gpu_pools: tuple[str, ...] = ()
    allowed_data_center_ids: tuple[str, ...] = ()
    max_hourly_rate: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_vram_gb, int)
            or isinstance(self.min_vram_gb, bool)
            or not isinstance(self.gpu_count, int)
            or isinstance(self.gpu_count, bool)
            or self.min_vram_gb < 1
            or self.gpu_count < 1
        ):
            raise ValueError("Ollama GPU memory and count must be positive")
        if not isinstance(self.cloud, CloudType):
            raise TypeError("Ollama cloud must be a CloudType")
        if self.max_hourly_rate is not None and not _positive_public_number(
            self.max_hourly_rate
        ):
            raise ValueError("Ollama max_hourly_rate must be positive")

    def requirements(self, product: ComputeProduct) -> PlacementRequirements:
        return PlacementRequirements(
            product=product,
            min_vram_gb=self.min_vram_gb,
            gpu_count=self.gpu_count,
            cloud=self.cloud,
            min_cuda_version=self.min_cuda_version,
            max_cost_per_hr=self.max_hourly_rate,
            allowed_gpu_ids=self.allowed_gpu_ids,
            allowed_gpu_pools=self.allowed_gpu_pools,
            allowed_data_center_ids=self.allowed_data_center_ids,
            benchmark_id="ollama-interactive",
        )


@dataclass(frozen=True)
class OllamaLeaseRequest:
    """Complete, idempotent request for bounded private inference capacity."""

    lease_id: str
    owner_id: str
    workload_id: str
    model: str
    constraints: OllamaResourceConstraints
    expected_session_seconds: int
    expected_active_seconds: int
    serverless_initialization_seconds: int
    serverless_idle_tail_seconds: int
    idle_timeout_seconds: int
    readiness_timeout_seconds: int
    hard_deadline: datetime
    max_authorized_cost: float
    mode: OllamaLeaseMode = OllamaLeaseMode.AUTO
    requested_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OllamaLeaseMode):
            raise TypeError("Ollama mode must be an OllamaLeaseMode")
        if not isinstance(self.constraints, OllamaResourceConstraints):
            raise TypeError("Ollama constraints must be OllamaResourceConstraints")
        if not isinstance(self.hard_deadline, datetime):
            raise TypeError("Ollama hard_deadline must be a datetime")
        if self.requested_at is not None and not isinstance(
            self.requested_at, datetime
        ):
            raise TypeError("Ollama requested_at must be a datetime when provided")
        for name, value in (
            ("lease_id", self.lease_id),
            ("owner_id", self.owner_id),
            ("workload_id", self.workload_id),
            ("model", self.model),
        ):
            if not isinstance(value, str):
                raise TypeError(f"Ollama {name} must be a string")
            if not value.strip():
                raise ValueError(f"Ollama {name} cannot be empty")
        positive = (
            self.expected_session_seconds,
            self.expected_active_seconds,
            self.idle_timeout_seconds,
            self.readiness_timeout_seconds,
        )
        nonnegative = (
            self.serverless_initialization_seconds,
            self.serverless_idle_tail_seconds,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive
        ):
            raise ValueError("Ollama duration fields must be positive integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in nonnegative
        ):
            raise ValueError("Ollama initialization/idle-tail fields must be integers")
        if self.expected_active_seconds > self.expected_session_seconds:
            raise ValueError("expected active time cannot exceed session time")
        if not _positive_public_number(self.max_authorized_cost):
            raise ValueError("max_authorized_cost must be positive")
        require_aware(self.hard_deadline, "hard_deadline")
        if self.requested_at is not None:
            require_aware(self.requested_at, "requested_at")
            if self.requested_at >= self.hard_deadline:
                raise ValueError("Ollama requested_at must precede hard_deadline")

    @property
    def fingerprint(self) -> str:
        constraints = asdict(self.constraints)
        constraints["cloud"] = self.constraints.cloud.value
        payload = {
            "owner_id": self.owner_id,
            "workload_id": self.workload_id,
            "model": self.model,
            "constraints": constraints,
            "expected_session_seconds": self.expected_session_seconds,
            "expected_active_seconds": self.expected_active_seconds,
            "serverless_initialization_seconds": (
                self.serverless_initialization_seconds
            ),
            "serverless_idle_tail_seconds": self.serverless_idle_tail_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
            "hard_deadline": iso_datetime(self.hard_deadline),
            "max_authorized_cost": self.max_authorized_cost,
            "mode": self.mode.value,
            "requested_at": (
                iso_datetime(self.requested_at) if self.requested_at else None
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def normalize_new_ollama_request_money(
    request: OllamaLeaseRequest,
) -> OllamaLeaseRequest:
    """Normalize public numeric spellings once before durable acceptance.

    Existing accepted requests retain their original fingerprint. Callers must
    therefore use this only when the lease ID has no durable row yet.
    """

    max_hourly_rate = request.constraints.max_hourly_rate
    constraints = request.constraints
    if isinstance(max_hourly_rate, int) and not isinstance(max_hourly_rate, bool):
        try:
            normalized_hourly_rate = conservative_decimal_ratio_float(
                Decimal(max_hourly_rate), 1
            )
        except ValueError as exc:
            raise ValueError(
                "Ollama max_hourly_rate cannot be represented as a finite float"
            ) from exc
        if not math.isfinite(normalized_hourly_rate):
            raise ValueError(
                "Ollama max_hourly_rate cannot be represented as a finite float"
            )
        constraints = replace(constraints, max_hourly_rate=normalized_hourly_rate)
    max_authorized_cost = request.max_authorized_cost
    if isinstance(max_authorized_cost, int) and not isinstance(
        max_authorized_cost, bool
    ):
        try:
            max_authorized_cost = conservative_decimal_ratio_float(
                Decimal(max_authorized_cost), 1
            )
        except ValueError as exc:
            raise ValueError(
                "Ollama max_authorized_cost cannot be represented as a finite float"
            ) from exc
        if not math.isfinite(max_authorized_cost):
            raise ValueError(
                "Ollama max_authorized_cost cannot be represented as a finite float"
            )
    if (
        constraints is request.constraints
        and max_authorized_cost is request.max_authorized_cost
    ):
        return request
    return replace(
        request,
        constraints=constraints,
        max_authorized_cost=max_authorized_cost,
    )


@dataclass(frozen=True)
class OllamaPlacementPlan:
    mode: OllamaLeaseMode
    resource_type: OllamaResourceType
    placement: PlacementDecision
    estimated_compute_cost: float
    maximum_compute_cost: float
    estimated_non_compute_cost: float
    maximum_non_compute_cost: float
    estimated_cost: float
    cost_ceiling: float
    estimated_billable_seconds: int
    maximum_billable_seconds: int
    maximum_concurrent_workers: int
    non_compute_components: tuple[OllamaNonComputeCostComponent, ...]
    maximum_serverless_cold_starts: int = 0

    def __post_init__(self) -> None:
        maximum = self.maximum_serverless_cold_starts
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
            raise ValueError("Ollama Serverless cold-start bound must be nonnegative")
        if self.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT and maximum < 1:
            raise ValueError("Ollama Serverless plans require a cold-start bound")
        if self.resource_type is OllamaResourceType.POD and maximum != 0:
            raise ValueError("Ollama Pod plans cannot declare Serverless cold starts")
        for name, value in (
            ("estimated_compute_cost", self.estimated_compute_cost),
            ("maximum_compute_cost", self.maximum_compute_cost),
            ("estimated_non_compute_cost", self.estimated_non_compute_cost),
            ("maximum_non_compute_cost", self.maximum_non_compute_cost),
            ("estimated_cost", self.estimated_cost),
            ("cost_ceiling", self.cost_ceiling),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Ollama plan {name} must be finite and nonnegative")
        for name, value in (
            ("estimated_billable_seconds", self.estimated_billable_seconds),
            ("maximum_billable_seconds", self.maximum_billable_seconds),
            ("maximum_concurrent_workers", self.maximum_concurrent_workers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Ollama plan {name} must be a positive integer")
        if self.maximum_compute_cost < self.estimated_compute_cost:
            raise ValueError("Ollama maximum compute cost must cover its estimate")
        if self.maximum_billable_seconds < self.estimated_billable_seconds:
            raise ValueError("Ollama maximum billable time must cover its estimate")
        if self.maximum_non_compute_cost < self.estimated_non_compute_cost:
            raise ValueError("Ollama maximum non-compute cost must cover its estimate")
        expected_compute = _rated_cost(
            self.placement.offered_cost_per_hr,
            self.estimated_billable_seconds,
            self.placement.gpu_count,
        )
        expected_maximum_compute = _rated_cost(
            self.placement.offered_cost_per_hr,
            self.maximum_billable_seconds,
            self.placement.gpu_count,
        )
        if not math.isclose(
            self.estimated_compute_cost,
            expected_compute,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.maximum_compute_cost,
            expected_maximum_compute,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Ollama compute costs are not derived from live rate")
        if (
            self.resource_type is OllamaResourceType.POD
            and self.maximum_concurrent_workers != 1
        ):
            raise ValueError("Ollama Pod plans require exactly one billable worker")
        if self.estimated_cost != _cost_sum(
            self.estimated_compute_cost, self.estimated_non_compute_cost
        ):
            raise ValueError("Ollama estimated total cost is not derived")
        if self.cost_ceiling != _cost_sum(
            self.maximum_compute_cost, self.maximum_non_compute_cost
        ):
            raise ValueError("Ollama total cost ceiling is not derived")
        if self.estimated_cost > self.cost_ceiling:
            raise ValueError("Ollama estimated total cost exceeds its ceiling")
        OllamaNonComputeCostPolicy(
            estimated_cost_usd=self.estimated_non_compute_cost,
            maximum_cost_usd=self.maximum_non_compute_cost,
            covered_components=self.non_compute_components,
        )


@dataclass(frozen=True)
class OllamaProvisioningIntent:
    """Content-safe original create authority persisted before provider I/O."""

    lease_id: str
    request_fingerprint: str
    provision_attempt_id: str
    mode: OllamaLeaseMode
    resource_type: OllamaResourceType
    resource_name: str
    provision_requested_at: datetime
    environment_sha256: str
    environment_keys: tuple[str, ...]
    provider_request_json: str
    provider_request_sha256: str
    selected_gpu_id: str
    selected_gpu_pool: str | None
    selected_gpu_name: str
    selected_gpu_count: int
    selected_cloud: CloudType
    allowed_data_center_ids: tuple[str, ...]
    accepted_whole_hourly_rate_usd: Decimal
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama provisioning intent version")
        for name, value in (
            ("lease_id", self.lease_id),
            ("provision_attempt_id", self.provision_attempt_id),
            ("resource_name", self.resource_name),
            ("selected_gpu_id", self.selected_gpu_id),
            ("selected_gpu_name", self.selected_gpu_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Ollama provisioning {name} must be nonempty")
        _sha256(self.request_fingerprint, "request_fingerprint")
        _sha256(self.environment_sha256, "environment_sha256")
        _sha256(self.provider_request_sha256, "provider_request_sha256")
        require_aware(self.provision_requested_at, "provision_requested_at")
        if (
            not isinstance(self.environment_keys, tuple)
            or not self.environment_keys
            or tuple(sorted(set(self.environment_keys))) != self.environment_keys
        ):
            raise ValueError("Ollama provisioning environment keys must be sorted")
        if any(not isinstance(item, str) or not item for item in self.environment_keys):
            raise ValueError("Ollama provisioning environment keys are invalid")
        if (
            not isinstance(self.selected_gpu_count, int)
            or isinstance(self.selected_gpu_count, bool)
            or self.selected_gpu_count < 1
        ):
            raise ValueError("Ollama provisioning GPU count must be positive")
        if not isinstance(self.selected_cloud, CloudType):
            raise TypeError("Ollama provisioning cloud is invalid")
        if (
            not isinstance(self.accepted_whole_hourly_rate_usd, Decimal)
            or not self.accepted_whole_hourly_rate_usd.is_finite()
            or self.accepted_whole_hourly_rate_usd <= 0
        ):
            raise ValueError("Ollama accepted whole hourly rate must be positive")
        request = _canonical_json_object(
            self.provider_request_json, "provider_request_json"
        )
        if _json_sha256(request) != self.provider_request_sha256:
            raise ValueError(
                "Ollama provider request digest does not match its payload"
            )
        environment = request.get("env")
        if (
            not isinstance(environment, Mapping)
            or set(environment) != {"keys", "sha256"}
            or _string_tuple_value(environment.get("keys")) != self.environment_keys
            or environment.get("sha256") != self.environment_sha256
        ):
            raise ValueError(
                "Ollama provider request must contain only bound environment evidence"
            )
        gpu = request.get("gpu")
        if not isinstance(gpu, Mapping):
            raise TypeError("Ollama provider request GPU evidence is invalid")
        if (
            request.get("name") != self.resource_name
            or gpu.get("count") != self.selected_gpu_count
            or _string_tuple_value(request.get("dataCenterIds", []))
            != self.allowed_data_center_ids
        ):
            raise ValueError("Ollama provider request does not bind its intent")
        if self.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            if (
                self.mode is not OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
                or not self.selected_gpu_pool
                or _string_tuple_value(gpu.get("pools")) != (self.selected_gpu_pool,)
                or request.get("type") != "LOAD_BALANCER"
            ):
                raise ValueError("Ollama Endpoint request does not bind its intent")
        elif (
            self.mode is not OllamaLeaseMode.DEDICATED_POD
            or self.selected_gpu_pool is not None
            or gpu.get("id") != self.selected_gpu_id
            or request.get("cloud") != self.selected_cloud.value
        ):
            raise ValueError("Ollama Pod request does not bind its intent")

    @property
    def intent_sha256(self) -> str:
        return _json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "lease_id": self.lease_id,
            "request_fingerprint": self.request_fingerprint,
            "provision_attempt_id": self.provision_attempt_id,
            "mode": self.mode.value,
            "resource_type": self.resource_type.value,
            "resource_name": self.resource_name,
            "provision_requested_at": iso_datetime(self.provision_requested_at),
            "environment_sha256": self.environment_sha256,
            "environment_keys": list(self.environment_keys),
            "provider_request": json.loads(self.provider_request_json),
            "provider_request_sha256": self.provider_request_sha256,
            "selected_gpu_id": self.selected_gpu_id,
            "selected_gpu_pool": self.selected_gpu_pool,
            "selected_gpu_name": self.selected_gpu_name,
            "selected_gpu_count": self.selected_gpu_count,
            "selected_cloud": self.selected_cloud.value,
            "allowed_data_center_ids": list(self.allowed_data_center_ids),
            "accepted_whole_hourly_rate_usd": decimal_text(
                self.accepted_whole_hourly_rate_usd
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaProvisioningIntent:
        _exact_keys(value, _INTENT_KEYS, "provisioning intent")
        request = _required_mapping(value, "provider_request")
        return cls(
            evidence_version=_required_int(value, "evidence_version"),
            lease_id=_required_string(value, "lease_id"),
            request_fingerprint=_required_string(value, "request_fingerprint"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            mode=OllamaLeaseMode(_required_string(value, "mode")),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            resource_name=_required_string(value, "resource_name"),
            provision_requested_at=_datetime(
                _required_string(value, "provision_requested_at")
            ),
            environment_sha256=_required_string(value, "environment_sha256"),
            environment_keys=_string_tuple_value(value.get("environment_keys")),
            provider_request_json=_canonical_json(request),
            provider_request_sha256=_required_string(value, "provider_request_sha256"),
            selected_gpu_id=_required_string(value, "selected_gpu_id"),
            selected_gpu_pool=_optional_string(value.get("selected_gpu_pool")),
            selected_gpu_name=_required_string(value, "selected_gpu_name"),
            selected_gpu_count=_required_int(value, "selected_gpu_count"),
            selected_cloud=CloudType(_required_string(value, "selected_cloud")),
            allowed_data_center_ids=_string_tuple_value(
                value.get("allowed_data_center_ids")
            ),
            accepted_whole_hourly_rate_usd=_required_decimal(
                value, "accepted_whole_hourly_rate_usd"
            ),
        )


@dataclass(frozen=True, repr=False)
class PreparedOllamaProvisioning:
    """Ephemeral exact create request paired with its safe durable intent."""

    intent: OllamaProvisioningIntent
    provider_request: PodCreateRequest | EndpointCreateRequest = field(repr=False)
    runtime_telemetry_binding_sha256: str

    def __post_init__(self) -> None:
        _sha256(
            self.runtime_telemetry_binding_sha256,
            "runtime telemetry binding",
        )
        if (
            self.runtime_telemetry_binding_sha256
            != runtime_telemetry_binding_for_intent(self.intent)
        ):
            raise ValueError("Prepared Ollama telemetry binding is inconsistent")

    def __repr__(self) -> str:
        return (
            "PreparedOllamaProvisioning("
            f"attempt={self.intent.provision_attempt_id!r}, "
            f"request_sha256={self.intent.provider_request_sha256!r})"
        )


@dataclass(frozen=True)
class OllamaProvisioningReceipt:
    """Exact-ID provider readback bound to one persisted original intent."""

    intent_sha256: str
    lease_id: str
    request_fingerprint: str
    provision_attempt_id: str
    resource_type: OllamaResourceType
    provider_resource_id: str
    resource_name: str
    provider_created_at: datetime
    readback_at: datetime
    readback_sha256: str
    environment_sha256: str
    reconciled_after_ambiguous_create: bool
    pod_realized_placement: PodRealizedPlacement | None = None
    provider_rate_precision: OllamaProviderRatePrecision | None = None
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama provisioning receipt version")
        for digest_name, digest in (
            ("intent_sha256", self.intent_sha256),
            ("request_fingerprint", self.request_fingerprint),
            ("readback_sha256", self.readback_sha256),
            ("environment_sha256", self.environment_sha256),
        ):
            _sha256(digest, digest_name)
        for name, value in (
            ("lease_id", self.lease_id),
            ("provision_attempt_id", self.provision_attempt_id),
            ("provider_resource_id", self.provider_resource_id),
            ("resource_name", self.resource_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Ollama provisioning receipt {name} is invalid")
        require_aware(self.provider_created_at, "provider_created_at")
        require_aware(self.readback_at, "readback_at")
        if self.resource_type is OllamaResourceType.POD:
            if (
                self.pod_realized_placement is None
                or self.provider_rate_precision
                is not OllamaProviderRatePrecision.PARSED_JSON_NUMBER
            ):
                raise ValueError(
                    "Ollama Pod receipt requires parsed realized placement"
                )
        elif (
            self.pod_realized_placement is not None
            or self.provider_rate_precision is not None
        ):
            raise ValueError(
                "Ollama Endpoint receipt cannot claim realized Pod billing"
            )

    @property
    def receipt_id(self) -> str:
        identity = {
            "intent_sha256": self.intent_sha256,
            "provider_resource_id": self.provider_resource_id,
            "readback_sha256": self.readback_sha256,
        }
        return "runpod-ollama-provisioning:" + _json_sha256(identity)

    def validate_intent(self, intent: OllamaProvisioningIntent) -> None:
        if (
            self.intent_sha256 != intent.intent_sha256
            or self.lease_id != intent.lease_id
            or self.request_fingerprint != intent.request_fingerprint
            or self.provision_attempt_id != intent.provision_attempt_id
            or self.resource_type is not intent.resource_type
            or self.resource_name != intent.resource_name
            or self.environment_sha256 != intent.environment_sha256
            or self.provider_created_at < intent.provision_requested_at
            or self.readback_at < intent.provision_requested_at
        ):
            raise ValueError("Ollama provisioning receipt does not bind its intent")
        placement = self.pod_realized_placement
        if placement is not None and (
            placement.provider_pod_id != self.provider_resource_id
            or placement.gpu_type_id != intent.selected_gpu_id
            or placement.gpu_display_name != intent.selected_gpu_name
            or placement.gpu_count != intent.selected_gpu_count
            or placement.cloud is not intent.selected_cloud
            or (
                intent.allowed_data_center_ids
                and placement.data_center_id not in intent.allowed_data_center_ids
            )
            or placement.hourly_rate_usd > intent.accepted_whole_hourly_rate_usd
        ):
            raise ValueError(
                "Ollama Pod provisioning receipt does not bind its accepted placement"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "receipt_id": self.receipt_id,
            "intent_sha256": self.intent_sha256,
            "lease_id": self.lease_id,
            "request_fingerprint": self.request_fingerprint,
            "provision_attempt_id": self.provision_attempt_id,
            "resource_type": self.resource_type.value,
            "provider_resource_id": self.provider_resource_id,
            "resource_name": self.resource_name,
            "provider_created_at": iso_datetime(self.provider_created_at),
            "readback_at": iso_datetime(self.readback_at),
            "readback_sha256": self.readback_sha256,
            "environment_sha256": self.environment_sha256,
            "reconciled_after_ambiguous_create": (
                self.reconciled_after_ambiguous_create
            ),
            "pod_realized_placement": (
                self.pod_realized_placement.to_dict()
                if self.pod_realized_placement
                else None
            ),
            "provider_rate_precision": (
                self.provider_rate_precision.value
                if self.provider_rate_precision
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaProvisioningReceipt:
        _exact_keys(value, _RECEIPT_KEYS, "provisioning receipt")
        raw_placement = value.get("pod_realized_placement")
        if raw_placement is not None and not isinstance(raw_placement, Mapping):
            raise TypeError("Ollama Pod realized placement must be an object or null")
        raw_precision = value.get("provider_rate_precision")
        receipt = cls(
            evidence_version=_required_int(value, "evidence_version"),
            intent_sha256=_required_string(value, "intent_sha256"),
            lease_id=_required_string(value, "lease_id"),
            request_fingerprint=_required_string(value, "request_fingerprint"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            provider_resource_id=_required_string(value, "provider_resource_id"),
            resource_name=_required_string(value, "resource_name"),
            provider_created_at=_datetime(
                _required_string(value, "provider_created_at")
            ),
            readback_at=_datetime(_required_string(value, "readback_at")),
            readback_sha256=_required_string(value, "readback_sha256"),
            environment_sha256=_required_string(value, "environment_sha256"),
            reconciled_after_ambiguous_create=_required_bool(
                value, "reconciled_after_ambiguous_create"
            ),
            pod_realized_placement=(
                PodRealizedPlacement.from_dict(raw_placement)
                if raw_placement is not None
                else None
            ),
            provider_rate_precision=(
                OllamaProviderRatePrecision(raw_precision)
                if isinstance(raw_precision, str)
                else None
            ),
        )
        if value.get("receipt_id") != receipt.receipt_id:
            raise ValueError("Ollama provisioning receipt identity is invalid")
        return receipt


@dataclass(frozen=True)
class OllamaProvisioningMismatchEvidence:
    """Sanitized exact-readback proof that a known resource is incompatible."""

    intent_sha256: str
    lease_id: str
    provision_attempt_id: str
    resource_type: OllamaResourceType
    provider_resource_id: str
    observed_resource_name: str
    observed_at: datetime
    readback_sha256: str
    reason: OllamaProvisioningMismatchReason
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama provisioning mismatch version")
        for digest_name, digest in (
            ("intent_sha256", self.intent_sha256),
            ("readback_sha256", self.readback_sha256),
        ):
            _sha256(digest, digest_name)
        for name, value in (
            ("lease_id", self.lease_id),
            ("provision_attempt_id", self.provision_attempt_id),
            ("provider_resource_id", self.provider_resource_id),
            ("observed_resource_name", self.observed_resource_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Ollama provisioning mismatch {name} is invalid")
        require_aware(self.observed_at, "observed_at")
        if not isinstance(self.reason, OllamaProvisioningMismatchReason):
            raise TypeError("Ollama provisioning mismatch reason is invalid")

    @property
    def evidence_id(self) -> str:
        identity = {
            "intent_sha256": self.intent_sha256,
            "provider_resource_id": self.provider_resource_id,
            "readback_sha256": self.readback_sha256,
            "reason": self.reason.value,
        }
        return "runpod-ollama-provisioning-mismatch:" + _json_sha256(identity)

    def validate_intent(self, intent: OllamaProvisioningIntent) -> None:
        if (
            self.intent_sha256 != intent.intent_sha256
            or self.lease_id != intent.lease_id
            or self.provision_attempt_id != intent.provision_attempt_id
            or self.resource_type is not intent.resource_type
            or self.observed_at < intent.provision_requested_at
        ):
            raise ValueError("Ollama provisioning mismatch does not bind its intent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "evidence_id": self.evidence_id,
            "intent_sha256": self.intent_sha256,
            "lease_id": self.lease_id,
            "provision_attempt_id": self.provision_attempt_id,
            "resource_type": self.resource_type.value,
            "provider_resource_id": self.provider_resource_id,
            "observed_resource_name": self.observed_resource_name,
            "observed_at": iso_datetime(self.observed_at),
            "readback_sha256": self.readback_sha256,
            "reason": self.reason.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaProvisioningMismatchEvidence:
        _exact_keys(value, _MISMATCH_KEYS, "provisioning mismatch")
        evidence = cls(
            evidence_version=_required_int(value, "evidence_version"),
            intent_sha256=_required_string(value, "intent_sha256"),
            lease_id=_required_string(value, "lease_id"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            provider_resource_id=_required_string(value, "provider_resource_id"),
            observed_resource_name=_required_string(value, "observed_resource_name"),
            observed_at=_datetime(_required_string(value, "observed_at")),
            readback_sha256=_required_string(value, "readback_sha256"),
            reason=OllamaProvisioningMismatchReason(_required_string(value, "reason")),
        )
        if value.get("evidence_id") != evidence.evidence_id:
            raise ValueError("Ollama provisioning mismatch identity is invalid")
        return evidence


@dataclass(frozen=True)
class OllamaCleanupIntent:
    """Immutable original exact-resource teardown admission."""

    intent_sha256: str
    provisioning_receipt_id: str | None
    lease_id: str
    request_fingerprint: str
    provision_attempt_id: str
    resource_type: OllamaResourceType
    provider_resource_id: str
    resource_name: str
    termination_reason: str
    delete_started_at: datetime
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama cleanup intent version")
        _sha256(self.intent_sha256, "cleanup intent_sha256")
        _sha256(self.request_fingerprint, "cleanup request_fingerprint")
        for name, value in (
            ("lease_id", self.lease_id),
            ("provision_attempt_id", self.provision_attempt_id),
            ("provider_resource_id", self.provider_resource_id),
            ("resource_name", self.resource_name),
            ("termination_reason", self.termination_reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Ollama cleanup {name} must be nonempty")
        if self.provisioning_receipt_id is not None and (
            not isinstance(self.provisioning_receipt_id, str)
            or not self.provisioning_receipt_id
        ):
            raise ValueError("Ollama cleanup provisioning receipt ID is invalid")
        require_aware(self.delete_started_at, "cleanup delete_started_at")

    @property
    def cleanup_intent_id(self) -> str:
        return "runpod-ollama-cleanup-intent:" + _json_sha256(
            {
                "intent_sha256": self.intent_sha256,
                "provider_resource_id": self.provider_resource_id,
                "delete_started_at": iso_datetime(self.delete_started_at),
            }
        )

    def validate_provisioning(
        self,
        intent: OllamaProvisioningIntent,
        receipt: OllamaProvisioningReceipt | None,
    ) -> None:
        if (
            self.intent_sha256 != intent.intent_sha256
            or self.lease_id != intent.lease_id
            or self.request_fingerprint != intent.request_fingerprint
            or self.provision_attempt_id != intent.provision_attempt_id
            or self.resource_type is not intent.resource_type
            or self.resource_name != intent.resource_name
            or self.delete_started_at < intent.provision_requested_at
        ):
            raise ValueError("Ollama cleanup intent does not bind provisioning")
        if receipt is None:
            if self.provisioning_receipt_id is not None:
                raise ValueError("Ollama cleanup intent claims an absent receipt")
        elif (
            self.provisioning_receipt_id != receipt.receipt_id
            or self.provider_resource_id != receipt.provider_resource_id
        ):
            raise ValueError("Ollama cleanup intent does not bind its receipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "cleanup_intent_id": self.cleanup_intent_id,
            "intent_sha256": self.intent_sha256,
            "provisioning_receipt_id": self.provisioning_receipt_id,
            "lease_id": self.lease_id,
            "request_fingerprint": self.request_fingerprint,
            "provision_attempt_id": self.provision_attempt_id,
            "resource_type": self.resource_type.value,
            "provider_resource_id": self.provider_resource_id,
            "resource_name": self.resource_name,
            "termination_reason": self.termination_reason,
            "delete_started_at": iso_datetime(self.delete_started_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaCleanupIntent:
        _exact_keys(value, _CLEANUP_INTENT_KEYS, "cleanup intent")
        result = cls(
            evidence_version=_required_int(value, "evidence_version"),
            intent_sha256=_required_string(value, "intent_sha256"),
            provisioning_receipt_id=_optional_string(
                value.get("provisioning_receipt_id")
            ),
            lease_id=_required_string(value, "lease_id"),
            request_fingerprint=_required_string(value, "request_fingerprint"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            provider_resource_id=_required_string(value, "provider_resource_id"),
            resource_name=_required_string(value, "resource_name"),
            termination_reason=_required_string(value, "termination_reason"),
            delete_started_at=_datetime(_required_string(value, "delete_started_at")),
        )
        if value.get("cleanup_intent_id") != result.cleanup_intent_id:
            raise ValueError("Ollama cleanup intent identity is invalid")
        return result


@dataclass(frozen=True)
class OllamaCleanupReceipt:
    """Exact-ID verified-absence upper-bound observation."""

    cleanup_intent_id: str
    verified_absent_at: datetime
    basis: OllamaCleanupBasis
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama cleanup receipt version")
        if not isinstance(self.cleanup_intent_id, str) or not self.cleanup_intent_id:
            raise ValueError("Ollama cleanup receipt intent ID is invalid")
        require_aware(self.verified_absent_at, "cleanup verified_absent_at")
        if not isinstance(self.basis, OllamaCleanupBasis):
            raise TypeError("Ollama cleanup basis is invalid")

    @property
    def cleanup_receipt_id(self) -> str:
        return "runpod-ollama-cleanup:" + _json_sha256(
            {
                "cleanup_intent_id": self.cleanup_intent_id,
                "verified_absent_at": iso_datetime(self.verified_absent_at),
                "basis": self.basis.value,
            }
        )

    def validate_intent(self, intent: OllamaCleanupIntent) -> None:
        if (
            self.cleanup_intent_id != intent.cleanup_intent_id
            or self.verified_absent_at < intent.delete_started_at
        ):
            raise ValueError("Ollama cleanup receipt does not bind its intent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "cleanup_receipt_id": self.cleanup_receipt_id,
            "cleanup_intent_id": self.cleanup_intent_id,
            "verified_absent_at": iso_datetime(self.verified_absent_at),
            "basis": self.basis.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaCleanupReceipt:
        _exact_keys(value, _CLEANUP_RECEIPT_KEYS, "cleanup receipt")
        result = cls(
            evidence_version=_required_int(value, "evidence_version"),
            cleanup_intent_id=_required_string(value, "cleanup_intent_id"),
            verified_absent_at=_datetime(_required_string(value, "verified_absent_at")),
            basis=OllamaCleanupBasis(_required_string(value, "basis")),
        )
        if value.get("cleanup_receipt_id") != result.cleanup_receipt_id:
            raise ValueError("Ollama cleanup receipt identity is invalid")
        return result


@dataclass(frozen=True)
class OllamaAdditionalCleanupTarget:
    """A second exact resource proven to belong to one admitted create attempt."""

    intent_sha256: str
    lease_id: str
    provision_attempt_id: str
    resource_type: OllamaResourceType
    resource_name: str
    provider_resource_id: str
    source: str
    provisioning_mismatch: OllamaProvisioningMismatchEvidence | None = None
    cleanup_intent: OllamaCleanupIntent | None = None
    cleanup_receipt: OllamaCleanupReceipt | None = None
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama additional cleanup target version")
        _sha256(self.intent_sha256, "additional cleanup intent")
        for name, value in (
            ("lease_id", self.lease_id),
            ("provision_attempt_id", self.provision_attempt_id),
            ("resource_name", self.resource_name),
            ("provider_resource_id", self.provider_resource_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Ollama additional cleanup {name} is invalid")
        if self.source != "original_post_ack":
            raise ValueError("Ollama additional cleanup source is invalid")
        if (
            self.provisioning_mismatch is not None
            and self.provisioning_mismatch.provider_resource_id
            != self.provider_resource_id
        ):
            raise ValueError("Ollama additional cleanup mismatch changed target")
        if self.cleanup_receipt is not None:
            if self.cleanup_intent is None:
                raise ValueError("Ollama additional cleanup receipt lacks intent")
            self.cleanup_receipt.validate_intent(self.cleanup_intent)

    @property
    def target_id(self) -> str:
        return "runpod-ollama-additional-cleanup:" + _json_sha256(
            {
                "intent_sha256": self.intent_sha256,
                "provider_resource_id": self.provider_resource_id,
                "source": self.source,
            }
        )

    def validate_intent(self, intent: OllamaProvisioningIntent) -> None:
        if (
            self.intent_sha256 != intent.intent_sha256
            or self.lease_id != intent.lease_id
            or self.provision_attempt_id != intent.provision_attempt_id
            or self.resource_type is not intent.resource_type
            or self.resource_name != intent.resource_name
        ):
            raise ValueError("Ollama additional cleanup target does not bind intent")
        if self.provisioning_mismatch is not None:
            self.provisioning_mismatch.validate_intent(intent)
        if self.cleanup_intent is not None:
            self.cleanup_intent.validate_provisioning(intent, None)
            if self.cleanup_intent.provider_resource_id != self.provider_resource_id:
                raise ValueError("Ollama additional cleanup intent changed target")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "target_id": self.target_id,
            "intent_sha256": self.intent_sha256,
            "lease_id": self.lease_id,
            "provision_attempt_id": self.provision_attempt_id,
            "resource_type": self.resource_type.value,
            "resource_name": self.resource_name,
            "provider_resource_id": self.provider_resource_id,
            "source": self.source,
            "provisioning_mismatch": (
                self.provisioning_mismatch.to_dict()
                if self.provisioning_mismatch
                else None
            ),
            "cleanup_intent": (
                self.cleanup_intent.to_dict() if self.cleanup_intent else None
            ),
            "cleanup_receipt": (
                self.cleanup_receipt.to_dict() if self.cleanup_receipt else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaAdditionalCleanupTarget:
        _exact_keys(value, _ADDITIONAL_CLEANUP_TARGET_KEYS, "additional cleanup target")
        raw_intent = value.get("cleanup_intent")
        raw_receipt = value.get("cleanup_receipt")
        raw_mismatch = value.get("provisioning_mismatch")
        result = cls(
            evidence_version=_required_int(value, "evidence_version"),
            intent_sha256=_required_string(value, "intent_sha256"),
            lease_id=_required_string(value, "lease_id"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            resource_name=_required_string(value, "resource_name"),
            provider_resource_id=_required_string(value, "provider_resource_id"),
            source=_required_string(value, "source"),
            provisioning_mismatch=(
                OllamaProvisioningMismatchEvidence.from_dict(raw_mismatch)
                if isinstance(raw_mismatch, Mapping)
                else None
            ),
            cleanup_intent=(
                OllamaCleanupIntent.from_dict(raw_intent)
                if isinstance(raw_intent, Mapping)
                else None
            ),
            cleanup_receipt=(
                OllamaCleanupReceipt.from_dict(raw_receipt)
                if isinstance(raw_receipt, Mapping)
                else None
            ),
        )
        if value.get("target_id") != result.target_id:
            raise ValueError("Ollama additional cleanup target identity is invalid")
        if (
            (raw_mismatch is not None and not isinstance(raw_mismatch, Mapping))
            or (raw_intent is not None and not isinstance(raw_intent, Mapping))
            or (raw_receipt is not None and not isinstance(raw_receipt, Mapping))
        ):
            raise ValueError("Ollama additional cleanup evidence is invalid")
        return result


def _validate_telemetry_phase(
    name: str,
    started_at: datetime | None,
    completed_at: datetime | None,
    duration_ms: int | None,
) -> None:
    if started_at is None or completed_at is None:
        if duration_ms is not None:
            raise ValueError(f"Ollama telemetry {name} duration is unbound")
        return
    if completed_at < started_at:
        raise ValueError(f"Ollama telemetry {name} order is invalid")
    derived_ms = (completed_at - started_at) // timedelta(milliseconds=1)
    # Go's RFC3339Nano wire preserves nanoseconds while Python ``datetime``
    # retains only microseconds.  Truncating each endpoint independently can
    # make Python's derived floor exactly one millisecond larger than Go's
    # ``time.Duration.Milliseconds`` result.  Accept only that bounded
    # quantization interval; larger discrepancies remain invalid.
    if duration_ms not in {derived_ms, max(0, derived_ms - 1)}:
        raise ValueError(f"Ollama telemetry {name} duration is inconsistent")


@dataclass(frozen=True)
class OllamaRuntimeTelemetrySnapshot:
    """Content-free cumulative observation from one runtime process."""

    runtime_version: str
    ollama_version: str
    telemetry_binding_sha256: str
    runtime_instance_id: str
    telemetry_revision: int
    provision_requested_at: datetime | None
    container_started_at: datetime
    ollama_process_started_at: datetime | None
    ollama_healthy_at: datetime | None
    model_pull_started_at: datetime | None
    model_pull_completed_at: datetime | None
    model_load_started_at: datetime | None
    model_ready_at: datetime | None
    ready_at: datetime | None
    placement_and_image_pull_wall_ms: int | None
    ollama_boot_ms: int | None
    model_pull_ms: int | None
    model_load_ms: int | None
    container_to_ready_ms: int | None
    cache_hit: bool
    startup_strategy: str
    ready: bool
    ollama_process_running: bool
    terminal_error: str | None
    inference_started: int
    inference_completed: int
    inference_failed: int
    inference_cancelled: int
    inference_in_flight: int
    first_inference_started_at: datetime | None
    last_inference_started_at: datetime | None
    last_inference_completed_at: datetime | None
    inference_busy_ms: int
    inference_idle_ms: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("Unsupported Ollama telemetry schema")
        _sha256(self.telemetry_binding_sha256, "telemetry binding")
        for name, value in (
            ("runtime_version", self.runtime_version),
            ("ollama_version", self.ollama_version),
            ("runtime_instance_id", self.runtime_instance_id),
            ("startup_strategy", self.startup_strategy),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Ollama telemetry {name} must be nonempty")
        if (
            not isinstance(self.telemetry_revision, int)
            or isinstance(self.telemetry_revision, bool)
            or self.telemetry_revision < 1
        ):
            raise ValueError("Ollama telemetry revision must be positive")
        require_aware(self.container_started_at, "telemetry container_started_at")
        for value in (
            self.provision_requested_at,
            self.ollama_process_started_at,
            self.ollama_healthy_at,
            self.model_pull_started_at,
            self.model_pull_completed_at,
            self.model_load_started_at,
            self.model_ready_at,
            self.ready_at,
            self.first_inference_started_at,
            self.last_inference_started_at,
            self.last_inference_completed_at,
        ):
            if value is not None:
                require_aware(value, "telemetry timestamp")
        for value in (
            self.placement_and_image_pull_wall_ms,
            self.ollama_boot_ms,
            self.model_pull_ms,
            self.model_load_ms,
            self.container_to_ready_ms,
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError("Ollama telemetry duration is invalid")
        for value in (
            self.inference_started,
            self.inference_completed,
            self.inference_failed,
            self.inference_cancelled,
            self.inference_in_flight,
            self.inference_busy_ms,
            self.inference_idle_ms,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Ollama telemetry counter is invalid")
        if (
            self.inference_completed
            + self.inference_failed
            + self.inference_cancelled
            + self.inference_in_flight
            != self.inference_started
        ):
            raise ValueError("Ollama telemetry inference counters are inconsistent")
        if self.ready != (self.ready_at is not None):
            raise ValueError("Ollama telemetry readiness is inconsistent")
        if self.terminal_error not in {None, "runtime_initialization_failed"}:
            raise ValueError("Ollama telemetry terminal error is invalid")
        _validate_telemetry_phase(
            "placement and image pull",
            self.provision_requested_at,
            self.container_started_at,
            self.placement_and_image_pull_wall_ms,
        )
        _validate_telemetry_phase(
            "Ollama boot",
            self.ollama_process_started_at,
            self.ollama_healthy_at,
            self.ollama_boot_ms,
        )
        _validate_telemetry_phase(
            "model pull",
            self.model_pull_started_at,
            self.model_pull_completed_at,
            self.model_pull_ms,
        )
        _validate_telemetry_phase(
            "model load",
            self.model_load_started_at,
            self.model_ready_at,
            self.model_load_ms,
        )
        if self.ready_at is not None and any(
            timestamp is not None and timestamp > self.ready_at
            for timestamp in (self.ollama_healthy_at, self.model_ready_at)
        ):
            raise ValueError("Ollama telemetry readiness order is invalid")
        if (
            self.first_inference_started_at is not None
            and self.last_inference_started_at is not None
            and self.first_inference_started_at > self.last_inference_started_at
        ):
            raise ValueError("Ollama telemetry inference order is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_version": self.runtime_version,
            "ollama_version": self.ollama_version,
            "telemetry_binding_sha256": self.telemetry_binding_sha256,
            "runtime_instance_id": self.runtime_instance_id,
            "telemetry_revision": self.telemetry_revision,
            "provision_requested_at": optional_iso_datetime(
                self.provision_requested_at
            ),
            "container_started_at": iso_datetime(self.container_started_at),
            "ollama_process_started_at": optional_iso_datetime(
                self.ollama_process_started_at
            ),
            "ollama_healthy_at": optional_iso_datetime(self.ollama_healthy_at),
            "model_pull_started_at": optional_iso_datetime(self.model_pull_started_at),
            "model_pull_completed_at": optional_iso_datetime(
                self.model_pull_completed_at
            ),
            "model_load_started_at": optional_iso_datetime(self.model_load_started_at),
            "model_ready_at": optional_iso_datetime(self.model_ready_at),
            "ready_at": optional_iso_datetime(self.ready_at),
            "placement_and_image_pull_wall_ms": self.placement_and_image_pull_wall_ms,
            "ollama_boot_ms": self.ollama_boot_ms,
            "model_pull_ms": self.model_pull_ms,
            "model_load_ms": self.model_load_ms,
            "container_to_ready_ms": self.container_to_ready_ms,
            "cache_hit": self.cache_hit,
            "startup_strategy": self.startup_strategy,
            "ready": self.ready,
            "ollama_process_running": self.ollama_process_running,
            "terminal_error": self.terminal_error,
            "inference_started": self.inference_started,
            "inference_completed": self.inference_completed,
            "inference_failed": self.inference_failed,
            "inference_cancelled": self.inference_cancelled,
            "inference_in_flight": self.inference_in_flight,
            "first_inference_started_at": optional_iso_datetime(
                self.first_inference_started_at
            ),
            "last_inference_started_at": optional_iso_datetime(
                self.last_inference_started_at
            ),
            "last_inference_completed_at": optional_iso_datetime(
                self.last_inference_completed_at
            ),
            "inference_busy_ms": self.inference_busy_ms,
            "inference_idle_ms": self.inference_idle_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaRuntimeTelemetrySnapshot:
        _exact_keys(value, _TELEMETRY_SNAPSHOT_KEYS, "telemetry snapshot")
        return cls(
            schema_version=_required_int(value, "schema_version"),
            runtime_version=_required_string(value, "runtime_version"),
            ollama_version=_required_string(value, "ollama_version"),
            telemetry_binding_sha256=_required_string(
                value, "telemetry_binding_sha256"
            ),
            runtime_instance_id=_required_string(value, "runtime_instance_id"),
            telemetry_revision=_required_int(value, "telemetry_revision"),
            provision_requested_at=_optional_datetime_value(
                value, "provision_requested_at"
            ),
            container_started_at=_datetime(
                _required_string(value, "container_started_at")
            ),
            ollama_process_started_at=_optional_datetime_value(
                value, "ollama_process_started_at"
            ),
            ollama_healthy_at=_optional_datetime_value(value, "ollama_healthy_at"),
            model_pull_started_at=_optional_datetime_value(
                value, "model_pull_started_at"
            ),
            model_pull_completed_at=_optional_datetime_value(
                value, "model_pull_completed_at"
            ),
            model_load_started_at=_optional_datetime_value(
                value, "model_load_started_at"
            ),
            model_ready_at=_optional_datetime_value(value, "model_ready_at"),
            ready_at=_optional_datetime_value(value, "ready_at"),
            placement_and_image_pull_wall_ms=_optional_int_value(
                value, "placement_and_image_pull_wall_ms"
            ),
            ollama_boot_ms=_optional_int_value(value, "ollama_boot_ms"),
            model_pull_ms=_optional_int_value(value, "model_pull_ms"),
            model_load_ms=_optional_int_value(value, "model_load_ms"),
            container_to_ready_ms=_optional_int_value(value, "container_to_ready_ms"),
            cache_hit=_required_bool(value, "cache_hit"),
            startup_strategy=_required_string(value, "startup_strategy"),
            ready=_required_bool(value, "ready"),
            ollama_process_running=_required_bool(value, "ollama_process_running"),
            terminal_error=_optional_string(value.get("terminal_error")),
            inference_started=_required_int(value, "inference_started"),
            inference_completed=_required_int(value, "inference_completed"),
            inference_failed=_required_int(value, "inference_failed"),
            inference_cancelled=_required_int(value, "inference_cancelled"),
            inference_in_flight=_required_int(value, "inference_in_flight"),
            first_inference_started_at=_optional_datetime_value(
                value, "first_inference_started_at"
            ),
            last_inference_started_at=_optional_datetime_value(
                value, "last_inference_started_at"
            ),
            last_inference_completed_at=_optional_datetime_value(
                value, "last_inference_completed_at"
            ),
            inference_busy_ms=_required_int(value, "inference_busy_ms"),
            inference_idle_ms=_required_int(value, "inference_idle_ms"),
        )


@dataclass(frozen=True)
class OllamaRuntimeTelemetryObservation:
    intent_sha256: str
    provisioning_receipt_id: str
    lease_id: str
    provision_attempt_id: str
    resource_type: OllamaResourceType
    provider_resource_id: str
    snapshot: OllamaRuntimeTelemetrySnapshot
    snapshot_sha256: str
    fetch_started_at: datetime
    observed_at: datetime
    authentication_basis: str = "exact_route_bearer"
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1:
            raise ValueError("Unsupported Ollama telemetry observation version")
        _sha256(self.intent_sha256, "telemetry intent")
        _sha256(self.snapshot_sha256, "telemetry snapshot")
        if self.snapshot_sha256 != _json_sha256(self.snapshot.to_dict()):
            raise ValueError("Ollama telemetry snapshot digest is invalid")
        if self.authentication_basis != "exact_route_bearer":
            raise ValueError("Ollama telemetry authentication basis is invalid")
        require_aware(self.fetch_started_at, "telemetry fetch_started_at")
        require_aware(self.observed_at, "telemetry observed_at")
        if self.observed_at < self.fetch_started_at:
            raise ValueError("Ollama telemetry observation interval is invalid")

    @property
    def observation_id(self) -> str:
        return "runpod-ollama-telemetry:" + _json_sha256(
            {
                "intent_sha256": self.intent_sha256,
                "provisioning_receipt_id": self.provisioning_receipt_id,
                "provider_resource_id": self.provider_resource_id,
                "snapshot_sha256": self.snapshot_sha256,
            }
        )

    def validate_provisioning(
        self,
        intent: OllamaProvisioningIntent,
        receipt: OllamaProvisioningReceipt,
        *,
        telemetry_binding_sha256: str,
    ) -> None:
        _sha256(telemetry_binding_sha256, "runtime telemetry binding")
        if (
            self.intent_sha256 != intent.intent_sha256
            or self.provisioning_receipt_id != receipt.receipt_id
            or self.lease_id != intent.lease_id
            or self.provision_attempt_id != intent.provision_attempt_id
            or self.resource_type is not intent.resource_type
            or self.provider_resource_id != receipt.provider_resource_id
            or self.snapshot.telemetry_binding_sha256 != telemetry_binding_sha256
            or self.fetch_started_at < receipt.readback_at
        ):
            raise ValueError("Ollama telemetry does not bind provisioning")

    def is_complete_startup_for(self, intent: OllamaProvisioningIntent) -> bool:
        """Return whether this snapshot proves the original runtime became ready.

        Ordinary authenticated telemetry remains useful when startup is not
        complete, but it must not authorize a route.  The runtime's original
        birth timestamp is part of the immutable provisioning intent rather
        than a value recoverable from a later observation.
        """

        snapshot = self.snapshot
        return (
            snapshot.provision_requested_at == intent.provision_requested_at
            and snapshot.ready is True
            and snapshot.ollama_process_running is True
            and snapshot.ready_at is not None
            and snapshot.terminal_error is None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "observation_id": self.observation_id,
            "intent_sha256": self.intent_sha256,
            "provisioning_receipt_id": self.provisioning_receipt_id,
            "lease_id": self.lease_id,
            "provision_attempt_id": self.provision_attempt_id,
            "resource_type": self.resource_type.value,
            "provider_resource_id": self.provider_resource_id,
            "snapshot": self.snapshot.to_dict(),
            "snapshot_sha256": self.snapshot_sha256,
            "fetch_started_at": iso_datetime(self.fetch_started_at),
            "observed_at": iso_datetime(self.observed_at),
            "authentication_basis": self.authentication_basis,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaRuntimeTelemetryObservation:
        _exact_keys(value, _TELEMETRY_OBSERVATION_KEYS, "telemetry observation")
        result = cls(
            evidence_version=_required_int(value, "evidence_version"),
            intent_sha256=_required_string(value, "intent_sha256"),
            provisioning_receipt_id=_required_string(value, "provisioning_receipt_id"),
            lease_id=_required_string(value, "lease_id"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            provider_resource_id=_required_string(value, "provider_resource_id"),
            snapshot=OllamaRuntimeTelemetrySnapshot.from_dict(
                _required_mapping(value, "snapshot")
            ),
            snapshot_sha256=_required_string(value, "snapshot_sha256"),
            fetch_started_at=_datetime(_required_string(value, "fetch_started_at")),
            observed_at=_datetime(_required_string(value, "observed_at")),
            authentication_basis=_required_string(value, "authentication_basis"),
        )
        if value.get("observation_id") != result.observation_id:
            raise ValueError("Ollama telemetry observation identity is invalid")
        return result


@dataclass(frozen=True)
class OllamaLifecycleBillingObservation:
    intent_sha256: str
    provisioning_receipt_id: str
    cleanup_receipt_id: str
    lease_id: str
    provision_attempt_id: str
    resource_type: OllamaResourceType
    provider_resource_id: str
    cost_scope: str
    provider_capacity_actual_cost_usd: Decimal
    first_observed_at: datetime
    pod_billing_receipt: PodBillingReceipt | None = None
    serverless_endpoint_hour_costs: tuple[ServerlessEndpointHourCost, ...] = ()
    provider_finality_confirmed: bool = False
    evidence_version: int = 1

    def __post_init__(self) -> None:
        if self.evidence_version != 1 or self.provider_finality_confirmed is not False:
            raise ValueError("Ollama billing cannot claim provider finality")
        _sha256(self.intent_sha256, "billing intent")
        require_aware(self.first_observed_at, "billing first_observed_at")
        if (
            not self.provider_capacity_actual_cost_usd.is_finite()
            or self.provider_capacity_actual_cost_usd < 0
        ):
            raise ValueError("Ollama provider actual cost is invalid")
        if self.resource_type is OllamaResourceType.POD:
            if (
                self.cost_scope != "pod_lifecycle"
                or self.pod_billing_receipt is None
                or self.serverless_endpoint_hour_costs
            ):
                raise ValueError("Ollama Pod billing payload is invalid")
            if (
                self.pod_billing_receipt.provider_pod_id != self.provider_resource_id
                or self.pod_billing_receipt.actual_cost_usd
                != self.provider_capacity_actual_cost_usd
            ):
                raise ValueError("Ollama Pod billing payload is unbound")
        else:
            if (
                self.cost_scope != "serverless_endpoint_lifecycle"
                or self.pod_billing_receipt is not None
                or not self.serverless_endpoint_hour_costs
            ):
                raise ValueError("Ollama Endpoint billing payload is invalid")
            if (
                any(
                    item.endpoint_id != self.provider_resource_id
                    for item in self.serverless_endpoint_hour_costs
                )
                or serverless_endpoint_hour_cost_totals(
                    self.serverless_endpoint_hour_costs
                )["actual_cost_usd"]
                != self.provider_capacity_actual_cost_usd
            ):
                raise ValueError("Ollama Endpoint billing payload is unbound")

    @property
    def observation_id(self) -> str:
        payload: dict[str, Any] = {
            "intent_sha256": self.intent_sha256,
            "provisioning_receipt_id": self.provisioning_receipt_id,
            "cleanup_receipt_id": self.cleanup_receipt_id,
            "provider_resource_id": self.provider_resource_id,
            "cost_scope": self.cost_scope,
            "provider_capacity_actual_cost_usd": decimal_text(
                self.provider_capacity_actual_cost_usd
            ),
            "pod_billing_receipt": self.pod_billing_receipt.to_dict()
            if self.pod_billing_receipt
            else None,
            "serverless_endpoint_hour_costs": [
                item.to_dict() for item in self.serverless_endpoint_hour_costs
            ],
        }
        return "runpod-ollama-lifecycle-billing:" + _json_sha256(payload)

    def validate_lifecycle(
        self,
        intent: OllamaProvisioningIntent,
        provisioning: OllamaProvisioningReceipt,
        cleanup_intent: OllamaCleanupIntent,
        cleanup: OllamaCleanupReceipt,
    ) -> None:
        cleanup_intent.validate_provisioning(intent, provisioning)
        cleanup.validate_intent(cleanup_intent)
        if (
            self.intent_sha256 != intent.intent_sha256
            or self.provisioning_receipt_id != provisioning.receipt_id
            or self.cleanup_receipt_id != cleanup.cleanup_receipt_id
            or self.lease_id != intent.lease_id
            or self.provision_attempt_id != intent.provision_attempt_id
            or self.resource_type is not intent.resource_type
            or self.provider_resource_id != provisioning.provider_resource_id
            or self.first_observed_at < cleanup.verified_absent_at
        ):
            raise ValueError("Ollama billing observation does not bind lifecycle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_version": self.evidence_version,
            "observation_id": self.observation_id,
            "intent_sha256": self.intent_sha256,
            "provisioning_receipt_id": self.provisioning_receipt_id,
            "cleanup_receipt_id": self.cleanup_receipt_id,
            "lease_id": self.lease_id,
            "provision_attempt_id": self.provision_attempt_id,
            "resource_type": self.resource_type.value,
            "provider_resource_id": self.provider_resource_id,
            "cost_scope": self.cost_scope,
            "provider_finality_confirmed": self.provider_finality_confirmed,
            "provider_capacity_actual_cost_usd": decimal_text(
                self.provider_capacity_actual_cost_usd
            ),
            "first_observed_at": iso_datetime(self.first_observed_at),
            "pod_billing_receipt": self.pod_billing_receipt.to_dict()
            if self.pod_billing_receipt
            else None,
            "serverless_endpoint_hour_costs": [
                item.to_dict() for item in self.serverless_endpoint_hour_costs
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaLifecycleBillingObservation:
        _exact_keys(value, _BILLING_OBSERVATION_KEYS, "billing observation")
        pod = value.get("pod_billing_receipt")
        hours = value.get("serverless_endpoint_hour_costs")
        if pod is not None and not isinstance(pod, Mapping):
            raise TypeError("Ollama Pod billing receipt must be an object")
        if not isinstance(hours, list):
            raise TypeError("Ollama Serverless billing hours must be a list")
        result = cls(
            evidence_version=_required_int(value, "evidence_version"),
            intent_sha256=_required_string(value, "intent_sha256"),
            provisioning_receipt_id=_required_string(value, "provisioning_receipt_id"),
            cleanup_receipt_id=_required_string(value, "cleanup_receipt_id"),
            lease_id=_required_string(value, "lease_id"),
            provision_attempt_id=_required_string(value, "provision_attempt_id"),
            resource_type=OllamaResourceType(_required_string(value, "resource_type")),
            provider_resource_id=_required_string(value, "provider_resource_id"),
            cost_scope=_required_string(value, "cost_scope"),
            provider_finality_confirmed=_required_bool(
                value, "provider_finality_confirmed"
            ),
            provider_capacity_actual_cost_usd=_required_decimal(
                value, "provider_capacity_actual_cost_usd"
            ),
            first_observed_at=_datetime(_required_string(value, "first_observed_at")),
            pod_billing_receipt=PodBillingReceipt.from_dict(pod)
            if pod is not None
            else None,
            serverless_endpoint_hour_costs=tuple(
                ServerlessEndpointHourCost.from_dict(item)
                for item in hours
                if isinstance(item, Mapping)
            ),
        )
        if len(result.serverless_endpoint_hour_costs) != len(hours):
            raise TypeError("Ollama Serverless billing hours must contain objects")
        if value.get("observation_id") != result.observation_id:
            raise ValueError("Ollama billing observation identity is invalid")
        return result


@dataclass(frozen=True)
class OllamaLifecycleBillingSelection:
    observation_id: str
    selected_at: datetime
    policy_version: int = 1

    def __post_init__(self) -> None:
        if not self.observation_id or self.policy_version != 1:
            raise ValueError("Ollama billing selection is invalid")
        require_aware(self.selected_at, "billing selected_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "selected_at": iso_datetime(self.selected_at),
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaLifecycleBillingSelection:
        _exact_keys(
            value,
            {"observation_id", "selected_at", "policy_version"},
            "billing selection",
        )
        return cls(
            observation_id=_required_string(value, "observation_id"),
            selected_at=_datetime(_required_string(value, "selected_at")),
            policy_version=_required_int(value, "policy_version"),
        )


@dataclass(frozen=True)
class OllamaLifecycleBillingCorrection:
    selected_observation_id: str
    revised_observation_id: str
    actual_cost_delta_usd: Decimal
    recorded_at: datetime

    def __post_init__(self) -> None:
        if (
            not self.selected_observation_id
            or not self.revised_observation_id
            or self.selected_observation_id == self.revised_observation_id
        ):
            raise ValueError("Ollama billing correction identity is invalid")
        if not self.actual_cost_delta_usd.is_finite():
            raise ValueError("Ollama billing correction delta is invalid")
        require_aware(self.recorded_at, "billing correction recorded_at")

    @property
    def correction_id(self) -> str:
        return "runpod-ollama-billing-correction:" + _json_sha256(
            {
                "selected_observation_id": self.selected_observation_id,
                "revised_observation_id": self.revised_observation_id,
                "actual_cost_delta_usd": decimal_text(self.actual_cost_delta_usd),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "selected_observation_id": self.selected_observation_id,
            "revised_observation_id": self.revised_observation_id,
            "actual_cost_delta_usd": decimal_text(self.actual_cost_delta_usd),
            "recorded_at": iso_datetime(self.recorded_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OllamaLifecycleBillingCorrection:
        _exact_keys(
            value,
            {
                "correction_id",
                "selected_observation_id",
                "revised_observation_id",
                "actual_cost_delta_usd",
                "recorded_at",
            },
            "billing correction",
        )
        result = cls(
            selected_observation_id=_required_string(value, "selected_observation_id"),
            revised_observation_id=_required_string(value, "revised_observation_id"),
            actual_cost_delta_usd=_required_decimal(value, "actual_cost_delta_usd"),
            recorded_at=_datetime(_required_string(value, "recorded_at")),
        )
        if value.get("correction_id") != result.correction_id:
            raise ValueError("Ollama billing correction identity is invalid")
        return result


@dataclass(frozen=True)
class ProvisionedOllamaResource:
    resource_type: OllamaResourceType
    provider_resource_id: str
    resource_name: str


@dataclass(frozen=True)
class OllamaReadinessObservation:
    provider_ready: bool
    route_url: str | None
    provider_health_url: str | None
    model_names: tuple[str, ...] = ()
    runtime_telemetry: OllamaRuntimeTelemetryObservation | None = None

    def has_model(self, requested_model: str) -> bool:
        requested = canonical_model_name(requested_model)
        return any(canonical_model_name(name) == requested for name in self.model_names)


@dataclass(frozen=True)
class OllamaLease:
    lease_id: str
    owner_id: str
    workload_id: str
    request_fingerprint: str
    model: str
    constraints_json: str
    mode: OllamaLeaseMode | None
    resource_type: OllamaResourceType | None
    provider_resource_id: str | None
    resource_name: str | None
    creation_uncertain: bool
    provision_attempt_id: str | None
    provision_attempts: int
    route_url: str | None
    provider_health_url: str | None
    state: OllamaLeaseState
    teardown_state: OllamaTeardownState
    created_at: datetime
    updated_at: datetime
    provisioning_started_at: datetime | None
    ready_at: datetime | None
    last_used_at: datetime
    idle_deadline: datetime
    hard_deadline: datetime
    readiness_deadline: datetime
    model_pull_started_at: datetime | None
    model_pull_attempts: int
    model_ready_at: datetime | None
    expected_session_seconds: int
    expected_active_seconds: int
    serverless_initialization_seconds: int
    serverless_idle_tail_seconds: int
    idle_timeout_seconds: int
    offered_rate_per_hr: float | None
    estimated_cost: float | None
    estimated_compute_cost: float | None
    maximum_compute_cost: float | None
    estimated_non_compute_cost: float | None
    maximum_non_compute_cost: float | None
    cost_ceiling: float | None
    cost_policy_components: tuple[OllamaNonComputeCostComponent, ...]
    maximum_concurrent_workers: int | None
    estimated_billable_seconds: int | None
    maximum_billable_seconds: int | None
    accrued_estimated_cost: float
    max_authorized_cost: float
    cold_start_seconds: float | None
    selected_gpu_id: str | None
    selected_gpu_pool: str | None
    selected_gpu_name: str | None
    catalog_observed_at: datetime | None
    last_provider_error: str | None
    termination_reason: str | None
    teardown_attempts: int
    revision: int
    # Inventory discovery is a durable reconciliation lead, never cleanup
    # authority. Only an acknowledged POST or a complete bound readback may
    # populate ``provider_resource_id``.
    ambiguous_create_candidate_id: str | None = None
    # How many GPUs the placement attaches. ``offered_rate_per_hr`` stays the
    # catalog's PER-GPU price, exactly as /catalog/gpus reported it, so the
    # count is carried alongside rather than folded in - a stored rate that
    # silently meant something new would misread every existing row. Defaults
    # to a single GPU, which is what every pre-existing lease had.
    placement_gpu_count: int = 1
    writer_contract_version: int = 4
    writer_epoch: int = 1
    reconcile_after: datetime | None = None
    provisioning_evidence_state: OllamaProvisioningEvidenceState = (
        OllamaProvisioningEvidenceState.REQUIRED
    )
    provisioning_intent: OllamaProvisioningIntent | None = None
    provisioning_receipt: OllamaProvisioningReceipt | None = None
    provisioning_mismatch: OllamaProvisioningMismatchEvidence | None = None
    # A discovered name match is only a reconciliation locator.  Its mismatch
    # evidence is kept separately so a later original POST acknowledgment can
    # establish a different, authoritative cleanup ID without rewriting
    # either observation.
    ambiguous_candidate_mismatch: OllamaProvisioningMismatchEvidence | None = None
    # These are original fingerprint inputs, not values derived from a later
    # restart clock or from mutable lifecycle deadlines.
    original_requested_at: datetime | None = None
    original_readiness_timeout_seconds: int | None = None
    runtime_telemetry_binding_sha256: str | None = None
    cleanup_intent: OllamaCleanupIntent | None = None
    cleanup_receipt: OllamaCleanupReceipt | None = None
    additional_cleanup_targets: tuple[OllamaAdditionalCleanupTarget, ...] = ()
    runtime_telemetry_state: OllamaRuntimeTelemetryState = (
        OllamaRuntimeTelemetryState.UNKNOWN
    )
    runtime_telemetry_observations: tuple[OllamaRuntimeTelemetryObservation, ...] = ()
    runtime_telemetry_last_provider_error: str | None = None
    lifecycle_billing_state: OllamaLifecycleBillingState = (
        OllamaLifecycleBillingState.UNKNOWN
    )
    lifecycle_billing_observations: tuple[OllamaLifecycleBillingObservation, ...] = ()
    lifecycle_billing_selection: OllamaLifecycleBillingSelection | None = None
    lifecycle_billing_corrections: tuple[OllamaLifecycleBillingCorrection, ...] = ()
    billing_last_checked_at: datetime | None = None
    billing_next_check_at: datetime | None = None
    billing_last_provider_error: str | None = None
    billing_retention_exhausted: bool = False

    @property
    def public_route_url(self) -> str | None:
        """Return a route only after both provider and requested-model readiness."""

        return self.route_url if self.state is OllamaLeaseState.READY else None

    @property
    def actual_cost_cap_status(self) -> OllamaActualCostCapStatus:
        if self.additional_cleanup_targets:
            return OllamaActualCostCapStatus.UNKNOWN
        selection = self.lifecycle_billing_selection
        if selection is None:
            return OllamaActualCostCapStatus.UNKNOWN
        selected = next(
            (
                item
                for item in self.lifecycle_billing_observations
                if item.observation_id == selection.observation_id
            ),
            None,
        )
        if selected is None:
            return OllamaActualCostCapStatus.UNKNOWN
        return (
            OllamaActualCostCapStatus.EXCEEDED
            if selected.provider_capacity_actual_cost_usd
            > Decimal(str(self.max_authorized_cost))
            else OllamaActualCostCapStatus.UNKNOWN
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "owner_id": self.owner_id,
            "workload_id": self.workload_id,
            "model": self.model,
            "mode": self.mode.value if self.mode else None,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "provider_resource_id": self.provider_resource_id,
            "ambiguous_create_candidate_id": self.ambiguous_create_candidate_id,
            "provision_attempt_id": self.provision_attempt_id,
            "creation_uncertain": self.creation_uncertain,
            "state": self.state.value,
            "teardown_state": self.teardown_state.value,
            "created_at": iso_datetime(self.created_at),
            "original_requested_at": optional_iso_datetime(self.original_requested_at),
            "original_readiness_timeout_seconds": (
                self.original_readiness_timeout_seconds
            ),
            "provisioning_started_at": optional_iso_datetime(
                self.provisioning_started_at
            ),
            "ready_at": optional_iso_datetime(self.ready_at),
            "readiness_deadline": iso_datetime(self.readiness_deadline),
            "last_used_at": iso_datetime(self.last_used_at),
            "idle_deadline": iso_datetime(self.idle_deadline),
            "hard_deadline": iso_datetime(self.hard_deadline),
            "offered_rate_per_hr": self.offered_rate_per_hr,
            "estimated_cost": self.estimated_cost,
            "estimated_compute_cost": self.estimated_compute_cost,
            "maximum_compute_cost": self.maximum_compute_cost,
            "estimated_non_compute_cost": self.estimated_non_compute_cost,
            "maximum_non_compute_cost": self.maximum_non_compute_cost,
            "cost_ceiling": self.cost_ceiling,
            "cost_policy_components": tuple(
                item.value for item in self.cost_policy_components
            ),
            "maximum_concurrent_workers": self.maximum_concurrent_workers,
            "estimated_billable_seconds": self.estimated_billable_seconds,
            "maximum_billable_seconds": self.maximum_billable_seconds,
            "expected_session_seconds": self.expected_session_seconds,
            "expected_active_seconds": self.expected_active_seconds,
            "serverless_initialization_seconds": (
                self.serverless_initialization_seconds
            ),
            "serverless_idle_tail_seconds": self.serverless_idle_tail_seconds,
            "accrued_estimated_cost": self.accrued_estimated_cost,
            "max_authorized_cost": self.max_authorized_cost,
            "cold_start_seconds": self.cold_start_seconds,
            "model_pull_started_at": optional_iso_datetime(self.model_pull_started_at),
            "model_ready_at": optional_iso_datetime(self.model_ready_at),
            "provision_attempts": self.provision_attempts,
            "model_pull_attempts": self.model_pull_attempts,
            "teardown_attempts": self.teardown_attempts,
            "selected_gpu_id": self.selected_gpu_id,
            "selected_gpu_pool": self.selected_gpu_pool,
            "selected_gpu_name": self.selected_gpu_name,
            "catalog_observed_at": optional_iso_datetime(self.catalog_observed_at),
            "last_provider_error": self.last_provider_error,
            "termination_reason": self.termination_reason,
            "writer_contract_version": self.writer_contract_version,
            "writer_epoch": self.writer_epoch,
            "reconcile_after": optional_iso_datetime(self.reconcile_after),
            "provisioning_evidence_state": self.provisioning_evidence_state.value,
            "provisioning_intent": (
                self.provisioning_intent.to_dict() if self.provisioning_intent else None
            ),
            "provisioning_receipt": (
                self.provisioning_receipt.to_dict()
                if self.provisioning_receipt
                else None
            ),
            "provisioning_mismatch": (
                self.provisioning_mismatch.to_dict()
                if self.provisioning_mismatch
                else None
            ),
            "ambiguous_candidate_mismatch": (
                self.ambiguous_candidate_mismatch.to_dict()
                if self.ambiguous_candidate_mismatch
                else None
            ),
            "runtime_telemetry_binding_sha256": (self.runtime_telemetry_binding_sha256),
            "cleanup_intent": (
                self.cleanup_intent.to_dict() if self.cleanup_intent else None
            ),
            "cleanup_receipt": (
                self.cleanup_receipt.to_dict() if self.cleanup_receipt else None
            ),
            "additional_cleanup_target_ids": tuple(
                item.target_id for item in self.additional_cleanup_targets
            ),
            "runtime_telemetry_state": self.runtime_telemetry_state.value,
            "runtime_telemetry_observation_ids": tuple(
                item.observation_id for item in self.runtime_telemetry_observations
            ),
            "runtime_telemetry_last_provider_error": (
                self.runtime_telemetry_last_provider_error
            ),
            "lifecycle_billing_state": self.lifecycle_billing_state.value,
            "lifecycle_billing_selected_observation_id": (
                self.lifecycle_billing_selection.observation_id
                if self.lifecycle_billing_selection
                else None
            ),
            "lifecycle_billing_latest_observation_id": (
                self.lifecycle_billing_observations[-1].observation_id
                if self.lifecycle_billing_observations
                else None
            ),
            "lifecycle_billing_correction_count": len(
                self.lifecycle_billing_corrections
            ),
            "provider_finality_confirmed": (
                False if self.lifecycle_billing_selection else None
            ),
            "actual_cost_cap_status": self.actual_cost_cap_status.value,
            "billing_last_checked_at": optional_iso_datetime(
                self.billing_last_checked_at
            ),
            "billing_next_check_at": optional_iso_datetime(self.billing_next_check_at),
            "billing_last_provider_error": self.billing_last_provider_error,
            "billing_retention_exhausted": self.billing_retention_exhausted,
        }


class OllamaCapacityProvider(Protocol):
    def validate_runtime_request(
        self,
        request: OllamaLeaseRequest,
        plan: OllamaPlacementPlan,
        *,
        provision_requested_at: datetime,
    ) -> PreparedOllamaProvisioning:
        """Prepare the exact create request without provider I/O."""
        ...

    async def plan(self, request: OllamaLeaseRequest) -> OllamaPlacementPlan: ...

    async def provision(
        self, prepared: PreparedOllamaProvisioning
    ) -> ProvisionedOllamaResource: ...

    async def observe_provisioning(
        self,
        intent: OllamaProvisioningIntent,
        resource: ProvisionedOllamaResource,
        *,
        reconciled_after_ambiguous_create: bool = False,
    ) -> OllamaProvisioningReceipt: ...

    async def discover_provisioning_resource(
        self, intent: OllamaProvisioningIntent
    ) -> ProvisionedOllamaResource | None: ...

    async def observe(
        self,
        resource: ProvisionedOllamaResource,
        intent: OllamaProvisioningIntent,
        provisioning: OllamaProvisioningReceipt,
    ) -> OllamaReadinessObservation: ...

    async def pull_model(
        self, resource: ProvisionedOllamaResource, route_url: str, model: str
    ) -> None: ...

    async def teardown(
        self,
        resource: ProvisionedOllamaResource,
        cleanup_intent: OllamaCleanupIntent,
    ) -> OllamaCleanupReceipt: ...

    async def observe_lifecycle_billing(
        self,
        intent: OllamaProvisioningIntent,
        provisioning: OllamaProvisioningReceipt,
        cleanup: OllamaCleanupReceipt,
    ) -> OllamaLifecycleBillingObservation | None: ...


def select_ollama_plan(
    request: OllamaLeaseRequest,
    decisions: Mapping[ComputeProduct, PlacementDecision],
    *,
    non_compute_cost_policies: Mapping[OllamaLeaseMode, OllamaNonComputeCostPolicy],
    planned_at: datetime,
    serverless_max_workers: int,
    failures: Sequence[str] = (),
) -> OllamaPlacementPlan:
    """Choose the lower all-in feasible mode from one conservative cost plan."""

    require_aware(planned_at, "planned_at")
    if (
        not isinstance(serverless_max_workers, int)
        or isinstance(serverless_max_workers, bool)
        or serverless_max_workers < 1
    ):
        raise ValueError("Ollama Serverless maximum workers must be positive")
    remaining_seconds = ceil_timedelta_seconds(request.hard_deadline - planned_at)
    if remaining_seconds < 1:
        raise RunPodManagerError("Ollama cost plan has no remaining lease runtime")

    candidates: list[OllamaPlacementPlan] = []
    serverless = decisions.get(ComputeProduct.SERVERLESS)
    if serverless is not None and request.mode in {
        OllamaLeaseMode.AUTO,
        OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
    }:
        maximum_cold_starts = maximum_serverless_cold_starts(
            expected_session_seconds=request.expected_session_seconds,
            idle_tail_seconds=request.serverless_idle_tail_seconds,
        )
        if maximum_cold_starts is not None:
            policy = _required_non_compute_cost_policy(
                non_compute_cost_policies,
                OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            )
            billable = request.expected_active_seconds + maximum_cold_starts * (
                request.serverless_initialization_seconds
                + request.serverless_idle_tail_seconds
            )
            maximum_billable = max(billable, remaining_seconds * serverless_max_workers)
            estimated_compute = _rated_cost(
                serverless.offered_cost_per_hr, billable, serverless.gpu_count
            )
            maximum_compute = _rated_cost(
                serverless.offered_cost_per_hr, maximum_billable, serverless.gpu_count
            )
            candidates.append(
                OllamaPlacementPlan(
                    mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
                    resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
                    placement=serverless,
                    estimated_compute_cost=estimated_compute,
                    maximum_compute_cost=maximum_compute,
                    estimated_non_compute_cost=policy.estimated_cost_usd,
                    maximum_non_compute_cost=policy.maximum_cost_usd,
                    estimated_cost=_cost_sum(
                        estimated_compute, policy.estimated_cost_usd
                    ),
                    cost_ceiling=_cost_sum(maximum_compute, policy.maximum_cost_usd),
                    estimated_billable_seconds=billable,
                    maximum_billable_seconds=maximum_billable,
                    maximum_concurrent_workers=serverless_max_workers,
                    non_compute_components=policy.covered_components,
                    maximum_serverless_cold_starts=maximum_cold_starts,
                )
            )
    pod = decisions.get(ComputeProduct.POD)
    # A Pod bills continuously from provisioning to its hard deadline, so its
    # estimate is the expected session and its ceiling is the time actually
    # left. Once less time remains than the session needs, no honest Pod plan
    # exists: constructing one would trip the plan's own
    # maximum >= estimate invariant and raise out of select_ollama_plan. That
    # escapes _provision_requested and reconcile(), which both catch only
    # RunPodManagerError, poisoning the whole reconcile pass and stranding
    # every later lease - including READY ones holding a running Pod past its
    # deadline. Decline the candidate instead; Serverless may still be viable.
    if (
        pod is not None
        and remaining_seconds >= request.expected_session_seconds
        and request.mode
        in {
            OllamaLeaseMode.AUTO,
            OllamaLeaseMode.DEDICATED_POD,
        }
    ):
        policy = _required_non_compute_cost_policy(
            non_compute_cost_policies,
            OllamaLeaseMode.DEDICATED_POD,
        )
        estimated_compute = _rated_cost(
            pod.offered_cost_per_hr, request.expected_session_seconds, pod.gpu_count
        )
        maximum_compute = _rated_cost(
            pod.offered_cost_per_hr, remaining_seconds, pod.gpu_count
        )
        candidates.append(
            OllamaPlacementPlan(
                mode=OllamaLeaseMode.DEDICATED_POD,
                resource_type=OllamaResourceType.POD,
                placement=pod,
                estimated_compute_cost=estimated_compute,
                maximum_compute_cost=maximum_compute,
                estimated_non_compute_cost=policy.estimated_cost_usd,
                maximum_non_compute_cost=policy.maximum_cost_usd,
                estimated_cost=_cost_sum(estimated_compute, policy.estimated_cost_usd),
                cost_ceiling=_cost_sum(maximum_compute, policy.maximum_cost_usd),
                estimated_billable_seconds=request.expected_session_seconds,
                maximum_billable_seconds=remaining_seconds,
                maximum_concurrent_workers=1,
                non_compute_components=policy.covered_components,
                maximum_serverless_cold_starts=0,
            )
        )
    affordable = [
        candidate
        for candidate in candidates
        if candidate.cost_ceiling <= request.max_authorized_cost
    ]
    if not affordable:
        context = "; ".join(failures)
        estimates = ", ".join(
            f"{item.mode.value} ceiling=${item.cost_ceiling:.6f}" for item in candidates
        )
        details = "; ".join(part for part in (estimates, context) if part)
        raise RunPodManagerError(
            "No Runpod Ollama mode satisfies the live catalog and authorized cost"
            + (f": {details}" if details else "")
        )
    return min(affordable, key=lambda candidate: candidate.estimated_cost)


def _required_non_compute_cost_policy(
    policies: Mapping[OllamaLeaseMode, OllamaNonComputeCostPolicy],
    mode: OllamaLeaseMode,
) -> OllamaNonComputeCostPolicy:
    policy = policies.get(mode)
    if not isinstance(policy, OllamaNonComputeCostPolicy):
        raise RunPodManagerError(
            f"Ollama {mode.value} non-compute cost policy is not configured"
        )
    return policy


def maximum_serverless_cold_starts(
    *, expected_session_seconds: int, idle_tail_seconds: int
) -> int | None:
    """Return a conservative cold-start bound for a scale-to-zero session.

    A new worker may be required after every complete idle-tail interval.  The
    extra initial start covers the inclusive session boundary.  A zero idle
    tail cannot provide a finite invocation-independent bound, so callers must
    omit Serverless rather than quote a knowingly incomplete cost.
    """

    if idle_tail_seconds <= 0:
        return None
    return 1 + expected_session_seconds // idle_tail_seconds


def accrued_cost(lease: OllamaLease, now: datetime) -> float:
    """Return a billing-safe upper bound from provisioning wall-clock time.

    Runpod does not expose active Serverless worker-seconds through the readiness
    surface.  Until billing reconciliation supplies that value, Serverless uses
    the same continuous-rate bound as Pods: it can terminate conservatively but
    cannot authorize spend beyond the caller's cap.
    """

    if lease.offered_rate_per_hr is None or lease.provisioning_started_at is None:
        return lease.accrued_estimated_cost
    end = min(now, lease.hard_deadline)
    elapsed = end - lease.provisioning_started_at
    elapsed_microseconds = max(
        0,
        (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds,
    )
    multiplier = lease.maximum_concurrent_workers
    if multiplier is None or multiplier < 1:
        return lease.max_authorized_cost
    return billing_exposure_ratio_float(
        Decimal(str(lease.offered_rate_per_hr)),
        multiplier=(elapsed_microseconds * multiplier * lease.placement_gpu_count),
        divisor=3_600_000_000,
    )


def authorized_cost_exposure(lease: OllamaLease, now: datetime) -> float:
    """Return accrued compute plus reserved maximum non-compute authorization."""

    overhead = lease.maximum_non_compute_cost
    if overhead is None:
        # A legacy compute-only row has no proof that storage/transfer exposure
        # was reserved.  Fail closed on its next lifecycle gate while keeping
        # teardown and state inspection available.
        return lease.max_authorized_cost
    return _cost_sum(accrued_cost(lease, now), overhead)


def over_authorized_cost(lease: OllamaLease, now: datetime) -> bool:
    """Return whether a lease has burned through its authorized spend.

    This is the single source of truth for the cost half of every lifecycle
    release gate.  It is one function rather than one comparison per call site
    on purpose: the rule was previously written out five times, and a rule
    duplicated across lifecycle sites cannot be pinned by a test, because
    deleting any single copy leaves the remaining copies to mask it.  The
    deadline halves stay at the call sites, because they genuinely differ
    (only a READY lease has a meaningful idle deadline).
    """

    return authorized_cost_exposure(lease, now) >= lease.max_authorized_cost


def authorized_cost_deadline(lease: OllamaLease) -> datetime:
    """Return the first exact microsecond at which the durable cap is spent."""

    if (
        lease.provisioning_started_at is None
        or lease.offered_rate_per_hr is None
        or lease.maximum_non_compute_cost is None
        or lease.maximum_concurrent_workers is None
        or lease.maximum_concurrent_workers < 1
    ):
        return lease.hard_deadline
    remaining = Fraction(Decimal(str(lease.max_authorized_cost))) - Fraction(
        Decimal(str(lease.maximum_non_compute_cost))
    )
    if remaining <= 0:
        return lease.provisioning_started_at
    hourly = (
        Fraction(Decimal(str(lease.offered_rate_per_hr)))
        * lease.maximum_concurrent_workers
        * lease.placement_gpu_count
    )
    if hourly <= 0:
        return lease.hard_deadline
    exact_microseconds = remaining * 3_600_000_000 / hourly
    microseconds = (
        exact_microseconds.numerator + exact_microseconds.denominator - 1
    ) // exact_microseconds.denominator
    try:
        crossing = lease.provisioning_started_at + timedelta(microseconds=microseconds)
    except OverflowError:
        return lease.hard_deadline
    return min(crossing, lease.hard_deadline)


def _rated_cost(hourly_rate: float, seconds: int, gpu_count: int = 1) -> float:
    """Rate billable seconds against a catalog offer.

    ``/catalog/gpus`` prices ``price.secure``/``price.community`` **per GPU**
    (``maxCount`` is a separate attachment limit), so a placement that asks
    for more than one GPU bills that multiple of the offered rate.  This is
    orthogonal to Serverless worker scaling, which multiplies billable
    *seconds* rather than the rate.
    """
    return billing_exposure_ratio_float(
        Decimal(str(hourly_rate)),
        multiplier=seconds * gpu_count,
        divisor=3600,
    )


def _cost_sum(*values: float) -> float:
    return billing_exposure_sum_float(Decimal(str(value)) for value in values)


def resource_from_lease(lease: OllamaLease) -> ProvisionedOllamaResource:
    if (
        not lease.resource_type
        or not lease.provider_resource_id
        or not lease.resource_name
    ):
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' has no attributable provider resource"
        )
    return ProvisionedOllamaResource(
        resource_type=lease.resource_type,
        provider_resource_id=lease.provider_resource_id,
        resource_name=lease.resource_name,
    )


def resource_name(lease_id: str) -> str:
    """Build a deterministic provider name without exposing owner/workload IDs."""

    digest = hashlib.sha256(lease_id.encode()).hexdigest()[:20]
    return f"kestrel-ollama-{digest}"


def provision_attempt_id(request: OllamaLeaseRequest) -> str:
    """Return a stable idempotency identity without exposing request contents."""

    digest = hashlib.sha256(
        f"{request.lease_id}:{request.fingerprint}".encode()
    ).hexdigest()
    return f"ollama-attempt-{digest[:24]}"


def canonical_model_name(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if ":" in normalized else f"{normalized}:latest"


def sanitize_provider_error(error: Exception) -> str:
    """Remove credentials and URLs before an adapter error enters durable state."""

    value = str(error)
    value = _URL_RE.sub("[REDACTED_URL]", value)
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    value = _SECRET_RE.sub(r"\1\2[REDACTED]", value)
    return value[:1000]


def require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"Ollama {name} must be timezone-aware")


def iso_datetime(value: datetime) -> str:
    require_aware(value, "timestamp")
    return value.astimezone(UTC).isoformat()


def optional_iso_datetime(value: datetime | None) -> str | None:
    return iso_datetime(value) if value else None


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_json_object(value: str, name: str) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"Ollama {name} must be JSON text")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama {name} is invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise TypeError(f"Ollama {name} must encode an object")
    canonical = _canonical_json(parsed)
    if canonical != value:
        raise ValueError(f"Ollama {name} must use canonical JSON")
    return parsed


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def ollama_runtime_telemetry_binding_sha256(
    *,
    lease_id: str,
    request_fingerprint: str,
    provision_attempt_id: str,
    mode: OllamaLeaseMode,
    resource_type: OllamaResourceType,
    resource_name: str,
    provision_requested_at: datetime,
    immutable_image: str,
) -> str:
    """Bind one runtime process to immutable, content-safe admission facts."""

    _sha256(request_fingerprint, "request fingerprint")
    require_aware(provision_requested_at, "runtime telemetry provision time")
    for name, value in (
        ("lease_id", lease_id),
        ("provision_attempt_id", provision_attempt_id),
        ("resource_name", resource_name),
        ("immutable_image", immutable_image),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Ollama telemetry {name} must be nonempty")
    return _json_sha256(
        {
            "binding_version": 1,
            "lease_id": lease_id,
            "request_fingerprint": request_fingerprint,
            "provision_attempt_id": provision_attempt_id,
            "mode": mode.value,
            "resource_type": resource_type.value,
            "resource_name": resource_name,
            "provision_requested_at": iso_datetime(provision_requested_at),
            "immutable_image": immutable_image,
        }
    )


def runtime_telemetry_binding_for_intent(
    intent: OllamaProvisioningIntent,
) -> str:
    request = _canonical_json_object(
        intent.provider_request_json, "provider_request_json"
    )
    image = request.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("Ollama provisioning intent omitted its immutable image")
    return ollama_runtime_telemetry_binding_sha256(
        lease_id=intent.lease_id,
        request_fingerprint=intent.request_fingerprint,
        provision_attempt_id=intent.provision_attempt_id,
        mode=intent.mode,
        resource_type=intent.resource_type,
        resource_name=intent.resource_name,
        provision_requested_at=intent.provision_requested_at,
        immutable_image=image,
    )


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Ollama {name} must be a lowercase SHA-256 digest")


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"Ollama {name} fields are invalid")


def _required_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise TypeError(f"Ollama {name} must be a nonempty string")
    return item


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("Ollama optional value must be a nonempty string")
    return value


def _required_int(value: Mapping[str, Any], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"Ollama {name} must be an integer")
    return item


def _required_bool(value: Mapping[str, Any], name: str) -> bool:
    item = value.get(name)
    if not isinstance(item, bool):
        raise TypeError(f"Ollama {name} must be a boolean")
    return item


def _required_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    item = value.get(name)
    if not isinstance(item, Mapping):
        raise TypeError(f"Ollama {name} must be an object")
    return item


def _optional_datetime_value(value: Mapping[str, Any], name: str) -> datetime | None:
    item = value.get(name)
    if item is None:
        return None
    if not isinstance(item, str):
        raise TypeError(f"Ollama {name} must be timestamp text or null")
    return _datetime(item)


def _optional_int_value(value: Mapping[str, Any], name: str) -> int | None:
    if value.get(name) is None:
        return None
    return _required_int(value, name)


def _string_tuple_value(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError("Ollama stored string tuple is invalid")
    return tuple(value)


def _required_decimal(value: Mapping[str, Any], name: str) -> Decimal:
    item = value.get(name)
    if not isinstance(item, str):
        raise TypeError(f"Ollama {name} must be decimal text")
    try:
        result = Decimal(item)
    except InvalidOperation as exc:
        raise ValueError(f"Ollama {name} is invalid decimal text") from exc
    if not result.is_finite():
        raise ValueError(f"Ollama {name} must be finite")
    return result


def _datetime(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Ollama stored timestamp is invalid") from exc
    require_aware(result, "stored timestamp")
    return result.astimezone(UTC)
