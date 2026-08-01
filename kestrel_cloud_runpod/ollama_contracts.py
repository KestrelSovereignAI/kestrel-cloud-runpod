"""Typed contracts and cost policy for durable private-Ollama leases."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

from .models import (
    CloudType,
    ComputeProduct,
    PlacementDecision,
    PlacementRequirements,
    RunPodManagerError,
)

_URL_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization)(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


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


class OllamaLeaseConflictError(RunPodManagerError):
    """A stable lease ID was reused for a different request."""


class OllamaLeaseAuthorizationError(RunPodManagerError):
    """The supplied owner/workload pair does not own the lease."""


class OllamaLeaseReadinessError(RunPodManagerError):
    """Capacity did not become model-ready within its bounded lease policy."""


class OllamaLeaseTeardownError(RunPodManagerError):
    """Provider teardown failed and remains durable/retryable."""


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
        if self.max_hourly_rate is not None and (
            not isinstance(self.max_hourly_rate, (int, float))
            or isinstance(self.max_hourly_rate, bool)
            or not math.isfinite(self.max_hourly_rate)
            or self.max_hourly_rate <= 0
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

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OllamaLeaseMode):
            raise TypeError("Ollama mode must be an OllamaLeaseMode")
        if not isinstance(self.constraints, OllamaResourceConstraints):
            raise TypeError("Ollama constraints must be OllamaResourceConstraints")
        if not isinstance(self.hard_deadline, datetime):
            raise TypeError("Ollama hard_deadline must be a datetime")
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
        if (
            not isinstance(self.max_authorized_cost, (int, float))
            or isinstance(self.max_authorized_cost, bool)
            or not math.isfinite(self.max_authorized_cost)
            or self.max_authorized_cost <= 0
        ):
            raise ValueError("max_authorized_cost must be positive")
        require_aware(self.hard_deadline, "hard_deadline")

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
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class OllamaPlacementPlan:
    mode: OllamaLeaseMode
    resource_type: OllamaResourceType
    placement: PlacementDecision
    estimated_cost: float
    estimated_billable_seconds: int


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
    estimated_billable_seconds: int | None
    accrued_estimated_cost: float
    max_authorized_cost: float
    cold_start_seconds: float | None
    selected_gpu_id: str | None
    selected_gpu_pool: str | None
    selected_gpu_name: str | None
    catalog_observed_at: datetime | None
    last_provider_error: str | None
    teardown_attempts: int
    revision: int

    @property
    def public_route_url(self) -> str | None:
        """Return a route only after both provider and requested-model readiness."""

        return self.route_url if self.state is OllamaLeaseState.READY else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "owner_id": self.owner_id,
            "workload_id": self.workload_id,
            "model": self.model,
            "mode": self.mode.value if self.mode else None,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "provider_resource_id": self.provider_resource_id,
            "provision_attempt_id": self.provision_attempt_id,
            "creation_uncertain": self.creation_uncertain,
            "route_url": self.public_route_url,
            "state": self.state.value,
            "teardown_state": self.teardown_state.value,
            "created_at": iso_datetime(self.created_at),
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
            "estimated_billable_seconds": self.estimated_billable_seconds,
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
        }


class OllamaCapacityProvider(Protocol):
    async def plan(self, request: OllamaLeaseRequest) -> OllamaPlacementPlan: ...

    async def provision(
        self,
        *,
        request: OllamaLeaseRequest,
        plan: OllamaPlacementPlan,
        resource_name: str,
    ) -> ProvisionedOllamaResource: ...

    async def find_resource(
        self, *, resource_type: OllamaResourceType, resource_name: str
    ) -> ProvisionedOllamaResource | None: ...

    async def observe(
        self, resource: ProvisionedOllamaResource
    ) -> OllamaReadinessObservation: ...

    async def pull_model(
        self, resource: ProvisionedOllamaResource, route_url: str, model: str
    ) -> None: ...

    async def teardown(self, resource: ProvisionedOllamaResource) -> None: ...


def select_ollama_plan(
    request: OllamaLeaseRequest,
    decisions: Mapping[ComputeProduct, PlacementDecision],
    *,
    failures: Sequence[str] = (),
) -> OllamaPlacementPlan:
    """Choose the lower live-cost feasible interactive mode without thresholds."""

    candidates: list[OllamaPlacementPlan] = []
    serverless = decisions.get(ComputeProduct.SERVERLESS)
    if serverless is not None and request.mode in {
        OllamaLeaseMode.AUTO,
        OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
    }:
        billable = (
            request.serverless_initialization_seconds
            + request.expected_active_seconds
            + request.serverless_idle_tail_seconds
        )
        candidates.append(
            OllamaPlacementPlan(
                mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
                resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
                placement=serverless,
                estimated_cost=serverless.offered_cost_per_hr * billable / 3600,
                estimated_billable_seconds=billable,
            )
        )
    pod = decisions.get(ComputeProduct.POD)
    if pod is not None and request.mode in {
        OllamaLeaseMode.AUTO,
        OllamaLeaseMode.DEDICATED_POD,
    }:
        candidates.append(
            OllamaPlacementPlan(
                mode=OllamaLeaseMode.DEDICATED_POD,
                resource_type=OllamaResourceType.POD,
                placement=pod,
                estimated_cost=(
                    pod.offered_cost_per_hr * request.expected_session_seconds / 3600
                ),
                estimated_billable_seconds=request.expected_session_seconds,
            )
        )
    affordable = [
        candidate
        for candidate in candidates
        if candidate.estimated_cost <= request.max_authorized_cost
    ]
    if not affordable:
        context = "; ".join(failures)
        estimates = ", ".join(
            f"{item.mode.value}=${item.estimated_cost:.6f}" for item in candidates
        )
        details = "; ".join(part for part in (estimates, context) if part)
        raise RunPodManagerError(
            "No Runpod Ollama mode satisfies the live catalog and authorized cost"
            + (f": {details}" if details else "")
        )
    return min(affordable, key=lambda candidate: candidate.estimated_cost)


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
    elapsed = max(0.0, (end - lease.provisioning_started_at).total_seconds())
    return lease.offered_rate_per_hr * elapsed / 3600


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
