"""Typed contracts and cost policy for durable private-Ollama leases."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
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
        if (
            not isinstance(self.max_authorized_cost, (int, float))
            or isinstance(self.max_authorized_cost, bool)
            or not math.isfinite(self.max_authorized_cost)
            or self.max_authorized_cost <= 0
        ):
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
        if not math.isclose(
            self.estimated_cost,
            self.estimated_compute_cost + self.estimated_non_compute_cost,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Ollama estimated total cost is not derived")
        if not math.isclose(
            self.cost_ceiling,
            self.maximum_compute_cost + self.maximum_non_compute_cost,
            rel_tol=1e-12,
            abs_tol=1e-12,
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
    # How many GPUs the placement attaches. ``offered_rate_per_hr`` stays the
    # catalog's PER-GPU price, exactly as /catalog/gpus reported it, so the
    # count is carried alongside rather than folded in - a stored rate that
    # silently meant something new would misread every existing row. Defaults
    # to a single GPU, which is what every pre-existing lease had.
    placement_gpu_count: int = 1

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
    remaining_seconds = math.ceil((request.hard_deadline - planned_at).total_seconds())
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
    elapsed = max(0.0, (end - lease.provisioning_started_at).total_seconds())
    multiplier = lease.maximum_concurrent_workers
    if multiplier is None or multiplier < 1:
        return lease.max_authorized_cost
    return float(
        Decimal(str(lease.offered_rate_per_hr))
        * Decimal(str(elapsed))
        * Decimal(multiplier)
        * Decimal(lease.placement_gpu_count)
        / Decimal(3600)
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


def _rated_cost(hourly_rate: float, seconds: int, gpu_count: int = 1) -> float:
    """Rate billable seconds against a catalog offer.

    ``/catalog/gpus`` prices ``price.secure``/``price.community`` **per GPU**
    (``maxCount`` is a separate attachment limit), so a placement that asks
    for more than one GPU bills that multiple of the offered rate.  This is
    orthogonal to Serverless worker scaling, which multiplies billable
    *seconds* rather than the rate.
    """
    return float(
        Decimal(str(hourly_rate))
        * Decimal(seconds)
        * Decimal(gpu_count)
        / Decimal(3600)
    )


def _cost_sum(*values: float) -> float:
    return float(sum((Decimal(str(value)) for value in values), Decimal(0)))


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
