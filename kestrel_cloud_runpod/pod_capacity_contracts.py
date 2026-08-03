"""Durable contracts for the canonical billable Runpod Pod lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from enum import Enum
from itertools import pairwise
from typing import Any, Protocol

from pydantic import SecretStr

from .models import (
    CloudType,
    ComputeProduct,
    PlacementDecision,
    PlacementRequirements,
    RunPodManagerError,
)

TRAINING_PROFILE_IDS = ("training", "training-h100", "training-flex")


class TrainingPodSource(str, Enum):
    """How a training Pod entered the durable lifecycle."""

    CONFIGURED_PERSISTENT = "configured_persistent"
    STOPPED_REUSE = "stopped_reuse"
    CREATED = "created"


class TrainingPodOwnership(str, Enum):
    """Whether this lease may stop the Pod when its work cannot continue."""

    PROVISIONAL = "provisional"
    OWNED = "owned"
    PREEXISTING_RUNNING = "preexisting_running"


class TrainingPodState(str, Enum):
    """Restart-safe training capacity and workload states."""

    REQUESTED = "requested"
    STARTING = "starting"
    READY = "ready"
    JOB_SUBMITTED = "job_submitted"
    JOB_COMPLETED = "job_completed"
    RESULT_RETRIEVED = "result_retrieved"
    CANCEL_REQUESTED = "cancel_requested"
    RECONCILE_REQUIRED = "reconcile_required"
    RELEASING = "releasing"
    RELEASED = "released"


class TrainingPodCleanupState(str, Enum):
    """Cleanup outcome retained separately from the workload state."""

    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    RETRYABLE_FAILURE = "retryable_failure"
    COMPLETE = "complete"
    NOT_OWNED = "not_owned"


class PodCapacityBillingState(str, Enum):
    """Whether the provider has supplied final authoritative Pod billing."""

    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    AUTHORITATIVE = "authoritative"
    UNRESOLVED = "unresolved"


class CatalogPodWorkloadState(str, Enum):
    """Content-free observation of the disposable worker process."""

    NOT_SUBMITTED = "not_submitted"
    IDLE = "idle"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PodCapacityConstraints:
    """Provider-neutral hard constraints for a dedicated Pod quote."""

    min_vram_gb: int
    gpu_count: int = 1
    cloud: CloudType = CloudType.SECURE
    min_cuda_version: str | None = None
    allowed_gpu_ids: tuple[str, ...] = ()
    allowed_data_center_ids: tuple[str, ...] = ()
    max_hourly_rate_usd: Decimal | None = None
    benchmark_id: str = "catalog-lora"
    allowed_products: tuple[ComputeProduct, ...] = (ComputeProduct.POD,)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_vram_gb, int)
            or isinstance(self.min_vram_gb, bool)
            or self.min_vram_gb < 1
        ):
            raise ValueError("Pod capacity min_vram_gb must be a positive integer")
        if (
            not isinstance(self.gpu_count, int)
            or isinstance(self.gpu_count, bool)
            or self.gpu_count < 1
        ):
            raise ValueError("Pod capacity gpu_count must be a positive integer")
        if not isinstance(self.cloud, CloudType):
            raise TypeError("Pod capacity cloud must be a CloudType")
        if self.allowed_products != (ComputeProduct.POD,):
            raise ValueError("Pod capacity leases support only the POD product")
        if not self.benchmark_id.strip():
            raise ValueError("Pod capacity benchmark_id must be non-empty")
        if self.max_hourly_rate_usd is not None and (
            not self.max_hourly_rate_usd.is_finite() or self.max_hourly_rate_usd <= 0
        ):
            raise ValueError("Pod capacity hourly-rate limit must be positive")

    def placement_requirements(self) -> PlacementRequirements:
        return PlacementRequirements(
            product=ComputeProduct.POD,
            min_vram_gb=self.min_vram_gb,
            gpu_count=self.gpu_count,
            cloud=self.cloud,
            min_cuda_version=self.min_cuda_version,
            max_cost_per_hr=(
                float(self.max_hourly_rate_usd)
                if self.max_hourly_rate_usd is not None
                else None
            ),
            allowed_gpu_ids=self.allowed_gpu_ids,
            allowed_data_center_ids=self.allowed_data_center_ids,
            benchmark_id=self.benchmark_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_vram_gb": self.min_vram_gb,
            "gpu_count": self.gpu_count,
            "cloud": self.cloud.value,
            "min_cuda_version": self.min_cuda_version,
            "allowed_gpu_ids": list(self.allowed_gpu_ids),
            "allowed_data_center_ids": list(self.allowed_data_center_ids),
            "max_hourly_rate_usd": decimal_text(self.max_hourly_rate_usd),
            "benchmark_id": self.benchmark_id,
            "allowed_products": [item.value for item in self.allowed_products],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodCapacityConstraints:
        return cls(
            min_vram_gb=_required_int(value, "min_vram_gb"),
            gpu_count=_required_int(value, "gpu_count"),
            cloud=CloudType(_required_string(value, "cloud")),
            min_cuda_version=_optional_string(value.get("min_cuda_version")),
            allowed_gpu_ids=_string_tuple(value.get("allowed_gpu_ids", ())),
            allowed_data_center_ids=_string_tuple(
                value.get("allowed_data_center_ids", ())
            ),
            max_hourly_rate_usd=_optional_decimal(value.get("max_hourly_rate_usd")),
            benchmark_id=_required_string(value, "benchmark_id"),
            allowed_products=tuple(
                ComputeProduct(item)
                for item in _string_tuple(value.get("allowed_products", ()))
            ),
        )


@dataclass(frozen=True)
class PodCapacityQuoteRequest:
    """Content-free inputs for a live dedicated-Pod capacity quote."""

    constraints: PodCapacityConstraints
    workload_kind: str
    parameters_sha256: str
    estimated_startup_seconds: int
    estimated_execution_seconds: int
    maximum_runtime_seconds: int
    quote_ttl_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.constraints, PodCapacityConstraints):
            raise TypeError("Pod quote constraints must be PodCapacityConstraints")
        _safe_identifier(self.workload_kind, "workload_kind")
        _sha256(self.parameters_sha256, "parameters_sha256")
        for name, value, allow_zero in (
            ("estimated_startup_seconds", self.estimated_startup_seconds, True),
            ("estimated_execution_seconds", self.estimated_execution_seconds, False),
            ("maximum_runtime_seconds", self.maximum_runtime_seconds, False),
            ("quote_ttl_seconds", self.quote_ttl_seconds, False),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if allow_zero else 1)
            ):
                raise ValueError(f"Pod quote {name} is invalid")
        if (
            self.estimated_startup_seconds + self.estimated_execution_seconds
            > self.maximum_runtime_seconds
        ):
            raise ValueError("Pod quote estimate exceeds maximum runtime")


@dataclass(frozen=True)
class PodCapacityQuote:
    """Exact live v2 offer and bounded timing/cost evidence."""

    schema_version: int
    capability_version: str
    provider_quote_id: str
    workload_kind: str
    parameters_sha256: str
    constraints: PodCapacityConstraints
    gpu_type_id: str
    gpu_display_name: str
    hourly_cost_usd: Decimal
    estimated_cost_usd: Decimal
    cost_ceiling_usd: Decimal
    estimated_startup_seconds: int
    estimated_execution_seconds: int
    maximum_runtime_seconds: int
    observed_at: datetime
    expires_at: datetime
    placement: PlacementDecision

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("Pod capacity quote must negotiate schema 3")
        for name, value in (
            ("capability_version", self.capability_version),
            ("provider_quote_id", self.provider_quote_id),
            ("workload_kind", self.workload_kind),
            ("gpu_type_id", self.gpu_type_id),
            ("gpu_display_name", self.gpu_display_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Pod capacity quote {name} must be non-empty")
        _sha256(self.parameters_sha256, "parameters_sha256")
        for name, value in (
            ("hourly_cost_usd", self.hourly_cost_usd),
            ("estimated_cost_usd", self.estimated_cost_usd),
            ("cost_ceiling_usd", self.cost_ceiling_usd),
        ):
            if not value.is_finite() or value <= 0:
                raise ValueError(f"Pod capacity quote {name} must be positive")
        if self.estimated_cost_usd > self.cost_ceiling_usd:
            raise ValueError("Pod capacity estimated cost exceeds its ceiling")
        for name, value, allow_zero in (
            ("estimated_startup_seconds", self.estimated_startup_seconds, True),
            ("estimated_execution_seconds", self.estimated_execution_seconds, False),
            ("maximum_runtime_seconds", self.maximum_runtime_seconds, False),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if allow_zero else 1)
            ):
                raise ValueError(f"Pod capacity quote {name} is invalid")
        estimated_seconds = (
            self.estimated_startup_seconds + self.estimated_execution_seconds
        )
        if estimated_seconds > self.maximum_runtime_seconds:
            raise ValueError("Pod capacity quote estimate exceeds maximum runtime")
        expected_estimate = pod_cost_usd(self.hourly_cost_usd, estimated_seconds)
        expected_ceiling = pod_cost_usd(
            self.hourly_cost_usd, self.maximum_runtime_seconds
        )
        if (
            self.estimated_cost_usd != expected_estimate
            or self.cost_ceiling_usd != expected_ceiling
        ):
            raise ValueError("Pod capacity quote cost does not match its rate/timing")
        require_aware(self.observed_at, "quote observed_at")
        require_aware(self.expires_at, "quote expires_at")
        if self.expires_at <= self.observed_at:
            raise ValueError("Pod capacity quote expiry is invalid")
        if self.placement.gpu_id != self.gpu_type_id:
            raise ValueError("Pod capacity quote GPU identity is inconsistent")
        if Decimal(str(self.placement.offered_cost_per_hr)) != self.hourly_cost_usd:
            raise ValueError("Pod capacity quote hourly rate is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability_version": self.capability_version,
            "provider_quote_id": self.provider_quote_id,
            "workload_kind": self.workload_kind,
            "parameters_sha256": self.parameters_sha256,
            "constraints": self.constraints.to_dict(),
            "gpu_type_id": self.gpu_type_id,
            "gpu_display_name": self.gpu_display_name,
            "hourly_cost_usd": decimal_text(self.hourly_cost_usd),
            "estimated_cost_usd": decimal_text(self.estimated_cost_usd),
            "cost_ceiling_usd": decimal_text(self.cost_ceiling_usd),
            "estimated_startup_seconds": self.estimated_startup_seconds,
            "estimated_execution_seconds": self.estimated_execution_seconds,
            "maximum_runtime_seconds": self.maximum_runtime_seconds,
            "observed_at": iso_datetime(self.observed_at),
            "expires_at": iso_datetime(self.expires_at),
            "placement": _placement_to_dict(self.placement),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodCapacityQuote:
        constraints = PodCapacityConstraints.from_dict(
            _required_mapping(value, "constraints")
        )
        return cls(
            schema_version=_required_int(value, "schema_version"),
            capability_version=_required_string(value, "capability_version"),
            provider_quote_id=_required_string(value, "provider_quote_id"),
            workload_kind=_required_string(value, "workload_kind"),
            parameters_sha256=_required_string(value, "parameters_sha256"),
            constraints=constraints,
            gpu_type_id=_required_string(value, "gpu_type_id"),
            gpu_display_name=_required_string(value, "gpu_display_name"),
            hourly_cost_usd=_required_decimal(value, "hourly_cost_usd"),
            estimated_cost_usd=_required_decimal(value, "estimated_cost_usd"),
            cost_ceiling_usd=_required_decimal(value, "cost_ceiling_usd"),
            estimated_startup_seconds=_required_int(value, "estimated_startup_seconds"),
            estimated_execution_seconds=_required_int(
                value, "estimated_execution_seconds"
            ),
            maximum_runtime_seconds=_required_int(value, "maximum_runtime_seconds"),
            observed_at=_datetime(_required_string(value, "observed_at")),
            expires_at=_datetime(_required_string(value, "expires_at")),
            placement=_placement_from_dict(
                _required_mapping(value, "placement"), constraints
            ),
        )


@dataclass(frozen=True, repr=False)
class CatalogAttemptCapability:
    """Restart-recoverable scoped bearer returned by an injected secret store."""

    secret_id: str
    bearer_token: SecretStr
    token_sha256: str
    expires_at: datetime

    def __post_init__(self) -> None:
        _safe_identifier(self.secret_id, "capability secret_id")
        token = self.bearer_token.get_secret_value()
        if not re.fullmatch(r"[A-Za-z0-9._~-]{32,4096}", token):
            raise ValueError("Catalog Pod bearer token is not a strong scoped token")
        _sha256(self.token_sha256, "capability token_sha256")
        if not hashlib.sha256(token.encode()).hexdigest() == self.token_sha256:
            raise ValueError("Catalog Pod bearer digest does not match its token")
        require_aware(self.expires_at, "capability expiry")


class CatalogAttemptCapabilityStore(Protocol):
    """Encrypted durable capability boundary supplied by the host service."""

    async def load_or_create(
        self, attempt_id: str, expires_at: datetime
    ) -> CatalogAttemptCapability: ...

    async def load(self, attempt_id: str) -> CatalogAttemptCapability | None: ...

    async def revoke(self, attempt_id: str) -> None: ...


@dataclass(frozen=True)
class CatalogPodCapacityRequest:
    """Secret-free, idempotent request for one catalog attempt Pod."""

    capacity_id: str
    cleanup_family_id: str
    owner_id: str
    workload_id: str
    attempt_id: str
    idempotency_key: str
    request_sha256: str
    workload_kind: str
    parameters_sha256: str
    image_reference: str
    profile_id: str
    quote: PodCapacityQuote
    accepted_max_cost_usd: Decimal
    created_at: datetime
    readiness_deadline: datetime
    hard_deadline: datetime
    bearer_expires_at: datetime
    idle_timeout_seconds: int
    cleanup_grace_seconds: int
    attempt_environment: Mapping[str, str]

    def __post_init__(self) -> None:
        for name, value in (
            ("capacity_id", self.capacity_id),
            ("cleanup_family_id", self.cleanup_family_id),
            ("owner_id", self.owner_id),
            ("workload_id", self.workload_id),
            ("attempt_id", self.attempt_id),
            ("idempotency_key", self.idempotency_key),
            ("workload_kind", self.workload_kind),
            ("profile_id", self.profile_id),
        ):
            _safe_identifier(value, name)
        _sha256(self.request_sha256, "request_sha256")
        _sha256(self.parameters_sha256, "parameters_sha256")
        if self.parameters_sha256 != self.quote.parameters_sha256:
            raise ValueError("Catalog Pod parameters do not match the accepted quote")
        if self.workload_kind != self.quote.workload_kind:
            raise ValueError("Catalog Pod workload kind does not match the quote")
        if self.accepted_max_cost_usd != self.quote.cost_ceiling_usd:
            raise ValueError(
                "Catalog Pod accepted maximum must equal the quote cost ceiling"
            )
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", self.image_reference):
            raise ValueError("Catalog Pod worker image must be pinned by digest")
        for name, value in (
            ("created_at", self.created_at),
            ("readiness_deadline", self.readiness_deadline),
            ("hard_deadline", self.hard_deadline),
            ("bearer_expires_at", self.bearer_expires_at),
        ):
            require_aware(value, name)
        if self.quote.expires_at <= self.created_at:
            raise ValueError("Catalog Pod quote has expired")
        if not (
            self.created_at
            < self.readiness_deadline
            < self.hard_deadline
            < self.bearer_expires_at
        ):
            raise ValueError("Catalog Pod deadlines or bearer expiry are invalid")
        if self.bearer_expires_at > self.created_at + timedelta(hours=24):
            raise ValueError("Catalog Pod bearer expiry cannot exceed 24 hours")
        for name, value in (
            ("idle_timeout_seconds", self.idle_timeout_seconds),
            ("cleanup_grace_seconds", self.cleanup_grace_seconds),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Catalog Pod {name} must be positive")
        if self.bearer_expires_at < self.hard_deadline + timedelta(
            seconds=self.cleanup_grace_seconds
        ):
            raise ValueError(
                "Catalog Pod bearer does not cover the cleanup grace period"
            )
        if self.hard_deadline > self.created_at + _seconds(
            self.quote.maximum_runtime_seconds
        ):
            raise ValueError("Catalog Pod hard deadline exceeds quoted maximum runtime")
        _validate_attempt_environment(self.attempt_environment)

    @property
    def resource_name(self) -> str:
        return durable_pod_capacity_name(self.capacity_id, self.fingerprint)

    @property
    def fingerprint(self) -> str:
        payload = {
            "capacity_id": self.capacity_id,
            "cleanup_family_id": self.cleanup_family_id,
            "owner_id": self.owner_id,
            "workload_id": self.workload_id,
            "attempt_id": self.attempt_id,
            "idempotency_key": self.idempotency_key,
            "request_sha256": self.request_sha256,
            "workload_kind": self.workload_kind,
            "parameters_sha256": self.parameters_sha256,
            "image_reference": self.image_reference,
            "profile_id": self.profile_id,
            "provider_quote_id": self.quote.provider_quote_id,
            "accepted_max_cost_usd": decimal_text(self.accepted_max_cost_usd),
            "created_at": iso_datetime(self.created_at),
            "readiness_deadline": iso_datetime(self.readiness_deadline),
            "hard_deadline": iso_datetime(self.hard_deadline),
            "bearer_expires_at": iso_datetime(self.bearer_expires_at),
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "cleanup_grace_seconds": self.cleanup_grace_seconds,
            "attempt_environment_keys": sorted(self.attempt_environment),
        }
        return _json_sha256(payload)


@dataclass(frozen=True)
class PodCapacitySpec:
    """Persisted generic identity, quote, policy, and non-secret capability data."""

    request: CatalogPodCapacityRequest
    capability_secret_id: str
    capability_token_sha256: str
    capability_expires_at: datetime
    attempt_environment_sha256: str

    def __post_init__(self) -> None:
        _safe_identifier(self.capability_secret_id, "capability_secret_id")
        _sha256(self.capability_token_sha256, "capability_token_sha256")
        _sha256(self.attempt_environment_sha256, "attempt_environment_sha256")
        require_aware(self.capability_expires_at, "capability_expires_at")
        if self.capability_expires_at != self.request.bearer_expires_at:
            raise ValueError("Persisted capability expiry does not match the request")

    def to_dict(self) -> dict[str, Any]:
        request = self.request
        return {
            "request": {
                "capacity_id": request.capacity_id,
                "cleanup_family_id": request.cleanup_family_id,
                "owner_id": request.owner_id,
                "workload_id": request.workload_id,
                "attempt_id": request.attempt_id,
                "idempotency_key": request.idempotency_key,
                "request_sha256": request.request_sha256,
                "workload_kind": request.workload_kind,
                "parameters_sha256": request.parameters_sha256,
                "image_reference": request.image_reference,
                "profile_id": request.profile_id,
                "quote": request.quote.to_dict(),
                "accepted_max_cost_usd": decimal_text(request.accepted_max_cost_usd),
                "created_at": iso_datetime(request.created_at),
                "readiness_deadline": iso_datetime(request.readiness_deadline),
                "hard_deadline": iso_datetime(request.hard_deadline),
                "bearer_expires_at": iso_datetime(request.bearer_expires_at),
                "idle_timeout_seconds": request.idle_timeout_seconds,
                "cleanup_grace_seconds": request.cleanup_grace_seconds,
                "attempt_environment": {
                    key: "[REDACTED]" for key in sorted(request.attempt_environment)
                },
            },
            "capability_secret_id": self.capability_secret_id,
            "capability_token_sha256": self.capability_token_sha256,
            "capability_expires_at": iso_datetime(self.capability_expires_at),
            "attempt_environment_sha256": self.attempt_environment_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodCapacitySpec:
        request_value = _required_mapping(value, "request")
        request = CatalogPodCapacityRequest(
            capacity_id=_required_string(request_value, "capacity_id"),
            cleanup_family_id=_required_string(request_value, "cleanup_family_id"),
            owner_id=_required_string(request_value, "owner_id"),
            workload_id=_required_string(request_value, "workload_id"),
            attempt_id=_required_string(request_value, "attempt_id"),
            idempotency_key=_required_string(request_value, "idempotency_key"),
            request_sha256=_required_string(request_value, "request_sha256"),
            workload_kind=_required_string(request_value, "workload_kind"),
            parameters_sha256=_required_string(request_value, "parameters_sha256"),
            image_reference=_required_string(request_value, "image_reference"),
            profile_id=_required_string(request_value, "profile_id"),
            quote=PodCapacityQuote.from_dict(_required_mapping(request_value, "quote")),
            accepted_max_cost_usd=_required_decimal(
                request_value, "accepted_max_cost_usd"
            ),
            created_at=_datetime(_required_string(request_value, "created_at")),
            readiness_deadline=_datetime(
                _required_string(request_value, "readiness_deadline")
            ),
            hard_deadline=_datetime(_required_string(request_value, "hard_deadline")),
            bearer_expires_at=_datetime(
                _required_string(request_value, "bearer_expires_at")
            ),
            idle_timeout_seconds=_required_int(request_value, "idle_timeout_seconds"),
            cleanup_grace_seconds=_required_int(request_value, "cleanup_grace_seconds"),
            attempt_environment={
                key: "[REDACTED]"
                for key in _string_tuple(
                    tuple(
                        _required_mapping(request_value, "attempt_environment").keys()
                    )
                )
            },
        )
        return cls(
            request=request,
            capability_secret_id=_required_string(value, "capability_secret_id"),
            capability_token_sha256=_required_string(value, "capability_token_sha256"),
            capability_expires_at=_datetime(
                _required_string(value, "capability_expires_at")
            ),
            attempt_environment_sha256=_required_string(
                value, "attempt_environment_sha256"
            ),
        )


@dataclass(frozen=True)
class PodBillingReceipt:
    """Content-free final provider billing evidence for one exact Pod."""

    provider_billing_id: str
    provider_pod_id: str | None
    billed_from: datetime
    billed_until: datetime
    billed_seconds: int
    hourly_price_usd: Decimal
    actual_cost_usd: Decimal
    reconciled_at: datetime

    def __post_init__(self) -> None:
        _safe_identifier(self.provider_billing_id, "provider_billing_id")
        if self.provider_pod_id is not None:
            _safe_identifier(self.provider_pod_id, "provider_pod_id")
        for name, value in (
            ("billed_from", self.billed_from),
            ("billed_until", self.billed_until),
            ("reconciled_at", self.reconciled_at),
        ):
            require_aware(value, name)
        if (
            isinstance(self.billed_seconds, bool)
            or not isinstance(self.billed_seconds, int)
            or self.billed_seconds < 0
            or self.billed_until < self.billed_from
        ):
            raise ValueError("Pod billing interval is invalid")
        expected_seconds = int((self.billed_until - self.billed_from).total_seconds())
        if self.billed_seconds != expected_seconds:
            raise ValueError(
                "Pod billed seconds must equal the truncated billing interval"
            )
        if self.reconciled_at < self.billed_until:
            raise ValueError(
                "Pod billing cannot be reconciled before its interval ends"
            )
        if (
            not self.hourly_price_usd.is_finite()
            or self.hourly_price_usd <= 0
            or not self.actual_cost_usd.is_finite()
            or self.actual_cost_usd < 0
        ):
            raise ValueError("Pod billing amount is invalid")
        if self.actual_cost_usd > 0 and self.provider_pod_id is None:
            raise ValueError("Nonzero Pod cost requires a provider Pod identity")
        if self.provider_pod_id is None and (
            self.billed_from != self.billed_until
            or self.billed_seconds != 0
            or self.actual_cost_usd != 0
        ):
            raise ValueError(
                "Missing provider Pod identity is valid only before billable capacity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_billing_id": self.provider_billing_id,
            "provider_pod_id": self.provider_pod_id,
            "billed_from": iso_datetime(self.billed_from),
            "billed_until": iso_datetime(self.billed_until),
            "billed_seconds": self.billed_seconds,
            "hourly_price_usd": decimal_text(self.hourly_price_usd),
            "actual_cost_usd": decimal_text(self.actual_cost_usd),
            "reconciled_at": iso_datetime(self.reconciled_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodBillingReceipt:
        return cls(
            provider_billing_id=_required_string(value, "provider_billing_id"),
            provider_pod_id=_optional_string(value.get("provider_pod_id")),
            billed_from=_datetime(_required_string(value, "billed_from")),
            billed_until=_datetime(_required_string(value, "billed_until")),
            billed_seconds=_required_int(value, "billed_seconds"),
            hourly_price_usd=_required_decimal(value, "hourly_price_usd"),
            actual_cost_usd=_required_decimal(value, "actual_cost_usd"),
            reconciled_at=_datetime(_required_string(value, "reconciled_at")),
        )


@dataclass(frozen=True)
class PodRealizedPlacement:
    """Realized v2 placement plus the catalog name bound by exact GPU ID."""

    provider_pod_id: str
    gpu_type_id: str
    gpu_display_name: str
    gpu_count: int
    cloud: CloudType
    data_center_id: str
    hourly_rate_usd: Decimal
    observed_at: datetime

    def __post_init__(self) -> None:
        for name, value in (("provider_pod_id", self.provider_pod_id),):
            _safe_identifier(value, name)
        _provider_identifier(self.data_center_id, "data_center_id")
        _display_name(self.gpu_type_id, "gpu_type_id")
        _display_name(self.gpu_display_name, "gpu_display_name")
        if (
            not isinstance(self.gpu_count, int)
            or isinstance(self.gpu_count, bool)
            or self.gpu_count < 1
        ):
            raise ValueError("Realized Pod GPU count must be positive")
        if not isinstance(self.cloud, CloudType):
            raise TypeError("Realized Pod cloud must be a CloudType")
        if not self.hourly_rate_usd.is_finite() or self.hourly_rate_usd <= 0:
            raise ValueError("Realized Pod hourly rate must be positive")
        require_aware(self.observed_at, "realized placement observed_at")

    def validate_against(self, quote: PodCapacityQuote) -> None:
        """Reject a realized Pod outside the exact accepted quote constraints."""

        if (
            self.gpu_type_id != quote.gpu_type_id
            or self.gpu_display_name != quote.gpu_display_name
            or self.gpu_count != quote.constraints.gpu_count
            or self.cloud is not quote.constraints.cloud
            or self.hourly_rate_usd > quote.hourly_cost_usd
            or (
                quote.constraints.allowed_data_center_ids
                and self.data_center_id not in quote.constraints.allowed_data_center_ids
            )
        ):
            raise ValueError("Realized Pod placement does not match accepted quote")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_pod_id": self.provider_pod_id,
            "gpu_type_id": self.gpu_type_id,
            "gpu_display_name": self.gpu_display_name,
            "gpu_count": self.gpu_count,
            "cloud": self.cloud.value,
            "data_center_id": self.data_center_id,
            "hourly_rate_usd": decimal_text(self.hourly_rate_usd),
            "observed_at": iso_datetime(self.observed_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodRealizedPlacement:
        _exact_keys(
            value,
            {
                "provider_pod_id",
                "gpu_type_id",
                "gpu_display_name",
                "gpu_count",
                "cloud",
                "data_center_id",
                "hourly_rate_usd",
                "observed_at",
            },
            "realized placement",
        )
        return cls(
            provider_pod_id=_required_string(value, "provider_pod_id"),
            gpu_type_id=_required_string(value, "gpu_type_id"),
            gpu_display_name=_required_string(value, "gpu_display_name"),
            gpu_count=_required_int(value, "gpu_count"),
            cloud=CloudType(_required_string(value, "cloud")),
            data_center_id=_required_string(value, "data_center_id"),
            hourly_rate_usd=_required_decimal(value, "hourly_rate_usd"),
            observed_at=_datetime(_required_string(value, "observed_at")),
        )


_WORKER_TIMING_FIELDS = frozenset(
    {
        "image_pull_and_container_boot_seconds",
        "image_pull_seconds",
        "container_boot_seconds",
        "model_load_seconds",
        "execution_seconds",
        "training_seconds",
        "artifact_upload_seconds",
    }
)
_WORKER_METRIC_FIELDS = frozenset(
    {"peak_vram_bytes", "peak_host_ram_bytes", "gpu_seconds", "idle_seconds"}
)


@dataclass(frozen=True)
class CatalogWorkerEvidence:
    """Strict content-free telemetry attested by one exact catalog worker."""

    schema_version: int
    attempt_id: str
    request_sha256: str
    image_digest: str
    container_process_started_at: datetime | None
    image_pull_and_container_boot_seconds: Decimal | None
    image_pull_seconds: Decimal | None
    container_boot_seconds: Decimal | None
    model_load_seconds: Decimal | None
    execution_seconds: Decimal | None
    training_seconds: Decimal | None
    artifact_upload_seconds: Decimal | None
    peak_vram_bytes: int | None
    peak_host_ram_bytes: int | None
    gpu_seconds: Decimal | None
    idle_seconds: Decimal | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Catalog worker evidence must use schema 1")
        _safe_identifier(self.attempt_id, "worker evidence attempt_id")
        _sha256(self.request_sha256, "worker evidence request_sha256")
        _image_digest(self.image_digest)
        if self.container_process_started_at is not None:
            require_aware(
                self.container_process_started_at,
                "worker evidence container_process_started_at",
            )
        for name in _WORKER_TIMING_FIELDS:
            value = getattr(self, name)
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError(
                    f"Catalog worker {name} must be finite and nonnegative"
                )
        for name in ("peak_vram_bytes", "peak_host_ram_bytes"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"Catalog worker {name} must be nonnegative")
        for name in ("gpu_seconds", "idle_seconds"):
            value = getattr(self, name)
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError(
                    f"Catalog worker {name} must be finite and nonnegative"
                )
        split = (self.image_pull_seconds, self.container_boot_seconds)
        if (split[0] is None) != (split[1] is None):
            raise ValueError(
                "Catalog worker image-pull and container-boot split must be complete"
            )
        if split[0] is not None and split[1] is not None:
            if self.image_pull_and_container_boot_seconds is None:
                raise ValueError(
                    "Catalog worker combined startup timing is required with a split"
                )
            if self.image_pull_and_container_boot_seconds != split[0] + split[1]:
                raise ValueError(
                    "Catalog worker combined startup timing contradicts its split"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "request_sha256": self.request_sha256,
            "image_digest": self.image_digest,
            "container_process_started_at": (
                iso_datetime(self.container_process_started_at)
                if self.container_process_started_at is not None
                else None
            ),
            "timings_seconds": {
                name: decimal_text(getattr(self, name))
                for name in sorted(_WORKER_TIMING_FIELDS)
            },
            "metrics": {
                "peak_vram_bytes": self.peak_vram_bytes,
                "peak_host_ram_bytes": self.peak_host_ram_bytes,
                "gpu_seconds": decimal_text(self.gpu_seconds),
                "idle_seconds": decimal_text(self.idle_seconds),
            },
        }

    @classmethod
    def from_envelope(cls, value: Mapping[str, Any]) -> CatalogWorkerEvidence:
        """Parse an allowlisted envelope; arbitrary private result fields fail."""

        _exact_keys(
            value,
            {
                "schema_version",
                "attempt_id",
                "request_sha256",
                "image_digest",
                "container_process_started_at",
                "timings_seconds",
                "metrics",
            },
            "worker evidence",
        )
        timings = _required_mapping(value, "timings_seconds")
        metrics = _required_mapping(value, "metrics")
        _exact_keys(timings, _WORKER_TIMING_FIELDS, "worker evidence timings")
        _exact_keys(metrics, _WORKER_METRIC_FIELDS, "worker evidence metrics")
        raw_started = value.get("container_process_started_at")
        if raw_started is not None and not isinstance(raw_started, str):
            raise ValueError(
                "Catalog worker container_process_started_at must be a timestamp or null"
            )
        return cls(
            schema_version=_required_int(value, "schema_version"),
            attempt_id=_required_string(value, "attempt_id"),
            request_sha256=_required_string(value, "request_sha256"),
            image_digest=_required_string(value, "image_digest"),
            container_process_started_at=(
                _datetime(raw_started) if isinstance(raw_started, str) else None
            ),
            **{
                name: _envelope_optional_decimal(timings.get(name), name)
                for name in _WORKER_TIMING_FIELDS
            },
            peak_vram_bytes=_optional_nonnegative_int(
                metrics.get("peak_vram_bytes"), "peak_vram_bytes"
            ),
            peak_host_ram_bytes=_optional_nonnegative_int(
                metrics.get("peak_host_ram_bytes"), "peak_host_ram_bytes"
            ),
            gpu_seconds=_envelope_optional_decimal(
                metrics.get("gpu_seconds"), "gpu_seconds"
            ),
            idle_seconds=_envelope_optional_decimal(
                metrics.get("idle_seconds"), "idle_seconds"
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CatalogWorkerEvidence:
        """Load the canonical JSON representation from durable storage."""

        return cls.from_envelope(value)


@dataclass(frozen=True)
class PodCapacityLifecycleEvidence:
    """First-observation timestamps; unavailable phases remain explicit nulls."""

    reservation_at: datetime
    provider_create_accepted_at: datetime | None = None
    provider_adopted_at: datetime | None = None
    first_running_observed_at: datetime | None = None
    worker_ready_at: datetime | None = None
    workload_submitted_at: datetime | None = None
    workload_running_at: datetime | None = None
    workload_terminal_at: datetime | None = None
    stop_confirmed_at: datetime | None = None
    billing_reconciled_at: datetime | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None:
                require_aware(value, f"capacity evidence {name}")
        # These observations share the host service clock. Missing intermediate
        # phases are valid, but timestamps that exist cannot run backwards.
        ordered = (
            self.reservation_at,
            self.provider_create_accepted_at or self.provider_adopted_at,
            self.first_running_observed_at,
            self.worker_ready_at,
            self.workload_submitted_at,
            self.workload_running_at,
            self.workload_terminal_at,
            self.stop_confirmed_at,
            self.billing_reconciled_at,
        )
        observed = tuple(value for value in ordered if value is not None)
        if any(later < earlier for earlier, later in pairwise(observed)):
            raise ValueError("Pod capacity host lifecycle timestamps must be monotonic")
        if (
            self.provider_create_accepted_at is not None
            and self.provider_adopted_at is not None
        ):
            raise ValueError(
                "Pod capacity cannot be both create-accepted and recovery-adopted"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: iso_datetime(value) if value is not None else None
            for name, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodCapacityLifecycleEvidence:
        expected = set(cls.__dataclass_fields__)
        _exact_keys(value, expected, "capacity lifecycle evidence")
        parsed: dict[str, datetime | None] = {}
        for name in expected:
            item = value.get(name)
            if item is not None and not isinstance(item, str):
                raise RunPodManagerError(
                    f"Stored capacity lifecycle evidence {name} is invalid"
                )
            parsed[name] = _datetime(item) if item is not None else None
        reservation = parsed.pop("reservation_at")
        if reservation is None:
            raise RunPodManagerError(
                "Stored capacity lifecycle evidence reservation_at is missing"
            )
        return cls(reservation_at=reservation, **parsed)


@dataclass(frozen=True)
class PodCapacityEvidence:
    """Immutable, versioned, content-free proof for one disposable Pod attempt."""

    schema_version: int
    capacity_id: str
    resource_name: str
    provider_quote_id: str
    catalog_observed_at: datetime
    attempt_id: str
    request_sha256: str
    image_digest: str
    accepted_gpu_type_id: str
    accepted_gpu_display_name: str
    accepted_gpu_count: int
    accepted_cloud: CloudType
    accepted_hourly_rate_usd: Decimal
    lifecycle: PodCapacityLifecycleEvidence
    realized_placement: PodRealizedPlacement | None = None
    worker: CatalogWorkerEvidence | None = None
    billing: PodBillingReceipt | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Pod capacity evidence must use schema 1")
        for name, value in (
            ("capacity_id", self.capacity_id),
            ("resource_name", self.resource_name),
            ("provider_quote_id", self.provider_quote_id),
            ("attempt_id", self.attempt_id),
        ):
            _safe_identifier(value, name)
        _display_name(self.accepted_gpu_type_id, "accepted_gpu_type_id")
        _display_name(self.accepted_gpu_display_name, "accepted_gpu_display_name")
        require_aware(self.catalog_observed_at, "evidence catalog_observed_at")
        _sha256(self.request_sha256, "evidence request_sha256")
        _image_digest(self.image_digest)
        if (
            not isinstance(self.accepted_gpu_count, int)
            or isinstance(self.accepted_gpu_count, bool)
            or self.accepted_gpu_count < 1
        ):
            raise ValueError("Accepted GPU count must be positive")
        if not isinstance(self.accepted_cloud, CloudType):
            raise TypeError("Accepted cloud must be a CloudType")
        if (
            not self.accepted_hourly_rate_usd.is_finite()
            or self.accepted_hourly_rate_usd <= 0
        ):
            raise ValueError("Accepted Pod hourly rate must be positive")
        if self.catalog_observed_at > self.lifecycle.reservation_at:
            raise ValueError("Accepted catalog observation cannot follow reservation")
        if self.realized_placement is not None:
            realized = self.realized_placement
            if (
                realized.gpu_type_id != self.accepted_gpu_type_id
                or realized.gpu_display_name != self.accepted_gpu_display_name
                or realized.gpu_count != self.accepted_gpu_count
                or realized.cloud is not self.accepted_cloud
                or realized.hourly_rate_usd > self.accepted_hourly_rate_usd
            ):
                raise ValueError(
                    "Persisted realized placement contradicts the accepted quote"
                )
        if self.worker is not None and (
            self.worker.attempt_id != self.attempt_id
            or self.worker.request_sha256 != self.request_sha256
            or self.worker.image_digest != self.image_digest
        ):
            raise ValueError("Persisted worker evidence binding is inconsistent")
        if self.billing is not None:
            if self.realized_placement is not None and (
                self.billing.provider_pod_id != self.realized_placement.provider_pod_id
                or self.billing.hourly_price_usd
                != self.realized_placement.hourly_rate_usd
            ):
                raise ValueError("Persisted billing evidence binding is inconsistent")
            if (
                self.lifecycle.billing_reconciled_at is not None
                and self.lifecycle.billing_reconciled_at != self.billing.reconciled_at
            ):
                raise ValueError(
                    "Persisted billing reconciliation timestamp is inconsistent"
                )

    @classmethod
    def from_spec(cls, spec: PodCapacitySpec) -> PodCapacityEvidence:
        request = spec.request
        quote = request.quote
        return cls(
            schema_version=1,
            capacity_id=request.capacity_id,
            resource_name=request.resource_name,
            provider_quote_id=quote.provider_quote_id,
            catalog_observed_at=quote.observed_at,
            attempt_id=request.attempt_id,
            request_sha256=request.request_sha256,
            image_digest=request.image_reference.rsplit("@", 1)[1],
            accepted_gpu_type_id=quote.gpu_type_id,
            accepted_gpu_display_name=quote.gpu_display_name,
            accepted_gpu_count=quote.constraints.gpu_count,
            accepted_cloud=quote.constraints.cloud,
            accepted_hourly_rate_usd=quote.hourly_cost_usd,
            lifecycle=PodCapacityLifecycleEvidence(reservation_at=request.created_at),
        )

    @property
    def is_complete(self) -> bool:
        lifecycle = self.lifecycle
        return (
            self.realized_placement is not None
            and self.worker is not None
            and (
                lifecycle.provider_create_accepted_at is not None
                or lifecycle.provider_adopted_at is not None
            )
            and lifecycle.first_running_observed_at is not None
            and lifecycle.worker_ready_at is not None
            and lifecycle.workload_submitted_at is not None
            and lifecycle.workload_running_at is not None
            and lifecycle.workload_terminal_at is not None
            and lifecycle.stop_confirmed_at is not None
            and lifecycle.billing_reconciled_at is not None
            and self.billing is not None
            and self.billing.billed_until >= lifecycle.stop_confirmed_at
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": {
                "capacity_id": self.capacity_id,
                "resource_name": self.resource_name,
                "attempt_id": self.attempt_id,
                "request_sha256": self.request_sha256,
                "image_digest": self.image_digest,
            },
            "accepted_quote": {
                "provider_quote_id": self.provider_quote_id,
                "catalog_observed_at": iso_datetime(self.catalog_observed_at),
                "gpu_type_id": self.accepted_gpu_type_id,
                "gpu_display_name": self.accepted_gpu_display_name,
                "gpu_count": self.accepted_gpu_count,
                "cloud": self.accepted_cloud.value,
                "hourly_rate_usd": decimal_text(self.accepted_hourly_rate_usd),
            },
            "realized_placement": (
                self.realized_placement.to_dict()
                if self.realized_placement is not None
                else None
            ),
            "lifecycle": self.lifecycle.to_dict(),
            "worker": self.worker.to_dict() if self.worker is not None else None,
            "billing": self.billing.to_dict() if self.billing is not None else None,
            "provenance": {
                "identity": "accepted_catalog_request",
                "accepted_quote": "runpod_rest_v2_catalog",
                "realized_placement": (
                    {
                        "gpu_id_cloud_data_center_rate": "runpod_rest_v2_pod",
                        "gpu_display_name": (
                            "accepted_catalog_name_after_exact_gpu_id_match"
                        ),
                        "observed_at": "pod_capacity_provider_clock",
                    }
                    if self.realized_placement
                    else None
                ),
                "lifecycle": "pod_capacity_lease_service_clock",
                "worker": (
                    {
                        "telemetry": "bound_catalog_worker",
                        "container_process_started_at": "bound_catalog_worker_clock",
                    }
                    if self.worker
                    else None
                ),
                "billing": "runpod_rest_v2_billing" if self.billing else None,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PodCapacityEvidence:
        _exact_keys(
            value,
            {
                "schema_version",
                "identity",
                "accepted_quote",
                "realized_placement",
                "lifecycle",
                "worker",
                "billing",
                "provenance",
            },
            "capacity evidence",
        )
        identity = _required_mapping(value, "identity")
        quote = _required_mapping(value, "accepted_quote")
        _exact_keys(
            identity,
            {
                "capacity_id",
                "resource_name",
                "attempt_id",
                "request_sha256",
                "image_digest",
            },
            "capacity evidence identity",
        )
        _exact_keys(
            quote,
            {
                "provider_quote_id",
                "catalog_observed_at",
                "gpu_type_id",
                "gpu_display_name",
                "gpu_count",
                "cloud",
                "hourly_rate_usd",
            },
            "capacity evidence quote",
        )
        # Provenance is fixed rather than caller-controlled metadata.
        provenance = _required_mapping(value, "provenance")
        _exact_keys(
            provenance,
            {
                "identity",
                "accepted_quote",
                "realized_placement",
                "lifecycle",
                "worker",
                "billing",
            },
            "capacity evidence provenance",
        )
        realized = value.get("realized_placement")
        worker = value.get("worker")
        billing = value.get("billing")
        for name, item in (
            ("realized_placement", realized),
            ("worker", worker),
            ("billing", billing),
        ):
            if item is not None and not isinstance(item, Mapping):
                raise RunPodManagerError(
                    f"Stored capacity evidence {name} must be an object or null"
                )
        if isinstance(billing, Mapping):
            _exact_keys(
                billing,
                {
                    "provider_billing_id",
                    "provider_pod_id",
                    "billed_from",
                    "billed_until",
                    "billed_seconds",
                    "hourly_price_usd",
                    "actual_cost_usd",
                    "reconciled_at",
                },
                "capacity evidence billing",
            )
        expected_provenance = {
            "identity": "accepted_catalog_request",
            "accepted_quote": "runpod_rest_v2_catalog",
            "realized_placement": (
                {
                    "gpu_id_cloud_data_center_rate": "runpod_rest_v2_pod",
                    "gpu_display_name": (
                        "accepted_catalog_name_after_exact_gpu_id_match"
                    ),
                    "observed_at": "pod_capacity_provider_clock",
                }
                if isinstance(realized, Mapping)
                else None
            ),
            "lifecycle": "pod_capacity_lease_service_clock",
            "worker": (
                {
                    "telemetry": "bound_catalog_worker",
                    "container_process_started_at": "bound_catalog_worker_clock",
                }
                if isinstance(worker, Mapping)
                else None
            ),
            "billing": (
                "runpod_rest_v2_billing" if isinstance(billing, Mapping) else None
            ),
        }
        if dict(provenance) != expected_provenance:
            raise RunPodManagerError(
                "Stored capacity evidence provenance is inconsistent"
            )
        return cls(
            schema_version=_required_int(value, "schema_version"),
            capacity_id=_required_string(identity, "capacity_id"),
            resource_name=_required_string(identity, "resource_name"),
            provider_quote_id=_required_string(quote, "provider_quote_id"),
            catalog_observed_at=_datetime(
                _required_string(quote, "catalog_observed_at")
            ),
            attempt_id=_required_string(identity, "attempt_id"),
            request_sha256=_required_string(identity, "request_sha256"),
            image_digest=_required_string(identity, "image_digest"),
            accepted_gpu_type_id=_required_string(quote, "gpu_type_id"),
            accepted_gpu_display_name=_required_string(quote, "gpu_display_name"),
            accepted_gpu_count=_required_int(quote, "gpu_count"),
            accepted_cloud=CloudType(_required_string(quote, "cloud")),
            accepted_hourly_rate_usd=_required_decimal(quote, "hourly_rate_usd"),
            lifecycle=PodCapacityLifecycleEvidence.from_dict(
                _required_mapping(value, "lifecycle")
            ),
            realized_placement=(
                PodRealizedPlacement.from_dict(realized)
                if isinstance(realized, Mapping)
                else None
            ),
            worker=(
                CatalogWorkerEvidence.from_dict(worker)
                if isinstance(worker, Mapping)
                else None
            ),
            billing=(
                PodBillingReceipt.from_dict(billing)
                if isinstance(billing, Mapping)
                else None
            ),
        )


class TrainingPodConflictError(RunPodManagerError):
    """A cleanup token or Pod is already claimed by another active lease."""


class TrainingPodLifecycleError(RunPodManagerError):
    """An operation failed while durable cleanup state remains observable."""

    reconcile_required = True

    def __init__(
        self,
        operation: str,
        *,
        cleanup_token: str,
        pod_id: str | None,
        cleanup_state: TrainingPodCleanupState,
        billing_risk: bool,
    ) -> None:
        self.operation = operation
        self.cleanup_token = cleanup_token
        self.pod_id = pod_id
        self.cleanup_state = cleanup_state
        self.billing_risk = billing_risk
        pod_context = f" Pod '{pod_id}'" if pod_id else " Pod with uncertain ID"
        risk = (
            "billable capacity may remain active"
            if billing_risk
            else "no owned billable capacity remains"
        )
        super().__init__(
            f"Training {operation} failed for{pod_context}; {risk}. "
            f"Reconcile cleanup token '{cleanup_token}' "
            f"(cleanup={cleanup_state.value})."
        )


class TrainingPodCleanupError(TrainingPodLifecycleError):
    """Owned capacity could not yet be confirmed stopped."""


@dataclass(frozen=True)
class TrainingPodRequest:
    """Stable, secret-free request persisted before provider I/O."""

    cleanup_token: str
    companion_id: str
    profile_id: str
    source: TrainingPodSource
    resource_name: str
    provider_pod_id: str | None
    created_at: datetime
    readiness_deadline: datetime
    hard_deadline: datetime
    root_cleanup_token: str | None = None
    capacity_spec: PodCapacitySpec | None = None

    def __post_init__(self) -> None:
        root_cleanup_token = self.root_cleanup_token
        if root_cleanup_token is None:
            root_cleanup_token = self.cleanup_token
            object.__setattr__(self, "root_cleanup_token", root_cleanup_token)
        for name, value in (
            ("cleanup_token", self.cleanup_token),
            ("root_cleanup_token", root_cleanup_token),
            ("companion_id", self.companion_id),
            ("profile_id", self.profile_id),
            ("resource_name", self.resource_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Training Pod {name} must be a non-empty string")
        for name, value in (
            ("cleanup_token", self.cleanup_token),
            ("root_cleanup_token", root_cleanup_token),
        ):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", value):
                raise ValueError(f"Training Pod {name} has an invalid format")
        if self.provider_pod_id is not None and not self.provider_pod_id.strip():
            raise ValueError("Training Pod provider_pod_id cannot be empty")
        for name, value in (
            ("created_at", self.created_at),
            ("readiness_deadline", self.readiness_deadline),
            ("hard_deadline", self.hard_deadline),
        ):
            require_aware(value, name)
        if self.readiness_deadline <= self.created_at:
            raise ValueError("Training Pod readiness deadline must be in the future")
        if self.hard_deadline <= self.readiness_deadline:
            raise ValueError(
                "Training Pod hard deadline must follow readiness deadline"
            )
        if (
            self.source is TrainingPodSource.CREATED
            and self.provider_pod_id is not None
        ):
            raise ValueError(
                "A create request cannot know its provider Pod ID in advance"
            )
        if (
            self.source is not TrainingPodSource.CREATED
            and self.provider_pod_id is None
        ):
            raise ValueError("A persistent/reused request requires a provider Pod ID")
        if self.capacity_spec is not None:
            catalog = self.capacity_spec.request
            if self.source is not TrainingPodSource.CREATED:
                raise ValueError("Catalog Pod capacity must create isolated capacity")
            if (
                self.cleanup_token != catalog.capacity_id
                or self.cleanup_family_token != catalog.cleanup_family_id
                or self.companion_id != catalog.owner_id
                or self.profile_id != catalog.profile_id
                or self.resource_name != catalog.resource_name
                or self.created_at != catalog.created_at
                or self.readiness_deadline != catalog.readiness_deadline
                or self.hard_deadline != catalog.hard_deadline
            ):
                raise ValueError("Catalog Pod capacity identity is inconsistent")

    @property
    def fingerprint(self) -> str:
        payload = {
            "cleanup_token": self.cleanup_token,
            "companion_id": self.companion_id,
            "profile_id": self.profile_id,
            "source": self.source.value,
            "resource_name": self.resource_name,
            "provider_pod_id": self.provider_pod_id,
            "readiness_deadline": iso_datetime(self.readiness_deadline),
            "hard_deadline": iso_datetime(self.hard_deadline),
        }
        # Preserve fingerprints for pre-family root/exact-token rows while binding
        # every new child attempt to its caller-owned cleanup family.
        if self.cleanup_family_token != self.cleanup_token:
            payload["root_cleanup_token"] = self.cleanup_family_token
        if self.capacity_spec is not None:
            payload["capacity_spec"] = self.capacity_spec.to_dict()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cleanup_family_token(self) -> str:
        """Return the validated root token after backward-compatible defaulting."""

        root = self.root_cleanup_token
        if root is None:  # pragma: no cover - guarded by frozen post-init validation
            raise ValueError("Training Pod root_cleanup_token was not initialized")
        return root


@dataclass(frozen=True)
class TrainingPodLease:
    """Durable ownership, workload, and teardown state for one Pod use."""

    cleanup_token: str
    root_cleanup_token: str
    request_fingerprint: str
    companion_id: str
    profile_id: str
    source: TrainingPodSource
    resource_name: str
    provider_pod_id: str | None
    ownership: TrainingPodOwnership
    state: TrainingPodState
    cleanup_state: TrainingPodCleanupState
    family_release_requested: bool
    family_release_complete: bool
    creation_uncertain: bool
    backend_base_url: str | None
    provider_job_id: str | None
    created_at: datetime
    updated_at: datetime
    last_heartbeat_at: datetime
    readiness_deadline: datetime
    hard_deadline: datetime
    last_provider_error: str | None
    stop_attempts: int
    revision: int
    capacity_spec: PodCapacitySpec | None = None
    workload_state: CatalogPodWorkloadState = CatalogPodWorkloadState.NOT_SUBMITTED
    workload_error_type: str | None = None
    billing_state: PodCapacityBillingState = PodCapacityBillingState.NOT_APPLICABLE
    billing_receipt: PodBillingReceipt | None = None
    terminated_at: datetime | None = None
    evidence: PodCapacityEvidence | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state is TrainingPodState.RELEASED

    @property
    def owns_billing_capacity(self) -> bool:
        return (
            self.ownership
            in {
                TrainingPodOwnership.PROVISIONAL,
                TrainingPodOwnership.OWNED,
            }
            and not self.is_terminal
        )

    @property
    def public_cleanup_token(self) -> str:
        return self.cleanup_token

    @property
    def capacity_id(self) -> str:
        return self.cleanup_token

    @property
    def is_catalog_attempt(self) -> bool:
        return self.capacity_spec is not None

    @property
    def settlement_ready(self) -> bool:
        return (
            self.is_terminal
            and self.billing_state is PodCapacityBillingState.AUTHORITATIVE
            and self.billing_receipt is not None
        )

    @property
    def terminal_success_evidence_complete(self) -> bool:
        """Success is proven only after teardown and authoritative full billing."""

        return (
            self.workload_state is CatalogPodWorkloadState.SUCCEEDED
            and self.settlement_ready
            and self.evidence is not None
            and self.evidence.is_complete
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return content-free operational state without a private route URL."""

        return {
            "cleanup_token": self.cleanup_token,
            "root_cleanup_token": self.root_cleanup_token,
            "companion_id": self.companion_id,
            "profile_id": self.profile_id,
            "source": self.source.value,
            "provider_pod_id": self.provider_pod_id,
            "ownership": self.ownership.value,
            "state": self.state.value,
            "cleanup_state": self.cleanup_state.value,
            "family_release_requested": self.family_release_requested,
            "family_release_complete": self.family_release_complete,
            "provider_job_id": self.provider_job_id,
            "creation_uncertain": self.creation_uncertain,
            "last_provider_error": self.last_provider_error,
            "stop_attempts": self.stop_attempts,
            "updated_at": iso_datetime(self.updated_at),
            "readiness_deadline": iso_datetime(self.readiness_deadline),
            "hard_deadline": iso_datetime(self.hard_deadline),
            "revision": self.revision,
            "workload_state": self.workload_state.value,
            "workload_error_type": self.workload_error_type,
            "billing_state": self.billing_state.value,
            "billing_receipt": (
                self.billing_receipt.to_dict()
                if self.billing_receipt is not None
                else None
            ),
            "terminated_at": (
                iso_datetime(self.terminated_at)
                if self.terminated_at is not None
                else None
            ),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "terminal_success_evidence_complete": (
                self.terminal_success_evidence_complete
            ),
            "capacity": (
                {
                    "owner_id": self.capacity_spec.request.owner_id,
                    "workload_id": self.capacity_spec.request.workload_id,
                    "attempt_id": self.capacity_spec.request.attempt_id,
                    "idempotency_key": self.capacity_spec.request.idempotency_key,
                    "workload_kind": self.capacity_spec.request.workload_kind,
                    "request_sha256": self.capacity_spec.request.request_sha256,
                    "parameters_sha256": (self.capacity_spec.request.parameters_sha256),
                    "quote": self.capacity_spec.request.quote.to_dict(),
                }
                if self.capacity_spec is not None
                else None
            ),
        }


def durable_training_name(cleanup_token: str) -> str:
    """Build a bounded deterministic Runpod resource name from a cleanup token."""

    digest = hashlib.sha256(cleanup_token.encode()).hexdigest()[:20]
    return f"kestrel-lora-{digest}"


def durable_pod_capacity_name(capacity_id: str, request_fingerprint: str) -> str:
    """Encode the stable capacity identity and immutable fingerprint in the name."""

    _safe_identifier(capacity_id, "capacity_id")
    _sha256(request_fingerprint, "request_fingerprint")
    identity = hashlib.sha256(capacity_id.encode()).hexdigest()[:12]
    return f"kestrel-cap-{identity}-{request_fingerprint[:12]}"


def fallback_training_cleanup_token(root_cleanup_token: str, profile_id: str) -> str:
    """Derive the legacy-compatible identity for one fallback profile attempt."""

    attempt_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"kestrel-runpod-training\0{root_cleanup_token}\0{profile_id}",
    )
    return f"training:{attempt_id}"


def sanitize_training_error(error: BaseException) -> str:
    """Persist only the safe exception type, never response bodies or URLs."""

    return type(error).__name__


def iso_datetime(value: datetime) -> str:
    require_aware(value, "timestamp")
    return value.astimezone(UTC).isoformat()


def require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"Training Pod {name} must be timezone-aware")


PodCapacitySource = TrainingPodSource
PodCapacityOwnership = TrainingPodOwnership
PodCapacityState = TrainingPodState
PodCapacityCleanupState = TrainingPodCleanupState
PodCapacityConflictError = TrainingPodConflictError
PodCapacityLifecycleError = TrainingPodLifecycleError
PodCapacityCleanupError = TrainingPodCleanupError
PodCapacityLeaseRequest = TrainingPodRequest
PodCapacityLease = TrainingPodLease


def decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def pod_cost_usd(hourly_cost_usd: Decimal, billed_seconds: int) -> Decimal:
    """Round a rate-derived upper bound up so reservations never underfund."""

    if (
        not hourly_cost_usd.is_finite()
        or hourly_cost_usd <= 0
        or not isinstance(billed_seconds, int)
        or isinstance(billed_seconds, bool)
        or billed_seconds < 0
    ):
        raise ValueError("Pod cost inputs are invalid")
    value = hourly_cost_usd * Decimal(billed_seconds) / Decimal(3600)
    return value.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)


def attempt_environment_sha256(environment: Mapping[str, str]) -> str:
    """Bind secret environment values without persisting or exposing them."""

    _validate_attempt_environment(environment)
    return _json_sha256(dict(sorted(environment.items())))


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", value
    ):
        raise ValueError(f"Pod capacity {name} has an invalid format")


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Pod capacity {name} must be a lowercase SHA-256 digest")


_FORBIDDEN_WORKLOAD_ENV = frozenset(
    {
        "DATABASE_URL",
        "CATALOG_DATABASE_URL",
        "CLOUD_SQL_CONNECTION_NAME",
        "PGPASSWORD",
        "RUNPOD_API_KEY",
        "RUNPOD_CONTROL_PLANE_API_KEY",
        "RUNPOD_SERVERLESS_API_KEY",
        "CATALOG_WORKER_MODE",
        "CATALOG_POD_ATTEMPT_ID",
        "CATALOG_POD_BEARER_TOKEN",
        "CATALOG_POD_BEARER_EXPIRES_AT",
        "CONTAINER_DIGEST",
    }
)


def _validate_attempt_environment(environment: Mapping[str, str]) -> None:
    if not isinstance(environment, Mapping):
        raise TypeError("Catalog Pod attempt_environment must be a mapping")
    invalid = []
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", key)
            or not isinstance(value, str)
            or "\x00" in value
            or "\r" in value
            or "\n" in value
        ):
            invalid.append(str(key))
    forbidden = sorted(_FORBIDDEN_WORKLOAD_ENV.intersection(environment))
    if invalid or forbidden:
        names = sorted(set(invalid + forbidden))
        raise ValueError(
            "Catalog Pod attempt environment contains forbidden or invalid keys: "
            + ", ".join(names)
        )


def _seconds(value: int) -> timedelta:
    return timedelta(seconds=value)


def _placement_to_dict(value: PlacementDecision) -> dict[str, Any]:
    return {
        "gpu_id": value.gpu_id,
        "gpu_pool": value.gpu_pool,
        "gpu_name": value.gpu_name,
        "memory_gb": value.memory_gb,
        "cloud": value.cloud.value,
        "gpu_count": value.gpu_count,
        "offered_cost_per_hr": value.offered_cost_per_hr,
        "availability": value.availability.value if value.availability else None,
        "catalog_observed_at": iso_datetime(value.catalog_observed_at),
    }


def _placement_from_dict(
    value: Mapping[str, Any], constraints: PodCapacityConstraints
) -> PlacementDecision:
    from .models import Availability

    raw_availability = value.get("availability")
    raw_hourly_rate = value.get("offered_cost_per_hr")
    if isinstance(raw_hourly_rate, bool) or not isinstance(
        raw_hourly_rate, (int, float)
    ):
        raise RunPodManagerError("Stored Pod capacity offered rate is invalid")
    return PlacementDecision(
        gpu_id=_required_string(value, "gpu_id"),
        gpu_pool=_optional_string(value.get("gpu_pool")),
        gpu_name=_required_string(value, "gpu_name"),
        memory_gb=_required_int(value, "memory_gb"),
        cloud=CloudType(_required_string(value, "cloud")),
        gpu_count=_required_int(value, "gpu_count"),
        offered_cost_per_hr=float(raw_hourly_rate),
        availability=(
            Availability(str(raw_availability))
            if raw_availability is not None
            else None
        ),
        catalog_observed_at=_datetime(_required_string(value, "catalog_observed_at")),
        requirements=constraints.placement_requirements(),
    )


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise RunPodManagerError(f"Stored Pod capacity {key} must be an object")
    return item


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], name: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise RunPodManagerError(f"{name} fields do not match the versioned contract")


def _display_name(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"Pod capacity {name} is invalid")


def _provider_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value
    ):
        raise ValueError(f"Pod capacity {name} is invalid")


def _image_digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Catalog worker image digest is invalid")


def _envelope_optional_decimal(value: Any, name: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise TypeError(f"Catalog worker {name} must be numeric or null")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Catalog worker {name} must be numeric or null") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"Catalog worker {name} must be finite and nonnegative")
    return parsed


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Catalog worker {name} must be a nonnegative integer or null")
    return value


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RunPodManagerError(f"Stored Pod capacity {key} must be a string")
    return item


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunPodManagerError("Stored Pod capacity optional string is invalid")
    return value


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise RunPodManagerError(f"Stored Pod capacity {key} must be an integer")
    return item


def _required_decimal(value: Mapping[str, Any], key: str) -> Decimal:
    item = value.get(key)
    if not isinstance(item, str):
        raise RunPodManagerError(f"Stored Pod capacity {key} must be a decimal string")
    try:
        parsed = Decimal(item)
    except (InvalidOperation, ValueError) as exc:
        raise RunPodManagerError(f"Stored Pod capacity {key} is not a decimal") from exc
    return parsed


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunPodManagerError("Stored Pod capacity decimal is invalid")
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise RunPodManagerError("Stored Pod capacity decimal is invalid") from exc


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RunPodManagerError("Stored Pod capacity string sequence is invalid")
    return tuple(value)


def _datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RunPodManagerError("Stored Pod capacity timestamp is invalid") from exc
    require_aware(parsed, "stored timestamp")
    return parsed.astimezone(UTC)
