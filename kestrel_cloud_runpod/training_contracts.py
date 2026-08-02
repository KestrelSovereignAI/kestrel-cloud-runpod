"""Durable contracts for billable Runpod training Pod ownership."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .models import RunPodManagerError


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

    def __post_init__(self) -> None:
        for name, value in (
            ("cleanup_token", self.cleanup_token),
            ("companion_id", self.companion_id),
            ("profile_id", self.profile_id),
            ("resource_name", self.resource_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Training Pod {name} must be a non-empty string")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", self.cleanup_token):
            raise ValueError("Training Pod cleanup_token has an invalid format")
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
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrainingPodLease:
    """Durable ownership, workload, and teardown state for one Pod use."""

    cleanup_token: str
    request_fingerprint: str
    companion_id: str
    profile_id: str
    source: TrainingPodSource
    resource_name: str
    provider_pod_id: str | None
    ownership: TrainingPodOwnership
    state: TrainingPodState
    cleanup_state: TrainingPodCleanupState
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

    def to_public_dict(self) -> dict[str, Any]:
        """Return content-free operational state without a private route URL."""

        return {
            "cleanup_token": self.cleanup_token,
            "companion_id": self.companion_id,
            "profile_id": self.profile_id,
            "source": self.source.value,
            "provider_pod_id": self.provider_pod_id,
            "ownership": self.ownership.value,
            "state": self.state.value,
            "cleanup_state": self.cleanup_state.value,
            "provider_job_id": self.provider_job_id,
            "creation_uncertain": self.creation_uncertain,
            "last_provider_error": self.last_provider_error,
            "stop_attempts": self.stop_attempts,
            "updated_at": iso_datetime(self.updated_at),
            "readiness_deadline": iso_datetime(self.readiness_deadline),
            "hard_deadline": iso_datetime(self.hard_deadline),
            "revision": self.revision,
        }


def durable_training_name(cleanup_token: str) -> str:
    """Build a bounded deterministic Runpod resource name from a cleanup token."""

    digest = hashlib.sha256(cleanup_token.encode()).hexdigest()[:20]
    return f"kestrel-lora-{digest}"


def sanitize_training_error(error: BaseException) -> str:
    """Persist only the safe exception type, never response bodies or URLs."""

    return type(error).__name__


def iso_datetime(value: datetime) -> str:
    require_aware(value, "timestamp")
    return value.astimezone(UTC).isoformat()


def require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"Training Pod {name} must be timezone-aware")
