"""Restart-safe lifecycle orchestration for private-Ollama leases."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from kestrel_sovereign.execution_custody import is_execution_control_error

from .billing import exact_decimal_integer_scale
from .effect_custody import await_owned_effect
from .models import (
    ComputeProduct,
    RunPodManagerError,
)
from .ollama_contracts import (
    OllamaCapacityProvider,
    OllamaLease,
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseReadinessError,
    OllamaLeaseRequest,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaLifecycleBillingObservation,
    OllamaLifecycleBillingState,
    OllamaPlacementPlan,
    OllamaProvisioningEvidenceError,
    OllamaProvisioningEvidenceState,
    OllamaProvisioningIntent,
    OllamaProvisioningMismatchError,
    OllamaProvisioningMismatchEvidence,
    OllamaProvisioningReceipt,
    OllamaResourceType,
    OllamaRuntimeTelemetryState,
    OllamaTeardownState,
    PreparedOllamaProvisioning,
    ProvisionedOllamaResource,
    accrued_cost,
    authorized_cost_deadline,
    normalize_new_ollama_request_money,
    over_authorized_cost,
    provision_attempt_id,
    require_aware,
    resource_from_lease,
    resource_name,
    sanitize_provider_error,
)
from .ollama_repository import SQLiteOllamaLeaseRepository, request_from_lease

logger = logging.getLogger(__name__)


def _validate_provisioning_receipt(
    receipt: OllamaProvisioningReceipt,
    intent: OllamaProvisioningIntent,
) -> None:
    if not isinstance(receipt, OllamaProvisioningReceipt):
        raise OllamaProvisioningEvidenceError(
            "Ollama provider returned invalid provisioning evidence"
        )
    try:
        receipt.validate_intent(intent)
    except ValueError as exc:
        raise OllamaProvisioningEvidenceError(
            "Ollama provider returned mismatched provisioning evidence"
        ) from exc


def _validate_provisioning_mismatch(
    evidence: OllamaProvisioningMismatchEvidence,
    intent: OllamaProvisioningIntent,
) -> None:
    if not isinstance(evidence, OllamaProvisioningMismatchEvidence):
        raise OllamaProvisioningEvidenceError(
            "Ollama provider returned invalid mismatch evidence"
        )
    try:
        evidence.validate_intent(intent)
    except ValueError as exc:
        raise OllamaProvisioningEvidenceError(
            "Ollama provider returned unbound mismatch evidence"
        ) from exc


def _ambiguous_candidate_from_lease(
    lease: OllamaLease,
) -> ProvisionedOllamaResource:
    if (
        not lease.ambiguous_create_candidate_id
        or lease.resource_type is None
        or not lease.resource_name
    ):
        raise OllamaProvisioningEvidenceError(
            "Stored Ollama ambiguous-create locator is incomplete"
        )
    return ProvisionedOllamaResource(
        resource_type=lease.resource_type,
        provider_resource_id=lease.ambiguous_create_candidate_id,
        resource_name=lease.resource_name,
    )


def _provisioning_deadline_reached(lease: OllamaLease, now: datetime) -> bool:
    return now >= min(lease.readiness_deadline, lease.hard_deadline)


_ALLOWED_TRANSITIONS: Mapping[OllamaLeaseState, frozenset[OllamaLeaseState]] = {
    OllamaLeaseState.REQUESTED: frozenset(
        {
            OllamaLeaseState.PROVISIONING,
            OllamaLeaseState.FAILED,
            OllamaLeaseState.TERMINATED,
        }
    ),
    OllamaLeaseState.PROVISIONING: frozenset(
        {
            OllamaLeaseState.RECONCILE_REQUIRED,
            OllamaLeaseState.WAITING_FOR_MODEL,
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.FAILED,
        }
    ),
    OllamaLeaseState.RECONCILE_REQUIRED: frozenset(
        {
            OllamaLeaseState.WAITING_FOR_MODEL,
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.FAILED,
        }
    ),
    OllamaLeaseState.WAITING_FOR_MODEL: frozenset(
        {
            OllamaLeaseState.READY,
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.FAILED,
        }
    ),
    OllamaLeaseState.READY: frozenset({OllamaLeaseState.RELEASING}),
    OllamaLeaseState.RELEASING: frozenset({OllamaLeaseState.TERMINATED}),
    OllamaLeaseState.FAILED: frozenset(
        {OllamaLeaseState.RELEASING, OllamaLeaseState.TERMINATED}
    ),
    OllamaLeaseState.TERMINATED: frozenset(),
}


class OllamaLeaseService:
    """Coordinate idempotent acquisition, readiness, expiry, and teardown."""

    def __init__(
        self,
        *,
        repository: SQLiteOllamaLeaseRepository,
        provider: OllamaCapacityProvider,
        poll_interval_seconds: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        billing_poll_interval_seconds: float = 3600,
        reconciliation_batch_size: int = 100,
        maximum_telemetry_observations: int = 256,
        maximum_billing_observations: int = 128,
        maximum_evidence_bytes: int = 1_048_576,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("Ollama lease poll interval must be positive")
        if billing_poll_interval_seconds <= 0:
            raise ValueError("Ollama billing poll interval must be positive")
        for name, value in (
            ("reconciliation_batch_size", reconciliation_batch_size),
            ("maximum_telemetry_observations", maximum_telemetry_observations),
            ("maximum_billing_observations", maximum_billing_observations),
            ("maximum_evidence_bytes", maximum_evidence_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Ollama {name} must be positive")
        self.repository = repository
        self.provider = provider
        self.poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self.billing_poll_interval_seconds = billing_poll_interval_seconds
        self.reconciliation_batch_size = reconciliation_batch_size
        self.maximum_telemetry_observations = maximum_telemetry_observations
        self.maximum_billing_observations = maximum_billing_observations
        self.maximum_evidence_bytes = maximum_evidence_bytes
        # In-process callers share the exact admitted physical cleanup task.
        # Process loss is handled by the durable PENDING state and exact-ID
        # provider idempotency; this map deliberately makes no cross-process
        # exactly-once claim.
        self._cleanup_tasks: dict[tuple[str, str], asyncio.Task[OllamaLease]] = {}
        self._billing_tasks: dict[tuple[str, str], asyncio.Task[OllamaLease]] = {}

    async def acquire(
        self,
        request: OllamaLeaseRequest,
        *,
        wait_until_ready: bool = True,
        plan: OllamaPlacementPlan | None = None,
    ) -> OllamaLease:
        now = self._now()
        existing = self.repository.get(request.lease_id)
        if existing is None:
            request = normalize_new_ollama_request_money(request)
            if request.hard_deadline <= now:
                raise ValueError("Ollama lease hard_deadline must be in the future")
            if (
                timedelta(seconds=request.expected_session_seconds)
                > request.hard_deadline - now
            ):
                raise ValueError("Expected Ollama session exceeds its hard deadline")
            requested_at = request.requested_at or now
            if now >= min(
                requested_at + timedelta(seconds=request.readiness_timeout_seconds),
                request.hard_deadline,
            ):
                raise ValueError("Ollama readiness deadline has already elapsed")
        elif existing.request_fingerprint != request.fingerprint:
            normalized_request = normalize_new_ollama_request_money(request)
            if existing.request_fingerprint == normalized_request.fingerprint:
                request = normalized_request
        lease, inserted = self.repository.insert_request(request, now=now)
        if not inserted:
            self._authorize(lease, request.owner_id, request.workload_id)
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        ):
            return lease
        if (
            lease.provisioning_mismatch is not None
            and lease.provider_resource_id is not None
        ):
            return await self._release(lease, reason="provisioning_evidence_mismatch")
        if lease.provisioning_mismatch is not None:
            return lease
        if lease.state is OllamaLeaseState.READY:
            return await self._ready_with_route(lease)
        if lease.state is OllamaLeaseState.REQUESTED:
            if _provisioning_deadline_reached(lease, now):
                return await self._release(lease, reason="expired")
            lease = await self._provision_requested(lease, request, plan=plan)
        elif lease.state in {
            OllamaLeaseState.PROVISIONING,
            OllamaLeaseState.RECONCILE_REQUIRED,
        }:
            lease = await self._reconcile_creation(lease)
        if lease.state is OllamaLeaseState.WAITING_FOR_MODEL and wait_until_ready:
            return await self._wait_until_ready(lease)
        return lease

    async def _provision_requested(
        self,
        lease: OllamaLease,
        request: OllamaLeaseRequest,
        *,
        plan: OllamaPlacementPlan | None = None,
    ) -> OllamaLease:
        if lease.state is not OllamaLeaseState.REQUESTED:
            raise RunPodManagerError(
                f"Ollama lease '{lease.lease_id}' is no longer awaiting provisioning"
            )
        try:
            plan = plan or await self.provider.plan(request)
            if _provisioning_deadline_reached(lease, self._now()):
                return await self._release(lease, reason="expired")
            self.validate_plan(request, plan)
            # This is still REQUESTED: proven deterministic validation failures
            # must not create uncertainty or admit a billable provider effect.
            provision_requested_at = self._now()
            prepared = self.provider.validate_runtime_request(
                request,
                plan,
                provision_requested_at=provision_requested_at,
            )
        except RunPodManagerError as exc:
            self._transition(
                lease,
                OllamaLeaseState.FAILED,
                last_provider_error=sanitize_provider_error(exc),
            )
            raise
        try:
            if not isinstance(prepared, PreparedOllamaProvisioning):
                raise RunPodManagerError(
                    "Ollama provider did not return a prepared provisioning request"
                )
            intent = prepared.intent
            accepted_whole_rate = exact_decimal_integer_scale(
                Decimal(str(plan.placement.offered_cost_per_hr)),
                plan.placement.gpu_count,
            )
            if (
                intent.lease_id != lease.lease_id
                or intent.request_fingerprint != lease.request_fingerprint
                or intent.provision_attempt_id != provision_attempt_id(request)
                or intent.resource_name != resource_name(request.lease_id)
                or intent.resource_type is not plan.resource_type
                or intent.mode is not plan.mode
                or intent.provision_requested_at != provision_requested_at
                or intent.selected_gpu_id != plan.placement.gpu_id
                or intent.selected_gpu_pool != plan.placement.gpu_pool
                or intent.selected_gpu_name != plan.placement.gpu_name
                or intent.selected_gpu_count != plan.placement.gpu_count
                or intent.selected_cloud is not plan.placement.cloud
                or intent.allowed_data_center_ids
                != request.constraints.allowed_data_center_ids
                or intent.accepted_whole_hourly_rate_usd != accepted_whole_rate
            ):
                raise RunPodManagerError(
                    "Prepared Ollama provisioning does not bind the accepted request"
                )
        except RunPodManagerError as exc:
            self._transition(
                lease,
                OllamaLeaseState.FAILED,
                last_provider_error=sanitize_provider_error(exc),
            )
            raise
        lease = self._transition(
            lease,
            OllamaLeaseState.PROVISIONING,
            mode=plan.mode,
            resource_type=plan.resource_type,
            resource_name=intent.resource_name,
            creation_uncertain=True,
            provision_attempt_id=intent.provision_attempt_id,
            provision_attempts=lease.provision_attempts + 1,
            provisioning_started_at=intent.provision_requested_at,
            offered_rate_per_hr=plan.placement.offered_cost_per_hr,
            placement_gpu_count=plan.placement.gpu_count,
            estimated_cost=plan.estimated_cost,
            estimated_compute_cost=plan.estimated_compute_cost,
            maximum_compute_cost=plan.maximum_compute_cost,
            estimated_non_compute_cost=plan.estimated_non_compute_cost,
            maximum_non_compute_cost=plan.maximum_non_compute_cost,
            cost_ceiling=plan.cost_ceiling,
            cost_policy_components_json=json.dumps(
                [item.value for item in plan.non_compute_components],
                separators=(",", ":"),
            ),
            maximum_concurrent_workers=plan.maximum_concurrent_workers,
            estimated_billable_seconds=plan.estimated_billable_seconds,
            maximum_billable_seconds=plan.maximum_billable_seconds,
            selected_gpu_id=plan.placement.gpu_id,
            selected_gpu_pool=plan.placement.gpu_pool,
            selected_gpu_name=plan.placement.gpu_name,
            catalog_observed_at=plan.placement.catalog_observed_at,
            provisioning_evidence_state=(
                OllamaProvisioningEvidenceState.INTENT_RECORDED
            ),
            provisioning_intent_json=intent,
            runtime_telemetry_binding_sha256=(
                prepared.runtime_telemetry_binding_sha256
            ),
        )
        return await await_owned_effect(self._provision_effect(lease, prepared))

    async def _provision_effect(
        self,
        lease: OllamaLease,
        prepared: PreparedOllamaProvisioning,
    ) -> OllamaLease:
        """Keep the original admitted create and its durable receipt together."""

        if _provisioning_deadline_reached(lease, self._now()):
            lease = self.repository.compare_and_set(
                lease,
                changes={"creation_uncertain": False},
            )
            return await self._release(lease, reason="expired")

        try:
            resource = await self.provider.provision(prepared)
        except OllamaProvisioningMismatchError as exc:
            lease = self._persist_provisioning_mismatch(
                lease, exc, acknowledged_create=True
            )
            await self._release(lease, reason="provisioning_evidence_mismatch")
            raise
        except RunPodManagerError as exc:
            # A response decoding/placement failure can happen AFTER the
            # provider accepted POST. Only planning failures before this
            # admitted effect prove that no create was attempted.
            self.repository.record_provisioning_outcome(
                lease,
                now=self._now(),
                provider_error=sanitize_provider_error(exc),
            )
            raise
        lease = self.repository.attach_provider_acknowledgment(
            lease,
            resource,
            now=self._now(),
        )
        if lease.state in {
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.FAILED,
            OllamaLeaseState.TERMINATED,
        }:
            return await self._release(lease, reason="late_create_acknowledgment")
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.COMPLETE
        ):
            if lease.state is OllamaLeaseState.READY:
                return await self._ready_with_route(lease)
            return lease
        if _provisioning_deadline_reached(lease, self._now()):
            return await self._release(lease, reason="expired")
        try:
            receipt = await self.provider.observe_provisioning(
                prepared.intent, resource
            )
            _validate_provisioning_receipt(receipt, prepared.intent)
        except OllamaProvisioningMismatchError as exc:
            lease = self._persist_provisioning_mismatch(lease, exc)
            await self._release(lease, reason="provisioning_evidence_mismatch")
            raise
        except RunPodManagerError as exc:
            self.repository.record_provisioning_outcome(
                lease,
                now=self._now(),
                provider_error=sanitize_provider_error(exc),
            )
            raise
        lease = self.repository.record_provisioning_outcome(
            lease,
            now=self._now(),
            receipt=receipt,
        )
        if _provisioning_deadline_reached(lease, self._now()):
            return await self._release(lease, reason="expired")
        return lease

    async def get(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        lease = self._required(lease_id)
        self._authorize(lease, owner_id, workload_id)
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        ):
            return lease
        if lease.state is OllamaLeaseState.READY:
            return await self._ready_with_route(lease)
        return lease

    async def touch(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        lease = self._required(lease_id)
        self._authorize(lease, owner_id, workload_id)
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        ):
            return lease
        if lease.state in {
            OllamaLeaseState.FAILED,
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.TERMINATED,
        }:
            # A lost touch response or an external reconciler may have already
            # moved the exact authorized lease to a route-less terminal path.
            # Return that durable truth idempotently so Core can detach its
            # route; renewal must never recreate capacity or retry teardown.
            return lease
        if lease.state is not OllamaLeaseState.READY:
            raise RunPodManagerError("Only a ready Ollama lease can be marked used")
        lease = await self._ready_with_route(lease)
        if lease.state is not OllamaLeaseState.READY:
            return lease
        now = self._now()
        accrued = accrued_cost(lease, now)
        if now >= lease.hard_deadline or over_authorized_cost(lease, now):
            return await self._release(lease, reason="deadline_or_cost_cap")
        touched = self.repository.compare_and_set(
            lease,
            changes={
                "last_used_at": now,
                "idle_deadline": min(
                    now + timedelta(seconds=lease.idle_timeout_seconds),
                    lease.hard_deadline,
                ),
                "accrued_estimated_cost": accrued,
            },
        )
        return replace(
            touched,
            route_url=lease.route_url,
            provider_health_url=lease.provider_health_url,
        )

    async def release(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        lease = self._required(lease_id)
        self._authorize(lease, owner_id, workload_id)
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        ):
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' has historical unknown "
                "provider authority and requires manual reconciliation"
            )
        return await self._release(lease, reason="caller_release")

    async def reconcile(self) -> tuple[OllamaLease, ...]:
        """Run one pass suitable for an external timer or long-lived service."""

        results: list[OllamaLease] = []
        now = self._now()
        for initial in self.repository.list_for_reconciliation(
            now=now, limit=self.reconciliation_batch_size
        ):
            claimed = self.repository.claim_for_reconciliation(
                initial,
                now=now,
                retry_at=self._next_reconcile_at(initial, now=now),
            )
            if claimed is None:
                continue
            try:
                lease = await self._reconcile_one(claimed.lease_id)
            except RunPodManagerError as exc:
                lease = self._record_reconcile_error(claimed.lease_id, exc)
            if lease is not None:
                results.append(lease)
        return tuple(results)

    def _next_reconcile_at(self, lease: OllamaLease, *, now: datetime) -> datetime:
        """Return the next durable visit without delaying an original cutoff."""

        retry = now + timedelta(seconds=self.poll_interval_seconds)
        if lease.state in {
            OllamaLeaseState.REQUESTED,
            OllamaLeaseState.PROVISIONING,
            OllamaLeaseState.RECONCILE_REQUIRED,
            OllamaLeaseState.WAITING_FOR_MODEL,
        }:
            for cutoff in (
                lease.hard_deadline,
                lease.readiness_deadline,
                authorized_cost_deadline(lease),
            ):
                if now < cutoff < retry:
                    retry = cutoff
        elif lease.state is OllamaLeaseState.READY:
            for cutoff in (
                lease.hard_deadline,
                lease.idle_deadline,
                authorized_cost_deadline(lease),
            ):
                if now < cutoff < retry:
                    retry = cutoff
        return retry

    async def reconcile_lease(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        """Reconcile one owner-scoped lease without exposing other tenants."""

        lease = self._required(lease_id)
        self._authorize(lease, owner_id, workload_id)
        try:
            reconciled = await self._reconcile_one(lease_id)
        except RunPodManagerError as exc:
            recorded = self._record_reconcile_error(lease_id, exc)
            if recorded is None:
                raise
            return recorded
        if reconciled is None:
            raise RunPodManagerError(f"Ollama lease '{lease_id}' was not found")
        return reconciled

    async def _reconcile_one(self, lease_id: str) -> OllamaLease | None:
        lease = self.repository.get(lease_id)
        if lease is None:
            return lease
        if lease.state is OllamaLeaseState.TERMINATED:
            if lease.additional_cleanup_targets:
                return lease
            if lease.lifecycle_billing_state in {
                OllamaLifecycleBillingState.PENDING,
                OllamaLifecycleBillingState.SELECTED,
                OllamaLifecycleBillingState.CORRECTION_PENDING,
            } and (
                lease.billing_next_check_at is None
                or lease.billing_next_check_at <= self._now()
            ):
                return await self._reconcile_lifecycle_billing(lease)
            return lease
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        ):
            return lease
        now = self._now()
        if lease.state is OllamaLeaseState.RELEASING:
            try:
                return await self._release(lease, reason="retry_teardown")
            except OllamaLeaseTeardownError:
                return self._required(lease.lease_id)
        if (
            lease.provisioning_mismatch is not None
            and lease.provider_resource_id is not None
        ):
            try:
                return await self._release(
                    lease, reason="provisioning_evidence_mismatch"
                )
            except OllamaLeaseTeardownError:
                return self._required(lease.lease_id)
        if lease.provisioning_mismatch is not None:
            return lease
        if lease.state is OllamaLeaseState.REQUESTED:
            if _provisioning_deadline_reached(lease, now):
                return await self._release(lease, reason="expired")
            lease = await self._provision_requested(lease, request_from_lease(lease))
        if lease.state in {
            OllamaLeaseState.PROVISIONING,
            OllamaLeaseState.RECONCILE_REQUIRED,
        }:
            lease = await self._reconcile_creation(lease)
        hard_expired = now >= lease.hard_deadline
        idle_expired = (
            lease.state is OllamaLeaseState.READY and now >= lease.idle_deadline
        )
        over_cost = over_authorized_cost(lease, now)
        readiness_expired = (
            lease.state
            in {
                OllamaLeaseState.PROVISIONING,
                OllamaLeaseState.RECONCILE_REQUIRED,
                OllamaLeaseState.WAITING_FOR_MODEL,
            }
            and now >= lease.readiness_deadline
        )
        if hard_expired or idle_expired or over_cost or readiness_expired:
            try:
                return await self._release(lease, reason="expired")
            except OllamaLeaseTeardownError:
                return self._required(lease.lease_id)
        if lease.state is OllamaLeaseState.WAITING_FOR_MODEL:
            return await self._check_readiness_once(lease)
        if lease.state is OllamaLeaseState.READY:
            return await self._ready_with_route(lease)
        return lease

    def _record_reconcile_error(
        self, lease_id: str, error: RunPodManagerError
    ) -> OllamaLease | None:
        """Keep one failed lease observable without starving the remaining pass."""

        lease = self.repository.get(lease_id)
        if lease is None or lease.state is OllamaLeaseState.TERMINATED:
            return lease
        try:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "last_provider_error": (
                        f"reconcile: {sanitize_provider_error(error)}"
                    )
                },
            )
        except OllamaLeaseConflictError:
            logger.warning(
                "Ollama lease %s changed again while recording reconcile failure",
                lease_id,
            )
            return self.repository.get(lease_id)

    async def _wait_until_ready(self, lease: OllamaLease) -> OllamaLease:
        while lease.state is OllamaLeaseState.WAITING_FOR_MODEL:
            lease = await self._check_readiness_once(lease)
            if lease.state is OllamaLeaseState.READY:
                return lease
            if (
                lease.state is OllamaLeaseState.TERMINATED
                and lease.termination_reason == "readiness_timeout"
            ):
                raise OllamaLeaseReadinessError(
                    f"Ollama lease '{lease.lease_id}' did not become model-ready"
                )
            now = self._now()
            if lease.state is OllamaLeaseState.WAITING_FOR_MODEL and (
                now >= lease.hard_deadline or over_authorized_cost(lease, now)
            ):
                return await self._release(lease, reason="deadline_or_cost_cap")
            if now >= lease.readiness_deadline:
                try:
                    await self._release(lease, reason="readiness_timeout")
                except OllamaLeaseTeardownError:
                    pass
                raise OllamaLeaseReadinessError(
                    f"Ollama lease '{lease.lease_id}' did not become model-ready"
                )
            await self._sleep(self.poll_interval_seconds)
        return lease

    async def _check_readiness_once(self, lease: OllamaLease) -> OllamaLease:
        resource = resource_from_lease(lease)
        intent, provisioning = self._provisioning_authority(lease)
        try:
            observation = await self.provider.observe(resource, intent, provisioning)
        except RunPodManagerError as exc:
            now = self._now()
            failed = self.repository.compare_and_set(
                lease,
                changes={
                    "last_provider_error": sanitize_provider_error(exc),
                    "accrued_estimated_cost": accrued_cost(lease, now),
                },
            )
            if now >= failed.hard_deadline or over_authorized_cost(failed, now):
                return await self._release(failed, reason="deadline_or_cost_cap")
            if now >= failed.readiness_deadline:
                return await self._release(failed, reason="readiness_timeout")
            return failed
        now = self._now()
        telemetry_committed = False
        if observation.runtime_telemetry is not None:
            lease = self.repository.record_runtime_telemetry(
                lease,
                observation.runtime_telemetry,
                now=now,
                maximum_observations=self.maximum_telemetry_observations,
                maximum_bytes=self.maximum_evidence_bytes,
            )
            telemetry_committed = (
                lease.runtime_telemetry_state
                is OllamaRuntimeTelemetryState.STARTUP_COMPLETE
                and any(
                    item.observation_id == observation.runtime_telemetry.observation_id
                    for item in lease.runtime_telemetry_observations
                )
            )
        accrued = accrued_cost(lease, now)
        lease = self.repository.compare_and_set(
            lease,
            changes={"accrued_estimated_cost": accrued},
        )
        if now >= lease.hard_deadline or over_authorized_cost(lease, now):
            return await self._release(lease, reason="deadline_or_cost_cap")
        if now >= lease.readiness_deadline:
            return await self._release(lease, reason="readiness_timeout")
        if (
            not observation.provider_ready
            or not observation.route_url
            or not telemetry_committed
        ):
            return lease
        if not observation.has_model(lease.model):
            lease = self.repository.compare_and_set(
                lease,
                changes={
                    "model_pull_started_at": lease.model_pull_started_at or now,
                    "model_pull_attempts": lease.model_pull_attempts + 1,
                    "last_provider_error": None,
                },
            )
            try:
                await self.provider.pull_model(
                    resource, observation.route_url, lease.model
                )
            except RunPodManagerError as exc:
                return self.repository.compare_and_set(
                    lease, changes={"last_provider_error": sanitize_provider_error(exc)}
                )
            now = self._now()
            if now >= lease.hard_deadline or over_authorized_cost(lease, now):
                return await self._release(lease, reason="deadline_or_cost_cap")
            if now >= lease.readiness_deadline:
                return await self._release(lease, reason="readiness_timeout")
            return lease
        provisioning_started = lease.provisioning_started_at or lease.created_at
        ready = self._transition(
            lease,
            OllamaLeaseState.READY,
            ready_at=now,
            model_ready_at=now,
            last_used_at=now,
            idle_deadline=min(
                now + timedelta(seconds=lease.idle_timeout_seconds),
                lease.hard_deadline,
            ),
            cold_start_seconds=(now - provisioning_started).total_seconds(),
            last_provider_error=None,
        )
        return replace(
            ready,
            route_url=observation.route_url,
            provider_health_url=observation.provider_health_url,
        )

    async def _ready_with_route(self, lease: OllamaLease) -> OllamaLease:
        """Re-observe a ready lease and attach its route only in host memory."""

        now = self._now()
        accrued = accrued_cost(lease, now)
        if (
            now >= lease.hard_deadline
            or now >= lease.idle_deadline
            or over_authorized_cost(lease, now)
        ):
            return await self._release(lease, reason="deadline_or_cost_cap")
        try:
            intent, provisioning = self._provisioning_authority(lease)
            observation = await self.provider.observe(
                resource_from_lease(lease), intent, provisioning
            )
        except RunPodManagerError as exc:
            now = self._now()
            accrued = accrued_cost(lease, now)
            if (
                now >= lease.hard_deadline
                or now >= lease.idle_deadline
                or over_authorized_cost(lease, now)
            ):
                return await self._release(lease, reason="deadline_or_cost_cap")
            self.repository.compare_and_set(
                lease,
                changes={
                    "last_provider_error": sanitize_provider_error(exc),
                    "accrued_estimated_cost": accrued,
                },
            )
            raise OllamaLeaseReadinessError(
                f"Ollama lease '{lease.lease_id}' ready route could not be observed"
            ) from exc
        now = self._now()
        if observation.runtime_telemetry is not None:
            lease = self.repository.record_runtime_telemetry(
                lease,
                observation.runtime_telemetry,
                now=now,
                maximum_observations=self.maximum_telemetry_observations,
                maximum_bytes=self.maximum_evidence_bytes,
            )
        accrued = accrued_cost(lease, now)
        if (
            now >= lease.hard_deadline
            or now >= lease.idle_deadline
            or over_authorized_cost(lease, now)
        ):
            return await self._release(lease, reason="deadline_or_cost_cap")
        if (
            not observation.provider_ready
            or not observation.route_url
            or not observation.has_model(lease.model)
            or observation.runtime_telemetry is None
            or not observation.runtime_telemetry.is_complete_startup_for(intent)
        ):
            self.repository.compare_and_set(
                lease,
                changes={
                    "last_provider_error": "ready route did not pass exact-model readiness",
                    "accrued_estimated_cost": accrued,
                },
            )
            raise OllamaLeaseReadinessError(
                f"Ollama lease '{lease.lease_id}' ready route is temporarily unavailable"
            )
        durable = self.repository.compare_and_set(
            lease,
            changes={
                "accrued_estimated_cost": accrued,
                "last_provider_error": None,
            },
        )
        return replace(
            durable,
            route_url=observation.route_url,
            provider_health_url=observation.provider_health_url,
        )

    async def _reconcile_creation(self, lease: OllamaLease) -> OllamaLease:
        intent = lease.provisioning_intent
        if intent is None:
            return lease
        try:
            if lease.provider_resource_id:
                resource = resource_from_lease(lease)
                reconciled_after_ambiguous_create = (
                    lease.ambiguous_create_candidate_id == lease.provider_resource_id
                )
            elif lease.ambiguous_create_candidate_id:
                resource = _ambiguous_candidate_from_lease(lease)
                reconciled_after_ambiguous_create = True
            else:
                resource = await self.provider.discover_provisioning_resource(intent)
                if resource is None:
                    return self.repository.record_provisioning_outcome(
                        lease,
                        now=self._now(),
                        provider_error=(
                            "ambiguous create discovery returned no attributable "
                            "resource"
                        ),
                    )
                lease = self.repository.record_provisioning_outcome(
                    lease,
                    now=self._now(),
                    ambiguous_candidate=resource,
                )
                reconciled_after_ambiguous_create = True
            receipt = await self.provider.observe_provisioning(
                intent,
                resource,
                reconciled_after_ambiguous_create=(reconciled_after_ambiguous_create),
            )
            _validate_provisioning_receipt(receipt, intent)
        except OllamaProvisioningMismatchError as exc:
            lease = self._persist_provisioning_mismatch(lease, exc)
            if lease.provider_resource_id is not None:
                return await self._release(
                    lease, reason="provisioning_evidence_mismatch"
                )
            return lease
        except RunPodManagerError as exc:
            failed = self.repository.record_provisioning_outcome(
                lease,
                now=self._now(),
                provider_error=sanitize_provider_error(exc),
            )
            if (
                failed.provider_resource_id is not None
                and _provisioning_deadline_reached(failed, self._now())
            ):
                return await self._release(failed, reason="expired")
            return failed
        completed = self.repository.record_provisioning_outcome(
            lease,
            now=self._now(),
            receipt=receipt,
        )
        if _provisioning_deadline_reached(completed, self._now()):
            return await self._release(completed, reason="expired")
        return completed

    async def _release(self, lease: OllamaLease, *, reason: str) -> OllamaLease:
        if (
            lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        ):
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' has historical unknown "
                "provider authority and requires manual reconciliation"
            )
        if lease.state is OllamaLeaseState.TERMINATED:
            return lease
        now = self._now()
        if (
            lease.state in {OllamaLeaseState.REQUESTED, OllamaLeaseState.FAILED}
            and lease.provisioning_evidence_state
            is OllamaProvisioningEvidenceState.REQUIRED
            and lease.provisioning_intent is None
            and lease.provider_resource_id is None
            and lease.ambiguous_create_candidate_id is None
            and lease.provision_attempt_id is None
            and lease.provision_attempts == 0
            and lease.creation_uncertain is False
        ):
            return self._transition(
                lease,
                OllamaLeaseState.TERMINATED,
                teardown_state=OllamaTeardownState.COMPLETE,
                accrued_estimated_cost=accrued_cost(lease, now),
                last_provider_error=(
                    lease.last_provider_error
                    if lease.provisioning_mismatch is not None
                    else None
                ),
                termination_reason=lease.termination_reason or reason,
                lifecycle_billing_state=OllamaLifecycleBillingState.NOT_APPLICABLE,
            )
        if lease.state is not OllamaLeaseState.RELEASING:
            lease = self._transition(
                lease,
                OllamaLeaseState.RELEASING,
                teardown_state=OllamaTeardownState.PENDING,
                route_url=None,
                accrued_estimated_cost=accrued_cost(lease, now),
                last_provider_error=(
                    lease.last_provider_error
                    if lease.provisioning_mismatch is not None
                    else None
                ),
                termination_reason=lease.termination_reason or reason,
            )
        intent = lease.provisioning_intent
        if intent is None:
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' has no original cleanup lineage"
            )
        key = (lease.lease_id, intent.intent_sha256)
        task = self._cleanup_tasks.get(key)
        if task is not None:
            return await await_owned_effect(task)
        task = asyncio.create_task(
            self._cleanup_effect(lease, reason=reason),
            name=f"ollama-cleanup-{lease.lease_id}",
        )
        self._cleanup_tasks[key] = task

        def forget(completed: asyncio.Task[OllamaLease]) -> None:
            if self._cleanup_tasks.get(key) is completed:
                self._cleanup_tasks.pop(key, None)

        task.add_done_callback(forget)
        return await await_owned_effect(task)

    async def _cleanup_effect(self, lease: OllamaLease, *, reason: str) -> OllamaLease:
        """Own uncertainty resolution and exact-ID teardown as one effect."""

        if lease.creation_uncertain:
            lease = await self._resolve_uncertain_release(lease, reason=reason)
            if lease.creation_uncertain:
                raise OllamaLeaseTeardownError(
                    f"Ollama lease '{lease.lease_id}' creation is still reconciling"
                )
        if not lease.provider_resource_id:
            return self._transition(
                lease,
                OllamaLeaseState.TERMINATED,
                teardown_state=OllamaTeardownState.COMPLETE,
                last_provider_error=(
                    lease.last_provider_error
                    if lease.provisioning_mismatch is not None
                    else None
                ),
                termination_reason=lease.termination_reason or reason,
            )
        lease = self.repository.record_cleanup_intent(
            lease,
            now=self._now(),
            termination_reason=reason,
        )
        telemetry_control_error: BaseException | None = None
        try:
            if (
                lease.cleanup_receipt is None
                and lease.provisioning_intent is not None
                and lease.provisioning_receipt is not None
                and lease.provisioning_mismatch is None
            ):
                try:
                    final_observation = await self.provider.observe(
                        resource_from_lease(lease),
                        lease.provisioning_intent,
                        lease.provisioning_receipt,
                    )
                except RunPodManagerError as exc:
                    lease = self.repository.record_runtime_telemetry(
                        lease,
                        None,
                        now=self._now(),
                        maximum_observations=self.maximum_telemetry_observations,
                        maximum_bytes=self.maximum_evidence_bytes,
                        provider_error=(
                            f"final telemetry: {sanitize_provider_error(exc)}"
                        ),
                    )
                else:
                    if final_observation.runtime_telemetry is not None:
                        lease = self.repository.record_runtime_telemetry(
                            lease,
                            final_observation.runtime_telemetry,
                            now=self._now(),
                            maximum_observations=self.maximum_telemetry_observations,
                            maximum_bytes=self.maximum_evidence_bytes,
                        )
                    else:
                        lease = self.repository.record_runtime_telemetry(
                            lease,
                            None,
                            now=self._now(),
                            maximum_observations=self.maximum_telemetry_observations,
                            maximum_bytes=self.maximum_evidence_bytes,
                            provider_error=(
                                "final telemetry: runtime telemetry is unavailable"
                            ),
                        )
        except BaseException as exc:
            if not is_execution_control_error(exc):
                raise
            telemetry_control_error = exc
        if telemetry_control_error is None:
            lease = self.repository.claim_cleanup_attempt(lease, now=self._now())
            return await self._teardown_all_effects(lease, reason=reason)
        try:
            lease = self.repository.claim_cleanup_attempt(lease, now=self._now())
            await self._teardown_all_effects(lease, reason=reason)
        # Preserve both original native control evidence and every possible
        # cleanup terminal, including cancellation/control BaseExceptions.
        except BaseException as cleanup_error:  # noqa: BLE001
            raise BaseExceptionGroup(
                "Ollama final telemetry control and cleanup both failed",
                [telemetry_control_error, cleanup_error],
            ) from None
        raise telemetry_control_error.with_traceback(
            telemetry_control_error.__traceback__
        )

    async def _teardown_all_effects(
        self, lease: OllamaLease, *, reason: str
    ) -> OllamaLease:
        """Join every exact resource proven for one admitted create lineage."""

        if lease.cleanup_receipt is None:
            lease = await self._teardown_effect(lease, reason=reason)
        for target in lease.additional_cleanup_targets:
            if target.cleanup_receipt is None:
                lease = await self._teardown_additional_effect(
                    lease,
                    provider_resource_id=target.provider_resource_id,
                    reason=reason,
                )
        return lease

    async def _teardown_effect(self, lease: OllamaLease, *, reason: str) -> OllamaLease:
        """Join physical deletion, exact absence readback and receipt persistence."""

        try:
            resource = resource_from_lease(lease)
            cleanup_intent = lease.cleanup_intent
            if cleanup_intent is None:
                raise OllamaLeaseTeardownError(
                    f"Ollama lease '{lease.lease_id}' cleanup intent disappeared"
                )
            cleanup_receipt = await self.provider.teardown(resource, cleanup_intent)
        except RunPodManagerError as exc:
            self.repository.record_teardown_outcome(
                lease,
                provider_resource_id=resource.provider_resource_id,
                now=self._now(),
                accrued_estimated_cost=accrued_cost(lease, self._now()),
                termination_reason=reason,
                provider_error=f"{reason}: {sanitize_provider_error(exc)}",
                retry_at=self._now() + timedelta(seconds=self.poll_interval_seconds),
            )
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' teardown remains retryable"
            ) from exc
        return self.repository.record_teardown_outcome(
            lease,
            provider_resource_id=resource.provider_resource_id,
            now=self._now(),
            accrued_estimated_cost=accrued_cost(lease, self._now()),
            termination_reason=reason,
            cleanup_receipt=cleanup_receipt,
            billing_next_check_at=self._now(),
        )

    async def _teardown_additional_effect(
        self,
        lease: OllamaLease,
        *,
        provider_resource_id: str,
        reason: str,
    ) -> OllamaLease:
        """Join cleanup of one separately retained original POST ACK."""

        lease = self.repository.record_additional_cleanup_intent(
            lease,
            provider_resource_id=provider_resource_id,
            now=self._now(),
            termination_reason=reason,
        )
        target = next(
            item
            for item in lease.additional_cleanup_targets
            if item.provider_resource_id == provider_resource_id
        )
        cleanup_intent = target.cleanup_intent
        if cleanup_intent is None:
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' additional cleanup intent disappeared"
            )
        resource = ProvisionedOllamaResource(
            resource_type=target.resource_type,
            provider_resource_id=target.provider_resource_id,
            resource_name=target.resource_name,
        )
        try:
            cleanup_receipt = await self.provider.teardown(resource, cleanup_intent)
        except RunPodManagerError as exc:
            self.repository.record_additional_cleanup_outcome(
                lease,
                provider_resource_id=provider_resource_id,
                now=self._now(),
                accrued_estimated_cost=accrued_cost(lease, self._now()),
                termination_reason=reason,
                provider_error=f"{reason}: {sanitize_provider_error(exc)}",
                retry_at=self._now() + timedelta(seconds=self.poll_interval_seconds),
            )
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' additional teardown remains retryable"
            ) from exc
        return self.repository.record_additional_cleanup_outcome(
            lease,
            provider_resource_id=provider_resource_id,
            now=self._now(),
            accrued_estimated_cost=accrued_cost(lease, self._now()),
            termination_reason=reason,
            cleanup_receipt=cleanup_receipt,
        )

    async def _resolve_uncertain_release(
        self, lease: OllamaLease, *, reason: str
    ) -> OllamaLease:
        intent = lease.provisioning_intent
        if intent is None:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": (
                        f"{reason}: uncertain creation has no original intent"
                    ),
                },
            )
        authoritative = lease.provider_resource_id is not None
        reconciled_after_ambiguous_create = False
        if authoritative:
            resource = resource_from_lease(lease)
            reconciled_after_ambiguous_create = (
                lease.ambiguous_create_candidate_id == lease.provider_resource_id
            )
        elif lease.provisioning_mismatch is not None:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": (
                        f"{reason}: ambiguous resource identity is untrusted"
                    ),
                },
            )
        elif lease.ambiguous_create_candidate_id:
            resource = _ambiguous_candidate_from_lease(lease)
            reconciled_after_ambiguous_create = True
        else:
            try:
                resource = await self.provider.discover_provisioning_resource(intent)
            except RunPodManagerError as exc:
                return self.repository.record_provisioning_outcome(
                    lease,
                    now=self._now(),
                    provider_error=f"{reason}: {sanitize_provider_error(exc)}",
                )
            if resource is None:
                return self.repository.record_provisioning_outcome(
                    lease,
                    now=self._now(),
                    provider_error=(
                        f"{reason}: awaiting ambiguous creation reconciliation"
                    ),
                )
            lease = self.repository.record_provisioning_outcome(
                lease,
                now=self._now(),
                ambiguous_candidate=resource,
            )
            reconciled_after_ambiguous_create = True
        try:
            receipt = await self.provider.observe_provisioning(
                intent,
                resource,
                reconciled_after_ambiguous_create=(reconciled_after_ambiguous_create),
            )
            _validate_provisioning_receipt(receipt, intent)
        except OllamaProvisioningMismatchError as exc:
            return self._persist_provisioning_mismatch(lease, exc)
        except OllamaProvisioningEvidenceError as exc:
            return self.repository.record_provisioning_outcome(
                lease,
                now=self._now(),
                provider_error=f"{reason}: {sanitize_provider_error(exc)}",
            )
        except RunPodManagerError as exc:
            return self.repository.record_provisioning_outcome(
                lease,
                now=self._now(),
                provider_error=f"{reason}: {sanitize_provider_error(exc)}",
            )
        return self.repository.record_provisioning_outcome(
            lease,
            now=self._now(),
            receipt=receipt,
        )

    async def _reconcile_lifecycle_billing(self, lease: OllamaLease) -> OllamaLease:
        intent = lease.provisioning_intent
        provisioning = lease.provisioning_receipt
        cleanup = lease.cleanup_receipt
        if intent is None or provisioning is None or cleanup is None:
            raise RunPodManagerError(
                f"Ollama lease '{lease.lease_id}' lacks lifecycle billing authority"
            )
        key = (lease.lease_id, cleanup.cleanup_receipt_id)
        task = self._billing_tasks.get(key)
        if task is not None:
            return await await_owned_effect(task)
        task = asyncio.create_task(
            self._billing_effect(lease),
            name=f"ollama-billing-{lease.lease_id}",
        )
        self._billing_tasks[key] = task

        def forget(completed: asyncio.Task[OllamaLease]) -> None:
            if self._billing_tasks.get(key) is completed:
                self._billing_tasks.pop(key, None)

        task.add_done_callback(forget)
        return await await_owned_effect(task)

    async def _billing_effect(self, lease: OllamaLease) -> OllamaLease:
        intent = lease.provisioning_intent
        provisioning = lease.provisioning_receipt
        cleanup_intent = lease.cleanup_intent
        cleanup = lease.cleanup_receipt
        if (
            intent is None
            or provisioning is None
            or cleanup_intent is None
            or cleanup is None
        ):
            raise RunPodManagerError(
                f"Ollama lease '{lease.lease_id}' lacks lifecycle billing evidence"
            )
        try:
            observation = await self.provider.observe_lifecycle_billing(
                intent, provisioning, cleanup
            )
            if observation is not None and not isinstance(
                observation, OllamaLifecycleBillingObservation
            ):
                raise RunPodManagerError(
                    "Ollama provider returned invalid lifecycle billing evidence"
                )
            if observation is not None:
                observation.validate_lifecycle(
                    intent, provisioning, cleanup_intent, cleanup
                )
        except RunPodManagerError as exc:
            checked_at = self._now()
            return self.repository.record_billing_outcome(
                lease,
                now=checked_at,
                next_check_at=checked_at
                + timedelta(seconds=self.billing_poll_interval_seconds),
                provider_error=sanitize_provider_error(exc),
                maximum_observations=self.maximum_billing_observations,
                maximum_bytes=self.maximum_evidence_bytes,
            )
        if observation is None:
            checked_at = self._now()
            return self.repository.record_billing_outcome(
                lease,
                now=checked_at,
                next_check_at=checked_at
                + timedelta(seconds=self.billing_poll_interval_seconds),
                provider_error="provider lifecycle billing history is incomplete",
                maximum_observations=self.maximum_billing_observations,
                maximum_bytes=self.maximum_evidence_bytes,
            )
        checked_at = self._now()
        return self.repository.record_billing_outcome(
            lease,
            now=checked_at,
            next_check_at=checked_at
            + timedelta(seconds=self.billing_poll_interval_seconds),
            observation=observation,
            maximum_observations=self.maximum_billing_observations,
            maximum_bytes=self.maximum_evidence_bytes,
        )

    @staticmethod
    def _provisioning_authority(
        lease: OllamaLease,
    ) -> tuple[OllamaProvisioningIntent, OllamaProvisioningReceipt]:
        intent = lease.provisioning_intent
        receipt = lease.provisioning_receipt
        if intent is None or receipt is None:
            raise OllamaProvisioningEvidenceError(
                f"Ollama lease '{lease.lease_id}' lacks readiness authority"
            )
        return intent, receipt

    def _persist_provisioning_mismatch(
        self,
        lease: OllamaLease,
        error: OllamaProvisioningMismatchError,
        *,
        acknowledged_create: bool = False,
    ) -> OllamaLease:
        intent = lease.provisioning_intent
        if intent is None:
            raise OllamaProvisioningEvidenceError(
                "Ollama provisioning mismatch has no original intent"
            ) from error
        evidence = error.evidence
        _validate_provisioning_mismatch(evidence, intent)
        if acknowledged_create:
            if lease.provider_resource_id not in {
                None,
                evidence.provider_resource_id,
            }:
                raise OllamaProvisioningEvidenceError(
                    "Ollama acknowledged create mismatch identifies another resource"
                ) from error
            authoritative = True
        else:
            authoritative = lease.provider_resource_id == evidence.provider_resource_id
        if not authoritative and (
            lease.ambiguous_create_candidate_id != evidence.provider_resource_id
        ):
            raise OllamaProvisioningEvidenceError(
                "Ollama provisioning mismatch identifies another resource"
            ) from error
        acknowledged_resource = (
            ProvisionedOllamaResource(
                resource_type=intent.resource_type,
                provider_resource_id=evidence.provider_resource_id,
                resource_name=evidence.observed_resource_name,
            )
            if acknowledged_create
            else None
        )
        return self.repository.record_provisioning_outcome(
            lease,
            now=self._now(),
            acknowledged_resource=acknowledged_resource,
            mismatch=evidence,
            candidate_mismatch=not authoritative,
            provider_error=sanitize_provider_error(error),
        )

    def _transition(
        self,
        lease: OllamaLease,
        target: OllamaLeaseState,
        **changes: Any,
    ) -> OllamaLease:
        if (
            target is not lease.state
            and target not in _ALLOWED_TRANSITIONS[lease.state]
        ):
            raise RunPodManagerError(
                f"Invalid Ollama lease transition {lease.state.value} -> {target.value}"
            )
        if target is OllamaLeaseState.RELEASING and lease.state is not target:
            changes.setdefault("reconcile_after", self._now())
        return self.repository.compare_and_set(
            lease, changes={"state": target, **changes}
        )

    def _required(self, lease_id: str) -> OllamaLease:
        lease = self.repository.get(lease_id)
        if lease is None:
            raise RunPodManagerError(f"Ollama lease '{lease_id}' was not found")
        return lease

    @staticmethod
    def _authorize(lease: OllamaLease, owner_id: str, workload_id: str) -> None:
        if lease.owner_id != owner_id or lease.workload_id != workload_id:
            raise OllamaLeaseAuthorizationError("Ollama lease ownership mismatch")

    @staticmethod
    def validate_plan(request: OllamaLeaseRequest, plan: OllamaPlacementPlan) -> None:
        """Validate a capacity plan against the caller's durable cost policy."""

        allowed_modes = (
            {request.mode}
            if request.mode is not OllamaLeaseMode.AUTO
            else {
                OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
                OllamaLeaseMode.DEDICATED_POD,
            }
        )
        if plan.mode not in allowed_modes:
            raise RunPodManagerError("Provider returned an unauthorized Ollama mode")
        expected_resource_type = (
            OllamaResourceType.SERVERLESS_ENDPOINT
            if plan.mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            else OllamaResourceType.POD
        )
        if plan.resource_type is not expected_resource_type:
            raise RunPodManagerError(
                "Provider returned an inconsistent Ollama resource type"
            )
        if plan.placement.requirements.product is not (
            ComputeProduct.SERVERLESS
            if plan.mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            else ComputeProduct.POD
        ):
            raise RunPodManagerError(
                "Provider returned an inconsistent Ollama catalog product"
            )
        if (
            not math.isfinite(plan.estimated_cost)
            or plan.estimated_cost < 0
            or not math.isfinite(plan.cost_ceiling)
            or plan.cost_ceiling < plan.estimated_cost
            or not math.isfinite(plan.placement.offered_cost_per_hr)
            or plan.placement.offered_cost_per_hr <= 0
            or not isinstance(plan.estimated_billable_seconds, int)
            or isinstance(plan.estimated_billable_seconds, bool)
            or plan.estimated_billable_seconds <= 0
        ):
            raise RunPodManagerError("Provider returned an invalid Ollama cost plan")
        if plan.cost_ceiling > request.max_authorized_cost:
            raise RunPodManagerError(
                "Provider Ollama all-in cost ceiling exceeds the maximum authorized cost"
            )
        if (
            request.constraints.max_hourly_rate is not None
            and plan.placement.offered_cost_per_hr > request.constraints.max_hourly_rate
        ):
            raise RunPodManagerError(
                "Provider Ollama plan exceeds the maximum authorized hourly rate"
            )

    def _now(self) -> datetime:
        value = self._clock()
        require_aware(value, "clock")
        return value
