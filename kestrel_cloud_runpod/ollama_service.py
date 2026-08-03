"""Restart-safe lifecycle orchestration for private-Ollama leases."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import (
    ComputeProduct,
    RunPodAmbiguousResultError,
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
    OllamaPlacementPlan,
    OllamaResourceType,
    OllamaTeardownState,
    accrued_cost,
    authorized_cost_exposure,
    provision_attempt_id,
    require_aware,
    resource_from_lease,
    resource_name,
    sanitize_provider_error,
)
from .ollama_repository import SQLiteOllamaLeaseRepository, request_from_lease

logger = logging.getLogger(__name__)

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
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("Ollama lease poll interval must be positive")
        self.repository = repository
        self.provider = provider
        self.poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._sleep = sleep

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
            if request.hard_deadline <= now:
                raise ValueError("Ollama lease hard_deadline must be in the future")
            if (
                request.expected_session_seconds
                > (request.hard_deadline - now).total_seconds()
            ):
                raise ValueError("Expected Ollama session exceeds its hard deadline")
            requested_at = request.requested_at or now
            if now >= min(
                requested_at + timedelta(seconds=request.readiness_timeout_seconds),
                request.hard_deadline,
            ):
                raise ValueError("Ollama readiness deadline has already elapsed")
        lease, inserted = self.repository.insert_request(request, now=now)
        if not inserted:
            self._authorize(lease, request.owner_id, request.workload_id)
        if lease.state is OllamaLeaseState.READY:
            return await self._ready_with_route(lease)
        if lease.state is OllamaLeaseState.REQUESTED:
            if now >= min(lease.readiness_deadline, lease.hard_deadline):
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
            self.validate_plan(request, plan)
        except RunPodManagerError as exc:
            self._transition(
                lease,
                OllamaLeaseState.FAILED,
                last_provider_error=sanitize_provider_error(exc),
            )
            raise
        provider_name = resource_name(request.lease_id)
        now = self._now()
        lease = self._transition(
            lease,
            OllamaLeaseState.PROVISIONING,
            mode=plan.mode,
            resource_type=plan.resource_type,
            resource_name=provider_name,
            creation_uncertain=True,
            provision_attempt_id=provision_attempt_id(request),
            provision_attempts=lease.provision_attempts + 1,
            provisioning_started_at=now,
            offered_rate_per_hr=plan.placement.offered_cost_per_hr,
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
        )
        try:
            resource = await self.provider.provision(
                request=request, plan=plan, resource_name=provider_name
            )
        except RunPodAmbiguousResultError as exc:
            self._transition(
                lease,
                OllamaLeaseState.RECONCILE_REQUIRED,
                last_provider_error=sanitize_provider_error(exc),
            )
            raise
        except RunPodManagerError as exc:
            self._transition(
                lease,
                OllamaLeaseState.FAILED,
                creation_uncertain=False,
                last_provider_error=sanitize_provider_error(exc),
            )
            raise
        lease = self._transition(
            lease,
            OllamaLeaseState.WAITING_FOR_MODEL,
            provider_resource_id=resource.provider_resource_id,
            creation_uncertain=False,
            last_provider_error=None,
        )
        return lease

    async def get(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        lease = self._required(lease_id)
        self._authorize(lease, owner_id, workload_id)
        if lease.state is OllamaLeaseState.READY:
            return await self._ready_with_route(lease)
        return lease

    async def touch(
        self, lease_id: str, *, owner_id: str, workload_id: str
    ) -> OllamaLease:
        lease = self._required(lease_id)
        self._authorize(lease, owner_id, workload_id)
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
        if (
            now >= lease.hard_deadline
            or authorized_cost_exposure(lease, now) >= lease.max_authorized_cost
        ):
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
        return await self._release(lease, reason="caller_release")

    async def reconcile(self) -> tuple[OllamaLease, ...]:
        """Run one pass suitable for an external timer or long-lived service."""

        results: list[OllamaLease] = []
        for initial in self.repository.list_for_reconciliation():
            try:
                lease = await self._reconcile_one(initial.lease_id)
            except RunPodManagerError as exc:
                lease = self._record_reconcile_error(initial.lease_id, exc)
            if lease is not None:
                results.append(lease)
        return tuple(results)

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
        if lease is None or lease.state is OllamaLeaseState.TERMINATED:
            return lease
        now = self._now()
        if lease.state is OllamaLeaseState.RELEASING:
            try:
                return await self._release(lease, reason="retry_teardown")
            except OllamaLeaseTeardownError:
                return self._required(lease.lease_id)
        if lease.state is OllamaLeaseState.REQUESTED:
            if now >= min(lease.readiness_deadline, lease.hard_deadline):
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
        over_cost = authorized_cost_exposure(lease, now) >= (lease.max_authorized_cost)
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
            now = self._now()
            if lease.state is OllamaLeaseState.WAITING_FOR_MODEL and (
                now >= lease.hard_deadline
                or authorized_cost_exposure(lease, now) >= lease.max_authorized_cost
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
        now = self._now()
        accrued = accrued_cost(lease, now)
        try:
            observation = await self.provider.observe(resource)
        except RunPodManagerError as exc:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "last_provider_error": sanitize_provider_error(exc),
                    "accrued_estimated_cost": accrued,
                },
            )
        lease = self.repository.compare_and_set(
            lease,
            changes={"accrued_estimated_cost": accrued},
        )
        if (
            now >= lease.hard_deadline
            or authorized_cost_exposure(lease, now) >= lease.max_authorized_cost
        ):
            return await self._release(lease, reason="deadline_or_cost_cap")
        if not observation.provider_ready or not observation.route_url:
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
            or authorized_cost_exposure(lease, now) >= lease.max_authorized_cost
        ):
            return await self._release(lease, reason="deadline_or_cost_cap")
        try:
            observation = await self.provider.observe(resource_from_lease(lease))
        except RunPodManagerError as exc:
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
        if (
            not observation.provider_ready
            or not observation.route_url
            or not observation.has_model(lease.model)
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
        if lease.provider_resource_id:
            return self._transition(
                lease,
                OllamaLeaseState.WAITING_FOR_MODEL,
                last_provider_error=None,
            )
        if lease.resource_type is None or lease.resource_name is None:
            return lease
        try:
            resource = await self.provider.find_resource(
                resource_type=lease.resource_type, resource_name=lease.resource_name
            )
        except RunPodManagerError as exc:
            return self.repository.compare_and_set(
                lease, changes={"last_provider_error": sanitize_provider_error(exc)}
            )
        if resource is None:
            return lease
        return self._transition(
            lease,
            OllamaLeaseState.WAITING_FOR_MODEL,
            provider_resource_id=resource.provider_resource_id,
            creation_uncertain=False,
            last_provider_error=None,
        )

    async def _release(self, lease: OllamaLease, *, reason: str) -> OllamaLease:
        if lease.state is OllamaLeaseState.TERMINATED:
            return lease
        now = self._now()
        if lease.state is OllamaLeaseState.REQUESTED and not lease.provider_resource_id:
            return self._transition(
                lease,
                OllamaLeaseState.TERMINATED,
                teardown_state=OllamaTeardownState.COMPLETE,
                accrued_estimated_cost=accrued_cost(lease, now),
                last_provider_error=None,
                termination_reason=lease.termination_reason or reason,
            )
        if lease.state is not OllamaLeaseState.RELEASING:
            lease = self._transition(
                lease,
                OllamaLeaseState.RELEASING,
                teardown_state=OllamaTeardownState.PENDING,
                route_url=None,
                accrued_estimated_cost=accrued_cost(lease, now),
                last_provider_error=None,
                termination_reason=lease.termination_reason or reason,
            )
        if not lease.provider_resource_id:
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
                        last_provider_error=None,
                        termination_reason=lease.termination_reason or reason,
                    )
            else:
                return self._transition(
                    lease,
                    OllamaLeaseState.TERMINATED,
                    teardown_state=OllamaTeardownState.COMPLETE,
                    last_provider_error=None,
                    termination_reason=lease.termination_reason or reason,
                )
        lease = self.repository.compare_and_set(
            lease,
            changes={"teardown_attempts": lease.teardown_attempts + 1},
        )
        try:
            await self.provider.teardown(resource_from_lease(lease))
        except RunPodManagerError as exc:
            self.repository.compare_and_set(
                lease,
                changes={
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": (
                        f"{reason}: {sanitize_provider_error(exc)}"
                    ),
                    "accrued_estimated_cost": accrued_cost(lease, self._now()),
                },
            )
            raise OllamaLeaseTeardownError(
                f"Ollama lease '{lease.lease_id}' teardown remains retryable"
            ) from exc
        return self._transition(
            lease,
            OllamaLeaseState.TERMINATED,
            teardown_state=OllamaTeardownState.COMPLETE,
            last_provider_error=None,
            accrued_estimated_cost=accrued_cost(lease, self._now()),
            termination_reason=lease.termination_reason or reason,
        )

    async def _resolve_uncertain_release(
        self, lease: OllamaLease, *, reason: str
    ) -> OllamaLease:
        if lease.resource_type is None or lease.resource_name is None:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": (
                        f"{reason}: uncertain creation has no resource identity"
                    ),
                },
            )
        try:
            resource = await self.provider.find_resource(
                resource_type=lease.resource_type,
                resource_name=lease.resource_name,
            )
        except RunPodManagerError as exc:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": (
                        f"{reason}: {sanitize_provider_error(exc)}"
                    ),
                },
            )
        if resource is not None:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "provider_resource_id": resource.provider_resource_id,
                    "creation_uncertain": False,
                    "teardown_state": OllamaTeardownState.PENDING,
                    "last_provider_error": None,
                },
            )
        if self._now() < lease.readiness_deadline:
            return self.repository.compare_and_set(
                lease,
                changes={
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": (
                        f"{reason}: awaiting ambiguous creation reconciliation"
                    ),
                },
            )
        return self.repository.compare_and_set(
            lease,
            changes={
                "creation_uncertain": False,
                "last_provider_error": None,
            },
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
