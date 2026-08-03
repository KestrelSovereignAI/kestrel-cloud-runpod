"""Restart-safe acquisition and cleanup state machine for billable Pods."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from .models import GPUProfile, RunPodAmbiguousResultError, RunPodManagerError
from .pod_capacity_contracts import (
    TRAINING_PROFILE_IDS,
    CatalogAttemptCapability,
    CatalogAttemptCapabilityStore,
    CatalogPodCapacityRequest,
    CatalogPodWorkloadState,
    CatalogWorkerEvidence,
    PodBillingReceipt,
    PodCapacityBillingState,
    PodCapacityLease,
    PodCapacityQuote,
    PodCapacityQuoteRequest,
    PodCapacitySpec,
    PodRealizedPlacement,
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodLease,
    TrainingPodLifecycleError,
    TrainingPodOwnership,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
    attempt_environment_sha256,
    fallback_training_cleanup_token,
    iso_datetime,
    sanitize_training_error,
)
from .pod_capacity_provider import (
    PodCapacityCreatedMismatchError,
    TrainingPodCapacityProvider,
)
from .pod_capacity_repository import SQLiteTrainingPodRepository
from .pod_transport import (
    CatalogPodTransportError,
    CatalogPodWorkloadObservation,
    CatalogPodWorkloadTransport,
)

logger = logging.getLogger(__name__)
_EVIDENCE_UNSET = object()


class PodCapacityLeaseService:
    """Own every transition from pre-mutation claim through final billing."""

    def __init__(
        self,
        *,
        repository: SQLiteTrainingPodRepository,
        provider: TrainingPodCapacityProvider,
        profiles: Mapping[str, GPUProfile],
        poll_interval_seconds: float,
        orphan_timeout_seconds: float,
        workload_status_observer: Callable[[TrainingPodLease], Awaitable[str | None]]
        | None = None,
        capability_store: CatalogAttemptCapabilityStore | None = None,
        workload_transport: CatalogPodWorkloadTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0 or orphan_timeout_seconds <= 0:
            raise ValueError(
                "Training Pod polling and orphan timeouts must be positive"
            )
        self.repository = repository
        self.provider = provider
        self.profiles = profiles
        self.poll_interval_seconds = poll_interval_seconds
        self.orphan_timeout_seconds = orphan_timeout_seconds
        self._workload_status_observer = workload_status_observer
        self._capability_store = capability_store
        self._workload_transport = workload_transport
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep

    async def quote(self, request: PodCapacityQuoteRequest) -> PodCapacityQuote:
        """Return the provider's exact live Pod offer without mutating capacity."""

        return await self.provider.quote(request)

    def get_catalog_capacity(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
    ) -> PodCapacityLease:
        """Return owner-bound capacity and canonical settlement state."""

        lease = self.find_catalog_capacity(
            capacity_id=capacity_id,
            owner_id=owner_id,
            workload_id=workload_id,
        )
        if lease is None:
            raise RunPodManagerError(
                f"Catalog Pod capacity '{capacity_id}' was not found"
            )
        return lease

    def find_catalog_capacity(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
    ) -> PodCapacityLease | None:
        """Return bound capacity, or ``None`` without provisioning when absent."""

        lease = self.repository.get(capacity_id)
        if lease is None:
            return None
        spec = self._capacity_spec(lease)
        if spec.request.owner_id != owner_id or spec.request.workload_id != workload_id:
            raise TrainingPodConflictError(
                "Pod capacity owner/workload identity does not match the lease"
            )
        return lease

    def validate_catalog_reconciler_dependencies(self) -> None:
        """Fail before polling when required host-owned dependencies are absent."""

        self._require_capability_store()
        self._require_workload_transport()
        if not self.profiles:
            raise RunPodManagerError(
                "Pod capacity reconciliation requires configured GPU profiles"
            )
        missing_profiles = {
            lease.profile_id
            for lease in self.repository.list_for_reconciliation()
            if lease.is_catalog_attempt and lease.profile_id not in self.profiles
        }
        if missing_profiles:
            raise RunPodManagerError(
                "Pod capacity reconciliation is missing durable catalog profiles"
            )

    async def acquire_catalog(
        self, request: CatalogPodCapacityRequest
    ) -> TrainingPodLease:
        """Acquire one isolated Pod for an opaque private catalog attempt."""

        self._require_workload_transport()
        existing = self.repository.get(request.capacity_id)
        if existing is not None:
            existing_spec = self._capacity_spec(existing)
            if (
                existing_spec.request.fingerprint != request.fingerprint
                or existing_spec.attempt_environment_sha256
                != attempt_environment_sha256(request.attempt_environment)
            ):
                raise TrainingPodConflictError(
                    f"Pod capacity '{request.capacity_id}' already represents "
                    "a different request"
                )
            if existing.state is TrainingPodState.READY:
                return self.heartbeat(existing.cleanup_token)
            raise TrainingPodConflictError(
                f"Pod capacity '{request.capacity_id}' is already in state "
                f"{existing.state.value}; reconcile it"
            )
        now = self._now()
        if request.created_at > now or request.quote.observed_at > request.created_at:
            raise RunPodManagerError("Catalog Pod request or quote is from the future")
        if request.quote.expires_at <= now:
            raise RunPodManagerError("Catalog Pod capacity quote has expired")
        store = self._require_capability_store()
        capability = await store.load_or_create(
            request.attempt_id, request.bearer_expires_at
        )
        _validate_capability(request, capability)
        capacity_spec = PodCapacitySpec(
            request=request,
            capability_secret_id=capability.secret_id,
            capability_token_sha256=capability.token_sha256,
            capability_expires_at=capability.expires_at,
            attempt_environment_sha256=attempt_environment_sha256(
                request.attempt_environment
            ),
        )
        internal = TrainingPodRequest(
            cleanup_token=request.capacity_id,
            root_cleanup_token=request.cleanup_family_id,
            companion_id=request.owner_id,
            profile_id=request.profile_id,
            source=TrainingPodSource.CREATED,
            resource_name=request.resource_name,
            provider_pod_id=None,
            created_at=request.created_at,
            readiness_deadline=request.readiness_deadline,
            hard_deadline=request.hard_deadline,
            capacity_spec=capacity_spec,
        )
        environment = {
            **dict(request.attempt_environment),
            "CATALOG_WORKER_MODE": "pod",
            "CATALOG_POD_ATTEMPT_ID": request.attempt_id,
            "CATALOG_POD_BEARER_TOKEN": capability.bearer_token.get_secret_value(),
            "CATALOG_POD_BEARER_EXPIRES_AT": iso_datetime(capability.expires_at),
            "CONTAINER_DIGEST": request.image_reference.rsplit("@", 1)[1],
        }
        return await self.acquire(internal, environment=environment)

    async def acquire(
        self,
        request: TrainingPodRequest,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> TrainingPodLease:
        """Acquire capacity, persisting authority before any provider mutation."""

        lease, inserted = self.repository.reserve(request)
        if not inserted:
            if lease.state is TrainingPodState.READY:
                return self.heartbeat(lease.cleanup_token)
            raise TrainingPodConflictError(
                f"Training cleanup token '{request.cleanup_token}' is already "
                f"in state {lease.state.value}; reconcile it"
            )
        profile = self._profile(lease.profile_id)
        if request.source is TrainingPodSource.CREATED:
            return await self._acquire_created(lease, profile, environment=environment)
        return await self._acquire_existing(lease, profile)

    async def _acquire_existing(
        self, lease: TrainingPodLease, profile: GPUProfile
    ) -> TrainingPodLease:
        pod_id = self._pod_id(lease)
        try:
            observation = await self.provider.observe(pod_id, profile=profile)
        except asyncio.CancelledError:
            self._release_unowned_claim(lease, error=None)
            raise
        except RunPodManagerError as exc:
            released = self._release_unowned_claim(lease, error=exc)
            raise self._lifecycle_error(
                "Pod discovery", released, billing_risk=False
            ) from exc

        if observation.is_running:
            lease = self.repository.compare_and_set(
                lease,
                changes={
                    "ownership": TrainingPodOwnership.PREEXISTING_RUNNING,
                    "state": TrainingPodState.STARTING,
                    "backend_base_url": observation.backend_base_url,
                    "last_heartbeat_at": self._now(),
                    "last_provider_error": None,
                },
            )
        elif observation.is_stopped:
            lease = self.repository.compare_and_set(
                lease,
                changes={
                    "ownership": TrainingPodOwnership.PROVISIONAL,
                    "state": TrainingPodState.STARTING,
                    "last_heartbeat_at": self._now(),
                },
            )
            try:
                await self.provider.start(pod_id, gpu_count=profile.gpu_count)
            except asyncio.CancelledError:
                await self._cleanup_cancelled_acquisition(
                    lease, "Pod start cancellation"
                )
                raise
            except RunPodManagerError as exc:
                lease = self._record_retryable_failure(lease, exc)
                try:
                    await self._release_one(
                        lease.cleanup_token, reason="Pod start failure"
                    )
                except TrainingPodCleanupError as cleanup_exc:
                    raise cleanup_exc from exc
                current = self._required(lease.cleanup_token)
                raise self._lifecycle_error(
                    "Pod start", current, billing_risk=False
                ) from exc
            lease = self.repository.compare_and_set(
                self._required(lease.cleanup_token),
                changes={
                    "ownership": TrainingPodOwnership.OWNED,
                    "last_heartbeat_at": self._now(),
                    "last_provider_error": None,
                },
            )
        else:
            released = self._release_unowned_claim(
                lease,
                error=RunPodManagerError(
                    f"Pod is not safely acquirable from state {observation.status}"
                ),
            )
            raise self._lifecycle_error("Pod acquisition", released, billing_risk=False)

        return await self._wait_ready(lease, profile)

    async def _acquire_created(
        self,
        lease: TrainingPodLease,
        profile: GPUProfile,
        *,
        environment: Mapping[str, str] | None,
    ) -> TrainingPodLease:
        lease = self.repository.compare_and_set(
            lease,
            changes={
                "state": TrainingPodState.STARTING,
                "ownership": TrainingPodOwnership.PROVISIONAL,
                # The to_thread create can continue after task cancellation.
                "creation_uncertain": True,
                "last_heartbeat_at": self._now(),
            },
        )
        try:
            create_args: dict[str, Any] = {
                "profile": profile,
                "resource_name": lease.resource_name,
                "companion_id": lease.companion_id,
            }
            if lease.capacity_spec is not None:
                create_args.update(
                    environment=dict(environment or {}),
                    capacity_spec=lease.capacity_spec,
                )
            created = await self.provider.create(**create_args)
            if lease.is_catalog_attempt and created.realized_placement is None:
                raise PodCapacityCreatedMismatchError(created.provider_pod_id)
        except asyncio.CancelledError:
            lease = self._required(lease.cleanup_token)
            await self._cleanup_cancelled_acquisition(lease, "Pod create cancellation")
            raise
        except PodCapacityCreatedMismatchError as exc:
            # The create response supplied a concrete ID, so this is owned
            # capacity rather than an ambiguous lost-response case. Persist the
            # ID before termination so a crash cannot strand a billable Pod.
            accepted_at = self._now()
            lease = self._transition_with_evidence(
                self._required(lease.cleanup_token),
                changes={
                    "provider_pod_id": exc.provider_pod_id,
                    "ownership": TrainingPodOwnership.OWNED,
                    "creation_uncertain": False,
                    "last_provider_error": sanitize_training_error(exc),
                    "last_heartbeat_at": accepted_at,
                },
                lifecycle={"provider_create_accepted_at": accepted_at},
            )
            try:
                released = await self._release_one(
                    lease.cleanup_token, reason="created Pod validation failure"
                )
            except TrainingPodCleanupError as cleanup_exc:
                raise cleanup_exc from exc
            raise self._lifecycle_error(
                "Pod create validation", released, billing_risk=False
            ) from exc
        except RunPodAmbiguousResultError as exc:
            lease = self._record_retryable_failure(lease, exc, creation_uncertain=True)
            if lease.is_catalog_attempt:
                raise self._lifecycle_error(
                    "Pod create", lease, billing_risk=True
                ) from exc
            try:
                await self._release_one(
                    lease.cleanup_token, reason="ambiguous Pod create"
                )
            except TrainingPodCleanupError as cleanup_exc:
                raise cleanup_exc from exc
            raise self._lifecycle_error(
                "Pod create", self._required(lease.cleanup_token), billing_risk=False
            ) from exc
        except RunPodManagerError as exc:
            # A typed non-ambiguous create failure means v2 did not accept a Pod.
            changes: dict[str, Any] = {
                "state": TrainingPodState.RELEASED,
                "cleanup_state": TrainingPodCleanupState.COMPLETE,
                "creation_uncertain": False,
                "last_provider_error": sanitize_training_error(exc),
                "last_heartbeat_at": self._now(),
            }
            evidence_lifecycle: dict[str, datetime] = {}
            evidence_billing: PodBillingReceipt | object = _EVIDENCE_UNSET
            if lease.is_catalog_attempt:
                receipt = self._zero_cost_receipt(lease, "create-rejected")
                reconciled_at = self._now()
                changes.update(
                    state=TrainingPodState.RELEASING,
                    billing_state=PodCapacityBillingState.AUTHORITATIVE,
                    billing_receipt_json=receipt,
                    terminated_at=reconciled_at,
                )
                evidence_lifecycle["billing_reconciled_at"] = reconciled_at
                evidence_billing = receipt
            released = self._transition_with_evidence(
                lease,
                changes=changes,
                lifecycle=evidence_lifecycle,
                billing=evidence_billing,
            )
            if released.is_catalog_attempt:
                released = await self._revoke_and_release(released)
            raise self._lifecycle_error(
                "Pod create", released, billing_risk=False
            ) from exc

        accepted_at = self._now()
        lease = self._transition_with_evidence(
            self._required(lease.cleanup_token),
            changes={
                "provider_pod_id": created.provider_pod_id,
                "ownership": TrainingPodOwnership.OWNED,
                "creation_uncertain": False,
                "last_heartbeat_at": accepted_at,
                "last_provider_error": None,
            },
            lifecycle={"provider_create_accepted_at": accepted_at},
            realized_placement=created.realized_placement,
        )
        return await self._wait_ready(lease, profile)

    async def _wait_ready(
        self, lease: TrainingPodLease, profile: GPUProfile
    ) -> TrainingPodLease:
        while self._now() < lease.readiness_deadline:
            try:
                observation = await self.provider.observe(
                    self._pod_id(lease), profile=profile
                )
            except asyncio.CancelledError:
                await self._cleanup_cancelled_acquisition(
                    lease, "readiness cancellation"
                )
                raise
            except RunPodManagerError as exc:
                lease = self._record_retryable_failure(lease, exc)
                return await self._raise_after_cleanup(
                    lease, operation="readiness observation", cause=exc
                )
            observed_at = self._now()
            lease = self._transition_with_evidence(
                self._required(lease.cleanup_token),
                changes={
                    "backend_base_url": observation.backend_base_url,
                    "last_heartbeat_at": observed_at,
                    "last_provider_error": None,
                },
                lifecycle=(
                    {"first_running_observed_at": observed_at}
                    if observation.is_running
                    else {}
                ),
            )
            if observation.is_failed or observation.is_stopped:
                cause = RunPodManagerError(
                    f"Training Pod entered unusable state {observation.status}"
                )
                return await self._raise_after_cleanup(
                    lease, operation="readiness", cause=cause
                )
            if observation.is_running and observation.backend_base_url:
                if lease.is_catalog_attempt:
                    try:
                        await self._require_workload_transport().health(
                            observation.backend_base_url
                        )
                    except CatalogPodTransportError as exc:
                        lease = self._record_retryable_failure(lease, exc)
                        try:
                            await self._sleep(self.poll_interval_seconds)
                        except asyncio.CancelledError:
                            await self._cleanup_cancelled_acquisition(
                                lease, "catalog health cancellation"
                            )
                            raise
                        lease = self._required(lease.cleanup_token)
                        continue
                ready_at = self._now()
                return self._transition_with_evidence(
                    lease,
                    changes={
                        "state": TrainingPodState.READY,
                        "last_heartbeat_at": ready_at,
                    },
                    lifecycle={"worker_ready_at": ready_at},
                )
            try:
                await self._sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                await self._cleanup_cancelled_acquisition(
                    lease, "readiness cancellation"
                )
                raise
            lease = self._required(lease.cleanup_token)
        cause = RunPodManagerError(
            "Training Pod did not become route-ready before timeout"
        )
        return await self._raise_after_cleanup(
            lease, operation="readiness timeout", cause=cause
        )

    async def _raise_after_cleanup(
        self,
        lease: TrainingPodLease,
        *,
        operation: str,
        cause: BaseException,
    ) -> TrainingPodLease:
        try:
            released = await self._release_one(lease.cleanup_token, reason=operation)
        except TrainingPodCleanupError as cleanup_exc:
            raise cleanup_exc from cause
        raise self._lifecycle_error(operation, released, billing_risk=False) from cause

    async def _cleanup_cancelled_acquisition(
        self, lease: TrainingPodLease, operation: str
    ) -> None:
        task = asyncio.create_task(
            self._release_one(lease.cleanup_token, reason=operation)
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A second cancellation must not detach the already-started cleanup task.
            await task
        except TrainingPodCleanupError:
            raise

    def heartbeat(self, cleanup_token: str) -> TrainingPodLease:
        lease = self._required(cleanup_token)
        if lease.is_terminal:
            raise TrainingPodLifecycleError(
                "heartbeat",
                cleanup_token=cleanup_token,
                pod_id=lease.provider_pod_id,
                cleanup_state=lease.cleanup_state,
                billing_risk=False,
            )
        return self.repository.compare_and_set(
            lease, changes={"last_heartbeat_at": self._now()}
        )

    def record_job(self, cleanup_token: str, provider_job_id: str) -> TrainingPodLease:
        if not provider_job_id:
            raise ValueError("Training provider_job_id must be non-empty")
        lease = self._required(cleanup_token)
        if lease.state not in {TrainingPodState.READY, TrainingPodState.JOB_SUBMITTED}:
            raise RunPodManagerError(
                f"Training cleanup token '{cleanup_token}' is not ready for submission"
            )
        if lease.provider_job_id not in {None, provider_job_id}:
            raise TrainingPodConflictError(
                f"Training cleanup token '{cleanup_token}' already owns another job"
            )
        return self.repository.compare_and_set(
            lease,
            changes={
                "state": TrainingPodState.JOB_SUBMITTED,
                "provider_job_id": provider_job_id,
                "last_heartbeat_at": self._now(),
                "last_provider_error": None,
            },
        )

    def record_operation_error(
        self, cleanup_token: str, error: BaseException
    ) -> TrainingPodLease:
        lease = self._required(cleanup_token)
        if lease.is_terminal:
            return lease
        return self._transition_nonterminal(
            lease,
            changes={
                "last_heartbeat_at": self._now(),
                "last_provider_error": sanitize_training_error(error),
            },
        )

    def record_status(self, cleanup_token: str, status: str) -> TrainingPodLease:
        lease = self._required(cleanup_token)
        if lease.is_terminal:
            return lease
        normalized = status.strip().lower()
        changes: dict[str, Any] = {
            "last_heartbeat_at": self._now(),
            "last_provider_error": None,
        }
        if normalized in {"completed", "succeeded"}:
            changes["state"] = TrainingPodState.JOB_COMPLETED
        elif normalized in {"cancelled", "canceled", "failed"}:
            changes["state"] = TrainingPodState.CANCEL_REQUESTED
        return self._transition_nonterminal(lease, changes=changes)

    def record_result_retrieved(self, cleanup_token: str) -> TrainingPodLease:
        lease = self._required(cleanup_token)
        if lease.state not in {
            TrainingPodState.JOB_SUBMITTED,
            TrainingPodState.JOB_COMPLETED,
        }:
            raise RunPodManagerError(
                f"Training cleanup token '{cleanup_token}' has no completed result"
            )
        return self.repository.compare_and_set(
            lease,
            changes={
                "state": TrainingPodState.RESULT_RETRIEVED,
                "last_heartbeat_at": self._now(),
                "last_provider_error": None,
            },
        )

    def request_cancellation(self, cleanup_token: str) -> TrainingPodLease:
        lease = self._required(cleanup_token)
        if lease.is_terminal:
            return lease
        return self.repository.compare_and_set(
            lease,
            changes={
                "state": TrainingPodState.CANCEL_REQUESTED,
                "last_heartbeat_at": self._now(),
            },
        )

    async def submit_catalog_workload(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
        payload: Mapping[str, Any],
    ) -> CatalogPodWorkloadObservation:
        """Submit one opaque request; identical replay is worker-idempotent."""

        lease = self._catalog_lease(capacity_id, owner_id, workload_id)
        if lease.state not in {TrainingPodState.READY, TrainingPodState.JOB_SUBMITTED}:
            raise RunPodManagerError(
                f"Pod capacity '{capacity_id}' is not ready for workload submission"
            )
        spec = self._capacity_spec(lease)
        capability = await self._load_capability(spec)
        observation = await self._require_workload_transport().submit(
            base_url=self._route(lease),
            attempt_id=spec.request.attempt_id,
            request_sha256=spec.request.request_sha256,
            bearer_token=capability.bearer_token,
            payload=payload,
        )
        self._record_catalog_observation(
            lease.cleanup_token, observation, submitted=True
        )
        return observation

    async def observe_catalog_workload(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
    ) -> CatalogPodWorkloadObservation:
        """Observe content-free status and tear down failed/cancelled work."""

        lease = self._catalog_lease(capacity_id, owner_id, workload_id)
        spec = self._capacity_spec(lease)
        capability = await self._load_capability(spec)
        observation = await self._require_workload_transport().status(
            base_url=self._route(lease),
            attempt_id=spec.request.attempt_id,
            request_sha256=spec.request.request_sha256,
            bearer_token=capability.bearer_token,
        )
        current = self._required(capacity_id)
        if current.workload_state is CatalogPodWorkloadState.CANCEL_REQUESTED:
            await self.release(capacity_id, reason="cancelled catalog attempt")
            return CatalogPodWorkloadObservation(
                attempt_id=observation.attempt_id,
                request_sha256=observation.request_sha256,
                state=CatalogPodWorkloadState.CANCEL_REQUESTED,
                error_type=None,
                result_available=False,
            )
        updated = self._record_catalog_observation(capacity_id, observation)
        if updated.workload_state in {
            CatalogPodWorkloadState.FAILED,
            CatalogPodWorkloadState.CANCELLED,
        }:
            await self.release(capacity_id, reason="terminal catalog workload")
        return observation

    async def retrieve_catalog_result(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
    ) -> Mapping[str, Any]:
        """Return an opaque result replay without acknowledging or tearing it down."""

        lease = self._catalog_lease(capacity_id, owner_id, workload_id)
        if lease.workload_state is not CatalogPodWorkloadState.SUCCEEDED:
            raise RunPodManagerError("Catalog Pod result is not ready")
        spec = self._capacity_spec(lease)
        capability = await self._load_capability(spec)
        lease = self.heartbeat(lease.cleanup_token)
        payload = await self._require_workload_transport().result(
            base_url=self._route(lease),
            attempt_id=spec.request.attempt_id,
            request_sha256=spec.request.request_sha256,
            bearer_token=capability.bearer_token,
        )
        current = self._required(capacity_id)
        if current.workload_state is CatalogPodWorkloadState.CANCEL_REQUESTED:
            await self.release(capacity_id, reason="late result after cancellation")
            raise RunPodManagerError("Cancelled catalog attempt cannot accept a result")
        return payload

    def record_catalog_worker_evidence(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
        envelope: Mapping[str, Any],
    ) -> TrainingPodLease:
        """Persist one independently projected content-free worker envelope."""

        lease = self._catalog_lease(capacity_id, owner_id, workload_id)
        spec = self._capacity_spec(lease)
        if lease.evidence is None:
            raise RunPodManagerError(
                "Legacy Pod capacity has explicit missing realized evidence"
            )
        worker = CatalogWorkerEvidence.from_envelope(envelope)
        expected_image_digest = spec.request.image_reference.rsplit("@", 1)[1]
        if (
            worker.attempt_id != spec.request.attempt_id
            or worker.request_sha256 != spec.request.request_sha256
            or worker.image_digest != expected_image_digest
        ):
            raise TrainingPodConflictError(
                "Catalog worker evidence does not match the exact attempt binding"
            )
        if lease.workload_state is not CatalogPodWorkloadState.SUCCEEDED:
            raise RunPodManagerError(
                "Catalog worker evidence cannot be accepted before workload success"
            )
        if (
            worker.container_process_started_at is not None
            and worker.container_process_started_at > self._now()
        ):
            raise RunPodManagerError(
                "Catalog worker process-start evidence is from the future"
            )
        if lease.evidence.worker is not None:
            if lease.evidence.worker != worker:
                raise TrainingPodConflictError(
                    "Catalog worker evidence was already recorded differently"
                )
            return lease
        if lease.is_terminal:
            raise RunPodManagerError(
                "Terminal catalog capacity cannot accept new worker evidence"
            )
        return self._transition_with_evidence(
            lease,
            changes={"last_heartbeat_at": self._now()},
            worker=worker,
        )

    async def acknowledge_catalog_result(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
    ) -> TrainingPodLease:
        """Acknowledge durable host persistence, then terminate disposable capacity."""

        lease = self._catalog_lease(capacity_id, owner_id, workload_id)
        if lease.workload_state is not CatalogPodWorkloadState.SUCCEEDED:
            raise RunPodManagerError(
                "Catalog Pod result cannot be acknowledged before success"
            )
        if lease.is_terminal:
            return lease
        if lease.state is TrainingPodState.JOB_COMPLETED:
            lease = self.repository.compare_and_set(
                lease,
                changes={
                    "state": TrainingPodState.RESULT_RETRIEVED,
                    "last_heartbeat_at": self._now(),
                    "last_provider_error": None,
                },
            )
        elif lease.state not in {
            TrainingPodState.RESULT_RETRIEVED,
            TrainingPodState.RELEASING,
        }:
            raise RunPodManagerError(
                "Catalog Pod result acknowledgement is inconsistent with lease state"
            )
        return await self.release(capacity_id, reason="catalog result acknowledged")

    async def cancel_catalog_workload(
        self,
        *,
        capacity_id: str,
        owner_id: str,
        workload_id: str,
    ) -> TrainingPodLease:
        """Persist cancellation intent, authenticate cancel, then terminate capacity."""

        lease = self._catalog_lease(capacity_id, owner_id, workload_id)
        if lease.is_terminal:
            return lease
        lease = self.repository.compare_and_set(
            lease,
            changes={
                "state": TrainingPodState.CANCEL_REQUESTED,
                "workload_state": CatalogPodWorkloadState.CANCEL_REQUESTED,
                "last_heartbeat_at": self._now(),
            },
        )
        transport_error: RunPodManagerError | None = None
        if lease.backend_base_url is not None:
            try:
                spec = self._capacity_spec(lease)
                capability = await self._load_capability(spec)
                await self._require_workload_transport().cancel(
                    base_url=lease.backend_base_url,
                    attempt_id=spec.request.attempt_id,
                    request_sha256=spec.request.request_sha256,
                    bearer_token=capability.bearer_token,
                )
            except RunPodManagerError as exc:
                transport_error = exc
                self.record_operation_error(capacity_id, exc)
        released = await self.release(capacity_id, reason="catalog cancellation")
        if transport_error is not None:
            logger.warning(
                "Catalog attempt %s was terminated after workload cancel failed (%s)",
                capacity_id,
                type(transport_error).__name__,
            )
        return released

    def get_active_family_attempt(
        self, root_cleanup_token: str
    ) -> TrainingPodLease | None:
        """Resolve the single active attempt owned by a caller's root token."""

        root = self.repository.get(root_cleanup_token)
        if root is None:
            return None
        if root.family_release_requested:
            raise TrainingPodConflictError(
                f"Training cleanup family '{root_cleanup_token}' is releasing"
            )
        active = tuple(
            lease
            for lease in self._cleanup_family(root_cleanup_token)
            if not lease.is_terminal
        )
        if len(active) > 1:
            raise TrainingPodConflictError(
                f"Training cleanup family '{root_cleanup_token}' has multiple "
                "active attempts; reconcile it"
            )
        return active[0] if active else None

    async def release(self, cleanup_token: str, *, reason: str) -> TrainingPodLease:
        """Release one exact attempt or every attempt addressed by a family root."""

        exact = self._required(cleanup_token)
        if exact.root_cleanup_token != cleanup_token:
            return await self._release_one(cleanup_token, reason=reason)

        root = self._mark_family_release_requested(exact)
        failures: list[TrainingPodCleanupError] = []
        for member in self._cleanup_family(cleanup_token):
            if member.is_terminal:
                continue
            try:
                await self._release_one(member.cleanup_token, reason=reason)
            except TrainingPodCleanupError as exc:
                failures.append(exc)
            except RunPodManagerError as exc:
                current = self._required(member.cleanup_token)
                failures.append(self._cleanup_error(reason, current))
                logger.warning(
                    "Training cleanup family %s could not release attempt %s (%s)",
                    cleanup_token,
                    member.cleanup_token,
                    type(exc).__name__,
                )

        if failures:
            failure = failures[-1]
            raise TrainingPodCleanupError(
                reason,
                cleanup_token=cleanup_token,
                pod_id=failure.pod_id,
                cleanup_state=failure.cleanup_state,
                billing_risk=True,
            ) from failure

        root = self._mark_family_release_complete(root)
        return root

    async def _release_one(
        self, cleanup_token: str, *, reason: str
    ) -> TrainingPodLease:
        """Stop one exact attempt idempotently, retaining its ID until confirmed."""

        lease = self._required(cleanup_token)
        if lease.is_terminal:
            return lease
        if (
            lease.is_catalog_attempt
            and lease.cleanup_state is TrainingPodCleanupState.COMPLETE
            and lease.terminated_at is not None
            and lease.billing_state
            in {
                PodCapacityBillingState.PENDING,
                PodCapacityBillingState.UNRESOLVED,
            }
        ):
            return await self._reconcile_billing(lease)
        if lease.ownership is TrainingPodOwnership.PREEXISTING_RUNNING:
            return self._transition_nonterminal(
                lease,
                changes={
                    "state": TrainingPodState.RELEASED,
                    "cleanup_state": TrainingPodCleanupState.NOT_OWNED,
                    "backend_base_url": None,
                    "last_provider_error": None,
                    "last_heartbeat_at": self._now(),
                },
            )
        if lease.provider_pod_id is None and lease.creation_uncertain:
            lease = await self._resolve_uncertain_creation(lease, reason=reason)
            if lease.provider_pod_id is None and lease.creation_uncertain:
                raise self._cleanup_error(reason, lease)
        if lease.provider_pod_id is None:
            terminal_at = self._now()
            changes: dict[str, Any] = {
                "state": TrainingPodState.RELEASED,
                "cleanup_state": TrainingPodCleanupState.COMPLETE,
                "backend_base_url": None,
                "last_provider_error": None,
                "last_heartbeat_at": terminal_at,
            }
            evidence_lifecycle: dict[str, datetime] = {}
            evidence_billing: PodBillingReceipt | object = _EVIDENCE_UNSET
            if lease.is_catalog_attempt:
                receipt = self._zero_cost_receipt(lease, "no-provider-capacity")
                changes.update(
                    state=TrainingPodState.RELEASING,
                    billing_state=PodCapacityBillingState.AUTHORITATIVE,
                    billing_receipt_json=receipt,
                    terminated_at=terminal_at,
                )
                evidence_lifecycle["billing_reconciled_at"] = terminal_at
                evidence_billing = receipt
            released = self._transition_with_evidence(
                lease,
                changes=changes,
                lifecycle=evidence_lifecycle,
                billing=evidence_billing,
            )
            if released.is_catalog_attempt:
                released = await self._revoke_and_release(released)
            return released
        if lease.state is not TrainingPodState.RELEASING:
            lease = self._transition_nonterminal(
                lease,
                changes={
                    "state": TrainingPodState.RELEASING,
                    "cleanup_state": TrainingPodCleanupState.PENDING,
                    "backend_base_url": None,
                    "last_heartbeat_at": self._now(),
                },
            )
        while not lease.is_terminal:
            try:
                lease = self.repository.compare_and_set(
                    lease, changes={"stop_attempts": lease.stop_attempts + 1}
                )
                break
            except TrainingPodConflictError:
                lease = self._required(cleanup_token)
        if lease.is_terminal:
            return lease
        try:
            if lease.is_catalog_attempt:
                confirmed = await self.provider.terminate(self._pod_id(lease))
            else:
                confirmed = await self.provider.stop(self._pod_id(lease))
        except RunPodManagerError as exc:
            lease = self._record_retryable_failure(lease, exc)
            if lease.is_terminal:
                return lease
            raise self._cleanup_error(reason, lease) from exc
        if not confirmed:
            lease = self._transition_nonterminal(
                lease,
                changes={
                    "cleanup_state": TrainingPodCleanupState.PENDING,
                    "last_provider_error": "StopPending",
                    "last_heartbeat_at": self._now(),
                },
            )
            if lease.is_terminal:
                return lease
            raise self._cleanup_error(reason, lease)
        completed_at = self._now()
        changes = {
            "state": (
                TrainingPodState.RELEASING
                if lease.is_catalog_attempt
                else TrainingPodState.RELEASED
            ),
            "cleanup_state": TrainingPodCleanupState.COMPLETE,
            "creation_uncertain": False,
            "backend_base_url": None,
            "last_provider_error": None,
            "last_heartbeat_at": completed_at,
        }
        if lease.is_catalog_attempt:
            changes.update(
                billing_state=PodCapacityBillingState.PENDING,
                terminated_at=completed_at,
            )
        released = self._transition_with_evidence(
            lease,
            changes=changes,
            lifecycle=(
                {"stop_confirmed_at": completed_at} if lease.is_catalog_attempt else {}
            ),
        )
        if released.is_catalog_attempt:
            return await self._reconcile_billing(released)
        return released

    async def reconcile(self) -> tuple[TrainingPodLease, ...]:
        """Run one pass suitable for a cheap external timer or webhook service."""

        results: list[TrainingPodLease] = []
        for root in self.repository.list_incomplete_family_releases():
            try:
                result = await self.release(
                    root.cleanup_token, reason="restart family release"
                )
            except (TrainingPodLifecycleError, RunPodManagerError) as exc:
                result = self._record_reconcile_error(root.cleanup_token, exc)
            if result is not None:
                results.append(result)
        for initial in self.repository.list_for_reconciliation():
            try:
                result = await self._reconcile_one(initial.cleanup_token)
            except (TrainingPodLifecycleError, RunPodManagerError) as exc:
                result = self._record_reconcile_error(initial.cleanup_token, exc)
            if result is not None:
                results.append(result)
        return tuple(results)

    async def _reconcile_one(self, cleanup_token: str) -> TrainingPodLease | None:
        lease = self.repository.get(cleanup_token)
        if lease is None:
            return lease
        if lease.is_terminal:
            if lease.billing_state in {
                PodCapacityBillingState.PENDING,
                PodCapacityBillingState.UNRESOLVED,
            }:
                return await self._reconcile_billing(lease)
            return lease
        now = self._now()
        idle_timeout = (
            self._capacity_spec(lease).request.idle_timeout_seconds
            if lease.is_catalog_attempt
            else self.orphan_timeout_seconds
        )
        stale = now >= lease.last_heartbeat_at + timedelta(seconds=idle_timeout)
        if lease.state is TrainingPodState.REQUESTED and (
            stale or now >= lease.readiness_deadline
        ):
            # REQUESTED is persisted before provider discovery/mutation. A crash
            # in this exact window must release the claim without stopping a Pod
            # that may have been running before this invocation existed.
            if lease.source is not TrainingPodSource.CREATED:
                return self._release_unowned_claim(lease, error=None)
            changes: dict[str, Any] = {
                "state": TrainingPodState.RELEASED,
                "cleanup_state": TrainingPodCleanupState.COMPLETE,
                "last_provider_error": None,
            }
            evidence_lifecycle: dict[str, datetime] = {}
            evidence_billing: PodBillingReceipt | object = _EVIDENCE_UNSET
            if lease.is_catalog_attempt:
                receipt = self._zero_cost_receipt(lease, "before-create")
                changes.update(
                    state=TrainingPodState.RELEASING,
                    billing_state=PodCapacityBillingState.AUTHORITATIVE,
                    billing_receipt_json=receipt,
                    terminated_at=now,
                )
                evidence_lifecycle["billing_reconciled_at"] = now
                evidence_billing = receipt
            released = self._transition_with_evidence(
                lease,
                changes=changes,
                lifecycle=evidence_lifecycle,
                billing=evidence_billing,
            )
            if released.is_catalog_attempt:
                released = await self._revoke_and_release(released)
            return released
        if lease.state is TrainingPodState.REQUESTED:
            return lease
        if lease.provider_pod_id is None and lease.creation_uncertain:
            # ``STARTING`` + ``creation_uncertain`` is also the normal window
            # while a live owner waits for the v2 create call to return.  A
            # concurrent reconciler must not discover and stop that owner's
            # freshly-created Pod.  Recovery begins only after the owner has
            # stopped heartbeating or the lease has reached its hard deadline.
            if not stale and now < lease.hard_deadline:
                return lease
            lease = await self._resolve_uncertain_creation(lease, reason="reconcile")
            if lease.provider_pod_id is not None:
                return await self._release_one(cleanup_token, reason="recovered create")
            return lease
        if lease.state in {
            TrainingPodState.RELEASING,
            TrainingPodState.CANCEL_REQUESTED,
            TrainingPodState.RECONCILE_REQUIRED,
            TrainingPodState.RESULT_RETRIEVED,
        }:
            return await self._release_one(cleanup_token, reason="retry teardown")
        if now >= lease.hard_deadline:
            return await self._release_one(cleanup_token, reason="hard deadline")
        if lease.is_catalog_attempt and lease.state is TrainingPodState.JOB_SUBMITTED:
            spec = self._capacity_spec(lease)
            await self.observe_catalog_workload(
                capacity_id=lease.cleanup_token,
                owner_id=spec.request.owner_id,
                workload_id=spec.request.workload_id,
            )
            return self._required(cleanup_token)
        if (
            lease.state is TrainingPodState.JOB_SUBMITTED
            and self._workload_status_observer is not None
        ):
            try:
                status = await self._workload_status_observer(lease)
            except RunPodManagerError as exc:
                return self.record_operation_error(cleanup_token, exc)
            if status is not None:
                lease = self.record_status(cleanup_token, status)
                if lease.state is TrainingPodState.CANCEL_REQUESTED:
                    return await self._release_one(
                        cleanup_token, reason="terminal training job"
                    )
                return lease
        if (
            lease.state
            in {
                TrainingPodState.STARTING,
                TrainingPodState.READY,
            }
            and stale
        ):
            return await self._release_one(cleanup_token, reason="orphaned acquisition")
        if lease.provider_pod_id and lease.state in {
            TrainingPodState.STARTING,
            TrainingPodState.READY,
        }:
            if lease.profile_id not in self.profiles:
                # A removed legacy fallback profile prevents route reconstruction,
                # but it must not prevent profile-free stale/deadline teardown.
                return lease
            observation = await self.provider.observe(
                lease.provider_pod_id, profile=self._profile(lease.profile_id)
            )
            if observation.is_stopped:
                changes: dict[str, Any] = {
                    "state": TrainingPodState.RELEASED,
                    "cleanup_state": TrainingPodCleanupState.COMPLETE,
                    "backend_base_url": None,
                    "last_provider_error": None,
                }
                if lease.is_catalog_attempt:
                    changes.update(
                        state=TrainingPodState.RELEASING,
                        billing_state=PodCapacityBillingState.PENDING,
                        terminated_at=now,
                    )
                stopped = self._transition_with_evidence(
                    lease,
                    changes=changes,
                    lifecycle=(
                        {"stop_confirmed_at": now} if lease.is_catalog_attempt else {}
                    ),
                )
                if stopped.is_catalog_attempt:
                    return await self._reconcile_billing(stopped)
                return stopped
            return self.repository.compare_and_set(
                lease,
                changes={
                    "backend_base_url": observation.backend_base_url,
                    "last_provider_error": None,
                },
            )
        return lease

    async def _resolve_uncertain_creation(
        self, lease: TrainingPodLease, *, reason: str
    ) -> TrainingPodLease:
        try:
            if lease.capacity_spec is not None:
                recovered = await self.provider.find_exact(
                    lease.resource_name, lease.capacity_spec
                )
                pod_id = recovered.provider_pod_id if recovered is not None else None
            else:
                recovered = None
                pod_id = await self.provider.find_by_name(lease.resource_name)
        except RunPodManagerError as exc:
            return self._record_retryable_failure(lease, exc, creation_uncertain=True)
        if pod_id is not None:
            adopted_at = self._now()
            return self._transition_with_evidence(
                lease,
                changes={
                    "provider_pod_id": pod_id,
                    "ownership": TrainingPodOwnership.OWNED,
                    "creation_uncertain": False,
                    "state": TrainingPodState.RECONCILE_REQUIRED,
                    "cleanup_state": TrainingPodCleanupState.PENDING,
                    "last_provider_error": None,
                },
                lifecycle=(
                    {"provider_adopted_at": adopted_at}
                    if lease.is_catalog_attempt
                    else {}
                ),
                realized_placement=(
                    recovered.realized_placement
                    if recovered is not None
                    else _EVIDENCE_UNSET
                ),
            )
        if self._now() < lease.readiness_deadline or lease.is_catalog_attempt:
            return self._transition_nonterminal(
                lease,
                changes={
                    "state": TrainingPodState.RECONCILE_REQUIRED,
                    "cleanup_state": TrainingPodCleanupState.RETRYABLE_FAILURE,
                    "billing_state": (
                        PodCapacityBillingState.UNRESOLVED
                        if lease.is_catalog_attempt
                        else lease.billing_state
                    ),
                    "last_provider_error": f"{reason}:AwaitingCreateReconciliation",
                },
            )
        return self._transition_nonterminal(
            lease,
            changes={
                "state": TrainingPodState.RELEASED,
                "cleanup_state": TrainingPodCleanupState.COMPLETE,
                "creation_uncertain": False,
                "last_provider_error": None,
            },
        )

    def _release_unowned_claim(
        self, lease: TrainingPodLease, *, error: BaseException | None
    ) -> TrainingPodLease:
        return self._transition_nonterminal(
            lease,
            changes={
                "ownership": TrainingPodOwnership.PREEXISTING_RUNNING,
                "state": TrainingPodState.RELEASED,
                "cleanup_state": TrainingPodCleanupState.NOT_OWNED,
                "backend_base_url": None,
                "last_provider_error": (
                    sanitize_training_error(error) if error is not None else None
                ),
                "last_heartbeat_at": self._now(),
            },
        )

    def _record_retryable_failure(
        self,
        lease: TrainingPodLease,
        error: BaseException,
        *,
        creation_uncertain: bool | None = None,
    ) -> TrainingPodLease:
        current = self._required(lease.cleanup_token)
        if current.is_terminal:
            return current
        changes: dict[str, Any] = {
            "state": TrainingPodState.RECONCILE_REQUIRED,
            "cleanup_state": TrainingPodCleanupState.RETRYABLE_FAILURE,
            "last_provider_error": sanitize_training_error(error),
            "last_heartbeat_at": self._now(),
        }
        if creation_uncertain is not None:
            changes["creation_uncertain"] = creation_uncertain
        return self._transition_nonterminal(current, changes=changes)

    def _cleanup_family(self, root_cleanup_token: str) -> tuple[TrainingPodLease, ...]:
        profile_ids = dict.fromkeys((*TRAINING_PROFILE_IDS, *self.profiles))
        legacy_tokens = tuple(
            fallback_training_cleanup_token(root_cleanup_token, profile_id)
            for profile_id in profile_ids
        )
        return self.repository.list_cleanup_family(
            root_cleanup_token, legacy_cleanup_tokens=legacy_tokens
        )

    def _mark_family_release_requested(
        self, root: TrainingPodLease
    ) -> TrainingPodLease:
        current = self._required(root.cleanup_token)
        while not current.family_release_requested:
            try:
                return self.repository.compare_and_set(
                    current,
                    changes={
                        "family_release_requested": True,
                        "family_release_complete": False,
                        "last_heartbeat_at": self._now(),
                    },
                )
            except TrainingPodConflictError:
                current = self._required(root.cleanup_token)
        return current

    def _mark_family_release_complete(self, root: TrainingPodLease) -> TrainingPodLease:
        current = self._required(root.cleanup_token)
        while not current.family_release_complete:
            try:
                return self.repository.compare_and_set(
                    current,
                    changes={
                        "family_release_complete": True,
                        "last_heartbeat_at": self._now(),
                    },
                )
            except TrainingPodConflictError:
                current = self._required(root.cleanup_token)
        return current

    def _transition_nonterminal(
        self, lease: TrainingPodLease, *, changes: Mapping[str, Any]
    ) -> TrainingPodLease:
        """Apply a CAS transition without ever reviving a concurrent terminal row."""

        current = lease
        while not current.is_terminal:
            try:
                return self.repository.compare_and_set(current, changes=changes)
            except TrainingPodConflictError:
                current = self._required(lease.cleanup_token)
        return current

    def _record_reconcile_error(
        self, cleanup_token: str, error: BaseException
    ) -> TrainingPodLease | None:
        lease = self.repository.get(cleanup_token)
        if lease is None:
            return lease
        if lease.is_terminal and lease.billing_state not in {
            PodCapacityBillingState.PENDING,
            PodCapacityBillingState.UNRESOLVED,
        }:
            return lease
        try:
            return self.repository.compare_and_set(
                lease,
                changes={"last_provider_error": sanitize_training_error(error)},
            )
        except TrainingPodConflictError:
            logger.warning(
                "Training cleanup token %s changed while recording reconcile error",
                cleanup_token,
            )
            return self.repository.get(cleanup_token)

    async def _reconcile_billing(self, lease: TrainingPodLease) -> TrainingPodLease:
        if not lease.is_catalog_attempt:
            return lease
        if lease.billing_state is PodCapacityBillingState.AUTHORITATIVE:
            return await self._revoke_and_release(lease)
        if lease.provider_pod_id is None:
            if lease.creation_uncertain:
                return self.repository.compare_and_set(
                    lease,
                    changes={
                        "billing_state": PodCapacityBillingState.UNRESOLVED,
                        "last_provider_error": "AmbiguousCreateBillingUnresolved",
                    },
                )
            receipt = self._zero_cost_receipt(lease, "no-provider-capacity")
        else:
            if lease.terminated_at is None:
                raise RunPodManagerError(
                    "Terminated catalog Pod omitted its durable termination time"
                )
            spec = self._capacity_spec(lease)
            receipt = await self.provider.final_billing(
                lease.provider_pod_id,
                capacity_spec=spec,
                created_at=lease.created_at,
                terminated_at=lease.terminated_at,
            )
            if receipt is None:
                if lease.billing_state is PodCapacityBillingState.PENDING:
                    return lease
                return self.repository.compare_and_set(
                    lease,
                    changes={"billing_state": PodCapacityBillingState.PENDING},
                )
            if receipt.provider_pod_id != lease.provider_pod_id:
                raise RunPodManagerError(
                    "Provider billing receipt does not match the leased Pod"
                )
            if (
                lease.terminated_at is None
                or receipt.billed_until < lease.terminated_at
            ):
                raise RunPodManagerError(
                    "Provider billing receipt does not cover confirmed Pod teardown"
                )
            if (
                lease.evidence is not None
                and lease.evidence.realized_placement is not None
                and receipt.hourly_price_usd
                != lease.evidence.realized_placement.hourly_rate_usd
            ):
                raise RunPodManagerError(
                    "Provider billing rate does not match realized Pod rate"
                )
        try:
            reconciled_at = receipt.reconciled_at
            reconciled = self._transition_with_evidence(
                lease,
                changes={
                    "billing_state": PodCapacityBillingState.AUTHORITATIVE,
                    "billing_receipt_json": receipt,
                    "last_provider_error": None,
                    "last_heartbeat_at": reconciled_at,
                },
                lifecycle={"billing_reconciled_at": reconciled_at},
                billing=receipt,
            )
        except TrainingPodConflictError:
            current = self._required(lease.cleanup_token)
            if current.billing_state is not PodCapacityBillingState.AUTHORITATIVE:
                raise
            reconciled = current
        return await self._revoke_and_release(reconciled)

    def _transition_with_evidence(
        self,
        lease: TrainingPodLease,
        *,
        changes: Mapping[str, Any],
        lifecycle: Mapping[str, datetime] | None = None,
        realized_placement: PodRealizedPlacement | None | object = _EVIDENCE_UNSET,
        worker: CatalogWorkerEvidence | object = _EVIDENCE_UNSET,
        billing: PodBillingReceipt | None | object = _EVIDENCE_UNSET,
    ) -> TrainingPodLease:
        """CAS state and first-observation evidence without stale overwrites."""

        current = lease
        while not current.is_terminal:
            persisted = dict(changes)
            evidence = current.evidence
            if evidence is not None:
                updated_lifecycle = evidence.lifecycle
                if lifecycle:
                    first_values = {
                        name: (
                            getattr(updated_lifecycle, name)
                            if getattr(updated_lifecycle, name) is not None
                            else value
                        )
                        for name, value in lifecycle.items()
                    }
                    updated_lifecycle = replace(updated_lifecycle, **first_values)
                realized = evidence.realized_placement
                if realized_placement is not _EVIDENCE_UNSET:
                    if realized_placement is None:
                        raise RunPodManagerError(
                            "Catalog Pod realized placement cannot be absent"
                        )
                    if realized is not None and realized != realized_placement:
                        raise TrainingPodConflictError(
                            "Catalog Pod realized placement evidence already differs"
                        )
                    realized = realized or realized_placement
                worker_value = evidence.worker
                if worker is not _EVIDENCE_UNSET:
                    if not isinstance(worker, CatalogWorkerEvidence):
                        raise TypeError("Catalog worker evidence is invalid")
                    if worker_value is not None and worker_value != worker:
                        raise TrainingPodConflictError(
                            "Catalog worker evidence was already recorded differently"
                        )
                    worker_value = worker_value or worker
                billing_value = evidence.billing
                if billing is not _EVIDENCE_UNSET:
                    if billing is not None and not isinstance(
                        billing, PodBillingReceipt
                    ):
                        raise TypeError("Catalog billing evidence is invalid")
                    if billing_value is not None and billing_value != billing:
                        raise TrainingPodConflictError(
                            "Catalog billing evidence was already recorded differently"
                        )
                    billing_value = billing_value or billing
                updated_evidence = replace(
                    evidence,
                    lifecycle=updated_lifecycle,
                    realized_placement=realized,
                    worker=worker_value,
                    billing=billing_value,
                )
                persisted["evidence_json"] = updated_evidence
            try:
                return self.repository.compare_and_set(current, changes=persisted)
            except TrainingPodConflictError:
                current = self._required(lease.cleanup_token)
        return current

    async def _revoke_and_release(self, lease: TrainingPodLease) -> TrainingPodLease:
        """Make bearer revocation part of the durable terminal transition."""

        if lease.billing_state is not PodCapacityBillingState.AUTHORITATIVE:
            raise RunPodManagerError(
                "Catalog Pod cannot become terminal before authoritative billing"
            )
        await self._require_capability_store().revoke(
            self._capacity_spec(lease).request.attempt_id
        )
        current = self._required(lease.cleanup_token)
        if current.is_terminal:
            return current
        return self.repository.compare_and_set(
            current,
            changes={
                "state": TrainingPodState.RELEASED,
                "last_heartbeat_at": self._now(),
            },
        )

    def _record_catalog_observation(
        self,
        capacity_id: str,
        observation: CatalogPodWorkloadObservation,
        *,
        submitted: bool = False,
    ) -> TrainingPodLease:
        lease = self._required(capacity_id)
        if lease.is_terminal:
            return lease
        if lease.workload_state is CatalogPodWorkloadState.CANCEL_REQUESTED:
            return lease
        spec = self._capacity_spec(lease)
        if (
            observation.attempt_id != spec.request.attempt_id
            or observation.request_sha256 != spec.request.request_sha256
        ):
            raise TrainingPodConflictError(
                "Catalog workload observation does not match the exact attempt binding"
            )
        observed_at = self._now()
        changes: dict[str, Any] = {
            "workload_state": observation.state,
            "workload_error_type": observation.error_type,
            "last_provider_error": None,
            "last_heartbeat_at": observed_at,
        }
        lifecycle: dict[str, datetime] = {}
        if submitted:
            lifecycle["workload_submitted_at"] = observed_at
        if observation.state is CatalogPodWorkloadState.RUNNING:
            changes.update(
                state=TrainingPodState.JOB_SUBMITTED,
                provider_job_id=observation.attempt_id,
            )
            lifecycle["workload_running_at"] = observed_at
        elif observation.state is CatalogPodWorkloadState.SUCCEEDED:
            changes.update(
                state=TrainingPodState.JOB_COMPLETED,
                provider_job_id=observation.attempt_id,
            )
            lifecycle["workload_terminal_at"] = observed_at
        elif observation.state in {
            CatalogPodWorkloadState.FAILED,
            CatalogPodWorkloadState.CANCELLED,
            CatalogPodWorkloadState.CANCEL_REQUESTED,
        }:
            changes["state"] = TrainingPodState.CANCEL_REQUESTED
            if observation.state in {
                CatalogPodWorkloadState.FAILED,
                CatalogPodWorkloadState.CANCELLED,
            }:
                lifecycle["workload_terminal_at"] = observed_at
        return self._transition_with_evidence(
            lease, changes=changes, lifecycle=lifecycle
        )

    def _catalog_lease(
        self, capacity_id: str, owner_id: str, workload_id: str
    ) -> TrainingPodLease:
        return self.get_catalog_capacity(
            capacity_id=capacity_id,
            owner_id=owner_id,
            workload_id=workload_id,
        )

    @staticmethod
    def _capacity_spec(lease: TrainingPodLease) -> PodCapacitySpec:
        if lease.capacity_spec is None:
            raise RunPodManagerError("Pod lease is not a catalog capacity attempt")
        return lease.capacity_spec

    async def _load_capability(self, spec: PodCapacitySpec) -> CatalogAttemptCapability:
        capability = await self._require_capability_store().load(
            spec.request.attempt_id
        )
        if capability is None:
            raise RunPodManagerError("Catalog Pod capability is unavailable")
        _validate_capability(spec.request, capability)
        if (
            capability.secret_id != spec.capability_secret_id
            or capability.token_sha256 != spec.capability_token_sha256
        ):
            raise RunPodManagerError(
                "Catalog Pod capability does not match the durable lease"
            )
        if capability.expires_at <= self._now():
            raise RunPodManagerError("Catalog Pod capability has expired")
        return capability

    def _require_capability_store(self) -> CatalogAttemptCapabilityStore:
        if self._capability_store is None:
            raise RunPodManagerError(
                "Catalog Pod capacity requires an encrypted capability store"
            )
        return self._capability_store

    def _require_workload_transport(self) -> CatalogPodWorkloadTransport:
        if self._workload_transport is None:
            raise RunPodManagerError(
                "Catalog Pod capacity requires the authenticated workload transport"
            )
        return self._workload_transport

    @staticmethod
    def _route(lease: TrainingPodLease) -> str:
        if not lease.backend_base_url:
            raise RunPodManagerError("Catalog Pod lease has no workload route")
        return lease.backend_base_url

    def _zero_cost_receipt(
        self, lease: TrainingPodLease, reason: str
    ) -> PodBillingReceipt:
        spec = self._capacity_spec(lease)
        now = self._now()
        digest = hashlib.sha256(
            f"{lease.cleanup_token}\0{reason}\0{lease.request_fingerprint}".encode()
        ).hexdigest()
        return PodBillingReceipt(
            provider_billing_id=f"runpod-no-pod:{digest}",
            provider_pod_id=None,
            billed_from=now,
            billed_until=now,
            billed_seconds=0,
            hourly_price_usd=spec.request.quote.hourly_cost_usd,
            actual_cost_usd=Decimal(0),
            reconciled_at=now,
        )

    def _profile(self, profile_id: str) -> GPUProfile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise RunPodManagerError(
                f"Stored training profile '{profile_id}' is no longer configured"
            ) from exc

    def _required(self, cleanup_token: str) -> TrainingPodLease:
        lease = self.repository.get(cleanup_token)
        if lease is None:
            raise RunPodManagerError(
                f"Training cleanup token '{cleanup_token}' was not found"
            )
        return lease

    @staticmethod
    def _pod_id(lease: TrainingPodLease) -> str:
        if not lease.provider_pod_id:
            raise RunPodManagerError(
                f"Training cleanup token '{lease.cleanup_token}' has no Pod ID"
            )
        return lease.provider_pod_id

    @staticmethod
    def _cleanup_error(reason: str, lease: TrainingPodLease) -> TrainingPodCleanupError:
        return TrainingPodCleanupError(
            reason,
            cleanup_token=lease.cleanup_token,
            pod_id=lease.provider_pod_id,
            cleanup_state=lease.cleanup_state,
            billing_risk=True,
        )

    @staticmethod
    def _lifecycle_error(
        operation: str, lease: TrainingPodLease, *, billing_risk: bool
    ) -> TrainingPodLifecycleError:
        return TrainingPodLifecycleError(
            operation,
            cleanup_token=lease.cleanup_token,
            pod_id=lease.provider_pod_id,
            cleanup_state=lease.cleanup_state,
            billing_risk=billing_risk,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Training Pod service clock must be timezone-aware")
        return value.astimezone(UTC)


TrainingPodLeaseService = PodCapacityLeaseService


def _validate_capability(
    request: CatalogPodCapacityRequest, capability: CatalogAttemptCapability
) -> None:
    if capability.expires_at != request.bearer_expires_at:
        raise RunPodManagerError(
            "Catalog Pod capability expiry does not match the attempt policy"
        )
