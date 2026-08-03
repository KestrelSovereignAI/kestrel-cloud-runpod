"""Restart-safe acquisition and cleanup state machine for training Pods."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import GPUProfile, RunPodAmbiguousResultError, RunPodManagerError
from .training_contracts import (
    TRAINING_PROFILE_IDS,
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodLease,
    TrainingPodLifecycleError,
    TrainingPodOwnership,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
    fallback_training_cleanup_token,
    sanitize_training_error,
)
from .training_provider import TrainingPodCapacityProvider
from .training_repository import SQLiteTrainingPodRepository

logger = logging.getLogger(__name__)


class TrainingPodLeaseService:
    """Own every transition from pre-mutation claim through confirmed stop."""

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
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep

    async def acquire(self, request: TrainingPodRequest) -> TrainingPodLease:
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
            return await self._acquire_created(lease, profile)
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
        self, lease: TrainingPodLease, profile: GPUProfile
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
            created = await self.provider.create(
                profile=profile,
                resource_name=lease.resource_name,
                companion_id=lease.companion_id,
            )
        except asyncio.CancelledError:
            lease = self._required(lease.cleanup_token)
            await self._cleanup_cancelled_acquisition(lease, "Pod create cancellation")
            raise
        except RunPodAmbiguousResultError as exc:
            lease = self._record_retryable_failure(lease, exc, creation_uncertain=True)
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
            released = self.repository.compare_and_set(
                lease,
                changes={
                    "state": TrainingPodState.RELEASED,
                    "cleanup_state": TrainingPodCleanupState.COMPLETE,
                    "creation_uncertain": False,
                    "last_provider_error": sanitize_training_error(exc),
                    "last_heartbeat_at": self._now(),
                },
            )
            raise self._lifecycle_error(
                "Pod create", released, billing_risk=False
            ) from exc

        lease = self.repository.compare_and_set(
            self._required(lease.cleanup_token),
            changes={
                "provider_pod_id": created.provider_pod_id,
                "ownership": TrainingPodOwnership.OWNED,
                "creation_uncertain": False,
                "last_heartbeat_at": self._now(),
                "last_provider_error": None,
            },
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
            lease = self.repository.compare_and_set(
                self._required(lease.cleanup_token),
                changes={
                    "backend_base_url": observation.backend_base_url,
                    "last_heartbeat_at": self._now(),
                    "last_provider_error": None,
                },
            )
            if observation.is_failed or observation.is_stopped:
                cause = RunPodManagerError(
                    f"Training Pod entered unusable state {observation.status}"
                )
                return await self._raise_after_cleanup(
                    lease, operation="readiness", cause=cause
                )
            if observation.is_running and observation.backend_base_url:
                return self.repository.compare_and_set(
                    lease,
                    changes={
                        "state": TrainingPodState.READY,
                        "last_heartbeat_at": self._now(),
                    },
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
            return self._transition_nonterminal(
                lease,
                changes={
                    "state": TrainingPodState.RELEASED,
                    "cleanup_state": TrainingPodCleanupState.COMPLETE,
                    "backend_base_url": None,
                    "last_provider_error": None,
                    "last_heartbeat_at": self._now(),
                },
            )
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
        return self._transition_nonterminal(
            lease,
            changes={
                "state": TrainingPodState.RELEASED,
                "cleanup_state": TrainingPodCleanupState.COMPLETE,
                "creation_uncertain": False,
                "backend_base_url": None,
                "last_provider_error": None,
                "last_heartbeat_at": self._now(),
            },
        )

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
        if lease is None or lease.is_terminal:
            return lease
        now = self._now()
        stale = now >= lease.last_heartbeat_at + timedelta(
            seconds=self.orphan_timeout_seconds
        )
        if lease.state is TrainingPodState.REQUESTED and (
            stale or now >= lease.readiness_deadline
        ):
            # REQUESTED is persisted before provider discovery/mutation. A crash
            # in this exact window must release the claim without stopping a Pod
            # that may have been running before this invocation existed.
            if lease.source is not TrainingPodSource.CREATED:
                return self._release_unowned_claim(lease, error=None)
            return self.repository.compare_and_set(
                lease,
                changes={
                    "state": TrainingPodState.RELEASED,
                    "cleanup_state": TrainingPodCleanupState.COMPLETE,
                    "last_provider_error": None,
                },
            )
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
        }:
            return await self._release_one(cleanup_token, reason="retry teardown")
        if now >= lease.hard_deadline:
            return await self._release_one(cleanup_token, reason="hard deadline")
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
                return self.repository.compare_and_set(
                    lease,
                    changes={
                        "state": TrainingPodState.RELEASED,
                        "cleanup_state": TrainingPodCleanupState.COMPLETE,
                        "backend_base_url": None,
                        "last_provider_error": None,
                    },
                )
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
            pod_id = await self.provider.find_by_name(lease.resource_name)
        except RunPodManagerError as exc:
            return self._record_retryable_failure(lease, exc, creation_uncertain=True)
        if pod_id is not None:
            return self._transition_nonterminal(
                lease,
                changes={
                    "provider_pod_id": pod_id,
                    "ownership": TrainingPodOwnership.OWNED,
                    "creation_uncertain": False,
                    "state": TrainingPodState.RECONCILE_REQUIRED,
                    "cleanup_state": TrainingPodCleanupState.PENDING,
                    "last_provider_error": None,
                },
            )
        if self._now() < lease.readiness_deadline:
            return self._transition_nonterminal(
                lease,
                changes={
                    "state": TrainingPodState.RECONCILE_REQUIRED,
                    "cleanup_state": TrainingPodCleanupState.RETRYABLE_FAILURE,
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
        if lease is None or lease.is_terminal:
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
