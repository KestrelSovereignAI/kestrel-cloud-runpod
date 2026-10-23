"""Billing-safe training acquisition, cancellation, and reconciliation tests."""

import asyncio
import inspect
from datetime import timedelta
from pathlib import Path

import pytest
from pod_capacity_test_support import (
    execution_control_carrier,
    record_control_injection,
)
from training_test_support import (
    FakeTrainingProvider,
    MutableClock,
    training_profile,
    training_request,
    training_service,
)

from kestrel_cloud_runpod.models import RunPodAmbiguousResultError, RunPodManagerError
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodLifecycleError,
    TrainingPodOwnership,
    TrainingPodSource,
    TrainingPodState,
    fallback_training_cleanup_token,
)
from kestrel_cloud_runpod.training_repository import SQLiteTrainingPodRepository
from kestrel_cloud_runpod.training_service import TrainingPodLeaseService


def _cleanup_carrier(boundary: str):
    return TrainingPodCleanupError(
        boundary,
        cleanup_token=f"training:{boundary}:0001",
        pod_id="pod-training-1",
        cleanup_state=TrainingPodCleanupState.RETRYABLE_FAILURE,
        billing_risk=True,
    )


@pytest.mark.asyncio
async def test_resumed_pod_missing_route_is_stopped(tmp_path: Path) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    service = training_service(tmp_path, clock, provider)

    with pytest.raises(TrainingPodLifecycleError) as raised:
        await service.acquire(training_request(clock, readiness_seconds=20))

    assert raised.value.billing_risk is False
    lease = service.repository.get("training:test-token-0001")
    assert lease is not None
    assert lease.state is TrainingPodState.RELEASED
    assert lease.cleanup_state is TrainingPodCleanupState.COMPLETE
    assert provider.start_calls == ["pod-training-1"]
    assert provider.stop_calls == ["pod-training-1"]


@pytest.mark.asyncio
async def test_readiness_exception_stop_failure_retains_retryable_pod_after_restart(
    tmp_path: Path,
) -> None:
    clock = MutableClock()

    class ReadinessErrorProvider(FakeTrainingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.observe_calls = 0

        async def observe(self, pod_id, *, profile):
            self.observe_calls += 1
            if self.observe_calls > 1:
                raise RunPodManagerError("status unavailable")
            return await super().observe(pod_id, profile=profile)

    provider = ReadinessErrorProvider()
    provider.stop_error = RunPodManagerError("stop unavailable")
    service = training_service(tmp_path, clock, provider)

    with pytest.raises(TrainingPodCleanupError) as raised:
        await service.acquire(training_request(clock))

    assert raised.value.pod_id == "pod-training-1"
    lease = service.repository.get(raised.value.cleanup_token)
    assert lease is not None
    assert lease.provider_pod_id == "pod-training-1"
    assert lease.cleanup_state is TrainingPodCleanupState.RETRYABLE_FAILURE

    provider.observe_calls = 0
    provider.stop_error = None
    restarted = training_service(tmp_path, clock, provider)
    reconciled = await restarted.reconcile()
    assert reconciled[0].state is TrainingPodState.RELEASED
    assert reconciled[0].stop_attempts == 2


@pytest.mark.asyncio
async def test_preexisting_running_pod_route_failure_is_never_stopped(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.status = "RUNNING"
    service = training_service(tmp_path, clock, provider)

    with pytest.raises(TrainingPodLifecycleError) as raised:
        await service.acquire(training_request(clock, readiness_seconds=20))

    assert raised.value.billing_risk is False
    assert provider.stop_calls == []
    lease = service.repository.get("training:test-token-0001")
    assert lease is not None
    assert lease.ownership is TrainingPodOwnership.PREEXISTING_RUNNING
    assert lease.cleanup_state is TrainingPodCleanupState.NOT_OWNED


@pytest.mark.asyncio
async def test_ambiguous_create_reconciles_exact_name_and_stops_pod(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.create_error = RunPodAmbiguousResultError(
        title="transport", detail="timeout", method="POST", resource="/pods"
    )
    provider.find_result = "pod-created-after-timeout"
    service = training_service(tmp_path, clock, provider)

    with pytest.raises(TrainingPodLifecycleError) as raised:
        await service.acquire(
            training_request(clock, source=TrainingPodSource.CREATED, pod_id=None)
        )

    assert raised.value.billing_risk is False
    assert provider.stop_calls == ["pod-created-after-timeout"]
    lease = service.repository.get("training:test-token-0001")
    assert lease is not None
    assert lease.state is TrainingPodState.RELEASED
    assert lease.provider_pod_id == "pod-created-after-timeout"


@pytest.mark.asyncio
async def test_restart_recovers_crash_between_create_and_id_persistence(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "training.sqlite3")
    lease, _ = repository.reserve(
        training_request(clock, source=TrainingPodSource.CREATED, pod_id=None)
    )
    repository.compare_and_set(
        lease,
        changes={
            "state": TrainingPodState.STARTING,
            "creation_uncertain": True,
            "ownership": TrainingPodOwnership.PROVISIONAL,
        },
    )
    provider = FakeTrainingProvider()
    provider.find_result = "pod-created-before-crash"
    clock.value += timedelta(seconds=31)
    restarted = TrainingPodLeaseService(
        repository=SQLiteTrainingPodRepository(tmp_path / "training.sqlite3"),
        provider=provider,
        profiles={"training": training_profile()},
        poll_interval_seconds=10,
        orphan_timeout_seconds=30,
        clock=clock,
        sleep=clock.sleep,
    )

    reconciled = await restarted.reconcile()
    assert reconciled[0].state is TrainingPodState.RELEASED
    assert provider.stop_calls == ["pod-created-before-crash"]


@pytest.mark.asyncio
async def test_reconciler_leaves_live_inflight_create_owned_by_acquirer(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    create_started = asyncio.Event()
    allow_create = asyncio.Event()

    class BlockingCreateProvider(FakeTrainingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.find_calls = 0

        async def create(self, *, profile, resource_name, companion_id):
            create_started.set()
            await allow_create.wait()
            return await super().create(
                profile=profile,
                resource_name=resource_name,
                companion_id=companion_id,
            )

        async def find_by_name(self, resource_name: str) -> str | None:
            self.find_calls += 1
            return await super().find_by_name(resource_name)

    provider = BlockingCreateProvider()
    provider.route = "https://pod-created-1-8888.proxy.runpod.net"
    acquirer = training_service(tmp_path, clock, provider)
    reconciler = training_service(tmp_path, clock, provider)
    acquisition = asyncio.create_task(
        acquirer.acquire(
            training_request(clock, source=TrainingPodSource.CREATED, pod_id=None)
        )
    )
    await create_started.wait()

    reconciled = await reconciler.reconcile()

    assert reconciled[0].state is TrainingPodState.STARTING
    assert reconciled[0].creation_uncertain is True
    assert provider.find_calls == 0
    assert provider.stop_calls == []

    allow_create.set()
    acquired = await acquisition
    assert acquired.state is TrainingPodState.READY
    assert acquired.provider_pod_id == "pod-created-1"
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_restart_before_persistent_discovery_never_stops_preexisting_pod(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "training.sqlite3")
    repository.reserve(training_request(clock))
    provider = FakeTrainingProvider()
    provider.status = "RUNNING"
    clock.value += timedelta(seconds=31)
    restarted = training_service(tmp_path, clock, provider)

    reconciled = await restarted.reconcile()

    assert reconciled[0].state is TrainingPodState.RELEASED
    assert reconciled[0].cleanup_state is TrainingPodCleanupState.NOT_OWNED
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_restart_reconciles_submitted_job_at_hard_deadline(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    service = training_service(tmp_path, clock, provider)
    lease = await service.acquire(training_request(clock, hard_seconds=60))
    service.record_job(lease.cleanup_token, "job-1")
    clock.value = lease.hard_deadline

    restarted = training_service(tmp_path, clock, provider)
    reconciled = await restarted.reconcile()

    assert reconciled[0].state is TrainingPodState.RELEASED
    assert provider.stop_calls == ["pod-training-1"]


@pytest.mark.asyncio
async def test_restart_recovers_submitted_job_status_before_hard_deadline(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    service = training_service(tmp_path, clock, provider)
    lease = await service.acquire(training_request(clock))
    service.record_job(lease.cleanup_token, "job-1")

    async def completed(_):
        return "completed"

    restarted = TrainingPodLeaseService(
        repository=SQLiteTrainingPodRepository(tmp_path / "training.sqlite3"),
        provider=provider,
        profiles={"training": training_profile()},
        poll_interval_seconds=10,
        orphan_timeout_seconds=30,
        workload_status_observer=completed,
        clock=clock,
        sleep=clock.sleep,
    )
    reconciled = await restarted.reconcile()

    assert reconciled[0].state is TrainingPodState.JOB_COMPLETED
    assert reconciled[0].provider_job_id == "job-1"
    assert provider.stop_calls == []


@pytest.mark.asyncio
async def test_restart_observes_failed_job_and_stops_owned_capacity(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    service = training_service(tmp_path, clock, provider)
    lease = await service.acquire(training_request(clock))
    service.record_job(lease.cleanup_token, "job-1")

    async def failed(_):
        return "failed"

    restarted = TrainingPodLeaseService(
        repository=SQLiteTrainingPodRepository(tmp_path / "training.sqlite3"),
        provider=provider,
        profiles={"training": training_profile()},
        poll_interval_seconds=10,
        orphan_timeout_seconds=30,
        workload_status_observer=failed,
        clock=clock,
        sleep=clock.sleep,
    )

    reconciled = await restarted.reconcile()

    assert reconciled[0].state is TrainingPodState.RELEASED
    assert reconciled[0].cleanup_state is TrainingPodCleanupState.COMPLETE
    assert provider.stop_calls == ["pod-training-1"]


@pytest.mark.asyncio
async def test_concurrent_acquisition_cannot_double_resume_one_pod(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    first = training_service(tmp_path, clock, provider)
    second = training_service(tmp_path, clock, provider)

    first_lease = await first.acquire(
        training_request(clock, token="training:first-token-0001")
    )
    with pytest.raises(TrainingPodConflictError):
        await second.acquire(
            training_request(clock, token="training:second-token-0002")
        )

    assert first_lease.state is TrainingPodState.READY
    assert provider.start_calls == ["pod-training-1"]


@pytest.mark.asyncio
async def test_cancelled_resume_runs_shielded_cleanup(tmp_path: Path) -> None:
    clock = MutableClock()
    started = asyncio.Event()
    never = asyncio.Event()

    class BlockingProvider(FakeTrainingProvider):
        async def start(self, pod_id: str, *, gpu_count: int) -> None:
            self.start_calls.append(pod_id)
            started.set()
            await never.wait()

    provider = BlockingProvider()
    service = training_service(tmp_path, clock, provider)
    task = asyncio.create_task(service.acquire(training_request(clock)))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.stop_calls == ["pod-training-1"]
    lease = service.repository.get("training:test-token-0001")
    assert lease is not None
    assert lease.state is TrainingPodState.RELEASED


@pytest.mark.asyncio
async def test_stop_pending_keeps_id_and_reconciler_retries(tmp_path: Path) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    provider.stop_confirms = False
    service = training_service(tmp_path, clock, provider)
    lease = await service.acquire(training_request(clock))

    with pytest.raises(TrainingPodCleanupError):
        await service.release(lease.cleanup_token, reason="test")
    retained = service.repository.get(lease.cleanup_token)
    assert retained is not None
    assert retained.provider_pod_id == "pod-training-1"
    assert retained.cleanup_state is TrainingPodCleanupState.PENDING

    provider.stop_confirms = True
    reconciled = await service.reconcile()
    assert reconciled[0].state is TrainingPodState.RELEASED


async def _persist_fallback_after_released_root(
    tmp_path: Path,
    clock: MutableClock,
    provider: FakeTrainingProvider,
    *,
    legacy: bool,
) -> tuple[TrainingPodLeaseService, str, str]:
    profiles = {
        "training": training_profile(),
        "training-h100": training_profile("training-h100"),
    }
    service = training_service(tmp_path, clock, provider, profiles=profiles)
    root_token = "training:fallback-family-root-0001"
    provider.create_error = RunPodManagerError("profile unavailable")
    with pytest.raises(TrainingPodLifecycleError):
        await service.acquire(
            training_request(
                clock,
                token=root_token,
                source=TrainingPodSource.CREATED,
                pod_id=None,
            )
        )
    provider.create_error = None
    provider.route = "https://pod-created-1-8888.proxy.runpod.net"
    child_token = fallback_training_cleanup_token(root_token, "training-h100")
    child = await service.acquire(
        training_request(
            clock,
            token=child_token,
            root_token=(None if legacy else root_token),
            profile_id="training-h100",
            source=TrainingPodSource.CREATED,
            pod_id=None,
        )
    )
    assert child.state is TrainingPodState.READY
    return service, root_token, child_token


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_root_token_releases_persisted_fallback_family_after_restart(
    tmp_path: Path, legacy: bool
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    _, root_token, child_token = await _persist_fallback_after_released_root(
        tmp_path, clock, provider, legacy=legacy
    )
    restarted_profiles = {"training": training_profile()}
    if not legacy:
        restarted_profiles["training-h100"] = training_profile("training-h100")
    restarted = training_service(
        tmp_path,
        clock,
        provider,
        profiles=restarted_profiles,
    )
    recovered = restarted.get_active_family_attempt(root_token)
    assert recovered is not None and recovered.cleanup_token == child_token

    released_root = await restarted.release(root_token, reason="crash recovery")

    released_child = restarted.repository.get(child_token)
    assert released_child is not None
    assert released_child.state is TrainingPodState.RELEASED
    assert provider.stop_calls == ["pod-created-1"]
    assert released_root.cleanup_token == root_token
    assert released_root.family_release_requested is True
    assert released_root.family_release_complete is True


@pytest.mark.asyncio
async def test_returned_child_token_keeps_exact_release_behavior(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    service, root_token, child_token = await _persist_fallback_after_released_root(
        tmp_path, clock, provider, legacy=False
    )

    released = await service.release(child_token, reason="returned session")

    root = service.repository.get(root_token)
    assert root is not None
    assert released.cleanup_token == child_token
    assert released.state is TrainingPodState.RELEASED
    assert root.family_release_requested is False


@pytest.mark.asyncio
async def test_reconcile_finishes_family_release_interrupted_before_child_stop(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    service, root_token, child_token = await _persist_fallback_after_released_root(
        tmp_path, clock, provider, legacy=True
    )
    root = service.repository.get(root_token)
    assert root is not None
    service.repository.compare_and_set(
        root,
        changes={
            "family_release_requested": True,
            "family_release_complete": False,
        },
    )
    restarted = training_service(
        tmp_path,
        clock,
        provider,
        profiles={"training": training_profile()},
    )

    await restarted.reconcile()

    child = restarted.repository.get(child_token)
    root = restarted.repository.get(root_token)
    assert child is not None and child.state is TrainingPodState.RELEASED
    assert root is not None and root.family_release_complete is True
    assert provider.stop_calls == ["pod-created-1"]


@pytest.mark.asyncio
async def test_status_and_error_recording_cannot_revive_released_attempt(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    service = training_service(tmp_path, clock, provider)
    acquired = await service.acquire(training_request(clock))
    released = await service.release(acquired.cleanup_token, reason="test")

    status = service.record_status(released.cleanup_token, "failed")
    error = service.record_operation_error(
        released.cleanup_token, RunPodManagerError("late observer")
    )

    assert status == released
    assert error == released
    assert service.repository.get(released.cleanup_token) == released


@pytest.mark.asyncio
async def test_reconcile_tolerates_removed_legacy_profile_until_orphan_cleanup(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    _, _, child_token = await _persist_fallback_after_released_root(
        tmp_path, clock, provider, legacy=True
    )
    restarted = training_service(
        tmp_path,
        clock,
        provider,
        profiles={"training": training_profile()},
    )

    healthy = await restarted.reconcile()

    child = restarted.repository.get(child_token)
    assert child is not None and child in healthy
    assert child.state is TrainingPodState.READY
    assert child.last_provider_error is None
    assert provider.stop_calls == []

    clock.value += timedelta(seconds=31)
    await restarted.reconcile()
    child = restarted.repository.get(child_token)
    assert child is not None and child.state is TrainingPodState.RELEASED
    assert provider.stop_calls == ["pod-created-1"]


@pytest.mark.asyncio
async def test_concurrent_family_release_and_reconcile_are_terminal_aware(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    first_stop_started = asyncio.Event()
    allow_stops = asyncio.Event()

    class BlockingStopProvider(FakeTrainingProvider):
        async def stop(self, pod_id):
            self.stop_calls.append(pod_id)
            first_stop_started.set()
            await allow_stops.wait()
            self.status = "EXITED"
            return True

    provider = BlockingStopProvider()
    first, root_token, child_token = await _persist_fallback_after_released_root(
        tmp_path, clock, provider, legacy=False
    )
    second = training_service(
        tmp_path,
        clock,
        provider,
        profiles={
            "training": training_profile(),
            "training-h100": training_profile("training-h100"),
        },
    )
    release_task = asyncio.create_task(first.release(root_token, reason="caller"))
    await first_stop_started.wait()
    reconcile_task = asyncio.create_task(second.reconcile())
    while len(provider.stop_calls) < 2:
        await asyncio.sleep(0)
    allow_stops.set()

    released_root, _ = await asyncio.gather(release_task, reconcile_task)

    child = first.repository.get(child_token)
    assert child is not None and child.state is TrainingPodState.RELEASED
    assert released_root.family_release_complete is True
    assert provider.stop_calls == ["pod-created-1", "pod-created-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "existing-observe",
        "existing-start",
        "existing-start-cleanup",
        "ambiguous-create-cleanup",
        "family-cleanup",
        "family-manager",
        "reconcile-family",
        "reconcile-lease",
        "reconcile-workload",
        "stop-attempt-conflict",
        "family-request-conflict",
        "family-complete-conflict",
        "transition-conflict",
        "record-reconcile-conflict",
    ],
)
async def test_training_service_preserves_exact_execution_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    record_property,
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    runtime = training_service(tmp_path, clock, provider)
    if boundary.endswith("-cleanup"):
        error = _cleanup_carrier(boundary)
    elif boundary.endswith("-conflict"):
        error = TrainingPodConflictError(f"{boundary} remains unknown")
    else:
        error = RunPodManagerError(f"{boundary} remains unknown")
    carrier, control, explicit = execution_control_carrier(
        error,
        grouped=boundary in {"family-manager", "reconcile-family"},
    )
    durable_at_injection = None
    family_at_injection = None

    def capture_injection(cleanup_token: str) -> None:
        nonlocal durable_at_injection, family_at_injection
        durable_at_injection = runtime.repository.get(cleanup_token)
        assert durable_at_injection is not None
        family_at_injection = runtime.repository.list_cleanup_family(
            durable_at_injection.root_cleanup_token
        )
        record_control_injection(record_property, boundary)

    if boundary == "existing-observe":
        provider.error_callback = lambda: capture_injection("training:test-token-0001")
        provider.observe_error = carrier
        operation = lambda: runtime.acquire(training_request(clock))
    elif boundary in {"existing-start", "existing-start-cleanup"}:
        provider.status = "EXITED"
        provider.start_error = (
            RunPodManagerError("ordinary Pod start failure")
            if boundary.endswith("-cleanup")
            else carrier
        )
        if boundary.endswith("-cleanup"):

            async def controlled_release(cleanup_token: str, *_args, **_kwargs):
                capture_injection(cleanup_token)
                raise carrier

            monkeypatch.setattr(runtime, "_release_one", controlled_release)
        else:
            provider.error_callback = lambda: capture_injection(
                "training:test-token-0001"
            )
        operation = lambda: runtime.acquire(training_request(clock))
    elif boundary == "ambiguous-create-cleanup":
        provider.create_error = RunPodAmbiguousResultError(
            title="unknown",
            detail="ordinary ambiguous create",
            method="POST",
            resource="/pods",
        )

        async def controlled_release(cleanup_token: str, *_args, **_kwargs):
            capture_injection(cleanup_token)
            raise carrier

        monkeypatch.setattr(runtime, "_release_one", controlled_release)
        operation = lambda: runtime.acquire(
            training_request(
                clock,
                token="training:ambiguous-create-cleanup:0001",
                source=TrainingPodSource.CREATED,
            )
        )
    elif boundary in {
        "family-cleanup",
        "family-manager",
        "reconcile-family",
    }:
        runtime, root_token, _child_token = await _persist_fallback_after_released_root(
            tmp_path, clock, provider, legacy=False
        )

        async def controlled_release_one(cleanup_token: str, *_args, **_kwargs):
            capture_injection(cleanup_token)
            raise carrier

        async def controlled_family_release(cleanup_token: str, *_args, **_kwargs):
            capture_injection(cleanup_token)
            raise carrier

        if boundary == "reconcile-family":
            root = runtime.repository.get(root_token)
            assert root is not None
            runtime.repository.compare_and_set(
                root,
                changes={
                    "family_release_requested": True,
                    "family_release_complete": False,
                },
            )
            monkeypatch.setattr(runtime, "release", controlled_family_release)
            operation = runtime.reconcile
        else:
            monkeypatch.setattr(runtime, "_release_one", controlled_release_one)
            operation = lambda: runtime.release(root_token, reason="controlled")
    elif boundary == "reconcile-lease":
        lease, _ = runtime.repository.reserve(training_request(clock))
        monkeypatch.setattr(
            runtime.repository,
            "list_for_reconciliation",
            lambda: (lease,),
        )

        async def controlled_reconcile_one(_cleanup_token):
            capture_injection(_cleanup_token)
            raise carrier

        monkeypatch.setattr(runtime, "_reconcile_one", controlled_reconcile_one)
        operation = runtime.reconcile
    elif boundary == "reconcile-workload":
        ready = await runtime.acquire(training_request(clock))
        runtime.record_job(ready.cleanup_token, "job-control")

        async def controlled_status(_lease):
            capture_injection(_lease.cleanup_token)
            raise carrier

        runtime._workload_status_observer = controlled_status
        operation = lambda: runtime._reconcile_one(ready.cleanup_token)
    else:
        if boundary == "stop-attempt-conflict":
            lease = await runtime.acquire(training_request(clock))
            changes_key = "stop_attempts"
            operation = lambda: runtime.release(
                lease.cleanup_token, reason="controlled"
            )
        else:
            lease, _ = runtime.repository.reserve(training_request(clock))
            if boundary == "family-complete-conflict":
                lease = runtime.repository.compare_and_set(
                    lease,
                    changes={
                        "family_release_requested": True,
                        "family_release_complete": False,
                    },
                )
                changes_key = "family_release_complete"
                operation = lambda: runtime._mark_family_release_complete(lease)
            elif boundary == "family-request-conflict":
                changes_key = "family_release_requested"
                operation = lambda: runtime._mark_family_release_requested(lease)
            elif boundary == "transition-conflict":
                changes_key = "last_provider_error"
                operation = lambda: runtime._transition_nonterminal(
                    lease, changes={"last_provider_error": "controlled"}
                )
            else:
                changes_key = "last_provider_error"
                operation = lambda: runtime._record_reconcile_error(
                    lease.cleanup_token, RunPodManagerError("ordinary reconcile error")
                )
        compare_and_set = runtime.repository.compare_and_set
        injected = False

        def controlled_compare_and_set(current, *, changes):
            nonlocal injected
            if not injected and changes_key in changes:
                injected = True
                capture_injection(current.cleanup_token)
                raise carrier
            return compare_and_set(current, changes=changes)

        monkeypatch.setattr(
            runtime.repository, "compare_and_set", controlled_compare_and_set
        )

    provider_calls = (
        tuple(provider.start_calls),
        tuple(provider.stop_calls),
        tuple(provider.create_calls),
    )
    with pytest.raises(type(carrier)) as raised:
        outcome = operation()
        if inspect.isawaitable(outcome):
            await outcome

    assert raised.value is carrier
    assert carrier.__cause__ is explicit
    if isinstance(explicit, BaseExceptionGroup):
        assert explicit.exceptions[0] is control
    else:
        assert explicit is control
    assert tuple(provider.start_calls) == provider_calls[0] + (
        ("pod-training-1",) if boundary.startswith("existing-start") else ()
    )
    assert tuple(provider.stop_calls) == provider_calls[1]
    assert len(provider.create_calls) == len(provider_calls[2]) + int(
        boundary == "ambiguous-create-cleanup"
    )
    assert durable_at_injection is not None
    assert family_at_injection is not None
    assert (
        runtime.repository.get(durable_at_injection.cleanup_token)
        == durable_at_injection
    )
    assert (
        runtime.repository.list_cleanup_family(durable_at_injection.root_cleanup_token)
        == family_at_injection
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "existing-observe",
        "existing-start",
        "existing-start-cleanup",
        "ambiguous-create-cleanup",
        "family-cleanup",
        "family-manager",
        "reconcile-family",
        "reconcile-lease",
        "reconcile-workload",
        "stop-attempt-conflict",
        "family-request-conflict",
        "family-complete-conflict",
        "transition-conflict",
        "record-reconcile-conflict",
    ],
)
async def test_training_service_retains_ordinary_normalization_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    clock = MutableClock()
    provider = FakeTrainingProvider()
    provider.route = "https://pod-training-1-8888.proxy.runpod.net"
    runtime = training_service(tmp_path, clock, provider)
    ordinary_manager = RunPodManagerError(f"ordinary {boundary} failure")
    ordinary_cleanup = _cleanup_carrier(f"ordinary-{boundary}")

    if boundary == "existing-observe":
        provider.observe_error = ordinary_manager
        with pytest.raises(TrainingPodLifecycleError) as raised:
            await runtime.acquire(training_request(clock))
        assert raised.value.__cause__ is ordinary_manager
        assert provider.stop_calls == []
        return

    if boundary in {"existing-start", "existing-start-cleanup"}:
        provider.status = "EXITED"
        provider.start_error = ordinary_manager
        if boundary.endswith("-cleanup"):

            async def failed_release(*_args, **_kwargs):
                raise ordinary_cleanup

            monkeypatch.setattr(runtime, "_release_one", failed_release)
            with pytest.raises(TrainingPodCleanupError) as raised:
                await runtime.acquire(training_request(clock))
            assert raised.value is ordinary_cleanup
            assert raised.value.__cause__ is ordinary_manager
            assert provider.stop_calls == []
        else:
            with pytest.raises(TrainingPodLifecycleError) as raised:
                await runtime.acquire(training_request(clock))
            assert raised.value.__cause__ is ordinary_manager
            assert provider.stop_calls == ["pod-training-1"]
        return

    if boundary == "ambiguous-create-cleanup":
        ambiguous = RunPodAmbiguousResultError(
            title="unknown",
            detail="ordinary ambiguous create",
            method="POST",
            resource="/pods",
        )
        provider.create_error = ambiguous

        async def failed_release(*_args, **_kwargs):
            raise ordinary_cleanup

        monkeypatch.setattr(runtime, "_release_one", failed_release)
        with pytest.raises(TrainingPodCleanupError) as raised:
            await runtime.acquire(
                training_request(
                    clock,
                    token="training:ambiguous-create-cleanup:0001",
                    source=TrainingPodSource.CREATED,
                )
            )
        assert raised.value is ordinary_cleanup
        assert raised.value.__cause__ is ambiguous
        assert len(provider.create_calls) == 1
        assert provider.stop_calls == []
        return

    if boundary in {"family-cleanup", "family-manager", "reconcile-family"}:
        runtime, root_token, _child_token = await _persist_fallback_after_released_root(
            tmp_path, clock, provider, legacy=False
        )

        async def failed_release(*_args, **_kwargs):
            raise ordinary_cleanup if boundary == "family-cleanup" else ordinary_manager

        if boundary == "reconcile-family":
            root = runtime.repository.get(root_token)
            assert root is not None
            runtime.repository.compare_and_set(
                root,
                changes={
                    "family_release_requested": True,
                    "family_release_complete": False,
                },
            )
            monkeypatch.setattr(runtime, "release", failed_release)
            results = await runtime.reconcile()
            assert any(lease.cleanup_token == root_token for lease in results)
            current = runtime.repository.get(root_token)
            assert current is not None
            assert current.last_provider_error == "RunPodManagerError"
        else:
            monkeypatch.setattr(runtime, "_release_one", failed_release)
            with pytest.raises(TrainingPodCleanupError) as raised:
                await runtime.release(root_token, reason="ordinary family release")
            if boundary == "family-cleanup":
                assert raised.value.__cause__ is ordinary_cleanup
            else:
                assert type(raised.value.__cause__) is TrainingPodCleanupError
        return

    if boundary == "reconcile-lease":
        lease, _ = runtime.repository.reserve(training_request(clock))
        monkeypatch.setattr(
            runtime.repository, "list_for_reconciliation", lambda: (lease,)
        )

        async def failed_reconcile(_cleanup_token):
            raise ordinary_manager

        monkeypatch.setattr(runtime, "_reconcile_one", failed_reconcile)
        results = await runtime.reconcile()
        assert results[0].cleanup_token == lease.cleanup_token
        assert results[0].last_provider_error == "RunPodManagerError"
        return

    if boundary == "reconcile-workload":
        ready = await runtime.acquire(training_request(clock))
        runtime.record_job(ready.cleanup_token, "job-ordinary")

        async def failed_status(_lease):
            raise ordinary_manager

        runtime._workload_status_observer = failed_status
        result = await runtime._reconcile_one(ready.cleanup_token)
        assert result is not None
        assert result.last_provider_error == "RunPodManagerError"
        assert provider.stop_calls == []
        return

    if boundary == "stop-attempt-conflict":
        lease = await runtime.acquire(training_request(clock))
        changes_key = "stop_attempts"
        operation = lambda: runtime.release(lease.cleanup_token, reason="ordinary")
    else:
        lease, _ = runtime.repository.reserve(training_request(clock))
        if boundary == "family-complete-conflict":
            lease = runtime.repository.compare_and_set(
                lease,
                changes={
                    "family_release_requested": True,
                    "family_release_complete": False,
                },
            )
            changes_key = "family_release_complete"
            operation = lambda: runtime._mark_family_release_complete(lease)
        elif boundary == "family-request-conflict":
            changes_key = "family_release_requested"
            operation = lambda: runtime._mark_family_release_requested(lease)
        elif boundary == "transition-conflict":
            changes_key = "last_provider_error"
            operation = lambda: runtime._transition_nonterminal(
                lease, changes={"last_provider_error": "ordinary"}
            )
        else:
            changes_key = "last_provider_error"
            operation = lambda: runtime._record_reconcile_error(
                lease.cleanup_token, ordinary_manager
            )

    compare_and_set = runtime.repository.compare_and_set
    injected = False

    def conflict_once(current, *, changes):
        nonlocal injected
        if not injected and changes_key in changes:
            injected = True
            raise TrainingPodConflictError("ordinary concurrent update")
        return compare_and_set(current, changes=changes)

    monkeypatch.setattr(runtime.repository, "compare_and_set", conflict_once)
    outcome = operation()
    result = await outcome if inspect.isawaitable(outcome) else outcome
    assert result is not None
    if boundary == "stop-attempt-conflict":
        assert result.state is TrainingPodState.RELEASED
        assert provider.stop_calls == ["pod-training-1"]
    elif boundary == "family-request-conflict":
        assert result.family_release_requested is True
    elif boundary == "family-complete-conflict":
        assert result.family_release_complete is True
    elif boundary == "transition-conflict":
        assert result.last_provider_error == "ordinary"
    else:
        assert result.last_provider_error is None
