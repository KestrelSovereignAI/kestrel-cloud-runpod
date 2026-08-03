"""Billing-safe training acquisition, cancellation, and reconciliation tests."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
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
