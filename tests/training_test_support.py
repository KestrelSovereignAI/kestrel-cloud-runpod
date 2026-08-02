"""Deterministic doubles for durable training Pod lifecycle tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from kestrel_cloud_runpod.models import GPUProfile, RunPodManagerError
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodRequest,
    TrainingPodSource,
    durable_training_name,
)
from kestrel_cloud_runpod.training_provider import (
    CreatedTrainingPod,
    TrainingPodObservation,
)
from kestrel_cloud_runpod.training_repository import SQLiteTrainingPodRepository
from kestrel_cloud_runpod.training_service import TrainingPodLeaseService


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 1, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def training_profile() -> GPUProfile:
    return GPUProfile(
        id="training",
        name="LoRA training",
        task_type="training",
        image_name="registry/trainer@sha256:" + "a" * 64,
        container_disk_gb=40,
        volume_gb=0,
        ports=["8888/http"],
        inference_port=8888,
        default_model="flux-lora-trainer",
        min_vram_gb=24,
        gpu_count=1,
    )


def training_request(
    clock: MutableClock,
    *,
    token: str = "training:test-token-0001",
    source: TrainingPodSource = TrainingPodSource.CONFIGURED_PERSISTENT,
    pod_id: str | None = "pod-training-1",
    readiness_seconds: int = 30,
    hard_seconds: int = 3600,
) -> TrainingPodRequest:
    return TrainingPodRequest(
        cleanup_token=token,
        companion_id="companion-1",
        profile_id="training",
        source=source,
        resource_name=durable_training_name(token),
        provider_pod_id=(None if source is TrainingPodSource.CREATED else pod_id),
        created_at=clock(),
        readiness_deadline=clock() + timedelta(seconds=readiness_seconds),
        hard_deadline=clock() + timedelta(seconds=hard_seconds),
    )


class FakeTrainingProvider:
    def __init__(self) -> None:
        self.status = "EXITED"
        self.route: str | None = None
        self.start_calls: list[str] = []
        self.create_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.find_result: str | None = None
        self.observe_error: RunPodManagerError | None = None
        self.start_error: RunPodManagerError | None = None
        self.create_error: RunPodManagerError | None = None
        self.stop_error: RunPodManagerError | None = None
        self.stop_confirms = True

    async def observe(
        self, pod_id: str, *, profile: GPUProfile
    ) -> TrainingPodObservation:
        assert profile.id == "training"
        if self.observe_error:
            raise self.observe_error
        return TrainingPodObservation(
            provider_pod_id=pod_id,
            status=self.status,
            backend_base_url=self.route,
            raw={"id": pod_id, "status": self.status},
        )

    async def start(self, pod_id: str, *, gpu_count: int) -> None:
        assert gpu_count == 1
        self.start_calls.append(pod_id)
        if self.start_error:
            raise self.start_error
        self.status = "RUNNING"

    async def create(
        self, *, profile: GPUProfile, resource_name: str, companion_id: str
    ) -> CreatedTrainingPod:
        assert profile.id == "training"
        assert companion_id == "companion-1"
        self.create_calls.append(resource_name)
        if self.create_error:
            raise self.create_error
        self.status = "RUNNING"
        return CreatedTrainingPod(provider_pod_id="pod-created-1", placement=None)

    async def find_by_name(self, resource_name: str) -> str | None:
        assert resource_name.startswith("kestrel-lora-")
        return self.find_result

    async def stop(self, pod_id: str, *, profile: GPUProfile) -> bool:
        assert profile.id == "training"
        self.stop_calls.append(pod_id)
        if self.stop_error:
            raise self.stop_error
        if self.stop_confirms:
            self.status = "EXITED"
        return self.stop_confirms


def training_service(
    tmp_path: Path,
    clock: MutableClock,
    provider: FakeTrainingProvider,
) -> TrainingPodLeaseService:
    return TrainingPodLeaseService(
        repository=SQLiteTrainingPodRepository(tmp_path / "training.sqlite3"),
        provider=provider,
        profiles={"training": training_profile()},
        poll_interval_seconds=10,
        orphan_timeout_seconds=30,
        clock=clock,
        sleep=clock.sleep,
    )
