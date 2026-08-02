"""Persistence, CAS, restart, and cross-process exclusion tests."""

from pathlib import Path

import pytest
from training_test_support import MutableClock, training_request

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodConflictError,
    TrainingPodState,
)
from kestrel_cloud_runpod.training_repository import (
    SQLiteTrainingPodRepository,
    training_database_path,
)


def test_reservation_survives_repository_restart(tmp_path: Path) -> None:
    clock = MutableClock()
    path = tmp_path / "state.sqlite3"
    lease, inserted = SQLiteTrainingPodRepository(path).reserve(training_request(clock))
    restarted = SQLiteTrainingPodRepository(path)
    assert inserted is True
    assert restarted.get(lease.cleanup_token) == lease


def test_revision_cas_rejects_stale_writer(tmp_path: Path) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    stale, _ = repository.reserve(training_request(clock))
    repository.compare_and_set(stale, changes={"state": TrainingPodState.STARTING})
    with pytest.raises(TrainingPodConflictError, match="concurrently"):
        repository.compare_and_set(stale, changes={"state": TrainingPodState.READY})


def test_second_active_token_cannot_claim_same_pod(tmp_path: Path) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    repository.reserve(training_request(clock, token="training:first-token-0001"))
    with pytest.raises(TrainingPodConflictError, match="already claimed"):
        repository.reserve(training_request(clock, token="training:second-token-0002"))


def test_database_path_is_explicit_absolute_and_expands_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAINING_STATE_ROOT", str(tmp_path))
    assert training_database_path(
        {"database_path": "${TRAINING_STATE_ROOT}/training.sqlite3"}
    ) == (tmp_path / "training.sqlite3")
    with pytest.raises(RunPodManagerError, match="absolute"):
        training_database_path({"database_path": "relative.sqlite3"})
