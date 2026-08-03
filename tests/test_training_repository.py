"""Persistence, CAS, restart, and cross-process exclusion tests."""

import sqlite3
from pathlib import Path

import pytest
from training_test_support import MutableClock, training_request

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodOwnership,
    TrainingPodSource,
    TrainingPodState,
    fallback_training_cleanup_token,
    iso_datetime,
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


def test_pre_family_database_migrates_with_legacy_attempt_self_rooted(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    root_token = "training:legacy-root-0001"
    child_token = fallback_training_cleanup_token(root_token, "training-h100")
    request = training_request(clock, token=child_token, profile_id="training-h100")
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE training_pod_leases (
                cleanup_token TEXT PRIMARY KEY,
                request_fingerprint TEXT NOT NULL,
                companion_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                source TEXT NOT NULL,
                resource_name TEXT NOT NULL,
                provider_pod_id TEXT,
                ownership TEXT NOT NULL,
                state TEXT NOT NULL,
                cleanup_state TEXT NOT NULL,
                creation_uncertain INTEGER NOT NULL DEFAULT 0,
                backend_base_url TEXT,
                provider_job_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_heartbeat_at TEXT NOT NULL,
                readiness_deadline TEXT NOT NULL,
                hard_deadline TEXT NOT NULL,
                last_provider_error TEXT,
                stop_attempts INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """INSERT INTO training_pod_leases (
                cleanup_token, request_fingerprint, companion_id, profile_id,
                source, resource_name, provider_pod_id, ownership, state,
                cleanup_state, created_at, updated_at, last_heartbeat_at,
                readiness_deadline, hard_deadline
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request.cleanup_token,
                request.fingerprint,
                request.companion_id,
                request.profile_id,
                request.source.value,
                request.resource_name,
                request.provider_pod_id,
                "owned",
                "ready",
                "not_requested",
                iso_datetime(request.created_at),
                iso_datetime(request.created_at),
                iso_datetime(request.created_at),
                iso_datetime(request.readiness_deadline),
                iso_datetime(request.hard_deadline),
            ),
        )

    repository = SQLiteTrainingPodRepository(database)
    migrated = repository.get(child_token)

    assert migrated is not None
    assert migrated.root_cleanup_token == child_token
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='training_pod_leases'"
            ).fetchone()
            is None
        )
        root_column = next(
            row
            for row in connection.execute("PRAGMA table_info(pod_capacity_leases)")
            if row[1] == "root_cleanup_token"
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert root_column[3] == 1
    family = repository.list_cleanup_family(
        root_token, legacy_cleanup_tokens=(child_token,)
    )
    assert family == (migrated,)


def test_family_release_gate_blocks_a_late_fallback_reservation(tmp_path: Path) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    root_request = training_request(clock, token="training:family-root-0001")
    root, _ = repository.reserve(root_request)
    root = repository.compare_and_set(
        root,
        changes={
            "state": TrainingPodState.RELEASED,
            "family_release_requested": True,
        },
    )
    child_token = fallback_training_cleanup_token(root.cleanup_token, "training-h100")

    with pytest.raises(TrainingPodConflictError, match="already releasing"):
        repository.reserve(
            training_request(
                clock,
                token=child_token,
                root_token=root.cleanup_token,
                profile_id="training-h100",
            )
        )


def test_versioned_table_migration_preserves_every_lifecycle_shape(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    database = tmp_path / "v1-state.sqlite3"
    repository = SQLiteTrainingPodRepository(database)
    expected = {}
    shapes = (
        ("requested", TrainingPodState.REQUESTED, False, None),
        ("uncertain", TrainingPodState.STARTING, True, None),
        ("ready", TrainingPodState.READY, False, "pod-ready"),
        ("releasing", TrainingPodState.RELEASING, False, "pod-releasing"),
        ("released", TrainingPodState.RELEASED, False, "pod-released"),
    )
    for name, state, uncertain, pod_id in shapes:
        source = (
            TrainingPodSource.CREATED
            if pod_id is None
            else TrainingPodSource.CONFIGURED_PERSISTENT
        )
        item = training_request(
            clock,
            token=f"training:migration-{name}-0001",
            source=source,
            pod_id=pod_id,
        )
        lease, _ = repository.reserve(item)
        changes = {
            "state": state,
            "creation_uncertain": uncertain,
            "ownership": (
                TrainingPodOwnership.PROVISIONAL
                if pod_id is None
                else TrainingPodOwnership.OWNED
            ),
        }
        if state is TrainingPodState.RELEASING:
            changes["cleanup_state"] = TrainingPodCleanupState.PENDING
        elif state is TrainingPodState.RELEASED:
            changes["cleanup_state"] = TrainingPodCleanupState.COMPLETE
        lease = repository.compare_and_set(lease, changes=changes)
        expected[lease.cleanup_token] = lease

    with sqlite3.connect(database) as connection:
        connection.execute(
            "ALTER TABLE pod_capacity_leases RENAME TO training_pod_leases"
        )
        connection.execute("PRAGMA user_version = 1")

    migrated = SQLiteTrainingPodRepository(database)
    for cleanup_token, lease in expected.items():
        restored = migrated.get(cleanup_token)
        assert restored is not None
        assert restored.state is lease.state
        assert restored.creation_uncertain is lease.creation_uncertain
        assert restored.provider_pod_id == lease.provider_pod_id
        assert restored.cleanup_state is lease.cleanup_state
