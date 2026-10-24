"""Persistence, CAS, restart, and cross-process exclusion tests."""

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
from training_test_support import MutableClock, training_request

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodOwnership,
    TrainingPodRevisionConflictError,
    TrainingPodSource,
    TrainingPodState,
    fallback_training_cleanup_token,
    iso_datetime,
)
from kestrel_cloud_runpod.training_repository import (
    SQLiteTrainingPodRepository,
    training_database_path,
)


@pytest.mark.parametrize("door", ["trigger", "not-null"])
def test_cas_preserves_nonunique_native_constraint(tmp_path: Path, door: str) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    lease, _ = repository.reserve(training_request(clock))
    before = repository.get(lease.cleanup_token)
    if door == "trigger":
        with repository._connect() as connection:
            connection.execute(
                "CREATE TRIGGER r27_cas_block BEFORE UPDATE ON pod_capacity_leases "
                "BEGIN SELECT RAISE(ABORT, 'original custody refused'); END"
            )
        changes = {"stop_attempts": 1}
        expected_code = sqlite3.SQLITE_CONSTRAINT_TRIGGER
        expected_name = "SQLITE_CONSTRAINT_TRIGGER"
    else:
        changes = {"state": None}
        expected_code = sqlite3.SQLITE_CONSTRAINT_NOTNULL
        expected_name = "SQLITE_CONSTRAINT_NOTNULL"
    caught = None
    try:
        repository.compare_and_set(lease, changes=changes)
    except (sqlite3.IntegrityError, TrainingPodConflictError) as error:
        caught = error
    assert caught is not None
    native = caught if isinstance(caught, sqlite3.IntegrityError) else caught.__cause__
    assert isinstance(native, sqlite3.IntegrityError)
    assert type(native.sqlite_errorcode) is int
    assert native.sqlite_errorcode == expected_code
    assert native.sqlite_errorname == expected_name
    assert repository.get(lease.cleanup_token) == before
    assert caught is native


def test_cas_real_unique_remains_ownership_conflict(tmp_path: Path) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    first, _ = repository.reserve(
        training_request(clock, token="training:r27-first-0001", pod_id="pod-r27-first")
    )
    second, _ = repository.reserve(
        training_request(
            clock, token="training:r27-second-0002", pod_id="pod-r27-second"
        )
    )
    with pytest.raises(TrainingPodConflictError) as caught:
        repository.compare_and_set(
            first, changes={"provider_pod_id": second.provider_pod_id}
        )
    native = caught.value.__cause__
    assert isinstance(native, sqlite3.IntegrityError)
    assert native.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE
    assert native.sqlite_errorname == "SQLITE_CONSTRAINT_UNIQUE"
    assert repository.get(first.cleanup_token) == first
    assert repository.get(second.cleanup_token) == second


@pytest.mark.parametrize("changed", [False, True])
def test_exact_reservation_replay_preserves_fingerprint(
    tmp_path: Path, changed: bool
) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    request = training_request(clock, token="training:r27-reserve-0001")
    original, inserted = repository.reserve(request)
    assert inserted
    if changed:
        value = training_request(
            clock, token=request.cleanup_token, pod_id="pod-r27-changed"
        )
        with pytest.raises(TrainingPodConflictError, match="different request"):
            repository.reserve(value)
    else:
        repeated, inserted = repository.reserve(request)
        assert not inserted and repeated == original
    assert repository.get(request.cleanup_token) == original


def test_reserve_preserves_native_trigger_refusal(tmp_path: Path) -> None:
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    request = training_request(clock, token="training:r27-trigger-0001")
    with repository._connect() as connection:
        connection.execute(
            "CREATE TRIGGER r27_reserve_block BEFORE INSERT ON pod_capacity_leases "
            "BEGIN SELECT RAISE(ABORT, 'reserve custody refused'); END"
        )
    caught = None
    try:
        repository.reserve(request)
    except (sqlite3.IntegrityError, TrainingPodConflictError) as error:
        caught = error
    assert caught is not None
    native = caught if isinstance(caught, sqlite3.IntegrityError) else caught.__cause__
    assert isinstance(native, sqlite3.IntegrityError)
    assert native.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER
    assert native.sqlite_errorname == "SQLITE_CONSTRAINT_TRIGGER"
    assert repository.get(request.cleanup_token) is None
    assert caught is native


@pytest.mark.parametrize("door", ["reserve", "cas"])
@pytest.mark.parametrize(
    "evidence", ["denial", "grouped-unknown", "ordinary", "implicit", "missing-code"]
)
def test_native_unique_error_control_precedes_ownership_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    door: str,
    evidence: str,
    record_property,
) -> None:
    """Real SQLite failure; attached typed causes are unit evidence, not COMMIT."""
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        ExecutionCommitOutcomeError,
    )

    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "native-carrier.sqlite3")
    first, _ = repository.reserve(
        training_request(clock, token="training:r27-carrier-first", pod_id="pod-A")
    )
    if door == "cas":
        second, _ = repository.reserve(
            training_request(clock, token="training:r27-carrier-second", pod_id="pod-B")
        )
    original_connect = repository._connect
    observed = []
    explicit = {
        "denial": ExecutionAuthorityError("unit explicit denial"),
        "grouped-unknown": BaseExceptionGroup(
            "unit explicit sources",
            [
                ValueError("original ordinary unit cause"),
                ExecutionCommitOutcomeError("unknown"),
            ],
        ),
        "ordinary": ValueError("ordinary explicit unit cause"),
    }.get(evidence)
    implicit = ExecutionAuthorityError("unit implicit context is not authority")

    class NativeErrorObservation:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, *args, **kwargs):
            try:
                return self.connection.execute(*args, **kwargs)
            except sqlite3.IntegrityError as error:
                assert error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE
                observed.append(error)
                if evidence == "missing-code":
                    # Intentionally remove carrier metadata from the SAME real
                    # native error. Absence must not authorize normalization.
                    del error.sqlite_errorcode
                    del error.sqlite_errorname
                elif evidence == "implicit":
                    error.__context__ = implicit
                else:
                    error.__cause__ = explicit
                raise

    @contextmanager
    def observe_native_error():
        # The canonical connection owner still performs transaction rollback
        # and physical close. This view only observes the actual SQL error.
        with original_connect() as connection:
            yield NativeErrorObservation(connection)

    monkeypatch.setattr(repository, "_connect", observe_native_error)
    caught = None
    try:
        if door == "reserve":
            repository.reserve(
                training_request(
                    clock, token="training:r27-carrier-second", pod_id="pod-A"
                )
            )
        else:
            repository.compare_and_set(
                first, changes={"provider_pod_id": second.provider_pod_id}
            )
    except (sqlite3.IntegrityError, TrainingPodConflictError) as error:
        caught = error

    assert len(observed) == 1
    native = observed[0]
    assert repository.get(first.cleanup_token) == first
    if door == "reserve":
        assert repository.get("training:r27-carrier-second") is None
    else:
        assert repository.get(second.cleanup_token) == second
    if evidence in {"denial", "grouped-unknown", "missing-code"}:
        assert caught is native
    else:
        assert type(caught) is TrainingPodConflictError
        assert caught.__cause__ is native
    if evidence == "missing-code":
        assert not hasattr(native, "sqlite_errorcode")
        assert not hasattr(native, "sqlite_errorname")
    elif evidence == "implicit":
        assert native.__context__ is implicit and native.__cause__ is None
    else:
        assert native.__cause__ is explicit
    record_property(
        "r27_native_carrier",
        {
            "door": door,
            "evidence": evidence,
            "actual_native_unique_before_tag": True,
            "same_original_error": caught is native,
            "durable_rows_unchanged": True,
            "typed_sources_are_unit_not_observed_commit": True,
        },
    )


def test_reservation_survives_repository_restart(tmp_path: Path) -> None:
    clock = MutableClock()
    path = tmp_path / "state.sqlite3"
    lease, inserted = SQLiteTrainingPodRepository(path).reserve(training_request(clock))
    restarted = SQLiteTrainingPodRepository(path)
    assert inserted is True
    assert restarted.get(lease.cleanup_token) == lease


def test_revision_cas_rejects_stale_writer(tmp_path: Path) -> None:
    from kestrel_cloud_runpod import (
        PodCapacityRevisionConflictError,
    )
    from kestrel_cloud_runpod import (
        TrainingPodRevisionConflictError as PublicTrainingRevisionConflict,
    )

    assert PodCapacityRevisionConflictError is TrainingPodRevisionConflictError
    assert PublicTrainingRevisionConflict is TrainingPodRevisionConflictError
    assert issubclass(TrainingPodRevisionConflictError, TrainingPodConflictError)
    clock = MutableClock()
    repository = SQLiteTrainingPodRepository(tmp_path / "state.sqlite3")
    stale, _ = repository.reserve(training_request(clock))
    repository.compare_and_set(stale, changes={"state": TrainingPodState.STARTING})
    with pytest.raises(TrainingPodRevisionConflictError, match="concurrently"):
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
    with closing(sqlite3.connect(database)) as connection, connection:
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
    with closing(sqlite3.connect(database)) as connection, connection:
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
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

    with closing(sqlite3.connect(database)) as connection, connection:
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
