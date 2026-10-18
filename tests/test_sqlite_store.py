"""Real SQLite connection ownership, transaction, and repository regressions."""

import sqlite3

import pytest
from ollama_test_support import MutableClock, make_request
from training_test_support import training_request

from kestrel_cloud_runpod.ollama_contracts import OllamaLeaseConflictError
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.pod_capacity_contracts import TrainingPodConflictError
from kestrel_cloud_runpod.pod_capacity_repository import SQLitePodCapacityRepository
from kestrel_cloud_runpod.sqlite_store import lease_database_connection


@pytest.fixture
def opened_connections(monkeypatch):
    original_connect = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        yield connections
    finally:
        for connection in connections:
            connection.close()


def _assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def test_connection_pragmas_commit_and_deterministic_close(
    tmp_path, opened_connections
):
    path = tmp_path / "leases.sqlite3"
    with lease_database_connection(path) as connection:
        assert connection.row_factory is sqlite3.Row
        assert connection.isolation_level is None
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        connection.execute("CREATE TABLE records (value INTEGER)")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO records VALUES (1)")
    _assert_closed(opened_connections)
    with lease_database_connection(path) as restarted:
        assert restarted.execute("SELECT value FROM records").fetchone()[0] == 1
    _assert_closed(opened_connections)


def test_failed_operation_rolls_back_and_closes(tmp_path, opened_connections):
    path = tmp_path / "leases.sqlite3"
    with lease_database_connection(path) as connection:
        connection.execute("CREATE TABLE records (value INTEGER UNIQUE)")
    with (
        pytest.raises(sqlite3.IntegrityError),
        lease_database_connection(path) as connection,
    ):
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO records VALUES (1)")
        connection.execute("INSERT INTO records VALUES (1)")
    _assert_closed(opened_connections)
    with lease_database_connection(path) as restarted:
        assert restarted.execute("SELECT count(*) FROM records").fetchone()[0] == 0
    _assert_closed(opened_connections)


def test_failed_commit_rolls_back_and_closes(tmp_path, opened_connections):
    path = tmp_path / "leases.sqlite3"
    with lease_database_connection(path) as connection:
        connection.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE children (parent_id INTEGER REFERENCES parents(id) "
            "DEFERRABLE INITIALLY DEFERRED)"
        )
    with (
        pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"),
        lease_database_connection(path) as connection,
    ):
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO children VALUES (1)")
    _assert_closed(opened_connections)
    with lease_database_connection(path) as restarted:
        assert restarted.execute("SELECT count(*) FROM children").fetchone()[0] == 0
    _assert_closed(opened_connections)


@pytest.mark.parametrize(
    "failed_pragma",
    [
        "PRAGMA foreign_keys = ON",
        "PRAGMA journal_mode = WAL",
        "PRAGMA busy_timeout = 30000",
    ],
)
def test_failed_configuration_closes_real_connection(
    tmp_path, monkeypatch, opened_connections, failed_pragma
):
    class ConfigurationFailure(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == failed_pragma:
                raise sqlite3.OperationalError("injected configuration failure")
            return super().execute(sql, parameters)

    captured_connect = sqlite3.connect

    def configured_connect(*args, **kwargs):
        return captured_connect(*args, factory=ConfigurationFailure, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", configured_connect)
    with (
        pytest.raises(sqlite3.OperationalError, match="configuration failure"),
        lease_database_connection(tmp_path / "leases.sqlite3"),
    ):
        pytest.fail("failed configuration yielded a connection")
    _assert_closed(opened_connections)


@pytest.mark.parametrize("kind", ["pod", "ollama"])
def test_repository_operations_and_failures_close_connections(
    tmp_path, opened_connections, kind
):
    clock = MutableClock()
    path = tmp_path / "leases.sqlite3"
    if kind == "pod":
        repository = SQLitePodCapacityRepository(path)
        request = training_request(clock)
        lease, inserted = repository.reserve(request)
        repeated, duplicate_inserted = repository.reserve(request)
        identifier = lease.cleanup_token
        conflict = TrainingPodConflictError
        assert repository.list_cleanup_family(identifier) == (lease,)
        assert repository.list_incomplete_family_releases() == ()
        table = "pod_capacity_leases"
    else:
        repository = SQLiteOllamaLeaseRepository(path)
        request = make_request(clock)
        lease, inserted = repository.insert_request(request, now=clock())
        repeated, duplicate_inserted = repository.insert_request(request, now=clock())
        identifier = lease.lease_id
        conflict = OllamaLeaseConflictError
        table = "ollama_leases"
    assert inserted and not duplicate_inserted and repeated == lease
    assert repository.get(identifier) == lease
    assert repository.get("missing-lease") is None
    assert repository.list_for_reconciliation() == (lease,)
    updated = repository.compare_and_set(lease, changes={"updated_at": clock()})
    assert updated.revision == lease.revision + 1
    with pytest.raises(conflict, match="concurrently"):
        repository.compare_and_set(lease, changes={"updated_at": clock()})
    _assert_closed(opened_connections)
    # Actual SQL failure after initialization must also relinquish ownership.
    with lease_database_connection(path) as connection:
        connection.execute(f"DROP TABLE {table}")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        repository.get(identifier)
    _assert_closed(opened_connections)


@pytest.mark.parametrize(
    "repository_class", [SQLitePodCapacityRepository, SQLiteOllamaLeaseRepository]
)
def test_repository_initialization_failure_closes_connection(
    tmp_path, opened_connections, repository_class
):
    path = tmp_path / "leases.sqlite3"
    with lease_database_connection(path) as connection:
        connection.execute("CREATE VIEW pod_capacity_leases AS SELECT 1")
        connection.execute("CREATE VIEW ollama_leases AS SELECT 1")
    with pytest.raises(sqlite3.OperationalError):
        repository_class(path)
    _assert_closed(opened_connections)
