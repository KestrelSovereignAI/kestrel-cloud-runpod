"""Persistence, idempotency, and restart tests for Ollama leases."""

import sqlite3
from dataclasses import replace

import pytest
from ollama_test_support import MutableClock, make_request

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseConflictError,
    OllamaLeaseState,
)
from kestrel_cloud_runpod.ollama_repository import (
    SQLiteOllamaLeaseRepository,
    lease_database_path,
    request_from_lease,
)


def test_insert_is_idempotent_and_survives_repository_restart(tmp_path):
    clock = MutableClock()
    database = tmp_path / "leases.sqlite3"
    first_repository = SQLiteOllamaLeaseRepository(database)
    first, inserted = first_repository.insert_request(make_request(clock), now=clock())

    restarted_repository = SQLiteOllamaLeaseRepository(database)
    duplicate, duplicate_inserted = restarted_repository.insert_request(
        make_request(clock), now=clock()
    )

    assert inserted is True
    assert duplicate_inserted is False
    assert duplicate == first
    assert duplicate.state is OllamaLeaseState.REQUESTED
    assert request_from_lease(duplicate) == make_request(clock)


def test_repository_additively_upgrades_compute_only_lease_schema(tmp_path):
    database = tmp_path / "leases.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    added = (
        "estimated_compute_cost",
        "maximum_compute_cost",
        "estimated_non_compute_cost",
        "maximum_non_compute_cost",
        "cost_ceiling",
        "cost_policy_components_json",
        "maximum_concurrent_workers",
        "maximum_billable_seconds",
    )
    with sqlite3.connect(database) as connection:
        for name in added:
            connection.execute(f"ALTER TABLE ollama_leases DROP COLUMN {name}")

    SQLiteOllamaLeaseRepository(database)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ollama_leases)")
        }
    assert set(added).issubset(columns)


def test_reused_lease_id_with_changed_request_is_rejected(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    repository.insert_request(make_request(clock), now=clock())

    with pytest.raises(OllamaLeaseConflictError, match="different request"):
        repository.insert_request(
            make_request(clock, model="different:latest"), now=clock()
        )


def test_compare_and_set_rejects_stale_revision(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, _ = repository.insert_request(make_request(clock), now=clock())
    updated = repository.compare_and_set(
        lease, changes={"state": OllamaLeaseState.PROVISIONING}
    )

    with pytest.raises(OllamaLeaseConflictError, match="changed concurrently"):
        repository.compare_and_set(lease, changes={"state": OllamaLeaseState.FAILED})

    assert updated.revision == lease.revision + 1


@pytest.mark.parametrize(
    "constraints_json",
    ["{", "[]", '{"min_vram_gb": 24, "gpu_count": 1, "cloud": "SECURE"}'],
)
def test_request_recovery_rejects_corrupt_constraints(tmp_path, constraints_json):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, _ = repository.insert_request(make_request(clock), now=clock())

    with pytest.raises(RunPodManagerError, match="corrupt constraints"):
        request_from_lease(replace(lease, constraints_json=constraints_json))


def test_request_recovery_rejects_non_requested_state(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, _ = repository.insert_request(make_request(clock), now=clock())

    with pytest.raises(RunPodManagerError, match="not a recoverable request"):
        request_from_lease(replace(lease, state=OllamaLeaseState.PROVISIONING))


def test_database_path_requires_absolute_explicit_configuration(monkeypatch):
    monkeypatch.delenv("RUNPOD_OLLAMA_LEASE_DB", raising=False)

    with pytest.raises(RunPodManagerError, match="database_path"):
        lease_database_path({})
    with pytest.raises(RunPodManagerError, match="absolute"):
        lease_database_path({"database_path": "relative.sqlite3"})


def test_database_environment_expansion_fails_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_OLLAMA_ROOT", raising=False)

    with pytest.raises(RunPodManagerError, match="MISSING_OLLAMA_ROOT"):
        lease_database_path({"database_path": "${MISSING_OLLAMA_ROOT}/leases.sqlite3"})
