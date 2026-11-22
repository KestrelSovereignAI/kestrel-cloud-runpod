"""Real SQLite tests for one registered endpoint and competing cache writers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from kestrel_cloud_runpod.ollama_contracts import OllamaLeaseConflictError
from kestrel_cloud_runpod.ollama_repository import SQLiteOllamaLeaseRepository
from kestrel_cloud_runpod.registered_ollama_journal import (
    RegisteredActivationIntent,
    RegisteredActivationState,
    RegisteredOllamaJournal,
    require_unreserved_registered_cache,
)
from kestrel_cloud_runpod.sqlite_store import lease_database_connection


@pytest.fixture
def authority(tmp_path):
    path = tmp_path / "leases.sqlite"
    SQLiteOllamaLeaseRepository(path)
    current = [datetime(2026, 11, 22, tzinfo=UTC)]

    def clock():
        return current[0]

    def advance(seconds):
        current[0] += timedelta(seconds=seconds)

    intent = RegisteredActivationIntent(
        "lease-original", "owner-original", "endpoint-one", "already-approved-cache",
        "a" * 64, "b" * 64, "0" * 64, "c" * 64, "d" * 64,
        clock() + timedelta(seconds=20), clock() + timedelta(seconds=120),
        clock() + timedelta(seconds=240), 15,
    )
    return path, clock, advance, intent


def test_restart_never_regrants_either_patch_or_unreserves_on_elapsed_time(authority):
    path, clock, advance, intent = authority
    journal = RegisteredOllamaJournal(path, clock=clock)
    arm, fresh = journal.reserve(intent)
    assert fresh and arm.state is RegisteredActivationState.ARMING
    assert arm.expires_at == intent.ready_before
    assert journal.due() == ()
    advance(130)
    restarted = RegisteredOllamaJournal(path, clock=clock)
    assert restarted.reserve(intent) == (arm, False)
    assert restarted.due() == (arm,)
    with pytest.raises(OllamaLeaseConflictError, match="already reserved"):
        restarted.reserve(replace(intent, lease_id="other", first_patch_before=clock() + timedelta(seconds=1),
                                  ready_before=clock() + timedelta(seconds=2)))
    with pytest.raises(OllamaLeaseConflictError, match="not confirmed"):
        restarted.claim_park(intent)
    restarted.record_active(intent)
    parking, claimed = restarted.claim_park(intent)
    assert claimed
    assert parking.state is RegisteredActivationState.PARKING
    restarted_again = RegisteredOllamaJournal(path, clock=clock)
    assert restarted_again.claim_park(intent) == (parking, False)
    assert restarted_again.due() == (parking,)
    parked = restarted_again.record_parked(intent)
    assert parked.state is RegisteredActivationState.PARKED
    assert restarted_again.due() == ()
    assert restarted_again.reserve(intent) == (parked, False)


def test_owner_admission_and_environment_digests_are_immutable(authority):
    path, clock, _advance, intent = authority
    journal = RegisteredOllamaJournal(path, clock=clock)
    journal.reserve(intent)
    for forged in (
        replace(intent, owner_id="foreign-owner"),
        replace(intent, admission_sha256="e" * 64),
        replace(intent, parked_environment_sha256="e" * 64),
    ):
        with pytest.raises(OllamaLeaseConflictError, match="changed its authority"):
            journal.reserve(forged)
        with pytest.raises(OllamaLeaseConflictError, match="original authority"):
            journal.record_active(forged)


def test_cold_deadline_and_idle_deadline_are_separate_and_hard_deadline_cannot_extend(authority):
    path, clock, advance, intent = authority
    journal = RegisteredOllamaJournal(path, clock=clock)
    journal.reserve(intent)
    advance(100)
    journal.record_active(intent)
    first = journal.touch(intent)
    assert first.expires_at == clock() + timedelta(seconds=intent.idle_seconds)
    advance(14)
    renewed = journal.touch(intent)
    assert renewed.expires_at > first.expires_at
    advance(16)
    with pytest.raises(OllamaLeaseConflictError, match="cannot be renewed"):
        journal.touch(intent)
    assert journal.due() == (renewed,)


def test_single_first_patch_claim_across_connections(authority):
    path, clock, _advance, intent = authority
    race = Barrier(2)
    first, second = (RegisteredOllamaJournal(path, clock=clock) for _ in range(2))

    def activate(journal):
        race.wait(timeout=5)
        return journal.reserve(intent)[1]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(activate, (first, second)))
    assert sorted(results) == [False, True]


def test_cache_reservation_blocks_the_existing_create_writer_until_exact_park(authority):
    path, clock, _advance, intent = authority
    journal = RegisteredOllamaJournal(path, clock=clock)
    journal.reserve(intent)
    with lease_database_connection(path) as db:
        db.execute("BEGIN IMMEDIATE")
        require_unreserved_registered_cache(db, frozenset({"other-cache"}))
        with pytest.raises(OllamaLeaseConflictError, match="reserved by a registered"):
            require_unreserved_registered_cache(db, frozenset({intent.cache_id}))
    journal.record_active(intent)
    journal.claim_park(intent)
    journal.record_parked(intent)
    assert journal.reserve(replace(
        intent, lease_id="next-lease",
        prior_environment_sha256=intent.parked_environment_sha256,
        parked_environment_sha256="e" * 64,
    ))[1]


def test_elapsed_offer_cannot_create_a_row(authority):
    path, clock, advance, intent = authority
    journal = RegisteredOllamaJournal(path, clock=clock)
    advance(21)
    with pytest.raises(OllamaLeaseConflictError, match="window has elapsed"):
        journal.reserve(intent)
    assert journal.get(intent.lease_id) is None
