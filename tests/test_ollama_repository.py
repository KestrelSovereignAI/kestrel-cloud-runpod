"""Persistence, idempotency, and restart tests for Ollama leases."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from multiprocessing import get_context

import pytest
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    make_request,
    pod_plan,
    serverless_plan,
)

from kestrel_cloud_runpod.inference_provider import _lease_metadata
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseConflictError,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaLifecycleBillingLatestReference,
    OllamaLifecycleBillingLatestState,
    OllamaLifecycleBillingObservation,
    OllamaLifecycleBillingSelection,
    OllamaLifecycleBillingState,
    OllamaProvisioningEvidenceState,
    OllamaProvisioningMismatchEvidence,
    OllamaProvisioningMismatchReason,
    OllamaRuntimeTelemetryObservation,
    OllamaRuntimeTelemetryState,
    OllamaTeardownState,
    ProvisionedOllamaResource,
    normalize_new_ollama_request_money,
)
from kestrel_cloud_runpod.ollama_repository import (
    OllamaWriterDrainAttestation,
    SQLiteOllamaLeaseRepository,
    _db_value,
    lease_database_path,
    request_from_lease,
)
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService
from kestrel_cloud_runpod.pod_capacity_contracts import PodBillingReceipt

_WRITER_TRIGGERS = (
    "ollama_writer_insert_v6",
    "ollama_writer_update_v6",
    "ollama_writer_delete_v6",
    "ollama_provisioning_evidence_insert_v6",
    "ollama_provisioning_evidence_update_v6",
    "ollama_lifecycle_evidence_update_v6",
)
_WRITER_COLUMNS = (
    "ambiguous_create_candidate_id",
    "original_requested_at",
    "original_readiness_timeout_seconds",
    "provisioning_mismatch_json",
    "ambiguous_candidate_mismatch_json",
    "provisioning_receipt_json",
    "provisioning_intent_json",
    "provisioning_evidence_state",
    "writer_epoch",
    "writer_contract_version",
    "runtime_telemetry_binding_sha256",
    "cleanup_intent_json",
    "cleanup_receipt_json",
    "runtime_telemetry_state",
    "runtime_telemetry_observations_json",
    "lifecycle_billing_state",
    "lifecycle_billing_observations_json",
    "lifecycle_billing_selection_json",
    "lifecycle_billing_corrections_json",
    "lifecycle_billing_latest_reference_json",
    "billing_last_checked_at",
    "billing_next_check_at",
    "billing_last_provider_error",
    "reconcile_after",
    "additional_cleanup_targets_json",
    "runtime_telemetry_last_provider_error",
    "billing_retention_exhausted",
)
_V5_ONLY_WRITER_COLUMNS = ("lifecycle_billing_latest_reference_json",)
_V4_ONLY_WRITER_COLUMNS = (
    "reconcile_after",
    "additional_cleanup_targets_json",
    "runtime_telemetry_last_provider_error",
    "billing_retention_exhausted",
)
_V3_ONLY_WRITER_COLUMNS = (
    "runtime_telemetry_binding_sha256",
    "cleanup_intent_json",
    "cleanup_receipt_json",
    "runtime_telemetry_state",
    "runtime_telemetry_observations_json",
    "lifecycle_billing_state",
    "lifecycle_billing_observations_json",
    "lifecycle_billing_selection_json",
    "lifecycle_billing_corrections_json",
    "billing_last_checked_at",
    "billing_next_check_at",
    "billing_last_provider_error",
)


def _rewrite_writer_contract_default(connection, version):
    """Give a synthetic predecessor its genuine versioned column default."""

    table_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'ollama_leases'"
    ).fetchone()[0]
    default_prefix = "writer_contract_version INTEGER NOT NULL DEFAULT "
    default_start = table_sql.find(default_prefix)
    if default_start < 0:
        raise AssertionError("synthetic predecessor has an unexpected writer column")
    value_start = default_start + len(default_prefix)
    value_end = value_start
    while value_end < len(table_sql) and table_sql[value_end].isdigit():
        value_end += 1
    if value_end == value_start:
        raise AssertionError("synthetic predecessor has an invalid writer default")
    index_sql = tuple(
        row[0]
        for row in connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'ollama_leases' "
            "AND sql IS NOT NULL ORDER BY name"
        )
    )
    for row in connection.execute("PRAGMA index_list(ollama_leases)").fetchall():
        if row[3] == "c":
            connection.execute(f"DROP INDEX {row[1]}")
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_xinfo(ollama_leases)")
    )
    names_sql = ", ".join(columns)
    connection.execute("ALTER TABLE ollama_leases RENAME TO _synthetic_predecessor")
    connection.execute(table_sql[:value_start] + str(version) + table_sql[value_end:])
    connection.execute(
        f"INSERT INTO ollama_leases ({names_sql}) "
        f"SELECT {names_sql} FROM _synthetic_predecessor"
    )
    connection.execute("DROP TABLE _synthetic_predecessor")
    for statement in index_sql:
        connection.execute(statement)


def _legacy_database(database, *, state="requested", saved_route=False):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(database)
    repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        if state != "requested":
            connection.execute(
                """UPDATE ollama_leases
                   SET state = ?, resource_type = 'serverless_endpoint',
                       provider_resource_id = 'historical-endpoint',
                       resource_name = 'historical-resource', route_url = ?""",
                (state, "https://historical.invalid" if saved_route else None),
            )
        connection.execute("DROP INDEX idx_ollama_leases_reconcile")
        for column in _WRITER_COLUMNS:
            connection.execute(f"ALTER TABLE ollama_leases DROP COLUMN {column}")
        connection.execute(
            "CREATE INDEX idx_ollama_leases_reconcile "
            "ON ollama_leases(state, idle_deadline, hard_deadline)"
        )
    return clock


def _downgrade_to_v1_schema(connection):
    _downgrade_to_v2_schema(connection)
    connection.execute(
        "ALTER TABLE ollama_leases DROP COLUMN ambiguous_candidate_mismatch_json"
    )
    connection.execute("UPDATE ollama_leases SET writer_contract_version = 1")
    _rewrite_writer_contract_default(connection, 1)


def _downgrade_to_v2_schema(connection):
    connection.execute("DROP INDEX idx_ollama_leases_reconcile")
    for column in (
        *_V5_ONLY_WRITER_COLUMNS,
        *_V4_ONLY_WRITER_COLUMNS,
        *_V3_ONLY_WRITER_COLUMNS,
    ):
        connection.execute(f"ALTER TABLE ollama_leases DROP COLUMN {column}")
    connection.execute(
        "CREATE INDEX idx_ollama_leases_reconcile "
        "ON ollama_leases(state, idle_deadline, hard_deadline)"
    )
    connection.execute("UPDATE ollama_leases SET writer_contract_version = 2")
    _rewrite_writer_contract_default(connection, 2)


def _downgrade_to_v3_schema(connection):
    connection.execute("DROP INDEX idx_ollama_leases_reconcile")
    connection.execute("UPDATE ollama_leases SET writer_contract_version = 3")
    for column in (*_V5_ONLY_WRITER_COLUMNS, *_V4_ONLY_WRITER_COLUMNS):
        connection.execute(f"ALTER TABLE ollama_leases DROP COLUMN {column}")
    connection.execute(
        "CREATE INDEX idx_ollama_leases_reconcile "
        "ON ollama_leases(state, lifecycle_billing_state, billing_next_check_at, "
        "idle_deadline, hard_deadline)"
    )
    _rewrite_writer_contract_default(connection, 3)


def _downgrade_to_v4_writer(connection):
    """Create a synthetic v4 schema while retaining row facts."""

    for trigger in _WRITER_TRIGGERS:
        connection.execute(f"DROP TRIGGER {trigger}")
    connection.execute("UPDATE ollama_leases SET writer_contract_version = 4")
    connection.execute(
        "ALTER TABLE ollama_leases DROP COLUMN lifecycle_billing_latest_reference_json"
    )
    _rewrite_writer_contract_default(connection, 4)


def _downgrade_to_v5_schema(connection):
    """Create the exact v5 column/row shape for current-migrator unit tests.

    Installed old-writer integration tests separately prove the genuine v5
    triggers and runtime fence. This helper intentionally does not pretend to
    reproduce those historical trigger bytes.
    """

    for trigger in _WRITER_TRIGGERS:
        connection.execute(f"DROP TRIGGER {trigger}")
    connection.execute("UPDATE ollama_leases SET writer_contract_version = 5")
    connection.execute(
        "ALTER TABLE ollama_leases DROP COLUMN lifecycle_billing_latest_reference_json"
    )
    _rewrite_writer_contract_default(connection, 5)


def _drain_attestation(database):
    return OllamaWriterDrainAttestation(
        database_path=database,
        drained_at=datetime(2026, 10, 20, tzinfo=UTC),
        drain_reference_sha256="a" * 64,
    )


def _replace_with_permissive_v1_contract_fixture(database):
    """Create a representative weak v1 schema for current-migrator tests."""

    old_triggers = (
        "ollama_writer_insert_v1",
        "ollama_writer_update_v1",
        "ollama_writer_delete_v1",
        "ollama_provisioning_evidence_insert_v1",
        "ollama_provisioning_evidence_update_v1",
    )
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        _downgrade_to_v1_schema(connection)
        for trigger in old_triggers:
            event = (
                "INSERT"
                if "insert" in trigger
                else "DELETE"
                if "delete" in trigger
                else "UPDATE"
            )
            connection.execute(
                f"""CREATE TRIGGER {trigger}
                    BEFORE {event} ON ollama_leases
                    BEGIN SELECT 1; END"""
            )


def _database_files(database):
    return {
        path.name: path.read_bytes()
        for path in database.parent.glob(database.name + "*")
    }


def _persistent_database_files(database):
    return {
        path.name: path.read_bytes()
        for path in database.parent.glob(database.name + "*")
        if not path.name.endswith("-shm")
    }


def _migrate_legacy_database_process(database, gate, results):
    gate.wait(timeout=10)
    try:
        repository = SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )
        lease = repository.get("lease-001")
        results.put(
            (
                "ok",
                lease.writer_contract_version,
                lease.provisioning_evidence_state.value,
            )
        )
    except (RunPodManagerError, sqlite3.Error, ValueError) as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def _claim_reconciliation_process(database, gate, results, now, retry_at):
    repository = SQLiteOllamaLeaseRepository(database)
    lease = repository.get("lease-001")
    assert lease is not None
    gate.wait(timeout=10)
    claimed = repository.claim_for_reconciliation(
        lease,
        now=now,
        retry_at=retry_at,
    )
    results.put(None if claimed is None else claimed.revision)


async def _terminated_pod_billing_evidence(
    tmp_path, *, actual=Decimal(1), database_name="selected-billing.sqlite3"
):
    clock = MutableClock()
    database = tmp_path / database_name
    repository = SQLiteOllamaLeaseRepository(database)
    provider = FakeOllamaProvider(pod_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    request = make_request(clock)
    ready = await service.acquire(request, plan=pod_plan())
    terminated = await service.release(
        ready.lease_id,
        owner_id=ready.owner_id,
        workload_id=ready.workload_id,
    )
    intent = terminated.provisioning_intent
    provisioning = terminated.provisioning_receipt
    cleanup_intent = terminated.cleanup_intent
    cleanup = terminated.cleanup_receipt
    assert intent is not None
    assert provisioning is not None
    assert provisioning.pod_realized_placement is not None
    assert cleanup_intent is not None
    assert cleanup is not None
    billed_from = provisioning.provider_created_at.replace(
        minute=0, second=0, microsecond=0
    )
    billed_until = billed_from + timedelta(hours=1)
    observed_at = max(billed_until, cleanup.verified_absent_at) + timedelta(hours=1)
    receipt = PodBillingReceipt(
        provider_billing_id="pod-billing-original",
        provider_pod_id=provisioning.provider_resource_id,
        billed_from=billed_from,
        billed_until=billed_until,
        billed_seconds=3600,
        hourly_price_usd=(provisioning.pod_realized_placement.hourly_rate_usd),
        actual_cost_usd=actual,
        reconciled_at=observed_at,
    )
    observation = OllamaLifecycleBillingObservation(
        intent_sha256=intent.intent_sha256,
        provisioning_receipt_id=provisioning.receipt_id,
        cleanup_receipt_id=cleanup.cleanup_receipt_id,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        provider_resource_id=provisioning.provider_resource_id,
        cost_scope="pod_lifecycle",
        provider_capacity_actual_cost_usd=actual,
        first_observed_at=observed_at,
        pod_billing_receipt=receipt,
    )
    return repository, terminated, observed_at, observation


async def _selected_pod_billing_lease(
    tmp_path, *, actual=Decimal(1), database_name="selected-billing.sqlite3"
):
    (
        repository,
        terminated,
        observed_at,
        observation,
    ) = await _terminated_pod_billing_evidence(
        tmp_path, actual=actual, database_name=database_name
    )
    selected = repository.record_billing_outcome(
        terminated,
        now=observed_at,
        next_check_at=observed_at + timedelta(hours=1),
        observation=observation,
        maximum_observations=8,
        maximum_bytes=100_000,
    )
    assert (
        selected.lifecycle_billing_latest_reference.state
        is OllamaLifecycleBillingLatestState.KNOWN
    )
    assert (
        selected.lifecycle_billing_latest_reference.observation_id
        == observation.observation_id
    )
    assert selected.lifecycle_billing_latest_reference.received_at == observed_at
    return repository, selected, observed_at


def _revised_telemetry(
    original: OllamaRuntimeTelemetryObservation,
    **snapshot_changes,
) -> OllamaRuntimeTelemetryObservation:
    snapshot = replace(original.snapshot, **snapshot_changes)
    digest = hashlib.sha256(
        json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return replace(
        original,
        snapshot=snapshot,
        snapshot_sha256=digest,
        observed_at=max(original.observed_at, snapshot.container_started_at),
    )


@contextmanager
def _live_wal_connection(database):
    with closing(sqlite3.connect(database, isolation_level=None)) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        yield connection


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


async def _cleanup_claim_with_late_target(tmp_path, *, database_name):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / database_name)
    provider = FakeOllamaProvider(serverless_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(make_request(clock), plan=serverless_plan())
    releasing = repository.compare_and_set(
        ready,
        changes={
            "state": OllamaLeaseState.RELEASING,
            "teardown_state": OllamaTeardownState.PENDING,
            "route_url": None,
            "termination_reason": "test cleanup claim",
        },
    )
    stale = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test cleanup claim",
    )
    assert stale.provisioning_intent is not None
    current = repository.attach_provider_acknowledgment(
        stale,
        ProvisionedOllamaResource(
            resource_type=stale.provisioning_intent.resource_type,
            provider_resource_id="late-original-ack",
            resource_name=stale.provisioning_intent.resource_name,
        ),
        now=clock(),
    )
    assert len(current.additional_cleanup_targets) == 1
    return clock, repository, stale, current


@pytest.mark.asyncio
async def test_cleanup_claim_folds_latest_same_lineage_owned_target(tmp_path):
    clock, repository, stale, current = await _cleanup_claim_with_late_target(
        tmp_path, database_name="cleanup-claim-fold.sqlite3"
    )
    claimed = repository.claim_cleanup_attempt(stale, now=clock())
    assert claimed.additional_cleanup_targets == current.additional_cleanup_targets
    latest = repository.latest_cleanup_lineage(stale)
    assert latest.additional_cleanup_targets == current.additional_cleanup_targets


@pytest.mark.asyncio
async def test_cleanup_claim_rejects_every_changed_lineage(tmp_path):
    clock, repository, stale, current = await _cleanup_claim_with_late_target(
        tmp_path, database_name="cleanup-claim-changed.sqlite3"
    )
    target = current.additional_cleanup_targets[0]
    changed_lineages = (
        replace(stale, request_fingerprint="f" * 64),
        replace(stale, provision_attempt_id="changed-attempt"),
        replace(
            stale,
            provisioning_intent=replace(
                stale.provisioning_intent,
                accepted_whole_hourly_rate_usd=Decimal("1.0"),
            ),
        ),
        replace(stale, provider_resource_id="foreign-primary-resource"),
        replace(
            stale,
            cleanup_intent=replace(
                stale.cleanup_intent,
                termination_reason="changed cleanup lineage",
            ),
        ),
        replace(
            stale,
            additional_cleanup_targets=(
                replace(target, provider_resource_id="foreign-resource"),
            ),
        ),
        replace(
            stale,
            additional_cleanup_targets=(
                replace(target, resource_name="changed-resource-name"),
            ),
        ),
    )
    for changed in changed_lineages:
        with pytest.raises(OllamaLeaseConflictError, match="cleanup lineage"):
            repository.claim_cleanup_attempt(changed, now=clock())
        with pytest.raises(OllamaLeaseConflictError, match="cleanup lineage"):
            repository.latest_cleanup_lineage(changed)


def test_two_repositories_cannot_claim_one_reconciliation_visit(tmp_path):
    clock = MutableClock()
    database = tmp_path / "leases.sqlite3"
    first = SQLiteOllamaLeaseRepository(database)
    lease, _ = first.insert_request(make_request(clock), now=clock())
    second = SQLiteOllamaLeaseRepository(database)
    same_revision = second.get(lease.lease_id)
    assert same_revision is not None

    claimed = first.claim_for_reconciliation(
        lease,
        now=clock(),
        retry_at=clock() + timedelta(seconds=1),
    )
    lost = second.claim_for_reconciliation(
        same_revision,
        now=clock(),
        retry_at=clock() + timedelta(seconds=1),
    )

    assert claimed is not None
    assert claimed.reconcile_after == clock() + timedelta(seconds=1)
    assert lost is None


def test_two_processes_cannot_claim_one_reconciliation_visit(tmp_path):
    clock = MutableClock()
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    repository.insert_request(make_request(clock), now=clock())
    context = get_context("spawn")
    gate = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_claim_reconciliation_process,
            args=(
                database,
                gate,
                results,
                clock(),
                clock() + timedelta(seconds=1),
            ),
        )
        for _index in range(2)
    ]
    for process in processes:
        process.start()
    gate.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0

    observed = [results.get(timeout=2) for _process in processes]
    results.close()
    results.join_thread()
    assert observed.count(None) == 1
    assert len([revision for revision in observed if revision is not None]) == 1


def test_runtime_refuses_incomplete_current_schema_before_any_ddl(tmp_path):
    database = tmp_path / "leases.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("ALTER TABLE ollama_leases DROP COLUMN maximum_compute_cost")
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


def test_attested_migration_repairs_supported_incomplete_current_schema(tmp_path):
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())
    current = SQLiteOllamaLeaseRepository(database).get("lease-001")
    assert current is not None
    assert current.writer_contract_version == 6
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("ALTER TABLE ollama_leases DROP COLUMN maximum_compute_cost")

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )

    lease = migrated.get("lease-001")
    assert lease.provisioning_evidence_state is OllamaProvisioningEvidenceState.REQUIRED
    assert lease.writer_contract_version == 6


@pytest.mark.asyncio
async def test_attested_v6_trigger_repair_preserves_every_current_fact(tmp_path):
    repository, selected, _observed_at = await _selected_pod_billing_lease(tmp_path)
    before = selected.to_dict()
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        connection.execute("DROP TRIGGER ollama_lifecycle_evidence_update_v6")

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        repository.path,
        attestation=_drain_attestation(repository.path),
    )
    retained = migrated.get(selected.lease_id)

    assert retained is not None
    assert retained.to_dict() == before
    assert retained.revision == selected.revision
    assert retained.writer_epoch == selected.writer_epoch
    assert retained.lifecycle_billing_latest_reference == (
        selected.lifecycle_billing_latest_reference
    )


def test_attested_migration_preserves_previous_writer_contract_authority(tmp_path):
    database = tmp_path / "previous-writer.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    original, _ = repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        _downgrade_to_v1_schema(connection)

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    retained = migrated.get(original.lease_id)

    assert retained.request_fingerprint == original.request_fingerprint
    assert (
        retained.provisioning_evidence_state is OllamaProvisioningEvidenceState.REQUIRED
    )
    assert retained.writer_contract_version == 6
    assert retained.ambiguous_candidate_mismatch is None


def test_attested_v2_to_v6_migration_retains_authority_but_not_cost_proof(tmp_path):
    database = tmp_path / "v2-writer.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    original, _ = repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("UPDATE ollama_leases SET writer_contract_version = 2")
        _downgrade_to_v2_schema(connection)

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    retained = migrated.get(original.lease_id)

    assert retained.request_fingerprint == original.request_fingerprint
    assert retained.writer_contract_version == 6
    assert retained.runtime_telemetry_state is OllamaRuntimeTelemetryState.UNKNOWN
    assert retained.runtime_telemetry_observations == ()
    assert retained.lifecycle_billing_state is OllamaLifecycleBillingState.UNKNOWN
    assert retained.lifecycle_billing_observations == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer_version", "downgrade"),
    ((4, _downgrade_to_v4_writer), (5, _downgrade_to_v5_schema)),
)
async def test_v4_or_v5_to_v6_migration_preserves_every_lifecycle_fact(
    tmp_path, writer_version, downgrade
):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path,
        actual=Decimal("0.12345678901234567890123456789"),
        database_name=f"v{writer_version}-to-v6.sqlite3",
    )
    original = selected.lifecycle_billing_observations[0]
    assert original.pod_billing_receipt is not None
    revised = replace(
        original,
        provider_capacity_actual_cost_usd=Decimal("0.2"),
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=replace(
            original.pod_billing_receipt,
            provider_billing_id="pod-billing-v5-correction",
            actual_cost_usd=Decimal("0.2"),
            reconciled_at=observed_at + timedelta(hours=1),
        ),
    )
    corrected = repository.record_billing_outcome(
        selected,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=revised,
        maximum_observations=8,
        maximum_bytes=100_000,
    )
    overflow = replace(
        revised,
        provider_capacity_actual_cost_usd=Decimal("0.3"),
        first_observed_at=observed_at + timedelta(hours=2),
        pod_billing_receipt=replace(
            revised.pod_billing_receipt,
            provider_billing_id="pod-billing-v5-retention-overflow",
            actual_cost_usd=Decimal("0.3"),
            reconciled_at=observed_at + timedelta(hours=2),
        ),
    )
    bounded = repository.record_billing_outcome(
        corrected,
        now=observed_at + timedelta(hours=2),
        next_check_at=observed_at + timedelta(hours=3),
        observation=overflow,
        maximum_observations=2,
        maximum_bytes=100_000,
    )
    assert len(bounded.lifecycle_billing_corrections) == 1
    assert bounded.billing_retention_exhausted is True

    before = bounded.to_dict()
    before_revision = bounded.revision
    before_epoch = bounded.writer_epoch
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        downgrade(connection)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(repository.path)

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        repository.path,
        attestation=_drain_attestation(repository.path),
    )
    retained = migrated.get(bounded.lease_id)
    assert retained is not None
    after = retained.to_dict()
    for field in (
        "writer_contract_version",
        "writer_epoch",
        "revision",
        "lifecycle_billing_latest_observation_id",
        "lifecycle_billing_latest_reference_state",
        "lifecycle_billing_latest_received_at",
    ):
        before.pop(field, None)
        after.pop(field, None)
    assert after == before
    assert retained.writer_contract_version == 6
    assert retained.writer_epoch == before_epoch + 1
    assert retained.revision == before_revision + 1
    assert retained.lifecycle_billing_corrections == (
        bounded.lifecycle_billing_corrections
    )
    assert (
        retained.lifecycle_billing_latest_reference.state
        is OllamaLifecycleBillingLatestState.HISTORICAL_UNKNOWN
    )
    assert retained.lifecycle_billing_latest_reference.observation_id is None
    assert retained.billing_retention_exhausted is True
    retained_public = retained.to_dict()
    retained_metadata = _lease_metadata(retained)
    assert retained_public["lifecycle_billing_latest_reference_state"] == (
        "historical_unknown"
    )
    assert retained_public["lifecycle_billing_latest_observation_id"] is None
    assert retained_public["lifecycle_billing_latest_received_at"] is None
    assert retained_metadata["lifecycle_billing_latest_reference_state"] == (
        "historical_unknown"
    )
    assert retained_metadata["lifecycle_billing_latest_observation_id"] is None
    assert retained_metadata["lifecycle_billing_latest_received_at"] is None


def test_v3_requires_drain_then_migrates_to_v6_without_restamping_authority(
    tmp_path,
):
    database = tmp_path / "v3-writer.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    original, _ = repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        _downgrade_to_v3_schema(connection)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)
    assert _database_files(database) == before

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    retained = migrated.get(original.lease_id)

    assert retained is not None
    assert retained.writer_contract_version == 6
    assert retained.provisioning_evidence_state is original.provisioning_evidence_state
    assert retained.reconcile_after == original.updated_at
    assert retained.additional_cleanup_targets == ()
    assert retained.runtime_telemetry_last_provider_error is None
    assert retained.billing_retention_exhausted is False


@pytest.mark.asyncio
async def test_v3_retention_sentinel_migrates_to_explicit_v6_retention_state(
    tmp_path,
) -> None:
    """Translate the exact v3 bounded-retention sentinel, not arbitrary errors."""

    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path, actual=Decimal("0.1")
    )
    original = selected.lifecycle_billing_observations[0]
    revised = replace(
        original,
        provider_capacity_actual_cost_usd=Decimal("0.2"),
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=replace(
            original.pod_billing_receipt,
            provider_billing_id="pod-billing-v3-retention-revision",
            actual_cost_usd=Decimal("0.2"),
        ),
    )
    bounded = repository.record_billing_outcome(
        selected,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=revised,
        maximum_observations=1,
        maximum_bytes=100_000,
    )
    assert bounded.lifecycle_billing_state is (
        OllamaLifecycleBillingState.CORRECTION_PENDING
    )
    assert bounded.lifecycle_billing_corrections == ()
    assert bounded.billing_retention_exhausted is True

    database = repository.path
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        _downgrade_to_v3_schema(connection)

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    retained = migrated.get(selected.lease_id)
    assert retained is not None
    assert retained.writer_contract_version == 6
    assert retained.lifecycle_billing_state is (
        OllamaLifecycleBillingState.CORRECTION_PENDING
    )
    assert retained.lifecycle_billing_selection == selected.lifecycle_billing_selection
    assert retained.lifecycle_billing_corrections == ()
    assert retained.billing_retention_exhausted is True


@pytest.mark.asyncio
async def test_v3_arbitrary_billing_error_is_not_promoted_to_retention_proof(
    tmp_path,
) -> None:
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path,
        actual=Decimal("0.1"),
        database_name="invalid-v3-retention.sqlite3",
    )
    original = selected.lifecycle_billing_observations[0]
    bounded = repository.record_billing_outcome(
        selected,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=replace(
            original,
            provider_capacity_actual_cost_usd=Decimal("0.2"),
            first_observed_at=observed_at + timedelta(hours=1),
            pod_billing_receipt=replace(
                original.pod_billing_receipt,
                provider_billing_id="pod-billing-invalid-v3-sentinel",
                actual_cost_usd=Decimal("0.2"),
            ),
        ),
        maximum_observations=1,
        maximum_bytes=100_000,
    )
    assert bounded.billing_retention_exhausted is True

    database = repository.path
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        _downgrade_to_v3_schema(connection)
        connection.execute(
            "UPDATE ollama_leases SET billing_last_provider_error = ?",
            ("another manual billing condition",),
        )

    with pytest.raises(RunPodManagerError, match="previous writer evidence"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )


def test_v1_writer_contract_requires_real_v6_drain_and_cannot_regain_authority(
    tmp_path,
):
    from test_ollama_service import _persist_uncertain_creation

    database = tmp_path / "v1-writer.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    request = make_request(clock)
    pending = _persist_uncertain_creation(repository, request, clock)
    intent = pending.provisioning_intent
    assert intent is not None
    mismatch = OllamaProvisioningMismatchEvidence(
        intent_sha256=intent.intent_sha256,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        provider_resource_id="owned-mismatch",
        observed_resource_name="wrong-runtime",
        observed_at=clock(),
        readback_sha256="d" * 64,
        reason=OllamaProvisioningMismatchReason.RUNTIME,
    )
    current_row = repository.record_provisioning_outcome(
        pending,
        now=clock(),
        mismatch=mismatch,
    )
    assert current_row.state is OllamaLeaseState.RELEASING
    _replace_with_permissive_v1_contract_fixture(database)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before
    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    retained = migrated.get(request.lease_id)
    assert retained.writer_contract_version == 6
    assert retained.provisioning_mismatch == mismatch
    assert retained.provider_resource_id == "owned-mismatch"
    assert retained.state is OllamaLeaseState.RELEASING

    with closing(sqlite3.connect(database)) as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="stale Ollama writer|invalid Ollama provisioning evidence mutation",
        ):
            connection.execute(
                """UPDATE ollama_leases
                   SET writer_contract_version = 1,
                       provisioning_mismatch_json = NULL,
                       state = 'ready',
                       route_url = 'https://stale.invalid',
                       revision = revision + 1,
                       writer_epoch = writer_epoch + 1
                   WHERE lease_id = ?""",
                (request.lease_id,),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="invalid Ollama provisioning evidence mutation",
        ):
            connection.execute(
                """UPDATE ollama_leases
                   SET provisioning_mismatch_json = NULL,
                       state = 'ready',
                       route_url = 'https://stale.invalid',
                       revision = revision + 1,
                       writer_epoch = writer_epoch + 1
                   WHERE lease_id = ?""",
                (request.lease_id,),
            )


def test_v1_drain_attestation_cannot_target_the_v6_contract(tmp_path):
    database = tmp_path / "wrong-target.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    repository.insert_request(make_request(MutableClock()), now=MutableClock()())
    before = _database_files(database)

    with pytest.raises(ValueError, match="Unsupported Ollama writer contract"):
        replace(_drain_attestation(database), target_writer_contract_version=1)

    assert _database_files(database) == before


def test_v6_runtime_rejects_unknown_future_writer_contract_without_repair(tmp_path):
    database = tmp_path / "future-writer.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        trigger_sql = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'ollama_leases' "
                "ORDER BY name"
            )
        )
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("UPDATE ollama_leases SET writer_contract_version = 99")
        for statement in trigger_sql:
            connection.execute(statement)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)
    assert _database_files(database) == before
    with pytest.raises(RunPodManagerError, match="not a supported"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )
    assert _database_files(database) == before


def test_migration_rejects_writer_version_with_impossible_column_contract(tmp_path):
    database = tmp_path / "impossible-v5-columns.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        trigger_sql = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'ollama_leases' "
                "ORDER BY name"
            )
        )
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("UPDATE ollama_leases SET writer_contract_version = 5")
        for statement in trigger_sql:
            connection.execute(statement)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="not a supported"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )

    assert _database_files(database) == before


def test_migration_rejects_mixed_previous_writer_versions(tmp_path):
    database = tmp_path / "mixed-previous-writers.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())
    repository.insert_request(
        replace(make_request(clock), lease_id="lease-002", workload_id="workload-002"),
        now=clock(),
    )
    with closing(sqlite3.connect(database)) as connection, connection:
        _downgrade_to_v5_schema(connection)
        connection.execute(
            "UPDATE ollama_leases SET writer_contract_version = 4 "
            "WHERE lease_id = 'lease-002'"
        )
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="not a supported"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )

    assert _database_files(database) == before


def test_attested_migration_separates_previous_untrusted_candidate_mismatch(
    tmp_path,
):
    from test_ollama_service import _persist_uncertain_creation

    database = tmp_path / "previous-candidate-mismatch.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    request = make_request(clock)
    original = _persist_uncertain_creation(repository, request, clock)
    intent = original.provisioning_intent
    assert intent is not None
    mismatch = OllamaProvisioningMismatchEvidence(
        intent_sha256=intent.intent_sha256,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        provider_resource_id="untrusted-candidate",
        observed_resource_name=intent.resource_name,
        observed_at=clock(),
        readback_sha256="c" * 64,
        reason=OllamaProvisioningMismatchReason.RUNTIME,
    )
    encoded = json.dumps(mismatch.to_dict(), sort_keys=True, separators=(",", ":"))
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        _downgrade_to_v1_schema(connection)
        connection.execute(
            """UPDATE ollama_leases
               SET ambiguous_create_candidate_id = ?,
                   provisioning_mismatch_json = ?,
                   revision = revision + 1,
                   writer_epoch = writer_epoch + 1
               WHERE lease_id = ?""",
            ("untrusted-candidate", encoded, request.lease_id),
        )

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    retained = migrated.get(request.lease_id)

    assert retained.provider_resource_id is None
    assert retained.ambiguous_create_candidate_id == "untrusted-candidate"
    assert retained.provisioning_mismatch is None
    assert retained.ambiguous_candidate_mismatch == mismatch
    with closing(sqlite3.connect(database)) as connection:
        stored = connection.execute(
            "SELECT provisioning_mismatch_json, "
            "ambiguous_candidate_mismatch_json FROM ollama_leases"
        ).fetchone()
    assert stored == (None, encoded)


def test_attestation_cannot_repair_an_incomplete_core_schema(tmp_path):
    database = tmp_path / "leases.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("ALTER TABLE ollama_leases DROP COLUMN owner_id")
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="not a supported"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )

    assert _database_files(database) == before


def test_runtime_refuses_wrong_current_column_definition_before_ddl(tmp_path):
    database = tmp_path / "wrong-column.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'ollama_leases'"
        ).fetchone()[0]
        index_sql = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_ollama_leases_reconcile'"
        ).fetchone()[0]
        trigger_sql = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
            )
        )
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("DROP INDEX idx_ollama_leases_reconcile")
        connection.execute("DROP TABLE ollama_leases")
        connection.execute(
            table_sql.replace("maximum_compute_cost REAL", "maximum_compute_cost TEXT")
        )
        connection.execute(index_sql)
        for statement in trigger_sql:
            connection.execute(statement)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


def test_runtime_refuses_hidden_extra_current_column_before_ddl(tmp_path):
    database = tmp_path / "hidden-column.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "ALTER TABLE ollama_leases ADD COLUMN owner_copy TEXT "
            "GENERATED ALWAYS AS (owner_id) VIRTUAL"
        )
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


@pytest.mark.parametrize(
    "extra_schema",
    [
        "CREATE INDEX unexpected_ollama_owner ON ollama_leases(owner_id)",
        """CREATE TRIGGER unexpected_ollama_update
           AFTER UPDATE ON ollama_leases BEGIN SELECT 1; END""",
    ],
)
def test_runtime_refuses_unexpected_current_schema_objects_before_ddl(
    tmp_path, extra_schema
):
    database = tmp_path / "extra-object.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(extra_schema)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


def _rebuild_with_owner_unique_constraint(database):
    with closing(sqlite3.connect(database)) as connection, connection:
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'ollama_leases'"
        ).fetchone()[0]
        index_sql = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_ollama_leases_reconcile'"
        ).fetchone()[0]
        trigger_sql = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
            )
        )
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("DROP INDEX idx_ollama_leases_reconcile")
        connection.execute("DROP TABLE ollama_leases")
        connection.execute(table_sql[:-1] + ", UNIQUE(owner_id))")
        connection.execute(index_sql)
        for statement in trigger_sql:
            connection.execute(statement)


def test_runtime_refuses_implicit_unique_index_before_any_ddl(tmp_path):
    database = tmp_path / "implicit-unique.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    _rebuild_with_owner_unique_constraint(database)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


def test_attestation_cannot_repair_unexpected_implicit_unique_index(tmp_path):
    database = tmp_path / "implicit-unique-attested.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    _rebuild_with_owner_unique_constraint(database)
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="not a supported"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database,
            attestation=_drain_attestation(database),
        )

    assert _database_files(database) == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sql: sql.replace("'requested'", "'REQUESTED'", 1),
        lambda sql: sql.replace("BEFORE INSERT", "BEFORE  INSERT", 1),
    ],
    ids=["literal-case", "whitespace"],
)
def test_runtime_refuses_noncanonical_trigger_sql_before_any_ddl(tmp_path, mutate):
    database = tmp_path / "noncanonical-trigger.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        name = "ollama_provisioning_evidence_insert_v6"
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (name,),
        ).fetchone()[0]
        connection.execute(f"DROP TRIGGER {name}")
        connection.execute(mutate(sql))
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


@pytest.mark.parametrize("missing_column", ["owner_id", "maximum_compute_cost"])
def test_live_wal_incomplete_current_schema_is_never_silently_repaired(
    tmp_path, missing_column
):
    database = tmp_path / "incomplete-current-wal.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with _live_wal_connection(database) as writer:
        for trigger in _WRITER_TRIGGERS:
            writer.execute(f"DROP TRIGGER {trigger}")
        writer.execute(f"ALTER TABLE ollama_leases DROP COLUMN {missing_column}")
        before = _persistent_database_files(database)

        with pytest.raises(RunPodManagerError, match="drained-writer migration"):
            SQLiteOllamaLeaseRepository(database)

        assert _persistent_database_files(database) == before


def test_legacy_database_without_drain_attestation_is_byte_for_byte_unchanged(
    tmp_path,
):
    database = tmp_path / "legacy.sqlite3"
    _legacy_database(database, state="ready")
    before = _database_files(database)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(database)

    assert _database_files(database) == before


def test_wal_only_legacy_schema_requires_drain_without_modification(tmp_path):
    database = tmp_path / "legacy-wal.sqlite3"
    with _live_wal_connection(database) as writer:
        writer.execute("CREATE TABLE ollama_leases (lease_id TEXT PRIMARY KEY)")
        before = _persistent_database_files(database)

        with pytest.raises(RunPodManagerError, match="drained-writer migration"):
            SQLiteOllamaLeaseRepository(database)

        assert _persistent_database_files(database) == before


def test_wal_resident_incomplete_writer_columns_are_not_silently_repaired(tmp_path):
    database = tmp_path / "partial-wal.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with _live_wal_connection(database) as writer:
        writer.execute("DROP TRIGGER ollama_provisioning_evidence_insert_v6")
        writer.execute("DROP TRIGGER ollama_provisioning_evidence_update_v6")
        writer.execute(
            "ALTER TABLE ollama_leases DROP COLUMN provisioning_mismatch_json"
        )
        before = _persistent_database_files(database)

        with pytest.raises(RunPodManagerError, match="drained-writer migration"):
            SQLiteOllamaLeaseRepository(database)

        assert _persistent_database_files(database) == before


def test_wal_resident_incomplete_writer_triggers_are_not_silently_repaired(tmp_path):
    database = tmp_path / "partial-trigger-wal.sqlite3"
    SQLiteOllamaLeaseRepository(database)
    with _live_wal_connection(database) as writer:
        writer.execute("DROP TRIGGER ollama_writer_delete_v6")
        writer.execute(
            """CREATE TRIGGER ollama_writer_delete_v6
               BEFORE DELETE ON ollama_leases
               BEGIN SELECT 1; END"""
        )
        before = _persistent_database_files(database)

        with pytest.raises(RunPodManagerError, match="drained-writer migration"):
            SQLiteOllamaLeaseRepository(database)

        assert _persistent_database_files(database) == before


def test_current_writer_contract_opens_with_outstanding_wal(tmp_path):
    database = tmp_path / "current-wal.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())
    with _live_wal_connection(database) as writer:
        writer.execute(
            """UPDATE ollama_leases
               SET last_provider_error = 'visible WAL value',
                   revision = revision + 1,
                   writer_epoch = writer_epoch + 1
               WHERE lease_id = 'lease-001'"""
        )

        reopened = SQLiteOllamaLeaseRepository(database)

        assert reopened.get("lease-001").last_provider_error == "visible WAL value"


@pytest.mark.parametrize("state", ["requested", "ready"])
def test_attested_legacy_migration_classifies_every_row_evidence_unknown(
    tmp_path, state
):
    database = tmp_path / "legacy.sqlite3"
    _legacy_database(database, state=state)

    repository = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    lease = repository.get("lease-001")

    assert lease.writer_contract_version == 6
    assert lease.writer_epoch == 0
    assert (
        lease.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
    )
    assert lease.provisioning_intent is None
    assert lease.provisioning_receipt is None
    assert lease.original_requested_at is None
    assert lease.original_readiness_timeout_seconds is None
    if state == "requested":
        with pytest.raises(RunPodManagerError, match="original request authority"):
            request_from_lease(lease)


@pytest.mark.asyncio
async def test_migrated_old_name_recovery_id_remains_untrusted_and_unfinalized(
    tmp_path,
):
    """Old recovery stored a name match without retaining an original POST ACK."""

    database = tmp_path / "legacy-name-collision.sqlite3"
    clock = _legacy_database(database, state="waiting_for_model")
    repository = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    provider = FakeOllamaProvider(serverless_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )

    migrated = repository.get("lease-001")
    assert migrated.provider_resource_id is None
    assert migrated.ambiguous_create_candidate_id == "historical-endpoint"

    assert await service.reconcile() == ()
    reconciled = repository.get("lease-001")
    assert reconciled is not None
    assert reconciled.state is OllamaLeaseState.WAITING_FOR_MODEL
    assert (
        reconciled.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
    )
    assert provider.teardown_calls == 0
    with pytest.raises(OllamaLeaseTeardownError, match="historical.*unknown"):
        await service.release(
            reconciled.lease_id,
            owner_id=reconciled.owner_id,
            workload_id=reconciled.workload_id,
        )
    assert (
        repository.get(reconciled.lease_id).state is OllamaLeaseState.WAITING_FOR_MODEL
    )


@pytest.mark.asyncio
async def test_migrated_terminal_flag_remains_visible_unknown_not_absence(tmp_path):
    database = tmp_path / "legacy-terminal-unknown.sqlite3"
    clock = _legacy_database(database, state="terminated")
    repository = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    provider = FakeOllamaProvider(serverless_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )

    reconciled = await service.reconcile()

    assert reconciled == ()
    retained = repository.get("lease-001")
    assert retained is not None
    assert retained.state is OllamaLeaseState.TERMINATED
    assert (
        retained.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
    )
    assert retained.ambiguous_create_candidate_id == "historical-endpoint"
    assert provider.teardown_calls == 0
    with pytest.raises(OllamaLeaseTeardownError, match="historical.*unknown"):
        await service.release(
            retained.lease_id,
            owner_id=retained.owner_id,
            workload_id=retained.workload_id,
        )


def test_attested_migration_atomically_scrubs_historical_ready_route(tmp_path):
    database = tmp_path / "legacy-ready-route.sqlite3"
    _legacy_database(database, state="ready", saved_route=True)

    repository = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )
    lease = repository.get("lease-001")

    assert lease.state is OllamaLeaseState.READY
    assert lease.route_url is None
    assert lease.provider_health_url is None
    assert (
        lease.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
    )


def test_attested_legacy_migration_rolls_back_every_schema_change_on_failure(
    tmp_path, monkeypatch
):
    database = tmp_path / "legacy.sqlite3"
    _legacy_database(database, state="ready")

    def interrupted(_connection):
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(
        "kestrel_cloud_runpod.ollama_repository._install_writer_triggers",
        interrupted,
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )

    with closing(sqlite3.connect(database)) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ollama_leases)")
        }
        row = connection.execute(
            "SELECT state, provider_resource_id FROM ollama_leases"
        ).fetchone()
    assert columns.isdisjoint(_WRITER_COLUMNS)
    assert row == ("ready", "historical-endpoint")


def test_concurrent_attested_initializers_observe_only_complete_writer_fence(
    tmp_path,
):
    database = tmp_path / "legacy.sqlite3"
    _legacy_database(database, state="ready")
    attestation = _drain_attestation(database)

    def migrate():
        repository = SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=attestation
        )
        return repository.get("lease-001")

    with ThreadPoolExecutor(max_workers=2) as executor:
        leases = tuple(executor.map(lambda _index: migrate(), range(2)))

    assert all(lease.writer_contract_version == 6 for lease in leases)
    assert all(
        lease.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        for lease in leases
    )
    with closing(sqlite3.connect(database)) as connection:
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    assert triggers == set(_WRITER_TRIGGERS)


def test_concurrent_process_initializers_observe_only_complete_writer_fence(
    tmp_path,
):
    database = tmp_path / "legacy-process.sqlite3"
    _legacy_database(database, state="ready", saved_route=True)
    context = get_context("spawn")
    gate = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_migrate_legacy_database_process,
            args=(database, gate, results),
        )
        for _index in range(2)
    ]
    for process in processes:
        process.start()
    gate.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=2) for _process in processes)
    results.close()
    results.join_thread()
    assert observed == [
        ("ok", 6, "historical_unknown"),
        ("ok", 6, "historical_unknown"),
    ]


@pytest.mark.parametrize(
    "assignment",
    [
        "state = 'failed', revision = revision + 1",
        "last_used_at = updated_at, revision = revision + 1",
        "teardown_state = 'pending', revision = revision + 1",
        "last_provider_error = 'old writer'",
    ],
)
def test_writer_epoch_fence_rejects_every_legacy_update_shape(tmp_path, assignment):
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())

    with (
        closing(sqlite3.connect(database)) as connection,
        pytest.raises(sqlite3.IntegrityError, match="stale Ollama writer"),
    ):
        connection.execute(f"UPDATE ollama_leases SET {assignment}")


@pytest.mark.parametrize(
    "assignment",
    [
        "owner_id = 'other-owner'",
        "workload_id = 'other-workload'",
        "request_fingerprint = 'other-fingerprint'",
        "model = 'other-model'",
        "constraints_json = '{}'",
        "original_requested_at = '2026-10-20T00:00:00Z'",
        "original_readiness_timeout_seconds = 999",
        "expected_session_seconds = expected_session_seconds + 1",
        "expected_active_seconds = expected_active_seconds + 1",
        "serverless_initialization_seconds = serverless_initialization_seconds + 1",
        "serverless_idle_tail_seconds = serverless_idle_tail_seconds + 1",
        "idle_timeout_seconds = idle_timeout_seconds + 1",
        "hard_deadline = '2026-10-21T00:00:00Z'",
        "readiness_deadline = '2026-10-21T00:00:00Z'",
        "max_authorized_cost = max_authorized_cost + 1",
        "mode = 'dedicated_pod'",
    ],
)
def test_original_request_authority_is_immutable(tmp_path, assignment):
    database = tmp_path / "immutable-request.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())

    with (
        closing(sqlite3.connect(database)) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="invalid Ollama provisioning evidence mutation",
        ),
    ):
        connection.execute(
            f"""UPDATE ollama_leases
                SET {assignment}, revision = revision + 1,
                    writer_epoch = writer_epoch + 1"""
        )


def test_writer_fence_rejects_legacy_insert_and_delete(tmp_path):
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())

    with closing(sqlite3.connect(database)) as connection:
        with pytest.raises(
            sqlite3.IntegrityError, match="current Ollama writer contract"
        ):
            connection.execute(
                """INSERT INTO ollama_leases (
                       lease_id, owner_id, workload_id, request_fingerprint, model,
                       original_requested_at,
                       original_readiness_timeout_seconds,
                       constraints_json, mode, state, teardown_state, created_at,
                       updated_at, last_used_at, idle_deadline, hard_deadline,
                       readiness_deadline, expected_session_seconds,
                       expected_active_seconds, serverless_initialization_seconds,
                       serverless_idle_tail_seconds, idle_timeout_seconds,
                       max_authorized_cost
                   )
                   SELECT 'old-writer-lease', owner_id, workload_id,
                          request_fingerprint, model, original_requested_at,
                          original_readiness_timeout_seconds, constraints_json,
                          mode, state,
                          teardown_state, created_at, updated_at, last_used_at,
                          idle_deadline, hard_deadline, readiness_deadline,
                          expected_session_seconds, expected_active_seconds,
                          serverless_initialization_seconds,
                          serverless_idle_tail_seconds, idle_timeout_seconds,
                          max_authorized_cost
                   FROM ollama_leases WHERE lease_id = 'lease-001'"""
            )
        with pytest.raises(
            sqlite3.IntegrityError, match="lease deletion is unsupported"
        ):
            connection.execute("DELETE FROM ollama_leases WHERE lease_id = 'lease-001'")


@pytest.mark.parametrize(
    "assignment",
    [
        "ambiguous_create_candidate_id = 'unbound-candidate'",
        "provisioning_intent_json = '{}'",
        "provisioning_receipt_json = '{}'",
        "provisioning_mismatch_json = '{}'",
        "provider_resource_id = 'unbound-resource'",
    ],
)
def test_current_writer_cannot_attach_evidence_without_its_typed_state(
    tmp_path, assignment
):
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())

    with (
        closing(sqlite3.connect(database)) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="invalid Ollama provisioning evidence mutation",
        ),
    ):
        connection.execute(
            f"""UPDATE ollama_leases
                SET {assignment}, revision = revision + 1,
                    writer_epoch = writer_epoch + 1"""
        )


def test_startup_route_scrub_advances_revision_and_writer_epoch(tmp_path):
    database = tmp_path / "leases.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    lease, _ = repository.insert_request(make_request(clock), now=clock())
    routed = repository.compare_and_set(
        lease,
        changes={
            "route_url": "https://private.invalid",
            "provider_health_url": "https://private.invalid/ping",
        },
    )

    restarted = SQLiteOllamaLeaseRepository(database).get(lease.lease_id)

    assert restarted.route_url is None
    assert restarted.provider_health_url is None
    assert restarted.revision == routed.revision + 1
    assert restarted.writer_epoch == routed.writer_epoch + 1


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
        lease, changes={"last_provider_error": "first writer"}
    )

    with pytest.raises(OllamaLeaseConflictError, match="changed concurrently"):
        repository.compare_and_set(
            lease, changes={"last_provider_error": "stale writer"}
        )

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


def test_request_recovery_rejects_changed_original_fingerprint_inputs(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "leases.sqlite3")
    lease, _ = repository.insert_request(
        make_request(clock, requested_at=clock(), readiness_timeout_seconds=73),
        now=clock(),
    )

    with pytest.raises(RunPodManagerError, match="authority is corrupt"):
        request_from_lease(
            replace(
                lease,
                original_readiness_timeout_seconds=74,
            )
        )


@pytest.mark.parametrize("integer_field", ["max_authorized_cost", "max_hourly_rate"])
def test_first_acceptance_normalizes_public_integer_money_identity(
    tmp_path, integer_field
):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "integer-money.sqlite3")
    request = make_request(clock)
    if integer_field == "max_authorized_cost":
        request = replace(request, max_authorized_cost=2)
    else:
        request = replace(
            request,
            constraints=replace(request.constraints, max_hourly_rate=1),
        )
    lease, inserted = repository.insert_request(request, now=clock())

    assert inserted is True
    recovered = request_from_lease(repository.get(lease.lease_id))
    canonical = normalize_new_ollama_request_money(request)
    assert recovered.fingerprint == canonical.fingerprint
    assert lease.request_fingerprint == canonical.fingerprint
    repeated, repeated_inserted = repository.insert_request(request, now=clock())
    float_spelling, float_spelling_inserted = repository.insert_request(
        canonical, now=clock()
    )
    assert repeated_inserted is False
    assert float_spelling_inserted is False
    assert repeated.request_fingerprint == canonical.fingerprint
    assert float_spelling.request_fingerprint == canonical.fingerprint


@pytest.mark.parametrize("integer_field", ["max_authorized_cost", "max_hourly_rate"])
def test_first_acceptance_integer_money_never_expands_authority(integer_field):
    original = 9_007_199_254_740_995
    request = make_request(MutableClock())
    if integer_field == "max_authorized_cost":
        request = replace(request, max_authorized_cost=original)
    else:
        request = replace(
            request,
            constraints=replace(request.constraints, max_hourly_rate=original),
        )

    accepted = normalize_new_ollama_request_money(request)
    normalized = (
        accepted.max_authorized_cost
        if integer_field == "max_authorized_cost"
        else accepted.constraints.max_hourly_rate
    )

    assert Decimal(str(normalized)) <= Decimal(original)


@pytest.mark.parametrize("integer_field", ["max_authorized_cost", "max_hourly_rate"])
def test_recovery_preserves_an_already_accepted_integer_fingerprint(
    tmp_path, integer_field
):
    clock = MutableClock()
    database = tmp_path / "accepted-integer-money.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    request = make_request(clock)
    if integer_field == "max_authorized_cost":
        request = replace(request, max_authorized_cost=2)
    else:
        request = replace(
            request,
            constraints=replace(request.constraints, max_hourly_rate=1),
        )
    repository.insert_request(normalize_new_ollama_request_money(request), now=clock())
    constraints = request.constraints.__dict__.copy()
    constraints["cloud"] = request.constraints.cloud.value
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("DROP TRIGGER ollama_writer_update_v6")
        connection.execute("DROP TRIGGER ollama_provisioning_evidence_update_v6")
        connection.execute(
            """UPDATE ollama_leases
               SET request_fingerprint = ?, constraints_json = ?,
                   max_authorized_cost = ?
               WHERE lease_id = ?""",
            (
                request.fingerprint,
                json.dumps(constraints, sort_keys=True, separators=(",", ":")),
                request.max_authorized_cost,
                request.lease_id,
            ),
        )

    recovered = request_from_lease(repository.get(request.lease_id))

    assert recovered.fingerprint == request.fingerprint


@pytest.mark.asyncio
async def test_runtime_telemetry_is_monotonic_and_restart_gaps_stay_partial(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "telemetry.sqlite3")
    provider = FakeOllamaProvider(pod_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(make_request(clock), plan=pod_plan())
    original = ready.runtime_telemetry_observations[-1]
    assert ready.runtime_telemetry_state is (
        OllamaRuntimeTelemetryState.STARTUP_COMPLETE
    )
    first_started = original.observed_at + timedelta(seconds=1)
    same_process = _revised_telemetry(
        original,
        telemetry_revision=2,
        inference_started=3,
        inference_completed=1,
        inference_failed=1,
        inference_cancelled=1,
        first_inference_started_at=first_started,
        last_inference_started_at=first_started + timedelta(seconds=1),
        last_inference_completed_at=first_started + timedelta(seconds=2),
        inference_busy_ms=11,
        inference_idle_ms=7,
    )
    updated = repository.record_runtime_telemetry(
        ready,
        same_process,
        now=clock(),
        maximum_observations=8,
        maximum_bytes=100_000,
    )
    assert updated.runtime_telemetry_state is (
        OllamaRuntimeTelemetryState.STARTUP_COMPLETE
    )

    restarted = _revised_telemetry(
        same_process,
        runtime_instance_id="runtime-test-2",
        telemetry_revision=1,
        inference_started=0,
        inference_completed=0,
        inference_failed=0,
        inference_cancelled=0,
        first_inference_started_at=None,
        last_inference_started_at=None,
        last_inference_completed_at=None,
        inference_busy_ms=0,
        inference_idle_ms=0,
    )
    partial = repository.record_runtime_telemetry(
        updated,
        restarted,
        now=clock(),
        maximum_observations=8,
        maximum_bytes=100_000,
    )
    assert partial.runtime_telemetry_state is OllamaRuntimeTelemetryState.PARTIAL

    later_restart = _revised_telemetry(restarted, telemetry_revision=2)
    retained = repository.record_runtime_telemetry(
        partial,
        later_restart,
        now=clock(),
        maximum_observations=8,
        maximum_bytes=100_000,
    )
    assert retained.runtime_telemetry_state is OllamaRuntimeTelemetryState.PARTIAL


@pytest.mark.asyncio
async def test_terminal_telemetry_stays_partial_without_population_quiescence(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "telemetry-terminal.sqlite3")
    provider = FakeOllamaProvider(pod_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(make_request(clock), plan=pod_plan())
    original = ready.runtime_telemetry_observations[-1]
    terminated = await service.release(
        ready.lease_id, owner_id=ready.owner_id, workload_id=ready.workload_id
    )
    final = _revised_telemetry(original, telemetry_revision=2)
    completed = repository.record_runtime_telemetry(
        terminated,
        final,
        now=clock(),
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert completed.runtime_telemetry_state is OllamaRuntimeTelemetryState.PARTIAL
    assert len(completed.runtime_telemetry_observations) == 2


def test_raw_sql_cannot_claim_no_provider_effect_after_admission(tmp_path):
    database = tmp_path / "false-no-effect.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())

    with (
        closing(sqlite3.connect(database)) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="invalid Ollama lifecycle evidence mutation",
        ),
    ):
        connection.execute(
            """UPDATE ollama_leases
                  SET state = 'terminated', teardown_state = 'complete',
                      provision_attempts = 1,
                      lifecycle_billing_state = 'not_applicable',
                      revision = revision + 1, writer_epoch = writer_epoch + 1
                WHERE lease_id = 'lease-001'"""
        )


def test_raw_sql_cannot_manufacture_historical_latest_billing_state(tmp_path):
    database = tmp_path / "false-historical-latest.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    repository.insert_request(make_request(clock), now=clock())

    with (
        closing(sqlite3.connect(database)) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="invalid Ollama lifecycle evidence mutation",
        ),
    ):
        connection.execute(
            """UPDATE ollama_leases
                  SET lifecycle_billing_latest_reference_json =
                        '{"observation_id":null,"received_at":null,"state":"historical_unknown"}',
                      revision = revision + 1, writer_epoch = writer_epoch + 1
                WHERE lease_id = 'lease-001'"""
        )


@pytest.mark.asyncio
async def test_sql_fence_preserves_selected_lifecycle_evidence(tmp_path):
    repository, selected, _observed_at = await _selected_pod_billing_lease(tmp_path)
    assert selected.lifecycle_billing_state is OllamaLifecycleBillingState.SELECTED
    mutations = (
        "lifecycle_billing_observations_json = '[]'",
        "lifecycle_billing_selection_json = NULL",
        "lifecycle_billing_state = 'unknown'",
        (
            "lifecycle_billing_state = 'pending', "
            "lifecycle_billing_selection_json = NULL, "
            "lifecycle_billing_latest_reference_json = "
            '\'{"observation_id":null,"received_at":null,'
            '"state":"not_observed"}\''
        ),
        "cleanup_receipt_json = NULL",
        (
            "lifecycle_billing_observations_json = "
            "json_insert(lifecycle_billing_observations_json, '$[#]', "
            "json_extract(lifecycle_billing_observations_json, '$[0]'))"
        ),
        (
            "lifecycle_billing_latest_reference_json = "
            '\'{"observation_id":null,"received_at":null,'
            '"state":"not_observed"}\''
        ),
        (
            "lifecycle_billing_latest_reference_json = "
            '\'{"observation_id":"foreign",'
            '"received_at":"2026-10-21T00:00:00.000000Z",'
            '"state":"known"}\', '
            "billing_last_checked_at = '2026-10-21T00:00:00.000000Z'"
        ),
        (
            "lifecycle_billing_latest_reference_json = "
            '\'{"observation_id":null,"received_at":null,'
            '"state":"historical_unknown"}\''
        ),
    )
    for mutation in mutations:
        with (
            closing(sqlite3.connect(repository.path)) as connection,
            pytest.raises(
                sqlite3.IntegrityError,
                match="invalid Ollama lifecycle evidence mutation",
            ),
        ):
            connection.execute(
                f"""UPDATE ollama_leases
                        SET {mutation}, revision = revision + 1,
                            writer_epoch = writer_epoch + 1
                      WHERE lease_id = ?""",
                (selected.lease_id,),
            )


@pytest.mark.asyncio
async def test_billing_correction_delta_is_exact_under_host_decimal_context(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    intent = selected.provisioning_intent
    provisioning = selected.provisioning_receipt
    cleanup = selected.cleanup_receipt
    original = selected.lifecycle_billing_observations[0]
    assert intent is not None
    assert provisioning is not None
    assert cleanup is not None
    revised_amount = Decimal("1e-29")
    revised_receipt = replace(
        original.pod_billing_receipt,
        provider_billing_id="pod-billing-revised",
        actual_cost_usd=revised_amount,
    )
    revised = replace(
        original,
        provider_capacity_actual_cost_usd=revised_amount,
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=revised_receipt,
    )

    with localcontext() as context:
        context.prec = 2
        context.Emax = 9
        context.Emin = -9
        corrected = repository.record_billing_outcome(
            selected,
            now=observed_at + timedelta(hours=1),
            next_check_at=observed_at + timedelta(hours=2),
            observation=revised,
            maximum_observations=8,
            maximum_bytes=100_000,
        )

    assert len(corrected.lifecycle_billing_corrections) == 1
    assert corrected.lifecycle_billing_corrections[0].actual_cost_delta_usd == (
        Decimal("-0.99999999999999999999999999999")
    )


@pytest.mark.asyncio
async def test_billing_correction_negation_is_context_independent(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path, actual=Decimal("0.12345")
    )
    original = selected.lifecycle_billing_observations[0]
    revised_receipt = replace(
        original.pod_billing_receipt,
        provider_billing_id="pod-billing-close-revision",
        actual_cost_usd=Decimal("0.12346"),
    )
    revised = replace(
        original,
        provider_capacity_actual_cost_usd=Decimal("0.12346"),
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=revised_receipt,
    )

    with localcontext() as context:
        context.prec = 2
        corrected = repository.record_billing_outcome(
            selected,
            now=observed_at + timedelta(hours=1),
            next_check_at=observed_at + timedelta(hours=2),
            observation=revised,
            maximum_observations=8,
            maximum_bytes=100_000,
        )

    assert corrected.lifecycle_billing_corrections[0].actual_cost_delta_usd == (
        Decimal("0.00001")
    )
    assert repository.get(selected.lease_id) == corrected


@pytest.mark.asyncio
async def test_billing_retention_status_survives_a_later_duplicate_poll(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path, actual=Decimal("0.1")
    )
    original = selected.lifecycle_billing_observations[0]
    revised = replace(
        original,
        provider_capacity_actual_cost_usd=Decimal("0.2"),
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=replace(
            original.pod_billing_receipt,
            provider_billing_id="pod-billing-retention-revision",
            actual_cost_usd=Decimal("0.2"),
        ),
    )
    bounded = repository.record_billing_outcome(
        selected,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=revised,
        maximum_observations=1,
        maximum_bytes=100_000,
    )
    assert bounded.lifecycle_billing_state is (
        OllamaLifecycleBillingState.CORRECTION_PENDING
    )

    repeated = repository.record_billing_outcome(
        bounded,
        now=observed_at + timedelta(hours=2),
        next_check_at=observed_at + timedelta(hours=3),
        observation=original,
        maximum_observations=1,
        maximum_bytes=100_000,
    )

    assert repeated.lifecycle_billing_state is (
        OllamaLifecycleBillingState.CORRECTION_PENDING
    )
    assert repeated.lifecycle_billing_observations == (
        selected.lifecycle_billing_observations
    )
    assert (
        repeated.lifecycle_billing_latest_reference
        == bounded.lifecycle_billing_latest_reference
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("later_capacity_bytes", [0, 100_000])
async def test_billing_retention_before_first_selection_remains_manual(
    tmp_path, later_capacity_bytes
):
    (
        repository,
        terminated,
        observed_at,
        oversized,
    ) = await _terminated_pod_billing_evidence(
        tmp_path,
        actual=Decimal("0.12345678901234567"),
        database_name=f"first-selection-retention-{later_capacity_bytes}.sqlite3",
    )
    assert oversized.pod_billing_receipt is not None
    shorter = replace(
        oversized,
        provider_capacity_actual_cost_usd=Decimal("0.1"),
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=replace(
            oversized.pod_billing_receipt,
            provider_billing_id="pod-billing-shorter-revision",
            actual_cost_usd=Decimal("0.1"),
            reconciled_at=observed_at + timedelta(hours=1),
        ),
    )
    shorter_selection = OllamaLifecycleBillingSelection(
        observation_id=shorter.observation_id,
        selected_at=observed_at + timedelta(hours=1),
    )
    retention_limit = sum(
        len(_db_value(value)) for value in ((shorter,), (), shorter_selection)
    )
    assert len(_db_value((oversized,))) > len(_db_value((shorter,)))

    bounded = repository.record_billing_outcome(
        terminated,
        now=observed_at,
        next_check_at=observed_at + timedelta(hours=1),
        observation=oversized,
        maximum_observations=8,
        maximum_bytes=retention_limit,
    )
    assert bounded.lifecycle_billing_state is OllamaLifecycleBillingState.PENDING
    assert bounded.lifecycle_billing_observations == ()
    assert bounded.lifecycle_billing_selection is None
    assert bounded.billing_retention_exhausted is True

    repeated = repository.record_billing_outcome(
        bounded,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=shorter,
        maximum_observations=8,
        maximum_bytes=retention_limit + later_capacity_bytes,
    )

    assert repeated.lifecycle_billing_state is OllamaLifecycleBillingState.PENDING
    assert repeated.lifecycle_billing_observations == ()
    assert repeated.lifecycle_billing_selection is None
    assert repeated.lifecycle_billing_corrections == ()
    assert repeated.billing_retention_exhausted is True
    assert (
        repeated.lifecycle_billing_latest_reference.state
        is OllamaLifecycleBillingLatestState.NOT_OBSERVED
    )
    assert repeated.billing_last_checked_at == observed_at + timedelta(hours=1)
    assert repeated.billing_last_provider_error == (
        "billing evidence retention requires manual reconciliation"
    )


@pytest.mark.asyncio
async def test_pod_billing_repoll_time_is_not_a_provider_cost_revision(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path, actual=Decimal("0.25")
    )
    original = selected.lifecycle_billing_observations[0]
    assert original.pod_billing_receipt is not None
    repolled = replace(
        original,
        first_observed_at=observed_at + timedelta(hours=1),
        pod_billing_receipt=replace(
            original.pod_billing_receipt,
            reconciled_at=observed_at + timedelta(hours=1),
        ),
    )

    assert repolled.observation_id == original.observation_id
    repeated = repository.record_billing_outcome(
        selected,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=repolled,
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert repeated.lifecycle_billing_observations == (original,)
    assert repeated.lifecycle_billing_corrections == ()
    assert repeated.lifecycle_billing_state is OllamaLifecycleBillingState.SELECTED
    assert repeated.lifecycle_billing_latest_reference.observation_id == (
        original.observation_id
    )
    assert repeated.lifecycle_billing_latest_reference.received_at == (
        observed_at + timedelta(hours=1)
    )


@pytest.mark.asyncio
async def test_billing_error_advances_schedule_without_changing_latest_reference(
    tmp_path,
):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path, actual=Decimal("0.25")
    )
    original_reference = selected.lifecycle_billing_latest_reference

    failed = repository.record_billing_outcome(
        selected,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        provider_error="provider billing evidence unavailable",
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert failed.lifecycle_billing_latest_reference == original_reference
    assert failed.billing_last_checked_at == observed_at + timedelta(hours=1)
    assert failed.billing_last_provider_error == (
        "provider billing evidence unavailable"
    )


@pytest.mark.asyncio
async def test_fresh_poll_resolves_migrated_v5_latest_reference_without_new_fact(
    tmp_path,
):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path,
        actual=Decimal("0.25"),
        database_name="v5-latest-resolution.sqlite3",
    )
    original = selected.lifecycle_billing_observations[0]
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        _downgrade_to_v5_schema(connection)
    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        repository.path,
        attestation=_drain_attestation(repository.path),
    )
    historical = migrated.get(selected.lease_id)
    assert historical is not None
    assert (
        historical.lifecycle_billing_latest_reference.state
        is OllamaLifecycleBillingLatestState.HISTORICAL_UNKNOWN
    )

    resolved = migrated.record_billing_outcome(
        historical,
        now=observed_at + timedelta(hours=1),
        next_check_at=observed_at + timedelta(hours=2),
        observation=original,
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert resolved.lifecycle_billing_observations == (original,)
    assert resolved.lifecycle_billing_corrections == ()
    assert (
        resolved.lifecycle_billing_latest_reference.state
        is OllamaLifecycleBillingLatestState.KNOWN
    )
    assert resolved.lifecycle_billing_latest_reference.observation_id == (
        original.observation_id
    )
    assert resolved.lifecycle_billing_latest_reference.received_at == (
        observed_at + timedelta(hours=1)
    )


def test_billing_latest_reference_constructor_and_parser_are_strict() -> None:
    received_at = datetime(2026, 10, 21, tzinfo=UTC)
    reference = OllamaLifecycleBillingLatestReference.known(
        "billing-observation", received_at=received_at
    )

    assert OllamaLifecycleBillingLatestReference.from_dict(reference.to_dict()) == (
        reference
    )
    with pytest.raises(TypeError, match="supported enum"):
        OllamaLifecycleBillingLatestReference(  # type: ignore[arg-type]
            state="known",
            observation_id="billing-observation",
            received_at=received_at,
        )
    with pytest.raises(TypeError, match="supported enum"):
        OllamaLifecycleBillingLatestReference(  # type: ignore[arg-type]
            state=object()
        )
    with pytest.raises(ValueError, match="not a valid"):
        OllamaLifecycleBillingLatestReference.from_dict(
            {
                "state": "future",
                "observation_id": None,
                "received_at": None,
            }
        )
    with pytest.raises(ValueError, match="fields are invalid"):
        OllamaLifecycleBillingLatestReference.from_dict(
            {
                "state": "not_observed",
                "observation_id": None,
                "received_at": None,
                "future": None,
            }
        )


@pytest.mark.asyncio
async def test_orphaned_billing_latest_reference_fails_readback_and_projection(
    tmp_path,
) -> None:
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    orphaned = replace(
        selected,
        lifecycle_billing_latest_reference=(
            OllamaLifecycleBillingLatestReference.known(
                "orphaned-observation", received_at=observed_at
            )
        ),
    )
    with pytest.raises(ValueError, match="reference is unbound"):
        orphaned.to_dict()

    with closing(sqlite3.connect(repository.path)) as connection, connection:
        connection.execute("DROP TRIGGER ollama_lifecycle_evidence_update_v6")
        connection.execute(
            """
            UPDATE ollama_leases
               SET lifecycle_billing_latest_reference_json = ?,
                   revision = revision + 1,
                   writer_epoch = writer_epoch + 1
             WHERE lease_id = ?
            """,
            (
                json.dumps(
                    orphaned.lifecycle_billing_latest_reference.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                selected.lease_id,
            ),
        )

    with pytest.raises(RunPodManagerError, match="latest reference is unbound"):
        repository.get(selected.lease_id)


@pytest.mark.asyncio
async def test_raw_provider_capacity_over_cap_is_explicit_not_all_in_finality(
    tmp_path,
):
    _repository, selected, _observed_at = await _selected_pod_billing_lease(
        tmp_path, actual=Decimal(3)
    )

    assert selected.max_authorized_cost == 2.0
    assert selected.actual_cost_cap_status.value == "exceeded"
    assert selected.lifecycle_billing_observations[0].provider_finality_confirmed is (
        False
    )


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
