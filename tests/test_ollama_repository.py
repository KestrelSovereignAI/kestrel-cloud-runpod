"""Persistence, idempotency, and restart tests for Ollama leases."""

import asyncio
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from multiprocessing import get_context
from pickle import PicklingError

import pytest
from kestrel_sovereign.execution_custody import is_execution_control_error
from ollama_test_support import (
    FakeOllamaProvider,
    MutableClock,
    OwnedSpawnProcesses,
    make_request,
    pod_plan,
    release_gates_and_boundedly_join_owned_tasks,
    serverless_plan,
)
from pod_capacity_test_support import execution_control_carrier

from kestrel_cloud_runpod.inference_provider import _lease_metadata
from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaAdditionalCleanupTarget,
    OllamaCleanupBasis,
    OllamaCleanupReceipt,
    OllamaLeaseConflictError,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaLifecycleBillingLatestReference,
    OllamaLifecycleBillingLatestState,
    OllamaLifecycleBillingObservation,
    OllamaLifecycleBillingSelection,
    OllamaLifecycleBillingState,
    OllamaProvisioningEvidenceError,
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
    _CURRENT_WRITER_CONTRACT_VERSION,
    OllamaWriterDrainAttestation,
    SQLiteOllamaLeaseRepository,
    _db_value,
    lease_database_path,
    request_from_lease,
)
from kestrel_cloud_runpod.ollama_service import OllamaLeaseService
from kestrel_cloud_runpod.pod_capacity_contracts import PodBillingReceipt

_WRITER_TRIGGERS = (
    "ollama_writer_insert_v7",
    "ollama_writer_update_v7",
    "ollama_writer_delete_v7",
    "ollama_provisioning_evidence_insert_v7",
    "ollama_provisioning_evidence_update_v7",
    "ollama_lifecycle_evidence_update_v7",
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


def _raw_database_snapshot(database):
    with closing(sqlite3.connect(database)) as connection:
        schema = tuple(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name"
            )
        )
        columns = tuple(connection.execute("PRAGMA table_xinfo(ollama_leases)"))
        rows = tuple(
            connection.execute("SELECT * FROM ollama_leases ORDER BY lease_id")
        )
    return schema, columns, rows


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


def _wait_for_owned_process_release(gate):
    gate.wait(timeout=10)


def _never_finishing_owned_process():
    while True:
        time.sleep(1)


def _put_owned_process_result(results):
    results.put("owned-result")


def _rescue_test_processes(processes, *gates):
    """Finitely rescue only this test's captured local spawn processes."""

    for gate in gates:
        gate.set()
    forced = []
    for process in processes:
        try:
            alive = process.is_alive()
        except ValueError:
            continue
        if alive:
            process.join(timeout=2)
        if process.is_alive():
            forced.append(process.pid)
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.is_alive():
            pytest.fail(f"Owned test process {process.pid} remained alive after rescue")
        process.close()
    return tuple(forced)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rate_case", "actual"),
    [
        pytest.param("exact", Decimal(1), id="exact"),
        pytest.param("mismatch", Decimal(1), id="mismatch"),
        pytest.param("mismatch", Decimal(0), id="mismatch-zero-cost"),
    ],
)
async def test_pod_lifecycle_billing_binds_original_realized_rate(
    tmp_path,
    rate_case,
    actual,
):
    (
        _repository,
        terminated,
        _observed_at,
        observation,
    ) = await _terminated_pod_billing_evidence(
        tmp_path,
        actual=actual,
        database_name=f"billing-rate-{rate_case}-{actual}.sqlite3",
    )
    intent = terminated.provisioning_intent
    provisioning = terminated.provisioning_receipt
    cleanup_intent = terminated.cleanup_intent
    cleanup = terminated.cleanup_receipt
    receipt = observation.pod_billing_receipt
    assert intent is not None
    assert provisioning is not None
    assert provisioning.pod_realized_placement is not None
    assert cleanup_intent is not None
    assert cleanup is not None
    assert receipt is not None
    assert observation.provider_capacity_actual_cost_usd == actual
    candidate = observation
    if rate_case == "mismatch":
        candidate = replace(
            observation,
            pod_billing_receipt=replace(
                receipt,
                hourly_price_usd=receipt.hourly_price_usd + Decimal("0.1"),
            ),
        )

    if rate_case == "exact":
        candidate.validate_lifecycle(intent, provisioning, cleanup_intent, cleanup)
    else:
        with pytest.raises(ValueError, match="realized.*rate|rate.*lifecycle"):
            candidate.validate_lifecycle(
                intent,
                provisioning,
                cleanup_intent,
                cleanup,
            )


@pytest.mark.asyncio
async def test_repository_rejects_pod_billing_rate_drift_atomically(tmp_path):
    (
        repository,
        terminated,
        observed_at,
        observation,
    ) = await _terminated_pod_billing_evidence(
        tmp_path,
        database_name="billing-rate-atomic.sqlite3",
    )
    receipt = observation.pod_billing_receipt
    assert receipt is not None
    drifted = replace(
        observation,
        pod_billing_receipt=replace(
            receipt,
            hourly_price_usd=receipt.hourly_price_usd + Decimal("0.1"),
        ),
    )
    durable_before = repository.get(terminated.lease_id)
    files_before = _database_files(repository.path)

    with pytest.raises(ValueError, match="realized.*rate|rate.*lifecycle"):
        repository.record_billing_outcome(
            terminated,
            now=observed_at,
            next_check_at=observed_at + timedelta(hours=1),
            observation=drifted,
            maximum_observations=8,
            maximum_bytes=100_000,
        )

    assert repository.get(terminated.lease_id) == durable_before
    assert _database_files(repository.path) == files_before


@pytest.mark.asyncio
async def test_repository_decode_rejects_fault_injected_pod_billing_rate_drift(
    tmp_path,
):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    original = selected.lifecycle_billing_observations[0]
    receipt = original.pod_billing_receipt
    assert receipt is not None
    drifted = replace(
        original,
        pod_billing_receipt=replace(
            receipt,
            hourly_price_usd=receipt.hourly_price_usd + Decimal("0.1"),
        ),
    )
    assert selected.lifecycle_billing_selection is not None
    selection = replace(
        selected.lifecycle_billing_selection,
        observation_id=drifted.observation_id,
    )
    latest = OllamaLifecycleBillingLatestReference.known(
        drifted.observation_id,
        received_at=observed_at,
    )
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        connection.execute("DROP TRIGGER ollama_lifecycle_evidence_update_v7")
        connection.execute(
            """
            UPDATE ollama_leases
               SET lifecycle_billing_observations_json = ?,
                   lifecycle_billing_selection_json = ?,
                   lifecycle_billing_latest_reference_json = ?,
                   revision = revision + 1,
                   writer_epoch = writer_epoch + 1
             WHERE lease_id = ?
            """,
            (
                json.dumps([drifted.to_dict()], sort_keys=True, separators=(",", ":")),
                json.dumps(selection.to_dict(), sort_keys=True, separators=(",", ":")),
                json.dumps(latest.to_dict(), sort_keys=True, separators=(",", ":")),
                selected.lease_id,
            ),
        )

    with pytest.raises(RunPodManagerError, match="Stored Ollama billing is unbound"):
        repository.get(selected.lease_id)


async def _versioned_contract_examples(tmp_path, *, database_name):
    repository, selected, _observed_at = await _selected_pod_billing_lease(
        tmp_path, database_name=database_name
    )
    intent = selected.provisioning_intent
    receipt = selected.provisioning_receipt
    cleanup_intent = selected.cleanup_intent
    cleanup_receipt = selected.cleanup_receipt
    telemetry = selected.runtime_telemetry_observations[-1]
    billing = selected.lifecycle_billing_observations[0]
    selection = selected.lifecycle_billing_selection
    assert intent is not None
    assert receipt is not None
    assert cleanup_intent is not None
    assert cleanup_receipt is not None
    assert selection is not None
    mismatch = OllamaProvisioningMismatchEvidence(
        intent_sha256=intent.intent_sha256,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        provider_resource_id=receipt.provider_resource_id,
        observed_resource_name=intent.resource_name,
        observed_at=receipt.readback_at,
        readback_sha256="c" * 64,
        reason=OllamaProvisioningMismatchReason.RUNTIME,
    )
    additional = OllamaAdditionalCleanupTarget(
        intent_sha256=intent.intent_sha256,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        resource_name=intent.resource_name,
        provider_resource_id="additional-owned-resource",
        source="original_post_ack",
    )
    return repository, {
        "provisioning-intent": (intent, "evidence_version", 1),
        "provisioning-receipt": (receipt, "evidence_version", 1),
        "provisioning-mismatch": (mismatch, "evidence_version", 1),
        "cleanup-intent": (cleanup_intent, "evidence_version", 1),
        "cleanup-receipt": (cleanup_receipt, "evidence_version", 1),
        "additional-cleanup-target": (additional, "evidence_version", 1),
        "runtime-snapshot": (telemetry.snapshot, "schema_version", 2),
        "runtime-observation": (telemetry, "evidence_version", 1),
        "billing-observation": (billing, "evidence_version", 1),
        "billing-selection": (selection, "policy_version", 1),
    }


_VERSIONED_CONTRACT_NAMES = (
    "provisioning-intent",
    "provisioning-receipt",
    "provisioning-mismatch",
    "cleanup-intent",
    "cleanup-receipt",
    "additional-cleanup-target",
    "runtime-snapshot",
    "runtime-observation",
    "billing-observation",
    "billing-selection",
)

_VERSIONED_CONTRACT_ERROR_MESSAGES = {
    "provisioning-intent": "Unsupported Ollama provisioning intent version",
    "provisioning-receipt": "Unsupported Ollama provisioning receipt version",
    "provisioning-mismatch": "Unsupported Ollama provisioning mismatch version",
    "cleanup-intent": "Unsupported Ollama cleanup intent version",
    "cleanup-receipt": "Unsupported Ollama cleanup receipt version",
    "additional-cleanup-target": "Unsupported Ollama additional cleanup target version",
    "runtime-snapshot": "Unsupported Ollama telemetry schema",
    "runtime-observation": "Unsupported Ollama telemetry observation version",
    "billing-observation": "Ollama billing cannot claim provider finality",
    "billing-selection": "Ollama billing selection is invalid",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_name", _VERSIONED_CONTRACT_NAMES)
@pytest.mark.parametrize(
    "bad_version",
    [pytest.param(True, id="bool"), pytest.param(1.0, id="equal-float")],
)
async def test_versioned_ollama_contracts_reject_non_integer_versions(
    tmp_path, contract_name, bad_version
):
    _repository, examples = await _versioned_contract_examples(
        tmp_path,
        database_name=f"contract-{contract_name}-{type(bad_version).__name__}.sqlite3",
    )
    value, field, expected = examples[contract_name]
    if expected == 2 and isinstance(bad_version, float):
        bad_version = 2.0

    with pytest.raises(ValueError) as error:
        replace(value, **{field: bad_version})
    assert type(error.value) is ValueError
    assert str(error.value) == _VERSIONED_CONTRACT_ERROR_MESSAGES[contract_name]


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_name", _VERSIONED_CONTRACT_NAMES)
async def test_versioned_ollama_contracts_round_trip_exact_integer_versions(
    tmp_path, contract_name
):
    _repository, examples = await _versioned_contract_examples(
        tmp_path, database_name=f"contract-exact-{contract_name}.sqlite3"
    )
    value, field, expected = examples[contract_name]

    assert type(getattr(value, field)) is int
    assert getattr(value, field) == expected
    assert type(value).from_dict(value.to_dict()) == value


class _UnavailableFirstProvisioningReadbackProvider(FakeOllamaProvider):
    def __init__(self):
        super().__init__(serverless_plan())
        self.readback_available = False

    async def observe_provisioning(self, intent, resource, **kwargs):
        if not self.readback_available:
            raise OllamaProvisioningEvidenceError(
                "original provisioning readback is unavailable"
            )
        return await super().observe_provisioning(intent, resource, **kwargs)


class _BlockedProvisioningProvider(FakeOllamaProvider):
    def __init__(self):
        super().__init__(serverless_plan())
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()

    async def provision(self, prepared):
        self.entered.set()
        await self.finish.wait()
        return await super().provision(prepared)


def _raw_lease_row(database, lease_id):
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease_id,)
        ).fetchone()
    assert row is not None
    return dict(row)


def _fork_repository(repository, tmp_path, *, name, lease_id):
    database = tmp_path / name
    with (
        closing(sqlite3.connect(repository.path)) as source,
        closing(sqlite3.connect(database)) as target,
    ):
        source.backup(target)
    forked = SQLiteOllamaLeaseRepository(database)
    lease = forked.get(lease_id)
    assert lease is not None
    return forked, lease


async def _pending_without_receipt(tmp_path, *, database_name):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / database_name)
    provider = _UnavailableFirstProvisioningReadbackProvider()
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(OllamaProvisioningEvidenceError):
        await service.acquire(
            make_request(clock), plan=serverless_plan(), wait_until_ready=False
        )
    pending = repository.get("lease-001")
    assert pending is not None
    assert pending.provisioning_intent is not None
    assert pending.provider_resource_id is not None
    assert pending.provisioning_receipt is None
    provider.readback_available = True
    return clock, repository, provider, pending


def _provisioning_mismatch(lease, *, provider_resource_id):
    intent = lease.provisioning_intent
    assert intent is not None
    return OllamaProvisioningMismatchEvidence(
        intent_sha256=intent.intent_sha256,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        provider_resource_id=provider_resource_id,
        observed_resource_name=intent.resource_name,
        observed_at=intent.provision_requested_at,
        readback_sha256="d" * 64,
        reason=OllamaProvisioningMismatchReason.RUNTIME,
    )


async def _first_write_version_transition(tmp_path, case):
    if case == "provisioning-intent":
        clock = MutableClock()
        repository = SQLiteOllamaLeaseRepository(tmp_path / "before.sqlite3")
        before, _inserted = repository.insert_request(make_request(clock), now=clock())
        candidate_repository, _candidate = _fork_repository(
            repository, tmp_path, name="candidate.sqlite3", lease_id=before.lease_id
        )
        provider = _BlockedProvisioningProvider()
        service = OllamaLeaseService(
            repository=candidate_repository,
            provider=provider,
            poll_interval_seconds=1,
            clock=clock,
            sleep=clock.sleep,
        )
        task = asyncio.create_task(
            service.acquire(
                make_request(clock), plan=serverless_plan(), wait_until_ready=False
            )
        )
        try:
            await asyncio.wait_for(provider.entered.wait(), timeout=5)
            candidate_row = _raw_lease_row(candidate_repository.path, before.lease_id)
        finally:
            await release_gates_and_boundedly_join_owned_tasks(task, provider.finish)
        task.result()
        return repository, before, candidate_row

    if case in {"billing-observation", "billing-selection"}:
        (
            repository,
            before,
            observed_at,
            observation,
        ) = await _terminated_pod_billing_evidence(
            tmp_path, database_name="before.sqlite3"
        )
        candidate_repository, candidate = _fork_repository(
            repository, tmp_path, name="candidate.sqlite3", lease_id=before.lease_id
        )
        candidate_repository.record_billing_outcome(
            candidate,
            now=observed_at,
            next_check_at=observed_at + timedelta(hours=1),
            observation=observation,
            maximum_observations=8,
            maximum_bytes=100_000,
        )
        return (
            repository,
            before,
            _raw_lease_row(candidate_repository.path, before.lease_id),
        )

    clock, repository, provider, pending = await _pending_without_receipt(
        tmp_path, database_name="before.sqlite3"
    )
    intent = pending.provisioning_intent
    assert intent is not None
    resource = ProvisionedOllamaResource(
        resource_type=intent.resource_type,
        provider_resource_id=pending.provider_resource_id,
        resource_name=intent.resource_name,
    )

    if case == "provisioning-receipt":
        before = pending
        candidate_repository, candidate = _fork_repository(
            repository, tmp_path, name="candidate.sqlite3", lease_id=before.lease_id
        )
        receipt = await provider.observe_provisioning(intent, resource)
        candidate_repository.record_provisioning_outcome(
            candidate, now=clock(), receipt=receipt
        )
    elif case == "provisioning-mismatch":
        before = pending
        candidate_repository, candidate = _fork_repository(
            repository, tmp_path, name="candidate.sqlite3", lease_id=before.lease_id
        )
        candidate_repository.record_provisioning_outcome(
            candidate,
            now=clock(),
            mismatch=_provisioning_mismatch(
                candidate, provider_resource_id=candidate.provider_resource_id
            ),
        )
    elif case == "ambiguous-candidate-mismatch":
        candidate_resource = replace(resource, provider_resource_id="candidate-id")
        before = repository.record_provisioning_outcome(
            pending, now=clock(), ambiguous_candidate=candidate_resource
        )
        candidate_repository, candidate = _fork_repository(
            repository, tmp_path, name="candidate.sqlite3", lease_id=before.lease_id
        )
        candidate_repository.record_provisioning_outcome(
            candidate,
            now=clock(),
            mismatch=_provisioning_mismatch(
                candidate, provider_resource_id="candidate-id"
            ),
            candidate_mismatch=True,
        )
    elif case in {
        "cleanup-intent",
        "cleanup-receipt",
        "additional-target",
        "additional-target-mismatch",
        "additional-cleanup-intent",
        "additional-cleanup-receipt",
        "runtime-observation",
        "runtime-snapshot",
    }:
        receipt = await provider.observe_provisioning(intent, resource)
        ready_for_observation = repository.record_provisioning_outcome(
            pending, now=clock(), receipt=receipt
        )
        if case in {"runtime-observation", "runtime-snapshot"}:
            before = ready_for_observation
            candidate_repository, candidate = _fork_repository(
                repository,
                tmp_path,
                name="candidate.sqlite3",
                lease_id=before.lease_id,
            )
            observation = await provider.observe(resource, intent, receipt)
            assert observation.runtime_telemetry is not None
            candidate_repository.record_runtime_telemetry(
                candidate,
                observation.runtime_telemetry,
                now=clock(),
                maximum_observations=8,
                maximum_bytes=100_000,
            )
        elif case in {"cleanup-intent", "cleanup-receipt"}:
            releasing = repository.compare_and_set(
                ready_for_observation,
                changes={
                    "state": OllamaLeaseState.RELEASING,
                    "teardown_state": OllamaTeardownState.PENDING,
                    "route_url": None,
                    "termination_reason": "test first-write evidence",
                },
            )
            before = releasing
            if case == "cleanup-receipt":
                before = repository.record_cleanup_intent(
                    releasing,
                    now=clock(),
                    termination_reason="test first-write evidence",
                )
            candidate_repository, candidate = _fork_repository(
                repository,
                tmp_path,
                name="candidate.sqlite3",
                lease_id=before.lease_id,
            )
            if case == "cleanup-intent":
                candidate_repository.record_cleanup_intent(
                    candidate,
                    now=clock(),
                    termination_reason="test first-write evidence",
                )
            else:
                cleanup_intent = candidate.cleanup_intent
                assert cleanup_intent is not None
                cleanup_receipt = OllamaCleanupReceipt(
                    cleanup_intent_id=cleanup_intent.cleanup_intent_id,
                    verified_absent_at=(
                        cleanup_intent.delete_started_at + timedelta(microseconds=1)
                    ),
                    basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
                )
                candidate_repository.record_teardown_outcome(
                    candidate,
                    provider_resource_id=candidate.provider_resource_id,
                    now=clock(),
                    accrued_estimated_cost=candidate.accrued_estimated_cost,
                    termination_reason="test first-write evidence",
                    cleanup_receipt=cleanup_receipt,
                    billing_next_check_at=clock() + timedelta(hours=1),
                )
        else:
            late_resource = replace(resource, provider_resource_id="late-original-ack")
            if case == "additional-target":
                before = ready_for_observation
            else:
                with_target = repository.record_provisioning_outcome(
                    ready_for_observation,
                    now=clock(),
                    acknowledged_resource=late_resource,
                )
                before = with_target
                if case == "additional-cleanup-receipt":
                    before = repository.record_additional_cleanup_intent(
                        with_target,
                        provider_resource_id=late_resource.provider_resource_id,
                        now=clock(),
                        termination_reason="test first-write evidence",
                    )
            candidate_repository, candidate = _fork_repository(
                repository,
                tmp_path,
                name="candidate.sqlite3",
                lease_id=before.lease_id,
            )
            if case == "additional-target":
                candidate_repository.record_provisioning_outcome(
                    candidate,
                    now=clock(),
                    acknowledged_resource=late_resource,
                )
            elif case == "additional-target-mismatch":
                candidate_repository.record_provisioning_outcome(
                    candidate,
                    now=clock(),
                    acknowledged_resource=late_resource,
                    mismatch=_provisioning_mismatch(
                        candidate,
                        provider_resource_id=late_resource.provider_resource_id,
                    ),
                )
            elif case == "additional-cleanup-intent":
                candidate_repository.record_additional_cleanup_intent(
                    candidate,
                    provider_resource_id=late_resource.provider_resource_id,
                    now=clock(),
                    termination_reason="test first-write evidence",
                )
            else:
                target = candidate.additional_cleanup_targets[0]
                assert target.cleanup_intent is not None
                candidate_repository.record_additional_cleanup_outcome(
                    candidate,
                    provider_resource_id=late_resource.provider_resource_id,
                    now=clock(),
                    accrued_estimated_cost=candidate.accrued_estimated_cost,
                    termination_reason="test first-write evidence",
                    cleanup_receipt=OllamaCleanupReceipt(
                        cleanup_intent_id=target.cleanup_intent.cleanup_intent_id,
                        verified_absent_at=(
                            target.cleanup_intent.delete_started_at
                            + timedelta(microseconds=1)
                        ),
                        basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
                    ),
                )
    else:
        raise AssertionError(f"Unsupported first-write transition case: {case}")

    return (
        repository,
        before,
        _raw_lease_row(candidate_repository.path, before.lease_id),
    )


_FIRST_WRITE_VERSION_CASES = (
    pytest.param(
        "provisioning-intent",
        "provisioning_intent_json",
        "$.evidence_version",
        "json('true')",
        id="provisioning-intent",
    ),
    pytest.param(
        "provisioning-receipt",
        "provisioning_receipt_json",
        "$.evidence_version",
        "json('true')",
        id="provisioning-receipt",
    ),
    pytest.param(
        "provisioning-mismatch",
        "provisioning_mismatch_json",
        "$.evidence_version",
        "json('true')",
        id="provisioning-mismatch",
    ),
    pytest.param(
        "ambiguous-candidate-mismatch",
        "ambiguous_candidate_mismatch_json",
        "$.evidence_version",
        "json('true')",
        id="ambiguous-candidate-mismatch",
    ),
    pytest.param(
        "cleanup-intent",
        "cleanup_intent_json",
        "$.evidence_version",
        "json('true')",
        id="cleanup-intent",
    ),
    pytest.param(
        "cleanup-receipt",
        "cleanup_receipt_json",
        "$.evidence_version",
        "json('true')",
        id="cleanup-receipt",
    ),
    pytest.param(
        "additional-target",
        "additional_cleanup_targets_json",
        "$[0].evidence_version",
        "json('true')",
        id="additional-target",
    ),
    pytest.param(
        "additional-target-mismatch",
        "additional_cleanup_targets_json",
        "$[0].provisioning_mismatch.evidence_version",
        "json('true')",
        id="additional-target-mismatch",
    ),
    pytest.param(
        "additional-cleanup-intent",
        "additional_cleanup_targets_json",
        "$[0].cleanup_intent.evidence_version",
        "json('true')",
        id="additional-cleanup-intent",
    ),
    pytest.param(
        "additional-cleanup-receipt",
        "additional_cleanup_targets_json",
        "$[0].cleanup_receipt.evidence_version",
        "json('true')",
        id="additional-cleanup-receipt",
    ),
    pytest.param(
        "runtime-observation",
        "runtime_telemetry_observations_json",
        "$[0].evidence_version",
        "json('true')",
        id="runtime-observation",
    ),
    pytest.param(
        "runtime-snapshot",
        "runtime_telemetry_observations_json",
        "$[0].snapshot.schema_version",
        "2.0",
        id="runtime-snapshot-equal-real",
    ),
    pytest.param(
        "billing-observation",
        "lifecycle_billing_observations_json",
        "$[0].evidence_version",
        "json('true')",
        id="billing-observation",
    ),
    pytest.param(
        "billing-selection",
        "lifecycle_billing_selection_json",
        "$.policy_version",
        "json('true')",
        id="billing-selection",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,column,json_path,replacement", _FIRST_WRITE_VERSION_CASES
)
async def test_writer_trigger_rejects_non_integer_first_write_version_types(
    tmp_path, case, column, json_path, replacement
):
    repository, before, candidate_row = await _first_write_version_transition(
        tmp_path, case
    )
    before_row = _raw_lease_row(repository.path, before.lease_id)
    assert candidate_row["lease_id"] == before_row["lease_id"]
    assert candidate_row["revision"] == before_row["revision"] + 1
    refusal = None
    accepted_row = None
    accepted_json_type = None
    with closing(sqlite3.connect(repository.path)) as connection:
        columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_xinfo(ollama_leases)")
            if row[1] != "lease_id"
        )
        assignments = []
        values = []
        for name in columns:
            if name == column:
                assignments.append(f'"{name}" = json_set(?, ?, {replacement})')
                values.extend((candidate_row[name], json_path))
            else:
                assignments.append(f'"{name}" = ?')
                values.append(candidate_row[name])
        values.append(before.lease_id)
        try:
            connection.execute(
                f"UPDATE ollama_leases SET {', '.join(assignments)} WHERE lease_id = ?",
                values,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            refusal = exc
            connection.rollback()
        else:
            accepted_row = _raw_lease_row(repository.path, before.lease_id)
            accepted_json_type = connection.execute(
                f'SELECT json_type("{column}", ?) FROM ollama_leases '
                "WHERE lease_id = ?",
                (json_path, before.lease_id),
            ).fetchone()[0]

    if refusal is not None:
        assert "invalid Ollama " in str(refusal)
        assert " evidence mutation" in str(refusal)
        assert _raw_lease_row(repository.path, before.lease_id) == before_row
    else:
        assert accepted_row is not None
        assert accepted_row[column] != candidate_row[column]
        assert accepted_json_type == ("real" if replacement == "2.0" else "true")
    assert isinstance(refusal, sqlite3.IntegrityError), (
        f"writer trigger admitted non-integer {case} version evidence"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale_outcome",
    [
        pytest.param("provider_error", id="provider-error"),
        pytest.param("exact_receipt", id="exact-receipt"),
    ],
)
async def test_stale_billing_outcome_cannot_poison_a_newer_successful_poll(
    tmp_path, stale_outcome
):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    observation = selected.lifecycle_billing_observations[0]
    stale_snapshot = selected
    newer_repository = SQLiteOllamaLeaseRepository(repository.path)
    newer_received_at = observed_at + timedelta(hours=2)
    newer = newer_repository.record_billing_outcome(
        selected,
        now=newer_received_at,
        next_check_at=newer_received_at + timedelta(hours=1),
        observation=observation,
        maximum_observations=8,
        maximum_bytes=100_000,
    )
    stale_received_at = observed_at + timedelta(hours=1)
    outcome = (
        {"provider_error": "stale provider billing failure"}
        if stale_outcome == "provider_error"
        else {"observation": observation}
    )

    with pytest.raises(OllamaLeaseConflictError, match="billing.*older|stale"):
        repository.record_billing_outcome(
            stale_snapshot,
            now=stale_received_at,
            next_check_at=stale_received_at + timedelta(hours=1),
            maximum_observations=8,
            maximum_bytes=100_000,
            **outcome,
        )

    restarted = SQLiteOllamaLeaseRepository(repository.path).get(selected.lease_id)
    assert restarted == newer
    assert restarted.billing_last_checked_at == newer_received_at
    assert restarted.lifecycle_billing_latest_reference.received_at == (
        newer_received_at
    )


@pytest.mark.asyncio
async def test_same_time_billing_error_replay_preserves_successful_reference(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    original_reference = selected.lifecycle_billing_latest_reference

    replayed = repository.record_billing_outcome(
        selected,
        now=observed_at,
        next_check_at=observed_at + timedelta(hours=1),
        provider_error="same-time provider billing retry",
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert replayed.lifecycle_billing_latest_reference == original_reference
    assert replayed.lifecycle_billing_selection == selected.lifecycle_billing_selection
    assert replayed.lifecycle_billing_observations == (
        selected.lifecycle_billing_observations
    )
    assert replayed.billing_last_checked_at == observed_at
    assert replayed.billing_last_provider_error == "same-time provider billing retry"


@pytest.mark.asyncio
async def test_same_time_success_after_error_refreshes_latest_reference(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    original = selected.lifecycle_billing_observations[0]
    retry_at = observed_at + timedelta(hours=1)
    failed = repository.record_billing_outcome(
        selected,
        now=retry_at,
        next_check_at=retry_at + timedelta(hours=1),
        provider_error="same-time provider billing retry",
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    refreshed = repository.record_billing_outcome(
        failed,
        now=retry_at,
        next_check_at=retry_at + timedelta(hours=1),
        observation=original,
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert refreshed.lifecycle_billing_observations == (original,)
    assert refreshed.lifecycle_billing_corrections == ()
    assert refreshed.lifecycle_billing_selection == selected.lifecycle_billing_selection
    assert refreshed.lifecycle_billing_latest_reference.observation_id == (
        original.observation_id
    )
    assert refreshed.lifecycle_billing_latest_reference.received_at == retry_at
    assert refreshed.billing_last_checked_at == retry_at
    assert refreshed.billing_last_provider_error is None
    assert SQLiteOllamaLeaseRepository(repository.path).get(selected.lease_id) == (
        refreshed
    )


@pytest.mark.asyncio
async def test_same_time_revised_success_after_error_appends_one_correction(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)
    original = selected.lifecycle_billing_observations[0]
    assert original.pod_billing_receipt is not None
    retry_at = observed_at + timedelta(hours=1)
    revised_receipt = replace(
        original.pod_billing_receipt,
        provider_billing_id="pod-billing-same-time-revision",
        actual_cost_usd=Decimal("0.5"),
        reconciled_at=retry_at,
    )
    revised = replace(
        original,
        provider_capacity_actual_cost_usd=Decimal("0.5"),
        first_observed_at=retry_at,
        pod_billing_receipt=revised_receipt,
    )
    failed = repository.record_billing_outcome(
        selected,
        now=retry_at,
        next_check_at=retry_at + timedelta(hours=1),
        provider_error="same-time provider billing retry",
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    corrected = repository.record_billing_outcome(
        failed,
        now=retry_at,
        next_check_at=retry_at + timedelta(hours=1),
        observation=revised,
        maximum_observations=8,
        maximum_bytes=100_000,
    )

    assert corrected.lifecycle_billing_observations == (original, revised)
    assert len(corrected.lifecycle_billing_corrections) == 1
    assert corrected.lifecycle_billing_selection == selected.lifecycle_billing_selection
    assert corrected.lifecycle_billing_latest_reference.observation_id == (
        revised.observation_id
    )
    assert corrected.lifecycle_billing_latest_reference.received_at == retry_at
    assert corrected.billing_last_checked_at == retry_at
    assert corrected.billing_last_provider_error is None
    assert SQLiteOllamaLeaseRepository(repository.path).get(selected.lease_id) == (
        corrected
    )


@pytest.mark.asyncio
async def test_invalid_candidate_row_is_decoded_before_repository_commit(tmp_path):
    repository, selected, observed_at = await _selected_pod_billing_lease(tmp_path)

    with pytest.raises(RunPodManagerError, match="billing latest reference"):
        repository.compare_and_set(
            selected,
            changes={
                "billing_last_checked_at": observed_at - timedelta(microseconds=1),
            },
        )

    restarted = SQLiteOllamaLeaseRepository(repository.path).get(selected.lease_id)
    assert restarted == selected


def _install_writer_contract_poison(database, *, event, lease_id):
    """Fault-inject a post-write decode failure without invalid public DTOs.

    This deliberately removes the canonical current-writer update guard before
    installing the observer. It is an atomicity control, not a valid-schema
    input incident.
    """

    quoted_lease_id = lease_id.replace("'", "''")
    update_trigger = f"ollama_writer_update_v{_CURRENT_WRITER_CONTRACT_VERSION}"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(f"DROP TRIGGER {update_trigger}")
        connection.execute(
            f"""CREATE TRIGGER test_poison_{event.lower()}ed_writer_version
                AFTER {event} ON ollama_leases
                WHEN NEW.lease_id = '{quoted_lease_id}'
                BEGIN
                  UPDATE ollama_leases
                     SET writer_contract_version = 99
                   WHERE lease_id = NEW.lease_id;
                END"""
        )


@pytest.mark.asyncio
async def test_provisioning_outcome_rolls_back_when_updated_row_cannot_decode(
    tmp_path,
):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "provisioning-decode.sqlite3")
    provider = _UnavailableFirstProvisioningReadbackProvider()
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(OllamaProvisioningEvidenceError):
        await service.acquire(
            make_request(clock), plan=serverless_plan(), wait_until_ready=False
        )
    pending = repository.get("lease-001")
    assert pending is not None
    assert pending.provisioning_intent is not None
    assert pending.provider_resource_id is not None
    assert pending.provisioning_receipt is None
    provider.readback_available = True
    receipt = await provider.observe_provisioning(
        pending.provisioning_intent,
        ProvisionedOllamaResource(
            resource_type=pending.resource_type,
            provider_resource_id=pending.provider_resource_id,
            resource_name=pending.resource_name,
        ),
    )
    before = _raw_lease_row(repository.path, pending.lease_id)
    _install_writer_contract_poison(
        repository.path, event="UPDATE", lease_id=pending.lease_id
    )

    with pytest.raises(RunPodManagerError, match="writer contract"):
        repository.record_provisioning_outcome(
            pending,
            now=clock(),
            receipt=receipt,
        )

    assert _raw_lease_row(repository.path, pending.lease_id) == before
    assert repository.get(pending.lease_id) == pending


@pytest.mark.asyncio
async def test_teardown_outcome_rolls_back_when_updated_row_cannot_decode(
    tmp_path,
):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "teardown-decode.sqlite3")
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
            "termination_reason": "test decode before teardown commit",
        },
    )
    admitted = repository.record_cleanup_intent(
        releasing,
        now=clock(),
        termination_reason="test decode before teardown commit",
    )
    assert admitted.cleanup_intent is not None
    cleanup_receipt = OllamaCleanupReceipt(
        cleanup_intent_id=admitted.cleanup_intent.cleanup_intent_id,
        verified_absent_at=(
            admitted.cleanup_intent.delete_started_at + timedelta(microseconds=1)
        ),
        basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
    )
    before = _raw_lease_row(repository.path, admitted.lease_id)
    _install_writer_contract_poison(
        repository.path, event="UPDATE", lease_id=admitted.lease_id
    )

    with pytest.raises(RunPodManagerError, match="writer contract"):
        repository.record_teardown_outcome(
            admitted,
            provider_resource_id=admitted.provider_resource_id,
            now=clock(),
            accrued_estimated_cost=admitted.accrued_estimated_cost,
            termination_reason="test decode before teardown commit",
            cleanup_receipt=cleanup_receipt,
            billing_next_check_at=clock() + timedelta(hours=1),
        )

    assert _raw_lease_row(repository.path, admitted.lease_id) == before
    assert repository.get(admitted.lease_id) == admitted


@pytest.mark.asyncio
async def test_reconciliation_claim_rolls_back_when_updated_row_cannot_decode(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "claim-decode.sqlite3")
    provider = FakeOllamaProvider(serverless_plan())
    service = OllamaLeaseService(
        repository=repository,
        provider=provider,
        poll_interval_seconds=1,
        clock=clock,
        sleep=clock.sleep,
    )
    ready = await service.acquire(make_request(clock), plan=serverless_plan())
    durable_before = repository.get(ready.lease_id)
    assert durable_before is not None
    assert ready.route_url is not None
    assert ready.provider_health_url is not None
    assert durable_before.route_url is None
    assert durable_before.provider_health_url is None
    before = _raw_lease_row(repository.path, ready.lease_id)
    _install_writer_contract_poison(
        repository.path, event="UPDATE", lease_id=ready.lease_id
    )

    with pytest.raises(RunPodManagerError, match="writer contract"):
        repository.claim_for_reconciliation(
            ready,
            now=clock(),
            retry_at=clock() + timedelta(seconds=1),
        )

    assert _raw_lease_row(repository.path, ready.lease_id) == before
    assert repository.get(ready.lease_id) == durable_before


def test_request_insert_rolls_back_when_inserted_row_cannot_decode(tmp_path):
    clock = MutableClock()
    repository = SQLiteOllamaLeaseRepository(tmp_path / "insert-decode.sqlite3")
    update_trigger = f"ollama_writer_update_v{_CURRENT_WRITER_CONTRACT_VERSION}"
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        connection.execute(f"DROP TRIGGER {update_trigger}")
        connection.execute(
            """CREATE TRIGGER test_poison_inserted_writer_version
               AFTER INSERT ON ollama_leases
               BEGIN
                 UPDATE ollama_leases
                    SET writer_contract_version = 99
                  WHERE lease_id = NEW.lease_id;
               END"""
        )

    with pytest.raises(RunPodManagerError, match="writer contract"):
        repository.insert_request(make_request(clock), now=clock())

    with closing(sqlite3.connect(repository.path)) as connection:
        assert (
            connection.execute("SELECT count(*) FROM ollama_leases").fetchone()[0] == 0
        )


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


def test_current_writer_contract_v7_is_installed_exactly(tmp_path):
    clock = MutableClock()
    database = tmp_path / "writer-v7.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    lease, inserted = repository.insert_request(make_request(clock), now=clock())
    assert inserted is True

    with closing(sqlite3.connect(database)) as connection:
        default = next(
            row[4]
            for row in connection.execute("PRAGMA table_xinfo(ollama_leases)")
            if row[1] == "writer_contract_version"
        )
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'ollama_leases'"
            )
        }

    assert lease.writer_contract_version == 7
    assert default == "7"
    assert triggers == {
        "ollama_writer_insert_v7",
        "ollama_writer_update_v7",
        "ollama_writer_delete_v7",
        "ollama_provisioning_evidence_insert_v7",
        "ollama_provisioning_evidence_update_v7",
        "ollama_lifecycle_evidence_update_v7",
    }


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
    with OwnedSpawnProcesses(
        tuple(processes), gates=(gate,), queues=(results,)
    ) as ownership:
        gate.set()
        ownership.join()
        for process in processes:
            assert not process.is_alive()
            assert process.exitcode == 0
        observed = [results.get(timeout=2) for _process in processes]
    assert observed.count(None) == 1
    assert len([revision for revision in observed if revision is not None]) == 1


def test_owned_processes_preserve_body_failure_and_join_every_child():
    context = get_context("spawn")
    gate = context.Event()
    processes = tuple(
        context.Process(target=_wait_for_owned_process_release, args=(gate,))
        for _index in range(2)
    )
    ownership = OwnedSpawnProcesses(processes, gates=(gate,), queues=())

    with pytest.raises(AssertionError, match="body failure is primary"), ownership:
        raise AssertionError("body failure is primary")

    assert ownership.cleanup_failures == []
    assert ownership.forced_cleanup == []
    assert ownership.final_exitcodes == [0, 0]
    assert ownership.closed_processes == 2


def test_owned_processes_join_first_child_when_second_start_fails():
    context = get_context("spawn")
    gate = context.Event()

    def unpickleable_local_target():
        pytest.fail("unpickleable child target unexpectedly ran")

    first = context.Process(target=_wait_for_owned_process_release, args=(gate,))
    second = context.Process(target=unpickleable_local_target)
    ownership = OwnedSpawnProcesses((first, second), gates=(gate,), queues=())

    with (
        pytest.raises((AttributeError, PicklingError), match="pickle") as caught,
        ownership,
    ):
        pytest.fail("unpickleable second process unexpectedly started")

    assert "unpickleable_local_target" in str(caught.value)
    assert ownership.cleanup_failures == []
    assert ownership.forced_cleanup == []
    assert ownership.final_exitcodes == [0]
    assert ownership.closed_processes == 1


def test_owned_processes_report_forced_termination_as_failure():
    context = get_context("spawn")
    process = context.Process(target=_never_finishing_owned_process)
    ownership = OwnedSpawnProcesses((process,), gates=(), queues=(), join_timeout=0.1)

    with pytest.raises(AssertionError, match="forced termination"), ownership:
        pass

    assert len(ownership.forced_cleanup) == 1
    assert ownership.forced_cleanup[0] is not None
    assert len(ownership.final_exitcodes) == 1
    assert ownership.closed_processes == 1


@pytest.mark.parametrize(
    "body_type",
    [
        pytest.param(RuntimeError, id="ordinary-control"),
        pytest.param(GeneratorExit, id="generator-close-control"),
    ],
)
def test_owned_processes_compose_body_and_all_cleanup_failures(body_type):
    context = get_context("spawn")
    gate = context.Event()
    cleanup_error = RuntimeError("owned gate release failed")
    events = []

    class FailingGate:
        def set(self):
            events.append("gate-set")
            gate.set()
            raise cleanup_error

    process = context.Process(target=_wait_for_owned_process_release, args=(gate,))
    ownership = OwnedSpawnProcesses((process,), gates=(FailingGate(),), queues=())
    body, control, explicit = execution_control_carrier(
        body_type("owned body failed"), grouped=True
    )
    forced_rescue = ()
    try:
        with pytest.raises(BaseExceptionGroup) as caught, ownership:
            raise body
    finally:
        forced_rescue = _rescue_test_processes((process,), gate)

    assert caught.value.exceptions == (body, cleanup_error)
    assert is_execution_control_error(caught.value)
    assert body.__cause__ is explicit
    assert explicit.exceptions[0] is control
    assert events == ["gate-set"]
    assert forced_rescue == ()
    assert ownership.final_exitcodes == [0]
    assert ownership.closed_processes == 1


def test_owned_processes_attempt_later_children_after_baseexception_group():
    context = get_context("spawn")
    first_gate = context.Event()
    second_gate = context.Event()
    ordinary = ValueError("first gate cleanup failed")
    grouped = BaseExceptionGroup("first gate group", [ordinary])
    events = []

    class FailingFirstGate:
        def set(self):
            events.append("first-gate")
            first_gate.set()
            raise grouped

    class ObservedSecondGate:
        def set(self):
            events.append("second-gate")
            second_gate.set()

    processes = (
        context.Process(target=_wait_for_owned_process_release, args=(first_gate,)),
        context.Process(target=_wait_for_owned_process_release, args=(second_gate,)),
    )
    ownership = OwnedSpawnProcesses(
        processes,
        gates=(FailingFirstGate(), ObservedSecondGate()),
        queues=(),
    )
    forced_rescue = ()
    try:
        with pytest.raises(BaseExceptionGroup) as caught, ownership:
            pass
    finally:
        forced_rescue = _rescue_test_processes(
            processes,
            first_gate,
            second_gate,
        )

    assert caught.value is grouped
    assert events == ["first-gate", "second-gate"]
    assert second_gate.is_set()
    assert ownership.final_exitcodes == [0, 0]
    assert ownership.closed_processes == 2
    assert forced_rescue == ()


def test_owned_processes_attempt_queue_join_after_close_failure():
    context = get_context("spawn")
    results = context.Queue()
    events = []
    close_error = ValueError("parent queue close failed")

    class FailingQueue:
        def close(self):
            events.append("queue-close")
            results.close()
            raise close_error

        def join_thread(self):
            events.append("queue-join")
            results.join_thread()

    process = context.Process(target=_put_owned_process_result, args=(results,))
    ownership = OwnedSpawnProcesses((process,), gates=(), queues=(FailingQueue(),))
    with pytest.raises(ValueError) as caught, ownership:
        ownership.join()
        assert results.get(timeout=2) == "owned-result"

    assert caught.value is close_error
    assert events == ["queue-close", "queue-join"]
    assert ownership.final_exitcodes == [0]
    assert ownership.closed_processes == 1
    assert ownership.queues_closed is True


def test_owned_processes_compose_second_start_and_prior_child_cleanup_failure():
    context = get_context("spawn")
    gate = context.Event()
    cleanup_error = RuntimeError("first child gate cleanup failed")

    class FailingGate:
        def set(self):
            gate.set()
            raise cleanup_error

    def unpickleable_local_target():
        pytest.fail("unpickleable child target unexpectedly ran")

    first = context.Process(target=_wait_for_owned_process_release, args=(gate,))
    second = context.Process(target=unpickleable_local_target)
    ownership = OwnedSpawnProcesses((first, second), gates=(FailingGate(),), queues=())
    forced_rescue = ()
    try:
        with (
            pytest.raises((AttributeError, PicklingError), match="pickle") as caught,
            ownership,
        ):
            pytest.fail("unpickleable second process unexpectedly started")
    finally:
        forced_rescue = _rescue_test_processes((first, second), gate)

    assert type(caught.value) in {AttributeError, PicklingError}
    assert "unpickleable_local_target" in str(caught.value)
    assert caught.value.__cause__ is cleanup_error
    assert ownership.cleanup_failures == [cleanup_error]
    assert ownership.final_exitcodes == [0]
    assert ownership.closed_processes == 1
    assert forced_rescue == ()


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
    assert current.writer_contract_version == 7
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("ALTER TABLE ollama_leases DROP COLUMN maximum_compute_cost")

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        database, attestation=_drain_attestation(database)
    )

    lease = migrated.get("lease-001")
    assert lease.provisioning_evidence_state is OllamaProvisioningEvidenceState.REQUIRED
    assert lease.writer_contract_version == 7


@pytest.mark.asyncio
async def test_attested_v7_trigger_repair_preserves_every_current_fact(tmp_path):
    repository, selected, _observed_at = await _selected_pod_billing_lease(tmp_path)
    before = selected.to_dict()
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        connection.execute("DROP TRIGGER ollama_lifecycle_evidence_update_v7")

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


@pytest.mark.asyncio
async def test_synthetic_v6_shape_migrates_to_v7_with_only_fence_counters_changed(
    tmp_path,
):
    """Exercise the migrator shape; native e091 proves authentic v6 separately."""

    repository, selected, _observed_at = await _selected_pod_billing_lease(
        tmp_path, database_name="synthetic-v6-to-v7.sqlite3"
    )
    with closing(sqlite3.connect(repository.path)) as connection, connection:
        for trigger in _WRITER_TRIGGERS:
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute("UPDATE ollama_leases SET writer_contract_version = 6")
        _rewrite_writer_contract_default(connection, 6)
    predecessor = _raw_lease_row(repository.path, selected.lease_id)
    before_files = _database_files(repository.path)

    with pytest.raises(RunPodManagerError, match="drained-writer migration"):
        SQLiteOllamaLeaseRepository(repository.path)
    assert _database_files(repository.path) == before_files

    migrated = SQLiteOllamaLeaseRepository.migrate_legacy_database(
        repository.path,
        attestation=_drain_attestation(repository.path),
    )
    retained = migrated.get(selected.lease_id)
    assert retained is not None
    migrated_row = _raw_lease_row(repository.path, selected.lease_id)
    expected = dict(predecessor)
    expected.update(
        {
            "writer_contract_version": 7,
            "revision": predecessor["revision"] + 1,
            "writer_epoch": predecessor["writer_epoch"] + 1,
        }
    )
    assert migrated_row == expected
    assert retained.writer_contract_version == 7
    assert retained.revision == selected.revision + 1
    assert retained.writer_epoch == selected.writer_epoch + 1


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
    assert retained.writer_contract_version == 7
    assert retained.ambiguous_candidate_mismatch is None


def test_attested_v2_to_v7_migration_retains_authority_but_not_cost_proof(tmp_path):
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
    assert retained.writer_contract_version == 7
    assert retained.runtime_telemetry_state is OllamaRuntimeTelemetryState.UNKNOWN
    assert retained.runtime_telemetry_observations == ()
    assert retained.lifecycle_billing_state is OllamaLifecycleBillingState.UNKNOWN
    assert retained.lifecycle_billing_observations == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer_version", "downgrade"),
    ((4, _downgrade_to_v4_writer), (5, _downgrade_to_v5_schema)),
)
async def test_v4_or_v5_to_v7_migration_preserves_every_lifecycle_fact(
    tmp_path, writer_version, downgrade
):
    repository, selected, observed_at = await _selected_pod_billing_lease(
        tmp_path,
        actual=Decimal("0.12345678901234567890123456789"),
        database_name=f"v{writer_version}-to-v7.sqlite3",
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
    assert retained.writer_contract_version == 7
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


def test_v3_requires_drain_then_migrates_to_v7_without_restamping_authority(
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
    assert retained.writer_contract_version == 7
    assert retained.provisioning_evidence_state is original.provisioning_evidence_state
    assert retained.reconcile_after == original.updated_at
    assert retained.additional_cleanup_targets == ()
    assert retained.runtime_telemetry_last_provider_error is None
    assert retained.billing_retention_exhausted is False


@pytest.mark.asyncio
async def test_v3_retention_sentinel_migrates_to_explicit_v7_retention_state(
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
    assert retained.writer_contract_version == 7
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


def test_v1_writer_contract_requires_real_v7_drain_and_cannot_regain_authority(
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
    assert retained.writer_contract_version == 7
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


def test_v1_drain_attestation_cannot_target_the_v7_contract(tmp_path):
    database = tmp_path / "wrong-target.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    repository.insert_request(make_request(MutableClock()), now=MutableClock()())
    before = _database_files(database)

    with pytest.raises(ValueError, match="Unsupported Ollama writer contract"):
        replace(_drain_attestation(database), target_writer_contract_version=1)

    assert _database_files(database) == before


def test_v7_runtime_rejects_unknown_future_writer_contract_without_repair(tmp_path):
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
        name = "ollama_provisioning_evidence_insert_v7"
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
        writer.execute("DROP TRIGGER ollama_provisioning_evidence_insert_v7")
        writer.execute("DROP TRIGGER ollama_provisioning_evidence_update_v7")
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
        writer.execute("DROP TRIGGER ollama_writer_delete_v7")
        writer.execute(
            """CREATE TRIGGER ollama_writer_delete_v7
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

    assert lease.writer_contract_version == 7
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


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        pytest.param("state", "not-a-lease-state", id="invalid-state"),
        pytest.param("updated_at", "not-an-aware-timestamp", id="invalid-timestamp"),
    ],
)
def test_historical_migration_decodes_every_candidate_before_destructive_ddl(
    tmp_path, column, invalid_value
):
    database = tmp_path / f"invalid-historical-{column}.sqlite3"
    _legacy_database(database, state="ready", saved_route=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            f"UPDATE ollama_leases SET {column} = ? WHERE lease_id = ?",
            (invalid_value, "lease-001"),
        )
    before = _raw_database_snapshot(database)

    with pytest.raises(RunPodManagerError, match="previous writer evidence is invalid"):
        SQLiteOllamaLeaseRepository.migrate_legacy_database(
            database, attestation=_drain_attestation(database)
        )

    assert _raw_database_snapshot(database) == before
    with closing(sqlite3.connect(database)) as connection:
        names = {
            row[1] for row in connection.execute("PRAGMA table_xinfo(ollama_leases)")
        }
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "writer_contract_version" not in names
    assert "_ollama_leases_drained_source" not in tables


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

    assert all(lease.writer_contract_version == 7 for lease in leases)
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
    with OwnedSpawnProcesses(
        tuple(processes), gates=(gate,), queues=(results,)
    ) as ownership:
        gate.set()
        ownership.join()
        for process in processes:
            assert not process.is_alive()
            assert process.exitcode == 0
        observed = sorted(results.get(timeout=2) for _process in processes)
    assert observed == [
        ("ok", 7, "historical_unknown"),
        ("ok", 7, "historical_unknown"),
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


def test_startup_route_scrub_decodes_every_changed_row_before_commit(tmp_path):
    """Fault-inject an undecodable row to prove the whole scrub rolls back."""

    database = tmp_path / "route-scrub-decode.sqlite3"
    repository = SQLiteOllamaLeaseRepository(database)
    clock = MutableClock()
    original, _ = repository.insert_request(make_request(clock), now=clock())
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.row_factory = sqlite3.Row
        columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_xinfo(ollama_leases)")
        )
        source = dict(
            connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?",
                (original.lease_id,),
            ).fetchone()
        )
        source.update(
            {
                "lease_id": "undecodable-route-lease",
                "workload_id": "undecodable-route-workload",
                "mode": "invalid-mode",
            }
        )
        connection.execute(
            f"INSERT INTO ollama_leases ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _column in columns)})",
            tuple(source[column] for column in columns),
        )
        connection.execute(
            """UPDATE ollama_leases
                  SET route_url = 'https://private.invalid',
                      provider_health_url = 'https://private.invalid/ping',
                      revision = revision + 1,
                      writer_epoch = writer_epoch + 1"""
        )
        before = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM ollama_leases ORDER BY lease_id"
            ).fetchall()
        )

    with pytest.raises(ValueError, match="invalid-mode.*OllamaLeaseMode"):
        SQLiteOllamaLeaseRepository(database)

    with closing(sqlite3.connect(database)) as connection:
        after = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM ollama_leases ORDER BY lease_id"
            ).fetchall()
        )
    assert after == before


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
        connection.execute("DROP TRIGGER ollama_writer_update_v7")
        connection.execute("DROP TRIGGER ollama_provisioning_evidence_update_v7")
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
        connection.execute("DROP TRIGGER ollama_lifecycle_evidence_update_v7")
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
