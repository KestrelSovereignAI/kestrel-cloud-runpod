"""Transactional SQLite persistence for private-Ollama leases."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping
from contextlib import AbstractContextManager, closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from .billing import exact_decimal_integer_scale, exact_decimal_sum
from .models import CloudType, RunPodManagerError
from .ollama_contracts import (
    OllamaAdditionalCleanupTarget,
    OllamaCleanupIntent,
    OllamaCleanupReceipt,
    OllamaLease,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaLeaseState,
    OllamaLifecycleBillingCorrection,
    OllamaLifecycleBillingLatestReference,
    OllamaLifecycleBillingLatestState,
    OllamaLifecycleBillingObservation,
    OllamaLifecycleBillingSelection,
    OllamaLifecycleBillingState,
    OllamaNonComputeCostComponent,
    OllamaProvisioningEvidenceState,
    OllamaProvisioningIntent,
    OllamaProvisioningMismatchEvidence,
    OllamaProvisioningReceipt,
    OllamaResourceConstraints,
    OllamaResourceType,
    OllamaRuntimeTelemetryObservation,
    OllamaRuntimeTelemetryState,
    OllamaTeardownState,
    ProvisionedOllamaResource,
    iso_datetime,
    normalize_new_ollama_request_money,
    require_aware,
    runtime_telemetry_binding_for_intent,
)
from .sqlite_store import lease_database_connection

_CURRENT_WRITER_CONTRACT_VERSION = 6
_BILLING_RETENTION_MANUAL_ERROR = (
    "billing evidence retention requires manual reconciliation"
)
_WRITER_TRIGGER_NAMES = (
    "ollama_writer_insert_v6",
    "ollama_writer_update_v6",
    "ollama_writer_delete_v6",
    "ollama_provisioning_evidence_insert_v6",
    "ollama_provisioning_evidence_update_v6",
    "ollama_lifecycle_evidence_update_v6",
)
_PREVIOUS_WRITER_TRIGGER_NAMES = (
    "ollama_writer_insert_v5",
    "ollama_writer_update_v5",
    "ollama_writer_delete_v5",
    "ollama_provisioning_evidence_insert_v5",
    "ollama_provisioning_evidence_update_v5",
    "ollama_lifecycle_evidence_update_v5",
    "ollama_writer_insert_v4",
    "ollama_writer_update_v4",
    "ollama_writer_delete_v4",
    "ollama_provisioning_evidence_insert_v4",
    "ollama_provisioning_evidence_update_v4",
    "ollama_lifecycle_evidence_update_v4",
    "ollama_writer_insert_v3",
    "ollama_writer_update_v3",
    "ollama_writer_delete_v3",
    "ollama_provisioning_evidence_insert_v3",
    "ollama_provisioning_evidence_update_v3",
    "ollama_lifecycle_evidence_update_v3",
    "ollama_writer_insert_v2",
    "ollama_writer_update_v2",
    "ollama_writer_delete_v2",
    "ollama_provisioning_evidence_insert_v2",
    "ollama_provisioning_evidence_update_v2",
    "ollama_writer_insert_v1",
    "ollama_writer_update_v1",
    "ollama_writer_delete_v1",
    "ollama_provisioning_evidence_insert_v1",
    "ollama_provisioning_evidence_update_v1",
)
_MIGRATABLE_WRITER_TRIGGER_NAMES = frozenset(
    (*_WRITER_TRIGGER_NAMES, *_PREVIOUS_WRITER_TRIGGER_NAMES)
)
_WRITER_CONTRACT_COLUMNS = frozenset(
    {
        "writer_contract_version",
        "writer_epoch",
        "reconcile_after",
        "provisioning_evidence_state",
        "provisioning_intent_json",
        "provisioning_receipt_json",
        "provisioning_mismatch_json",
        "ambiguous_candidate_mismatch_json",
        "ambiguous_create_candidate_id",
        "original_requested_at",
        "original_readiness_timeout_seconds",
        "runtime_telemetry_binding_sha256",
        "cleanup_intent_json",
        "cleanup_receipt_json",
        "additional_cleanup_targets_json",
        "runtime_telemetry_state",
        "runtime_telemetry_observations_json",
        "runtime_telemetry_last_provider_error",
        "lifecycle_billing_state",
        "lifecycle_billing_observations_json",
        "lifecycle_billing_selection_json",
        "lifecycle_billing_corrections_json",
        "lifecycle_billing_latest_reference_json",
        "billing_last_checked_at",
        "billing_next_check_at",
        "billing_last_provider_error",
        "billing_retention_exhausted",
    }
)
_V5_WRITER_CONTRACT_COLUMNS = _WRITER_CONTRACT_COLUMNS - {
    "lifecycle_billing_latest_reference_json",
}
_V3_WRITER_CONTRACT_COLUMNS = _V5_WRITER_CONTRACT_COLUMNS - {
    "reconcile_after",
    "additional_cleanup_targets_json",
    "runtime_telemetry_last_provider_error",
    "billing_retention_exhausted",
}
_V2_WRITER_CONTRACT_COLUMNS = _V3_WRITER_CONTRACT_COLUMNS - {
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
}
_V1_WRITER_CONTRACT_COLUMNS = _V2_WRITER_CONTRACT_COLUMNS - {
    "ambiguous_candidate_mismatch_json"
}
_LEGACY_ADDITIVE_COLUMNS = frozenset(
    {
        "termination_reason",
        "estimated_compute_cost",
        "maximum_compute_cost",
        "estimated_non_compute_cost",
        "maximum_non_compute_cost",
        "cost_ceiling",
        "cost_policy_components_json",
        "maximum_concurrent_workers",
        "maximum_billable_seconds",
        "placement_gpu_count",
        *_WRITER_CONTRACT_COLUMNS,
    }
)


@dataclass(frozen=True)
class _LeaseColumn:
    name: str
    data_type: str
    not_null: bool = False
    default_sql: str | None = None
    primary_key: bool = False

    def declaration(self) -> str:
        parts = [self.name, self.data_type]
        if self.not_null:
            parts.append("NOT NULL")
        if self.default_sql is not None:
            parts.extend(("DEFAULT", self.default_sql))
        if self.primary_key:
            parts.append("PRIMARY KEY")
        return " ".join(parts)

    def signature(self) -> tuple[str, bool, str | None, bool, int]:
        return self.data_type, self.not_null, self.default_sql, self.primary_key, 0


_LEASE_COLUMNS = (
    _LeaseColumn("lease_id", "TEXT", primary_key=True),
    _LeaseColumn("owner_id", "TEXT", not_null=True),
    _LeaseColumn("workload_id", "TEXT", not_null=True),
    _LeaseColumn("request_fingerprint", "TEXT", not_null=True),
    _LeaseColumn("original_requested_at", "TEXT"),
    _LeaseColumn("original_readiness_timeout_seconds", "INTEGER"),
    _LeaseColumn("model", "TEXT", not_null=True),
    _LeaseColumn("constraints_json", "TEXT", not_null=True),
    _LeaseColumn("mode", "TEXT"),
    _LeaseColumn("resource_type", "TEXT"),
    _LeaseColumn("provider_resource_id", "TEXT"),
    _LeaseColumn("ambiguous_create_candidate_id", "TEXT"),
    _LeaseColumn("resource_name", "TEXT"),
    _LeaseColumn("creation_uncertain", "INTEGER", not_null=True, default_sql="0"),
    _LeaseColumn("provision_attempt_id", "TEXT"),
    _LeaseColumn("provision_attempts", "INTEGER", not_null=True, default_sql="0"),
    _LeaseColumn("route_url", "TEXT"),
    _LeaseColumn("provider_health_url", "TEXT"),
    _LeaseColumn("state", "TEXT", not_null=True),
    _LeaseColumn("teardown_state", "TEXT", not_null=True),
    _LeaseColumn("created_at", "TEXT", not_null=True),
    _LeaseColumn("updated_at", "TEXT", not_null=True),
    _LeaseColumn("provisioning_started_at", "TEXT"),
    _LeaseColumn("ready_at", "TEXT"),
    _LeaseColumn("last_used_at", "TEXT", not_null=True),
    _LeaseColumn("idle_deadline", "TEXT", not_null=True),
    _LeaseColumn("hard_deadline", "TEXT", not_null=True),
    _LeaseColumn("readiness_deadline", "TEXT", not_null=True),
    _LeaseColumn("model_pull_started_at", "TEXT"),
    _LeaseColumn("model_pull_attempts", "INTEGER", not_null=True, default_sql="0"),
    _LeaseColumn("model_ready_at", "TEXT"),
    _LeaseColumn("expected_session_seconds", "INTEGER", not_null=True),
    _LeaseColumn("expected_active_seconds", "INTEGER", not_null=True),
    _LeaseColumn("serverless_initialization_seconds", "INTEGER", not_null=True),
    _LeaseColumn("serverless_idle_tail_seconds", "INTEGER", not_null=True),
    _LeaseColumn("idle_timeout_seconds", "INTEGER", not_null=True),
    _LeaseColumn("offered_rate_per_hr", "REAL"),
    _LeaseColumn("placement_gpu_count", "INTEGER", not_null=True, default_sql="1"),
    _LeaseColumn("estimated_cost", "REAL"),
    _LeaseColumn("estimated_compute_cost", "REAL"),
    _LeaseColumn("maximum_compute_cost", "REAL"),
    _LeaseColumn("estimated_non_compute_cost", "REAL"),
    _LeaseColumn("maximum_non_compute_cost", "REAL"),
    _LeaseColumn("cost_ceiling", "REAL"),
    _LeaseColumn("cost_policy_components_json", "TEXT"),
    _LeaseColumn("maximum_concurrent_workers", "INTEGER"),
    _LeaseColumn("estimated_billable_seconds", "INTEGER"),
    _LeaseColumn("maximum_billable_seconds", "INTEGER"),
    _LeaseColumn("accrued_estimated_cost", "REAL", not_null=True, default_sql="0"),
    _LeaseColumn("max_authorized_cost", "REAL", not_null=True),
    _LeaseColumn("cold_start_seconds", "REAL"),
    _LeaseColumn("selected_gpu_id", "TEXT"),
    _LeaseColumn("selected_gpu_pool", "TEXT"),
    _LeaseColumn("selected_gpu_name", "TEXT"),
    _LeaseColumn("catalog_observed_at", "TEXT"),
    _LeaseColumn("last_provider_error", "TEXT"),
    _LeaseColumn("termination_reason", "TEXT"),
    _LeaseColumn("teardown_attempts", "INTEGER", not_null=True, default_sql="0"),
    _LeaseColumn("revision", "INTEGER", not_null=True, default_sql="0"),
    _LeaseColumn("writer_contract_version", "INTEGER", not_null=True, default_sql="6"),
    _LeaseColumn("writer_epoch", "INTEGER", not_null=True, default_sql="0"),
    _LeaseColumn("reconcile_after", "TEXT"),
    _LeaseColumn(
        "provisioning_evidence_state",
        "TEXT",
        not_null=True,
        default_sql="'required'",
    ),
    _LeaseColumn("provisioning_intent_json", "TEXT"),
    _LeaseColumn("provisioning_receipt_json", "TEXT"),
    _LeaseColumn("provisioning_mismatch_json", "TEXT"),
    _LeaseColumn("ambiguous_candidate_mismatch_json", "TEXT"),
    _LeaseColumn("runtime_telemetry_binding_sha256", "TEXT"),
    _LeaseColumn("cleanup_intent_json", "TEXT"),
    _LeaseColumn("cleanup_receipt_json", "TEXT"),
    _LeaseColumn(
        "additional_cleanup_targets_json",
        "TEXT",
        not_null=True,
        default_sql="'[]'",
    ),
    _LeaseColumn(
        "runtime_telemetry_state", "TEXT", not_null=True, default_sql="'unknown'"
    ),
    _LeaseColumn(
        "runtime_telemetry_observations_json",
        "TEXT",
        not_null=True,
        default_sql="'[]'",
    ),
    _LeaseColumn("runtime_telemetry_last_provider_error", "TEXT"),
    _LeaseColumn(
        "lifecycle_billing_state", "TEXT", not_null=True, default_sql="'unknown'"
    ),
    _LeaseColumn(
        "lifecycle_billing_observations_json",
        "TEXT",
        not_null=True,
        default_sql="'[]'",
    ),
    _LeaseColumn("lifecycle_billing_selection_json", "TEXT"),
    _LeaseColumn(
        "lifecycle_billing_corrections_json",
        "TEXT",
        not_null=True,
        default_sql="'[]'",
    ),
    _LeaseColumn(
        "lifecycle_billing_latest_reference_json",
        "TEXT",
        not_null=True,
        default_sql=(
            '\'{"observation_id":null,"received_at":null,"state":"not_observed"}\''
        ),
    ),
    _LeaseColumn("billing_last_checked_at", "TEXT"),
    _LeaseColumn("billing_next_check_at", "TEXT"),
    _LeaseColumn("billing_last_provider_error", "TEXT"),
    _LeaseColumn(
        "billing_retention_exhausted", "INTEGER", not_null=True, default_sql="0"
    ),
)
_LEASE_COLUMN_BY_NAME = {column.name: column for column in _LEASE_COLUMNS}
_CREATE_LEASE_TABLE_SQL = (
    "CREATE TABLE ollama_leases ("
    + ", ".join(column.declaration() for column in _LEASE_COLUMNS)
    + ")"
)
_RECONCILE_INDEX_NAME = "idx_ollama_leases_reconcile"
_RECONCILE_INDEX_SQL = (
    "CREATE INDEX idx_ollama_leases_reconcile "
    "ON ollama_leases(reconcile_after, state, provisioning_evidence_state, "
    "lifecycle_billing_state, billing_next_check_at, idle_deadline, hard_deadline)"
)


@dataclass(frozen=True)
class OllamaWriterDrainAttestation:
    """Operator evidence that the exact legacy DB has no admitted old writers.

    This typed value records an operator-controlled deployment receipt.  The
    SDK cannot observe whether old processes, routes, or HTTP effects have
    drained, and therefore never manufactures this attestation itself.
    """

    database_path: Path
    drained_at: datetime
    drain_reference_sha256: str
    target_writer_contract_version: int = _CURRENT_WRITER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.database_path, Path)
            or not self.database_path.is_absolute()
        ):
            raise ValueError(
                "Ollama writer drain attestation requires an absolute path"
            )
        require_aware(self.drained_at, "writer drain attestation")
        if (
            not isinstance(self.drain_reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.drain_reference_sha256) is None
        ):
            raise ValueError("Ollama writer drain reference must be a SHA-256 digest")
        if self.target_writer_contract_version != _CURRENT_WRITER_CONTRACT_VERSION:
            raise ValueError("Unsupported Ollama writer contract migration target")


def _complete_sql_statements(script: str) -> tuple[str, ...]:
    pending = ""
    statements: list[str] = []
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending.strip())
            pending = ""
    if pending.strip():
        raise RunPodManagerError("Incomplete Ollama writer contract SQL")
    return tuple(statements)


def _execute_sql_script_transactionally(
    connection: sqlite3.Connection, script: str
) -> None:
    """Execute complete SQLite statements without ``executescript``'s commit."""

    for statement in _complete_sql_statements(script):
        connection.execute(statement)


def _writer_trigger_script() -> str:
    return """
        CREATE TRIGGER ollama_writer_insert_v6
        BEFORE INSERT ON ollama_leases
        WHEN NEW.writer_contract_version IS NOT 6
          OR NEW.writer_epoch IS NOT 1
          OR NEW.revision IS NOT 0
          OR NEW.reconcile_after IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'current Ollama writer contract required');
        END;

        CREATE TRIGGER ollama_writer_update_v6
        BEFORE UPDATE ON ollama_leases
        WHEN OLD.writer_contract_version IS NOT 6
          OR NEW.writer_contract_version IS NOT OLD.writer_contract_version
          OR NEW.writer_epoch IS NOT OLD.writer_epoch + 1
          OR NEW.revision IS NOT OLD.revision + 1
        BEGIN
            SELECT RAISE(ABORT, 'stale Ollama writer');
        END;

        CREATE TRIGGER ollama_writer_delete_v6
        BEFORE DELETE ON ollama_leases
        BEGIN
            SELECT RAISE(ABORT, 'Ollama lease deletion is unsupported');
        END;

        CREATE TRIGGER ollama_provisioning_evidence_insert_v6
        BEFORE INSERT ON ollama_leases
        WHEN NEW.state IS NOT 'requested'
          OR NEW.provisioning_evidence_state IS NOT 'required'
          OR NEW.original_readiness_timeout_seconds IS NULL
          OR NEW.original_readiness_timeout_seconds < 1
          OR NEW.provisioning_intent_json IS NOT NULL
          OR NEW.provisioning_receipt_json IS NOT NULL
          OR NEW.provisioning_mismatch_json IS NOT NULL
          OR NEW.ambiguous_candidate_mismatch_json IS NOT NULL
          OR NEW.provider_resource_id IS NOT NULL
          OR NEW.ambiguous_create_candidate_id IS NOT NULL
          OR NEW.resource_type IS NOT NULL
          OR NEW.resource_name IS NOT NULL
          OR NEW.provision_attempt_id IS NOT NULL
          OR NEW.provisioning_started_at IS NOT NULL
          OR NEW.creation_uncertain IS NOT 0
          OR NEW.runtime_telemetry_binding_sha256 IS NOT NULL
          OR NEW.cleanup_intent_json IS NOT NULL
          OR NEW.cleanup_receipt_json IS NOT NULL
          OR NEW.additional_cleanup_targets_json IS NOT '[]'
          OR NEW.runtime_telemetry_state IS NOT 'unknown'
          OR NEW.runtime_telemetry_observations_json IS NOT '[]'
          OR NEW.runtime_telemetry_last_provider_error IS NOT NULL
          OR NEW.lifecycle_billing_state IS NOT 'unknown'
          OR NEW.lifecycle_billing_observations_json IS NOT '[]'
          OR NEW.lifecycle_billing_selection_json IS NOT NULL
          OR NEW.lifecycle_billing_corrections_json IS NOT '[]'
          OR NEW.lifecycle_billing_latest_reference_json IS NOT
             '{"observation_id":null,"received_at":null,"state":"not_observed"}'
          OR NEW.billing_last_checked_at IS NOT NULL
          OR NEW.billing_next_check_at IS NOT NULL
          OR NEW.billing_last_provider_error IS NOT NULL
          OR NEW.billing_retention_exhausted IS NOT 0
        BEGIN
            SELECT RAISE(ABORT, 'new Ollama lease must begin without provider evidence');
        END;

        CREATE TRIGGER ollama_provisioning_evidence_update_v6
        BEFORE UPDATE ON ollama_leases
        WHEN
          (OLD.provisioning_evidence_state = 'required'
             AND NEW.provisioning_evidence_state NOT IN ('required', 'intent_recorded'))
          OR (OLD.provisioning_evidence_state = 'intent_recorded'
             AND NEW.provisioning_evidence_state NOT IN ('intent_recorded', 'complete'))
          OR (OLD.provisioning_evidence_state = 'complete'
             AND NEW.provisioning_evidence_state IS NOT 'complete')
          OR (OLD.provisioning_evidence_state = 'historical_unknown'
             AND NEW.provisioning_evidence_state IS NOT 'historical_unknown')
          OR (NEW.provisioning_evidence_state IN ('required', 'historical_unknown')
             AND (NEW.provisioning_intent_json IS NOT NULL
                  OR NEW.provisioning_receipt_json IS NOT NULL
                  OR NEW.provisioning_mismatch_json IS NOT NULL
                  OR NEW.ambiguous_candidate_mismatch_json IS NOT NULL))
          OR (NEW.provisioning_evidence_state = 'required'
             AND (NEW.provider_resource_id IS NOT NULL
                  OR NEW.ambiguous_create_candidate_id IS NOT NULL
                  OR NEW.resource_type IS NOT NULL
                  OR NEW.resource_name IS NOT NULL
                  OR NEW.provision_attempt_id IS NOT NULL
                  OR NEW.provisioning_started_at IS NOT NULL
                  OR NEW.creation_uncertain IS NOT 0))
          OR (NEW.provisioning_evidence_state = 'intent_recorded'
             AND NEW.provisioning_receipt_json IS NOT NULL)
          OR (NEW.provisioning_evidence_state = 'intent_recorded'
             AND NEW.state NOT IN ('provisioning', 'reconcile_required',
                                   'releasing', 'terminated', 'failed'))
          OR (NEW.provisioning_evidence_state = 'complete'
             AND NEW.state NOT IN ('waiting_for_model', 'ready',
                                   'releasing', 'terminated', 'failed'))
          OR (NEW.provisioning_evidence_state IN ('intent_recorded', 'complete')
             AND (NEW.provisioning_intent_json IS NULL
                  OR json_valid(NEW.provisioning_intent_json) IS NOT 1
                  OR json_extract(NEW.provisioning_intent_json, '$.evidence_version') IS NOT 1
                  OR json_extract(NEW.provisioning_intent_json, '$.lease_id') IS NOT NEW.lease_id
                  OR json_extract(NEW.provisioning_intent_json, '$.request_fingerprint') IS NOT NEW.request_fingerprint
                  OR json_extract(NEW.provisioning_intent_json, '$.provision_attempt_id') IS NOT NEW.provision_attempt_id
                  OR json_extract(NEW.provisioning_intent_json, '$.mode') IS NOT NEW.mode
                  OR json_extract(NEW.provisioning_intent_json, '$.resource_type') IS NOT NEW.resource_type
                  OR json_extract(NEW.provisioning_intent_json, '$.resource_name') IS NOT NEW.resource_name
                  OR json_extract(NEW.provisioning_intent_json, '$.provision_requested_at') IS NOT NEW.provisioning_started_at
                  OR json_extract(NEW.provisioning_intent_json, '$.selected_gpu_id') IS NOT NEW.selected_gpu_id
                  OR json_extract(NEW.provisioning_intent_json, '$.selected_gpu_pool') IS NOT NEW.selected_gpu_pool
                  OR json_extract(NEW.provisioning_intent_json, '$.selected_gpu_name') IS NOT NEW.selected_gpu_name
                  OR json_extract(NEW.provisioning_intent_json, '$.selected_gpu_count') IS NOT NEW.placement_gpu_count))
          OR (NEW.provisioning_evidence_state = 'complete'
             AND (NEW.provisioning_receipt_json IS NULL
                  OR json_valid(NEW.provisioning_receipt_json) IS NOT 1
                  OR json_extract(NEW.provisioning_receipt_json, '$.evidence_version') IS NOT 1
                  OR json_extract(NEW.provisioning_receipt_json, '$.lease_id') IS NOT NEW.lease_id
                  OR json_extract(NEW.provisioning_receipt_json, '$.request_fingerprint') IS NOT NEW.request_fingerprint
                  OR json_extract(NEW.provisioning_receipt_json, '$.provision_attempt_id') IS NOT NEW.provision_attempt_id
                  OR json_extract(NEW.provisioning_receipt_json, '$.resource_type') IS NOT NEW.resource_type
                  OR json_extract(NEW.provisioning_receipt_json, '$.provider_resource_id') IS NOT NEW.provider_resource_id
                  OR json_extract(NEW.provisioning_receipt_json, '$.resource_name') IS NOT NEW.resource_name
                  OR NEW.provider_resource_id IS NULL
                  OR (json_extract(NEW.provisioning_receipt_json, '$.reconciled_after_ambiguous_create') IS 1
                      AND (NEW.ambiguous_create_candidate_id IS NULL
                           OR NEW.provider_resource_id IS NOT NEW.ambiguous_create_candidate_id))
                  OR NEW.creation_uncertain IS NOT 0))
          OR (NEW.provisioning_mismatch_json IS NOT NULL
             AND (NEW.provisioning_evidence_state NOT IN
                     ('intent_recorded', 'complete')
                  OR NEW.provisioning_intent_json IS NULL
                  OR NEW.provider_resource_id IS NULL
                  OR json_valid(NEW.provisioning_mismatch_json) IS NOT 1
                  OR json_extract(NEW.provisioning_mismatch_json, '$.evidence_version') IS NOT 1
                  OR json_extract(NEW.provisioning_mismatch_json, '$.lease_id') IS NOT NEW.lease_id
                  OR json_extract(NEW.provisioning_mismatch_json, '$.provision_attempt_id') IS NOT NEW.provision_attempt_id
                  OR json_extract(NEW.provisioning_mismatch_json, '$.resource_type') IS NOT NEW.resource_type
                  OR json_extract(NEW.provisioning_mismatch_json, '$.provider_resource_id')
                     IS NOT NEW.provider_resource_id))
          OR (NEW.ambiguous_candidate_mismatch_json IS NOT NULL
             AND (NEW.provisioning_evidence_state NOT IN ('intent_recorded', 'complete')
                  OR NEW.provisioning_intent_json IS NULL
                  OR NEW.ambiguous_create_candidate_id IS NULL
                  OR json_valid(NEW.ambiguous_candidate_mismatch_json) IS NOT 1
                  OR json_extract(NEW.ambiguous_candidate_mismatch_json, '$.evidence_version') IS NOT 1
                  OR json_extract(NEW.ambiguous_candidate_mismatch_json, '$.lease_id') IS NOT NEW.lease_id
                  OR json_extract(NEW.ambiguous_candidate_mismatch_json, '$.provision_attempt_id') IS NOT NEW.provision_attempt_id
                  OR json_extract(NEW.ambiguous_candidate_mismatch_json, '$.resource_type') IS NOT NEW.resource_type
                  OR json_extract(NEW.ambiguous_candidate_mismatch_json, '$.provider_resource_id')
                     IS NOT NEW.ambiguous_create_candidate_id))
          OR (NEW.ambiguous_create_candidate_id IS NOT NULL
             AND NEW.provisioning_evidence_state NOT IN
                 ('intent_recorded', 'complete', 'historical_unknown'))
          OR (NEW.state IN ('provisioning', 'reconcile_required')
             AND NEW.provisioning_evidence_state NOT IN ('intent_recorded', 'complete'))
          OR (NEW.state IN ('waiting_for_model', 'ready')
             AND NEW.provisioning_evidence_state IS NOT 'complete')
          OR (OLD.provisioning_evidence_state = 'historical_unknown'
             AND NEW.state NOT IN ('releasing', 'terminated', 'failed'))
          OR (OLD.provisioning_intent_json IS NOT NULL
             AND NEW.provisioning_intent_json IS NOT OLD.provisioning_intent_json)
          OR (OLD.provisioning_receipt_json IS NOT NULL
             AND NEW.provisioning_receipt_json IS NOT OLD.provisioning_receipt_json)
          OR (OLD.provisioning_mismatch_json IS NOT NULL
             AND NEW.provisioning_mismatch_json IS NOT OLD.provisioning_mismatch_json)
          OR (OLD.ambiguous_candidate_mismatch_json IS NOT NULL
             AND NEW.ambiguous_candidate_mismatch_json
                 IS NOT OLD.ambiguous_candidate_mismatch_json)
          OR (OLD.provider_resource_id IS NOT NULL
             AND NEW.provider_resource_id IS NOT OLD.provider_resource_id)
          OR (OLD.ambiguous_create_candidate_id IS NOT NULL
             AND NEW.ambiguous_create_candidate_id IS NOT OLD.ambiguous_create_candidate_id)
          OR (OLD.provisioning_intent_json IS NOT NULL AND (
                NEW.lease_id IS NOT OLD.lease_id
             OR NEW.request_fingerprint IS NOT OLD.request_fingerprint
             OR NEW.mode IS NOT OLD.mode
             OR NEW.resource_type IS NOT OLD.resource_type
             OR NEW.resource_name IS NOT OLD.resource_name
             OR NEW.provision_attempt_id IS NOT OLD.provision_attempt_id
             OR NEW.provisioning_started_at IS NOT OLD.provisioning_started_at
             OR NEW.selected_gpu_id IS NOT OLD.selected_gpu_id
             OR NEW.selected_gpu_pool IS NOT OLD.selected_gpu_pool
             OR NEW.selected_gpu_name IS NOT OLD.selected_gpu_name
             OR NEW.placement_gpu_count IS NOT OLD.placement_gpu_count
             OR NEW.offered_rate_per_hr IS NOT OLD.offered_rate_per_hr
             OR NEW.catalog_observed_at IS NOT OLD.catalog_observed_at
             OR NEW.runtime_telemetry_binding_sha256
                IS NOT OLD.runtime_telemetry_binding_sha256
          ))
          OR NEW.original_requested_at IS NOT OLD.original_requested_at
          OR NEW.original_readiness_timeout_seconds
             IS NOT OLD.original_readiness_timeout_seconds
          OR NEW.reconcile_after IS NULL
          OR NEW.owner_id IS NOT OLD.owner_id
          OR NEW.workload_id IS NOT OLD.workload_id
          OR NEW.request_fingerprint IS NOT OLD.request_fingerprint
          OR NEW.model IS NOT OLD.model
          OR NEW.constraints_json IS NOT OLD.constraints_json
          OR NEW.expected_session_seconds IS NOT OLD.expected_session_seconds
          OR NEW.expected_active_seconds IS NOT OLD.expected_active_seconds
          OR NEW.serverless_initialization_seconds
             IS NOT OLD.serverless_initialization_seconds
          OR NEW.serverless_idle_tail_seconds
             IS NOT OLD.serverless_idle_tail_seconds
          OR NEW.idle_timeout_seconds IS NOT OLD.idle_timeout_seconds
          OR NEW.hard_deadline IS NOT OLD.hard_deadline
          OR NEW.readiness_deadline IS NOT OLD.readiness_deadline
          OR NEW.max_authorized_cost IS NOT OLD.max_authorized_cost
          OR (OLD.state = 'requested' AND NEW.state = 'requested'
             AND NEW.mode IS NOT OLD.mode)
        BEGIN
            SELECT RAISE(ABORT, 'invalid Ollama provisioning evidence mutation');
        END;

        CREATE TRIGGER ollama_lifecycle_evidence_update_v6
        BEFORE UPDATE ON ollama_leases
        WHEN
          (NEW.cleanup_intent_json IS NOT NULL AND (
                NEW.provisioning_evidence_state NOT IN ('intent_recorded', 'complete')
             OR NEW.provisioning_intent_json IS NULL
             OR NEW.provider_resource_id IS NULL
             OR json_valid(NEW.cleanup_intent_json) IS NOT 1
             OR json_extract(NEW.cleanup_intent_json, '$.evidence_version') IS NOT 1
             OR json_extract(NEW.cleanup_intent_json, '$.lease_id') IS NOT NEW.lease_id
             OR json_extract(NEW.cleanup_intent_json, '$.request_fingerprint')
                IS NOT NEW.request_fingerprint
             OR json_extract(NEW.cleanup_intent_json, '$.provision_attempt_id')
                IS NOT NEW.provision_attempt_id
             OR json_extract(NEW.cleanup_intent_json, '$.resource_type')
                IS NOT NEW.resource_type
             OR json_extract(NEW.cleanup_intent_json, '$.provider_resource_id')
                IS NOT NEW.provider_resource_id
             OR (json_extract(NEW.cleanup_intent_json,
                              '$.provisioning_receipt_id') IS NOT NULL
                 AND (NEW.provisioning_receipt_json IS NULL
                      OR json_extract(NEW.cleanup_intent_json,
                                      '$.provisioning_receipt_id')
                         IS NOT json_extract(NEW.provisioning_receipt_json,
                                             '$.receipt_id')))
          ))
          OR (OLD.cleanup_intent_json IS NOT NULL
             AND NEW.cleanup_intent_json IS NOT OLD.cleanup_intent_json)
          OR (NEW.cleanup_receipt_json IS NOT NULL AND (
                NEW.cleanup_intent_json IS NULL
             OR json_valid(NEW.cleanup_receipt_json) IS NOT 1
             OR json_extract(NEW.cleanup_receipt_json, '$.evidence_version') IS NOT 1
             OR json_extract(NEW.cleanup_receipt_json, '$.cleanup_intent_id')
                IS NOT json_extract(NEW.cleanup_intent_json, '$.cleanup_intent_id')
             OR NOT (
                  (NEW.state = 'terminated'
                   AND NEW.teardown_state = 'complete')
                  OR (NEW.state = 'releasing'
                      AND NEW.teardown_state IN ('pending', 'retryable_failure')
                      AND json_array_length(
                            NEW.additional_cleanup_targets_json
                          ) > 0)
             )
          ))
          OR (OLD.cleanup_receipt_json IS NOT NULL
             AND NEW.cleanup_receipt_json IS NOT OLD.cleanup_receipt_json)
          OR json_valid(NEW.additional_cleanup_targets_json) IS NOT 1
          OR json_type(NEW.additional_cleanup_targets_json) IS NOT 'array'
          OR json_array_length(NEW.additional_cleanup_targets_json) > 1
          OR json_array_length(NEW.additional_cleanup_targets_json)
             < json_array_length(OLD.additional_cleanup_targets_json)
          OR EXISTS (
               SELECT 1 FROM json_each(OLD.additional_cleanup_targets_json) old_item
               LEFT JOIN json_each(NEW.additional_cleanup_targets_json) new_item
                 ON new_item.key = old_item.key
              WHERE new_item.value IS NULL
                 OR json_extract(new_item.value, '$.evidence_version')
                    IS NOT json_extract(old_item.value, '$.evidence_version')
                 OR json_extract(new_item.value, '$.target_id')
                    IS NOT json_extract(old_item.value, '$.target_id')
                 OR json_extract(new_item.value, '$.intent_sha256')
                    IS NOT json_extract(old_item.value, '$.intent_sha256')
                 OR json_extract(new_item.value, '$.lease_id')
                    IS NOT json_extract(old_item.value, '$.lease_id')
                 OR json_extract(new_item.value, '$.provision_attempt_id')
                    IS NOT json_extract(old_item.value, '$.provision_attempt_id')
                 OR json_extract(new_item.value, '$.resource_type')
                    IS NOT json_extract(old_item.value, '$.resource_type')
                 OR json_extract(new_item.value, '$.resource_name')
                    IS NOT json_extract(old_item.value, '$.resource_name')
                 OR json_extract(new_item.value, '$.provider_resource_id')
                    IS NOT json_extract(old_item.value, '$.provider_resource_id')
                 OR json_extract(new_item.value, '$.source')
                    IS NOT json_extract(old_item.value, '$.source')
                 OR (json_extract(old_item.value, '$.provisioning_mismatch')
                         IS NOT NULL
                     AND json_extract(new_item.value, '$.provisioning_mismatch')
                         IS NOT json_extract(old_item.value,
                                             '$.provisioning_mismatch'))
                 OR (json_extract(old_item.value, '$.cleanup_intent') IS NOT NULL
                     AND json_extract(new_item.value, '$.cleanup_intent')
                         IS NOT json_extract(old_item.value, '$.cleanup_intent'))
                 OR (json_extract(old_item.value, '$.cleanup_receipt') IS NOT NULL
                     AND json_extract(new_item.value, '$.cleanup_receipt')
                         IS NOT json_extract(old_item.value, '$.cleanup_receipt'))
          )
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.additional_cleanup_targets_json) item
                WHERE json_type(item.value) IS NOT 'object'
                   OR json_extract(item.value, '$.evidence_version') IS NOT 1
                   OR json_extract(item.value, '$.intent_sha256')
                      IS NOT json_extract(NEW.provisioning_receipt_json,
                                          '$.intent_sha256')
                   OR json_extract(item.value, '$.lease_id') IS NOT NEW.lease_id
                   OR json_extract(item.value, '$.provision_attempt_id')
                      IS NOT NEW.provision_attempt_id
                   OR json_extract(item.value, '$.resource_type')
                      IS NOT NEW.resource_type
                   OR json_extract(item.value, '$.resource_name')
                      IS NOT NEW.resource_name
                   OR json_extract(item.value, '$.provider_resource_id') IS NULL
                   OR json_extract(item.value, '$.provider_resource_id')
                      IS NEW.provider_resource_id
                   OR json_extract(item.value, '$.source')
                      IS NOT 'original_post_ack'
                   OR (json_extract(item.value, '$.provisioning_mismatch')
                         IS NOT NULL
                       AND json_extract(item.value,
                             '$.provisioning_mismatch.provider_resource_id')
                           IS NOT json_extract(item.value,
                                               '$.provider_resource_id'))
                   OR (json_extract(item.value, '$.cleanup_intent') IS NOT NULL
                       AND json_extract(item.value,
                                        '$.cleanup_intent.provider_resource_id')
                           IS NOT json_extract(item.value,
                                               '$.provider_resource_id'))
                   OR (json_extract(item.value, '$.cleanup_receipt') IS NOT NULL
                       AND (json_extract(item.value, '$.cleanup_intent') IS NULL
                            OR json_extract(item.value,
                                 '$.cleanup_receipt.cleanup_intent_id')
                               IS NOT json_extract(item.value,
                                 '$.cleanup_intent.cleanup_intent_id')))
          )
          OR (NEW.state = 'terminated'
             AND EXISTS (
                  SELECT 1
                    FROM json_each(NEW.additional_cleanup_targets_json) item
                   WHERE json_extract(item.value, '$.cleanup_receipt') IS NULL
             ))
          OR json_valid(NEW.runtime_telemetry_observations_json) IS NOT 1
          OR json_type(NEW.runtime_telemetry_observations_json) IS NOT 'array'
          OR json_array_length(NEW.runtime_telemetry_observations_json)
             < json_array_length(OLD.runtime_telemetry_observations_json)
          OR EXISTS (
               SELECT 1 FROM json_each(OLD.runtime_telemetry_observations_json) old_item
               LEFT JOIN json_each(NEW.runtime_telemetry_observations_json) new_item
                 ON new_item.key = old_item.key
              WHERE new_item.value IS NOT old_item.value
          )
          OR NEW.runtime_telemetry_state NOT IN
             ('unknown', 'startup_complete', 'partial', 'complete')
          OR (OLD.runtime_telemetry_state = 'partial'
             AND NEW.runtime_telemetry_state IS NOT 'partial')
          OR (OLD.runtime_telemetry_state = 'complete'
             AND NEW.runtime_telemetry_state IS NOT 'complete')
          OR (NEW.runtime_telemetry_state IN ('startup_complete', 'complete')
             AND json_array_length(NEW.runtime_telemetry_observations_json) < 1)
          OR (NEW.runtime_telemetry_state = 'complete' AND (
                NEW.cleanup_intent_json IS NULL
             OR NEW.state NOT IN ('releasing', 'terminated')
             OR json_extract(
                  NEW.runtime_telemetry_observations_json,
                  '$[#-1].snapshot.inference_in_flight'
                ) IS NOT 0
          ))
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.runtime_telemetry_observations_json) item
                WHERE json_type(item.value) IS NOT 'object'
                   OR json_extract(item.value, '$.observation_id') IS NULL
                   OR json_extract(item.value, '$.lease_id') IS NOT NEW.lease_id
                   OR json_extract(item.value, '$.provision_attempt_id')
                      IS NOT NEW.provision_attempt_id
                   OR json_extract(item.value, '$.provider_resource_id')
                      IS NOT NEW.provider_resource_id
                   OR json_extract(item.value, '$.snapshot.telemetry_binding_sha256')
                      IS NOT NEW.runtime_telemetry_binding_sha256
          )
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.runtime_telemetry_observations_json) first
                 JOIN json_each(NEW.runtime_telemetry_observations_json) duplicate
                   ON duplicate.key > first.key
                  AND json_extract(duplicate.value, '$.observation_id')
                      = json_extract(first.value, '$.observation_id')
          )
          OR json_valid(NEW.lifecycle_billing_observations_json) IS NOT 1
          OR json_type(NEW.lifecycle_billing_observations_json) IS NOT 'array'
          OR json_array_length(NEW.lifecycle_billing_observations_json)
             < json_array_length(OLD.lifecycle_billing_observations_json)
          OR EXISTS (
               SELECT 1 FROM json_each(OLD.lifecycle_billing_observations_json) old_item
               LEFT JOIN json_each(NEW.lifecycle_billing_observations_json) new_item
                 ON new_item.key = old_item.key
              WHERE new_item.value IS NOT old_item.value
          )
          OR NEW.lifecycle_billing_state NOT IN
             ('unknown', 'not_applicable', 'pending', 'selected',
              'correction_pending')
          OR (OLD.lifecycle_billing_state = 'not_applicable'
             AND NEW.lifecycle_billing_state IS NOT 'not_applicable')
          OR (OLD.lifecycle_billing_state = 'pending'
             AND NEW.lifecycle_billing_state NOT IN ('pending', 'selected'))
          OR (OLD.lifecycle_billing_state = 'selected'
             AND NEW.lifecycle_billing_state NOT IN
                 ('selected', 'correction_pending'))
          OR (OLD.lifecycle_billing_state = 'correction_pending'
             AND NEW.lifecycle_billing_state IS NOT 'correction_pending')
          OR (NEW.lifecycle_billing_state = 'unknown' AND (
                json_array_length(NEW.lifecycle_billing_observations_json) != 0
             OR NEW.lifecycle_billing_selection_json IS NOT NULL
             OR json_array_length(NEW.lifecycle_billing_corrections_json) != 0
             OR json_extract(NEW.lifecycle_billing_latest_reference_json,
                             '$.state') IS NOT 'not_observed'
          ))
          OR (NEW.lifecycle_billing_state = 'not_applicable' AND (
                NEW.writer_contract_version IS NOT 6
             OR NEW.provisioning_evidence_state IS NOT 'required'
             OR NEW.provisioning_intent_json IS NOT NULL
             OR NEW.provisioning_receipt_json IS NOT NULL
             OR NEW.provider_resource_id IS NOT NULL
             OR NEW.ambiguous_create_candidate_id IS NOT NULL
             OR NEW.provision_attempt_id IS NOT NULL
             OR NEW.provisioning_started_at IS NOT NULL
             OR NEW.provision_attempts IS NOT 0
             OR NEW.creation_uncertain IS NOT 0
             OR NEW.cleanup_intent_json IS NOT NULL
             OR NEW.cleanup_receipt_json IS NOT NULL
             OR NEW.state IS NOT 'terminated'
             OR NEW.teardown_state IS NOT 'complete'
             OR json_array_length(NEW.lifecycle_billing_observations_json) != 0
             OR NEW.lifecycle_billing_selection_json IS NOT NULL
             OR json_array_length(NEW.lifecycle_billing_corrections_json) != 0
             OR json_extract(NEW.lifecycle_billing_latest_reference_json,
                             '$.state') IS NOT 'not_observed'
          ))
          OR (NEW.lifecycle_billing_state IN
                ('pending', 'selected', 'correction_pending')
             AND json_array_length(NEW.additional_cleanup_targets_json) = 0
             AND (NEW.state IS NOT 'terminated'
                  OR NEW.cleanup_receipt_json IS NULL
                  OR NEW.billing_next_check_at IS NULL))
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.lifecycle_billing_observations_json) item
                WHERE json_type(item.value) IS NOT 'object'
                   OR json_extract(item.value, '$.observation_id') IS NULL
                   OR json_extract(item.value, '$.lease_id') IS NOT NEW.lease_id
                   OR json_extract(item.value, '$.provision_attempt_id')
                      IS NOT NEW.provision_attempt_id
                   OR json_extract(item.value, '$.provider_resource_id')
                      IS NOT NEW.provider_resource_id
                   OR json_extract(item.value, '$.cleanup_receipt_id')
                      IS NOT json_extract(NEW.cleanup_receipt_json,
                                          '$.cleanup_receipt_id')
          )
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.lifecycle_billing_observations_json) first
                 JOIN json_each(NEW.lifecycle_billing_observations_json) duplicate
                   ON duplicate.key > first.key
                  AND json_extract(duplicate.value, '$.observation_id')
                      = json_extract(first.value, '$.observation_id')
          )
          OR (NEW.lifecycle_billing_selection_json IS NOT NULL AND (
                json_valid(NEW.lifecycle_billing_selection_json) IS NOT 1
             OR json_extract(NEW.lifecycle_billing_selection_json,
                             '$.observation_id') IS NULL
             OR NOT EXISTS (
                  SELECT 1
                    FROM json_each(NEW.lifecycle_billing_observations_json)
                   WHERE json_extract(value, '$.observation_id')
                         = json_extract(NEW.lifecycle_billing_selection_json,
                                        '$.observation_id')
             )
          ))
          OR (OLD.lifecycle_billing_selection_json IS NOT NULL
             AND NEW.lifecycle_billing_selection_json
                 IS NOT OLD.lifecycle_billing_selection_json)
          OR json_valid(NEW.lifecycle_billing_corrections_json) IS NOT 1
          OR json_type(NEW.lifecycle_billing_corrections_json) IS NOT 'array'
          OR json_array_length(NEW.lifecycle_billing_corrections_json)
             < json_array_length(OLD.lifecycle_billing_corrections_json)
          OR EXISTS (
               SELECT 1 FROM json_each(OLD.lifecycle_billing_corrections_json) old_item
               LEFT JOIN json_each(NEW.lifecycle_billing_corrections_json) new_item
                 ON new_item.key = old_item.key
              WHERE new_item.value IS NOT old_item.value
          )
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.lifecycle_billing_corrections_json) item
                WHERE json_type(item.value) IS NOT 'object'
                   OR json_extract(item.value, '$.correction_id') IS NULL
                   OR NEW.lifecycle_billing_selection_json IS NULL
                   OR json_extract(item.value, '$.selected_observation_id')
                      IS NOT json_extract(NEW.lifecycle_billing_selection_json,
                                          '$.observation_id')
                   OR NOT EXISTS (
                        SELECT 1
                          FROM json_each(NEW.lifecycle_billing_observations_json) obs
                         WHERE json_extract(obs.value, '$.observation_id')
                               = json_extract(item.value,
                                              '$.revised_observation_id')
                   )
          )
          OR EXISTS (
               SELECT 1
                 FROM json_each(NEW.lifecycle_billing_corrections_json) first
                 JOIN json_each(NEW.lifecycle_billing_corrections_json) duplicate
                   ON duplicate.key > first.key
                  AND json_extract(duplicate.value, '$.correction_id')
                      = json_extract(first.value, '$.correction_id')
          )
          OR json_valid(NEW.lifecycle_billing_latest_reference_json) IS NOT 1
          OR json_type(NEW.lifecycle_billing_latest_reference_json) IS NOT 'object'
          OR (SELECT count(*)
                FROM json_each(NEW.lifecycle_billing_latest_reference_json)) IS NOT 3
          OR json_extract(NEW.lifecycle_billing_latest_reference_json, '$.state')
             NOT IN ('not_observed', 'historical_unknown', 'known')
          OR (json_extract(NEW.lifecycle_billing_latest_reference_json, '$.state')
                 IN ('not_observed', 'historical_unknown')
             AND (json_type(NEW.lifecycle_billing_latest_reference_json,
                            '$.observation_id') IS NOT 'null'
                  OR json_type(NEW.lifecycle_billing_latest_reference_json,
                               '$.received_at') IS NOT 'null'))
          OR (json_extract(NEW.lifecycle_billing_latest_reference_json, '$.state')
                 = 'known'
             AND (json_type(NEW.lifecycle_billing_latest_reference_json,
                            '$.observation_id') IS NOT 'text'
                  OR json_type(NEW.lifecycle_billing_latest_reference_json,
                               '$.received_at') IS NOT 'text'
                  OR NOT EXISTS (
                       SELECT 1
                         FROM json_each(NEW.lifecycle_billing_observations_json)
                        WHERE json_extract(value, '$.observation_id') =
                              json_extract(
                                NEW.lifecycle_billing_latest_reference_json,
                                '$.observation_id'
                              )
                  )))
          OR (json_extract(OLD.lifecycle_billing_latest_reference_json, '$.state')
                 = 'known'
             AND json_extract(NEW.lifecycle_billing_latest_reference_json, '$.state')
                 IS NOT 'known')
          OR (json_extract(NEW.lifecycle_billing_latest_reference_json, '$.state')
                 = 'historical_unknown'
             AND json_extract(OLD.lifecycle_billing_latest_reference_json, '$.state')
                 IS NOT 'historical_unknown')
          OR (NEW.lifecycle_billing_latest_reference_json
                 IS NOT OLD.lifecycle_billing_latest_reference_json
             AND (json_extract(NEW.lifecycle_billing_latest_reference_json, '$.state')
                    IS NOT 'known'
                  OR NEW.billing_last_provider_error IS NOT NULL
                  OR NEW.billing_last_checked_at IS OLD.billing_last_checked_at
                  OR json_extract(NEW.lifecycle_billing_latest_reference_json,
                                  '$.received_at')
                     IS NOT NEW.billing_last_checked_at))
          OR (NEW.lifecycle_billing_state IN ('selected', 'correction_pending')
             AND (NEW.lifecycle_billing_selection_json IS NULL
                  OR json_extract(NEW.lifecycle_billing_latest_reference_json,
                                  '$.state') NOT IN
                     ('known', 'historical_unknown')))
          OR (NEW.lifecycle_billing_state = 'pending' AND (
                json_array_length(NEW.lifecycle_billing_observations_json) != 0
             OR NEW.lifecycle_billing_selection_json IS NOT NULL
             OR json_array_length(NEW.lifecycle_billing_corrections_json) != 0
             OR json_extract(NEW.lifecycle_billing_latest_reference_json,
                             '$.state') IS NOT 'not_observed'
          ))
          OR (NEW.lifecycle_billing_state = 'selected'
             AND json_array_length(NEW.lifecycle_billing_corrections_json) != 0)
          OR (NEW.lifecycle_billing_state = 'correction_pending'
             AND json_array_length(NEW.lifecycle_billing_corrections_json) < 1
             AND NEW.billing_retention_exhausted IS NOT 1)
          OR NEW.billing_retention_exhausted NOT IN (0, 1)
          OR (OLD.billing_retention_exhausted = 1
             AND (NEW.billing_retention_exhausted IS NOT 1
                  OR NEW.lifecycle_billing_latest_reference_json
                     IS NOT OLD.lifecycle_billing_latest_reference_json))
          OR (NEW.billing_retention_exhausted = 1
             AND NEW.lifecycle_billing_state NOT IN
                 ('pending', 'correction_pending'))
          OR (json_array_length(OLD.additional_cleanup_targets_json) > 0 AND (
                NEW.lifecycle_billing_state IS NOT OLD.lifecycle_billing_state
             OR NEW.lifecycle_billing_observations_json
                IS NOT OLD.lifecycle_billing_observations_json
             OR NEW.lifecycle_billing_selection_json
                IS NOT OLD.lifecycle_billing_selection_json
             OR NEW.lifecycle_billing_corrections_json
                IS NOT OLD.lifecycle_billing_corrections_json
             OR NEW.lifecycle_billing_latest_reference_json
                IS NOT OLD.lifecycle_billing_latest_reference_json
          ))
        BEGIN
            SELECT RAISE(ABORT, 'invalid Ollama lifecycle evidence mutation');
        END;
        """


@dataclass(frozen=True)
class _LeaseSchemaInventory:
    table_sql: str
    columns: tuple[tuple[Any, ...], ...]
    indexes: tuple[
        tuple[str, int, str, int, str | None, tuple[tuple[Any, ...], ...]], ...
    ]
    triggers: tuple[tuple[str, str], ...]


def _lease_schema_inventory(connection: sqlite3.Connection) -> _LeaseSchemaInventory:
    table = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'ollama_leases'"
    ).fetchone()
    if table is None or not isinstance(table["sql"], str):
        raise RunPodManagerError("Ollama lease table schema is unavailable")
    columns = tuple(
        tuple(row)
        for row in connection.execute("PRAGMA table_xinfo(ollama_leases)").fetchall()
    )
    indexes = []
    for row in connection.execute("PRAGMA index_list(ollama_leases)").fetchall():
        name = row["name"]
        sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (name,),
        ).fetchone()
        indexes.append(
            (
                name,
                row["unique"],
                row["origin"],
                row["partial"],
                sql_row["sql"] if sql_row is not None else None,
                tuple(
                    tuple(item)
                    for item in connection.execute(
                        f"PRAGMA index_xinfo({name})"
                    ).fetchall()
                ),
            )
        )
    triggers = tuple(
        (row["name"], row["sql"])
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name = 'ollama_leases' ORDER BY name"
        ).fetchall()
    )
    return _LeaseSchemaInventory(
        table_sql=table["sql"],
        columns=columns,
        indexes=tuple(sorted(indexes)),
        triggers=triggers,
    )


def _canonical_lease_schema_inventory() -> _LeaseSchemaInventory:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(_CREATE_LEASE_TABLE_SQL)
        connection.execute(_RECONCILE_INDEX_SQL)
        _install_writer_triggers(connection)
        inventory = _lease_schema_inventory(connection)
    return inventory


def _writer_contract_is_current(
    connection: sqlite3.Connection, *, columns: tuple[sqlite3.Row, ...]
) -> bool:
    if not _lease_schema_is_current(columns):
        return False
    if _lease_schema_inventory(connection) != _canonical_lease_schema_inventory():
        return False
    stale_row = connection.execute(
        "SELECT 1 FROM ollama_leases WHERE writer_contract_version IS NOT ? LIMIT 1",
        (_CURRENT_WRITER_CONTRACT_VERSION,),
    ).fetchone()
    return stale_row is None


def _column_signatures(
    columns: tuple[sqlite3.Row, ...],
) -> dict[str, tuple[str, bool, str | None, bool, int]]:
    return {
        row["name"]: (
            row["type"].upper(),
            bool(row["notnull"]),
            row["dflt_value"],
            bool(row["pk"]),
            row["hidden"],
        )
        for row in columns
    }


def _lease_schema_is_current(columns: tuple[sqlite3.Row, ...]) -> bool:
    return _column_signatures(columns) == {
        name: column.signature() for name, column in _LEASE_COLUMN_BY_NAME.items()
    }


def _legacy_schema_is_supported(
    connection: sqlite3.Connection, columns: tuple[sqlite3.Row, ...]
) -> bool:
    """Accept only the complete historical core or a trigger-only repair."""

    actual = _column_signatures(columns)
    expected = {
        name: column.signature() for name, column in _LEASE_COLUMN_BY_NAME.items()
    }

    def supported_signature(
        name: str, signature: tuple[str, bool, str | None, bool, int]
    ) -> bool:
        if signature == expected[name]:
            return True
        return name == "writer_contract_version" and signature in (
            ("INTEGER", True, "0", False, 0),
            ("INTEGER", True, "1", False, 0),
            ("INTEGER", True, "2", False, 0),
            ("INTEGER", True, "3", False, 0),
            ("INTEGER", True, "4", False, 0),
            ("INTEGER", True, "5", False, 0),
        )

    if not set(actual).issubset(expected) or any(
        not supported_signature(name, signature) for name, signature in actual.items()
    ):
        return False
    names = set(actual)
    writer_columns = names & _WRITER_CONTRACT_COLUMNS
    if writer_columns and writer_columns not in {
        _WRITER_CONTRACT_COLUMNS,
        _V5_WRITER_CONTRACT_COLUMNS,
        _V3_WRITER_CONTRACT_COLUMNS,
        _V2_WRITER_CONTRACT_COLUMNS,
        _V1_WRITER_CONTRACT_COLUMNS,
    }:
        return False
    if "writer_contract_version" in names:
        versions = {
            row["writer_contract_version"]
            for row in connection.execute(
                "SELECT DISTINCT writer_contract_version FROM ollama_leases"
            ).fetchall()
        }
        if not versions.issubset({1, 2, 3, 4, 5, _CURRENT_WRITER_CONTRACT_VERSION}):
            return False
        if len(versions) > 1:
            return False
        columns_for_version = {
            1: _V1_WRITER_CONTRACT_COLUMNS,
            2: _V2_WRITER_CONTRACT_COLUMNS,
            3: _V3_WRITER_CONTRACT_COLUMNS,
            4: _V5_WRITER_CONTRACT_COLUMNS,
            5: _V5_WRITER_CONTRACT_COLUMNS,
            _CURRENT_WRITER_CONTRACT_VERSION: _WRITER_CONTRACT_COLUMNS,
        }
        writer_default = actual["writer_contract_version"][2]
        if writer_default is None or not writer_default.isdecimal():
            return False
        default_version = int(writer_default)
        if (
            default_version not in columns_for_version
            or writer_columns != columns_for_version[default_version]
            or (versions and versions != {default_version})
        ):
            return False
    required_legacy = set(expected) - _LEGACY_ADDITIVE_COLUMNS
    if not required_legacy.issubset(names):
        return False
    triggers = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'trigger' AND tbl_name = 'ollama_leases'"
    ).fetchall()
    if any(row["name"] not in _MIGRATABLE_WRITER_TRIGGER_NAMES for row in triggers):
        return False
    allowed_indexes = {_RECONCILE_INDEX_NAME, "sqlite_autoindex_ollama_leases_1"}
    index_rows = connection.execute("PRAGMA index_list(ollama_leases)").fetchall()
    if any(row["name"] not in allowed_indexes for row in index_rows):
        return False
    primary = next(
        (
            row
            for row in index_rows
            if row["name"] == "sqlite_autoindex_ollama_leases_1"
        ),
        None,
    )
    if primary is None or primary["unique"] != 1 or primary["origin"] != "pk":
        return False
    primary_columns = tuple(
        row["name"]
        for row in connection.execute(
            "PRAGMA index_xinfo(sqlite_autoindex_ollama_leases_1)"
        ).fetchall()
        if row["key"]
    )
    return primary_columns == ("lease_id",)


def _drop_writer_triggers(connection: sqlite3.Connection) -> None:
    for name in _MIGRATABLE_WRITER_TRIGGER_NAMES:
        connection.execute(f"DROP TRIGGER IF EXISTS {name}")


def _install_writer_triggers(connection: sqlite3.Connection) -> None:
    _drop_writer_triggers(connection)
    _execute_sql_script_transactionally(connection, _writer_trigger_script())


def _source_writer_contract_versions(
    connection: sqlite3.Connection, columns: tuple[sqlite3.Row, ...]
) -> set[int]:
    if "writer_contract_version" not in {row["name"] for row in columns}:
        return set()
    return {
        row["writer_contract_version"]
        for row in connection.execute(
            "SELECT DISTINCT writer_contract_version FROM ollama_leases"
        ).fetchall()
    }


def _rebuild_current_lease_schema(
    connection: sqlite3.Connection,
    *,
    columns: tuple[sqlite3.Row, ...],
    classify_historical: bool,
) -> None:
    """Rebuild from the canonical DDL after the source was fully validated."""

    source_table = "_ollama_leases_drained_source"
    source_names = {row["name"] for row in columns}
    source_versions = _source_writer_contract_versions(connection, columns)
    upgrading_v3 = (
        source_names & _WRITER_CONTRACT_COLUMNS == _V3_WRITER_CONTRACT_COLUMNS
        and source_versions == {3}
    )
    repairing_v6 = (
        source_names & _WRITER_CONTRACT_COLUMNS == _WRITER_CONTRACT_COLUMNS
        and source_versions == {_CURRENT_WRITER_CONTRACT_VERSION}
    )
    upgrading_v5 = (
        source_names & _WRITER_CONTRACT_COLUMNS == _V5_WRITER_CONTRACT_COLUMNS
        and source_versions == {5}
    )
    upgrading_v4 = (
        source_names & _WRITER_CONTRACT_COLUMNS == _V5_WRITER_CONTRACT_COLUMNS
        and source_versions == {4}
    )
    copied_names = [
        column.name
        for column in _LEASE_COLUMNS
        if column.name in source_names
        and (not classify_historical or column.name not in _WRITER_CONTRACT_COLUMNS)
    ]
    _drop_writer_triggers(connection)
    connection.execute(f"DROP INDEX IF EXISTS {_RECONCILE_INDEX_NAME}")
    connection.execute(f"ALTER TABLE ollama_leases RENAME TO {source_table}")
    connection.execute(_CREATE_LEASE_TABLE_SQL)
    names_sql = ", ".join(copied_names)
    connection.execute(
        f"INSERT INTO ollama_leases ({names_sql}) "
        f"SELECT {names_sql} FROM {source_table}"
    )
    if classify_historical:
        connection.execute(
            """
            UPDATE ollama_leases
            SET writer_contract_version = 6,
                writer_epoch = 0,
                reconcile_after = updated_at,
                provisioning_evidence_state = 'historical_unknown',
                provisioning_intent_json = NULL,
                provisioning_receipt_json = NULL,
                provisioning_mismatch_json = NULL,
                ambiguous_candidate_mismatch_json = NULL,
                ambiguous_create_candidate_id = provider_resource_id,
                provider_resource_id = NULL,
                original_requested_at = NULL,
                original_readiness_timeout_seconds = NULL,
                route_url = NULL,
                provider_health_url = NULL,
                runtime_telemetry_binding_sha256 = NULL,
                cleanup_intent_json = NULL,
                cleanup_receipt_json = NULL,
                additional_cleanup_targets_json = '[]',
                runtime_telemetry_state = 'unknown',
                runtime_telemetry_observations_json = '[]',
                runtime_telemetry_last_provider_error = NULL,
                lifecycle_billing_state = 'unknown',
                lifecycle_billing_observations_json = '[]',
                lifecycle_billing_selection_json = NULL,
                lifecycle_billing_corrections_json = '[]',
                lifecycle_billing_latest_reference_json =
                    '{"observation_id":null,"received_at":null,"state":"not_observed"}',
                billing_last_checked_at = NULL,
                billing_next_check_at = NULL,
                billing_last_provider_error = NULL,
                billing_retention_exhausted = 0
            """
        )
    else:
        if "ambiguous_candidate_mismatch_json" not in source_names:
            _translate_previous_candidate_mismatches(connection)
        if repairing_v6:
            pass
        elif upgrading_v4 or upgrading_v5:
            connection.execute(
                """
                UPDATE ollama_leases
                   SET writer_contract_version = 6,
                       lifecycle_billing_latest_reference_json = CASE
                           WHEN json_array_length(
                                  lifecycle_billing_observations_json
                                ) > 0
                           THEN '{"observation_id":null,"received_at":null,"state":"historical_unknown"}'
                           ELSE '{"observation_id":null,"received_at":null,"state":"not_observed"}'
                       END,
                       revision = revision + 1,
                       writer_epoch = writer_epoch + 1
                """
            )
        elif upgrading_v3:
            connection.execute(
                """
                UPDATE ollama_leases
                   SET writer_contract_version = 6,
                       reconcile_after = updated_at,
                       lifecycle_billing_latest_reference_json = CASE
                           WHEN json_array_length(
                                  lifecycle_billing_observations_json
                                ) > 0
                           THEN '{"observation_id":null,"received_at":null,"state":"historical_unknown"}'
                           ELSE '{"observation_id":null,"received_at":null,"state":"not_observed"}'
                       END,
                       billing_retention_exhausted = CASE
                           WHEN lifecycle_billing_state = 'correction_pending'
                            AND lifecycle_billing_selection_json IS NOT NULL
                            AND lifecycle_billing_corrections_json = '[]'
                            AND billing_last_provider_error =
                                'billing evidence retention requires manual reconciliation'
                           THEN 1 ELSE billing_retention_exhausted END,
                       revision = revision + 1,
                       writer_epoch = writer_epoch + 1
                """
            )
        else:
            connection.execute(
                """
                UPDATE ollama_leases
                    SET writer_contract_version = 6,
                        lifecycle_billing_latest_reference_json =
                            '{"observation_id":null,"received_at":null,"state":"not_observed"}',
                        reconcile_after = updated_at,
                        route_url = NULL,
                        provider_health_url = NULL,
                        runtime_telemetry_state = CASE
                            WHEN provisioning_evidence_state = 'complete'
                            THEN 'partial' ELSE 'unknown' END,
                        lifecycle_billing_state = 'unknown',
                        revision = revision + 1,
                        writer_epoch = writer_epoch + 1
                    """
            )
        if not repairing_v6 and not upgrading_v4 and not upgrading_v5:
            for migrated in connection.execute(
                "SELECT lease_id, provisioning_intent_json FROM ollama_leases "
                "WHERE provisioning_intent_json IS NOT NULL"
            ).fetchall():
                intent = _stored_provisioning_intent(
                    migrated["provisioning_intent_json"]
                )
                if intent is None:
                    raise RunPodManagerError(
                        "Ollama previous writer intent disappeared during migration"
                    )
                connection.execute(
                    "UPDATE ollama_leases "
                    "SET runtime_telemetry_binding_sha256 = ? WHERE lease_id = ?",
                    (
                        runtime_telemetry_binding_for_intent(intent),
                        migrated["lease_id"],
                    ),
                )
        if (
            not repairing_v6
            and not upgrading_v3
            and not upgrading_v4
            and not upgrading_v5
        ):
            connection.execute(
                """
            UPDATE ollama_leases
               SET state = 'releasing',
                   teardown_state = 'pending',
                   termination_reason = COALESCE(
                       termination_reason, 'writer_contract_upgrade'
                   )
             WHERE state IN ('waiting_for_model', 'ready')
               AND provider_resource_id IS NOT NULL
            """
            )
        try:
            for row in connection.execute("SELECT * FROM ollama_leases"):
                _lease_from_row(row)
        except (KeyError, TypeError, ValueError, RunPodManagerError) as exc:
            raise RunPodManagerError(
                "Ollama previous writer evidence is invalid"
            ) from exc
    connection.execute(f"DROP TABLE {source_table}")
    connection.execute(_RECONCILE_INDEX_SQL)
    _install_writer_triggers(connection)


def _translate_previous_candidate_mismatches(
    connection: sqlite3.Connection,
) -> None:
    """Separate old untrusted-candidate evidence without granting ownership.

    The immediately previous writer stored either an owned mismatch or an
    untrusted discovery-candidate mismatch in one column.  Its trigger bound
    that JSON to ``COALESCE(provider_resource_id, candidate_id)``.  An absent
    owned ID therefore identifies the candidate-only case unambiguously.  Move
    those exact bytes to the new slot while leaving the locator untrusted.
    """

    connection.execute(
        """
        UPDATE ollama_leases
        SET ambiguous_candidate_mismatch_json = provisioning_mismatch_json,
            provisioning_mismatch_json = NULL
        WHERE provider_resource_id IS NULL
          AND ambiguous_create_candidate_id IS NOT NULL
          AND provisioning_mismatch_json IS NOT NULL
        """
    )


class SQLiteOllamaLeaseRepository:
    """SQLite WAL store with revision-based compare-and-set updates."""

    _COLUMNS = frozenset(
        {
            "mode",
            "resource_type",
            "provider_resource_id",
            "ambiguous_create_candidate_id",
            "resource_name",
            "creation_uncertain",
            "provision_attempt_id",
            "provision_attempts",
            "route_url",
            "provider_health_url",
            "state",
            "teardown_state",
            "updated_at",
            "provisioning_started_at",
            "ready_at",
            "last_used_at",
            "idle_deadline",
            "model_pull_started_at",
            "model_pull_attempts",
            "model_ready_at",
            "offered_rate_per_hr",
            "placement_gpu_count",
            "estimated_cost",
            "estimated_compute_cost",
            "maximum_compute_cost",
            "estimated_non_compute_cost",
            "maximum_non_compute_cost",
            "cost_ceiling",
            "cost_policy_components_json",
            "maximum_concurrent_workers",
            "estimated_billable_seconds",
            "maximum_billable_seconds",
            "accrued_estimated_cost",
            "cold_start_seconds",
            "selected_gpu_id",
            "selected_gpu_pool",
            "selected_gpu_name",
            "catalog_observed_at",
            "last_provider_error",
            "termination_reason",
            "teardown_attempts",
            "reconcile_after",
            "provisioning_evidence_state",
            "provisioning_intent_json",
            "provisioning_receipt_json",
            "provisioning_mismatch_json",
            "ambiguous_candidate_mismatch_json",
            "runtime_telemetry_binding_sha256",
            "cleanup_intent_json",
            "cleanup_receipt_json",
            "additional_cleanup_targets_json",
            "runtime_telemetry_state",
            "runtime_telemetry_observations_json",
            "runtime_telemetry_last_provider_error",
            "lifecycle_billing_state",
            "lifecycle_billing_observations_json",
            "lifecycle_billing_selection_json",
            "lifecycle_billing_corrections_json",
            "lifecycle_billing_latest_reference_json",
            "billing_last_checked_at",
            "billing_next_check_at",
            "billing_last_provider_error",
            "billing_retention_exhausted",
        }
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("Ollama lease database path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def migrate_legacy_database(
        cls,
        path: Path | str,
        *,
        attestation: OllamaWriterDrainAttestation,
    ) -> SQLiteOllamaLeaseRepository:
        """Install the v6 writer fence after an operator-controlled drain."""

        resolved = Path(path)
        if not resolved.is_absolute():
            raise ValueError("Ollama lease database path must be absolute")
        if resolved.resolve() != attestation.database_path.resolve():
            raise ValueError(
                "Ollama writer drain attestation targets a different database"
            )
        repository = cls.__new__(cls)
        repository.path = resolved
        repository.path.parent.mkdir(parents=True, exist_ok=True)
        repository._initialize(allow_legacy_migration=True)
        return repository

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return lease_database_connection(self.path)

    def _initialize(self, *, allow_legacy_migration: bool = False) -> None:
        with (
            closing(
                sqlite3.connect(self.path, timeout=30, isolation_level=None)
            ) as connection,
            connection,
        ):
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("BEGIN EXCLUSIVE")
            table_exists = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'ollama_leases'"
                ).fetchone()
                is not None
            )
            columns = (
                tuple(
                    connection.execute("PRAGMA table_xinfo(ollama_leases)").fetchall()
                )
                if table_exists
                else ()
            )
            contract_is_current = table_exists and _writer_contract_is_current(
                connection, columns=columns
            )
            source_writer_columns = {
                row["name"] for row in columns
            } & _WRITER_CONTRACT_COLUMNS
            source_writer_versions = _source_writer_contract_versions(
                connection, columns
            )
            migrating_complete_predecessor = (
                table_exists
                and not contract_is_current
                and (
                    (
                        source_writer_columns == _V5_WRITER_CONTRACT_COLUMNS
                        and source_writer_versions in ({4}, {5})
                    )
                    or (
                        source_writer_columns == _WRITER_CONTRACT_COLUMNS
                        and source_writer_versions == {_CURRENT_WRITER_CONTRACT_VERSION}
                    )
                )
            )
            if table_exists and not contract_is_current:
                if not allow_legacy_migration:
                    raise RunPodManagerError(
                        "Legacy Ollama lease database requires an explicit "
                        "drained-writer migration before runtime startup"
                    )
                if not _legacy_schema_is_supported(connection, columns):
                    raise RunPodManagerError(
                        "Ollama lease database schema is not a supported "
                        "drained-writer migration source"
                    )
            if not table_exists:
                connection.execute(_CREATE_LEASE_TABLE_SQL)
                connection.execute(_RECONCILE_INDEX_SQL)
                _install_writer_triggers(connection)
            elif not contract_is_current:
                names = {row["name"] for row in columns}
                added_writer_contract = "writer_contract_version" not in names
                # The operator attested that old writers and routes have
                # drained. Rebuild from the same canonical DDL used for a new
                # database; a hand-normalized SQL comparison could otherwise
                # bless behaviorally different literals, whitespace, or
                # implicit indexes.
                _rebuild_current_lease_schema(
                    connection,
                    columns=columns,
                    classify_historical=added_writer_contract,
                )
            current_columns = tuple(
                connection.execute("PRAGMA table_xinfo(ollama_leases)").fetchall()
            )
            if not _writer_contract_is_current(connection, columns=current_columns):
                raise RunPodManagerError(
                    "Ollama lease database writer contract installation failed"
                )
            # Routes are re-observed from Runpod and exist only in the host
            # process.  Clear values written by releases before the SDK
            # inference boundary made addressable endpoints host-only.
            if not migrating_complete_predecessor:
                connection.execute(
                    """
                    UPDATE ollama_leases
                    SET route_url = NULL,
                        provider_health_url = NULL,
                        revision = revision + 1,
                        writer_epoch = writer_epoch + 1
                    WHERE route_url IS NOT NULL OR provider_health_url IS NOT NULL
                    """
                )
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.execute("PRAGMA journal_mode = WAL")

    def insert_request(
        self, request: OllamaLeaseRequest, *, now: datetime
    ) -> tuple[OllamaLease, bool]:
        inserted = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (request.lease_id,)
            ).fetchone()
            if row is None:
                request = normalize_new_ollama_request_money(request)
                request_started_at = request.requested_at or now
                constraints = asdict(request.constraints)
                constraints["cloud"] = request.constraints.cloud.value
                values = (
                    request.lease_id,
                    request.owner_id,
                    request.workload_id,
                    request.fingerprint,
                    (
                        iso_datetime(request.requested_at)
                        if request.requested_at is not None
                        else None
                    ),
                    request.readiness_timeout_seconds,
                    request.model,
                    json.dumps(constraints, sort_keys=True, separators=(",", ":")),
                    request.mode.value,
                    OllamaLeaseState.REQUESTED.value,
                    OllamaTeardownState.NOT_REQUESTED.value,
                    iso_datetime(now),
                    iso_datetime(now),
                    iso_datetime(now),
                    iso_datetime(
                        min(
                            now + timedelta(seconds=request.idle_timeout_seconds),
                            request.hard_deadline,
                        )
                    ),
                    iso_datetime(request.hard_deadline),
                    iso_datetime(
                        min(
                            request_started_at
                            + timedelta(seconds=request.readiness_timeout_seconds),
                            request.hard_deadline,
                        )
                    ),
                    request.expected_session_seconds,
                    request.expected_active_seconds,
                    request.serverless_initialization_seconds,
                    request.serverless_idle_tail_seconds,
                    request.idle_timeout_seconds,
                    request.max_authorized_cost,
                    _CURRENT_WRITER_CONTRACT_VERSION,
                    1,
                    iso_datetime(now),
                    OllamaProvisioningEvidenceState.REQUIRED.value,
                )
                connection.execute(
                    """
                    INSERT INTO ollama_leases (
                        lease_id, owner_id, workload_id, request_fingerprint,
                        original_requested_at,
                        original_readiness_timeout_seconds, model,
                        constraints_json, mode, state, teardown_state, created_at,
                        updated_at, last_used_at, idle_deadline, hard_deadline,
                        readiness_deadline, expected_session_seconds,
                        expected_active_seconds, serverless_initialization_seconds,
                        serverless_idle_tail_seconds, idle_timeout_seconds,
                        max_authorized_cost, writer_contract_version, writer_epoch,
                        reconcile_after,
                        provisioning_evidence_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                inserted = True
                row = connection.execute(
                    "SELECT * FROM ollama_leases WHERE lease_id = ?",
                    (request.lease_id,),
                ).fetchone()
            elif row["request_fingerprint"] != request.fingerprint:
                normalized_request = normalize_new_ollama_request_money(request)
                if row["request_fingerprint"] == normalized_request.fingerprint:
                    request = normalized_request
            connection.commit()
        if row is None:
            raise RunPodManagerError("Failed to persist or load Ollama lease")
        lease = _lease_from_row(row)
        if lease.request_fingerprint != request.fingerprint:
            raise OllamaLeaseConflictError(
                f"Lease '{request.lease_id}' already represents a different request"
            )
        return lease, inserted

    def get(self, lease_id: str) -> OllamaLease | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
        return _lease_from_row(row) if row is not None else None

    def compare_and_set(
        self, lease: OllamaLease, *, changes: Mapping[str, Any]
    ) -> OllamaLease:
        unknown = set(changes).difference(self._COLUMNS)
        if unknown:
            raise ValueError(f"Unsupported Ollama lease updates: {sorted(unknown)}")
        changes = dict(changes)
        changes.setdefault("updated_at", datetime.now(UTC))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._update_locked(
                connection,
                lease,
                changes,
                "compare and set",
            )

    def claim_cleanup_attempt(
        self, lease: OllamaLease, *, now: datetime
    ) -> OllamaLease:
        """Claim cleanup against the latest row in one immutable effect lineage."""

        require_aware(now, "cleanup attempt")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if not _cleanup_lineage_is_compatible(lease, current):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed cleanup lineage"
                )
            if current.state is OllamaLeaseState.TERMINATED:
                connection.commit()
                return current
            if current.state is not OllamaLeaseState.RELEASING:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' no longer permits cleanup"
                )
            return self._update_locked(
                connection,
                current,
                {
                    "teardown_attempts": current.teardown_attempts + 1,
                    "updated_at": now,
                },
                "cleanup attempt",
            )

    def latest_cleanup_lineage(self, lease: OllamaLease) -> OllamaLease:
        """Reload a monotonic extension of one already-claimed cleanup lineage."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if not _cleanup_lineage_is_compatible(lease, current):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed cleanup lineage"
                )
            connection.commit()
        return current

    def validate_stored_cleanup_authority(self, lease: OllamaLease) -> OllamaLease:
        """Validate one supplied cleanup snapshot without discovering new targets.

        Cleanup can continue from an already-durable snapshot when an optional
        later refresh fails, but only after the snapshot is proven to be a
        prefix of the current stored authority.  Returning the supplied value
        deliberately prevents this admission check from turning an unseen
        concurrent suffix into provider-delete authority.
        """

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if not _cleanup_lineage_is_compatible(lease, current):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed cleanup lineage"
                )
            if current.state is OllamaLeaseState.TERMINATED:
                connection.commit()
                return current
            if (
                lease.state is not OllamaLeaseState.RELEASING
                or current.state is not OllamaLeaseState.RELEASING
                or lease.cleanup_intent is None
                or lease.provider_resource_id is None
                or lease.cleanup_intent.provider_resource_id
                != lease.provider_resource_id
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' lacks stored cleanup authority"
                )
            connection.commit()
        return lease

    def attach_provider_acknowledgment(
        self,
        lease: OllamaLease,
        resource: ProvisionedOllamaResource,
        *,
        now: datetime,
    ) -> OllamaLease:
        """Attach an original POST ACK to the latest immutable lease lineage."""

        return self.record_provisioning_outcome(
            lease,
            now=now,
            acknowledged_resource=resource,
        )

    def record_provisioning_outcome(
        self,
        lease: OllamaLease,
        *,
        now: datetime,
        acknowledged_resource: ProvisionedOllamaResource | None = None,
        ambiguous_candidate: ProvisionedOllamaResource | None = None,
        receipt: OllamaProvisioningReceipt | None = None,
        mismatch: OllamaProvisioningMismatchEvidence | None = None,
        candidate_mismatch: bool = False,
        provider_error: str | None = None,
    ) -> OllamaLease:
        """Fold one post-await result into the latest immutable create lineage.

        The caller's revision only identifies the lineage that admitted the
        effect.  State and cleanup decisions are made from the row locked in
        this transaction so a concurrent release cannot discard an original
        provider acknowledgment or revive work.
        """

        require_aware(now, "provider acknowledgment")
        intent = lease.provisioning_intent
        if intent is None:
            raise OllamaLeaseConflictError(
                f"Lease '{lease.lease_id}' has no original provisioning intent"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' disappeared before create acknowledgment"
                )
            current = _lease_from_row(row)
            current_intent = current.provisioning_intent
            if (
                current_intent is None
                or current_intent.intent_sha256 != intent.intent_sha256
                or current.request_fingerprint != lease.request_fingerprint
                or current.provision_attempt_id != lease.provision_attempt_id
                or (
                    acknowledged_resource is not None
                    and current.resource_type is not acknowledged_resource.resource_type
                )
                or (
                    ambiguous_candidate is not None
                    and current.resource_type is not ambiguous_candidate.resource_type
                )
                or current.resource_type is not intent.resource_type
                or current.resource_name != intent.resource_name
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed provisioning lineage before "
                    "create acknowledgment"
                )
            acknowledged_id = (
                acknowledged_resource.provider_resource_id
                if acknowledged_resource is not None
                else None
            )
            candidate_id = (
                ambiguous_candidate.provider_resource_id
                if ambiguous_candidate is not None
                else None
            )
            if ambiguous_candidate is not None and (
                ambiguous_candidate.resource_name != intent.resource_name
                or (
                    current.ambiguous_create_candidate_id is not None
                    and current.ambiguous_create_candidate_id != candidate_id
                )
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' received a foreign discovery locator"
                )
            if receipt is not None:
                try:
                    receipt.validate_intent(intent)
                except ValueError as exc:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' received an unbound receipt"
                    ) from exc
                if acknowledged_id not in {None, receipt.provider_resource_id}:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' receipt changed its create ID"
                    )
                acknowledged_id = receipt.provider_resource_id
            if mismatch is not None:
                try:
                    mismatch.validate_intent(intent)
                except ValueError as exc:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' received unbound mismatch evidence"
                    ) from exc
                if not candidate_mismatch:
                    if acknowledged_id not in {
                        None,
                        mismatch.provider_resource_id,
                    }:
                        connection.rollback()
                        raise OllamaLeaseConflictError(
                            f"Lease '{lease.lease_id}' mismatch changed its create ID"
                        )
                    acknowledged_id = mismatch.provider_resource_id
            additional_ack_target = (
                current.provider_resource_id is not None
                and acknowledged_id is not None
                and current.provider_resource_id != acknowledged_id
                and acknowledged_resource is not None
            )
            if (
                current.provider_resource_id is not None
                and acknowledged_id is not None
                and current.provider_resource_id != acknowledged_id
                and not additional_ack_target
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' received a foreign create acknowledgment"
                )
            if additional_ack_target and receipt is not None:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' cannot bind one receipt to two resources"
                )
            if current.state not in {
                OllamaLeaseState.PROVISIONING,
                OllamaLeaseState.RECONCILE_REQUIRED,
                OllamaLeaseState.WAITING_FOR_MODEL,
                OllamaLeaseState.READY,
                OllamaLeaseState.RELEASING,
                OllamaLeaseState.FAILED,
                OllamaLeaseState.TERMINATED,
            }:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' cannot accept a create acknowledgment "
                    "in its current state"
                )
            changes: dict[str, Any] = {}
            new_additional_ack_target = False
            if acknowledged_id is not None and not additional_ack_target:
                changes["provider_resource_id"] = acknowledged_id
            if additional_ack_target:
                target = OllamaAdditionalCleanupTarget(
                    intent_sha256=intent.intent_sha256,
                    lease_id=current.lease_id,
                    provision_attempt_id=intent.provision_attempt_id,
                    resource_type=intent.resource_type,
                    resource_name=intent.resource_name,
                    provider_resource_id=acknowledged_id,
                    source="original_post_ack",
                    provisioning_mismatch=(
                        mismatch if not candidate_mismatch else None
                    ),
                )
                existing_target = next(
                    (
                        item
                        for item in current.additional_cleanup_targets
                        if item.provider_resource_id == acknowledged_id
                    ),
                    None,
                )
                if existing_target is None:
                    if current.additional_cleanup_targets:
                        connection.rollback()
                        raise OllamaLeaseConflictError(
                            f"Lease '{lease.lease_id}' has conflicting cleanup targets"
                        )
                    changes["additional_cleanup_targets_json"] = (target,)
                    new_additional_ack_target = True
                elif (
                    mismatch is not None
                    and existing_target.provisioning_mismatch is None
                ):
                    changes["additional_cleanup_targets_json"] = (
                        replace(existing_target, provisioning_mismatch=mismatch),
                    )
                elif (
                    mismatch is not None
                    and existing_target.provisioning_mismatch != mismatch
                ):
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' additional mismatch is immutable"
                    )
            if candidate_id is not None:
                changes["ambiguous_create_candidate_id"] = candidate_id
            if candidate_mismatch:
                if mismatch is None or (
                    current.ambiguous_create_candidate_id
                    != mismatch.provider_resource_id
                ):
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' candidate mismatch changed its locator"
                    )
                if (
                    current.ambiguous_candidate_mismatch is not None
                    and current.ambiguous_candidate_mismatch != mismatch
                ):
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' candidate mismatch is immutable"
                    )
                changes["ambiguous_candidate_mismatch_json"] = mismatch
            elif mismatch is not None and not additional_ack_target:
                if (
                    current.provisioning_mismatch is not None
                    and current.provisioning_mismatch != mismatch
                ):
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' provisioning mismatch is immutable"
                    )
                changes.update(
                    {
                        "provisioning_mismatch_json": mismatch,
                        "creation_uncertain": False,
                        "route_url": None,
                        "provider_health_url": None,
                    }
                )
            if receipt is not None:
                if current.provisioning_mismatch is not None:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' cannot replace mismatch evidence"
                    )
                if current.provider_resource_id is None and (
                    not receipt.reconciled_after_ambiguous_create
                    or current.ambiguous_create_candidate_id
                    != receipt.provider_resource_id
                ):
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' receipt has no original authority"
                    )
                if (
                    current.provisioning_receipt is not None
                    and current.provisioning_receipt != receipt
                ):
                    if not _same_provisioning_receipt_authority(
                        current.provisioning_receipt, receipt
                    ):
                        connection.rollback()
                        raise OllamaLeaseConflictError(
                            f"Lease '{lease.lease_id}' provisioning receipt is immutable"
                        )
                else:
                    changes["provisioning_receipt_json"] = receipt
                changes.update(
                    {
                        "provisioning_evidence_state": (
                            OllamaProvisioningEvidenceState.COMPLETE
                        ),
                        "creation_uncertain": False,
                        "last_provider_error": None,
                    }
                )
                if (
                    current.provisioning_receipt is None
                    and current.state is OllamaLeaseState.TERMINATED
                    and current.teardown_state is OllamaTeardownState.COMPLETE
                    and current.cleanup_intent is not None
                    and current.cleanup_receipt is not None
                    and not current.additional_cleanup_targets
                    and current.provisioning_mismatch is None
                    and current.lifecycle_billing_state
                    is OllamaLifecycleBillingState.UNKNOWN
                    and not current.lifecycle_billing_observations
                    and current.lifecycle_billing_selection is None
                    and not current.lifecycle_billing_corrections
                    and current.lifecycle_billing_latest_reference.state
                    is OllamaLifecycleBillingLatestState.NOT_OBSERVED
                    and not current.billing_retention_exhausted
                ):
                    changes.update(
                        {
                            "lifecycle_billing_state": (
                                OllamaLifecycleBillingState.PENDING
                            ),
                            "billing_next_check_at": now,
                        }
                    )
            elif provider_error is not None:
                changes["last_provider_error"] = provider_error

            if additional_ack_target and not changes:
                connection.commit()
                return current

            new_state = current.state
            teardown_state = current.teardown_state
            if new_additional_ack_target:
                new_state = OllamaLeaseState.RELEASING
                teardown_state = OllamaTeardownState.PENDING
                changes.update(
                    {
                        "creation_uncertain": False,
                        "route_url": None,
                        "provider_health_url": None,
                        "billing_next_check_at": None,
                        "billing_last_provider_error": (
                            "multiple exact resources require manual billing "
                            "reconciliation"
                        ),
                        "reconcile_after": now,
                    }
                )
            elif (
                acknowledged_id is not None
                and current.provider_resource_id is None
                and current.state
                in {OllamaLeaseState.FAILED, OllamaLeaseState.TERMINATED}
            ) or (
                mismatch is not None
                and not candidate_mismatch
                and (
                    current.provider_resource_id is not None
                    or acknowledged_id is not None
                )
                and current.state
                not in {
                    OllamaLeaseState.RELEASING,
                    OllamaLeaseState.TERMINATED,
                    OllamaLeaseState.FAILED,
                }
            ):
                new_state = OllamaLeaseState.RELEASING
                teardown_state = OllamaTeardownState.PENDING
            elif (
                mismatch is not None or provider_error is not None
            ) and current.state is OllamaLeaseState.PROVISIONING:
                new_state = OllamaLeaseState.RECONCILE_REQUIRED
            elif receipt is not None and current.state in {
                OllamaLeaseState.PROVISIONING,
                OllamaLeaseState.RECONCILE_REQUIRED,
            }:
                new_state = OllamaLeaseState.WAITING_FOR_MODEL
            if (
                current.state is OllamaLeaseState.RELEASING
                and not additional_ack_target
                and (mismatch is not None or provider_error is not None)
            ):
                if (
                    current.provider_resource_id is not None
                    or acknowledged_id is not None
                ):
                    changes["creation_uncertain"] = False
                    teardown_state = OllamaTeardownState.PENDING
                else:
                    teardown_state = OllamaTeardownState.RETRYABLE_FAILURE
            changes.update(
                {
                    "state": new_state,
                    "teardown_state": teardown_state,
                    "updated_at": now,
                }
            )
            serialized = {key: _db_value(value) for key, value in changes.items()}
            assignments = ", ".join(f"{key} = ?" for key in serialized)
            cursor = connection.execute(
                f"""UPDATE ollama_leases
                    SET {assignments}, revision = revision + 1,
                        writer_epoch = writer_epoch + 1
                    WHERE lease_id = ? AND revision = ?""",
                (*serialized.values(), current.lease_id, current.revision),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed concurrently; reconcile it"
                )
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
            ).fetchone()
            connection.commit()
        if row is None:
            raise RunPodManagerError("Acknowledged Ollama lease disappeared")
        return _lease_from_row(row)

    def record_cleanup_intent(
        self,
        lease: OllamaLease,
        *,
        now: datetime,
        termination_reason: str,
    ) -> OllamaLease:
        """Freeze the original teardown start before admitting provider I/O."""

        require_aware(now, "cleanup intent")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
            ).fetchone()
            if row is None:
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' disappeared before cleanup"
                )
            current = _lease_from_row(row)
            if not _cleanup_lineage_is_compatible(
                lease,
                current,
                phase=_CleanupLineagePhase.PRIMARY_INTENT,
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed cleanup lineage"
                )
            if current.cleanup_intent is not None:
                connection.commit()
                return current
            intent = current.provisioning_intent
            provisioning = current.provisioning_receipt
            if (
                intent is None
                or current.provider_resource_id is None
                or current.state is not OllamaLeaseState.RELEASING
            ):
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' lacks exact cleanup authority"
                )
            cleanup = OllamaCleanupIntent(
                intent_sha256=intent.intent_sha256,
                provisioning_receipt_id=(
                    provisioning.receipt_id if provisioning is not None else None
                ),
                lease_id=current.lease_id,
                request_fingerprint=current.request_fingerprint,
                provision_attempt_id=intent.provision_attempt_id,
                resource_type=intent.resource_type,
                provider_resource_id=current.provider_resource_id,
                resource_name=intent.resource_name,
                termination_reason=current.termination_reason or termination_reason,
                delete_started_at=now,
            )
            cleanup.validate_provisioning(intent, provisioning)
            return self._update_locked(
                connection,
                current,
                {"cleanup_intent_json": cleanup, "updated_at": now},
                "cleanup intent",
            )

    def record_additional_cleanup_intent(
        self,
        lease: OllamaLease,
        *,
        provider_resource_id: str,
        now: datetime,
        termination_reason: str,
    ) -> OllamaLease:
        """Freeze teardown start for one separately proven original POST ACK."""

        require_aware(now, "additional cleanup intent")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if not _cleanup_lineage_is_compatible(
                lease,
                current,
                phase=_CleanupLineagePhase.ADDITIONAL_INTENT,
                provider_resource_id=provider_resource_id,
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed additional cleanup lineage"
                )
            intent = current.provisioning_intent
            if intent is None or current.state is not OllamaLeaseState.RELEASING:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' lacks additional cleanup authority"
                )
            targets = list(current.additional_cleanup_targets)
            target_match = _additional_cleanup_target(
                current.additional_cleanup_targets,
                provider_resource_id,
            )
            if target_match is None:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' has no matching cleanup target"
                )
            index, target = target_match
            if target.cleanup_receipt is not None:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed additional cleanup lineage"
                )
            if target.cleanup_intent is not None:
                connection.commit()
                return current
            cleanup = OllamaCleanupIntent(
                intent_sha256=intent.intent_sha256,
                provisioning_receipt_id=None,
                lease_id=current.lease_id,
                request_fingerprint=current.request_fingerprint,
                provision_attempt_id=intent.provision_attempt_id,
                resource_type=intent.resource_type,
                provider_resource_id=target.provider_resource_id,
                resource_name=intent.resource_name,
                termination_reason=current.termination_reason or termination_reason,
                delete_started_at=now,
            )
            cleanup.validate_provisioning(intent, None)
            targets[index] = replace(target, cleanup_intent=cleanup)
            return self._update_locked(
                connection,
                current,
                {
                    "additional_cleanup_targets_json": tuple(targets),
                    "updated_at": now,
                },
                "additional cleanup intent",
            )

    def record_additional_cleanup_outcome(
        self,
        lease: OllamaLease,
        *,
        provider_resource_id: str,
        now: datetime,
        accrued_estimated_cost: float,
        termination_reason: str,
        cleanup_receipt: OllamaCleanupReceipt | None = None,
        provider_error: str | None = None,
        retry_at: datetime | None = None,
    ) -> OllamaLease:
        """Fold one exact additional-target teardown into the same aggregate."""

        require_aware(now, "additional teardown outcome")
        if (cleanup_receipt is None) == (provider_error is None):
            raise ValueError("Additional cleanup requires one outcome")
        if provider_error is not None:
            if retry_at is None:
                raise ValueError("Failed cleanup requires a retry schedule")
            require_aware(retry_at, "additional cleanup retry")
            if retry_at <= now:
                raise ValueError("Additional cleanup retry must be in the future")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if not _cleanup_lineage_is_compatible(
                lease,
                current,
                phase=_CleanupLineagePhase.ADDITIONAL_OUTCOME,
                provider_resource_id=provider_resource_id,
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed additional cleanup lineage"
                )
            targets = list(current.additional_cleanup_targets)
            target_match = _additional_cleanup_target(
                current.additional_cleanup_targets,
                provider_resource_id,
            )
            if target_match is None:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' has no matching cleanup target"
                )
            index, target = target_match
            if cleanup_receipt is not None:
                if target.cleanup_intent is None:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' additional cleanup lacks intent"
                    )
                try:
                    cleanup_receipt.validate_intent(target.cleanup_intent)
                except ValueError as exc:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' received unbound additional cleanup"
                    ) from exc
            if current.state is not OllamaLeaseState.RELEASING and not (
                current.state is OllamaLeaseState.TERMINATED
                and target.cleanup_receipt is not None
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed additional cleanup lineage"
                )
            if target.cleanup_receipt is not None:
                connection.commit()
                return current
            if provider_error is not None:
                changes: dict[str, Any] = {
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": provider_error,
                    "reconcile_after": retry_at,
                }
            else:
                if cleanup_receipt is None:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' additional cleanup lacks intent"
                    )
                targets[index] = replace(target, cleanup_receipt=cleanup_receipt)
                all_cleanup_complete = current.cleanup_receipt is not None and all(
                    item.cleanup_receipt is not None for item in targets
                )
                changes = {
                    "additional_cleanup_targets_json": tuple(targets),
                    "state": (
                        OllamaLeaseState.TERMINATED
                        if all_cleanup_complete
                        else OllamaLeaseState.RELEASING
                    ),
                    "teardown_state": (
                        OllamaTeardownState.COMPLETE
                        if all_cleanup_complete
                        else OllamaTeardownState.PENDING
                    ),
                    "last_provider_error": None,
                    "termination_reason": (
                        current.termination_reason or termination_reason
                    ),
                }
            changes.update(
                {
                    "route_url": None,
                    "provider_health_url": None,
                    "accrued_estimated_cost": accrued_estimated_cost,
                    "updated_at": now,
                }
            )
            return self._update_locked(
                connection, current, changes, "additional teardown outcome"
            )

    def record_runtime_telemetry(
        self,
        lease: OllamaLease,
        observation: OllamaRuntimeTelemetryObservation | None,
        *,
        now: datetime,
        maximum_observations: int,
        maximum_bytes: int,
        provider_error: str | None = None,
    ) -> OllamaLease:
        """Append one authenticated snapshot without rewriting prior evidence."""

        if maximum_observations < 1 or maximum_bytes < 1:
            raise ValueError("Ollama telemetry retention bounds must be positive")
        if (observation is None) == (provider_error is None):
            raise ValueError("Ollama telemetry outcome must be evidence or one error")
        require_aware(now, "runtime telemetry persistence")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if (
                current.request_fingerprint != lease.request_fingerprint
                or current.provision_attempt_id != lease.provision_attempt_id
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed telemetry lineage"
                )
            if observation is None:
                state = (
                    OllamaRuntimeTelemetryState.COMPLETE
                    if current.runtime_telemetry_state
                    is OllamaRuntimeTelemetryState.COMPLETE
                    else OllamaRuntimeTelemetryState.PARTIAL
                )
                return self._update_locked(
                    connection,
                    current,
                    {
                        "runtime_telemetry_state": state,
                        "runtime_telemetry_last_provider_error": provider_error,
                        "updated_at": now,
                    },
                    "runtime telemetry error",
                )
            intent = current.provisioning_intent
            provisioning = current.provisioning_receipt
            binding = current.runtime_telemetry_binding_sha256
            if intent is None or provisioning is None or binding is None:
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' lacks telemetry authority"
                )
            observation.validate_provisioning(
                intent, provisioning, telemetry_binding_sha256=binding
            )
            if any(
                item.observation_id == observation.observation_id
                for item in current.runtime_telemetry_observations
            ):
                connection.commit()
                return current
            observations = (*current.runtime_telemetry_observations, observation)
            serialized = _db_value(observations)
            if (
                len(observations) > maximum_observations
                or len(serialized) > maximum_bytes
            ):
                return self._update_locked(
                    connection,
                    current,
                    {
                        "runtime_telemetry_state": (
                            OllamaRuntimeTelemetryState.COMPLETE
                            if current.runtime_telemetry_state
                            is OllamaRuntimeTelemetryState.COMPLETE
                            else OllamaRuntimeTelemetryState.PARTIAL
                        ),
                        "runtime_telemetry_last_provider_error": (
                            "runtime telemetry retention requires manual reconciliation"
                        ),
                        "updated_at": now,
                    },
                    "runtime telemetry limit",
                )
            previous = (
                current.runtime_telemetry_observations[-1]
                if current.runtime_telemetry_observations
                else None
            )
            lineage_complete = previous is None or _telemetry_continues_lineage(
                previous, observation
            )
            terminal_without_startup = previous is None and current.state in {
                OllamaLeaseState.RELEASING,
                OllamaLeaseState.TERMINATED,
            }
            state = (
                OllamaRuntimeTelemetryState.COMPLETE
                if current.runtime_telemetry_state
                is OllamaRuntimeTelemetryState.COMPLETE
                else OllamaRuntimeTelemetryState.PARTIAL
                if current.runtime_telemetry_state
                is OllamaRuntimeTelemetryState.PARTIAL
                or terminal_without_startup
                or not lineage_complete
                or current.state
                in {OllamaLeaseState.RELEASING, OllamaLeaseState.TERMINATED}
                or not observation.is_complete_startup_for(intent)
                else OllamaRuntimeTelemetryState.STARTUP_COMPLETE
            )
            return self._update_locked(
                connection,
                current,
                {
                    "runtime_telemetry_observations_json": observations,
                    "runtime_telemetry_state": state,
                    "runtime_telemetry_last_provider_error": None,
                    "updated_at": now,
                },
                "runtime telemetry",
            )

    def record_billing_outcome(
        self,
        lease: OllamaLease,
        *,
        now: datetime,
        next_check_at: datetime,
        observation: OllamaLifecycleBillingObservation | None = None,
        provider_error: str | None = None,
        maximum_observations: int,
        maximum_bytes: int,
    ) -> OllamaLease:
        """Select once, then append immutable correction evidence."""

        require_aware(now, "billing observation persistence")
        require_aware(next_check_at, "billing next check")
        if next_check_at <= now:
            raise ValueError("Ollama billing retry must be in the future")
        if (observation is None) == (provider_error is None):
            raise ValueError("Ollama billing outcome must be evidence or one error")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._locked_lease(connection, lease.lease_id)
            if (
                current.billing_last_checked_at is not None
                and now < current.billing_last_checked_at
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' rejected a stale billing poll clock"
                )
            if (
                current.state is not OllamaLeaseState.TERMINATED
                or current.cleanup_intent is None
                or current.cleanup_receipt is None
                or current.provisioning_intent is None
                or current.provisioning_receipt is None
                or current.additional_cleanup_targets
            ):
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' lacks closed lifecycle billing authority"
                )
            changes: dict[str, Any] = {
                "billing_last_checked_at": now,
                "billing_next_check_at": next_check_at,
                "billing_last_provider_error": provider_error,
                "reconcile_after": next_check_at,
                "updated_at": now,
            }
            if observation is not None:
                observation.validate_lifecycle(
                    current.provisioning_intent,
                    current.provisioning_receipt,
                    current.cleanup_intent,
                    current.cleanup_receipt,
                )
            if current.billing_retention_exhausted:
                changes.update(
                    {
                        "lifecycle_billing_state": (
                            OllamaLifecycleBillingState.CORRECTION_PENDING
                            if current.lifecycle_billing_selection is not None
                            else OllamaLifecycleBillingState.PENDING
                        ),
                        "billing_last_provider_error": (
                            _BILLING_RETENTION_MANUAL_ERROR
                        ),
                    }
                )
                return self._update_locked(
                    connection, current, changes, "lifecycle billing"
                )
            if observation is not None:
                existing = next(
                    (
                        item
                        for item in current.lifecycle_billing_observations
                        if item.observation_id == observation.observation_id
                    ),
                    None,
                )
                if existing is not None:
                    changes["lifecycle_billing_latest_reference_json"] = (
                        OllamaLifecycleBillingLatestReference.known(
                            existing.observation_id,
                            received_at=now,
                        )
                    )
                else:
                    observations = (
                        *current.lifecycle_billing_observations,
                        observation,
                    )
                    selection = current.lifecycle_billing_selection
                    if selection is None:
                        candidate_selection = OllamaLifecycleBillingSelection(
                            observation_id=observation.observation_id,
                            selected_at=now,
                        )
                        corrections = current.lifecycle_billing_corrections
                    else:
                        candidate_selection = selection
                        selected = next(
                            item
                            for item in current.lifecycle_billing_observations
                            if item.observation_id == selection.observation_id
                        )
                        correction = OllamaLifecycleBillingCorrection(
                            selected_observation_id=selection.observation_id,
                            revised_observation_id=observation.observation_id,
                            actual_cost_delta_usd=exact_decimal_sum(
                                (
                                    observation.provider_capacity_actual_cost_usd,
                                    selected.provider_capacity_actual_cost_usd.copy_negate(),
                                )
                            ),
                            recorded_at=now,
                        )
                        corrections = (
                            *current.lifecycle_billing_corrections,
                            correction,
                        )
                    latest_reference = OllamaLifecycleBillingLatestReference.known(
                        observation.observation_id,
                        received_at=now,
                    )
                    serialized_bytes = sum(
                        len(_db_value(value))
                        for value in (
                            observations,
                            corrections,
                            candidate_selection,
                            latest_reference,
                        )
                    )
                    exceeds_retention = (
                        len(observations) > maximum_observations
                        or len(corrections) > maximum_observations
                        or serialized_bytes > maximum_bytes
                    )
                    if exceeds_retention:
                        changes.update(
                            {
                                "lifecycle_billing_state": (
                                    OllamaLifecycleBillingState.CORRECTION_PENDING
                                    if selection is not None
                                    else OllamaLifecycleBillingState.PENDING
                                ),
                                "billing_last_provider_error": (
                                    _BILLING_RETENTION_MANUAL_ERROR
                                ),
                                "billing_retention_exhausted": True,
                            }
                        )
                    else:
                        changes.update(
                            {
                                "lifecycle_billing_observations_json": observations,
                                "lifecycle_billing_selection_json": (
                                    candidate_selection
                                ),
                                "lifecycle_billing_corrections_json": corrections,
                                "lifecycle_billing_latest_reference_json": (
                                    latest_reference
                                ),
                                "lifecycle_billing_state": (
                                    OllamaLifecycleBillingState.SELECTED
                                    if selection is None
                                    else OllamaLifecycleBillingState.CORRECTION_PENDING
                                ),
                            }
                        )
            return self._update_locked(
                connection, current, changes, "lifecycle billing"
            )

    def list_for_reconciliation(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> tuple[OllamaLease, ...]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Ollama reconciliation limit must be positive")
        now = now or datetime.now(UTC)
        require_aware(now, "reconciliation clock")
        now_text = iso_datetime(now)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM ollama_leases
                   WHERE provisioning_evidence_state != ?
                     AND reconcile_after <= ?
                     AND (state != 'terminated'
                          OR (json_array_length(additional_cleanup_targets_json) = 0
                              AND lifecycle_billing_state IN
                                ('pending', 'selected', 'correction_pending')
                              AND (billing_next_check_at IS NULL
                                   OR billing_next_check_at <= ?)))
                   ORDER BY
                     CASE
                       WHEN state NOT IN ('releasing', 'terminated')
                            AND hard_deadline <= ? THEN 0
                       WHEN state IN ('provisioning', 'reconcile_required',
                                      'waiting_for_model')
                            AND readiness_deadline <= ? THEN 0
                       WHEN state = 'ready' AND idle_deadline <= ? THEN 0
                       ELSE 1
                     END,
                     reconcile_after, created_at, lease_id
                   LIMIT ?""",
                (
                    OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN.value,
                    now_text,
                    now_text,
                    now_text,
                    now_text,
                    now_text,
                    limit,
                ),
            ).fetchall()
        return tuple(_lease_from_row(row) for row in rows)

    def claim_for_reconciliation(
        self,
        lease: OllamaLease,
        *,
        now: datetime,
        retry_at: datetime,
    ) -> OllamaLease | None:
        """Durably rotate one due row before any provider await.

        The claim is scheduling metadata only.  It neither creates provider
        authority nor changes a lifecycle timestamp.  A competing process
        wins through the same revision/epoch CAS used for every write.
        """

        require_aware(now, "reconciliation claim")
        require_aware(retry_at, "reconciliation retry")
        if retry_at <= now:
            raise ValueError("Ollama reconciliation retry must be in the future")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE ollama_leases
                      SET reconcile_after = ?, revision = revision + 1,
                          writer_epoch = writer_epoch + 1
                    WHERE lease_id = ? AND revision = ?
                      AND provisioning_evidence_state != ?
                      AND reconcile_after <= ?""",
                (
                    iso_datetime(retry_at),
                    lease.lease_id,
                    lease.revision,
                    OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN.value,
                    iso_datetime(now),
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
            ).fetchone()
            connection.commit()
        if row is None:
            raise RunPodManagerError("Claimed Ollama lease disappeared")
        return _lease_from_row(row)

    def _locked_lease(
        self, connection: sqlite3.Connection, lease_id: str
    ) -> OllamaLease:
        row = connection.execute(
            "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease_id,)
        ).fetchone()
        if row is None:
            raise OllamaLeaseConflictError(f"Lease '{lease_id}' disappeared")
        return _lease_from_row(row)

    def _update_locked(
        self,
        connection: sqlite3.Connection,
        lease: OllamaLease,
        changes: Mapping[str, Any],
        context: str,
    ) -> OllamaLease:
        serialized = {key: _db_value(value) for key, value in changes.items()}
        assignments = ", ".join(f"{key} = ?" for key in serialized)
        cursor = connection.execute(
            f"""UPDATE ollama_leases
                SET {assignments}, revision = revision + 1,
                    writer_epoch = writer_epoch + 1
                WHERE lease_id = ? AND revision = ?""",
            (*serialized.values(), lease.lease_id, lease.revision),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise OllamaLeaseConflictError(
                f"Lease '{lease.lease_id}' changed concurrently during {context}"
            )
        row = connection.execute(
            "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
        ).fetchone()
        if row is None:
            connection.rollback()
            raise RunPodManagerError(f"Ollama lease disappeared during {context}")
        updated = _lease_from_row(row)
        connection.commit()
        return updated

    def record_teardown_outcome(
        self,
        lease: OllamaLease,
        *,
        provider_resource_id: str,
        now: datetime,
        accrued_estimated_cost: float,
        termination_reason: str,
        cleanup_receipt: OllamaCleanupReceipt | None = None,
        billing_next_check_at: datetime | None = None,
        provider_error: str | None = None,
        retry_at: datetime | None = None,
    ) -> OllamaLease:
        """Fold an exact-ID teardown outcome into the latest lease revision."""

        require_aware(now, "teardown outcome")
        if not provider_resource_id:
            raise ValueError("Ollama teardown outcome requires a provider ID")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' disappeared during teardown"
                )
            current = _lease_from_row(row)
            if (
                not _cleanup_lineage_is_compatible(lease, current)
                or current.provider_resource_id != provider_resource_id
            ):
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' changed teardown lineage"
                )
            if provider_error is None:
                if current.cleanup_intent is None or cleanup_receipt is None:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' teardown lacks typed cleanup"
                    )
                try:
                    cleanup_receipt.validate_intent(current.cleanup_intent)
                except ValueError as exc:
                    connection.rollback()
                    raise OllamaLeaseConflictError(
                        f"Lease '{lease.lease_id}' received unbound cleanup"
                    ) from exc
                if (
                    current.provisioning_receipt is not None
                    and billing_next_check_at is None
                ):
                    connection.rollback()
                    raise ValueError("Successful cleanup requires a billing schedule")
                if billing_next_check_at is not None:
                    require_aware(billing_next_check_at, "billing schedule")
                if current.cleanup_receipt is not None:
                    connection.commit()
                    return current
                all_cleanup_complete = all(
                    target.cleanup_receipt is not None
                    for target in current.additional_cleanup_targets
                )
                changes: dict[str, Any] = {
                    "state": (
                        OllamaLeaseState.TERMINATED
                        if all_cleanup_complete
                        else OllamaLeaseState.RELEASING
                    ),
                    "teardown_state": (
                        OllamaTeardownState.COMPLETE
                        if all_cleanup_complete
                        else OllamaTeardownState.PENDING
                    ),
                    "last_provider_error": (
                        current.last_provider_error
                        if current.provisioning_mismatch is not None
                        else None
                    ),
                    "termination_reason": (
                        current.termination_reason or termination_reason
                    ),
                    "cleanup_receipt_json": cleanup_receipt,
                    "lifecycle_billing_state": (
                        current.lifecycle_billing_state
                        if current.additional_cleanup_targets
                        else OllamaLifecycleBillingState.PENDING
                        if current.provisioning_receipt is not None
                        else OllamaLifecycleBillingState.UNKNOWN
                    ),
                    "billing_next_check_at": (
                        current.billing_next_check_at
                        if current.additional_cleanup_targets
                        else billing_next_check_at
                        if current.provisioning_receipt is not None
                        else None
                    ),
                }
            else:
                if retry_at is None:
                    connection.rollback()
                    raise ValueError("Failed cleanup requires a retry schedule")
                require_aware(retry_at, "cleanup retry")
                if retry_at <= now:
                    connection.rollback()
                    raise ValueError("Cleanup retry must be in the future")
                if current.cleanup_receipt is not None:
                    connection.commit()
                    return current
                changes = {
                    "teardown_state": OllamaTeardownState.RETRYABLE_FAILURE,
                    "last_provider_error": provider_error,
                    "reconcile_after": retry_at,
                }
            if current.state is OllamaLeaseState.TERMINATED:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' has no retained teardown outcome"
                )
            if current.state is not OllamaLeaseState.RELEASING:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' is not releasing its exact resource"
                )
            changes.update(
                {
                    "route_url": None,
                    "provider_health_url": None,
                    "accrued_estimated_cost": accrued_estimated_cost,
                    "updated_at": now,
                }
            )
            serialized = {key: _db_value(value) for key, value in changes.items()}
            assignments = ", ".join(f"{key} = ?" for key in serialized)
            cursor = connection.execute(
                f"""UPDATE ollama_leases
                    SET {assignments}, revision = revision + 1,
                        writer_epoch = writer_epoch + 1
                    WHERE lease_id = ? AND revision = ?""",
                (*serialized.values(), current.lease_id, current.revision),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise OllamaLeaseConflictError(
                    f"Lease '{lease.lease_id}' teardown changed concurrently"
                )
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (lease.lease_id,)
            ).fetchone()
            connection.commit()
        if row is None:
            raise RunPodManagerError("Torn-down Ollama lease disappeared")
        return _lease_from_row(row)


def lease_database_path(config: Mapping[str, Any]) -> Path:
    """Resolve the explicitly configured lease DB without a local fallback."""

    raw = os.getenv("RUNPOD_OLLAMA_LEASE_DB") or config.get("database_path")
    if not isinstance(raw, str) or not raw.strip():
        raise RunPodManagerError(
            "Configure RUNPOD_OLLAMA_LEASE_DB or ollama_leases.database_path"
        )
    path = Path(_expand_required_environment(raw))
    if not path.is_absolute():
        raise RunPodManagerError("Ollama lease database path must be absolute")
    return path


def request_from_lease(lease: OllamaLease) -> OllamaLeaseRequest:
    """Reconstruct a REQUESTED lease after a process crash before provisioning."""

    if lease.state is not OllamaLeaseState.REQUESTED or lease.mode is None:
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' is not a recoverable request"
        )
    if (
        lease.provisioning_evidence_state
        is not OllamaProvisioningEvidenceState.REQUIRED
        or lease.original_readiness_timeout_seconds is None
    ):
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' lacks original request authority"
        )
    try:
        raw: object = json.loads(lease.constraints_json)
    except json.JSONDecodeError as exc:
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' has corrupt constraints"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' has corrupt constraints"
        )
    try:
        constraints = OllamaResourceConstraints(
            min_vram_gb=_required_stored_int(raw, "min_vram_gb"),
            gpu_count=_required_stored_int(raw, "gpu_count"),
            cloud=CloudType(_required_stored_string(raw, "cloud")),
            min_cuda_version=_optional_stored_string(raw, "min_cuda_version"),
            allowed_gpu_ids=_string_tuple(raw, "allowed_gpu_ids"),
            allowed_gpu_pools=_string_tuple(raw, "allowed_gpu_pools"),
            allowed_data_center_ids=_string_tuple(raw, "allowed_data_center_ids"),
            max_hourly_rate=_optional_stored_number(raw, "max_hourly_rate"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' has corrupt constraints"
        ) from exc
    request = OllamaLeaseRequest(
        lease_id=lease.lease_id,
        owner_id=lease.owner_id,
        workload_id=lease.workload_id,
        model=lease.model,
        constraints=constraints,
        expected_session_seconds=lease.expected_session_seconds,
        expected_active_seconds=lease.expected_active_seconds,
        serverless_initialization_seconds=lease.serverless_initialization_seconds,
        serverless_idle_tail_seconds=lease.serverless_idle_tail_seconds,
        idle_timeout_seconds=lease.idle_timeout_seconds,
        readiness_timeout_seconds=lease.original_readiness_timeout_seconds,
        hard_deadline=lease.hard_deadline,
        max_authorized_cost=lease.max_authorized_cost,
        mode=lease.mode,
        requested_at=lease.original_requested_at,
    )
    if request.fingerprint == lease.request_fingerprint:
        return request
    # Writer-contract v1 accepted public integer costs before SQLite's REAL
    # affinity erased the spelling of max_authorized_cost. Preserve the exact
    # old fingerprint when it can be proven; never restamp it with the new
    # first-acceptance normalization.
    if (
        isinstance(lease.max_authorized_cost, float)
        and lease.max_authorized_cost.is_integer()
    ):
        legacy_request = replace(
            request, max_authorized_cost=int(lease.max_authorized_cost)
        )
        if legacy_request.fingerprint == lease.request_fingerprint:
            return legacy_request
    raise RunPodManagerError(
        f"Ollama lease '{lease.lease_id}' original request authority is corrupt"
    )


def _lease_from_row(row: sqlite3.Row) -> OllamaLease:
    intent = _stored_provisioning_intent(row["provisioning_intent_json"])
    receipt = _stored_provisioning_receipt(row["provisioning_receipt_json"])
    mismatch = _stored_provisioning_mismatch(row["provisioning_mismatch_json"])
    candidate_mismatch = _stored_provisioning_mismatch(
        row["ambiguous_candidate_mismatch_json"]
    )
    cleanup_intent = _stored_cleanup_intent(row["cleanup_intent_json"])
    cleanup_receipt = _stored_cleanup_receipt(row["cleanup_receipt_json"])
    additional_cleanup_targets = _stored_json_array(
        row["additional_cleanup_targets_json"],
        "additional cleanup targets",
        OllamaAdditionalCleanupTarget.from_dict,
    )
    telemetry_observations = _stored_json_array(
        row["runtime_telemetry_observations_json"],
        "runtime telemetry observations",
        OllamaRuntimeTelemetryObservation.from_dict,
    )
    billing_observations = _stored_json_array(
        row["lifecycle_billing_observations_json"],
        "lifecycle billing observations",
        OllamaLifecycleBillingObservation.from_dict,
    )
    billing_selection = _stored_billing_selection(
        row["lifecycle_billing_selection_json"]
    )
    billing_corrections = _stored_json_array(
        row["lifecycle_billing_corrections_json"],
        "lifecycle billing corrections",
        OllamaLifecycleBillingCorrection.from_dict,
    )
    billing_latest_reference = _stored_billing_latest_reference(
        row["lifecycle_billing_latest_reference_json"]
    )
    if receipt is not None:
        if intent is None:
            raise RunPodManagerError(
                "Stored Ollama provisioning receipt has no original intent"
            )
        try:
            receipt.validate_intent(intent)
        except ValueError as exc:
            raise RunPodManagerError(
                "Stored Ollama provisioning receipt does not bind its intent"
            ) from exc
    if mismatch is not None:
        if intent is None:
            raise RunPodManagerError(
                "Stored Ollama provisioning mismatch has no original intent"
            )
        try:
            mismatch.validate_intent(intent)
        except ValueError as exc:
            raise RunPodManagerError(
                "Stored Ollama provisioning mismatch does not bind its intent"
            ) from exc
    if candidate_mismatch is not None:
        if intent is None:
            raise RunPodManagerError(
                "Stored Ollama candidate mismatch has no original intent"
            )
        try:
            candidate_mismatch.validate_intent(intent)
        except ValueError as exc:
            raise RunPodManagerError(
                "Stored Ollama candidate mismatch does not bind its intent"
            ) from exc
    lease = OllamaLease(
        lease_id=row["lease_id"],
        owner_id=row["owner_id"],
        workload_id=row["workload_id"],
        request_fingerprint=row["request_fingerprint"],
        original_requested_at=_optional_datetime(row["original_requested_at"]),
        original_readiness_timeout_seconds=row["original_readiness_timeout_seconds"],
        model=row["model"],
        constraints_json=row["constraints_json"],
        mode=OllamaLeaseMode(row["mode"]) if row["mode"] else None,
        resource_type=(
            OllamaResourceType(row["resource_type"]) if row["resource_type"] else None
        ),
        provider_resource_id=row["provider_resource_id"],
        ambiguous_create_candidate_id=row["ambiguous_create_candidate_id"],
        resource_name=row["resource_name"],
        creation_uncertain=bool(row["creation_uncertain"]),
        provision_attempt_id=row["provision_attempt_id"],
        provision_attempts=row["provision_attempts"],
        route_url=row["route_url"],
        provider_health_url=row["provider_health_url"],
        state=OllamaLeaseState(row["state"]),
        teardown_state=OllamaTeardownState(row["teardown_state"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        provisioning_started_at=_optional_datetime(row["provisioning_started_at"]),
        ready_at=_optional_datetime(row["ready_at"]),
        last_used_at=_datetime(row["last_used_at"]),
        idle_deadline=_datetime(row["idle_deadline"]),
        hard_deadline=_datetime(row["hard_deadline"]),
        readiness_deadline=_datetime(row["readiness_deadline"]),
        model_pull_started_at=_optional_datetime(row["model_pull_started_at"]),
        model_pull_attempts=row["model_pull_attempts"],
        model_ready_at=_optional_datetime(row["model_ready_at"]),
        expected_session_seconds=row["expected_session_seconds"],
        expected_active_seconds=row["expected_active_seconds"],
        serverless_initialization_seconds=row["serverless_initialization_seconds"],
        serverless_idle_tail_seconds=row["serverless_idle_tail_seconds"],
        idle_timeout_seconds=row["idle_timeout_seconds"],
        offered_rate_per_hr=row["offered_rate_per_hr"],
        placement_gpu_count=(
            row["placement_gpu_count"] if row["placement_gpu_count"] is not None else 1
        ),
        estimated_cost=row["estimated_cost"],
        estimated_compute_cost=row["estimated_compute_cost"],
        maximum_compute_cost=row["maximum_compute_cost"],
        estimated_non_compute_cost=row["estimated_non_compute_cost"],
        maximum_non_compute_cost=row["maximum_non_compute_cost"],
        cost_ceiling=row["cost_ceiling"],
        cost_policy_components=_stored_cost_components(
            row["cost_policy_components_json"]
        ),
        maximum_concurrent_workers=row["maximum_concurrent_workers"],
        estimated_billable_seconds=row["estimated_billable_seconds"],
        maximum_billable_seconds=row["maximum_billable_seconds"],
        accrued_estimated_cost=row["accrued_estimated_cost"],
        max_authorized_cost=row["max_authorized_cost"],
        cold_start_seconds=row["cold_start_seconds"],
        selected_gpu_id=row["selected_gpu_id"],
        selected_gpu_pool=row["selected_gpu_pool"],
        selected_gpu_name=row["selected_gpu_name"],
        catalog_observed_at=_optional_datetime(row["catalog_observed_at"]),
        last_provider_error=row["last_provider_error"],
        termination_reason=row["termination_reason"],
        teardown_attempts=row["teardown_attempts"],
        revision=row["revision"],
        writer_contract_version=row["writer_contract_version"],
        writer_epoch=row["writer_epoch"],
        reconcile_after=_optional_datetime(row["reconcile_after"]),
        provisioning_evidence_state=OllamaProvisioningEvidenceState(
            row["provisioning_evidence_state"]
        ),
        provisioning_intent=intent,
        provisioning_receipt=receipt,
        provisioning_mismatch=mismatch,
        ambiguous_candidate_mismatch=candidate_mismatch,
        runtime_telemetry_binding_sha256=row["runtime_telemetry_binding_sha256"],
        cleanup_intent=cleanup_intent,
        cleanup_receipt=cleanup_receipt,
        additional_cleanup_targets=additional_cleanup_targets,
        runtime_telemetry_state=OllamaRuntimeTelemetryState(
            row["runtime_telemetry_state"]
        ),
        runtime_telemetry_observations=telemetry_observations,
        runtime_telemetry_last_provider_error=(
            row["runtime_telemetry_last_provider_error"]
        ),
        lifecycle_billing_state=OllamaLifecycleBillingState(
            row["lifecycle_billing_state"]
        ),
        lifecycle_billing_observations=billing_observations,
        lifecycle_billing_selection=billing_selection,
        lifecycle_billing_corrections=billing_corrections,
        lifecycle_billing_latest_reference=billing_latest_reference,
        billing_last_checked_at=_optional_datetime(row["billing_last_checked_at"]),
        billing_next_check_at=_optional_datetime(row["billing_next_check_at"]),
        billing_last_provider_error=row["billing_last_provider_error"],
        billing_retention_exhausted=bool(row["billing_retention_exhausted"]),
    )
    _validate_stored_provisioning_evidence(lease)
    _validate_stored_lifecycle_evidence(lease)
    return lease


def _validate_stored_lifecycle_evidence(lease: OllamaLease) -> None:
    intent = lease.provisioning_intent
    provisioning = lease.provisioning_receipt
    if intent is None and (
        lease.runtime_telemetry_binding_sha256 is not None
        or lease.cleanup_intent is not None
        or lease.cleanup_receipt is not None
        or lease.additional_cleanup_targets
        or lease.runtime_telemetry_observations
        or lease.lifecycle_billing_observations
        or lease.lifecycle_billing_selection is not None
        or lease.lifecycle_billing_corrections
        or lease.lifecycle_billing_latest_reference.state
        is not OllamaLifecycleBillingLatestState.NOT_OBSERVED
    ):
        raise RunPodManagerError(
            "Stored Ollama lifecycle evidence has no original intent"
        )
    if intent is not None and (
        lease.runtime_telemetry_binding_sha256
        != runtime_telemetry_binding_for_intent(intent)
    ):
        raise RunPodManagerError("Stored Ollama telemetry binding is invalid")
    if lease.cleanup_intent is not None:
        if intent is None:
            raise RunPodManagerError("Stored Ollama cleanup has no original intent")
        try:
            lease.cleanup_intent.validate_provisioning(intent, provisioning)
        except ValueError as exc:
            raise RunPodManagerError("Stored Ollama cleanup intent is unbound") from exc
    if lease.cleanup_receipt is not None:
        if lease.cleanup_intent is None:
            raise RunPodManagerError("Stored Ollama cleanup receipt has no intent")
        try:
            lease.cleanup_receipt.validate_intent(lease.cleanup_intent)
        except ValueError as exc:
            raise RunPodManagerError(
                "Stored Ollama cleanup receipt is unbound"
            ) from exc
        primary_cleanup_is_partial = (
            lease.state is OllamaLeaseState.RELEASING
            and lease.teardown_state
            in {
                OllamaTeardownState.PENDING,
                OllamaTeardownState.RETRYABLE_FAILURE,
            }
            and bool(lease.additional_cleanup_targets)
        )
        if not primary_cleanup_is_partial and (
            lease.state is not OllamaLeaseState.TERMINATED
            or lease.teardown_state is not OllamaTeardownState.COMPLETE
        ):
            raise RunPodManagerError("Stored Ollama cleanup receipt is not terminal")
    for target in lease.additional_cleanup_targets:
        if intent is None:
            raise RunPodManagerError(
                "Stored Ollama additional cleanup has no original intent"
            )
        try:
            target.validate_intent(intent)
        except ValueError as exc:
            raise RunPodManagerError(
                "Stored Ollama additional cleanup target is unbound"
            ) from exc
    if lease.state is OllamaLeaseState.TERMINATED and any(
        target.cleanup_receipt is None for target in lease.additional_cleanup_targets
    ):
        raise RunPodManagerError("Stored Ollama additional cleanup is incomplete")
    telemetry_ids: set[str] = set()
    for observation in lease.runtime_telemetry_observations:
        if (
            intent is None
            or provisioning is None
            or lease.runtime_telemetry_binding_sha256 is None
        ):
            raise RunPodManagerError("Stored Ollama telemetry lacks provisioning")
        try:
            observation.validate_provisioning(
                intent,
                provisioning,
                telemetry_binding_sha256=lease.runtime_telemetry_binding_sha256,
            )
        except ValueError as exc:
            raise RunPodManagerError("Stored Ollama telemetry is unbound") from exc
        if observation.observation_id in telemetry_ids:
            raise RunPodManagerError("Stored Ollama telemetry is duplicated")
        telemetry_ids.add(observation.observation_id)
    if (
        lease.runtime_telemetry_state
        in {
            OllamaRuntimeTelemetryState.STARTUP_COMPLETE,
            OllamaRuntimeTelemetryState.COMPLETE,
        }
        and not lease.runtime_telemetry_observations
    ):
        raise RunPodManagerError("Stored Ollama telemetry state has no observation")
    if lease.runtime_telemetry_state is OllamaRuntimeTelemetryState.COMPLETE and (
        lease.cleanup_intent is None
        or lease.state not in {OllamaLeaseState.RELEASING, OllamaLeaseState.TERMINATED}
        or lease.runtime_telemetry_observations[-1].snapshot.inference_in_flight != 0
    ):
        raise RunPodManagerError("Stored Ollama lifecycle telemetry is incomplete")
    billing_by_id: dict[str, OllamaLifecycleBillingObservation] = {}
    for observation in lease.lifecycle_billing_observations:
        if (
            intent is None
            or provisioning is None
            or lease.cleanup_intent is None
            or lease.cleanup_receipt is None
        ):
            raise RunPodManagerError("Stored Ollama billing lacks lifecycle evidence")
        try:
            observation.validate_lifecycle(
                intent,
                provisioning,
                lease.cleanup_intent,
                lease.cleanup_receipt,
            )
        except ValueError as exc:
            raise RunPodManagerError("Stored Ollama billing is unbound") from exc
        if observation.observation_id in billing_by_id:
            raise RunPodManagerError("Stored Ollama billing is duplicated")
        billing_by_id[observation.observation_id] = observation
    selection = lease.lifecycle_billing_selection
    if selection is not None and selection.observation_id not in billing_by_id:
        raise RunPodManagerError("Stored Ollama billing selection is unbound")
    latest = lease.lifecycle_billing_latest_reference
    if latest.state is OllamaLifecycleBillingLatestState.KNOWN:
        if (
            latest.observation_id not in billing_by_id
            or latest.received_at is None
            or lease.billing_last_checked_at is None
            or latest.received_at > lease.billing_last_checked_at
        ):
            raise RunPodManagerError(
                "Stored Ollama billing latest reference is unbound"
            )
    elif latest.state is OllamaLifecycleBillingLatestState.HISTORICAL_UNKNOWN:
        if not billing_by_id:
            raise RunPodManagerError(
                "Stored Ollama historical billing reference has no evidence"
            )
    elif billing_by_id:
        raise RunPodManagerError("Stored Ollama billing evidence has no latest state")
    correction_ids: set[str] = set()
    for correction in lease.lifecycle_billing_corrections:
        selected = billing_by_id.get(correction.selected_observation_id)
        revised = billing_by_id.get(correction.revised_observation_id)
        if (
            selected is None
            or revised is None
            or (
                selection is None
                or selection.observation_id != correction.selected_observation_id
            )
        ):
            raise RunPodManagerError("Stored Ollama billing correction is unbound")
        expected = exact_decimal_sum(
            (
                revised.provider_capacity_actual_cost_usd,
                selected.provider_capacity_actual_cost_usd.copy_negate(),
            )
        )
        if correction.actual_cost_delta_usd != expected:
            raise RunPodManagerError("Stored Ollama billing correction is inexact")
        if correction.correction_id in correction_ids:
            raise RunPodManagerError("Stored Ollama billing correction is duplicated")
        correction_ids.add(correction.correction_id)
    state = lease.lifecycle_billing_state
    if state is OllamaLifecycleBillingState.UNKNOWN and (
        lease.lifecycle_billing_observations
        or selection is not None
        or lease.lifecycle_billing_corrections
        or latest.state is not OllamaLifecycleBillingLatestState.NOT_OBSERVED
    ):
        raise RunPodManagerError("Stored Ollama unknown billing has evidence")
    if state is OllamaLifecycleBillingState.NOT_APPLICABLE and (
        lease.writer_contract_version != _CURRENT_WRITER_CONTRACT_VERSION
        or lease.provisioning_evidence_state
        is not OllamaProvisioningEvidenceState.REQUIRED
        or intent is not None
        or lease.provider_resource_id is not None
        or lease.ambiguous_create_candidate_id is not None
        or lease.provision_attempt_id is not None
        or lease.provisioning_started_at is not None
        or lease.provision_attempts != 0
        or lease.creation_uncertain
        or lease.state is not OllamaLeaseState.TERMINATED
        or lease.teardown_state is not OllamaTeardownState.COMPLETE
        or latest.state is not OllamaLifecycleBillingLatestState.NOT_OBSERVED
    ):
        raise RunPodManagerError("Stored Ollama no-effect billing proof is invalid")
    if state is OllamaLifecycleBillingState.PENDING and (
        lease.lifecycle_billing_observations
        or selection is not None
        or lease.lifecycle_billing_corrections
        or latest.state is not OllamaLifecycleBillingLatestState.NOT_OBSERVED
    ):
        raise RunPodManagerError("Stored Ollama pending billing selected evidence")
    if state is OllamaLifecycleBillingState.SELECTED and (
        selection is None
        or lease.lifecycle_billing_corrections
        or latest.state
        not in {
            OllamaLifecycleBillingLatestState.KNOWN,
            OllamaLifecycleBillingLatestState.HISTORICAL_UNKNOWN,
        }
    ):
        raise RunPodManagerError("Stored Ollama selected billing state is invalid")
    if state is OllamaLifecycleBillingState.CORRECTION_PENDING and (
        selection is None
        or latest.state
        not in {
            OllamaLifecycleBillingLatestState.KNOWN,
            OllamaLifecycleBillingLatestState.HISTORICAL_UNKNOWN,
        }
        or (
            not lease.lifecycle_billing_corrections
            and not lease.billing_retention_exhausted
        )
    ):
        raise RunPodManagerError("Stored Ollama correction state is invalid")
    if lease.billing_retention_exhausted and state not in {
        OllamaLifecycleBillingState.PENDING,
        OllamaLifecycleBillingState.CORRECTION_PENDING,
    }:
        raise RunPodManagerError("Stored Ollama billing retention state is invalid")
    if (
        state
        in {
            OllamaLifecycleBillingState.PENDING,
            OllamaLifecycleBillingState.SELECTED,
            OllamaLifecycleBillingState.CORRECTION_PENDING,
        }
        and not lease.additional_cleanup_targets
        and (
            lease.state is not OllamaLeaseState.TERMINATED
            or lease.cleanup_receipt is None
            or lease.billing_next_check_at is None
        )
    ):
        raise RunPodManagerError("Stored Ollama billing schedule is invalid")


def _telemetry_continues_lineage(
    previous: OllamaRuntimeTelemetryObservation,
    current: OllamaRuntimeTelemetryObservation,
) -> bool:
    """Validate one cumulative snapshot without reconstructing missing lineage."""

    before = previous.snapshot
    after = current.snapshot
    stable_before = (
        before.runtime_version,
        before.ollama_version,
        before.telemetry_binding_sha256,
        before.runtime_instance_id,
        before.provision_requested_at,
        before.container_started_at,
        before.ollama_process_started_at,
        before.ollama_healthy_at,
        before.model_pull_started_at,
        before.model_pull_completed_at,
        before.model_load_started_at,
        before.model_ready_at,
        before.ready_at,
        before.placement_and_image_pull_wall_ms,
        before.ollama_boot_ms,
        before.model_pull_ms,
        before.model_load_ms,
        before.container_to_ready_ms,
        before.cache_hit,
        before.startup_strategy,
        before.ready,
    )
    stable_after = (
        after.runtime_version,
        after.ollama_version,
        after.telemetry_binding_sha256,
        after.runtime_instance_id,
        after.provision_requested_at,
        after.container_started_at,
        after.ollama_process_started_at,
        after.ollama_healthy_at,
        after.model_pull_started_at,
        after.model_pull_completed_at,
        after.model_load_started_at,
        after.model_ready_at,
        after.ready_at,
        after.placement_and_image_pull_wall_ms,
        after.ollama_boot_ms,
        after.model_pull_ms,
        after.model_load_ms,
        after.container_to_ready_ms,
        after.cache_hit,
        after.startup_strategy,
        after.ready,
    )
    monotonic_before = (
        before.inference_started,
        before.inference_completed,
        before.inference_failed,
        before.inference_cancelled,
        before.inference_busy_ms,
        before.inference_idle_ms,
    )
    monotonic_after = (
        after.inference_started,
        after.inference_completed,
        after.inference_failed,
        after.inference_cancelled,
        after.inference_busy_ms,
        after.inference_idle_ms,
    )
    return (
        current.observed_at >= previous.observed_at
        and after.telemetry_revision > before.telemetry_revision
        and stable_after == stable_before
        and all(new >= old for old, new in zip(monotonic_before, monotonic_after))
        and _first_timestamp_is_monotonic(
            before.first_inference_started_at, after.first_inference_started_at
        )
        and _latest_timestamp_is_monotonic(
            before.last_inference_started_at, after.last_inference_started_at
        )
        and _latest_timestamp_is_monotonic(
            before.last_inference_completed_at, after.last_inference_completed_at
        )
    )


def _additional_cleanup_targets_are_evidence_prefix(
    stale: tuple[OllamaAdditionalCleanupTarget, ...],
    current: tuple[OllamaAdditionalCleanupTarget, ...],
) -> bool:
    """Accept only monotonic evidence added to the same owned target lineage."""

    if len(stale) > len(current):
        return False
    for before, after in zip(stale, current):
        if (
            before.intent_sha256 != after.intent_sha256
            or before.lease_id != after.lease_id
            or before.provision_attempt_id != after.provision_attempt_id
            or before.resource_type is not after.resource_type
            or before.resource_name != after.resource_name
            or before.provider_resource_id != after.provider_resource_id
            or before.source != after.source
            or (
                before.provisioning_mismatch is not None
                and before.provisioning_mismatch != after.provisioning_mismatch
            )
            or (
                before.cleanup_intent is not None
                and before.cleanup_intent != after.cleanup_intent
            )
            or (
                before.cleanup_receipt is not None
                and before.cleanup_receipt != after.cleanup_receipt
            )
        ):
            return False
    return True


class _CleanupLineagePhase(Enum):
    STRICT = "strict"
    PRIMARY_INTENT = "primary_intent"
    ADDITIONAL_INTENT = "additional_intent"
    ADDITIONAL_OUTCOME = "additional_outcome"


def _additional_cleanup_target(
    targets: tuple[OllamaAdditionalCleanupTarget, ...],
    provider_resource_id: str,
) -> tuple[int, OllamaAdditionalCleanupTarget] | None:
    return next(
        (
            (offset, target)
            for offset, target in enumerate(targets)
            if target.provider_resource_id == provider_resource_id
        ),
        None,
    )


def _cleanup_lineage_is_compatible(
    stale: OllamaLease,
    current: OllamaLease,
    *,
    phase: _CleanupLineagePhase = _CleanupLineagePhase.STRICT,
    provider_resource_id: str | None = None,
) -> bool:
    """Validate the immutable cleanup authority plus a monotonic target suffix."""

    if not isinstance(phase, _CleanupLineagePhase):
        return False
    if phase in {
        _CleanupLineagePhase.ADDITIONAL_INTENT,
        _CleanupLineagePhase.ADDITIONAL_OUTCOME,
    }:
        if provider_resource_id is None:
            return False
        stale_target_match = _additional_cleanup_target(
            stale.additional_cleanup_targets,
            provider_resource_id,
        )
        current_target_match = _additional_cleanup_target(
            current.additional_cleanup_targets,
            provider_resource_id,
        )
        if stale_target_match is None or current_target_match is None:
            return False
        if (
            phase is _CleanupLineagePhase.ADDITIONAL_OUTCOME
            and stale_target_match[1].cleanup_intent is None
        ):
            return False
    elif provider_resource_id is not None:
        return False

    primary_intent_is_compatible = current.cleanup_intent == stale.cleanup_intent
    if (
        phase is not _CleanupLineagePhase.STRICT
        and stale.cleanup_intent is None
    ):
        primary_intent_is_compatible = True

    return (
        current.lease_id == stale.lease_id
        and stale.writer_contract_version == _CURRENT_WRITER_CONTRACT_VERSION
        and current.writer_contract_version == stale.writer_contract_version
        and 0 <= stale.writer_epoch <= current.writer_epoch
        and 0 <= stale.revision <= current.revision
        and current.request_fingerprint == stale.request_fingerprint
        and current.provision_attempt_id == stale.provision_attempt_id
        and current.provisioning_intent == stale.provisioning_intent
        and current.resource_type is stale.resource_type
        and current.resource_name == stale.resource_name
        and current.provider_resource_id == stale.provider_resource_id
        and primary_intent_is_compatible
        and (
            stale.cleanup_receipt is None
            or current.cleanup_receipt == stale.cleanup_receipt
        )
        and _additional_cleanup_targets_are_evidence_prefix(
            stale.additional_cleanup_targets,
            current.additional_cleanup_targets,
        )
    )


def _same_provisioning_receipt_authority(
    first: OllamaProvisioningReceipt, second: OllamaProvisioningReceipt
) -> bool:
    """Compare immutable provider facts while ignoring poll-local metadata."""

    return first.same_provider_authority(second)


def _first_timestamp_is_monotonic(
    before: datetime | None, after: datetime | None
) -> bool:
    return before is None or before == after


def _latest_timestamp_is_monotonic(
    before: datetime | None, after: datetime | None
) -> bool:
    return before is None or (after is not None and after >= before)


def _validate_stored_provisioning_evidence(lease: OllamaLease) -> None:
    if (
        lease.writer_contract_version != _CURRENT_WRITER_CONTRACT_VERSION
        or lease.writer_epoch < 0
        or lease.reconcile_after is None
    ):
        raise RunPodManagerError("Stored Ollama writer contract is invalid")
    if (
        lease.provisioning_evidence_state
        is not OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
    ):
        if (
            lease.original_readiness_timeout_seconds is None
            or lease.original_readiness_timeout_seconds < 1
        ):
            raise RunPodManagerError(
                "Stored Ollama request lacks original fingerprint inputs"
            )
    elif (
        lease.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN
        and (
            lease.original_requested_at is not None
            or lease.original_readiness_timeout_seconds is not None
        )
    ):
        raise RunPodManagerError(
            "Historical Ollama request cannot claim original fingerprint inputs"
        )
    intent = lease.provisioning_intent
    if intent is None:
        if (
            lease.provisioning_evidence_state
            in {
                OllamaProvisioningEvidenceState.INTENT_RECORDED,
                OllamaProvisioningEvidenceState.COMPLETE,
            }
            or lease.provisioning_receipt is not None
            or lease.provisioning_mismatch is not None
            or lease.ambiguous_candidate_mismatch is not None
            or (
                lease.provisioning_evidence_state
                is OllamaProvisioningEvidenceState.REQUIRED
                and (
                    lease.provider_resource_id is not None
                    or lease.ambiguous_create_candidate_id is not None
                    or lease.resource_type is not None
                    or lease.resource_name is not None
                    or lease.provision_attempt_id is not None
                    or lease.provisioning_started_at is not None
                    or lease.creation_uncertain
                )
            )
        ):
            raise RunPodManagerError(
                "Stored Ollama provisioning evidence has no original intent"
            )
        return
    if lease.provisioning_evidence_state in {
        OllamaProvisioningEvidenceState.REQUIRED,
        OllamaProvisioningEvidenceState.HISTORICAL_UNKNOWN,
    }:
        raise RunPodManagerError(
            "Stored Ollama original intent has an incompatible evidence state"
        )
    if (
        lease.provisioning_evidence_state
        is OllamaProvisioningEvidenceState.INTENT_RECORDED
        and lease.state
        not in {
            OllamaLeaseState.PROVISIONING,
            OllamaLeaseState.RECONCILE_REQUIRED,
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.TERMINATED,
            OllamaLeaseState.FAILED,
        }
    ) or (
        lease.provisioning_evidence_state is OllamaProvisioningEvidenceState.COMPLETE
        and lease.state
        not in {
            OllamaLeaseState.WAITING_FOR_MODEL,
            OllamaLeaseState.READY,
            OllamaLeaseState.RELEASING,
            OllamaLeaseState.TERMINATED,
            OllamaLeaseState.FAILED,
        }
    ):
        raise RunPodManagerError(
            "Stored Ollama lifecycle state does not match its provisioning evidence"
        )
    if (
        lease.provisioning_evidence_state is OllamaProvisioningEvidenceState.COMPLETE
    ) != (lease.provisioning_receipt is not None):
        raise RunPodManagerError(
            "Stored Ollama provisioning receipt does not match its evidence state"
        )
    try:
        raw_constraints: object = json.loads(lease.constraints_json)
    except json.JSONDecodeError as exc:
        raise RunPodManagerError(
            "Stored Ollama provisioning intent has corrupt constraints"
        ) from exc
    if not isinstance(raw_constraints, Mapping):
        raise RunPodManagerError(
            "Stored Ollama provisioning intent has corrupt constraints"
        )
    offered_whole_rate = (
        exact_decimal_integer_scale(
            Decimal(str(lease.offered_rate_per_hr)), lease.placement_gpu_count
        )
        if lease.offered_rate_per_hr is not None
        else None
    )
    if (
        intent.lease_id != lease.lease_id
        or intent.request_fingerprint != lease.request_fingerprint
        or intent.provision_attempt_id != lease.provision_attempt_id
        or intent.mode is not lease.mode
        or intent.resource_type is not lease.resource_type
        or intent.resource_name != lease.resource_name
        or intent.provision_requested_at != lease.provisioning_started_at
        or intent.selected_gpu_id != lease.selected_gpu_id
        or intent.selected_gpu_pool != lease.selected_gpu_pool
        or intent.selected_gpu_name != lease.selected_gpu_name
        or intent.selected_gpu_count != lease.placement_gpu_count
        or intent.selected_cloud.value != raw_constraints.get("cloud")
        or intent.allowed_data_center_ids
        != _string_tuple(raw_constraints, "allowed_data_center_ids")
        or intent.accepted_whole_hourly_rate_usd != offered_whole_rate
    ):
        raise RunPodManagerError(
            "Stored Ollama provisioning intent does not bind its lease"
        )
    if lease.provisioning_receipt is not None and (
        lease.provider_resource_id != lease.provisioning_receipt.provider_resource_id
        or (
            lease.provisioning_receipt.reconciled_after_ambiguous_create
            and (
                lease.ambiguous_create_candidate_id is None
                or lease.provider_resource_id != lease.ambiguous_create_candidate_id
            )
        )
    ):
        raise RunPodManagerError(
            "Stored Ollama provisioning receipt does not bind its resource"
        )
    if (
        lease.ambiguous_create_candidate_id is not None
        and not lease.ambiguous_create_candidate_id
    ):
        raise RunPodManagerError("Stored Ollama ambiguous-create candidate is invalid")
    if lease.provisioning_mismatch is not None and (
        lease.provisioning_mismatch.provider_resource_id != lease.provider_resource_id
        or lease.provisioning_evidence_state
        not in {
            OllamaProvisioningEvidenceState.INTENT_RECORDED,
            OllamaProvisioningEvidenceState.COMPLETE,
        }
    ):
        raise RunPodManagerError(
            "Stored Ollama provisioning mismatch does not bind its resource"
        )
    if lease.ambiguous_candidate_mismatch is not None and (
        lease.ambiguous_candidate_mismatch.provider_resource_id
        != lease.ambiguous_create_candidate_id
        or lease.provisioning_evidence_state
        not in {
            OllamaProvisioningEvidenceState.INTENT_RECORDED,
            OllamaProvisioningEvidenceState.COMPLETE,
        }
    ):
        raise RunPodManagerError(
            "Stored Ollama candidate mismatch does not bind its locator"
        )


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_required_environment(value: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if not resolved:
            raise RunPodManagerError(
                f"Ollama lease database references unset environment variable '{name}'"
            )
        return resolved

    return _ENV_REFERENCE.sub(replace_match, value)


def _db_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return iso_datetime(value)
    if isinstance(
        value,
        (
            OllamaProvisioningIntent,
            OllamaProvisioningReceipt,
            OllamaProvisioningMismatchEvidence,
            OllamaCleanupIntent,
            OllamaCleanupReceipt,
            OllamaLifecycleBillingSelection,
            OllamaLifecycleBillingLatestReference,
            OllamaAdditionalCleanupTarget,
        ),
    ):
        return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))
    if isinstance(value, tuple) and all(
        isinstance(
            item,
            (
                OllamaRuntimeTelemetryObservation,
                OllamaLifecycleBillingObservation,
                OllamaLifecycleBillingCorrection,
                OllamaAdditionalCleanupTarget,
            ),
        )
        for item in value
    ):
        return json.dumps(
            [item.to_dict() for item in value],
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _stored_json(value: str | None, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RunPodManagerError(f"Stored Ollama {name} is corrupt") from exc
    if not isinstance(decoded, Mapping):
        raise RunPodManagerError(f"Stored Ollama {name} is corrupt")
    return decoded


def _stored_json_array(
    value: str,
    name: str,
    decoder: Any,
) -> tuple[Any, ...]:
    try:
        decoded: object = json.loads(value)
        if not isinstance(decoded, list):
            raise TypeError
        return tuple(decoder(item) for item in decoded)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RunPodManagerError(f"Stored Ollama {name} is corrupt") from exc


def _stored_cleanup_intent(value: str | None) -> OllamaCleanupIntent | None:
    decoded = _stored_json(value, "cleanup intent")
    if decoded is None:
        return None
    try:
        return OllamaCleanupIntent.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError("Stored Ollama cleanup intent is corrupt") from exc


def _stored_cleanup_receipt(value: str | None) -> OllamaCleanupReceipt | None:
    decoded = _stored_json(value, "cleanup receipt")
    if decoded is None:
        return None
    try:
        return OllamaCleanupReceipt.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError("Stored Ollama cleanup receipt is corrupt") from exc


def _stored_billing_selection(
    value: str | None,
) -> OllamaLifecycleBillingSelection | None:
    decoded = _stored_json(value, "lifecycle billing selection")
    if decoded is None:
        return None
    try:
        return OllamaLifecycleBillingSelection.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError(
            "Stored Ollama lifecycle billing selection is corrupt"
        ) from exc


def _stored_billing_latest_reference(
    value: str,
) -> OllamaLifecycleBillingLatestReference:
    decoded = _stored_json(value, "lifecycle billing latest reference")
    if decoded is None:
        raise RunPodManagerError(
            "Stored Ollama lifecycle billing latest reference is missing"
        )
    try:
        return OllamaLifecycleBillingLatestReference.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError(
            "Stored Ollama lifecycle billing latest reference is corrupt"
        ) from exc


def _stored_provisioning_intent(value: str | None) -> OllamaProvisioningIntent | None:
    decoded = _stored_json(value, "provisioning intent")
    if decoded is None:
        return None
    try:
        return OllamaProvisioningIntent.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError(
            "Stored Ollama provisioning intent is corrupt"
        ) from exc


def _stored_provisioning_receipt(
    value: str | None,
) -> OllamaProvisioningReceipt | None:
    decoded = _stored_json(value, "provisioning receipt")
    if decoded is None:
        return None
    try:
        return OllamaProvisioningReceipt.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError(
            "Stored Ollama provisioning receipt is corrupt"
        ) from exc


def _stored_provisioning_mismatch(
    value: str | None,
) -> OllamaProvisioningMismatchEvidence | None:
    decoded = _stored_json(value, "provisioning mismatch")
    if decoded is None:
        return None
    try:
        return OllamaProvisioningMismatchEvidence.from_dict(decoded)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError(
            "Stored Ollama provisioning mismatch is corrupt"
        ) from exc


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    require_aware(parsed, "stored timestamp")
    return parsed.astimezone(UTC)


def _optional_datetime(value: str | None) -> datetime | None:
    return _datetime(value) if value else None


def _string_tuple(value: Mapping[str, Any], name: str) -> tuple[str, ...]:
    items = value.get(name, ())
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise TypeError(f"Stored Ollama {name} must be a string list")
    if any(not item for item in items):
        raise ValueError(f"Stored Ollama {name} must be a string list")
    return tuple(items)


def _required_stored_int(value: Mapping[str, Any], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"Stored Ollama {name} must be an integer")
    return item


def _required_stored_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str):
        raise TypeError(f"Stored Ollama {name} must be a string")
    if not item:
        raise ValueError(f"Stored Ollama {name} must be a string")
    return item


def _optional_stored_string(value: Mapping[str, Any], name: str) -> str | None:
    item = value.get(name)
    if item is not None and not isinstance(item, str):
        raise TypeError(f"Stored Ollama {name} must be a string or null")
    return item


def _optional_stored_number(value: Mapping[str, Any], name: str) -> int | float | None:
    item = value.get(name)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise TypeError(f"Stored Ollama {name} must be a finite number or null")
    if not math.isfinite(item):
        raise ValueError(f"Stored Ollama {name} must be a finite number or null")
    return item


def _stored_cost_components(
    value: str | None,
) -> tuple[OllamaNonComputeCostComponent, ...]:
    if value is None:
        return ()
    try:
        raw: object = json.loads(value)
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise TypeError
        components = tuple(OllamaNonComputeCostComponent(item) for item in raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RunPodManagerError(
            "Stored Ollama non-compute cost components are corrupt"
        ) from exc
    if len(set(components)) != len(components):
        raise RunPodManagerError(
            "Stored Ollama non-compute cost components are corrupt"
        )
    return components
