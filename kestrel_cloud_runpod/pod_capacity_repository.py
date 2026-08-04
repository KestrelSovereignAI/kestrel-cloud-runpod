"""SQLite WAL persistence for restart-safe Pod capacity ownership."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .models import RunPodManagerError
from .pod_capacity_contracts import (
    CatalogPodWorkloadState,
    PodBillingReceipt,
    PodCapacityBillingState,
    PodCapacityEvidence,
    PodCapacitySpec,
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodLease,
    TrainingPodOwnership,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
    iso_datetime,
)


class SQLitePodCapacityRepository:
    """Transactional canonical lease store with revision CAS and migration."""

    _COLUMNS = frozenset(
        {
            "provider_pod_id",
            "ownership",
            "state",
            "cleanup_state",
            "family_release_requested",
            "family_release_complete",
            "creation_uncertain",
            "backend_base_url",
            "provider_job_id",
            "updated_at",
            "last_heartbeat_at",
            "last_provider_error",
            "stop_attempts",
            "workload_state",
            "workload_error_type",
            "billing_state",
            "billing_receipt_json",
            "terminated_at",
            "evidence_json",
        }
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("Training Pod database path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            existing = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if (
                "training_pod_leases" in existing
                and "pod_capacity_leases" not in existing
            ):
                connection.execute(
                    "ALTER TABLE training_pod_leases RENAME TO pod_capacity_leases"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pod_capacity_leases (
                    cleanup_token TEXT PRIMARY KEY,
                    root_cleanup_token TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    companion_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    resource_name TEXT NOT NULL,
                    provider_pod_id TEXT,
                    ownership TEXT NOT NULL,
                    state TEXT NOT NULL,
                    cleanup_state TEXT NOT NULL,
                    family_release_requested INTEGER NOT NULL DEFAULT 0,
                    family_release_complete INTEGER NOT NULL DEFAULT 0,
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
                    capacity_spec_json TEXT,
                    workload_state TEXT NOT NULL DEFAULT 'not_submitted',
                    workload_error_type TEXT,
                    billing_state TEXT NOT NULL DEFAULT 'not_applicable',
                    billing_receipt_json TEXT,
                    terminated_at TEXT,
                    evidence_json TEXT,
                    revision INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(pod_capacity_leases)"
                ).fetchall()
            }
            if "root_cleanup_token" not in columns:
                connection.execute(
                    "ALTER TABLE pod_capacity_leases ADD COLUMN "
                    "root_cleanup_token TEXT NOT NULL DEFAULT ''"
                )
            # Old fallback hashes cannot be inverted. They are initially
            # self-rooted and are associated with a caller root at query time
            # by recomputing deterministic fallback identities. Run this on
            # every open so an interrupted ALTER/backfill resumes safely.
            connection.execute(
                "UPDATE pod_capacity_leases "
                "SET root_cleanup_token = cleanup_token "
                "WHERE root_cleanup_token IS NULL OR root_cleanup_token = ''"
            )
            if "family_release_requested" not in columns:
                connection.execute(
                    "ALTER TABLE pod_capacity_leases "
                    "ADD COLUMN family_release_requested INTEGER NOT NULL DEFAULT 0"
                )
            if "family_release_complete" not in columns:
                connection.execute(
                    "ALTER TABLE pod_capacity_leases "
                    "ADD COLUMN family_release_complete INTEGER NOT NULL DEFAULT 0"
                )
            generic_columns = {
                "capacity_spec_json": "TEXT",
                "workload_state": "TEXT NOT NULL DEFAULT 'not_submitted'",
                "workload_error_type": "TEXT",
                "billing_state": "TEXT NOT NULL DEFAULT 'not_applicable'",
                "billing_receipt_json": "TEXT",
                "terminated_at": "TEXT",
                "evidence_json": "TEXT",
            }
            for name, definition in generic_columns.items():
                if name not in columns:
                    connection.execute(
                        "ALTER TABLE pod_capacity_leases ADD COLUMN "
                        f"{name} {definition}"
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_pod_capacity_active_resource_name
                ON pod_capacity_leases(resource_name)
                WHERE state != 'released'
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_pod_capacity_active_provider_pod
                ON pod_capacity_leases(provider_pod_id)
                WHERE state != 'released' AND provider_pod_id IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_pod_capacity_active_cleanup_family
                ON pod_capacity_leases(root_cleanup_token)
                WHERE state != 'released'
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pod_capacity_reconcile
                ON pod_capacity_leases(state, hard_deadline, last_heartbeat_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pod_capacity_cleanup_family
                ON pod_capacity_leases(root_cleanup_token, state, created_at)
                """
            )
            for legacy_index in (
                "uq_training_active_resource_name",
                "uq_training_active_provider_pod",
                "uq_training_active_cleanup_family",
                "idx_training_pods_reconcile",
                "idx_training_pods_cleanup_family",
            ):
                connection.execute(f"DROP INDEX IF EXISTS {legacy_index}")
            connection.execute("PRAGMA user_version = 3")

    def reserve(self, request: TrainingPodRequest) -> tuple[TrainingPodLease, bool]:
        """Persist an acquisition claim before any provider network operation."""

        now = iso_datetime(request.created_at)
        values = (
            request.cleanup_token,
            request.cleanup_family_token,
            request.fingerprint,
            request.companion_id,
            request.profile_id,
            request.source.value,
            request.resource_name,
            request.provider_pod_id,
            TrainingPodOwnership.PROVISIONAL.value,
            TrainingPodState.REQUESTED.value,
            TrainingPodCleanupState.NOT_REQUESTED.value,
            (
                json.dumps(
                    request.capacity_spec.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if request.capacity_spec is not None
                else None
            ),
            (
                PodCapacityBillingState.PENDING.value
                if request.capacity_spec is not None
                else PodCapacityBillingState.NOT_APPLICABLE.value
            ),
            (
                json.dumps(
                    PodCapacityEvidence.from_spec(request.capacity_spec).to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if request.capacity_spec is not None
                else None
            ),
            now,
            now,
            now,
            iso_datetime(request.readiness_deadline),
            iso_datetime(request.hard_deadline),
        )
        inserted = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if request.cleanup_family_token != request.cleanup_token:
                    root = connection.execute(
                        """SELECT family_release_requested
                           FROM pod_capacity_leases
                           WHERE cleanup_token = ?
                             AND root_cleanup_token = cleanup_token""",
                        (request.cleanup_family_token,),
                    ).fetchone()
                    if root is None:
                        connection.rollback()
                        raise TrainingPodConflictError(
                            "A fallback training attempt requires its persisted "
                            "root cleanup token"
                        )
                    if bool(root["family_release_requested"]):
                        connection.rollback()
                        raise TrainingPodConflictError(
                            f"Training cleanup family '{request.cleanup_family_token}' "
                            "is already releasing"
                        )
                connection.execute(
                    """
                    INSERT INTO pod_capacity_leases (
                        cleanup_token, root_cleanup_token, request_fingerprint,
                        companion_id, profile_id, source, resource_name,
                        provider_pod_id, ownership, state, cleanup_state,
                        capacity_spec_json, billing_state, evidence_json,
                        family_release_requested, family_release_complete, created_at,
                        updated_at, last_heartbeat_at, readiness_deadline, hard_deadline
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (*values[:14], False, False, *values[14:]),
                )
                inserted = True
            except sqlite3.IntegrityError as exc:
                row = connection.execute(
                    "SELECT * FROM pod_capacity_leases WHERE cleanup_token = ?",
                    (request.cleanup_token,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    raise TrainingPodConflictError(
                        "The requested training Pod is already claimed by another "
                        "active cleanup token"
                    ) from exc
            row = connection.execute(
                "SELECT * FROM pod_capacity_leases WHERE cleanup_token = ?",
                (request.cleanup_token,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RunPodManagerError("Failed to persist training Pod ownership")
        lease = _lease_from_row(row)
        if lease.request_fingerprint != request.fingerprint:
            raise TrainingPodConflictError(
                f"Training cleanup token '{request.cleanup_token}' already represents "
                "a different request"
            )
        return lease, inserted

    def get(self, cleanup_token: str) -> TrainingPodLease | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pod_capacity_leases WHERE cleanup_token = ?",
                (cleanup_token,),
            ).fetchone()
        return _lease_from_row(row) if row is not None else None

    def list_cleanup_family(
        self,
        root_cleanup_token: str,
        *,
        legacy_cleanup_tokens: tuple[str, ...] = (),
    ) -> tuple[TrainingPodLease, ...]:
        """Return a root family, including configured pre-migration child hashes."""

        candidate_tokens = tuple(
            dict.fromkeys((root_cleanup_token, *legacy_cleanup_tokens))
        )
        placeholders = ", ".join("?" for _ in candidate_tokens)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM pod_capacity_leases
                    WHERE root_cleanup_token = ?
                       OR cleanup_token IN ({placeholders})
                    ORDER BY created_at, cleanup_token""",
                (root_cleanup_token, *candidate_tokens),
            ).fetchall()
        return tuple(_lease_from_row(row) for row in rows)

    def list_incomplete_family_releases(self) -> tuple[TrainingPodLease, ...]:
        """Return durable root cleanup requests that a restart must finish."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM pod_capacity_leases
                   WHERE cleanup_token = root_cleanup_token
                     AND family_release_requested = 1
                     AND family_release_complete = 0
                   ORDER BY created_at, cleanup_token"""
            ).fetchall()
        return tuple(_lease_from_row(row) for row in rows)

    def compare_and_set(
        self, lease: TrainingPodLease, *, changes: Mapping[str, Any]
    ) -> TrainingPodLease:
        unknown = set(changes).difference(self._COLUMNS)
        if unknown:
            raise ValueError(f"Unsupported training Pod updates: {sorted(unknown)}")
        serialized = {key: _db_value(value) for key, value in changes.items()}
        serialized["updated_at"] = serialized.get(
            "updated_at", iso_datetime(datetime.now(UTC))
        )
        assignments = ", ".join(f"{key} = ?" for key in serialized)
        values = [*serialized.values(), lease.cleanup_token, lease.revision]
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    f"""UPDATE pod_capacity_leases
                        SET {assignments}, revision = revision + 1
                        WHERE cleanup_token = ? AND revision = ?""",
                    values,
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise TrainingPodConflictError(
                        f"Training cleanup token '{lease.cleanup_token}' changed "
                        "concurrently; reconcile it"
                    )
                row = connection.execute(
                    "SELECT * FROM pod_capacity_leases WHERE cleanup_token = ?",
                    (lease.cleanup_token,),
                ).fetchone()
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise TrainingPodConflictError(
                "The provider Pod is already claimed by another active training lease"
            ) from exc
        if row is None:
            raise RunPodManagerError("Updated training Pod lease disappeared")
        return _lease_from_row(row)

    def list_for_reconciliation(self) -> tuple[TrainingPodLease, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM pod_capacity_leases
                   WHERE state != 'released'
                      OR billing_state IN ('pending', 'unresolved')
                   ORDER BY created_at"""
            ).fetchall()
        return tuple(_lease_from_row(row) for row in rows)


def pod_capacity_database_path(config: Mapping[str, Any]) -> Path:
    """Resolve the mandatory durable state path without a local fallback."""

    raw = (
        os.getenv("RUNPOD_POD_CAPACITY_LEASE_DB")
        or os.getenv("RUNPOD_TRAINING_LEASE_DB")
        or config.get("database_path")
    )
    if not isinstance(raw, str) or not raw.strip():
        raise RunPodManagerError(
            "Configure RUNPOD_POD_CAPACITY_LEASE_DB or an absolute database_path"
        )
    path = Path(_expand_required_environment(raw))
    if not path.is_absolute():
        raise RunPodManagerError("Training Pod database path must be absolute")
    return path


SQLiteTrainingPodRepository = SQLitePodCapacityRepository
training_database_path = pod_capacity_database_path


def _lease_from_row(row: sqlite3.Row) -> TrainingPodLease:
    billing_receipt = (
        PodBillingReceipt.from_dict(json.loads(row["billing_receipt_json"]))
        if row["billing_receipt_json"] is not None
        else None
    )
    evidence = (
        PodCapacityEvidence.from_dict(json.loads(row["evidence_json"]))
        if row["evidence_json"] is not None
        else None
    )
    if evidence is not None and evidence.billing != billing_receipt:
        raise RunPodManagerError(
            "Stored Pod capacity billing and evidence receipts are inconsistent"
        )
    return TrainingPodLease(
        cleanup_token=row["cleanup_token"],
        root_cleanup_token=row["root_cleanup_token"],
        request_fingerprint=row["request_fingerprint"],
        companion_id=row["companion_id"],
        profile_id=row["profile_id"],
        source=TrainingPodSource(row["source"]),
        resource_name=row["resource_name"],
        provider_pod_id=row["provider_pod_id"],
        ownership=TrainingPodOwnership(row["ownership"]),
        state=TrainingPodState(row["state"]),
        cleanup_state=TrainingPodCleanupState(row["cleanup_state"]),
        family_release_requested=bool(row["family_release_requested"]),
        family_release_complete=bool(row["family_release_complete"]),
        creation_uncertain=bool(row["creation_uncertain"]),
        backend_base_url=row["backend_base_url"],
        provider_job_id=row["provider_job_id"],
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        last_heartbeat_at=_datetime(row["last_heartbeat_at"]),
        readiness_deadline=_datetime(row["readiness_deadline"]),
        hard_deadline=_datetime(row["hard_deadline"]),
        last_provider_error=row["last_provider_error"],
        stop_attempts=row["stop_attempts"],
        revision=row["revision"],
        capacity_spec=(
            PodCapacitySpec.from_dict(json.loads(row["capacity_spec_json"]))
            if row["capacity_spec_json"] is not None
            else None
        ),
        workload_state=CatalogPodWorkloadState(row["workload_state"]),
        workload_error_type=row["workload_error_type"],
        billing_state=PodCapacityBillingState(row["billing_state"]),
        billing_receipt=billing_receipt,
        terminated_at=(
            _datetime(row["terminated_at"])
            if row["terminated_at"] is not None
            else None
        ),
        evidence=evidence,
    )


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_required_environment(value: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if not resolved:
            raise RunPodManagerError(
                f"Training Pod database references unset environment variable '{name}'"
            )
        return resolved

    return _ENV_REFERENCE.sub(replace_match, value)


def _db_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return iso_datetime(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, PodBillingReceipt):
        return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))
    if isinstance(value, PodCapacityEvidence):
        return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))
    return value


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunPodManagerError("Stored training Pod timestamp is not timezone-aware")
    return parsed.astimezone(UTC)
