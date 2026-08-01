"""Transactional SQLite persistence for private-Ollama leases."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .models import CloudType, RunPodManagerError
from .ollama_contracts import (
    OllamaLease,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaLeaseState,
    OllamaResourceConstraints,
    OllamaResourceType,
    OllamaTeardownState,
    iso_datetime,
    require_aware,
)


class SQLiteOllamaLeaseRepository:
    """SQLite WAL store with revision-based compare-and-set updates."""

    _COLUMNS = frozenset(
        {
            "mode",
            "resource_type",
            "provider_resource_id",
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
            "estimated_cost",
            "estimated_billable_seconds",
            "accrued_estimated_cost",
            "cold_start_seconds",
            "selected_gpu_id",
            "selected_gpu_pool",
            "selected_gpu_name",
            "catalog_observed_at",
            "last_provider_error",
            "teardown_attempts",
        }
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("Ollama lease database path must be absolute")
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ollama_leases (
                    lease_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workload_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    model TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    mode TEXT,
                    resource_type TEXT,
                    provider_resource_id TEXT,
                    resource_name TEXT,
                    creation_uncertain INTEGER NOT NULL DEFAULT 0,
                    provision_attempt_id TEXT,
                    provision_attempts INTEGER NOT NULL DEFAULT 0,
                    route_url TEXT,
                    provider_health_url TEXT,
                    state TEXT NOT NULL,
                    teardown_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    provisioning_started_at TEXT,
                    ready_at TEXT,
                    last_used_at TEXT NOT NULL,
                    idle_deadline TEXT NOT NULL,
                    hard_deadline TEXT NOT NULL,
                    readiness_deadline TEXT NOT NULL,
                    model_pull_started_at TEXT,
                    model_pull_attempts INTEGER NOT NULL DEFAULT 0,
                    model_ready_at TEXT,
                    expected_session_seconds INTEGER NOT NULL,
                    expected_active_seconds INTEGER NOT NULL,
                    serverless_initialization_seconds INTEGER NOT NULL,
                    serverless_idle_tail_seconds INTEGER NOT NULL,
                    idle_timeout_seconds INTEGER NOT NULL,
                    offered_rate_per_hr REAL,
                    estimated_cost REAL,
                    estimated_billable_seconds INTEGER,
                    accrued_estimated_cost REAL NOT NULL DEFAULT 0,
                    max_authorized_cost REAL NOT NULL,
                    cold_start_seconds REAL,
                    selected_gpu_id TEXT,
                    selected_gpu_pool TEXT,
                    selected_gpu_name TEXT,
                    catalog_observed_at TEXT,
                    last_provider_error TEXT,
                    teardown_attempts INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ollama_leases_reconcile
                ON ollama_leases(state, idle_deadline, hard_deadline)
                """
            )

    def insert_request(
        self, request: OllamaLeaseRequest, *, now: datetime
    ) -> tuple[OllamaLease, bool]:
        constraints = asdict(request.constraints)
        constraints["cloud"] = request.constraints.cloud.value
        values = (
            request.lease_id,
            request.owner_id,
            request.workload_id,
            request.fingerprint,
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
                    now + timedelta(seconds=request.readiness_timeout_seconds),
                    request.hard_deadline,
                )
            ),
            request.expected_session_seconds,
            request.expected_active_seconds,
            request.serverless_initialization_seconds,
            request.serverless_idle_tail_seconds,
            request.idle_timeout_seconds,
            request.max_authorized_cost,
        )
        inserted = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO ollama_leases (
                        lease_id, owner_id, workload_id, request_fingerprint, model,
                        constraints_json, mode, state, teardown_state, created_at,
                        updated_at, last_used_at, idle_deadline, hard_deadline,
                        readiness_deadline, expected_session_seconds,
                        expected_active_seconds, serverless_initialization_seconds,
                        serverless_idle_tail_seconds, idle_timeout_seconds,
                        max_authorized_cost
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                inserted = True
            except sqlite3.IntegrityError:
                pass
            row = connection.execute(
                "SELECT * FROM ollama_leases WHERE lease_id = ?", (request.lease_id,)
            ).fetchone()
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
        serialized = {key: _db_value(value) for key, value in changes.items()}
        serialized["updated_at"] = serialized.get(
            "updated_at", iso_datetime(datetime.now(UTC))
        )
        assignments = ", ".join(f"{key} = ?" for key in serialized)
        values = [*serialized.values(), lease.lease_id, lease.revision]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""UPDATE ollama_leases
                    SET {assignments}, revision = revision + 1
                    WHERE lease_id = ? AND revision = ?""",
                values,
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
            raise RunPodManagerError("Updated Ollama lease disappeared")
        return _lease_from_row(row)

    def list_for_reconciliation(self) -> tuple[OllamaLease, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ollama_leases WHERE state != ? ORDER BY created_at",
                (OllamaLeaseState.TERMINATED.value,),
            ).fetchall()
        return tuple(_lease_from_row(row) for row in rows)


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
            max_hourly_rate=_optional_stored_float(raw, "max_hourly_rate"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunPodManagerError(
            f"Ollama lease '{lease.lease_id}' has corrupt constraints"
        ) from exc
    readiness_seconds = max(
        1,
        math.ceil((lease.readiness_deadline - lease.created_at).total_seconds()),
    )
    return OllamaLeaseRequest(
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
        readiness_timeout_seconds=readiness_seconds,
        hard_deadline=lease.hard_deadline,
        max_authorized_cost=lease.max_authorized_cost,
        mode=lease.mode,
    )


def _lease_from_row(row: sqlite3.Row) -> OllamaLease:
    return OllamaLease(
        lease_id=row["lease_id"],
        owner_id=row["owner_id"],
        workload_id=row["workload_id"],
        request_fingerprint=row["request_fingerprint"],
        model=row["model"],
        constraints_json=row["constraints_json"],
        mode=OllamaLeaseMode(row["mode"]) if row["mode"] else None,
        resource_type=(
            OllamaResourceType(row["resource_type"]) if row["resource_type"] else None
        ),
        provider_resource_id=row["provider_resource_id"],
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
        estimated_cost=row["estimated_cost"],
        estimated_billable_seconds=row["estimated_billable_seconds"],
        accrued_estimated_cost=row["accrued_estimated_cost"],
        max_authorized_cost=row["max_authorized_cost"],
        cold_start_seconds=row["cold_start_seconds"],
        selected_gpu_id=row["selected_gpu_id"],
        selected_gpu_pool=row["selected_gpu_pool"],
        selected_gpu_name=row["selected_gpu_name"],
        catalog_observed_at=_optional_datetime(row["catalog_observed_at"]),
        last_provider_error=row["last_provider_error"],
        teardown_attempts=row["teardown_attempts"],
        revision=row["revision"],
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
    return value


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


def _optional_stored_float(value: Mapping[str, Any], name: str) -> float | None:
    item = value.get(name)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise TypeError(f"Stored Ollama {name} must be a finite number or null")
    if not math.isfinite(item):
        raise ValueError(f"Stored Ollama {name} must be a finite number or null")
    return float(item)
