"""Durable custody of an activation on an existing Ollama endpoint.

An ARMING or PARKING row means its PATCH *may* have reached Runpod. Reopening
the journal never reacquires that effect right. Only exact provider evidence
can advance a row; uncertainty keeps the endpoint and its cache reserved.
This module neither calls Runpod nor constructs a create/deletion receipt.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Self

from .ollama_contracts import OllamaLeaseConflictError, OllamaProvisioningIntent, iso_datetime
from .sqlite_store import lease_database_connection

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}")
_DIGEST = re.compile(r"[a-f0-9]{64}")
_TABLE = "ollama_registered_activations"
_CREATE = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    lease_id TEXT PRIMARY KEY,
    endpoint_id TEXT NOT NULL,
    cache_id TEXT NOT NULL,
    intent TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('arming', 'active', 'parking', 'parked', 'fenced', 'fenced_zero')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idle_deadline TEXT NOT NULL,
    hard_deadline TEXT NOT NULL
)
"""


class RegisteredActivationState(str, Enum):
    ARMING = "arming"
    ACTIVE = "active"
    PARKING = "parking"
    PARKED = "parked"
    FENCED = "fenced"
    FENCED_ZERO = "fenced_zero"


@dataclass(frozen=True, slots=True)
class RegisteredActivationIntent:
    lease_id: str
    owner_id: str
    endpoint_id: str
    cache_id: str
    admission_sha256: str
    registration_sha256: str
    prior_environment_sha256: str
    active_environment_sha256: str
    parked_environment_sha256: str
    first_patch_before: datetime
    ready_before: datetime
    hard_deadline: datetime
    idle_seconds: int

    def __post_init__(self) -> None:
        for name in ("lease_id", "owner_id", "endpoint_id", "cache_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ValueError("Invalid registered activation identity")
        for name in (
            "admission_sha256", "registration_sha256", "prior_environment_sha256",
            "active_environment_sha256", "parked_environment_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _DIGEST.fullmatch(value):
                raise ValueError("Invalid registered activation digest")
        if (
            len({self.prior_environment_sha256, self.active_environment_sha256,
                 self.parked_environment_sha256}) != 3
            or type(self.idle_seconds) is not int
            or self.idle_seconds < 1
        ):
            raise ValueError("Invalid registered activation environment or idle bound")
        for name in ("first_patch_before", "ready_before", "hard_deadline"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError("Registered activation deadlines must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if not self.first_patch_before <= self.ready_before < self.hard_deadline:
            raise ValueError("Registered activation cannot outlive its hard deadline")

    def _encode(self) -> str:
        wire = {item.name: getattr(self, item.name) for item in fields(self)}
        for key in ("first_patch_before", "ready_before", "hard_deadline"):
            wire[key] = iso_datetime(wire[key])
        return json.dumps(wire, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _decode(cls, value: str) -> Self:
        wire = json.loads(value)
        for key in ("first_patch_before", "ready_before", "hard_deadline"):
            wire[key] = datetime.fromisoformat(wire[key])
        intent = cls(**wire)
        if intent._encode() != value:
            raise ValueError("Noncanonical registered activation intent")
        return intent


@dataclass(frozen=True, slots=True)
class RegisteredActivation:
    intent: RegisteredActivationIntent
    state: RegisteredActivationState
    created_at: datetime
    updated_at: datetime
    idle_deadline: datetime

    @property
    def expires_at(self) -> datetime:
        return min(self.intent.hard_deadline, self.idle_deadline)


def _row_activation(row: sqlite3.Row) -> RegisteredActivation:
    intent = RegisteredActivationIntent._decode(row["intent"])
    if (
        row["lease_id"] != intent.lease_id or row["endpoint_id"] != intent.endpoint_id
        or row["cache_id"] != intent.cache_id
        or row["hard_deadline"] != iso_datetime(intent.hard_deadline)
    ):
        raise OllamaLeaseConflictError("Registered activation journal identity drift")
    return RegisteredActivation(
        intent, RegisteredActivationState(row["state"]),
        datetime.fromisoformat(row["created_at"]), datetime.fromisoformat(row["updated_at"]),
        datetime.fromisoformat(row["idle_deadline"]),
    )


def require_unreserved_registered_cache(
    connection: sqlite3.Connection, volume_ids: frozenset[str],
) -> None:
    """Called under the same admitted-create write transaction."""
    if not volume_ids or connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_TABLE,),
    ).fetchone() is None:
        return
    placeholders = ",".join("?" for _ in volume_ids)
    if connection.execute(
        f"SELECT 1 FROM {_TABLE} WHERE state != 'parked'"
        f" AND cache_id IN ({placeholders}) LIMIT 1",
        tuple(sorted(volume_ids)),
    ).fetchone() is not None:
        raise OllamaLeaseConflictError("Model cache is reserved by a registered endpoint")


class RegisteredOllamaJournal:
    """One exclusive activation per endpoint/cache in the existing lease DB."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None:
        if not path.is_absolute():
            raise ValueError("Registered lease journal path must be absolute")
        self._path = path
        self._clock = clock
        with lease_database_connection(path) as db:
            db.execute("BEGIN IMMEDIATE")
            # This is an additive family, not an alternative standalone DB.
            if db.execute(
                "SELECT 1 FROM ollama_lease_journal_identity WHERE singleton = 1"
            ).fetchone() is None:
                raise OllamaLeaseConflictError("Original Ollama lease journal identity is missing")
            db.execute(_CREATE)
            for column in ("endpoint_id", "cache_id"):
                db.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS registered_active_{column}"
                    f" ON {_TABLE} ({column}) WHERE state != 'parked'"
                )

    def get(self, lease_id: str) -> RegisteredActivation | None:
        with lease_database_connection(self._path) as db:
            row = self._get(db, lease_id)
            return _row_activation(row) if row is not None else None

    def reserve(self, intent: RegisteredActivationIntent) -> tuple[RegisteredActivation, bool]:
        """Only inserted=True grants the current caller a single first PATCH."""
        with lease_database_connection(self._path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._get(db, intent.lease_id)
            if row is not None:
                existing = _row_activation(row)
                if existing.intent != intent:
                    raise OllamaLeaseConflictError("Registered lease replay changed its authority")
                return existing, False
            now = self._now()
            if now >= intent.first_patch_before:
                raise OllamaLeaseConflictError("Registered activation window has elapsed")
            if db.execute(
                f"SELECT 1 FROM {_TABLE} WHERE state != 'parked'"
                " AND (endpoint_id = ? OR cache_id = ?) LIMIT 1",
                (intent.endpoint_id, intent.cache_id),
            ).fetchone() is not None:
                raise OllamaLeaseConflictError("Registered endpoint or cache is already reserved")
            last = self._last_parked(db, intent.endpoint_id)
            if last is not None and (
                last.intent.parked_environment_sha256 != intent.prior_environment_sha256
                or last.intent.registration_sha256 != intent.registration_sha256
            ):
                raise OllamaLeaseConflictError("Registered endpoint predecessor changed before admission")
            for old in db.execute(
                "SELECT provisioning_intent_json FROM ollama_leases "
                "WHERE provisioning_intent_json IS NOT NULL "
                "AND (state != 'terminated' OR creation_uncertain != 0)"
            ):
                original = OllamaProvisioningIntent.from_dict(json.loads(old[0]))
                if intent.cache_id in original.network_volume_ids:
                    raise OllamaLeaseConflictError("Model cache is reserved by an existing lease")
            # Cold startup has a different bound from inactivity after the
            # first authenticated ready readback/route delivery.
            idle = intent.ready_before
            db.execute(
                f"INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, 'arming', ?, ?, ?, ?)",
                (
                    intent.lease_id, intent.endpoint_id, intent.cache_id, intent._encode(),
                    iso_datetime(now), iso_datetime(now), iso_datetime(idle),
                    iso_datetime(intent.hard_deadline),
                ),
            )
            return RegisteredActivation(intent, RegisteredActivationState.ARMING, now, now, idle), True

    def last_parked(self, endpoint_id: str) -> RegisteredActivation | None:
        with lease_database_connection(self._path) as db:
            return self._last_parked(db, endpoint_id)

    @staticmethod
    def _last_parked(db: sqlite3.Connection, endpoint_id: str) -> RegisteredActivation | None:
        row = db.execute(
            f"SELECT * FROM {_TABLE} WHERE endpoint_id = ? AND state = 'parked'"
            " ORDER BY rowid DESC LIMIT 1",
            (endpoint_id,),
        ).fetchone()
        return _row_activation(row) if row is not None else None

    def due(self) -> tuple[RegisteredActivation, ...]:
        """Parking remains due until a readback proves the physically zero state."""
        now = iso_datetime(self._now())
        with lease_database_connection(self._path) as db:
            rows = db.execute(
                f"SELECT * FROM {_TABLE} WHERE state IN ('parking', 'fenced') OR"
                " (state IN ('arming', 'active') AND (hard_deadline <= ? OR idle_deadline <= ?))"
                " ORDER BY created_at",
                (now, now),
            ).fetchall()
            return tuple(_row_activation(row) for row in rows)

    def record_active(self, original: RegisteredActivationIntent) -> RegisteredActivation:
        """Record the controller's exact fingerprint readback, not a create."""
        return self._advance(original, RegisteredActivationState.ARMING, RegisteredActivationState.ACTIVE)

    def claim_park(self, original: RegisteredActivationIntent) -> tuple[RegisteredActivation, bool]:
        """Claim the single park PATCH after original active authority is proved."""
        with lease_database_connection(self._path) as db:
            db.execute("BEGIN IMMEDIATE")
            record = self._required(db, original)
            if record.state in (
                RegisteredActivationState.PARKING, RegisteredActivationState.PARKED,
                RegisteredActivationState.FENCED, RegisteredActivationState.FENCED_ZERO,
            ):
                return record, False
            if record.state is not RegisteredActivationState.ACTIVE:
                raise OllamaLeaseConflictError("Registered activation is not confirmed for parking")
            return self._write(db, record, RegisteredActivationState.PARKING), True

    def record_parked(self, original: RegisteredActivationIntent) -> RegisteredActivation:
        """Caller must verify the exact park marker, zero cap and zero workers."""
        return self._advance(original, RegisteredActivationState.PARKING, RegisteredActivationState.PARKED)

    def fence_uncertain_parking(self, original: RegisteredActivationIntent) -> RegisteredActivation:
        """Permanently forbid re-admission before retrying a missing park effect."""
        with lease_database_connection(self._path) as db:
            db.execute("BEGIN IMMEDIATE")
            record = self._required(db, original)
            if record.state in (RegisteredActivationState.FENCED, RegisteredActivationState.FENCED_ZERO):
                return record
            if record.state is not RegisteredActivationState.PARKING or self._now() < original.hard_deadline:
                raise OllamaLeaseConflictError("Uncertain parking cannot be fenced yet")
            return self._write(db, record, RegisteredActivationState.FENCED)

    def record_fenced_zero(self, original: RegisteredActivationIntent) -> RegisteredActivation:
        """Record zero without granting a new lease after a possible late PATCH."""
        return self._advance(original, RegisteredActivationState.FENCED, RegisteredActivationState.FENCED_ZERO)

    def touch(self, original: RegisteredActivationIntent) -> RegisteredActivation:
        with lease_database_connection(self._path) as db:
            db.execute("BEGIN IMMEDIATE")
            record = self._required(db, original)
            now = self._now()
            if record.state is not RegisteredActivationState.ACTIVE or now >= record.expires_at:
                raise OllamaLeaseConflictError("Registered lease cannot be renewed")
            idle = min(original.hard_deadline, now + timedelta(seconds=original.idle_seconds))
            return self._write(db, record, record.state, idle=idle)

    def _advance(
        self, original: RegisteredActivationIntent,
        before: RegisteredActivationState, after: RegisteredActivationState,
    ) -> RegisteredActivation:
        with lease_database_connection(self._path) as db:
            db.execute("BEGIN IMMEDIATE")
            record = self._required(db, original)
            if record.state is after:
                return record
            if record.state is not before:
                raise OllamaLeaseConflictError("Registered activation transition changed")
            return self._write(db, record, after)

    def _write(
        self, db: sqlite3.Connection, record: RegisteredActivation,
        state: RegisteredActivationState, *, idle: datetime | None = None,
    ) -> RegisteredActivation:
        now = max(record.updated_at, self._now())
        deadline = record.idle_deadline if idle is None else idle
        db.execute(
            f"UPDATE {_TABLE} SET state = ?, updated_at = ?, idle_deadline = ? WHERE lease_id = ?",
            (state.value, iso_datetime(now), iso_datetime(deadline), record.intent.lease_id),
        )
        return RegisteredActivation(record.intent, state, record.created_at, now, deadline)

    @staticmethod
    def _get(db: sqlite3.Connection, lease_id: str) -> sqlite3.Row | None:
        return db.execute(f"SELECT * FROM {_TABLE} WHERE lease_id = ?", (lease_id,)).fetchone()

    def _required(self, db: sqlite3.Connection, intent: RegisteredActivationIntent) -> RegisteredActivation:
        row = self._get(db, intent.lease_id)
        if row is None or (record := _row_activation(row)).intent != intent:
            raise OllamaLeaseConflictError("Registered activation original authority is missing")
        return record

    def _now(self) -> datetime:
        value = self._clock()
        if value.utcoffset() is None:
            raise ValueError("Registered reconciliation clock must be timezone-aware")
        return value.astimezone(UTC)
