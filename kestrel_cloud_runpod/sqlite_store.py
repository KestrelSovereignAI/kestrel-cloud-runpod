"""Canonical transaction and connection lifetime for durable lease stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path


@contextmanager
def lease_database_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Keep existing WAL/transaction semantics and always close the connection.

    A sqlite3 connection's own context manager commits or rolls back, but does
    not close it. The outer owner also covers failed PRAGMA initialization and
    failed commit/rollback, not only a successfully yielded repository operation.
    """
    with (
        closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as connection,
        connection,
    ):
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        yield connection
