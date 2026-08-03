"""Installed one-shot command for externally scheduled capacity reconciliation."""

from __future__ import annotations

import argparse
import asyncio
import errno
import importlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TextIO

from .models import RunPodManagerError
from .pod_capacity_contracts import (
    PodCapacityBillingState,
    PodCapacityLease,
    PodCapacityState,
)
from .pod_capacity_repository import SQLitePodCapacityRepository
from .pod_capacity_service import PodCapacityLeaseService

SERVICE_FACTORY_ENV = "RUNPOD_POD_CAPACITY_SERVICE_FACTORY"
TIMEOUT_ENV = "RUNPOD_POD_CAPACITY_RECONCILE_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS = 300.0

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 70
EXIT_CONFIGURATION_ERROR = 78
EXIT_RETRYABLE = 75

_FACTORY_REFERENCE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)

PodCapacityServiceFactory = Callable[[], PodCapacityLeaseService]
ServiceFactoryLoader = Callable[[str], PodCapacityServiceFactory]


class PodCapacityReconcilerConfigurationError(RunPodManagerError):
    """The command cannot safely construct its explicit dependencies."""


class PodCapacityReconcilerBusyError(RunPodManagerError):
    """Another process owns the canonical database reconciliation lock."""


@dataclass(frozen=True)
class PodCapacityReconcileSummary:
    """Content-free counts suitable for a scheduler or metrics collector."""

    reconciled_count: int
    catalog_count: int
    retryable_error_count: int
    unresolved_billing_count: int
    pending_billing_count: int
    states: Mapping[str, int]

    @property
    def requires_retry(self) -> bool:
        return self.retryable_error_count > 0 or self.unresolved_billing_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "retryable" if self.requires_retry else "ok",
            "reconciled_count": self.reconciled_count,
            "catalog_count": self.catalog_count,
            "retryable_error_count": self.retryable_error_count,
            "unresolved_billing_count": self.unresolved_billing_count,
            "pending_billing_count": self.pending_billing_count,
            "states": dict(sorted(self.states.items())),
        }


async def reconcile_once(
    service_factory: PodCapacityServiceFactory,
) -> tuple[PodCapacityLease, ...]:
    """Construct the canonical service and run exactly one reconciliation pass."""

    service = _construct_service(service_factory)
    return await service.reconcile()


def load_service_factory(reference: str) -> PodCapacityServiceFactory:
    """Resolve one explicit host-owned ``module:callable`` service factory."""

    if not isinstance(reference, str) or not _FACTORY_REFERENCE.fullmatch(reference):
        raise PodCapacityReconcilerConfigurationError(
            "The Pod capacity service factory reference is invalid"
        )
    module_name, attribute_path = reference.split(":", 1)
    try:
        value: Any = importlib.import_module(module_name)
    except (ImportError, ValueError) as exc:
        raise PodCapacityReconcilerConfigurationError(
            "The Pod capacity service factory module could not be imported"
        ) from exc
    try:
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise PodCapacityReconcilerConfigurationError(
            "The Pod capacity service factory callable was not found"
        ) from exc
    if not callable(value):
        raise PodCapacityReconcilerConfigurationError(
            "The Pod capacity service factory target is not callable"
        )
    return value


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    factory_loader: ServiceFactoryLoader = load_service_factory,
) -> int:
    """Run one locked pass and return a stable scheduler-facing exit status."""

    output = stdout or sys.stdout
    environment = environ if environ is not None else os.environ
    try:
        arguments = _parse_arguments(argv, environment)
        factory = factory_loader(arguments.service_factory)
        try:
            service = _construct_service(factory)
        except (RunPodManagerError, OSError, TypeError, ValueError) as exc:
            raise PodCapacityReconcilerConfigurationError(
                "The Pod capacity service factory could not construct its dependencies"
            ) from exc
        with _exclusive_reconcile_lock(service.repository.path):
            try:
                leases = asyncio.run(
                    asyncio.wait_for(
                        service.reconcile(), timeout=arguments.timeout_seconds
                    )
                )
            except TimeoutError:
                _write_json(output, {"status": "timeout"})
                return EXIT_RETRYABLE
            except RunPodManagerError as exc:
                _write_json(
                    output,
                    {"status": "error", "error_type": type(exc).__name__},
                )
                return EXIT_RUNTIME_ERROR
            except (OSError, sqlite3.Error) as exc:
                _write_json(
                    output,
                    {"status": "error", "error_type": type(exc).__name__},
                )
                return EXIT_RUNTIME_ERROR
    except PodCapacityReconcilerBusyError:
        _write_json(output, {"status": "busy"})
        return EXIT_RETRYABLE
    except PodCapacityReconcilerConfigurationError:
        _write_json(output, {"status": "configuration_error"})
        return EXIT_CONFIGURATION_ERROR

    summary = _summarize(leases)
    _write_json(output, summary.to_dict())
    return EXIT_RETRYABLE if summary.requires_retry else EXIT_OK


def main() -> NoReturn:
    """Run one fail-fast, non-provisioning pass for cron or a systemd timer."""

    raise SystemExit(run_cli())


def _construct_service(
    service_factory: PodCapacityServiceFactory,
) -> PodCapacityLeaseService:
    service = service_factory()
    if not isinstance(service, PodCapacityLeaseService):
        raise PodCapacityReconcilerConfigurationError(
            "The configured factory did not return PodCapacityLeaseService"
        )
    if not isinstance(service.repository, SQLitePodCapacityRepository):
        raise PodCapacityReconcilerConfigurationError(
            "The capacity reconciler requires SQLitePodCapacityRepository"
        )
    service.validate_catalog_reconciler_dependencies()
    return service


def _summarize(
    leases: Sequence[PodCapacityLease],
) -> PodCapacityReconcileSummary:
    unique = {lease.cleanup_token: lease for lease in leases}
    values = tuple(unique.values())
    catalog = tuple(lease for lease in values if lease.is_catalog_attempt)
    retryable_errors = sum(
        1
        for lease in values
        if lease.state is not PodCapacityState.RELEASED
        and lease.last_provider_error is not None
    )
    return PodCapacityReconcileSummary(
        reconciled_count=len(values),
        catalog_count=len(catalog),
        retryable_error_count=retryable_errors,
        unresolved_billing_count=sum(
            lease.billing_state is PodCapacityBillingState.UNRESOLVED
            for lease in catalog
        ),
        pending_billing_count=sum(
            lease.billing_state is PodCapacityBillingState.PENDING for lease in catalog
        ),
        states=Counter(lease.state.value for lease in values),
    )


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise PodCapacityReconcilerConfigurationError(
            "The Pod capacity reconciler arguments are invalid"
        )


def _parse_arguments(
    argv: Sequence[str] | None, environment: Mapping[str, str]
) -> argparse.Namespace:
    parser = _ArgumentParser(
        prog="kestrel-runpod-reconcile-capacity",
        description="Run one bounded Pod-capacity reconciliation pass.",
    )
    parser.add_argument(
        "--service-factory",
        default=environment.get(SERVICE_FACTORY_ENV),
        help=(
            "Host-owned module:callable returning a fully configured "
            "PodCapacityLeaseService"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        default=environment.get(TIMEOUT_ENV, str(DEFAULT_TIMEOUT_SECONDS)),
        type=_positive_timeout,
        help="Upper bound for one reconciliation pass",
    )
    parsed = parser.parse_args(argv)
    if not parsed.service_factory:
        raise PodCapacityReconcilerConfigurationError(
            f"Configure --service-factory or {SERVICE_FACTORY_ENV}"
        )
    return parsed


def _positive_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be numeric") from exc
    if not 0 < parsed <= 3600:
        raise argparse.ArgumentTypeError("timeout must be between 0 and 3600 seconds")
    return parsed


@contextmanager
def _exclusive_reconcile_lock(database_path: Path) -> Iterator[None]:
    """Take a process-scoped advisory lock derived from the canonical DB path."""

    lock_path = database_path.with_name(database_path.name + ".reconcile.lock")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
    except OSError as exc:
        raise PodCapacityReconcilerConfigurationError(
            "The Pod capacity reconciliation lock could not be opened"
        ) from exc
    locked = False
    try:
        locked = _lock_descriptor(descriptor)
        if not locked:
            raise PodCapacityReconcilerBusyError(
                "Another Pod capacity reconciliation pass is active"
            )
        yield
    finally:
        if locked:
            _unlock_descriptor(descriptor)
        os.close(descriptor)


def _lock_descriptor(descriptor: int) -> bool:
    if os.name == "posix":
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise PodCapacityReconcilerConfigurationError(
                "The Pod capacity reconciliation lock failed"
            ) from exc
        return True
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise PodCapacityReconcilerConfigurationError(
                "The Pod capacity reconciliation lock failed"
            ) from exc
        return True
    raise PodCapacityReconcilerConfigurationError(
        "The Pod capacity reconciler does not support file locking on this platform"
    )


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


def _write_json(output: TextIO, value: Mapping[str, Any]) -> None:
    output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
