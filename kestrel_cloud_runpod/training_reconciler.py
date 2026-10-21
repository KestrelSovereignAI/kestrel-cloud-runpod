"""One-shot entry point for externally scheduled training Pod reconciliation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NoReturn, Protocol, TextIO

from ._one_shot_reconciler import run_one_shot_reconciler
from .manager import RunPodManager
from .training_contracts import TrainingPodLease


class TrainingReconcileManager(Protocol):
    async def reconcile_training_pods(self) -> tuple[TrainingPodLease, ...]: ...


async def reconcile_once(
    manager_factory: Callable[[], TrainingReconcileManager] = RunPodManager,
) -> tuple[TrainingPodLease, ...]:
    """Construct one manager and perform a fail-fast reconciliation pass."""

    manager = manager_factory()
    return await manager.reconcile_training_pods()


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    manager_factory: Callable[[], TrainingReconcileManager] = RunPodManager,
    stderr: TextIO | None = None,
) -> int:
    """Validate arguments, then execute exactly one fail-fast pass."""

    return run_one_shot_reconciler(
        argv,
        program="kestrel-runpod-reconcile-training",
        description="Run one training Pod reconciliation pass.",
        operation_factory=lambda: reconcile_once(manager_factory),
        stderr=stderr,
    )


def main() -> NoReturn:
    """Run one pass; an external scheduler owns cadence and alerting."""

    raise SystemExit(run_cli())
