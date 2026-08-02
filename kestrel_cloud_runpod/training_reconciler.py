"""One-shot entry point for externally scheduled training Pod reconciliation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

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


def main() -> None:
    """Run one pass; an external scheduler owns cadence and alerting."""

    asyncio.run(reconcile_once())
