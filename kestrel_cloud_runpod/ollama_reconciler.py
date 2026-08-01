"""One-shot entry point for externally scheduled Ollama lease reconciliation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .manager import RunPodManager
from .ollama_contracts import OllamaLease


async def reconcile_once(
    manager_factory: Callable[[], RunPodManager] = RunPodManager,
) -> tuple[OllamaLease, ...]:
    """Reconcile every nonterminal lease using a freshly initialized manager."""

    manager = manager_factory()
    return await manager.reconcile_ollama_leases()


def main() -> None:
    """Execute one fail-fast pass for a timer, cron job, or job runner."""

    asyncio.run(reconcile_once())
