"""One-shot entry point for externally scheduled Ollama lease reconciliation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NoReturn, TextIO

from ._one_shot_reconciler import run_one_shot_reconciler
from .manager import RunPodManager
from .ollama_contracts import OllamaLease


async def reconcile_once(
    manager_factory: Callable[[], RunPodManager] = RunPodManager,
) -> tuple[OllamaLease, ...]:
    """Reconcile every nonterminal lease using a freshly initialized manager."""

    manager = manager_factory()
    return await manager.reconcile_ollama_leases()


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    manager_factory: Callable[[], RunPodManager] = RunPodManager,
    stderr: TextIO | None = None,
) -> int:
    """Validate arguments, then execute exactly one fail-fast pass."""

    return run_one_shot_reconciler(
        argv,
        program="kestrel-runpod-reconcile-ollama",
        description="Run one Ollama lease reconciliation pass.",
        operation_factory=lambda: reconcile_once(manager_factory),
        stderr=stderr,
    )


def main() -> NoReturn:
    """Execute one fail-fast pass for a timer, cron job, or job runner."""

    raise SystemExit(run_cli())
