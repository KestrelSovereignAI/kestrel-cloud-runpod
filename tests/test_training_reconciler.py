"""External one-shot training reconciliation entry point."""

import pytest

from kestrel_cloud_runpod.training_reconciler import reconcile_once, run_cli


@pytest.mark.asyncio
async def test_reconcile_once_constructs_manager_and_runs_one_pass() -> None:
    class Manager:
        async def reconcile_training_pods(self):
            return ("reconciled",)

    assert await reconcile_once(Manager) == ("reconciled",)


def test_cli_empty_arguments_preserve_one_shot_manager_contract() -> None:
    calls = 0

    class Manager:
        async def reconcile_training_pods(self):
            nonlocal calls
            calls += 1
            return ()

    assert run_cli([], manager_factory=Manager) == 0
    assert calls == 1
