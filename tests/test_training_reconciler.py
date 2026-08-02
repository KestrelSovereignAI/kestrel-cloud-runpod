"""External one-shot training reconciliation entry point."""

import pytest

from kestrel_cloud_runpod.training_reconciler import reconcile_once


@pytest.mark.asyncio
async def test_reconcile_once_constructs_manager_and_runs_one_pass() -> None:
    class Manager:
        async def reconcile_training_pods(self):
            return ("reconciled",)

    assert await reconcile_once(Manager) == ("reconciled",)
