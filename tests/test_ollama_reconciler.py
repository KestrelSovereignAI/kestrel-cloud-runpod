"""External one-shot reconciler entry point tests."""

import pytest

from kestrel_cloud_runpod.ollama_reconciler import reconcile_once


class _Manager:
    def __init__(self):
        self.calls = 0

    async def reconcile_ollama_leases(self):
        self.calls += 1
        return ("lease",)


@pytest.mark.asyncio
async def test_reconcile_once_constructs_manager_and_runs_single_pass():
    manager = _Manager()

    result = await reconcile_once(lambda: manager)

    assert result == ("lease",)
    assert manager.calls == 1
