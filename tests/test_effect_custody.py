"""Owned effects keep cancellation from detaching mutation/receipt tails."""

import asyncio
import threading

import pytest

from kestrel_cloud_runpod import await_owned_effect


@pytest.mark.asyncio
async def test_success_and_failure_preserve_original_outcomes():
    async def success():
        return "receipt"

    async def failure():
        raise ValueError("receipt persistence failed")

    assert await await_owned_effect(success()) == "receipt"
    with pytest.raises(ValueError, match="receipt persistence failed"):
        await await_owned_effect(failure())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_repeated_cancellation_joins_real_thread_and_receipt_tail(failure):
    entered = threading.Event()
    finish = threading.Event()
    receipt_entered = asyncio.Event()
    receipt_finish = asyncio.Event()
    receipts = []

    def mutation():
        entered.set()
        assert finish.wait(5), "test did not release mutation thread"
        return "provider-id"

    async def effect():
        provider_id = await asyncio.to_thread(mutation)
        receipt_entered.set()
        await receipt_finish.wait()
        if failure:
            raise ValueError("durable receipt failed")
        receipts.append(provider_id)

    task = asyncio.create_task(await_owned_effect(effect()))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(4):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finish.set()
        await asyncio.wait_for(receipt_entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        finish.set()
        receipt_finish.set()
    if failure:
        with pytest.raises(ValueError, match="durable receipt failed"):
            await task
        assert receipts == []
    else:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert receipts == ["provider-id"]


@pytest.mark.asyncio
async def test_effect_self_cancellation_does_not_loop_forever():
    async def effect():
        raise asyncio.CancelledError("effect cancelled itself")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(await_owned_effect(effect()), 5)
