"""Owned effects keep cancellation from detaching mutation/receipt tails."""

import asyncio
import gc
import threading

import pytest

from kestrel_cloud_runpod import await_owned_effect


def _original_errors(error):
    pending, seen = [error], set()
    while pending:
        original = pending.pop()
        if original is None or id(original) in seen:
            continue
        seen.add(id(original))
        yield original
        pending.append(original.__cause__)
        if isinstance(original, BaseExceptionGroup):
            pending.extend(original.exceptions)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ordinary", "denied", "unknown", "grouped_control"])
async def test_late_effect_failure_retains_original_cancellation_and_native_evidence(kind):
    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        ExecutionCommitOutcomeError,
        execution_commit_outcome,
        is_execution_control_error,
    )

    sql_error = ValueError("original SQL boundary")
    native = ExecutionCommitOutcomeError("unknown")
    failures = {
        "ordinary": sql_error,
        "denied": ExecutionAuthorityError("original native session lost"),
        "unknown": native,
        "grouped_control": BaseExceptionGroup("original SQL and native", [sql_error, native]),
    }
    original = failures[kind]
    entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def effect():
        entered.set()
        await release.wait()
        completed.set()
        raise original

    owner = asyncio.create_task(await_owned_effect(effect()))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        owner.cancel("first original observer cancellation")
        for _ in range(3):
            await asyncio.sleep(0)
            owner.cancel("later original observer cancellation")
            assert not owner.done()
    finally:
        release.set()
        outcome = await await_owned_task(owner)
    assert completed.is_set()
    errors = tuple(_original_errors(outcome.error))
    assert any(error is original for error in errors)
    assert any(
        isinstance(error, asyncio.CancelledError)
        and error.args == ("first original observer cancellation",)
        for error in errors
    ), "late failure must not discard original observer cancellation"
    if kind != "ordinary":
        assert is_execution_control_error(outcome.error)
        assert original.__cause__ is None, "finite observation must not mutate shared native source"
    if kind in {"unknown", "grouped_control"}:
        assert execution_commit_outcome(outcome.error) == "unknown"
    if kind == "ordinary":
        assert outcome.error is original  # Preserve documented effect-failure precedence.


@pytest.mark.asyncio
async def test_joined_failure_emits_no_orphaned_shield_future_diagnostics():
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    diagnostics = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def effect():
        entered.set()
        await release.wait()
        raise ValueError("late joined failure")

    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
    owner = asyncio.create_task(await_owned_effect(effect()))
    try:
        await entered.wait()
        for _ in range(4):
            owner.cancel()
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(ValueError, match="late joined failure"):
            await owner
        del owner
        gc.collect()
        for _ in range(3):
            await asyncio.sleep(0)
        assert diagnostics == []
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["task", "future", "custom_awaitable"])
async def test_existing_awaitable_contract_is_not_narrowed_to_coroutines(kind):
    async def success():
        return "exact existing effect receipt"

    class ExistingAwaitable:
        def __await__(self):
            return success().__await__()

    if kind == "task":
        effect = asyncio.create_task(success())
    elif kind == "future":
        effect = asyncio.get_running_loop().create_future()
        effect.set_result("exact existing effect receipt")
    else:
        effect = ExistingAwaitable()
    assert await await_owned_effect(effect) == "exact existing effect receipt"
