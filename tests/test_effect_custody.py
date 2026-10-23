"""Owned effects keep cancellation from detaching mutation/receipt tails."""

import asyncio
import gc
import inspect
import threading
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from kestrel_cloud_runpod import await_owned_effect


def test_owned_effect_metadata_requires_fixed_core_and_test_loop_custody():
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    dependencies = [Requirement(value) for value in project["dependencies"]]
    (core,) = [value for value in dependencies if value.name == "kestrel-sovereign"]
    assert "0.53.34" not in core.specifier
    assert "0.53.36" not in core.specifier
    assert "0.53.37" not in core.specifier
    assert "0.53.40" not in core.specifier
    assert "0.53.41" in core.specifier
    test_dependencies = [
        Requirement(value) for value in project["optional-dependencies"]["test"]
    ]
    (test_loop,) = [
        value for value in test_dependencies if value.name == "pytest-asyncio"
    ]
    assert "1.3.0" not in test_loop.specifier
    assert "1.4.0" in test_loop.specifier


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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["coroutine", "closeable_awaitable"])
async def test_real_task_admission_refusal_closes_only_the_original_unstarted_effect(kind):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    original = RuntimeError("original loop refused owned task admission")
    body, closed = [], []
    release = asyncio.Event()

    async def coroutine():
        body.append("effect body entered")
        await release.wait()
        return "existing accepted effect"

    class CloseableAwaitable:
        def __await__(self):
            return coroutine().__await__()

        def close(self):
            closed.append("original unstarted close")

    effect = coroutine() if kind == "coroutine" else CloseableAwaitable()

    def refuse(_loop, _coroutine, **_kwargs):
        raise original

    try:
        loop.set_task_factory(refuse)
        with pytest.raises(RuntimeError) as caught:
            await await_owned_effect(effect)
        assert caught.value is original
    finally:
        loop.set_task_factory(previous_factory)
        release.set()
        if kind == "coroutine":
            assert inspect.getcoroutinestate(effect) == inspect.CORO_CLOSED
    assert body == []
    assert closed == (["original unstarted close"] if kind == "closeable_awaitable" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["existing_task", "existing_future", "future_like", "readonly_future"])
@pytest.mark.parametrize("failure", [False, True])
async def test_existing_admitted_awaitable_is_joined_even_when_new_task_admission_is_refused(kind, failure):
    from kestrel_sovereign._async_ownership import await_owned_task

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    release, tail = asyncio.Event(), []
    original = ValueError("original accepted effect failure")
    refused = []

    class FutureLike:
        """The standard asyncio Future-like protocol, backed by a REAL Future."""

        __slots__ = ("_future",)
        _asyncio_future_blocking = False

        def __init__(self, future):
            self._future = future

        def __await__(self):
            return self._future.__await__()

        def __getattr__(self, name):
            return getattr(self._future, name)

    class ReadOnlyFuture(asyncio.Future):
        def __setattr__(self, name, value):
            if name.startswith("_kestrel_"):
                raise AttributeError("foreign Future does not admit private Core metadata")
            return super().__setattr__(name, value)

    async def accepted():
        await release.wait()
        tail.append("original physical tail completed")
        if failure:
            raise original
        return "original accepted receipt"

    if kind == "existing_task":
        effect = asyncio.create_task(accepted())
    elif kind == "readonly_future":
        effect = ReadOnlyFuture(loop=loop)
    else:
        effect = loop.create_future()
    if kind == "future_like":
        effect = FutureLike(effect)
        assert asyncio.isfuture(effect) and not isinstance(effect, asyncio.Future)
    owner = asyncio.create_task(await_owned_effect(effect))

    def refuse(_loop, _coroutine, **_kwargs):
        refused.append("new task allocation attempted")
        raise RuntimeError("new task admission refused")

    try:
        loop.set_task_factory(refuse)
        for _ in range(3):
            await asyncio.sleep(0)
            assert not owner.done(), "an accepted original cannot lose its join to wrapper refusal"
            owner.cancel("original existing-effect caller cancellation")
    finally:
        loop.set_task_factory(previous_factory)
        release.set()
        if kind in {"existing_future", "future_like", "readonly_future"} and not effect.done():
            if failure:
                effect.set_exception(original)
            else:
                effect.set_result("original accepted receipt")
        try:
            effect_outcome = await await_owned_task(effect)
        finally:
            outcome = await await_owned_task(owner)
    assert refused == [], "existing Task/Future needs no surrogate Task admission"
    if kind == "existing_task":
        assert tail == ["original physical tail completed"]
    assert not effect.cancelled()
    if failure:
        assert effect_outcome.error is original and outcome.error is original
    else:
        assert effect_outcome.result == "original accepted receipt"
        assert isinstance(outcome.error, asyncio.CancelledError)
    assert any(isinstance(error, asyncio.CancelledError) for error in _original_errors(outcome.error))
