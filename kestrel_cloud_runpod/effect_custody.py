"""Retain admitted effects until their actual mutation and receipt tail finishes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar, cast

_T = TypeVar("_T")


async def await_owned_effect(effect: Awaitable[_T]) -> _T:
    """Join an admitted effect despite repeated caller cancellation.

    This is not an admission check or a retry. The host must persist intent and
    validate authorization before calling it. Include durable receipt/error
    persistence inside ``effect`` and retain the host's admission lease outside
    this await. Synchronous provider calls can use ``asyncio.to_thread`` here;
    cancelling a coroutine cannot stop their threads.

    Caller cancellation is propagated only after the effect finishes. An effect
    failure takes precedence so that cleanup/ambiguity errors remain visible.
    Process death still requires reconciliation from the persisted intent.
    """

    from kestrel_sovereign._async_ownership import (
        await_owned_operation,
        await_owned_task,
    )
    from kestrel_sovereign.execution_custody import execution_terminal_error

    if asyncio.isfuture(effect):
        # Already-admitted Tasks/Futures need no new Task allocation. Refusing
        # a surrogate join task cannot abandon the original physical effect.
        outcome = await await_owned_task(effect)
    else:
        # Core retains the EXACT unstarted close boundary on admission refusal.
        # A redundant coroutine wrapper would hide that original from cleanup.
        outcome = await await_owned_operation(effect, retain_originals=True)
    terminal = execution_terminal_error(
        outcome.error, outcome.cancellation,
        retain_originals=True, isolate_control_originals=True,
    )
    if terminal is not None:
        raise terminal
    return cast(_T, outcome.result)
