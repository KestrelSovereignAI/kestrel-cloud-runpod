"""Retain admitted effects until their actual mutation and receipt tail finishes."""

from __future__ import annotations

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

    from kestrel_sovereign._async_ownership import await_owned_operation
    from kestrel_sovereign.execution_custody import execution_terminal_error

    async def joined() -> _T:
        # The public contract accepts any Awaitable, including a Future or an
        # already-admitted Task. Core owns the coroutine join, not a second
        # mutation/admission or an asyncio shield algorithm copied here.
        return await effect

    outcome = await await_owned_operation(joined(), retain_originals=True)
    terminal = execution_terminal_error(
        outcome.error, outcome.cancellation,
        retain_originals=True, isolate_control_originals=True,
    )
    if terminal is not None:
        raise terminal
    return cast(_T, outcome.result)
