"""Retain admitted effects until their actual mutation and receipt tail finishes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

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

    task = asyncio.ensure_future(effect)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result
