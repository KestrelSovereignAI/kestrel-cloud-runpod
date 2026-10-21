"""Shared, content-free argument boundary for one-shot reconcilers."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import NoReturn, TextIO, TypeVar

_Result = TypeVar("_Result")


class _ContentFreeArgumentParser(argparse.ArgumentParser):
    """Reject unsupported arguments without reflecting their raw values."""

    def error(self, _message: str) -> NoReturn:
        self.print_usage(self._stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")

    def __init__(self, *args, stderr: TextIO, **kwargs) -> None:
        self._stderr = stderr
        super().__init__(*args, **kwargs)

    def _print_message(self, message: str | None, file: TextIO | None = None) -> None:
        super()._print_message(message, self._stderr if file is sys.stderr else file)


def run_one_shot_reconciler(
    argv: Sequence[str] | None,
    *,
    program: str,
    description: str,
    operation_factory: Callable[[], Awaitable[_Result]],
    stderr: TextIO | None = None,
) -> int:
    """Parse before constructing any coroutine, manager, or event loop."""

    parser = _ContentFreeArgumentParser(
        prog=program,
        description=description,
        stderr=stderr or sys.stderr,
    )
    parser.parse_args(argv)
    asyncio.run(operation_factory())
    return 0
