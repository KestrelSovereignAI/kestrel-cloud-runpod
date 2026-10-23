"""Shared argument boundary for installed one-shot reconcilers."""

from __future__ import annotations

from io import StringIO

import pytest

from kestrel_cloud_runpod._one_shot_reconciler import run_one_shot_reconciler


def test_help_exits_before_coroutine_factory_or_event_loop(capsys) -> None:
    called = False

    def operation_factory():
        nonlocal called
        called = True
        raise AssertionError("help must not create the operation")

    with pytest.raises(SystemExit) as raised:
        run_one_shot_reconciler(
            ["--help"],
            program="content-free-test",
            description="one pass",
            operation_factory=operation_factory,
        )

    assert raised.value.code == 0
    assert called is False
    assert "usage: content-free-test" in capsys.readouterr().out


def test_invalid_argument_is_generic_and_never_constructs_operation() -> None:
    secret_canary = "--credential-secret-value"
    error = StringIO()
    called = False

    def operation_factory():
        nonlocal called
        called = True
        raise AssertionError("invalid arguments must not create the operation")

    with pytest.raises(SystemExit) as raised:
        run_one_shot_reconciler(
            [secret_canary],
            program="content-free-test",
            description="one pass",
            operation_factory=operation_factory,
            stderr=error,
        )

    assert raised.value.code == 2
    assert called is False
    assert "invalid arguments" in error.getvalue()
    assert secret_canary not in error.getvalue()


def test_empty_arguments_run_exactly_one_operation() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1

    assert (
        run_one_shot_reconciler(
            [],
            program="content-free-test",
            description="one pass",
            operation_factory=operation,
        )
        == 0
    )
    assert calls == 1
