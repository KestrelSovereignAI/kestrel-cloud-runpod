"""Installed, locked, content-free Pod-capacity reconciler command tests."""

import asyncio
import json
from importlib.metadata import distribution
from io import StringIO

import pytest
from pod_capacity_test_support import (
    TOKEN,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    request,
    service,
)

from kestrel_cloud_runpod.models import RunPodAmbiguousResultError, RunPodManagerError
from kestrel_cloud_runpod.pod_capacity_contracts import PodCapacityLifecycleError
from kestrel_cloud_runpod.pod_capacity_reconciler import (
    EXIT_CONFIGURATION_ERROR,
    EXIT_OK,
    EXIT_RETRYABLE,
    EXIT_RUNTIME_ERROR,
    _exclusive_reconcile_lock,
    reconcile_once,
    run_cli,
)

FACTORY_REFERENCE = "catalog_host.runpod:build_capacity_service"


def test_distribution_installs_capacity_reconciler_entry_point() -> None:
    scripts = {
        entry.name: entry.value
        for entry in distribution("kestrel-cloud-runpod").entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["kestrel-runpod-reconcile-capacity"] == (
        "kestrel_cloud_runpod.pod_capacity_reconciler:main"
    )


def _runtime(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    return (
        service(tmp_path, clock, provider, store, transport),
        provider,
    )


@pytest.mark.asyncio
async def test_reconcile_once_constructs_service_and_runs_one_pass(tmp_path) -> None:
    runtime, provider = _runtime(tmp_path)

    assert await reconcile_once(lambda: runtime) == ()
    assert provider.create_calls == []


def test_cli_runs_once_without_provisioning_and_emits_content_free_json(
    tmp_path,
) -> None:
    runtime, provider = _runtime(tmp_path)
    output = StringIO()

    exit_status = run_cli(
        ["--service-factory", FACTORY_REFERENCE],
        stdout=output,
        factory_loader=lambda _: lambda: runtime,
    )

    assert exit_status == EXIT_OK
    assert json.loads(output.getvalue()) == {
        "catalog_count": 0,
        "pending_billing_count": 0,
        "reconciled_count": 0,
        "retryable_error_count": 0,
        "states": {},
        "status": "ok",
        "unresolved_billing_count": 0,
    }
    assert provider.create_calls == []


def test_cli_missing_factory_fails_before_loading_or_provisioning() -> None:
    output = StringIO()
    loader_called = False

    def loader(_):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("missing configuration must fail first")

    exit_status = run_cli([], environ={}, stdout=output, factory_loader=loader)

    assert exit_status == EXIT_CONFIGURATION_ERROR
    assert json.loads(output.getvalue()) == {"status": "configuration_error"}
    assert loader_called is False


def test_cli_auth_configuration_error_never_exposes_message_or_secret() -> None:
    output = StringIO()
    leaked_secret = "runpod-secret-value"

    def failed_factory():
        raise RunPodManagerError(f"RUNPOD_API_KEY missing: {leaked_secret}")

    exit_status = run_cli(
        ["--service-factory", FACTORY_REFERENCE],
        stdout=output,
        factory_loader=lambda _: failed_factory,
    )

    assert exit_status == EXIT_CONFIGURATION_ERROR
    assert json.loads(output.getvalue()) == {"status": "configuration_error"}
    assert leaked_secret not in output.getvalue()


def test_cli_missing_catalog_dependency_fails_before_reconcile(tmp_path) -> None:
    runtime, provider = _runtime(tmp_path)
    runtime._capability_store = None
    output = StringIO()

    exit_status = run_cli(
        ["--service-factory", FACTORY_REFERENCE],
        stdout=output,
        factory_loader=lambda _: lambda: runtime,
    )

    assert exit_status == EXIT_CONFIGURATION_ERROR
    assert json.loads(output.getvalue()) == {"status": "configuration_error"}
    assert provider.create_calls == []


def test_cli_lock_contention_is_retryable_and_never_runs_service(tmp_path) -> None:
    runtime, provider = _runtime(tmp_path)
    output = StringIO()

    with _exclusive_reconcile_lock(runtime.repository.path):
        exit_status = run_cli(
            ["--service-factory", FACTORY_REFERENCE],
            stdout=output,
            factory_loader=lambda _: lambda: runtime,
        )

    assert exit_status == EXIT_RETRYABLE
    assert json.loads(output.getvalue()) == {"status": "busy"}
    assert provider.create_calls == []


def test_cli_timeout_is_retryable_and_content_free(tmp_path) -> None:
    runtime, _ = _runtime(tmp_path)

    async def slow_reconcile():
        await asyncio.sleep(1)
        return ()

    runtime.reconcile = slow_reconcile
    output = StringIO()

    exit_status = run_cli(
        [
            "--service-factory",
            FACTORY_REFERENCE,
            "--timeout-seconds",
            "0.001",
        ],
        stdout=output,
        factory_loader=lambda _: lambda: runtime,
    )

    assert exit_status == EXIT_RETRYABLE
    assert json.loads(output.getvalue()) == {"status": "timeout"}


def test_cli_reports_durable_retry_without_provisioning_a_replacement(tmp_path) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    provider.create_error = RunPodAmbiguousResultError(
        title="timeout",
        detail=f"response lost {TOKEN}",
        method="POST",
        resource="/pods",
    )
    runtime = service(
        tmp_path,
        clock,
        provider,
        FakeCapabilityStore(clock),
        FakeWorkloadTransport(),
    )
    with pytest.raises(PodCapacityLifecycleError):
        asyncio.run(runtime.acquire_catalog(request(clock)))
    output = StringIO()

    exit_status = run_cli(
        ["--service-factory", FACTORY_REFERENCE],
        stdout=output,
        factory_loader=lambda _: lambda: runtime,
    )

    payload = json.loads(output.getvalue())
    assert exit_status == EXIT_RETRYABLE
    assert payload["status"] == "retryable"
    assert payload["retryable_error_count"] == 1
    assert payload["catalog_count"] == 1
    assert TOKEN not in output.getvalue()
    assert len(provider.create_calls) == 1


def test_cli_runtime_error_is_typed_without_exposing_message(tmp_path) -> None:
    runtime, _ = _runtime(tmp_path)
    leaked_secret = "runtime-secret-value"

    async def failed_reconcile():
        raise RunPodManagerError(f"provider failed with {leaked_secret}")

    runtime.reconcile = failed_reconcile
    output = StringIO()

    exit_status = run_cli(
        ["--service-factory", FACTORY_REFERENCE],
        stdout=output,
        factory_loader=lambda _: lambda: runtime,
    )

    assert exit_status == EXIT_RUNTIME_ERROR
    assert json.loads(output.getvalue()) == {
        "error_type": "RunPodManagerError",
        "status": "error",
    }
    assert leaked_secret not in output.getvalue()
