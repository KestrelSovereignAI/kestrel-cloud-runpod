"""Installed, locked, content-free Pod-capacity reconciler command tests."""

import asyncio
import errno
import json
import os
import sys
from importlib.metadata import distribution
from io import StringIO
from types import SimpleNamespace

import pytest
from pod_capacity_test_support import (
    TOKEN,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    execution_control_carrier,
    record_control_injection,
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
    PodCapacityReconcilerBusyError,
    PodCapacityReconcilerConfigurationError,
    _exclusive_reconcile_lock,
    load_service_factory,
    reconcile_once,
    run_cli,
)

FACTORY_REFERENCE = "catalog_host.runpod:build_capacity_service"


def _module_os_proxy(**overrides):
    attributes = vars(os).copy()
    attributes.update(overrides)
    return SimpleNamespace(**attributes)


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


@pytest.mark.parametrize(
    "boundary",
    [
        "module-import",
        "attribute",
        "factory",
        "runtime-timeout",
        "runtime-manager",
        "runtime-os",
        "outer-busy",
        "outer-configuration",
        "lock-open",
        "lock-posix",
        "lock-windows",
    ],
)
def test_reconciler_does_not_convert_execution_control_to_runtime_exit(
    tmp_path, monkeypatch, boundary, record_property
):
    host_os_name = os.name
    runtime, provider = _runtime(tmp_path)
    error_type = {
        "module-import": ImportError,
        "attribute": AttributeError,
        "runtime-timeout": TimeoutError,
        "runtime-os": OSError,
        "lock-open": OSError,
        "lock-posix": OSError,
        "lock-windows": OSError,
        "outer-busy": PodCapacityReconcilerBusyError,
        "outer-configuration": PodCapacityReconcilerConfigurationError,
    }.get(boundary, RunPodManagerError)
    carrier, control, explicit = execution_control_carrier(
        error_type(f"reconciler {boundary} outcome remains unknown"),
        grouped=True,
    )

    def inject() -> None:
        record_control_injection(record_property, f"reconciler-{boundary}")

    async def controlled_reconcile():
        inject()
        raise carrier

    output = StringIO()

    with pytest.raises(type(carrier)) as raised:
        if boundary == "module-import":
            monkeypatch.setattr(
                "kestrel_cloud_runpod.pod_capacity_reconciler.importlib.import_module",
                lambda _name: (inject(), (_ for _ in ()).throw(carrier))[1],
            )
            load_service_factory(FACTORY_REFERENCE)
        elif boundary == "attribute":

            class ControlledModule:
                def __getattribute__(self, _name):
                    inject()
                    raise carrier

            monkeypatch.setattr(
                "kestrel_cloud_runpod.pod_capacity_reconciler.importlib.import_module",
                lambda _name: ControlledModule(),
            )
            load_service_factory(FACTORY_REFERENCE)
        elif boundary.startswith("lock-"):
            if boundary == "lock-open":
                monkeypatch.setattr(
                    "kestrel_cloud_runpod.pod_capacity_reconciler.os",
                    _module_os_proxy(
                        open=lambda *_args, **_kwargs: (
                            inject(),
                            (_ for _ in ()).throw(carrier),
                        )[1]
                    ),
                )
            elif boundary == "lock-posix":
                monkeypatch.setattr(
                    "fcntl.flock",
                    lambda *_args, **_kwargs: (
                        inject(),
                        (_ for _ in ()).throw(carrier),
                    )[1],
                )
            else:
                monkeypatch.setattr(
                    "kestrel_cloud_runpod.pod_capacity_reconciler.os",
                    _module_os_proxy(name="nt"),
                )
                monkeypatch.setitem(
                    sys.modules,
                    "msvcrt",
                    SimpleNamespace(
                        LK_NBLCK=1,
                        locking=lambda *_args, **_kwargs: (
                            inject(),
                            (_ for _ in ()).throw(carrier),
                        )[1],
                    ),
                )
            with _exclusive_reconcile_lock(tmp_path / "capacity.sqlite3"):
                raise AssertionError("controlled lock failure was suppressed")
        else:
            if boundary.startswith("runtime-"):
                runtime.reconcile = controlled_reconcile

            def factory():
                if boundary in {"factory", "outer-busy", "outer-configuration"}:
                    inject()
                    raise carrier
                return runtime

            factory_loader = lambda _reference: (
                (inject(), (_ for _ in ()).throw(carrier))[1]
                if boundary in {"outer-busy", "outer-configuration"}
                else factory
            )
            run_cli(
                ["--service-factory", FACTORY_REFERENCE],
                stdout=output,
                factory_loader=factory_loader,
            )

    assert raised.value is carrier
    assert carrier.__cause__ is explicit
    assert isinstance(explicit, BaseExceptionGroup)
    assert explicit.exceptions[0] is control
    assert output.getvalue() == ""
    assert provider.create_calls == []
    assert os.name == host_os_name


@pytest.mark.parametrize(
    "boundary",
    [
        "module-import",
        "attribute",
        "runtime-os",
        "lock-open",
        "lock-posix",
        "lock-windows",
    ],
)
def test_reconciler_retains_ordinary_normalization_contract(
    tmp_path, monkeypatch, boundary
):
    host_os_name = os.name
    runtime, provider = _runtime(tmp_path)
    output = StringIO()

    if boundary == "module-import":
        monkeypatch.setattr(
            "kestrel_cloud_runpod.pod_capacity_reconciler.importlib.import_module",
            lambda _name: (_ for _ in ()).throw(ImportError("missing module")),
        )
        with pytest.raises(PodCapacityReconcilerConfigurationError):
            load_service_factory(FACTORY_REFERENCE)
    elif boundary == "attribute":
        monkeypatch.setattr(
            "kestrel_cloud_runpod.pod_capacity_reconciler.importlib.import_module",
            lambda _name: SimpleNamespace(),
        )
        with pytest.raises(PodCapacityReconcilerConfigurationError):
            load_service_factory(FACTORY_REFERENCE)
    elif boundary == "runtime-os":

        async def failed_reconcile():
            raise OSError("ordinary runtime failure")

        runtime.reconcile = failed_reconcile
        assert (
            run_cli(
                ["--service-factory", FACTORY_REFERENCE],
                stdout=output,
                factory_loader=lambda _reference: lambda: runtime,
            )
            == EXIT_RUNTIME_ERROR
        )
        assert json.loads(output.getvalue()) == {
            "status": "error",
            "error_type": "OSError",
        }
    elif boundary == "lock-open":
        monkeypatch.setattr(
            "kestrel_cloud_runpod.pod_capacity_reconciler.os",
            _module_os_proxy(
                open=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    OSError(errno.EIO, "ordinary lock-open failure")
                )
            ),
        )
        with pytest.raises(PodCapacityReconcilerConfigurationError):
            with _exclusive_reconcile_lock(runtime.repository.path):
                raise AssertionError("lock-open failure was suppressed")
    else:
        if boundary == "lock-posix":
            monkeypatch.setattr(
                "fcntl.flock",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    OSError(errno.EIO, "ordinary POSIX lock failure")
                ),
            )
        else:
            monkeypatch.setattr(
                "kestrel_cloud_runpod.pod_capacity_reconciler.os",
                _module_os_proxy(name="nt"),
            )
            monkeypatch.setitem(
                sys.modules,
                "msvcrt",
                SimpleNamespace(
                    LK_NBLCK=1,
                    locking=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError(errno.EIO, "ordinary Windows lock failure")
                    ),
                ),
            )
        with pytest.raises(PodCapacityReconcilerConfigurationError):
            with _exclusive_reconcile_lock(runtime.repository.path):
                raise AssertionError("lock failure was suppressed")

    assert provider.create_calls == []
    assert os.name == host_os_name
