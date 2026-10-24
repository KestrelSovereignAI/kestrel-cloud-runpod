"""Runpod REST v2 training capacity adapter contracts."""

import asyncio
import threading
import json
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from pod_capacity_test_support import complete_pod_payload, native_pod_action_harness
from training_test_support import (
    MutableClock,
    training_profile,
    training_request,
    training_service,
)

from kestrel_cloud_runpod.clients import is_exact_runpod_not_found
from kestrel_cloud_runpod.models import (
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)
from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupState,
    TrainingPodOwnership,
    TrainingPodSource,
    TrainingPodState,
)
from kestrel_cloud_runpod.providers import DirectRunPodProvider
from kestrel_cloud_runpod.training_provider import RunpodTrainingPodProvider


@pytest.mark.asyncio
async def test_observe_resolves_exact_http_proxy_route() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.get_status.return_value = {
        "id": "pod-1",
        "status": "RUNNING",
        "runtime": {"ports": [{"private": 8888, "type": "http"}]},
    }
    adapter = RunpodTrainingPodProvider(provider)

    observed = await adapter.observe("pod-1", profile=training_profile())

    assert observed.is_running is True
    assert observed.backend_base_url == "https://pod-1-8888.proxy.runpod.net"


@pytest.mark.asyncio
async def test_create_uses_provider_v2_boundary_and_validates_id() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.start_pod.return_value = {"id": "pod-created"}
    adapter = RunpodTrainingPodProvider(provider)

    created = await adapter.create(
        profile=training_profile(),
        resource_name="kestrel-lora-durable-name",
        companion_id="companion-1",
    )

    assert created.provider_pod_id == "pod-created"
    assert provider.start_pod.call_args.args[1] == {
        "name": "kestrel-lora-durable-name",
        "companion_id": "companion-1",
        "purpose": "lora_training",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"status": "RUNNING"},
        {"id": "pod-created", "_kestrel_placement": "invalid"},
    ],
)
async def test_successful_create_with_unusable_identity_remains_reconcilable(
    response,
) -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.start_pod.return_value = response
    adapter = RunpodTrainingPodProvider(provider)

    with pytest.raises(RunPodAmbiguousResultError) as raised:
        await adapter.create(
            profile=training_profile(),
            resource_name="kestrel-lora-durable-name",
            companion_id="companion-1",
        )

    assert raised.value.reconcile_required is True


@pytest.mark.asyncio
async def test_find_by_name_rejects_duplicate_provider_resources() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.list_pods.return_value = [
        {"id": "pod-1", "name": "same"},
        {"id": "pod-2", "name": "same"},
    ]
    adapter = RunpodTrainingPodProvider(provider)

    with pytest.raises(RunPodManagerError, match="Multiple"):
        await adapter.find_by_name("same")


@pytest.mark.asyncio
async def test_stop_requires_v2_confirmation() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    provider.stop_pod.return_value = {"id": "pod-1", "status": "STOPPING"}
    provider.get_status.return_value = {"id": "pod-1", "status": "RUNNING"}
    adapter = RunpodTrainingPodProvider(provider)

    assert await adapter.stop("pod-1") is False


@pytest.mark.asyncio
async def test_cancelled_start_waits_for_inflight_v2_mutation_to_resolve() -> None:
    provider = MagicMock(spec=DirectRunPodProvider)
    entered = threading.Event()
    finish = threading.Event()

    def blocking_start(*_):
        entered.set()
        finish.wait(timeout=5)
        return {"id": "pod-1", "status": "RUNNING"}

    provider.resume_pod.side_effect = blocking_start
    adapter = RunpodTrainingPodProvider(provider)
    task = asyncio.create_task(adapter.start("pod-1", gpu_count=1))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False

    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_id,status,expected_error,running,stopped",
    [
        pytest.param("pod-foreign", "STOPPED", "Runpod v2 Pod status response did not match the requested Pod ID", None, None, id="foreign-stopped"),
        pytest.param("pod-foreign", "EXITED", "Runpod v2 Pod status response did not match the requested Pod ID", None, None, id="foreign-exited"),
        pytest.param("pod-foreign", "RUNNING", "Runpod v2 Pod status response did not match the requested Pod ID", None, None, id="foreign-running"),
        pytest.param(None, "STOPPED", "Invalid Pod.id response", None, None, id="missing-id"),
        pytest.param("pod-catalog-1", "RUNNING", None, True, False, id="owned-running"),
        pytest.param("pod-catalog-1", "STOPPED", None, False, True, id="owned-stopped"),
        pytest.param("pod-catalog-1", "EXITED", None, False, True, id="owned-exited"),
    ],
)
async def test_r29_native_observe_requires_exact_pod_identity(
    response_id, status, expected_error, running, stopped, record_property
):
    """Use the real transport/client/provider/adapter, not a replaced observe."""
    with native_pod_action_harness() as (state, _direct, adapter):
        payload = complete_pod_payload(
            pod_id=state["pod_id"] if response_id is None else response_id,
            status=status,
        )
        payload["runtime"] = {"ports": [{"private": 8888, "type": "http"}]}
        if response_id is None:
            del payload["id"]
        state["readback_payloads"].append(payload)
        record_property("injection_entry", "native-readback-before-exact-identity")
        observed = error = None
        try:
            observed = await adapter.observe(state["pod_id"], profile=training_profile())
        except RunPodManagerError as original:
            error = original
        record_property("r29_observe", json.dumps({
            "requested_id": state["pod_id"],
            "served": state["served_readbacks"],
            "calls": state["calls"],
            "error_type": None if error is None else type(error).__name__,
            "error": None if error is None else str(error),
            "observation": None if observed is None else {
                "id": observed.provider_pod_id, "status": observed.status,
                "raw_id": observed.raw.get("id"), "route": observed.backend_base_url,
            },
        }, sort_keys=True))
        assert state["served_readbacks"] == [payload]
        assert state["calls"] == [("GET", f"/v2/pods/{state['pod_id']}", None)]
        if expected_error is not None:
            assert type(error) is RunPodManagerError
            assert str(error) == expected_error
            assert observed is None
        else:
            assert error is None and observed is not None
            assert observed.provider_pod_id == observed.raw["id"] == state["pod_id"]
            assert observed.status == status
            assert observed.is_running is running
            assert observed.is_stopped is stopped
            assert observed.backend_base_url == (
                f"https://{state['pod_id']}-8888.proxy.runpod.net"
            )


@pytest.mark.asyncio
async def test_r29_native_observe_preserves_typed_404_cleanup_semantics(record_property):
    with native_pod_action_harness() as (state, _direct, adapter):
        state["readback"] = "absent"
        error = None
        try:
            await adapter.observe(state["pod_id"], profile=training_profile())
        except RunPodAPIError as original:
            error = original
        # Observation still propagates 404; the existing cleanup operation,
        # separately addressed to that same ID, retains canonical idempotency.
        state["action_status"] = 404
        state["action_payload"] = {"title": "not found"}
        confirmed = await adapter.stop(state["pod_id"])
        record_property("r29_observe", json.dumps({
            "requested_id": state["pod_id"], "calls": state["calls"],
            "error_type": None if error is None else type(error).__name__,
            "status_code": None if error is None else error.status_code,
            "method": None if error is None else error.method,
            "resource": None if error is None else error.resource,
            "exact_not_found": error is not None and is_exact_runpod_not_found(error),
            "separate_stop_confirmed": confirmed,
        }, sort_keys=True))
        assert type(error) is RunPodAPIError and error.status_code == 404
        assert error.method == "GET" and error.resource == f"/pods/{state['pod_id']}"
        assert error.__cause__ is None and is_exact_runpod_not_found(error)
        assert confirmed is True
        assert state["calls"] == [
            ("GET", f"/v2/pods/{state['pod_id']}", None),
            ("POST", f"/v2/pods/{state['pod_id']}/action", {"action": "stop"}),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["foreign-stopped", "foreign-exited", "owned-running", "owned-stopped", "not-found"],
)
async def test_r29_native_training_reconcile_does_not_complete_foreign_pod(
    tmp_path, case, record_property
):
    """Acquire owned capacity normally, then observe through the REAL adapter."""
    clock = MutableClock()
    with native_pod_action_harness() as (state, _direct, adapter):
        initially_stopped = complete_pod_payload(pod_id=state["pod_id"], status="EXITED")
        initially_running = complete_pod_payload(pod_id=state["pod_id"], status="RUNNING")
        initially_running["runtime"] = {"ports": [{"private": 8888, "type": "http"}]}
        state["readback_payloads"].extend([initially_stopped, initially_running])
        runtime = training_service(tmp_path, clock, adapter)
        lease = await runtime.acquire(training_request(
            clock, source=TrainingPodSource.STOPPED_REUSE, pod_id=state["pod_id"]
        ))
        record_property("r29_fixture", json.dumps({
            "original": lease.to_public_dict(), "clock": clock().isoformat(),
            "calls": state["calls"], "served": state["served_readbacks"],
        }, sort_keys=True))
        assert lease.state is TrainingPodState.READY
        assert lease.ownership is TrainingPodOwnership.OWNED
        assert lease.cleanup_state is TrainingPodCleanupState.NOT_REQUESTED
        assert lease.owns_billing_capacity
        assert lease.provider_pod_id == state["pod_id"]
        assert state["served_readbacks"] == [initially_stopped, initially_running]
        assert state["calls"] == [
            ("GET", f"/v2/pods/{state['pod_id']}", None),
            ("POST", f"/v2/pods/{state['pod_id']}/action", {"action": "start"}),
            ("GET", f"/v2/pods/{state['pod_id']}", None),
        ]
        assert clock() < lease.hard_deadline
        assert clock() < lease.last_heartbeat_at + timedelta(seconds=runtime.orphan_timeout_seconds)

        record_property("injection_entry", "owned-ready-lease-before-observation")
        response = None
        if case == "not-found":
            state["readback"] = "absent"
        else:
            status = "RUNNING" if case == "owned-running" else (
                "EXITED" if case == "foreign-exited" else "STOPPED"
            )
            response = complete_pod_payload(
                pod_id="pod-foreign" if case.startswith("foreign-") else state["pod_id"],
                status=status,
            )
            response["runtime"] = {"ports": [{"private": 8888, "type": "http"}]}
            state["readback_payloads"].append(response)
        before_observe = len(state["calls"])
        results = await runtime.reconcile()
        retained = runtime.repository.get(lease.cleanup_token)
        scheduled = runtime.repository.list_for_reconciliation()
        observation_calls = state["calls"][before_observe:]

        # A correct-ID terminal action is now available for explicit owner
        # cleanup. A false RELEASED record would incorrectly skip this stop.
        state["action_status"] = 200
        state["action_payload"] = complete_pod_payload(
            pod_id=state["pod_id"], status="STOPPED"
        )
        before_release = len(state["calls"])
        released = await runtime.release(lease.cleanup_token, reason="r29 exact owner cleanup")
        release_calls = state["calls"][before_release:]
        record_property("r29_training", json.dumps({
            "case": case, "requested_id": state["pod_id"],
            "original": lease.to_public_dict(),
            "served_observation": response,
            "retained": None if retained is None else retained.to_public_dict(),
            "results": [item.to_public_dict() for item in results],
            "reconciliation_ids": [item.cleanup_token for item in scheduled],
            "observation_calls": observation_calls,
            "explicit_release": released.to_public_dict(),
            "release_calls": release_calls,
            "actual_live_provider_or_money_proof": False,
        }, sort_keys=True))
        assert retained is not None and len(results) == 1
        assert observation_calls == [("GET", f"/v2/pods/{state['pod_id']}", None)]
        assert retained.provider_pod_id == lease.provider_pod_id
        assert retained.request_fingerprint == lease.request_fingerprint
        assert retained.readiness_deadline == lease.readiness_deadline
        assert retained.hard_deadline == lease.hard_deadline
        if case == "owned-stopped":
            assert retained.is_terminal
            assert retained.cleanup_state is TrainingPodCleanupState.COMPLETE
            assert retained.last_provider_error is None
            assert lease.cleanup_token not in [item.cleanup_token for item in scheduled]
            assert release_calls == []
        else:
            assert retained.state is TrainingPodState.READY
            assert retained.cleanup_state is TrainingPodCleanupState.NOT_REQUESTED
            assert retained.ownership is TrainingPodOwnership.OWNED
            assert retained.owns_billing_capacity and not retained.is_terminal
            assert lease.cleanup_token in [item.cleanup_token for item in scheduled]
            assert retained.last_provider_error == (
                "RunPodAPIError" if case == "not-found" else
                None if case == "owned-running" else "RunPodManagerError"
            )
            assert release_calls == [
                ("POST", f"/v2/pods/{state['pod_id']}/action", {"action": "stop"}),
            ]
        assert released.state is TrainingPodState.RELEASED
        assert released.cleanup_state is TrainingPodCleanupState.COMPLETE
        assert released.provider_pod_id == lease.provider_pod_id
