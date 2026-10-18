"""Public opaque recovery boundary tests; not private-worker or GPU proof."""

import asyncio
import copy
import json
from datetime import timedelta

import httpx
import pytest
from pod_capacity_test_support import (
    ACK_SHA,
    IMAGE,
    RECEIPT_SHA,
    REQUEST_SHA,
    RESULT_SHA,
    TOKEN,
    FakeWorkloadTransport,
    acknowledgement,
    service,
)
from pydantic import SecretStr
from test_catalog_artifact_handoff import pending

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.pod_capacity_contracts import (
    CatalogPodWorkloadState,
    TrainingPodConflictError,
    TrainingPodState,
)
from kestrel_cloud_runpod.pod_transport import (
    CatalogPodTransportConflictError,
    CatalogPodTransportError,
    CatalogPodWorkloadTransport,
)


def recovery_payload():
    envelope = copy.deepcopy(FakeWorkloadTransport().result_payload)
    envelope["receipt"]["image_digest"] = IMAGE.rsplit("@", 1)[1]
    envelope["result"]["opaque_private_result"] = "not-retained-in-sdk-state"
    return {"handoff": envelope, "acknowledgement": acknowledgement()}


class RecoveryTransport(FakeWorkloadTransport):
    def __init__(self):
        super().__init__()
        self.recovery_calls = []
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()
        self.block = False
        self.fail = False

    async def recover_handoff(self, **values):
        self.recovery_calls.append(values["payload"])
        self.entered.set()
        if self.block:
            await self.finish.wait()
        if self.fail:
            raise CatalogPodTransportError("injected recovery response loss")
        self.status_value = CatalogPodWorkloadState.ACKNOWLEDGED
        return await self.status(**values)


@pytest.mark.asyncio
async def test_recovery_records_actual_proof_without_execution_or_private_persistence(
    tmp_path,
):
    runtime, scope, clock, provider, store, transport = await pending(
        tmp_path, RecoveryTransport()
    )
    payload = recovery_payload()
    lease = await runtime.recover_catalog_result(**scope, recovery=payload)
    assert lease.state is TrainingPodState.JOB_COMPLETED
    assert lease.workload_state is CatalogPodWorkloadState.ACKNOWLEDGED
    assert lease.artifact_handoff.remote_ack_observed_at == clock()
    assert lease.artifact_handoff.host_confirmation_declared_at is None
    assert (
        len(transport.recovery_calls)
        == len(transport.submit_calls)
        == len(provider.create_calls)
        == 1
    )
    assert not transport.acknowledgement_calls and not provider.terminate_calls
    assert (
        b"not-retained-in-sdk-state" not in (tmp_path / "capacity.sqlite3").read_bytes()
    )
    await store.revoke(lease.capacity_spec.request.attempt_id)
    clock.value += timedelta(days=1)
    store.capabilities.clear()
    restarted = service(tmp_path, clock, provider, store, transport)
    assert await restarted.recover_catalog_result(**scope, recovery=payload) == lease
    assert len(transport.recovery_calls) == 1
    changed = copy.deepcopy(payload)
    changed["acknowledgement"]["opaque_private_signature"] = "changed"
    with pytest.raises(TrainingPodConflictError):
        await restarted.recover_catalog_result(**scope, recovery=changed)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ack", "recovery"])
@pytest.mark.parametrize("during_load", [False, True])
async def test_fresh_ack_or_recovery_checks_hard_deadline_twice(
    tmp_path, operation, during_load
):
    runtime, scope, clock, provider, store, transport = await pending(
        tmp_path, RecoveryTransport()
    )
    lease = runtime.repository.get(scope["capacity_id"])
    if during_load:
        load = store.load

        async def expire_then_load(attempt_id):
            clock.value = lease.hard_deadline
            return await load(attempt_id)

        store.load = expire_then_load
    else:
        clock.value = lease.hard_deadline
    with pytest.raises(RunPodManagerError, match="hard deadline"):
        if operation == "ack":
            await runtime.acknowledge_catalog_result(
                **scope, acknowledgement=acknowledgement()
            )
        else:
            await runtime.recover_catalog_result(**scope, recovery=recovery_payload())
    assert not transport.recovery_calls and not transport.acknowledgement_calls
    assert not provider.terminate_calls and len(transport.submit_calls) == 1
    assert (
        runtime.repository.get(
            scope["capacity_id"]
        ).artifact_handoff.remote_ack_observed_at
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [False, True])
async def test_recovery_tail_owns_snapshot_and_actual_proof_through_cancellation(
    tmp_path, stop
):
    transport = RecoveryTransport()
    transport.block = True
    runtime, scope, clock, provider, _store, transport = await pending(
        tmp_path, transport
    )
    payload = recovery_payload()
    original = copy.deepcopy(payload)
    task = asyncio.create_task(
        runtime.recover_catalog_result(**scope, recovery=payload)
    )
    await asyncio.wait_for(transport.entered.wait(), 5)
    payload["handoff"]["result"]["opaque_private_result"] = "caller mutation"
    payload["acknowledgement"]["opaque_private_signature"] = "caller mutation"
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    if stop:
        runtime.request_cancellation(scope["capacity_id"])
        await runtime.reconcile()
    transport.finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.recovery_calls == [original]
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.artifact_handoff.remote_ack_observed_at == clock()
    assert saved.artifact_handoff.host_confirmation_declared_at is None
    assert saved.state is (
        TrainingPodState.RELEASING if stop else TrainingPodState.JOB_COMPLETED
    )
    assert len(provider.terminate_calls) == int(stop)


@pytest.mark.asyncio
async def test_stop_during_capability_load_prevents_recovery_and_keeps_only_intent(
    tmp_path,
):
    runtime, scope, _clock, _provider, store, transport = await pending(
        tmp_path, RecoveryTransport()
    )
    load = store.load

    async def stop_then_load(attempt_id):
        runtime.request_cancellation(scope["capacity_id"])
        return await load(attempt_id)

    store.load = stop_then_load
    with pytest.raises(RunPodManagerError, match="stopped before dispatch"):
        await runtime.recover_catalog_result(**scope, recovery=recovery_payload())
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.artifact_handoff.acknowledgement_sha256 == ACK_SHA
    assert saved.artifact_handoff.remote_ack_observed_at is None
    assert not transport.recovery_calls


@pytest.mark.asyncio
async def test_recovery_without_live_capability_never_invents_actual_proof(tmp_path):
    runtime, scope, _clock, provider, store, transport = await pending(
        tmp_path, RecoveryTransport()
    )
    store.capabilities.clear()
    with pytest.raises(RunPodManagerError, match="capability is unavailable"):
        await runtime.recover_catalog_result(**scope, recovery=recovery_payload())
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.artifact_handoff.acknowledgement_sha256 == ACK_SHA
    assert saved.artifact_handoff.remote_ack_observed_at is None
    assert not transport.recovery_calls and not provider.terminate_calls


@pytest.mark.asyncio
async def test_recovery_transport_failure_preserves_intent_and_retries_without_resubmission(
    tmp_path,
):
    transport = RecoveryTransport()
    transport.fail = True
    runtime, scope, clock, provider, store, transport = await pending(
        tmp_path, transport
    )
    with pytest.raises(CatalogPodTransportError):
        await runtime.recover_catalog_result(**scope, recovery=recovery_payload())
    saved = runtime.repository.get(scope["capacity_id"])
    assert saved.artifact_handoff.acknowledgement_sha256 == ACK_SHA
    assert saved.artifact_handoff.remote_ack_observed_at is None
    transport.fail = False
    restarted = service(tmp_path, clock, provider, store, transport)
    assert (
        await restarted.recover_catalog_result(**scope, recovery=recovery_payload())
    ).artifact_handoff.remote_ack_observed_at == clock()
    assert (
        len(transport.recovery_calls) == 2
        and len(transport.submit_calls) == len(provider.create_calls) == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ["attempt", "request", "image", "result", "receipt", "ack", "protocol", "extra"],
)
async def test_recovery_rejects_binding_changes_before_transport(tmp_path, field):
    runtime, scope, _clock, _provider, _store, transport = await pending(
        tmp_path, RecoveryTransport()
    )
    payload = recovery_payload()
    if field == "attempt":
        payload["handoff"]["receipt"]["attempt_id"] = "another-attempt"
    elif field == "request":
        payload["handoff"]["receipt"]["request_sha256"] = "a" * 64
    elif field == "image":
        payload["handoff"]["receipt"]["image_digest"] = "sha256:" + "b" * 64
    elif field == "result":
        payload["handoff"]["result"]["request_sha256"] = "a" * 64
    elif field == "receipt":
        payload["handoff"]["receipt"]["receipt_sha256"] = "a" * 64
    elif field == "ack":
        payload["acknowledgement"]["result_sha256"] = "a" * 64
    elif field == "protocol":
        payload["handoff"]["protocol_version"] = True
    else:
        payload["execution_request"] = {}
    with pytest.raises(ValueError):
        await runtime.recover_catalog_result(**scope, recovery=payload)
    assert not transport.recovery_calls and not transport.acknowledgement_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 401, 409, 500])
async def test_opaque_recovery_transport_uses_only_recovery_route_and_actual_response(
    status,
):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            status,
            json={
                "attempt_id": "attempt:catalog-run-0001",
                "request_sha256": REQUEST_SHA,
                "state": "acknowledged",
                "error_type": None,
                "result_available": True,
                "result_sha256": RESULT_SHA,
                "receipt_sha256": RECEIPT_SHA,
                "acknowledgement_sha256": ACK_SHA,
            },
        )

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    values = {
        "base_url": "https://pod-1-8080.proxy.runpod.net",
        "attempt_id": "attempt:catalog-run-0001",
        "request_sha256": REQUEST_SHA,
        "bearer_token": SecretStr(TOKEN),
        "payload": recovery_payload(),
    }
    if status == 200:
        assert (
            await transport.recover_handoff(**values)
        ).acknowledgement_sha256 == ACK_SHA
    else:
        with pytest.raises(CatalogPodTransportError):
            await transport.recover_handoff(**values)
    assert len(seen) == 1 and seen[0].url.path.endswith("/handoff-recovery")
    assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(seen[0].content) == recovery_payload()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["result_sha256", "receipt_sha256", "acknowledgement_sha256"]
)
async def test_recovery_response_must_match_exact_host_intent(name):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "attempt_id": "attempt:catalog-run-0001",
                "request_sha256": REQUEST_SHA,
                "state": "acknowledged",
                "error_type": None,
                "result_available": True,
                "result_sha256": RESULT_SHA,
                "receipt_sha256": RECEIPT_SHA,
                "acknowledgement_sha256": ACK_SHA,
                name: "a" * 64,
            },
        )

    transport = CatalogPodWorkloadTransport(http_transport=httpx.MockTransport(handler))
    with pytest.raises(CatalogPodTransportConflictError):
        await transport.recover_handoff(
            base_url="https://pod-1-8080.proxy.runpod.net",
            attempt_id="attempt:catalog-run-0001",
            request_sha256=REQUEST_SHA,
            bearer_token=SecretStr(TOKEN),
            payload=recovery_payload(),
        )
