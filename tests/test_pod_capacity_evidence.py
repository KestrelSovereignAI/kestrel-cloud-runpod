"""Durable content-free realized Pod and worker evidence contract tests."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta

import pytest
from pod_capacity_test_support import (
    IMAGE,
    REQUEST_SHA,
    FakeCapabilityStore,
    FakeCapacityProvider,
    FakeWorkloadTransport,
    MutableClock,
    final_receipt,
    request,
    service,
)

from kestrel_cloud_runpod.models import RunPodManagerError
from kestrel_cloud_runpod.pod_capacity_contracts import (
    CatalogPodWorkloadState,
    CatalogWorkerEvidence,
    PodCapacityBillingState,
    PodCapacityLifecycleEvidence,
    TrainingPodConflictError,
)
from kestrel_cloud_runpod.pod_capacity_repository import SQLitePodCapacityRepository


def worker_envelope(clock: MutableClock) -> dict[str, object]:
    return {
        "schema_version": 1,
        "attempt_id": "attempt:catalog-run-0001",
        "request_sha256": REQUEST_SHA,
        "image_digest": IMAGE.rsplit("@", 1)[1],
        "container_process_started_at": clock().isoformat(),
        "timings_seconds": {
            "image_pull_and_container_boot_seconds": "15.5",
            "image_pull_seconds": "10.25",
            "container_boot_seconds": "5.25",
            "model_load_seconds": "7.75",
            "execution_seconds": "91.125",
            "training_seconds": "80.5",
            "artifact_upload_seconds": "2.875",
        },
        "metrics": {
            "peak_vram_bytes": 24_000_000_000,
            "peak_host_ram_bytes": 8_000_000_000,
            "gpu_seconds": "88.5",
            "idle_seconds": "18.25",
        },
    }


async def successful_workload(tmp_path):
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    store = FakeCapabilityStore(clock)
    transport = FakeWorkloadTransport()
    runtime = service(tmp_path, clock, provider, store, transport)
    catalog_request = request(clock)
    ready = await runtime.acquire_catalog(catalog_request)
    clock.value += timedelta(seconds=1)
    await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload={
            "dispatch_attempt_id": catalog_request.attempt_id,
            "request_sha256": catalog_request.request_sha256,
        },
    )
    clock.value += timedelta(seconds=1)
    transport.status_value = CatalogPodWorkloadState.SUCCEEDED
    await runtime.observe_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    return runtime, catalog_request, clock, provider, ready


@pytest.mark.asyncio
async def test_realized_lifecycle_worker_and_billing_evidence_survive_restart(
    tmp_path,
) -> None:
    runtime, catalog_request, clock, provider, ready = await successful_workload(
        tmp_path
    )
    assert ready.evidence is not None
    assert ready.evidence.realized_placement is not None
    assert ready.evidence.realized_placement.data_center_id == "US-TX-3"
    assert (
        ready.evidence.lifecycle.provider_create_accepted_at
        == request(MutableClock()).created_at
    )
    assert ready.evidence.lifecycle.first_running_observed_at is not None
    assert ready.evidence.lifecycle.worker_ready_at is not None

    evidence_envelope = worker_envelope(clock)
    recorded = runtime.record_catalog_worker_evidence(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        envelope=evidence_envelope,
    )
    replay = runtime.record_catalog_worker_evidence(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        envelope=evidence_envelope,
    )
    assert replay.evidence == recorded.evidence

    clock.value += timedelta(seconds=1)
    pending = await runtime.acknowledge_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    assert pending.billing_state is PodCapacityBillingState.PENDING
    assert pending.evidence is not None
    assert pending.evidence.lifecycle.stop_confirmed_at == clock()
    assert pending.terminal_success_evidence_complete is False

    provider.billing_receipt = final_receipt(clock)
    await runtime.reconcile()
    expected = runtime.repository.get(catalog_request.capacity_id)
    assert expected is not None
    assert expected.terminal_success_evidence_complete is True
    assert expected.evidence is not None
    assert expected.evidence.billing == expected.billing_receipt
    assert expected.evidence.lifecycle.billing_reconciled_at == clock()

    restarted = SQLitePodCapacityRepository(runtime.repository.path).get(
        catalog_request.capacity_id
    )
    assert restarted == expected
    identical_terminal_replay = runtime.record_catalog_worker_evidence(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        envelope=evidence_envelope,
    )
    assert identical_terminal_replay == expected
    mismatched_terminal_replay = deepcopy(evidence_envelope)
    mismatched_terminal_replay["metrics"]["idle_seconds"] = "19"  # type: ignore[index]
    with pytest.raises(TrainingPodConflictError, match="already recorded"):
        runtime.record_catalog_worker_evidence(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
            envelope=mismatched_terminal_replay,
        )


@pytest.mark.asyncio
async def test_duplicate_observations_and_cas_race_preserve_first_timestamps(
    tmp_path,
) -> None:
    clock = MutableClock()
    provider = FakeCapacityProvider(clock)
    runtime = service(
        tmp_path,
        clock,
        provider,
        FakeCapabilityStore(clock),
        FakeWorkloadTransport(),
    )
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    first = runtime.repository.get(catalog_request.capacity_id)
    assert first is not None and first.evidence is not None
    first_running = first.evidence.lifecycle.first_running_observed_at
    first_ready = first.evidence.lifecycle.worker_ready_at

    clock.value += timedelta(seconds=2)
    payload = {
        "dispatch_attempt_id": catalog_request.attempt_id,
        "request_sha256": catalog_request.request_sha256,
    }
    await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload=payload,
    )
    submitted = runtime.repository.get(catalog_request.capacity_id)
    assert submitted is not None and submitted.evidence is not None
    first_submitted = submitted.evidence.lifecycle.workload_submitted_at
    first_workload_running = submitted.evidence.lifecycle.workload_running_at

    clock.value += timedelta(seconds=2)
    await runtime.submit_catalog_workload(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        payload=payload,
    )
    duplicate = runtime.repository.get(catalog_request.capacity_id)
    assert duplicate is not None and duplicate.evidence is not None
    assert duplicate.evidence.lifecycle.first_running_observed_at == first_running
    assert duplicate.evidence.lifecycle.worker_ready_at == first_ready
    assert duplicate.evidence.lifecycle.workload_submitted_at == first_submitted
    assert duplicate.evidence.lifecycle.workload_running_at == first_workload_running

    # Two processes race from the same stale revision. The loser reloads the row
    # and must retain the winner's first terminal observation.
    stale = duplicate
    first_candidate = clock() + timedelta(seconds=1)
    second_candidate = clock() + timedelta(seconds=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda candidate: runtime._transition_with_evidence(
                    stale,
                    changes={"last_heartbeat_at": candidate},
                    lifecycle={"workload_terminal_at": candidate},
                ),
                (first_candidate, second_candidate),
            )
        )
    durable = runtime.repository.get(catalog_request.capacity_id)
    assert durable is not None and durable.evidence is not None
    first_terminal = durable.evidence.lifecycle.workload_terminal_at
    assert first_terminal in {first_candidate, second_candidate}
    assert all(
        result.evidence is not None
        and result.evidence.lifecycle.workload_terminal_at == first_terminal
        for result in results
    )


@pytest.mark.asyncio
async def test_worker_evidence_binding_replay_and_strict_allowlist(tmp_path) -> None:
    runtime, catalog_request, clock, _, _ = await successful_workload(tmp_path)
    valid = worker_envelope(clock)
    runtime.record_catalog_worker_evidence(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        envelope=valid,
    )

    changed = deepcopy(valid)
    changed["metrics"]["gpu_seconds"] = "89"  # type: ignore[index]
    with pytest.raises(TrainingPodConflictError, match="already recorded"):
        runtime.record_catalog_worker_evidence(
            capacity_id=catalog_request.capacity_id,
            owner_id=catalog_request.owner_id,
            workload_id=catalog_request.workload_id,
            envelope=changed,
        )

    for field, value in (
        ("attempt_id", "attempt:other-run-0002"),
        ("request_sha256", "f" * 64),
        ("image_digest", "sha256:" + "e" * 64),
    ):
        mismatched = deepcopy(valid)
        mismatched[field] = value
        with pytest.raises(TrainingPodConflictError, match="exact attempt binding"):
            runtime.record_catalog_worker_evidence(
                capacity_id=catalog_request.capacity_id,
                owner_id=catalog_request.owner_id,
                workload_id=catalog_request.workload_id,
                envelope=mismatched,
            )

    invalid_envelopes = []
    for private_key in ("prompt", "response", "signed_url", "weights", "capability"):
        value = deepcopy(valid)
        value[private_key] = "private"
        invalid_envelopes.append(value)
    unknown_timing = deepcopy(valid)
    unknown_timing["timings_seconds"]["private_phase"] = 1  # type: ignore[index]
    invalid_envelopes.append(unknown_timing)
    unknown_metric = deepcopy(valid)
    unknown_metric["metrics"]["temperature"] = 1  # type: ignore[index]
    invalid_envelopes.append(unknown_metric)
    invalid_timestamp = deepcopy(valid)
    invalid_timestamp["container_process_started_at"] = 123
    invalid_envelopes.append(invalid_timestamp)
    negative = deepcopy(valid)
    negative["metrics"]["gpu_seconds"] = -1  # type: ignore[index]
    invalid_envelopes.append(negative)
    not_finite = deepcopy(valid)
    not_finite["timings_seconds"]["training_seconds"] = float("nan")  # type: ignore[index]
    invalid_envelopes.append(not_finite)
    contradictory = deepcopy(valid)
    contradictory["timings_seconds"][  # type: ignore[index]
        "image_pull_and_container_boot_seconds"
    ] = "16"
    invalid_envelopes.append(contradictory)
    incomplete_split = deepcopy(valid)
    incomplete_split["timings_seconds"]["image_pull_seconds"] = None  # type: ignore[index]
    invalid_envelopes.append(incomplete_split)

    for invalid in invalid_envelopes:
        with pytest.raises((ValueError, RunPodManagerError)):
            CatalogWorkerEvidence.from_envelope(invalid)


@pytest.mark.asyncio
async def test_public_projection_is_content_free_and_omits_capabilities(
    tmp_path,
) -> None:
    runtime, catalog_request, clock, provider, _ = await successful_workload(tmp_path)
    runtime.record_catalog_worker_evidence(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
        envelope=worker_envelope(clock),
    )
    provider.billing_receipt = final_receipt(clock)
    await runtime.acknowledge_catalog_result(
        capacity_id=catalog_request.capacity_id,
        owner_id=catalog_request.owner_id,
        workload_id=catalog_request.workload_id,
    )
    lease = runtime.repository.get(catalog_request.capacity_id)
    assert lease is not None
    projection = lease.to_public_dict()
    serialized = json.dumps(projection, sort_keys=True)
    assert provider.route not in serialized
    assert "catalog-pod-token" not in serialized
    assert catalog_request.image_reference not in serialized

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    public_keys = keys(projection)
    assert public_keys.isdisjoint(
        {
            "backend_base_url",
            "bearer_token",
            "capability_secret_id",
            "capability_token_sha256",
            "capability_expires_at",
            "image_reference",
            "prompt",
            "response",
            "signed_url",
            "url",
            "weights",
            "artifact_capability",
            "raw",
        }
    )


@pytest.mark.asyncio
async def test_v2_catalog_row_migrates_with_explicit_missing_evidence(tmp_path) -> None:
    clock = MutableClock()
    runtime = service(
        tmp_path,
        clock,
        FakeCapacityProvider(clock),
        FakeCapabilityStore(clock),
        FakeWorkloadTransport(),
    )
    catalog_request = request(clock)
    await runtime.acquire_catalog(catalog_request)
    database = runtime.repository.path
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE pod_capacity_leases DROP COLUMN evidence_json")
        connection.execute("PRAGMA user_version = 2")

    migrated = SQLitePodCapacityRepository(database)
    legacy = migrated.get(catalog_request.capacity_id)
    assert legacy is not None
    assert legacy.evidence is None
    assert legacy.to_public_dict()["evidence"] is None
    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(pod_capacity_leases)")
        }
        assert "evidence_json" in columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_lifecycle_timestamp_order_and_worker_startup_split_are_validated() -> None:
    clock = MutableClock()
    with pytest.raises(ValueError, match="monotonic"):
        PodCapacityLifecycleEvidence(
            reservation_at=clock(),
            provider_create_accepted_at=clock() + timedelta(seconds=2),
            first_running_observed_at=clock() + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="both create-accepted"):
        PodCapacityLifecycleEvidence(
            reservation_at=clock(),
            provider_create_accepted_at=clock(),
            provider_adopted_at=clock(),
        )
