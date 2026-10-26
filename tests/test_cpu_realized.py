"""Exact original CPU resource observation; production v2 parser, no paid calls."""

from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import PodResource
from kestrel_cloud_runpod.cpu_realized import (
    CpuPodMismatch,
    CpuPodReadinessUnknown,
    inspect_cpu_readiness,
    reconcile_cpu_candidates,
)
from test_cpu_environment import NOW, make_plan
from test_cpu_environment import stable_catalog_time as stable_catalog_time


def body(plan, **changes):
    return dict(
        id="owned-cpu",
        name="kestrel-cpu-" + plan.request.digest()[:32],
        image=plan.profile.image,
        cloud="SECURE",
        disk=20,
        dataCenterId="US-TX-3",
        cpu={"id": "cpu5c", "vcpuCount": 4, "memory": 10},
        gpu=None,
        globalNetworking=False,
        status="RUNNING",
        cost=0.16,
        startedAt=(NOW + timedelta(seconds=1)).isoformat(),
        **changes,
    )


def inspect(plan, raw=None, at=None):
    return inspect_cpu_readiness(
        PodResource.from_dict(body(plan) if raw is None else raw),
        plan,
        expected_pod_id="owned-cpu",
        at=at or NOW + timedelta(seconds=2),
    )


def test_real_control_plane_read_validates_exact_original_running_cpu_and_compute_rate():
    plan = make_plan()
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET" and request.url.path == "/v2/pods/owned-cpu"
        return httpx.Response(200, json=body(plan))

    with closing(
        RunpodControlPlaneClient(api_key="synthetic", http_transport=httpx.MockTransport(handle))
    ) as client:
        observed = inspect_cpu_readiness(
            client.get_pod("owned-cpu"),
            plan,
            expected_pod_id="owned-cpu",
            at=NOW + timedelta(seconds=2),
        )
    assert len(calls) == 1 and observed.phase == "running"
    assert observed.compute_hourly_usd == plan.catalog.compute_cost_per_hr
    assert observed.request_sha256 == plan.request.digest()
    observed.require_same_incarnation(observed)
    with pytest.raises(CpuPodMismatch) as error:
        replace(observed, started_at=NOW + timedelta(seconds=2)).require_same_incarnation(observed)
    assert error.value.pod_id == "owned-cpu" and error.value.requires_teardown


@pytest.mark.parametrize(
    "change",
    [
        {"name": "another"},
        {"image": "worker@sha256:" + "b" * 64},
        {"cloud": "COMMUNITY"},
        {"disk": 21},
        {"disk": True},
        {"dataCenterId": "EU-RO-1"},
        {"cpu": {"id": "cpu4c", "vcpuCount": 4, "memory": 10}},
        {"cpu": {"id": "cpu5c", "vcpuCount": 8, "memory": 10}},
        {"cpu": {"id": "cpu5c", "vcpuCount": 4, "memory": 11}},
        {"gpu": {"id": "GPU", "count": 1}},
        {"globalNetworking": True},
        {"cost": 0.17},  # Still below all-in0.18, but above accepted COMPUTE0.16.
        {"status": "EXITED", "cost": 0.0},
        {"status": "TERMINATED", "cost": 0.0},
    ],
)
def test_realized_mismatch_preserves_original_id_and_requires_teardown(change):
    plan = make_plan()
    raw = body(plan)
    raw.update(change)
    with pytest.raises(CpuPodMismatch) as error:
        inspect(plan, raw)
    assert error.value.pod_id == "owned-cpu" and error.value.requires_teardown


@pytest.mark.parametrize(
    "change",
    [
        {"dataCenterId": None},
        {"cpu": None},
        {"cost": None},
        {"startedAt": None},
        {"startedAt": "2026-10-26T12:00:01"},
        {"startedAt": (NOW - timedelta(seconds=1)).isoformat()},
    ],
)
def test_unknown_running_facts_never_allow_private_connection_or_discard_custody(change):
    plan = make_plan()
    raw = body(plan)
    raw.update(change)
    with pytest.raises(CpuPodReadinessUnknown) as error:
        inspect(plan, raw)
    assert error.value.pod_id == "owned-cpu"


@pytest.mark.parametrize("status", ["PROVISIONING", "STARTING"])
def test_zero_pending_rate_never_becomes_running_and_deadline_never_renews(status):
    plan = make_plan()
    raw = body(plan)
    raw.update(status=status, cost=0.0, startedAt=None)
    observed = inspect(plan, raw)
    assert observed.phase == "pending" and observed.compute_hourly_usd is None
    for at in (NOW + timedelta(seconds=60), plan.request.expires_at):
        with pytest.raises(CpuPodMismatch):
            inspect(plan, raw, at=at)


def test_candidate_absence_or_name_match_is_not_proof_of_create_failure_or_ownership():
    plan = make_plan()
    assert reconcile_cpu_candidates((), plan) is None
    raw = body(plan)
    pod = PodResource.from_dict(raw)
    assert reconcile_cpu_candidates((pod,), plan) is pod
    with pytest.raises(ResourceOutcomeUnknown):
        reconcile_cpu_candidates((pod, replace(pod, id="another-cpu")), plan)
    raw["id"] = "another-cpu"
    with pytest.raises(CpuPodReadinessUnknown) as error:
        inspect(plan, raw)
    assert error.value.pod_id == "owned-cpu"
    with pytest.raises(ResourceUnsupported):
        inspect(plan, at=NOW.replace(tzinfo=None))
