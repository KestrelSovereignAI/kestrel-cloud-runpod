"""Exact original CPU resource observation; production v2 parser, no paid calls."""

from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

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
from scripts.check_runpod_openapi import load_openapi
from test_cpu_environment import NOW, make_plan
from test_cpu_environment import stable_catalog_time as stable_catalog_time


def body(plan, **changes):
    result = dict(
        id="owned-cpu",
        name="kestrel-cpu-" + plan.request.digest()[:32],
        image=plan.profile.image,
        cloud="SECURE",
        disk=20,
        dataCenterId="US-TX-3",
        cpu={"id": "cpu5c", "vcpuCount": 4, "memory": 10},
        globalNetworking={"enabled": False},
        actions=["stop", "terminate"], args="", cmd=[], entrypoint=[],
        ports=[], env={}, mounts={}, registry=None, locked=False,
        cudaVersion=None, ssh={"proxy": None, "direct": None}, template=None,
        runtime={}, createdAt=NOW.isoformat(),
        status="RUNNING",
        cost=0.16,
        startedAt=(NOW + timedelta(seconds=1)).isoformat(),
    )
    # Validate the nominal complete fixture before intentional malformed/missing
    # response tests change it; RUNNING and pending successes use the pin too.
    pod_schema().validate(result)
    return result | changes


@lru_cache
def pod_schema():
    document = load_openapi(
        (Path(__file__).resolve().parents[1] / "vendor/runpod-v2-openapi.yaml").read_text()
    )
    return Draft202012Validator({**document, "$ref": "#/components/schemas/Pod"})


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
        {"globalNetworking": True},  # Request-side boolean is invalid readback.
        {"globalNetworking": {"enabled": True}},
        {"globalNetworking": {"enabled": False, "ip": "10.0.0.1"}},
        {"globalNetworking": {"enabled": False, "internalDns": "owned.internal"}},
        {"ports": ["443/http"]}, {"env": {"EXTRA": "value"}},
        {"mounts": {"network": [{"volumeId": "extra", "path": "/extra"}]}},
        {"args": "extra"}, {"cmd": ["extra"]}, {"entrypoint": ["extra"]},
        {"registry": "extra"}, {"locked": True},
        {"cluster": {"id": "extra", "rank": 0}}, {"template": "extra"},
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
        {"cost": 0},
        {"cost": -0.0},
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
    pod_schema().validate(raw)
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


@pytest.mark.parametrize("status", ["RUNNING", "PROVISIONING", "STARTING"])
@pytest.mark.parametrize("field", [
    "image", "ports", "env", "mounts", "args", "registry", "locked", "template",
    "cpu", "cpu.id", "cpu.vcpuCount", "cpu.memory", "globalNetworking",
    "globalNetworking.enabled",
])
def test_missing_required_facts_are_unknown_instead_of_a_known_mismatch(status, field):
    plan = make_plan()
    raw = body(plan, status=status)
    key, _, nested = field.partition(".")
    if nested:
        raw[key].pop(nested)
    else:
        raw.pop(key)
    with pytest.raises(CpuPodReadinessUnknown):
        inspect(plan, raw)


def test_readiness_clock_before_planning_is_unknown_and_start_cannot_predate_plan():
    times = iter([NOW, NOW + timedelta(seconds=1)])
    plan = make_plan(clock=lambda: next(times))
    with pytest.raises(CpuPodReadinessUnknown):
        inspect(plan, at=NOW)
    with pytest.raises(CpuPodReadinessUnknown):
        inspect(plan, body(plan, startedAt=NOW.isoformat()))
    assert inspect(plan).phase == "running"



def test_known_terminal_state_requires_teardown_even_with_other_unknown_facts():
    plan = make_plan()
    raw = body(plan, status="EXITED")
    raw.pop("cpu")
    with pytest.raises(CpuPodMismatch):
        inspect(plan, raw)
