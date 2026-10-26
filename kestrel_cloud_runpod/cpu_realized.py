"""Runpod CPU placement observations; neither authority nor browser readiness.

Call only for the exact original Pod ID retained by canonical host custody.
These checks never create, adopt, connect, delete or infer physical release.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, ResourceUnsupported

from .cpu_environment import CpuEnvironmentAcquisitionPlan
from .models import PodResource


class CpuPodMismatch(ResourceConflict):
    """Original owned Pod requires teardown; no replacement is authorized."""

    requires_teardown = True

    def __init__(self, pod_id: str):
        self.pod_id = pod_id
        super().__init__("owned CPU Pod contradicts the admitted resource or readiness bounds")


class CpuPodReadinessUnknown(ResourceOutcomeUnknown):
    """Keep original Pod custody; unknown observations grant no data access."""

    def __init__(self, pod_id: str):
        self.pod_id = pod_id
        super().__init__("owned CPU Pod placement or authoritative running rate is unknown")


@dataclass(frozen=True)
class CpuPodObservation:
    pod_id: str
    request_sha256: str
    observed_at: datetime
    phase: str
    started_at: datetime | None = None
    compute_hourly_usd: Decimal | None = None

    def require_same_incarnation(self, original: CpuPodObservation) -> None:
        if (
            (self.pod_id, self.request_sha256, self.started_at)
            != (original.pod_id, original.request_sha256, original.started_at)
            or self.phase != "running"
            or original.phase != "running"
        ):
            raise CpuPodMismatch(original.pod_id)


def inspect_cpu_readiness(
    pod: PodResource,
    plan: CpuEnvironmentAcquisitionPlan,
    *,
    expected_pod_id: str,
    at: datetime,
) -> CpuPodObservation:
    """Validate first readiness facts inside the original fixed ready window.

    A running result proves API placement/rate facts only. TLS peer identity,
    actual sandbox/egress, worker admission and current grants remain mandatory.
    Runpod `startedAt` is an incarnation hint, not cryptographic attestation.
    """
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ResourceUnsupported("CPU readiness requires an aware trusted time")
    if not isinstance(expected_pod_id, str) or not expected_pod_id:
        raise ResourceUnsupported("CPU readiness requires original Pod custody")
    raw = pod.raw
    if (
        not isinstance(raw, Mapping)
        or pod.id != expected_pod_id
        or raw.get("id") != expected_pod_id
    ):
        raise CpuPodReadinessUnknown(expected_pod_id)
    deadline = min(
        plan.request.expires_at,
        plan.planned_at + timedelta(seconds=plan.request.ready_timeout_seconds),
    )
    if at < plan.planned_at:
        raise CpuPodReadinessUnknown(expected_pod_id)
    if at >= deadline:
        raise CpuPodMismatch(expected_pod_id)
    expected = {
        "name": "kestrel-cpu-" + plan.request.digest()[:32],
        "image": plan.profile.image,
        "cloud": "SECURE",
        "disk": plan.profile.disk_gb,
        "dataCenterId": plan.profile.data_center_id,
    }
    # Any known contradiction requires cleanup even while provisioning. Missing
    # scheduler facts may remain pending only within the original deadline.
    for name, value in expected.items():
        observed = raw.get(name)
        if observed is not None and (
            observed != value or (name == "disk" and type(observed) is not int)
        ):
            raise CpuPodMismatch(expected_pod_id)
    # Known changes to the empty create intent are contradictions. Missing
    # required readback fields stay unknown, including during provisioning.
    empty = {
        "ports": [], "env": {}, "mounts": {}, "args": "", "registry": None,
        "locked": False, "template": None,
    }
    unknown = any(raw.get(name) is None for name in expected)
    for name, value in empty.items():
        if name not in raw:
            unknown = True
        elif raw[name] != value or type(raw[name]) is not type(value):
            raise CpuPodMismatch(expected_pod_id)
    for name in ("cmd", "entrypoint"):
        if name in raw and (type(raw[name]) is not list or raw[name] != []):
            raise CpuPodMismatch(expected_pod_id)
    # These optional fields are omitted, never null, on schema-valid CPU Pods.
    if "gpu" in raw or "cluster" in raw:
        raise CpuPodMismatch(expected_pod_id)
    networking = raw.get("globalNetworking")
    if networking is None:
        unknown = True
    elif not isinstance(networking, Mapping):
        raise CpuPodMismatch(expected_pod_id)
    elif "ip" in networking or "internalDns" in networking:
        raise CpuPodMismatch(expected_pod_id)
    elif "enabled" not in networking:
        unknown = True
    elif networking["enabled"] is not False:
        raise CpuPodMismatch(expected_pod_id)
    cpu = raw.get("cpu")
    if cpu is None:
        unknown = True
    elif not isinstance(cpu, Mapping):
        raise CpuPodMismatch(expected_pod_id)
    else:
        for name, value in {
            "id": plan.profile.cpu_id, "vcpuCount": plan.profile.vcpu_count,
            "memory": int(plan.catalog.ram_gb),
        }.items():
            if name not in cpu or cpu[name] is None:
                unknown = True
            elif cpu[name] != value or type(cpu[name]) is not type(value):
                raise CpuPodMismatch(expected_pod_id)
    status = raw.get("status")
    if status is not None and status not in ("RUNNING", "PROVISIONING", "STARTING"):
        raise CpuPodMismatch(expected_pod_id)
    if unknown or status is None:
        raise CpuPodReadinessUnknown(expected_pod_id)
    if status in {"PROVISIONING", "STARTING"}:
        return CpuPodObservation(expected_pod_id, plan.request.digest(), at, "pending")
    cost = raw.get("cost")
    if type(cost) not in (int, float):
        raise CpuPodReadinessUnknown(expected_pod_id)
    rate = Decimal(str(cost))
    if not rate.is_finite() or rate <= 0:
        raise CpuPodReadinessUnknown(expected_pod_id)
    if rate > plan.catalog.compute_cost_per_hr:
        raise CpuPodMismatch(expected_pod_id)
    try:
        started = datetime.fromisoformat(raw["startedAt"])
        if started.utcoffset() is None or not plan.planned_at <= started <= at:
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise CpuPodReadinessUnknown(expected_pod_id) from None
    return CpuPodObservation(expected_pod_id, plan.request.digest(), at, "running", started, rate)


def reconcile_cpu_candidates(
    pods: tuple[PodResource, ...],
    plan: CpuEnvironmentAcquisitionPlan,
) -> PodResource | None:
    """Find one exact-name candidate without adopting or claiming absent creates.

    None means unresolved: an accepted POST may not yet be visible. A candidate
    must still pass immutable identity, private worker proof and canonical
    custody admission; the deterministic name is never an ownership credential.
    """
    name = "kestrel-cpu-" + plan.request.digest()[:32]
    matches = [pod for pod in pods if pod.name == name]
    if len(matches) > 1:
        raise ResourceOutcomeUnknown("multiple CPU Pods match the original create intent")
    return matches[0] if matches else None
