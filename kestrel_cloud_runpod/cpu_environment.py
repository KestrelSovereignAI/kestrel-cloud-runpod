"""Read-only CPU acquisition prerequisites, not an admitted visual provider.

The admitting host must own grants, accepted quotes, request/lease custody,
unknown-create reconciliation and teardown. No method in this module creates a
Pod or asserts that a CPU allocation provides a secure browser transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from types import MappingProxyType
from typing import Callable, Literal, Protocol, Self

from pydantic import AwareDatetime, Field, model_validator

from kestrel_sdk.computer_use import EnvironmentRequest
from kestrel_sdk.computer_use.models import ImageDigest, PinnedImage
from kestrel_sdk.resources import (
    Identifier,
    PositiveInt,
    ResourceContract,
    ResourceUnsupported,
    USD,
)
from .billing import billing_identity_amount, exact_decimal_sum
from .cpu_catalog import CpuCatalogSnapshot, CpuPodCatalogMatch
from .models import ComputeProduct, CpuPodCreateRequest


def _aware(at: datetime) -> None:
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError("CPU planning requires a trusted aware current time")


def _utc_iso(at: datetime) -> str:
    _aware(at)
    try:
        return at.astimezone(UTC).isoformat()
    except OverflowError:
        raise ValueError("CPU timestamp is outside the supported UTC range") from None


def _digest(kind: str, payload: dict) -> str:
    envelope = {"kind": kind, "version": 1, "payload": payload}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_catalog_age(value: timedelta) -> None:
    if not isinstance(value, timedelta) or not timedelta(0) < value <= timedelta(minutes=1):
        raise ValueError("CPU catalog freshness must be positive and at most one minute")


class CpuEnvironmentProfile(ResourceContract):
    """Exact requested shape/image; datacenter placement remains a preference."""

    provider_id: Identifier
    profile_id: Identifier
    image: PinnedImage
    cpu_id: Identifier
    vcpu_count: PositiveInt
    data_center_id: Identifier
    disk_gb: PositiveInt
    max_lifetime_seconds: PositiveInt = Field(le=7 * 86400)
    cleanup_grace_seconds: PositiveInt = Field(le=86400)

    @model_validator(mode="after")
    def cpu_shape(self) -> Self:
        if self.vcpu_count < 2 or self.vcpu_count & (self.vcpu_count - 1):
            raise ValueError("CPU profile requires power-of-two vCPU count >= 2")
        return self

    def require_request(self, request: EnvironmentRequest, *, at: datetime) -> None:
        _aware(at)
        if (request.provider_id, request.profile_id, request.image, request.region) != (
            self.provider_id,
            self.profile_id,
            self.image,
            self.data_center_id,
        ):
            raise ResourceUnsupported("CPU request differs from the exact configured profile")
        if request.location != "remote" or request.surface != "browser":
            raise ResourceUnsupported("CPU planning requires an explicit remote browser request")
        if not request.required_capabilities <= {
            "screenshot",
            "click",
            "type",
            "key",
            "scroll",
            "wait",
            "navigate",
            "takeover",
        }:
            raise ResourceUnsupported(
                "CPU planning has no file/exec/clipboard/DOM/accessibility conformance"
            )
        if not request.requested_at <= at < request.expires_at:
            raise ResourceUnsupported("CPU request is outside its fixed lifetime")
        if request.expires_at - request.requested_at > timedelta(seconds=self.max_lifetime_seconds):
            raise ResourceUnsupported("CPU request exceeds the configured lifetime")
        if at >= request.expires_at - timedelta(seconds=request.ready_timeout_seconds):
            raise ResourceUnsupported("CPU request has insufficient remaining readiness time")

    def digest(self) -> str:
        return _digest("runpod.cpu-environment-profile", self.model_dump(mode="json"))

    def _pod_request(self, request: EnvironmentRequest) -> CpuPodCreateRequest:
        """Internal construction after plan admission; never paid authority."""
        return CpuPodCreateRequest(
            name="kestrel-cpu-" + request.digest()[:32],
            image=self.image,
            cpu_id=self.cpu_id,
            vcpu_count=self.vcpu_count,
            data_center_ids=(self.data_center_id,),
            disk_gb=self.disk_gb,
            env=MappingProxyType({}),
        )


class CpuNonComputeCostPolicy(ResourceContract):
    """Explicit operator-supplied disk/transfer/cleanup estimate provenance.

    The CPU API's compute price does not cover these costs. Both values must be
    provided, even for an explicitly justified zero. This is a finite estimate,
    not a provider-enforced billing cap or evidence that cleanup completed.
    No network volumes/persistent storage are supported by this planner.
    """

    policy_id: Identifier
    provider_id: Identifier
    profile_id: Identifier
    profile_sha256: ImageDigest
    disk_gb: PositiveInt
    data_center_id: Identifier
    covered_components: frozenset[Literal["container_disk", "transfer", "cleanup"]]
    observed_at: AwareDatetime
    expires_at: AwareDatetime
    disk_transfer_cleanup_hourly_usd: USD = Field(max_digits=36, decimal_places=12)
    disk_transfer_cleanup_fixed_usd: USD = Field(max_digits=36, decimal_places=12)

    @model_validator(mode="after")
    def coverage_window(self) -> Self:
        _utc_iso(self.observed_at)
        _utc_iso(self.expires_at)
        if self.expires_at <= self.observed_at:
            raise ValueError("CPU cost policy expiry must follow observation")
        if self.covered_components != {"container_disk", "transfer", "cleanup"}:
            raise ValueError("CPU cost policy must explicitly cover disk, transfer and cleanup")
        return self

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        payload["covered_components"] = sorted(self.covered_components)
        for name in ("disk_transfer_cleanup_hourly_usd", "disk_transfer_cleanup_fixed_usd"):
            payload[name] = billing_identity_amount(getattr(self, name))
        for name in ("observed_at", "expires_at"):
            payload[name] = _utc_iso(getattr(self, name))
        return _digest("runpod.cpu-noncompute-policy", payload)

    def require_coverage(
        self,
        request: EnvironmentRequest,
        profile: CpuEnvironmentProfile,
        *,
        at: datetime,
    ) -> None:
        _aware(at)
        if (
            self.provider_id,
            self.profile_id,
            self.profile_sha256,
            self.disk_gb,
            self.data_center_id,
        ) != (
            profile.provider_id,
            profile.profile_id,
            profile.digest(),
            profile.disk_gb,
            profile.data_center_id,
        ) or self.covered_components != {"container_disk", "transfer", "cleanup"}:
            raise ResourceUnsupported("CPU noncompute policy prices another profile/disk/region")
        if not self.observed_at <= at < self.expires_at:
            raise ResourceUnsupported("CPU noncompute cost policy is not current")
        if self.expires_at - request.expires_at < timedelta(seconds=profile.cleanup_grace_seconds):
            raise ResourceUnsupported("CPU cost policy does not cover lifetime and cleanup grace")


def _duration_cost(rate: Decimal, duration: timedelta) -> Decimal:
    """Conservative 12-decimal USD ceiling, independent of ambient precision."""
    micros = (duration.days * 86400 + duration.seconds) * 1_000_000 + duration.microseconds
    numerator, denominator = rate.as_integer_ratio()
    denominator *= 3_600_000_000
    scaled, remainder = divmod(numerator * micros * 10**12, denominator)
    # Constructing from a tuple avoids ambient Decimal multiplication/scaleb.
    return Decimal((0, tuple(map(int, str(scaled + bool(remainder)))), -12))


class CpuCatalogReader(Protocol):
    def list_cpus(self, *, product: ComputeProduct, vcpu_count: int) -> CpuCatalogSnapshot: ...


@dataclass(frozen=True)
class CpuEnvironmentAcquisitionPlan:
    """Immutable original planning facts; not a quote acceptance or reservation."""

    request: EnvironmentRequest
    profile: CpuEnvironmentProfile
    noncompute: CpuNonComputeCostPolicy
    catalog: CpuPodCatalogMatch
    planned_at: datetime
    expires_at: datetime
    catalog_max_age: timedelta = timedelta(seconds=30)

    def __post_init__(self):
        # model_copy deliberately bypasses Pydantic validation. Retained plan
        # facts must satisfy the original typed contracts before any arithmetic.
        object.__setattr__(self, "request", EnvironmentRequest.model_validate(self.request))
        object.__setattr__(self, "profile", CpuEnvironmentProfile.model_validate(self.profile))
        object.__setattr__(
            self, "noncompute", CpuNonComputeCostPolicy.model_validate(self.noncompute)
        )
        _aware(self.planned_at)
        _aware(self.expires_at)
        _aware(self.catalog.observed_at)
        _require_catalog_age(self.catalog_max_age)
        self.profile.require_request(self.request, at=self.planned_at)
        self.noncompute.require_coverage(self.request, self.profile, at=self.planned_at)
        if (self.catalog.cpu_id, self.catalog.vcpu_count, self.catalog.data_center_ids) != (
            self.profile.cpu_id,
            self.profile.vcpu_count,
            (self.profile.data_center_id,),
        ):
            raise ValueError("CPU catalog match differs from the exact profile")
        if (
            not self.catalog.ram_gb.is_finite()
            or self.catalog.ram_gb <= 0
            or not self.catalog.compute_cost_per_hr.is_finite()
            or self.catalog.compute_cost_per_hr <= 0
        ):
            raise ValueError("CPU planning requires positive finite known resource/price facts")
        if (
            not self.catalog.observed_at
            <= self.planned_at
            < self.expires_at
            <= min(
                self.request.expires_at,
                self.noncompute.expires_at,
                self.catalog.observed_at + self.catalog_max_age,
                self.request.expires_at - timedelta(seconds=self.request.ready_timeout_seconds),
            )
        ):
            raise ValueError("CPU plan lifetime exceeds current catalog/request/policy facts")

    @property
    def all_in_hourly_usd(self) -> Decimal:
        return exact_decimal_sum(
            (self.catalog.compute_cost_per_hr, self.noncompute.disk_transfer_cleanup_hourly_usd)
        )

    @property
    def estimated_total_usd(self) -> Decimal:
        # Price the entire admitted lifetime, including elapsed provisioning,
        # and a configured cleanup grace. A late create never widens TTL.
        duration = (
            self.request.expires_at
            - self.request.requested_at
            + timedelta(seconds=self.profile.cleanup_grace_seconds)
        )
        return exact_decimal_sum(
            (
                _duration_cost(self.all_in_hourly_usd, duration),
                self.noncompute.disk_transfer_cleanup_fixed_usd,
            )
        )

    def digest(self) -> str:
        payload = dict(
            request_sha256=self.request.digest(),
            profile_sha256=self.profile.digest(),
            noncompute_sha256=self.noncompute.digest(),
            cpu_id=self.catalog.cpu_id,
            vcpu_count=self.catalog.vcpu_count,
            data_center_ids=self.catalog.data_center_ids,
            ram_gb=billing_identity_amount(self.catalog.ram_gb),
            compute_hourly_usd=billing_identity_amount(self.catalog.compute_cost_per_hr),
            all_in_hourly_usd=billing_identity_amount(self.all_in_hourly_usd),
            estimated_total_usd=billing_identity_amount(self.estimated_total_usd),
            planned_at=_utc_iso(self.planned_at),
            expires_at=_utc_iso(self.expires_at),
            catalog_observed_at=_utc_iso(self.catalog.observed_at),
            catalog_max_age_us=self.catalog_max_age // timedelta(microseconds=1),
        )
        return _digest("runpod.cpu-environment-plan", payload)

    def pod_request(
        self,
        request: EnvironmentRequest,
        profile: CpuEnvironmentProfile,
        noncompute: CpuNonComputeCostPolicy,
        *,
        at: datetime,
    ) -> CpuPodCreateRequest:
        """Revalidate originals immediately before a separate admitted create.

        The caller must persist the accepted quote and original create intent,
        hold current grants/privacy, and reconcile any unknown create by the
        original request. This plan owns no mutation or second custody store.
        """
        if (
            request.digest() != self.request.digest()
            or profile.digest() != self.profile.digest()
            or noncompute.digest() != self.noncompute.digest()
        ):
            raise ResourceUnsupported(
                "CPU plan differs from the original request/profile/cost policy"
            )
        self.profile.require_request(self.request, at=at)
        self.noncompute.require_coverage(self.request, self.profile, at=at)
        if not self.planned_at <= at < self.expires_at:
            raise ResourceUnsupported("CPU acquisition plan expired or is from the future")
        if (
            self.all_in_hourly_usd > self.request.max_hourly_cost_usd
            or self.estimated_total_usd > self.request.max_total_cost_usd
        ):
            raise ResourceUnsupported("CPU all-in estimate exceeds the original cost ceilings")
        return self.profile._pod_request(self.request)


class CpuEnvironmentPlanner:
    """Uses the production v2 catalog client without exposing a create method."""

    def __init__(
        self,
        client: CpuCatalogReader,
        *,
        catalog_max_age: timedelta = timedelta(seconds=30),
    ):
        _require_catalog_age(catalog_max_age)
        self.client, self.catalog_max_age = client, catalog_max_age

    def plan(
        self,
        request: EnvironmentRequest,
        profile: CpuEnvironmentProfile,
        noncompute: CpuNonComputeCostPolicy,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> CpuEnvironmentAcquisitionPlan:
        # Refuse invalid trusted-host copies before even issuing a catalog read.
        request = EnvironmentRequest.model_validate(request)
        profile = CpuEnvironmentProfile.model_validate(profile)
        noncompute = CpuNonComputeCostPolicy.model_validate(noncompute)
        # Production list_cpus records observation before its real HTTP request;
        # the second clock sample includes provider latency in freshness checks.
        initial = clock()
        profile.require_request(request, at=initial)
        noncompute.require_coverage(request, profile, at=initial)
        snapshot = self.client.list_cpus(
            product=ComputeProduct.POD,
            vcpu_count=profile.vcpu_count,
        )
        now = clock()
        profile.require_request(request, at=now)
        noncompute.require_coverage(request, profile, at=now)
        if (
            not isinstance(snapshot.observed_at, datetime)
            or snapshot.observed_at.utcoffset() is None
        ):
            raise ResourceUnsupported("CPU catalog observation must be timezone-aware")
        if not initial <= snapshot.observed_at <= now:
            raise ResourceUnsupported("CPU catalog clock does not cover the current API request")
        if now >= snapshot.observed_at + self.catalog_max_age:
            raise ResourceUnsupported("CPU catalog freshness expired during planning")
        match = snapshot.match_pod(
            profile._pod_request(request),
            now=now,
            max_age=self.catalog_max_age,
        )
        if match.compute_cost_per_hr <= 0:
            raise ResourceUnsupported("CPU catalog requires a positive compute price")
        plan = CpuEnvironmentAcquisitionPlan(
            request,
            profile,
            noncompute,
            match,
            now,
            min(
                request.expires_at - timedelta(seconds=request.ready_timeout_seconds),
                noncompute.expires_at,
                snapshot.observed_at + self.catalog_max_age,
            ),
            self.catalog_max_age,
        )
        plan.pod_request(request, profile, noncompute, at=now)
        return plan
