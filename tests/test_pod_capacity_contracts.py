"""Generic quote, identity, cost, environment, and persistence contracts."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from pod_capacity_test_support import (
    IMAGE,
    PARAMETERS_SHA,
    TOKEN,
    MutableClock,
    quote,
    request,
)

from kestrel_cloud_runpod.pod_capacity_contracts import (
    PodBillingReceipt,
    PodCapacitySpec,
    PodRealizedPlacement,
    attempt_environment_sha256,
    pod_cost_usd,
)


def test_request_binds_exact_parameter_digest_quote_cost_and_image() -> None:
    clock = MutableClock()
    original = request(clock)

    with pytest.raises(ValueError, match="parameters"):
        replace(original, parameters_sha256="f" * 64)
    with pytest.raises(ValueError, match="must equal"):
        replace(original, accepted_max_cost_usd=Decimal("0.07"))
    with pytest.raises(ValueError, match="pinned by digest"):
        replace(original, image_reference="ghcr.io/kestrel/catalog-worker:latest")

    assert original.parameters_sha256 == PARAMETERS_SHA
    assert original.image_reference == IMAGE
    assert original.resource_name.startswith("kestrel-cap-")


@pytest.mark.parametrize(
    "key",
    [
        "DATABASE_URL",
        "CATALOG_DATABASE_URL",
        "RUNPOD_API_KEY",
        "RUNPOD_CONTROL_PLANE_API_KEY",
        "RUNPOD_SERVERLESS_API_KEY",
        "CATALOG_POD_BEARER_TOKEN",
        "CONTAINER_DIGEST",
    ],
)
def test_request_refuses_control_database_and_service_owned_environment(
    key: str,
) -> None:
    clock = MutableClock()
    with pytest.raises(ValueError, match=key):
        request(clock, attempt_environment={key: "must-not-reach-worker"})


def test_persisted_spec_contains_only_secret_identity_digest_and_expiry() -> None:
    clock = MutableClock()
    original = request(clock)
    spec = PodCapacitySpec(
        request=original,
        capability_secret_id="secret:catalog-capability-0001",
        capability_token_sha256="a" * 64,
        capability_expires_at=original.bearer_expires_at,
        attempt_environment_sha256=attempt_environment_sha256(
            original.attempt_environment
        ),
    )

    serialized = spec.to_dict()
    assert TOKEN not in str(serialized)
    assert "private/model" not in str(serialized)
    assert serialized["request"]["attempt_environment"] == {
        "MODEL_REPOSITORY": "[REDACTED]"
    }
    assert PodCapacitySpec.from_dict(serialized).to_dict() == serialized


def test_changed_attempt_environment_changes_durable_identity() -> None:
    clock = MutableClock()
    first = request(clock, attempt_environment={"MODEL_REPOSITORY": "model/a"})
    second = request(clock, attempt_environment={"MODEL_REPOSITORY": "model/b"})
    assert first.resource_name == second.resource_name
    assert attempt_environment_sha256(first.attempt_environment) != (
        attempt_environment_sha256(second.attempt_environment)
    )


def test_billing_receipt_matches_private_terminal_contract() -> None:
    clock = MutableClock()
    billed_from = clock() - timedelta(seconds=1, microseconds=750_000)
    receipt = PodBillingReceipt(
        provider_billing_id="runpod-billing:" + "a" * 64,
        provider_pod_id="pod-catalog-1",
        billed_from=billed_from,
        billed_until=clock(),
        billed_seconds=1,
        hourly_price_usd=Decimal("0.40"),
        actual_cost_usd=Decimal("0.001"),
        reconciled_at=clock(),
    )

    assert receipt.billed_seconds == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"billed_seconds": True}, "interval"),
        ({"billed_seconds": 1.0}, "interval"),
        ({"billed_seconds": 2}, "truncated"),
        (
            {"reconciled_at": MutableClock()() - timedelta(seconds=1)},
            "before its interval ends",
        ),
        ({"provider_pod_id": None}, "provider Pod identity"),
        (
            {"provider_pod_id": None, "actual_cost_usd": Decimal(0)},
            "before billable capacity",
        ),
    ],
)
def test_billing_receipt_rejects_non_authoritative_dimensions(
    changes: dict[str, object], message: str
) -> None:
    clock = MutableClock()
    receipt = PodBillingReceipt(
        provider_billing_id="runpod-billing:" + "b" * 64,
        provider_pod_id="pod-catalog-1",
        billed_from=clock() - timedelta(seconds=1),
        billed_until=clock(),
        billed_seconds=1,
        hourly_price_usd=Decimal("0.40"),
        actual_cost_usd=Decimal("0.001"),
        reconciled_at=clock(),
    )

    with pytest.raises(ValueError, match=message):
        replace(receipt, **changes)


def test_zero_cost_receipt_requires_no_billed_interval_without_a_pod() -> None:
    clock = MutableClock()
    receipt = PodBillingReceipt(
        provider_billing_id="runpod-no-pod:" + "c" * 64,
        provider_pod_id=None,
        billed_from=clock(),
        billed_until=clock(),
        billed_seconds=0,
        hourly_price_usd=Decimal("0.40"),
        actual_cost_usd=Decimal(0),
        reconciled_at=clock(),
    )

    assert receipt.provider_pod_id is None


def test_multi_gpu_quote_prices_and_authorizes_the_same_pod():
    """Cost is derived from placement.gpu_count; authorization reads
    constraints.gpu_count, and the Pod is created from constraints. If the two
    can diverge the quote authorizes a different Pod than it priced, and the
    divergence survives a durable round trip via the stored placement blob.
    """
    clock = MutableClock()
    quad = quote(clock, gpu_count=4)

    assert quad.constraints.gpu_count == 4
    assert quad.placement.gpu_count == 4
    # 0.40/GPU x 4 x 600s = 0.266667, not 0.066667.
    assert quad.cost_ceiling_usd == Decimal("0.266667")
    assert quad.cost_ceiling_usd == pod_cost_usd(Decimal("0.4"), 600, 4)

    with pytest.raises(ValueError, match="GPU count is inconsistent"):
        replace(quad, placement=replace(quad.placement, gpu_count=1))


def test_realized_multi_gpu_pod_is_accepted_against_its_own_quote():
    """Pod.cost is the WHOLE Pod's burn; the quote rate is per-GPU.

    Comparing them directly rejected every conforming multi-GPU Pod - and in
    the ambiguous-create recovery path that left a running, billing Pod
    unadoptable and therefore never terminated.
    """
    clock = MutableClock()
    quad = quote(clock, gpu_count=4)
    realized = PodRealizedPlacement(
        provider_pod_id="pod-catalog-1",
        gpu_type_id=quad.gpu_type_id,
        gpu_display_name=quad.gpu_display_name,
        gpu_count=4,
        cloud=quad.constraints.cloud,
        data_center_id="US-TX-3",
        hourly_rate_usd=Decimal("1.6"),  # 0.40/GPU x 4
        observed_at=clock(),
    )

    realized.validate_against(quad)

    # A whole-Pod rate above the accepted whole-Pod rate is still rejected.
    with pytest.raises(ValueError):
        replace(realized, hourly_rate_usd=Decimal("1.61")).validate_against(quad)


def test_terminated_pod_reporting_zero_cost_is_adoptable():
    """The v2 spec documents cost 0.0 for EXITED/TERMINATED pods."""
    clock = MutableClock()
    single = quote(clock)
    realized = PodRealizedPlacement(
        provider_pod_id="pod-catalog-1",
        gpu_type_id=single.gpu_type_id,
        gpu_display_name=single.gpu_display_name,
        gpu_count=1,
        cloud=single.constraints.cloud,
        data_center_id="US-TX-3",
        hourly_rate_usd=Decimal("0"),
        observed_at=clock(),
    )

    realized.validate_against(single)
    assert realized.hourly_rate_usd == Decimal("0")
