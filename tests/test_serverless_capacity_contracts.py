"""Contract tests for finite Serverless quotes and billing evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from serverless_capacity_test_support import (
    PARAMETERS_SHA256,
    MutableClock,
    ambiguous_window,
    attempt,
    constraints,
    endpoint_spec,
    planned_endpoint,
    profile,
    quote,
    request,
)

import kestrel_cloud_runpod
from kestrel_cloud_runpod.models import Availability, CloudType, RunPodManagerError
from kestrel_cloud_runpod.serverless_capacity_contracts import (
    RUNPOD_V2_OPENAPI_SHA256,
    RUNPOD_V2_OPENAPI_SOURCE_URL,
    PlannedServerlessEndpoint,
    ServerlessAmbiguousBillingWindow,
    ServerlessAmbiguousWindowBillingReceipt,
    ServerlessBillingReceipt,
    ServerlessCapacityQuote,
    ServerlessEndpointHourCost,
    ServerlessEndpointSpec,
    serverless_worker_cost_usd,
)


def test_endpoint_spec_round_trip_is_canonical_and_digest_protected() -> None:
    item = endpoint_spec(registry_id="registry-private-01")
    serialized = item.to_dict()

    assert serialized["spec_sha256"] == item.spec_sha256
    assert ServerlessEndpointSpec.from_dict(serialized).to_dict() == serialized
    with pytest.raises(RunPodManagerError, match="digest mismatches"):
        ServerlessEndpointSpec.from_dict({**serialized, "disk_gb": 21})

    planned = planned_endpoint(spec=item)
    assert PlannedServerlessEndpoint.from_dict(planned.to_dict()) == planned
    assert "endpoint_id" not in planned.to_dict()


def test_endpoint_create_shape_is_pinned_to_api_served_openapi() -> None:
    payload = (
        endpoint_spec(registry_id="registry-private-01")
        .create_request(
            "kestrel-selfie-run-0001",
            gpu_pool="BLACKWELL_24",
            data_center_id="US-TX-3",
        )
        .to_payload()
    )

    assert RUNPOD_V2_OPENAPI_SOURCE_URL == "https://api.runpod.io/v2/openapi.json"
    assert RUNPOD_V2_OPENAPI_SHA256 == (
        "0bbdd828569233765e310e773e34586b33a6e38f55afde989ebd670152ed5c13"
    )
    assert payload["type"] == "QUEUE"
    assert payload["workers"] == {"min": 0, "max": 1, "idleTimeout": 10}
    assert payload["scaling"] == {"type": "QUEUE_DELAY", "queueDelay": 4.0}
    assert payload["gpu"] == {"pools": ["BLACKWELL_24"], "count": 1}
    assert payload["dataCenterIds"] == ["US-TX-3"]
    assert payload["networkVolumes"] == []
    assert payload["env"] == {}
    assert payload["ports"] == []
    assert payload["registry"] == "registry-private-01"


def test_public_package_exports_serverless_construction_dependencies() -> None:
    assert kestrel_cloud_runpod.Availability is Availability
    assert kestrel_cloud_runpod.CloudType is CloudType
    assert kestrel_cloud_runpod.serverless_worker_cost_usd is serverless_worker_cost_usd
    assert callable(kestrel_cloud_runpod.serverless_billing_hour_starts)


def test_quote_round_trip_binds_normalized_parameters_and_exact_cost_math() -> None:
    item = quote()
    serialized = item.to_dict()

    assert item.parameters_sha256 == PARAMETERS_SHA256
    assert item.estimated_worker_cost_usd == Decimal("0.019167")
    assert item.maximum_worker_cost_usd == Decimal("0.034500")
    assert item.estimated_cost_usd == Decimal("0.019667")
    assert item.cost_ceiling_usd == Decimal("0.035500")
    assert item.job_execution_timeout_ms == 120_000
    assert item.job_ttl_ms == 300_000
    assert ServerlessCapacityQuote.from_dict(serialized).to_dict() == serialized


def test_ambiguous_window_and_receipt_round_trip_are_content_free() -> None:
    accepted = quote()
    allocation = ambiguous_window(accepted)
    allocation.validate_quote(accepted)
    serialized_window = allocation.to_dict()
    assert (
        ServerlessAmbiguousBillingWindow.from_dict(serialized_window).to_dict()
        == serialized_window
    )
    receipt = ServerlessAmbiguousWindowBillingReceipt(
        schema_version=1,
        contract_version="serverless-capacity-v1",
        provider_billing_id="runpod-serverless-ambiguous-billing:" + "6" * 64,
        provider_quote_id=accepted.provider_quote_id,
        endpoint_profile_sha256=accepted.endpoint_profile_sha256,
        endpoint_id=allocation.endpoint_id,
        attempt_id=allocation.attempt_id,
        exclusive_window_sha256=allocation.exclusive_window_sha256,
        exclusive_billing_hour_starts=allocation.exclusive_billing_hour_starts,
        attempted_at=allocation.attempted_at,
        billable_coverage_until=allocation.billable_coverage_until,
        billing_window_from=allocation.exclusive_billing_hour_starts[0],
        billing_window_until=allocation.exclusive_billing_hour_starts[-1]
        + timedelta(hours=1),
        accepted_cost_ceiling_usd=accepted.cost_ceiling_usd,
        endpoint_hour_costs=(
            ServerlessEndpointHourCost(
                provider_observation_id="runpod-serverless-hour:" + "5" * 64,
                endpoint_id=allocation.endpoint_id,
                utc_hour_start=allocation.exclusive_billing_hour_starts[0],
                utc_hour_end=allocation.exclusive_billing_hour_starts[0]
                + timedelta(hours=1),
                gpu_cost_usd=Decimal("0.047"),
                cpu_cost_usd=Decimal(0),
                disk_cost_usd=Decimal("0.001"),
                fee_cost_usd=Decimal("0.002"),
                actual_cost_usd=Decimal("0.050"),
            ),
        ),
        gpu_cost_usd=Decimal("0.047"),
        cpu_cost_usd=Decimal(0),
        disk_cost_usd=Decimal("0.001"),
        fee_cost_usd=Decimal("0.002"),
        actual_cost_usd=Decimal("0.050"),
        capped_cost_usd=accepted.cost_ceiling_usd,
        operator_loss_usd=Decimal("0.0145"),
        reconciled_at=allocation.exclusive_billing_hour_starts[-1] + timedelta(hours=1),
    )
    serialized_receipt = receipt.to_dict()
    assert (
        ServerlessAmbiguousWindowBillingReceipt.from_dict(serialized_receipt).to_dict()
        == serialized_receipt
    )
    nested_extra = {
        **serialized_receipt,
        "endpoint_hour_costs": [
            {**serialized_receipt["endpoint_hour_costs"][0], "raw": "private"}
        ],
    }
    with pytest.raises(RunPodManagerError, match="unsupported"):
        ServerlessAmbiguousWindowBillingReceipt.from_dict(nested_extra)
    encoded = json.dumps(
        {"window": serialized_window, "receipt": serialized_receipt},
        sort_keys=True,
    ).lower()
    assert not any(
        marker in encoded
        for marker in ("prompt", "response", "signed_url", "image", '"raw"')
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"parameters_sha256": "A" * 64}, "SHA-256"),
        ({"hourly_worker_rate_usd": Decimal("NaN")}, "finite and positive"),
        ({"hourly_worker_rate_usd": Decimal(-1)}, "finite and positive"),
        ({"availability": Availability.NONE}, "unavailable"),
        ({"estimated_billable_seconds": 101}, "inconsistent"),
        ({"cost_ceiling_usd": Decimal("0.03")}, "cost is inconsistent"),
    ],
)
def test_quote_rejects_unsafe_or_inconsistent_dimensions(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(quote(), **changes)


def test_quote_deserialization_rejects_raw_or_content_bearing_fields() -> None:
    serialized = quote().to_dict()
    for field in ("raw", "prompt", "response", "signed_url", "weight"):
        with pytest.raises(RunPodManagerError, match="unsupported"):
            ServerlessCapacityQuote.from_dict({**serialized, field: "private"})

    with pytest.raises(RunPodManagerError, match="decimal string"):
        ServerlessCapacityQuote.from_dict(
            {**serialized, "hourly_worker_rate_usd": 0.69}
        )


def test_quote_serialization_is_content_free() -> None:
    encoded = json.dumps(quote().to_dict(), sort_keys=True).lower()
    forbidden = (
        "http://",
        "https://",
        "api_key",
        "authorization",
        "prompt",
        "response",
        "signed_url",
        "worker_reference",
        "private_runtime_value",
        "weight",
        "capability",
        '"raw"',
    )
    assert not any(marker in encoded for marker in forbidden)


def test_endpoint_spec_requires_safe_scale_to_zero_placement_constraints() -> None:
    with pytest.raises(ValueError, match="single-worker"):
        profile(workers_max=2)
    assert constraints(
        allowed_gpu_pools=("BLACKWELL_24", "ADA_24")
    ).allowed_gpu_pools == ("BLACKWELL_24", "ADA_24")
    with pytest.raises(ValueError, match="allowed endpoint data center"):
        constraints(allowed_data_center_ids=())
    with pytest.raises(ValueError, match="idle_tail_seconds"):
        profile(idle_tail_seconds=3_601)
    with pytest.raises(ValueError, match="at least 0.5"):
        profile(scaling_value=Decimal("0.4"))
    with pytest.raises(ValueError, match="QUEUE_DELAY"):
        profile(scaling_type="REQUEST_COUNT", scaling_value=Decimal(1))
    with pytest.raises(ValueError, match="network volume cost"):
        profile(network_volume_ids=("volume-private-01",))
    with pytest.raises(ValueError, match="too large"):
        constraints(
            max_hourly_worker_rate_usd=Decimal("1e9999")
        ).placement_requirements()


def test_request_requires_bounded_queue_billable_and_non_worker_estimates() -> None:
    with pytest.raises(ValueError, match="queue delay exceeds"):
        request(estimated_queue_delay_seconds=121)
    with pytest.raises(ValueError, match="maximum billable time is inconsistent"):
        request(maximum_billable_seconds=99)
    with pytest.raises(ValueError, match="non-worker cost exceeds"):
        request(
            estimated_non_worker_cost_usd=Decimal("0.002"),
            maximum_non_worker_cost_usd=Decimal("0.001"),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        request(maximum_non_worker_cost_usd=Decimal(0))
    with pytest.raises(ValueError, match="five minutes"):
        request(quote_ttl_seconds=301)
    with pytest.raises(ValueError, match="endpoint timeout"):
        request(
            estimated_execution_seconds=121,
            maximum_execution_seconds=121,
            maximum_worker_start_seconds=169,
            maximum_billable_seconds=300,
            job_execution_timeout_ms=121_000,
        )


def test_serverless_job_policy_accepts_provider_boundaries_and_rejects_overflow() -> (
    None
):
    minimum_profile = profile(execution_timeout_ms=5_000)
    request(
        profile=minimum_profile,
        estimated_execution_seconds=5,
        maximum_execution_seconds=5,
        maximum_billable_seconds=135,
        job_execution_timeout_ms=5_000,
    )
    request(
        estimated_queue_delay_seconds=4,
        estimated_worker_start_seconds=1,
        estimated_execution_seconds=5,
        maximum_queue_delay_seconds=4,
        maximum_worker_start_seconds=1,
        maximum_execution_seconds=5,
        maximum_billable_seconds=16,
        job_ttl_ms=10_000,
    )
    maximum_profile = profile(execution_timeout_ms=604_800_000)
    request(
        profile=maximum_profile,
        job_execution_timeout_ms=604_800_000,
        job_ttl_ms=604_800_000,
    )

    for timeout_ms in (4_999, 604_800_001):
        with pytest.raises(ValueError, match="between 5 seconds and 7 days"):
            profile(execution_timeout_ms=timeout_ms)
        with pytest.raises(ValueError, match="between 5 seconds and 7 days"):
            request(job_execution_timeout_ms=timeout_ms)
    for ttl_ms in (9_999, 604_800_001):
        with pytest.raises(ValueError, match="between 10 seconds and 7 days"):
            request(job_ttl_ms=ttl_ms)

    request(job_ttl_ms=290_000)
    with pytest.raises(ValueError, match="maximum job lifespan"):
        request(job_ttl_ms=289_999)
    with pytest.raises(ValueError, match="maximum job lifespan"):
        request(
            profile=maximum_profile,
            estimated_queue_delay_seconds=1,
            maximum_queue_delay_seconds=1,
            maximum_worker_start_seconds=1,
            estimated_worker_start_seconds=1,
            estimated_execution_seconds=604_799,
            maximum_execution_seconds=604_799,
            maximum_billable_seconds=604_810,
            job_execution_timeout_ms=604_800_000,
            job_ttl_ms=604_800_000,
        )


def test_quote_expiry_and_exact_accepted_ceiling_fail_closed() -> None:
    clock = MutableClock()
    item = quote(clock)
    item.assert_fresh(now=clock(), accepted_cost_ceiling_usd=item.cost_ceiling_usd)

    with pytest.raises(RunPodManagerError, match="does not match"):
        item.assert_fresh(
            now=clock(), accepted_cost_ceiling_usd=item.cost_ceiling_usd + Decimal(1)
        )
    with pytest.raises(RunPodManagerError, match="expired"):
        item.assert_fresh(
            now=item.expires_at, accepted_cost_ceiling_usd=item.cost_ceiling_usd
        )


def test_attempt_binds_exact_endpoint_job_quote_and_authorized_interval() -> None:
    item = quote()
    valid = attempt(item)
    valid.validate_quote(item)

    with pytest.raises(RunPodManagerError, match="endpoint"):
        replace(valid, endpoint_id="endpoint-other").validate_quote(item)
    with pytest.raises(RunPodManagerError, match="quote identity"):
        replace(
            valid, provider_quote_id="runpod-serverless:" + "f" * 64
        ).validate_quote(item)
    with pytest.raises(RunPodManagerError, match="accepted quote interval"):
        replace(valid, submitted_at=item.expires_at).validate_quote(item)
    with pytest.raises(RunPodManagerError, match="maximum interval"):
        replace(
            valid,
            completed_at=valid.submitted_at + timedelta(seconds=301),
        ).validate_quote(item)
    first_hour = valid.exclusive_billing_hour_starts[0]
    with pytest.raises(ValueError, match="exceeds policy bounds"):
        replace(
            valid,
            exclusive_billing_hour_starts=tuple(
                first_hour + timedelta(hours=index) for index in range(1_000)
            ),
        )


def _receipt() -> ServerlessBillingReceipt:
    item = quote()
    billing_attempt = attempt(item)
    return ServerlessBillingReceipt(
        schema_version=1,
        contract_version="serverless-capacity-v1",
        provider_billing_id="runpod-serverless-billing:" + "f" * 64,
        provider_quote_id=item.provider_quote_id,
        endpoint_profile_sha256=item.endpoint_profile_sha256,
        endpoint_id=billing_attempt.endpoint_id,
        job_id=billing_attempt.job_id,
        attempt_id=billing_attempt.attempt_id,
        exclusive_window_sha256=billing_attempt.exclusive_window_sha256,
        exclusive_billing_hour_starts=(billing_attempt.exclusive_billing_hour_starts),
        attempt_started_at=billing_attempt.submitted_at,
        attempt_completed_at=billing_attempt.completed_at,
        billable_coverage_until=billing_attempt.completed_at
        + timedelta(seconds=item.idle_tail_seconds),
        billing_window_from=billing_attempt.submitted_at.replace(
            minute=0, second=0, microsecond=0
        ),
        billing_window_until=billing_attempt.submitted_at.replace(
            minute=0, second=0, microsecond=0
        )
        + timedelta(hours=1),
        hourly_worker_rate_usd=item.hourly_worker_rate_usd,
        pre_execution_delay_ms=2_000,
        worker_startup_ms=None,
        execution_ms=30_000,
        accepted_idle_tail_ms=item.idle_tail_seconds * 1_000,
        idle_tail_ms=None,
        gpu_cost_usd=Decimal("0.02"),
        cpu_cost_usd=Decimal(0),
        disk_cost_usd=Decimal("0.001"),
        fee_cost_usd=Decimal("0.002"),
        actual_cost_usd=Decimal("0.023"),
        reconciled_at=billing_attempt.submitted_at.replace(
            minute=0, second=0, microsecond=0
        )
        + timedelta(hours=1),
    )


def test_receipt_round_trip_keeps_unobservable_components_null_and_content_free() -> (
    None
):
    receipt = _receipt()
    serialized = receipt.to_dict()
    assert serialized["worker_startup_ms"] is None
    assert serialized["accepted_idle_tail_ms"] == 10_000
    assert serialized["idle_tail_ms"] is None
    assert ServerlessBillingReceipt.from_dict(serialized).to_dict() == serialized
    with pytest.raises(RunPodManagerError, match="unsupported"):
        ServerlessBillingReceipt.from_dict(
            {**serialized, "response": "private-provider-body"}
        )

    encoded = json.dumps(serialized, sort_keys=True).lower()
    assert not any(
        marker in encoded
        for marker in (
            "http://",
            "https://",
            "prompt",
            "response",
            "signed_url",
            "image",
            "weight",
            "capability",
            '"raw"',
        )
    )


def test_receipt_rejects_nonfinite_negative_and_mismatched_components() -> None:
    receipt = _receipt()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(receipt, fee_cost_usd=Decimal("NaN"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(receipt, gpu_cost_usd=Decimal(-1))
    with pytest.raises(ValueError, match="do not equal"):
        replace(receipt, actual_cost_usd=Decimal("0.024"))
    with pytest.raises(ValueError, match="intervals"):
        replace(
            receipt,
            reconciled_at=receipt.billing_window_until - timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="timing exceeds"):
        replace(receipt, execution_ms=300_000)


def test_worker_cost_is_ceiled_and_rejects_nan() -> None:
    assert serverless_worker_cost_usd(Decimal("0.69"), 1, 1) == Decimal("0.000192")
    with pytest.raises(ValueError, match="finite and positive"):
        serverless_worker_cost_usd(Decimal("NaN"), 1, 1)


def test_serverless_worker_cost_bills_every_gpu_in_the_worker() -> None:
    """The catalog prices per GPU; a worker attaches gpu_count of them.

    This is the sole rule converting the per-GPU rate into worker cost, and
    ServerlessCapacityQuote re-uses it to derive-check itself - so without the
    multiplier an understated quote validates cleanly and the accepted ceiling
    binds to it. The identical defect was fixed and pinned for Pods; every
    Serverless fixture uses a single GPU, which is why this was invisible.
    """
    one_hour = 3600
    assert serverless_worker_cost_usd(Decimal("0.44"), 1, one_hour) == Decimal(
        "0.440000"
    )
    assert serverless_worker_cost_usd(Decimal("0.44"), 4, one_hour) == Decimal(
        "1.760000"
    )
    # Authorization rounds UP, never in the payer's favour.
    assert serverless_worker_cost_usd(Decimal("0.44"), 3, 1) > Decimal("0.000366")
    for bad in (0, -1, True):
        with pytest.raises((ValueError, TypeError)):
            serverless_worker_cost_usd(Decimal("0.44"), bad, one_hour)
