"""Exact shared billing primitives, independent of a caller's Decimal context."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from kestrel_cloud_runpod.billing import (
    billing_amounts,
    billing_hour_bounds,
    billing_identity_amount,
    exact_decimal_sum,
)
from kestrel_cloud_runpod.models import RunPodManagerError


def test_signed_deltas_retain_one_to_one_e_minus_29_without_rounding():
    with localcontext() as context:
        context.prec = 5
        old = Decimal(1)
        new = Decimal("1e-29")
        delta = exact_decimal_sum((new, old.copy_negate()))
        assert delta == Decimal("-0.99999999999999999999999999999")
        assert exact_decimal_sum((old, delta)) == new


@pytest.mark.parametrize("order", [(0, 1, 2), (0, 2, 1), (2, 1, 0)])
def test_signed_sum_across_two_hundred_orders_of_magnitude(order):
    values = (Decimal("1e100"), Decimal("-1e100"), Decimal("1e-100"))
    with localcontext() as context:
        context.prec = 2
        assert exact_decimal_sum(values[index] for index in order) == Decimal("1e-100")
        assert exact_decimal_sum((values[0], Decimal("-1e-100"))) == Decimal(
            (0, (9,) * 200, -100)
        )


@pytest.mark.parametrize("invalid", [Decimal("NaN"), Decimal("Infinity"), 1])
def test_signed_sum_rejects_nonfinite_or_wrong_unit(invalid):
    with pytest.raises(ValueError, match="finite Decimals"):
        exact_decimal_sum((invalid,))
    with pytest.raises(ValueError, match="finite Decimal"):
        billing_identity_amount(invalid)


def test_empty_sum_is_exact_zero():
    assert exact_decimal_sum(()) == Decimal(0)


@pytest.mark.parametrize(
    "amount,expected",
    [("1.000", "1"), ("10", "10"), ("-0.00", "0"), ("0.0100", "0.01")],
)
def test_identity_amount_is_numerical_not_decimal_spelling(amount, expected):
    with localcontext() as context:
        context.prec = 2
        assert billing_identity_amount(Decimal(amount)) == expected
        assert billing_identity_amount(Decimal("1.00000000000000000000000000001")) == (
            "1.00000000000000000000000000001"
        )


def test_component_arithmetic_does_not_round_away_a_missing_component():
    with localcontext() as context:
        context.prec = 2
        with pytest.raises(RunPodManagerError, match="components"):
            billing_amounts(
                {"totalAmount": 1, "gpuAmount": 1, "diskAmount": 1e-29},
                components=("gpuAmount", "diskAmount"),
                context="billing record",
            )


@pytest.mark.parametrize("invalid", [True, "0.1", -1, float("nan"), float("inf")])
def test_component_parser_rejects_invalid_api_amounts(invalid):
    with pytest.raises(RunPodManagerError, match="Invalid Runpod"):
        billing_amounts(
            {"totalAmount": invalid, "gpuAmount": 0},
            components=("gpuAmount",),
            context="billing record",
        )


def test_utc_hour_bounds_are_shared_and_half_open_across_offsets():
    start = datetime(2026, 10, 20, 3, 10, tzinfo=timezone(timedelta(hours=3)))
    end = datetime(2026, 10, 20, 2, tzinfo=UTC)
    assert billing_hour_bounds(start, end) == (datetime(2026, 10, 20, tzinfo=UTC), end)


@pytest.mark.parametrize(
    "start,end",
    [
        (datetime(2026, 10, 20), datetime(2026, 10, 21, tzinfo=UTC)),
        (datetime(2026, 10, 20, tzinfo=UTC), datetime(2026, 10, 20, tzinfo=UTC)),
    ],
)
def test_hour_bounds_reject_unknown_or_empty_intervals(start, end):
    with pytest.raises(ValueError):
        billing_hour_bounds(start, end)


def test_serverless_hour_receipt_and_totals_use_exact_shared_arithmetic():
    from kestrel_cloud_runpod.serverless_capacity_contracts import (
        ServerlessEndpointHourCost,
        serverless_endpoint_hour_cost_totals,
    )

    total = Decimal((0, (1,) + (0,) * 28 + (1,), -29))
    start = datetime(2026, 10, 20, tzinfo=UTC)
    with localcontext() as context:
        context.prec = 5
        hour = ServerlessEndpointHourCost(
            provider_observation_id="billing-test-hour",
            endpoint_id="owned-endpoint",
            utc_hour_start=start,
            utc_hour_end=start + timedelta(hours=1),
            gpu_cost_usd=Decimal(1),
            cpu_cost_usd=Decimal(0),
            disk_cost_usd=Decimal("1e-29"),
            fee_cost_usd=Decimal(0),
            actual_cost_usd=total,
        )
        assert serverless_endpoint_hour_cost_totals((hour,))["actual_cost_usd"] == total
