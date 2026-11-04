"""Exact shared billing primitives, independent of a caller's Decimal context."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from kestrel_cloud_runpod.billing import (
    billing_amounts,
    billing_exposure_ratio_float,
    billing_exposure_sum_float,
    billing_hour_bounds,
    billing_identity_amount,
    ceil_timedelta_seconds,
    conservative_decimal_ratio_float,
    exact_decimal_integer_scale,
    exact_decimal_sum,
    floor_timedelta_seconds,
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


def test_sum_ignores_ambient_exponent_limits_and_traps():
    with localcontext() as context:
        context.prec = 2
        context.Emin = -1
        context.Emax = 1
        assert exact_decimal_sum((Decimal("1e-5"), Decimal("1e-5"))) == Decimal("2e-5")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((Decimal("-0.0"),), Decimal("0.0")),
        ((Decimal("1"), Decimal("0.0")), Decimal("1.0")),  # noqa: FURB157
        ((Decimal("0"),), Decimal("0")),  # noqa: FURB157
    ],
)
def test_exact_decimal_sum_preserves_published_zero_exponent(values, expected):
    result = exact_decimal_sum(values)

    assert result.as_tuple() == expected.as_tuple()


def test_sum_rejects_pathological_exponent_span_before_large_allocation():
    with pytest.raises(ValueError, match="supported exact range"):
        exact_decimal_sum((Decimal("1e4096"), Decimal("1e-1")))


def test_exposure_sum_uses_one_context_independent_directed_boundary():
    with localcontext() as context:
        context.prec = 2
        context.Emin = -1
        context.Emax = 1
        assert (
            billing_exposure_sum_float((Decimal("0.05091666666666667"), Decimal("0.2")))
            == 0.2509166666666667
        )


@pytest.mark.parametrize(
    ("value", "ceiling", "floor"),
    [
        (timedelta(seconds=1, microseconds=1), 2, 1),
        (timedelta(seconds=2**34, microseconds=1), 2**34 + 1, 2**34),
        (timedelta(microseconds=-1), 0, -1),
    ],
)
def test_timedelta_second_bounds_are_integer_exact(value, ceiling, floor):
    assert ceil_timedelta_seconds(value) == ceiling
    assert floor_timedelta_seconds(value) == floor


@pytest.mark.parametrize(
    ("value", "multiplier", "expected"),
    [
        ("0.99", 1, "0.99"),
        (
            "9.999999999999999999999999999e99",
            1001,
            "1.0009999999999999999999999998999e103",
        ),
        ("1e-100", 1001, "1.001e-97"),
        ("-0", 7, "0"),
    ],
)
def test_integer_rate_scale_is_independent_of_ambient_context(
    value, multiplier, expected
):
    with localcontext() as context:
        context.prec = 1
        assert exact_decimal_integer_scale(Decimal(value), multiplier) == Decimal(
            expected
        )


@pytest.mark.parametrize(
    ("value", "multiplier"),
    [(Decimal("NaN"), 1), (Decimal(1), -1), (Decimal(1), True), (1, 1)],
)
def test_integer_rate_scale_rejects_invalid_inputs(value, multiplier):
    with pytest.raises(ValueError):
        exact_decimal_integer_scale(value, multiplier)


@pytest.mark.parametrize(
    ("value", "divisor", "expected"),
    [
        ("0.99", 1, 0.99),
        ("2.97", 3, 0.99),
        ("1.41", 3, 0.47),
        ("5e-324", 1, 5e-324),
        ("1.7976931348623157e308", 1, 1.7976931348623157e308),
    ],
)
def test_conservative_ratio_uses_canonical_float_decimal_semantics(
    value, divisor, expected
):
    with localcontext() as context:
        context.prec = 1
        result = conservative_decimal_ratio_float(Decimal(value), divisor)

    assert result == expected
    assert exact_decimal_integer_scale(Decimal(str(result)), divisor) <= Decimal(value)


def test_conservative_ratio_rounds_toward_the_authorized_cap():
    value = Decimal("1.9999999999999999")
    result = conservative_decimal_ratio_float(value, 1)

    assert result < 2.0
    assert Decimal(str(result)) <= value


@pytest.mark.parametrize(
    ("value", "divisor"),
    [
        (Decimal("NaN"), 1),
        (Decimal("Infinity"), 1),
        (Decimal(0), 1),
        (Decimal(-1), 1),
        (Decimal("1e309"), 1),
        (Decimal("1e-325"), 1),
        (Decimal(1), 0),
        (Decimal(1), True),
        (1, 1),
    ],
)
def test_conservative_ratio_rejects_unrepresentable_or_invalid_inputs(value, divisor):
    with pytest.raises(ValueError):
        conservative_decimal_ratio_float(value, divisor)


@pytest.mark.parametrize(
    ("value", "multiplier", "divisor", "minimum"),
    [
        ("0", 1, 1, "0"),
        ("0.47", 3, 1, "1.41"),
        ("1", 1, 3_600_000_000, "2.777777777777777777777777778e-10"),
        ("1e-325", 1, 1, "1e-325"),
        ("9.999999999999999e307", 1, 1, "9.999999999999999e307"),
    ],
)
def test_exposure_ratio_never_rounds_below_exact_amount(
    value, multiplier, divisor, minimum
):
    with localcontext() as context:
        context.prec = 1
        context.Emax = 1
        context.Emin = -1
        result = billing_exposure_ratio_float(
            Decimal(value), multiplier=multiplier, divisor=divisor
        )

    assert Decimal(str(result)) >= Decimal(minimum)


@pytest.mark.parametrize(
    ("value", "multiplier", "divisor"),
    [
        (Decimal("NaN"), 1, 1),
        (Decimal("Infinity"), 1, 1),
        (Decimal(-1), 1, 1),
        (Decimal(1), -1, 1),
        (Decimal(1), True, 1),
        (Decimal(1), 1, 0),
        (Decimal(1), 1, True),
        (Decimal("1e309"), 1, 1),
        (1, 1, 1),
    ],
)
def test_exposure_ratio_rejects_invalid_or_overflowing_inputs(
    value, multiplier, divisor
):
    with pytest.raises(ValueError):
        billing_exposure_ratio_float(value, multiplier=multiplier, divisor=divisor)


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
        (datetime(2026, 10, 20), datetime(2026, 10, 21, tzinfo=UTC)),  # noqa: DTZ001
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
