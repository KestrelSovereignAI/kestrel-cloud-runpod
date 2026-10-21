"""Shared UTC coverage and exact monetary arithmetic for provider observations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .models import RunPodManagerError

_MAX_EXACT_DECIMAL_DIGITS = 4096


def billing_hour_bounds(
    started_at: datetime, coverage_until: datetime
) -> tuple[datetime, datetime]:
    """Whole UTC hours touched by a positive half-open lifecycle interval."""

    for name, value in (("started_at", started_at), ("coverage_until", coverage_until)):
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError(f"Billing {name} must be a timezone-aware datetime")
    start = started_at.astimezone(UTC)
    end = coverage_until.astimezone(UTC)
    if end <= start:
        raise ValueError("Billing coverage interval is invalid")
    first = start.replace(minute=0, second=0, microsecond=0)
    until = end.replace(minute=0, second=0, microsecond=0)
    if end != until:
        until += timedelta(hours=1)
    return first, until


def exact_decimal_sum(values: Iterable[Decimal]) -> Decimal:
    """Sum finite money without consulting the ambient Decimal context.

    Billing observations are exact decimal quantities.  Aligning their integer
    coefficients directly avoids every ambient context setting, including
    precision, exponent bounds, rounding, and traps.  The explicit span bound
    rejects pathological inputs before allocating an enormous power of ten.
    """

    items = tuple(values)
    if any(not isinstance(item, Decimal) or not item.is_finite() for item in items):
        raise ValueError("Billing amounts must be finite Decimals")
    if not items:
        return Decimal(0)
    tuples = tuple(item.as_tuple() for item in items)
    exponents = tuple(item.exponent for item in tuples)
    if any(not isinstance(exponent, int) for exponent in exponents):
        raise ValueError("Billing amounts must have finite decimal exponents")
    common_exponent = min(exponents)
    required_digits = max(
        len(item.digits) + item.exponent - common_exponent for item in tuples
    )
    required_digits += len(str(len(tuples)))
    if required_digits > _MAX_EXACT_DECIMAL_DIGITS:
        raise ValueError("Billing amount precision exceeds the supported exact range")

    total = 0
    for item in tuples:
        coefficient = int("".join(str(digit) for digit in item.digits))
        if item.sign:
            coefficient = -coefficient
        total += coefficient * (10 ** (item.exponent - common_exponent))
    if total == 0:
        # Published v0.15 receipts retain the finest input exponent in totals
        # (while ordinary Decimal addition canonicalizes a negative-zero sign
        # to positive). Preserve that wire identity without ambient precision.
        return Decimal((0, (0,), common_exponent))
    sign = int(total < 0)
    digits = tuple(int(digit) for digit in str(abs(total)))
    return Decimal((sign, digits, common_exponent))


def exact_timedelta_microseconds(value: timedelta) -> int:
    """Return exact signed microseconds without a binary-float boundary."""

    if not isinstance(value, timedelta):
        raise TypeError("Billing duration must be a timedelta")
    return (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds


def ceil_timedelta_seconds(value: timedelta) -> int:
    """Return the exact mathematical ceiling of a timedelta in seconds."""

    microseconds = exact_timedelta_microseconds(value)
    return (microseconds + 999_999) // 1_000_000


def floor_timedelta_seconds(value: timedelta) -> int:
    """Return the exact mathematical floor of a timedelta in seconds."""

    return exact_timedelta_microseconds(value) // 1_000_000


def exact_decimal_integer_scale(value: Decimal, multiplier: int) -> Decimal:
    """Scale a finite Decimal by an integer without consulting ambient context."""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("Billing rate must be a finite Decimal")
    if (
        not isinstance(multiplier, int)
        or isinstance(multiplier, bool)
        or multiplier < 0
    ):
        raise ValueError("Billing rate multiplier must be a nonnegative integer")
    if value.is_zero() or multiplier == 0:
        return Decimal(0)
    value_tuple = value.as_tuple()
    coefficient = int("".join(str(digit) for digit in value_tuple.digits))
    product = coefficient * multiplier
    return Decimal(
        (
            value_tuple.sign,
            tuple(int(digit) for digit in str(product)),
            value_tuple.exponent,
        )
    )


def _directed_decimal_ratio_float(
    value: Decimal,
    *,
    multiplier: int,
    divisor: int,
    ceiling: bool,
) -> float:
    """Convert an exact nonnegative ratio at one explicit float boundary.

    Public and provider floats in this package use their shortest decimal
    string as their canonical numeric value.  Authorization caps therefore
    round down on that decimal boundary, while exposure estimates round up.
    Integer/Fraction arithmetic keeps both directions independent of the
    process-global Decimal context.
    """

    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("Billing amount must be a finite nonnegative Decimal")
    if (
        not isinstance(multiplier, int)
        or isinstance(multiplier, bool)
        or multiplier < 0
    ):
        raise ValueError("Billing multiplier must be a nonnegative integer")
    if not isinstance(divisor, int) or isinstance(divisor, bool) or divisor < 1:
        raise ValueError("Billing divisor must be a positive integer")
    exact = Fraction(value) * multiplier / divisor
    if exact == 0:
        return 0.0
    try:
        candidate = float(exact)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "Billing amount cannot be represented as a finite float"
        ) from exc
    if not math.isfinite(candidate):
        raise ValueError("Billing amount cannot be represented as a finite float")
    if candidate <= 0:
        if not ceiling:
            raise ValueError("Billing amount is below the positive float range")
        candidate = math.nextafter(0.0, math.inf)
    candidate_value = Fraction(Decimal(str(candidate)))
    if ceiling and candidate_value < exact:
        candidate = math.nextafter(candidate, math.inf)
    elif not ceiling and candidate_value > exact:
        candidate = math.nextafter(candidate, -math.inf)
    if not math.isfinite(candidate) or candidate <= 0:
        raise ValueError("Billing amount cannot be represented as a finite float")
    candidate_value = Fraction(Decimal(str(candidate)))
    if (ceiling and candidate_value < exact) or (
        not ceiling and candidate_value > exact
    ):
        raise ValueError("Billing amount cannot be represented as a directed float")
    return candidate


def conservative_decimal_ratio_float(value: Decimal, divisor: int) -> float:
    """Return a finite positive float that never exceeds an exact ratio."""

    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError("Billing cap must be a finite positive Decimal")
    return _directed_decimal_ratio_float(
        value,
        multiplier=1,
        divisor=divisor,
        ceiling=False,
    )


def billing_exposure_ratio_float(
    value: Decimal, *, multiplier: int = 1, divisor: int = 1
) -> float:
    """Return a finite float that never understates exact provider exposure."""

    return _directed_decimal_ratio_float(
        value,
        multiplier=multiplier,
        divisor=divisor,
        ceiling=True,
    )


def billing_exposure_sum_float(values: Iterable[Decimal]) -> float:
    """Sum exact components and expose one conservatively rounded float."""

    return billing_exposure_ratio_float(exact_decimal_sum(values))


def billing_identity_amount(value: Decimal) -> str:
    """Canonical numerical identity, without changing published wire formatting."""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("Billing identity amount must be a finite Decimal")
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def billing_amounts(
    value: Mapping[str, Any], *, components: Sequence[str], context: str
) -> dict[str, Decimal]:
    """Validate API numeric amounts and their exact component/total equality."""

    amounts: dict[str, Decimal] = {}
    for name in ("totalAmount", *components):
        raw = value.get(name)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise RunPodManagerError(f"Invalid Runpod {context}.{name}")
        amount = Decimal(str(raw))
        if not amount.is_finite() or amount < 0:
            raise RunPodManagerError(f"Invalid Runpod {context}.{name}")
        # Only new API observations are normalized. Published receipt codecs
        # preserve their original decimal strings and digest identities.
        amounts[name] = Decimal(billing_identity_amount(amount))
    if amounts["totalAmount"] != exact_decimal_sum(
        amounts[name] for name in components
    ):
        raise RunPodManagerError(f"Runpod {context} components do not equal total cost")
    return amounts
