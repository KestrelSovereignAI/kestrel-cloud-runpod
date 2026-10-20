"""Shared UTC coverage and exact monetary arithmetic for provider observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from typing import Any

from .models import RunPodManagerError


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
    """Sum finite money without ambient Decimal precision silently rounding it."""

    items = tuple(values)
    if any(not isinstance(item, Decimal) or not item.is_finite() for item in items):
        raise ValueError("Billing amounts must be finite Decimals")
    if not items:
        return Decimal(0)
    exponent = min(item.as_tuple().exponent for item in items)
    digits = max(
        len(item.as_tuple().digits) + item.as_tuple().exponent - exponent
        for item in items
    )
    with localcontext() as context:
        context.prec = max(context.prec, digits + len(str(len(items))) + 1)
        return sum(items, start=Decimal(0))


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
