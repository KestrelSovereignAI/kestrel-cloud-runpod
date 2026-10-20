"""One exact-Pod billing observation path for training and inference consumers.

This is historical cost observation, not spend authority or invoice finality.
Callers retain original resource/authorization and verified termination custody.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from .billing import (
    billing_amounts,
    billing_hour_bounds,
    billing_identity_amount,
    exact_decimal_sum,
)
from .clients import RunpodControlPlaneClient
from .effect_custody import await_owned_effect
from .models import BillingPage, RunPodManagerError
from .pod_capacity_contracts import (
    PodBillingReceipt,
    _safe_identifier,
    iso_datetime,
    require_aware,
)
from .signed_invocations import canonical_sha256


async def read_exact_pod_billing(
    client: RunpodControlPlaneClient,
    *,
    pod_id: str,
    created_at: datetime,
    terminated_at: datetime,
    realized_hourly_rate_usd: Decimal,
    clock: Callable[[], datetime],
) -> PodBillingReceipt | None:
    """Observe complete closed-hour costs; unknown history stays pending.

    No synthetic capacity spec is required. A rate is provenance carried beside
    actual provider amounts, never a multiplier used to manufacture those costs.
    """

    _safe_identifier(pod_id, "billing pod_id")
    if (
        not isinstance(realized_hourly_rate_usd, Decimal)
        or not realized_hourly_rate_usd.is_finite()
        or realized_hourly_rate_usd < 0
    ):
        raise ValueError(
            "Pod billing realized hourly rate must be finite and nonnegative"
        )
    window_from, window_until = billing_hour_bounds(created_at, terminated_at)
    now = clock()
    require_aware(now, "Pod billing observation clock")
    if now < window_until:
        return None
    page = await await_owned_effect(
        asyncio.to_thread(
            client.pod_billing,
            start_time=iso_datetime(window_from),
            end_time=iso_datetime(window_until),
            bucket_size="hour",
            pod_id=pod_id,
        )
    )
    rows = _exact_hours(page, pod_id=pod_id, first=window_from, until=window_until)
    if rows is None:
        return None
    identity = {
        "source": "runpod-v2-pod-billing",
        "pod_id": pod_id,
        "hourly_price_usd": billing_identity_amount(realized_hourly_rate_usd),
        "hours": rows,
    }
    reconciled_at = clock()
    require_aware(reconciled_at, "Pod billing observation clock")
    return PodBillingReceipt(
        provider_billing_id="runpod-billing:"
        + canonical_sha256(identity).removeprefix("sha256:"),
        provider_pod_id=pod_id,
        billed_from=window_from,
        billed_until=window_until,
        billed_seconds=int((window_until - window_from).total_seconds()),
        hourly_price_usd=realized_hourly_rate_usd,
        actual_cost_usd=exact_decimal_sum(Decimal(row["totalAmount"]) for row in rows),
        reconciled_at=reconciled_at,
    )


def _exact_hours(
    page: BillingPage, *, pod_id: str, first: datetime, until: datetime
) -> tuple[dict[str, str], ...] | None:
    components = ("gpuAmount", "cpuAmount", "diskAmount")
    amount_keys = {"totalAmount", *components}
    metadata = page.metadata
    _keys(
        metadata,
        {"query", "recordCount", "uniquePodCount", "totals"},
        "billing metadata",
    )
    query = metadata.get("query")
    totals = metadata.get("totals")
    _keys(query, {"startTime", "endTime", "bucketSize", "podId"}, "billing query")
    _keys(totals, amount_keys, "billing totals")
    if (
        query["podId"] != pod_id
        or query["bucketSize"] != "hour"
        or _timestamp(query["startTime"]) != first
        or _timestamp(query["endTime"]) != until
    ):
        raise RunPodManagerError(
            "Runpod Pod billing query echo differs from requested coverage"
        )
    if type(metadata["recordCount"]) is not int or metadata["recordCount"] != len(
        page.records
    ):
        raise RunPodManagerError("Runpod Pod billing record count is inconsistent")
    unique_count = metadata["uniquePodCount"]
    if type(unique_count) is not int or unique_count != (1 if page.records else 0):
        raise RunPodManagerError("Runpod Pod billing unique Pod count is inconsistent")
    expected_totals = billing_amounts(
        totals, components=components, context="billing totals"
    )
    parsed: list[tuple[datetime, datetime, dict[str, Decimal]]] = []
    for row in page.records:
        _keys(
            row, {"podId", "startTime", "endTime", *amount_keys}, "Pod billing record"
        )
        if row["podId"] != pod_id:
            raise RunPodManagerError("Runpod Pod billing record identity mismatch")
        parsed.append(
            (
                _timestamp(row["startTime"]),
                _timestamp(row["endTime"]),
                billing_amounts(
                    row, components=components, context="Pod billing record"
                ),
            )
        )
    if any(
        exact_decimal_sum(amounts[key] for _, _, amounts in parsed)
        != expected_totals[key]
        for key in amount_keys
    ):
        raise RunPodManagerError("Runpod Pod billing totals do not equal its records")
    cursor = first
    incomplete = not parsed
    normalized: list[dict[str, str]] = []
    for start, end, amounts in sorted(parsed, key=lambda item: item[0]):
        if start < cursor or start < first or end <= start or end > until:
            raise RunPodManagerError(
                "Runpod Pod billing hours overlap or escape coverage"
            )
        if start != start.replace(minute=0, second=0, microsecond=0):
            raise RunPodManagerError(
                "Runpod Pod billing record is not an exact UTC hour"
            )
        if start > cursor or end < start + timedelta(hours=1):
            incomplete = True
        if end > start + timedelta(hours=1):
            raise RunPodManagerError("Runpod Pod billing record exceeds one hour")
        normalized.append(
            {
                "podId": pod_id,
                "startTime": iso_datetime(start),
                "endTime": iso_datetime(end),
                **{
                    name: billing_identity_amount(amount)
                    for name, amount in sorted(amounts.items())
                },
            }
        )
        cursor = end
    if incomplete or cursor < until:
        return None
    return tuple(normalized)


def _keys(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RunPodManagerError(
            f"Runpod {context} contains missing or unsupported fields"
        )


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RunPodManagerError("Runpod Pod billing timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RunPodManagerError("Runpod Pod billing timestamp is invalid") from error
    if parsed.utcoffset() is None:
        raise RunPodManagerError("Runpod Pod billing timestamp must include a timezone")
    return parsed.astimezone(UTC)
