"""Read-only Serverless cold-start observations, not spend or readiness evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from kestrel_sovereign.execution_custody import is_execution_control_error

from .models import (
    CursorPagination,
    RunPodManagerError,
    _integer,
    _mapping,
    _optional_decimal,
    _optional_response_datetime,
)


class EndpointMetricsInterval(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    SIX_HOURS = "6h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"

    @property
    def duration(self) -> timedelta:
        unit = (
            timedelta(minutes=1)
            if self.value.endswith("m")
            else (timedelta(hours=1) if self.value.endswith("h") else timedelta(days=1))
        )
        return int(self.value[:-1]) * unit


@dataclass(frozen=True)
class _MetricsTimestamp:
    """One exact UTC instant plus its original RFC3339 representation."""

    utc_second: datetime
    fraction_digits: str
    text: str

    @property
    def key(self) -> tuple[datetime, str]:
        # Fraction strings have no trailing zeros. Lexicographic order is
        # exact within one UTC second, without float/Decimal precision limits.
        return self.utc_second, self.fraction_digits

    @property
    def datetime_view(self) -> datetime:
        """A microsecond-limited view; never used for window comparisons."""

        return self.utc_second.replace(
            microsecond=int((self.fraction_digits + "000000")[:6])
        )


def _parse_metrics_timestamp(
    value: Any,
    context: str,
    *,
    whole_seconds: bool = False,
) -> _MetricsTimestamp:
    # fromisoformat alone also accepts date-only, space-separated and naive
    # values. Require the documented RFC3339 representation before parsing.
    if not isinstance(value, str):
        raise RunPodManagerError(f"Invalid {context} timestamp")
    matched = re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}"
        r"(?:\.(?P<fraction>\d+))?(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)",
        value,
        flags=re.ASCII,
    )
    if matched is None:
        raise RunPodManagerError(f"Invalid {context} timestamp")
    # All supported buckets are at least one minute. Inspect the original
    # fraction before datetime can truncate sub-microsecond nonzero digits.
    if whole_seconds and (matched.group("fraction") or "").strip("0"):
        raise RunPodManagerError("Cold-start bucket is not interval-aligned")
    parsed = _optional_response_datetime(
        value[:-1] + "Z" if value.endswith("z") else value, context
    )
    if parsed is None:
        raise RunPodManagerError(f"Invalid {context} timestamp")
    return _MetricsTimestamp(
        parsed.replace(microsecond=0),
        (matched.group("fraction") or "").rstrip("0"),
        value,
    )


def cold_start_query(
    *,
    start_time: str | None,
    end_time: str | None,
    interval: EndpointMetricsInterval | str | None,
    limit: int | None,
) -> dict[str, str | int]:
    """Validate caller syntax/bounds; provider owns its clock and retention cut."""

    params: dict[str, str | int] = {}
    parsed: dict[str, _MetricsTimestamp] = {}
    for key, value in (("startTime", start_time), ("endTime", end_time)):
        if value is not None:
            try:
                parsed[key] = _parse_metrics_timestamp(value, "metrics query")
            except RunPodManagerError as error:
                if is_execution_control_error(error):
                    raise
                raise ValueError("Metrics query requires RFC3339 timestamps") from None
            params[key] = value
    if len(parsed) == 2 and parsed["startTime"].key >= parsed["endTime"].key:
        raise ValueError("Metrics start_time must precede end_time")
    if interval is not None:
        if (
            not isinstance(interval, str)
            or interval not in EndpointMetricsInterval._value2member_map_
        ):
            raise ValueError("Unsupported endpoint metrics interval")
        params["interval"] = EndpointMetricsInterval(interval).value
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Metrics limit must be an integer from 1 to 1000")
        params["limit"] = limit
    return params


@dataclass(frozen=True)
class EndpointDurationStats:
    """Vendor bucket statistics in milliseconds; percentiles are not additive."""

    minimum_ms: int
    maximum_ms: int
    total_ms: Decimal
    p70_ms: int
    p90_ms: int
    p95_ms: int
    p98_ms: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointDurationStats":
        integers = [
            _integer(value, key, "cold-start durationMs")
            for key in ("min", "max", "p70", "p90", "p95", "p98")
        ]
        total = _optional_decimal(value.get("total"), "cold-start durationMs.total")
        ordered = (integers[0], *integers[2:], integers[1])
        if (
            total is None
            or any(item < 0 for item in integers)
            or any(lower > upper for lower, upper in zip(ordered, ordered[1:]))
        ):
            raise RunPodManagerError("Invalid cold-start duration statistics")
        return cls(integers[0], integers[1], total, *integers[2:])


@dataclass(frozen=True)
class EndpointColdStartBucket:
    time: datetime
    count: int
    duration_ms: EndpointDurationStats

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointColdStartBucket":
        count = _integer(value, "count", "cold-start bucket")
        if count < 0:
            raise RunPodManagerError("Invalid cold-start bucket count")
        return cls(
            time=_parse_metrics_timestamp(
                value.get("time"), "cold-start bucket", whole_seconds=True
            ).datetime_view,
            count=count,
            duration_ms=EndpointDurationStats.from_dict(
                _mapping(value, "durationMs", "cold-start bucket")
            ),
        )


@dataclass(frozen=True)
class EndpointColdStartMetricsPage:
    _start: _MetricsTimestamp
    _end: _MetricsTimestamp
    interval: EndpointMetricsInterval
    records: tuple[EndpointColdStartBucket, ...]
    pagination: CursorPagination

    @property
    def start_time(self) -> datetime:
        return self._start.datetime_view

    @property
    def end_time(self) -> datetime:
        return self._end.datetime_view

    @property
    def start_time_text(self) -> str:
        return self._start.text

    @property
    def end_time_text(self) -> str:
        return self._end.text

    @property
    def window_identity(
        self,
    ) -> tuple[tuple[datetime, str], tuple[datetime, str], EndpointMetricsInterval]:
        return self._start.key, self._end.key, self.interval

    def require_query(self, params: Mapping[str, str | int]) -> None:
        """Reject a response that contradicts the original explicit query."""

        expected_interval = params.get(
            "interval", EndpointMetricsInterval.ONE_HOUR.value
        )
        if self.interval.value != expected_interval:
            raise RunPodManagerError("Cold-start response changed requested interval")
        if (
            "startTime" in params
            and self._start.key
            != _parse_metrics_timestamp(params["startTime"], "metrics query").key
        ):
            raise RunPodManagerError("Cold-start response changed requested startTime")
        # A future request end may be clamped by Runpod's clock. It can never
        # extend the response beyond the caller's exclusive end boundary.
        if (
            "endTime" in params
            and self._end.key
            > _parse_metrics_timestamp(params["endTime"], "metrics query").key
        ):
            raise RunPodManagerError("Cold-start response exceeded requested endTime")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        pagination: CursorPagination,
    ) -> "EndpointColdStartMetricsPage":
        start = _parse_metrics_timestamp(value.get("startTime"), "cold-start startTime")
        end = _parse_metrics_timestamp(value.get("endTime"), "cold-start endTime")
        interval = value.get("interval")
        if (
            not isinstance(interval, str)
            or interval not in EndpointMetricsInterval._value2member_map_
        ):
            raise RunPodManagerError("Invalid cold-start metrics interval")
        resolved_interval = EndpointMetricsInterval(interval)
        records = value.get("records")
        if not isinstance(records, list) or any(
            not isinstance(item, Mapping) for item in records
        ):
            raise RunPodManagerError("Invalid cold-start records envelope")
        if start.key >= end.key:
            raise RunPodManagerError("Invalid cold-start resolved window")
        buckets = tuple(EndpointColdStartBucket.from_dict(item) for item in records)
        for index, bucket in enumerate(buckets):
            if (
                bucket.time - datetime(1970, 1, 1, tzinfo=UTC)
            ) % resolved_interval.duration:
                raise RunPodManagerError("Cold-start bucket is not interval-aligned")
            # A boundary bucket can begin before an unaligned window start.
            if (
                (bucket.time, "") >= end.key
                or start.utc_second - bucket.time >= resolved_interval.duration
            ):
                raise RunPodManagerError("Cold-start bucket is outside resolved window")
            if index and buckets[index - 1].time >= bucket.time:
                raise RunPodManagerError("Cold-start buckets are not strictly ordered")
        return cls(
            start,
            end,
            resolved_interval,
            buckets,
            pagination,
        )


@dataclass(frozen=True)
class EndpointColdStartObservation:
    """Complete bounded read; sparse/empty buckets remain sparse/empty."""

    endpoint_id: str
    observed_at: datetime
    pages: tuple[EndpointColdStartMetricsPage, ...]

    @property
    def records(self) -> tuple[EndpointColdStartBucket, ...]:
        return tuple(record for page in self.pages for record in page.records)

    @property
    def start_time(self) -> datetime:
        return self.pages[0].start_time

    @property
    def end_time(self) -> datetime:
        return self.pages[0].end_time

    @property
    def interval(self) -> EndpointMetricsInterval:
        return self.pages[0].interval

    @property
    def start_time_text(self) -> str:
        return self.pages[0].start_time_text

    @property
    def end_time_text(self) -> str:
        return self.pages[0].end_time_text
