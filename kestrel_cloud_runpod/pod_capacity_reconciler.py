"""One-shot entry point for a cheap external Pod-capacity poller."""

from __future__ import annotations

from collections.abc import Callable

from .pod_capacity_contracts import PodCapacityLease
from .pod_capacity_service import PodCapacityLeaseService


async def reconcile_once(
    service_factory: Callable[[], PodCapacityLeaseService],
) -> tuple[PodCapacityLease, ...]:
    """Construct a restart-safe service and run exactly one bounded pass."""

    return await service_factory().reconcile()
