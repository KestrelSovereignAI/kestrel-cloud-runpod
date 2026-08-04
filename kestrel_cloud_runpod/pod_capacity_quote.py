"""Read-only composition for live Runpod Pod capacity quotes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .models import GPUProfile
from .pod_capacity_contracts import PodCapacityQuote, PodCapacityQuoteRequest


class PodCapacityQuoteProvider(Protocol):
    """The deliberately narrow provider surface needed by API replicas."""

    async def quote(self, request: PodCapacityQuoteRequest) -> PodCapacityQuote: ...


class PodCapacityQuoteService:
    """Expose live placement quotes without constructing writable lease state."""

    def __init__(
        self,
        *,
        provider: PodCapacityQuoteProvider,
        profiles: Mapping[str, GPUProfile],
    ) -> None:
        if not profiles:
            raise ValueError("Pod capacity quotes require configured GPU profiles")
        self.provider = provider
        self.profiles = profiles

    async def quote(self, request: PodCapacityQuoteRequest) -> PodCapacityQuote:
        """Return the provider's exact live offer without mutating capacity."""

        return await self.provider.quote(request)
