"""Tests for read-only Pod capacity quote composition."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kestrel_cloud_runpod import PodCapacityQuoteService
from tests.pod_capacity_test_support import profile


def test_quote_service_requires_an_explicit_profile_catalog() -> None:
    with pytest.raises(ValueError, match="configured GPU profiles"):
        PodCapacityQuoteService(provider=AsyncMock(), profiles={})


@pytest.mark.asyncio
async def test_quote_service_delegates_without_a_repository() -> None:
    provider = AsyncMock()
    request = object()
    expected = object()
    provider.quote.return_value = expected
    profiles = {"catalog-lora": profile()}

    service = PodCapacityQuoteService(provider=provider, profiles=profiles)

    assert service.profiles is profiles
    assert await service.quote(request) is expected
    provider.quote.assert_awaited_once_with(request)
