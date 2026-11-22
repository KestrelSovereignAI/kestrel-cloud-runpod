"""Registered offer never weakens ownership or the budget before effects."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from kestrel_sdk.llm import InferenceLeaseConstraintError, InferenceLeaseRequest, InferencePrivacy

from kestrel_cloud_runpod.registered_ollama_admission import RegisteredQuotePolicy


NOW = datetime(2026, 11, 22, tzinfo=UTC)
RATE = Decimal("1.34")


@pytest.fixture
def original():
    return RegisteredQuotePolicy(
        "a" * 64, "journal", "US-NC-1", "gpt-oss:20b", "sha256:" + "b" * 64, 60, 45, 5,
        180, 120, Decimal("0.01"),
    ), InferenceLeaseRequest(
        request_id="kite-request", owner_id="kite-owner", runtime="ollama", model="gpt-oss:20b",
        allowed_regions=("us-nc-1",), capabilities=("chat", "streaming"),
        max_hourly_cost_usd=Decimal("1.4"), max_total_cost_usd=Decimal("0.26"),
        expected_session_seconds=120, ready_deadline_seconds=180, idle_ttl_seconds=25,
        requested_at=NOW,
    )


def test_registered_offer_bounds_worst_case_continuous_compute_and_idle(original):
    policy, request = original
    quote = policy.quote(request, rate=RATE, now=NOW)
    assert quote.estimated_total_cost_usd == Decimal("0.123527778")
    assert quote.expires_at == NOW + timedelta(seconds=45)
    assert quote.region == "us-nc-1"
    digest = policy.accept(request, quote, refreshed_rate=RATE, now=NOW + timedelta(seconds=25))
    assert digest == policy.accept(request, quote, refreshed_rate=RATE, now=NOW + timedelta(seconds=27))


def test_registered_original_offer_cannot_be_rebound_or_used_after_drift(original):
    policy, request = original
    quote = policy.quote(request, rate=RATE, now=NOW)
    for changed, rate, now in (
        (replace(request, owner_id="someone-else"), RATE, NOW),
        (replace(request, idle_ttl_seconds=15), RATE, NOW),
        (request, Decimal("1.35"), NOW),
        (request, RATE, quote.expires_at),
    ):
        with pytest.raises(InferenceLeaseConstraintError):
            policy.accept(changed, quote, refreshed_rate=rate, now=now)
    with pytest.raises(InferenceLeaseConstraintError):
        replace(policy, journal_id="different").accept(request, quote, refreshed_rate=RATE, now=NOW)
    with pytest.raises(InferenceLeaseConstraintError):
        replace(policy, model_digest="sha256:" + "c" * 64).accept(request, quote, refreshed_rate=RATE, now=NOW)


@pytest.mark.parametrize("changes", [
    {"allowed_regions": ("ca-mtl-1",)}, {"privacy": InferencePrivacy.PRIVATE_NETWORK},
    {"expected_concurrency": 2}, {"model": "different:20b"},
    {"max_total_cost_usd": Decimal("0.08")}, {"expected_session_seconds": 121},
])
def test_registered_quote_refuses_wrong_region_capacity_and_budget(original, changes):
    policy, request = original
    with pytest.raises(InferenceLeaseConstraintError):
        policy.quote(replace(request, **changes), rate=RATE, now=NOW)
