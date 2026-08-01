"""Cost selection and public-route contracts for Ollama leases."""

from dataclasses import replace

import pytest
from ollama_test_support import MutableClock, make_decision, make_request

from kestrel_cloud_runpod.models import ComputeProduct, RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseMode,
    canonical_model_name,
    sanitize_provider_error,
    select_ollama_plan,
)


def test_bursty_session_selects_lower_effective_serverless_cost():
    clock = MutableClock()
    request = make_request(clock)
    plan = select_ollama_plan(
        request,
        {
            ComputeProduct.SERVERLESS: make_decision(
                ComputeProduct.SERVERLESS,
                rate=1.0,
                gpu_id="serverless",
                pool="pool-24",
            ),
            ComputeProduct.POD: make_decision(
                ComputeProduct.POD, rate=0.5, gpu_id="pod", pool=None
            ),
        },
    )

    assert plan.mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    assert plan.estimated_cost == pytest.approx(390 / 3600)


def test_sustained_session_selects_pod_at_live_rates():
    clock = MutableClock()
    request = make_request(
        clock,
        expected_active_seconds=3500,
        serverless_initialization_seconds=120,
        serverless_idle_tail_seconds=120,
    )
    plan = select_ollama_plan(
        request,
        {
            ComputeProduct.SERVERLESS: make_decision(
                ComputeProduct.SERVERLESS,
                rate=1.0,
                gpu_id="serverless",
                pool="pool-24",
            ),
            ComputeProduct.POD: make_decision(
                ComputeProduct.POD, rate=0.6, gpu_id="pod", pool=None
            ),
        },
    )

    assert plan.mode is OllamaLeaseMode.DEDICATED_POD
    assert plan.estimated_cost == pytest.approx(0.6)


def test_forced_mode_and_cost_cap_fail_closed():
    clock = MutableClock()
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=0.1,
    )

    with pytest.raises(RunPodManagerError, match="authorized cost"):
        select_ollama_plan(
            request,
            {
                ComputeProduct.POD: make_decision(
                    ComputeProduct.POD, rate=1.0, gpu_id="pod", pool=None
                )
            },
        )


def test_request_fingerprint_changes_with_billing_policy():
    clock = MutableClock()
    request = make_request(clock)

    assert request.fingerprint != replace(request, max_authorized_cost=1.5).fingerprint


def test_model_tags_normalize_implicit_latest():
    assert canonical_model_name("Qwen3:8B") == "qwen3:8b"
    assert canonical_model_name("qwen3") == "qwen3:latest"


@pytest.mark.parametrize("invalid_cost", [float("nan"), float("inf"), 0, -1])
def test_request_rejects_non_finite_or_non_positive_cost(invalid_cost):
    clock = MutableClock()

    with pytest.raises(ValueError, match="max_authorized_cost"):
        make_request(clock, max_authorized_cost=invalid_cost)


def test_request_rejects_fractional_duration():
    clock = MutableClock()

    with pytest.raises(ValueError, match="positive integers"):
        make_request(clock, expected_session_seconds=1.5)


def test_provider_error_is_redacted_before_durable_state():
    error = RuntimeError(
        "authorization=Bearer-secret token=private https://signed.example/path?key=x"
    )

    redacted = sanitize_provider_error(error)

    assert "Bearer-secret" not in redacted
    assert "private" not in redacted
    assert "signed.example" not in redacted
