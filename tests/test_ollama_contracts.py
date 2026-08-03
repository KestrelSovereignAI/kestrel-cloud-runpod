"""Cost selection and public-route contracts for Ollama leases."""

from dataclasses import replace

import pytest
from ollama_test_support import MutableClock, make_decision, make_request

from kestrel_cloud_runpod.models import ComputeProduct, RunPodManagerError
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseMode,
    OllamaPlacementPlan,
    OllamaResourceType,
    canonical_model_name,
    maximum_serverless_cold_starts,
    sanitize_provider_error,
    select_ollama_plan,
)


def test_bursty_session_selects_lower_effective_serverless_cost():
    clock = MutableClock()
    request = make_request(clock, max_authorized_cost=5.0)
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
                ComputeProduct.POD, rate=4.0, gpu_id="pod", pool=None
            ),
        },
    )

    assert plan.mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
    assert plan.maximum_serverless_cold_starts == 121
    assert plan.estimated_billable_seconds == 11_190
    assert plan.estimated_cost == pytest.approx(11_190 / 3600)


def test_serverless_quote_covers_every_possible_scale_to_zero_cycle():
    clock = MutableClock()
    request = make_request(
        clock,
        expected_session_seconds=300,
        expected_active_seconds=60,
        serverless_initialization_seconds=20,
        serverless_idle_tail_seconds=60,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
    )

    plan = select_ollama_plan(
        request,
        {
            ComputeProduct.SERVERLESS: make_decision(
                ComputeProduct.SERVERLESS,
                rate=1.0,
                gpu_id="serverless",
                pool="pool-24",
            )
        },
    )

    assert plan.maximum_serverless_cold_starts == 6
    assert plan.estimated_billable_seconds == 540
    assert plan.estimated_cost == pytest.approx(540 / 3600)


def test_zero_idle_tail_has_no_finite_serverless_cold_start_bound():
    clock = MutableClock()
    request = make_request(
        clock,
        serverless_idle_tail_seconds=0,
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
    )

    assert (
        maximum_serverless_cold_starts(
            expected_session_seconds=300,
            idle_tail_seconds=0,
        )
        is None
    )
    with pytest.raises(RunPodManagerError, match="No Runpod Ollama mode"):
        select_ollama_plan(
            request,
            {
                ComputeProduct.SERVERLESS: make_decision(
                    ComputeProduct.SERVERLESS,
                    rate=1.0,
                    gpu_id="serverless",
                    pool="pool-24",
                )
            },
        )


def test_placement_plan_rejects_missing_or_cross_product_cold_start_bound():
    serverless = make_decision(
        ComputeProduct.SERVERLESS,
        rate=1.0,
        gpu_id="serverless",
        pool="pool-24",
    )
    with pytest.raises(ValueError, match="require a cold-start bound"):
        OllamaPlacementPlan(
            mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
            placement=serverless,
            estimated_cost=0.1,
            estimated_billable_seconds=100,
        )
    with pytest.raises(ValueError, match="cannot declare Serverless"):
        OllamaPlacementPlan(
            mode=OllamaLeaseMode.DEDICATED_POD,
            resource_type=OllamaResourceType.POD,
            placement=make_decision(
                ComputeProduct.POD,
                rate=1.0,
                gpu_id="pod",
                pool=None,
            ),
            estimated_cost=0.1,
            estimated_billable_seconds=100,
            maximum_serverless_cold_starts=1,
        )


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
