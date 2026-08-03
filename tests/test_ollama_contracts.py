"""Cost selection and public-route contracts for Ollama leases."""

from dataclasses import replace

import pytest
from ollama_test_support import (
    MutableClock,
    make_decision,
    make_request,
    non_compute_cost_policies,
    non_compute_cost_policy,
)

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
        non_compute_cost_policies=non_compute_cost_policies(),
        planned_at=clock(),
        serverless_max_workers=1,
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
        non_compute_cost_policies=non_compute_cost_policies(),
        planned_at=clock(),
        serverless_max_workers=1,
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
            non_compute_cost_policies=non_compute_cost_policies(),
            planned_at=clock(),
            serverless_max_workers=1,
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
            estimated_compute_cost=0.1,
            maximum_compute_cost=0.1,
            estimated_non_compute_cost=0.0,
            maximum_non_compute_cost=0.0,
            estimated_cost=0.1,
            cost_ceiling=0.1,
            estimated_billable_seconds=100,
            maximum_billable_seconds=100,
            maximum_concurrent_workers=1,
            non_compute_components=non_compute_cost_policies()[
                OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            ].covered_components,
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
            estimated_compute_cost=0.1,
            maximum_compute_cost=0.1,
            estimated_non_compute_cost=0.0,
            maximum_non_compute_cost=0.0,
            estimated_cost=0.1,
            cost_ceiling=0.1,
            estimated_billable_seconds=100,
            maximum_billable_seconds=100,
            maximum_concurrent_workers=1,
            non_compute_components=non_compute_cost_policies()[
                OllamaLeaseMode.DEDICATED_POD
            ].covered_components,
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
        non_compute_cost_policies=non_compute_cost_policies(),
        planned_at=clock(),
        serverless_max_workers=1,
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
            non_compute_cost_policies=non_compute_cost_policies(),
            planned_at=clock(),
            serverless_max_workers=1,
        )


def test_auto_uses_all_in_ceiling_not_compute_only_affordability():
    clock = MutableClock()
    request = make_request(clock, max_authorized_cost=1.5)
    policies = non_compute_cost_policies()
    policies[OllamaLeaseMode.SERVERLESS_LOAD_BALANCER] = non_compute_cost_policy(
        estimated=0.01, maximum=0.6
    )

    plan = select_ollama_plan(
        request,
        {
            ComputeProduct.SERVERLESS: make_decision(
                ComputeProduct.SERVERLESS,
                rate=0.5,
                gpu_id="serverless",
                pool="pool-24",
            ),
            ComputeProduct.POD: make_decision(
                ComputeProduct.POD,
                rate=0.5,
                gpu_id="pod",
                pool=None,
            ),
        },
        non_compute_cost_policies=policies,
        planned_at=clock(),
        serverless_max_workers=1,
    )

    assert plan.mode is OllamaLeaseMode.DEDICATED_POD
    assert plan.cost_ceiling == pytest.approx(1.0)


@pytest.mark.parametrize("extra_overhead, affordable", [(0.0, True), (1e-9, False)])
def test_all_in_ceiling_exact_boundary_and_tiny_overage(extra_overhead, affordable):
    clock = MutableClock()
    request = make_request(
        clock,
        mode=OllamaLeaseMode.DEDICATED_POD,
        max_authorized_cost=1.0,
    )
    policies = non_compute_cost_policies()
    policies[OllamaLeaseMode.DEDICATED_POD] = non_compute_cost_policy(
        estimated=0.0,
        maximum=extra_overhead,
    )
    arguments = {
        "request": request,
        "decisions": {
            ComputeProduct.POD: make_decision(
                ComputeProduct.POD,
                rate=0.5,
                gpu_id="pod",
                pool=None,
            )
        },
        "non_compute_cost_policies": policies,
        "planned_at": clock(),
        "serverless_max_workers": 1,
    }

    if affordable:
        plan = select_ollama_plan(**arguments)
        assert plan.cost_ceiling == 1.0
    else:
        with pytest.raises(RunPodManagerError, match="all-in|ceiling|authorized cost"):
            select_ollama_plan(**arguments)


@pytest.mark.parametrize(
    "mode, product",
    [
        (OllamaLeaseMode.SERVERLESS_LOAD_BALANCER, ComputeProduct.SERVERLESS),
        (OllamaLeaseMode.DEDICATED_POD, ComputeProduct.POD),
    ],
)
def test_missing_mode_cost_policy_fails_closed(mode, product):
    clock = MutableClock()
    pool = "pool-24" if product is ComputeProduct.SERVERLESS else None

    with pytest.raises(RunPodManagerError, match="cost policy is not configured"):
        select_ollama_plan(
            make_request(clock, mode=mode, max_authorized_cost=10.0),
            {
                product: make_decision(
                    product,
                    rate=0.5,
                    gpu_id="gpu",
                    pool=pool,
                )
            },
            non_compute_cost_policies={},
            planned_at=clock(),
            serverless_max_workers=1,
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


def test_pod_compute_authorizes_every_gpu_in_a_multi_gpu_placement():
    """A 4-GPU Pod must authorize 4x the per-GPU catalog rate.

    ``/catalog/gpus`` prices ``price.secure`` per GPU, so rating a multi-GPU
    Pod at the bare offered rate under-authorizes the plan by ``gpu_count``
    and lets a lease bill past its approved ceiling.
    """
    clock = MutableClock()
    request = make_request(clock, max_authorized_cost=10.0)
    single, multi = (
        select_ollama_plan(
            request,
            {
                ComputeProduct.POD: make_decision(
                    ComputeProduct.POD,
                    rate=0.6,
                    gpu_id="pod",
                    pool=None,
                    gpu_count=count,
                )
            },
            non_compute_cost_policies=non_compute_cost_policies(),
            planned_at=clock(),
            serverless_max_workers=1,
        )
        for count in (1, 4)
    )

    assert single.mode is OllamaLeaseMode.DEDICATED_POD
    assert multi.mode is OllamaLeaseMode.DEDICATED_POD
    # A Pod is one worker regardless of how many GPUs it attaches; the GPU
    # count must not be smuggled into the worker dimension.
    assert single.maximum_concurrent_workers == 1
    assert multi.maximum_concurrent_workers == 1
    assert multi.estimated_billable_seconds == single.estimated_billable_seconds
    assert multi.maximum_billable_seconds == single.maximum_billable_seconds

    assert single.estimated_compute_cost == pytest.approx(0.6)
    assert multi.estimated_compute_cost == pytest.approx(2.4)
    assert multi.maximum_compute_cost == pytest.approx(single.maximum_compute_cost * 4)


def test_serverless_gpu_count_and_worker_ceiling_stay_orthogonal():
    """GPUs-per-worker scales the rate; max workers scales billable seconds.

    Conflating the two would either double-count or hide one of them, so this
    pins that the factors compose independently.
    """
    clock = MutableClock()
    request = make_request(clock, max_authorized_cost=200.0)

    def plan_for(*, gpu_count: int, workers: int):
        return select_ollama_plan(
            request,
            {
                ComputeProduct.SERVERLESS: make_decision(
                    ComputeProduct.SERVERLESS,
                    rate=1.0,
                    gpu_id="serverless",
                    pool="pool-24",
                    gpu_count=gpu_count,
                )
            },
            non_compute_cost_policies=non_compute_cost_policies(),
            planned_at=clock(),
            serverless_max_workers=workers,
        )

    base = plan_for(gpu_count=1, workers=1)
    more_gpus = plan_for(gpu_count=3, workers=1)
    more_workers = plan_for(gpu_count=1, workers=5)

    # GPUs multiply the rate and leave billable seconds untouched.
    assert more_gpus.estimated_billable_seconds == base.estimated_billable_seconds
    assert more_gpus.maximum_concurrent_workers == 1
    assert more_gpus.estimated_compute_cost == pytest.approx(
        base.estimated_compute_cost * 3
    )

    # Workers multiply the billable ceiling and leave the rate untouched.
    assert more_workers.maximum_concurrent_workers == 5
    assert more_workers.estimated_compute_cost == pytest.approx(
        base.estimated_compute_cost
    )
    assert more_workers.maximum_billable_seconds > base.maximum_billable_seconds
