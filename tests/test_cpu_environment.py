"""Production v2 CPU planning boundary with synthetic provider traffic only."""

from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
import json
import tomllib

import httpx
from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from kestrel_sdk.computer_use import EnvironmentRequest
from kestrel_sdk.resources import ResourceScope, ResourceUnsupported
from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.cpu_environment import (
    CpuEnvironmentPlanner,
    CpuEnvironmentProfile,
    CpuNonComputeCostPolicy,
    _duration_cost,
)
from kestrel_cloud_runpod.models import RunPodManagerError
from scripts.check_runpod_openapi import load_openapi

NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)
IMAGE = "registry.example/browser@sha256:" + "a" * 64


def profile(**changes):
    return CpuEnvironmentProfile(
        **(
            dict(
                provider_id="runpod-cpu",
                profile_id="cpu-browser",
                image=IMAGE,
                cpu_id="cpu5c",
                vcpu_count=4,
                data_center_id="US-TX-3",
                disk_gb=20,
                max_lifetime_seconds=3600,
                cleanup_grace_seconds=60,
            )
            | changes
        )
    )


def policy(*, priced_profile=None, **changes):
    priced = priced_profile or profile()
    return CpuNonComputeCostPolicy(
        **(
            dict(
                policy_id="explicit-estimate-v1",
                observed_at=NOW - timedelta(minutes=1),
                provider_id=priced.provider_id,
                profile_id=priced.profile_id,
                profile_sha256=priced.digest(),
                disk_gb=priced.disk_gb,
                data_center_id=priced.data_center_id,
                covered_components=frozenset({"container_disk", "transfer", "cleanup"}),
                expires_at=NOW + timedelta(hours=2),
                disk_transfer_cleanup_hourly_usd=Decimal("0.02"),
                disk_transfer_cleanup_fixed_usd=Decimal("0.01"),
            )
            | changes
        )
    )


def request(**changes):
    return EnvironmentRequest(
        **(
            dict(
                request_id="original",
                scope=ResourceScope(tenant_id="tenant", principal_id="owner", agent_id="agent"),
                provider_id="runpod-cpu",
                profile_id="cpu-browser",
                image=IMAGE,
                surface="browser",
                location="remote",
                region="US-TX-3",
                required_capabilities=frozenset({"screenshot", "click"}),
                requested_at=NOW,
                expires_at=NOW + timedelta(minutes=30),
                idle_ttl_seconds=60,
                ready_timeout_seconds=60,
                max_hourly_cost_usd=Decimal("0.20"),
                max_total_cost_usd=Decimal("0.11"),
            )
            | changes
        )
    )


def cpu(**changes):
    fixture = Path(__file__).parent / "fixtures/cpu_environment_catalog_v2.json"
    return json.loads(fixture.read_text())["cpus"][0] | changes


def make_plan(
    *, original=None, configured=None, costs=None, raw=None, clock=lambda: NOW, calls=None
):
    calls = [] if calls is None else calls

    def handler(http_request):
        calls.append(http_request)
        assert http_request.method == "GET" and http_request.url.path == "/v2/catalog/cpus"
        assert dict(http_request.url.params) == {
            "include": "AVAILABILITY",
            "product": "POD",
            "vcpuCount": "4",
        }
        return httpx.Response(200, json={"cpus": [cpu() if raw is None else raw]})

    with closing(
        RunpodControlPlaneClient(api_key="synthetic", http_transport=httpx.MockTransport(handler))
    ) as client:
        plan = CpuEnvironmentPlanner(client).plan(
            original or request(),
            configured or profile(),
            costs or policy(),
            clock=clock,
        )
        assert len(calls) == 1
        return plan


@pytest.fixture(autouse=True)
def stable_catalog_time(monkeypatch):
    from kestrel_cloud_runpod import clients

    class Clock:
        @staticmethod
        def now(_):
            return NOW

    monkeypatch.setattr(clients, "datetime", Clock)


def test_real_client_only_reads_catalog_and_emits_exact_cpu_intent_matching_pinned_schema():
    plan = make_plan()
    assert plan.catalog.compute_cost_per_hr == Decimal("0.16")
    assert plan.catalog.ram_gb == Decimal("10")
    assert plan.all_in_hourly_usd == Decimal("0.18")
    assert plan.estimated_total_usd == Decimal("0.103")  # 31min plus fixed reserve
    assert plan.expires_at == NOW + timedelta(seconds=30)
    intent = plan.pod_request(request(), profile(), policy(), at=NOW)
    with pytest.raises(TypeError):
        intent.env["UNADMITTED"] = "value"
    wire = intent.to_payload()
    assert wire == dict(
        name="kestrel-cpu-" + request().digest()[:32],
        image=IMAGE,
        cpu={"id": "cpu5c", "vcpuCount": 4},
        dataCenterIds=["US-TX-3"],
        disk=20,
        cloud="SECURE",
        ports=[],
        env={},
        globalNetworking=False,
    )
    document = load_openapi(
        (Path(__file__).resolve().parents[1] / "vendor/runpod-v2-openapi.yaml").read_text()
    )
    Draft202012Validator(
        {"$ref": "#/components/schemas/CreatePodRequest", "components": document["components"]}
    ).validate(wire)
    assert make_plan().digest() == plan.digest()


def test_cpu_catalog_fixture_matches_the_actual_pinned_response_schema():
    root = Path(__file__).resolve().parents[1]
    document = load_openapi((root / "vendor/runpod-v2-openapi.yaml").read_text())
    Draft202012Validator(
        {
            "$ref": "#/components/schemas/ListCpuTypesResponse",
            "components": document["components"],
        }
    ).validate({"cpus": [cpu()]})


@pytest.mark.parametrize(
    "changes",
    [
        {"provider_id": "other"},
        {"profile_id": "other"},
        {"image": "registry.example/browser@sha256:" + "b" * 64},
        {"region": None},
        {"region": "US-KS-2"},
        {"location": "local"},
        {"surface": "desktop"},
        {"required_capabilities": frozenset({"screenshot", "exec"})},
        {"required_capabilities": frozenset({"screenshot", "files"})},
        {"required_capabilities": frozenset({"screenshot", "dom"})},
        {"required_capabilities": frozenset({"screenshot", "clipboard"})},
        {"requested_at": NOW + timedelta(seconds=1)},
        {"expires_at": NOW + timedelta(hours=2)},
    ],
)
def test_exact_profile_destination_and_known_contract_prerequisites_are_required(changes):
    calls = []
    with pytest.raises(ResourceUnsupported):
        make_plan(original=request(**changes), calls=calls)
    assert not calls


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": NOW + timedelta(seconds=1)},
        {"expires_at": NOW},
        {"expires_at": NOW + timedelta(minutes=30, seconds=59)},
    ],
)
def test_noncompute_policy_must_be_current_and_cover_cleanup(changes):
    with pytest.raises(ResourceUnsupported):
        make_plan(costs=policy(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"max_hourly_cost_usd": Decimal("0.179999")},
        {"max_total_cost_usd": Decimal("0.102999")},
    ],
)
def test_disk_transfer_and_cleanup_costs_are_included_in_original_ceilings(changes):
    with pytest.raises(ResourceUnsupported, match="ceilings"):
        make_plan(original=request(**changes))


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (cpu(availability="NONE"), "CPU availability is unknown or insufficient"),
        (
            cpu(dataCenters=[{"id": "US-KS-2", "name": "Kansas", "availability": "HIGH"}]),
            "Every requested CPU datacenter must have sufficient known stock",
        ),
        (
            {key: value for key, value in cpu().items() if key != "dataCenters"},
            "Every requested CPU datacenter must have sufficient known stock",
        ),
        (
            cpu(dataCenters=[{"id": "US-TX-3", "name": "Texas", "availability": "NONE"}]),
            "Every requested CPU datacenter must have sufficient known stock",
        ),
        (cpu(vcpu={"min": 8, "max": 32}), "CPU flavor or count is not offered"),
    ],
)
def test_schema_valid_stock_region_or_shape_is_refused_for_its_intended_reason(raw, reason):
    root = Path(__file__).resolve().parents[1]
    document = load_openapi((root / "vendor/runpod-v2-openapi.yaml").read_text())
    Draft202012Validator(
        {
            "$ref": "#/components/schemas/ListCpuTypesResponse",
            "components": document["components"],
        }
    ).validate({"cpus": [raw]})
    with pytest.raises(RunPodManagerError, match=reason):
        make_plan(raw=raw)


def test_missing_required_cpu_price_does_not_become_an_unknown_zero_plan():
    # A malformed provider response is a distinct fail-closed case.
    with pytest.raises(RunPodManagerError, match="CPU Pod rate is unknown"):
        make_plan(raw=cpu(price={}))


def test_catalog_latency_counts_against_freshness_and_request_lifetime():
    times = iter([NOW, NOW + timedelta(seconds=31)])
    with pytest.raises(ResourceUnsupported, match="freshness"):
        make_plan(clock=lambda: next(times))


def test_plan_cannot_be_used_after_expiry_or_with_changed_original_bindings():
    plan = make_plan()
    for at in (
        NOW - timedelta(microseconds=1),
        NOW + timedelta(seconds=30),
        NOW.replace(tzinfo=None),
    ):
        with pytest.raises((ValueError, ResourceUnsupported)):
            plan.pod_request(request(), profile(), policy(), at=at)
    for original, configured, costs in (
        (request(request_id="changed"), profile(), policy()),
        (
            request(
                scope=ResourceScope(tenant_id="tenant", principal_id="other", agent_id="agent")
            ),
            profile(),
            policy(),
        ),
        (request(), profile(disk_gb=21), policy()),
        (request(), profile(cpu_id="other"), policy()),
        (request(), profile(), policy(policy_id="other")),
    ):
        with pytest.raises(ResourceUnsupported, match="original"):
            plan.pod_request(original, configured, costs, at=NOW)


def test_plan_digest_binds_cpu_economics_placement_owner_and_policy_provenance():
    original = make_plan()
    larger = profile(disk_gb=21)
    for changed in (
        replace(original, request=request(request_id="other")),
        replace(original, profile=larger, noncompute=policy(priced_profile=larger)),
        replace(original, noncompute=policy(policy_id="other")),
        replace(original, catalog=replace(original.catalog, ram_gb=Decimal("11"))),
        replace(original, catalog=replace(original.catalog, compute_cost_per_hr=Decimal("0.17"))),
        replace(original, expires_at=NOW + timedelta(seconds=29)),
    ):
        assert changed.digest() != original.digest()
    offset = timezone(timedelta(hours=-5))
    assert (
        replace(
            original,
            noncompute=policy(
                observed_at=policy().observed_at.astimezone(offset),
                expires_at=policy().expires_at.astimezone(offset),
            ),
        ).digest()
        == original.digest()
    )


def test_exact_cost_arithmetic_is_independent_of_ambient_precision_and_rounds_up():
    with localcontext() as context:
        context.prec = 2
        plan = make_plan()
        assert plan.all_in_hourly_usd == Decimal("0.18")
        assert plan.estimated_total_usd == Decimal("0.103")
        fractional = make_plan(
            original=request(expires_at=NOW + timedelta(minutes=30, microseconds=1))
        )
        assert fractional.estimated_total_usd == Decimal("0.10300000005")


def test_strict_profiles_and_explicit_costs_have_no_unknown_zero_defaults():
    for changes in (
        {"vcpu_count": True},
        {"vcpu_count": 3},
        {"image": "mutable:latest"},
        {"disk_gb": 0},
    ):
        with pytest.raises(ValidationError):
            profile(**changes)
    for field in ("disk_transfer_cleanup_hourly_usd", "disk_transfer_cleanup_fixed_usd"):
        values = policy().model_dump()
        values.pop(field)
        with pytest.raises(ValidationError):
            CpuNonComputeCostPolicy(**values)
    for value in (Decimal("NaN"), Decimal("Infinity"), Decimal("-1")):
        with pytest.raises(ValidationError):
            policy(disk_transfer_cleanup_hourly_usd=value)


def test_plan_cannot_embed_mismatched_catalog_facts_or_extend_freshness():
    original = make_plan()
    for changes in (
        {"cpu_id": "other"},
        {"vcpu_count": 8},
        {"data_center_ids": ("US-KS-2",)},
        {"ram_gb": Decimal("0")},
        {"compute_cost_per_hr": Decimal("NaN")},
    ):
        with pytest.raises(ValueError):
            replace(original, catalog=replace(original.catalog, **changes))
    with pytest.raises(ValueError):
        replace(original, expires_at=NOW + timedelta(seconds=61))


@pytest.mark.parametrize("age", [timedelta(0), timedelta(seconds=-1), timedelta(seconds=61), 30])
def test_catalog_freshness_configuration_is_bounded(age):
    with pytest.raises(ValueError):
        CpuEnvironmentPlanner(object(), catalog_max_age=age)


def test_no_fake_provider_registration_or_cpu_custody_store():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert not any(
        group in pyproject["project"]["entry-points"]
        for group in (
            "kestrel_sovereign.environment_providers",
            "kestrel_sovereign.computer_drivers",
        )
    )
    assert not hasattr(CpuEnvironmentPlanner, "acquire")
    assert not hasattr(CpuEnvironmentPlanner, "release")
    assert not hasattr(CpuEnvironmentProfile, "pod_request")
    assert "kestrel-sovereign-sdk>=0.43.0,<1" in pyproject["project"]["dependencies"]
    assert "kestrel-sovereign>=0.54.4,<1" in pyproject["project"]["dependencies"]


def test_request_identity_uses_exact_sdk_digest_even_for_equal_decimal_values():
    original = make_plan()
    changed = request(max_hourly_cost_usd=Decimal("0.2"), max_total_cost_usd=Decimal("0.110"))
    assert changed == original.request and changed.digest() != original.request.digest()
    with pytest.raises(ResourceUnsupported, match="original"):
        original.pod_request(changed, profile(), policy(), at=NOW)
    assert original.pod_request(request(), profile(), policy(), at=NOW).name == (
        "kestrel-cpu-" + original.request.digest()[:32]
    )


def test_final_intent_emitter_receives_the_stored_original_objects(monkeypatch):
    plan = make_plan()
    received = []
    emitter = CpuEnvironmentProfile._pod_request

    def capture(instance, original):
        received.append((instance, original))
        return emitter(instance, original)

    monkeypatch.setattr(CpuEnvironmentProfile, "_pod_request", capture)
    clone = EnvironmentRequest.model_validate_json(request().model_dump_json())
    assert clone is not plan.request and clone.digest() == plan.request.digest()
    plan.pod_request(clone, profile(), policy(), at=NOW)
    assert len(received) == 1
    assert received[0][0] is plan.profile and received[0][1] is plan.request


@pytest.mark.parametrize(
    "changes",
    [
        {"provider_id": "other"},
        {"profile_id": "other"},
        {"profile_sha256": "b" * 64},
        {"disk_gb": 2000},
        {"data_center_id": "EU-RO-1"},
    ],
)
def test_noncompute_policy_prices_exactly_the_bound_profile(changes):
    with pytest.raises(ResourceUnsupported, match="another profile"):
        make_plan(costs=policy(**changes))
    with pytest.raises(ResourceUnsupported, match="another profile"):
        make_plan(configured=profile(disk_gb=2000))


def test_cost_policy_requires_all_components_and_prices_numeric_identity_canonically():
    with pytest.raises(ValidationError, match="explicitly cover"):
        policy(covered_components=frozenset({"container_disk", "transfer"}))
    copied = policy().model_copy(update={"covered_components": frozenset({"container_disk"})})
    with pytest.raises(ValidationError, match="explicitly cover"):
        make_plan(costs=copied)
    equivalent = policy(
        disk_transfer_cleanup_hourly_usd=Decimal("0.020"),
        disk_transfer_cleanup_fixed_usd=Decimal("0.01000"),
    )
    assert policy().digest() == equivalent.digest()
    original = make_plan()
    assert replace(original, noncompute=equivalent).digest() == original.digest()
    assert original.pod_request(request(), profile(), equivalent, at=NOW) == (
        original.pod_request(request(), profile(), policy(), at=NOW)
    )


def test_fractional_cost_rounds_up_and_exact_cost_ceiling_equality_is_accepted():
    assert _duration_cost(Decimal("0.01"), timedelta(microseconds=1)) == Decimal("3E-12")
    original = request(max_hourly_cost_usd=Decimal("0.18"), max_total_cost_usd=Decimal("0.103"))
    plan = make_plan(original=original)
    assert plan.pod_request(original, profile(), policy(), at=NOW).disk_gb == 20


def test_half_open_catalog_expiry_has_a_domain_error_at_the_exact_boundary():
    times = iter([NOW, NOW + timedelta(seconds=30)])
    with pytest.raises(ResourceUnsupported, match="freshness"):
        make_plan(clock=lambda: next(times))


@pytest.mark.parametrize("initial", [NOW + timedelta(seconds=1), NOW - timedelta(seconds=1)])
def test_catalog_observation_must_belong_to_the_same_api_call_clock_window(initial):
    times = iter([initial, initial])
    with pytest.raises(ResourceUnsupported, match="clock"):
        make_plan(
            original=request(requested_at=NOW - timedelta(minutes=1)), clock=lambda: next(times)
        )


def test_plan_expiry_reserves_full_readiness_time_and_never_widens_request_ttl():
    short = request(expires_at=NOW + timedelta(seconds=61))
    plan = make_plan(original=short)
    assert plan.expires_at == NOW + timedelta(seconds=1)
    with pytest.raises(ResourceUnsupported, match="readiness"):
        plan.pod_request(short, profile(), policy(), at=NOW + timedelta(seconds=1))
    times = iter([NOW, NOW + timedelta(seconds=1)])
    with pytest.raises(ResourceUnsupported, match="readiness"):
        make_plan(original=short, clock=lambda: next(times))


def test_takeover_is_only_a_logical_controller_fence_not_manual_transport_support():
    plan = make_plan(original=request(required_capabilities=frozenset({"screenshot", "takeover"})))
    assert plan.request.required_capabilities == {"screenshot", "takeover"}
    with pytest.raises(ResourceUnsupported, match="accessibility"):
        make_plan(
            original=request(required_capabilities=frozenset({"screenshot", "accessibility"}))
        )


@pytest.mark.parametrize("kind", ["request", "profile", "noncompute"])
def test_unchecked_copies_are_revalidated_at_plan_construction(kind):
    plan = make_plan()
    if kind == "request":
        changes = {"request": request().model_copy(update={"max_total_cost_usd": Decimal("-1")})}
    elif kind == "profile":
        forged = profile().model_copy(update={"cleanup_grace_seconds": -1800})
        changes = {"profile": forged, "noncompute": policy(priced_profile=forged)}
    else:
        changes = {
            "noncompute": policy().model_copy(
                update={"disk_transfer_cleanup_hourly_usd": Decimal("-0.16")}
            )
        }
    with pytest.raises(ValidationError):
        replace(plan, **changes)


@pytest.mark.parametrize("kind", ["request", "profile", "noncompute"])
def test_unchecked_copies_are_rejected_before_any_catalog_read(kind):
    original, configured, costs = request(), profile(), policy()
    if kind == "request":
        original = original.model_copy(update={"required_capabilities": frozenset({"click"})})
    elif kind == "profile":
        configured = configured.model_copy(update={"cleanup_grace_seconds": -1800})
        costs = policy(priced_profile=configured)
    else:
        costs = costs.model_copy(update={"disk_transfer_cleanup_hourly_usd": Decimal("-0.16")})
    calls = []
    with pytest.raises(ValidationError):
        make_plan(original=original, configured=configured, costs=costs, calls=calls)
    assert not calls


def test_exact_lifetime_cleanup_and_policy_observation_limits_are_inclusive():
    original = request(expires_at=NOW + timedelta(hours=1), max_total_cost_usd=Decimal("0.20"))
    costs = policy(
        observed_at=NOW,
        expires_at=original.expires_at + timedelta(seconds=profile().cleanup_grace_seconds),
    )
    plan = make_plan(original=original, costs=costs)
    assert plan.request.expires_at - plan.request.requested_at == timedelta(
        seconds=plan.profile.max_lifetime_seconds
    )
    assert plan.noncompute.observed_at == NOW
    assert plan.pod_request(original, profile(), costs, at=NOW).disk_gb == 20


def test_naive_snapshot_timestamp_refuses_with_a_domain_error():
    from kestrel_cloud_runpod.cpu_catalog import CpuCatalogSnapshot
    from kestrel_cloud_runpod.models import ComputeProduct

    class Reader:
        def list_cpus(self, **kwargs):
            return CpuCatalogSnapshot(ComputeProduct.POD, 4, NOW.replace(tzinfo=None), ())

    with pytest.raises(ResourceUnsupported, match="timezone-aware"):
        CpuEnvironmentPlanner(Reader()).plan(request(), profile(), policy(), clock=lambda: NOW)


def test_actual_production_cpu_query_values_validate_against_the_pinned_parameters():
    calls = []
    make_plan(calls=calls)
    root = Path(__file__).resolve().parents[1]
    document = load_openapi((root / "vendor/runpod-v2-openapi.yaml").read_text())
    operation = document["paths"]["/v2/catalog/cpus"]["get"]
    parameters = [
        document["components"]["parameters"][item["$ref"].rsplit("/", 1)[1]]
        for item in operation["parameters"]
    ]
    sent = dict(calls[0].url.params)
    assert set(sent) == {item["name"] for item in parameters}
    for parameter in parameters:
        assert parameter["in"] == "query"
        wire = sent[parameter["name"]]
        value = wire.split(",") if parameter["schema"]["type"] == "array" else int(wire)
        Draft202012Validator(
            {**parameter["schema"], "components": document["components"]}
        ).validate(value)
