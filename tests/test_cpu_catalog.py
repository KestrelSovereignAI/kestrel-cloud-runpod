"""Read-only CPU discovery and matching; all provider traffic is synthetic."""

from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.cpu_catalog import CpuCatalogSnapshot, CpuOffer
from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    ComputeProduct,
    CpuPodCreateRequest,
    RunPodManagerError,
)
from scripts.check_runpod_openapi import load_openapi

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


def offer(**changes):
    return {
        "id": "cpu5c",
        "name": "Compute",
        "group": "Gen 5",
        "vcpu": {"min": 2, "max": 32},
        "ramGbPerVcpu": 2.5,
        "price": {"securePerVcpu": 0.04, "serverlessPerVcpu": 0.03},
        "availability": "HIGH",
        "dataCenters": [
            {"id": "US-TX-3", "name": "Texas", "availability": "MEDIUM"},
            {"id": "US-KS-2", "name": "Kansas", "availability": "LOW"},
        ],
        **changes,
    }


def request(**changes):
    return CpuPodCreateRequest(
        **{
            "name": "cpu",
            "image": "image",
            "cpu_id": "cpu5c",
            "vcpu_count": 4,
            "data_center_ids": ("US-TX-3",),
            **changes,
        }
    )


def snapshot(*, raw=None, product=ComputeProduct.POD, count=4, at=NOW):
    return CpuCatalogSnapshot.from_response(
        {"cpus": [offer() if raw is None else raw]},
        product=product,
        vcpu_count=count,
        observed_at=at,
    )


def match(catalog=None, pod_request=None, **changes):
    return (catalog or snapshot()).match_pod(
        pod_request or request(),
        **{"now": NOW, "max_age": timedelta(seconds=30), **changes},
    )


def test_cpu_catalog_fixture_matches_pinned_schema():
    root = Path(__file__).resolve().parents[1]
    document = load_openapi((root / "vendor/runpod-v2-openapi.yaml").read_text())
    Draft202012Validator(
        {
            "$ref": "#/components/schemas/ListCpuTypesResponse",
            "components": document["components"],
        }
    ).validate({"cpus": [offer()]})
    result = match()
    assert result.cpu_id == "cpu5c" and result.vcpu_count == 4
    assert result.ram_gb == Decimal(10) and result.compute_cost_per_hr == Decimal(
        "0.16"
    )
    assert result.data_center_ids == ("US-TX-3",) and result.observed_at == NOW


@pytest.mark.parametrize("product", [ComputeProduct.POD, ComputeProduct.SERVERLESS])
def test_query_binds_product_count_and_observation_time(product):
    calls = []

    def handler(req):
        calls.append(req)
        assert req.method == "GET" and req.url.path == "/v2/catalog/cpus"
        assert dict(req.url.params) == {
            "include": "AVAILABILITY",
            "product": product.value,
            "vcpuCount": "4",
        }
        return httpx.Response(200, json={"cpus": [offer()]})

    before = datetime.now(UTC)
    with closing(
        RunpodControlPlaneClient(
            api_key="synthetic", http_transport=httpx.MockTransport(handler)
        )
    ) as client:
        catalog = client.list_cpus(product=product, vcpu_count=4)
    assert len(calls) == 1 and catalog.product is product and catalog.vcpu_count == 4
    assert before <= catalog.observed_at <= datetime.now(UTC)
    assert catalog.offers[0].serverless_price_per_vcpu == Decimal("0.03")


@pytest.mark.parametrize(
    "product,count",
    [
        (None, 4),
        ("POD", 4),
        ((ComputeProduct.POD, ComputeProduct.SERVERLESS), 4),
        (ComputeProduct.POD, 1),
        (ComputeProduct.POD, 3),
        (ComputeProduct.POD, True),
        (ComputeProduct.POD, 4.0),
    ],
)
def test_invalid_query_never_reaches_transport(product, count):
    calls = []
    with (
        closing(
            RunpodControlPlaneClient(
                api_key="synthetic",
                http_transport=httpx.MockTransport(lambda req: calls.append(req)),
            )
        ) as client,
        pytest.raises((ValueError, TypeError)),
    ):
        client.list_cpus(product=product, vcpu_count=count)
    assert not calls


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"id": "x"},
        offer(vcpu={"min": True, "max": 8}),
        offer(vcpu={"min": 2, "max": 1}),
        offer(vcpu={"min": 2, "max": "8"}),
        offer(ramGbPerVcpu=0),
        offer(ramGbPerVcpu=None),
        offer(ramGbPerVcpu=float("nan")),
        offer(price={"securePerVcpu": -1}),
        offer(price={"securePerVcpu": True}),
        offer(price={"securePerVcpu": "0.04"}),
        offer(price={"securePerVcpu": float("inf")}),
        offer(availability="invalid"),
        offer(dataCenters=None),
        offer(dataCenters=[{}]),
        offer(dataCenters=[{"id": "x", "availability": "invalid"}]),
        offer(dataCenters=[{"id": "x"}, {"id": "x"}]),
    ],
)
def test_malformed_catalog_is_refused_without_private_echo(raw):
    with pytest.raises(RunPodManagerError, match="CPU catalog"):
        CpuOffer.from_dict(raw)


@pytest.mark.parametrize(
    "raw",
    [{}, {"cpus": None}, {"cpus": {}}, {"cpus": [None]}, {"cpus": [offer(), offer()]}],
)
def test_malformed_snapshot_refused(raw):
    with pytest.raises(RunPodManagerError):
        CpuCatalogSnapshot.from_response(
            raw, product=ComputeProduct.POD, vcpu_count=4, observed_at=NOW
        )


def test_empty_catalog_remains_empty():
    catalog = CpuCatalogSnapshot.from_response(
        {"cpus": []}, product=ComputeProduct.POD, vcpu_count=4, observed_at=NOW
    )
    assert catalog.offers == ()
    with pytest.raises(RunPodManagerError):
        match(catalog)


@pytest.mark.parametrize(
    "raw", [offer(price={}), offer(price={"serverlessPerVcpu": 0.03})]
)
def test_missing_pod_price_does_not_borrow_serverless_or_zero(raw):
    catalog = snapshot(raw=raw)
    assert catalog.offers[0].secure_price_per_vcpu is None
    with pytest.raises(RunPodManagerError, match="rate is unknown"):
        match(catalog)


def test_actual_zero_rate_is_preserved():
    assert match(
        snapshot(raw=offer(price={"securePerVcpu": 0}))
    ).compute_cost_per_hr == Decimal(0)


@pytest.mark.parametrize(
    "changes",
    [
        {"cpu_id": "other"},
        {"vcpu_count": 8},
        {"cloud": CloudType.COMMUNITY},
        {"data_center_ids": ()},
        {"data_center_ids": ("elsewhere",)},
        {"data_center_ids": ("US-TX-3", "elsewhere")},
    ],
)
def test_request_cannot_reuse_stock_for_another_flavor_count_cloud_or_region(changes):
    with pytest.raises(RunPodManagerError):
        match(pod_request=request(**changes))


def test_serverless_stock_is_not_pod_stock():
    with pytest.raises(RunPodManagerError, match="query does not match"):
        match(snapshot(product=ComputeProduct.SERVERLESS))


def test_requested_count_must_be_inside_current_flavor_bounds():
    with pytest.raises(RunPodManagerError, match="not offered"):
        match(snapshot(raw=offer(vcpu={"min": 8, "max": 32})))
    assert match(snapshot(raw=offer(vcpu={"min": 3, "max": 6}))).vcpu_count == 4


@pytest.mark.parametrize(
    "raw",
    [
        offer(availability=None),
        offer(availability="NONE"),
        offer(dataCenters=[]),
        offer(dataCenters=[{"id": "US-TX-3"}]),
        offer(dataCenters=[{"id": "US-TX-3", "availability": "NONE"}]),
    ],
)
def test_missing_or_absent_stock_is_not_deployable(raw):
    with pytest.raises(RunPodManagerError):
        match(snapshot(raw=raw))


def test_minimum_availability_applies_to_overall_and_each_requested_region():
    with pytest.raises(RunPodManagerError):
        match(minimum_availability=Availability.HIGH)
    with pytest.raises(RunPodManagerError):
        match(
            snapshot(raw=offer(availability="LOW")),
            minimum_availability=Availability.MEDIUM,
        )
    assert match(minimum_availability=Availability.MEDIUM).data_center_ids == (
        "US-TX-3",
    )
    with pytest.raises(RunPodManagerError):
        match(
            pod_request=request(data_center_ids=("US-TX-3", "US-KS-2")),
            minimum_availability=Availability.MEDIUM,
        )


@pytest.mark.parametrize(
    "at", [NOW - timedelta(seconds=31), NOW + timedelta(seconds=1)]
)
def test_stale_or_future_catalog_is_refused(at):
    with pytest.raises(RunPodManagerError, match="stale or from the future"):
        match(snapshot(at=at))


def test_snapshot_age_boundary_and_timezone_normalization():
    shifted = NOW.astimezone(timezone(timedelta(hours=-5)))
    assert snapshot(at=shifted).observed_at == NOW
    assert match(
        snapshot(at=NOW - timedelta(seconds=30))
    ).observed_at == NOW - timedelta(seconds=30)


@pytest.mark.parametrize(
    "changes",
    [
        {"now": NOW.replace(tzinfo=None)},
        {"max_age": timedelta(0)},
        {"minimum_availability": Availability.NONE},
        {"minimum_availability": "LOW"},
    ],
)
def test_invalid_matching_policy_is_refused(changes):
    with pytest.raises(ValueError):
        match(**changes)


def test_snapshot_copies_provider_owned_mutable_data():
    raw = deepcopy(offer())
    catalog = snapshot(raw=raw)
    raw["price"]["securePerVcpu"] = 99
    raw["dataCenters"][0]["availability"] = "NONE"
    assert match(catalog).compute_cost_per_hr == Decimal("0.16")


def test_compute_cost_and_ram_are_independent_of_ambient_decimal_precision():
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 2
        result = match(
            snapshot(raw=offer(ramGbPerVcpu=2.12345, price={"securePerVcpu": 0.12345}))
        )
    assert result.compute_cost_per_hr == Decimal("0.49380")
    assert result.ram_gb == Decimal("8.49380")


def test_catalog_time_includes_request_latency(monkeypatch):
    from kestrel_cloud_runpod import clients

    clock = [NOW]

    class Clock:
        @staticmethod
        def now(_):
            return clock[0]

    monkeypatch.setattr(clients, "datetime", Clock)

    def handler(_):
        clock[0] += timedelta(seconds=40)
        return httpx.Response(200, json={"cpus": [offer()]})

    with closing(
        RunpodControlPlaneClient(
            api_key="synthetic", http_transport=httpx.MockTransport(handler)
        )
    ) as client:
        catalog = client.list_cpus(product=ComputeProduct.POD, vcpu_count=4)
    assert catalog.observed_at == NOW
    with pytest.raises(RunPodManagerError, match="stale"):
        match(catalog, now=clock[0])
