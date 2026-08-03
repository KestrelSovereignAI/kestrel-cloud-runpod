"""Runpod v2 quote, exact recovery, termination, and billing adapter tests."""

import hashlib
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pod_capacity_test_support import (
    PARAMETERS_SHA,
    MutableClock,
    profile,
    request,
)

from kestrel_cloud_runpod.models import (
    Availability,
    BillingPage,
    GPUOffer,
    RunPodManagerError,
)
from kestrel_cloud_runpod.pod_capacity_contracts import (
    PodCapacityQuoteRequest,
    PodCapacitySpec,
    attempt_environment_sha256,
)
from kestrel_cloud_runpod.pod_capacity_provider import (
    PodCapacityCreatedMismatchError,
    RunpodPodCapacityProvider,
)


def _offer() -> GPUOffer:
    return GPUOffer(
        id="NVIDIA RTX PRO 4500",
        name="RTX PRO 4500",
        pool=None,
        manufacturer="NVIDIA",
        memory_gb=32,
        secure=True,
        community=False,
        secure_price_per_hr=0.4,
        community_price_per_hr=0.0,
        secure_max_count=8,
        community_max_count=0,
        availability=Availability.HIGH,
        data_centers=({"id": "US-TX-3", "availability": "HIGH"},),
        availability_min_cuda_version="12.8",
    )


def _spec(clock: MutableClock) -> PodCapacitySpec:
    item = request(clock)
    return PodCapacitySpec(
        request=item,
        capability_secret_id="secret:catalog-capability-0001",
        capability_token_sha256="a" * 64,
        capability_expires_at=item.bearer_expires_at,
        attempt_environment_sha256=attempt_environment_sha256(item.attempt_environment),
    )


@pytest.mark.asyncio
async def test_quote_binds_parameter_digest_exact_gpu_rate_and_timing() -> None:
    clock = MutableClock()
    client = SimpleNamespace(list_gpus=lambda **_: [_offer()])
    direct = SimpleNamespace(client=client)
    provider = RunpodPodCapacityProvider(direct, clock=clock)

    result = await provider.quote(
        PodCapacityQuoteRequest(
            constraints=request(clock).quote.constraints,
            workload_kind="catalog-lora",
            parameters_sha256=PARAMETERS_SHA,
            estimated_startup_seconds=60,
            estimated_execution_seconds=300,
            maximum_runtime_seconds=600,
        )
    )

    assert result.gpu_type_id == "NVIDIA RTX PRO 4500"
    assert result.gpu_display_name == "RTX PRO 4500"
    assert result.hourly_cost_usd == Decimal("0.4")
    assert result.estimated_cost_usd == Decimal("0.040000")
    assert result.cost_ceiling_usd == Decimal("0.066667")
    assert result.parameters_sha256 == PARAMETERS_SHA
    assert result.provider_quote_id.startswith("runpod-pod:")


@pytest.mark.asyncio
async def test_create_injects_exact_image_gpu_and_attempt_environment() -> None:
    clock = MutableClock()
    spec = _spec(clock)
    captured = {}

    def start_pod(effective_profile, metadata):
        captured.update(profile=effective_profile, metadata=metadata)
        return {
            "id": "pod-catalog-1",
            "name": spec.request.resource_name,
            "image": spec.request.image_reference,
            "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
            "cloud": "SECURE",
            "dataCenterId": "US-TX-3",
            "cost": 0.4,
            "_kestrel_placement": spec.request.quote.placement,
        }

    direct = SimpleNamespace(start_pod=start_pod)
    provider = RunpodPodCapacityProvider(direct, clock=clock)
    environment = {"CATALOG_WORKER_MODE": "pod"}

    created = await provider.create(
        profile=profile(),
        resource_name=spec.request.resource_name,
        companion_id=spec.request.owner_id,
        environment=environment,
        capacity_spec=spec,
    )

    assert created.provider_pod_id == "pod-catalog-1"
    assert captured["profile"].image_name == spec.request.image_reference
    assert captured["profile"].allowed_gpu_ids == (spec.request.quote.gpu_type_id,)
    assert captured["metadata"]["env_overrides"] == environment


@pytest.mark.asyncio
async def test_created_metadata_mismatch_preserves_known_pod_id() -> None:
    clock = MutableClock()
    spec = _spec(clock)

    def start_pod(effective_profile, metadata):
        return {
            "id": "pod-catalog-mismatch",
            "name": spec.request.resource_name,
            "image": "wrong@sha256:" + "f" * 64,
            "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
            "cloud": "SECURE",
            "_kestrel_placement": spec.request.quote.placement,
        }

    provider = RunpodPodCapacityProvider(
        SimpleNamespace(start_pod=start_pod), clock=clock
    )

    with pytest.raises(PodCapacityCreatedMismatchError) as raised:
        await provider.create(
            profile=profile(),
            resource_name=spec.request.resource_name,
            companion_id=spec.request.owner_id,
            environment={"CATALOG_WORKER_MODE": "pod"},
            capacity_spec=spec,
        )

    assert raised.value.provider_pod_id == "pod-catalog-mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"dataCenterId": "EU-RO-1"}, "immutable placement"),
        ({"cost": 0.41}, "immutable placement"),
        ({"cloud": "COMMUNITY"}, "immutable placement"),
    ],
)
async def test_create_rejects_realized_placement_outside_quote(
    override, expected_message
) -> None:
    clock = MutableClock()
    spec = _spec(clock)

    def start_pod(effective_profile, metadata):
        payload = {
            "id": "pod-catalog-mismatch",
            "name": spec.request.resource_name,
            "image": spec.request.image_reference,
            "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
            "cloud": "SECURE",
            "dataCenterId": "US-TX-3",
            "cost": 0.4,
            "_kestrel_placement": spec.request.quote.placement,
        }
        payload.update(override)
        return payload

    provider = RunpodPodCapacityProvider(
        SimpleNamespace(start_pod=start_pod), clock=clock
    )
    with pytest.raises(PodCapacityCreatedMismatchError, match=expected_message):
        await provider.create(
            profile=profile(),
            resource_name=spec.request.resource_name,
            companion_id=spec.request.owner_id,
            environment={"CATALOG_WORKER_MODE": "pod"},
            capacity_spec=spec,
        )


@pytest.mark.asyncio
async def test_exact_recovery_rejects_mismatch_and_multiple_matches() -> None:
    clock = MutableClock()
    spec = _spec(clock)
    valid = {
        "id": "pod-catalog-1",
        "name": spec.request.resource_name,
        "image": spec.request.image_reference,
        "gpu": {"id": spec.request.quote.gpu_type_id, "count": 1},
        "cloud": "SECURE",
        "dataCenterId": "US-TX-3",
        "cost": 0.4,
    }
    direct = SimpleNamespace(list_pods=lambda: [valid])
    provider = RunpodPodCapacityProvider(direct, clock=clock)
    recovered = await provider.find_exact(spec.request.resource_name, spec)
    assert recovered is not None
    assert recovered.provider_pod_id == "pod-catalog-1"
    assert recovered.realized_placement is not None
    assert recovered.realized_placement.data_center_id == "US-TX-3"

    direct.list_pods = lambda: [{**valid, "image": "wrong@sha256:" + "f" * 64}]
    with pytest.raises(RunPodManagerError, match="mismatched"):
        await provider.find_exact(spec.request.resource_name, spec)

    direct.list_pods = lambda: [valid, {**valid, "id": "pod-catalog-2"}]
    with pytest.raises(RunPodManagerError, match="Multiple"):
        await provider.find_exact(spec.request.resource_name, spec)


@pytest.mark.asyncio
async def test_final_billing_waits_for_records_then_returns_content_free_receipt() -> (
    None
):
    clock = MutableClock()
    spec = _spec(clock)
    terminated = clock()
    created = terminated - timedelta(minutes=10)
    pages = [
        BillingPage(
            records=(),
            metadata={
                "query": {
                    "podId": "pod-catalog-1",
                    "startTime": created.isoformat(),
                    "endTime": terminated.isoformat(),
                },
                "totals": {"totalAmount": 0},
            },
        ),
        BillingPage(
            records=(
                {
                    "podId": "pod-catalog-1",
                    "startTime": created.isoformat(),
                    "endTime": terminated.isoformat(),
                    "totalAmount": 0.061,
                },
            ),
            metadata={
                "query": {
                    "podId": "pod-catalog-1",
                    "startTime": created.isoformat(),
                    "endTime": terminated.isoformat(),
                },
                "totals": {"totalAmount": 0.061},
            },
        ),
    ]
    client = SimpleNamespace(pod_billing=lambda **_: pages.pop(0))
    provider = RunpodPodCapacityProvider(SimpleNamespace(client=client), clock=clock)

    assert (
        await provider.final_billing(
            "pod-catalog-1",
            capacity_spec=spec,
            created_at=created,
            terminated_at=terminated,
            realized_hourly_rate_usd=Decimal("0.4"),
        )
        is None
    )
    receipt = await provider.final_billing(
        "pod-catalog-1",
        capacity_spec=spec,
        created_at=created,
        terminated_at=terminated,
        realized_hourly_rate_usd=Decimal("0.4"),
    )
    assert receipt is not None
    assert receipt.actual_cost_usd == Decimal("0.061")
    assert receipt.billed_seconds == 600
    assert receipt.provider_billing_id.startswith("runpod-billing:")
    assert hashlib.sha256(b"private").hexdigest() not in str(receipt)


@pytest.mark.asyncio
async def test_final_billing_waits_until_records_cover_termination() -> None:
    clock = MutableClock()
    spec = _spec(clock)
    terminated = clock()
    created = terminated - timedelta(minutes=10)
    partial_end = terminated - timedelta(seconds=1)
    page = BillingPage(
        records=(
            {
                "podId": "pod-catalog-1",
                "startTime": created.isoformat(),
                "endTime": partial_end.isoformat(),
                "totalAmount": 0.06,
            },
        ),
        metadata={
            "query": {
                "podId": "pod-catalog-1",
                "startTime": created.isoformat(),
                "endTime": terminated.isoformat(),
            },
            "totals": {"totalAmount": 0.06},
        },
    )
    provider = RunpodPodCapacityProvider(
        SimpleNamespace(client=SimpleNamespace(pod_billing=lambda **_: page)),
        clock=clock,
    )

    assert (
        await provider.final_billing(
            "pod-catalog-1",
            capacity_spec=spec,
            created_at=created,
            terminated_at=terminated,
            realized_hourly_rate_usd=Decimal("0.4"),
        )
        is None
    )
