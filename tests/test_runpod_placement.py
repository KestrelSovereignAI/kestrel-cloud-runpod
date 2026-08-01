"""Live-catalog placement policy tests."""

from datetime import datetime, timezone

import pytest

from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    ComputeProduct,
    GPUOffer,
    PlacementRequirements,
    RunPodManagerError,
)
from kestrel_cloud_runpod.placement import select_gpu


def _offer(
    gpu_id: str,
    *,
    pool: str | None = "BLACKWELL_24",
    memory: int = 24,
    price: float = 0.69,
    availability: Availability = Availability.HIGH,
    max_count: int = 1,
    data_centers=(),
    availability_min_cuda_version: str | None = None,
) -> GPUOffer:
    return GPUOffer(
        id=gpu_id,
        name=gpu_id,
        pool=pool,
        manufacturer="NVIDIA",
        memory_gb=memory,
        secure=True,
        community=False,
        secure_price_per_hr=price,
        community_price_per_hr=0.0,
        secure_max_count=max_count,
        community_max_count=0,
        availability=availability,
        data_centers=tuple(data_centers),
        availability_min_cuda_version=availability_min_cuda_version,
    )


def test_placement_prefers_availability_then_live_price_and_records_snapshot():
    observed = datetime(2026, 8, 1, tzinfo=timezone.utc)
    requirements = PlacementRequirements(
        product=ComputeProduct.POD,
        min_vram_gb=24,
        cloud=CloudType.SECURE,
        max_cost_per_hr=1.5,
        benchmark_id="catalog-selfie",
    )
    decision = select_gpu(
        [
            _offer("cheap-low", price=0.4, availability=Availability.LOW),
            _offer("expensive-high", price=1.2),
            _offer("cheap-high", price=0.7),
        ],
        requirements,
        observed_at=observed,
    )

    assert decision.gpu_id == "cheap-high"
    assert decision.offered_cost_per_hr == 0.7
    assert decision.catalog_observed_at == observed
    assert decision.requirements.benchmark_id == "catalog-selfie"


def test_serverless_mig_uses_product_availability_not_pod_max_count():
    """MIG Serverless offers can be HIGH while maxCount.secure is zero."""

    requirements = PlacementRequirements(
        product=ComputeProduct.SERVERLESS,
        min_vram_gb=24,
        cloud=CloudType.SECURE,
    )
    decision = select_gpu(
        [_offer("PRO 6000 MIG 1g.24gb", pool="BLACKWELL_24", max_count=0)],
        requirements,
    )

    assert decision.gpu_id == "PRO 6000 MIG 1g.24gb"
    assert decision.gpu_pool == "BLACKWELL_24"


@pytest.mark.parametrize(
    "offer,requirements",
    [
        (
            _offer("too-small", memory=16),
            PlacementRequirements(product=ComputeProduct.POD, min_vram_gb=24),
        ),
        (
            _offer("too-expensive", price=2.0),
            PlacementRequirements(
                product=ComputeProduct.POD,
                min_vram_gb=24,
                max_cost_per_hr=1.0,
            ),
        ),
        (
            _offer("no-pool", pool=None),
            PlacementRequirements(product=ComputeProduct.SERVERLESS, min_vram_gb=24),
        ),
        (
            _offer("none", availability=Availability.NONE),
            PlacementRequirements(product=ComputeProduct.POD, min_vram_gb=24),
        ),
    ],
)
def test_placement_fails_closed_for_incompatible_offers(offer, requirements):
    with pytest.raises(RunPodManagerError, match="No live Runpod GPU offer"):
        select_gpu([offer], requirements)


def test_placement_enforces_allowed_data_centers():
    offer = _offer(
        "regional",
        data_centers=(
            {"id": "EU-RO-1", "availability": "HIGH"},
            {"id": "US-TX-3", "availability": "NONE"},
        ),
    )
    requirements = PlacementRequirements(
        product=ComputeProduct.POD,
        min_vram_gb=24,
        allowed_data_center_ids=("US-TX-3",),
    )

    with pytest.raises(RunPodManagerError):
        select_gpu([offer], requirements)


def test_placement_requires_cuda_filtered_catalog_provenance():
    requirements = PlacementRequirements(
        product=ComputeProduct.POD,
        min_vram_gb=24,
        min_cuda_version="12.8",
    )

    with pytest.raises(RunPodManagerError):
        select_gpu([_offer("unfiltered")], requirements)

    decision = select_gpu(
        [_offer("filtered", availability_min_cuda_version="12.8")],
        requirements,
    )
    assert decision.gpu_id == "filtered"
