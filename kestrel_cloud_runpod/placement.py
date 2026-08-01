"""Deterministic GPU placement from live Runpod v2 catalog offers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from .models import (
    Availability,
    ComputeProduct,
    GPUOffer,
    PlacementDecision,
    PlacementRequirements,
    RunPodManagerError,
)

_AVAILABILITY_RANK = {
    Availability.NONE: 0,
    Availability.LOW: 1,
    Availability.MEDIUM: 2,
    Availability.HIGH: 3,
}


def select_gpu(
    offers: Iterable[GPUOffer],
    requirements: PlacementRequirements,
    *,
    observed_at: datetime | None = None,
) -> PlacementDecision:
    """Choose the highest-availability, lowest-cost compatible catalog offer."""

    compatible = [offer for offer in offers if _is_compatible(offer, requirements)]
    if not compatible:
        raise RunPodManagerError(
            "No live Runpod GPU offer satisfies the workload's VRAM, CUDA, "
            "availability, cloud, data-center, pool, and cost constraints"
        )
    compatible.sort(
        key=lambda offer: (
            -_AVAILABILITY_RANK.get(offer.availability or Availability.NONE, 0),
            offer.price_for(requirements.cloud),
            offer.memory_gb,
            offer.id,
        )
    )
    selected = compatible[0]
    return PlacementDecision(
        gpu_id=selected.id,
        gpu_pool=selected.pool,
        gpu_name=selected.name,
        memory_gb=selected.memory_gb,
        cloud=requirements.cloud,
        gpu_count=requirements.gpu_count,
        offered_cost_per_hr=selected.price_for(requirements.cloud),
        availability=selected.availability,
        catalog_observed_at=observed_at or datetime.now(timezone.utc),
        requirements=requirements,
    )


def _is_compatible(offer: GPUOffer, requirements: PlacementRequirements) -> bool:
    if offer.memory_gb < requirements.min_vram_gb:
        return False
    if (
        requirements.min_cuda_version is not None
        and offer.availability_min_cuda_version != requirements.min_cuda_version
    ):
        # The v2 catalog applies minCudaVersion to its product-specific
        # availability calculation but does not echo CUDA capability per GPU.
        # Accept only offers carrying provenance from that exact filtered query.
        return False
    if not offer.supports_cloud(requirements.cloud):
        return False
    # ``maxCount`` describes Pod capacity. Serverless placement is pool-based;
    # its product-specific stock signal is the availability expansion returned
    # for product=SERVERLESS (MIG offers can legitimately report maxCount=0).
    if (
        requirements.product is ComputeProduct.POD
        and offer.max_count_for(requirements.cloud) < requirements.gpu_count
    ):
        return False
    if requirements.product is ComputeProduct.SERVERLESS and not offer.pool:
        return False
    if requirements.allowed_gpu_ids and offer.id not in requirements.allowed_gpu_ids:
        return False
    if (
        requirements.allowed_gpu_pools
        and offer.pool not in requirements.allowed_gpu_pools
    ):
        return False
    price = offer.price_for(requirements.cloud)
    if (
        requirements.max_cost_per_hr is not None
        and price > requirements.max_cost_per_hr
    ):
        return False
    if (
        _AVAILABILITY_RANK.get(offer.availability or Availability.NONE, 0)
        < _AVAILABILITY_RANK[requirements.minimum_availability]
    ):
        return False
    if requirements.allowed_data_center_ids:
        allowed = set(requirements.allowed_data_center_ids)
        if not any(
            item.get("id") in allowed
            and item.get("availability") != Availability.NONE.value
            for item in offer.data_centers
        ):
            return False
    return True
