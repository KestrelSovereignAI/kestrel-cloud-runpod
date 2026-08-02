"""Opt-in, read-only authenticated smoke checks for the beta v2 API."""

import os

import pytest

from kestrel_cloud_runpod.clients import RunpodControlPlaneClient
from kestrel_cloud_runpod.models import ComputeProduct


@pytest.mark.cloud_resource
def test_live_v2_gpu_catalog_is_read_only_and_typed():
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        pytest.skip("RUNPOD_API_KEY is not configured")
    client = RunpodControlPlaneClient(api_key=api_key)
    try:
        offers = client.list_gpus(products=(ComputeProduct.SERVERLESS,))
    finally:
        client.close()

    assert offers
    assert all(offer.id and offer.memory_gb > 0 for offer in offers)


@pytest.mark.cloud_resource
def test_live_v2_network_volume_inventory_is_read_only_and_typed():
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        pytest.skip("RUNPOD_API_KEY is not configured")
    client = RunpodControlPlaneClient(api_key=api_key)
    try:
        volumes = client.list_network_volumes()
    finally:
        client.close()

    assert all(volume.id and volume.size_gb >= 10 for volume in volumes)
