"""Contracts for RunPod profile-owned model defaults."""

from types import SimpleNamespace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kestrel_cloud_runpod.core import RunPodManagerCore
from kestrel_cloud_runpod.models import (
    RunPodLegacyAdmissionRetiredError,
    normalize_cuda_floor,
    _optional_response_datetime,
)


@pytest.mark.parametrize("floor", ["１２.８", "١٢", "12.８"])
def test_cuda_floor_requires_ascii_wire_numbers(floor):
    with pytest.raises(ValueError, match="CUDA major"):
        normalize_cuda_floor(floor)


def test_resource_authority_version_is_exported_with_public_contracts():
    from kestrel_cloud_runpod import EndpointResourceVersion

    assert EndpointResourceVersion.GA.value == 2


def test_optional_response_datetime_preserves_default_utc_and_explicit_offset_mode():
    value = "2026-11-14T13:00:00+01:00"
    default = _optional_response_datetime(value, "test")
    retained = _optional_response_datetime(value, "test", normalize_utc=False)
    assert default == datetime(2026, 11, 14, 12, tzinfo=UTC)
    assert default.utcoffset() == timedelta(0)
    assert retained == default
    assert retained.isoformat() == value
    assert retained.utcoffset() == timedelta(hours=1)


@pytest.mark.parametrize("normalize_utc", [True, False])
def test_optional_response_datetime_none_contract_is_unchanged(normalize_utc):
    assert (
        _optional_response_datetime(None, "test", normalize_utc=normalize_utc) is None
    )


class _ResumeHarness:
    resume_stopped_pod = RunPodManagerCore.resume_stopped_pod
    _refuse_legacy_admission = staticmethod(RunPodManagerCore._refuse_legacy_admission)

    def __init__(self, profile):
        self.provider = SimpleNamespace(resume_pod=MagicMock())
        self._lock = None
        self._session = None
        self.wait_called = False
        self.profile = profile

    async def _wait_until_ready(self):
        self.wait_called = True


class _AsyncNullLock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_resume_stopped_pod_refuses_ownerless_admission_before_provider_action():
    profile = SimpleNamespace(
        id="training",
        task_type="training",
        default_model=None,
        pod_type="a100",
    )
    harness = _ResumeHarness(profile)
    harness._lock = _AsyncNullLock()

    with pytest.raises(RunPodLegacyAdmissionRetiredError, match="durable"):
        await harness.resume_stopped_pod(
            {"id": "pod-123", "gpuCount": 1}, profile, 3600
        )

    harness.provider.resume_pod.assert_not_called()
    assert harness._session is None
    assert harness.wait_called is False
