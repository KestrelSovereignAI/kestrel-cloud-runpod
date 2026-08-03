"""Public durable training ownership contracts."""

from dataclasses import replace
from datetime import timedelta

import pytest
from training_test_support import MutableClock, training_request

from kestrel_cloud_runpod.training_contracts import (
    TrainingPodCleanupState,
    TrainingPodLifecycleError,
    TrainingPodSource,
)


def test_request_requires_ordered_aware_deadlines() -> None:
    clock = MutableClock()
    with pytest.raises(ValueError, match="hard deadline"):
        training_request(clock, readiness_seconds=30, hard_seconds=30)


def test_create_request_does_not_accept_a_provider_id() -> None:
    clock = MutableClock()
    request = training_request(clock, source=TrainingPodSource.CREATED, pod_id=None)
    assert request.provider_pod_id is None
    assert len(request.fingerprint) == 64


def test_lifecycle_error_exposes_cleanup_authority_without_provider_detail() -> None:
    error = TrainingPodLifecycleError(
        "readiness",
        cleanup_token="training:test-token-0001",
        pod_id="pod-1",
        cleanup_state=TrainingPodCleanupState.RETRYABLE_FAILURE,
        billing_risk=True,
    )
    assert error.reconcile_required is True
    assert error.cleanup_token == "training:test-token-0001"
    assert error.pod_id == "pod-1"
    assert "billable capacity may remain active" in str(error)
    assert "https://" not in str(error)


def test_request_fingerprint_changes_with_deadline() -> None:
    clock = MutableClock()
    first = training_request(clock)
    clock.value += timedelta(seconds=1)
    second = training_request(clock)
    assert first.fingerprint != second.fingerprint


def test_child_request_fingerprint_is_bound_to_its_cleanup_family() -> None:
    clock = MutableClock()
    child = training_request(
        clock,
        token="training:family-child-0001",
        root_token="training:family-root-0001",
    )

    assert child.root_cleanup_token == "training:family-root-0001"
    assert (
        child.fingerprint
        != training_request(clock, token="training:family-child-0001").fingerprint
    )


def test_request_defaults_root_for_pre_family_callers() -> None:
    request = training_request(MutableClock())
    compatible = replace(request, root_cleanup_token=None)

    assert compatible.cleanup_family_token == compatible.cleanup_token
    assert compatible.fingerprint == request.fingerprint
