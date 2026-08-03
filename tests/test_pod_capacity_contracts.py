"""Generic quote, identity, cost, environment, and persistence contracts."""

from dataclasses import replace
from decimal import Decimal

import pytest
from pod_capacity_test_support import (
    IMAGE,
    PARAMETERS_SHA,
    TOKEN,
    MutableClock,
    request,
)

from kestrel_cloud_runpod.pod_capacity_contracts import (
    PodCapacitySpec,
    attempt_environment_sha256,
)


def test_request_binds_exact_parameter_digest_quote_cost_and_image() -> None:
    clock = MutableClock()
    original = request(clock)

    with pytest.raises(ValueError, match="parameters"):
        replace(original, parameters_sha256="f" * 64)
    with pytest.raises(ValueError, match="must equal"):
        replace(original, accepted_max_cost_usd=Decimal("0.07"))
    with pytest.raises(ValueError, match="pinned by digest"):
        replace(original, image_reference="ghcr.io/kestrel/catalog-worker:latest")

    assert original.parameters_sha256 == PARAMETERS_SHA
    assert original.image_reference == IMAGE
    assert original.resource_name.startswith("kestrel-cap-")


@pytest.mark.parametrize(
    "key",
    [
        "DATABASE_URL",
        "CATALOG_DATABASE_URL",
        "RUNPOD_API_KEY",
        "RUNPOD_CONTROL_PLANE_API_KEY",
        "RUNPOD_SERVERLESS_API_KEY",
        "CATALOG_POD_BEARER_TOKEN",
        "CONTAINER_DIGEST",
    ],
)
def test_request_refuses_control_database_and_service_owned_environment(
    key: str,
) -> None:
    clock = MutableClock()
    with pytest.raises(ValueError, match=key):
        request(clock, attempt_environment={key: "must-not-reach-worker"})


def test_persisted_spec_contains_only_secret_identity_digest_and_expiry() -> None:
    clock = MutableClock()
    original = request(clock)
    spec = PodCapacitySpec(
        request=original,
        capability_secret_id="secret:catalog-capability-0001",
        capability_token_sha256="a" * 64,
        capability_expires_at=original.bearer_expires_at,
        attempt_environment_sha256=attempt_environment_sha256(
            original.attempt_environment
        ),
    )

    serialized = spec.to_dict()
    assert TOKEN not in str(serialized)
    assert "private/model" not in str(serialized)
    assert serialized["request"]["attempt_environment"] == {
        "MODEL_REPOSITORY": "[REDACTED]"
    }
    assert PodCapacitySpec.from_dict(serialized).to_dict() == serialized


def test_changed_attempt_environment_changes_durable_identity() -> None:
    clock = MutableClock()
    first = request(clock, attempt_environment={"MODEL_REPOSITORY": "model/a"})
    second = request(clock, attempt_environment={"MODEL_REPOSITORY": "model/b"})
    assert first.resource_name == second.resource_name
    assert attempt_environment_sha256(first.attempt_environment) != (
        attempt_environment_sha256(second.attempt_environment)
    )
