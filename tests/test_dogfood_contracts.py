"""Coverage for the product-neutral contracts extracted from ``dogfood.py``.

These types are imported by production Frinz code, so they ship independently of
the live harness. The extraction is mechanical, which makes the two risks (a)
the surface silently shrinking and (b) the content-free guard - the only
security-relevant behaviour here - silently weakening.
"""

from __future__ import annotations

import pytest

from kestrel_cloud_runpod import dogfood_contracts as dc

# Exactly what production Frinz imports today. Shrinking this breaks the
# consumer at import time, in another repository, which no test there can catch
# before this package is released.
FRINZ_SURFACE = (
    "DogfoodLane",
    "DogfoodPhase",
    "ExpectedResource",
    "PhaseObservation",
    "ProviderAttemptIdentity",
    "ResourceIdentity",
    "ResourcePlan",
    "ResourceType",
    "SpendQuote",
)


@pytest.mark.parametrize("name", FRINZ_SURFACE)
def test_frinz_facing_surface_is_present(name):
    assert hasattr(dc, name), f"{name} is imported by production Frinz code"


def test_module_does_not_pull_in_the_harness():
    """The whole point: importing contracts must not import the orchestrator.

    ``dogfood.py`` owns the CLI, workspace management, spend gate and live-run
    behaviour. If this module ever imports it, production Frinz acquires the
    entire live-test harness as a runtime dependency.
    """
    import inspect

    source = inspect.getsource(dc)
    assert "from .dogfood import" not in source
    assert "from kestrel_cloud_runpod.dogfood import" not in source


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "a private prompt"},
        {"nested": {"api_key": "secret"}},
        [{"api_key": "hunter2"}],  # list recursion, sensitive key inside
        {"image_url": "https://private.example/artifact.jpg"},
    ],
)
def test_content_free_guard_rejects_private_material(payload):
    """Evidence is content-free: digests and identities, never the content."""
    with pytest.raises(dc.DogfoodSafetyError):
        dc._assert_content_free(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"artifact_sha256": "a" * 64},
        {"request_digest": "b" * 64},
        {"elapsed_ms": 1234},
        {"nested": {"lora_plaintext_sha256": "c" * 64}},
    ],
)
def test_content_free_guard_admits_digests_and_timings(payload):
    """A *_sha256 / *_digest key is admissible even though it names an artifact."""
    dc._assert_content_free(payload)


def test_content_free_guard_rejects_a_non_string_key():
    with pytest.raises(dc.DogfoodSafetyError):
        dc._assert_content_free({1: "value"})


def test_phases_and_lanes_are_stable_wire_values():
    """These are persisted into attestation evidence, so the strings matter."""
    assert dc.DogfoodLane("ollama") is dc.DogfoodLane.OLLAMA
    assert {lane.value for lane in dc.DogfoodLane} >= {"ollama", "lora"}
    # Every phase value round-trips through its own string form.
    for phase in dc.DogfoodPhase:
        assert dc.DogfoodPhase(phase.value) is phase
