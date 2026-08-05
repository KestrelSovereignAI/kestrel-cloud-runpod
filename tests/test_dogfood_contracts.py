"""Coverage for the product-neutral contracts extracted from ``dogfood.py``.

These types are imported by production Frinz code, so they ship independently of
the live harness. The extraction is mechanical, which makes the two risks (a)
the surface silently shrinking and (b) the content-free guard - the only
security-relevant behaviour here - silently weakening.

On (b), note where the guard actually sits on the production path.
``_assert_content_free`` runs only inside ``PhaseObservation.to_evidence``, and
Frinz does not call that - it calls ``binding_payload()``. So in production,
content-freeness is enforced entirely by the ``__post_init__`` validators, and
those are what the bulk of this module tests.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

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

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
SHA = "sha256:" + "a" * 64


@pytest.mark.parametrize("name", FRINZ_SURFACE)
def test_frinz_facing_surface_is_present(name):
    assert hasattr(dc, name), f"{name} is imported by production Frinz code"


# --------------------------------------------------------------------------
# Import graph
# --------------------------------------------------------------------------


def _imports_of(source: str) -> set[str]:
    """Every module `source` imports, from its AST rather than its text."""

    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `from X import y` is ambiguous in the AST: `y` may be a submodule
            # or just a name inside X. Record BOTH `X` and `X.y`, so a package
            # import of a forbidden submodule is caught whichever it turns out
            # to be. Missing this is how `from kestrel_cloud_runpod import
            # clients` slipped past an earlier version of this guard - the very
            # spelling used at the top of this file.
            base = (
                f"kestrel_cloud_runpod.{node.module or ''}".rstrip(".")
                if node.level  # relative: resolve against this package
                else node.module
            )
            if base:
                modules.add(base)
                modules.update(f"{base}.{alias.name}" for alias in node.names)
    return modules


def _declared_imports() -> set[str]:
    return _imports_of(inspect.getsource(dc))


FORBIDDEN_EDGES = {
    "kestrel_cloud_runpod.dogfood",
    "kestrel_cloud_runpod.clients",
    "kestrel_cloud_runpod.manager",
    "kestrel_cloud_runpod.core",
    "kestrel_cloud_runpod.feature",
    "httpx",
    "subprocess",
}


def test_module_does_not_import_the_harness_or_the_transport():
    """The whole point: contracts must not drag in the orchestrator.

    ``dogfood.py`` owns the CLI, workspace management, spend gate and live-run
    behaviour; ``clients.py`` owns the authenticated Runpod control-plane
    transport. An edge to either makes this file unseverable from the harness.

    This walks the AST, not the source text, because a text search for two
    literal import spellings does not survive a third spelling - ``from .
    import dogfood`` and ``import kestrel_cloud_runpod.dogfood as _h`` both
    defeat it while regressing exactly what the docstring promises.
    """

    declared = _declared_imports()
    # A blind extractor would satisfy the line below vacuously, so assert it
    # actually saw this module before trusting the intersection.
    assert "re" in declared and "dataclasses" in declared
    assert not (declared & FORBIDDEN_EDGES)


@pytest.mark.parametrize(
    "source",
    [
        "from . import dogfood\n",
        "import kestrel_cloud_runpod.dogfood as _h\n",
        "from .dogfood import DogfoodPhase\n",
        "from kestrel_cloud_runpod.dogfood import DogfoodPhase\n",
        "from .clients import RunpodControlPlaneClient\n",
        "import kestrel_cloud_runpod.clients\n",
        # The bare-package spelling. This is the one this file itself uses to
        # import the module under test, and an earlier version of the guard
        # recorded only "kestrel_cloud_runpod" for it - so a real edge to the
        # live control-plane transport passed the whole suite.
        "from kestrel_cloud_runpod import dogfood\n",
        "from kestrel_cloud_runpod import clients as _transport\n",
        "from kestrel_cloud_runpod import clients, models\n",
        "import httpx\n",
        "import subprocess\n",
    ],
)
def test_the_import_guard_catches_every_spelling(source):
    """Pins the extractor itself, by CALLING it rather than re-implementing it.

    An inline copy of the AST walk would pass even if `_imports_of` were made
    structurally blind (`ast.parse("")`), which would also make the guard above
    pass vacuously — the regression the guard exists to catch.
    """

    assert _imports_of(source) & FORBIDDEN_EDGES, f"not caught: {source!r}"


# --------------------------------------------------------------------------
# Wire values
# --------------------------------------------------------------------------

# Persisted verbatim into attestation evidence and read back by
# `DogfoodPhase(request.phase)` in Frinz's projector. Renaming any one of these
# breaks every evidence row already written, in another repository. Pinning the
# literal strings is the point; a round-trip through `DogfoodPhase(p.value)` is
# a tautology for any StrEnum and pins nothing.
PHASE_WIRE_VALUES = {
    "PRE_LORA_SELFIE": "pre_lora_selfie",
    "OLLAMA_QUOTE": "ollama_quote",
    "OLLAMA_ACQUIRE": "ollama_acquire",
    "OLLAMA_INFERENCE": "ollama_inference",
    "OLLAMA_STREAM": "ollama_stream",
    "OLLAMA_RESTART_RECONCILE": "ollama_restart_reconcile",
    "OLLAMA_REPLAY": "ollama_replay",
    "OLLAMA_CROSS_OWNER": "ollama_cross_owner",
    "OLLAMA_RELEASE": "ollama_release",
    "LORA_QUOTE": "lora_quote",
    "LORA_SUBMIT": "lora_submit",
    "LORA_POLL": "lora_poll",
    "LORA_CANCEL_LATE_RESULT": "lora_cancel_late_result",
    "LORA_UPLOAD_ACK_INTERRUPT": "lora_upload_ack_interrupt",
    "LORA_REPLAY": "lora_replay",
    "LORA_CROSS_OWNER": "lora_cross_owner",
    "LORA_PROMOTE": "lora_promote",
    "SELFIE_QUOTE": "selfie_quote",
    "EXPIRED_CAPABILITY": "expired_capability",
    "COST_CAP_REFUSAL": "cost_cap_refusal",
    "PRIVACY_CLOUD_REFUSAL": "privacy_cloud_refusal",
    "POST_LORA_SELFIE": "post_lora_selfie",
    "BILLING_RECONCILE": "billing_reconcile",
    "CLEANUP": "cleanup",
}


def test_every_phase_wire_value_is_pinned():
    assert {p.name: p.value for p in dc.DogfoodPhase} == PHASE_WIRE_VALUES


def test_every_lane_and_resource_type_wire_value_is_pinned():
    assert {lane.name: lane.value for lane in dc.DogfoodLane} == {
        "OLLAMA": "ollama",
        "LORA": "lora",
        "SELFIE": "selfie",
    }
    assert {item.name: item.value for item in dc.ResourceType} == {
        "POD": "pod",
        "SERVERLESS_ENDPOINT": "serverless_endpoint",
        "NETWORK_VOLUME": "network_volume",
    }


def test_resource_mutating_phases_are_pinned():
    """Only these phases may register a plan or a billable attempt."""

    assert {p.value for p in dc._RESOURCE_MUTATING_PHASES} == {
        "ollama_acquire",
        "lora_submit",
        "lora_cancel_late_result",
        "lora_upload_ack_interrupt",
        "post_lora_selfie",
    }


# --------------------------------------------------------------------------
# _assert_content_free
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# ResourceIdentity / ExpectedResource
# --------------------------------------------------------------------------


def _identity(**over) -> dc.ResourceIdentity:
    return dc.ResourceIdentity(
        **{
            "resource_type": dc.ResourceType.POD,
            "resource_id": "pod-0001",
            "resource_name": "kite-dogfood-pod",
            **over,
        }
    )


def test_resource_identity_round_trips_through_its_payload():
    identity = _identity()
    assert dc.ResourceIdentity.from_payload(identity.to_payload()) == identity
    assert identity.to_payload() == {
        "resource_type": "pod",
        "resource_id": "pod-0001",
        "resource_name": "kite-dogfood-pod",
    }


@pytest.mark.parametrize("field", ["resource_id", "resource_name"])
def test_resource_identity_rejects_a_url_bearing_field(field):
    """`://` is how a private endpoint would leak into evidence."""
    with pytest.raises(dc.DogfoodSafetyError):
        _identity(**{field: "https://private.example/pod"})


def test_safe_identifier_alphabet_is_what_excludes_urls():
    """Pins the clause that does the work, not the one that reads like it does.

    ``_safe_identifier`` also tests ``"://" in value``, but that branch is
    unreachable: a URL is rejected because ``/`` is absent from ``_SAFE_ID``'s
    character class, so no input can reach the ``://`` test. Deleting the
    ``://`` clause therefore breaks nothing, and a test written against it
    pins nothing. Widen this alphabet and evidence starts accepting routes.
    """

    assert dc._SAFE_ID.pattern == r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,254}$"
    for char in "/ \t\n?#%&=\\":
        assert not dc._SAFE_ID.fullmatch(f"a{char}b"), f"{char!r} became admissible"
    # No string can satisfy _SAFE_ID and still contain "://", which is why the
    # second clause cannot fire.
    assert not dc._SAFE_ID.fullmatch("a://b")


def test_resource_identity_rejects_an_untyped_resource_type():
    with pytest.raises(dc.DogfoodSafetyError):
        _identity(resource_type="pod")


def test_expected_resource_rejects_an_untyped_lane():
    with pytest.raises(dc.DogfoodSafetyError):
        dc.ExpectedResource(
            resource_type=dc.ResourceType.POD, resource_name="kite-pod", lane="lora"
        )


# --------------------------------------------------------------------------
# ResourcePlan
# --------------------------------------------------------------------------


def _expected(name="kite-pod", lane=dc.DogfoodLane.LORA) -> dc.ExpectedResource:
    return dc.ExpectedResource(
        resource_type=dc.ResourceType.POD, resource_name=name, lane=lane
    )


def _plan(**over) -> dc.ResourcePlan:
    return dc.ResourcePlan(
        **{
            "run_id": "run-0001",
            "phase": dc.DogfoodPhase.LORA_SUBMIT,
            "lane": dc.DogfoodLane.LORA,
            "plan_id": "plan-0001",
            "cleanup_family_id": "family-0001",
            "expected_resources": (_expected(),),
            **over,
        }
    )


def test_resource_plan_defaults_initial_to_the_full_expected_set():
    plan = _plan()
    assert plan.initial_resources == plan.expected_resources
    assert plan.digest.startswith("sha256:")


def test_resource_plan_rejects_a_non_mutating_phase():
    """A plan registers capacity; only a mutating phase may do that."""
    with pytest.raises(dc.DogfoodSafetyError):
        _plan(phase=dc.DogfoodPhase.LORA_QUOTE)


def test_resource_plan_rejects_a_resource_from_another_lane():
    with pytest.raises(dc.DogfoodSafetyError):
        _plan(expected_resources=(_expected(lane=dc.DogfoodLane.OLLAMA),))


def test_resource_plan_rejects_an_empty_expected_set():
    with pytest.raises(dc.DogfoodSafetyError):
        _plan(expected_resources=())


def test_resource_plan_rejects_duplicate_identities():
    with pytest.raises(dc.DogfoodSafetyError):
        _plan(expected_resources=(_expected(), _expected()))


def test_resource_plan_rejects_initial_resources_outside_expected():
    with pytest.raises(dc.DogfoodSafetyError):
        _plan(initial_resources=(_expected(name="not-in-the-plan"),))


def test_resource_plan_digest_changes_with_its_content():
    """The digest is what binds a plan to an attempt; it must not be constant."""
    assert _plan().digest != _plan(plan_id="plan-0002").digest


# --------------------------------------------------------------------------
# ProviderAttemptIdentity
# --------------------------------------------------------------------------


def _attempt(**over) -> dc.ProviderAttemptIdentity:
    return dc.ProviderAttemptIdentity(
        **{
            "run_id": "run-0001",
            "attempt_id": "attempt-0001",
            "phase": dc.DogfoodPhase.LORA_SUBMIT,
            "lane": dc.DogfoodLane.LORA,
            "plan_digest": SHA,
            "quote_digest": SHA,
            "resource": _identity(),
            "provider_operation_id": "op-0001",
            "exclusive_window_sha256": SHA,
            "started_at": NOW,
            "completed_at": NOW + timedelta(seconds=30),
            **over,
        }
    )


def test_provider_attempt_round_trips_through_its_payload():
    assert dc.ProviderAttemptIdentity.from_payload(_attempt().to_payload()) == _attempt()


def test_provider_attempt_rejects_a_non_mutating_phase():
    with pytest.raises(dc.DogfoodSafetyError):
        _attempt(phase=dc.DogfoodPhase.LORA_QUOTE)


@pytest.mark.parametrize(
    "field", ["plan_digest", "quote_digest", "exclusive_window_sha256"]
)
def test_provider_attempt_rejects_a_malformed_digest(field):
    with pytest.raises(dc.DogfoodSafetyError):
        _attempt(**{field: "a" * 64})  # no sha256: prefix


def test_provider_attempt_rejects_an_inverted_interval():
    with pytest.raises(dc.DogfoodSafetyError):
        _attempt(started_at=NOW + timedelta(seconds=30), completed_at=NOW)


def test_provider_attempt_rejects_a_naive_timestamp():
    with pytest.raises(dc.DogfoodSafetyError):
        _attempt(started_at=datetime(2026, 8, 3, 12, 0))


def test_provider_attempt_rejects_an_untyped_resource():
    with pytest.raises(dc.DogfoodSafetyError):
        _attempt(resource={"resource_type": "pod", "resource_id": "x"})


# --------------------------------------------------------------------------
# SpendQuote — the money invariants
# --------------------------------------------------------------------------


def _quote(**over) -> dc.SpendQuote:
    return dc.SpendQuote(
        **{
            "run_id": "run-0001",
            "lane": dc.DogfoodLane.LORA,
            "quote_id": "quote-0001",
            "estimated_cost_usd": Decimal("1.50"),
            "hard_cap_usd": Decimal("5.00"),
            "observed_at": NOW,
            "expires_at": NOW + timedelta(minutes=5),
            **over,
        }
    )


def test_spend_quote_rejects_an_estimate_above_its_hard_cap():
    """The whole purpose of the type: a quote can never authorize past the cap."""
    with pytest.raises(dc.DogfoodSafetyError):
        _quote(estimated_cost_usd=Decimal("5.01"))


def test_spend_quote_admits_an_estimate_exactly_at_its_cap():
    assert _quote(estimated_cost_usd=Decimal("5.00")).hard_cap_usd == Decimal("5.00")


@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1"), Decimal("NaN")])
def test_spend_quote_rejects_a_non_positive_or_non_finite_cost(amount):
    with pytest.raises(dc.DogfoodSafetyError):
        _quote(estimated_cost_usd=amount)


def test_spend_quote_rejects_a_float_cost():
    """Decimal is required: binary floats silently misprice money."""
    with pytest.raises(dc.DogfoodSafetyError):
        _quote(estimated_cost_usd=1.5)


def test_spend_quote_expiry_window_is_at_most_five_minutes():
    _quote(expires_at=NOW + timedelta(minutes=5))  # boundary is inclusive
    with pytest.raises(dc.DogfoodSafetyError):
        _quote(expires_at=NOW + timedelta(minutes=5, seconds=1))


def test_spend_quote_rejects_an_expiry_at_or_before_observation():
    for expires_at in (NOW, NOW - timedelta(seconds=1)):
        with pytest.raises(dc.DogfoodSafetyError):
            _quote(expires_at=expires_at)


def test_spend_quote_digest_changes_with_its_amounts():
    assert _quote().digest != _quote(estimated_cost_usd=Decimal("1.51")).digest


# --------------------------------------------------------------------------
# PhaseObservation — what Frinz actually binds
# --------------------------------------------------------------------------


def _observation(**over) -> dc.PhaseObservation:
    return dc.PhaseObservation(
        **{
            "phase": dc.DogfoodPhase.LORA_SUBMIT,
            "state_transitions": ("queued", "running"),
            "timings_ms": {"total": 12},
            **over,
        }
    )


# Frinz feeds binding_payload() to phase_evidence_sha256 and the server receipt
# binds the result, so this key set IS the cross-repo digest contract. Adding,
# removing or renaming a key changes every digest and fails verification in the
# other repository.
BINDING_PAYLOAD_KEYS = {
    "phase",
    "state_transitions",
    "timings_ms",
    "artifact_digests",
    "estimated_cost_usd",
    "actual_cost_usd",
    "billing_receipt_digest",
    "provider",
    "model",
    "product_consent_count",
    "trained_weight_digest",
    "promoted_weight_digest",
    "weight_digest_used",
    "output_image_digest",
    "uploaded_artifact_digest",
    "recovered_artifact_digest",
    "recovered_resource_plan_digest",
    "provider_ack_interruption_count",
    "recovery_count",
    "publication_count",
    "promotion_count",
    "provider_attempts",
}


def test_binding_payload_key_set_is_the_pinned_cross_repo_contract():
    assert set(_observation().binding_payload()) == BINDING_PAYLOAD_KEYS


def test_binding_payload_sorts_timings_deterministically():
    """A digest over an unordered mapping is not reproducible."""
    payload = _observation(timings_ms={"total": 2, "cleanup": 1}).binding_payload()
    assert list(payload["timings_ms"]) == ["cleanup", "total"]


def test_observation_requires_live_state_transitions():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(state_transitions=())


def test_observation_rejects_a_url_in_a_state_transition():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(state_transitions=("https://private.example/x",))


def test_observation_rejects_an_unknown_timing_field():
    """The timing whitelist is what keeps free-form keys out of evidence."""
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(timings_ms={"user_prompt": 1})


@pytest.mark.parametrize("value", [-1, True, "12"])
def test_observation_rejects_a_non_millisecond_timing(value):
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(timings_ms={"total": value})


@pytest.mark.parametrize("field", ["provider", "model"])
def test_observation_rejects_a_url_bearing_attribute(field):
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(**{field: "https://private.example/v1"})


def test_observation_rejects_a_malformed_artifact_digest():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(artifact_digests=("a" * 64,))


def test_observation_rejects_duplicate_provider_attempts():
    """Two rows for one attempt_id would double-count real spend."""
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(provider_attempts=(_attempt(), _attempt()))


def test_observation_admits_distinct_provider_attempts():
    observation = _observation(
        provider_attempts=(_attempt(), _attempt(attempt_id="attempt-0002"))
    )
    assert len(observation.binding_payload()["provider_attempts"]) == 2


def test_observation_rejects_an_untyped_provider_attempt():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(provider_attempts=({"attempt_id": "attempt-0001"},))


def test_observation_rejects_a_negative_cost():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(actual_cost_usd=Decimal("-0.01"))


def test_to_evidence_shape():
    payload = _observation().to_evidence(run_id="run-0001", observed_at=NOW)
    assert payload["contract"] == dc.DOGFOOD_CONTRACT
    assert payload["run_id"] == "run-0001"
    assert set(payload) == BINDING_PAYLOAD_KEYS | {
        "contract",
        "event",
        "run_id",
        "observed_at",
    }


@pytest.mark.parametrize(
    "run_id",
    [
        "https://private.example/secret-run",
        "Bearer sk-live-0123456789",
        "run id with spaces",
        "",
    ],
)
def test_to_evidence_rejects_a_content_bearing_run_id(run_id):
    """`run_id` is caller-supplied and lands verbatim in persisted evidence.

    Asserting on the returned payload's shape does not test this: removing
    either `_safe_identifier` or `_assert_content_free` from `to_evidence`
    leaves every shape assertion green while a signed URL or bearer token is
    written straight into the attestation record.
    """

    with pytest.raises(dc.DogfoodSafetyError):
        _observation().to_evidence(run_id=run_id, observed_at=NOW)


@pytest.mark.parametrize(
    "value",
    [
        "https://private.example/artifact.jpg",
        "http://10.0.0.1/internal",
        "Bearer sk-live-0123456789",
        "api_key=hunter2",
        "api-key: hunter2",
        "token=abcdef",
        "password=hunter2",
        "secret: shhh",
        "signature=deadbeef",
    ],
)
def test_content_free_guard_rejects_a_sensitive_value_under_a_benign_key(value):
    """Pins the VALUE regex, which the key rule otherwise masks.

    `{"image_url": "https://..."}` is rejected for its key ("url" is in
    `_SENSITIVE_KEY_PARTS`), so it proves nothing about `_SENSITIVE_VALUE`.
    Under a benign key the value branch is the only thing that can fire.
    """

    with pytest.raises(dc.DogfoodSafetyError, match="URL or credential-like"):
        dc._assert_content_free({"note": value})


def test_content_free_guard_admits_a_benign_value_under_a_benign_key():
    dc._assert_content_free({"note": "queued then running", "count": 3})


def test_to_evidence_rejects_unvalidated_content_from_binding_payload(monkeypatch):
    """Pins `to_evidence`'s `_assert_content_free` call, which nothing else can.

    Every field `binding_payload()` emits is validated at construction, and
    `run_id` is now validated by `_safe_identifier` directly above the guard.
    So no input through the public API reaches this call first — meaning a
    test built from valid objects cannot detect its removal, which is exactly
    how it went unpinned.

    What the call actually defends is the case the validators do not cover: a
    future field added to `binding_payload()` without one. Simulating that is
    the only way to exercise it, and it tests the real contract — whatever the
    projection emits must be content-free before it becomes evidence.
    """

    observation = _observation()
    monkeypatch.setattr(
        type(observation),
        "binding_payload",
        lambda self: {"phase": self.phase.value, "prompt": "a private prompt"},
    )
    with pytest.raises(dc.DogfoodSafetyError):
        observation.to_evidence(run_id="run-0001", observed_at=NOW)


def test_to_evidence_rejects_a_naive_observed_at():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation().to_evidence(
            run_id="run-0001", observed_at=datetime(2026, 8, 3, 12, 0)
        )


# --------------------------------------------------------------------------
# Cross-module agreement
# --------------------------------------------------------------------------


def test_shared_validators_agree_with_signed_invocations():
    """These helpers are deliberately parallel, not shared — so pin the drift.

    ``dogfood_contracts`` and ``signed_invocations`` each define ``_SHA256``,
    ``_required_payload_string``, ``_iso`` and ``_parse_time``. They are NOT
    deduplicated on purpose: the two modules raise different exception types
    (``DogfoodSafetyError`` vs ``SignedInvocationError``), and that type IS the
    contract each module advertises to its callers. Sharing one implementation
    would either force one error contract on both or add an exception-type
    parameter that buys nothing.

    What the duplication does risk is the copies drifting apart, so this
    asserts they still agree on everything except the exception type. Both are
    consumed by the same Frinz projector, and a divergent digest or timestamp
    rule between them is a cross-repo bug that neither module's own tests would
    see.
    """

    from kestrel_cloud_runpod import signed_invocations as si

    assert dc._SHA256.pattern == si._SHA256.pattern

    aware = datetime(2026, 8, 3, 12, 0, 30, tzinfo=UTC)
    assert dc._iso(aware) == si._iso(aware) == "2026-08-03T12:00:30Z"
    # Both normalize a non-UTC offset to the same Z-suffixed instant.
    offset = datetime(2026, 8, 3, 7, 0, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert dc._iso(offset) == si._iso(offset) == "2026-08-03T12:00:30Z"

    assert dc._parse_time("2026-08-03T12:00:30Z", "t") == si._parse_time(
        "2026-08-03T12:00:30Z", "t"
    )
    assert dc._required_payload_string("v", "n") == si._required_payload_string("v", "n")

    # Same rejections, different exception types — the contract boundary.
    for bad in (None, 12, b"bytes"):
        with pytest.raises(dc.DogfoodSafetyError):
            dc._required_payload_string(bad, "n")
        with pytest.raises(si.SignedInvocationError):
            si._required_payload_string(bad, "n")
    for bad in ("not-a-timestamp", "2026-08-03T12:00:30", None):
        with pytest.raises(dc.DogfoodSafetyError):
            dc._parse_time(bad, "t")
        with pytest.raises(si.SignedInvocationError):
            si._parse_time(bad, "t")
    naive = datetime(2026, 8, 3, 12, 0)
    with pytest.raises(dc.DogfoodSafetyError):
        dc._iso(naive)
    with pytest.raises(si.SignedInvocationError):
        si._iso(naive)
    # Non-datetime input is where the two used to diverge: dc._iso lacked the
    # isinstance check and raised AttributeError instead of its own error type.
    for bad in ("2026-08-03T12:00:00Z", None, 1754308800, object()):
        with pytest.raises(dc.DogfoodSafetyError):
            dc._iso(bad)
        with pytest.raises(si.SignedInvocationError):
            si._iso(bad)


def test_to_evidence_rejects_a_non_datetime_observed_at():
    """`observed_at` is caller-supplied and had no validator of its own.

    A deserialized ISO string is the realistic input: it produced
    `AttributeError: 'str' object has no attribute 'tzinfo'` out of a module
    whose contract is `DogfoodSafetyError`.
    """

    for bad in ("2026-08-03T12:00:00Z", None, 1754308800):
        with pytest.raises(dc.DogfoodSafetyError):
            _observation().to_evidence(run_id="run-0001", observed_at=bad)


# --------------------------------------------------------------------------
# Digest validation must not escape the module's error contract
# --------------------------------------------------------------------------

# `re.Pattern.fullmatch` raises TypeError on non-str input, so a bytes or int
# digest used to escape DogfoodSafetyError entirely — the same escape class as
# the UnsupportedAlgorithm leak. `hashlib.sha256(x).digest()` where
# `.hexdigest()` was meant is the ordinary way to produce one, and these are
# CONSTRUCTOR paths, so `_required_payload_string` never runs.
NON_STRING_DIGESTS = [b"\x00" * 32, 7, None, ["sha256:" + "a" * 64], 3.14]


@pytest.mark.parametrize("bad", NON_STRING_DIGESTS)
@pytest.mark.parametrize(
    "field", ["plan_digest", "quote_digest", "exclusive_window_sha256"]
)
def test_provider_attempt_contains_a_non_string_digest(field, bad):
    with pytest.raises(dc.DogfoodSafetyError):
        _attempt(**{field: bad})


@pytest.mark.parametrize(
    "bad", [v for v in NON_STRING_DIGESTS if v is not None]
)
@pytest.mark.parametrize(
    "field", ["operation_digest", "provider_quote_sha256", "endpoint_plan_sha256"]
)
def test_spend_quote_contains_a_non_string_digest(field, bad):
    """SpendQuote's optional-digest loop had no test at all.

    `None` is excluded: these fields are Optional and default to None, so the
    guard is `value is not None and not _is_sha256(value)`. Asserting None is
    rejected would test the opposite of the contract.
    """

    with pytest.raises(dc.DogfoodSafetyError):
        _quote(**{field: bad})


@pytest.mark.parametrize(
    "field", ["operation_digest", "provider_quote_sha256", "endpoint_plan_sha256"]
)
def test_spend_quote_optional_digests_admit_none_and_a_real_digest(field):
    assert getattr(_quote(**{field: None}), field) is None
    assert getattr(_quote(**{field: SHA}), field) == SHA


@pytest.mark.parametrize("bad", NON_STRING_DIGESTS)
def test_observation_artifact_digests_reject_a_non_string(bad):
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(artifact_digests=(bad,))


@pytest.mark.parametrize(
    "field",
    [
        "billing_receipt_digest",
        "trained_weight_digest",
        "promoted_weight_digest",
        "weight_digest_used",
        "output_image_digest",
        "uploaded_artifact_digest",
        "recovered_artifact_digest",
        "recovered_resource_plan_digest",
    ],
)
def test_observation_optional_digest_fields_are_validated(field):
    """All eight had no test: replacing the loop with `pass` stayed green."""

    with pytest.raises(dc.DogfoodSafetyError):
        _observation(**{field: "a" * 64})  # missing the sha256: prefix
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(**{field: b"\x00" * 32})  # non-string: used to be TypeError
    assert getattr(_observation(**{field: SHA}), field) == SHA


def test_is_sha256_accepts_any_input_type_without_raising():
    """The helper itself must never raise — it returns a bool for anything."""

    for value in (b"bytes", 7, None, [], {}, object(), "sha256:" + "a" * 64):
        assert isinstance(dc._is_sha256(value), bool)
    assert dc._is_sha256("sha256:" + "a" * 64)
    assert not dc._is_sha256("sha256:" + "A" * 64)  # uppercase hex rejected
    assert not dc._is_sha256("a" * 64)


# --------------------------------------------------------------------------
# Frozen, digest-bearing fields must not alias the caller's containers
# --------------------------------------------------------------------------


def test_observation_detaches_state_transitions_from_the_caller():
    """Validated content must not be mutable after validation.

    `_safe_identifier` passes over every element at construction, then the
    caller appends a URL to their own list and it appears in the projection
    Frinz feeds to `phase_evidence_sha256` for the server to sign.
    """

    transitions = ["queued", "running"]
    observation = _observation(state_transitions=transitions)
    transitions.append("https://private.example/leak")
    assert observation.binding_payload()["state_transitions"] == ["queued", "running"]


def test_observation_detaches_timings_from_the_caller():
    timings = {"total": 12}
    observation = _observation(timings_ms=timings)
    timings["total"] = -1  # would have been rejected at construction
    assert observation.binding_payload()["timings_ms"] == {"total": 12}
    assert observation.to_evidence(run_id="run-0001", observed_at=NOW)["timings_ms"] == {
        "total": 12
    }


def test_observation_detaches_artifact_digests_from_the_caller():
    """Passing a LIST is the case that aliases; a tuple cannot.

    `artifact_digests` has no isinstance-tuple check, so a list is accepted and
    was stored live. (`provider_attempts` does have one, so it needs no
    detaching — that is why there is no sibling assertion here.)
    """

    digests = [SHA]
    observation = _observation(artifact_digests=digests)
    digests.append("not-a-digest")
    assert observation.binding_payload()["artifact_digests"] == [SHA]
    assert isinstance(observation.artifact_digests, tuple)


def test_observation_requires_a_tuple_of_provider_attempts():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(provider_attempts=[_attempt()])


def test_resource_plan_digest_is_stable_after_construction():
    """`plan.digest` is what `ProviderAttemptIdentity.plan_digest` pins.

    Appending to the caller's list defeated both the same-lane and uniqueness
    checks AND changed the digest recorded against a billable attempt.
    """

    # Pass the LIST, not tuple(list). Converting first means the append below
    # was never connected to anything - the same mistake that made the
    # artifact_digests aliasing test vacuous.
    resources = [_expected()]
    plan = _plan(expected_resources=resources)
    before = plan.digest
    resources.append(
        dc.ExpectedResource(
            resource_type=dc.ResourceType.NETWORK_VOLUME,
            resource_name="extra-volume",
            lane=dc.DogfoodLane.OLLAMA,
        )
    )
    assert plan.digest == before
    assert [r["lane"] for r in plan.to_payload()["expected_resources"]] == ["lora"]


def test_resource_plan_accepts_a_list_and_stores_a_tuple():
    plan = _plan(expected_resources=[_expected()])
    assert isinstance(plan.expected_resources, tuple)
    assert isinstance(plan.initial_resources, tuple)


# --------------------------------------------------------------------------
# Enum fields must be type-guarded, not just membership-checked
# --------------------------------------------------------------------------


def test_resource_plan_refuses_an_untyped_phase():
    """`DogfoodPhase` is a StrEnum, so membership alone admits a raw str.

    `"lora_submit" in _RESOURCE_MUTATING_PHASES` is True, so the plan
    constructed, and `.digest` then raised AttributeError ('str' has no
    attribute 'value') — escaping DogfoodSafetyError on the signed path.
    """

    with pytest.raises(dc.DogfoodSafetyError, match="phase is invalid"):
        _plan(phase="lora_submit")


def test_provider_attempt_refuses_an_untyped_phase():
    with pytest.raises(dc.DogfoodSafetyError, match="phase is invalid"):
        _attempt(phase="lora_submit")


def test_an_untyped_phase_never_reaches_the_signed_projection():
    """The end-to-end consequence: binding_payload() is what gets signed."""

    with pytest.raises(dc.DogfoodSafetyError):
        _observation(provider_attempts=(_attempt(phase="lora_submit"),))


@pytest.mark.parametrize(
    ("factory", "field", "bad", "message"),
    [
        # Exact messages, because these guards shadow each other. A raw "lora"
        # lane also trips ResourcePlan's same-lane check on expected_resources
        # (`item.lane is not self.lane`), so a bare `raises` is satisfied by
        # whichever fires first and pins neither.
        (lambda **k: _plan(**k), "lane", "lora", "resource plan lane is invalid"),
        (lambda **k: _attempt(**k), "lane", "lora", "provider attempt lane is invalid"),
        (lambda **k: _quote(**k), "lane", "lora", "spend quote lane is invalid"),
        (
            lambda **k: _observation(**k),
            "phase",
            "lora_submit",
            "phase observation has an invalid phase",
        ),
    ],
)
def test_enum_fields_reject_the_equal_string(factory, field, bad, message):
    """Each of these guards was individually removable with the suite green."""

    with pytest.raises(dc.DogfoodSafetyError, match=message):
        factory(**{field: bad})


# --------------------------------------------------------------------------
# Validate-then-recopy: a one-shot iterable must not empty itself
# --------------------------------------------------------------------------


def test_observation_accepts_a_generator_of_state_transitions():
    """Validating first and copying afterwards emptied the signed projection.

    A generator is always truthy, so the "requires live state transitions"
    guard passed, then the second iteration saw an exhausted iterator and the
    projection silently carried `[]`. `(t.name for t in log)` is ordinary
    Python.
    """

    observation = _observation(state_transitions=(t for t in ["queued", "running"]))
    assert observation.state_transitions == ("queued", "running")
    assert observation.binding_payload()["state_transitions"] == ["queued", "running"]


def test_observation_still_rejects_an_empty_generator():
    with pytest.raises(dc.DogfoodSafetyError, match="live state transitions"):
        _observation(state_transitions=(t for t in []))


def test_observation_validates_a_generator_element():
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(state_transitions=(t for t in ["https://private.example/x"]))


def test_observation_accepts_a_generator_of_artifact_digests():
    observation = _observation(artifact_digests=(d for d in [SHA]))
    assert observation.artifact_digests == (SHA,)


def test_resource_plan_accepts_a_generator_of_expected_resources():
    plan = _plan(expected_resources=(r for r in [_expected()]))
    assert len(plan.expected_resources) == 1
    assert plan.digest.startswith("sha256:")


# --------------------------------------------------------------------------
# Materialization must not let a bad type escape the contract
#
# Round 5 fixed one-shot iterables by copying BEFORE validating. That broke
# the other direction: `tuple(None)` / `dict(None)` raise bare TypeError, and
# DogfoodError subclasses RuntimeError, so a caller's `except DogfoodError`
# never sees them. The order that satisfies both is type-check, copy, validate.
# --------------------------------------------------------------------------

NON_SEQUENCES = [None, 12, 3.5, object(), True]
NON_MAPPINGS = [None, 12, "total=12", {"total"}, ["total", 12], object()]


@pytest.mark.parametrize("bad", NON_SEQUENCES)
def test_observation_rejects_a_non_sequence_state_transitions(bad):
    with pytest.raises(dc.DogfoodSafetyError, match="must be a sequence"):
        _observation(state_transitions=bad)


@pytest.mark.parametrize("bad", NON_SEQUENCES)
def test_observation_rejects_a_non_sequence_artifact_digests(bad):
    with pytest.raises(dc.DogfoodSafetyError, match="must be a sequence"):
        _observation(artifact_digests=bad)


@pytest.mark.parametrize("bad", NON_MAPPINGS)
def test_observation_rejects_a_non_mapping_timings(bad):
    """`None` is the ordinary shape of an absent field from a deserialized row.

    A list of pairs is also refused rather than coerced: the declared type is
    `Mapping[str, int]`, and silently accepting `[("total", 12)]` would put a
    shape the annotation excludes into the signature-bound projection.
    """

    with pytest.raises(dc.DogfoodSafetyError, match="must be a mapping"):
        _observation(timings_ms=bad)


@pytest.mark.parametrize("bad", NON_SEQUENCES)
def test_resource_plan_rejects_a_non_sequence_expected_resources(bad):
    with pytest.raises(dc.DogfoodSafetyError, match="must be a sequence"):
        _plan(expected_resources=bad)


@pytest.mark.parametrize("bad", [v for v in NON_SEQUENCES if v is not None])
def test_resource_plan_rejects_a_non_sequence_initial_resources(bad):
    """`None` is excluded: it is the documented default, meaning "same as
    expected_resources". Asserting it is rejected would test the opposite of
    the contract - the same mistake made once already on SpendQuote's optional
    digest fields."""

    with pytest.raises(dc.DogfoodSafetyError, match="must be a sequence"):
        _plan(initial_resources=bad)


def test_resource_plan_initial_resources_defaults_to_expected():
    plan = _plan(initial_resources=None)
    assert plan.initial_resources == plan.expected_resources


@pytest.mark.parametrize("field", ["state_transitions", "artifact_digests"])
def test_observation_refuses_a_bare_string_sequence(field):
    """A str is Iterable; exploding it into characters would be silent damage."""

    with pytest.raises(dc.DogfoodSafetyError, match="must be a sequence"):
        _observation(**{field: "queued"})


def test_expected_resource_rejects_an_untyped_resource_type():
    """The one isinstance guard in either module that survived its own removal.

    Without it `ExpectedResource(resource_type="pod", ...)` constructs, and the
    failure surfaces later at `to_payload()` -> `self.resource_type.value` ->
    AttributeError, reached through `ResourcePlan.digest` — the signed path.
    """

    with pytest.raises(dc.DogfoodSafetyError, match="must be a ResourceType"):
        dc.ExpectedResource(
            resource_type="pod", resource_name="kite-pod", lane=dc.DogfoodLane.LORA
        )


# --------------------------------------------------------------------------
# from_payload round trips — on FRINZ_SURFACE, previously untested
# --------------------------------------------------------------------------


def test_expected_resource_round_trips_through_its_payload():
    resource = _expected()
    assert dc.ExpectedResource.from_payload(resource.to_payload()) == resource
    assert resource.to_payload() == {
        "resource_type": "pod",
        "resource_name": "kite-pod",
        "lane": "lora",
    }


def test_resource_plan_round_trips_through_its_payload():
    plan = _plan()
    assert dc.ResourcePlan.from_payload(plan.to_payload()) == plan
    assert dc.ResourcePlan.from_payload(plan.to_payload()).digest == plan.digest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("lane"),
        lambda p: p.update(unexpected=True),
        lambda p: p.update(resource_type="not-a-type"),
        lambda p: p.update(lane="not-a-lane"),
    ],
)
def test_expected_resource_from_payload_rejects_a_bad_envelope(mutate):
    payload = _expected().to_payload()
    mutate(payload)
    with pytest.raises(dc.DogfoodSafetyError):
        dc.ExpectedResource.from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("plan_id"),
        lambda p: p.update(unexpected=True),
        lambda p: p.update(expected_resources="not-a-list"),
        lambda p: p.update(phase="lora_quote"),
    ],
)
def test_resource_plan_from_payload_rejects_a_bad_envelope(mutate):
    payload = _plan().to_payload()
    mutate(payload)
    with pytest.raises(dc.DogfoodSafetyError):
        dc.ResourcePlan.from_payload(payload)


@pytest.mark.parametrize(
    "field", ["provider_ack_interruption_count", "recovery_count",
              "publication_count", "promotion_count", "product_consent_count"]
)
@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None])
def test_observation_integer_counters_are_validated(field, bad):
    with pytest.raises(dc.DogfoodSafetyError):
        _observation(**{field: bad})


@pytest.mark.parametrize("field", ["observed_at", "expires_at"])
def test_spend_quote_rejects_a_naive_or_non_datetime_timestamp(field):
    for bad in (datetime(2026, 8, 3, 12, 0), "2026-08-03T12:00:00Z", None):
        with pytest.raises(dc.DogfoodSafetyError):
            _quote(**{field: bad})
