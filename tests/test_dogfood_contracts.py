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
            # level>0 is a relative import; resolve it against this package so
            # `from . import dogfood` and `from .dogfood import x` both land on
            # the same name as the absolute form.
            if node.level:
                modules.add(f"kestrel_cloud_runpod.{node.module or ''}".rstrip("."))
                if node.module is None:
                    modules.update(
                        f"kestrel_cloud_runpod.{alias.name}" for alias in node.names
                    )
            elif node.module:
                modules.add(node.module)
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
