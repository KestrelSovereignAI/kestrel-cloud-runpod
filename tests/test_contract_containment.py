"""Every public entry point raises ONLY its module's declared error type.

Seven review rounds on this stack found seven separate escapes from the two
modules' error contracts, one at a time:

    UnsupportedAlgorithm   not a ValueError        (key loading)
    AttributeError         from a missing isinstance on an enum
    AttributeError         from _iso on a non-datetime
    TypeError              from _SHA256.fullmatch on non-str
    TypeError/ValueError   from tuple()/dict() on a non-iterable
    UnicodeEncodeError     from .encode() on a lone surrogate
    OverflowError          from astimezone() near datetime.min/max

Mutation testing cannot find any of them. It measures whether an EXISTING
guard is pinned; a MISSING guard has no mutant, so it is invisible by
construction. Every one of the seven was a check written for the type of a
value at the point of assignment, with no check for the operation applied to
it afterwards - `astimezone`, `.encode()`, `dict()`, `fullmatch`, attribute
access.

This file is the regression guard for the class rather than the instances. It
throws a boundary corpus at every public entry point and asserts that whatever
comes back out is the module's own error type. It does not care WHICH inputs
are rejected - only that rejection stays inside the contract.

Adding a new public name or a new field without a guard should fail here.
"""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from kestrel_cloud_runpod import dogfood_contracts as dc
from kestrel_cloud_runpod import signed_invocations as si

# Inputs chosen because each one broke something, or is the boundary next to
# something that broke. Kept as data so a new hostile shape is one line.
LONE_SURROGATE = json.loads('{"v": "truncated emoji \\ud83d"}')["v"]


def _key_blobs() -> list[str]:
    """base64url blobs that are well-formed DER but wrong in a specific way.

    A generic corpus of strings and ints cannot reach the key-loading arms:
    every one of them fails the base64url alphabet or the DER parse long
    before the interesting branch. These are the shapes that got past those
    and into `UnsupportedAlgorithm` / `TypeError` territory - both real
    escapes found in earlier rounds, and neither reachable without a blob
    that actually parses.
    """

    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    key = Ed25519PrivateKey.generate()
    pkcs8 = key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    spki = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Rewrite the Ed25519 OID 1.3.101.112 to an unassigned 1.2.3.4 ->
    # UnsupportedAlgorithm, which is not a ValueError.
    bad_oid = bytes.fromhex("06032a0304")
    ed_oid = bytes.fromhex("06032b6570")
    # An ENCRYPTED PKCS8 with password=None -> TypeError, which a random-byte
    # sweep can never synthesize.
    encrypted = key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"passphrase"),
    )
    rsa_der = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    ).private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return [
        b64(pkcs8.replace(ed_oid, bad_oid)),
        b64(spki.replace(ed_oid, bad_oid)),
        b64(encrypted),
        b64(rsa_der),
        b64(pkcs8[:-1]),
        b64(b"\x30\x82\xff\xffnot-der"),
    ]


KEY_BLOBS = _key_blobs()

HOSTILE_VALUES = [
    None,
    True,
    0,
    -1,
    7,
    3.14,
    float("nan"),
    float("inf"),
    Decimal("NaN"),
    Decimal("Infinity"),
    Decimal("-1"),
    "",
    "not-a-value",
    LONE_SURROGATE,
    "sha256:" + "a" * 64,
    b"bytes",
    bytearray(b"bytes"),
    [],
    ["item"],
    (),
    ("item",),
    {},
    {"k": "v"},
    {1: "non-string-key"},
    {"nested"},
    object(),
    # Datetime boundaries — the OverflowError class.
    datetime.min,
    datetime.max,
    datetime.min.replace(tzinfo=UTC),
    datetime.max.replace(tzinfo=UTC),
    datetime.min.replace(tzinfo=timezone(timedelta(minutes=1))),
    datetime.max.replace(tzinfo=timezone(timedelta(minutes=-1))),
    datetime(2026, 8, 3, 12, 0),  # naive
    datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    # ISO strings at the same boundaries, for the from_payload paths.
    "0001-01-01T00:00:00+00:01",
    "9999-12-31T23:59:59-00:01",
    "2026-08-03T12:00:00Z",
    "2026-08-03T12:00:00",  # no timezone
    *KEY_BLOBS,
]


def _recursive_values() -> list[object]:
    """Self-referential and deeply-nested inputs.

    The file's own docstring names "recursion depth" as a class mutation
    testing cannot find, and then did not supply a single value that reaches
    it. Both shapes below escaped as RecursionError until the depth bound.
    """

    cyclic_dict: dict = {}
    cyclic_dict["self"] = cyclic_dict
    cyclic_list: list = []
    cyclic_list.append(cyclic_list)
    deep = json.loads('{"a":' * 1500 + "1" + "}" * 1500)
    deep_list = json.loads("[" * 1500 + "1" + "]" * 1500)
    return [cyclic_dict, cyclic_list, deep, deep_list]


HOSTILE_VALUES += _recursive_values()

# Exhausted and one-shot iterables: the validate-then-recopy class.
HOSTILE_VALUES += [
    iter([]),
    iter(["item"]),
    (x for x in []),
    (x for x in ["item"]),
    range(0),
    range(3),
]


def _stable_id(value: object) -> str:
    """Deterministic test ids. `repr(object())` embeds a memory address, which
    makes ids differ per run and breaks `pytest --lf` reproducibility."""

    try:
        text = repr(value)
    except Exception:  # a hostile __repr__ must not break collection
        return f"{type(value).__name__}-unreprable"
    if " at 0x" in text or len(text) > 40:
        return f"{type(value).__name__}-{abs(hash(text)) % 10000:04d}"
    return text


NOW_AWARE = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def _trust_payload(trust) -> dict:
    return {
        "target": trust.target,
        "route": trust.route,
        "key_id": trust.key_id,
        "public_key_spki_b64": trust.public_key_spki_b64,
        "public_key_sha256": trust.public_key_sha256,
        "owner_binding_sha256": trust.owner_binding_sha256,
        "companion_id": trust.companion_id,
        "agent_id": trust.agent_id,
    }


def _signed_response_fixture():
    """A valid (request, response, trust) triple, built once per call."""

    import base64
    import hashlib

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    key = Ed25519PrivateKey.generate()
    private_der = key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_der = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    route = (
        "/api/kestrel/companions/00000000-0000-4000-8000-000000000001"
        "/agent/invoke/attested"
    )
    signer = si.InvokeReceiptSigner(
        private_key_pkcs8_b64=b64(private_der), key_id="frinz-test-key"
    )
    trust = si.ReceiptTrust(
        target="frinz_companion_kite",
        route=route,
        key_id=signer.key_id,
        public_key_spki_b64=b64(public_der),
        public_key_sha256="sha256:" + hashlib.sha256(public_der).hexdigest(),
        owner_binding_sha256="sha256:" + "1" * 64,
        companion_id="00000000-0000-4000-8000-000000000001",
        agent_id="kite",
    )
    request = si.AttestedInvokeRequest(
        run_id="run-20260803-0001",
        phase="lora_submit",
        request_id="request-lora-submit-0001",
        input="private invocation input",
        model="qwen3:8b",
        provider="runpod",
        session_id="session-0001",
        operation_digest="sha256:" + "2" * 64,
        quote_digest="sha256:" + "3" * 64,
        resource_plan_digest="sha256:" + "4" * 64,
    )
    body = "private invocation response"
    evidence = {
        "phase": request.phase,
        "state_transitions": ["queued"],
        "timings_ms": {"total": 1},
    }
    receipt = signer.sign(
        {
            "run_id": request.run_id,
            "phase": request.phase,
            "route": route,
            "request_id": request.request_id,
            "owner_binding_sha256": trust.owner_binding_sha256,
            "companion_id": trust.companion_id,
            "agent_id": trust.agent_id,
            "input_sha256": si.utf8_sha256(request.input),
            "response_sha256": si.utf8_sha256(body),
            "transport": si.FRINZ_AUTHENTICATED_HTTP_TRANSPORT,
            "model": request.model,
            "provider": request.provider,
            "session_id": request.session_id,
            "operation_digest": request.operation_digest,
            "quote_digest": request.quote_digest,
            "resource_plan_digest": request.resource_plan_digest,
            "evidence_digest": si.phase_evidence_sha256(request.phase, evidence),
            "started_at": "2026-08-03T12:00:00Z",
            "completed_at": "2026-08-03T12:00:00.001000Z",
            "elapsed_ms": 1,
            "issued_at": "2026-08-03T12:00:00.001000Z",
            "receipt_id": "receipt-lora-submit-0001",
        }
    )
    response = si.AttestedInvokeResponse(
        response=body,
        model=request.model,
        provider=request.provider,
        session_id=request.session_id,
        phase_evidence=evidence,
        invocation_receipt=receipt,
    )
    return request, response, trust


def _assert_contained(fn, contract: type[Exception], label: str) -> None:
    """Call `fn`; anything but `contract` or a clean return is an escape."""

    try:
        fn()
    except contract:
        return
    except Exception as exc:  # noqa: BLE001 - the whole point is the open set
        pytest.fail(
            f"{label} escaped {contract.__name__}: "
            f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers that take arbitrary caller input
# ---------------------------------------------------------------------------

SIGNED_CALLABLES = [
    ("canonical_json", lambda v: si.canonical_json(v)),
    ("canonical_sha256", lambda v: si.canonical_sha256(v)),
    ("utf8_sha256", lambda v: si.utf8_sha256(v)),
    ("base64url_encode", lambda v: si.base64url_encode(v)),
    ("base64url_decode", lambda v: si.base64url_decode(v)),
    ("phase_evidence_sha256(phase)", lambda v: si.phase_evidence_sha256(v, {})),
    (
        "phase_evidence_sha256(evidence)",
        lambda v: si.phase_evidence_sha256("lora_submit", v),
    ),
    ("_iso", lambda v: si._iso(v)),
    ("_parse_time", lambda v: si._parse_time(v, "t")),
    ("_safe_identifier", lambda v: si._safe_identifier("n", v)),
    ("_sha256", lambda v: si._sha256("n", v)),
    ("_optional_sha256", lambda v: si._optional_sha256("n", v)),
    ("_optional_identifier", lambda v: si._optional_identifier("n", v)),
    ("_relative_route", lambda v: si._relative_route("n", v)),
    ("_strict_int", lambda v: si._strict_int(v, "n")),
    ("_canonical_json_object", lambda v: si._canonical_json_object(v, "n")),
    ("_require_exact_json_values", lambda v: si._require_exact_json_values(v, "n")),
]

DOGFOOD_CALLABLES = [
    ("_iso", lambda v: dc._iso(v)),
    ("_parse_time", lambda v: dc._parse_time(v, "t")),
    ("_safe_identifier", lambda v: dc._safe_identifier("n", v)),
    ("_is_sha256", lambda v: dc._is_sha256(v)),
    ("_required_payload_string", lambda v: dc._required_payload_string(v, "n")),
    ("_assert_content_free", lambda v: dc._assert_content_free(v)),
    ("_materialized_sequence", lambda v: dc._materialized_sequence(v, "n")),
    ("_materialized_mapping", lambda v: dc._materialized_mapping(v, "n")),
]


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
@pytest.mark.parametrize("label,fn", SIGNED_CALLABLES, ids=lambda x: x if isinstance(x, str) else "")
def test_signed_invocations_helpers_contain_every_hostile_value(label, fn, value):
    _assert_contained(lambda: fn(value), si.SignedInvocationError, f"si.{label}")


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
@pytest.mark.parametrize("label,fn", DOGFOOD_CALLABLES, ids=lambda x: x if isinstance(x, str) else "")
def test_dogfood_helpers_contain_every_hostile_value(label, fn, value):
    _assert_contained(lambda: fn(value), dc.DogfoodError, f"dc.{label}")


# ---------------------------------------------------------------------------
# from_payload: one hostile value per field slot, per type
# ---------------------------------------------------------------------------


def _valid_payloads() -> list[tuple[str, type, dict]]:
    """A well-formed payload for every from_payload on the public surface."""

    identity = dc.ResourceIdentity(
        resource_type=dc.ResourceType.POD,
        resource_id="pod-0001",
        resource_name="kite-pod",
    )
    expected = dc.ExpectedResource(
        resource_type=dc.ResourceType.POD,
        resource_name="kite-pod",
        lane=dc.DogfoodLane.LORA,
    )
    plan = dc.ResourcePlan(
        run_id="run-0001",
        phase=dc.DogfoodPhase.LORA_SUBMIT,
        lane=dc.DogfoodLane.LORA,
        plan_id="plan-0001",
        cleanup_family_id="family-0001",
        expected_resources=(expected,),
    )
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    attempt = dc.ProviderAttemptIdentity(
        run_id="run-0001",
        attempt_id="attempt-0001",
        phase=dc.DogfoodPhase.LORA_SUBMIT,
        lane=dc.DogfoodLane.LORA,
        plan_digest="sha256:" + "a" * 64,
        quote_digest="sha256:" + "a" * 64,
        resource=identity,
        provider_operation_id="op-0001",
        exclusive_window_sha256="sha256:" + "a" * 64,
        started_at=now,
        completed_at=now + timedelta(seconds=30),
    )
    return [
        ("ResourceIdentity", dc.ResourceIdentity, identity.to_payload()),
        ("ExpectedResource", dc.ExpectedResource, expected.to_payload()),
        ("ResourcePlan", dc.ResourcePlan, plan.to_payload()),
        ("ProviderAttemptIdentity", dc.ProviderAttemptIdentity, attempt.to_payload()),
    ]


@pytest.mark.parametrize("name,cls,payload", _valid_payloads(), ids=lambda x: x if isinstance(x, str) else "")
def test_dogfood_from_payload_contains_a_hostile_value_in_every_field(
    name, cls, payload
):
    """Every field slot, every hostile value — nothing but DogfoodError."""

    for field in payload:
        for value in HOSTILE_VALUES:
            corrupted = {**payload, field: value}
            _assert_contained(
                lambda p=corrupted: cls.from_payload(p),
                dc.DogfoodError,
                f"{name}.from_payload[{field}={value!r}]",
            )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
def test_dogfood_from_payload_contains_a_hostile_envelope(value):
    for name, cls, _payload in _valid_payloads():
        _assert_contained(
            lambda c=cls: c.from_payload(value),
            dc.DogfoodError,
            f"{name}.from_payload(envelope)",
        )


def test_signed_from_payload_contains_a_hostile_value_in_every_field():
    """Same sweep for the signed-invocation payload types."""

    request = si.AttestedInvokeRequest(
        run_id="run-0001",
        phase="lora_submit",
        request_id="request-0001",
        input="private input",
    )
    payload = request.to_payload()
    for field in payload:
        for value in HOSTILE_VALUES:
            corrupted = {**payload, field: value}
            _assert_contained(
                lambda p=corrupted: si.AttestedInvokeRequest.from_payload(p),
                si.SignedInvocationError,
                f"AttestedInvokeRequest.from_payload[{field}={value!r}]",
            )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
def test_signed_from_payload_contains_a_hostile_envelope(value):
    for cls in (
        si.AttestedInvokeRequest,
        si.AttestedInvokeResponse,
        si.ReceiptTrust,
        si.ServerInvokeReceipt,
    ):
        _assert_contained(
            lambda c=cls: c.from_payload(value),
            si.SignedInvocationError,
            f"{cls.__name__}.from_payload(envelope)",
        )


# ---------------------------------------------------------------------------
# Constructors and the verification boundary
# ---------------------------------------------------------------------------


def test_dogfood_constructors_contain_a_hostile_value_in_every_field():
    """Direct construction is the harness's route and bypasses from_payload."""

    baselines = [
        (
            dc.ResourceIdentity,
            dict(
                resource_type=dc.ResourceType.POD,
                resource_id="pod-0001",
                resource_name="kite-pod",
            ),
        ),
        (
            dc.ExpectedResource,
            dict(
                resource_type=dc.ResourceType.POD,
                resource_name="kite-pod",
                lane=dc.DogfoodLane.LORA,
            ),
        ),
        (
            dc.PhaseObservation,
            dict(
                phase=dc.DogfoodPhase.LORA_SUBMIT,
                state_transitions=("queued",),
                timings_ms={"total": 1},
            ),
        ),
        (
            dc.SpendQuote,
            dict(
                run_id="run-0001",
                lane=dc.DogfoodLane.LORA,
                quote_id="quote-0001",
                estimated_cost_usd=Decimal("1.00"),
                hard_cap_usd=Decimal("5.00"),
                observed_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
                expires_at=datetime(2026, 8, 3, 12, 5, tzinfo=UTC),
            ),
        ),
    ]
    for cls, baseline in baselines:
        for field in baseline:
            for value in HOSTILE_VALUES:
                kwargs = {**baseline, field: value}
                _assert_contained(
                    lambda c=cls, k=kwargs: c(**k),
                    dc.DogfoodError,
                    f"{cls.__name__}({field}={value!r})",
                )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
def test_signer_and_verifier_boundaries_contain_every_hostile_value(value):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import base64

    key = Ed25519PrivateKey.generate()
    der = key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    b64 = base64.urlsafe_b64encode(der).decode().rstrip("=")
    signer = si.InvokeReceiptSigner(private_key_pkcs8_b64=b64, key_id="k")
    SWEPT_SI_ENTRY_POINTS.add("InvokeReceiptSigner")

    _assert_contained(
        lambda: si.InvokeReceiptSigner(private_key_pkcs8_b64=value, key_id="k"),
        si.SignedInvocationError,
        "InvokeReceiptSigner(private_key)",
    )
    _assert_contained(
        lambda: signer.sign(value), si.SignedInvocationError, "signer.sign(payload)"
    )
    _assert_contained(
        lambda: si.InvokeReceiptVerifier(value),
        si.SignedInvocationError,
        "InvokeReceiptVerifier(trusts)",
    )
    _assert_contained(
        lambda: si.InvokeReceiptVerifier.verify_phase_evidence(value, {}),
        si.SignedInvocationError,
        "verify_phase_evidence(receipt)",
    )


# ---------------------------------------------------------------------------
# The surface this file actually sweeps, declared as data.
#
# The previous rot guard was name-level: `covered_si` was a hardcoded string
# set, so listing a class name satisfied it whether or not any input ever
# reached that class. It reported full coverage while never sweeping
# AttestedInvokeResponse, ServerInvokeReceipt or ReceiptTrust constructors or
# from_payload slots, InvokeReceiptVerifier.verify in ANY of its argument
# positions, ResourcePlan/ProviderAttemptIdentity constructors, 19 of
# PhaseObservation's 22 fields, or 3 of SpendQuote's 10 - and the escape found
# this round lived in one of the skipped positions.
#
# Coverage is now derived from the sweeps below rather than asserted alongside
# them.
# ---------------------------------------------------------------------------

SWEPT_SI_ENTRY_POINTS: set[str] = set()
SWEPT_DC_ENTRY_POINTS: set[str] = set()


def _sweep_callable(label, fn, value, contract, registry):
    registry.add(label.split("(")[0].split(".")[0])
    _assert_contained(lambda: fn(value), contract, label)


def _sweep_every_field(cls, baseline, value, contract, registry):
    """Every field of `cls`, one hostile value at a time."""

    registry.add(cls.__name__)
    for field in baseline:
        kwargs = {**baseline, field: value}
        _assert_contained(
            lambda c=cls, k=kwargs: c(**k),
            contract,
            f"{cls.__name__}({field}=<hostile>)",
        )


def _sweep_every_payload_slot(cls, payload, value, contract, registry):
    registry.add(cls.__name__)
    for field in payload:
        corrupted = {**payload, field: value}
        _assert_contained(
            lambda c=cls, p=corrupted: c.from_payload(p),
            contract,
            f"{cls.__name__}.from_payload({field}=<hostile>)",
        )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
def test_signed_payload_types_contain_a_hostile_value_in_every_slot(value):
    """Constructors AND from_payload slots for every signed payload type.

    None of these were swept before; the RecursionError escape lived in
    AttestedInvokeResponse's phase_evidence slot.
    """

    _request, response, trust = _signed_response_fixture()
    receipt = response.invocation_receipt
    for cls, payload in (
        (si.AttestedInvokeRequest, _request.to_payload()),
        (si.AttestedInvokeResponse, response.to_payload()),
        (si.ReceiptTrust, _trust_payload(trust)),
        (si.ServerInvokeReceipt, receipt.to_payload()),
    ):
        _sweep_every_payload_slot(
            cls, payload, value, si.SignedInvocationError, SWEPT_SI_ENTRY_POINTS
        )

    # Constructors, via dataclasses.replace on a valid instance.
    from dataclasses import fields, replace

    for instance in (_request, response, trust, receipt):
        SWEPT_SI_ENTRY_POINTS.add(type(instance).__name__)
        for f in fields(instance):
            _assert_contained(
                lambda i=instance, n=f.name: replace(i, **{n: value}),
                si.SignedInvocationError,
                f"{type(instance).__name__}({f.name}=<hostile>)",
            )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
def test_verify_contains_a_hostile_value_in_every_argument_position(value):
    """`verify` is the most authority-bearing entry point and was never swept."""

    request, response, trust = _signed_response_fixture()
    verifier = si.InvokeReceiptVerifier((trust,))
    SWEPT_SI_ENTRY_POINTS.add("InvokeReceiptVerifier")
    base = {
        "run_id": request.run_id,
        "phase": request.phase,
        "request_id": request.request_id,
        "input_text": request.input,
        "operation_digest": request.operation_digest,
        "quote_digest": request.quote_digest,
        "resource_plan_digest": request.resource_plan_digest,
    }
    for position in base:
        kwargs = {**base, position: value}
        _assert_contained(
            lambda k=kwargs: verifier.verify(response.invocation_receipt, **k),
            si.SignedInvocationError,
            f"verify({position}=<hostile>)",
        )
    # Both positional slots of verify_phase_evidence, not just the receipt.
    _assert_contained(
        lambda: si.InvokeReceiptVerifier.verify_phase_evidence(
            response.invocation_receipt, value
        ),
        si.SignedInvocationError,
        "verify_phase_evidence(evidence=<hostile>)",
    )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=_stable_id)
def test_dogfood_remaining_constructors_contain_a_hostile_value(value):
    """ResourcePlan and ProviderAttemptIdentity were never swept; and
    PhaseObservation over 3 of its 22 fields."""

    from dataclasses import fields, replace

    for _name, cls, payload in _valid_payloads():
        if cls in (dc.ResourcePlan, dc.ProviderAttemptIdentity):
            instance = cls.from_payload(payload)
            SWEPT_DC_ENTRY_POINTS.add(cls.__name__)
            for f in fields(instance):
                _assert_contained(
                    lambda i=instance, n=f.name: replace(i, **{n: value}),
                    dc.DogfoodError,
                    f"{cls.__name__}({f.name}=<hostile>)",
                )

    observation = dc.PhaseObservation(
        phase=dc.DogfoodPhase.LORA_SUBMIT,
        state_transitions=("queued",),
        timings_ms={"total": 1},
    )
    SWEPT_DC_ENTRY_POINTS.add("PhaseObservation")
    for f in fields(observation):
        _assert_contained(
            lambda n=f.name: replace(observation, **{n: value}),
            dc.DogfoodError,
            f"PhaseObservation({f.name}=<hostile>)",
        )
    # to_evidence's two argument positions.
    _assert_contained(
        lambda: observation.to_evidence(run_id=value, observed_at=NOW_AWARE),
        dc.DogfoodError,
        "to_evidence(run_id=<hostile>)",
    )
    _assert_contained(
        lambda: observation.to_evidence(run_id="run-0001", observed_at=value),
        dc.DogfoodError,
        "to_evidence(observed_at=<hostile>)",
    )


def test_no_public_name_in_either_module_is_left_unswept():
    """Rot guard for BOTH modules, derived from what the sweeps actually hit.

    dogfood_contracts had no rot guard at all and twelve public names; adding a
    thirteenth shipped uncovered, which is the exact failure the docstring
    claims this file prevents.
    """

    def public_names(module):
        return {
            name
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", module.__name__) == module.__name__
        }

    # Error types and enums carry no caller input of their own.
    inert = {
        "SignedInvocationError", "DogfoodError", "DogfoodSafetyError",
        "DogfoodLane", "DogfoodPhase", "ResourceType",
    }
    si_public = public_names(si) - inert
    dc_public = public_names(dc) - inert
    si_covered = SWEPT_SI_ENTRY_POINTS | {
        label.split("(")[0] for label, _ in SIGNED_CALLABLES
    }
    dc_covered = SWEPT_DC_ENTRY_POINTS | {
        label.split("(")[0] for label, _ in DOGFOOD_CALLABLES
    } | {"ResourceIdentity", "ExpectedResource", "SpendQuote"}
    assert not (si_public - si_covered), (
        f"unswept in signed_invocations: {sorted(si_public - si_covered)}"
    )
    assert not (dc_public - dc_covered), (
        f"unswept in dogfood_contracts: {sorted(dc_public - dc_covered)}"
    )
