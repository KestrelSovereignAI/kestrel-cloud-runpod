"""Production contract tests for server-issued invocation receipts."""

from __future__ import annotations

import base64
import copy
import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kestrel_cloud_runpod.signed_invocations import (
    FRINZ_AUTHENTICATED_HTTP_TRANSPORT,
    AttestedInvokeRequest,
    AttestedInvokeResponse,
    InvokeReceiptSigner,
    InvokeReceiptVerifier,
    ReceiptTrust,
    ServerInvokeReceipt,
    SignedInvocationError,
    base64url_decode,
    base64url_encode,
    canonical_json,
    canonical_sha256,
    phase_evidence_sha256,
    utf8_sha256,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROUTE = (
    "/api/kestrel/companions/00000000-0000-4000-8000-000000000001/agent/invoke/attested"
)


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _authority() -> tuple[InvokeReceiptSigner, ReceiptTrust]:
    private_key = Ed25519PrivateKey.generate()
    private_der = private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = InvokeReceiptSigner(
        private_key_pkcs8_b64=_b64url(private_der),
        key_id="frinz-test-key",
    )
    trust = ReceiptTrust(
        target="frinz_companion_kite",
        route=ROUTE,
        key_id=signer.key_id,
        public_key_spki_b64=_b64url(public_der),
        public_key_sha256="sha256:" + hashlib.sha256(public_der).hexdigest(),
        owner_binding_sha256="sha256:" + "1" * 64,
        companion_id="00000000-0000-4000-8000-000000000001",
        agent_id="kite",
    )
    return signer, trust


def _request() -> AttestedInvokeRequest:
    return AttestedInvokeRequest(
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


def _receipt_payload(
    request: AttestedInvokeRequest,
    trust: ReceiptTrust,
    response: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "phase": request.phase,
        "route": ROUTE,
        "request_id": request.request_id,
        "owner_binding_sha256": trust.owner_binding_sha256,
        "companion_id": trust.companion_id,
        "agent_id": trust.agent_id,
        "input_sha256": utf8_sha256(request.input),
        "response_sha256": utf8_sha256(response),
        "transport": FRINZ_AUTHENTICATED_HTTP_TRANSPORT,
        "model": request.model,
        "provider": request.provider,
        "session_id": request.session_id,
        "operation_digest": request.operation_digest,
        "quote_digest": request.quote_digest,
        "resource_plan_digest": request.resource_plan_digest,
        "evidence_digest": phase_evidence_sha256(request.phase, evidence),
        "started_at": "2026-08-03T12:00:00Z",
        "completed_at": "2026-08-03T12:00:00.001000Z",
        "elapsed_ms": 1,
        "issued_at": "2026-08-03T12:00:00.001000Z",
        "receipt_id": "receipt-lora-submit-0001",
    }


def _signed_response() -> tuple[
    AttestedInvokeRequest, AttestedInvokeResponse, ReceiptTrust
]:
    signer, trust = _authority()
    request = _request()
    response = "private invocation response"
    evidence = {
        "phase": request.phase,
        "state_transitions": ["queued"],
        "timings_ms": {"total": 1},
    }
    receipt = signer.sign(_receipt_payload(request, trust, response, evidence))
    return (
        request,
        AttestedInvokeResponse(
            response=response,
            model=request.model,
            provider=request.provider,
            session_id=request.session_id,
            phase_evidence=evidence,
            invocation_receipt=receipt,
        ),
        trust,
    )


def test_attested_request_is_exact_digest_bound_and_hides_input_from_repr():
    request = _request()
    assert AttestedInvokeRequest.from_payload(request.to_payload()) == request
    assert request.input not in repr(request)

    unknown = {**request.to_payload(), "unknown": True}
    with pytest.raises(SignedInvocationError, match="request fields differ"):
        AttestedInvokeRequest.from_payload(unknown)
    with pytest.raises(SignedInvocationError, match="operation_digest"):
        replace(request, operation_digest="not-a-digest")


def test_safe_identifiers_are_limited_to_255_characters():
    request = replace(_request(), run_id="a" * 255)
    assert request.run_id == "a" * 255
    with pytest.raises(SignedInvocationError, match="safe identifier"):
        replace(request, run_id="a" * 256)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "phase",
        "request_id",
        "input",
        "model",
        "provider",
        "session_id",
        "operation_digest",
        "quote_digest",
        "resource_plan_digest",
    ],
)
@pytest.mark.parametrize("wrong_value", [7, True])
def test_attested_request_rejects_every_wrong_string_field_type(field, wrong_value):
    payload = _request().to_payload()
    payload[field] = wrong_value
    with pytest.raises(SignedInvocationError):
        AttestedInvokeRequest.from_payload(payload)


def test_canonical_digest_and_phase_evidence_reject_ambiguous_values():
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert canonical_sha256({"a": 1}) == canonical_sha256({"a": 1})
    with pytest.raises(SignedInvocationError, match="canonical JSON"):
        canonical_json({"not_finite": float("nan")})
    with pytest.raises(SignedInvocationError, match="exact invocation phase"):
        phase_evidence_sha256("lora_submit", {"phase": "lora_poll"})
    with pytest.raises(SignedInvocationError, match="string keys"):
        phase_evidence_sha256("lora_submit", {"phase": "lora_submit", 1: "bad"})


def test_signed_response_round_trips_and_rejects_unknown_fields():
    _request_value, response, _trust = _signed_response()
    payload = response.to_payload()
    assert AttestedInvokeResponse.from_payload(payload) == response
    assert response.response not in repr(response)

    with pytest.raises(SignedInvocationError, match="response fields differ"):
        AttestedInvokeResponse.from_payload({**payload, "unknown": None})
    receipt_payload = response.invocation_receipt.to_payload()
    receipt_payload["payload"]["unknown"] = None
    with pytest.raises(SignedInvocationError, match="payload fields differ"):
        ServerInvokeReceipt.from_payload(receipt_payload)


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "route",
        "key_id",
        "public_key_spki_b64",
        "public_key_sha256",
        "owner_binding_sha256",
        "companion_id",
        "agent_id",
    ],
)
@pytest.mark.parametrize("wrong_value", [7, True])
def test_receipt_trust_rejects_every_wrong_string_field_type(field, wrong_value):
    _signer, trust = _authority()
    payload = trust.to_payload()
    payload[field] = wrong_value
    with pytest.raises(SignedInvocationError):
        ReceiptTrust.from_payload(payload)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "phase",
        "route",
        "request_id",
        "owner_binding_sha256",
        "companion_id",
        "agent_id",
        "input_sha256",
        "response_sha256",
        "transport",
        "model",
        "provider",
        "session_id",
        "operation_digest",
        "quote_digest",
        "resource_plan_digest",
        "evidence_digest",
        "started_at",
        "completed_at",
        "issued_at",
        "receipt_id",
    ],
)
@pytest.mark.parametrize("wrong_value", [7, True])
def test_receipt_rejects_every_wrong_payload_string_field_type(field, wrong_value):
    _request_value, response, _trust = _signed_response()
    envelope = copy.deepcopy(response.invocation_receipt.to_payload())
    envelope["payload"][field] = wrong_value
    with pytest.raises(SignedInvocationError):
        ServerInvokeReceipt.from_payload(envelope)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "contract"),
        ("signature", "algorithm"),
        ("signature", "key_id"),
        ("signature", "public_key_sha256"),
        ("signature", "value"),
    ],
)
@pytest.mark.parametrize("wrong_value", [7, True])
def test_receipt_rejects_every_wrong_envelope_string_field_type(
    section, field, wrong_value
):
    _request_value, response, _trust = _signed_response()
    envelope = copy.deepcopy(response.invocation_receipt.to_payload())
    target = envelope if section is None else envelope[section]
    target[field] = wrong_value
    with pytest.raises(SignedInvocationError):
        ServerInvokeReceipt.from_payload(envelope)


def test_verifier_binds_request_signature_evidence_and_infers_target():
    request, response, trust = _signed_response()
    receipt = response.invocation_receipt
    verifier = InvokeReceiptVerifier((trust,))
    assert (
        verifier.verify(
            receipt,
            run_id=request.run_id,
            phase=request.phase,
            request_id=request.request_id,
            input_text=request.input,
            operation_digest=request.operation_digest,
            quote_digest=request.quote_digest,
            resource_plan_digest=request.resource_plan_digest,
        )
        == trust.target
    )
    verifier.verify_phase_evidence(receipt, response.phase_evidence)

    with pytest.raises(SignedInvocationError, match="exact run, route, or request"):
        verifier.verify(
            receipt,
            run_id=request.run_id,
            phase=request.phase,
            request_id=request.request_id,
            input_text="different input",
            operation_digest=request.operation_digest,
            quote_digest=request.quote_digest,
            resource_plan_digest=request.resource_plan_digest,
        )
    with pytest.raises(SignedInvocationError, match="signature is invalid"):
        verifier.verify(
            replace(receipt, signature_b64="A" * 86),
            run_id=request.run_id,
            phase=request.phase,
            request_id=request.request_id,
            input_text=request.input,
            operation_digest=request.operation_digest,
            quote_digest=request.quote_digest,
            resource_plan_digest=request.resource_plan_digest,
        )
    with pytest.raises(SignedInvocationError, match="exact phase evidence"):
        verifier.verify_phase_evidence(
            receipt,
            {**response.phase_evidence, "timings_ms": {"total": 2}},
        )


def test_signer_normalizes_utc_offsets_before_signing_the_returned_receipt():
    signer, trust = _authority()
    request = _request()
    response = "private invocation response"
    evidence = {
        "phase": request.phase,
        "state_transitions": ["queued"],
        "timings_ms": {"total": 1},
    }
    payload = _receipt_payload(request, trust, response, evidence)
    payload["started_at"] = "2026-08-03T12:00:00+00:00"
    payload["completed_at"] = "2026-08-03T12:00:00.001000+00:00"
    payload["issued_at"] = "2026-08-03T12:00:00.001000+00:00"

    receipt = signer.sign(payload)

    assert receipt.signed_payload()["started_at"] == "2026-08-03T12:00:00Z"
    assert (
        InvokeReceiptVerifier((trust,)).verify(
            receipt,
            run_id=request.run_id,
            phase=request.phase,
            request_id=request.request_id,
            input_text=request.input,
            operation_digest=request.operation_digest,
            quote_digest=request.quote_digest,
            resource_plan_digest=request.resource_plan_digest,
        )
        == trust.target
    )


def test_receipt_elapsed_time_must_equal_rounded_timestamp_interval():
    _request_value, response, _trust = _signed_response()

    with pytest.raises(SignedInvocationError, match="differs from its timestamps"):
        replace(response.invocation_receipt, elapsed_ms=2)


def test_response_rejects_unsigned_result_and_evidence_changes():
    _request_value, response, _trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="signed invocation result"):
        replace(response, response="changed response")
    with pytest.raises(SignedInvocationError, match="signed phase evidence"):
        replace(
            response,
            phase_evidence={
                **response.phase_evidence,
                "state_transitions": ["changed"],
            },
        )
    with pytest.raises(SignedInvocationError, match="misordered"):
        replace(
            response.invocation_receipt,
            completed_at=NOW + timedelta(seconds=2),
            issued_at=NOW + timedelta(seconds=1),
        )


# ---------------------------------------------------------------------------
# Key loading: the module's error contract is SignedInvocationError(ValueError)
# ---------------------------------------------------------------------------


def _mangled_oid(der: bytes) -> bytes:
    """Rewrite the Ed25519 OID 1.3.101.112 to an unassigned 1.2.3.4.

    The result is well-formed DER whose algorithm is unrecognized, which is the
    one input class that raises `UnsupportedAlgorithm` rather than `ValueError`.
    """

    ed25519_oid = bytes.fromhex("06032b6570")
    assert ed25519_oid in der
    return der.replace(ed25519_oid, bytes.fromhex("06032a0304"))


def test_signer_contains_an_unsupported_key_algorithm():
    """`UnsupportedAlgorithm` is not a `ValueError`, so it escaped the contract.

    `frinz/app_lifecycle.py` reads the signer key from the environment and
    wraps construction in `except ValueError`, correct because
    `SignedInvocationError` subclasses it. Without this containment, an
    unassigned-OID blob in that env var kills startup with an unhandled
    cryptography traceback instead of the intended configuration error.
    """

    private_key = Ed25519PrivateKey.generate()
    der = private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with pytest.raises(SignedInvocationError, match="invalid DER"):
        InvokeReceiptSigner(
            private_key_pkcs8_b64=_b64url(_mangled_oid(der)), key_id="frinz-test-key"
        )


def test_receipt_trust_contains_an_unsupported_key_algorithm():
    private_key = Ed25519PrivateKey.generate()
    der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    mangled = _mangled_oid(der)
    with pytest.raises(SignedInvocationError, match="invalid DER"):
        ReceiptTrust(
            target="frinz_companion_kite",
            route=ROUTE,
            key_id="frinz-test-key",
            public_key_spki_b64=_b64url(mangled),
            public_key_sha256="sha256:" + hashlib.sha256(mangled).hexdigest(),
            owner_binding_sha256="sha256:" + "1" * 64,
            companion_id="00000000-0000-4000-8000-000000000001",
            agent_id="kite",
        )


def test_receipt_trust_rejects_an_rsa_key():
    """A supported-but-wrong algorithm takes the isinstance path, not the DER one."""

    from cryptography.hazmat.primitives.asymmetric import rsa

    der = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(SignedInvocationError, match="must be Ed25519"):
        ReceiptTrust(
            target="frinz_companion_kite",
            route=ROUTE,
            key_id="frinz-test-key",
            public_key_spki_b64=_b64url(der),
            public_key_sha256="sha256:" + hashlib.sha256(der).hexdigest(),
            owner_binding_sha256="sha256:" + "1" * 64,
            companion_id="00000000-0000-4000-8000-000000000001",
            agent_id="kite",
        )


def test_receipt_trust_rejects_a_public_key_digest_that_does_not_match_the_key():
    """The digest is what a verifier pins; it must be bound to the actual key."""

    other_der = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    der = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(SignedInvocationError, match="digest differs"):
        ReceiptTrust(
            target="frinz_companion_kite",
            route=ROUTE,
            key_id="frinz-test-key",
            public_key_spki_b64=_b64url(der),
            public_key_sha256="sha256:" + hashlib.sha256(other_der).hexdigest(),
            owner_binding_sha256="sha256:" + "1" * 64,
            companion_id="00000000-0000-4000-8000-000000000001",
            agent_id="kite",
        )


# ---------------------------------------------------------------------------
# Route strictness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route",
    [
        "/api//kestrel/invoke",  # empty segment
        "/api/../admin/invoke",  # parent traversal
        "/api/./invoke",  # current-segment no-op
        "/api/kestrel/..",  # trailing traversal
        "/api/kestrel/.",  # trailing no-op
        "api/kestrel/invoke",  # not absolute
        "https://host/api/invoke",  # absolute URL
        "/",  # no segment
    ],
)
def test_receipt_trust_rejects_a_non_strict_route(route):
    """A pinned route that normalizes differently is a pin on nothing.

    Unlike the identifier alphabet, `_RELATIVE_ROUTE` does admit `/`, so these
    clauses are reachable and each is load-bearing.
    """

    der = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(SignedInvocationError, match="strict relative HTTP route"):
        ReceiptTrust(
            target="frinz_companion_kite",
            route=route,
            key_id="frinz-test-key",
            public_key_spki_b64=_b64url(der),
            public_key_sha256="sha256:" + hashlib.sha256(der).hexdigest(),
            owner_binding_sha256="sha256:" + "1" * 64,
            companion_id="00000000-0000-4000-8000-000000000001",
            agent_id="kite",
        )


# ---------------------------------------------------------------------------
# Trust-set resolution: exactly one target, or nothing
# ---------------------------------------------------------------------------


def _verify(verifier, receipt, request):
    return verifier.verify(
        receipt,
        run_id=request.run_id,
        phase=request.phase,
        request_id=request.request_id,
        input_text=request.input,
        operation_digest=request.operation_digest,
        quote_digest=request.quote_digest,
        resource_plan_digest=request.resource_plan_digest,
    )


def test_verifier_refuses_a_receipt_matching_no_pinned_trust():
    request, response, _trust = _signed_response()
    _signer_b, trust_b = _authority()  # a different key entirely
    verifier = InvokeReceiptVerifier((trust_b,))
    with pytest.raises(SignedInvocationError, match="one pinned execution target"):
        _verify(verifier, response.invocation_receipt, request)


def test_verifier_refuses_a_receipt_matching_more_than_one_trust():
    """Ambiguity must fail closed: two pinned targets means no inferred target."""

    request, response, trust = _signed_response()
    # Same key, route, companion, agent - only `target` differs, and `target`
    # is what verify() returns rather than something it matches on.
    twin = replace(trust, target="frinz_companion_kite_twin")
    verifier = InvokeReceiptVerifier((trust, twin))
    with pytest.raises(SignedInvocationError, match="one pinned execution target"):
        _verify(verifier, response.invocation_receipt, request)


def test_verifier_refuses_a_receipt_signed_by_another_pinned_key():
    """Key A's receipt presented against a trust pinning key B.

    The trust here is otherwise identical - same route, companion, agent and
    key_id - so only the key material and its digest distinguish them.
    """

    request, response, trust = _signed_response()
    other_public_der = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    impostor = replace(
        trust,
        public_key_spki_b64=_b64url(other_public_der),
        public_key_sha256="sha256:" + hashlib.sha256(other_public_der).hexdigest(),
    )
    verifier = InvokeReceiptVerifier((impostor,))
    with pytest.raises(SignedInvocationError, match="one pinned execution target"):
        _verify(verifier, response.invocation_receipt, request)


def test_verifier_rejects_duplicate_targets_and_an_empty_trust_set():
    _request_value, _response, trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="trust set is invalid"):
        InvokeReceiptVerifier(())
    with pytest.raises(SignedInvocationError, match="targets must be unique"):
        InvokeReceiptVerifier((trust, replace(trust, key_id="other-key-id")))


# ---------------------------------------------------------------------------
# The five identity pins the module docstring advertises
#
# "an external verifier can pin the exact route, owner, companion, agent, and
# public key that identify one execution target."  All five are matched in
# `verify()`, and each needs its own test: a trust that differs only by key
# material collapses onto the `public_key_sha256` clause and proves nothing
# about the other four.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("route", ROUTE.replace("invoke/attested", "invoke/other")),
        ("owner_binding_sha256", "sha256:" + "9" * 64),
        ("companion_id", "00000000-0000-4000-8000-00000000dead"),
        ("agent_id", "not-kite"),
        ("key_id", "another-key-id"),
    ],
)
def test_verifier_refuses_a_receipt_whose_identity_differs_in_one_field(field, value):
    """One differing pin is enough to refuse; each is matched independently.

    `owner_binding_sha256` is the cross-owner case: without that clause a
    receipt legitimately issued for owner A verifies against a trust pinned to
    owner B (same route, companion, agent and key) and `verify()` hands back
    B's target.
    """

    request, response, trust = _signed_response()
    verifier = InvokeReceiptVerifier((replace(trust, **{field: value}),))
    with pytest.raises(SignedInvocationError, match="one pinned execution target"):
        _verify(verifier, response.invocation_receipt, request)


# ---------------------------------------------------------------------------
# Transport: the authenticated-transport guarantee
# ---------------------------------------------------------------------------


def test_receipt_refuses_an_unauthenticated_transport():
    """`__post_init__` is the ONLY enforcement of the transport constant.

    `verify()` never reads `receipt.transport`, so if this check goes, a
    receipt carrying `transport: "public_http"` signed by the pinned key
    verifies cleanly and returns the target. The constant is exported in
    `__all__` and advertised in the CHANGELOG as the authenticated-transport
    guarantee, so it needs a test of its own.
    """

    _request_value, response, _trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="authenticated Frinz HTTP"):
        replace(response.invocation_receipt, transport="public_http")


def test_receipt_refuses_a_foreign_contract_or_signature_algorithm():
    _request_value, response, _trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="contract differs"):
        replace(response.invocation_receipt, contract="some-other-contract-v1")
    with pytest.raises(SignedInvocationError, match="signature algorithm differs"):
        replace(response.invocation_receipt, signature_algorithm="ES256")


@pytest.mark.parametrize(
    "route",
    ["/api//kestrel/invoke", "/api/../admin/invoke", "/api/./invoke", "relative/path"],
)
def test_receipt_refuses_a_non_strict_route(route):
    """`ServerInvokeReceipt` calls `_relative_route` too, not just `ReceiptTrust`."""

    _request_value, response, _trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="strict relative HTTP route"):
        replace(response.invocation_receipt, route=route)


# ---------------------------------------------------------------------------
# Binding: every field verify() compares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["run_id", "phase", "request_id", "operation_digest", "quote_digest",
     "resource_plan_digest"],
)
def test_verifier_refuses_a_receipt_not_bound_to_the_exact_request(field):
    """Each equality in the binding check is load-bearing and independent."""

    request, response, trust = _signed_response()
    verifier = InvokeReceiptVerifier((trust,))
    kwargs = {
        "run_id": request.run_id,
        "phase": request.phase,
        "request_id": request.request_id,
        "input_text": request.input,
        "operation_digest": request.operation_digest,
        "quote_digest": request.quote_digest,
        "resource_plan_digest": request.resource_plan_digest,
    }
    kwargs[field] = (
        "sha256:" + "7" * 64 if field.endswith("_digest") else "something-else"
    )
    with pytest.raises(SignedInvocationError, match="exact run, route, or request"):
        verifier.verify(response.invocation_receipt, **kwargs)


def test_verifier_refuses_a_receipt_whose_key_is_not_ed25519():
    """The isinstance guard inside verify(), distinct from ReceiptTrust's."""

    from cryptography.hazmat.primitives.asymmetric import rsa

    request, response, trust = _signed_response()
    der = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    # ReceiptTrust rejects a non-Ed25519 key at construction, so reach the
    # verify()-side guard by swapping the key material in afterwards.
    impostor = replace(trust)
    object.__setattr__(impostor, "public_key_spki_b64", _b64url(der))
    object.__setattr__(
        impostor, "public_key_sha256", "sha256:" + hashlib.sha256(der).hexdigest()
    )
    receipt = replace(
        response.invocation_receipt, public_key_sha256=impostor.public_key_sha256
    )
    verifier = InvokeReceiptVerifier((impostor,))
    with pytest.raises(SignedInvocationError, match="Ed25519"):
        _verify(verifier, receipt, request)


def test_to_payload_does_not_expose_the_internal_phase_evidence_mapping():
    """`to_payload()` handed out its own dict on a frozen dataclass.

    Mutating the returned payload rewrote `response.phase_evidence` in place,
    after which `verify_phase_evidence` failed against the signed digest.
    `ServerInvokeReceipt.to_payload()` already builds fresh dicts.
    """

    _request_value, response, _trust = _signed_response()
    payload = response.to_payload()
    payload["phase_evidence"]["timings_ms"]["total"] = 999
    assert response.phase_evidence["timings_ms"]["total"] == 1


def test_base64url_alphabet_is_what_rejects_padding():
    """Pins the clause that does the work, not the one that reads like it does.

    `base64url_decode` also tests `"=" in value`, but that branch is
    unreachable: `[A-Za-z0-9_-]+` excludes `=`, so nothing that fullmatches can
    contain one. Deleting the `=` clause breaks no test and never could. The
    character class is the real guard; widen it and padded or
    standard-alphabet base64 starts being accepted.
    """

    for padded in ("YWJj=", "YWJ=jZA", "==", "YWJjZA=="):
        with pytest.raises(SignedInvocationError, match="unpadded base64url"):
            base64url_decode(padded, "test value")
    # Standard-alphabet base64 uses "+" and "/"; base64url must reject both.
    for standard in ("ab+cd", "ab/cd"):
        with pytest.raises(SignedInvocationError, match="unpadded base64url"):
            base64url_decode(standard, "test value")
    # Empty is rejected by the `+` quantifier, not by the `=` clause.
    with pytest.raises(SignedInvocationError, match="unpadded base64url"):
        base64url_decode("", "test value")
    assert base64url_decode(base64url_encode(b"\x00\xff round trip")) == (
        b"\x00\xff round trip"
    )


def test_signer_rejects_malformed_der_before_the_ed25519_check():
    """The ValueError arm, plus the base64url rejection that precedes it."""

    # Valid base64url, not valid DER -> the ValueError arm.
    with pytest.raises(SignedInvocationError, match="invalid DER"):
        InvokeReceiptSigner(
            private_key_pkcs8_b64=_b64url(b"\x30\x82\xff\xffnot-a-key"),
            key_id="frinz-test-key",
        )
    # Not valid base64url at all -> rejected before load_der_private_key runs.
    with pytest.raises(SignedInvocationError, match="unpadded base64url"):
        InvokeReceiptSigner(private_key_pkcs8_b64="", key_id="frinz-test-key")


def test_signer_contains_an_encrypted_private_key():
    """The `TypeError` arm — reachable, and by a realistic operator mistake.

    `load_der_private_key(<encrypted PKCS8>, password=None)` raises TypeError,
    which is not a ValueError, so without this arm the blob escapes
    `SignedInvocationError` — the module's whole error contract — and bypasses
    the `except ValueError` in `frinz/app_lifecycle.py` that reads this key
    from the environment.

    `openssl pkcs8 -topk8` produces ENCRYPTED output by default, so an operator
    exporting a passphrase-protected key hits exactly this. `base64url_decode`
    accepts the blob byte-for-byte, so nothing rejects it earlier.
    """

    encrypted_der = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"a-passphrase"),
    )
    with pytest.raises(SignedInvocationError, match="invalid DER"):
        InvokeReceiptSigner(
            private_key_pkcs8_b64=_b64url(encrypted_der), key_id="frinz-test-key"
        )


def test_signer_refuses_a_valid_non_ed25519_private_key():
    """The signer's own Ed25519 isinstance guard — a third such guard."""

    from cryptography.hazmat.primitives.asymmetric import rsa

    der = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    ).private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with pytest.raises(SignedInvocationError, match="must be Ed25519"):
        InvokeReceiptSigner(private_key_pkcs8_b64=_b64url(der), key_id="frinz-test-key")


# ---------------------------------------------------------------------------
# The response envelope's own bindings
#
# `verify()` compares run_id, phase, request_id, input_sha256 and the three
# digests — never model, provider or session_id. `__post_init__` is the sole
# enforcement for those three, structurally identical to `transport`, so each
# needs its own test. Without them a consumer reading `response.model` gets a
# field no signature covers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "gpt-4o-mystery"),
        ("provider", "openai"),
        ("session_id", "attacker-session"),
        ("response", "a substituted response body"),
    ],
)
def test_response_envelope_must_match_its_signed_receipt(field, value):
    _request_value, response, _trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="signed invocation result"):
        replace(response, **{field: value})


def test_response_rejects_an_untyped_receipt_and_a_non_string_body():
    _request_value, response, _trust = _signed_response()
    with pytest.raises(SignedInvocationError, match="receipt is not typed"):
        replace(response, invocation_receipt={"not": "a receipt"})
    # Match the EXACT message. `utf8_sha256` has its own "signed text must be a
    # string" guard two lines later, so a loose "must be a string" pattern is
    # satisfied by either and pins neither.
    with pytest.raises(
        SignedInvocationError, match="attested invoke response must be a string"
    ):
        replace(response, response=b"bytes are not a response body")


def test_utf8_sha256_rejects_non_string_input():
    """The inner guard that shadows the one above; both need their own test."""

    for bad in (b"bytes", None, 7, ["list"]):
        with pytest.raises(SignedInvocationError, match="signed text must be a string"):
            utf8_sha256(bad)


def test_verifier_rejects_an_untyped_receipt():
    request, _response, trust = _signed_response()
    verifier = InvokeReceiptVerifier((trust,))
    with pytest.raises(SignedInvocationError, match="not typed"):
        _verify(verifier, {"not": "a receipt"}, request)


# ---------------------------------------------------------------------------
# Identifier alphabet and signature shape
# ---------------------------------------------------------------------------


def test_safe_identifier_alphabet_is_what_keeps_routes_out_of_receipts():
    """Widening this class makes `https://…` a legal run_id in a SIGNED receipt.

    `dogfood_contracts` pins its twin; this one was unpinned, so admitting "/"
    survived the whole suite. These identifiers are persisted into attestation
    evidence and bound by the signature, so the alphabet is security-relevant.
    """

    from kestrel_cloud_runpod import signed_invocations as si

    assert si._SAFE_IDENTIFIER.pattern == r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,254}$"
    for char in "/ \t\n?#%&=\\":
        assert not si._SAFE_IDENTIFIER.fullmatch(f"a{char}b"), f"{char!r} admitted"
    assert not si._SAFE_IDENTIFIER.fullmatch("https://private.example/x")
    assert not si._SAFE_IDENTIFIER.fullmatch("")  # must start with alphanumeric
    assert not si._SAFE_IDENTIFIER.fullmatch("-leading-dash")
    assert not si._SAFE_IDENTIFIER.fullmatch("a" * 256)  # 255-char ceiling
    assert si._SAFE_IDENTIFIER.fullmatch("a" * 255)


def test_receipt_signature_must_be_exactly_64_bytes():
    """Ed25519 signatures are 64 bytes; a short one must not reach verify()."""

    _request_value, response, _trust = _signed_response()
    for size in (32, 63, 65, 128):
        with pytest.raises(SignedInvocationError, match="64 bytes"):
            replace(response.invocation_receipt, signature_b64=_b64url(b"\x01" * size))


@pytest.mark.parametrize(
    "field", ["run_id", "phase", "request_id"]
)
def test_verify_rejects_a_content_bearing_caller_identifier(field):
    """verify()'s own `_safe_identifier` loop on caller-supplied arguments."""

    request, response, trust = _signed_response()
    verifier = InvokeReceiptVerifier((trust,))
    kwargs = {
        "run_id": request.run_id,
        "phase": request.phase,
        "request_id": request.request_id,
        "input_text": request.input,
        "operation_digest": request.operation_digest,
        "quote_digest": request.quote_digest,
        "resource_plan_digest": request.resource_plan_digest,
    }
    kwargs[field] = "https://private.example/leak"
    with pytest.raises(SignedInvocationError, match="must be a safe identifier"):
        verifier.verify(response.invocation_receipt, **kwargs)


def test_phase_evidence_must_be_an_exact_json_object():
    """`_require_exact_json_values` is removable wholesale without this."""

    signer, trust = _authority()
    request = _request()
    for evidence in (
        {"phase": "lora_submit", "nan": float("nan")},
        {"phase": "lora_submit", "inf": float("inf")},
        {"phase": "lora_submit", "nested": {"bad": float("-inf")}},
        {"phase": "lora_submit", "list": [1, float("nan")]},
    ):
        with pytest.raises(SignedInvocationError):
            phase_evidence_sha256(request.phase, evidence)
    with pytest.raises(SignedInvocationError, match="string keys"):
        phase_evidence_sha256(request.phase, {1: "non-string key"})
    with pytest.raises(SignedInvocationError):
        phase_evidence_sha256(request.phase, "not a mapping")
