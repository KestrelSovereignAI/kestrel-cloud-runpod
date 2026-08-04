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
