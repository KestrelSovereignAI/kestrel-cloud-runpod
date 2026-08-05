"""Canonical Ed25519 receipts for authenticated agent invocations.

This module is intentionally independent of the dogfood orchestrator.  A
serving boundary can import the signer and receipt types without importing any
provider lifecycle code, while an external verifier can pin the exact route,
owner, companion, agent, and public key that identify the execution target.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

INVOCATION_RECEIPT_CONTRACT = "frinz-agent-invocation-receipt-v1"
INVOCATION_RECEIPT_SIGNATURE_ALGORITHM = "Ed25519"
FRINZ_AUTHENTICATED_HTTP_TRANSPORT = "frinz_authenticated_http"

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,254}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELATIVE_ROUTE = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@/-]{1,511}$")


class SignedInvocationError(ValueError):
    """A signed invocation object failed its exact contract."""


def canonical_json(value: object) -> str:
    """Serialize one JSON value into the only signed representation."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SignedInvocationError("signed value is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    """Return the prefixed digest of a canonical JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def utf8_sha256(value: str) -> str:
    """Return the prefixed digest of exact UTF-8 text."""

    if not isinstance(value, str):
        raise SignedInvocationError("signed text must be a string")
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def phase_evidence_sha256(phase: str, evidence_payload: Mapping[str, Any]) -> str:
    """Digest exact phase evidence after proving its phase discriminator."""

    _safe_identifier("phase evidence phase", phase)
    normalized = _canonical_json_object(evidence_payload, "phase evidence")
    if normalized.get("phase") != phase:
        raise SignedInvocationError(
            "phase evidence does not bind the exact invocation phase"
        )
    return canonical_sha256(normalized)


def base64url_encode(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise SignedInvocationError("base64url input must be bytes")
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def base64url_decode(value: str, name: str = "signed value") -> bytes:
    # The `"=" in value` clause is unreachable for the same reason as the
    # `"://"` clause in dogfood_contracts._safe_identifier: the preceding
    # character class excludes "=", so nothing that fullmatches can contain
    # one. Kept as a second line of defence should the class ever be widened
    # to admit padding. What actually rejects padded input is the class, and
    # that is what the tests pin - see
    # test_base64url_alphabet_is_what_rejects_padding.
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
        or "=" in value
    ):
        raise SignedInvocationError(f"{name} must be unpadded base64url")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as exc:
        raise SignedInvocationError(f"{name} is not valid base64url") from exc


def _safe_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise SignedInvocationError(f"{name} must be a safe identifier")
    return value


def _required_payload_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SignedInvocationError(f"{name} must be a string")
    return value


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SignedInvocationError(f"{name} must be a JSON object with string keys")
    _require_exact_json_values(value, name)
    normalized = json.loads(canonical_json(value))
    # Unreachable in the same structural sense as the `"="` and `"://"`
    # clauses: `value` is a proven Mapping by this line, so its canonical JSON
    # always round-trips to a dict. Kept as defence in depth, and narrated here
    # so the next reader does not spend time looking for the input that hits
    # it.
    if not isinstance(normalized, dict):
        raise SignedInvocationError(f"{name} must be a JSON object")
    return normalized


def _require_exact_json_values(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        canonical_json(value)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_exact_json_values(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SignedInvocationError(
                    f"{name} must be a JSON object with string keys"
                )
            _require_exact_json_values(item, f"{name}.{key}")
        return
    raise SignedInvocationError(f"{name} contains a non-JSON value")


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SignedInvocationError(f"{name} must be a prefixed SHA-256 digest")
    return value


def _optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _sha256(name, value)


def _optional_identifier(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _safe_identifier(name, value)


def _relative_route(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not _RELATIVE_ROUTE.fullmatch(value)
        or "//" in value
        or "/../" in f"{value}/"
        or "/./" in f"{value}/"
    ):
        raise SignedInvocationError(f"{name} must be a strict relative HTTP route")
    return value


def _iso(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SignedInvocationError("signed timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise SignedInvocationError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SignedInvocationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SignedInvocationError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class AttestedInvokeRequest:
    """Exact body accepted by an authenticated attested invoke endpoint."""

    run_id: str
    phase: str
    request_id: str
    input: str = field(repr=False)
    model: str | None = None
    provider: str | None = None
    session_id: str | None = None
    operation_digest: str | None = None
    quote_digest: str | None = None
    resource_plan_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "phase", "request_id"):
            _safe_identifier(f"attested invoke {name}", getattr(self, name))
        if not isinstance(self.input, str) or not self.input:
            raise SignedInvocationError("attested invoke input cannot be empty")
        for name in ("model", "provider", "session_id"):
            _optional_identifier(f"attested invoke {name}", getattr(self, name))
        for name in ("operation_digest", "quote_digest", "resource_plan_digest"):
            _optional_sha256(f"attested invoke {name}", getattr(self, name))

    def to_payload(self) -> dict[str, str | None]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "request_id": self.request_id,
            "input": self.input,
            "model": self.model,
            "provider": self.provider,
            "session_id": self.session_id,
            "operation_digest": self.operation_digest,
            "quote_digest": self.quote_digest,
            "resource_plan_digest": self.resource_plan_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> AttestedInvokeRequest:
        fields = {
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
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise SignedInvocationError("attested invoke request fields differ")
        return cls(
            run_id=_required_payload_string(value["run_id"], "request.run_id"),
            phase=_required_payload_string(value["phase"], "request.phase"),
            request_id=_required_payload_string(
                value["request_id"], "request.request_id"
            ),
            input=_required_payload_string(value["input"], "request.input"),
            model=_optional_payload_string(value["model"], "request.model"),
            provider=_optional_payload_string(value["provider"], "request.provider"),
            session_id=_optional_payload_string(
                value["session_id"], "request.session_id"
            ),
            operation_digest=_optional_sha256(
                "request.operation_digest", value["operation_digest"]
            ),
            quote_digest=_optional_sha256(
                "request.quote_digest", value["quote_digest"]
            ),
            resource_plan_digest=_optional_sha256(
                "request.resource_plan_digest", value["resource_plan_digest"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ReceiptTrust:
    """Pinned signing authority and semantic identity for one execution target."""

    target: str
    route: str
    key_id: str
    public_key_spki_b64: str
    public_key_sha256: str
    owner_binding_sha256: str
    companion_id: str
    agent_id: str

    def __post_init__(self) -> None:
        _safe_identifier("receipt trust target", self.target)
        _relative_route("receipt trust route", self.route)
        for name in ("key_id", "companion_id", "agent_id"):
            _safe_identifier(f"receipt trust {name}", getattr(self, name))
        for name in ("public_key_sha256", "owner_binding_sha256"):
            _sha256(f"receipt trust {name}", getattr(self, name))
        encoded = base64url_decode(self.public_key_spki_b64, "receipt trust public key")
        actual_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if actual_digest != self.public_key_sha256:
            raise SignedInvocationError("receipt trust public key digest differs")
        try:
            public_key = serialization.load_der_public_key(encoded)
        except (UnsupportedAlgorithm, ValueError) as exc:
            # UnsupportedAlgorithm is not a ValueError: well-formed DER whose
            # algorithm OID is unrecognized raises it, so without this the
            # isinstance check below never runs and the module's
            # SignedInvocationError contract leaks a cryptography exception.
            raise SignedInvocationError(
                "receipt trust public key is invalid DER"
            ) from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise SignedInvocationError("receipt trust key must be Ed25519")

    @classmethod
    def from_payload(cls, value: object) -> ReceiptTrust:
        expected = {
            "target",
            "route",
            "key_id",
            "public_key_spki_b64",
            "public_key_sha256",
            "owner_binding_sha256",
            "companion_id",
            "agent_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SignedInvocationError("receipt trust fields differ")
        return cls(
            **{
                name: _required_payload_string(value[name], f"receipt trust {name}")
                for name in expected
            }
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "target": self.target,
            "route": self.route,
            "key_id": self.key_id,
            "public_key_spki_b64": self.public_key_spki_b64,
            "public_key_sha256": self.public_key_sha256,
            "owner_binding_sha256": self.owner_binding_sha256,
            "companion_id": self.companion_id,
            "agent_id": self.agent_id,
        }


@dataclass(frozen=True, slots=True)
class ServerInvokeReceipt:
    """Server-issued content-free invocation evidence signed with Ed25519."""

    receipt_id: str
    run_id: str
    phase: str
    route: str
    request_id: str
    owner_binding_sha256: str
    companion_id: str
    agent_id: str
    input_sha256: str
    response_sha256: str
    transport: str
    model: str | None
    provider: str | None
    session_id: str | None
    operation_digest: str | None
    quote_digest: str | None
    resource_plan_digest: str | None
    evidence_digest: str
    started_at: datetime
    completed_at: datetime
    elapsed_ms: int
    issued_at: datetime
    key_id: str
    public_key_sha256: str
    signature_b64: str
    contract: str = INVOCATION_RECEIPT_CONTRACT
    signature_algorithm: str = INVOCATION_RECEIPT_SIGNATURE_ALGORITHM

    def __post_init__(self) -> None:
        if self.contract != INVOCATION_RECEIPT_CONTRACT:
            raise SignedInvocationError("invocation receipt contract differs")
        if self.signature_algorithm != INVOCATION_RECEIPT_SIGNATURE_ALGORITHM:
            raise SignedInvocationError(
                "invocation receipt signature algorithm differs"
            )
        for name in (
            "receipt_id",
            "run_id",
            "phase",
            "request_id",
            "companion_id",
            "agent_id",
            "transport",
            "key_id",
        ):
            _safe_identifier(f"invocation receipt {name}", getattr(self, name))
        if self.transport != FRINZ_AUTHENTICATED_HTTP_TRANSPORT:
            raise SignedInvocationError(
                "invocation receipt transport is not authenticated Frinz HTTP"
            )
        _relative_route("invocation receipt route", self.route)
        for name in (
            "owner_binding_sha256",
            "input_sha256",
            "response_sha256",
            "evidence_digest",
            "public_key_sha256",
        ):
            _sha256(f"invocation receipt {name}", getattr(self, name))
        for name in ("operation_digest", "quote_digest", "resource_plan_digest"):
            _optional_sha256(f"invocation receipt {name}", getattr(self, name))
        for name in ("model", "provider", "session_id"):
            _optional_identifier(f"invocation receipt {name}", getattr(self, name))
        for name in ("started_at", "completed_at", "issued_at"):
            _iso(getattr(self, name))
        if not self.started_at <= self.completed_at <= self.issued_at:
            raise SignedInvocationError("invocation receipt timestamps are misordered")
        if (
            not isinstance(self.elapsed_ms, int)
            or isinstance(self.elapsed_ms, bool)
            or self.elapsed_ms < 0
        ):
            raise SignedInvocationError("invocation receipt elapsed time is invalid")
        expected_elapsed_ms = round(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )
        if self.elapsed_ms != expected_elapsed_ms:
            raise SignedInvocationError(
                "invocation receipt elapsed time differs from its timestamps"
            )
        signature = base64url_decode(self.signature_b64, "invocation receipt signature")
        if len(signature) != 64:
            raise SignedInvocationError(
                "invocation receipt signature must contain 64 bytes"
            )

    def signed_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "route": self.route,
            "request_id": self.request_id,
            "owner_binding_sha256": self.owner_binding_sha256,
            "companion_id": self.companion_id,
            "agent_id": self.agent_id,
            "input_sha256": self.input_sha256,
            "response_sha256": self.response_sha256,
            "transport": self.transport,
            "model": self.model,
            "provider": self.provider,
            "session_id": self.session_id,
            "operation_digest": self.operation_digest,
            "quote_digest": self.quote_digest,
            "resource_plan_digest": self.resource_plan_digest,
            "evidence_digest": self.evidence_digest,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "elapsed_ms": self.elapsed_ms,
            "issued_at": _iso(self.issued_at),
            "receipt_id": self.receipt_id,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "payload": self.signed_payload(),
            "signature": {
                "algorithm": self.signature_algorithm,
                "key_id": self.key_id,
                "public_key_sha256": self.public_key_sha256,
                "value": self.signature_b64,
            },
        }

    @classmethod
    def from_payload(cls, value: object) -> ServerInvokeReceipt:
        if not isinstance(value, Mapping) or set(value) != {
            "contract",
            "payload",
            "signature",
        }:
            raise SignedInvocationError("invocation receipt envelope fields differ")
        payload = value["payload"]
        signature = value["signature"]
        payload_fields = {
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
            "elapsed_ms",
            "issued_at",
            "receipt_id",
        }
        signature_fields = {"algorithm", "key_id", "public_key_sha256", "value"}
        if (
            not isinstance(payload, Mapping)
            or set(payload) != payload_fields
            or not isinstance(signature, Mapping)
            or set(signature) != signature_fields
        ):
            raise SignedInvocationError("invocation receipt payload fields differ")
        return cls(
            contract=_required_payload_string(value["contract"], "receipt.contract"),
            receipt_id=_required_payload_string(
                payload["receipt_id"], "receipt.receipt_id"
            ),
            run_id=_required_payload_string(payload["run_id"], "receipt.run_id"),
            phase=_required_payload_string(payload["phase"], "receipt.phase"),
            route=_required_payload_string(payload["route"], "receipt.route"),
            request_id=_required_payload_string(
                payload["request_id"], "receipt.request_id"
            ),
            owner_binding_sha256=_required_payload_string(
                payload["owner_binding_sha256"], "receipt.owner_binding_sha256"
            ),
            companion_id=_required_payload_string(
                payload["companion_id"], "receipt.companion_id"
            ),
            agent_id=_required_payload_string(payload["agent_id"], "receipt.agent_id"),
            input_sha256=_required_payload_string(
                payload["input_sha256"], "receipt.input_sha256"
            ),
            response_sha256=_required_payload_string(
                payload["response_sha256"], "receipt.response_sha256"
            ),
            transport=_required_payload_string(
                payload["transport"], "receipt.transport"
            ),
            model=_optional_payload_string(payload["model"], "receipt.model"),
            provider=_optional_payload_string(payload["provider"], "receipt.provider"),
            session_id=_optional_payload_string(
                payload["session_id"], "receipt.session_id"
            ),
            operation_digest=_optional_sha256(
                "receipt.operation_digest", payload["operation_digest"]
            ),
            quote_digest=_optional_sha256(
                "receipt.quote_digest", payload["quote_digest"]
            ),
            resource_plan_digest=_optional_sha256(
                "receipt.resource_plan_digest", payload["resource_plan_digest"]
            ),
            evidence_digest=_required_payload_string(
                payload["evidence_digest"], "receipt.evidence_digest"
            ),
            started_at=_parse_time(payload["started_at"], "receipt.started_at"),
            completed_at=_parse_time(payload["completed_at"], "receipt.completed_at"),
            elapsed_ms=_strict_int(payload["elapsed_ms"], "receipt.elapsed_ms"),
            issued_at=_parse_time(payload["issued_at"], "receipt.issued_at"),
            key_id=_required_payload_string(
                signature["key_id"], "receipt.signature.key_id"
            ),
            public_key_sha256=_required_payload_string(
                signature["public_key_sha256"],
                "receipt.signature.public_key_sha256",
            ),
            signature_b64=_required_payload_string(
                signature["value"], "receipt.signature.value"
            ),
            signature_algorithm=_required_payload_string(
                signature["algorithm"], "receipt.signature.algorithm"
            ),
        )


@dataclass(frozen=True, slots=True)
class AttestedInvokeResponse:
    """Exact successful response returned by an attested invoke endpoint."""

    response: str = field(repr=False)
    model: str | None
    provider: str | None
    session_id: str | None
    phase_evidence: Mapping[str, Any]
    invocation_receipt: ServerInvokeReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.response, str):
            raise SignedInvocationError("attested invoke response must be a string")
        for name in ("model", "provider", "session_id"):
            _optional_identifier(f"attested response {name}", getattr(self, name))
        if not isinstance(self.invocation_receipt, ServerInvokeReceipt):
            raise SignedInvocationError("attested response receipt is not typed")
        receipt = self.invocation_receipt
        if (
            receipt.response_sha256 != utf8_sha256(self.response)
            or receipt.model != self.model
            or receipt.provider != self.provider
            or receipt.session_id != self.session_id
        ):
            raise SignedInvocationError(
                "attested response differs from its signed invocation result"
            )
        normalized_evidence = _canonical_json_object(
            self.phase_evidence, "attested phase evidence"
        )
        if receipt.evidence_digest != phase_evidence_sha256(
            receipt.phase, normalized_evidence
        ):
            raise SignedInvocationError(
                "attested response differs from its signed phase evidence"
            )
        object.__setattr__(self, "phase_evidence", normalized_evidence)

    def to_payload(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "model": self.model,
            "provider": self.provider,
            "session_id": self.session_id,
            # Deep-copied, not aliased. This is a frozen dataclass whose
            # evidence is bound by the receipt's signed digest; handing out the
            # live mapping let a caller mutate `phase_evidence` in place
            # through the returned payload, after which `verify_phase_evidence`
            # failed against the signature. `ServerInvokeReceipt.to_payload()`
            # already builds fresh dicts.
            "phase_evidence": deepcopy(dict(self.phase_evidence)),
            "invocation_receipt": self.invocation_receipt.to_payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> AttestedInvokeResponse:
        fields = {
            "response",
            "model",
            "provider",
            "session_id",
            "phase_evidence",
            "invocation_receipt",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise SignedInvocationError("attested invoke response fields differ")
        evidence = value["phase_evidence"]
        if not isinstance(evidence, Mapping):
            raise SignedInvocationError("attested phase evidence must be an object")
        raw_response = value["response"]
        if not isinstance(raw_response, str):
            raise SignedInvocationError("attested invoke response must be a string")
        return cls(
            response=raw_response,
            model=_optional_payload_string(value["model"], "response.model"),
            provider=_optional_payload_string(value["provider"], "response.provider"),
            session_id=_optional_payload_string(
                value["session_id"], "response.session_id"
            ),
            phase_evidence=dict(evidence),
            invocation_receipt=ServerInvokeReceipt.from_payload(
                value["invocation_receipt"]
            ),
        )


def _optional_payload_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SignedInvocationError(f"{name} must be a string or null")
    return value


def _strict_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SignedInvocationError(f"{name} must be an integer")
    return value


class InvokeReceiptVerifier:
    """Verify a receipt and infer its target only from pinned signed identity."""

    def __init__(self, trusts: Sequence[ReceiptTrust]) -> None:
        # Materialize BEFORE validating, and validate the COPY. Validating the
        # argument and then re-iterating it drains a one-shot input: passing a
        # generator made `any(...)` consume it, `tuple(trusts)` come back
        # empty, and the "trust set is invalid" guard pass over nothing. The
        # verifier then trusted nothing at all, and every later verify() blamed
        # the receipt ("does not identify one pinned execution target") for a
        # defect in the trust set. It fails closed, but silently.
        if isinstance(trusts, (str, bytes)) or not isinstance(trusts, Iterable):
            raise SignedInvocationError("receipt verifier trust set is invalid")
        self._trusts = tuple(trusts)
        if not self._trusts or any(
            not isinstance(item, ReceiptTrust) for item in self._trusts
        ):
            raise SignedInvocationError("receipt verifier trust set is invalid")
        targets = [item.target for item in self._trusts]
        if len(targets) != len(set(targets)):
            raise SignedInvocationError("receipt verifier targets must be unique")

    def verify(
        self,
        receipt: ServerInvokeReceipt,
        *,
        run_id: str,
        phase: str,
        request_id: str,
        input_text: str,
        operation_digest: str | None = None,
        quote_digest: str | None = None,
        resource_plan_digest: str | None = None,
    ) -> str:
        if not isinstance(receipt, ServerInvokeReceipt):
            raise SignedInvocationError("invocation receipt is not typed")
        matching_trusts = tuple(
            trust
            for trust in self._trusts
            if receipt.route == trust.route
            and receipt.owner_binding_sha256 == trust.owner_binding_sha256
            and receipt.companion_id == trust.companion_id
            and receipt.agent_id == trust.agent_id
            and receipt.key_id == trust.key_id
            and receipt.public_key_sha256 == trust.public_key_sha256
        )
        if len(matching_trusts) != 1:
            raise SignedInvocationError(
                "invocation receipt does not identify one pinned execution target"
            )
        trust = matching_trusts[0]
        for name, value in (
            ("run_id", run_id),
            ("phase", phase),
            ("request_id", request_id),
        ):
            _safe_identifier(f"invocation verification {name}", value)
        if (
            receipt.run_id != run_id
            or receipt.phase != phase
            or receipt.request_id != request_id
            or receipt.input_sha256 != utf8_sha256(input_text)
            or receipt.operation_digest != operation_digest
            or receipt.quote_digest != quote_digest
            or receipt.resource_plan_digest != resource_plan_digest
        ):
            raise SignedInvocationError(
                "invocation receipt is not bound to the exact run, route, or request"
            )
        encoded_key = base64url_decode(
            trust.public_key_spki_b64, "receipt trust public key"
        )
        public_key = serialization.load_der_public_key(encoded_key)
        if not isinstance(public_key, Ed25519PublicKey):
            raise SignedInvocationError("pinned receipt key is not Ed25519")
        try:
            public_key.verify(
                base64url_decode(receipt.signature_b64, "invocation receipt signature"),
                canonical_json(receipt.signed_payload()).encode(),
            )
        except InvalidSignature as exc:
            raise SignedInvocationError(
                "invocation receipt signature is invalid"
            ) from exc
        return trust.target

    @staticmethod
    def verify_phase_evidence(
        receipt: ServerInvokeReceipt, evidence_payload: Mapping[str, Any]
    ) -> None:
        if receipt.evidence_digest != phase_evidence_sha256(
            receipt.phase, evidence_payload
        ):
            raise SignedInvocationError(
                "invocation receipt did not bind the exact phase evidence"
            )


class InvokeReceiptSigner:
    """Sign canonical receipt payloads at the authenticated serving boundary."""

    __slots__ = ("_key", "key_id", "public_key_sha256")

    def __init__(self, *, private_key_pkcs8_b64: str, key_id: str) -> None:
        self.key_id = _safe_identifier("invocation receipt signer key_id", key_id)
        encoded = base64url_decode(
            private_key_pkcs8_b64, "invocation receipt signer private key"
        )
        try:
            key = serialization.load_der_private_key(encoded, password=None)
        except (TypeError, UnsupportedAlgorithm, ValueError) as exc:
            # All three arms are reachable and none is a ValueError subclass
            # except the last, while app_lifecycle wraps signer construction in
            # `except ValueError` because SignedInvocationError is this
            # module's whole error contract:
            #   TypeError            - an ENCRYPTED PKCS8 key with
            #     password=None ("Password was not given but private key is
            #     encrypted"). `openssl pkcs8 -topk8` produces encrypted output
            #     by default, so an operator exporting a passphrase-protected
            #     key hits this. base64url_decode accepts the blob, so nothing
            #     rejects it earlier.
            #   UnsupportedAlgorithm - well-formed DER, unrecognized OID.
            #   ValueError           - malformed DER.
            raise SignedInvocationError(
                "invocation receipt signer private key is invalid DER"
            ) from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise SignedInvocationError("invocation receipt signer key must be Ed25519")
        self._key = key
        public_der = key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.public_key_sha256 = "sha256:" + hashlib.sha256(public_der).hexdigest()

    def sign(self, payload: Mapping[str, Any]) -> ServerInvokeReceipt:
        unsigned_envelope = {
            "contract": INVOCATION_RECEIPT_CONTRACT,
            "payload": dict(payload),
            "signature": {
                "algorithm": INVOCATION_RECEIPT_SIGNATURE_ALGORITHM,
                "key_id": self.key_id,
                "public_key_sha256": self.public_key_sha256,
                "value": base64url_encode(bytes(64)),
            },
        }
        unsigned_receipt = ServerInvokeReceipt.from_payload(unsigned_envelope)
        normalized_payload = unsigned_receipt.signed_payload()
        signature = base64url_encode(
            self._key.sign(canonical_json(normalized_payload).encode())
        )
        return ServerInvokeReceipt.from_payload(
            {
                "contract": INVOCATION_RECEIPT_CONTRACT,
                "payload": normalized_payload,
                "signature": {
                    "algorithm": INVOCATION_RECEIPT_SIGNATURE_ALGORITHM,
                    "key_id": self.key_id,
                    "public_key_sha256": self.public_key_sha256,
                    "value": signature,
                },
            }
        )


__all__ = [
    "FRINZ_AUTHENTICATED_HTTP_TRANSPORT",
    "INVOCATION_RECEIPT_CONTRACT",
    "INVOCATION_RECEIPT_SIGNATURE_ALGORITHM",
    "AttestedInvokeRequest",
    "AttestedInvokeResponse",
    "InvokeReceiptSigner",
    "InvokeReceiptVerifier",
    "ReceiptTrust",
    "ServerInvokeReceipt",
    "SignedInvocationError",
    "base64url_decode",
    "base64url_encode",
    "canonical_json",
    "canonical_sha256",
    "phase_evidence_sha256",
    "utf8_sha256",
]
