"""Product-neutral typed contracts shared with the live dogfood harness.

Extracted from ``dogfood.py`` so production code - notably Frinz - can depend
on these identities, phases and observation shapes WITHOUT importing the
5,000-line live-test orchestrator. Nothing here executes a run, provisions a
resource, or spends money: it is data shapes and their validation only.

The orchestrator, CLI, workspace management, spend gate and benchmarks are to
stay in ``dogfood.py``, which will import these names rather than redefining
them. That rebase has not landed yet: on the branch carrying ``dogfood.py``
those definitions are still declared there, and the structural guard that they
are re-exported rather than re-declared belongs with it.

Imports here are deliberately minimal, and ``test_dogfood_contracts`` pins the
module's own import graph: it must not acquire an edge to ``dogfood`` itself,
to the Runpod control-plane transport (``.clients``), or to any provider
lifecycle code.

That is a guarantee about *this module*, not about the distribution. The
package ``__init__`` is an eager aggregator, so ``import
kestrel_cloud_runpod.dogfood_contracts`` still executes every sibling module;
what the guard buys is that this file stays severable, so the contracts can be
lifted into a lighter distribution without untangling a transport edge first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, cast

from .signed_invocations import (
    canonical_sha256 as _digest,
)


DOGFOOD_CONTRACT = "runpod-kite-live-dogfood-v1"


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,254}$")


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


_TIMING_FIELDS = frozenset(
    {
        "placement",
        "image_pull",
        "container_boot",
        "queue_delay",
        "model_load",
        "input_fetch",
        "inference",
        "training",
        "upload",
        "total",
    }
)


_OPERATIONAL_TIMING_FIELDS = _TIMING_FIELDS | {"cleanup", "billing_reconcile"}


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "bearer",
    "credential",
    "endpoint",
    "prompt",
    "response",
    "secret",
    "signed_url",
    "token",
    "url",
    "weight_bytes",
)


_SENSITIVE_VALUE = re.compile(
    r"(?i)(https?://|\bBearer\s+|api[_-]?key\s*[=:]|"
    r"(?:token|password|secret|signature)\s*[=:])"
)


class DogfoodError(RuntimeError):
    """A required live or isolation invariant was not satisfied."""


class DogfoodSafetyError(DogfoodError):
    """Continuing could touch unowned state, spend money, or leak content."""


class ResourceType(StrEnum):
    POD = "pod"
    SERVERLESS_ENDPOINT = "serverless_endpoint"
    NETWORK_VOLUME = "network_volume"


class DogfoodLane(StrEnum):
    OLLAMA = "ollama"
    LORA = "lora"
    SELFIE = "selfie"


class DogfoodPhase(StrEnum):
    PRE_LORA_SELFIE = "pre_lora_selfie"
    OLLAMA_QUOTE = "ollama_quote"
    OLLAMA_ACQUIRE = "ollama_acquire"
    OLLAMA_INFERENCE = "ollama_inference"
    OLLAMA_STREAM = "ollama_stream"
    OLLAMA_RESTART_RECONCILE = "ollama_restart_reconcile"
    OLLAMA_REPLAY = "ollama_replay"
    OLLAMA_CROSS_OWNER = "ollama_cross_owner"
    OLLAMA_RELEASE = "ollama_release"
    LORA_QUOTE = "lora_quote"
    LORA_SUBMIT = "lora_submit"
    LORA_POLL = "lora_poll"
    LORA_CANCEL_LATE_RESULT = "lora_cancel_late_result"
    LORA_UPLOAD_ACK_INTERRUPT = "lora_upload_ack_interrupt"
    LORA_REPLAY = "lora_replay"
    LORA_CROSS_OWNER = "lora_cross_owner"
    LORA_PROMOTE = "lora_promote"
    SELFIE_QUOTE = "selfie_quote"
    EXPIRED_CAPABILITY = "expired_capability"
    COST_CAP_REFUSAL = "cost_cap_refusal"
    PRIVACY_CLOUD_REFUSAL = "privacy_cloud_refusal"
    POST_LORA_SELFIE = "post_lora_selfie"
    BILLING_RECONCILE = "billing_reconcile"
    CLEANUP = "cleanup"


_RESOURCE_MUTATING_PHASES = (
    DogfoodPhase.OLLAMA_ACQUIRE,
    DogfoodPhase.LORA_SUBMIT,
    DogfoodPhase.LORA_CANCEL_LATE_RESULT,
    DogfoodPhase.LORA_UPLOAD_ACK_INTERRUPT,
    DogfoodPhase.POST_LORA_SELFIE,
)


def _require_aware(value: object) -> datetime:
    """The single place either module decides a datetime is usable.

    `utcoffset()` executes caller-supplied `tzinfo` code, so the CHECK itself
    can raise: a tzinfo returning a non-timedelta gives TypeError, one
    returning an out-of-range offset gives ValueError. Both escape the module
    contract, and DogfoodSafetyError derives from
    RuntimeError.

    This existed inline at six sites; the round that first fixed it patched
    two and left four live. One helper, one handler, so there is no next four.
    """

    try:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise DogfoodSafetyError("dogfood timestamps must be timezone-aware")
    except Exception as exc:  # noqa: BLE001 - utcoffset() is arbitrary caller code
        # Deliberately open. `utcoffset()` executes a caller-supplied tzinfo's
        # own implementation, so the set of exceptions it can raise is not
        # ours to enumerate: TypeError and ValueError are the documented ones,
        # but a tzinfo may raise anything at all. An enumerated tuple here was
        # the previous version of this fix, and a tzinfo raising RuntimeError
        # walked straight through it. This is a trust boundary, not a
        # type check.
        if isinstance(exc, DogfoodSafetyError):
            raise
        raise DogfoodSafetyError("dogfood timestamps must be timezone-aware") from exc
    return value


def _iso(value: datetime) -> str:
    # The isinstance check matters as much as the tz one: `observed_at` reaches
    # here straight from `to_evidence`'s caller, so a deserialized ISO *string*
    # used to raise AttributeError ('str' object has no attribute 'tzinfo') out
    # of a module whose advertised contract is DogfoodSafetyError. That is the
    # same escape class as the UnsupportedAlgorithm leak in signed_invocations.
    # This now matches si._iso exactly, modulo the exception type.
    _require_aware(value)
    try:
        # See the note in signed_invocations._iso: aware does not imply
        # representable, and OverflowError is an ArithmeticError that escapes
        # DogfoodError entirely.
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError) as exc:
        raise DogfoodSafetyError(
            "dogfood timestamp is out of representable range"
        ) from exc


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise DogfoodSafetyError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DogfoodSafetyError(f"{name} must be an ISO-8601 timestamp") from exc
    try:
        _require_aware(parsed)
    except DogfoodSafetyError as exc:
        raise DogfoodSafetyError(f"{name} must include a timezone") from exc
    try:
        return parsed.astimezone(UTC)
    except (OverflowError, OSError) as exc:
        raise DogfoodSafetyError(
            f"{name} is out of representable range"
        ) from exc


def _safe_identifier(name: str, value: object) -> str:
    # The `"://" in value` clause is currently unreachable: _SAFE_ID's character
    # class omits "/", so no string can both fullmatch and contain "://". It is
    # kept as a second line of defence should that class ever be widened, and
    # to stay identical to the definition in dogfood.py. What actually enforces
    # URL-freeness here is the character class, so that is what the tests pin -
    # see test_safe_identifier_alphabet_is_what_excludes_urls.
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value) or "://" in value:
        raise DogfoodSafetyError(f"{name} must be a content-free safe identifier")
    return value


def _is_sha256(value: object) -> bool:
    """Whether `value` is a prefixed SHA-256 digest, for ANY input type.

    `re.Pattern.fullmatch` raises TypeError on non-str input, so applying
    `_SHA256.fullmatch` straight to a caller-supplied attribute let a bytes or
    int digest escape `DogfoodSafetyError` — this module's whole contract — the
    same way `UnsupportedAlgorithm` escaped `SignedInvocationError`. Passing
    `hashlib.sha256(x).digest()` where `.hexdigest()` was meant is the ordinary
    way to hit it. `signed_invocations._sha256` has always guarded this; these
    are its twins.
    """

    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _materialized_sequence(value: object, name: str) -> tuple[Any, ...]:
    """Copy a caller-supplied sequence exactly once, type-checking FIRST.

    Ordering matters in both directions and this is the shape that satisfies
    both. Validating and THEN copying re-iterates the input, so a one-shot
    iterable validates on the first pass and is copied from an exhausted
    iterator on the second - the field silently empties. Copying and THEN
    validating fixes that but lets `tuple(None)` raise a bare TypeError,
    escaping DogfoodSafetyError. So: type-check, copy, then validate content.

    str/bytes are rejected rather than exploded into characters.
    """

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise DogfoodSafetyError(f"{name} must be a sequence")
    return tuple(value)


def _materialized_mapping(value: object, name: str) -> dict[str, Any]:
    """Same contract as _materialized_sequence, for a mapping field."""

    if not isinstance(value, Mapping):
        raise DogfoodSafetyError(f"{name} must be a mapping")
    return dict(value)


def _required_payload_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise DogfoodSafetyError(f"{name} must be a string")
    return value


# Matches signed_invocations._MAX_JSON_DEPTH: deeper than any legitimate
# evidence payload, well under the interpreter's recursion limit.
_MAX_EVIDENCE_DEPTH = 64


def _assert_content_free(
    value: object, *, path: str = "evidence", depth: int = 0
) -> None:
    # Same unbounded-recursion shape as the signed_invocations twin. Reaching
    # it today requires calling this helper directly, because `to_evidence`
    # only ever hands it the bounded projection from `binding_payload()` - but
    # it is a module-level helper, the bound costs nothing, and an unbounded
    # recursion raises RecursionError, which is a RuntimeError and so escapes
    # DogfoodError entirely. The bound doubles as cycle detection.
    if depth > _MAX_EVIDENCE_DEPTH:
        raise DogfoodSafetyError(f"{path} nests deeper than {_MAX_EVIDENCE_DEPTH}")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise DogfoodSafetyError(f"{path} has a non-string key")
            normalized = raw_key.lower()
            digest_only = normalized.endswith(("_digest", "_sha256"))
            if not digest_only and any(
                part in normalized for part in _SENSITIVE_KEY_PARTS
            ):
                raise DogfoodSafetyError(
                    f"{path}.{raw_key} is not content-free evidence"
                )
            _assert_content_free(item, path=f"{path}.{raw_key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_content_free(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise DogfoodSafetyError(f"{path} contains a URL or credential-like value")


@dataclass(frozen=True, slots=True, order=True)
class ResourceIdentity:
    """One exact provider identity; URLs and credentials are structurally absent."""

    resource_type: ResourceType
    resource_id: str
    resource_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.resource_type, ResourceType):
            raise DogfoodSafetyError("resource_type must be a ResourceType")
        _safe_identifier("resource_id", self.resource_id)
        _safe_identifier("resource_name", self.resource_name)

    @classmethod
    def from_payload(cls, value: object) -> ResourceIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "resource_type",
            "resource_id",
            "resource_name",
        }:
            raise DogfoodSafetyError("resource identity fields differ")
        try:
            resource_type = ResourceType(
                _required_payload_string(
                    value["resource_type"], "resource.resource_type"
                )
            )
        except (TypeError, ValueError) as exc:
            raise DogfoodSafetyError("resource type is invalid") from exc
        return cls(
            resource_type=resource_type,
            resource_id=_required_payload_string(
                value["resource_id"], "resource.resource_id"
            ),
            resource_name=_required_payload_string(
                value["resource_name"], "resource.resource_name"
            ),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
        }


@dataclass(frozen=True, slots=True)
class ExpectedResource:
    resource_type: ResourceType
    resource_name: str
    lane: DogfoodLane

    def __post_init__(self) -> None:
        if not isinstance(self.resource_type, ResourceType):
            raise DogfoodSafetyError("expected resource_type must be a ResourceType")
        if not isinstance(self.lane, DogfoodLane):
            raise DogfoodSafetyError("expected resource lane must be a DogfoodLane")
        _safe_identifier("expected resource_name", self.resource_name)

    def to_payload(self) -> dict[str, str]:
        return {
            "resource_type": self.resource_type.value,
            "resource_name": self.resource_name,
            "lane": self.lane.value,
        }

    @classmethod
    def from_payload(cls, value: object) -> ExpectedResource:
        if not isinstance(value, Mapping) or set(value) != {
            "resource_type",
            "resource_name",
            "lane",
        }:
            raise DogfoodSafetyError("planned resource fields differ")
        try:
            return cls(
                resource_type=ResourceType(
                    _required_payload_string(
                        value["resource_type"], "planned_resource.resource_type"
                    )
                ),
                resource_name=_required_payload_string(
                    value["resource_name"], "planned_resource.resource_name"
                ),
                lane=DogfoodLane(
                    _required_payload_string(value["lane"], "planned_resource.lane")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DogfoodSafetyError("planned resource values are invalid") from exc


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """Durable provider-ineligible plan registered before dispatch eligibility."""

    run_id: str
    phase: DogfoodPhase
    lane: DogfoodLane
    plan_id: str
    cleanup_family_id: str
    expected_resources: tuple[ExpectedResource, ...]
    initial_resources: tuple[ExpectedResource, ...] | None = None

    def __post_init__(self) -> None:
        # Materialize before validating - see the note in PhaseObservation.
        object.__setattr__(
            self,
            "expected_resources",
            _materialized_sequence(self.expected_resources, "expected_resources"),
        )
        if self.initial_resources is not None:
            object.__setattr__(
                self,
                "initial_resources",
                _materialized_sequence(self.initial_resources, "initial_resources"),
            )
        _safe_identifier("resource plan run_id", self.run_id)
        # isinstance FIRST. DogfoodPhase is a StrEnum, so
        # `"lora_submit" in _RESOURCE_MUTATING_PHASES` is True for a raw str
        # and membership alone lets an untyped phase through. It then reaches
        # `self.phase.value` in to_payload/digest and raises AttributeError -
        # escaping DogfoodSafetyError, this module's whole contract, on the
        # signature-bound path. `lane` below has always had this guard.
        if not isinstance(self.phase, DogfoodPhase):
            raise DogfoodSafetyError("resource plan phase is invalid")
        if self.phase not in _RESOURCE_MUTATING_PHASES:
            raise DogfoodSafetyError("resource plan phase is not mutating")
        if not isinstance(self.lane, DogfoodLane):
            raise DogfoodSafetyError("resource plan lane is invalid")
        _safe_identifier("resource plan_id", self.plan_id)
        _safe_identifier("resource cleanup_family_id", self.cleanup_family_id)
        if not self.expected_resources or any(
            not isinstance(item, ExpectedResource) or item.lane is not self.lane
            for item in self.expected_resources
        ):
            raise DogfoodSafetyError(
                "resource plan requires typed resources in its exact lane"
            )
        identities = {
            (item.resource_type, item.resource_name) for item in self.expected_resources
        }
        if len(identities) != len(self.expected_resources):
            raise DogfoodSafetyError("resource plan identities must be unique")
        initial = (
            self.expected_resources
            if self.initial_resources is None
            else self.initial_resources
        )
        if (
            not initial
            or any(not isinstance(item, ExpectedResource) for item in initial)
            or not set(initial).issubset(set(self.expected_resources))
        ):
            raise DogfoodSafetyError(
                "resource plan initial resources must be a nonempty expected subset"
            )
        # `initial_resources` was already detached; `expected_resources` was
        # not, so appending to the caller's list after construction defeated
        # the same-lane and uniqueness checks above AND changed `digest` -
        # which `ProviderAttemptIdentity.plan_digest` pins against a billable
        # attempt, so the recorded plan digest was not stable.
        object.__setattr__(self, "initial_resources", tuple(initial))

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "lane": self.lane.value,
            "plan_id": self.plan_id,
            "cleanup_family_id": self.cleanup_family_id,
            "expected_resources": [
                item.to_payload() for item in self.expected_resources
            ],
            "initial_resources": [
                item.to_payload()
                for item in cast(tuple[ExpectedResource, ...], self.initial_resources)
            ],
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> ResourcePlan:
        if not isinstance(value, Mapping) or set(value) != {
            "run_id",
            "phase",
            "lane",
            "plan_id",
            "cleanup_family_id",
            "expected_resources",
            "initial_resources",
        }:
            raise DogfoodSafetyError("resource plan fields differ")
        expected = value["expected_resources"]
        initial = value["initial_resources"]
        if not isinstance(expected, list) or not isinstance(initial, list):
            raise DogfoodSafetyError(
                "resource plan expected_resources must be an array"
            )
        try:
            return cls(
                run_id=_required_payload_string(
                    value["run_id"], "resource_plan.run_id"
                ),
                phase=DogfoodPhase(
                    _required_payload_string(value["phase"], "resource_plan.phase")
                ),
                lane=DogfoodLane(
                    _required_payload_string(value["lane"], "resource_plan.lane")
                ),
                plan_id=_required_payload_string(
                    value["plan_id"], "resource_plan.plan_id"
                ),
                cleanup_family_id=_required_payload_string(
                    value["cleanup_family_id"],
                    "resource_plan.cleanup_family_id",
                ),
                expected_resources=tuple(
                    ExpectedResource.from_payload(item) for item in expected
                ),
                initial_resources=tuple(
                    ExpectedResource.from_payload(item) for item in initial
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DogfoodSafetyError("resource plan values are invalid") from exc


@dataclass(frozen=True, slots=True, order=True)
class ProviderAttemptIdentity:
    """One actual billable provider attempt bound to its owned capacity."""

    run_id: str
    attempt_id: str
    phase: DogfoodPhase
    lane: DogfoodLane
    plan_digest: str
    quote_digest: str
    resource: ResourceIdentity
    provider_operation_id: str
    exclusive_window_sha256: str
    started_at: datetime
    completed_at: datetime

    def __post_init__(self) -> None:
        _safe_identifier("provider attempt run_id", self.run_id)
        _safe_identifier("provider attempt attempt_id", self.attempt_id)
        _safe_identifier(
            "provider attempt provider_operation_id", self.provider_operation_id
        )
        # isinstance FIRST - see the note in ResourcePlan.
        if not isinstance(self.phase, DogfoodPhase):
            raise DogfoodSafetyError("provider attempt phase is invalid")
        if self.phase not in _RESOURCE_MUTATING_PHASES:
            raise DogfoodSafetyError("provider attempt phase is not resource mutating")
        if not isinstance(self.lane, DogfoodLane):
            raise DogfoodSafetyError("provider attempt lane is invalid")
        if (
            not _is_sha256(self.plan_digest)
            or not _is_sha256(self.quote_digest)
            or not _is_sha256(self.exclusive_window_sha256)
        ):
            raise DogfoodSafetyError("provider attempt quote or plan digest is invalid")
        if not isinstance(self.resource, ResourceIdentity):
            raise DogfoodSafetyError("provider attempt resource identity is invalid")
        # Awareness through the one helper, so a hostile tzinfo cannot raise
        # from inside the check. The comparison is safe only afterwards.
        _require_aware(self.started_at)
        _require_aware(self.completed_at)
        if self.started_at > self.completed_at:
            raise DogfoodSafetyError("provider attempt interval is invalid")
        # Serializability is part of the contract, not a to_payload() concern:
        # without this an extreme timestamp constructed successfully and raised
        # OverflowError later from PhaseObservation.binding_payload() - the
        # projection Frinz feeds to phase_evidence_sha256.
        _iso(self.started_at)
        _iso(self.completed_at)

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "phase": self.phase.value,
            "lane": self.lane.value,
            "plan_digest": self.plan_digest,
            "quote_digest": self.quote_digest,
            "resource": self.resource.to_payload(),
            "provider_operation_id": self.provider_operation_id,
            "exclusive_window_sha256": self.exclusive_window_sha256,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }

    @classmethod
    def from_payload(cls, value: object) -> ProviderAttemptIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "run_id",
            "attempt_id",
            "phase",
            "lane",
            "plan_digest",
            "quote_digest",
            "resource",
            "provider_operation_id",
            "exclusive_window_sha256",
            "started_at",
            "completed_at",
        }:
            raise DogfoodSafetyError("provider attempt fields differ")
        try:
            phase = DogfoodPhase(
                _required_payload_string(value["phase"], "provider_attempt.phase")
            )
            lane = DogfoodLane(
                _required_payload_string(value["lane"], "provider_attempt.lane")
            )
        except (TypeError, ValueError) as exc:
            raise DogfoodSafetyError(
                "provider attempt phase or lane is invalid"
            ) from exc
        return cls(
            run_id=_required_payload_string(value["run_id"], "provider_attempt.run_id"),
            attempt_id=_required_payload_string(
                value["attempt_id"], "provider_attempt.attempt_id"
            ),
            phase=phase,
            lane=lane,
            plan_digest=_required_payload_string(
                value["plan_digest"], "provider_attempt.plan_digest"
            ),
            quote_digest=_required_payload_string(
                value["quote_digest"], "provider_attempt.quote_digest"
            ),
            resource=ResourceIdentity.from_payload(value["resource"]),
            provider_operation_id=_required_payload_string(
                value["provider_operation_id"],
                "provider_attempt.provider_operation_id",
            ),
            exclusive_window_sha256=_required_payload_string(
                value["exclusive_window_sha256"],
                "provider_attempt.exclusive_window_sha256",
            ),
            started_at=_parse_time(value["started_at"], "attempt.started_at"),
            completed_at=_parse_time(value["completed_at"], "attempt.completed_at"),
        )


@dataclass(frozen=True, slots=True)
class SpendQuote:
    run_id: str
    lane: DogfoodLane
    quote_id: str
    estimated_cost_usd: Decimal
    hard_cap_usd: Decimal
    observed_at: datetime
    expires_at: datetime
    operation_digest: str | None = None
    provider_quote_sha256: str | None = None
    endpoint_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        _safe_identifier("run_id", self.run_id)
        _safe_identifier("quote_id", self.quote_id)
        if not isinstance(self.lane, DogfoodLane):
            raise DogfoodSafetyError("spend quote lane is invalid")
        for name in ("estimated_cost_usd", "hard_cap_usd"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise DogfoodSafetyError(f"spend quote {name} must be positive")
        if self.estimated_cost_usd > self.hard_cap_usd:
            raise DogfoodSafetyError("spend quote estimate exceeds its hard cap")
        for name in (
            "operation_digest",
            "provider_quote_sha256",
            "endpoint_plan_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise DogfoodSafetyError(f"spend quote {name} is invalid")
        try:
            _require_aware(self.observed_at)
            _require_aware(self.expires_at)
        except DogfoodSafetyError as exc:
            raise DogfoodSafetyError("spend quote timestamps must be aware") from exc
        try:
            # `observed_at + timedelta(minutes=5)` overflows near datetime.max,
            # raising OverflowError out of a money-invariant validator.
            latest = self.observed_at + timedelta(minutes=5)
        except (OverflowError, OSError) as exc:
            raise DogfoodSafetyError(
                "spend quote observed_at is out of representable range"
            ) from exc
        if not self.observed_at < self.expires_at <= latest:
            raise DogfoodSafetyError(
                "spend quote may remain valid for at most five minutes"
            )
        # Prove both timestamps are serializable NOW rather than at to_payload
        # time: `digest` is bound into attempts, so a deferred failure would
        # surface on the signature-bound path.
        _iso(self.observed_at)
        _iso(self.expires_at)

    @property
    def digest(self) -> str:
        return _digest(self.to_payload())

    def to_payload(self) -> dict[str, str | None]:
        return {
            "run_id": self.run_id,
            "lane": self.lane.value,
            "quote_id": self.quote_id,
            "estimated_cost_usd": str(self.estimated_cost_usd),
            "hard_cap_usd": str(self.hard_cap_usd),
            "observed_at": _iso(self.observed_at),
            "expires_at": _iso(self.expires_at),
            "operation_digest": self.operation_digest,
            "provider_quote_sha256": self.provider_quote_sha256,
            "endpoint_plan_sha256": self.endpoint_plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class PhaseObservation:
    """Content-free result returned by the trusted live binding."""

    phase: DogfoodPhase
    state_transitions: tuple[str, ...]
    timings_ms: Mapping[str, int]
    artifact_digests: tuple[str, ...] = ()
    estimated_cost_usd: Decimal | None = None
    actual_cost_usd: Decimal | None = None
    billing_receipt_digest: str | None = None
    provider: str | None = None
    model: str | None = None
    product_consent_count: int = 0
    trained_weight_digest: str | None = None
    promoted_weight_digest: str | None = None
    weight_digest_used: str | None = None
    output_image_digest: str | None = None
    uploaded_artifact_digest: str | None = None
    recovered_artifact_digest: str | None = None
    recovered_resource_plan_digest: str | None = None
    provider_ack_interruption_count: int = 0
    recovery_count: int = 0
    publication_count: int = 0
    promotion_count: int = 0
    provider_attempts: tuple[ProviderAttemptIdentity, ...] = ()

    def __post_init__(self) -> None:
        # Materialize BEFORE validating, not after. Validating first and
        # copying afterwards means the content that gets copied is not the
        # content that was validated: for a one-shot iterable the second pass
        # sees an exhausted iterator, so
        #
        #   state_transitions=(t.name for t in log)   # ordinary Python
        #
        # silently became `()` in the signature-bound projection, and the
        # emptiness guard below was bypassed because a generator is always
        # truthy. Detaching here also closes the aliasing case: the caller
        # cannot mutate a validated container afterwards.
        object.__setattr__(
            self,
            "state_transitions",
            _materialized_sequence(self.state_transitions, "state_transitions"),
        )
        object.__setattr__(
            self, "timings_ms", _materialized_mapping(self.timings_ms, "timings_ms")
        )
        object.__setattr__(
            self,
            "artifact_digests",
            _materialized_sequence(self.artifact_digests, "artifact_digests"),
        )
        if not isinstance(self.phase, DogfoodPhase):
            raise DogfoodSafetyError("phase observation has an invalid phase")
        if not self.state_transitions:
            raise DogfoodSafetyError(
                "phase observation requires live state transitions"
            )
        for item in self.state_transitions:
            _safe_identifier("state transition", item)
        # The Mapping type check lives in `_materialized_mapping` above; by
        # here `timings_ms` is a dict, so testing it again would pin nothing.
        if set(self.timings_ms) - _OPERATIONAL_TIMING_FIELDS:
            raise DogfoodSafetyError("phase observation has unknown timing fields")
        for name, value in self.timings_ms.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DogfoodSafetyError(
                    f"timings_ms.{name} must be nonnegative milliseconds"
                )
        for item in self.artifact_digests:
            if not _is_sha256(item):
                raise DogfoodSafetyError("artifact digest must be sha256")
        for name in (
            "billing_receipt_digest",
            "trained_weight_digest",
            "promoted_weight_digest",
            "weight_digest_used",
            "output_image_digest",
            "uploaded_artifact_digest",
            "recovered_artifact_digest",
            "recovered_resource_plan_digest",
        ):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise DogfoodSafetyError(f"{name} must be sha256")
        for name in ("estimated_cost_usd", "actual_cost_usd"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, Decimal) or not value.is_finite() or value < 0
            ):
                raise DogfoodSafetyError(f"{name} must be finite and nonnegative")
        for name in ("provider", "model"):
            value = getattr(self, name)
            if value is not None:
                _safe_identifier(name, value)
        if (
            not isinstance(self.product_consent_count, int)
            or isinstance(self.product_consent_count, bool)
            or self.product_consent_count < 0
        ):
            raise DogfoodSafetyError("product_consent_count must be nonnegative")
        for name in (
            "provider_ack_interruption_count",
            "recovery_count",
            "publication_count",
            "promotion_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DogfoodSafetyError(f"{name} must be nonnegative")
        if not isinstance(self.provider_attempts, tuple) or any(
            not isinstance(item, ProviderAttemptIdentity)
            for item in self.provider_attempts
        ):
            raise DogfoodSafetyError(
                "phase observation provider attempts must be typed identities"
            )
        attempt_ids = [item.attempt_id for item in self.provider_attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise DogfoodSafetyError(
                "phase observation contains duplicate provider attempts"
            )
        # `provider_attempts` needs no detaching: the isinstance check above
        # already requires a tuple, and tuples cannot alias. The other three
        # were materialized at the TOP of this method, before validation.

    def binding_payload(self) -> dict[str, Any]:
        """Canonical content-free projection bound by the server receipt."""

        return {
            "phase": self.phase.value,
            "state_transitions": list(self.state_transitions),
            "timings_ms": dict(sorted(self.timings_ms.items())),
            "artifact_digests": list(self.artifact_digests),
            "estimated_cost_usd": (
                str(self.estimated_cost_usd)
                if self.estimated_cost_usd is not None
                else None
            ),
            "actual_cost_usd": (
                str(self.actual_cost_usd) if self.actual_cost_usd is not None else None
            ),
            "billing_receipt_digest": self.billing_receipt_digest,
            "provider": self.provider,
            "model": self.model,
            "product_consent_count": self.product_consent_count,
            "trained_weight_digest": self.trained_weight_digest,
            "promoted_weight_digest": self.promoted_weight_digest,
            "weight_digest_used": self.weight_digest_used,
            "output_image_digest": self.output_image_digest,
            "uploaded_artifact_digest": self.uploaded_artifact_digest,
            "recovered_artifact_digest": self.recovered_artifact_digest,
            "recovered_resource_plan_digest": self.recovered_resource_plan_digest,
            "provider_ack_interruption_count": self.provider_ack_interruption_count,
            "recovery_count": self.recovery_count,
            "publication_count": self.publication_count,
            "promotion_count": self.promotion_count,
            "provider_attempts": [item.to_payload() for item in self.provider_attempts],
        }

    def to_evidence(self, *, run_id: str, observed_at: datetime) -> dict[str, Any]:
        # ``run_id`` is caller-supplied and was the one run_id in either module
        # that reached a persisted record without passing through
        # ``_safe_identifier`` — every other one is validated at construction.
        # ``_assert_content_free`` below does catch a URL, but as the record's
        # last line of defence rather than as a field validator; validate here
        # too so a malformed run_id is rejected by the same rule everywhere.
        _safe_identifier("evidence run_id", run_id)
        payload: dict[str, Any] = {
            "contract": DOGFOOD_CONTRACT,
            "event": "phase_observation",
            "run_id": run_id,
            "observed_at": _iso(observed_at),
            **self.binding_payload(),
        }
        _assert_content_free(payload)
        return payload

