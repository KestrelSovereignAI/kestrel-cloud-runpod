"""Owner and budget binding for a registered Ollama inference offer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_UP, Decimal
import re

from kestrel_sdk.llm import (
    InferenceLeaseConstraintError, InferenceLeaseQuote, InferenceLeaseRequest, InferencePrivacy,
)

from .signed_invocations import canonical_sha256

REGISTERED_PROVIDER_NAME = "runpod_registered"
REGISTERED_CAPABILITIES = ("chat", "completions", "embeddings", "streaming", "tools")
_USD_PRECISION = Decimal("0.000000001")


@dataclass(frozen=True, slots=True)
class RegisteredQuotePolicy:
    registration_sha256: str
    journal_id: str
    region: str
    model: str
    model_digest: str
    estimated_ready_seconds: int
    quote_ttl_seconds: int
    worker_idle_seconds: int
    max_readiness_seconds: int
    max_session_seconds: int
    non_compute_cost_bound_usd: Decimal

    def __post_init__(self) -> None:
        if (
            not self.region.startswith("US-")
            or re.fullmatch(r"sha256:[a-f0-9]{64}", self.model_digest) is None
            or any(type(value) is not int or value < 1 for value in (
                self.estimated_ready_seconds, self.quote_ttl_seconds, self.worker_idle_seconds,
                self.max_readiness_seconds, self.max_session_seconds,
            ))
            or self.estimated_ready_seconds > self.max_readiness_seconds
            or not self.non_compute_cost_bound_usd.is_finite()
            or self.non_compute_cost_bound_usd < 0
        ):
            raise ValueError("Invalid registered inference quote policy")

    def deadlines(self, request: InferenceLeaseRequest) -> tuple[datetime, datetime]:
        if (
            request.model != self.model or request.runtime != "ollama"
            or request.expected_concurrency != 1
            or request.privacy is InferencePrivacy.PRIVATE_NETWORK
            or set(request.capabilities).difference(REGISTERED_CAPABILITIES)
            or (request.allowed_regions and self.region.lower() not in request.allowed_regions)
            or request.ready_deadline_seconds > self.max_readiness_seconds
            or request.expected_session_seconds > self.max_session_seconds
            or request.idle_ttl_seconds > request.expected_session_seconds
        ):
            raise InferenceLeaseConstraintError("Request is outside registered inference policy")
        ready_before = request.requested_at + timedelta(seconds=request.ready_deadline_seconds)
        hard_deadline = ready_before + timedelta(seconds=request.expected_session_seconds)
        return ready_before, hard_deadline

    def quote(
        self, request: InferenceLeaseRequest, *, rate: Decimal, now: datetime,
    ) -> InferenceLeaseQuote:
        ready_before, hard_deadline = self.deadlines(request)
        cutoff = min(
            now + timedelta(seconds=self.quote_ttl_seconds),
            ready_before - timedelta(seconds=self.estimated_ready_seconds),
        )
        if cutoff <= now or now < request.requested_at:
            raise InferenceLeaseConstraintError("Insufficient registered worker startup window")
        identity, binding = self._identity(request)
        quote = InferenceLeaseQuote(
            quote_id=identity, request_id=request.request_id,
            provider_name=REGISTERED_PROVIDER_NAME, runtime="ollama", region=self.region.lower(),
            privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT, hourly_cost_usd=rate,
            estimated_total_cost_usd=self._cost(request, rate),
            estimated_ready_seconds=self.estimated_ready_seconds, expires_at=cutoff,
            metadata={
                "request_binding": binding, "registration_binding": self.registration_sha256,
                "journal_identity": self.journal_id,
                "model_digest": self.model_digest,
                "hard_deadline": hard_deadline.isoformat(),
            },
        )
        quote.validate_for(request, now=now)
        return quote

    def accept(
        self, request: InferenceLeaseRequest, quote: InferenceLeaseQuote, *,
        refreshed_rate: Decimal, now: datetime,
    ) -> str:
        """Revalidate the original approved scope, not a replacement timestamp."""
        quote.validate_for(request, now=now)
        ready_before, hard_deadline = self.deadlines(request)
        identity, binding = self._identity(request)
        if (
            quote.quote_id != identity or quote.provider_name != REGISTERED_PROVIDER_NAME
            or quote.region != self.region.lower()
            or quote.privacy is not InferencePrivacy.AUTHENTICATED_ENDPOINT
            or quote.estimated_ready_seconds != self.estimated_ready_seconds
            or quote.estimated_total_cost_usd != self._cost(request, quote.hourly_cost_usd)
            or refreshed_rate <= 0 or refreshed_rate > quote.hourly_cost_usd
            or quote.expires_at > ready_before - timedelta(seconds=self.estimated_ready_seconds)
            or quote.metadata != {
                "request_binding": binding, "registration_binding": self.registration_sha256,
                "journal_identity": self.journal_id, "model_digest": self.model_digest,
                "hard_deadline": hard_deadline.isoformat(),
            }
        ):
            raise InferenceLeaseConstraintError("Registered offer no longer binds the exact admission")
        return canonical_sha256({
            "contract": "registered-inference-admission-v1",
            "request_binding": binding, "quote": quote.to_public_dict(),
        }).removeprefix("sha256:")

    def _cost(self, request: InferenceLeaseRequest, rate: Decimal) -> Decimal:
        if not rate.is_finite() or rate <= 0:
            raise InferenceLeaseConstraintError("Registered worker rate is invalid")
        # One worker can at worst be active continuously through the full
        # approved window plus the configured provider idle tail.
        bound = request.ready_deadline_seconds + request.expected_session_seconds + self.worker_idle_seconds
        return (rate * Decimal(bound) / Decimal(3600) + self.non_compute_cost_bound_usd).quantize(
            _USD_PRECISION, rounding=ROUND_UP,
        )

    def _identity(self, request: InferenceLeaseRequest) -> tuple[str, str]:
        digest = canonical_sha256({
            "contract": "registered-inference-offer-v1", "owner": request.owner_id,
            "request": request.to_public_dict(), "registration": self.registration_sha256,
            "journal": self.journal_id, "model_digest": self.model_digest,
        })
        return "runpod-registered-offer:" + digest.removeprefix("sha256:"), digest
