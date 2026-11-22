"""Exact control-plane effects for an already registered private endpoint."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime

from kestrel_sovereign.execution_custody import is_execution_control_error

from .clients import RunpodControlPlaneClient, endpoint_worker_population_is_empty
from .models import EndpointCreateRequest, EndpointResource, EndpointUpdateRequest, environment_values_sha256
from .ollama_contracts import (
    OLLAMA_REQUIRE_CACHED_MODEL_ENV, OllamaLeaseAuthorizationError, OllamaLeaseConflictError,
    OllamaProvisioningEvidenceError, iso_datetime,
)
from .ollama_provider import validate_registered_ollama_endpoint
from .registered_ollama_journal import RegisteredActivation, RegisteredActivationIntent, RegisteredActivationState, RegisteredOllamaJournal
from .signed_invocations import canonical_sha256

_PARK_MARKER_ENV = "KESTREL_OLLAMA_PARKED_ACTIVATION"
_ACTIVE_MARKER_ENV = "KESTREL_OLLAMA_REGISTERED_ACTIVATION"
_SECRET_REFERENCE = re.compile(r"\{\{ RUNPOD_SECRET_[A-Za-z0-9_]+ \}\}")


@dataclass(frozen=True, slots=True)
class RegisteredOllamaEndpoint:
    endpoint_id: str
    original_created_at: datetime
    baseline: EndpointCreateRequest

    def __post_init__(self) -> None:
        if (
            self.baseline.workers_max != 0 or self.baseline.workers_min != 0
            or len(self.baseline.network_volume_ids) != 1
            or {_PARK_MARKER_ENV, _ACTIVE_MARKER_ENV}.intersection(self.baseline.env)
            or any(
                key in self.baseline.env for key in (
                    "KESTREL_OLLAMA_BEARER_TOKEN", "KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT",
                )
            )
        ):
            raise ValueError("Registered Ollama baseline must be parked without a route bearer")

    @property
    def digest(self) -> str:
        policy = self.baseline.to_payload()
        policy["env"] = environment_values_sha256(self.baseline.env)
        return canonical_sha256({
            "contract": "registered-ollama-endpoint-v1", "id": self.endpoint_id,
            "original_created_at": iso_datetime(self.original_created_at),
            "policy": policy,
        }).removeprefix("sha256:")

    def verify(self, endpoint: EndpointResource, *, max_workers: int, env_sha256: str) -> None:
        validate_registered_ollama_endpoint(
            endpoint, endpoint_id=self.endpoint_id, original_created_at=self.original_created_at,
            expected=replace(self.baseline, workers_max=max_workers),
            expected_environment_sha256=env_sha256,
        )

    def park_environment(self, lease_id: str, admission_sha256: str) -> dict[str, str]:
        # Different markers for distinct operations make a lost PATCH response
        # provable without ever granting a second potentially late write.
        marker = canonical_sha256({
            "contract": "registered-ollama-park-v1", "registration": self.digest,
            "lease_id": lease_id, "admission_sha256": admission_sha256,
        })
        return {**self.baseline.env, _PARK_MARKER_ENV: marker}


class RegisteredOllamaLifecycle:
    """Admitted activation; an uncertain parking retry permanently fences re-use.

    A caller must persist/verify its provider-neutral quote and ownership
    before acquire. Network operations here are control-plane only. A caller
    may wake exactly the fresh accepted activation after checking its state,
    but status, parking and restart reconciliation cannot wake it.
    """

    def __init__(
        self, *, client: RunpodControlPlaneClient, endpoint: RegisteredOllamaEndpoint,
        journal: RegisteredOllamaJournal, clock: Callable[[], datetime],
    ) -> None:
        self.client = client
        self.endpoint = endpoint
        self.journal = journal
        self.clock = clock

    def activate(
        self, *, lease_id: str, owner_id: str, admission_sha256: str,
        active_env: Mapping[str, str], first_patch_before: datetime,
        ready_before: datetime, hard_deadline: datetime, idle_seconds: int,
    ) -> tuple[RegisteredActivation, bool]:
        try:
            expiry = datetime.fromisoformat(active_env.get("KESTREL_OLLAMA_BEARER_TOKEN_EXPIRES_AT", ""))
        except (TypeError, ValueError):
            raise OllamaLeaseAuthorizationError("Registered runtime bearer expiry is invalid") from None
        if (
            not _SECRET_REFERENCE.fullmatch(active_env.get("KESTREL_OLLAMA_BEARER_TOKEN", ""))
            or active_env.get(OLLAMA_REQUIRE_CACHED_MODEL_ENV) != "true"
            or expiry != hard_deadline
            or {_PARK_MARKER_ENV, _ACTIVE_MARKER_ENV}.intersection(active_env)
        ):
            raise OllamaLeaseAuthorizationError("Registered runtime requires a bounded cache-only secret reference")
        active_env = {
            **active_env,
            _ACTIVE_MARKER_ENV: canonical_sha256({
                "contract": "registered-ollama-arm-v1", "registration": self.endpoint.digest,
                "lease_id": lease_id, "admission_sha256": admission_sha256,
            }),
        }
        last = self.journal.last_parked(self.endpoint.endpoint_id)
        prior_sha = (
            last.intent.parked_environment_sha256 if last
            else environment_values_sha256(self.endpoint.baseline.env)
        )
        existing = self.journal.get(lease_id)
        park_env = self.endpoint.park_environment(lease_id, admission_sha256)
        intent = RegisteredActivationIntent(
            lease_id, owner_id, self.endpoint.endpoint_id, self.endpoint.baseline.network_volume_ids[0],
            admission_sha256, self.endpoint.digest,
            existing.intent.prior_environment_sha256 if existing else prior_sha,
            environment_values_sha256(active_env), environment_values_sha256(park_env),
            first_patch_before, ready_before, hard_deadline, idle_seconds,
        )
        if existing is None:
            self._verify_parked(prior_sha)
        row, fresh = self.journal.reserve(intent)
        if not fresh:
            return row, False
        self._verify_parked(prior_sha)
        if self.clock() >= min(row.expires_at, intent.first_patch_before):
            raise OllamaLeaseConflictError("Registered activation expired before its first PATCH")
        ack = self.client.update_endpoint(intent.endpoint_id, EndpointUpdateRequest(
            workers_min=0, workers_max=1, env=active_env,
        ))
        self.endpoint.verify(ack, max_workers=1, env_sha256=intent.active_environment_sha256)
        self._verify_active(intent)
        return self.journal.record_active(intent), True

    def park(self, row: RegisteredActivation) -> RegisteredActivation:
        """Park only the exact original effect; an unresolved write stays due."""
        intent = row.intent
        if intent.endpoint_id != self.endpoint.endpoint_id or intent.registration_sha256 != self.endpoint.digest:
            raise OllamaLeaseAuthorizationError("Registered endpoint changed from original authority")
        latest = self.journal.get(intent.lease_id)
        if latest is None or latest.intent != intent:
            raise OllamaLeaseConflictError("No original registered activation to park")
        if latest.state in (RegisteredActivationState.PARKED, RegisteredActivationState.FENCED_ZERO):
            return latest
        if latest.state is RegisteredActivationState.ARMING:
            # A parked predecessor does not prove an ambiguous arm was never
            # sent. Its reservation cannot be released or reused.
            if not self._observed_active(intent):
                return latest
            latest = self.journal.record_active(intent)
        if latest.state is RegisteredActivationState.ACTIVE:
            if not self._observed_active(intent):
                return latest
            latest, claimed = self.journal.claim_park(intent)
            if claimed:
                self._patch_parked(intent)
        if latest.state is RegisteredActivationState.PARKING:
            if self._observed_parked(intent):
                return self.journal.record_parked(intent)
            if self.clock() >= intent.hard_deadline and self._observed_active(intent):
                latest = self.journal.fence_uncertain_parking(intent)
        if latest.state is RegisteredActivationState.FENCED:
            if self._observed_active(intent):
                # Both the original and rescue effects only ever set zero.
                # This endpoint and cache can never be reused by a later lease.
                self._patch_parked(intent)
            if self._observed_parked(intent):
                return self.journal.record_fenced_zero(intent)
        return latest

    def reap_due(self) -> tuple[RegisteredActivation, ...]:
        """Run one independent control-plane pass; it cannot invoke inference."""
        return tuple(
            self.park(row) for row in self.journal.due()
            if row.intent.endpoint_id == self.endpoint.endpoint_id
        )

    def _patch_parked(self, intent: RegisteredActivationIntent) -> None:
        env = self.endpoint.park_environment(intent.lease_id, intent.admission_sha256)
        if environment_values_sha256(env) != intent.parked_environment_sha256:
            raise OllamaLeaseConflictError("Park effect no longer matches original intent")
        ack = self.client.update_endpoint(intent.endpoint_id, EndpointUpdateRequest(
            workers_min=0, workers_max=0, env=env,
        ))
        self.endpoint.verify(ack, max_workers=0, env_sha256=intent.parked_environment_sha256)

    def _verify_parked(self, env_sha256: str) -> None:
        self.endpoint.verify(
            self.client.get_endpoint(self.endpoint.endpoint_id), max_workers=0, env_sha256=env_sha256,
        )
        if not endpoint_worker_population_is_empty(self.client.list_endpoint_workers(self.endpoint.endpoint_id)):
            raise OllamaProvisioningEvidenceError("Registered endpoint has not reached zero workers")

    def _verify_active(self, intent: RegisteredActivationIntent) -> None:
        self.endpoint.verify(
            self.client.get_endpoint(intent.endpoint_id), max_workers=1,
            env_sha256=intent.active_environment_sha256,
        )

    def _observed_active(self, intent: RegisteredActivationIntent) -> bool:
        try:
            self._verify_active(intent)
        except OllamaProvisioningEvidenceError as error:
            if is_execution_control_error(error):
                raise
            return False
        return True

    def _observed_parked(self, intent: RegisteredActivationIntent) -> bool:
        try:
            self._verify_parked(intent.parked_environment_sha256)
        except OllamaProvisioningEvidenceError as error:
            if is_execution_control_error(error):
                raise
            return False
        return True
