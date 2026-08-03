"""SDK inference-lease adapter backed by durable Runpod Ollama capacity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseConstraintError,
    InferenceLeaseFailure,
    InferenceLeaseNotFoundError,
    InferenceLeaseOwnershipError,
    InferenceLeaseProviderUnavailableError,
    InferenceLeaseProvisioningError,
    InferenceLeaseQuote,
    InferenceLeaseRequest,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceProviderCapability,
    InferenceRoute,
)
from pydantic import SecretStr

from .models import GPUProfile, RunPodManagerError
from .ollama import _required_int
from .ollama_contracts import (
    OllamaLease,
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaLeaseState,
    OllamaPlacementPlan,
    OllamaResourceConstraints,
    OllamaResourceType,
    maximum_serverless_cold_starts,
    require_aware,
)
from .ollama_provider import RunpodOllamaCapacityProvider
from .ollama_runtime import parse_ollama_model_allowlist
from .ollama_service import OllamaLeaseService
from .providers import _resolve_env_vars

_PROVIDER_NAME = "runpod"
_RUNTIME = "ollama"
_CAPABILITIES = ("chat", "completions", "embeddings", "streaming", "tools")


@dataclass(frozen=True)
class _InferencePolicy:
    profile: GPUProfile
    regions: Mapping[str, str]
    allowed_models: frozenset[str]
    quote_ttl_seconds: int
    serverless_ready_seconds: int
    pod_ready_seconds: int

    def ready_seconds(self, mode: OllamaLeaseMode) -> int:
        return (
            self.serverless_ready_seconds
            if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            else self.pod_ready_seconds
        )


@dataclass(frozen=True)
class _Runtime:
    service: OllamaLeaseService
    policy: _InferencePolicy
    route_key: Callable[[OllamaResourceType], str]


class RunpodInferenceLeaseProvider:
    """Provider-neutral Ollama/OpenAI leases on Runpod v2 capacity.

    Construction is deliberately lazy so entry-point discovery never needs a
    Runpod credential or writes a lease database.  The first availability or
    quote call loads the package's canonical manager configuration.
    """

    def __init__(
        self,
        *,
        service: OllamaLeaseService | None = None,
        profile: GPUProfile | None = None,
        settings: Mapping[str, Any] | None = None,
        route_key: Callable[[OllamaResourceType], str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        supplied = (service is not None, profile is not None, settings is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("service, profile, and settings must be injected together")
        self._clock = clock
        self._runtime: _Runtime | None = None
        self._injected = (
            (service, profile, settings, route_key) if all(supplied) else None
        )

    @property
    def provider_name(self) -> str:
        return _PROVIDER_NAME

    def capabilities(self) -> Sequence[InferenceProviderCapability]:
        try:
            runtime = self._get_runtime()
        except InferenceLeaseProviderUnavailableError:
            return ()
        return (
            InferenceProviderCapability(
                runtime=_RUNTIME,
                privacy=(InferencePrivacy.AUTHENTICATED_ENDPOINT,),
                capabilities=_CAPABILITIES,
                regions=tuple(sorted(runtime.policy.regions)),
                max_concurrency=1,
            ),
        )

    def is_available(self) -> bool:
        try:
            self._get_runtime()
        except InferenceLeaseProviderUnavailableError:
            return False
        return True

    async def quote(self, request: InferenceLeaseRequest) -> InferenceLeaseQuote:
        runtime = self._get_runtime()
        now = self._now()
        region, provider_region = self._validate_request(
            request, policy=runtime.policy, now=now
        )
        candidates: list[tuple[OllamaPlacementPlan, OllamaLeaseRequest]] = []
        for mode in (
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
            OllamaLeaseMode.DEDICATED_POD,
        ):
            internal_request = self._internal_request(
                request,
                policy=runtime.policy,
                mode=mode,
                provider_region=provider_region,
                max_hourly_cost=request.max_hourly_cost_usd,
                max_total_cost=request.max_total_cost_usd,
            )
            try:
                _validate_runtime_preflight(runtime.service, internal_request)
                plan = await runtime.service.provider.plan(internal_request)
                runtime.service.validate_plan(internal_request, plan)
            except RunPodManagerError:
                continue
            ready_seconds = runtime.policy.ready_seconds(mode)
            if ready_seconds > self._remaining_readiness_seconds(request, now):
                continue
            candidates.append((plan, internal_request))
        if not candidates:
            raise InferenceLeaseProviderUnavailableError(
                "Runpod has no configured Ollama capacity satisfying the model, "
                "region, readiness, privacy, and cost constraints"
            )
        plan, _internal_request = min(
            candidates,
            key=lambda item: (
                item[0].estimated_cost,
                runtime.policy.ready_seconds(item[0].mode),
                item[0].placement.offered_cost_per_hr,
                item[0].mode.value,
            ),
        )
        hourly = _money(plan.placement.offered_cost_per_hr)
        total = _money(plan.estimated_cost)
        readiness_deadline = request.requested_at + timedelta(
            seconds=request.ready_deadline_seconds
        )
        expires_at = min(
            now + timedelta(seconds=runtime.policy.quote_ttl_seconds),
            readiness_deadline
            - timedelta(seconds=runtime.policy.ready_seconds(plan.mode)),
        )
        if expires_at <= now:
            raise InferenceLeaseConstraintError(
                "Runpod quote cannot outlive the requested readiness deadline"
            )
        metadata = _quote_metadata(plan)
        return InferenceLeaseQuote(
            quote_id=_quote_id(request),
            request_id=request.request_id,
            provider_name=self.provider_name,
            runtime=_RUNTIME,
            region=region,
            privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
            hourly_cost_usd=hourly,
            estimated_total_cost_usd=total,
            estimated_ready_seconds=runtime.policy.ready_seconds(plan.mode),
            expires_at=expires_at,
            metadata=metadata,
        )

    async def acquire(
        self,
        request: InferenceLeaseRequest,
        quote: InferenceLeaseQuote,
    ) -> InferenceLease:
        runtime = self._get_runtime()
        now = self._now()
        lease_id = _lease_id(request)
        existing = runtime.service.repository.get(lease_id)
        if existing is not None:
            self._authorize(existing, request.owner_id)
        self._validate_quote(
            request,
            quote,
            policy=runtime.policy,
            now=now,
            allow_elapsed_deadline=existing is not None,
        )
        if existing is None and quote.expires_at <= now:
            raise InferenceLeaseConstraintError("Runpod quote is expired")

        mode = _quote_mode(quote)
        provider_region = runtime.policy.regions[quote.region]
        internal_request = self._internal_request(
            request,
            policy=runtime.policy,
            mode=mode,
            provider_region=provider_region,
            max_hourly_cost=quote.hourly_cost_usd,
            max_total_cost=request.max_total_cost_usd,
        )
        internal_request = _bind_request_to_quoted_placement(internal_request, quote)
        plan: OllamaPlacementPlan | None = None
        readiness_elapsed = now >= (
            request.requested_at + timedelta(seconds=request.ready_deadline_seconds)
        )
        requires_provision = existing is None or (
            existing.state is OllamaLeaseState.REQUESTED and not readiness_elapsed
        )
        if requires_provision:
            self._validate_cold_start_window(
                request,
                policy=runtime.policy,
                mode=mode,
                now=now,
            )
            try:
                _validate_runtime_preflight(runtime.service, internal_request)
                plan = await runtime.service.provider.plan(internal_request)
                runtime.service.validate_plan(internal_request, plan)
            except RunPodManagerError as exc:
                raise InferenceLeaseProvisioningError(
                    "Runpod could not refresh the selected Ollama capacity quote"
                ) from exc
            self._validate_realized_plan(plan, quote, request=request)
            self._validate_cold_start_window(
                request,
                policy=runtime.policy,
                mode=mode,
                now=self._now(),
            )
        try:
            internal = await runtime.service.acquire(
                internal_request,
                wait_until_ready=False,
                plan=plan,
            )
        except OllamaLeaseAuthorizationError as exc:
            raise InferenceLeaseOwnershipError(
                "inference lease is owned by another agent"
            ) from exc
        except OllamaLeaseConflictError as exc:
            raise InferenceLeaseConstraintError(
                "request_id already represents different Runpod constraints"
            ) from exc
        except (RunPodManagerError, ValueError) as exc:
            persisted = runtime.service.repository.get(lease_id)
            if persisted is None:
                raise InferenceLeaseProvisioningError(
                    "Runpod could not persist the Ollama acquisition"
                ) from exc
            self._authorize(persisted, request.owner_id)
            internal = persisted
        return self._to_sdk_lease(internal, runtime=runtime)

    async def status(self, owner_id: str, lease_id: str) -> InferenceLease:
        runtime = self._get_runtime()
        internal = self._required(runtime.service, lease_id)
        self._authorize(internal, owner_id)
        try:
            internal = await runtime.service.reconcile_lease(
                lease_id,
                owner_id=owner_id,
                workload_id=internal.workload_id,
            )
        except OllamaLeaseAuthorizationError as exc:
            raise InferenceLeaseOwnershipError(
                "inference lease is owned by another agent"
            ) from exc
        return self._to_sdk_lease(internal, runtime=runtime)

    async def touch(self, owner_id: str, lease_id: str) -> InferenceLease:
        """Renew one exact ready lease and return its live authoritative route."""

        runtime = self._get_runtime()
        internal = self._required(runtime.service, lease_id)
        self._authorize(internal, owner_id)
        workload_id = internal.workload_id
        try:
            internal = await runtime.service.touch(
                lease_id,
                owner_id=owner_id,
                workload_id=workload_id,
            )
        except OllamaLeaseAuthorizationError as exc:
            raise InferenceLeaseOwnershipError(
                "inference lease is owned by another agent"
            ) from exc
        except RunPodManagerError as exc:
            raise InferenceLeaseProvisioningError(
                "Runpod could not renew the inference lease idle deadline"
            ) from exc
        if (
            internal.lease_id != lease_id
            or internal.owner_id != owner_id
            or internal.workload_id != workload_id
        ):
            raise InferenceLeaseProvisioningError(
                "Runpod returned an inference lease with inconsistent identity"
            )
        return self._to_sdk_lease(internal, runtime=runtime)

    async def release(self, owner_id: str, lease_id: str) -> InferenceLease:
        runtime = self._get_runtime()
        internal = self._required(runtime.service, lease_id)
        self._authorize(internal, owner_id)
        try:
            internal = await runtime.service.release(
                lease_id,
                owner_id=owner_id,
                workload_id=internal.workload_id,
            )
        except OllamaLeaseAuthorizationError as exc:
            raise InferenceLeaseOwnershipError(
                "inference lease is owned by another agent"
            ) from exc
        except RunPodManagerError:
            # Teardown failures are durable and retryable.  Returning RELEASING
            # lets core persist that route-less state and call release again.
            internal = self._required(runtime.service, lease_id)
        return self._to_sdk_lease(internal, runtime=runtime)

    def _get_runtime(self) -> _Runtime:
        if self._runtime is not None:
            return self._runtime
        try:
            if self._injected is not None:
                service, profile, settings, route_key = self._injected
                assert service is not None and profile is not None
                assert settings is not None
            else:
                from .manager import RunPodManager

                manager = RunPodManager()
                service = manager._get_ollama_lease_service()
                profile = manager._select_profile("ollama")
                raw_settings = manager.config.get("ollama_leases")
                if not isinstance(raw_settings, Mapping):
                    raise RunPodManagerError(
                        "Configure the ollama_leases section before inference"
                    )
                settings = raw_settings
                route_key = None
            policy = _policy(profile, settings)
            if route_key is None:
                provider = service.provider
                if not isinstance(provider, RunpodOllamaCapacityProvider):
                    raise RunPodManagerError(
                        "Inference leases require the Runpod Ollama capacity provider"
                    )
                route_key = provider.bearer_token_for
            runtime = _Runtime(
                service=service,
                policy=policy,
                route_key=route_key,
            )
        except (ImportError, OSError, RunPodManagerError, TypeError, ValueError) as exc:
            raise InferenceLeaseProviderUnavailableError(
                "Runpod inference provider configuration is unavailable"
            ) from exc
        self._runtime = runtime
        return runtime

    def _validate_request(
        self,
        request: InferenceLeaseRequest,
        *,
        policy: _InferencePolicy,
        now: datetime,
    ) -> tuple[str, str]:
        if request.runtime != _RUNTIME:
            raise InferenceLeaseConstraintError(
                "Runpod inference supports only the Ollama runtime"
            )
        if request.privacy is InferencePrivacy.PRIVATE_NETWORK:
            raise InferenceLeaseConstraintError(
                "Runpod Ollama exposes an authenticated endpoint, not a private network"
            )
        if request.expected_concurrency != 1:
            raise InferenceLeaseConstraintError(
                "Runpod Ollama currently supports one expected concurrent request"
            )
        unsupported = set(request.capabilities).difference(_CAPABILITIES)
        if unsupported:
            raise InferenceLeaseConstraintError(
                "Runpod Ollama does not satisfy all requested capabilities"
            )
        if request.model not in policy.allowed_models:
            raise InferenceLeaseConstraintError(
                "requested model is outside the Runpod Ollama operator allowlist"
            )
        if request.max_hourly_cost_usd <= 0 or request.max_total_cost_usd <= 0:
            raise InferenceLeaseConstraintError(
                "Runpod inference requires positive hourly and total cost limits"
            )
        if self._remaining_readiness_seconds(request, now) < 1:
            raise InferenceLeaseConstraintError(
                "requested Runpod readiness deadline has elapsed"
            )
        permitted = (
            sorted(set(request.allowed_regions).intersection(policy.regions))
            if request.allowed_regions
            else sorted(policy.regions)
        )
        if not permitted:
            raise InferenceLeaseConstraintError(
                "Runpod inference has no configured allowed region"
            )
        region = permitted[0]
        return region, policy.regions[region]

    def _validate_quote(
        self,
        request: InferenceLeaseRequest,
        quote: InferenceLeaseQuote,
        *,
        policy: _InferencePolicy,
        now: datetime,
        allow_elapsed_deadline: bool = False,
    ) -> None:
        validation_time = min(now, quote.expires_at - timedelta(microseconds=1))
        quote.validate_for(request, now=validation_time)
        if allow_elapsed_deadline:
            self._validate_request(
                request,
                policy=policy,
                now=min(now, request.requested_at),
            )
        else:
            self._validate_request(request, policy=policy, now=now)
        if quote.provider_name != self.provider_name:
            raise InferenceLeaseConstraintError("quote provider does not match Runpod")
        if quote.quote_id != _quote_id(request):
            raise InferenceLeaseConstraintError(
                "quote does not match the owner-scoped Runpod request"
            )
        if quote.region not in policy.regions:
            raise InferenceLeaseConstraintError(
                "quote region is no longer allowed by Runpod policy"
            )
        mode = _quote_mode(quote)
        if quote.estimated_ready_seconds != policy.ready_seconds(mode):
            raise InferenceLeaseConstraintError(
                "quote readiness estimate does not match Runpod policy"
            )
        if _quoted_cost(quote, "all_in_cost_ceiling_usd") > (
            request.max_total_cost_usd
        ):
            raise InferenceLeaseConstraintError(
                "Runpod quote all-in ceiling exceeds maximum total cost"
            )
        _validate_quote_cost_breakdown(quote)

    def _internal_request(
        self,
        request: InferenceLeaseRequest,
        *,
        policy: _InferencePolicy,
        mode: OllamaLeaseMode,
        provider_region: str,
        max_hourly_cost: Decimal,
        max_total_cost: Decimal,
    ) -> OllamaLeaseRequest:
        ready_seconds = policy.ready_seconds(mode)
        expected_session_seconds = request.expected_session_seconds
        if mode is OllamaLeaseMode.DEDICATED_POD:
            expected_session_seconds += ready_seconds
        hourly_limit = float(max_hourly_cost)
        if policy.profile.max_cost_per_hr is not None:
            hourly_limit = min(hourly_limit, policy.profile.max_cost_per_hr)
        return OllamaLeaseRequest(
            lease_id=_lease_id(request),
            owner_id=request.owner_id,
            workload_id=request.request_id,
            model=request.model,
            constraints=OllamaResourceConstraints(
                min_vram_gb=policy.profile.min_vram_gb,
                gpu_count=policy.profile.gpu_count,
                cloud=policy.profile.cloud,
                min_cuda_version=policy.profile.min_cuda_version,
                allowed_gpu_ids=policy.profile.allowed_gpu_ids,
                allowed_data_center_ids=(provider_region,),
                max_hourly_rate=hourly_limit,
            ),
            expected_session_seconds=expected_session_seconds,
            expected_active_seconds=request.expected_session_seconds,
            serverless_initialization_seconds=policy.serverless_ready_seconds,
            serverless_idle_tail_seconds=request.idle_ttl_seconds,
            idle_timeout_seconds=request.idle_ttl_seconds,
            readiness_timeout_seconds=request.ready_deadline_seconds,
            hard_deadline=(
                request.requested_at
                + timedelta(
                    seconds=(
                        request.ready_deadline_seconds
                        + request.expected_session_seconds
                    )
                )
            ),
            max_authorized_cost=float(max_total_cost),
            mode=mode,
            requested_at=request.requested_at,
        )

    @staticmethod
    def _remaining_readiness_seconds(
        request: InferenceLeaseRequest, now: datetime
    ) -> int:
        deadline = request.requested_at + timedelta(
            seconds=request.ready_deadline_seconds
        )
        return max(0, math.floor((deadline - now).total_seconds()))

    def _validate_cold_start_window(
        self,
        request: InferenceLeaseRequest,
        *,
        policy: _InferencePolicy,
        mode: OllamaLeaseMode,
        now: datetime,
    ) -> None:
        if policy.ready_seconds(mode) > self._remaining_readiness_seconds(request, now):
            raise InferenceLeaseConstraintError(
                "Runpod estimated cold start no longer fits the readiness deadline"
            )

    @staticmethod
    def _validate_realized_plan(
        plan: OllamaPlacementPlan,
        quote: InferenceLeaseQuote,
        *,
        request: InferenceLeaseRequest,
    ) -> None:
        if plan.mode is not _quote_mode(quote):
            raise InferenceLeaseConstraintError(
                "live Runpod plan changed the quoted execution mode"
            )
        for metadata_name, realized in (
            ("gpu_id", plan.placement.gpu_id),
            ("gpu_pool", plan.placement.gpu_pool),
            ("gpu_name", plan.placement.gpu_name),
        ):
            if quote.metadata.get(metadata_name) != realized:
                raise InferenceLeaseConstraintError(
                    "live Runpod placement differs from the accepted quote"
                )
        if _money(plan.placement.offered_cost_per_hr) > quote.hourly_cost_usd:
            raise InferenceLeaseConstraintError(
                "live Runpod hourly cost exceeds the accepted quote"
            )
        if _money(plan.estimated_cost) > quote.estimated_total_cost_usd:
            raise InferenceLeaseConstraintError(
                "live Runpod total cost exceeds the accepted quote"
            )
        if _money(plan.estimated_compute_cost) > _quoted_cost(
            quote, "estimated_compute_cost_usd"
        ) or _money(plan.maximum_compute_cost) > _quoted_cost(
            quote, "maximum_compute_cost_usd"
        ):
            raise InferenceLeaseConstraintError(
                "live Runpod compute exposure exceeds the accepted quote"
            )
        if _money(plan.estimated_non_compute_cost) != _quoted_cost(
            quote, "estimated_non_compute_cost_usd"
        ) or _money(plan.maximum_non_compute_cost) != _quoted_cost(
            quote, "maximum_non_compute_cost_usd"
        ):
            raise InferenceLeaseConstraintError(
                "live Runpod non-compute policy differs from the accepted quote"
            )
        if tuple(item.value for item in plan.non_compute_components) != (
            _quoted_cost_components(quote)
        ):
            raise InferenceLeaseConstraintError(
                "live Runpod non-compute components differ from the accepted quote"
            )
        realized_ceiling = _money(plan.cost_ceiling)
        if (
            realized_ceiling > _quoted_cost(quote, "all_in_cost_ceiling_usd")
            or realized_ceiling > request.max_total_cost_usd
        ):
            raise InferenceLeaseConstraintError(
                "live Runpod all-in ceiling exceeds the accepted authorization"
            )
        if (
            plan.maximum_serverless_cold_starts
            != _quoted_maximum_serverless_cold_starts(quote)
        ):
            raise InferenceLeaseConstraintError(
                "live Runpod cold-start exposure differs from the accepted quote"
            )

    def _to_sdk_lease(self, lease: OllamaLease, *, runtime: _Runtime) -> InferenceLease:
        now = self._now()
        state = _sdk_state(lease, now)
        updated_at = max(lease.created_at, min(lease.updated_at, now))
        expires_at = lease.hard_deadline
        if state is InferenceLeaseState.EXPIRED:
            expires_at = max(
                lease.created_at + timedelta(microseconds=1),
                min(lease.hard_deadline, now),
            )
            updated_at = max(updated_at, expires_at)
        route = None
        if state is InferenceLeaseState.READY:
            if lease.resource_type is None or not lease.public_route_url:
                raise InferenceLeaseProvisioningError(
                    "Runpod marked an Ollama lease ready without a usable route"
                )
            route = InferenceRoute(
                endpoint=SecretStr(f"{lease.public_route_url.rstrip('/')}/v1"),
                model=lease.model,
                api_key=SecretStr(runtime.route_key(lease.resource_type)),
                context_window=runtime.policy.profile.max_context_window,
                protocol="openai",
            )
        failure = None
        if state is InferenceLeaseState.FAILED:
            failure = InferenceLeaseFailure(
                code="runpod_provisioning_failed",
                message=lease.last_provider_error or "Runpod provisioning failed",
                retryable=False,
            )
        hourly, total = _lease_costs(lease)
        return InferenceLease(
            lease_id=lease.lease_id,
            quote_id=_quote_id_from_ids(lease.owner_id, lease.workload_id),
            request_id=lease.workload_id,
            owner_id=lease.owner_id,
            provider_name=self.provider_name,
            state=state,
            model=lease.model,
            runtime=_RUNTIME,
            privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
            created_at=lease.created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            region=_lease_region(lease),
            hourly_cost_usd=hourly,
            estimated_total_cost_usd=total,
            route=route,
            failure=failure,
            metadata=_lease_metadata(lease),
        )

    def _now(self) -> datetime:
        value = self._clock()
        require_aware(value, "inference provider clock")
        return value.astimezone(UTC)

    @staticmethod
    def _required(service: OllamaLeaseService, lease_id: str) -> OllamaLease:
        lease = service.repository.get(lease_id)
        if lease is None:
            raise InferenceLeaseNotFoundError(
                "inference lease was not found for this agent"
            )
        return lease

    @staticmethod
    def _authorize(lease: OllamaLease, owner_id: str) -> None:
        if lease.owner_id != owner_id:
            raise InferenceLeaseOwnershipError(
                "inference lease is owned by another agent"
            )


def _policy(profile: GPUProfile, settings: Mapping[str, Any]) -> _InferencePolicy:
    resolved_environment = _resolve_env_vars(profile.env)
    raw_models = resolved_environment.get("KESTREL_OLLAMA_ALLOWED_MODELS")
    if not raw_models:
        raise RunPodManagerError(
            "profiles.ollama.env.KESTREL_OLLAMA_ALLOWED_MODELS is required"
        )
    allowed_models = frozenset(parse_ollama_model_allowlist(raw_models))
    regions: dict[str, str] = {}
    for provider_region in profile.allowed_data_center_ids:
        normalized = provider_region.strip().lower()
        if not normalized or normalized in regions:
            raise RunPodManagerError(
                "profiles.ollama.allowed_data_center_ids must be unique"
            )
        regions[normalized] = provider_region
    if not regions:
        raise RunPodManagerError(
            "profiles.ollama.allowed_data_center_ids must constrain inference region"
        )
    return _InferencePolicy(
        profile=profile,
        regions=regions,
        allowed_models=allowed_models,
        quote_ttl_seconds=_required_int(settings, "quote_ttl_seconds"),
        serverless_ready_seconds=_required_int(
            settings, "serverless_estimated_ready_seconds"
        ),
        pod_ready_seconds=_required_int(settings, "pod_estimated_ready_seconds"),
    )


def _quote_mode(quote: InferenceLeaseQuote) -> OllamaLeaseMode:
    raw = quote.metadata.get("mode")
    try:
        mode = OllamaLeaseMode(raw)
    except (TypeError, ValueError) as exc:
        raise InferenceLeaseConstraintError(
            "Runpod quote has an invalid execution mode"
        ) from exc
    if mode is OllamaLeaseMode.AUTO:
        raise InferenceLeaseConstraintError(
            "Runpod quote must select a concrete execution mode"
        )
    return mode


def _quote_id(request: InferenceLeaseRequest) -> str:
    return _quote_id_from_ids(request.owner_id, request.request_id)


def _quote_id_from_ids(owner_id: str, request_id: str) -> str:
    digest = hashlib.sha256(
        f"{_PROVIDER_NAME}\0{owner_id}\0{request_id}".encode()
    ).hexdigest()[:32]
    return f"runpod-quote-{digest}"


def _lease_id(request: InferenceLeaseRequest) -> str:
    digest = hashlib.sha256(
        f"{_PROVIDER_NAME}\0{request.owner_id}\0{request.request_id}".encode()
    ).hexdigest()[:32]
    return f"runpod-lease-{digest}"


def _quote_metadata(plan: OllamaPlacementPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode.value,
        "gpu_id": plan.placement.gpu_id,
        "gpu_pool": plan.placement.gpu_pool,
        "gpu_name": plan.placement.gpu_name,
        "estimated_billable_seconds": plan.estimated_billable_seconds,
        "maximum_billable_seconds": plan.maximum_billable_seconds,
        "maximum_concurrent_workers": plan.maximum_concurrent_workers,
        "maximum_serverless_cold_starts": (plan.maximum_serverless_cold_starts),
        "catalog_observed_at": plan.placement.catalog_observed_at.isoformat(),
        "estimated_compute_cost_usd": str(_money(plan.estimated_compute_cost)),
        "maximum_compute_cost_usd": str(_money(plan.maximum_compute_cost)),
        "estimated_non_compute_cost_usd": str(_money(plan.estimated_non_compute_cost)),
        "maximum_non_compute_cost_usd": str(_money(plan.maximum_non_compute_cost)),
        "estimated_all_in_cost_usd": str(_money(plan.estimated_cost)),
        "all_in_cost_ceiling_usd": str(_money(plan.cost_ceiling)),
        "non_compute_components": tuple(
            item.value for item in plan.non_compute_components
        ),
        "cost_kind": "conservative_authorization_not_observed_billing",
    }


def _lease_metadata(lease: OllamaLease) -> dict[str, Any]:
    return {
        "mode": lease.mode.value if lease.mode else None,
        "resource_type": lease.resource_type.value if lease.resource_type else None,
        "gpu_id": lease.selected_gpu_id,
        "gpu_pool": lease.selected_gpu_pool,
        "gpu_name": lease.selected_gpu_name,
        "estimated_billable_seconds": lease.estimated_billable_seconds,
        "maximum_billable_seconds": lease.maximum_billable_seconds,
        "maximum_concurrent_workers": lease.maximum_concurrent_workers,
        "maximum_serverless_cold_starts": (
            maximum_serverless_cold_starts(
                expected_session_seconds=lease.expected_session_seconds,
                idle_tail_seconds=lease.serverless_idle_tail_seconds,
            )
            if lease.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
            else 0
        ),
        "cold_start_seconds": lease.cold_start_seconds,
        "estimated_compute_cost_usd": _optional_money_text(
            lease.estimated_compute_cost
        ),
        "maximum_compute_cost_usd": _optional_money_text(lease.maximum_compute_cost),
        "estimated_non_compute_cost_usd": _optional_money_text(
            lease.estimated_non_compute_cost
        ),
        "maximum_non_compute_cost_usd": _optional_money_text(
            lease.maximum_non_compute_cost
        ),
        "estimated_all_in_cost_usd": _optional_money_text(lease.estimated_cost),
        "all_in_cost_ceiling_usd": _optional_money_text(lease.cost_ceiling),
        "non_compute_components": tuple(
            item.value for item in lease.cost_policy_components
        ),
        "cost_kind": "conservative_authorization_not_observed_billing",
    }


def _quoted_maximum_serverless_cold_starts(
    quote: InferenceLeaseQuote,
) -> int:
    value = quote.metadata.get("maximum_serverless_cold_starts")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InferenceLeaseConstraintError(
            "Runpod quote has an invalid Serverless cold-start bound"
        )
    return value


def _lease_costs(lease: OllamaLease) -> tuple[Decimal, Decimal]:
    constraints: object
    try:
        constraints = json.loads(lease.constraints_json)
    except json.JSONDecodeError as exc:
        raise InferenceLeaseProvisioningError(
            "Runpod inference lease has corrupt durable constraints"
        ) from exc
    if not isinstance(constraints, Mapping):
        raise InferenceLeaseProvisioningError(
            "Runpod inference lease has corrupt durable constraints"
        )
    hourly = lease.offered_rate_per_hr
    if hourly is None:
        raw_hourly = constraints.get("max_hourly_rate")
        if not isinstance(raw_hourly, (int, float)) or isinstance(raw_hourly, bool):
            raise InferenceLeaseProvisioningError(
                "Runpod inference lease has no durable hourly cost bound"
            )
        hourly = float(raw_hourly)
    total = lease.estimated_cost
    if total is None:
        total = lease.max_authorized_cost
    return _money(hourly), _money(total)


def _lease_region(lease: OllamaLease) -> str:
    try:
        constraints: object = json.loads(lease.constraints_json)
    except json.JSONDecodeError as exc:
        raise InferenceLeaseProvisioningError(
            "Runpod inference lease has corrupt durable region policy"
        ) from exc
    if not isinstance(constraints, Mapping):
        raise InferenceLeaseProvisioningError(
            "Runpod inference lease has corrupt durable region policy"
        )
    regions = constraints.get("allowed_data_center_ids")
    if (
        not isinstance(regions, Sequence)
        or isinstance(regions, (str, bytes))
        or len(regions) != 1
        or not isinstance(regions[0], str)
    ):
        raise InferenceLeaseProvisioningError(
            "Runpod inference lease has no exact durable region"
        )
    return regions[0].lower()


def _sdk_state(lease: OllamaLease, now: datetime) -> InferenceLeaseState:
    del now
    if lease.state is OllamaLeaseState.READY:
        return InferenceLeaseState.READY
    if lease.state is OllamaLeaseState.FAILED:
        return InferenceLeaseState.FAILED
    if lease.state is OllamaLeaseState.RELEASING:
        return InferenceLeaseState.RELEASING
    if lease.state is OllamaLeaseState.TERMINATED:
        if lease.termination_reason in {
            "deadline_or_cost_cap",
            "expired",
            "readiness_timeout",
        }:
            return InferenceLeaseState.EXPIRED
        return InferenceLeaseState.RELEASED
    return InferenceLeaseState.PENDING


def _money(value: float) -> Decimal:
    if not math.isfinite(value) or value < 0:
        raise InferenceLeaseProvisioningError(
            "Runpod returned an invalid inference cost"
        )
    return Decimal(str(value))


def _optional_money_text(value: float | None) -> str | None:
    return str(_money(value)) if value is not None else None


def _quoted_cost(quote: InferenceLeaseQuote, name: str) -> Decimal:
    value = quote.metadata.get(name)
    if not isinstance(value, str):
        raise InferenceLeaseConstraintError(
            f"Runpod quote has no valid {name.replace('_', ' ')}"
        )
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise InferenceLeaseConstraintError(
            f"Runpod quote has no valid {name.replace('_', ' ')}"
        ) from exc
    if not amount.is_finite() or amount < 0:
        raise InferenceLeaseConstraintError(
            f"Runpod quote has no valid {name.replace('_', ' ')}"
        )
    return amount


def _quoted_cost_components(quote: InferenceLeaseQuote) -> tuple[str, ...]:
    value = quote.metadata.get("non_compute_components")
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
        or tuple(sorted(value)) != value
    ):
        raise InferenceLeaseConstraintError(
            "Runpod quote has invalid non-compute cost components"
        )
    return value


def _validate_quote_cost_breakdown(quote: InferenceLeaseQuote) -> None:
    estimated_compute = _quoted_cost(quote, "estimated_compute_cost_usd")
    maximum_compute = _quoted_cost(quote, "maximum_compute_cost_usd")
    estimated_overhead = _quoted_cost(quote, "estimated_non_compute_cost_usd")
    maximum_overhead = _quoted_cost(quote, "maximum_non_compute_cost_usd")
    estimated_total = _quoted_cost(quote, "estimated_all_in_cost_usd")
    ceiling = _quoted_cost(quote, "all_in_cost_ceiling_usd")
    _quoted_cost_components(quote)
    if (
        maximum_compute < estimated_compute
        or maximum_overhead < estimated_overhead
        or estimated_total != estimated_compute + estimated_overhead
        or ceiling != maximum_compute + maximum_overhead
        or ceiling < estimated_total
        or estimated_total != quote.estimated_total_cost_usd
    ):
        raise InferenceLeaseConstraintError(
            "Runpod quote has an inconsistent all-in cost breakdown"
        )


def _bind_request_to_quoted_placement(
    request: OllamaLeaseRequest, quote: InferenceLeaseQuote
) -> OllamaLeaseRequest:
    gpu_id = quote.metadata.get("gpu_id")
    gpu_pool = quote.metadata.get("gpu_pool")
    gpu_name = quote.metadata.get("gpu_name")
    if not isinstance(gpu_id, str) or not gpu_id or not isinstance(gpu_name, str):
        raise InferenceLeaseConstraintError(
            "Runpod quote has an invalid exact GPU placement"
        )
    if gpu_pool is not None and (not isinstance(gpu_pool, str) or not gpu_pool):
        raise InferenceLeaseConstraintError(
            "Runpod quote has an invalid exact GPU pool"
        )
    return replace(
        request,
        constraints=replace(
            request.constraints,
            allowed_gpu_ids=(gpu_id,),
            allowed_gpu_pools=((gpu_pool,) if gpu_pool is not None else ()),
        ),
    )


def _validate_runtime_preflight(
    service: OllamaLeaseService, request: OllamaLeaseRequest
) -> None:
    provider = service.provider
    if isinstance(provider, RunpodOllamaCapacityProvider):
        provider.validate_runtime_request(request)


__all__ = ["RunpodInferenceLeaseProvider"]
