"""Provider-neutral inference adapter for the explicitly registered endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
from kestrel_sdk.llm import (
    InferenceLease, InferenceLeaseConstraintError, InferenceLeaseNotFoundError,
    InferenceLeaseOwnershipError, InferenceLeaseProviderUnavailableError, InferenceLeaseQuote,
    InferenceLeaseRequest, InferenceLeaseState, InferencePrivacy, InferenceProviderCapability,
    InferenceRoute,
)
from kestrel_sovereign.execution_custody import is_execution_control_error, owned_native_http_client
from pydantic import SecretStr

from .clients import RunpodControlPlaneClient
from .effect_custody import await_owned_effect
from .models import SERVERLESS_DATA_PLANE_BASE_URL, CloudType, ComputeProduct, RunPodManagerError, VolumeType
from .ollama_contracts import OllamaLeaseMode, OllamaProvisioningEvidenceError
from .ollama_repository import SQLiteOllamaLeaseRepository, lease_database_path
from .ollama_runtime import (
    build_ollama_runtime_environment, cache_only_ollama_runtime_version, parse_ollama_model_allowlist,
)
from .registered_ollama import RegisteredOllamaLifecycle
from .registered_ollama_admission import REGISTERED_CAPABILITIES, REGISTERED_PROVIDER_NAME, RegisteredQuotePolicy
from .registered_ollama_journal import RegisteredActivation, RegisteredActivationState, RegisteredInferenceTerms, RegisteredOllamaJournal
from .registered_ollama_readiness import RegisteredWorkerReadback, verify_registered_runtime
from .registered_ollama_reconciler import load_registered_endpoint
from .signed_invocations import canonical_sha256


class RegisteredInferenceLeaseProvider:
    """No status, release, or expired replay ever makes a data-plane request."""

    provider_name = REGISTERED_PROVIDER_NAME

    def __init__(self) -> None:
        self._lifecycle: RegisteredOllamaLifecycle | None = None

    def is_available(self) -> bool:
        # Absence is normal discovery. An opt-in with invalid settings fails
        # explicitly at its first quote instead of falling back to a Pod.
        return bool(os.getenv("RUNPOD_REGISTERED_OLLAMA_CONFIG"))

    def capabilities(self) -> Sequence[InferenceProviderCapability]:
        if not self.is_available():
            return ()
        _lifecycle, policy = self._runtime()
        return (InferenceProviderCapability(
            runtime="ollama", privacy=(InferencePrivacy.AUTHENTICATED_ENDPOINT,),
            capabilities=REGISTERED_CAPABILITIES, regions=(policy.region.lower(),),
            max_concurrency=1,
        ),)

    def _runtime(self) -> tuple[RegisteredOllamaLifecycle, RegisteredQuotePolicy]:
        if self._lifecycle is not None:
            return self._lifecycle, self._policy
        raw = os.getenv("RUNPOD_REGISTERED_OLLAMA_CONFIG", "")
        if not raw or not Path(raw).is_absolute():
            raise InferenceLeaseProviderUnavailableError("Registered Ollama is not explicitly configured")
        key, control = os.getenv("RUNPOD_REGISTERED_OLLAMA_KEY", ""), os.getenv("RUNPOD_API_KEY", "")
        if len(key) < 32 or not control or key == control:
            raise InferenceLeaseProviderUnavailableError("Registered Ollama credentials must have separate scopes")
        config = tomllib.loads(Path(raw).read_text(encoding="utf-8"))
        if set(config) != {"endpoint", "inference"}:
            raise InferenceLeaseProviderUnavailableError("Invalid registered inference config shape")
        endpoint = load_registered_endpoint(config["endpoint"])
        if re.fullmatch(r"[a-zA-Z0-9_-]+", endpoint.endpoint_id) is None:
            raise InferenceLeaseProviderUnavailableError("Invalid endpoint ID for an exact route")
        settings = config["inference"]
        if set(settings) != {
            "model_pin", "secret_name", "gpu_id", "model_storage_path",
            "estimated_ready_seconds", "quote_ttl_seconds", "max_readiness_seconds",
            "max_session_seconds", "non_compute_cost_bound_usd", "cache_size_gb",
            "invocation_key_sha256",
        }:
            raise InferenceLeaseProviderUnavailableError("Invalid registered inference policy shape")
        models = parse_ollama_model_allowlist(settings["model_pin"])
        if (
            len(models) != 1 or not re.fullmatch(r"[a-zA-Z0-9_]+", settings["secret_name"])
            or type(settings["cache_size_gb"]) is not int or settings["cache_size_gb"] < 1
        ):
            raise InferenceLeaseProviderUnavailableError("Registered route requires an exact model and secret")
        pin = settings["invocation_key_sha256"]
        if (
            not isinstance(pin, str) or re.fullmatch(r"[a-f0-9]{64}", pin) is None
            or not hmac.compare_digest(hashlib.sha256(key.encode()).hexdigest(), pin)
        ):
            raise InferenceLeaseProviderUnavailableError("Registered invocation key differs from the scoped enrollment")
        repo = SQLiteOllamaLeaseRepository(lease_database_path({}))
        policy = RegisteredQuotePolicy(
            endpoint.digest, repo.journal_identity(), endpoint.baseline.data_center_ids[0],
            next(iter(models)), next(iter(models.values())),
            settings["estimated_ready_seconds"], settings["quote_ttl_seconds"],
            endpoint.baseline.idle_timeout_seconds, settings["max_readiness_seconds"],
            settings["max_session_seconds"], Decimal(settings["non_compute_cost_bound_usd"]),
        )
        clock = lambda: datetime.now(UTC)
        lifecycle = RegisteredOllamaLifecycle(
            client=RunpodControlPlaneClient(api_key=control, read_timeout=10),
            endpoint=endpoint, journal=RegisteredOllamaJournal(repo.path, clock=clock), clock=clock,
        )
        self._gpu_id = settings["gpu_id"]
        self._model_pin = settings["model_pin"]
        self._model_digest = next(iter(models.values()))
        self._model_path = settings["model_storage_path"]
        self._secret_ref = "{{ RUNPOD_SECRET_" + settings["secret_name"] + " }}"
        self._key = SecretStr(key)
        self._cache_size_gb = settings["cache_size_gb"]
        self._health_base_url = SERVERLESS_DATA_PLANE_BASE_URL
        self._base_url = "https://" + endpoint.endpoint_id + ".api.runpod.ai"
        self._policy = policy
        self._lifecycle = lifecycle
        return lifecycle, policy

    def _rate(self, lifecycle: RegisteredOllamaLifecycle, policy: RegisteredQuotePolicy) -> Decimal:
        baseline = lifecycle.endpoint.baseline
        matching = [
            offer for offer in lifecycle.client.list_gpus(
                products=(ComputeProduct.SERVERLESS,), min_cuda_version=baseline.min_cuda_version,
            )
            if offer.pool == baseline.gpu_pools[0]
        ]
        if len(matching) != 1 or matching[0].id != self._gpu_id or not any(
            region.get("id") == policy.region and region.get("availability") in ("LOW", "MEDIUM", "HIGH")
            for region in matching[0].data_centers
        ):
            raise InferenceLeaseProviderUnavailableError("Registered GPU has no exact local capacity offer")
        return Decimal(str(matching[0].price_for(CloudType.SECURE, ComputeProduct.SERVERLESS)))

    def _verify_cache(self, lifecycle: RegisteredOllamaLifecycle, policy: RegisteredQuotePolicy) -> None:
        cache_id = lifecycle.endpoint.baseline.network_volume_ids[0]
        matches = [volume for volume in lifecycle.client.list_network_volumes() if volume.id == cache_id]
        if (
            len(matches) != 1 or matches[0].data_center != policy.region
            or matches[0].size_gb != self._cache_size_gb or matches[0].volume_type is not VolumeType.STANDARD
        ):
            raise InferenceLeaseProviderUnavailableError("Registered cache has drifted or is not colocated")

    async def _verify_key(self, lifecycle: RegisteredOllamaLifecycle) -> None:
        # The v2 health envelope is the service control view, not an LB
        # workload (the host below is distinct from the worker ingress).
        endpoint_id = lifecycle.endpoint.endpoint_id
        url = f"{self._health_base_url}/{endpoint_id}/health"
        async with owned_native_http_client(httpx.AsyncClient(
            follow_redirects=False, trust_env=False, timeout=5,
        )) as http:
            denied = await http.get(url)
            bearer = {"Authorization": "Bearer " + self._key.get_secret_value()}
            allowed = await http.get(url, headers=bearer)
            if denied.status_code != 401 or allowed.status_code != 200:
                raise InferenceLeaseProviderUnavailableError("Registered endpoint scope was not proved")
            management = lifecycle.client.transport.base_url
            for path in ("/account", f"/serverless/{endpoint_id}"):
                result = await http.get(management + path, headers=bearer)
                if result.status_code != 403:
                    raise InferenceLeaseProviderUnavailableError("Registered invocation key is not excluded from management")

    async def quote(self, request: InferenceLeaseRequest) -> InferenceLeaseQuote:
        lifecycle, policy = self._runtime()
        rate = await await_owned_effect(asyncio.to_thread(self._rate, lifecycle, policy))
        return policy.quote(request, rate=rate, now=lifecycle.clock())

    @staticmethod
    def _lease_id(request: InferenceLeaseRequest) -> str:
        digest = canonical_sha256({
            "owner": request.owner_id, "request": request.request_id, "provider": REGISTERED_PROVIDER_NAME,
        })
        return "registered-runpod-lease:" + digest.removeprefix("sha256:")

    async def acquire(
        self, request: InferenceLeaseRequest, quote: InferenceLeaseQuote,
    ) -> InferenceLease:
        lifecycle, policy = self._runtime()
        lease_id = self._lease_id(request)
        ready_before, hard_deadline = policy.deadlines(request)
        existing = lifecycle.journal.get(lease_id)
        if existing is not None:
            self._validate_replay(policy, request, quote, existing, now=lifecycle.clock())
            return await self.status(request.owner_id, lease_id)
        rate = await await_owned_effect(asyncio.to_thread(self._rate, lifecycle, policy))
        await await_owned_effect(asyncio.to_thread(self._verify_cache, lifecycle, policy))
        await self._verify_key(lifecycle)
        admission = policy.accept(request, quote, refreshed_rate=rate, now=lifecycle.clock())
        active_env = build_ollama_runtime_environment(
            {"KESTREL_OLLAMA_ALLOWED_MODELS": self._model_pin}, requested_model=request.model,
            bearer_token=self._secret_ref, bearer_token_expires_at=hard_deadline,
            mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER, model_storage_path=self._model_path,
            provision_requested_at=request.requested_at, telemetry_binding_sha256=admission,
            require_cached_model=True,
        )
        terms = RegisteredInferenceTerms(
            request.request_id, quote.quote_id, request.model, policy.model_digest, quote.region,
            format(quote.hourly_cost_usd, "f"), format(quote.estimated_total_cost_usd, "f"),
        )
        row, fresh = await await_owned_effect(asyncio.to_thread(
            lifecycle.activate, lease_id=lease_id, owner_id=request.owner_id, admission_sha256=admission,
            active_env=active_env, first_patch_before=quote.expires_at, ready_before=ready_before,
            hard_deadline=hard_deadline, idle_seconds=request.idle_ttl_seconds, terms=terms,
        ))
        if fresh:
            await self._await_startup(lifecycle, row)
        return await self.status(request.owner_id, lease_id)

    async def _await_startup(
        self, lifecycle: RegisteredOllamaLifecycle, row: RegisteredActivation,
    ) -> RegisteredActivation:
        # Only the fresh original admission may send a cold data-plane probe.
        # A restarted caller observes and parks; it never enters this method.
        intent = row.intent
        async with owned_native_http_client(httpx.AsyncClient(
            base_url=self._base_url, headers={"Authorization": "Bearer " + self._key.get_secret_value()},
            follow_redirects=False, trust_env=False,
        )) as http:
            while (seconds := (intent.ready_before - lifecycle.clock()).total_seconds()) > 0:
                current, _proof, revoked = lifecycle.journal.route_view(intent)
                if (
                    revoked or current.state is not RegisteredActivationState.ACTIVE
                    or lifecycle.clock() >= current.expires_at
                ):
                    break
                try:
                    async with asyncio.timeout(seconds):
                        probe = await http.get("/kestrel/telemetry", timeout=seconds)
                        if probe.status_code == 200:
                            if intent.terms is None:
                                raise InferenceLeaseConstraintError("Registered startup lacks its original approved terms")
                            # A waking probe's body is not evidence. Bracket
                            # a new authenticated snapshot with worker reads.
                            worker = await await_owned_effect(asyncio.to_thread(self._worker, lifecycle, row))
                            evidence = await http.get("/kestrel/telemetry", timeout=seconds)
                            if evidence.status_code != 200:
                                raise OllamaProvisioningEvidenceError("Registered runtime is not yet observable")
                            tags = await http.get("/api/tags", timeout=seconds)
                            tags.raise_for_status()
                            confirmed = await await_owned_effect(asyncio.to_thread(self._worker, lifecycle, row))
                            snapshot = verify_registered_runtime(
                                evidence.json(), tags.json(), telemetry_binding_sha256=intent.admission_sha256,
                                expected_model=intent.terms.model, model_digest=self._model_digest,
                                expected_runtime_version=cache_only_ollama_runtime_version(lifecycle.endpoint.baseline.image),
                                first_worker=worker, last_worker=confirmed,
                            )
                            if snapshot.model_load_ms is None:
                                raise InferenceLeaseConstraintError("Registered startup lacks measured load evidence")
                            return lifecycle.journal.record_ready(
                                intent, worker_id=worker.worker_id, cache_hit=True,
                                startup_ms=(lifecycle.clock() - row.created_at) // timedelta(milliseconds=1),
                                model_load_ms=snapshot.model_load_ms,
                            )
                        if probe.status_code not in (409, 425, 429, 502, 503, 504):
                            raise InferenceLeaseConstraintError("Registered endpoint rejected the scoped readiness probe")
                except (TimeoutError, httpx.TransportError, OllamaProvisioningEvidenceError) as error:
                    if is_execution_control_error(error):
                        raise
                await asyncio.sleep(min(2, max(0, (intent.ready_before - lifecycle.clock()).total_seconds())))
        return await await_owned_effect(asyncio.to_thread(lifecycle.park, row))

    def _worker(self, lifecycle: RegisteredOllamaLifecycle, row: RegisteredActivation) -> RegisteredWorkerReadback:
        return RegisteredWorkerReadback.verified(
            lifecycle.client.list_endpoint_workers(row.intent.endpoint_id), data_center=self._policy.region,
            gpu_id=self._gpu_id, image=lifecycle.endpoint.baseline.image,
            activated_not_before=row.created_at.replace(microsecond=0),
        )

    async def status(self, owner_id: str, lease_id: str) -> InferenceLease:
        lifecycle, _policy = self._runtime()
        row = self._owned(lifecycle, owner_id, lease_id, historical_parked=True)
        if row.intent.registration_sha256 != lifecycle.endpoint.digest:
            # This original lease already has durable, observed zero evidence.
            # The current enrollment owns no effects for it, including PARK.
            return self._sdk(lifecycle, row)
        row, proof, revoked = lifecycle.journal.route_view(row.intent)
        if revoked or lifecycle.clock() >= row.expires_at or row.state in (
            RegisteredActivationState.PARKING, RegisteredActivationState.FENCED,
        ):
            row = await await_owned_effect(asyncio.to_thread(lifecycle.park, row))
        verified_worker_id = None
        if row.state is RegisteredActivationState.ACTIVE:
            if proof is not None and not await self._owns_ready_worker(lifecycle, row, proof["worker_id"]):
                return self._sdk(lifecycle, lifecycle.journal.revoke(row.intent), releasing=True)
            if proof is not None:
                verified_worker_id = proof["worker_id"]
        return self._sdk(lifecycle, row, verified_worker_id=verified_worker_id)

    async def touch(self, owner_id: str, lease_id: str) -> InferenceLease:
        lifecycle, _policy = self._runtime()
        row = self._owned(lifecycle, owner_id, lease_id)
        proof = lifecycle.journal.readiness(lease_id)
        if proof is None:
            raise InferenceLeaseConstraintError("Registered workload has no original readiness evidence")
        if not await self._owns_ready_worker(lifecycle, row, proof["worker_id"]):
            lifecycle.journal.revoke(row.intent)
            raise InferenceLeaseConstraintError("Registered workload changed before inference")
        row = lifecycle.journal.touch(row.intent)
        return self._sdk(lifecycle, row, verified_worker_id=proof["worker_id"])

    async def _owns_ready_worker(
        self, lifecycle: RegisteredOllamaLifecycle, row: RegisteredActivation, worker_id: object,
    ) -> bool:
        def observe() -> bool:
            try:
                if not lifecycle._observed_active(row.intent):
                    return False
                return self._worker(lifecycle, row).worker_id == worker_id
            except RunPodManagerError as error:
                if is_execution_control_error(error):
                    raise
                return False
        return await await_owned_effect(asyncio.to_thread(observe))

    async def release(self, owner_id: str, lease_id: str) -> InferenceLease:
        lifecycle, _policy = self._runtime()
        row = self._owned(lifecycle, owner_id, lease_id)
        row = lifecycle.journal.revoke(row.intent)
        return self._sdk(
            lifecycle, await await_owned_effect(asyncio.to_thread(lifecycle.park, row)),
            releasing=True,
        )

    async def reconcile_expired_acquisition(
        self, request: InferenceLeaseRequest, quote: InferenceLeaseQuote,
    ) -> InferenceLease:
        lifecycle, policy = self._runtime()
        now = lifecycle.clock()
        if now < quote.expires_at:
            raise InferenceLeaseConstraintError("Registered acquisition is not expired")
        row = self._owned(lifecycle, request.owner_id, self._lease_id(request))
        self._validate_replay(policy, request, quote, row, now=now)
        return await self.status(request.owner_id, row.intent.lease_id)

    @staticmethod
    def _validate_replay(
        policy: RegisteredQuotePolicy, request: InferenceLeaseRequest, quote: InferenceLeaseQuote,
        row: RegisteredActivation, *, now: datetime,
    ) -> None:
        # Recover an original signed offer without charging or refreshing
        # availability. Occupied stock cannot reject a lease already admitted.
        admission = policy.accept(
            request, quote, refreshed_rate=quote.hourly_cost_usd,
            now=min(now, quote.expires_at - timedelta(microseconds=1)),
        )
        if admission != row.intent.admission_sha256 or row.intent.owner_id != request.owner_id:
            raise InferenceLeaseConstraintError("Offer does not match its original registered activation")

    @staticmethod
    def _owned(
        lifecycle: RegisteredOllamaLifecycle, owner_id: str, lease_id: str, *,
        historical_parked: bool = False,
    ) -> RegisteredActivation:
        row = lifecycle.journal.get(lease_id)
        if row is None or row.intent.terms is None:
            raise InferenceLeaseNotFoundError("No original registered lease was found")
        if row.intent.owner_id != owner_id:
            raise InferenceLeaseOwnershipError("Registered lease is owned by another agent")
        if row.intent.registration_sha256 != lifecycle.endpoint.digest:
            if not (
                historical_parked and row.state is RegisteredActivationState.PARKED
                and row.intent.endpoint_id == lifecycle.endpoint.endpoint_id
                and lifecycle.endpoint.baseline.network_volume_ids == (row.intent.cache_id,)
                and lifecycle.endpoint.matches_parked_registration(row.intent.registration_sha256)
            ):
                raise InferenceLeaseConstraintError("Registered lease's original endpoint changed")
        return row

    def _sdk(
        self, lifecycle: RegisteredOllamaLifecycle, row: RegisteredActivation, *,
        releasing: bool = False, verified_worker_id: object = None,
    ) -> InferenceLease:
        row, proof, revoked = lifecycle.journal.route_view(row.intent)
        terms = row.intent.terms
        if terms is None:
            raise InferenceLeaseConstraintError("Registered lease is missing its approved terms")
        if (
            row.state not in (RegisteredActivationState.PARKED, RegisteredActivationState.FENCED_ZERO)
            and proof is not None and terms.model_digest != self._model_digest
        ):
            row = lifecycle.journal.revoke(row.intent)
            releasing = True
        route = None
        now = lifecycle.clock()
        if row.state in (RegisteredActivationState.PARKED, RegisteredActivationState.FENCED_ZERO):
            state = InferenceLeaseState.RELEASED
        elif (
            releasing or revoked or row.state in (RegisteredActivationState.PARKING, RegisteredActivationState.FENCED)
            or now >= row.expires_at
        ):
            state = InferenceLeaseState.RELEASING
        elif (
            row.state is RegisteredActivationState.ACTIVE and proof is not None
            and isinstance(verified_worker_id, str) and proof["worker_id"] == verified_worker_id
        ):
            state = InferenceLeaseState.READY
            route = InferenceRoute(endpoint=SecretStr(self._base_url + "/v1"), model=terms.model, api_key=self._key)
        else:
            state = InferenceLeaseState.PENDING
        return InferenceLease(
            lease_id=row.intent.lease_id, quote_id=terms.quote_id, request_id=terms.request_id,
            owner_id=row.intent.owner_id, provider_name=self.provider_name, state=state, model=terms.model,
            runtime="ollama", privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT, created_at=row.created_at,
            updated_at=row.updated_at,
            expires_at=row.expires_at, region=terms.region,
            hourly_cost_usd=Decimal(terms.hourly_usd), estimated_total_cost_usd=Decimal(terms.estimated_total_usd),
            route=route, metadata=proof or {},
        )
