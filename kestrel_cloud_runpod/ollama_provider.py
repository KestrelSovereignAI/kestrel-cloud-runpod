"""Runpod v2 capacity, readiness, and teardown adapter for Ollama leases."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from .billing import exact_decimal_integer_scale
from .clients import (
    RunpodControlPlaneClient,
    is_exact_runpod_not_found,
    is_retryable_runpod_api_error,
)
from .models import (
    Availability,
    ComputeProduct,
    EndpointCreateRequest,
    EndpointResource,
    FlashBoot,
    GPUProfile,
    PlacementDecision,
    PodCreateRequest,
    PodResource,
    ResourceFieldState,
    RunPodAPIError,
    RunPodManagerError,
    assess_endpoint_mounts,
    assess_pod_mounts,
    environment_values_sha256,
    normalize_cuda_floor,
)
from .ollama_contracts import (
    OllamaCleanupBasis,
    OllamaCleanupIntent,
    OllamaCleanupReceipt,
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaLeaseTeardownError,
    OllamaLifecycleBillingObservation,
    OllamaNonComputeCostComponent,
    OllamaNonComputeCostPolicy,
    OllamaPlacementPlan,
    OllamaProviderRatePrecision,
    OllamaProvisioningEvidenceError,
    OllamaProvisioningIntent,
    OllamaProvisioningMismatchError,
    OllamaProvisioningMismatchEvidence,
    OllamaProvisioningMismatchReason,
    OllamaProvisioningReceipt,
    OllamaReadinessObservation,
    OllamaResourceType,
    OllamaRuntimeTelemetryObservation,
    OllamaRuntimeTelemetrySnapshot,
    PreparedOllamaProvisioning,
    ProvisionedOllamaResource,
    iso_datetime,
    ollama_runtime_telemetry_binding_sha256,
    provision_attempt_id,
    resource_name,
    runtime_telemetry_binding_for_intent,
    select_ollama_plan,
)
from .ollama_runtime import (
    build_ollama_runtime_environment,
    require_immutable_ollama_image,
)
from .placement import select_gpu
from .pod_billing import read_exact_pod_billing
from .pod_capacity_contracts import PodRealizedPlacement
from .providers import _resolve_env_vars
from .serverless_capacity_contracts import serverless_endpoint_hour_cost_totals
from .serverless_capacity_provider import read_exact_serverless_billing

RUNPOD_SERVERLESS_MAX_IDLE_TIMEOUT_SECONDS = 3600


def _raise_teardown_api_error(error: RunPodAPIError) -> None:
    """Translate only canonical provider retry/auth outcomes after exact I/O."""

    if error.status_code in {401, 403}:
        raise OllamaLeaseAuthorizationError(
            "Runpod rejected Ollama cleanup authorization"
        ) from error
    if is_retryable_runpod_api_error(error):
        raise OllamaLeaseTeardownError(
            "Runpod Ollama cleanup remains retryable"
        ) from error
    raise error.with_traceback(error.__traceback__)


def _validate_exact_cleanup_readback(
    observed: object,
    resource: ProvisionedOllamaResource,
) -> None:
    expected_type = (
        EndpointResource
        if resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
        else PodResource
    )
    if (
        type(observed) is not expected_type
        or observed.id != resource.provider_resource_id
    ):
        raise OllamaLeaseConflictError(
            "Runpod cleanup readback did not match the original resource identity"
        )


@dataclass(frozen=True)
class RunpodOllamaDeployment:
    """Explicit runtime settings; image and region policy remain profile-owned."""

    profile: GPUProfile
    serverless_workers_max: int
    serverless_request_count: int
    serverless_execution_timeout_ms: int
    serverless_flashboot: FlashBoot
    http_timeout_seconds: float
    serverless_non_compute_cost: OllamaNonComputeCostPolicy
    pod_non_compute_cost: OllamaNonComputeCostPolicy
    billing_batch_hours: int = 24

    def __post_init__(self) -> None:
        if self.serverless_workers_max < 1 or self.serverless_request_count < 1:
            raise ValueError(
                "Runpod Ollama Serverless worker settings must be positive"
            )
        if self.serverless_execution_timeout_ms < 1 or self.http_timeout_seconds <= 0:
            raise ValueError("Runpod Ollama endpoint timeouts must be positive")
        if (
            not isinstance(self.billing_batch_hours, int)
            or isinstance(self.billing_batch_hours, bool)
            or self.billing_batch_hours < 1
        ):
            raise ValueError("Runpod Ollama billing batch size must be positive")
        for name, policy in (
            ("Serverless", self.serverless_non_compute_cost),
            ("Pod", self.pod_non_compute_cost),
        ):
            if not isinstance(policy, OllamaNonComputeCostPolicy):
                raise TypeError(f"Runpod Ollama {name} cost policy is required")
        common = {
            OllamaNonComputeCostComponent.CONTAINER_DISK,
            OllamaNonComputeCostComponent.MODEL_TRANSFER,
            OllamaNonComputeCostComponent.RETRY_ALLOWANCE,
        }
        # NETWORK_VOLUME means exactly one thing: an explicitly attached
        # network volume, identified by ``network_volume_id``.  A Pod's
        # ``volume_gb`` is persistent container storage — the Pod create path
        # below maps it to ``mounts.persistent``, never ``mounts.network`` —
        # and is already priced under CONTAINER_DISK, which every policy
        # covers unconditionally.  Treating it as a network
        # volume made operators declare a component their deployment does not
        # attach, and blurred the only signal that says whether a shared
        # network volume is in play.
        required = set(common)
        if self.profile.network_volume_id:
            required.add(OllamaNonComputeCostComponent.NETWORK_VOLUME)
        for name, policy in (
            ("Serverless", self.serverless_non_compute_cost),
            ("Pod", self.pod_non_compute_cost),
        ):
            if set(policy.covered_components) != required:
                raise ValueError(
                    f"Runpod Ollama {name} cost policy does not cover the exact "
                    "configured storage, transfer, and retry components"
                )

    @property
    def non_compute_cost_policies(
        self,
    ) -> Mapping[OllamaLeaseMode, OllamaNonComputeCostPolicy]:
        return {
            OllamaLeaseMode.SERVERLESS_LOAD_BALANCER: (
                self.serverless_non_compute_cost
            ),
            OllamaLeaseMode.DEDICATED_POD: self.pod_non_compute_cost,
        }


class RunpodOllamaCapacityProvider:
    """Provision only v2 load-balanced Serverless endpoints or dedicated Pods."""

    def __init__(
        self,
        *,
        client: RunpodControlPlaneClient,
        deployment: RunpodOllamaDeployment,
        serverless_api_key: str | None,
        pod_bearer_token: str | None,
        control_plane_api_key: str | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not serverless_api_key and not pod_bearer_token:
            raise RunPodManagerError(
                "A restricted Serverless key or Pod inference token is required"
            )
        workload_credentials = tuple(
            credential
            for credential in (serverless_api_key, pod_bearer_token)
            if credential
        )
        if control_plane_api_key and any(
            hmac.compare_digest(control_plane_api_key, credential)
            for credential in workload_credentials
        ):
            raise RunPodManagerError(
                "Ollama workload credentials must differ from RUNPOD_API_KEY"
            )
        if (
            serverless_api_key
            and pod_bearer_token
            and hmac.compare_digest(serverless_api_key, pod_bearer_token)
        ):
            raise RunPodManagerError(
                "Serverless and Pod Ollama credentials must be distinct"
            )
        self.client = client
        self.deployment = deployment
        self._serverless_api_key = serverless_api_key
        self._pod_bearer_token = pod_bearer_token
        self._http_transport = http_transport
        self._clock = clock

    def bearer_token_for(self, resource_type: OllamaResourceType) -> str:
        """Return the host-only workload credential for an acquired route."""

        token = (
            self._serverless_api_key
            if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
            else self._pod_bearer_token
        )
        if not token:
            raise RunPodManagerError(
                f"No workload credential is configured for {resource_type.value}"
            )
        return token

    async def plan(self, request: OllamaLeaseRequest) -> OllamaPlacementPlan:
        products = (
            (ComputeProduct.SERVERLESS,)
            if request.mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            else (ComputeProduct.POD,)
            if request.mode is OllamaLeaseMode.DEDICATED_POD
            else (ComputeProduct.SERVERLESS, ComputeProduct.POD)
        )
        decisions: dict[ComputeProduct, PlacementDecision] = {}
        failures: list[str] = []
        for product in products:
            if product is ComputeProduct.SERVERLESS and not self._serverless_api_key:
                failures.append(
                    "SERVERLESS: RUNPOD_SERVERLESS_API_KEY is not configured"
                )
                continue
            if product is ComputeProduct.POD and not self._pod_bearer_token:
                failures.append("POD: RUNPOD_OLLAMA_BEARER_TOKEN is not configured")
                continue
            if (
                product is ComputeProduct.SERVERLESS
                and request.idle_timeout_seconds
                > RUNPOD_SERVERLESS_MAX_IDLE_TIMEOUT_SECONDS
            ):
                failures.append(
                    "SERVERLESS: idle timeout exceeds the v2 endpoint maximum"
                )
                continue
            requirements = request.constraints.requirements(product)
            try:
                offers = await asyncio.to_thread(
                    self.client.list_gpus,
                    products=(product,),
                    count=requirements.gpu_count,
                    cloud=requirements.cloud,
                    min_cuda_version=requirements.min_cuda_version,
                )
                if product is ComputeProduct.SERVERLESS:
                    poolless_available = any(
                        offer.pool is None
                        and offer.availability not in {None, Availability.NONE}
                        for offer in offers
                    )
                    offers = tuple(offer for offer in offers if offer.pool)
                    if poolless_available and not offers:
                        raise RunPodManagerError(
                            "Runpod v2 advertised Serverless GPU availability "
                            "without the canonical pool ID required by endpoint "
                            "creation; refusing to guess a billable placement"
                        )
                decisions[product] = select_gpu(offers, requirements)
            except RunPodManagerError as exc:
                failures.append(f"{product.value}: {exc}")
        return select_ollama_plan(
            request,
            decisions,
            non_compute_cost_policies=self.deployment.non_compute_cost_policies,
            planned_at=self._clock(),
            serverless_max_workers=self.deployment.serverless_workers_max,
            failures=failures,
        )

    def validate_runtime_request(
        self,
        request: OllamaLeaseRequest,
        plan: OllamaPlacementPlan,
        *,
        provision_requested_at: datetime,
    ) -> PreparedOllamaProvisioning:
        """Prepare exactly one secret-bearing create request without provider I/O."""

        mode = plan.mode
        if mode is OllamaLeaseMode.AUTO or plan.resource_type is not (
            OllamaResourceType.SERVERLESS_ENDPOINT
            if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            else OllamaResourceType.POD
        ):
            raise RunPodManagerError("Runtime preflight requires a consistent plan")
        profile = self.deployment.profile
        image = require_immutable_ollama_image(
            _resolve_env_vars({"image": profile.image_name})["image"]
        )
        name = resource_name(request.lease_id)
        attempt_id = provision_attempt_id(request)
        telemetry_binding = ollama_runtime_telemetry_binding_sha256(
            lease_id=request.lease_id,
            request_fingerprint=request.fingerprint,
            provision_attempt_id=attempt_id,
            mode=mode,
            resource_type=plan.resource_type,
            resource_name=name,
            provision_requested_at=provision_requested_at,
            immutable_image=image,
        )
        environment = self._runtime_environment(
            request=request,
            mode=mode,
            provision_requested_at=provision_requested_at,
            telemetry_binding_sha256=telemetry_binding,
        )
        if plan.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            pool = plan.placement.gpu_pool
            if not pool:
                raise RunPodManagerError("Selected Serverless GPU has no pool ID")
            provider_request: PodCreateRequest | EndpointCreateRequest = (
                EndpointCreateRequest(
                    name=name,
                    image=image,
                    gpu_pools=(pool,),
                    endpoint_type="LOAD_BALANCER",
                    scaling={
                        "type": "REQUEST_COUNT",
                        "requestCount": self.deployment.serverless_request_count,
                    },
                    gpu_count=plan.placement.gpu_count,
                    min_cuda_version=request.constraints.min_cuda_version,
                    workers_min=0,
                    workers_max=self.deployment.serverless_workers_max,
                    idle_timeout_seconds=request.idle_timeout_seconds,
                    disk_gb=profile.container_disk_gb,
                    ports=tuple(profile.ports),
                    env=dict(environment),
                    registry_id=profile.registry_id,
                    data_center_ids=request.constraints.allowed_data_center_ids,
                    network_volume_ids=(
                        (profile.network_volume_id,)
                        if profile.network_volume_id
                        else ()
                    ),
                    execution_timeout_ms=(
                        self.deployment.serverless_execution_timeout_ms
                    ),
                    flashboot=self.deployment.serverless_flashboot,
                )
            )
        else:
            mounts: Mapping[str, Any] | None = None
            if profile.network_volume_id:
                mounts = {
                    "network": [
                        {
                            "volumeId": profile.network_volume_id,
                            "path": profile.volume_mount_path or "/workspace",
                        }
                    ]
                }
            elif profile.volume_gb:
                mounts = {
                    "persistent": {
                        "size": profile.volume_gb,
                        "path": profile.volume_mount_path or "/workspace",
                    }
                }
            provider_request = PodCreateRequest(
                name=name,
                image=image,
                gpu_id=plan.placement.gpu_id,
                gpu_count=plan.placement.gpu_count,
                cloud=request.constraints.cloud,
                disk_gb=profile.container_disk_gb,
                ports=tuple(profile.ports),
                env=dict(environment),
                registry_id=profile.registry_id,
                data_center_ids=request.constraints.allowed_data_center_ids,
                mounts=mounts,
                min_cuda_version=request.constraints.min_cuda_version,
            )
        safe_request = _safe_provider_request(provider_request.to_payload())
        request_json = _canonical_json(safe_request)
        intent = OllamaProvisioningIntent(
            lease_id=request.lease_id,
            request_fingerprint=request.fingerprint,
            provision_attempt_id=attempt_id,
            mode=mode,
            resource_type=plan.resource_type,
            resource_name=name,
            provision_requested_at=provision_requested_at,
            environment_sha256=environment_values_sha256(environment),
            environment_keys=tuple(sorted(environment)),
            provider_request_json=request_json,
            provider_request_sha256=_json_sha256(safe_request),
            selected_gpu_id=plan.placement.gpu_id,
            selected_gpu_pool=plan.placement.gpu_pool,
            selected_gpu_name=plan.placement.gpu_name,
            selected_gpu_count=plan.placement.gpu_count,
            selected_cloud=plan.placement.cloud,
            allowed_data_center_ids=request.constraints.allowed_data_center_ids,
            accepted_whole_hourly_rate_usd=(
                exact_decimal_integer_scale(
                    Decimal(str(plan.placement.offered_cost_per_hr)),
                    plan.placement.gpu_count,
                )
            ),
        )
        return PreparedOllamaProvisioning(
            intent=intent,
            provider_request=provider_request,
            runtime_telemetry_binding_sha256=telemetry_binding,
        )

    async def provision(
        self, prepared: PreparedOllamaProvisioning
    ) -> ProvisionedOllamaResource:
        intent = prepared.intent
        current = _safe_provider_request(prepared.provider_request.to_payload())
        if _json_sha256(current) != intent.provider_request_sha256:
            raise RunPodManagerError(
                "Prepared Ollama create request changed before POST"
            )
        if intent.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            if not isinstance(prepared.provider_request, EndpointCreateRequest):
                raise RunPodManagerError(
                    "Prepared Ollama resource type is inconsistent"
                )
            created = await asyncio.to_thread(
                self.client.create_endpoint, prepared.provider_request
            )
        else:
            if not isinstance(prepared.provider_request, PodCreateRequest):
                raise RunPodManagerError(
                    "Prepared Ollama resource type is inconsistent"
                )
            created = await asyncio.to_thread(
                self.client.create_pod, prepared.provider_request
            )
        if created.name != intent.resource_name:
            _raise_mismatch(
                created,
                intent,
                provider_resource_id=created.id,
                observed_at=self._clock(),
                reason=OllamaProvisioningMismatchReason.RESOURCE_IDENTITY,
                detail="Created Ollama resource identity is inconsistent",
            )
        return ProvisionedOllamaResource(
            resource_type=intent.resource_type,
            provider_resource_id=created.id,
            resource_name=created.name,
        )

    async def _read_original_provisioning(
        self,
        intent: OllamaProvisioningIntent,
        resource: ProvisionedOllamaResource,
    ) -> tuple[
        EndpointResource | PodResource,
        Mapping[str, Any],
        PodRealizedPlacement | None,
        OllamaProviderRatePrecision | None,
        datetime,
    ]:
        if (
            resource.resource_type is not intent.resource_type
            or resource.resource_name != intent.resource_name
        ):
            raise OllamaProvisioningEvidenceError(
                "Ollama provisioning readback resource does not bind its intent"
            )
        readback_at = self._clock()
        if intent.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            observed = await asyncio.to_thread(
                self.client.get_endpoint, resource.provider_resource_id
            )
        else:
            observed = await asyncio.to_thread(
                self.client.get_pod, resource.provider_resource_id
            )
        if observed.id != resource.provider_resource_id:
            _raise_mismatch(
                observed,
                intent,
                provider_resource_id=resource.provider_resource_id,
                observed_at=readback_at,
                reason=OllamaProvisioningMismatchReason.RESOURCE_IDENTITY,
                detail="Ollama provisioning exact-ID readback returned another resource",
            )
        try:
            if intent.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
                normalized, placement = (
                    _validate_endpoint_provisioning(
                        observed,
                        intent,
                        provider_resource_id=resource.provider_resource_id,
                        observed_at=readback_at,
                    ),
                    None,
                )
                precision = None
            else:
                normalized, placement = _validate_pod_provisioning(
                    observed,
                    intent,
                    provider_resource_id=resource.provider_resource_id,
                    observed_at=readback_at,
                )
                precision = OllamaProviderRatePrecision.PARSED_JSON_NUMBER
        except (OllamaProvisioningEvidenceError, OllamaProvisioningMismatchError):
            raise
        except RunPodManagerError as exc:
            raise OllamaProvisioningEvidenceError(
                "Ollama provider readback evidence is incomplete or invalid"
            ) from exc
        return observed, normalized, placement, precision, readback_at

    async def observe_provisioning(
        self,
        intent: OllamaProvisioningIntent,
        resource: ProvisionedOllamaResource,
        *,
        reconciled_after_ambiguous_create: bool = False,
    ) -> OllamaProvisioningReceipt:
        observed, normalized, placement, precision, readback_at = (
            await self._read_original_provisioning(intent, resource)
        )
        if observed.created_at is None or observed.environment_sha256 is None:
            raise OllamaProvisioningEvidenceError(
                "Ollama provider omitted original provisioning evidence"
            )
        return OllamaProvisioningReceipt(
            intent_sha256=intent.intent_sha256,
            lease_id=intent.lease_id,
            request_fingerprint=intent.request_fingerprint,
            provision_attempt_id=intent.provision_attempt_id,
            resource_type=intent.resource_type,
            provider_resource_id=resource.provider_resource_id,
            resource_name=intent.resource_name,
            provider_created_at=observed.created_at,
            readback_at=readback_at,
            readback_sha256=_json_sha256(normalized),
            environment_sha256=observed.environment_sha256,
            reconciled_after_ambiguous_create=reconciled_after_ambiguous_create,
            pod_realized_placement=placement,
            provider_rate_precision=precision,
        )

    async def discover_provisioning_resource(
        self, intent: OllamaProvisioningIntent
    ) -> ProvisionedOllamaResource | None:
        if intent.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            resources = await asyncio.to_thread(self.client.list_endpoints)
            matches = [item for item in resources if item.name == intent.resource_name]
        else:
            resources = await asyncio.to_thread(self.client.list_pods)
            matches = [item for item in resources if item.name == intent.resource_name]
        if len(matches) > 1:
            raise OllamaProvisioningEvidenceError(
                "Multiple Runpod resources match the original Ollama create name"
            )
        if not matches:
            return None
        return ProvisionedOllamaResource(
            resource_type=intent.resource_type,
            provider_resource_id=matches[0].id,
            resource_name=intent.resource_name,
        )

    async def observe(
        self,
        resource: ProvisionedOllamaResource,
        intent: OllamaProvisioningIntent | None = None,
        provisioning: OllamaProvisioningReceipt | None = None,
    ) -> OllamaReadinessObservation:
        if intent is None or provisioning is None:
            raise OllamaProvisioningEvidenceError(
                "Ollama readiness requires original provisioning authority"
            )
        try:
            provisioning.validate_intent(intent)
        except ValueError as exc:
            raise OllamaProvisioningEvidenceError(
                "Ollama readiness provisioning receipt is unbound"
            ) from exc
        if (
            provisioning.provider_resource_id != resource.provider_resource_id
            or provisioning.resource_type is not resource.resource_type
            or provisioning.resource_name != resource.resource_name
        ):
            raise OllamaProvisioningEvidenceError(
                "Ollama readiness resource does not bind its provisioning receipt"
            )
        observed, _normalized, _placement, _precision, _readback_at = (
            await self._read_original_provisioning(intent, resource)
        )
        if resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            if not isinstance(observed, EndpointResource):
                raise OllamaProvisioningEvidenceError(
                    "Ollama Endpoint readiness readback has the wrong type"
                )
            endpoint = observed
            base_url = endpoint.request_urls.get("base")
            health_url = endpoint.request_urls.get("health")
            provider_ready = bool(
                health_url
                and await self._healthy(
                    health_url, bearer_token=self._serverless_api_key
                )
            )
            models = (
                await self._models(base_url, bearer_token=self._serverless_api_key)
                if base_url
                else ()
            )
            bearer_token = self._serverless_api_key
        else:
            if not isinstance(observed, PodResource):
                raise OllamaProvisioningEvidenceError(
                    "Ollama Pod readiness readback has the wrong type"
                )
            pod = observed
            base_url = _pod_base_url(
                resource.provider_resource_id,
                pod.raw,
                self.deployment.profile.inference_port,
                self.deployment.profile.inference_protocol,
            )
            health_url = None
            private_route = bool(
                base_url and await self._rejects_anonymous_requests(base_url)
            )
            runtime_ready = bool(
                base_url
                and private_route
                and await self._healthy(
                    f"{base_url.rstrip('/')}/ping",
                    bearer_token=self._pod_bearer_token,
                )
            )
            provider_ready = (
                pod.status is not None
                and pod.status.upper() == "RUNNING"
                and private_route
                and runtime_ready
            )
            models = (
                await self._models(base_url, bearer_token=self._pod_bearer_token)
                if base_url and provider_ready
                else ()
            )
            bearer_token = self._pod_bearer_token
        telemetry = (
            await self._runtime_telemetry(
                base_url,
                bearer_token=bearer_token,
                intent=intent,
                provisioning=provisioning,
            )
            if base_url and provider_ready
            else None
        )
        return OllamaReadinessObservation(
            provider_ready=provider_ready,
            route_url=base_url,
            provider_health_url=health_url,
            model_names=models,
            runtime_telemetry=telemetry,
        )

    async def pull_model(
        self, resource: ProvisionedOllamaResource, route_url: str, model: str
    ) -> None:
        del resource, route_url, model
        raise RunPodManagerError(
            "The reviewed Ollama runtime owns model pulls and digest verification"
        )

    async def teardown(
        self,
        resource: ProvisionedOllamaResource,
        cleanup_intent: OllamaCleanupIntent,
    ) -> OllamaCleanupReceipt:
        """Request teardown, then require the exact resource GET to be absent.

        An acknowledged asynchronous action (or its own 404) is not a physical
        absence observation. Authentication failures are blockers. Canonical
        transient transport outcomes and a still-present exact resource use
        the provider's retry disposition for durable service reconciliation.
        """

        try:
            cleanup_intent.validate_resource(resource)
        except ValueError as exc:
            raise OllamaLeaseConflictError(
                "Ollama cleanup resource is unbound"
            ) from exc
        delete_acknowledged = True
        try:
            if resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
                await asyncio.to_thread(
                    self.client.delete_endpoint, resource.provider_resource_id
                )
            else:
                await asyncio.to_thread(
                    self.client.pod_action, resource.provider_resource_id, "terminate"
                )
        except RunPodAPIError as exc:
            if not is_exact_runpod_not_found(exc):
                _raise_teardown_api_error(exc)
            delete_acknowledged = False
        read_resource = (
            self.client.get_endpoint
            if resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT
            else self.client.get_pod
        )
        try:
            observed = await asyncio.to_thread(
                read_resource, resource.provider_resource_id
            )
        except RunPodAPIError as exc:
            if is_exact_runpod_not_found(exc):
                receipt = OllamaCleanupReceipt(
                    cleanup_intent_id=cleanup_intent.cleanup_intent_id,
                    verified_absent_at=self._clock(),
                    basis=(
                        OllamaCleanupBasis.DELETE_ACKNOWLEDGED
                        if delete_acknowledged
                        else OllamaCleanupBasis.AMBIGUOUS_DELETE_VERIFIED_ABSENT
                    ),
                )
                receipt.validate_intent(cleanup_intent)
                return receipt
            _raise_teardown_api_error(exc)
        _validate_exact_cleanup_readback(observed, resource)
        raise OllamaLeaseTeardownError(
            "Original Ollama resource is still present after teardown; reconcile it"
        )

    async def observe_lifecycle_billing(
        self,
        intent: OllamaProvisioningIntent,
        provisioning: OllamaProvisioningReceipt,
        cleanup: OllamaCleanupReceipt,
    ) -> OllamaLifecycleBillingObservation | None:
        provisioning.validate_intent(intent)
        if cleanup.verified_absent_at < provisioning.provider_created_at:
            raise RunPodManagerError("Ollama cleanup precedes provider creation")
        first_observed_at = self._clock()
        if provisioning.resource_type is OllamaResourceType.POD:
            placement = provisioning.pod_realized_placement
            if placement is None:
                raise RunPodManagerError(
                    "Ollama Pod billing lacks original realized placement"
                )
            pod = await read_exact_pod_billing(
                self.client,
                pod_id=provisioning.provider_resource_id,
                created_at=provisioning.provider_created_at,
                terminated_at=cleanup.verified_absent_at,
                realized_hourly_rate_usd=placement.hourly_rate_usd,
                clock=self._clock,
            )
            if pod is None:
                return None
            return OllamaLifecycleBillingObservation(
                intent_sha256=intent.intent_sha256,
                provisioning_receipt_id=provisioning.receipt_id,
                cleanup_receipt_id=cleanup.cleanup_receipt_id,
                lease_id=intent.lease_id,
                provision_attempt_id=intent.provision_attempt_id,
                resource_type=intent.resource_type,
                provider_resource_id=provisioning.provider_resource_id,
                cost_scope="pod_lifecycle",
                provider_capacity_actual_cost_usd=pod.actual_cost_usd,
                first_observed_at=first_observed_at,
                pod_billing_receipt=pod,
            )
        hours = await read_exact_serverless_billing(
            self.client,
            endpoint_id=provisioning.provider_resource_id,
            created_at=provisioning.provider_created_at,
            verified_absent_at=cleanup.verified_absent_at,
            clock=self._clock,
            batch_hours=self.deployment.billing_batch_hours,
        )
        if hours is None:
            return None
        return OllamaLifecycleBillingObservation(
            intent_sha256=intent.intent_sha256,
            provisioning_receipt_id=provisioning.receipt_id,
            cleanup_receipt_id=cleanup.cleanup_receipt_id,
            lease_id=intent.lease_id,
            provision_attempt_id=intent.provision_attempt_id,
            resource_type=intent.resource_type,
            provider_resource_id=provisioning.provider_resource_id,
            cost_scope="serverless_endpoint_lifecycle",
            provider_capacity_actual_cost_usd=serverless_endpoint_hour_cost_totals(
                hours
            )["actual_cost_usd"],
            first_observed_at=first_observed_at,
            serverless_endpoint_hour_costs=hours,
        )

    async def _runtime_telemetry(
        self,
        base_url: str,
        *,
        bearer_token: str | None,
        intent: OllamaProvisioningIntent,
        provisioning: OllamaProvisioningReceipt,
    ) -> OllamaRuntimeTelemetryObservation:
        if not bearer_token:
            raise RunPodManagerError("Ollama runtime telemetry credential is absent")
        fetch_started_at = self._clock()
        try:
            async with self._http_client(bearer_token=bearer_token) as client:
                response = await client.get(f"{base_url.rstrip('/')}/kestrel/telemetry")
                if response.status_code != 200:
                    raise RunPodManagerError("Ollama runtime telemetry is unavailable")
                raw: object = response.json()
        except httpx.HTTPError as exc:
            raise RunPodManagerError("Ollama runtime telemetry is unavailable") from exc
        except ValueError as exc:
            raise RunPodManagerError("Ollama runtime telemetry is invalid") from exc
        if not isinstance(raw, Mapping):
            raise RunPodManagerError("Ollama runtime telemetry is invalid")
        try:
            snapshot = OllamaRuntimeTelemetrySnapshot.from_dict(raw)
            observation = OllamaRuntimeTelemetryObservation(
                intent_sha256=intent.intent_sha256,
                provisioning_receipt_id=provisioning.receipt_id,
                lease_id=intent.lease_id,
                provision_attempt_id=intent.provision_attempt_id,
                resource_type=intent.resource_type,
                provider_resource_id=provisioning.provider_resource_id,
                snapshot=snapshot,
                snapshot_sha256=_json_sha256(snapshot.to_dict()),
                fetch_started_at=fetch_started_at,
                observed_at=self._clock(),
            )
            observation.validate_provisioning(
                intent,
                provisioning,
                telemetry_binding_sha256=(runtime_telemetry_binding_for_intent(intent)),
            )
            return observation
        except (TypeError, ValueError) as exc:
            raise RunPodManagerError("Ollama runtime telemetry is invalid") from exc

    async def _healthy(self, url: str, *, bearer_token: str | None) -> bool:
        try:
            async with self._http_client(bearer_token=bearer_token) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def _rejects_anonymous_requests(self, base_url: str) -> bool:
        try:
            async with self._http_client(bearer_token=None) as client:
                response = await client.get(f"{base_url.rstrip('/')}/api/tags")
                return response.status_code in {401, 403}
        except httpx.HTTPError:
            return False

    async def _models(
        self, base_url: str, *, bearer_token: str | None
    ) -> tuple[str, ...]:
        try:
            async with self._http_client(bearer_token=bearer_token) as client:
                response = await client.get(f"{base_url.rstrip('/')}/api/tags")
                if response.status_code < 200 or response.status_code >= 300:
                    return ()
                payload: object = response.json()
        except (httpx.HTTPError, ValueError):
            return ()
        if not isinstance(payload, Mapping):
            return ()
        models = payload.get("models")
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            return ()
        names: list[str] = []
        for item in models:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(names)

    def _http_client(self, *, bearer_token: str | None) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else None
        return httpx.AsyncClient(
            headers=headers,
            timeout=self.deployment.http_timeout_seconds,
            transport=self._http_transport,
        )

    def _runtime_environment(
        self,
        *,
        request: OllamaLeaseRequest,
        mode: OllamaLeaseMode,
        provision_requested_at: datetime,
        telemetry_binding_sha256: str,
    ) -> dict[str, str]:
        profile = self.deployment.profile
        require_immutable_ollama_image(
            _resolve_env_vars({"image": profile.image_name})["image"]
        )
        resource_type = (
            OllamaResourceType.SERVERLESS_ENDPOINT
            if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER
            else OllamaResourceType.POD
        )
        token = self.bearer_token_for(resource_type)
        if mode is OllamaLeaseMode.SERVERLESS_LOAD_BALANCER:
            if (
                profile.network_volume_id
                and self.deployment.serverless_workers_max != 1
            ):
                raise RunPodManagerError(
                    "A shared Ollama network-volume cache requires exactly one "
                    "Serverless worker to prevent concurrent mutation"
                )
            storage_path = (
                "/runpod-volume/ollama" if profile.network_volume_id else "/models"
            )
        else:
            storage_path = (
                f"{(profile.volume_mount_path or '/workspace').rstrip('/')}/ollama"
                if profile.network_volume_id or profile.volume_gb
                else "/models"
            )
        return build_ollama_runtime_environment(
            _resolve_env_vars(profile.env),
            requested_model=request.model,
            bearer_token=token,
            bearer_token_expires_at=request.hard_deadline,
            mode=mode,
            model_storage_path=storage_path,
            provision_requested_at=provision_requested_at,
            telemetry_binding_sha256=telemetry_binding_sha256,
        )


def _safe_provider_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    environment = payload.get("env")
    if not isinstance(environment, Mapping):
        raise RunPodManagerError("Prepared Ollama request omitted its environment")
    try:
        environment_sha256 = environment_values_sha256(environment)
    except TypeError as exc:
        raise RunPodManagerError("Prepared Ollama environment is invalid") from exc
    safe = dict(payload)
    safe["env"] = {
        "keys": sorted(environment),
        "sha256": environment_sha256,
    }
    return safe


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _intent_request(intent: OllamaProvisioningIntent) -> Mapping[str, Any]:
    value: object = json.loads(intent.provider_request_json)
    if not isinstance(value, Mapping):
        raise OllamaProvisioningEvidenceError(
            "Stored Ollama provisioning request is invalid"
        )
    return value


@dataclass
class _ProvisioningAssessment:
    """Collect every independent readback fact before choosing the outcome."""

    observed: EndpointResource | PodResource
    intent: OllamaProvisioningIntent
    provider_resource_id: str
    observed_at: datetime
    missing_common: list[str] = field(default_factory=list)
    missing_policy: list[str] = field(default_factory=list)
    missing_placement: list[str] = field(default_factory=list)
    missing_runtime: list[str] = field(default_factory=list)
    invalid_common: list[str] = field(default_factory=list)
    invalid_policy: list[str] = field(default_factory=list)
    invalid_placement: list[str] = field(default_factory=list)
    invalid_runtime: list[str] = field(default_factory=list)
    mismatches: list[tuple[OllamaProvisioningMismatchReason, str]] = field(
        default_factory=list
    )

    def missing(self, category: str, name: str) -> None:
        fields = getattr(self, f"missing_{category}")
        if name not in fields:
            fields.append(name)

    def invalid(self, category: str, name: str) -> None:
        fields = getattr(self, f"invalid_{category}")
        if name not in fields:
            fields.append(name)

    def record_resource_field(
        self,
        category: str,
        field_path: str,
        *,
        required: bool = True,
    ) -> None:
        state = self.observed.field_evidence.state(field_path)
        if state is ResourceFieldState.MISSING and required:
            self.missing(category, field_path)
        elif state is ResourceFieldState.INVALID:
            self.invalid(category, field_path)

    def unavailable(self, *categories: str) -> list[str]:
        missing = {
            item
            for category in categories
            for item in getattr(self, f"missing_{category}")
        }
        invalid = {
            item
            for category in categories
            for item in getattr(self, f"invalid_{category}")
        }
        return sorted({*missing, *(f"{item} (invalid)" for item in invalid)})

    def mismatch(self, reason: OllamaProvisioningMismatchReason, detail: str) -> None:
        self.mismatches.append((reason, detail))

    def raise_mismatch_if_any(self) -> None:
        if not self.mismatches:
            return
        reason, detail = self.mismatches[0]
        _raise_mismatch(
            self.observed,
            self.intent,
            provider_resource_id=self.provider_resource_id,
            observed_at=self.observed_at,
            reason=reason,
            detail=detail,
        )

    def finish_endpoint(self) -> None:
        self.raise_mismatch_if_any()
        unavailable = self.unavailable("common", "policy", "runtime")
        if unavailable:
            raise OllamaProvisioningEvidenceError(
                "Ollama Endpoint omitted original policy evidence: "
                + ", ".join(unavailable)
            )

    def finish_pod(self) -> None:
        self.raise_mismatch_if_any()
        common = self.unavailable("common")
        if common:
            raise OllamaProvisioningEvidenceError(
                "Ollama provider omitted original provisioning evidence: "
                + ", ".join(common)
            )
        placement = self.unavailable("placement")
        if placement:
            if placement == ["cudaVersion"]:
                raise OllamaProvisioningEvidenceError(
                    "Ollama Pod omitted CUDA readback evidence"
                )
            raise OllamaProvisioningEvidenceError(
                "Ollama Pod omitted realized placement or rate evidence: "
                + ", ".join(placement)
            )
        runtime = self.unavailable("runtime")
        if runtime:
            if runtime in (["globalNetworking"], ["globalNetworking (invalid)"]):
                raise OllamaProvisioningEvidenceError(
                    "Ollama Pod omitted global networking evidence"
                )
            raise OllamaProvisioningEvidenceError(
                "Ollama Pod omitted runtime or mount evidence: " + ", ".join(runtime)
            )


def _validate_common_provisioning(
    observed: EndpointResource | PodResource,
    intent: OllamaProvisioningIntent,
    *,
    provider_resource_id: str,
    observed_at: datetime,
) -> tuple[Mapping[str, Any], Mapping[str, Any], _ProvisioningAssessment]:
    expected = _intent_request(intent)
    assessment = _ProvisioningAssessment(
        observed=observed,
        intent=intent,
        provider_resource_id=provider_resource_id,
        observed_at=observed_at,
    )
    if observed.name != intent.resource_name:
        assessment.mismatch(
            OllamaProvisioningMismatchReason.RESOURCE_IDENTITY,
            "Ollama provisioning readback identity does not match intent",
        )
    if (
        observed.created_at is not None
        and observed.created_at < intent.provision_requested_at
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.RESOURCE_IDENTITY,
            "Ollama provisioning readback predates the original create",
        )
    expected_environment = expected.get("env")
    if (
        not isinstance(expected_environment, Mapping)
        or expected_environment.get("sha256") != intent.environment_sha256
    ):
        raise OllamaProvisioningEvidenceError(
            "Stored Ollama image or environment intent is invalid"
        )
    assessment.record_resource_field("common", "createdAt")
    assessment.record_resource_field("common", "image")
    assessment.record_resource_field("common", "env")
    if (
        observed.field_evidence.state("createdAt") is ResourceFieldState.PRESENT
        and observed.created_at is None
    ):
        assessment.missing("common", "createdAt")
    if (
        observed.field_evidence.state("image") is ResourceFieldState.PRESENT
        and observed.image is None
    ):
        assessment.missing("common", "image")
    if (
        observed.field_evidence.state("env") is ResourceFieldState.PRESENT
        and observed.environment_sha256 is None
    ):
        assessment.missing("common", "env")
    if (
        observed.field_evidence.state("image") is ResourceFieldState.PRESENT
        and observed.image != expected.get("image")
    ) or (
        observed.field_evidence.state("env") is ResourceFieldState.PRESENT
        and observed.environment_sha256 != intent.environment_sha256
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.IMAGE_OR_ENVIRONMENT,
            "Ollama provider readback image or environment does not match intent",
        )
    return expected, observed.raw, assessment


def _validate_endpoint_provisioning(
    endpoint: EndpointResource,
    intent: OllamaProvisioningIntent,
    *,
    provider_resource_id: str,
    observed_at: datetime,
) -> dict[str, Any]:
    expected, raw, assessment = _validate_common_provisioning(
        endpoint,
        intent,
        provider_resource_id=provider_resource_id,
        observed_at=observed_at,
    )
    assessment.record_resource_field("runtime", "requestUrls", required=False)
    assessment.record_resource_field("policy", "type")
    if (
        endpoint.field_evidence.state("type") is ResourceFieldState.PRESENT
        and endpoint.endpoint_type is None
    ):
        assessment.missing("policy", "type")
    if (
        endpoint.field_evidence.state("type") is ResourceFieldState.PRESENT
        and endpoint.endpoint_type != "LOAD_BALANCER"
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
            "Ollama Endpoint type does not match original request",
        )
    for key in ("args", "disk", "ports", "registry", "timeout", "flashboot"):
        if key not in raw:
            if (
                key in {"disk", "ports", "timeout", "flashboot"}
                or expected.get(key) is not None
            ):
                assessment.missing("runtime", key)
            continue
        try:
            actual = _typed_runtime_value(key, raw[key], context="Endpoint")
        except OllamaProvisioningEvidenceError:
            assessment.invalid("runtime", key)
            continue
        accepted = _typed_runtime_value(key, expected.get(key), context="intent")
        if actual != accepted:
            assessment.mismatch(
                OllamaProvisioningMismatchReason.RUNTIME,
                "Ollama Endpoint runtime configuration does not match intent",
            )
    expected_gpu = _required_mapping(expected.get("gpu"), "intent.gpu")
    expected_workers = _required_mapping(expected.get("workers"), "intent.workers")
    expected_scaling = _required_mapping(expected.get("scaling"), "intent.scaling")

    gpu_value = raw.get("gpu")
    if gpu_value is None:
        assessment.missing("policy", "gpu")
    elif not isinstance(gpu_value, Mapping):
        assessment.invalid("policy", "gpu")
    else:
        for key in (
            "pools",
            "count",
            "minCudaVersion",
            "allowedCudaVersions",
            "excludedTypes",
        ):
            if key not in expected_gpu:
                if key in gpu_value and gpu_value[key] not in (None, (), []):
                    assessment.mismatch(
                        OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
                        "Ollama Endpoint placement policy does not match intent",
                    )
                continue
            if key not in gpu_value:
                assessment.missing("policy", f"gpu.{key}")
                continue
            try:
                actual: object
                accepted: object
                if key in {"pools", "allowedCudaVersions", "excludedTypes"}:
                    actual = _string_tuple(gpu_value[key], f"Endpoint.gpu.{key}")
                    accepted = _string_tuple(expected_gpu.get(key), f"intent.gpu.{key}")
                elif key == "count":
                    actual = _required_int(gpu_value[key], "Endpoint.gpu.count")
                    accepted = _required_int(expected_gpu.get(key), "intent.gpu.count")
                else:
                    actual = gpu_value[key]
                    accepted = expected_gpu.get(key)
            except OllamaProvisioningEvidenceError:
                assessment.invalid("policy", f"gpu.{key}")
                continue
            if actual != accepted or (
                key in {"allowedCudaVersions", "excludedTypes"} and actual
            ):
                assessment.mismatch(
                    OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
                    "Ollama Endpoint placement policy does not match intent",
                )

    integer_fields = {
        "workers": frozenset({"min", "max", "idleTimeout"}),
        "scaling": frozenset({"requestCount"}),
    }
    string_fields = {"scaling": frozenset({"type"})}
    for field_name, accepted_mapping in (
        ("workers", expected_workers),
        ("scaling", expected_scaling),
    ):
        value = raw.get(field_name)
        if value is None:
            assessment.missing("policy", field_name)
            continue
        if not isinstance(value, Mapping):
            assessment.invalid("policy", field_name)
            continue
        for key, accepted in accepted_mapping.items():
            if key not in value:
                assessment.missing("policy", f"{field_name}.{key}")
                continue
            supplied = value[key]
            if key in integer_fields.get(field_name, ()) and (
                not isinstance(supplied, int) or isinstance(supplied, bool)
            ):
                assessment.invalid("policy", f"{field_name}.{key}")
                continue
            if key in string_fields.get(field_name, ()) and not isinstance(
                supplied, str
            ):
                assessment.invalid("policy", f"{field_name}.{key}")
                continue
            if supplied != accepted:
                assessment.mismatch(
                    OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
                    "Ollama Endpoint placement policy does not match intent",
                )
        if set(value).difference(accepted_mapping):
            assessment.mismatch(
                OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
                "Ollama Endpoint placement policy does not match intent",
            )

    for key in ("dataCenterIds", "networkVolumes"):
        if key not in raw:
            assessment.missing("policy", key)
            continue
        try:
            actual_values = _string_tuple(raw[key], f"Endpoint.{key}")
            accepted_values = _string_tuple(expected.get(key), f"intent.{key}")
        except OllamaProvisioningEvidenceError:
            assessment.invalid("policy", key)
            continue
        if actual_values != accepted_values:
            assessment.mismatch(
                OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
                "Ollama Endpoint placement policy does not match intent",
            )
    mount_assessment = assess_endpoint_mounts(
        raw,
        _string_tuple(expected.get("networkVolumes"), "intent.networkVolumes"),
    )
    if mount_assessment.mismatched_kinds:
        assessment.mismatch(
            OllamaProvisioningMismatchReason.ENDPOINT_POLICY,
            "Ollama Endpoint storage policy does not match intent",
        )
    for kind in mount_assessment.missing_kinds:
        assessment.missing("policy", f"mounts.{kind}")
    for kind in mount_assessment.invalid_kinds:
        assessment.invalid("policy", f"mounts.{kind}")
    assessment.finish_endpoint()
    return {
        "provider_resource_id": endpoint.id,
        "provider_created_at": (
            iso_datetime(endpoint.created_at) if endpoint.created_at else None
        ),
        "provider_request": expected,
    }


def _validate_pod_provisioning(
    pod: PodResource,
    intent: OllamaProvisioningIntent,
    *,
    provider_resource_id: str,
    observed_at: datetime,
) -> tuple[dict[str, Any], PodRealizedPlacement]:
    expected, raw, assessment = _validate_common_provisioning(
        pod,
        intent,
        provider_resource_id=provider_resource_id,
        observed_at=observed_at,
    )
    assessment.record_resource_field("common", "status")
    if (
        pod.field_evidence.state("status") is ResourceFieldState.PRESENT
        and pod.status is None
    ):
        assessment.missing("common", "status")
    if (
        pod.field_evidence.state("status") is ResourceFieldState.PRESENT
        and pod.status is not None
        and pod.status.upper() not in {"PROVISIONING", "STARTING", "RUNNING"}
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.POD_NOT_LIVE,
            "Ollama Pod is not live for original-rate evidence",
        )
    expected_gpu = _required_mapping(expected.get("gpu"), "intent.gpu")
    placement_detail = "Ollama Pod realized placement exceeds or differs from intent"
    assessment.record_resource_field("placement", "gpu")
    assessment.record_resource_field("placement", "gpu.id")
    if (
        pod.field_evidence.state("gpu.id") is ResourceFieldState.PRESENT
        and pod.gpu_id is None
    ):
        assessment.missing("placement", "gpu.id")
    if pod.field_evidence.state(
        "gpu.id"
    ) is ResourceFieldState.PRESENT and pod.gpu_id != expected_gpu.get("id"):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.POD_PLACEMENT_OR_RATE,
            placement_detail,
        )
    assessment.record_resource_field("placement", "gpu.count")
    if (
        pod.field_evidence.state("gpu.count") is ResourceFieldState.PRESENT
        and pod.gpu_count < 1
    ):
        assessment.missing("placement", "gpu.count")
    if pod.field_evidence.state(
        "gpu.count"
    ) is ResourceFieldState.PRESENT and pod.gpu_count != expected_gpu.get("count"):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.POD_PLACEMENT_OR_RATE,
            placement_detail,
        )
    assessment.record_resource_field("placement", "cloud")
    if (
        pod.field_evidence.state("cloud") is ResourceFieldState.PRESENT
        and pod.cloud is None
    ):
        assessment.missing("placement", "cloud")
    if (
        pod.field_evidence.state("cloud") is ResourceFieldState.PRESENT
        and pod.cloud is not None
        and pod.cloud.value != expected.get("cloud")
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.POD_PLACEMENT_OR_RATE,
            placement_detail,
        )
    assessment.record_resource_field("placement", "dataCenterId")
    if (
        pod.field_evidence.state("dataCenterId") is ResourceFieldState.PRESENT
        and pod.data_center_id is None
    ):
        assessment.missing("placement", "dataCenterId")
    if (
        pod.field_evidence.state("dataCenterId") is ResourceFieldState.PRESENT
        and pod.data_center_id is not None
        and intent.allowed_data_center_ids
        and pod.data_center_id not in intent.allowed_data_center_ids
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.POD_PLACEMENT_OR_RATE,
            placement_detail,
        )
    assessment.record_resource_field("placement", "cost")
    if (
        pod.field_evidence.state("cost") is ResourceFieldState.PRESENT
        and pod.cost_per_hr is None
    ):
        assessment.missing("placement", "cost")
    if (
        pod.field_evidence.state("cost") is ResourceFieldState.PRESENT
        and pod.cost_per_hr is not None
        and pod.cost_per_hr > intent.accepted_whole_hourly_rate_usd
    ):
        assessment.mismatch(
            OllamaProvisioningMismatchReason.POD_PLACEMENT_OR_RATE,
            placement_detail,
        )
    expected_floor = expected_gpu.get("minCudaVersion")
    if expected_floor is not None:
        assessment.record_resource_field("placement", "cudaVersion")
        if (
            pod.field_evidence.state("cudaVersion") is ResourceFieldState.PRESENT
            and pod.cuda_version is None
        ):
            assessment.missing("placement", "cudaVersion")
        if pod.field_evidence.state(
            "cudaVersion"
        ) is ResourceFieldState.PRESENT and not _cuda_meets_floor(
            pod.cuda_version, expected_floor
        ):
            assessment.mismatch(
                OllamaProvisioningMismatchReason.RUNTIME,
                "Ollama Pod CUDA readback does not satisfy intent",
            )
    required_runtime = {
        "args",
        "disk",
        "ports",
        "registry",
        "mounts",
        "globalNetworking",
    }
    accepted_runtime = {
        key: _typed_runtime_value(key, expected.get(key), context="intent")
        for key in ("args", "disk", "ports", "registry")
    }
    for key in required_runtime.difference(raw):
        assessment.missing("runtime", key)
    for key in ("args", "disk", "ports", "registry"):
        if key not in raw:
            continue
        try:
            actual = _typed_runtime_value(key, raw[key], context="Pod")
        except OllamaProvisioningEvidenceError:
            assessment.invalid("runtime", key)
            continue
        if actual != accepted_runtime[key]:
            assessment.mismatch(
                OllamaProvisioningMismatchReason.RUNTIME,
                "Ollama Pod runtime configuration does not match intent",
            )
    if "mounts" in raw:
        accepted_mounts = expected.get("mounts")
        if accepted_mounts is not None and not isinstance(accepted_mounts, Mapping):
            raise OllamaProvisioningEvidenceError(
                "Stored Ollama Pod mounts are invalid"
            )
        mount_assessment = assess_pod_mounts(raw, accepted_mounts)
        if mount_assessment.mismatched_kinds:
            assessment.mismatch(
                OllamaProvisioningMismatchReason.RUNTIME,
                "Ollama Pod runtime configuration does not match intent",
            )
        for kind in mount_assessment.missing_kinds:
            assessment.missing(
                "runtime", "mounts" if kind == "mounts" else f"mounts.{kind}"
            )
        for kind in mount_assessment.invalid_kinds:
            assessment.invalid(
                "runtime", "mounts" if kind == "mounts" else f"mounts.{kind}"
            )
    if "globalNetworking" in raw:
        try:
            networking_enabled = _pod_global_networking_enabled(raw["globalNetworking"])
        except OllamaProvisioningEvidenceError:
            assessment.invalid("runtime", "globalNetworking")
        else:
            if networking_enabled != expected.get("globalNetworking"):
                assessment.mismatch(
                    OllamaProvisioningMismatchReason.RUNTIME,
                    "Ollama Pod runtime configuration does not match intent",
                )
    assessment.finish_pod()
    try:
        placement = PodRealizedPlacement(
            provider_pod_id=pod.id,
            gpu_type_id=pod.gpu_id,
            gpu_display_name=intent.selected_gpu_name,
            gpu_count=pod.gpu_count,
            cloud=pod.cloud,
            data_center_id=pod.data_center_id,
            hourly_rate_usd=pod.cost_per_hr,
            observed_at=observed_at,
        )
    except (TypeError, ValueError) as exc:
        raise OllamaProvisioningEvidenceError(
            "Ollama Pod realized placement evidence is invalid"
        ) from exc
    return (
        {
            "provider_resource_id": pod.id,
            "provider_created_at": (
                iso_datetime(pod.created_at) if pod.created_at else None
            ),
            "provider_request": expected,
            "realized_placement": {
                key: value
                for key, value in placement.to_dict().items()
                if key != "observed_at"
            },
            "rate_precision": OllamaProviderRatePrecision.PARSED_JSON_NUMBER.value,
        },
        placement,
    )


def _normalized_response_argument(value: object) -> str | None:
    """Normalize the documented empty stored command to an omitted override."""

    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise OllamaProvisioningEvidenceError(
            "Ollama provider returned an invalid runtime argument"
        )
    return value


def _typed_runtime_value(key: str, value: object, *, context: str) -> object:
    """Decode one documented runtime field without bool/int equivalence."""

    if key == "args":
        return _normalized_response_argument(value)
    if key in {"disk", "timeout"}:
        return _required_int(value, f"{context}.{key}")
    if key == "ports":
        return _string_tuple(value, f"{context}.ports")
    if key in {"registry", "flashboot"}:
        if value is None:
            return None
        if not isinstance(value, str):
            raise OllamaProvisioningEvidenceError(
                f"{context}.{key} is missing or invalid"
            )
        return value
    raise ValueError(f"Unsupported Ollama runtime evidence field: {key}")


def _pod_global_networking_enabled(value: object) -> bool:
    """Read the response-only PodGlobalNetworking object from the v2 schema."""

    if not isinstance(value, Mapping) or not isinstance(value.get("enabled"), bool):
        raise OllamaProvisioningEvidenceError(
            "Ollama Pod omitted global networking evidence"
        )
    if not set(value).issubset({"enabled", "ip", "internalDns"}):
        raise OllamaProvisioningEvidenceError(
            "Ollama Pod returned invalid global networking evidence"
        )
    return value["enabled"]


def _raise_mismatch(
    observed: EndpointResource | PodResource,
    intent: OllamaProvisioningIntent,
    *,
    provider_resource_id: str,
    observed_at: datetime,
    reason: OllamaProvisioningMismatchReason,
    detail: str,
) -> None:
    safe_readback = {
        "provider_resource_id": provider_resource_id,
        "observed_resource_id": observed.id,
        "observed_resource_name": observed.name,
        "provider_created_at": (
            iso_datetime(observed.created_at) if observed.created_at else None
        ),
        "image": observed.image,
        "environment_sha256": observed.environment_sha256,
        "sanitized_provider_readback": observed.raw,
    }
    evidence = OllamaProvisioningMismatchEvidence(
        intent_sha256=intent.intent_sha256,
        lease_id=intent.lease_id,
        provision_attempt_id=intent.provision_attempt_id,
        resource_type=intent.resource_type,
        provider_resource_id=provider_resource_id,
        observed_resource_name=observed.name,
        observed_at=observed_at,
        readback_sha256=_json_sha256(safe_readback),
        reason=reason,
    )
    raise OllamaProvisioningMismatchError(detail, evidence)


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OllamaProvisioningEvidenceError(f"{name} is missing or invalid")
    return value


def _required_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise OllamaProvisioningEvidenceError(f"{name} is missing or invalid")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise OllamaProvisioningEvidenceError(f"{name} is missing or invalid")
    return tuple(value)


def _cuda_meets_floor(actual: str | None, expected: Any) -> bool:
    if actual is None or not isinstance(expected, str):
        return False
    try:
        actual_normalized = normalize_cuda_floor(actual)
        expected_normalized = normalize_cuda_floor(expected)
    except ValueError:
        return False
    assert actual_normalized is not None and expected_normalized is not None
    actual_parts = tuple(int(part) for part in actual_normalized.split("."))
    expected_parts = tuple(int(part) for part in expected_normalized.split("."))
    return actual_parts >= expected_parts


def _pod_base_url(
    pod_id: str,
    payload: Mapping[str, Any],
    inference_port: int,
    protocol: str,
) -> str | None:
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    ports = runtime.get("ports")
    if not isinstance(ports, Sequence) or isinstance(ports, (str, bytes)):
        return None
    for item in ports:
        if not isinstance(item, Mapping):
            continue
        private = item.get("private") or item.get("privatePort")
        try:
            matches = (
                isinstance(private, (str, int))
                and not isinstance(private, bool)
                and int(private) == inference_port
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            continue
        port_type = str(item.get("type", "")).lower()
        if port_type == "http":
            return f"https://{pod_id}-{inference_port}.proxy.runpod.net"
        if protocol.strip().lower() != "https":
            return None
        ip = item.get("ip")
        public = item.get("public") or item.get("publicPort")
        if (
            isinstance(ip, str)
            and isinstance(public, (str, int))
            and not isinstance(public, bool)
        ):
            return f"https://{ip}:{int(public)}"
    return None
