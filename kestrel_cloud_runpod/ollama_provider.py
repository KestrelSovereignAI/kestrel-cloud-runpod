"""Runpod v2 capacity, readiness, and teardown adapter for Ollama leases."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from .clients import RunpodControlPlaneClient
from .models import (
    Availability,
    ComputeProduct,
    EndpointCreateRequest,
    FlashBoot,
    GPUProfile,
    PlacementDecision,
    PodCreateRequest,
    RunPodAPIError,
    RunPodManagerError,
)
from .ollama_contracts import (
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaNonComputeCostComponent,
    OllamaNonComputeCostPolicy,
    OllamaPlacementPlan,
    OllamaReadinessObservation,
    OllamaResourceType,
    ProvisionedOllamaResource,
    select_ollama_plan,
)
from .ollama_runtime import (
    build_ollama_runtime_environment,
    require_immutable_ollama_image,
)
from .placement import select_gpu
from .providers import _resolve_env_vars

RUNPOD_SERVERLESS_MAX_IDLE_TIMEOUT_SECONDS = 3600


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

    def __post_init__(self) -> None:
        if self.serverless_workers_max < 1 or self.serverless_request_count < 1:
            raise ValueError(
                "Runpod Ollama Serverless worker settings must be positive"
            )
        if self.serverless_execution_timeout_ms < 1 or self.http_timeout_seconds <= 0:
            raise ValueError("Runpod Ollama endpoint timeouts must be positive")
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

    def validate_runtime_request(self, request: OllamaLeaseRequest) -> None:
        """Validate workload image, model, and credential before catalog access."""

        if request.mode is OllamaLeaseMode.AUTO:
            raise RunPodManagerError(
                "Runtime preflight requires a concrete Ollama execution mode"
            )
        self._runtime_environment(
            request=request,
            mode=request.mode,
            provision_requested_at=self._clock(),
        )

    async def provision(
        self,
        *,
        request: OllamaLeaseRequest,
        plan: OllamaPlacementPlan,
        resource_name: str,
    ) -> ProvisionedOllamaResource:
        profile = self.deployment.profile
        image = require_immutable_ollama_image(
            _resolve_env_vars({"image": profile.image_name})["image"]
        )
        provision_requested_at = self._clock()
        if plan.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            env = self._runtime_environment(
                request=request,
                mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
                provision_requested_at=provision_requested_at,
            )
            pool = plan.placement.gpu_pool
            if not pool:
                raise RunPodManagerError("Selected Serverless GPU has no pool ID")
            endpoint = await asyncio.to_thread(
                self.client.create_endpoint,
                EndpointCreateRequest(
                    name=resource_name,
                    image=image,
                    gpu_pools=(pool,),
                    endpoint_type="LOAD_BALANCER",
                    scaling={
                        "type": "REQUEST_COUNT",
                        "requestCount": self.deployment.serverless_request_count,
                    },
                    gpu_count=plan.placement.gpu_count,
                    workers_min=0,
                    workers_max=self.deployment.serverless_workers_max,
                    idle_timeout_seconds=request.idle_timeout_seconds,
                    disk_gb=profile.container_disk_gb,
                    ports=tuple(profile.ports),
                    env=env,
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
                ),
            )
            return ProvisionedOllamaResource(
                resource_type=plan.resource_type,
                provider_resource_id=endpoint.id,
                resource_name=resource_name,
            )
        env = self._runtime_environment(
            request=request,
            mode=OllamaLeaseMode.DEDICATED_POD,
            provision_requested_at=provision_requested_at,
        )
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
        pod = await asyncio.to_thread(
            self.client.create_pod,
            PodCreateRequest(
                name=resource_name,
                image=image,
                gpu_id=plan.placement.gpu_id,
                gpu_count=plan.placement.gpu_count,
                cloud=request.constraints.cloud,
                disk_gb=profile.container_disk_gb,
                ports=tuple(profile.ports),
                env=env,
                registry_id=profile.registry_id,
                data_center_ids=request.constraints.allowed_data_center_ids,
                mounts=mounts,
            ),
        )
        return ProvisionedOllamaResource(
            resource_type=plan.resource_type,
            provider_resource_id=pod.id,
            resource_name=resource_name,
        )

    async def find_resource(
        self, *, resource_type: OllamaResourceType, resource_name: str
    ) -> ProvisionedOllamaResource | None:
        if resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            resources = await asyncio.to_thread(self.client.list_endpoints)
            matches = [item for item in resources if item.name == resource_name]
        else:
            resources = await asyncio.to_thread(self.client.list_pods)
            matches = [item for item in resources if item.name == resource_name]
        if len(matches) > 1:
            raise RunPodManagerError(
                f"Multiple Runpod resources match durable name '{resource_name}'"
            )
        if not matches:
            return None
        return ProvisionedOllamaResource(
            resource_type=resource_type,
            provider_resource_id=matches[0].id,
            resource_name=resource_name,
        )

    async def observe(
        self, resource: ProvisionedOllamaResource
    ) -> OllamaReadinessObservation:
        if resource.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            endpoint = await asyncio.to_thread(
                self.client.get_endpoint, resource.provider_resource_id
            )
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
        else:
            pod = await asyncio.to_thread(
                self.client.get_pod, resource.provider_resource_id
            )
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
                pod.status.upper() == "RUNNING" and private_route and runtime_ready
            )
            models = (
                await self._models(base_url, bearer_token=self._pod_bearer_token)
                if base_url and provider_ready
                else ()
            )
        return OllamaReadinessObservation(
            provider_ready=provider_ready,
            route_url=base_url,
            provider_health_url=health_url,
            model_names=models,
        )

    async def pull_model(
        self, resource: ProvisionedOllamaResource, route_url: str, model: str
    ) -> None:
        del resource, route_url, model
        raise RunPodManagerError(
            "The reviewed Ollama runtime owns model pulls and digest verification"
        )

    async def teardown(self, resource: ProvisionedOllamaResource) -> None:
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
            if exc.status_code != 404:
                raise

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
        )


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
