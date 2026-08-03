"""Shared deterministic fixtures for Ollama lease tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    ComputeProduct,
    PlacementDecision,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaPlacementPlan,
    OllamaReadinessObservation,
    OllamaResourceConstraints,
    OllamaResourceType,
    ProvisionedOllamaResource,
)


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 1, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def make_request(clock: MutableClock, **changes) -> OllamaLeaseRequest:
    request = OllamaLeaseRequest(
        lease_id="lease-001",
        owner_id="owner-001",
        workload_id="workload-001",
        model="qwen3:8b",
        constraints=OllamaResourceConstraints(min_vram_gb=24),
        expected_session_seconds=3600,
        expected_active_seconds=300,
        serverless_initialization_seconds=60,
        serverless_idle_tail_seconds=30,
        idle_timeout_seconds=300,
        readiness_timeout_seconds=120,
        hard_deadline=clock() + timedelta(hours=2),
        max_authorized_cost=2.0,
    )
    return replace(request, **changes)


def make_decision(
    product: ComputeProduct,
    *,
    rate: float,
    gpu_id: str,
    pool: str | None,
) -> PlacementDecision:
    requirements = OllamaResourceConstraints(min_vram_gb=24).requirements(product)
    return PlacementDecision(
        gpu_id=gpu_id,
        gpu_pool=pool,
        gpu_name=gpu_id,
        memory_gb=24,
        cloud=CloudType.SECURE,
        gpu_count=1,
        offered_cost_per_hr=rate,
        availability=Availability.HIGH,
        catalog_observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        requirements=requirements,
    )


class FakeOllamaProvider:
    def __init__(self, plan: OllamaPlacementPlan) -> None:
        self.selected_plan = plan
        self.provision_calls = 0
        self.pull_calls = 0
        self.teardown_calls = 0
        self.teardown_failures = 0
        self.models: tuple[str, ...] = ("qwen3:8b",)
        self.provider_ready = True
        self.route_url = "https://private.example"
        self.health_url = "https://private.example/ping"
        self.resource = ProvisionedOllamaResource(
            resource_type=plan.resource_type,
            provider_resource_id="provider-001",
            resource_name="unset",
        )
        self.find_result: ProvisionedOllamaResource | None = None

    async def plan(self, request: OllamaLeaseRequest) -> OllamaPlacementPlan:
        del request
        return self.selected_plan

    async def provision(
        self,
        *,
        request: OllamaLeaseRequest,
        plan: OllamaPlacementPlan,
        resource_name: str,
    ) -> ProvisionedOllamaResource:
        del request, plan
        self.provision_calls += 1
        self.resource = replace(self.resource, resource_name=resource_name)
        return self.resource

    async def find_resource(
        self, *, resource_type: OllamaResourceType, resource_name: str
    ) -> ProvisionedOllamaResource | None:
        del resource_type, resource_name
        return self.find_result

    async def observe(
        self, resource: ProvisionedOllamaResource
    ) -> OllamaReadinessObservation:
        del resource
        return OllamaReadinessObservation(
            provider_ready=self.provider_ready,
            route_url=self.route_url,
            provider_health_url=self.health_url,
            model_names=self.models,
        )

    async def pull_model(
        self, resource: ProvisionedOllamaResource, route_url: str, model: str
    ) -> None:
        del resource, route_url
        self.pull_calls += 1
        self.models = (model,)

    async def teardown(self, resource: ProvisionedOllamaResource) -> None:
        del resource
        self.teardown_calls += 1
        if self.teardown_failures:
            self.teardown_failures -= 1
            from kestrel_cloud_runpod.models import RunPodManagerError

            raise RunPodManagerError("temporary teardown failure")


def serverless_plan(
    rate: float = 0.9, *, estimated_cost: float = 0.1
) -> OllamaPlacementPlan:
    decision = make_decision(
        ComputeProduct.SERVERLESS,
        rate=rate,
        gpu_id="gpu-serverless",
        pool="pool-24",
    )
    return OllamaPlacementPlan(
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        placement=decision,
        estimated_cost=estimated_cost,
        estimated_billable_seconds=390,
        maximum_serverless_cold_starts=1,
    )
