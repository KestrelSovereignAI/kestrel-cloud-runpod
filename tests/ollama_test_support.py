"""Shared deterministic fixtures for Ollama lease tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    ComputeProduct,
    PlacementDecision,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaNonComputeCostComponent,
    OllamaNonComputeCostPolicy,
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
    gpu_count: int = 1,
) -> PlacementDecision:
    """Build a placement decision.

    ``rate`` is the catalog's **per-GPU** hourly price, matching
    ``/catalog/gpus`` ``price.secure``/``price.community``; ``gpu_count`` is
    how many of them the placement attaches.
    """
    requirements = OllamaResourceConstraints(
        min_vram_gb=24, gpu_count=gpu_count
    ).requirements(product)
    return PlacementDecision(
        gpu_id=gpu_id,
        gpu_pool=pool,
        gpu_name=gpu_id,
        memory_gb=24,
        cloud=CloudType.SECURE,
        gpu_count=gpu_count,
        offered_cost_per_hr=rate,
        availability=Availability.HIGH,
        catalog_observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        requirements=requirements,
    )


_BASE_COST_COMPONENTS = (
    OllamaNonComputeCostComponent.CONTAINER_DISK,
    OllamaNonComputeCostComponent.MODEL_TRANSFER,
    OllamaNonComputeCostComponent.RETRY_ALLOWANCE,
)


def non_compute_cost_policy(
    *,
    estimated: float = 0.0,
    maximum: float = 0.0,
    include_network_volume: bool = False,
) -> OllamaNonComputeCostPolicy:
    components = _BASE_COST_COMPONENTS
    if include_network_volume:
        components = tuple(
            sorted(
                (*components, OllamaNonComputeCostComponent.NETWORK_VOLUME),
                key=lambda item: item.value,
            )
        )
    return OllamaNonComputeCostPolicy(
        estimated_cost_usd=estimated,
        maximum_cost_usd=maximum,
        covered_components=components,
    )


def non_compute_cost_policies(
    *, estimated: float = 0.0, maximum: float = 0.0
) -> dict[OllamaLeaseMode, OllamaNonComputeCostPolicy]:
    return {
        OllamaLeaseMode.SERVERLESS_LOAD_BALANCER: non_compute_cost_policy(
            estimated=estimated, maximum=maximum
        ),
        OllamaLeaseMode.DEDICATED_POD: non_compute_cost_policy(
            estimated=estimated, maximum=maximum
        ),
    }


class FakeOllamaProvider:
    def __init__(self, plan: OllamaPlacementPlan) -> None:
        self.selected_plan = plan
        self.provision_calls = 0
        self.plan_calls = 0
        self.drift_plan: OllamaPlacementPlan | None = None
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
        self.plan_calls += 1
        # The live catalog can move between quoting and acquiring. Returning
        # the identical plan for both hides every acceptance guard that exists
        # to catch that drift, so tests can supply a second plan.
        if self.drift_plan is not None and self.plan_calls > 1:
            return self.drift_plan
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
    rate: float = 0.9,
    *,
    estimated_cost: float = 0.1,
    maximum_compute_cost: float | None = None,
    estimated_non_compute_cost: float = 0.0,
    maximum_non_compute_cost: float = 0.0,
    gpu_count: int = 1,
) -> OllamaPlacementPlan:
    decision = make_decision(
        ComputeProduct.SERVERLESS,
        rate=rate,
        gpu_id="gpu-serverless",
        pool="pool-24",
        gpu_count=gpu_count,
    )
    compute_ceiling = (
        estimated_cost if maximum_compute_cost is None else maximum_compute_cost
    )
    # Costs are rated as rate x seconds x gpu_count / 3600, so the seconds that
    # produce a given cost shrink as GPUs are added.
    unit_rate = Decimal(str(rate)) * Decimal(gpu_count)
    estimated_billable_seconds = round(
        Decimal(str(estimated_cost)) * Decimal(3600) / unit_rate
    )
    maximum_billable_seconds = round(
        Decimal(str(compute_ceiling)) * Decimal(3600) / unit_rate
    )

    # Re-derive the costs FROM the rounded seconds, in the same direction and
    # with the same formula the plan's own consistency check uses, so a rate
    # that does not divide evenly cannot make the fixture self-inconsistent.
    def _rate(seconds: int) -> float:
        return float(
            Decimal(str(rate)) * Decimal(seconds) * Decimal(gpu_count) / Decimal(3600)
        )

    estimated_cost = _rate(estimated_billable_seconds)
    compute_ceiling = _rate(maximum_billable_seconds)
    estimated_total = float(
        Decimal(str(estimated_cost)) + Decimal(str(estimated_non_compute_cost))
    )
    total_ceiling = float(
        Decimal(str(compute_ceiling)) + Decimal(str(maximum_non_compute_cost))
    )
    return OllamaPlacementPlan(
        mode=OllamaLeaseMode.SERVERLESS_LOAD_BALANCER,
        resource_type=OllamaResourceType.SERVERLESS_ENDPOINT,
        placement=decision,
        estimated_compute_cost=estimated_cost,
        maximum_compute_cost=compute_ceiling,
        estimated_non_compute_cost=estimated_non_compute_cost,
        maximum_non_compute_cost=maximum_non_compute_cost,
        estimated_cost=estimated_total,
        cost_ceiling=total_ceiling,
        estimated_billable_seconds=estimated_billable_seconds,
        maximum_billable_seconds=maximum_billable_seconds,
        maximum_concurrent_workers=1,
        non_compute_components=_BASE_COST_COMPONENTS,
        maximum_serverless_cold_starts=1,
    )
