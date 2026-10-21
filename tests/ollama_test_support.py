"""Shared deterministic fixtures for Ollama lease tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

from kestrel_cloud_runpod.billing import (
    billing_exposure_ratio_float,
    exact_decimal_integer_scale,
    exact_decimal_sum,
)
from kestrel_cloud_runpod.models import (
    Availability,
    CloudType,
    ComputeProduct,
    EndpointCreateRequest,
    PlacementDecision,
    PodCreateRequest,
    environment_values_sha256,
)
from kestrel_cloud_runpod.ollama_contracts import (
    OllamaCleanupBasis,
    OllamaCleanupIntent,
    OllamaCleanupReceipt,
    OllamaLeaseMode,
    OllamaLeaseRequest,
    OllamaLeaseTeardownError,
    OllamaNonComputeCostComponent,
    OllamaNonComputeCostPolicy,
    OllamaPlacementPlan,
    OllamaProviderRatePrecision,
    OllamaProvisioningIntent,
    OllamaProvisioningReceipt,
    OllamaReadinessObservation,
    OllamaResourceConstraints,
    OllamaResourceType,
    OllamaRuntimeTelemetryObservation,
    OllamaRuntimeTelemetrySnapshot,
    PreparedOllamaProvisioning,
    ProvisionedOllamaResource,
    ollama_runtime_telemetry_binding_sha256,
    provision_attempt_id,
    resource_name,
    runtime_telemetry_binding_for_intent,
)
from kestrel_cloud_runpod.pod_capacity_contracts import PodRealizedPlacement


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


def runtime_telemetry_payload(binding: str, timestamp: str) -> dict[str, object]:
    """Build one exact authenticated runtime snapshot for native service tests."""

    return {
        "schema_version": 2,
        "runtime_version": "test",
        "ollama_version": "test",
        "telemetry_binding_sha256": binding,
        "runtime_instance_id": "runtime-instance-1",
        "telemetry_revision": 1,
        "provision_requested_at": timestamp,
        "container_started_at": timestamp,
        "ollama_process_started_at": timestamp,
        "ollama_healthy_at": timestamp,
        "model_pull_started_at": None,
        "model_pull_completed_at": None,
        "model_load_started_at": timestamp,
        "model_ready_at": timestamp,
        "ready_at": timestamp,
        "placement_and_image_pull_wall_ms": 0,
        "ollama_boot_ms": 0,
        "model_pull_ms": None,
        "model_load_ms": 0,
        "container_to_ready_ms": 0,
        "cache_hit": True,
        "startup_strategy": "cache_hit",
        "ready": True,
        "ollama_process_running": True,
        "terminal_error": None,
        "inference_started": 0,
        "inference_completed": 0,
        "inference_failed": 0,
        "inference_cancelled": 0,
        "inference_in_flight": 0,
        "first_inference_started_at": None,
        "last_inference_started_at": None,
        "last_inference_completed_at": None,
        "inference_busy_ms": 0,
        "inference_idle_ms": 0,
    }


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

    def validate_runtime_request(
        self,
        request: OllamaLeaseRequest,
        plan: OllamaPlacementPlan,
        *,
        provision_requested_at: datetime,
    ) -> PreparedOllamaProvisioning:
        name = resource_name(request.lease_id)
        image = "example.invalid/ollama@sha256:" + "a" * 64
        attempt_id = provision_attempt_id(request)
        binding = ollama_runtime_telemetry_binding_sha256(
            lease_id=request.lease_id,
            request_fingerprint=request.fingerprint,
            provision_attempt_id=attempt_id,
            mode=plan.mode,
            resource_type=plan.resource_type,
            resource_name=name,
            provision_requested_at=provision_requested_at,
            immutable_image=image,
        )
        environment = {
            "KESTREL_OLLAMA_BEARER_TOKEN": "test-" + "s" * 32,
            "KESTREL_OLLAMA_PROVISION_REQUESTED_AT": (
                provision_requested_at.isoformat()
            ),
            "KESTREL_OLLAMA_TELEMETRY_BINDING_SHA256": binding,
        }
        if plan.resource_type is OllamaResourceType.SERVERLESS_ENDPOINT:
            provider_request = EndpointCreateRequest(
                name=name,
                image=image,
                gpu_pools=(plan.placement.gpu_pool or "test-pool",),
                endpoint_type="LOAD_BALANCER",
                scaling={"type": "REQUEST_COUNT", "requestCount": 1},
                gpu_count=plan.placement.gpu_count,
                env=environment,
                data_center_ids=request.constraints.allowed_data_center_ids,
            )
        else:
            provider_request = PodCreateRequest(
                name=name,
                image=image,
                gpu_id=plan.placement.gpu_id,
                gpu_count=plan.placement.gpu_count,
                cloud=plan.placement.cloud,
                env=environment,
                data_center_ids=request.constraints.allowed_data_center_ids,
            )
        safe = provider_request.to_payload()
        safe["env"] = {
            "keys": sorted(environment),
            "sha256": environment_values_sha256(environment),
        }
        canonical = json.dumps(safe, sort_keys=True, separators=(",", ":"))
        intent = OllamaProvisioningIntent(
            lease_id=request.lease_id,
            request_fingerprint=request.fingerprint,
            provision_attempt_id=attempt_id,
            mode=plan.mode,
            resource_type=plan.resource_type,
            resource_name=resource_name(request.lease_id),
            provision_requested_at=provision_requested_at,
            environment_sha256=environment_values_sha256(environment),
            environment_keys=tuple(sorted(environment)),
            provider_request_json=canonical,
            provider_request_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
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
        return PreparedOllamaProvisioning(intent, provider_request, binding)

    async def plan(self, request: OllamaLeaseRequest) -> OllamaPlacementPlan:
        del request
        # The live catalog can move between quoting and acquiring. Returning
        # the identical plan for both hides every acceptance guard that exists
        # to catch that drift, so tests can supply a second plan.
        #
        # Drift applies from the moment the test assigns it, which is the
        # phase boundary the test already expresses by assigning it after
        # quoting. It is deliberately NOT keyed off the call count: quote()
        # calls plan() once per candidate mode, so a `plan_calls > 1` rule
        # would also mean "the POD candidate inside this same quote()" and
        # would silently drift it — quietly changing which candidate min()
        # selects, with no error anywhere.
        if self.drift_plan is not None:
            return self.drift_plan
        return self.selected_plan

    async def provision(
        self, prepared: PreparedOllamaProvisioning
    ) -> ProvisionedOllamaResource:
        self.provision_calls += 1
        self.resource = replace(
            self.resource,
            resource_type=prepared.intent.resource_type,
            resource_name=prepared.intent.resource_name,
        )
        return self.resource

    async def observe_provisioning(
        self,
        intent: OllamaProvisioningIntent,
        resource: ProvisionedOllamaResource,
        *,
        reconciled_after_ambiguous_create: bool = False,
    ) -> OllamaProvisioningReceipt:
        placement = None
        precision = None
        if intent.resource_type is OllamaResourceType.POD:
            placement = PodRealizedPlacement(
                provider_pod_id=resource.provider_resource_id,
                gpu_type_id=intent.selected_gpu_id,
                gpu_display_name=intent.selected_gpu_name,
                gpu_count=intent.selected_gpu_count,
                cloud=intent.selected_cloud,
                data_center_id=(intent.allowed_data_center_ids or ("US-TX-1",))[0],
                hourly_rate_usd=intent.accepted_whole_hourly_rate_usd,
                observed_at=intent.provision_requested_at,
            )
            precision = OllamaProviderRatePrecision.PARSED_JSON_NUMBER
        return OllamaProvisioningReceipt(
            intent_sha256=intent.intent_sha256,
            lease_id=intent.lease_id,
            request_fingerprint=intent.request_fingerprint,
            provision_attempt_id=intent.provision_attempt_id,
            resource_type=intent.resource_type,
            provider_resource_id=resource.provider_resource_id,
            resource_name=intent.resource_name,
            provider_created_at=intent.provision_requested_at,
            readback_at=intent.provision_requested_at,
            readback_sha256="b" * 64,
            environment_sha256=intent.environment_sha256,
            reconciled_after_ambiguous_create=reconciled_after_ambiguous_create,
            pod_realized_placement=placement,
            provider_rate_precision=precision,
        )

    async def discover_provisioning_resource(
        self, intent: OllamaProvisioningIntent
    ) -> ProvisionedOllamaResource | None:
        del intent
        if self.find_result is None:
            return None
        return self.find_result

    async def observe(
        self,
        resource: ProvisionedOllamaResource,
        intent: OllamaProvisioningIntent,
        provisioning: OllamaProvisioningReceipt,
    ) -> OllamaReadinessObservation:
        binding = runtime_telemetry_binding_for_intent(intent)
        snapshot = OllamaRuntimeTelemetrySnapshot(
            runtime_version="test",
            ollama_version="test",
            telemetry_binding_sha256=binding,
            runtime_instance_id="runtime-test-1",
            telemetry_revision=1,
            provision_requested_at=intent.provision_requested_at,
            container_started_at=intent.provision_requested_at,
            ollama_process_started_at=intent.provision_requested_at,
            ollama_healthy_at=intent.provision_requested_at,
            model_pull_started_at=None,
            model_pull_completed_at=None,
            model_load_started_at=intent.provision_requested_at,
            model_ready_at=intent.provision_requested_at,
            ready_at=intent.provision_requested_at,
            placement_and_image_pull_wall_ms=0,
            ollama_boot_ms=0,
            model_pull_ms=None,
            model_load_ms=0,
            container_to_ready_ms=0,
            cache_hit=True,
            startup_strategy="cache_hit",
            ready=True,
            ollama_process_running=True,
            terminal_error=None,
            inference_started=0,
            inference_completed=0,
            inference_failed=0,
            inference_cancelled=0,
            inference_in_flight=0,
            first_inference_started_at=None,
            last_inference_started_at=None,
            last_inference_completed_at=None,
            inference_busy_ms=0,
            inference_idle_ms=0,
        )
        observation = OllamaRuntimeTelemetryObservation(
            intent_sha256=intent.intent_sha256,
            provisioning_receipt_id=provisioning.receipt_id,
            lease_id=intent.lease_id,
            provision_attempt_id=intent.provision_attempt_id,
            resource_type=intent.resource_type,
            provider_resource_id=resource.provider_resource_id,
            snapshot=snapshot,
            snapshot_sha256=hashlib.sha256(
                json.dumps(
                    snapshot.to_dict(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            fetch_started_at=provisioning.readback_at,
            observed_at=provisioning.readback_at,
        )
        return OllamaReadinessObservation(
            provider_ready=self.provider_ready,
            route_url=self.route_url,
            provider_health_url=self.health_url,
            model_names=self.models,
            runtime_telemetry=observation,
        )

    async def pull_model(
        self, resource: ProvisionedOllamaResource, route_url: str, model: str
    ) -> None:
        del resource, route_url
        self.pull_calls += 1
        self.models = (model,)

    async def teardown(
        self,
        resource: ProvisionedOllamaResource,
        cleanup_intent: OllamaCleanupIntent,
    ) -> OllamaCleanupReceipt:
        del resource
        self.teardown_calls += 1
        if self.teardown_failures:
            self.teardown_failures -= 1
            raise OllamaLeaseTeardownError("temporary teardown remains retryable")
        return OllamaCleanupReceipt(
            cleanup_intent_id=cleanup_intent.cleanup_intent_id,
            verified_absent_at=max(
                cleanup_intent.delete_started_at,
                cleanup_intent.delete_started_at + timedelta(microseconds=1),
            ),
            basis=OllamaCleanupBasis.DELETE_ACKNOWLEDGED,
        )

    async def observe_lifecycle_billing(self, intent, provisioning, cleanup):
        del intent, provisioning, cleanup


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
    unit_rate = Fraction(Decimal(str(rate))) * gpu_count
    estimated_billable_seconds = round(
        Fraction(Decimal(str(estimated_cost))) * 3600 / unit_rate
    )
    maximum_billable_seconds = round(
        Fraction(Decimal(str(compute_ceiling))) * 3600 / unit_rate
    )

    # Re-derive the costs FROM the rounded seconds, in the same direction and
    # with the same formula the plan's own consistency check uses, so a rate
    # that does not divide evenly cannot make the fixture self-inconsistent.
    def _rate(seconds: int) -> float:
        return billing_exposure_ratio_float(
            Decimal(str(rate)), multiplier=seconds * gpu_count, divisor=3600
        )

    estimated_cost = _rate(estimated_billable_seconds)
    compute_ceiling = _rate(maximum_billable_seconds)
    estimated_total = billing_exposure_ratio_float(
        exact_decimal_sum(
            (Decimal(str(estimated_cost)), Decimal(str(estimated_non_compute_cost)))
        )
    )
    total_ceiling = billing_exposure_ratio_float(
        exact_decimal_sum(
            (Decimal(str(compute_ceiling)), Decimal(str(maximum_non_compute_cost)))
        )
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


def pod_plan(
    rate: float = 0.9,
    *,
    estimated_cost: float = 0.1,
    maximum_compute_cost: float | None = None,
    estimated_non_compute_cost: float = 0.0,
    maximum_non_compute_cost: float = 0.0,
    gpu_count: int = 1,
) -> OllamaPlacementPlan:
    serverless = serverless_plan(
        rate,
        estimated_cost=estimated_cost,
        maximum_compute_cost=maximum_compute_cost,
        estimated_non_compute_cost=estimated_non_compute_cost,
        maximum_non_compute_cost=maximum_non_compute_cost,
        gpu_count=gpu_count,
    )
    return replace(
        serverless,
        mode=OllamaLeaseMode.DEDICATED_POD,
        resource_type=OllamaResourceType.POD,
        placement=make_decision(
            ComputeProduct.POD,
            rate=rate,
            gpu_id="gpu-pod",
            pool=None,
            gpu_count=gpu_count,
        ),
        maximum_serverless_cold_starts=0,
    )
