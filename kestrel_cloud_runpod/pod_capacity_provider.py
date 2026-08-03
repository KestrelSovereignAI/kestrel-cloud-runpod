"""Runpod REST v2 adapter for the canonical durable Pod capacity lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol, TypeVar

from .models import (
    CloudType,
    ComputeProduct,
    GPUProfile,
    PlacementDecision,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
)
from .placement import select_gpu
from .pod_capacity_contracts import (
    PodBillingReceipt,
    PodCapacityQuote,
    PodCapacityQuoteRequest,
    PodCapacitySpec,
    PodRealizedPlacement,
    decimal_text,
    iso_datetime,
    pod_cost_usd,
)
from .providers import DirectRunPodProvider

_T = TypeVar("_T")


@dataclass(frozen=True)
class TrainingPodObservation:
    """Content-free capacity state needed by the lifecycle service."""

    provider_pod_id: str
    status: str
    backend_base_url: str | None
    raw: Mapping[str, Any]

    @property
    def is_running(self) -> bool:
        return self.status.upper() == "RUNNING"

    @property
    def is_stopped(self) -> bool:
        return self.status.upper() in {"EXITED", "STOPPED", "TERMINATED"}

    @property
    def is_failed(self) -> bool:
        return self.status.upper() in {"FAILED", "ERROR"}


@dataclass(frozen=True)
class CreatedTrainingPod:
    """Provider identity and live placement returned by a v2 Pod create."""

    provider_pod_id: str
    placement: PlacementDecision | None
    realized_placement: PodRealizedPlacement | None = None
    raw: Mapping[str, Any] | None = None


class PodCapacityCreatedMismatchError(RunPodManagerError):
    """A created Pod is known by ID but failed immutable validation."""

    def __init__(self, provider_pod_id: str) -> None:
        if not provider_pod_id.strip():
            raise ValueError("Created Pod mismatch requires a provider Pod ID")
        self.provider_pod_id = provider_pod_id
        super().__init__(
            "Runpod v2 created a Pod whose immutable placement did not match "
            "the accepted capacity request"
        )


class TrainingPodCapacityProvider(Protocol):
    """Capacity operations required by the durable training lifecycle."""

    async def observe(
        self, pod_id: str, *, profile: GPUProfile
    ) -> TrainingPodObservation: ...

    async def start(self, pod_id: str, *, gpu_count: int) -> None: ...

    async def create(
        self,
        *,
        profile: GPUProfile,
        resource_name: str,
        companion_id: str,
        environment: Mapping[str, str] | None = None,
        capacity_spec: PodCapacitySpec | None = None,
    ) -> CreatedTrainingPod: ...

    async def find_by_name(self, resource_name: str) -> str | None: ...

    async def stop(self, pod_id: str) -> bool: ...

    async def terminate(self, pod_id: str) -> bool: ...

    async def quote(self, request: PodCapacityQuoteRequest) -> PodCapacityQuote: ...

    async def find_exact(
        self, resource_name: str, capacity_spec: PodCapacitySpec
    ) -> CreatedTrainingPod | None: ...

    async def final_billing(
        self,
        pod_id: str,
        *,
        capacity_spec: PodCapacitySpec,
        created_at: datetime,
        terminated_at: datetime,
    ) -> PodBillingReceipt | None: ...


class RunpodPodCapacityProvider:
    """Expose only the v2 Pod operations the durable lifecycle requires."""

    def __init__(
        self,
        provider: DirectRunPodProvider,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self._clock = clock or (lambda: datetime.now(UTC))

    async def quote(self, request: PodCapacityQuoteRequest) -> PodCapacityQuote:
        """Map one live v2 catalog decision into the public neutral quote."""

        requirements = request.constraints.placement_requirements()
        offers = await asyncio.to_thread(
            self.provider.client.list_gpus,
            products=(ComputeProduct.POD,),
            count=requirements.gpu_count,
            cloud=requirements.cloud,
            min_cuda_version=requirements.min_cuda_version,
        )
        observed_at = self._now()
        placement = select_gpu(offers, requirements, observed_at=observed_at)
        hourly = Decimal(str(placement.offered_cost_per_hr))
        estimated_seconds = (
            request.estimated_startup_seconds + request.estimated_execution_seconds
        )
        estimated = pod_cost_usd(hourly, estimated_seconds, placement.gpu_count)
        ceiling = pod_cost_usd(
            hourly, request.maximum_runtime_seconds, placement.gpu_count
        )
        identity = {
            "capability_version": "runpod-pod-capacity-v1",
            "workload_kind": request.workload_kind,
            "parameters_sha256": request.parameters_sha256,
            "gpu_type_id": placement.gpu_id,
            "gpu_display_name": placement.gpu_name,
            "hourly_cost_usd": decimal_text(hourly),
            "estimated_cost_usd": decimal_text(estimated),
            "cost_ceiling_usd": decimal_text(ceiling),
            "estimated_startup_seconds": request.estimated_startup_seconds,
            "estimated_execution_seconds": request.estimated_execution_seconds,
            "maximum_runtime_seconds": request.maximum_runtime_seconds,
            "observed_at": iso_datetime(observed_at),
            "constraints": request.constraints.to_dict(),
        }
        provider_quote_id = (
            "runpod-pod:"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        return PodCapacityQuote(
            schema_version=3,
            capability_version="runpod-pod-capacity-v1",
            provider_quote_id=provider_quote_id,
            workload_kind=request.workload_kind,
            parameters_sha256=request.parameters_sha256,
            constraints=request.constraints,
            gpu_type_id=placement.gpu_id,
            gpu_display_name=placement.gpu_name,
            hourly_cost_usd=hourly,
            estimated_cost_usd=estimated,
            cost_ceiling_usd=ceiling,
            estimated_startup_seconds=request.estimated_startup_seconds,
            estimated_execution_seconds=request.estimated_execution_seconds,
            maximum_runtime_seconds=request.maximum_runtime_seconds,
            observed_at=observed_at,
            expires_at=observed_at + timedelta(seconds=request.quote_ttl_seconds),
            placement=placement,
        )

    async def observe(
        self, pod_id: str, *, profile: GPUProfile
    ) -> TrainingPodObservation:
        raw = await asyncio.to_thread(self.provider.get_status, pod_id)
        status = raw.get("status") or raw.get("desiredStatus")
        if not isinstance(status, str) or not status:
            raise RunPodManagerError("Runpod v2 Pod status response omitted status")
        return TrainingPodObservation(
            provider_pod_id=pod_id,
            status=status,
            backend_base_url=_pod_base_url(pod_id, raw, profile),
            raw=raw,
        )

    async def start(self, pod_id: str, *, gpu_count: int) -> None:
        await _mutation_to_thread(self.provider.resume_pod, pod_id, gpu_count)

    async def create(
        self,
        *,
        profile: GPUProfile,
        resource_name: str,
        companion_id: str,
        environment: Mapping[str, str] | None = None,
        capacity_spec: PodCapacitySpec | None = None,
    ) -> CreatedTrainingPod:
        effective_profile = profile
        purpose = "lora_training"
        if capacity_spec is not None:
            request = capacity_spec.request
            quote = request.quote
            if profile.network_volume_id or profile.volume_gb:
                raise RunPodManagerError(
                    "Catalog attempt Pods cannot use persistent or network volumes"
                )
            effective_profile = replace(
                profile,
                image_name=request.image_reference,
                min_vram_gb=quote.constraints.min_vram_gb,
                min_cuda_version=quote.constraints.min_cuda_version,
                max_cost_per_hr=float(quote.hourly_cost_usd),
                gpu_count=quote.constraints.gpu_count,
                cloud=quote.constraints.cloud,
                allowed_gpu_ids=(quote.gpu_type_id,),
                allowed_data_center_ids=quote.constraints.allowed_data_center_ids,
                env={},
            )
            purpose = request.workload_kind
        metadata: dict[str, Any] = {
            "name": resource_name,
            "companion_id": companion_id,
            "purpose": purpose,
        }
        if environment is not None:
            metadata["env_overrides"] = dict(environment)
        result = await _mutation_to_thread(
            self.provider.start_pod, effective_profile, metadata
        )
        pod_id = result.get("id") or result.get("podId")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodAmbiguousResultError(
                title="Runpod v2 create response was incomplete",
                detail="A successful Pod create response omitted the Pod ID",
                method="POST",
                resource="/pods",
            )
        placement = result.get("_kestrel_placement")
        if placement is not None and not isinstance(placement, PlacementDecision):
            raise RunPodAmbiguousResultError(
                title="Runpod v2 create response was incomplete",
                detail=(
                    "A successful Pod create response had invalid placement metadata"
                ),
                method="POST",
                resource="/pods",
            )
        realized: PodRealizedPlacement | None = None
        if capacity_spec is not None:
            try:
                _validate_realized_placement(placement, capacity_spec)
                _validate_recovery_payload(result, resource_name, capacity_spec)
                realized = _realized_placement(
                    result, pod_id, capacity_spec, self._now()
                )
            except RunPodManagerError as exc:
                raise PodCapacityCreatedMismatchError(pod_id) from exc
        return CreatedTrainingPod(
            provider_pod_id=pod_id,
            placement=placement,
            realized_placement=realized,
            raw=dict(result),
        )

    async def find_by_name(self, resource_name: str) -> str | None:
        pods = await asyncio.to_thread(self.provider.list_pods)
        matches = [pod for pod in pods if pod.get("name") == resource_name]
        if len(matches) > 1:
            raise RunPodManagerError(
                f"Multiple Runpod Pods match durable name '{resource_name}'"
            )
        if not matches:
            return None
        pod_id = matches[0].get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodManagerError("Runpod v2 Pod list item omitted its ID")
        return pod_id

    async def find_exact(
        self, resource_name: str, capacity_spec: PodCapacitySpec
    ) -> CreatedTrainingPod | None:
        """Recover only one Pod whose v2 shape matches the immutable request."""

        pods = await asyncio.to_thread(self.provider.list_pods)
        matches = [pod for pod in pods if pod.get("name") == resource_name]
        if not matches:
            return None
        if len(matches) > 1:
            raise RunPodManagerError(
                f"Multiple Runpod Pods match capacity identity '{resource_name}'"
            )
        match = matches[0]
        _validate_recovery_payload(match, resource_name, capacity_spec)
        pod_id = match.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise RunPodManagerError("Runpod v2 Pod list item omitted its ID")
        return CreatedTrainingPod(
            provider_pod_id=pod_id,
            placement=None,
            realized_placement=_realized_placement(
                match, pod_id, capacity_spec, self._now()
            ),
            raw=dict(match),
        )

    async def stop(self, pod_id: str) -> bool:
        """Request idempotent stop and report whether v2 confirms non-billing state."""

        try:
            result = await _mutation_to_thread(self.provider.stop_pod, pod_id)
        except RunPodAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        status = result.get("status") or result.get("desiredStatus")
        if isinstance(status, str) and status.upper() in {
            "EXITED",
            "STOPPED",
            "TERMINATED",
        }:
            return True
        try:
            raw = await asyncio.to_thread(self.provider.get_status, pod_id)
        except RunPodAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        observed_status = raw.get("status") or raw.get("desiredStatus")
        if not isinstance(observed_status, str) or not observed_status:
            raise RunPodManagerError("Runpod v2 Pod status response omitted status")
        return observed_status.upper() in {"EXITED", "STOPPED", "TERMINATED"}

    async def terminate(self, pod_id: str) -> bool:
        """Permanently terminate disposable capacity and confirm non-billing state."""

        try:
            result = await _mutation_to_thread(self.provider.terminate_pod, pod_id)
        except RunPodAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        status = result.get("status") or result.get("desiredStatus")
        if isinstance(status, str) and status.upper() == "TERMINATED":
            return True
        try:
            observed = await asyncio.to_thread(self.provider.get_status, pod_id)
        except RunPodAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        observed_status = observed.get("status") or observed.get("desiredStatus")
        return isinstance(observed_status, str) and observed_status.upper() in {
            "TERMINATED",
            "EXITED",
        }

    async def final_billing(
        self,
        pod_id: str,
        *,
        capacity_spec: PodCapacitySpec,
        created_at: datetime,
        terminated_at: datetime,
    ) -> PodBillingReceipt | None:
        """Return final v2 billing only after the requested interval is covered."""

        page = await asyncio.to_thread(
            self.provider.client.pod_billing,
            start_time=iso_datetime(created_at),
            end_time=iso_datetime(terminated_at),
            bucket_size="hour",
            pod_id=pod_id,
        )
        query = page.metadata.get("query")
        totals = page.metadata.get("totals")
        if not isinstance(query, Mapping) or not isinstance(totals, Mapping):
            raise RunPodManagerError("Runpod v2 Pod billing metadata is incomplete")
        if query.get("podId") != pod_id:
            raise RunPodManagerError("Runpod v2 Pod billing identity mismatch")
        start_time = _provider_datetime(
            query.get("startTime"), "billing query startTime"
        )
        end_time = _provider_datetime(query.get("endTime"), "billing query endTime")
        if start_time > created_at or end_time < terminated_at:
            raise RunPodManagerError(
                "Runpod v2 Pod billing query did not cover the requested interval"
            )
        if not page.records:
            return None
        for record in page.records:
            if record.get("podId") != pod_id:
                raise RunPodManagerError(
                    "Runpod v2 Pod billing record identity mismatch"
                )
        actual = _provider_decimal(totals.get("totalAmount"), "billing totalAmount")
        billed_from = min(
            _provider_datetime(record.get("startTime"), "billing startTime")
            for record in page.records
        )
        billed_until = max(
            _provider_datetime(record.get("endTime"), "billing endTime")
            for record in page.records
        )
        if billed_from > created_at or billed_until < terminated_at:
            return None
        billed_seconds = max(0, int((billed_until - billed_from).total_seconds()))
        billing_digest = hashlib.sha256(
            f"{pod_id}\0{iso_datetime(billed_from)}\0{iso_datetime(billed_until)}\0{actual}".encode()
        ).hexdigest()
        return PodBillingReceipt(
            provider_billing_id=f"runpod-billing:{billing_digest}",
            provider_pod_id=pod_id,
            billed_from=billed_from,
            billed_until=billed_until,
            billed_seconds=billed_seconds,
            # actual_cost_usd is the whole Pod's spend, so the price beside
            # it must be the whole Pod's hourly rate, not the per-GPU catalog
            # price. Pairing the two units made the evidence invariant
            # unsatisfiable for any multi-GPU Pod.
            hourly_price_usd=(
                capacity_spec.request.quote.hourly_cost_usd
                * Decimal(capacity_spec.request.quote.constraints.gpu_count)
            ),
            actual_cost_usd=actual,
            reconciled_at=self._now(),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Pod capacity provider clock must be timezone-aware")
        return value.astimezone(UTC)


def _pod_base_url(
    pod_id: str, payload: Mapping[str, Any], profile: GPUProfile
) -> str | None:
    """Resolve the workload route from the exact v2 runtime port response."""

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
                and int(private) == profile.inference_port
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            continue
        port_type = str(item.get("type", "")).lower()
        if port_type == "http":
            return f"https://{pod_id}-{profile.inference_port}.proxy.runpod.net"
        ip = item.get("ip")
        public = item.get("public") or item.get("publicPort")
        if (
            isinstance(ip, str)
            and isinstance(public, (str, int))
            and not isinstance(public, bool)
        ):
            protocol = profile.inference_protocol.strip().lower()
            if protocol not in {"http", "https"}:
                raise RunPodManagerError(
                    "Training profile inference_protocol must be http or https"
                )
            return f"{protocol}://{ip}:{int(public)}"
    return None


async def _mutation_to_thread(operation: Callable[..., _T], /, *args: Any) -> _T:
    """Do not propagate cancellation until an in-flight v2 mutation resolves."""

    task: asyncio.Task[_T] = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Threads cannot be cancelled. Waiting here closes the race where a
        # cleanup stop could run before a delayed create/start completed.
        await task
        raise


PodCapacityObservation = TrainingPodObservation
CreatedPodCapacity = CreatedTrainingPod
PodCapacityProvider = TrainingPodCapacityProvider
RunpodTrainingPodProvider = RunpodPodCapacityProvider


def _validate_realized_placement(
    placement: PlacementDecision | None, capacity_spec: PodCapacitySpec
) -> None:
    quote = capacity_spec.request.quote
    if placement is None:
        raise RunPodAmbiguousResultError(
            title="Runpod v2 create response was incomplete",
            detail="A successful Pod create response omitted placement evidence",
            method="POST",
            resource="/pods",
        )
    if (
        placement.gpu_id != quote.gpu_type_id
        or placement.gpu_count != quote.constraints.gpu_count
        or placement.cloud is not quote.constraints.cloud
        or Decimal(str(placement.offered_cost_per_hr)) > quote.hourly_cost_usd
    ):
        raise RunPodManagerError("Created Pod placement does not match accepted quote")


def _validate_recovery_payload(
    payload: Mapping[str, Any], resource_name: str, capacity_spec: PodCapacitySpec
) -> None:
    request = capacity_spec.request
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping):
        raise RunPodManagerError("Matching Runpod Pod omitted GPU placement")
    image = payload.get("image") or payload.get("imageName")
    raw_gpu_count = gpu.get("count")
    try:
        if isinstance(raw_gpu_count, bool) or not isinstance(raw_gpu_count, (int, str)):
            raise TypeError
        gpu_count = int(raw_gpu_count)
    except (TypeError, ValueError):
        gpu_count = 0
    expected = (
        payload.get("name") == resource_name
        and image == request.image_reference
        and gpu.get("id") == request.quote.gpu_type_id
        and gpu_count == request.quote.constraints.gpu_count
        and payload.get("cloud") == request.quote.constraints.cloud.value
    )
    if not expected:
        raise RunPodManagerError(
            "Runpod Pod with the deterministic name has mismatched immutable metadata"
        )


def _realized_placement(
    payload: Mapping[str, Any],
    pod_id: str,
    capacity_spec: PodCapacitySpec,
    observed_at: datetime,
) -> PodRealizedPlacement:
    """Project only provider placement fields needed for durable public evidence."""

    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping):
        raise RunPodManagerError("Runpod v2 Pod omitted realized GPU placement")
    raw_gpu_count = gpu.get("count")
    try:
        if isinstance(raw_gpu_count, bool) or not isinstance(raw_gpu_count, (int, str)):
            raise TypeError
        gpu_count = int(raw_gpu_count)
    except (TypeError, ValueError) as exc:
        raise RunPodManagerError("Runpod v2 Pod GPU count is invalid") from exc
    raw_data_center = payload.get("dataCenterId")
    if not isinstance(raw_data_center, str) or not raw_data_center.strip():
        raise RunPodManagerError("Runpod v2 Pod omitted realized data center")
    raw_cloud = payload.get("cloud")
    if not isinstance(raw_cloud, str):
        raise RunPodManagerError("Runpod v2 Pod omitted realized cloud")
    try:
        cloud = CloudType(raw_cloud.upper())
    except ValueError as exc:
        raise RunPodManagerError("Runpod v2 Pod cloud is invalid") from exc
    raw_rate = payload.get("cost")
    rate = _provider_decimal(raw_rate, "Pod hourly rate")
    quote = capacity_spec.request.quote
    try:
        # Construction validates too, so it belongs inside the converter. A
        # bare ValueError here escaped every handler in the reconcile chain
        # and exited the process, permanently skipping every other lease in
        # the pass - the same failure mode already closed on the Ollama side.
        realized = PodRealizedPlacement(
            provider_pod_id=pod_id,
            gpu_type_id=str(gpu.get("id", "")),
            gpu_display_name=quote.gpu_display_name,
            gpu_count=gpu_count,
            cloud=cloud,
            data_center_id=raw_data_center,
            hourly_rate_usd=rate,
            observed_at=observed_at,
        )
        realized.validate_against(quote)
    except ValueError as exc:
        raise RunPodManagerError(
            "Created Pod placement does not match accepted quote"
        ) from exc
    return realized


def _provider_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise RunPodManagerError(f"Runpod v2 {name} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RunPodManagerError(f"Runpod v2 {name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunPodManagerError(f"Runpod v2 {name} is not timezone-aware")
    return parsed.astimezone(UTC)


def _provider_decimal(value: Any, name: str) -> Decimal:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RunPodManagerError(f"Runpod v2 {name} is invalid")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed < 0:
        raise RunPodManagerError(f"Runpod v2 {name} is invalid")
    return parsed
