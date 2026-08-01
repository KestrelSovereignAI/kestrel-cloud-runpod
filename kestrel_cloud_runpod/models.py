"""Typed contracts for the Runpod v2 control and data planes."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

CONTROL_PLANE_BASE_URL = "https://v2-rest.runpod.io/v2"
SERVERLESS_DATA_PLANE_BASE_URL = "https://api.runpod.ai/v2"
_URL_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization)(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


class RunPodManagerError(Exception):
    """Base exception for Runpod manager and provider failures."""


@dataclass(frozen=True)
class RateLimit:
    """Rate-limit information returned by either Runpod v2 service."""

    raw: Optional[str] = None
    policy: Optional[str] = None
    retry_after_seconds: Optional[float] = None


class RunPodAPIError(RunPodManagerError):
    """An actionable HTTP or RFC 9457 error returned by Runpod."""

    def __init__(
        self,
        *,
        title: str,
        detail: str,
        status_code: Optional[int] = None,
        errors: Sequence[str] = (),
        method: Optional[str] = None,
        resource: Optional[str] = None,
        rate_limit: Optional[RateLimit] = None,
    ) -> None:
        self.title = _redact_error_text(title)
        self.detail = _redact_error_text(detail)
        self.status_code = status_code
        self.errors = tuple(_redact_error_text(error) for error in errors)
        self.method = method
        self.resource = resource
        self.rate_limit = rate_limit
        context = " ".join(part for part in (method, resource) if part)
        status = f" ({status_code})" if status_code is not None else ""
        suffix = f" [{context}]" if context else ""
        validation = f" Validation: {'; '.join(self.errors)}" if self.errors else ""
        super().__init__(f"{self.title}{status}: {self.detail}{validation}{suffix}")


class RunPodAmbiguousResultError(RunPodAPIError):
    """A mutating request may have succeeded and must be reconciled first."""

    reconcile_required = True


class CloudType(str, Enum):
    SECURE = "SECURE"
    COMMUNITY = "COMMUNITY"


class ComputeProduct(str, Enum):
    POD = "POD"
    SERVERLESS = "SERVERLESS"


class Availability(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FlashBoot(str, Enum):
    """Runpod Serverless cold-start acceleration policy."""

    OFF = "OFF"
    FLASHBOOT = "FLASHBOOT"
    PRIORITY_FLASHBOOT = "PRIORITY_FLASHBOOT"


@dataclass(frozen=True)
class PlacementRequirements:
    """Stable workload constraints used to select a live catalog offer."""

    product: ComputeProduct
    min_vram_gb: int
    gpu_count: int = 1
    cloud: CloudType = CloudType.SECURE
    min_cuda_version: Optional[str] = None
    max_cost_per_hr: Optional[float] = None
    allowed_gpu_ids: tuple[str, ...] = ()
    allowed_gpu_pools: tuple[str, ...] = ()
    allowed_data_center_ids: tuple[str, ...] = ()
    minimum_availability: Availability = Availability.LOW
    benchmark_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.min_vram_gb < 1:
            raise ValueError("min_vram_gb must be at least 1")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        if self.max_cost_per_hr is not None and self.max_cost_per_hr <= 0:
            raise ValueError("max_cost_per_hr must be positive")
        if self.min_cuda_version is not None and not re.fullmatch(
            r"\d+(?:\.\d+)?", self.min_cuda_version
        ):
            raise ValueError("min_cuda_version must be a CUDA major or major.minor")


@dataclass(frozen=True)
class GPUOffer:
    """A GPU type, live price, and availability snapshot from the v2 catalog."""

    id: str
    name: str
    pool: Optional[str]
    manufacturer: str
    memory_gb: int
    secure: bool
    community: bool
    secure_price_per_hr: float
    community_price_per_hr: float
    secure_max_count: int
    community_max_count: int
    availability: Optional[Availability] = None
    data_centers: tuple[Mapping[str, Any], ...] = ()
    availability_min_cuda_version: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        availability_min_cuda_version: Optional[str] = None,
    ) -> "GPUOffer":
        price = _mapping(value, "price", "GPU offer")
        max_count = _mapping(value, "maxCount", "GPU offer")
        availability = value.get("availability")
        try:
            return cls(
                id=_string(value, "id", "GPU offer"),
                name=_string(value, "name", "GPU offer"),
                pool=_optional_string(value.get("pool"), "GPU offer.pool"),
                manufacturer=_string(value, "manufacturer", "GPU offer"),
                memory_gb=_integer(value, "memory", "GPU offer"),
                secure=_boolean(value, "secure", "GPU offer"),
                community=_boolean(value, "community", "GPU offer"),
                secure_price_per_hr=_number(price, "secure", "GPU offer.price"),
                community_price_per_hr=_number(price, "community", "GPU offer.price"),
                secure_max_count=_integer(max_count, "secure", "GPU offer.maxCount"),
                community_max_count=_integer(
                    max_count, "community", "GPU offer.maxCount"
                ),
                availability=(
                    Availability(availability) if availability is not None else None
                ),
                data_centers=tuple(
                    _mapping_item(item, "GPU offer.dataCenters")
                    for item in value.get("dataCenters", ())
                ),
                availability_min_cuda_version=availability_min_cuda_version,
            )
        except ValueError as exc:
            raise RunPodManagerError(
                f"Invalid GPU offer availability: {availability}"
            ) from exc

    def price_for(self, cloud: CloudType) -> float:
        return (
            self.secure_price_per_hr
            if cloud is CloudType.SECURE
            else self.community_price_per_hr
        )

    def max_count_for(self, cloud: CloudType) -> int:
        return (
            self.secure_max_count
            if cloud is CloudType.SECURE
            else self.community_max_count
        )

    def supports_cloud(self, cloud: CloudType) -> bool:
        return self.secure if cloud is CloudType.SECURE else self.community


@dataclass(frozen=True)
class PlacementDecision:
    """Auditable result of selecting one live catalog offer."""

    gpu_id: str
    gpu_pool: Optional[str]
    gpu_name: str
    memory_gb: int
    cloud: CloudType
    gpu_count: int
    offered_cost_per_hr: float
    availability: Optional[Availability]
    catalog_observed_at: datetime
    requirements: PlacementRequirements


@dataclass(frozen=True)
class PodCreateRequest:
    """Typed subset of the v2 CreatePodRequest used by Kestrel."""

    name: str
    image: str
    gpu_id: str
    gpu_count: int = 1
    cloud: CloudType = CloudType.SECURE
    disk_gb: int = 50
    ports: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    args: Optional[str] = None
    registry_id: Optional[str] = None
    data_center_ids: tuple[str, ...] = ()
    mounts: Optional[Mapping[str, Any]] = None
    global_networking: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.image or not self.gpu_id:
            raise ValueError("Pod name, image, and gpu_id are required")
        if self.gpu_count < 1 or self.disk_gb < 1:
            raise ValueError("Pod gpu_count and disk_gb must be positive")

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "gpu": {"id": self.gpu_id, "count": self.gpu_count},
            "cloud": self.cloud.value,
            "disk": self.disk_gb,
            "ports": list(self.ports),
            "env": dict(self.env),
            "globalNetworking": self.global_networking,
        }
        if self.args is not None:
            payload["args"] = self.args
        if self.registry_id is not None:
            payload["registry"] = self.registry_id
        if self.data_center_ids:
            payload["dataCenterIds"] = list(self.data_center_ids)
        if self.mounts is not None:
            payload["mounts"] = dict(self.mounts)
        return payload


@dataclass(frozen=True)
class EndpointCreateRequest:
    """Typed subset of the v2 CreateEndpointRequest used by Kestrel."""

    name: str
    image: str
    gpu_pools: tuple[str, ...]
    endpoint_type: str
    scaling: Mapping[str, Any]
    gpu_count: int = 1
    workers_min: int = 0
    workers_max: int = 3
    idle_timeout_seconds: Optional[int] = 10
    disk_gb: int = 20
    ports: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    args: Optional[str] = None
    registry_id: Optional[str] = None
    data_center_ids: tuple[str, ...] = ()
    network_volume_ids: tuple[str, ...] = ()
    execution_timeout_ms: int = 300_000
    flashboot: FlashBoot = FlashBoot.FLASHBOOT

    def __post_init__(self) -> None:
        if not self.name or not self.image or not self.gpu_pools:
            raise ValueError(
                "Endpoint name, image, and at least one GPU pool are required"
            )
        if self.endpoint_type not in {"QUEUE", "LOAD_BALANCER"}:
            raise ValueError("endpoint_type must be QUEUE or LOAD_BALANCER")
        scaling_type = self.scaling.get("type")
        if scaling_type not in {"QUEUE_DELAY", "REQUEST_COUNT"}:
            raise ValueError("scaling.type must be QUEUE_DELAY or REQUEST_COUNT")
        if self.endpoint_type == "LOAD_BALANCER" and scaling_type != "REQUEST_COUNT":
            raise ValueError("LOAD_BALANCER endpoints require REQUEST_COUNT scaling")
        if (
            self.endpoint_type == "QUEUE"
            and scaling_type == "REQUEST_COUNT"
            and self.idle_timeout_seconds is not None
        ):
            raise ValueError(
                "QUEUE endpoints using REQUEST_COUNT must omit idle_timeout_seconds"
            )
        if (
            self.gpu_count < 1
            or self.workers_min < 0
            or self.workers_max < self.workers_min
        ):
            raise ValueError("Endpoint GPU and worker counts are invalid")
        if not isinstance(self.flashboot, FlashBoot):
            raise ValueError("flashboot must be a FlashBoot enum value")

    def to_payload(self) -> Dict[str, Any]:
        workers: Dict[str, int] = {"min": self.workers_min, "max": self.workers_max}
        if self.idle_timeout_seconds is not None:
            workers["idleTimeout"] = self.idle_timeout_seconds
        payload: Dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "type": self.endpoint_type,
            "gpu": {"pools": list(self.gpu_pools), "count": self.gpu_count},
            "workers": workers,
            "scaling": dict(self.scaling),
            "disk": self.disk_gb,
            "ports": list(self.ports),
            "env": dict(self.env),
            "dataCenterIds": list(self.data_center_ids),
            "networkVolumes": list(self.network_volume_ids),
            "timeout": self.execution_timeout_ms,
            "flashboot": self.flashboot.value,
        }
        if self.args is not None:
            payload["args"] = self.args
        if self.registry_id is not None:
            payload["registry"] = self.registry_id
        return payload


@dataclass(frozen=True)
class EndpointUpdateRequest:
    """Typed PATCH fields supported by Kestrel's endpoint manager."""

    name: Optional[str] = None
    image: Optional[str] = None
    gpu_pools: Optional[tuple[str, ...]] = None
    gpu_count: Optional[int] = None
    workers_min: Optional[int] = None
    workers_max: Optional[int] = None
    idle_timeout_seconds: Optional[int] = None
    scaling: Optional[Mapping[str, Any]] = None
    data_center_ids: Optional[tuple[str, ...]] = None
    network_volume_ids: Optional[tuple[str, ...]] = None
    execution_timeout_ms: Optional[int] = None
    flashboot: Optional[FlashBoot] = None

    def __post_init__(self) -> None:
        if self.flashboot is not None and not isinstance(self.flashboot, FlashBoot):
            raise ValueError("flashboot must be a FlashBoot enum value")

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in (("name", self.name), ("image", self.image)):
            if value is not None:
                payload[key] = value
        if self.gpu_pools is not None:
            gpu: Dict[str, Any] = {"pools": list(self.gpu_pools)}
            if self.gpu_count is not None:
                gpu["count"] = self.gpu_count
            payload["gpu"] = gpu
        elif self.gpu_count is not None:
            raise ValueError("gpu_count requires gpu_pools in an endpoint update")
        workers = {
            key: value
            for key, value in (
                ("min", self.workers_min),
                ("max", self.workers_max),
                ("idleTimeout", self.idle_timeout_seconds),
            )
            if value is not None
        }
        if workers:
            payload["workers"] = workers
        if self.scaling is not None:
            payload["scaling"] = dict(self.scaling)
        if self.data_center_ids is not None:
            payload["dataCenterIds"] = list(self.data_center_ids)
        if self.network_volume_ids is not None:
            payload["networkVolumes"] = list(self.network_volume_ids)
        if self.execution_timeout_ms is not None:
            payload["timeout"] = self.execution_timeout_ms
        if self.flashboot is not None:
            payload["flashboot"] = self.flashboot.value
        if not payload:
            raise ValueError("Endpoint update cannot be empty")
        return payload


@dataclass(frozen=True)
class PodResource:
    id: str
    name: str
    status: str
    gpu_id: Optional[str]
    gpu_count: int
    cost_per_hr: float
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PodResource":
        gpu = value.get("gpu") or {}
        if not isinstance(gpu, Mapping):
            raise RunPodManagerError("Invalid v2 Pod.gpu response")
        return cls(
            id=_string(value, "id", "Pod"),
            name=_string(value, "name", "Pod"),
            status=_string(value, "status", "Pod"),
            gpu_id=_optional_string(gpu.get("id"), "Pod.gpu.id"),
            gpu_count=_optional_integer(gpu.get("count"), "Pod.gpu.count") or 0,
            cost_per_hr=_optional_number(value.get("cost"), "Pod.cost") or 0.0,
            raw=sanitize_resource_payload(value),
        )


@dataclass(frozen=True)
class EndpointResource:
    id: str
    name: str
    endpoint_type: Optional[str]
    request_urls: Mapping[str, str]
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointResource":
        urls = value.get("requestUrls") or {}
        if not isinstance(urls, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in urls.items()
        ):
            raise RunPodManagerError("Invalid v2 Endpoint.requestUrls response")
        return cls(
            id=_string(value, "id", "Endpoint"),
            name=_string(value, "name", "Endpoint"),
            endpoint_type=_optional_string(value.get("type"), "Endpoint.type"),
            request_urls=dict(urls),
            raw=sanitize_resource_payload(value),
        )


@dataclass(frozen=True)
class BillingPage:
    records: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BillingPage":
        records = value.get("records")
        metadata = value.get("metadata")
        if not isinstance(records, list) or not isinstance(metadata, Mapping):
            raise RunPodManagerError("Invalid v2 billing list envelope")
        return cls(
            records=tuple(_mapping_item(item, "billing.records") for item in records),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ServerlessJob:
    id: str
    status: str
    output: Any = None
    error: Optional[str] = None
    delay_time_ms: Optional[int] = None
    execution_time_ms: Optional[int] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ServerlessJob":
        return cls(
            id=_string(value, "id", "Serverless job"),
            status=_string(value, "status", "Serverless job"),
            output=value.get("output"),
            error=_optional_string(value.get("error"), "Serverless job.error"),
            delay_time_ms=_optional_integer(
                value.get("delayTime"), "Serverless job.delayTime"
            ),
            execution_time_ms=_optional_integer(
                value.get("executionTime"), "Serverless job.executionTime"
            ),
            raw=dict(value),
        )


def _mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise RunPodManagerError(f"Invalid {context}.{key} response")
    return item


def _redact_error_text(value: str) -> str:
    redacted = _URL_RE.sub("[REDACTED_URL]", value)
    redacted = _BEARER_RE.sub("Bearer [REDACTED]", redacted)
    return _SECRET_RE.sub(r"\1\2[REDACTED]", redacted)


def sanitize_resource_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy a provider resource while removing echoed credential values."""

    return {
        str(key): _sanitize_resource_value(str(key), item)
        for key, item in value.items()
    }


def _sanitize_resource_value(key: str, value: Any) -> Any:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    if normalized == "env" and isinstance(value, Mapping):
        return {str(env_key): "[REDACTED]" for env_key in value}
    if any(
        marker in normalized
        for marker in ("apikey", "authorization", "password", "secret", "token")
    ):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return sanitize_resource_payload(value)
    if isinstance(value, list):
        return [_sanitize_resource_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_resource_value(key, item) for item in value)
    return value


def _mapping_item(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunPodManagerError(f"Invalid {context} response")
    return dict(value)


def _string(value: Mapping[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RunPodManagerError(f"Invalid {context}.{key} response")
    return item


def _optional_string(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunPodManagerError(f"Invalid {context} response")
    return value


def _integer(value: Mapping[str, Any], key: str, context: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise RunPodManagerError(f"Invalid {context}.{key} response")
    return item


def _optional_integer(value: Any, context: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunPodManagerError(f"Invalid {context} response")
    return value


def _number(value: Mapping[str, Any], key: str, context: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise RunPodManagerError(f"Invalid {context}.{key} response")
    return float(item)


def _optional_number(value: Any, context: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RunPodManagerError(f"Invalid {context} response")
    return float(value)


def _boolean(value: Mapping[str, Any], key: str, context: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise RunPodManagerError(f"Invalid {context}.{key} response")
    return item


class PodStatus(Enum):
    """Lifecycle states returned by RunPod."""

    OFFLINE = "offline"
    PROVISIONING = "provisioning"
    LOADING = "loading"
    READY = "ready"
    TERMINATING = "terminating"
    ERROR = "error"


@dataclass
class GPUProfile:
    """Workload profile with placement constraints, not a hardcoded SKU."""

    id: str
    name: str
    task_type: str
    image_name: str
    container_disk_gb: int
    volume_gb: int
    ports: List[str]
    inference_port: int
    inference_protocol: str = "http"
    inference_base_path: str = "/v1"
    image_invoke_path: Optional[str] = None
    default_model: Optional[str] = None
    pod_type: Optional[str] = None
    min_vram_gb: int = 1
    min_cuda_version: Optional[str] = None
    max_cost_per_hr: Optional[float] = None
    gpu_count: int = 1
    cloud: CloudType = CloudType.SECURE
    allowed_gpu_ids: tuple[str, ...] = ()
    allowed_data_center_ids: tuple[str, ...] = ()
    max_context_window: Optional[int] = None
    readiness_timeout_seconds: Optional[int] = None
    registry_id: Optional[str] = None  # Runpod v2 registry credential ID
    network_volume_id: Optional[str] = None  # Network volume ID for persistent storage
    volume_mount_path: Optional[str] = (
        None  # Mount path for network volume (e.g., /workspace)
    )
    persistent_pod_id: Optional[str] = (
        None  # Use existing pod instead of creating new (resume/pause mode)
    )
    env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.container_disk_gb < 1:
            raise ValueError("container_disk_gb must be at least 1")
        if self.volume_gb != 0 and self.volume_gb < 10:
            raise ValueError("volume_gb must be 0 (disabled) or at least 10")
        if self.min_vram_gb < 1:
            raise ValueError("min_vram_gb must be at least 1")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        if self.max_cost_per_hr is not None and self.max_cost_per_hr <= 0:
            raise ValueError("max_cost_per_hr must be positive")


@dataclass
class RunPodSession:
    """Tracks the currently active RunPod session."""

    pod_id: str
    profile: GPUProfile
    task_profile: str
    model_name: str
    pod_type: Optional[str]
    status: PodStatus
    ttl_seconds: int
    started_at: datetime
    expires_at: datetime
    backend_base_url: Optional[str] = None
    inference_url: Optional[str] = None
    image_endpoint: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    gpu_type_id: Optional[str] = None
    cost_per_hr: Optional[float] = None
    placement: Optional[PlacementDecision] = None
    # Metering context (optional, for usage billing)
    companion_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "profile": self.profile.id,
            "task_profile": self.task_profile,
            "model_name": self.model_name,
            "status": self.status.value,
            "backend_base_url": self.backend_base_url,
            "inference_url": self.inference_url,
            "image_endpoint": self.image_endpoint,
            "ttl_seconds": self.ttl_seconds,
            "remaining_ttl_seconds": self.remaining_ttl_seconds,
            "gpu_type_id": self.gpu_type_id,
            "cost_per_hr": self.cost_per_hr,
            "vram_gb": self.placement.memory_gb if self.placement else None,
            "runtime": self.runtime,
        }

    @property
    def remaining_ttl_seconds(self) -> int:
        delta = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delta))

    @property
    def is_active(self) -> bool:
        return self.status not in {
            PodStatus.OFFLINE,
            PodStatus.TERMINATING,
            PodStatus.ERROR,
        }
