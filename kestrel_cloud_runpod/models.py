"""Typed contracts for the Runpod v2 control and data planes."""

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence

CONTROL_PLANE_BASE_URL = "https://api.runpod.io/v2"
SERVERLESS_DATA_PLANE_BASE_URL = "https://api.runpod.ai/v2"
_URL_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization)(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


class RunPodManagerError(Exception):
    """Base exception for Runpod manager and provider failures."""


class RunPodLegacyAdmissionRetiredError(RunPodManagerError):
    """Legacy session admission has no durable original-effect owner."""

    def __init__(self) -> None:
        super().__init__(
            "Legacy Runpod session admission is retired; use the inference-lease "
            "provider or durable Pod-capacity service"
        )


class RunPodMountPolicyMismatchError(RunPodManagerError):
    """A complete endpoint readback confirms storage outside policy."""


class ResourceFieldState(str, Enum):
    """Whether one optional provider-readback field is usable as evidence."""

    PRESENT = "present"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True)
class ResourceFieldEvidence:
    """Secret-safe field-local validity for a partially usable v2 response."""

    missing_fields: tuple[str, ...] = ()
    invalid_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = tuple(sorted(set(self.missing_fields)))
        invalid = tuple(sorted(set(self.invalid_fields)))
        if any(
            not isinstance(item, str) or not item for item in (*missing, *invalid)
        ) or set(missing).intersection(invalid):
            raise ValueError("Runpod resource field evidence is invalid")
        object.__setattr__(self, "missing_fields", missing)
        object.__setattr__(self, "invalid_fields", invalid)

    def state(self, field_path: str) -> ResourceFieldState:
        if field_path in self.invalid_fields:
            return ResourceFieldState.INVALID
        if field_path in self.missing_fields:
            return ResourceFieldState.MISSING
        return ResourceFieldState.PRESENT

    def require_valid(
        self, *, context: str, required_fields: Sequence[str] = ()
    ) -> None:
        unavailable = sorted(
            {
                *self.invalid_fields,
                *(field for field in required_fields if field in self.missing_fields),
            }
        )
        if unavailable:
            raise RunPodManagerError(
                f"Invalid {context} response fields: " + ", ".join(unavailable)
            )


def environment_values_sha256(environment: Mapping[str, str]) -> str:
    """Bind exact environment values without imposing a product's key policy."""

    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise TypeError("Runpod environment must be a string mapping")
    encoded = json.dumps(
        dict(sorted(environment.items())), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json_value(value: Any) -> Any:
    """Own and recursively freeze one JSON-compatible request value."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _mutable_json_value(value: Any) -> Any:
    """Return a fresh JSON shape from an immutable request snapshot."""

    if isinstance(value, Mapping):
        return {key: _mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json_value(item) for item in value]
    return value


def normalize_cuda_floor(value: Optional[str]) -> Optional[str]:
    """Represent a configured CUDA floor in the API's major.minor format."""

    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        raise ValueError("min_cuda_version must be a CUDA major or major.minor")
    major, _, minor = value.partition(".")
    return f"{int(major)}.{int(minor or '0')}"


@dataclass(frozen=True)
class EndpointMountAssessment:
    """Complete resource mount evidence, before outcome precedence is chosen."""

    missing_kinds: tuple[str, ...]
    invalid_kinds: tuple[str, ...]
    mismatched_kinds: tuple[str, ...]


def _mount_sequence(
    value: object, *, required_keys: frozenset[str]
) -> tuple[tuple[tuple[object, ...], ...], bool, bool]:
    """Return typed members, invalid evidence and whether evidence is complete."""

    if not isinstance(value, list):
        return (), True, False
    valid: list[tuple[object, ...]] = []
    invalid = False
    for mount in value:
        if not isinstance(mount, Mapping) or set(mount) != required_keys:
            invalid = True
            continue
        fields: list[object] = []
        for key in sorted(required_keys):
            field = mount.get(key)
            if not isinstance(field, str) or not field:
                invalid = True
                break
            fields.append(field)
        else:
            valid.append(tuple(fields))
    return tuple(valid), invalid, not invalid


def _persistent_mount(
    value: object,
) -> tuple[tuple[object, ...] | None, bool, bool]:
    if not isinstance(value, Mapping) or set(value) != {"path", "size"}:
        return None, True, False
    path = value.get("path")
    size = value.get("size")
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
    ):
        return None, True, False
    return (path, size), False, True


def _has_confirmed_foreign_mount_member(actual: object, accepted: object) -> bool:
    """Return whether partial sequence evidence proves an unaccepted member."""

    if not isinstance(actual, tuple) or not isinstance(accepted, tuple):
        return False
    remaining = list(accepted)
    for member in actual:
        try:
            remaining.remove(member)
        except ValueError:
            return True
    return False


def _assess_mounts(
    raw: Mapping[str, Any],
    *,
    expected: Mapping[str, object],
    parsers: Mapping[str, Callable[[object], tuple[object, bool, bool]]],
    required_kinds: frozenset[str],
) -> EndpointMountAssessment:
    """Assess every mount kind without inferring absence from malformed siblings."""

    if "mounts" not in raw or raw.get("mounts") is None:
        return EndpointMountAssessment(tuple(sorted(required_kinds)), (), ())
    mounts = raw.get("mounts")
    if not isinstance(mounts, Mapping):
        return EndpointMountAssessment((), tuple(sorted(parsers)), ())
    missing: list[str] = []
    invalid: list[str] = []
    mismatched: list[str] = []
    if set(mounts).difference(parsers):
        mismatched.append("unsupported")
    for kind, parser in parsers.items():
        if kind not in mounts:
            if kind in required_kinds:
                missing.append(kind)
            elif kind in expected:
                mismatched.append(kind)
            continue
        actual, actual_invalid, complete = parser(mounts[kind])
        if actual_invalid:
            invalid.append(kind)
        if kind not in expected:
            if complete or _has_confirmed_foreign_mount_member(actual, ()):
                mismatched.append(kind)
            continue
        accepted, accepted_invalid, accepted_complete = parser(expected[kind])
        if accepted_invalid or not accepted_complete:
            raise ValueError("Accepted mount policy is invalid")
        if (complete and actual != accepted) or (
            actual_invalid and _has_confirmed_foreign_mount_member(actual, accepted)
        ):
            mismatched.append(kind)
    return EndpointMountAssessment(tuple(missing), tuple(invalid), tuple(mismatched))


def assess_endpoint_mounts(
    raw: Mapping[str, Any], expected_network_volume_ids: Sequence[str]
) -> EndpointMountAssessment:
    """Assess network and global policy independently without short-circuiting."""

    return _assess_mounts(
        raw,
        expected={
            "network": [
                {"volumeId": volume_id} for volume_id in expected_network_volume_ids
            ],
            "global": [],
        },
        parsers={
            "network": lambda value: _mount_sequence(
                value, required_keys=frozenset({"volumeId"})
            ),
            "global": lambda value: _mount_sequence(
                value, required_keys=frozenset({"volumeId"})
            ),
        },
        required_kinds=frozenset({"network", "global"}),
    )


def assess_pod_mounts(
    raw: Mapping[str, Any], expected_mounts: Mapping[str, Any] | None
) -> EndpointMountAssessment:
    """Assess Pod network and persistent mounts as independent typed evidence."""

    if "mounts" not in raw or raw.get("mounts") is None:
        return EndpointMountAssessment(("mounts",), (), ())
    if not isinstance(raw.get("mounts"), Mapping):
        return EndpointMountAssessment((), ("mounts",), ())
    return _assess_mounts(
        raw,
        expected=expected_mounts or {},
        parsers={
            "network": lambda value: _mount_sequence(
                value, required_keys=frozenset({"path", "volumeId"})
            ),
            "persistent": _persistent_mount,
        },
        required_kinds=frozenset(),
    )


def validate_endpoint_mounts(
    raw: Mapping[str, Any], expected_network_volume_ids: Sequence[str]
) -> None:
    """Verify all endpoint storage kinds against the accepted storage policy."""

    assessment = assess_endpoint_mounts(raw, expected_network_volume_ids)
    if assessment.mismatched_kinds:
        if assessment.mismatched_kinds == ("global",):
            raise RunPodMountPolicyMismatchError(
                "Endpoint global mounts are outside the accepted policy"
            )
        raise RunPodMountPolicyMismatchError(
            "Endpoint mounts contain storage outside the accepted policy"
        )
    if assessment.invalid_kinds:
        raise RunPodManagerError("Endpoint mounts contain invalid policy evidence")
    if assessment.missing_kinds:
        raise RunPodManagerError("Endpoint mounts omit required policy evidence")


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


class RunPodAcknowledgedResourceError(RunPodAmbiguousResultError):
    """A mutation returned an exact resource with invalid sibling evidence."""

    def __init__(self, *, resource_id: str, resource_name: str) -> None:
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("Acknowledged resource ID must be nonempty")
        if not isinstance(resource_name, str) or not resource_name:
            raise ValueError("Acknowledged resource name must be nonempty")
        self.resource_id = resource_id
        self.resource_name = resource_name
        super().__init__(
            title="Runpod v2 create evidence is invalid",
            detail=(
                "The Pod create was acknowledged by exact ID, but its optional "
                "response evidence was invalid"
            ),
            method="POST",
            resource="/pods",
        )


class CloudType(str, Enum):
    SECURE = "SECURE"
    COMMUNITY = "COMMUNITY"

    @classmethod
    def from_config(cls, value: "str | CloudType") -> "CloudType":
        """Normalize operator configuration, not persisted resource authority."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("Cloud type must be SECURE or COMMUNITY")
        try:
            return cls(value.strip().upper())
        except ValueError as exc:
            raise ValueError("Cloud type must be SECURE or COMMUNITY") from exc


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


class VolumeType(str, Enum):
    """Runpod network-volume storage tier."""

    STANDARD = "STANDARD"
    HIGH_PERFORMANCE = "HIGH_PERFORMANCE"


class EndpointResourceVersion(Enum):
    """Version the saved create/readback authority independently of the API URL."""

    LEGACY = 1
    GA = 2


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
        normalize_cuda_floor(self.min_cuda_version)


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
    serverless_price_per_hr: Optional[float] = None

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
                serverless_price_per_hr=(
                    _number(price, "serverless", "GPU offer.price")
                    if "serverless" in price
                    else None
                ),
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

    def price_for(
        self, cloud: CloudType, product: ComputeProduct = ComputeProduct.POD
    ) -> float:
        """Return the per-GPU product rate; omitted product retains Pod API compatibility."""
        if product is ComputeProduct.SERVERLESS:
            if self.serverless_price_per_hr is None:
                raise RunPodManagerError("GPU offer has no Serverless pool price")
            return self.serverless_price_per_hr
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


@dataclass(frozen=True, kw_only=True)
class _PodContainerRequest:
    """One container envelope for both v2 Pod compute variants."""

    name: str
    image: str
    cloud: CloudType = CloudType.SECURE
    disk_gb: int = 50
    ports: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    args: Optional[str] = field(default=None, repr=False)
    registry_id: Optional[str] = None
    data_center_ids: tuple[str, ...] = ()
    mounts: Optional[Mapping[str, Any]] = None
    global_networking: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.image:
            raise ValueError("Pod name and image are required")
        if type(self.disk_gb) is not int or self.disk_gb < 1:
            raise ValueError("Pod disk_gb must be a positive integer")
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "data_center_ids", tuple(self.data_center_ids))
        object.__setattr__(self, "env", _freeze_json_value(self.env))
        if self.mounts is not None:
            object.__setattr__(self, "mounts", _freeze_json_value(self.mounts))

    def _container_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "cloud": self.cloud.value,
            "disk": self.disk_gb,
            "ports": list(self.ports),
            "env": _mutable_json_value(self.env),
            "globalNetworking": self.global_networking,
        }
        if self.args is not None:
            payload["args"] = self.args
        if self.registry_id is not None:
            payload["registry"] = self.registry_id
        if self.data_center_ids:
            payload["dataCenterIds"] = list(self.data_center_ids)
        if self.mounts is not None:
            payload["mounts"] = _mutable_json_value(self.mounts)
        return payload


@dataclass(frozen=True, kw_only=True)
class PodCreateRequest(_PodContainerRequest):
    """GPU Pod variant; CPU requests cannot carry these GPU constraints."""

    gpu_id: str
    gpu_count: int = 1
    min_cuda_version: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.gpu_id:
            raise ValueError("Pod gpu_id is required")
        if type(self.gpu_count) is not int or self.gpu_count < 1:
            raise ValueError("Pod gpu_count must be a positive integer")
        normalize_cuda_floor(self.min_cuda_version)

    def to_payload(self) -> Dict[str, Any]:
        payload = self._container_payload()
        payload["gpu"] = {"id": self.gpu_id, "count": self.gpu_count}
        if self.min_cuda_version is not None:
            payload["gpu"]["minCudaVersion"] = normalize_cuda_floor(
                self.min_cuda_version
            )
        return payload


@dataclass(frozen=True, kw_only=True)
class CpuPodCreateRequest(_PodContainerRequest):
    """CPU Pod wire request, not a capacity quote or visual capability grant.

    The caller still selects a currently offered flavor/count and owns paid
    resource custody. RAM is assigned by Runpod, never requested here. CPU Pods
    have no GPU/CUDA settings, persistent container mount or global networking.
    """

    cpu_id: str
    vcpu_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.cpu_id, str) or not self.cpu_id:
            raise ValueError("Pod cpu_id is required")
        count = self.vcpu_count
        if type(count) is not int or count < 2 or count & (count - 1):
            raise ValueError("Pod vcpu_count must be a power of two of at least 2")
        if self.global_networking is not False:
            raise ValueError("CPU Pods do not support global networking")
        if self.mounts is not None:
            if not isinstance(self.mounts, Mapping) or set(self.mounts) - {"network"}:
                raise ValueError("CPU Pods support only an explicit network mount")
            network = self.mounts.get("network", [])
            if not isinstance(network, (list, tuple)) or len(network) > 1:
                raise ValueError("CPU Pods support at most one network mount")
            for mount in network:
                if (
                    not isinstance(mount, Mapping)
                    or set(mount) != {"volumeId", "path"}
                    or not all(
                        isinstance(mount[key], str) and mount[key]
                        for key in ("volumeId", "path")
                    )
                ):
                    raise ValueError("CPU Pod network mount requires volumeId and path")
        super().__post_init__()

    def to_payload(self) -> Dict[str, Any]:
        payload = self._container_payload()
        payload["cpu"] = {"id": self.cpu_id, "vcpuCount": self.vcpu_count}
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
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    args: Optional[str] = None
    registry_id: Optional[str] = None
    data_center_ids: tuple[str, ...] = ()
    network_volume_ids: tuple[str, ...] = ()
    execution_timeout_ms: int = 300_000
    flashboot: FlashBoot = FlashBoot.FLASHBOOT
    min_cuda_version: Optional[str] = None
    resource_schema_version: EndpointResourceVersion = EndpointResourceVersion.GA

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
        normalize_cuda_floor(self.min_cuda_version)
        if not isinstance(self.resource_schema_version, EndpointResourceVersion):
            raise ValueError("Endpoint resource_schema_version is invalid")
        object.__setattr__(self, "gpu_pools", tuple(self.gpu_pools))
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "data_center_ids", tuple(self.data_center_ids))
        object.__setattr__(self, "network_volume_ids", tuple(self.network_volume_ids))
        object.__setattr__(self, "scaling", _freeze_json_value(self.scaling))
        object.__setattr__(self, "env", _freeze_json_value(self.env))

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
            "scaling": _mutable_json_value(self.scaling),
            "disk": self.disk_gb,
            "ports": list(self.ports),
            "env": _mutable_json_value(self.env),
            "dataCenterIds": list(self.data_center_ids),
            "networkVolumes": list(self.network_volume_ids),
            "timeout": self.execution_timeout_ms,
            "flashboot": self.flashboot.value,
        }
        if self.args is not None:
            payload["args"] = self.args
        if self.registry_id is not None:
            payload["registry"] = self.registry_id
        if self.resource_schema_version is EndpointResourceVersion.GA:
            payload["mounts"] = {
                "network": [
                    {"volumeId": volume_id} for volume_id in self.network_volume_ids
                ],
                "global": [],
            }
        if (
            self.resource_schema_version is EndpointResourceVersion.GA
            and self.min_cuda_version is not None
        ):
            payload["gpu"]["minCudaVersion"] = normalize_cuda_floor(
                self.min_cuda_version
            )
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


def _optional_evidence_value(
    source: Mapping[str, Any],
    key: str,
    field_path: str,
    parser: Any,
    missing: list[str],
    invalid: list[str],
) -> Any:
    """Parse one optional field without suppressing usable sibling evidence."""

    if key not in source or source[key] is None:
        missing.append(field_path)
        return None
    try:
        return parser(source[key])
    except (RunPodManagerError, TypeError, ValueError, OverflowError):
        invalid.append(field_path)
        return None


@dataclass(frozen=True)
class PodResource:
    id: str
    name: str
    status: str | None
    gpu_id: Optional[str]
    gpu_count: int
    cost_per_hr: Decimal | None
    raw: Mapping[str, Any]
    cloud: CloudType | None = None
    data_center_id: str | None = None
    cuda_version: str | None = None
    image: str | None = None
    environment_sha256: str | None = None
    created_at: datetime | None = None
    field_evidence: ResourceFieldEvidence = field(default_factory=ResourceFieldEvidence)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PodResource":
        missing: list[str] = []
        invalid: list[str] = []
        gpu_value = value.get("gpu")
        if gpu_value is None:
            missing.append("gpu")
            gpu: Mapping[str, Any] = {}
        elif not isinstance(gpu_value, Mapping):
            invalid.append("gpu")
            gpu = {}
        else:
            gpu = gpu_value

        gpu_id = _optional_evidence_value(
            gpu,
            "id",
            "gpu.id",
            lambda item: _optional_string(item, "Pod.gpu.id"),
            missing,
            invalid,
        )
        gpu_count = _optional_evidence_value(
            gpu,
            "count",
            "gpu.count",
            lambda item: _optional_integer(item, "Pod.gpu.count"),
            missing,
            invalid,
        )
        cost = _optional_evidence_value(
            value,
            "cost",
            "cost",
            lambda item: _optional_decimal(item, "Pod.cost"),
            missing,
            invalid,
        )
        raw_cloud = value.get("cloud")
        if raw_cloud is None:
            missing.append("cloud")
            cloud = None
        else:
            try:
                cloud = CloudType(raw_cloud)
            except (TypeError, ValueError):
                invalid.append("cloud")
                cloud = None
        environment_sha256 = _optional_evidence_value(
            value,
            "env",
            "env",
            environment_values_sha256,
            missing,
            invalid,
        )
        return cls(
            id=_string(value, "id", "Pod"),
            name=_string(value, "name", "Pod"),
            status=_optional_evidence_value(
                value,
                "status",
                "status",
                lambda item: _nonempty_response_string(item, "Pod.status"),
                missing,
                invalid,
            ),
            gpu_id=gpu_id,
            gpu_count=gpu_count or 0,
            cost_per_hr=cost,
            raw=sanitize_resource_payload(value),
            cloud=cloud,
            data_center_id=_optional_evidence_value(
                value,
                "dataCenterId",
                "dataCenterId",
                lambda item: _optional_string(item, "Pod.dataCenterId"),
                missing,
                invalid,
            ),
            cuda_version=_optional_evidence_value(
                value,
                "cudaVersion",
                "cudaVersion",
                lambda item: _optional_string(item, "Pod.cudaVersion"),
                missing,
                invalid,
            ),
            image=_optional_evidence_value(
                value,
                "image",
                "image",
                lambda item: _optional_string(item, "Pod.image"),
                missing,
                invalid,
            ),
            environment_sha256=environment_sha256,
            created_at=_optional_evidence_value(
                value,
                "createdAt",
                "createdAt",
                lambda item: _optional_response_datetime(item, "Pod.createdAt"),
                missing,
                invalid,
            ),
            field_evidence=ResourceFieldEvidence(tuple(missing), tuple(invalid)),
        )

    def require_valid_fields(self, *, context: str) -> "PodResource":
        self.field_evidence.require_valid(context=context)
        return self


@dataclass(frozen=True)
class EndpointResource:
    id: str
    name: str
    endpoint_type: Optional[str]
    request_urls: Mapping[str, str]
    raw: Mapping[str, Any]
    image: str | None = None
    environment_sha256: str | None = None
    created_at: datetime | None = None
    field_evidence: ResourceFieldEvidence = field(default_factory=ResourceFieldEvidence)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointResource":
        missing: list[str] = []
        invalid: list[str] = []
        urls = _optional_evidence_value(
            value,
            "requestUrls",
            "requestUrls",
            lambda item: _string_mapping(item, "Endpoint.requestUrls"),
            missing,
            invalid,
        )
        environment_sha256 = _optional_evidence_value(
            value,
            "env",
            "env",
            environment_values_sha256,
            missing,
            invalid,
        )
        return cls(
            id=_string(value, "id", "Endpoint"),
            name=_string(value, "name", "Endpoint"),
            endpoint_type=_optional_evidence_value(
                value,
                "type",
                "type",
                lambda item: _optional_string(item, "Endpoint.type"),
                missing,
                invalid,
            ),
            request_urls=dict(urls or {}),
            raw=sanitize_resource_payload(value),
            image=_optional_evidence_value(
                value,
                "image",
                "image",
                lambda item: _optional_string(item, "Endpoint.image"),
                missing,
                invalid,
            ),
            environment_sha256=environment_sha256,
            created_at=_optional_evidence_value(
                value,
                "createdAt",
                "createdAt",
                lambda item: _optional_response_datetime(item, "Endpoint.createdAt"),
                missing,
                invalid,
            ),
            field_evidence=ResourceFieldEvidence(tuple(missing), tuple(invalid)),
        )

    def require_valid_fields(self, *, context: str) -> "EndpointResource":
        self.field_evidence.require_valid(context=context)
        return self


@dataclass(frozen=True)
class NetworkVolume:
    """Read-only network-volume inventory used by cleanup verification."""

    id: str
    name: str
    size_gb: int
    data_center: str
    volume_type: VolumeType

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NetworkVolume":
        raw_type = _string(value, "type", "Network volume")
        try:
            volume_type = VolumeType(raw_type)
        except ValueError as exc:
            raise RunPodManagerError(
                f"Invalid network volume type: {raw_type}"
            ) from exc
        size_gb = _integer(value, "size", "Network volume")
        if not 10 <= size_gb <= 4096:
            raise RunPodManagerError("Invalid network volume size")
        return cls(
            id=_string(value, "id", "Network volume"),
            name=_string(value, "name", "Network volume"),
            size_gb=size_gb,
            data_center=_string(value, "dataCenter", "Network volume"),
            volume_type=volume_type,
        )


@dataclass(frozen=True)
class BillingPage:
    records: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BillingPage":
        if set(value) != {"records", "metadata"}:
            raise RunPodManagerError(
                "Invalid v2 billing envelope fields or unsupported pagination"
            )
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


def _nonempty_response_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunPodManagerError(f"Invalid {context} response")
    return value


def _string_mapping(value: Any, context: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RunPodManagerError(f"Invalid {context} response")
    return dict(value)


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


def _optional_decimal(value: Any, context: str) -> Decimal | None:
    """Preserve the JSON parser's numeric value; not the original wire lexeme."""

    if value is None:
        return None
    if not isinstance(value, (int, float, Decimal)) or isinstance(value, bool):
        raise RunPodManagerError(f"Invalid {context} response")
    try:
        parsed = Decimal(str(value))
        binary = float(value)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise RunPodManagerError(f"Invalid {context} response") from exc
    if (
        not parsed.is_finite()
        or parsed < 0
        or not math.isfinite(binary)
        or (parsed != 0 and binary == 0)
    ):
        raise RunPodManagerError(f"Invalid {context} response")
    return parsed


def _optional_response_datetime(value: Any, context: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunPodManagerError(f"Invalid {context} response")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RunPodManagerError(f"Invalid {context} response") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunPodManagerError(f"Invalid {context} response")
    return parsed.astimezone(UTC)


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
    training_cleanup_token: Optional[str] = None

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
            "training_cleanup_token": self.training_cleanup_token,
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
