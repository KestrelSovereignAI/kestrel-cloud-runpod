"""CPU catalog facts and explicit Pod matching, never paid-create authority."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, InvalidOperation
from typing import Any

from .models import (
    Availability,
    CloudType,
    ComputeProduct,
    CpuPodCreateRequest,
    RunPodManagerError,
)


def validate_cpu_query(product: ComputeProduct, vcpu_count: int) -> None:
    if not isinstance(product, ComputeProduct):
        raise TypeError("CPU availability requires one explicit compute product")
    if type(vcpu_count) is not int or vcpu_count < 2 or vcpu_count & (vcpu_count - 1):
        raise ValueError("CPU availability requires a power-of-two vcpu_count >= 2")


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid catalog identifier")
    return value


def _number(value: Any, *, positive: bool = False) -> Decimal | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("invalid catalog number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("invalid catalog number") from None
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError("invalid catalog number")
    return number


def _availability(value: Any) -> Availability | None:
    return Availability(value) if value is not None else None


def _scale(number: Decimal, count: int) -> Decimal:
    # Provider facts must not change with another library's decimal context.
    context = Context(
        prec=max(28, len(number.as_tuple().digits) + len(str(count))),
        Emax=MAX_EMAX,
        Emin=MIN_EMIN,
    )
    return context.multiply(number, Decimal(count))


@dataclass(frozen=True)
class CpuDataCenter:
    id: str
    availability: Availability | None


@dataclass(frozen=True)
class CpuOffer:
    """Immutable facts; omitted prices or stock remain unknown."""

    id: str
    name: str
    group: str
    min_vcpu: int
    max_vcpu: int
    ram_gb_per_vcpu: Decimal
    secure_price_per_vcpu: Decimal | None
    serverless_price_per_vcpu: Decimal | None
    availability: Availability | None
    data_centers: tuple[CpuDataCenter, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CpuOffer":
        try:
            bounds = value["vcpu"]
            minimum, maximum = bounds["min"], bounds["max"]
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or minimum < 2
                or minimum > maximum
            ):
                raise ValueError("inverted CPU bounds")
            ram = _number(value["ramGbPerVcpu"], positive=True)
            if ram is None:
                raise ValueError("missing RAM ratio")
            price = value.get("price", {})
            if not isinstance(price, Mapping):
                raise TypeError("invalid prices")
            raw_centers = value.get("dataCenters", [])
            if not isinstance(raw_centers, list):
                raise TypeError("invalid datacenters")
            centers = tuple(
                CpuDataCenter(
                    _text(item["id"]), _availability(item.get("availability"))
                )
                for item in raw_centers
            )
            if len({item.id for item in centers}) != len(centers):
                raise ValueError("duplicate datacenters")
            return cls(
                id=_text(value["id"]),
                name=_text(value["name"]),
                group=_text(value["group"]),
                min_vcpu=minimum,
                max_vcpu=maximum,
                ram_gb_per_vcpu=ram,
                secure_price_per_vcpu=_number(price.get("securePerVcpu")),
                serverless_price_per_vcpu=_number(price.get("serverlessPerVcpu")),
                availability=_availability(value.get("availability")),
                data_centers=centers,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise RunPodManagerError("Invalid v2 CPU catalog response") from None


@dataclass(frozen=True)
class CpuPodCatalogMatch:
    """Read-only catalog match, not a reservation or an accepted spend quote."""

    cpu_id: str
    vcpu_count: int
    data_center_ids: tuple[str, ...]
    ram_gb: Decimal
    compute_cost_per_hr: Decimal
    observed_at: datetime


@dataclass(frozen=True)
class CpuCatalogSnapshot:
    """Bind availability to the exact single-product/count query and its time."""

    product: ComputeProduct
    vcpu_count: int
    observed_at: datetime
    offers: tuple[CpuOffer, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
        *,
        product: ComputeProduct,
        vcpu_count: int,
        observed_at: datetime,
    ) -> "CpuCatalogSnapshot":
        validate_cpu_query(product, vcpu_count)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("CPU catalog observation must be timezone-aware")
        raw = value.get("cpus")
        if not isinstance(raw, list) or not all(
            isinstance(item, Mapping) for item in raw
        ):
            raise RunPodManagerError("Invalid v2 CPU catalog response")
        offers = tuple(CpuOffer.from_dict(item) for item in raw)
        if len({item.id for item in offers}) != len(offers):
            raise RunPodManagerError("Duplicate v2 CPU catalog flavor")
        return cls(product, vcpu_count, observed_at.astimezone(UTC), offers)

    def match_pod(
        self,
        request: CpuPodCreateRequest,
        *,
        now: datetime,
        max_age: timedelta,
        minimum_availability: Availability = Availability.LOW,
    ) -> CpuPodCatalogMatch:
        """Refuse unknown/stale/mismatched facts; never issue a create or fallback.

        Compute rate excludes disk, network storage, transfer and other charges.
        Network-volume placement and accepted all-in spend need separate custody.
        """
        request.to_payload()
        if now.tzinfo is None or now.utcoffset() is None or max_age <= timedelta(0):
            raise ValueError(
                "CPU catalog matching requires aware time and positive max_age"
            )
        if (
            not isinstance(minimum_availability, Availability)
            or minimum_availability is Availability.NONE
        ):
            raise ValueError("CPU matching requires positive availability")
        age = now - self.observed_at
        if age < timedelta(0) or age > max_age:
            raise RunPodManagerError("CPU catalog snapshot is stale or from the future")
        if (
            self.product is not ComputeProduct.POD
            or self.vcpu_count != request.vcpu_count
        ):
            raise RunPodManagerError(
                "CPU catalog availability query does not match Pod request"
            )
        if request.cloud is not CloudType.SECURE:
            raise RunPodManagerError("CPU catalog provides no community Pod rate")
        offer = next((item for item in self.offers if item.id == request.cpu_id), None)
        if offer is None or not offer.min_vcpu <= request.vcpu_count <= offer.max_vcpu:
            raise RunPodManagerError("CPU flavor or count is not offered")
        ranks = {
            Availability.NONE: 0,
            Availability.LOW: 1,
            Availability.MEDIUM: 2,
            Availability.HIGH: 3,
        }
        floor = ranks[minimum_availability]
        if offer.availability is None or ranks[offer.availability] < floor:
            raise RunPodManagerError("CPU availability is unknown or insufficient")
        centers = {item.id: item.availability for item in offer.data_centers}
        if not request.data_center_ids or any(
            centers.get(item) is None or ranks[centers[item]] < floor
            for item in request.data_center_ids
        ):
            raise RunPodManagerError(
                "Every requested CPU datacenter must have sufficient known stock"
            )
        rate = offer.secure_price_per_vcpu
        if rate is None:
            raise RunPodManagerError("CPU Pod rate is unknown")
        return CpuPodCatalogMatch(
            offer.id,
            request.vcpu_count,
            tuple(request.data_center_ids),
            _scale(offer.ram_gb_per_vcpu, request.vcpu_count),
            _scale(rate, request.vcpu_count),
            self.observed_at,
        )
