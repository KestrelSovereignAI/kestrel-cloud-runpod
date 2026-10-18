"""Narrow authenticated transport for one disposable catalog Pod attempt."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import SecretStr

from .models import RunPodManagerError
from .pod_capacity_contracts import CatalogPodWorkloadState


class CatalogPodTransportError(RunPodManagerError):
    """The worker transport failed without exposing private response content."""


class CatalogPodTransportConflictError(CatalogPodTransportError):
    """The attempt is already bound to a different canonical request."""


@dataclass(frozen=True)
class CatalogPodWorkloadObservation:
    """Content-free status from the single-attempt worker."""

    attempt_id: str
    request_sha256: str | None
    state: CatalogPodWorkloadState
    error_type: str | None
    result_available: bool
    result_sha256: str | None = None
    receipt_sha256: str | None = None
    acknowledgement_sha256: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.result_sha256,
            self.receipt_sha256,
            self.acknowledgement_sha256,
        ):
            if value is not None:
                try:
                    _sha256(value)
                except ValueError as exc:
                    raise CatalogPodTransportError(
                        "Catalog Pod status digest is invalid"
                    ) from exc
        if self.state in {
            CatalogPodWorkloadState.AWAITING_ACK,
            CatalogPodWorkloadState.ACKNOWLEDGED,
        } and (
            not self.result_available
            or self.request_sha256 is None
            or self.result_sha256 is None
            or self.receipt_sha256 is None
        ):
            raise CatalogPodTransportError(
                "Catalog Pod handoff status is missing result/receipt evidence"
            )
        if (
            self.state is CatalogPodWorkloadState.ACKNOWLEDGED
            and self.acknowledgement_sha256 is None
        ):
            raise CatalogPodTransportError(
                "Catalog Pod acknowledged status is missing ACK evidence"
            )


class CatalogPodWorkloadTransport:
    """Carry opaque catalog envelopes without importing their private schema."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Catalog Pod transport timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self.http_transport = http_transport

    async def health(self, base_url: str) -> None:
        response = await self._request("GET", base_url, "/health", bearer=None)
        if response.status_code != 200:
            raise CatalogPodTransportError(
                f"Catalog Pod health returned HTTP {response.status_code}"
            )
        payload = _json_object(response, "health")
        if payload != {"status": "ready"}:
            raise CatalogPodTransportError("Catalog Pod health response is invalid")

    async def submit(
        self,
        *,
        base_url: str,
        attempt_id: str,
        request_sha256: str,
        bearer_token: SecretStr,
        payload: Mapping[str, Any],
    ) -> CatalogPodWorkloadObservation:
        _attempt(attempt_id)
        _sha256(request_sha256)
        if payload.get("dispatch_attempt_id") != attempt_id:
            raise CatalogPodTransportConflictError(
                "Opaque catalog request attempt binding does not match the lease"
            )
        if payload.get("request_sha256") != request_sha256:
            raise CatalogPodTransportConflictError(
                "Opaque catalog request hash does not match the lease"
            )
        response = await self._request(
            "POST",
            base_url,
            _job_path(attempt_id),
            bearer=bearer_token,
            json_payload=payload,
        )
        if response.status_code == 409:
            raise CatalogPodTransportConflictError(
                "Catalog Pod attempt is bound to another request hash"
            )
        if response.status_code not in {200, 202}:
            raise CatalogPodTransportError(
                f"Catalog Pod submit returned HTTP {response.status_code}"
            )
        return _observation(response, attempt_id, request_sha256)

    async def status(
        self,
        *,
        base_url: str,
        attempt_id: str,
        request_sha256: str,
        bearer_token: SecretStr,
    ) -> CatalogPodWorkloadObservation:
        response = await self._request(
            "GET",
            base_url,
            _job_path(attempt_id),
            bearer=bearer_token,
        )
        if response.status_code != 200:
            raise CatalogPodTransportError(
                f"Catalog Pod status returned HTTP {response.status_code}"
            )
        return _observation(response, attempt_id, request_sha256)

    async def result(
        self,
        *,
        base_url: str,
        attempt_id: str,
        request_sha256: str,
        bearer_token: SecretStr,
        artifact_handoff_protocol_version: int | None = 1,
    ) -> Mapping[str, Any]:
        response = await self._request(
            "GET",
            base_url,
            _job_path(attempt_id) + "/result",
            bearer=bearer_token,
        )
        if response.status_code == 409:
            raise CatalogPodTransportConflictError(
                "Catalog Pod result is not available"
            )
        if response.status_code != 200:
            raise CatalogPodTransportError(
                f"Catalog Pod result returned HTTP {response.status_code}"
            )
        payload = _json_object(response, "result")
        if artifact_handoff_protocol_version is None:
            bindings = (payload,)
        elif (
            type(artifact_handoff_protocol_version) is int
            and artifact_handoff_protocol_version == 1
        ):
            if (
                type(payload.get("protocol_version")) is not int
                or payload["protocol_version"] != 1
            ):
                raise CatalogPodTransportError(
                    "Catalog Pod result protocol does not match the lease"
                )
            receipt, result = payload.get("receipt"), payload.get("result")
            if not isinstance(receipt, Mapping) or not isinstance(result, Mapping):
                raise CatalogPodTransportError(
                    "Catalog Pod result handoff envelope is invalid"
                )
            if (
                receipt.get("attempt_id") != attempt_id
                or receipt.get("request_sha256") != request_sha256
            ):
                raise CatalogPodTransportConflictError(
                    "Catalog Pod receipt does not match the lease"
                )
            bindings = (result,)
        else:
            raise CatalogPodTransportError("Catalog Pod result protocol is unsupported")
        if any(
            binding.get("dispatch_attempt_id") != attempt_id
            or binding.get("request_sha256") != request_sha256
            for binding in bindings
        ):
            raise CatalogPodTransportConflictError(
                "Catalog Pod result is not bound to the leased attempt"
            )
        return dict(payload)

    async def acknowledge(
        self,
        *,
        base_url: str,
        attempt_id: str,
        request_sha256: str,
        bearer_token: SecretStr,
        payload: Mapping[str, Any],
    ) -> CatalogPodWorkloadObservation:
        """Forward the private host's opaque signed ACK; never implement its HMAC."""

        if (
            payload.get("attempt_id") != attempt_id
            or payload.get("request_sha256") != request_sha256
        ):
            raise CatalogPodTransportConflictError(
                "Catalog Pod ACK does not match the lease"
            )
        response = await self._request(
            "POST",
            base_url,
            _job_path(attempt_id) + "/acknowledgement",
            bearer=bearer_token,
            json_payload=payload,
        )
        if response.status_code == 409:
            raise CatalogPodTransportConflictError(
                "Catalog Pod rejected the artifact acknowledgement"
            )
        if response.status_code != 200:
            raise CatalogPodTransportError(
                f"Catalog Pod acknowledgement returned HTTP {response.status_code}"
            )
        observation = _observation(response, attempt_id, request_sha256)
        if observation.state is not CatalogPodWorkloadState.ACKNOWLEDGED or any(
            getattr(observation, name) != payload.get(name)
            for name in ("result_sha256", "receipt_sha256", "acknowledgement_sha256")
        ):
            raise CatalogPodTransportConflictError(
                "Catalog Pod ACK observation does not match the exact host intent"
            )
        return observation

    async def cancel(
        self,
        *,
        base_url: str,
        attempt_id: str,
        request_sha256: str,
        bearer_token: SecretStr,
    ) -> CatalogPodWorkloadObservation:
        response = await self._request(
            "POST",
            base_url,
            _job_path(attempt_id) + "/cancel",
            bearer=bearer_token,
        )
        if response.status_code not in {200, 404}:
            raise CatalogPodTransportError(
                f"Catalog Pod cancel returned HTTP {response.status_code}"
            )
        if response.status_code == 404:
            return CatalogPodWorkloadObservation(
                attempt_id=attempt_id,
                request_sha256=None,
                state=CatalogPodWorkloadState.CANCELLED,
                error_type=None,
                result_available=False,
            )
        return _observation(response, attempt_id, request_sha256)

    async def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        bearer: SecretStr | None,
        json_payload: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        base = _tls_base_url(base_url)
        headers = {
            "Accept": "application/json",
            "Cache-Control": "no-store",
        }
        if bearer is not None:
            headers["Authorization"] = f"Bearer {bearer.get_secret_value()}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.http_transport,
                follow_redirects=False,
                headers=headers,
            ) as client:
                return await client.request(method, f"{base}{path}", json=json_payload)
        except httpx.TimeoutException as exc:
            raise CatalogPodTransportError(
                f"Catalog Pod {method.lower()} timed out"
            ) from exc
        except httpx.TransportError as exc:
            raise CatalogPodTransportError(
                f"Catalog Pod {method.lower()} transport failed"
            ) from exc


def _observation(
    response: httpx.Response, attempt_id: str, request_sha256: str
) -> CatalogPodWorkloadObservation:
    payload = _json_object(response, "status")
    if payload.get("attempt_id") != attempt_id:
        raise CatalogPodTransportConflictError(
            "Catalog Pod status attempt binding does not match the lease"
        )
    raw_hash = payload.get("request_sha256")
    if raw_hash is not None and raw_hash != request_sha256:
        raise CatalogPodTransportConflictError(
            "Catalog Pod status request hash does not match the lease"
        )
    try:
        state = CatalogPodWorkloadState(str(payload.get("state")))
    except ValueError as exc:
        raise CatalogPodTransportError("Catalog Pod status state is invalid") from exc
    raw_error = payload.get("error_type")
    if raw_error is not None and (
        not isinstance(raw_error, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", raw_error)
    ):
        raise CatalogPodTransportError("Catalog Pod status error type is invalid")
    result_available = payload.get("result_available")
    if not isinstance(result_available, bool):
        raise CatalogPodTransportError(
            "Catalog Pod status result availability is invalid"
        )
    return CatalogPodWorkloadObservation(
        attempt_id=attempt_id,
        request_sha256=raw_hash,
        state=state,
        error_type=raw_error,
        result_available=result_available,
        result_sha256=payload.get("result_sha256"),
        receipt_sha256=payload.get("receipt_sha256"),
        acknowledgement_sha256=payload.get("acknowledgement_sha256"),
    )


def _json_object(response: httpx.Response, operation: str) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise CatalogPodTransportError(
            f"Catalog Pod {operation} response is not JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CatalogPodTransportError(
            f"Catalog Pod {operation} response must be an object"
        )
    return payload


def _tls_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CatalogPodTransportError(
            "Catalog Pod workload route must be an origin-only HTTPS URL"
        )
    path = parsed.path.rstrip("/")
    return f"https://{parsed.netloc}{path}"


def _job_path(attempt_id: str) -> str:
    _attempt(attempt_id)
    return f"/v1/catalog/jobs/{quote(attempt_id, safe='')}"


def _attempt(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", value
    ):
        raise ValueError("Catalog Pod attempt_id is invalid")


def _sha256(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Catalog Pod request_sha256 is invalid")
