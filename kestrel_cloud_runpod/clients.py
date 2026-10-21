"""Runpod v2 HTTP clients with one authenticated transport boundary."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional
from urllib.parse import quote, urlparse

import httpx

from .cpu_catalog import CpuCatalogSnapshot, validate_cpu_query
from .models import (
    CONTROL_PLANE_BASE_URL,
    SERVERLESS_DATA_PLANE_BASE_URL,
    BillingPage,
    CloudType,
    ComputeProduct,
    CpuPodCreateRequest,
    EndpointCreateRequest,
    EndpointResource,
    EndpointResourceVersion,
    EndpointUpdateRequest,
    GPUOffer,
    NetworkVolume,
    PodCreateRequest,
    PodResource,
    RateLimit,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
    ServerlessJob,
)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_RATE_LIMIT_RESET_RE = re.compile(r"(?:^|[,;]\s*)reset=(\d+(?:\.\d+)?)", re.IGNORECASE)
DEFAULT_USER_AGENT = "kestrel-cloud-runpod/v2"


def _is_retryable_runpod_status(status_code: int | None) -> bool:
    return status_code in _RETRYABLE_STATUSES


def is_retryable_runpod_api_error(error: RunPodAPIError) -> bool:
    """Classify one exact API outcome for durable reconciliation.

    A missing status is retryable only when the canonical client retained its
    original HTTPX failure as the explicit cause.  This prevents callers from
    manufacturing retry authority with a nullable status alone.  This result
    does not grant permission to repeat an arbitrary HTTP method.
    """

    if type(error) not in {RunPodAPIError, RunPodAmbiguousResultError}:
        return False
    if _is_retryable_runpod_status(error.status_code):
        return True
    return error.status_code is None and isinstance(error.__cause__, httpx.HTTPError)


class RunpodTransport:
    """Synchronous JSON transport shared by both Runpod v2 services."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        bearer_auth: bool,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_safe_retries: int = 2,
        user_agent: str = DEFAULT_USER_AGENT,
        sleep: Callable[[float], None] = time.sleep,
        http_transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise RunPodManagerError("A Runpod API key is required")
        if not user_agent.strip():
            raise ValueError("Runpod User-Agent cannot be empty")
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("Runpod transport timeouts must be positive")
        if not isinstance(max_safe_retries, int) or max_safe_retries < 0:
            raise ValueError("max_safe_retries must be a non-negative integer")
        self.base_url = _validate_v2_base_url(base_url)
        self.max_safe_retries = max_safe_retries
        self._sleep = sleep
        auth_value = f"Bearer {api_key}" if bearer_auth else api_key
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )
        self._client = httpx.Client(
            base_url=f"{self.base_url}/",
            headers={
                "Authorization": auth_value,
                "Accept": "application/json",
                "User-Agent": user_agent,
            },
            timeout=timeout,
            transport=http_transport,
        )
        self.last_rate_limit = RateLimit()
        self.last_sse_event_id: Optional[str] = None

    def close(self) -> None:
        self._client.close()

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        expected_statuses: Sequence[int] = (200,),
        ambiguous_on_failure: bool = False,
    ) -> Optional[Mapping[str, Any]]:
        response = self._request(
            method,
            path,
            params=params,
            json_body=json_body,
            expected_statuses=expected_statuses,
            ambiguous_on_failure=ambiguous_on_failure,
        )
        if response.status_code == 204 or not response.content:
            return None
        error_type = RunPodAmbiguousResultError if ambiguous_on_failure else RunPodAPIError
        try:
            payload = response.json()
        except (ValueError, RecursionError) as exc:
            raise error_type(
                title="Invalid Runpod response",
                detail="Expected a JSON object from the v2 API",
                status_code=response.status_code,
                method=method.upper(),
                resource=path,
                rate_limit=self.last_rate_limit,
            ) from exc
        if not isinstance(payload, Mapping):
            raise error_type(
                title="Invalid Runpod response",
                detail="Expected a JSON object from the v2 API",
                status_code=response.status_code,
                method=method.upper(),
                resource=path,
                rate_limit=self.last_rate_limit,
            )
        return payload

    def iter_sse(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        last_event_id: Optional[str] = None,
        stream_window_seconds: Optional[float] = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Yield parsed SSE data objects without buffering an unbounded stream."""

        if stream_window_seconds is not None and stream_window_seconds <= 0:
            raise ValueError("stream_window_seconds must be positive")
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        resource = path.lstrip("/")
        try:
            with self._client.stream(
                "GET", resource, params=params, headers=headers
            ) as response:
                self.last_rate_limit = _rate_limit_from_headers(response.headers)
                if response.status_code != 200:
                    response.read()
                    self._raise_response_error(response, "GET", path)
                deadline_expired = threading.Event()

                def close_at_deadline() -> None:
                    deadline_expired.set()
                    response.close()

                timer = None
                if stream_window_seconds is not None:
                    timer = threading.Timer(stream_window_seconds, close_at_deadline)
                    timer.daemon = True
                    timer.start()
                try:
                    data_lines: list[str] = []
                    event_id: Optional[str] = None
                    for line in response.iter_lines():
                        if not line:
                            if not data_lines:
                                continue
                            payload = self._parse_sse_data(
                                "\n".join(data_lines), path=path
                            )
                            if event_id is not None:
                                self.last_sse_event_id = event_id
                            yield payload
                            data_lines.clear()
                            event_id = None
                            continue
                        if line.startswith(":"):
                            continue
                        field, separator, raw_value = line.partition(":")
                        if not separator:
                            raw_value = ""
                        value = (
                            raw_value[1:] if raw_value.startswith(" ") else raw_value
                        )
                        if field == "data":
                            data_lines.append(value)
                        elif field == "id" and "\x00" not in value:
                            event_id = value
                    if data_lines:
                        payload = self._parse_sse_data("\n".join(data_lines), path=path)
                        if event_id is not None:
                            self.last_sse_event_id = event_id
                        yield payload
                except httpx.HTTPError:
                    if deadline_expired.is_set():
                        return
                    raise
                finally:
                    if timer is not None:
                        timer.cancel()
        except httpx.HTTPError as exc:
            raise RunPodAPIError(
                title="Runpod transport failed",
                detail=type(exc).__name__,
                method="GET",
                resource=path,
                rate_limit=self.last_rate_limit,
            ) from exc

    def _parse_sse_data(self, raw_data: str, *, path: str) -> Mapping[str, Any]:
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise RunPodAPIError(
                title="Invalid Runpod log event",
                detail="SSE data was not valid JSON",
                status_code=200,
                method="GET",
                resource=path,
                rate_limit=self.last_rate_limit,
            ) from exc
        if not isinstance(payload, Mapping):
            raise RunPodAPIError(
                title="Invalid Runpod log event",
                detail="SSE data was not a JSON object",
                status_code=200,
                method="GET",
                resource=path,
                rate_limit=self.last_rate_limit,
            )
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]],
        json_body: Optional[Mapping[str, Any]],
        expected_statuses: Sequence[int],
        ambiguous_on_failure: bool,
    ) -> httpx.Response:
        normalized_method = method.upper()
        safe_to_retry = normalized_method in _SAFE_METHODS
        resource = path.lstrip("/")
        attempt = 0
        while True:
            try:
                response = self._client.request(
                    normalized_method,
                    resource,
                    params=params,
                    json=json_body,
                )
            except httpx.HTTPError as exc:
                if safe_to_retry and attempt < self.max_safe_retries:
                    self._sleep(_fallback_backoff(attempt))
                    attempt += 1
                    continue
                error_type = (
                    RunPodAmbiguousResultError
                    if ambiguous_on_failure
                    else RunPodAPIError
                )
                raise error_type(
                    title="Runpod transport failed",
                    detail=type(exc).__name__,
                    method=normalized_method,
                    resource=path,
                    rate_limit=self.last_rate_limit,
                ) from exc

            self.last_rate_limit = _rate_limit_from_headers(response.headers)
            if response.status_code in expected_statuses:
                return response
            if (
                safe_to_retry
                and _is_retryable_runpod_status(response.status_code)
                and attempt < self.max_safe_retries
            ):
                self._sleep(_retry_delay(response.headers, attempt))
                attempt += 1
                continue
            if ambiguous_on_failure and (
                200 <= response.status_code < 300 or response.status_code >= 500
            ):
                problem = _problem_from_response(response)
                raise RunPodAmbiguousResultError(
                    title=problem["title"],
                    detail=problem["detail"],
                    status_code=response.status_code,
                    errors=problem["errors"],
                    method=normalized_method,
                    resource=path,
                    rate_limit=self.last_rate_limit,
                )
            self._raise_response_error(response, normalized_method, path)

    def _raise_response_error(
        self, response: httpx.Response, method: str, path: str
    ) -> None:
        problem = _problem_from_response(response)
        raise RunPodAPIError(
            title=problem["title"],
            detail=problem["detail"],
            status_code=response.status_code,
            errors=problem["errors"],
            method=method,
            resource=path,
            rate_limit=self.last_rate_limit,
        )


class RunpodControlPlaneClient:
    """Typed client for ``https://api.runpod.io/v2`` resources."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = CONTROL_PLANE_BASE_URL,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_safe_retries: int = 2,
        max_inventory_pages: int = 1000,
        user_agent: str = DEFAULT_USER_AGENT,
        sleep: Callable[[float], None] = time.sleep,
        http_transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if type(max_inventory_pages) is not int or max_inventory_pages < 1:
            raise ValueError("max_inventory_pages must be a positive integer")
        self.max_inventory_pages = max_inventory_pages
        self.transport = RunpodTransport(
            api_key=api_key,
            base_url=base_url,
            bearer_auth=True,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_safe_retries=max_safe_retries,
            user_agent=user_agent,
            sleep=sleep,
            http_transport=http_transport,
        )

    def close(self) -> None:
        self.transport.close()

    def list_gpus(
        self,
        *,
        include_availability: bool = True,
        products: Sequence[ComputeProduct] = (ComputeProduct.POD,),
        count: int = 1,
        cloud: CloudType = CloudType.SECURE,
        min_cuda_version: Optional[str] = None,
    ) -> tuple[GPUOffer, ...]:
        params: dict[str, Any] = {}
        if include_availability:
            if len(products) != 1:
                raise ValueError(
                    "Query exactly one product when requesting availability so the "
                    "returned stock signal remains product-specific"
                )
            params.update(
                {
                    "include": "AVAILABILITY",
                    "product": ",".join(product.value for product in products),
                    "count": count,
                    "cloud": cloud.value,
                }
            )
            if min_cuda_version:
                params["minCudaVersion"] = min_cuda_version
        payload = _required_payload(
            self.transport.request_json("GET", "/catalog/gpus", params=params)
        )
        return tuple(
            GPUOffer.from_dict(
                item,
                availability_min_cuda_version=(
                    min_cuda_version if include_availability else None
                ),
            )
            for item in _list_envelope(payload, "gpus")
        )

    def get_gpu(self, gpu_id: str) -> GPUOffer:
        payload = _required_payload(
            self.transport.request_json("GET", f"/catalog/gpus/{_segment(gpu_id)}")
        )
        return GPUOffer.from_dict(payload)

    def list_cpus(
        self, *, product: ComputeProduct, vcpu_count: int
    ) -> CpuCatalogSnapshot:
        """Read stock for one explicit product/count, preserving query context."""
        validate_cpu_query(product, vcpu_count)
        observed_at = datetime.now(timezone.utc)
        payload = _required_payload(
            self.transport.request_json(
                "GET",
                "/catalog/cpus",
                params={
                    "include": "AVAILABILITY",
                    "product": product.value,
                    "vcpuCount": vcpu_count,
                },
            )
        )
        return CpuCatalogSnapshot.from_response(
            payload, product=product, vcpu_count=vcpu_count, observed_at=observed_at
        )

    def create_pod(self, request: PodCreateRequest | CpuPodCreateRequest) -> PodResource:
        payload = self.transport.request_json(
            "POST",
            "/pods",
            json_body=request.to_payload(),
            expected_statuses=(201,),
            ambiguous_on_failure=True,
        )
        try:
            return PodResource.from_dict(_required_payload(payload))
        except (RunPodManagerError, RecursionError):
            # A malformed success receipt cannot turn a paid create into a
            # known failure. The caller retains custody and reconciles it.
            raise RunPodAmbiguousResultError(
                title="Unverified Pod creation response",
                detail="The successful create response was invalid; reconcile before retrying",
                status_code=201,
                method="POST",
                resource="/pods",
                rate_limit=self.transport.last_rate_limit,
            ) from None

    def get_pod(self, pod_id: str) -> PodResource:
        payload = _required_payload(
            self.transport.request_json("GET", f"/pods/{_segment(pod_id)}")
        )
        return PodResource.from_dict(payload)

    def list_pods(self) -> tuple[PodResource, ...]:
        return tuple(
            PodResource.from_dict(item)
            for item in self._list_resources(
                "/pods", "pods", params={"includeClusterPods": True}
            )
        )

    def list_network_volumes(self) -> tuple[NetworkVolume, ...]:
        """List account volumes for read-only cleanup and cost verification."""

        return tuple(
            NetworkVolume.from_dict(item)
            for item in self._list_resources("/network-volumes", "networkVolumes")
        )

    def pod_action(self, pod_id: str, action: str) -> Optional[PodResource]:
        if action not in {"start", "stop", "restart", "terminate"}:
            raise ValueError(f"Unsupported pod action: {action}")
        payload = self.transport.request_json(
            "POST",
            f"/pods/{_segment(pod_id)}/action",
            json_body={"action": action},
            expected_statuses=(200, 204),
            ambiguous_on_failure=True,
        )
        return PodResource.from_dict(payload) if payload is not None else None

    def delete_pod(self, pod_id: str) -> None:
        self.transport.request_json(
            "DELETE",
            f"/pods/{_segment(pod_id)}",
            expected_statuses=(204,),
        )

    def iter_pod_logs(
        self,
        pod_id: str,
        *,
        tail: int = 100,
        source: Optional[str] = None,
        since: Optional[str] = None,
        last_event_id: Optional[str] = None,
        stream_window_seconds: Optional[float] = None,
    ) -> Iterator[Mapping[str, Any]]:
        params: dict[str, Any] = {"tail": tail}
        if source:
            params["source"] = source
        if since:
            params["since"] = since
        return self.transport.iter_sse(
            f"/pods/{_segment(pod_id)}/logs",
            params=params,
            last_event_id=last_event_id,
            stream_window_seconds=stream_window_seconds,
        )

    def create_endpoint(self, request: EndpointCreateRequest) -> EndpointResource:
        if request.resource_schema_version is not EndpointResourceVersion.GA:
            raise RunPodManagerError(
                "Legacy endpoint authority must upgrade before new spend"
            )
        payload = self.transport.request_json(
            "POST",
            "/serverless",
            json_body=request.to_payload(),
            expected_statuses=(201,),
            ambiguous_on_failure=True,
        )
        try:
            return EndpointResource.from_dict(_required_payload(payload))
        except (RunPodManagerError, RecursionError):
            raise RunPodAmbiguousResultError(
                title="Unverified Endpoint creation response",
                detail=(
                    "The successful create response omitted a usable identity; "
                    "reconcile before retrying"
                ),
                status_code=201,
                method="POST",
                resource="/serverless",
                rate_limit=self.transport.last_rate_limit,
            ) from None

    def get_endpoint(self, endpoint_id: str) -> EndpointResource:
        payload = _required_payload(
            self.transport.request_json("GET", f"/serverless/{_segment(endpoint_id)}")
        )
        return EndpointResource.from_dict(payload)

    def list_endpoints(self) -> tuple[EndpointResource, ...]:
        return tuple(
            EndpointResource.from_dict(item)
            for item in self._list_resources("/serverless", "endpoints")
        )

    def _list_resources(
        self, path: str, key: str, *, params: Optional[Mapping[str, Any]] = None
    ) -> Iterator[Mapping[str, Any]]:
        """Walk the complete inventory; incomplete pages cannot prove cleanup."""

        query = dict(params or {})
        seen_cursors: set[str] = set()
        seen_ids: set[str] = set()
        pages = 0
        while True:
            if pages >= self.max_inventory_pages:
                raise RunPodManagerError(
                    "Runpod inventory exceeded max_inventory_pages; completeness is unknown"
                )
            payload = _required_payload(
                self.transport.request_json("GET", path, params=query)
            )
            pages += 1
            items = _list_envelope(payload, key)
            pagination = payload.get("pagination")
            if not isinstance(pagination, Mapping):
                raise RunPodManagerError("Invalid v2 pagination metadata")
            has_next = pagination.get("hasNextPage")
            if type(has_next) is not bool or "nextCursor" not in pagination:
                raise RunPodManagerError("Invalid v2 pagination metadata")
            cursor = pagination["nextCursor"]
            if has_next:
                if (
                    not isinstance(cursor, str)
                    or not cursor.strip()
                    or cursor in seen_cursors
                ):
                    raise RunPodManagerError("Invalid or repeated v2 pagination cursor")
                seen_cursors.add(cursor)
            elif cursor is not None:
                raise RunPodManagerError("Invalid final v2 pagination cursor")
            for item in items:
                resource_id = item.get("id")
                if not isinstance(resource_id, str) or not resource_id:
                    raise RunPodManagerError("Invalid v2 inventory resource identity")
                if resource_id in seen_ids:
                    raise RunPodManagerError("Duplicate v2 inventory resource identity")
                seen_ids.add(resource_id)
                yield item
            if not has_next:
                return
            query["cursor"] = cursor

    def update_endpoint(
        self, endpoint_id: str, changes: EndpointUpdateRequest
    ) -> EndpointResource:
        payload = _required_payload(
            self.transport.request_json(
                "PATCH",
                f"/serverless/{_segment(endpoint_id)}",
                json_body=changes.to_payload(),
            )
        )
        return EndpointResource.from_dict(payload)

    def delete_endpoint(self, endpoint_id: str) -> None:
        self.transport.request_json(
            "DELETE",
            f"/serverless/{_segment(endpoint_id)}",
            expected_statuses=(204,),
            ambiguous_on_failure=True,
        )

    def list_endpoint_workers(self, endpoint_id: str) -> Mapping[str, Any]:
        payload = _required_payload(
            self.transport.request_json(
                "GET", f"/serverless/{_segment(endpoint_id)}/workers"
            )
        )
        _list_envelope(payload, "workers")
        if not isinstance(payload.get("summary"), Mapping):
            raise RunPodManagerError("Invalid v2 workers summary envelope")
        return payload

    def iter_worker_logs(
        self,
        endpoint_id: str,
        worker_id: str,
        *,
        tail: int = 100,
        source: Optional[str] = None,
        since: Optional[str] = None,
        last_event_id: Optional[str] = None,
        stream_window_seconds: Optional[float] = None,
    ) -> Iterator[Mapping[str, Any]]:
        params: dict[str, Any] = {"tail": tail}
        if source:
            params["source"] = source
        if since:
            params["since"] = since
        return self.transport.iter_sse(
            f"/serverless/{_segment(endpoint_id)}/workers/{_segment(worker_id)}/logs",
            params=params,
            last_event_id=last_event_id,
            stream_window_seconds=stream_window_seconds,
        )

    def pod_billing(
        self,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        bucket_size: Optional[str] = None,
        last_n: Optional[int] = None,
        pod_id: Optional[str] = None,
    ) -> BillingPage:
        params = _billing_params(start_time, end_time, bucket_size, last_n)
        if pod_id:
            params["podId"] = pod_id
        payload = _required_payload(
            self.transport.request_json("GET", "/billing/pods", params=params)
        )
        return BillingPage.from_dict(payload)

    def serverless_billing(
        self,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        bucket_size: Optional[str] = None,
        last_n: Optional[int] = None,
        endpoint_id: Optional[str] = None,
    ) -> BillingPage:
        params = _billing_params(start_time, end_time, bucket_size, last_n)
        if endpoint_id:
            params["serverlessId"] = endpoint_id
        payload = _required_payload(
            self.transport.request_json("GET", "/billing/serverless", params=params)
        )
        return BillingPage.from_dict(payload)


class RunpodServerlessClient:
    """Typed client for queue jobs on ``https://api.runpod.ai/v2``."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = SERVERLESS_DATA_PLANE_BASE_URL,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_safe_retries: int = 2,
        user_agent: str = DEFAULT_USER_AGENT,
        sleep: Callable[[float], None] = time.sleep,
        http_transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.transport = RunpodTransport(
            api_key=api_key,
            base_url=base_url,
            bearer_auth=True,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_safe_retries=max_safe_retries,
            user_agent=user_agent,
            sleep=sleep,
            http_transport=http_transport,
        )

    def close(self) -> None:
        self.transport.close()

    def run(
        self,
        endpoint_id: str,
        input_data: Mapping[str, Any],
        *,
        webhook_url: Optional[str] = None,
        execution_timeout_ms: Optional[int] = None,
        ttl_ms: Optional[int] = None,
    ) -> ServerlessJob:
        body: dict[str, Any] = {"input": dict(input_data)}
        if webhook_url:
            body["webhook"] = webhook_url
        policy: dict[str, int] = {}
        if execution_timeout_ms is not None:
            policy["executionTimeout"] = execution_timeout_ms
        if ttl_ms is not None:
            policy["ttl"] = ttl_ms
        if policy:
            body["policy"] = policy
        payload = _required_payload(
            self.transport.request_json(
                "POST",
                f"/{_segment(endpoint_id)}/run",
                json_body=body,
                ambiguous_on_failure=True,
            )
        )
        return ServerlessJob.from_dict(payload)

    def status(
        self, endpoint_id: str, job_id: str, *, ttl_ms: Optional[int] = None
    ) -> ServerlessJob:
        params = {"ttl": ttl_ms} if ttl_ms is not None else None
        payload = _required_payload(
            self.transport.request_json(
                "GET",
                f"/{_segment(endpoint_id)}/status/{_segment(job_id)}",
                params=params,
            )
        )
        return ServerlessJob.from_dict(payload)

    def cancel(self, endpoint_id: str, job_id: str) -> ServerlessJob:
        payload = _required_payload(
            self.transport.request_json(
                "POST",
                f"/{_segment(endpoint_id)}/cancel/{_segment(job_id)}",
            )
        )
        return ServerlessJob.from_dict(payload)

    def retry(self, endpoint_id: str, job_id: str) -> ServerlessJob:
        payload = _required_payload(
            self.transport.request_json(
                "POST",
                f"/{_segment(endpoint_id)}/retry/{_segment(job_id)}",
            )
        )
        return ServerlessJob.from_dict(payload)

    def health(self, endpoint_id: str) -> Mapping[str, Any]:
        return _required_payload(
            self.transport.request_json("GET", f"/{_segment(endpoint_id)}/health")
        )


def _validate_v2_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Runpod base URL must be an absolute HTTP(S) URL")
    if not parsed.path.endswith("/v2"):
        raise ValueError(
            "Runpod base URL must end in /v2; v1 and GraphQL are unsupported"
        )
    return normalized


def _segment(value: str) -> str:
    if not value:
        raise ValueError("Runpod resource identifier cannot be empty")
    return quote(value, safe="")


def _required_payload(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if value is None:
        raise RunPodManagerError("Runpod v2 returned an empty response")
    return value


def _list_envelope(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    items = payload.get(key)
    if not isinstance(items, list):
        raise RunPodManagerError(f"Invalid v2 list envelope: expected '{key}'")
    result: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise RunPodManagerError(f"Invalid v2 list envelope item in '{key}'")
        result.append(item)
    return result


def _billing_params(
    start_time: Optional[str],
    end_time: Optional[str],
    bucket_size: Optional[str],
    last_n: Optional[int],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in (
        ("startTime", start_time),
        ("endTime", end_time),
        ("bucketSize", bucket_size),
        ("lastN", last_n),
    ):
        if value is not None:
            params[key] = value
    return params


def _problem_from_response(response: httpx.Response) -> dict[str, Any]:
    title = "Runpod API error"
    detail = f"Runpod returned HTTP {response.status_code}"
    errors: tuple[str, ...] = ()
    try:
        payload = response.json()
    except (ValueError, RecursionError):
        payload = None
    if isinstance(payload, Mapping):
        if isinstance(payload.get("title"), str):
            title = payload["title"]
        if isinstance(payload.get("detail"), str):
            detail = payload["detail"]
        raw_errors = payload.get("errors")
        if isinstance(raw_errors, list):
            errors = tuple(item for item in raw_errors if isinstance(item, str))
    return {"title": title, "detail": detail, "errors": errors}


def _rate_limit_from_headers(headers: httpx.Headers) -> RateLimit:
    return RateLimit(
        raw=headers.get("RateLimit"),
        policy=headers.get("RateLimit-Policy"),
        retry_after_seconds=_parse_retry_after(headers.get("Retry-After")),
    )


def _retry_delay(headers: httpx.Headers, attempt: int) -> float:
    retry_after = _parse_retry_after(headers.get("Retry-After"))
    if retry_after is not None:
        return retry_after
    raw_rate_limit = headers.get("RateLimit")
    if raw_rate_limit:
        match = _RATE_LIMIT_RESET_RE.search(raw_rate_limit)
        if match:
            return float(match.group(1))
    return _fallback_backoff(attempt)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _fallback_backoff(attempt: int) -> float:
    return min(0.5 * (2**attempt), 4.0)
