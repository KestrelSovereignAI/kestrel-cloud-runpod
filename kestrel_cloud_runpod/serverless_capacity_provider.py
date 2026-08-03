"""Runpod REST v2 adapter for finite Serverless quotes and billing."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from .clients import RunpodControlPlaneClient, RunpodServerlessClient
from .models import (
    Availability,
    BillingPage,
    ComputeProduct,
    EndpointResource,
    GPUOffer,
    RunPodAmbiguousResultError,
    RunPodManagerError,
    ServerlessJob,
)
from .placement import select_gpu
from .serverless_capacity_contracts import (
    SERVERLESS_CAPACITY_CONTRACT_VERSION,
    SERVERLESS_CAPACITY_SCHEMA_VERSION,
    PlannedServerlessCapacityQuote,
    PlannedServerlessCapacityQuoteRequest,
    ServerlessActivatedSubmission,
    ServerlessAmbiguousBillingWindow,
    ServerlessAmbiguousWindowBillingReceipt,
    ServerlessBillingAttempt,
    ServerlessBillingReceipt,
    ServerlessCapacityQuote,
    ServerlessCapacityQuoteRequest,
    ServerlessEndpointActivationReceipt,
    ServerlessEndpointCleanupReceipt,
    ServerlessEndpointHourCost,
    ServerlessEndpointProfile,
    ServerlessEndpointSpec,
    decimal_text,
    iso_datetime,
    json_sha256,
    parse_datetime,
    serverless_billing_hour_starts,
    serverless_worker_cost_usd,
)

_TERMINAL_JOB_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})
_BILLING_RECORD_KEYS = frozenset(
    {
        "startTime",
        "endTime",
        "serverlessId",
        "totalAmount",
        "gpuAmount",
        "cpuAmount",
        "diskAmount",
        "feeAmount",
    }
)
_BILLING_METADATA_KEYS = frozenset(
    {"query", "recordCount", "uniqueServerlessCount", "totals"}
)
_BILLING_QUERY_KEYS = frozenset({"startTime", "endTime", "bucketSize", "serverlessId"})
_BILLING_TOTAL_KEYS = frozenset(
    {"totalAmount", "gpuAmount", "cpuAmount", "diskAmount", "feeAmount"}
)
_WORKER_SUMMARY_KEYS = frozenset(
    {"running", "idle", "initializing", "throttled", "unhealthy", "total"}
)
_AVAILABILITY_RANK = {
    Availability.NONE.value: 0,
    Availability.LOW.value: 1,
    Availability.MEDIUM.value: 2,
    Availability.HIGH.value: 3,
}


class ServerlessEndpointActivationError(RunPodManagerError):
    """Endpoint activation failed; exact ledger cleanup identity is attached."""

    def __init__(
        self,
        message: str,
        *,
        endpoint_name: str,
        observed_endpoint_ids: tuple[str, ...] = (),
    ) -> None:
        self.endpoint_name = endpoint_name
        self.observed_endpoint_ids = observed_endpoint_ids
        super().__init__(message)


class ServerlessEndpointCleanupError(RunPodManagerError):
    """Exact activated endpoint could not be safely deleted or verified absent."""

    def __init__(
        self,
        message: str,
        *,
        endpoint_name: str,
        endpoint_id: str,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.endpoint_id = endpoint_id
        super().__init__(message)


@dataclass(frozen=True)
class _CatalogSelection:
    offer: GPUOffer
    observed_at: datetime
    gpu_id: str
    gpu_pool: str
    gpu_name: str
    vram_gb: int
    data_center_id: str
    availability: Availability
    hourly_worker_rate_usd: Decimal
    observation_sha256: str


class RunpodServerlessCapacityProvider:
    """Read-only v2 catalog/endpoint quote and billing reconciliation surface."""

    def __init__(
        self,
        *,
        control_client: RunpodControlPlaneClient,
        job_client: RunpodServerlessClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.control_client = control_client
        self.job_client = job_client
        self._clock = clock

    async def quote(
        self, request: ServerlessCapacityQuoteRequest
    ) -> ServerlessCapacityQuote:
        """Observe catalog capacity and endpoint configuration using GET only."""

        return await self._observe_quote(request)

    async def quote_planned_endpoint(
        self, request: PlannedServerlessCapacityQuoteRequest
    ) -> PlannedServerlessCapacityQuote:
        """Quote a pre-registered endpoint plan using GET-only catalog evidence."""

        return await self._observe_planned_quote(request)

    async def activate_planned_endpoint(
        self,
        request: PlannedServerlessCapacityQuoteRequest,
        quote: PlannedServerlessCapacityQuote,
        *,
        accepted_cost_ceiling_usd: Decimal,
        prior_receipt: ServerlessEndpointActivationReceipt | None = None,
    ) -> ServerlessEndpointActivationReceipt:
        """Create once after authorization, then validate exact REST v2 readback."""

        _validate_planned_request_binding(request, quote)
        if prior_receipt is not None:
            prior_receipt.validate_quote(quote)
            endpoint = await asyncio.to_thread(
                self.control_client.get_endpoint, prior_receipt.endpoint_id
            )
            readback_digest = _validate_planned_endpoint_binding(
                endpoint,
                request.endpoint,
                expected_endpoint_id=prior_receipt.endpoint_id,
                gpu_pool=quote.gpu_pool,
                data_center_id=quote.data_center_id,
            )
            if readback_digest != prior_receipt.endpoint_readback_sha256:
                raise ServerlessEndpointActivationError(
                    "Replayed Serverless activation readback drifted",
                    endpoint_name=quote.endpoint_name,
                    observed_endpoint_ids=(prior_receipt.endpoint_id,),
                )
            return prior_receipt

        now = self._now()
        quote.assert_fresh(now=now, accepted_cost_ceiling_usd=accepted_cost_ceiling_usd)
        current = await self._observe_planned_quote(request)
        _validate_planned_quote_drift(
            current,
            quote,
            accepted_cost_ceiling_usd=accepted_cost_ceiling_usd,
        )

        existing = await asyncio.to_thread(self.control_client.list_endpoints)
        named_before = tuple(
            item for item in existing if item.name == quote.endpoint_name
        )
        if named_before:
            raise ServerlessEndpointActivationError(
                "Planned Serverless endpoint name already exists before activation",
                endpoint_name=quote.endpoint_name,
                observed_endpoint_ids=tuple(item.id for item in named_before),
            )

        create_request = request.endpoint.spec.create_request(
            quote.endpoint_name,
            gpu_pool=quote.gpu_pool,
            data_center_id=quote.data_center_id,
        )
        if json_sha256(create_request.to_payload()) != quote.operation_digest:
            raise RunPodManagerError(
                "Planned Serverless create request does not match the accepted digest"
            )
        create_started_at = self._now()
        ambiguous = False
        try:
            created = await asyncio.to_thread(
                self.control_client.create_endpoint, create_request
            )
        except RunPodAmbiguousResultError as exc:
            ambiguous = True
            after = await asyncio.to_thread(self.control_client.list_endpoints)
            named_after = tuple(
                item for item in after if item.name == quote.endpoint_name
            )
            if len(named_after) != 1:
                raise ServerlessEndpointActivationError(
                    "Ambiguous Serverless create did not reconcile to exactly one "
                    "planned endpoint",
                    endpoint_name=quote.endpoint_name,
                    observed_endpoint_ids=tuple(item.id for item in named_after),
                ) from exc
            created = named_after[0]
        except RunPodManagerError as exc:
            raise ServerlessEndpointActivationError(
                "Planned Serverless endpoint create failed",
                endpoint_name=quote.endpoint_name,
            ) from exc

        activated_at = self._now()
        if created.name != quote.endpoint_name:
            raise ServerlessEndpointActivationError(
                "Serverless create returned a different endpoint identity",
                endpoint_name=quote.endpoint_name,
                observed_endpoint_ids=(created.id,),
            )
        try:
            endpoint = await asyncio.to_thread(
                self.control_client.get_endpoint, created.id
            )
            readback_digest = _validate_planned_endpoint_binding(
                endpoint,
                request.endpoint,
                expected_endpoint_id=created.id,
                gpu_pool=quote.gpu_pool,
                data_center_id=quote.data_center_id,
            )
        except RunPodManagerError as exc:
            raise ServerlessEndpointActivationError(
                "Created Serverless endpoint failed exact readback validation",
                endpoint_name=quote.endpoint_name,
                observed_endpoint_ids=(created.id,),
            ) from exc
        readback_at = self._now()
        activation_identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "provider_quote_id": quote.provider_quote_id,
            "quote_sha256": quote.quote_sha256,
            "plan_id": quote.plan_id,
            "plan_sha256": quote.plan_sha256,
            "endpoint_name": quote.endpoint_name,
            "endpoint_id": endpoint.id,
            "endpoint_spec_sha256": quote.endpoint_spec_sha256,
            "operation_digest": quote.operation_digest,
            "endpoint_readback_sha256": readback_digest,
            "create_started_at": iso_datetime(create_started_at),
            "activated_at": iso_datetime(activated_at),
            "readback_at": iso_datetime(readback_at),
            "reconciled_after_ambiguous_create": ambiguous,
        }
        return ServerlessEndpointActivationReceipt(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_activation_id=(
                "runpod-serverless-activation:" + json_sha256(activation_identity)
            ),
            provider_quote_id=quote.provider_quote_id,
            quote_sha256=quote.quote_sha256,
            plan_id=quote.plan_id,
            plan_sha256=quote.plan_sha256,
            endpoint_name=quote.endpoint_name,
            endpoint_id=endpoint.id,
            endpoint_spec_sha256=quote.endpoint_spec_sha256,
            operation_digest=quote.operation_digest,
            endpoint_readback_sha256=readback_digest,
            create_started_at=create_started_at,
            activated_at=activated_at,
            readback_at=readback_at,
            reconciled_after_ambiguous_create=ambiguous,
        )

    async def authorize_planned_submission(
        self,
        request: PlannedServerlessCapacityQuoteRequest,
        quote: PlannedServerlessCapacityQuote,
        activation: ServerlessEndpointActivationReceipt,
        *,
        accepted_cost_ceiling_usd: Decimal,
        exclusive_window_sha256: str,
    ) -> ServerlessActivatedSubmission:
        """Return the only provider-eligible child of a planned quote."""

        _validate_planned_request_binding(request, quote)
        activation.validate_quote(quote)
        now = self._now()
        quote.assert_fresh(now=now, accepted_cost_ceiling_usd=accepted_cost_ceiling_usd)
        current = await self._observe_planned_quote(request)
        _validate_planned_quote_drift(
            current,
            quote,
            accepted_cost_ceiling_usd=accepted_cost_ceiling_usd,
        )
        endpoint = await asyncio.to_thread(
            self.control_client.get_endpoint, activation.endpoint_id
        )
        readback_digest = _validate_planned_endpoint_binding(
            endpoint,
            request.endpoint,
            expected_endpoint_id=activation.endpoint_id,
            gpu_pool=quote.gpu_pool,
            data_center_id=quote.data_center_id,
        )
        if readback_digest != activation.endpoint_readback_sha256:
            raise RunPodManagerError(
                "Activated Serverless endpoint drifted before submission"
            )
        identity = {
            "provider_quote_id": quote.provider_quote_id,
            "quote_sha256": quote.quote_sha256,
            "provider_activation_id": activation.provider_activation_id,
            "endpoint_name": activation.endpoint_name,
            "endpoint_id": activation.endpoint_id,
            "endpoint_spec_sha256": activation.endpoint_spec_sha256,
            "operation_digest": activation.operation_digest,
            "exclusive_window_sha256": exclusive_window_sha256,
            "authorized_at": iso_datetime(now),
            "expires_at": iso_datetime(quote.expires_at),
        }
        return ServerlessActivatedSubmission(
            provider_submission_id=(
                "runpod-serverless-submission:" + json_sha256(identity)
            ),
            provider_quote_id=quote.provider_quote_id,
            quote_sha256=quote.quote_sha256,
            provider_activation_id=activation.provider_activation_id,
            endpoint_name=activation.endpoint_name,
            endpoint_id=activation.endpoint_id,
            endpoint_spec_sha256=activation.endpoint_spec_sha256,
            operation_digest=activation.operation_digest,
            exclusive_window_sha256=exclusive_window_sha256,
            authorized_at=now,
            expires_at=quote.expires_at,
        )

    async def cleanup_planned_endpoint(
        self,
        request: PlannedServerlessCapacityQuoteRequest,
        quote: PlannedServerlessCapacityQuote,
        activation: ServerlessEndpointActivationReceipt,
        *,
        prior_receipt: ServerlessEndpointCleanupReceipt | None = None,
    ) -> ServerlessEndpointCleanupReceipt:
        """Delete only the exact idle activated child and verify account absence."""

        _validate_planned_request_binding(request, quote)
        activation.validate_quote(quote)
        if prior_receipt is not None:
            prior_receipt.validate_activation(activation)
            resources = await asyncio.to_thread(self.control_client.list_endpoints)
            if any(
                item.id == activation.endpoint_id
                or item.name == activation.endpoint_name
                for item in resources
            ):
                raise ServerlessEndpointCleanupError(
                    "Replayed Serverless cleanup found the deleted endpoint identity",
                    endpoint_name=activation.endpoint_name,
                    endpoint_id=activation.endpoint_id,
                )
            return prior_receipt

        endpoint = await asyncio.to_thread(
            self.control_client.get_endpoint, activation.endpoint_id
        )
        readback_digest = _validate_planned_endpoint_binding(
            endpoint,
            request.endpoint,
            expected_endpoint_id=activation.endpoint_id,
            gpu_pool=quote.gpu_pool,
            data_center_id=quote.data_center_id,
        )
        if readback_digest != activation.endpoint_readback_sha256:
            raise ServerlessEndpointCleanupError(
                "Serverless endpoint drifted before cleanup",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            )
        worker_page = await asyncio.to_thread(
            self.control_client.list_endpoint_workers, activation.endpoint_id
        )
        workers = worker_page.get("workers")
        summary = worker_page.get("summary")
        if not isinstance(workers, list) or not isinstance(summary, Mapping):
            raise ServerlessEndpointCleanupError(
                "Serverless worker inventory is invalid before cleanup",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            )
        if set(summary) != _WORKER_SUMMARY_KEYS:
            raise ServerlessEndpointCleanupError(
                "Serverless worker summary is invalid before cleanup",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            )
        try:
            workers_are_zero = all(
                _integer(summary.get(key), f"workers summary.{key}") == 0
                for key in _WORKER_SUMMARY_KEYS
            )
        except RunPodManagerError as exc:
            raise ServerlessEndpointCleanupError(
                "Serverless worker summary is invalid before cleanup",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            ) from exc
        if workers or not workers_are_zero:
            raise ServerlessEndpointCleanupError(
                "Serverless endpoint still has active workers before cleanup",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            )
        workers_zero_observed_at = self._now()
        delete_started_at = self._now()
        reconciled = False
        try:
            await asyncio.to_thread(
                self.control_client.delete_endpoint, activation.endpoint_id
            )
        except RunPodAmbiguousResultError:
            reconciled = True
        except RunPodManagerError as exc:
            raise ServerlessEndpointCleanupError(
                "Serverless endpoint delete failed before absence was ambiguous",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            ) from exc
        resources = await asyncio.to_thread(self.control_client.list_endpoints)
        if any(
            item.id == activation.endpoint_id or item.name == activation.endpoint_name
            for item in resources
        ):
            raise ServerlessEndpointCleanupError(
                "Serverless endpoint remains present after cleanup",
                endpoint_name=activation.endpoint_name,
                endpoint_id=activation.endpoint_id,
            )
        verified_absent_at = self._now()
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "provider_quote_id": quote.provider_quote_id,
            "quote_sha256": quote.quote_sha256,
            "provider_activation_id": activation.provider_activation_id,
            "endpoint_name": activation.endpoint_name,
            "endpoint_id": activation.endpoint_id,
            "endpoint_spec_sha256": activation.endpoint_spec_sha256,
            "operation_digest": activation.operation_digest,
            "endpoint_readback_sha256": activation.endpoint_readback_sha256,
            "workers_zero_observed_at": iso_datetime(workers_zero_observed_at),
            "delete_started_at": iso_datetime(delete_started_at),
            "verified_absent_at": iso_datetime(verified_absent_at),
            "reconciled_after_delete_error": reconciled,
        }
        return ServerlessEndpointCleanupReceipt(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_cleanup_id=("runpod-serverless-cleanup:" + json_sha256(identity)),
            provider_quote_id=quote.provider_quote_id,
            quote_sha256=quote.quote_sha256,
            provider_activation_id=activation.provider_activation_id,
            endpoint_name=activation.endpoint_name,
            endpoint_id=activation.endpoint_id,
            endpoint_spec_sha256=activation.endpoint_spec_sha256,
            operation_digest=activation.operation_digest,
            endpoint_readback_sha256=activation.endpoint_readback_sha256,
            workers_zero_observed_at=workers_zero_observed_at,
            delete_started_at=delete_started_at,
            verified_absent_at=verified_absent_at,
            reconciled_after_delete_error=reconciled,
        )

    async def validate_quote_for_submission(
        self,
        request: ServerlessCapacityQuoteRequest,
        quote: ServerlessCapacityQuote,
        *,
        accepted_cost_ceiling_usd: Decimal,
    ) -> ServerlessCapacityQuote:
        """Re-read capacity immediately before submission and reject drift."""

        now = self._now()
        quote.assert_fresh(now=now, accepted_cost_ceiling_usd=accepted_cost_ceiling_usd)
        _validate_request_binding(request, quote)
        current = await self._observe_quote(request)
        stable_dimensions = (
            "profile_id",
            "endpoint_profile_sha256",
            "endpoint_id",
            "gpu_id",
            "gpu_pool",
            "vram_gb",
            "data_center_id",
            "cloud",
            "gpu_count",
            "min_cuda_version",
            "benchmark_id",
        )
        if any(
            getattr(current, name) != getattr(quote, name) for name in stable_dimensions
        ):
            raise RunPodManagerError(
                "Serverless endpoint or live placement drifted after quote acceptance"
            )
        if current.hourly_worker_rate_usd > quote.hourly_worker_rate_usd:
            raise RunPodManagerError(
                "Serverless worker rate increased after quote acceptance"
            )
        if current.cost_ceiling_usd > accepted_cost_ceiling_usd:
            raise RunPodManagerError(
                "Current Serverless cost exceeds the accepted ceiling"
            )
        return quote

    async def final_billing(
        self,
        attempt: ServerlessBillingAttempt,
        quote: ServerlessCapacityQuote,
    ) -> ServerlessBillingReceipt | None:
        """Return complete endpoint-window billing or ``None`` while it settles.

        REST v2 currently aggregates Serverless billing by endpoint and hour.
        ``exclusive_window_sha256`` is therefore an external durable proof that
        no other job shares the resolved endpoint window.  This provider binds
        that proof to the exact terminal job but does not own queue/job state.
        """

        attempt.validate_quote(quote)
        return await self._final_billing_bound(
            attempt,
            quote,
            endpoint_profile_sha256=quote.endpoint_profile_sha256,
        )

    async def final_activated_billing(
        self,
        attempt: ServerlessBillingAttempt,
        quote: PlannedServerlessCapacityQuote,
        activation: ServerlessEndpointActivationReceipt,
        submission: ServerlessActivatedSubmission,
    ) -> ServerlessBillingReceipt | None:
        """Settle only the exact activated child and exclusive submission window."""

        submission.validate(quote, activation)
        if (
            attempt.endpoint_id != activation.endpoint_id
            or attempt.provider_quote_id != quote.provider_quote_id
            or attempt.exclusive_window_sha256 != submission.exclusive_window_sha256
        ):
            raise RunPodManagerError(
                "Serverless billing attempt is not bound to the activated submission"
            )
        if not submission.authorized_at <= attempt.submitted_at < submission.expires_at:
            raise RunPodManagerError(
                "Serverless activated submission falls outside its authorization"
            )
        maximum_elapsed = (
            quote.maximum_queue_delay_seconds
            + quote.maximum_worker_start_seconds
            + quote.maximum_execution_seconds
        )
        if (
            attempt.completed_at - attempt.submitted_at
        ).total_seconds() > maximum_elapsed:
            raise RunPodManagerError(
                "Serverless activated attempt exceeds the accepted maximum interval"
            )
        coverage_until = attempt.completed_at + timedelta(
            seconds=quote.idle_tail_seconds
        )
        expected_hours = tuple(
            value.astimezone(UTC) for value in attempt.exclusive_billing_hour_starts
        )
        required_hours = serverless_billing_hour_starts(
            attempt.submitted_at, coverage_until
        )
        if (
            not expected_hours
            or expected_hours[0] > required_hours[0]
            or expected_hours[-1] < required_hours[-1]
        ):
            raise RunPodManagerError(
                "Serverless activated billing allocation does not cover its window"
            )
        return await self._final_billing_bound(
            attempt,
            quote,
            endpoint_profile_sha256=quote.endpoint_spec_sha256,
        )

    async def _final_billing_bound(
        self,
        attempt: ServerlessBillingAttempt,
        quote: ServerlessCapacityQuote | PlannedServerlessCapacityQuote,
        *,
        endpoint_profile_sha256: str,
    ) -> ServerlessBillingReceipt | None:
        if self.job_client is None:
            raise RunPodManagerError(
                "Serverless billing requires a restricted job-status client"
            )
        now = self._now()
        if now < attempt.completed_at:
            raise RunPodManagerError(
                "Serverless billing cannot be reconciled before attempt completion"
            )
        job = await asyncio.to_thread(
            self.job_client.status, attempt.endpoint_id, attempt.job_id
        )
        _validate_terminal_job(job, attempt)
        if (
            job.delay_time_ms is not None
            and job.delay_time_ms
            > (quote.maximum_queue_delay_seconds + quote.maximum_worker_start_seconds)
            * 1_000
        ):
            raise RunPodManagerError(
                "Serverless job pre-execution delay exceeds the accepted maximum"
            )
        if (
            job.execution_time_ms is not None
            and job.execution_time_ms > quote.maximum_execution_seconds * 1_000
        ):
            raise RunPodManagerError(
                "Serverless job execution exceeds the accepted maximum"
            )
        billable_coverage_until = attempt.completed_at + timedelta(
            seconds=quote.idle_tail_seconds
        )
        billing_hour_starts = tuple(
            value.astimezone(UTC) for value in attempt.exclusive_billing_hour_starts
        )
        window_from = billing_hour_starts[0]
        window_until = billing_hour_starts[-1] + timedelta(hours=1)
        if now < window_until:
            return None
        page = await asyncio.to_thread(
            self.control_client.serverless_billing,
            start_time=iso_datetime(window_from),
            end_time=iso_datetime(window_until),
            bucket_size="hour",
            endpoint_id=attempt.endpoint_id,
        )
        endpoint_hour_costs = _authoritative_hour_costs(
            page,
            endpoint_id=attempt.endpoint_id,
            window_from=window_from,
            window_until=window_until,
        )
        if endpoint_hour_costs is None:
            return None
        amounts = _sum_hour_costs(endpoint_hour_costs)
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "provider_quote_id": quote.provider_quote_id,
            "endpoint_profile_sha256": endpoint_profile_sha256,
            "endpoint_id": attempt.endpoint_id,
            "job_id": attempt.job_id,
            "attempt_id": attempt.attempt_id,
            "exclusive_window_sha256": attempt.exclusive_window_sha256,
            "exclusive_billing_hour_starts": [
                iso_datetime(value) for value in billing_hour_starts
            ],
            "attempt_started_at": iso_datetime(attempt.submitted_at),
            "attempt_completed_at": iso_datetime(attempt.completed_at),
            "billable_coverage_until": iso_datetime(billable_coverage_until),
            "billing_window_from": iso_datetime(window_from),
            "billing_window_until": iso_datetime(window_until),
            "hourly_worker_rate_usd": decimal_text(quote.hourly_worker_rate_usd),
            "pre_execution_delay_ms": job.delay_time_ms,
            "worker_startup_ms": None,
            "execution_ms": job.execution_time_ms,
            "accepted_idle_tail_ms": quote.idle_tail_seconds * 1_000,
            "idle_tail_ms": None,
            **{name: decimal_text(value) for name, value in amounts.items()},
        }
        return ServerlessBillingReceipt(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_billing_id="runpod-serverless-billing:" + json_sha256(identity),
            provider_quote_id=quote.provider_quote_id,
            endpoint_profile_sha256=endpoint_profile_sha256,
            endpoint_id=attempt.endpoint_id,
            job_id=attempt.job_id,
            attempt_id=attempt.attempt_id,
            exclusive_window_sha256=attempt.exclusive_window_sha256,
            exclusive_billing_hour_starts=billing_hour_starts,
            attempt_started_at=attempt.submitted_at,
            attempt_completed_at=attempt.completed_at,
            billable_coverage_until=billable_coverage_until,
            billing_window_from=window_from,
            billing_window_until=window_until,
            hourly_worker_rate_usd=quote.hourly_worker_rate_usd,
            pre_execution_delay_ms=job.delay_time_ms,
            worker_startup_ms=None,
            execution_ms=job.execution_time_ms,
            accepted_idle_tail_ms=quote.idle_tail_seconds * 1_000,
            idle_tail_ms=None,
            gpu_cost_usd=amounts["gpu_cost_usd"],
            cpu_cost_usd=amounts["cpu_cost_usd"],
            disk_cost_usd=amounts["disk_cost_usd"],
            fee_cost_usd=amounts["fee_cost_usd"],
            actual_cost_usd=amounts["actual_cost_usd"],
            reconciled_at=now,
        )

    async def final_ambiguous_window_billing(
        self,
        window: ServerlessAmbiguousBillingWindow,
        quote: ServerlessCapacityQuote,
    ) -> ServerlessAmbiguousWindowBillingReceipt | None:
        """Settle an exclusive endpoint-hour window without a provider job ID.

        This is the fail-closed recovery path for a submission whose acceptance
        was ambiguous. It performs no job lookup and trusts no estimate as
        actual cost: the last allocated hour must close and REST v2 must return
        a complete set of endpoint-hour aggregates before a receipt is emitted.
        """

        window.validate_quote(quote)
        return await self._final_ambiguous_window_billing_bound(
            window,
            quote,
            endpoint_profile_sha256=quote.endpoint_profile_sha256,
        )

    async def final_activated_ambiguous_window_billing(
        self,
        window: ServerlessAmbiguousBillingWindow,
        quote: PlannedServerlessCapacityQuote,
        activation: ServerlessEndpointActivationReceipt,
        submission: ServerlessActivatedSubmission,
    ) -> ServerlessAmbiguousWindowBillingReceipt | None:
        """Settle one ambiguous exclusive window on an activated endpoint."""

        window.validate_activated_submission(quote, activation, submission)
        return await self._final_ambiguous_window_billing_bound(
            window,
            quote,
            endpoint_profile_sha256=quote.endpoint_spec_sha256,
        )

    async def _final_ambiguous_window_billing_bound(
        self,
        window: ServerlessAmbiguousBillingWindow,
        quote: ServerlessCapacityQuote | PlannedServerlessCapacityQuote,
        *,
        endpoint_profile_sha256: str,
    ) -> ServerlessAmbiguousWindowBillingReceipt | None:
        now = self._now()
        billing_hour_starts = tuple(
            value.astimezone(UTC) for value in window.exclusive_billing_hour_starts
        )
        window_from = billing_hour_starts[0]
        window_until = billing_hour_starts[-1] + timedelta(hours=1)
        if now < window_until:
            return None
        page = await asyncio.to_thread(
            self.control_client.serverless_billing,
            start_time=iso_datetime(window_from),
            end_time=iso_datetime(window_until),
            bucket_size="hour",
            endpoint_id=window.endpoint_id,
        )
        endpoint_hour_costs = _authoritative_hour_costs(
            page,
            endpoint_id=window.endpoint_id,
            window_from=window_from,
            window_until=window_until,
        )
        if endpoint_hour_costs is None:
            return None
        amounts = _sum_hour_costs(endpoint_hour_costs)
        actual_cost = amounts["actual_cost_usd"]
        capped_cost = min(actual_cost, window.accepted_cost_ceiling_usd)
        operator_loss = actual_cost - capped_cost
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "provider_quote_id": quote.provider_quote_id,
            "endpoint_profile_sha256": endpoint_profile_sha256,
            "endpoint_id": window.endpoint_id,
            "attempt_id": window.attempt_id,
            "exclusive_window_sha256": window.exclusive_window_sha256,
            "exclusive_billing_hour_starts": [
                iso_datetime(value) for value in billing_hour_starts
            ],
            "attempted_at": iso_datetime(window.attempted_at),
            "billable_coverage_until": iso_datetime(window.billable_coverage_until),
            "billing_window_from": iso_datetime(window_from),
            "billing_window_until": iso_datetime(window_until),
            "accepted_cost_ceiling_usd": decimal_text(window.accepted_cost_ceiling_usd),
            "endpoint_hour_costs": [item.to_dict() for item in endpoint_hour_costs],
            **{name: decimal_text(value) for name, value in amounts.items()},
            "capped_cost_usd": decimal_text(capped_cost),
            "operator_loss_usd": decimal_text(operator_loss),
        }
        return ServerlessAmbiguousWindowBillingReceipt(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_billing_id=(
                "runpod-serverless-ambiguous-billing:" + json_sha256(identity)
            ),
            provider_quote_id=quote.provider_quote_id,
            endpoint_profile_sha256=endpoint_profile_sha256,
            endpoint_id=window.endpoint_id,
            attempt_id=window.attempt_id,
            exclusive_window_sha256=window.exclusive_window_sha256,
            exclusive_billing_hour_starts=billing_hour_starts,
            attempted_at=window.attempted_at,
            billable_coverage_until=window.billable_coverage_until,
            billing_window_from=window_from,
            billing_window_until=window_until,
            accepted_cost_ceiling_usd=window.accepted_cost_ceiling_usd,
            endpoint_hour_costs=endpoint_hour_costs,
            gpu_cost_usd=amounts["gpu_cost_usd"],
            cpu_cost_usd=amounts["cpu_cost_usd"],
            disk_cost_usd=amounts["disk_cost_usd"],
            fee_cost_usd=amounts["fee_cost_usd"],
            actual_cost_usd=actual_cost,
            capped_cost_usd=capped_cost,
            operator_loss_usd=operator_loss,
            reconciled_at=now,
        )

    async def _observe_quote(
        self, request: ServerlessCapacityQuoteRequest
    ) -> ServerlessCapacityQuote:
        profile = request.profile
        selection = await self._observe_catalog(profile.spec)
        requirements = profile.spec.constraints.placement_requirements()
        endpoint = await asyncio.to_thread(
            self.control_client.get_endpoint, profile.endpoint_id
        )
        _validate_endpoint_binding(
            endpoint,
            profile,
            gpu_pool=selection.gpu_pool,
            data_center_id=selection.data_center_id,
        )
        amounts = _quote_amounts(
            request,
            profile.spec,
            hourly=selection.hourly_worker_rate_usd,
        )
        expires_at = selection.observed_at + timedelta(
            seconds=request.quote_ttl_seconds
        )
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "workload_kind": request.workload_kind,
            "parameters_sha256": request.parameters_sha256,
            "profile_id": profile.profile_id,
            "endpoint_profile_sha256": profile.profile_sha256,
            "endpoint_id": profile.endpoint_id,
            "catalog_observation_sha256": selection.observation_sha256,
            "gpu_id": selection.gpu_id,
            "gpu_pool": selection.gpu_pool,
            "vram_gb": selection.vram_gb,
            "data_center_id": selection.data_center_id,
            "cloud": requirements.cloud.value,
            "gpu_count": requirements.gpu_count,
            "min_cuda_version": requirements.min_cuda_version,
            "availability": selection.availability.value,
            "benchmark_id": profile.constraints.benchmark_id,
            "hourly_worker_rate_usd": decimal_text(selection.hourly_worker_rate_usd),
            **_quote_request_identity(request, profile.spec),
            "catalog_observed_at": iso_datetime(selection.observed_at),
            "expires_at": iso_datetime(expires_at),
        }
        return ServerlessCapacityQuote(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_quote_id="runpod-serverless:" + json_sha256(identity),
            workload_kind=request.workload_kind,
            parameters_sha256=request.parameters_sha256,
            profile_id=profile.profile_id,
            endpoint_profile_sha256=profile.profile_sha256,
            endpoint_id=profile.endpoint_id,
            catalog_observation_sha256=selection.observation_sha256,
            gpu_id=selection.gpu_id,
            gpu_pool=selection.gpu_pool,
            gpu_name=selection.gpu_name,
            vram_gb=selection.vram_gb,
            data_center_id=selection.data_center_id,
            cloud=requirements.cloud,
            gpu_count=requirements.gpu_count,
            min_cuda_version=requirements.min_cuda_version,
            availability=selection.availability,
            benchmark_id=profile.constraints.benchmark_id,
            hourly_worker_rate_usd=selection.hourly_worker_rate_usd,
            estimated_queue_delay_seconds=request.estimated_queue_delay_seconds,
            estimated_worker_start_seconds=request.estimated_worker_start_seconds,
            estimated_execution_seconds=request.estimated_execution_seconds,
            idle_tail_seconds=profile.idle_tail_seconds,
            maximum_queue_delay_seconds=request.maximum_queue_delay_seconds,
            maximum_worker_start_seconds=request.maximum_worker_start_seconds,
            maximum_execution_seconds=request.maximum_execution_seconds,
            job_execution_timeout_ms=request.job_execution_timeout_ms,
            job_ttl_ms=request.job_ttl_ms,
            estimated_billable_seconds=amounts["estimated_billable_seconds"],
            maximum_billable_seconds=request.maximum_billable_seconds,
            estimated_worker_cost_usd=amounts["estimated_worker_cost_usd"],
            maximum_worker_cost_usd=amounts["maximum_worker_cost_usd"],
            estimated_non_worker_cost_usd=request.estimated_non_worker_cost_usd,
            maximum_non_worker_cost_usd=request.maximum_non_worker_cost_usd,
            estimated_cost_usd=amounts["estimated_cost_usd"],
            cost_ceiling_usd=amounts["cost_ceiling_usd"],
            catalog_observed_at=selection.observed_at,
            expires_at=expires_at,
        )

    async def _observe_planned_quote(
        self, request: PlannedServerlessCapacityQuoteRequest
    ) -> PlannedServerlessCapacityQuote:
        planned = request.endpoint
        spec = planned.spec
        selection = await self._observe_catalog(spec)
        requirements = spec.constraints.placement_requirements()
        operation_digest = spec.operation_digest(
            planned.endpoint_name,
            gpu_pool=selection.gpu_pool,
            data_center_id=selection.data_center_id,
        )
        amounts = _quote_amounts(
            request,
            spec,
            hourly=selection.hourly_worker_rate_usd,
        )
        expires_at = selection.observed_at + timedelta(
            seconds=request.quote_ttl_seconds
        )
        identity = {
            "contract_version": SERVERLESS_CAPACITY_CONTRACT_VERSION,
            "workload_kind": request.workload_kind,
            "parameters_sha256": request.parameters_sha256,
            "plan_id": planned.plan_id,
            "plan_sha256": planned.plan_sha256,
            "endpoint_name": planned.endpoint_name,
            "endpoint_spec_sha256": spec.spec_sha256,
            "operation_digest": operation_digest,
            "worker_reference": spec.worker_reference,
            "catalog_observation_sha256": selection.observation_sha256,
            "gpu_id": selection.gpu_id,
            "gpu_pool": selection.gpu_pool,
            "vram_gb": selection.vram_gb,
            "data_center_id": selection.data_center_id,
            "cloud": requirements.cloud.value,
            "gpu_count": requirements.gpu_count,
            "min_cuda_version": requirements.min_cuda_version,
            "availability": selection.availability.value,
            "benchmark_id": spec.constraints.benchmark_id,
            "hourly_worker_rate_usd": decimal_text(selection.hourly_worker_rate_usd),
            **_quote_request_identity(request, spec),
            "catalog_observed_at": iso_datetime(selection.observed_at),
            "expires_at": iso_datetime(expires_at),
        }
        return PlannedServerlessCapacityQuote(
            schema_version=SERVERLESS_CAPACITY_SCHEMA_VERSION,
            contract_version=SERVERLESS_CAPACITY_CONTRACT_VERSION,
            provider_quote_id=("runpod-serverless-plan:" + json_sha256(identity)),
            workload_kind=request.workload_kind,
            parameters_sha256=request.parameters_sha256,
            plan_id=planned.plan_id,
            plan_sha256=planned.plan_sha256,
            endpoint_name=planned.endpoint_name,
            endpoint_spec_sha256=spec.spec_sha256,
            operation_digest=operation_digest,
            worker_reference=spec.worker_reference,
            catalog_observation_sha256=selection.observation_sha256,
            gpu_id=selection.gpu_id,
            gpu_pool=selection.gpu_pool,
            gpu_name=selection.gpu_name,
            vram_gb=selection.vram_gb,
            data_center_id=selection.data_center_id,
            cloud=requirements.cloud,
            gpu_count=requirements.gpu_count,
            min_cuda_version=requirements.min_cuda_version,
            availability=selection.availability,
            benchmark_id=spec.constraints.benchmark_id,
            hourly_worker_rate_usd=selection.hourly_worker_rate_usd,
            estimated_queue_delay_seconds=request.estimated_queue_delay_seconds,
            estimated_worker_start_seconds=request.estimated_worker_start_seconds,
            estimated_execution_seconds=request.estimated_execution_seconds,
            idle_tail_seconds=spec.idle_tail_seconds,
            maximum_queue_delay_seconds=request.maximum_queue_delay_seconds,
            maximum_worker_start_seconds=request.maximum_worker_start_seconds,
            maximum_execution_seconds=request.maximum_execution_seconds,
            job_execution_timeout_ms=request.job_execution_timeout_ms,
            job_ttl_ms=request.job_ttl_ms,
            estimated_billable_seconds=amounts["estimated_billable_seconds"],
            maximum_billable_seconds=request.maximum_billable_seconds,
            estimated_worker_cost_usd=amounts["estimated_worker_cost_usd"],
            maximum_worker_cost_usd=amounts["maximum_worker_cost_usd"],
            estimated_non_worker_cost_usd=request.estimated_non_worker_cost_usd,
            maximum_non_worker_cost_usd=request.maximum_non_worker_cost_usd,
            estimated_cost_usd=amounts["estimated_cost_usd"],
            cost_ceiling_usd=amounts["cost_ceiling_usd"],
            catalog_observed_at=selection.observed_at,
            expires_at=expires_at,
        )

    async def _observe_catalog(self, spec: ServerlessEndpointSpec) -> _CatalogSelection:
        requirements = spec.constraints.placement_requirements()
        offers = await asyncio.to_thread(
            self.control_client.list_gpus,
            include_availability=True,
            products=(ComputeProduct.SERVERLESS,),
            count=requirements.gpu_count,
            cloud=requirements.cloud,
            min_cuda_version=requirements.min_cuda_version,
        )
        _validate_catalog_numbers(offers)
        observed_at = self._now()
        placement = select_gpu(offers, requirements, observed_at=observed_at)
        if placement.gpu_pool is None or placement.availability is None:
            raise RunPodManagerError(
                "Serverless catalog selection omitted pool or availability"
            )
        matching_pool = tuple(
            offer for offer in offers if offer.pool == placement.gpu_pool
        )
        if len(matching_pool) != 1 or matching_pool[0].id != placement.gpu_id:
            raise RunPodManagerError(
                "Serverless endpoint pool does not identify one exact catalog GPU"
            )
        data_center_id = _select_data_center(matching_pool[0], spec)
        catalog_observation_sha256 = _catalog_observation_sha256(
            matching_pool[0],
            spec,
            data_center_id=data_center_id,
        )
        hourly = Decimal(str(placement.offered_cost_per_hr))
        return _CatalogSelection(
            offer=matching_pool[0],
            observed_at=observed_at,
            gpu_id=placement.gpu_id,
            gpu_pool=placement.gpu_pool,
            gpu_name=placement.gpu_name,
            vram_gb=placement.memory_gb,
            data_center_id=data_center_id,
            availability=placement.availability,
            hourly_worker_rate_usd=hourly,
            observation_sha256=catalog_observation_sha256,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Serverless capacity clock must be timezone-aware")
        return value.astimezone(UTC)


def _validate_request_binding(
    request: ServerlessCapacityQuoteRequest, quote: ServerlessCapacityQuote
) -> None:
    profile = request.profile
    if (
        request.workload_kind != quote.workload_kind
        or request.parameters_sha256 != quote.parameters_sha256
        or profile.profile_id != quote.profile_id
        or profile.profile_sha256 != quote.endpoint_profile_sha256
        or profile.endpoint_id != quote.endpoint_id
        or profile.constraints.gpu_count != quote.gpu_count
        or profile.constraints.cloud != quote.cloud
        or profile.constraints.min_cuda_version != quote.min_cuda_version
        or quote.gpu_pool not in profile.constraints.allowed_gpu_pools
        or quote.data_center_id not in profile.constraints.allowed_data_center_ids
        or profile.constraints.benchmark_id != quote.benchmark_id
        or request.estimated_queue_delay_seconds != quote.estimated_queue_delay_seconds
        or request.estimated_worker_start_seconds
        != quote.estimated_worker_start_seconds
        or request.estimated_execution_seconds != quote.estimated_execution_seconds
        or profile.idle_tail_seconds != quote.idle_tail_seconds
        or request.maximum_queue_delay_seconds != quote.maximum_queue_delay_seconds
        or request.maximum_worker_start_seconds != quote.maximum_worker_start_seconds
        or request.maximum_execution_seconds != quote.maximum_execution_seconds
        or request.job_execution_timeout_ms != quote.job_execution_timeout_ms
        or request.job_ttl_ms != quote.job_ttl_ms
        or request.maximum_billable_seconds != quote.maximum_billable_seconds
        or request.estimated_non_worker_cost_usd != quote.estimated_non_worker_cost_usd
        or request.maximum_non_worker_cost_usd != quote.maximum_non_worker_cost_usd
    ):
        raise RunPodManagerError(
            "Serverless quote does not match the workload or configured profile"
        )


def _validate_planned_request_binding(
    request: PlannedServerlessCapacityQuoteRequest,
    quote: PlannedServerlessCapacityQuote,
) -> None:
    endpoint = request.endpoint
    spec = endpoint.spec
    if (
        request.workload_kind != quote.workload_kind
        or request.parameters_sha256 != quote.parameters_sha256
        or endpoint.plan_id != quote.plan_id
        or endpoint.plan_sha256 != quote.plan_sha256
        or endpoint.endpoint_name != quote.endpoint_name
        or spec.spec_sha256 != quote.endpoint_spec_sha256
        or spec.worker_reference != quote.worker_reference
        or spec.operation_digest(
            endpoint.endpoint_name,
            gpu_pool=quote.gpu_pool,
            data_center_id=quote.data_center_id,
        )
        != quote.operation_digest
        or spec.constraints.gpu_count != quote.gpu_count
        or spec.constraints.cloud != quote.cloud
        or spec.constraints.min_cuda_version != quote.min_cuda_version
        or quote.gpu_pool not in spec.constraints.allowed_gpu_pools
        or quote.data_center_id not in spec.constraints.allowed_data_center_ids
        or spec.constraints.benchmark_id != quote.benchmark_id
        or request.estimated_queue_delay_seconds != quote.estimated_queue_delay_seconds
        or request.estimated_worker_start_seconds
        != quote.estimated_worker_start_seconds
        or request.estimated_execution_seconds != quote.estimated_execution_seconds
        or spec.idle_tail_seconds != quote.idle_tail_seconds
        or request.maximum_queue_delay_seconds != quote.maximum_queue_delay_seconds
        or request.maximum_worker_start_seconds != quote.maximum_worker_start_seconds
        or request.maximum_execution_seconds != quote.maximum_execution_seconds
        or request.job_execution_timeout_ms != quote.job_execution_timeout_ms
        or request.job_ttl_ms != quote.job_ttl_ms
        or request.maximum_billable_seconds != quote.maximum_billable_seconds
        or request.estimated_non_worker_cost_usd != quote.estimated_non_worker_cost_usd
        or request.maximum_non_worker_cost_usd != quote.maximum_non_worker_cost_usd
    ):
        raise RunPodManagerError(
            "Planned Serverless quote does not match its workload or endpoint plan"
        )


def _validate_planned_quote_drift(
    current: PlannedServerlessCapacityQuote,
    accepted: PlannedServerlessCapacityQuote,
    *,
    accepted_cost_ceiling_usd: Decimal,
) -> None:
    stable_dimensions = (
        "workload_kind",
        "parameters_sha256",
        "plan_id",
        "plan_sha256",
        "endpoint_name",
        "endpoint_spec_sha256",
        "operation_digest",
        "worker_reference",
        "gpu_id",
        "gpu_pool",
        "vram_gb",
        "data_center_id",
        "cloud",
        "gpu_count",
        "min_cuda_version",
        "availability",
        "benchmark_id",
    )
    if any(
        getattr(current, name) != getattr(accepted, name) for name in stable_dimensions
    ):
        raise RunPodManagerError(
            "Planned Serverless placement or create configuration drifted"
        )
    if current.hourly_worker_rate_usd > accepted.hourly_worker_rate_usd:
        raise RunPodManagerError(
            "Planned Serverless worker rate increased after quote acceptance"
        )
    if current.cost_ceiling_usd > accepted_cost_ceiling_usd:
        raise RunPodManagerError(
            "Current planned Serverless cost exceeds the accepted ceiling"
        )


def _quote_request_identity(
    request: Any, spec: ServerlessEndpointSpec
) -> dict[str, Any]:
    return {
        "estimated_queue_delay_seconds": request.estimated_queue_delay_seconds,
        "estimated_worker_start_seconds": request.estimated_worker_start_seconds,
        "estimated_execution_seconds": request.estimated_execution_seconds,
        "idle_tail_seconds": spec.idle_tail_seconds,
        "maximum_queue_delay_seconds": request.maximum_queue_delay_seconds,
        "maximum_worker_start_seconds": request.maximum_worker_start_seconds,
        "maximum_execution_seconds": request.maximum_execution_seconds,
        "job_execution_timeout_ms": request.job_execution_timeout_ms,
        "job_ttl_ms": request.job_ttl_ms,
        "maximum_billable_seconds": request.maximum_billable_seconds,
        "estimated_non_worker_cost_usd": decimal_text(
            request.estimated_non_worker_cost_usd
        ),
        "maximum_non_worker_cost_usd": decimal_text(
            request.maximum_non_worker_cost_usd
        ),
    }


def _quote_amounts(
    request: Any,
    spec: ServerlessEndpointSpec,
    *,
    hourly: Decimal,
) -> dict[str, Any]:
    estimated_billable = (
        request.estimated_worker_start_seconds
        + request.estimated_execution_seconds
        + spec.idle_tail_seconds
    )
    estimated_worker = serverless_worker_cost_usd(
        hourly, spec.constraints.gpu_count, estimated_billable
    )
    maximum_worker = serverless_worker_cost_usd(
        hourly, spec.constraints.gpu_count, request.maximum_billable_seconds
    )
    return {
        "estimated_billable_seconds": estimated_billable,
        "estimated_worker_cost_usd": estimated_worker,
        "maximum_worker_cost_usd": maximum_worker,
        "estimated_cost_usd": estimated_worker + request.estimated_non_worker_cost_usd,
        "cost_ceiling_usd": maximum_worker + request.maximum_non_worker_cost_usd,
    }


def _validate_catalog_numbers(offers: Sequence[GPUOffer]) -> None:
    for offer in offers:
        rates = (offer.secure_price_per_hr, offer.community_price_per_hr)
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise RunPodManagerError("Serverless catalog contained an unsafe GPU rate")
        if (
            offer.memory_gb < 0
            or offer.secure_max_count < 0
            or offer.community_max_count < 0
        ):
            raise RunPodManagerError("Serverless catalog contained an unsafe GPU value")


def _select_data_center(offer: GPUOffer, spec: ServerlessEndpointSpec) -> str:
    allowed = set(spec.constraints.allowed_data_center_ids)
    matches: list[tuple[str, str]] = []
    for item in offer.data_centers:
        data_center_id = item.get("id")
        if data_center_id not in allowed:
            continue
        availability = item.get("availability")
        if not isinstance(availability, str) or availability not in _AVAILABILITY_RANK:
            raise RunPodManagerError(
                "Serverless catalog data-center availability is invalid"
            )
        if (
            _AVAILABILITY_RANK[availability]
            >= _AVAILABILITY_RANK[spec.constraints.minimum_availability.value]
        ):
            matches.append((data_center_id, availability))
    if not matches:
        raise RunPodManagerError(
            "Serverless catalog returned no allowed data center at required availability"
        )
    matches.sort(key=lambda item: (-_AVAILABILITY_RANK[item[1]], item[0]))
    return matches[0][0]


def _catalog_observation_sha256(
    offer: GPUOffer,
    spec: ServerlessEndpointSpec,
    *,
    data_center_id: str,
) -> str:
    selected_data_center = next(
        item for item in offer.data_centers if item.get("id") == data_center_id
    )
    return json_sha256(
        {
            "product": ComputeProduct.SERVERLESS.value,
            "gpu_id": offer.id,
            "gpu_pool": offer.pool,
            "gpu_name": offer.name,
            "manufacturer": offer.manufacturer,
            "vram_gb": offer.memory_gb,
            "cloud": spec.constraints.cloud.value,
            "gpu_count": spec.constraints.gpu_count,
            "hourly_worker_rate_usd": decimal_text(
                Decimal(str(offer.price_for(spec.constraints.cloud)))
            ),
            "availability": offer.availability.value if offer.availability else None,
            "data_center_id": data_center_id,
            "data_center_availability": selected_data_center.get("availability"),
            "min_cuda_version": offer.availability_min_cuda_version,
        }
    )


def _validate_endpoint_binding(
    endpoint: EndpointResource,
    profile: ServerlessEndpointProfile,
    *,
    gpu_pool: str,
    data_center_id: str,
) -> None:
    if (
        endpoint.id != profile.endpoint_id
        or endpoint.name != profile.endpoint_name
        or endpoint.endpoint_type != "QUEUE"
    ):
        raise RunPodManagerError(
            "Configured Serverless endpoint identity or type does not match profile"
        )
    _validate_endpoint_config(
        endpoint,
        profile.spec,
        gpu_pool=gpu_pool,
        data_center_id=data_center_id,
    )


def _validate_planned_endpoint_binding(
    endpoint: EndpointResource,
    planned: Any,
    *,
    expected_endpoint_id: str,
    gpu_pool: str,
    data_center_id: str,
) -> str:
    if (
        endpoint.id != expected_endpoint_id
        or endpoint.name != planned.endpoint_name
        or endpoint.endpoint_type != "QUEUE"
    ):
        raise RunPodManagerError(
            "Created Serverless endpoint identity or type does not match the plan"
        )
    normalized = _validate_endpoint_config(
        endpoint,
        planned.spec,
        gpu_pool=gpu_pool,
        data_center_id=data_center_id,
    )
    return json_sha256(
        {
            "endpoint_id": endpoint.id,
            "endpoint_name": endpoint.name,
            "immutable_config": normalized,
        }
    )


def _validate_endpoint_config(
    endpoint: EndpointResource,
    spec: ServerlessEndpointSpec,
    *,
    gpu_pool: str,
    data_center_id: str,
) -> dict[str, Any]:
    raw = endpoint.raw
    if not isinstance(raw, Mapping):
        raise RunPodManagerError("Serverless endpoint omitted its configuration")
    if raw.get("type") != "QUEUE":
        raise RunPodManagerError(
            "Configured Serverless endpoint identity or type does not match profile"
        )
    gpu = _mapping(raw.get("gpu"), "endpoint.gpu")
    workers = _mapping(raw.get("workers"), "endpoint.workers")
    scaling = _mapping(raw.get("scaling"), "endpoint.scaling")
    if _string_tuple(gpu.get("pools"), "endpoint.gpu.pools") != (gpu_pool,):
        raise RunPodManagerError(
            "Serverless endpoint does not constrain execution to the quoted pool"
        )
    if _integer(gpu.get("count"), "endpoint.gpu.count") != spec.constraints.gpu_count:
        raise RunPodManagerError("Serverless endpoint GPU count does not match profile")
    if (
        _integer(workers.get("min"), "endpoint.workers.min") != spec.workers_min
        or _integer(workers.get("max"), "endpoint.workers.max") != spec.workers_max
        or _integer(workers.get("idleTimeout"), "endpoint.workers.idleTimeout")
        != spec.idle_tail_seconds
    ):
        raise RunPodManagerError(
            "Serverless endpoint worker policy does not match profile"
        )
    scaling_type = scaling.get("type")
    scaling_key = "queueDelay"
    if (
        scaling_type != spec.scaling_type
        or _decimal(scaling.get(scaling_key), f"endpoint.scaling.{scaling_key}")
        != spec.scaling_value
    ):
        raise RunPodManagerError(
            "Serverless endpoint scaling policy does not match profile"
        )
    if _string_tuple(raw.get("dataCenterIds"), "endpoint.dataCenterIds") != (
        data_center_id,
    ):
        raise RunPodManagerError(
            "Serverless endpoint does not constrain execution to the quoted data center"
        )
    if _string_tuple(raw.get("networkVolumes"), "endpoint.networkVolumes") != (
        spec.network_volume_ids
    ):
        raise RunPodManagerError(
            "Serverless endpoint network volumes do not match profile"
        )
    expected_scalars = (
        ("image", spec.worker_reference),
        ("timeout", spec.execution_timeout_ms),
        ("flashboot", spec.flashboot.value),
        ("disk", spec.disk_gb),
    )
    if any(raw.get(name) != expected for name, expected in expected_scalars):
        raise RunPodManagerError(
            "Serverless endpoint runtime or billing profile does not match configuration"
        )
    if _string_tuple(raw.get("ports"), "endpoint.ports"):
        raise RunPodManagerError("Serverless endpoint ports must be empty")
    env = _mapping(raw.get("env"), "endpoint.env")
    if env:
        raise RunPodManagerError("Serverless endpoint environment must be empty")
    if raw.get("args") is not None:
        raise RunPodManagerError("Serverless endpoint args must be empty")
    if raw.get("registry") != spec.registry_id:
        raise RunPodManagerError("Serverless endpoint registry does not match the spec")
    return {
        "name": endpoint.name,
        "image": spec.worker_reference,
        "type": "QUEUE",
        "gpu": {"pools": [gpu_pool], "count": spec.constraints.gpu_count},
        "workers": {
            "min": spec.workers_min,
            "max": spec.workers_max,
            "idleTimeout": spec.idle_tail_seconds,
        },
        "scaling": {
            "type": spec.scaling_type,
            "queueDelay": decimal_text(spec.scaling_value),
        },
        "disk": spec.disk_gb,
        "ports": [],
        "env": {},
        "args": None,
        "registry": spec.registry_id,
        "dataCenterIds": [data_center_id],
        "networkVolumes": [],
        "timeout": spec.execution_timeout_ms,
        "flashboot": spec.flashboot.value,
    }


def _validate_terminal_job(
    job: ServerlessJob, attempt: ServerlessBillingAttempt
) -> None:
    if not isinstance(job.id, str) or job.id != attempt.job_id:
        raise RunPodManagerError("Runpod returned a mismatched Serverless job")
    if (
        not isinstance(job.status, str)
        or job.status.upper() not in _TERMINAL_JOB_STATES
    ):
        raise RunPodManagerError("Serverless job is not terminal for billing")
    for name, value in (
        ("delayTime", job.delay_time_ms),
        ("executionTime", job.execution_time_ms),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise RunPodManagerError(f"Serverless job {name} is unsafe")


def _authoritative_hour_costs(
    page: BillingPage,
    *,
    endpoint_id: str,
    window_from: datetime,
    window_until: datetime,
) -> tuple[ServerlessEndpointHourCost, ...] | None:
    metadata = page.metadata
    _exact_keys(metadata, _BILLING_METADATA_KEYS, "billing metadata")
    query = _mapping(metadata.get("query"), "billing metadata.query")
    totals = _mapping(metadata.get("totals"), "billing metadata.totals")
    _exact_keys(query, _BILLING_QUERY_KEYS, "billing query")
    _exact_keys(totals, _BILLING_TOTAL_KEYS, "billing totals")
    if (
        query.get("serverlessId") != endpoint_id
        or query.get("bucketSize") != "hour"
        or parse_datetime(_string(query.get("startTime"), "billing query start"))
        != window_from
        or parse_datetime(_string(query.get("endTime"), "billing query end"))
        != window_until
    ):
        raise RunPodManagerError("Runpod billing query echo does not match attempt")
    record_count = _integer(metadata.get("recordCount"), "billing recordCount")
    unique_count = _integer(
        metadata.get("uniqueServerlessCount"), "billing uniqueServerlessCount"
    )
    if record_count != len(page.records):
        raise RunPodManagerError("Runpod billing record count is inconsistent")
    if not page.records:
        if unique_count not in {0, 1}:
            raise RunPodManagerError("Runpod empty billing endpoint count is invalid")
        return None
    if unique_count != 1:
        raise RunPodManagerError("Runpod billing spans multiple Serverless endpoints")
    parsed: list[tuple[datetime, datetime, dict[str, Decimal]]] = []
    for record in page.records:
        _exact_keys(record, _BILLING_RECORD_KEYS, "billing record")
        if record.get("serverlessId") != endpoint_id:
            raise RunPodManagerError("Runpod billing record endpoint is mismatched")
        record_from = parse_datetime(
            _string(record.get("startTime"), "billing record start")
        )
        record_until = parse_datetime(
            _string(record.get("endTime"), "billing record end")
        )
        amounts = _amounts_from_mapping(record, "billing record")
        parsed.append((record_from, record_until, amounts))
    parsed.sort(key=lambda item: item[0])
    cursor = window_from
    hourly_costs: list[ServerlessEndpointHourCost] = []
    for record_from, record_until, amounts in parsed:
        if record_until <= record_from:
            raise RunPodManagerError("Runpod billing record interval is invalid")
        if record_from > cursor and record_from < window_until:
            return None
        if record_from != cursor:
            raise RunPodManagerError(
                "Runpod billing records overlap or escape the window"
            )
        if record_until > window_until:
            raise RunPodManagerError(
                "Runpod billing record exceeds the requested window"
            )
        expected_record_until = record_from + timedelta(hours=1)
        if record_until < expected_record_until:
            return None
        if record_until != expected_record_until:
            raise RunPodManagerError(
                "Runpod hourly billing record does not cover exactly one hour"
            )
        observation_identity = {
            "source": "runpod-v2-serverless-billing",
            "startTime": iso_datetime(record_from),
            "endTime": iso_datetime(record_until),
            "serverlessId": endpoint_id,
            "totalAmount": decimal_text(amounts["actual_cost_usd"]),
            "gpuAmount": decimal_text(amounts["gpu_cost_usd"]),
            "cpuAmount": decimal_text(amounts["cpu_cost_usd"]),
            "diskAmount": decimal_text(amounts["disk_cost_usd"]),
            "feeAmount": decimal_text(amounts["fee_cost_usd"]),
        }
        try:
            hourly_costs.append(
                ServerlessEndpointHourCost(
                    provider_observation_id=(
                        "runpod-serverless-hour:" + json_sha256(observation_identity)
                    ),
                    endpoint_id=endpoint_id,
                    utc_hour_start=record_from,
                    utc_hour_end=record_until,
                    gpu_cost_usd=amounts["gpu_cost_usd"],
                    cpu_cost_usd=amounts["cpu_cost_usd"],
                    disk_cost_usd=amounts["disk_cost_usd"],
                    fee_cost_usd=amounts["fee_cost_usd"],
                    actual_cost_usd=amounts["actual_cost_usd"],
                )
            )
        except ValueError as exc:
            raise RunPodManagerError(
                "Runpod billing record is not an exact UTC hour"
            ) from exc
        cursor = record_until
    if cursor < window_until:
        return None
    expected_totals = _amounts_from_mapping(totals, "billing totals")
    summed = _sum_hour_costs(tuple(hourly_costs))
    if summed != expected_totals:
        raise RunPodManagerError("Runpod billing totals do not equal its records")
    return tuple(hourly_costs)


_AMOUNT_NAMES = (
    "actual_cost_usd",
    "gpu_cost_usd",
    "cpu_cost_usd",
    "disk_cost_usd",
    "fee_cost_usd",
)
_AMOUNT_KEYS = {
    "actual_cost_usd": "totalAmount",
    "gpu_cost_usd": "gpuAmount",
    "cpu_cost_usd": "cpuAmount",
    "disk_cost_usd": "diskAmount",
    "fee_cost_usd": "feeAmount",
}


def _sum_hour_costs(
    values: tuple[ServerlessEndpointHourCost, ...],
) -> dict[str, Decimal]:
    return {
        name: sum((getattr(item, name) for item in values), start=Decimal(0))
        for name in _AMOUNT_NAMES
    }


def _amounts_from_mapping(value: Mapping[str, Any], context: str) -> dict[str, Decimal]:
    amounts = {
        name: _decimal(value.get(key), f"{context}.{key}")
        for name, key in _AMOUNT_KEYS.items()
    }
    if amounts["actual_cost_usd"] != (
        amounts["gpu_cost_usd"]
        + amounts["cpu_cost_usd"]
        + amounts["disk_cost_usd"]
        + amounts["fee_cost_usd"]
    ):
        raise RunPodManagerError(f"{context} cost components do not equal total")
    return amounts


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return value


def _integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return value


def _decimal(value: Any, context: str) -> Decimal:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RunPodManagerError(f"Invalid Runpod {context}")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise RunPodManagerError(f"Invalid Runpod {context}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return parsed


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RunPodManagerError(f"Invalid Runpod {context}")
    return tuple(value)


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    if set(value) != expected:
        raise RunPodManagerError(
            f"Runpod {context} contains missing or unsupported fields"
        )


__all__ = [
    "RunpodServerlessCapacityProvider",
    "ServerlessEndpointActivationError",
    "ServerlessEndpointCleanupError",
]
