"""
Historical Runpod LoRA workload observation and original capacity custody.

NEW LoRA acquisition/submission/generation belongs to the authorized catalog
workflow. These compatibility names refuse before effects; the SAME capacity
ledger still owns original status, cancellation, download and release.
"""

# This mixin intentionally declares its manager host contract in the class
# docstring; attributes and lifecycle methods are supplied by RunPodManagerCore.
# pyright: reportAttributeAccessIssue=false, reportUninitializedInstanceVariable=false

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn, overload

from kestrel_sdk.config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_DOWNLOAD,
)
from kestrel_sovereign.async_ownership import await_owned_operation
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
    owned_native_http_client,
)

from .models import (
    RunPodManagerError,
    RunPodSession,
)
from .providers import DirectRunPodProvider
from .training_contracts import TrainingPodLease
from .training_provider import RunpodTrainingPodProvider
from .training_repository import SQLiteTrainingPodRepository, training_database_path
from .training_service import TrainingPodLeaseService

logger = logging.getLogger(__name__)


class RunPodTrainingMixin:
    """
    Mixin for LoRA training operations on RunPod.

    Requires RunPodManagerCore as base class.
    """

    _training_pod_lease_service: TrainingPodLeaseService | None = None

    def set_training_pod_lease_service(self, service: TrainingPodLeaseService) -> None:
        """Inject the durable training lifecycle service for hosting or tests."""

        self._training_pod_lease_service = service

    def _get_training_pod_lease_service(self) -> TrainingPodLeaseService:
        existing = self._training_pod_lease_service
        if existing is not None:
            return existing
        if not isinstance(self.provider, DirectRunPodProvider):
            raise RunPodManagerError(
                "Durable training Pods require the direct Runpod v2 provider or an "
                "explicitly injected TrainingPodLeaseService"
            )
        settings = self.config.get("training_pods")
        if not isinstance(settings, Mapping):
            raise RunPodManagerError(
                "Configure the training_pods section before acquiring training capacity"
            )
        service = TrainingPodLeaseService(
            repository=SQLiteTrainingPodRepository(training_database_path(settings)),
            provider=RunpodTrainingPodProvider(self.provider),
            profiles=self.profiles,
            poll_interval_seconds=_required_positive_number(
                settings, "poll_interval_seconds"
            ),
            orphan_timeout_seconds=_required_positive_number(
                settings, "orphan_timeout_seconds"
            ),
            workload_status_observer=self._reconcile_training_workload_status,
        )
        self._training_pod_lease_service = service
        return service

    def get_training_pod_lease(self, cleanup_token: str) -> TrainingPodLease:
        """Return durable operational state for an authorized cleanup token."""

        lease = self._get_training_pod_lease_service().repository.get(cleanup_token)
        if lease is None:
            raise RunPodManagerError(
                f"Training cleanup token '{cleanup_token}' was not found"
            )
        return lease

    async def release_training_pod(
        self, cleanup_token: str, *, reason: str = "caller release"
    ) -> TrainingPodLease:
        """Stop owned training capacity or keep a retryable cleanup record."""

        service = self._get_training_pod_lease_service()
        lease = await service.release(cleanup_token, reason=reason)
        async with self._lock:
            session_token = (
                self._session.training_cleanup_token
                if self._session is not None
                else None
            )
            session_lease = (
                service.repository.get(session_token) if session_token else None
            )
            if self._session is not None and (
                session_token == cleanup_token
                or (
                    session_lease is not None
                    and session_lease.root_cleanup_token == cleanup_token
                )
            ):
                self._session.status = self._map_status("EXITED")
                self._session = None
        return lease

    async def reconcile_training_pods(self) -> tuple[TrainingPodLease, ...]:
        """Run one restart-safe cleanup pass from an external scheduler."""

        return await self._get_training_pod_lease_service().reconcile()

    async def _reconcile_training_workload_status(
        self, lease: TrainingPodLease
    ) -> str | None:
        """Recover webhook/poller loss without publishing or downloading output."""

        import httpx

        if not lease.backend_base_url or not lease.provider_job_id:
            return None
        url = f"{lease.backend_base_url}/status/{lease.provider_job_id}"
        try:
            async with owned_native_http_client(
                httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if is_execution_control_error(exc):
                raise
            raise RunPodManagerError(
                f"Training reconciliation status failed ({type(exc).__name__})"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RunPodManagerError(
                "Training reconciliation status returned a non-object"
            )
        status = payload.get("status") or payload.get("state")
        if not isinstance(status, str) or not status:
            raise RunPodManagerError("Training reconciliation status omitted status")
        return status

    async def _wait_for_training_ready(
        self,
        session: RunPodSession,
        timeout: int = 600,  # 10 minutes default (model loading can take 5-10 min)
        poll_interval: int = 15,
    ) -> None:
        """
        Wait for training pod's /ready endpoint to return 200.

        This is critical because:
        - The FLUX.2 model is ~24GB and takes 5-10 minutes to download on first run
        - Even cached, loading to GPU takes 1-2 minutes
        - /health returns OK while model is still loading
        - /ready returns 503 until model is fully loaded and GPU-ready

        Args:
            session: Active RunPod session with backend_base_url
            timeout: Max seconds to wait (default 10 minutes)
            poll_interval: Seconds between /ready checks

        Raises:
            RunPodManagerError: If not ready within timeout
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()

        ready_url = f"{session.backend_base_url}/ready"
        deadline = datetime.now(UTC) + timedelta(seconds=timeout)

        logger.info(
            "Waiting for training model to load at %s "
            "(may take 5-10 min on first run)...",
            ready_url,
        )

        attempts = 0
        last_status = None
        last_detail = None

        async with owned_native_http_client(
            httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
        ) as client:
            while datetime.now(UTC) < deadline:
                attempts += 1
                service.heartbeat(cleanup_token)
                try:
                    response = await client.get(ready_url)

                    if response.status_code == 200:
                        data = response.json()
                        gpu = data.get("gpu", "unknown")
                        gpu_memory = data.get("gpu_memory_gb", "?")
                        logger.info(f"Training pod ready! GPU: {gpu} ({gpu_memory}GB)")
                        return

                    elif response.status_code == 503:
                        # Could be loading OR training in progress
                        try:
                            data = response.json()
                            detail = data.get("detail", "loading")
                        except Exception as exc:
                            if is_execution_control_error(exc):
                                raise
                            detail = response.text[:100]

                        # Check if another training is already running
                        if "Training in progress" in str(detail):
                            # Extract the job ID from the message
                            # Format: "Training in progress: {job_id}"
                            existing_job = (
                                str(detail).split(":")[-1].strip()
                                if ":" in str(detail)
                                else "unknown"
                            )
                            raise RunPodManagerError(
                                "Cannot start training - another job is already "
                                f"running on this pod: {existing_job}. Wait for it "
                                "to complete or cancel it first."
                            )

                        if detail != last_detail:
                            logger.info(
                                f"Training pod not ready (attempt {attempts}): {detail}"
                            )
                            last_detail = detail

                    elif response.status_code == 404:
                        # SimpleTuner loads on demand when /ready is unavailable.
                        # Skip the wait and proceed directly to training
                        logger.info(
                            "Training Pod has no /ready endpoint; proceeding with "
                            "SimpleTuner's on-demand model load"
                        )
                        return

                    else:
                        # Unexpected status
                        logger.warning(
                            f"Unexpected /ready response: {response.status_code}"
                        )

                except httpx.ConnectError as exc:
                    if is_execution_control_error(exc):
                        raise
                    service.record_operation_error(cleanup_token, exc)
                    if last_status != "connect_error":
                        logger.info(
                            f"Training pod not yet reachable (attempt {attempts})"
                        )
                        last_status = "connect_error"

                except httpx.TimeoutException as exc:
                    if is_execution_control_error(exc):
                        raise
                    service.record_operation_error(cleanup_token, exc)
                    if last_status != "timeout":
                        logger.warning(
                            f"Training pod /ready timed out (attempt {attempts})"
                        )
                        last_status = "timeout"

                await asyncio.sleep(poll_interval)

        # Timeout reached
        raise RunPodManagerError(
            f"Training pod model not ready after {timeout}s ({attempts} attempts). "
            f"The FLUX model may still be downloading. Check pod logs."
        )

    async def start_training_pod(
        self, companion_id: str, *, cleanup_token: str | None = None
    ) -> NoReturn:
        """Retired: new LoRA work belongs to the authorized host catalog.

        A retained capacity token is recovery identity, not fresh model/input
        authority. No profile fallback, persistent-Pod resume or replacement
        acquisition is performed here.
        """
        self._refuse_legacy_admission()

    async def start_inference_pod(self, companion_id: str) -> NoReturn:
        """Refuse legacy fresh LoRA inference before configuration or effects."""
        self._refuse_legacy_admission()

    async def submit_training_job(
        self,
        session: RunPodSession,
        avatar_data: bytes,
        companion_id: str,
        callback_url: str | None = None,
        wait_for_model_ready: bool = True,
    ) -> NoReturn:
        """Refuse a NEW legacy /train POST, even with a retained session.

        Existing job status/cancel/download/resource release and the original
        durable capacity ledger remain available. A session/token alone cannot
        establish accepted catalog input, artifact, deadline or spend authority.
        """
        self._refuse_legacy_admission()

    async def get_current_job(self, session: RunPodSession) -> dict[str, Any] | None:
        """
        Check if a training job is currently running on the pod.

        Args:
            session: Active RunPod session

        Returns:
            Job info dict if training in progress, None if idle
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()
        service.heartbeat(cleanup_token)

        url = f"{session.backend_base_url}/current-job"

        async with owned_native_http_client(
            httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
        ) as client:
            try:
                response = await client.get(url)
                if response.status_code == 404:
                    # Endpoint not available in older container versions
                    return None
                response.raise_for_status()
                data = response.json()
                if data.get("current_job"):
                    return data
                return None
            except (httpx.HTTPError, ValueError) as exc:
                if is_execution_control_error(exc):
                    raise
                service.record_operation_error(cleanup_token, exc)
                raise RunPodManagerError(
                    f"Training current-job observation failed ({type(exc).__name__})"
                ) from exc

    async def cancel_training_job(
        self, session: RunPodSession, job_id: str
    ) -> dict[str, Any]:
        """
        Cancel a training job.

        Note: This marks the job as cancelled but may not stop the actual
        training process. For stuck jobs, pod restart may be needed.

        Args:
            session: Active RunPod session
            job_id: Job ID to cancel

        Returns:
            Cancellation result
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()
        service.request_cancellation(cleanup_token)

        url = f"{session.backend_base_url}/cancel/{job_id}"

        cancellation: dict[str, Any]
        try:
            async with owned_native_http_client(
                httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
            ) as client:
                response = await client.post(url)
                response.raise_for_status()
                payload: object = response.json()
                if not isinstance(payload, dict):
                    raise RunPodManagerError(
                        "Training cancellation returned a non-object response"
                    )
                cancellation = payload
        except asyncio.CancelledError as cancellation_error:
            if is_execution_control_error(cancellation_error):
                raise
            await self._release_after_cancellation(
                cleanup_token,
                "job cancellation",
                cancellation_error,
                persist_cancellation=False,
            )
        except (httpx.HTTPError, ValueError, RunPodManagerError) as exc:
            if is_execution_control_error(exc):
                raise
            service.record_operation_error(cleanup_token, exc)
            await self._release_after_cancellation(
                cleanup_token,
                "job cancellation",
                exc,
                persist_cancellation=False,
                reason="job cancellation failure",
            )
            raise RunPodManagerError(
                f"Training cancellation failed ({type(exc).__name__}); "
                "capacity was stopped"
            ) from exc
        await service.release(cleanup_token, reason="job cancelled")
        return cancellation

    async def clear_current_job(self, session: RunPodSession) -> dict[str, Any]:
        """
        Force-clear the current job lock on the pod.

        USE WITH CAUTION: Only use when a job is stuck and unresponsive.
        This clears the lock but does NOT kill any running processes.

        Args:
            session: Active RunPod session

        Returns:
            Result with cleared_job info
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()
        service.heartbeat(cleanup_token)

        url = f"{session.backend_base_url}/clear-current-job"

        try:
            async with owned_native_http_client(
                httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
            ) as client:
                response = await client.post(url)
                response.raise_for_status()
                result: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if is_execution_control_error(exc):
                raise
            service.record_operation_error(cleanup_token, exc)
            raise RunPodManagerError(
                f"Training lock clear failed ({type(exc).__name__})"
            ) from exc
        if not isinstance(result, dict):
            error = RunPodManagerError("Training lock clear returned a non-object")
            service.record_operation_error(cleanup_token, error)
            raise error
        logger.warning("Force-cleared the current training job lock")
        return result

    async def poll_training_status(
        self, session: RunPodSession, job_id: str
    ) -> dict[str, Any]:
        """
        Get training job status from pod.

        Args:
            session: Active RunPod session
            job_id: Training job ID

        Returns:
            Status dict with: status, progress, error, output_path
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()
        service.heartbeat(cleanup_token)

        status_url = f"{session.backend_base_url}/status/{job_id}"

        try:
            async with owned_native_http_client(
                httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
            ) as client:
                response = await client.get(status_url)
                response.raise_for_status()
                result: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if is_execution_control_error(exc):
                raise
            service.record_operation_error(cleanup_token, exc)
            raise RunPodManagerError(
                f"Training status observation failed ({type(exc).__name__}); "
                f"reconcile cleanup token '{cleanup_token}'"
            ) from exc
        if not isinstance(result, dict):
            error = RunPodManagerError("Training status returned a non-object response")
            service.record_operation_error(cleanup_token, error)
            raise error
        raw_status = result.get("status") or result.get("state")
        if not isinstance(raw_status, str) or not raw_status:
            error = RunPodManagerError("Training status response omitted status")
            service.record_operation_error(cleanup_token, error)
            raise error
        service.record_status(cleanup_token, raw_status)
        return result

    async def download_lora(self, session: RunPodSession, job_id: str) -> bytes:
        """
        Download trained LoRA file from pod.

        Args:
            session: Active RunPod session
            job_id: Completed training job ID

        Returns:
            LoRA file bytes (.safetensors)
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()
        service.heartbeat(cleanup_token)

        download_url = f"{session.backend_base_url}/download/{job_id}"
        logger.info(f"Downloading LoRA from {download_url}")

        try:
            async with owned_native_http_client(
                httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD)
            ) as client:
                response = await client.get(download_url)
                response.raise_for_status()
                content = response.content
        except httpx.HTTPError as exc:
            if is_execution_control_error(exc):
                raise
            service.record_operation_error(cleanup_token, exc)
            raise RunPodManagerError(
                f"Training result download failed ({type(exc).__name__}); "
                f"reconcile cleanup token '{cleanup_token}'"
            ) from exc
        service.record_result_retrieved(cleanup_token)
        return content

    @staticmethod
    def _training_token(session: RunPodSession) -> str:
        token = session.training_cleanup_token
        if not token:
            raise RunPodManagerError(
                "Training session has no durable cleanup token; reconcile the "
                "retained original ledger/resource identity, not a replacement pod"
            )
        return token

    def _session_lease(self, session: RunPodSession) -> TrainingPodLease:
        """Bind original cleanup identity independently of lifecycle state.

        This binding is used only by termination. A separate canonical
        controller may already have completed release while the same session
        remains cached. Workload eligibility is fenced by the service's own
        heartbeat/state boundaries, not by this cleanup identity projection.
        """

        cleanup_token = self._training_token(session)
        lease = self.get_training_pod_lease(cleanup_token)
        if (
            session.task_profile != "training"
            or lease.provider_pod_id != session.pod_id
            or lease.profile_id != session.profile.id
            or lease.companion_id != session.companion_id
        ):
            raise RunPodManagerError(
                "Training session no longer matches its durable capacity lease"
            )
        return lease

    @overload
    async def _release_after_cancellation(
        self,
        cleanup_token: str,
        operation: str,
        cancellation: asyncio.CancelledError,
        *,
        persist_cancellation: bool,
        reason: str | None = None,
    ) -> NoReturn: ...

    @overload
    async def _release_after_cancellation(
        self,
        cleanup_token: str,
        operation: str,
        cancellation: BaseException,
        *,
        persist_cancellation: bool,
        reason: str | None = None,
    ) -> None: ...

    async def _release_after_cancellation(
        self,
        cleanup_token: str,
        operation: str,
        cancellation: BaseException,
        *,
        persist_cancellation: bool,
        reason: str | None = None,
    ) -> None:
        """Join cleanup; healthy ordinary primaries retain caller normalization."""
        service = self._get_training_pod_lease_service()

        async def request_and_release() -> TrainingPodLease:
            if persist_cancellation:
                service.request_cancellation(cleanup_token)
            return await service.release(
                cleanup_token,
                reason=reason if reason is not None else f"{operation} cancellation",
            )

        cleanup = await await_owned_operation(
            request_and_release(),
            retain_originals=True,
        )
        if (
            cleanup.error is None
            and cleanup.cancellation is None
            and not isinstance(cancellation, asyncio.CancelledError)
        ):
            return
        terminal = execution_terminal_error(
            cleanup.error,
            cancellation,
            cleanup.cancellation,
            retain_originals=True,
            prefer_cancellation=True,
            isolate_control_originals=True,
        )
        if terminal is None:
            raise AssertionError("Training cleanup lost its terminal outcome")
        raise terminal

    async def generate_with_lora(
        self, session: RunPodSession, prompt: str, lora_path: str, num_outputs: int = 1
    ) -> NoReturn:
        """Refuse fresh legacy generation; old FLUX paths are not SDXL inputs."""
        self._refuse_legacy_admission()


def _required_positive_number(settings: Mapping[str, Any], name: str) -> float:
    value = settings.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RunPodManagerError(f"training_pods.{name} must be a positive number")
    return float(value)
