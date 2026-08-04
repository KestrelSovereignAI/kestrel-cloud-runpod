"""
RunPod LoRA Training Methods.

Contains SSH-based and HTTP-based training methods for
LoRA training on RunPod GPU instances.
"""

# This mixin intentionally declares its manager host contract in the class
# docstring; attributes and lifecycle methods are supplied by RunPodManagerCore.
# pyright: reportAttributeAccessIssue=false, reportUninitializedInstanceVariable=false

import asyncio
import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.config.constants import (
    HTTP_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_DOWNLOAD,
    HTTP_TIMEOUT_UPLOAD,
    POD_READY_TIMEOUT,
)

from .models import (
    GPUProfile,
    RunPodAmbiguousResultError,
    RunPodManagerError,
    RunPodSession,
)
from .providers import DirectRunPodProvider
from .training_contracts import (
    TRAINING_PROFILE_IDS,
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodLease,
    TrainingPodLifecycleError,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
    durable_training_name,
    fallback_training_cleanup_token,
)
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
    _training_admission_lock: asyncio.Lock | None = None

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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
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
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        logger.info(
            "Waiting for training model to load at %s "
            "(may take 5-10 min on first run)...",
            ready_url,
        )

        attempts = 0
        last_status = None
        last_detail = None

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
            while datetime.now(timezone.utc) < deadline:
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
                        except Exception:
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

                except httpx.ConnectError:
                    service.record_operation_error(
                        cleanup_token, httpx.ConnectError("training readiness")
                    )
                    if last_status != "connect_error":
                        logger.info(
                            f"Training pod not yet reachable (attempt {attempts})"
                        )
                        last_status = "connect_error"

                except httpx.TimeoutException:
                    service.record_operation_error(
                        cleanup_token, httpx.TimeoutException("training readiness")
                    )
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
    ) -> RunPodSession:
        """
        Start a pod for LoRA training using the training profile.

        If persistent_pod_id is configured:
        - Ask v2 to start the existing pod if stopped
        - Use the existing pod if already running (instant)
        - After training, pod should be stopped (paused) not terminated

        Otherwise tries profiles in order:
        1. "training" - A100 80GB in US-TX-3 (has network volume cache)
        2. "training-h100" - H100 80GB in US-TX-3 (faster but more expensive)
        3. "training-flex" - A100 80GB any datacenter (no network volume)

        Args:
            companion_id: Companion being trained (for naming/tracking)

        Returns:
            A route-ready session carrying its durable cleanup token.
        """
        ttl_seconds = self._validate_ttl(3600)  # 1 hour requested training cap
        root_cleanup_token = cleanup_token or f"training:{uuid.uuid4()}"

        service = self._get_training_pod_lease_service()
        active_attempt = service.get_active_family_attempt(root_cleanup_token)
        if active_attempt is not None:
            if active_attempt.companion_id != companion_id:
                raise RunPodManagerError(
                    f"Training cleanup token '{root_cleanup_token}' already belongs "
                    "to another companion"
                )
            if active_attempt.state is not TrainingPodState.READY:
                raise RunPodManagerError(
                    f"Training cleanup token '{root_cleanup_token}' has an active "
                    f"attempt in state {active_attempt.state.value}; reconcile it"
                )
            active_attempt = service.heartbeat(active_attempt.cleanup_token)
            profile = self._select_profile(active_attempt.profile_id)
            return await self._record_training_session(
                lease=active_attempt,
                profile=profile,
                companion_id=companion_id,
                ttl_seconds=ttl_seconds,
            )

        # Check if training profile has a persistent pod configured.
        if "training" in self.profiles:
            profile = self.profiles["training"]
            # Expand at runtime so a server restart is unnecessary.
            persistent_pod_id = self._expand_single_env_var(profile.persistent_pod_id)
            if persistent_pod_id:
                logger.info(f"Using persistent training pod: {persistent_pod_id}")
                return await self._acquire_training_session(
                    service=service,
                    companion_id=companion_id,
                    profile=profile,
                    ttl_seconds=ttl_seconds,
                    source=TrainingPodSource.CONFIGURED_PERSISTENT,
                    provider_pod_id=persistent_pod_id,
                    cleanup_token=root_cleanup_token,
                    root_cleanup_token=root_cleanup_token,
                )

        # Reuse an explicitly stopped Pod when the workload's storage policy permits it.
        stopped_pod = await self.find_stopped_pod("lora_training", "training")
        if stopped_pod:
            try:
                profile = self._select_profile("training")
                pod_id = stopped_pod.get("id")
                if not isinstance(pod_id, str) or not pod_id:
                    raise RunPodManagerError("Stopped training Pod omitted its v2 ID")
                logger.info("Starting existing stopped training Pod through Runpod v2")
                return await self._acquire_training_session(
                    service=service,
                    companion_id=companion_id,
                    profile=profile,
                    ttl_seconds=ttl_seconds,
                    source=TrainingPodSource.STOPPED_REUSE,
                    provider_pod_id=pod_id,
                    cleanup_token=root_cleanup_token,
                    root_cleanup_token=root_cleanup_token,
                )
            except RunPodManagerError as e:
                logger.warning(f"Failed to resume stopped pod: {e}")
                raise

        # Try each profile in order
        last_error = None
        attempted_profiles = 0
        for profile_name in TRAINING_PROFILE_IDS:
            if profile_name not in self.profiles:
                continue

            profile_cleanup_token = root_cleanup_token
            if attempted_profiles > 0:
                # One logical request may make several confirmed-safe capacity
                # attempts.  Each attempt needs its own durable primary key so
                # a released first profile cannot collide with the fallback,
                # while the first attempt retains the caller's stable token.
                profile_cleanup_token = fallback_training_cleanup_token(
                    root_cleanup_token, profile_name
                )
            attempted_profiles += 1
            try:
                logger.info(f"Trying training profile: {profile_name}")
                return await self._acquire_training_session(
                    service=service,
                    companion_id=companion_id,
                    profile=self.profiles[profile_name],
                    ttl_seconds=ttl_seconds,
                    source=TrainingPodSource.CREATED,
                    provider_pod_id=None,
                    cleanup_token=profile_cleanup_token,
                    root_cleanup_token=root_cleanup_token,
                )
            except TrainingPodLifecycleError as e:
                logger.warning(f"Profile {profile_name} failed: {e}")
                last_error = e
                if e.billing_risk or e.cleanup_state not in {
                    TrainingPodCleanupState.COMPLETE,
                    TrainingPodCleanupState.NOT_OWNED,
                }:
                    raise
                # A known create rejection with confirmed no capacity can try
                # the next configured profile. Reuse/resume never reaches here.
                continue

        # All profiles failed
        if last_error:
            logger.error(f"All training profiles failed. Last error: {last_error}")
            raise last_error
        raise RunPodManagerError("No training profile is configured")

    async def _acquire_training_session(
        self,
        *,
        service: TrainingPodLeaseService,
        companion_id: str,
        profile: GPUProfile,
        ttl_seconds: int,
        source: TrainingPodSource,
        provider_pod_id: str | None,
        cleanup_token: str,
        root_cleanup_token: str,
    ) -> RunPodSession:
        if self._training_admission_lock is None:
            self._training_admission_lock = asyncio.Lock()
        async with self._training_admission_lock:
            async with self._lock:
                if self._session and self._session.is_active:
                    raise RunPodManagerError("A RunPod session is already active")
            now = datetime.now(timezone.utc)
            readiness_seconds = profile.readiness_timeout_seconds or POD_READY_TIMEOUT
            readiness_seconds = min(readiness_seconds, ttl_seconds - 1)
            request = TrainingPodRequest(
                cleanup_token=cleanup_token,
                root_cleanup_token=root_cleanup_token,
                companion_id=companion_id,
                profile_id=profile.id,
                source=source,
                resource_name=durable_training_name(cleanup_token),
                provider_pod_id=provider_pod_id,
                created_at=now,
                readiness_deadline=now + timedelta(seconds=readiness_seconds),
                hard_deadline=now + timedelta(seconds=ttl_seconds),
            )
            lease = await service.acquire(request)
            if not lease.provider_pod_id or not lease.backend_base_url:
                raise RunPodManagerError(
                    "Durable training acquisition returned no route"
                )
            return await self._record_training_session(
                lease=lease,
                profile=profile,
                companion_id=companion_id,
                ttl_seconds=ttl_seconds,
            )

    async def _record_training_session(
        self,
        *,
        lease: TrainingPodLease,
        profile: GPUProfile,
        companion_id: str,
        ttl_seconds: int,
    ) -> RunPodSession:
        if not lease.provider_pod_id or not lease.backend_base_url:
            raise RunPodManagerError("Durable training lease has no route")
        session = RunPodSession(
            pod_id=lease.provider_pod_id,
            profile=profile,
            task_profile="training",
            model_name="flux-lora-trainer",
            pod_type=profile.pod_type,
            status=self._map_status("RUNNING"),
            ttl_seconds=ttl_seconds,
            started_at=lease.created_at,
            expires_at=lease.hard_deadline,
            backend_base_url=lease.backend_base_url,
            companion_id=companion_id,
            training_cleanup_token=lease.cleanup_token,
        )
        async with self._lock:
            self._session = session
        return session

    async def start_inference_pod(self, companion_id: str) -> Optional[RunPodSession]:
        """
        Start a pod for LoRA-based image generation.

        Reuses an eligible stopped Pod before creating a new one.

        Args:
            companion_id: Companion for tracking

        Returns:
            RunPodSession if started successfully, None otherwise
        """
        try:
            profile = self._select_profile("image")
            ttl_seconds = 600  # 10 min for inference

            # Try to resume a stopped inference pod first (much faster)
            stopped_pod = await self.find_stopped_pod("lora_inference", "image")
            if stopped_pod:
                logger.info("Starting existing stopped inference Pod through Runpod v2")
                return await self.resume_stopped_pod(stopped_pod, profile, ttl_seconds)

            # No stopped pod found, create new one
            await self.start_session(
                task_profile="image",
                model_name="flux-with-lora",
                ttl_seconds=ttl_seconds,
                metadata={
                    "name": f"kestrel-selfie-{companion_id[:8]}",
                    "companion_id": companion_id,
                    "purpose": "lora_inference",
                },
            )

            async with self._lock:
                return self._session

        except RunPodAmbiguousResultError:
            raise
        except RunPodManagerError as e:
            logger.error(f"Failed to start inference pod for {companion_id}: {e}")
            return None

    async def submit_training_job(
        self,
        session: RunPodSession,
        avatar_data: bytes,
        companion_id: str,
        callback_url: Optional[str] = None,
        wait_for_model_ready: bool = True,
    ) -> str:
        """
        Submit training job to pod's /train endpoint.

        IMPORTANT: The training pod needs 5-10 minutes to load the FLUX model
        after startup. We wait for the /ready endpoint before submitting.

        Args:
            session: Active RunPod session
            avatar_data: Avatar image bytes from sovereign storage
            companion_id: Companion ID
            callback_url: Optional webhook for completion
            wait_for_model_ready: If True, wait for /ready endpoint before submitting.
                                  Model loading can take 5-10 minutes on first run.

        Returns:
            Training job ID from the pod
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")
        cleanup_token = self._training_token(session)
        service = self._get_training_pod_lease_service()

        try:
            # /health can be healthy while the large trainer model is loading.
            if wait_for_model_ready:
                await self._wait_for_training_ready(session)
        except asyncio.CancelledError:
            service.request_cancellation(cleanup_token)
            await self._release_after_cancellation(cleanup_token, "model readiness")
            raise
        except RunPodManagerError as exc:
            service.record_operation_error(cleanup_token, exc)
            try:
                await service.release(cleanup_token, reason="model readiness failure")
            except TrainingPodCleanupError as cleanup_exc:
                raise cleanup_exc from exc
            raise

        train_url = f"{session.backend_base_url}/train"
        logger.info(f"Submitting training job to {train_url}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_UPLOAD) as client:
            try:
                # Use avatar bytes directly from sovereign storage
                logger.info(
                    "Using avatar data from sovereign storage: %s bytes",
                    len(avatar_data),
                )

                # Detect content type from magic bytes
                if avatar_data[:8] == b"\x89PNG\r\n\x1a\n":
                    content_type = "image/png"
                    filename = "avatar.png"
                else:
                    content_type = "image/jpeg"
                    filename = "avatar.jpg"

                # Submit as multipart form with image file
                files = {"image": (filename, avatar_data, content_type)}
                data = {
                    "companion_id": companion_id,
                }
                if callback_url:
                    data["callback_url"] = callback_url

                response = await client.post(train_url, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, Mapping):
                    raise RunPodManagerError(
                        "Training workload returned a non-object submission response"
                    )
            except asyncio.CancelledError:
                service.request_cancellation(cleanup_token)
                await self._release_after_cancellation(
                    cleanup_token, "training submission"
                )
                raise
            except httpx.ConnectError as e:
                return await self._recover_or_cleanup_submission(
                    client=client,
                    session=session,
                    companion_id=companion_id,
                    cleanup_token=cleanup_token,
                    cause=e,
                )
            except httpx.HTTPStatusError as e:
                return await self._recover_or_cleanup_submission(
                    client=client,
                    session=session,
                    companion_id=companion_id,
                    cleanup_token=cleanup_token,
                    cause=e,
                )
            except httpx.TimeoutException as e:
                return await self._recover_or_cleanup_submission(
                    client=client,
                    session=session,
                    companion_id=companion_id,
                    cleanup_token=cleanup_token,
                    cause=e,
                )
            except (ValueError, RunPodManagerError) as exc:
                service.record_operation_error(cleanup_token, exc)
                try:
                    await service.release(
                        cleanup_token, reason="invalid submission response"
                    )
                except TrainingPodCleanupError as cleanup_exc:
                    raise cleanup_exc from exc
                raise RunPodManagerError(
                    f"Training submission failed ({type(exc).__name__})"
                ) from exc

        job_id = result.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            error = RunPodManagerError("Training workload omitted its job ID")
            service.record_operation_error(cleanup_token, error)
            try:
                await service.release(cleanup_token, reason="missing training job ID")
            except TrainingPodCleanupError as cleanup_exc:
                raise cleanup_exc from error
            raise error

        service.record_job(cleanup_token, job_id)
        logger.info(f"Training job submitted: {job_id}")
        return job_id

    async def _recover_or_cleanup_submission(
        self,
        *,
        client: Any,
        session: RunPodSession,
        companion_id: str,
        cleanup_token: str,
        cause: BaseException,
    ) -> str:
        """Recover an accepted job ID or stop capacity after an ambiguous POST."""

        import httpx

        service = self._get_training_pod_lease_service()
        current_url = f"{session.backend_base_url}/current-job"
        recovered_job_id: str | None = None
        try:
            response = await client.get(current_url)
            if response.status_code == 200:
                payload: object = response.json()
                if isinstance(payload, Mapping):
                    current = payload.get("current_job")
                    if isinstance(current, Mapping):
                        raw_id = current.get("job_id") or current.get("id")
                        raw_companion = current.get("companion_id")
                    else:
                        raw_id = payload.get("job_id")
                        raw_companion = payload.get("companion_id")
                    if (
                        isinstance(raw_id, str)
                        and raw_id
                        and raw_companion
                        in {
                            None,
                            companion_id,
                        }
                    ):
                        recovered_job_id = raw_id
        except (httpx.HTTPError, ValueError):
            recovered_job_id = None
        if recovered_job_id:
            service.record_job(cleanup_token, recovered_job_id)
            return recovered_job_id
        service.record_operation_error(cleanup_token, cause)
        try:
            await service.release(cleanup_token, reason="training submission failure")
        except TrainingPodCleanupError as cleanup_exc:
            raise cleanup_exc from cause
        raise RunPodManagerError(
            f"Training submission failed ({type(cause).__name__}); capacity was stopped"
        ) from cause

    async def get_current_job(self, session: RunPodSession) -> Optional[Dict[str, Any]]:
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

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
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
                service.record_operation_error(cleanup_token, exc)
                raise RunPodManagerError(
                    f"Training current-job observation failed ({type(exc).__name__})"
                ) from exc

    async def cancel_training_job(
        self, session: RunPodSession, job_id: str
    ) -> Dict[str, Any]:
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

        cancellation: Dict[str, Any]
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                response = await client.post(url)
                response.raise_for_status()
                payload: object = response.json()
                if not isinstance(payload, dict):
                    raise RunPodManagerError(
                        "Training cancellation returned a non-object response"
                    )
                cancellation = payload
        except asyncio.CancelledError:
            await self._release_after_cancellation(cleanup_token, "job cancellation")
            raise
        except (httpx.HTTPError, ValueError, RunPodManagerError) as exc:
            service.record_operation_error(cleanup_token, exc)
            try:
                await service.release(cleanup_token, reason="job cancellation failure")
            except TrainingPodCleanupError as cleanup_exc:
                raise cleanup_exc from exc
            raise RunPodManagerError(
                f"Training cancellation failed ({type(exc).__name__}); "
                "capacity was stopped"
            ) from exc
        await service.release(cleanup_token, reason="job cancelled")
        return cancellation

    async def clear_current_job(self, session: RunPodSession) -> Dict[str, Any]:
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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                response = await client.post(url)
                response.raise_for_status()
                result: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
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
    ) -> Dict[str, Any]:
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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT) as client:
                response = await client.get(status_url)
                response.raise_for_status()
                result: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_DOWNLOAD) as client:
                response = await client.get(download_url)
                response.raise_for_status()
                content = response.content
        except httpx.HTTPError as exc:
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
                "Training session has no durable cleanup token; reacquire it through "
                "start_training_pod"
            )
        return token

    async def _release_after_cancellation(
        self, cleanup_token: str, operation: str
    ) -> None:
        service = self._get_training_pod_lease_service()
        cleanup = asyncio.create_task(
            service.release(cleanup_token, reason=f"{operation} cancellation")
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup

    async def generate_with_lora(
        self, session: RunPodSession, prompt: str, lora_path: str, num_outputs: int = 1
    ) -> Dict[str, Any]:
        """
        Generate images using loaded LoRA model.

        Args:
            session: Active RunPod session
            prompt: Image generation prompt
            lora_path: Path to LoRA file (in Kestrel storage)
            num_outputs: Number of images to generate

        Returns:
            Dict with "images" list of URLs/base64
        """
        import httpx

        if not session.backend_base_url:
            raise RunPodManagerError("Session has no backend URL")

        # Use the image generation endpoint
        generate_url = f"{session.backend_base_url}/generate"
        logger.info(f"Generating with LoRA at {generate_url}")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_UPLOAD) as client:
            response = await client.post(
                generate_url,
                json={
                    "prompt": prompt,
                    "lora_path": lora_path,
                    "num_outputs": num_outputs,
                    "aspect_ratio": "1:1",
                    "output_format": "jpg",
                },
            )
            response.raise_for_status()
            return response.json()


def _required_positive_number(settings: Mapping[str, Any], name: str) -> float:
    value = settings.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RunPodManagerError(f"training_pods.{name} must be a positive number")
    return float(value)
