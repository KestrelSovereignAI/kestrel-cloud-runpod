"""
RunPod GPU management for Kestrel.

Modular structure for the RunPod GPU instance manager:
- models.py: Data models, enums, exceptions
- providers.py: GPU provider abstractions (direct, managed proxy)
- core.py: Core SDK operations, profile loading, session management
- training.py: LoRA training methods (HTTP API)
- ollama.py: durable private-Ollama lease integration
- manager.py: Combined RunPodManager class
- feature.py: Kestrel feature integration

Usage:
    from kestrel_cloud_runpod import RunPodManager, RunPodFeature

    # Direct manager usage
    manager = RunPodManager()

    # LoRA Training workflow
    session = await manager.start_training_pod("companion-123")
    job_id = await manager.submit_training_job(session, avatar_data, "companion-123")
    status = await manager.poll_training_status(session, job_id)
    lora_data = await manager.download_lora(session, job_id)

    # Private Ollama uses an explicit OllamaLeaseRequest:
    lease = await manager.acquire_ollama_lease(request)
    route = lease.public_route_url

    # Or as a Kestrel feature
    feature = RunPodFeature(agent)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .clients import RunpodControlPlaneClient, RunpodServerlessClient
from .feature import RunPodFeature
from .manager import RunPodManager
from .models import (
    FlashBoot,
    GPUProfile,
    PlacementRequirements,
    PodStatus,
    RunPodAmbiguousResultError,
    RunPodAPIError,
    RunPodManagerError,
    RunPodSession,
)
from .ollama_contracts import (
    OllamaLease,
    OllamaLeaseAuthorizationError,
    OllamaLeaseConflictError,
    OllamaLeaseMode,
    OllamaLeaseReadinessError,
    OllamaLeaseRequest,
    OllamaLeaseState,
    OllamaLeaseTeardownError,
    OllamaResourceConstraints,
    OllamaResourceType,
    OllamaTeardownState,
)
from .ollama_service import OllamaLeaseService

try:
    __version__ = _version("kestrel-cloud-runpod")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "FlashBoot",
    "GPUProfile",
    "OllamaLease",
    "OllamaLeaseAuthorizationError",
    "OllamaLeaseConflictError",
    "OllamaLeaseMode",
    "OllamaLeaseReadinessError",
    "OllamaLeaseRequest",
    "OllamaLeaseService",
    "OllamaLeaseState",
    "OllamaLeaseTeardownError",
    "OllamaResourceConstraints",
    "OllamaResourceType",
    "OllamaTeardownState",
    "PlacementRequirements",
    "PodStatus",
    "RunPodAPIError",
    "RunPodAmbiguousResultError",
    "RunPodFeature",
    "RunPodManager",
    "RunPodManagerError",
    "RunPodSession",
    "RunpodControlPlaneClient",
    "RunpodServerlessClient",
    "__version__",
]
