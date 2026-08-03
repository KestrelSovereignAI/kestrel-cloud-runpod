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
from .inference_provider import RunpodInferenceLeaseProvider
from .manager import RunPodManager
from .models import (
    Availability,
    CloudType,
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
    maximum_serverless_cold_starts,
)
from .ollama_service import OllamaLeaseService
from .pod_capacity_contracts import (
    CatalogAttemptCapability,
    CatalogAttemptCapabilityStore,
    CatalogPodCapacityRequest,
    CatalogPodWorkloadState,
    CatalogWorkerEvidence,
    PodBillingReceipt,
    PodCapacityBillingState,
    PodCapacityCleanupError,
    PodCapacityCleanupState,
    PodCapacityConflictError,
    PodCapacityConstraints,
    PodCapacityEvidence,
    PodCapacityLease,
    PodCapacityLeaseRequest,
    PodCapacityLifecycleError,
    PodCapacityLifecycleEvidence,
    PodCapacityOwnership,
    PodCapacityQuote,
    PodCapacityQuoteRequest,
    PodCapacitySource,
    PodCapacitySpec,
    PodCapacityState,
    PodRealizedPlacement,
    pod_cost_usd,
)
from .pod_capacity_provider import (
    CreatedPodCapacity,
    PodCapacityCreatedMismatchError,
    PodCapacityObservation,
    PodCapacityProvider,
    RunpodPodCapacityProvider,
)
from .pod_capacity_quote import PodCapacityQuoteProvider, PodCapacityQuoteService
from .pod_capacity_repository import (
    SQLitePodCapacityRepository,
    pod_capacity_database_path,
)
from .pod_capacity_service import PodCapacityLeaseService
from .pod_transport import (
    CatalogPodTransportConflictError,
    CatalogPodTransportError,
    CatalogPodWorkloadObservation,
    CatalogPodWorkloadTransport,
)
from .serverless_capacity_contracts import (
    SERVERLESS_CAPACITY_CONTRACT_VERSION,
    SERVERLESS_CAPACITY_SCHEMA_VERSION,
    ServerlessAmbiguousBillingWindow,
    ServerlessAmbiguousWindowBillingReceipt,
    ServerlessBillingAttempt,
    ServerlessBillingReceipt,
    ServerlessCapacityConstraints,
    ServerlessCapacityQuote,
    ServerlessCapacityQuoteRequest,
    ServerlessEndpointProfile,
    serverless_billing_hour_starts,
    serverless_worker_cost_usd,
)
from .serverless_capacity_provider import RunpodServerlessCapacityProvider
from .training_contracts import (
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodLease,
    TrainingPodLifecycleError,
    TrainingPodOwnership,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
)
from .training_service import TrainingPodLeaseService

try:
    __version__ = _version("kestrel-cloud-runpod")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "SERVERLESS_CAPACITY_CONTRACT_VERSION",
    "SERVERLESS_CAPACITY_SCHEMA_VERSION",
    "Availability",
    "CatalogAttemptCapability",
    "CatalogAttemptCapabilityStore",
    "CatalogPodCapacityRequest",
    "CatalogPodTransportConflictError",
    "CatalogPodTransportError",
    "CatalogPodWorkloadObservation",
    "CatalogPodWorkloadState",
    "CatalogPodWorkloadTransport",
    "CatalogWorkerEvidence",
    "CloudType",
    "CreatedPodCapacity",
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
    "PodBillingReceipt",
    "PodCapacityBillingState",
    "PodCapacityCleanupError",
    "PodCapacityCleanupState",
    "PodCapacityConflictError",
    "PodCapacityConstraints",
    "PodCapacityCreatedMismatchError",
    "PodCapacityEvidence",
    "PodCapacityLease",
    "PodCapacityLeaseRequest",
    "PodCapacityLeaseService",
    "PodCapacityLifecycleError",
    "PodCapacityLifecycleEvidence",
    "PodCapacityObservation",
    "PodCapacityOwnership",
    "PodCapacityProvider",
    "PodCapacityQuote",
    "PodCapacityQuoteProvider",
    "PodCapacityQuoteRequest",
    "PodCapacityQuoteService",
    "PodCapacitySource",
    "PodCapacitySpec",
    "PodCapacityState",
    "PodRealizedPlacement",
    "PodStatus",
    "RunPodAPIError",
    "RunPodAmbiguousResultError",
    "RunPodFeature",
    "RunPodManager",
    "RunPodManagerError",
    "RunPodSession",
    "RunpodControlPlaneClient",
    "RunpodInferenceLeaseProvider",
    "RunpodPodCapacityProvider",
    "RunpodServerlessCapacityProvider",
    "RunpodServerlessClient",
    "SQLitePodCapacityRepository",
    "ServerlessAmbiguousBillingWindow",
    "ServerlessAmbiguousWindowBillingReceipt",
    "ServerlessBillingAttempt",
    "ServerlessBillingReceipt",
    "ServerlessCapacityConstraints",
    "ServerlessCapacityQuote",
    "ServerlessCapacityQuoteRequest",
    "ServerlessEndpointProfile",
    "TrainingPodCleanupError",
    "TrainingPodCleanupState",
    "TrainingPodConflictError",
    "TrainingPodLease",
    "TrainingPodLeaseService",
    "TrainingPodLifecycleError",
    "TrainingPodOwnership",
    "TrainingPodRequest",
    "TrainingPodSource",
    "TrainingPodState",
    "__version__",
    "maximum_serverless_cold_starts",
    "pod_capacity_database_path",
    "pod_cost_usd",
    "serverless_billing_hour_starts",
    "serverless_worker_cost_usd",
]
