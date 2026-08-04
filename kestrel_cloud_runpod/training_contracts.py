"""Compatibility names for the canonical Pod capacity contracts.

New integrations must import :mod:`kestrel_cloud_runpod.pod_capacity_contracts`.
"""

from .pod_capacity_contracts import (
    TRAINING_PROFILE_IDS,
    TrainingPodCleanupError,
    TrainingPodCleanupState,
    TrainingPodConflictError,
    TrainingPodLease,
    TrainingPodLifecycleError,
    TrainingPodOwnership,
    TrainingPodRequest,
    TrainingPodSource,
    TrainingPodState,
    durable_training_name,
    fallback_training_cleanup_token,
    iso_datetime,
    require_aware,
    sanitize_training_error,
)

__all__ = [
    "TRAINING_PROFILE_IDS",
    "TrainingPodCleanupError",
    "TrainingPodCleanupState",
    "TrainingPodConflictError",
    "TrainingPodLease",
    "TrainingPodLifecycleError",
    "TrainingPodOwnership",
    "TrainingPodRequest",
    "TrainingPodSource",
    "TrainingPodState",
    "durable_training_name",
    "fallback_training_cleanup_token",
    "iso_datetime",
    "require_aware",
    "sanitize_training_error",
]
