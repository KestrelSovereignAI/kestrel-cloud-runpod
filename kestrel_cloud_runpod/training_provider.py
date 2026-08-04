"""Compatibility names for the canonical Pod capacity provider."""

from .pod_capacity_provider import (
    CreatedPodCapacity,
    PodCapacityProvider,
    RunpodPodCapacityProvider,
    TrainingPodObservation,
)

CreatedTrainingPod = CreatedPodCapacity
TrainingPodCapacityProvider = PodCapacityProvider
RunpodTrainingPodProvider = RunpodPodCapacityProvider

__all__ = [
    "CreatedTrainingPod",
    "RunpodTrainingPodProvider",
    "TrainingPodCapacityProvider",
    "TrainingPodObservation",
]
