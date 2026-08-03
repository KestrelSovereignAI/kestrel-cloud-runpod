"""Compatibility import for the canonical Pod capacity lease service."""

from .pod_capacity_service import PodCapacityLeaseService

TrainingPodLeaseService = PodCapacityLeaseService

__all__ = ["TrainingPodLeaseService"]
