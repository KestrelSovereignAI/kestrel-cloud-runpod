"""Compatibility import for the canonical Pod capacity repository."""

from .pod_capacity_repository import (
    SQLitePodCapacityRepository,
    pod_capacity_database_path,
)

SQLiteTrainingPodRepository = SQLitePodCapacityRepository
training_database_path = pod_capacity_database_path

__all__ = ["SQLiteTrainingPodRepository", "training_database_path"]
