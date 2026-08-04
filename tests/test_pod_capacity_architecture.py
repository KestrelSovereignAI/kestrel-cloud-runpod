"""One-source-of-truth and public/private package-boundary tests."""

from pathlib import Path

from kestrel_cloud_runpod.pod_capacity_provider import RunpodPodCapacityProvider
from kestrel_cloud_runpod.pod_capacity_repository import SQLitePodCapacityRepository
from kestrel_cloud_runpod.pod_capacity_service import PodCapacityLeaseService
from kestrel_cloud_runpod.training_provider import RunpodTrainingPodProvider
from kestrel_cloud_runpod.training_repository import SQLiteTrainingPodRepository
from kestrel_cloud_runpod.training_service import TrainingPodLeaseService


def test_training_names_are_aliases_not_a_second_writable_lifecycle() -> None:
    assert TrainingPodLeaseService is PodCapacityLeaseService
    assert SQLiteTrainingPodRepository is SQLitePodCapacityRepository
    assert RunpodTrainingPodProvider is RunpodPodCapacityProvider


def test_public_cloud_package_has_no_private_catalog_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "kestrel_cloud_runpod").glob("*.py")
    )
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "frinz_catalog" not in source
    assert "frinz-catalog" not in project
