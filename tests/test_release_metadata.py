"""Release metadata alignment tests."""

import tomllib
from pathlib import Path


def test_project_version_has_matching_release_notes():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        version = tomllib.load(project_file)["project"]["version"]

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{version}]" in changelog


def test_published_core_floor_matches_operator_and_release_metadata():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        dependencies = tomllib.load(project_file)["project"]["dependencies"]

    expected = "kestrel-sovereign>=0.63.0,<1"
    assert expected in dependencies
    assert f"`{expected}`" in (root / "README.md").read_text(encoding="utf-8")
    assert "Require published Core 0.63.0" in (root / "CHANGELOG.md").read_text(
        encoding="utf-8"
    )
    expected_sdk = "kestrel-sovereign-sdk>=0.49.0,<1"
    assert expected_sdk in dependencies
    assert f"`{expected_sdk}`" in (root / "README.md").read_text(encoding="utf-8")


def test_installed_core_retires_new_runpod_work_but_keeps_recovery():
    from kestrel_sovereign.features.training.factory import TrainingProviderFactory

    capabilities = TrainingProviderFactory.get_capabilities("runpod")
    assert capabilities is not None
    assert capabilities.training is False
    assert capabilities.generation is False
    assert capabilities.supports_lora_download is True
