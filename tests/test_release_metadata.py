"""Release metadata alignment tests."""

import tomllib
from pathlib import Path


def test_project_version_has_matching_release_notes():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        version = tomllib.load(project_file)["project"]["version"]

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{version}]" in changelog
