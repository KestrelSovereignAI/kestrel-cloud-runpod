"""Execute real release guards and audit every reusable checkout's source."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _workflows():
    root = Path(__file__).resolve().parents[1] / ".github/workflows"
    return tuple(
        yaml.load((root / name).read_text(), Loader=yaml.BaseLoader)
        for name in ("ci.yml", "publish.yml")
    )


def _assert_checkout_bindings(ci):
    assert ci["on"]["workflow_call"]["inputs"]["ref"]["type"] == "string"
    checkouts = [
        step
        for job in ci["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) >= 4
    assert all(
        step.get("with", {}).get("ref") == "${{ inputs.ref }}" for step in checkouts
    )


def test_every_reusable_ci_checkout_honors_the_explicit_ref():
    ci, _ = _workflows()
    _assert_checkout_bindings(ci)


@pytest.mark.parametrize(
    "job", ["lint-and-imports", "build-check", "install-check", "test"]
)
def test_omitting_any_checkout_ref_breaks_the_regression_guard(job):
    ci, _ = _workflows()
    altered = copy.deepcopy(ci)
    checkout = next(
        s
        for s in altered["jobs"][job]["steps"]
        if s.get("uses", "").startswith("actions/checkout@")
    )
    checkout.setdefault("with", {}).pop("ref", None)
    with pytest.raises(AssertionError):
        _assert_checkout_bindings(altered)


def test_publication_tests_and_builds_one_resolved_commit_before_upload():
    _, publish = _workflows()
    jobs = publish["jobs"]
    assert jobs["resolve"]["outputs"]["sha"] == "${{ steps.source.outputs.sha }}"
    assert jobs["ci"]["needs"] == "resolve"
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["ci"]["with"]["ref"] == "${{ needs.resolve.outputs.sha }}"
    assert set(jobs["build"]["needs"]) == {"resolve", "ci"}
    checkouts = [
        s
        for s in jobs["build"]["steps"]
        if s.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 1
    assert checkouts[0]["with"]["ref"] == "${{ needs.resolve.outputs.sha }}"
    guard = next(
        s
        for s in jobs["build"]["steps"]
        if s["name"] == "Verify the version metadata matches the tag"
    )
    assert "if" not in guard
    assert guard["env"]["EXPECTED_VERSION"] == "${{ needs.resolve.outputs.version }}"
    assert jobs["publish"]["needs"] == "build"
    assert jobs["publish"]["environment"] == "pypi"


def _run_step(root, *, step_id=None, name=None, environment):
    _, workflow = _workflows()
    job = "resolve" if step_id is not None else "build"
    step = next(
        s
        for s in workflow["jobs"][job]["steps"]
        if (s.get("id") == step_id if step_id is not None else s.get("name") == name)
    )
    output = root / ("output-" + (step_id or "version"))
    result = subprocess.run(
        ["/bin/bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=root,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "GITHUB_OUTPUT": str(output),
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    values = (
        dict(row.split("=", 1) for row in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, values


@pytest.mark.parametrize("release_ref", ["v0.9.0", "refs/tags/v0.9.0"])
def test_real_manual_request_differs_from_dispatch_branch_but_freezes_release_source(
    tmp_path, release_ref
):
    def git(*arguments):
        return subprocess.check_output(
            ["git", *arguments], cwd=tmp_path, text=True
        ).strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "UncleSaurus")
    git("config", "user.email", "unclesaurus@proton.me")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion = "0.9.0"\n')
    git("add", "pyproject.toml")
    git("commit", "-qm", "release source")
    release_sha = git("rev-parse", "HEAD")
    git("tag", "-a", "v0.9.0", "-m", "immutable release fixture")
    project.write_text('[project]\nversion = "1.0.0"\n')
    git("commit", "-qam", "dispatch branch source")
    assert git("rev-parse", "HEAD") != release_sha

    validation, tag = _run_step(
        tmp_path, step_id="tag", environment={"RELEASE_REF": release_ref}
    )
    assert validation.returncode == 0, validation.stderr
    assert tag == {"ref": "refs/tags/v0.9.0", "version": "0.9.0"}
    wrong, _ = _run_step(
        tmp_path, step_id="source", environment={"EXPECTED_VERSION": tag["version"]}
    )
    assert wrong.returncode != 0
    git("checkout", "-q", "--detach", tag["ref"])
    resolution, source = _run_step(
        tmp_path, step_id="source", environment={"EXPECTED_VERSION": tag["version"]}
    )
    assert resolution.returncode == 0, resolution.stderr
    assert source == {"sha": release_sha}
    # The reusable jobs and build consume this same SHA, not the mutable tag.
    git("checkout", "-q", "main")
    git("tag", "-f", "v0.9.0")
    git("checkout", "-q", "--detach", source["sha"])
    built, _ = _run_step(
        tmp_path,
        name="Verify the version metadata matches the tag",
        environment={"EXPECTED_VERSION": tag["version"]},
    )
    assert built.returncode == 0, built.stderr
    assert git("rev-parse", "HEAD") == release_sha


@pytest.mark.parametrize(
    "release_ref",
    [
        "",
        "main",
        "v1.2",
        "1.2.3",
        "v1.2.3rc1",
        "refs/heads/v1.2.3",
        "v1.2.3\nsha=unreviewed",
        "v1.2.3;touch injected",
        "$(touch injected)",
    ],
)
def test_nonrelease_or_injection_shaped_input_refuses_before_checkout(
    tmp_path, release_ref
):
    result, outputs = _run_step(
        tmp_path, step_id="tag", environment={"RELEASE_REF": release_ref}
    )
    assert result.returncode != 0
    assert not outputs
    assert not (tmp_path / "injected").exists()
