"""Build real archives with unrelated local metadata and inspect their bytes."""

import base64
import csv
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.check_distribution import verify_distribution


def _build(root, output, *, targets=("sdist", "wheel")):
    command = [sys.executable, "-m", "hatchling", "build", "-d", str(output)]
    for target in targets:
        command.extend(("-t", target))
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_gate(root, output):
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_distribution.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-root",
            str(root),
            "--dist",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture
def native_distribution(tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "project"
    root.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore"):
        shutil.copyfile(source / name, root / name)
    shutil.copytree(
        source / "kestrel_cloud_runpod",
        root / "kestrel_cloud_runpod",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # Neither a single virtualenv basename nor VCS ignore rules define the
    # trust boundary. Include both hidden/ignored and ordinary unrelated trees.
    for relative in (".environment-other/dependency", "unrelated/nested"):
        nested = root / relative
        nested.mkdir(parents=True)
        for name in ("README.md", "LICENSE", "pyproject.toml"):
            (nested / name).write_text("unrelated-build-sentinel", encoding="utf-8")
    output = tmp_path / "dist"
    _build(root, output)
    return root, output


def test_ci_and_publication_use_the_same_gate_before_artifact_upload():
    source = Path(__file__).resolve().parents[1]
    expected = "uv run --no-project python scripts/check_distribution.py --dist dist"
    for workflow, job in (("ci.yml", "build-check"), ("publish.yml", "build")):
        definition = yaml.safe_load(
            (source / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        )
        steps = definition["jobs"][job]["steps"]
        (gate,) = [
            index for index, step in enumerate(steps) if step.get("run") == expected
        ]
        (build,) = [
            index for index, step in enumerate(steps) if step.get("run") == "uv build"
        ]
        assert build < gate
        for index, step in enumerate(steps):
            if step.get("uses", "").startswith("actions/upload-artifact@"):
                assert gate < index


def test_real_sdist_rebuilds_the_exact_wheel(native_distribution, tmp_path):
    root, output = native_distribution
    artifact = next(output.glob("*.tar.gz"))
    verify_distribution(artifact, source_root=root)
    extracted = tmp_path / "rebuild"
    with tarfile.open(artifact) as archive:
        archive.extractall(extracted, filter="data")
    _build(next(extracted.iterdir()), tmp_path / "rebuilt-dist", targets=("wheel",))
    assert (
        next(output.glob("*.whl")).read_bytes()
        == next((tmp_path / "rebuilt-dist").glob("*.whl")).read_bytes()
    )


def test_real_sdist_and_wheel_cannot_collect_nested_metadata(native_distribution):
    root, output = native_distribution
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        paths = [member.name.split("/", 1)[1] for member in archive.getmembers()]
        assert all(
            not path.startswith((".environment-other/", "unrelated/")) for path in paths
        )
        for name in ("README.md", "LICENSE", "pyproject.toml"):
            (member,) = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/" + name)
            ]
            with archive.extractfile(member) as stream:
                assert stream.read() == (root / name).read_bytes()
    with zipfile.ZipFile(next(output.glob("*.whl"))) as archive:
        assert all(
            not name.startswith((".environment-other/", "unrelated/"))
            for name in archive.namelist()
        )
        for path in (root / "kestrel_cloud_runpod").glob("*.py"):
            assert (
                hashlib.sha256(
                    archive.read("kestrel_cloud_runpod/" + path.name)
                ).digest()
                == hashlib.sha256(path.read_bytes()).digest()
            )


def test_real_archive_envelope_verifier_and_cli(native_distribution):
    root, output = native_distribution
    for artifact in output.iterdir():
        verify_distribution(artifact, source_root=root)
    checked = _run_gate(root, output)
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize(
    "mutation",
    ["dependencies", "entry_points", "wheel_tag", "record", "sdist_dependencies"],
)
def test_install_metadata_cannot_drift_with_valid_source_and_record(
    native_distribution, tmp_path, mutation
):
    root, output = native_distribution
    source = next(
        output.glob("*.tar.gz" if mutation == "sdist_dependencies" else "*.whl")
    )
    verify_distribution(source, source_root=root)
    changed = tmp_path / "install-metadata" / source.name
    changed.parent.mkdir()
    if mutation == "sdist_dependencies":
        with tarfile.open(source) as archive:
            entries = []
            for member in archive.getmembers():
                with archive.extractfile(member) as stream:
                    entries.append((member, stream.read()))
        with tarfile.open(changed, "w:gz") as archive:
            for member, value in entries:
                if member.name.endswith("/PKG-INFO"):
                    assert b"Requires-Dist:" in value
                    value = b"\n".join(
                        line
                        for line in value.split(b"\n")
                        if not line.startswith(b"Requires-Dist:")
                    )
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    else:
        with zipfile.ZipFile(source) as archive:
            entries = {
                entry.filename: (entry, archive.read(entry.filename))
                for entry in archive.infolist()
            }
        metadata = next(name for name in entries if name.endswith("/METADATA"))
        folder = metadata.rsplit("/", 1)[0]
        target = {
            "dependencies": metadata,
            "entry_points": folder + "/entry_points.txt",
            "wheel_tag": folder + "/WHEEL",
            "record": folder + "/RECORD",
        }[mutation]
        info, value = entries[target]
        if mutation == "dependencies":
            assert b"Requires-Dist:" in value
            value = b"\n".join(
                line
                for line in value.split(b"\n")
                if not line.startswith(b"Requires-Dist:")
            )
        elif mutation == "wheel_tag":
            assert b"Tag: py3-none-any" in value
            value = value.replace(
                b"Tag: py3-none-any", b"Tag: py3-none-missing_platform"
            )
        else:
            value = b""
        entries[target] = info, value
        if mutation != "record":
            # Preserve valid RECORD hashes, so a checksum-only patch cannot
            # falsely satisfy the semantic metadata corruption regressions.
            record = io.StringIO(newline="")
            writer = csv.writer(record, lineterminator="\n")
            for name, (_entry, body) in entries.items():
                if name.endswith("/RECORD"):
                    writer.writerow((name, "", ""))
                else:
                    digest = (
                        base64.urlsafe_b64encode(hashlib.sha256(body).digest())
                        .rstrip(b"=")
                        .decode("ascii")
                    )
                    writer.writerow((name, "sha256=" + digest, len(body)))
            info, _value = entries[folder + "/RECORD"]
            entries[folder + "/RECORD"] = info, record.getvalue().encode("utf-8")
        with zipfile.ZipFile(changed, "w") as archive:
            for entry, value in entries.values():
                archive.writestr(entry, value)
    with pytest.raises(ValueError, match="install metadata"):
        verify_distribution(changed, source_root=root)


def test_metadata_frontend_absence_refuses_without_backend_fallback(
    native_distribution, monkeypatch
):
    root, output = native_distribution
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ValueError, match="Canonical install metadata build failed"):
        verify_distribution(next(output.glob("*.whl")), source_root=root)


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_native_frontend_failure_refuses_install_metadata(
    native_distribution, tmp_path, monkeypatch, failure
):
    root, output = native_distribution
    verify_distribution(next(output.glob("*.whl")), source_root=root)
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    executable = frontend / "uv"
    operation = (
        "import sys; sys.exit(17)"
        if failure == "exit"
        else "import time; time.sleep(5)"
    )
    executable.write_text(f"#!{sys.executable}\n{operation}\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(frontend))
    if failure == "timeout":
        # Only bound the genuine child process's wait; never synthesize its
        # result/error or bypass native subprocess cleanup.
        original_run = subprocess.run

        def bounded_run(*args, **kwargs):
            return original_run(*args, **{**kwargs, "timeout": 0.1})

        monkeypatch.setattr(subprocess, "run", bounded_run)
    with pytest.raises(
        ValueError, match="Canonical install metadata build failed"
    ) as error:
        verify_distribution(next(output.glob("*.whl")), source_root=root)
    if failure == "exit":
        assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
        assert error.value.__cause__.returncode == 17
    else:
        assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)


@pytest.mark.parametrize("marker", ["uv_marker", "modified_marker", "linked_marker"])
def test_only_exact_regular_uv_frontend_marker_is_admitted(native_distribution, marker):
    root, output = native_distribution
    path = output / ".gitignore"
    if marker == "linked_marker":
        path.symlink_to(root / "README.md")
    else:
        path.write_bytes(b"*" if marker == "uv_marker" else b"foreign-build-sentinel")
    checked = _run_gate(root, output)
    if marker == "uv_marker":
        assert checked.returncode == 0, checked.stdout + checked.stderr
    else:
        assert checked.returncode != 0
        assert "Unexpected build frontend marker" in checked.stderr


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "duplicate", "symlink", "bytes", "version"]
)
def test_actual_mutated_archive_cannot_pass_gate(
    native_distribution, tmp_path, kind, mutation
):
    root, output = native_distribution
    source = next(output.glob("*.whl" if kind == "wheel" else "*.tar.gz"))
    verify_distribution(source, source_root=root)
    changed = tmp_path / "mutated" / source.name
    changed.parent.mkdir()
    if kind == "wheel":
        with zipfile.ZipFile(source) as archive:
            entries = [
                (entry, archive.read(entry.filename)) for entry in archive.infolist()
            ]
        target = "kestrel_cloud_runpod/__init__.py"
        metadata = next(
            entry.filename
            for entry, _ in entries
            if entry.filename.endswith("/METADATA")
        )
        with zipfile.ZipFile(changed, "w") as archive:
            for entry, value in entries:
                if entry.filename == target:
                    if mutation == "missing":
                        continue
                    if mutation == "bytes":
                        value += b"\n# changed source bytes\n"
                    if mutation == "symlink":
                        entry.external_attr = 0o120777 << 16
                if entry.filename == metadata and mutation == "version":
                    value = value.replace(b"\nVersion: ", b"\nVersion: wrong-", 1)
                archive.writestr(entry, value)
                if mutation == "duplicate" and entry.filename == target:
                    with pytest.warns(UserWarning, match="Duplicate name:"):
                        archive.writestr(entry, value)
            if mutation == "extra":
                archive.writestr("unrelated/README.md", b"unrelated-build-sentinel")
    else:
        with tarfile.open(source) as archive:
            entries = []
            for member in archive.getmembers():
                with archive.extractfile(member) as stream:
                    entries.append((member, stream.read()))
        target = next(
            member.name
            for member, _ in entries
            if member.name.endswith("/kestrel_cloud_runpod/__init__.py")
        )
        with tarfile.open(changed, "w:gz") as archive:
            for member, value in entries:
                if member.name == target:
                    if mutation == "missing":
                        continue
                    if mutation == "bytes":
                        value += b"\n# changed source bytes\n"
                    if mutation == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "outside-source"
                        value = b""
                if member.name.endswith("/PKG-INFO") and mutation == "version":
                    value = value.replace(b"\nVersion: ", b"\nVersion: wrong-", 1)
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
                if mutation == "duplicate" and member.name == target:
                    archive.addfile(member, io.BytesIO(value))
            if mutation == "extra":
                member = tarfile.TarInfo(
                    target.split("/", 1)[0] + "/unrelated/README.md"
                )
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
    expected_error = {
        "extra": "unexpected paths",
        "missing": "missing",
        "duplicate": "duplicate",
        "symlink": "nonregular",
        "bytes": "canonical source",
        "version": "metadata differs",
    }[mutation]
    with pytest.raises(ValueError, match=expected_error):
        verify_distribution(changed, source_root=root)


@pytest.mark.parametrize("mutation", ["extra_artifact", "renamed", "source_symlink"])
def test_cli_and_source_boundary_fail_closed(native_distribution, mutation):
    root, output = native_distribution
    artifact = next(output.glob("*.whl"))
    if mutation == "extra_artifact":
        (output / "unrelated.txt").write_text(
            "unrelated-build-sentinel", encoding="utf-8"
        )
    elif mutation == "renamed":
        artifact.rename(output / "foreign.whl")
    else:
        (root / "kestrel_cloud_runpod" / "foreign.py").symlink_to(root / "README.md")
    checked = _run_gate(root, output)
    assert checked.returncode != 0
