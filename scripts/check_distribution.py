"""Verify actual release archives contain only canonical source and metadata."""

from __future__ import annotations

import argparse
import logging
import stat
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Mapping
from email.parser import BytesParser
from pathlib import Path


def verify_distribution(artifact: Path, *, source_root: Path) -> None:
    """Check exact names and physical source bytes without extracting archives."""

    project = tomllib.loads(
        (source_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    name = project["name"].replace("-", "_")
    version = project["version"]
    package = source_root / name
    sources = {}
    for path in package.rglob("*.py"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release source must contain regular Python files")
        sources[path.relative_to(source_root).as_posix()] = path.read_bytes()
    if f"{name}/__init__.py" not in sources:
        raise ValueError("Release source package is missing")

    if artifact.name == f"{name}-{version}-py3-none-any.whl":
        metadata = f"{name}-{version}.dist-info"
        expected = {
            *sources,
            *(
                f"{metadata}/{item}"
                for item in (
                    "METADATA",
                    "WHEEL",
                    "RECORD",
                    "entry_points.txt",
                    "licenses/LICENSE",
                )
            ),
        }
        with zipfile.ZipFile(artifact) as archive:
            entries = archive.infolist()
            if (
                len(entries) != len(expected)
                or {entry.filename for entry in entries} != expected
            ):
                raise ValueError("Wheel has missing, duplicate or unexpected paths")
            if any(
                entry.is_dir()
                # Hatch records permission-only modes for generated metadata;
                # explicit symlink/device/directory types are never accepted.
                or stat.S_IFMT(entry.external_attr >> 16) not in {0, stat.S_IFREG}
                for entry in entries
            ):
                raise ValueError("Wheel contains a nonregular member")
            _check_bytes(sources, archive.read)
            if (
                archive.read(f"{metadata}/licenses/LICENSE")
                != (source_root / "LICENSE").read_bytes()
            ):
                raise ValueError("Wheel license differs from canonical root license")
            _check_metadata(archive.read(f"{metadata}/METADATA"), project)
            _check_install_metadata(
                artifact_name=artifact.name,
                source_root=source_root,
                kind="wheel",
                expected_names=tuple(
                    f"{metadata}/{item}"
                    for item in ("METADATA", "WHEEL", "RECORD", "entry_points.txt")
                ),
                read=archive.read,
            )
        return

    if artifact.name != f"{name}-{version}.tar.gz":
        raise ValueError("Unexpected release artifact filename")
    expected_bytes = {
        **sources,
        **{
            item: (source_root / item).read_bytes()
            for item in ("README.md", "LICENSE", "pyproject.toml", ".gitignore")
        },
    }
    prefix = f"{name}-{version}/"
    expected = {prefix + item for item in (*expected_bytes, "PKG-INFO")}
    with tarfile.open(artifact, "r:gz") as archive:
        members = archive.getmembers()
        if (
            len(members) != len(expected)
            or {member.name for member in members} != expected
        ):
            raise ValueError("Sdist has missing, duplicate or unexpected paths")
        if any(not member.isfile() for member in members):
            raise ValueError("Sdist contains a nonregular member")

        def read(name: str) -> bytes:
            with archive.extractfile(prefix + name) as stream:
                return stream.read()

        _check_bytes(expected_bytes, read)
        _check_metadata(read("PKG-INFO"), project)
        _check_install_metadata(
            artifact_name=artifact.name,
            source_root=source_root,
            kind="sdist",
            expected_names=(prefix + "PKG-INFO",),
            read=lambda name: read(name.removeprefix(prefix)),
        )


def _check_bytes(expected: Mapping[str, bytes], read: Callable[[str], bytes]) -> None:
    if any(read(name) != value for name, value in expected.items()):
        raise ValueError("Archive bytes differ from canonical source")


def _check_metadata(value: bytes, project: Mapping[str, object]) -> None:
    metadata = BytesParser().parsebytes(value)
    if metadata.get_all("Name") != [project["name"]] or metadata.get_all("Version") != [
        project["version"]
    ]:
        raise ValueError("Archive metadata differs from canonical project identity")


def _check_install_metadata(
    *,
    artifact_name: str,
    source_root: Path,
    kind: str,
    expected_names: tuple[str, ...],
    read: Callable[[str], bytes],
) -> None:
    """Use the actual configured build tool, not another metadata serializer.

    The independent finite envelope/source checks still constrain what the
    backend may collect. Rebuilding only supplies canonical generated install
    metadata, including dependency/entry-point declarations and RECORD hashes.
    The same uv frontend resolves the source's declared build-system backend;
    the gate needs no globally installed or separately chosen metadata backend.
    """
    with tempfile.TemporaryDirectory(prefix="runpod-canonical-metadata-") as temporary:
        try:
            subprocess.run(
                [
                    "uv",
                    "build",
                    "--wheel" if kind == "wheel" else "--sdist",
                    "--out-dir",
                    temporary,
                ],
                cwd=source_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise ValueError("Canonical install metadata build failed") from error
        canonical = Path(temporary) / artifact_name
        if kind == "wheel":
            with zipfile.ZipFile(canonical) as archive:
                if any(read(name) != archive.read(name) for name in expected_names):
                    raise ValueError(
                        "Archive install metadata differs from canonical build"
                    )
        else:
            with tarfile.open(canonical, "r:gz") as archive:
                for name in expected_names:
                    with archive.extractfile(name) as stream:
                        if read(name) != stream.read():
                            raise ValueError(
                                "Archive install metadata differs from canonical build"
                            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--dist", type=Path, required=True)
    arguments = parser.parse_args()
    artifacts = []
    for path in arguments.dist.iterdir():
        if path.name == ".gitignore":
            # The supported uv frontend creates this exact non-artifact marker.
            # It is never an archive or permission to ignore other contents.
            if path.is_symlink() or not path.is_file() or path.read_bytes() != b"*":
                raise ValueError("Unexpected build frontend marker")
        else:
            artifacts.append(path)
    if len(artifacts) != 2 or any(
        not path.is_file() or path.is_symlink() for path in artifacts
    ):
        raise ValueError("Release directory must contain exactly one wheel and sdist")
    for artifact in artifacts:
        verify_distribution(artifact, source_root=arguments.source_root)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Verified exact wheel and sdist envelope and canonical source bytes"
    )


if __name__ == "__main__":
    main()
