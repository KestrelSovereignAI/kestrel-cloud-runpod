"""Validate the pinned Runpod v2 OpenAPI file and report upstream drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "vendor" / "runpod-v2-openapi.yaml"
LOCK_PATH = REPO_ROOT / "vendor" / "runpod-v2-openapi.lock.json"
USER_AGENT = "kestrel-cloud-runpod-openapi-drift/1.0"
HTTP_METHODS = frozenset({"get", "post", "patch", "put", "delete"})


class _OpenAPI31Loader(yaml.SafeLoader):
    """Parse OpenAPI's YAML 1.2 scalars instead of PyYAML's YAML 1.1 aliases."""


_OpenAPI31Loader.yaml_implicit_resolvers = {
    first_character: [
        resolver
        for resolver in resolvers
        if not (
            resolver[0] == "tag:yaml.org,2002:bool"
            and first_character in {"O", "o", "Y", "y", "N", "n"}
        )
    ]
    for first_character, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load_openapi(source: str | bytes) -> dict[str, Any]:
    """Load an OpenAPI 3.1 YAML document using YAML 1.2 boolean semantics."""

    document = yaml.load(source, Loader=_OpenAPI31Loader)
    if not isinstance(document, dict):
        raise ValueError("Runpod OpenAPI schema is not a YAML object")
    return document


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_lock() -> dict[str, str]:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    required = {"source", "retrieved_at", "sha256"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"{LOCK_PATH} must contain {sorted(required)}")
    return {key: str(value[key]) for key in required}


def check_pin() -> None:
    lock = read_lock()
    actual = sha256(SCHEMA_PATH.read_bytes())
    if actual != lock["sha256"]:
        raise ValueError(
            f"Pinned schema checksum mismatch: lock={lock['sha256']} actual={actual}"
        )
    document = load_openapi(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("openapi") != "3.1.0":
        raise ValueError(
            "Pinned Runpod schema is not the expected OpenAPI 3.1 document"
        )


def fetch_live(source: str) -> bytes:
    request = Request(
        source, headers={"User-Agent": USER_AGENT, "Accept": "application/yaml"}
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Runpod schema fetch returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Runpod schema fetch failed: {exc.reason}") from exc


def semantic_diff(pinned: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Return high-signal API changes; the full checksum still gates every drift."""

    changes: list[str] = []
    pinned_ops = _operations(pinned)
    live_ops = _operations(live)
    for operation in sorted(pinned_ops - live_ops):
        changes.append(f"REMOVED operation {operation}")
    for operation in sorted(live_ops - pinned_ops):
        changes.append(f"ADDED operation {operation}")

    pinned_schemas = _schemas(pinned)
    live_schemas = _schemas(live)
    for name in sorted(pinned_schemas.keys() - live_schemas.keys()):
        changes.append(f"REMOVED schema {name}")
    for name in sorted(live_schemas.keys() - pinned_schemas.keys()):
        changes.append(f"ADDED schema {name}")
    for name in sorted(pinned_schemas.keys() & live_schemas.keys()):
        before = pinned_schemas[name]
        after = live_schemas[name]
        before_required = set(before.get("required") or ())
        after_required = set(after.get("required") or ())
        for field in sorted(after_required - before_required):
            changes.append(f"NEW required field {name}.{field}")
        before_enum = set(before.get("enum") or ())
        after_enum = set(after.get("enum") or ())
        for value in sorted(before_enum - after_enum, key=str):
            changes.append(f"REMOVED enum value {name}.{value}")
        for value in sorted(after_enum - before_enum, key=str):
            changes.append(f"ADDED enum value {name}.{value}")
    return changes


def _operations(document: dict[str, Any]) -> set[str]:
    paths = document.get("paths") or {}
    if not isinstance(paths, dict):
        return set()
    return {
        f"{method.upper()} {path}"
        for path, item in paths.items()
        if isinstance(item, dict)
        for method in item
        if method in HTTP_METHODS
    }


def _schemas(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schemas = (document.get("components") or {}).get("schemas") or {}
    if not isinstance(schemas, dict):
        return {}
    return {key: value for key, value in schemas.items() if isinstance(value, dict)}


def check_live() -> None:
    lock = read_lock()
    pinned_bytes = SCHEMA_PATH.read_bytes()
    live_bytes = fetch_live(lock["source"])
    live_hash = sha256(live_bytes)
    if live_hash == lock["sha256"]:
        print(f"Runpod v2 OpenAPI pin is current ({live_hash})")
        return
    pinned = load_openapi(pinned_bytes)
    live = load_openapi(live_bytes)
    changes = semantic_diff(pinned, live)
    print(
        f"Runpod v2 OpenAPI drift detected: pinned={lock['sha256']} live={live_hash}",
        file=sys.stderr,
    )
    for change in changes[:100]:
        print(f"- {change}", file=sys.stderr)
    if len(changes) > 100:
        print(f"- ... {len(changes) - 100} more semantic changes", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-pin", action="store_true")
    parser.add_argument("--check-live", action="store_true")
    args = parser.parse_args()
    if not args.check_pin and not args.check_live:
        parser.error("choose --check-pin and/or --check-live")
    if args.check_pin:
        check_pin()
        print("Pinned Runpod v2 OpenAPI checksum and document are valid")
    if args.check_live:
        check_live()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
