from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import load_contract, validate
from .paths import PipelinePaths, ensure_parent


@dataclass(frozen=True)
class ManifestArtifact:
    kind: str
    path: str
    sensitivity: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_public_manifest(paths: PipelinePaths, *, artifacts: list[ManifestArtifact], public_safe: bool) -> None:
    existing_artifacts: list[dict] = []
    if paths.public_manifest_path.exists():
        try:
            existing_payload = read_json(paths.public_manifest_path)
            if isinstance(existing_payload, dict):
                found = existing_payload.get("artifacts", [])
                if isinstance(found, list):
                    existing_artifacts = [item for item in found if isinstance(item, dict)]
        except json.JSONDecodeError:
            existing_artifacts = []

    merged = {(item.get("kind"), item.get("path")): item for item in existing_artifacts}
    for item in artifacts:
        merged[(item.kind, item.path)] = {
            "kind": item.kind,
            "path": item.path,
            "sensitivity": item.sensitivity,
        }

    payload = {
        "generated_utc": utc_now(),
        "public_safe": public_safe,
        "artifacts": sorted(
            merged.values(),
            key=lambda item: (str(item.get("sensitivity", "")), str(item.get("kind", "")), str(item.get("path", ""))),
        ),
    }
    contract = load_contract("public_artifact_manifest", paths.root)
    validate(contract, payload, where="public_artifact_manifest")
    write_json(paths.public_manifest_path, payload)
