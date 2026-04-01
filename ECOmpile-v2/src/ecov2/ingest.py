from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .contracts import ContractError, load_contract, validate
from .manifest import ManifestArtifact, write_jsonl, write_public_manifest
from .paths import PipelinePaths
from .redaction import redact_path, redact_text


SUPPORTED_FAMILIES = ("codex", "vscode", "claude")
ROLE_HEADER_RE = re.compile(r"^\*\*(User|Assistant):\*\*(.*)$")


@dataclass(frozen=True)
class IngestResult:
    source_root: Path
    records_count: int
    skipped_files: int
    duplicate_sessions_skipped: int


def ingest_linked_sessions(paths: PipelinePaths, source_root: Path, public_safe: bool = True) -> IngestResult:
    contract = load_contract("normalized_session", paths.root)
    records: list[dict] = []
    public_rows: list[dict] = []
    skipped = 0
    dedupe_hashes: set[str] = set()
    duplicate_sessions = 0

    for family, file_path in iter_transcript_files(source_root):
        try:
            record = parse_transcript_file(file_path, default_family=family)
            validate(contract, record, where=f"normalized_session:{file_path.name}")
        except (ContractError, ValueError):
            skipped += 1
            continue
        stable_key = stable_session_hash(record)
        if stable_key in dedupe_hashes:
            duplicate_sessions += 1
            continue
        dedupe_hashes.add(stable_key)
        records.append(record)
        if public_safe:
            public_rows.append(redacted_record(record))
        else:
            public_rows.append(record)

    write_jsonl(paths.normalized_path, records)
    public_path = paths.public_artifacts_dir / "normalized_sessions.public.jsonl"
    write_jsonl(public_path, public_rows)
    write_public_manifest(
        paths,
        artifacts=[
            ManifestArtifact("normalized_sessions", str(paths.normalized_path), "internal"),
            ManifestArtifact("normalized_sessions_public", str(public_path), "public"),
        ],
        public_safe=public_safe,
    )
    return IngestResult(
        source_root=source_root,
        records_count=len(records),
        skipped_files=skipped,
        duplicate_sessions_skipped=duplicate_sessions,
    )


def iter_transcript_files(source_root: Path):
    for family in SUPPORTED_FAMILIES:
        family_dir = source_root / family
        if not family_dir.exists():
            continue
        for path in family_dir.rglob("*.md"):
            yield family, path


def parse_transcript_file(path: Path, *, default_family: str = "codex") -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    title = _extract_title(lines) or path.stem
    meta = _extract_meta(lines)
    messages = _extract_messages(lines)
    if len(messages) < 2:
        raise ValueError(f"insufficient messages in {path}")

    family = str(meta.get("Family", default_family)).strip().lower()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported family {family}")

    session_id = _extract_session_id(path)
    return {
        "session_id": session_id,
        "family": family,
        "provider": str(meta.get("Provider", "unknown")).strip() or "unknown",
        "updated_at": str(meta.get("Updated", "unknown")).strip() or "unknown",
        "title": title,
        "scope": str(meta.get("Scope", "")),
        "source_path": str(path.resolve()),
        "messages": messages,
    }


def _extract_title(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_meta(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        if line.startswith("## Conversation"):
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"Provider", "Family", "Scope", "Updated"}:
            out[key] = value
    return out


def _extract_messages(lines: list[str]) -> list[dict]:
    messages: list[dict] = []
    role: str | None = None
    buffer: list[str] = []
    anchor = "line:1"
    conversation_started = False

    def flush() -> None:
        nonlocal buffer, role, anchor
        if role is None:
            buffer = []
            return
        text = "\n".join(buffer).strip()
        if text:
            messages.append({"role": role, "text": text, "anchor": anchor})
        buffer = []
        role = None
        anchor = "line:1"

    for index, line in enumerate(lines, start=1):
        if not conversation_started:
            if line.startswith("## Conversation"):
                conversation_started = True
            continue
        if line.startswith("Original file:"):
            break
        stripped = line.strip()
        role_match = ROLE_HEADER_RE.match(stripped)
        if role_match:
            flush()
            role = role_match.group(1).lower()
            anchor = f"line:{index}"
            inline = role_match.group(2).strip()
            if inline:
                buffer.append(inline)
            continue
        if role is not None:
            buffer.append(line)
    flush()
    return messages


def _extract_session_id(path: Path) -> str:
    suffix = path.stem.rsplit("__", 1)
    if len(suffix) == 2 and suffix[1]:
        return suffix[1]
    return path.stem


def stable_session_hash(record: dict) -> str:
    messages = record.get("messages", [])
    normalized_parts: list[str] = [
        str(record.get("family", "")).strip().lower(),
        str(record.get("provider", "")).strip().lower(),
    ]
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        text = str(msg.get("text", ""))
        text = re.sub(r"\s+", " ", text).strip().lower()
        normalized_parts.append(f"{role}:{text}")
    raw = "|".join(normalized_parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def redacted_record(record: dict) -> dict:
    clone = dict(record)
    clone["source_path"] = redact_path(str(clone["source_path"]))
    clone["messages"] = [
        {
            "role": msg["role"],
            "anchor": msg["anchor"],
            "text": redact_text(msg["text"]),
        }
        for msg in clone["messages"]
    ]
    if "title" in clone:
        clone["title"] = redact_text(str(clone["title"]))
    return clone
