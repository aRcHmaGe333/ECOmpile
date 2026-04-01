from __future__ import annotations

import hashlib
import re

from .contracts import load_contract, validate
from .manifest import ManifestArtifact, read_jsonl, write_jsonl, write_public_manifest
from .paths import PipelinePaths
from .redaction import redact_text


NEGATIVE_MARKERS = (
    "wrong",
    "no,",
    "no.",
    "not what",
    "idiot",
    "hallucinating",
    "waste",
    "doesn't work",
    "didn't",
    "useless",
    "fail",
    "stop",
)
COMMAND_PREFIXES = (
    "cmd /c",
    "icacls",
    "python",
    "python3",
    "powershell",
    "git ",
    "cargo ",
    "pytest",
    "Get-ChildItem",
    "Select-String",
    "mkdir",
    "ls ",
)
CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", flags=re.DOTALL)


def extract_outcomes(paths: PipelinePaths, *, public_safe: bool = True) -> int:
    sessions = read_jsonl(paths.normalized_path)
    contract = load_contract("extracted_outcome", paths.root)
    outcomes: list[dict] = []

    for session in sessions:
        outcomes.extend(extract_from_session(session))

    validated: list[dict] = []
    for index, outcome in enumerate(outcomes):
        outcome["outcome_id"] = outcome_id(outcome, index)
        validate(contract, outcome, where=f"extracted_outcome[{index}]")
        validated.append(outcome)

    write_jsonl(paths.outcomes_path, validated)
    public_rows = [redact_outcome(item) if public_safe else item for item in validated]
    public_path = paths.public_artifacts_dir / "extracted_outcomes.public.jsonl"
    write_jsonl(public_path, public_rows)
    write_public_manifest(
        paths,
        artifacts=[
            ManifestArtifact("extracted_outcomes", str(paths.outcomes_path), "internal"),
            ManifestArtifact("extracted_outcomes_public", str(public_path), "public"),
        ],
        public_safe=public_safe,
    )
    return len(validated)


def extract_from_session(session: dict) -> list[dict]:
    messages = session.get("messages", [])
    out: list[dict] = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        intent_message = _find_previous_user(messages, idx)
        if intent_message is None:
            continue
        payloads = _candidate_payloads(message.get("text", ""))
        if not payloads:
            continue
        acceptance = _acceptance_signal(messages, idx)
        for payload, candidate_type in payloads:
            out.append(
                {
                    "outcome_id": "",
                    "session_id": session["session_id"],
                    "family": session["family"],
                    "intent_text": intent_message.get("text", ""),
                    "context": _context_for_family(session.get("family", "")),
                    "candidate_type": candidate_type,
                    "candidate_payload": payload,
                    "source_anchor": message.get("anchor", "line:0"),
                    "acceptance_signal": acceptance,
                    "determinism_shape": _determinism_shape(candidate_type, payload),
                }
            )
    return out


def _find_previous_user(messages: list[dict], index: int) -> dict | None:
    for cursor in range(index - 1, -1, -1):
        if messages[cursor].get("role") == "user":
            return messages[cursor]
    return None


def _acceptance_signal(messages: list[dict], assistant_index: int) -> float:
    for cursor in range(assistant_index + 1, len(messages)):
        if messages[cursor].get("role") != "user":
            continue
        text = str(messages[cursor].get("text", "")).lower()
        if any(marker in text for marker in NEGATIVE_MARKERS):
            return 0.2
        return 0.9
    return 0.75


def _candidate_payloads(text: str) -> list[tuple[str, str]]:
    payloads: list[tuple[str, str]] = []
    for block in CODE_FENCE_RE.findall(text):
        cleaned = block.strip()
        if not cleaned:
            continue
        candidate_type = "command" if _looks_command(cleaned) else "template"
        payloads.append((cleaned, candidate_type))
    if payloads:
        return payloads

    if _looks_procedure(text):
        procedure = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        payloads.append((procedure, "procedure"))
        return payloads

    command_lines = [line.strip() for line in text.splitlines() if _looks_command(line.strip())]
    if command_lines:
        payloads.append(("\n".join(command_lines), "command"))
    return payloads


def _looks_command(text: str) -> bool:
    lowered = text.lower()
    if any(lowered.startswith(prefix.lower()) for prefix in COMMAND_PREFIXES):
        return True
    if "\n" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return bool(lines) and all(_looks_command(line) for line in lines)
    return False


def _looks_procedure(text: str) -> bool:
    steps = [line for line in text.splitlines() if re.match(r"^\s*\d+\.", line)]
    return len(steps) >= 2


def _determinism_shape(candidate_type: str, payload: str) -> float:
    if candidate_type == "command":
        if "<" in payload and ">" in payload:
            return 0.9
        return 1.0
    if candidate_type == "template":
        return 0.8
    return 0.65


def _context_for_family(family: str) -> dict:
    family_lower = family.lower()
    if family_lower == "vscode":
        return {"platform": "windows", "surface": "ide"}
    if family_lower == "claude":
        return {"platform": "windows", "surface": "claude_project"}
    return {"platform": "windows", "surface": "cli"}


def outcome_id(outcome: dict, index: int) -> str:
    digest = hashlib.sha1(
        f"{outcome['session_id']}|{outcome['source_anchor']}|{outcome['candidate_payload']}|{index}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"outcome_{digest[:12]}"


def redact_outcome(outcome: dict) -> dict:
    clone = dict(outcome)
    clone["intent_text"] = redact_text(str(clone["intent_text"]))
    clone["candidate_payload"] = redact_text(str(clone["candidate_payload"]))
    return clone
