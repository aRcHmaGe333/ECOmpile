"""
OpenAI handoff compiler for ECOmpile case evidence.

Purpose:
- package externally provable evidence from a case log,
- declare the internal telemetry contract required for true introspection,
- generate an email-aligned handoff summary with compensation guardrail.

This script does NOT claim model-internal introspection from chat logs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROMPT_HDR = "## Prompt:"
RESPONSE_HDR = "## Response:"


@dataclass
class Turn:
    role: str
    text: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(read_text(path))


def parse_turns(conversation_md: str) -> List[Turn]:
    lines = conversation_md.splitlines()
    turns: List[Turn] = []
    current_role: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_role
        if current_role is None:
            buffer = []
            return
        turns.append(Turn(role=current_role, text="\n".join(buffer).strip()))
        buffer = []
        current_role = None

    for line in lines:
        marker = line.strip()
        if marker == PROMPT_HDR:
            flush()
            current_role = "prompt"
            continue
        if marker == RESPONSE_HDR:
            flush()
            current_role = "response"
            continue
        if current_role is not None:
            buffer.append(line)
    flush()
    return turns


def group_exchanges(turns: Sequence[Turn]) -> List[Tuple[Turn, Turn]]:
    out: List[Tuple[Turn, Turn]] = []
    i = 0
    while i + 1 < len(turns):
        a = turns[i]
        b = turns[i + 1]
        if a.role == "prompt" and b.role == "response":
            out.append((a, b))
            i += 2
            continue
        i += 1
    return out


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_all(text: str, required: Sequence[str]) -> bool:
    n = normalize(text)
    return all(token.lower() in n for token in required)


def contains_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            return True
    return False


def compact_excerpt(text: str, max_lines: int = 6) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def build_kernel_candidate_md(
    spec: Dict[str, object],
    source_case_rel: str,
) -> str:
    kernel_id = str(spec["kernel_id"])
    platform = str(spec["platform"])
    context = str(spec["context"])
    intent_tokens = list(spec["intent_required"])
    emit_lines = list(spec["emit_template_lines"])
    forbid_list = list(spec["forbid_patterns"])

    req_block = "\n".join(f"- {token}" for token in intent_tokens)
    emit_block = "\n".join(f"- `{line}`" for line in emit_lines)
    forbid_block = "\n".join(f"- {item}" for item in forbid_list)

    return f"""# Kernel Candidate: {kernel_id}

KERNEL_ID: {kernel_id}
STATUS: candidate
PLATFORM: {platform}
CONTEXT: {context}

## Intent Signature
{req_block}

## Stop Condition
On intent match and validated primitive availability, stop exploratory branch generation and emit deterministic primitive template.

## Emit Template
{emit_block}

## Forbid
{forbid_block}

## Source Case
- `{source_case_rel}`
"""


def build_handoff_summary_md(
    spec: Dict[str, object],
    case_id: str,
    evidence_path: Path,
    trace_contract_path: Path,
    kernel_path: Path,
) -> str:
    title = str(spec.get("email_subject_line", "ECOmpile OpenAI Handoff Package"))
    comp = str(spec["compensation_guardrail"])
    kernel_id = str(spec["kernel_id"])

    return f"""# {title}

Case: `{case_id}`
Kernel candidate: `{kernel_id}`

## What Is Already In Place (External)
- Behavioral failure/correction evidence is packaged in `behavioral_evidence.json`.
- Deterministic kernel candidate is packaged in `{kernel_path.name}`.
- Routing intent and forbid lists are explicit and testable.

## What Is Internal To OpenAI
- True model-internal introspection requires internal telemetry.
- Required event contract is packaged in `internal_trace_contract.json`.
- Internal replay/trace validation is the remaining validation step.

## Commercial Guardrail
{comp}

## Package Files
- `{evidence_path.name}`
- `{trace_contract_path.name}`
- `{kernel_path.name}`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile OpenAI handoff package from ECOmpile case log.")
    parser.add_argument("--case-file", required=True, help="Path to case conversation markdown.")
    parser.add_argument("--spec-file", required=True, help="Path to OpenAI handoff spec JSON.")
    default_out = Path(__file__).resolve().parents[2] / "artifacts" / "openai_handoff"
    parser.add_argument("--out-dir", default=str(default_out), help="Output directory for package.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required behavioral evidence or contract fields are missing.",
    )
    args = parser.parse_args()

    case_file = Path(args.case_file).resolve()
    spec_file = Path(args.spec_file).resolve()
    out_dir = Path(args.out_dir).resolve()

    spec = load_json(spec_file)
    conversation = read_text(case_file)
    turns = parse_turns(conversation)
    exchanges = group_exchanges(turns)

    if not exchanges:
        raise ValueError("No prompt/response exchanges detected in case log.")

    required_spec_keys = [
        "kernel_id",
        "platform",
        "context",
        "intent_required",
        "failure_patterns",
        "correction_patterns",
        "emit_template_lines",
        "forbid_patterns",
        "internal_trace_events_required",
        "compensation_guardrail",
    ]
    for key in required_spec_keys:
        if key not in spec:
            raise ValueError(f"Spec missing required key: {key}")

    all_prompts = "\n".join(prompt.text for prompt, _ in exchanges)
    intent_hit = contains_all(all_prompts, list(spec["intent_required"]))

    failure_idx: Optional[int] = None
    correction_idx: Optional[int] = None
    failure_excerpt = ""
    correction_excerpt = ""

    for idx, (_prompt, response) in enumerate(exchanges):
        if failure_idx is None and contains_any_pattern(response.text, list(spec["failure_patterns"])):
            failure_idx = idx
            failure_excerpt = compact_excerpt(response.text)
            continue
        if failure_idx is not None and contains_any_pattern(response.text, list(spec["correction_patterns"])):
            correction_idx = idx
            correction_excerpt = compact_excerpt(response.text)
            break

    behavioral_evidence_ok = intent_hit and failure_idx is not None and correction_idx is not None
    trace_contract_ok = bool(spec["internal_trace_events_required"])

    if args.strict and not (behavioral_evidence_ok and trace_contract_ok):
        raise RuntimeError(
            "Strict mode failed: missing behavioral evidence chain or internal trace contract requirements."
        )

    case_id = case_file.parent.name
    case_out_dir = out_dir / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    behavioral_evidence = {
        "case_id": case_id,
        "kernel_id": spec["kernel_id"],
        "introspection_level": "behavioral_only",
        "requires_internal_telemetry_for_true_introspection": True,
        "intent_hit": intent_hit,
        "failure_exchange_index": failure_idx,
        "correction_exchange_index": correction_idx,
        "failure_excerpt": failure_excerpt,
        "correction_excerpt": correction_excerpt,
        "strict_behavioral_evidence_ok": behavioral_evidence_ok,
        "source_case_file": str(case_file),
        "source_spec_file": str(spec_file),
    }

    trace_contract = {
        "contract_id": f"{spec['kernel_id']}_INTERNAL_TRACE_CONTRACT",
        "kernel_id": spec["kernel_id"],
        "required_events": list(spec["internal_trace_events_required"]),
        "required_event_fields": [
            "run_id",
            "timestamp_utc",
            "event_type",
            "model_build",
            "context",
            "decision_payload",
        ],
        "determinism_requirements": [
            "Pinned model build/version",
            "Pinned system instructions and policy snapshot",
            "Pinned decoding parameters or replay-grade event capture",
            "Consistent run_id across prompt, trace events, and output",
        ],
    }

    source_case_rel = f"cases/{case_file.parent.name}/{case_file.name}"
    kernel_md = build_kernel_candidate_md(spec, source_case_rel)

    evidence_path = case_out_dir / "behavioral_evidence.json"
    trace_contract_path = case_out_dir / "internal_trace_contract.json"
    kernel_path = case_out_dir / f"{spec['kernel_id']}.kernel.candidate.md"

    write_text(evidence_path, json.dumps(behavioral_evidence, indent=2))
    write_text(trace_contract_path, json.dumps(trace_contract, indent=2))
    write_text(kernel_path, kernel_md)

    summary_md = build_handoff_summary_md(spec, case_id, evidence_path, trace_contract_path, kernel_path)
    summary_path = case_out_dir / "handoff_summary.md"
    write_text(summary_path, summary_md)

    print(f"[handoff] behavioral evidence: {evidence_path}")
    print(f"[handoff] trace contract: {trace_contract_path}")
    print(f"[handoff] kernel candidate: {kernel_path}")
    print(f"[handoff] summary: {summary_path}")
    print(f"[handoff] strict_behavioral_evidence_ok={behavioral_evidence_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
