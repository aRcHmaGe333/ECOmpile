from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import load_contract, validate
from .manifest import ManifestArtifact, read_json, write_json, write_public_manifest
from .paths import PipelinePaths, ensure_parent
from .redaction import redact_text


def build_review_dilemmas(
    paths: PipelinePaths,
    *,
    public_safe: bool = True,
    include_non_eligible: bool = False,
    similarity_threshold: float = 0.6,
) -> dict:
    payload = read_json(paths.candidates_path)
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    filtered = [
        item for item in candidates if include_non_eligible or bool(item.get("promotion_eligible", False))
    ]
    dilemmas = _cluster_dilemmas(filtered, similarity_threshold=similarity_threshold)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "include_non_eligible": include_non_eligible,
        "similarity_threshold": similarity_threshold,
        "candidate_count": len(filtered),
        "dilemma_count": len(dilemmas),
        "dilemmas": dilemmas,
    }
    internal_path = paths.review_records_dir / "review_dilemmas.json"
    write_json(internal_path, result)
    public_path = paths.public_artifacts_dir / "review_dilemmas.public.json"
    write_json(public_path, _public_dilemmas(result) if public_safe else result)
    write_public_manifest(
        paths,
        artifacts=[
            ManifestArtifact("review_dilemmas", str(internal_path), "internal"),
            ManifestArtifact("review_dilemmas_public", str(public_path), "public"),
        ],
        public_safe=public_safe,
    )
    return {
        "generated_utc": result["generated_utc"],
        "candidate_count": result["candidate_count"],
        "dilemma_count": result["dilemma_count"],
        "dilemmas_path": str(internal_path),
        "dilemmas_public_path": str(public_path),
    }


def apply_review_decisions(paths: PipelinePaths, decisions_file: Path, *, public_safe: bool = True) -> dict:
    payload = json.loads(decisions_file.read_text(encoding="utf-8"))
    decisions = _expand_to_decisions(payload)

    contract = load_contract("review_decision", paths.root)
    validated: list[dict] = []
    for idx, decision in enumerate(decisions):
        validate(contract, decision, where=f"review_decision[{idx}]")
        validated.append(dict(decision))

    # Last explicit decision wins per candidate_id.
    unique: dict[str, dict] = {}
    for decision in validated:
        unique[str(decision["candidate_id"])] = decision
    validated = list(unique.values())

    staging_rows = _read_tsv_map(paths.staging_index_path)
    approved_ids = [item["candidate_id"] for item in validated if item["decision"] == "approve"]
    rejected_ids = [item["candidate_id"] for item in validated if item["decision"] == "reject"]
    promoted = []

    for kernel_id in approved_ids:
        source = paths.kernels_staging_dir / f"{kernel_id}.kernel.md"
        if not source.exists():
            continue
        target = paths.kernels_active_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if kernel_id in staging_rows:
            promoted.append(staging_rows[kernel_id])

    _merge_active_index(paths.kernels_index_path, promoted)
    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decisions_file": str(decisions_file.resolve()),
        "approved": approved_ids,
        "rejected": rejected_ids,
        "promoted_count": len(promoted),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    review_record = paths.review_records_dir / f"review_{timestamp}.json"
    write_json(review_record, record)
    public_record = paths.public_artifacts_dir / f"review_{timestamp}.public.json"
    write_json(public_record, record if not public_safe else _public_record(record))
    write_public_manifest(
        paths,
        artifacts=[
            ManifestArtifact("review_record", str(review_record), "internal"),
            ManifestArtifact("review_record_public", str(public_record), "public"),
            ManifestArtifact("active_kernel_index", str(paths.kernels_index_path), "public"),
        ],
        public_safe=public_safe,
    )
    return record


def _expand_to_decisions(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return _ensure_list(payload, "decisions")
    if not isinstance(payload, dict):
        raise ValueError("decisions file must contain a list or an object payload")
    if "decisions" in payload:
        return _ensure_list(payload["decisions"], "decisions")
    if "dilemmas" in payload:
        dilemmas = _ensure_list(payload["dilemmas"], "dilemmas")
        expanded: list[dict] = []
        for idx, dilemma in enumerate(dilemmas):
            if not isinstance(dilemma, dict):
                raise ValueError(f"dilemmas[{idx}] must be an object")
            decision = str(dilemma.get("decision", "")).strip().lower()
            if decision not in {"approve", "reject"}:
                continue
            reviewer = str(dilemma.get("reviewer", "")).strip()
            timestamp_utc = str(dilemma.get("timestamp_utc", "")).strip()
            reason = str(dilemma.get("reason", "")).strip()
            candidate_ids = dilemma.get("candidate_ids", [])
            if not isinstance(candidate_ids, list):
                raise ValueError(f"dilemmas[{idx}].candidate_ids must be a list")
            for candidate_id in candidate_ids:
                expanded.append(
                    {
                        "candidate_id": str(candidate_id),
                        "decision": decision,
                        "reviewer": reviewer,
                        "timestamp_utc": timestamp_utc,
                        "reason": reason,
                    }
                )
        return expanded
    raise ValueError("decisions file must contain 'decisions' or 'dilemmas'")


def _ensure_list(value: object, field: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _read_tsv_map(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        row = {header[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(header))}
        rows[row["kernel_id"]] = row
    return rows


def _merge_active_index(path: Path, promoted_rows: list[dict]) -> None:
    existing = _read_tsv_map(path)
    for row in promoted_rows:
        existing[row["kernel_id"]] = row
    header = [
        "kernel_id",
        "platform",
        "context",
        "tokens",
        "primitive",
        "score",
        "promotion_eligible",
        "occurrences",
    ]
    lines = ["\t".join(header)]
    for kernel_id in sorted(existing.keys()):
        row = existing[kernel_id]
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _public_record(record: dict) -> dict:
    clone = dict(record)
    clone["decisions_file"] = Path(clone["decisions_file"]).name
    return clone


def _cluster_dilemmas(candidates: list[dict], *, similarity_threshold: float) -> list[dict]:
    remaining = sorted(candidates, key=lambda item: (-float(item.get("score", 0)), item.get("candidate_id", "")))
    groups: list[list[dict]] = []

    while remaining:
        pivot = remaining.pop(0)
        group = [pivot]
        keep: list[dict] = []
        pivot_ctx = _ctx_key(pivot)
        pivot_tokens = _token_set(pivot)
        pivot_emit = _emit_key(pivot)
        for candidate in remaining:
            same_context = _ctx_key(candidate) == pivot_ctx
            similar_tokens = _jaccard(pivot_tokens, _token_set(candidate)) >= similarity_threshold
            similar_emit = pivot_emit and pivot_emit == _emit_key(candidate)
            if same_context and (similar_tokens or similar_emit):
                group.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        groups.append(group)

    dilemmas: list[dict] = []
    for index, group in enumerate(groups, start=1):
        representative = group[0]
        scores = [float(item.get("score", 0.0)) for item in group]
        total_occurrences = sum(int(item.get("metrics", {}).get("occurrences", 0)) for item in group)
        all_eligible = all(bool(item.get("promotion_eligible", False)) for item in group)
        recommendation = "approve" if all_eligible and (sum(scores) / max(1, len(scores))) >= 0.8 else "defer"
        rationale = (
            "Grouped similar deterministic candidates to minimize manual review load; "
            "one decision can apply to the full dilemma."
        )
        dilemmas.append(
            {
                "dilemma_id": f"DLM_{index:04d}",
                "candidate_ids": [str(item.get("candidate_id", "")) for item in group],
                "representative_candidate_id": str(representative.get("candidate_id", "")),
                "context": representative.get("context", {}),
                "intent_signature_tokens": representative.get("intent_signature_tokens", []),
                "preview_emit_template": [str(line) for line in representative.get("emit_template", [])[:3]],
                "score_range": {"min": round(min(scores), 4), "max": round(max(scores), 4)},
                "occurrences_total": total_occurrences,
                "recommendation": recommendation,
                "reason": rationale,
            }
        )
    return dilemmas


def _ctx_key(candidate: dict) -> str:
    ctx = candidate.get("context", {})
    return f"{ctx.get('platform', '')}|{ctx.get('surface', '')}"


def _token_set(candidate: dict) -> set[str]:
    return {str(token).strip().lower() for token in candidate.get("intent_signature_tokens", []) if str(token).strip()}


def _emit_key(candidate: dict) -> str:
    template = candidate.get("emit_template", [])
    if not template:
        return ""
    return str(template[0]).strip().lower()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _public_dilemmas(payload: dict) -> dict:
    clone = json.loads(json.dumps(payload))
    for dilemma in clone.get("dilemmas", []):
        dilemma["preview_emit_template"] = [redact_text(line) for line in dilemma.get("preview_emit_template", [])]
    return clone
