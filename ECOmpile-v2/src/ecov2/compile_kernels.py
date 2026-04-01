from __future__ import annotations

from pathlib import Path

from .manifest import ManifestArtifact, read_json, write_public_manifest
from .paths import PipelinePaths, ensure_parent


def compile_staging_kernels(paths: PipelinePaths, *, public_safe: bool = True) -> int:
    payload = read_json(paths.candidates_path)
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    paths.kernels_staging_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = [
        "kernel_id\tplatform\tcontext\ttokens\tprimitive\tscore\tpromotion_eligible\toccurrences"
    ]
    count = 0
    for candidate in candidates:
        kernel_id = str(candidate["candidate_id"])
        kernel_path = paths.kernels_staging_dir / f"{kernel_id}.kernel.md"
        ensure_parent(kernel_path)
        kernel_path.write_text(render_kernel(candidate), encoding="utf-8")
        primitive = candidate.get("emit_template", [""])[0] if candidate.get("emit_template") else ""
        rows.append(
            "\t".join(
                [
                    kernel_id,
                    str(candidate.get("context", {}).get("platform", "unknown")),
                    str(candidate.get("context", {}).get("surface", "unknown")),
                    ",".join(candidate.get("intent_signature_tokens", [])),
                    primitive.replace("\t", " ").replace("\n", " "),
                    str(candidate.get("score", 0)),
                    str(bool(candidate.get("promotion_eligible", False))).lower(),
                    str(candidate.get("metrics", {}).get("occurrences", 0)),
                ]
            )
        )
        count += 1

    ensure_parent(paths.staging_index_path)
    paths.staging_index_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    write_public_manifest(
        paths,
        artifacts=[
            ManifestArtifact("kernel_staging", str(paths.kernels_staging_dir), "public"),
            ManifestArtifact("kernel_staging_index", str(paths.staging_index_path), "public"),
        ],
        public_safe=public_safe,
    )
    return count


def render_kernel(candidate: dict) -> str:
    tokens = candidate.get("intent_signature_tokens", [])
    emit_template = candidate.get("emit_template", [])
    forbid = candidate.get("forbid", [])
    provenance = candidate.get("provenance", [])
    context = candidate.get("context", {})
    lines = [
        f"# Kernel: {candidate['candidate_id']}",
        "",
        f"KERNEL_ID: {candidate['candidate_id']}",
        "STATUS: staging",
        f"PLATFORM: {context.get('platform', 'unknown')}",
        f"CONTEXT: {context.get('surface', 'unknown')}",
        "",
        "## Intent Signature",
        "Required tokens:",
    ]
    lines.extend(f"- {token}" for token in tokens)
    lines.extend(
        [
            "",
            "## Stop Condition",
            str(candidate.get("stop_condition", "")),
            "",
            "## Emit Template",
            "```text",
        ]
    )
    lines.extend(str(item) for item in emit_template)
    lines.extend(["```", "", "## Forbid"])
    lines.extend(f"- {item}" for item in forbid)
    lines.extend(["", "## Provenance"])
    lines.extend(f"- session={item['session_id']} anchor={item['source_anchor']}" for item in provenance)
    lines.extend(
        [
            "",
            "## Metrics",
            f"- score: {candidate.get('score', 0)}",
            f"- recurrence: {candidate.get('metrics', {}).get('recurrence', 0)}",
            f"- acceptance_signal: {candidate.get('metrics', {}).get('acceptance_signal', 0)}",
            f"- cross_session_reuse: {candidate.get('metrics', {}).get('cross_session_reuse', 0)}",
            f"- determinism_shape: {candidate.get('metrics', {}).get('determinism_shape', 0)}",
            f"- occurrences: {candidate.get('metrics', {}).get('occurrences', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"
