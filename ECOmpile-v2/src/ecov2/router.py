from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .paths import PipelinePaths


TOKEN_RE = re.compile(r"[a-zA-Z0-9_\\/-]+")


@dataclass(frozen=True)
class RouteResult:
    status: str
    reason: str
    kernel_id: str | None
    emitted_template: list[str]
    score: int

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "kernel_id": self.kernel_id,
            "emitted_template": self.emitted_template,
            "score": self.score,
        }


def route_prompt(paths: PipelinePaths, prompt: str, *, platform: str = "windows", surface: str = "cli") -> RouteResult:
    rows = _read_index(paths.kernels_index_path)
    if not rows:
        return RouteResult(
            status="fallback",
            reason="No active kernels available.",
            kernel_id=None,
            emitted_template=[],
            score=0,
        )
    prompt_tokens = {item.lower() for item in TOKEN_RE.findall(prompt)}
    best: tuple[int, dict] | None = None
    for row in rows:
        if row["platform"] and row["platform"].lower() != platform.lower():
            continue
        if row.get("context") and row["context"].lower() != surface.lower():
            continue
        tokens = [token.strip().lower() for token in row["tokens"].split(",") if token.strip()]
        if not tokens:
            continue
        matches = sum(1 for token in tokens if token in prompt_tokens)
        if matches == len(tokens):
            score = matches * 10
            if best is None or score > best[0]:
                best = (score, row)
    if best is None:
        return RouteResult(
            status="fallback",
            reason="No kernel matched all required intent tokens.",
            kernel_id=None,
            emitted_template=[],
            score=0,
        )
    row = best[1]
    template = _read_emit_template(paths.kernels_active_dir / f"{row['kernel_id']}.kernel.md")
    if not template and row.get("primitive"):
        template = [row["primitive"]]
    return RouteResult(
        status="kernel_hit",
        reason="Deterministic kernel match; exploration stopped.",
        kernel_id=row["kernel_id"],
        emitted_template=template,
        score=best[0],
    )


def dump_route_json(result: RouteResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=True)


def _read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows: list[dict] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        row = {header[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(header))}
        rows.append(row)
    return rows


def _read_emit_template(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_emit = False
    in_code = False
    for line in lines:
        if line.strip().startswith("## Emit Template"):
            in_emit = True
            continue
        if not in_emit:
            continue
        if line.strip().startswith("## ") and not line.strip().startswith("## Emit Template"):
            break
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code and line.strip():
            output.append(line.rstrip())
    return output
