from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "artifact_harvest" / "artifacts.jsonl"
DEFAULT_ROOTS_FILE = Path(__file__).with_name("artifact_source_roots.sample.json")

ALLOWED_EXTS = {".md", ".txt", ".log", ".json", ".jsonl"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "state", "history"}

TRAIT_SIGNALS = [
    "default",
    "prefer",
    "default to",
    "should",
    "should be",
    "use",
    "preserve",
    "strengthen",
    "promote",
    "lock",
    "bounded",
    "provenance",
    "trait",
]

FAILURE_SIGNALS = [
    "failure",
    "anti-pattern",
    "avoid",
    "do not",
    "should not",
    "don't",
    "misleading",
    "distortion",
    "waste",
    "garbage",
    "wrong",
    "overstated",
]

PREVENTION_SIGNALS = [
    "prevent",
    "prevention",
    "correction",
    "loss",
    "protect",
    "avoid",
    "guard",
]

RULE_SIGNALS = [
    "rule",
    "guideline",
    "threshold",
    "task",
    "policy",
    "must",
    "should",
    "prefer",
    "default",
]

CONVERSATIONAL_PREFIX_RE = re.compile(r"^\*{0,2}(assistant|user|system)\*{0,2}:\s*", re.IGNORECASE)
QUESTION_LEAD_RE = re.compile(r"^(should i|what would you prefer|do you want|can you|could you|would you|is this|are you)", re.IGNORECASE)
CHANGELOG_RE = re.compile(r"^(\d+\.|#+\s*\d+\.?|added\b|updated\b|fixed\b|implemented\b)", re.IGNORECASE)


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_class: str
    title: str
    summary: str
    source_path: str
    source_timestamp: str
    source_anchor: str
    confidence: str
    repeat_count: int
    context_span: str
    causal_confidence: str
    promotion_candidate: bool
    enforcement_candidate: bool
    paired_artifact_id: Optional[str]
    transformation_history: List[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_roots(path: Path) -> List[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    roots = payload.get("roots", [])
    return [Path(root).expanduser() for root in roots]


def iter_files(roots: List[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for filename in filenames:
                    candidates.append(Path(dirpath) / filename)
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTS:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return json.dumps(payload, ensure_ascii=True, indent=2)
        except json.JSONDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")
    return path.read_text(encoding="utf-8", errors="ignore")


def split_candidates(text: str) -> List[tuple[str, str]]:
    chunks = re.split(r"\n\s*\n+", text)
    results: List[tuple[str, str]] = []
    line_no = 1
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            line_no += chunk.count("\n") + 1
            continue
        results.append((stripped, f"line:{line_no}"))
        line_no += chunk.count("\n") + 2
    return results


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_candidate_text(text: str) -> str:
    cleaned = normalize(text)
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
    cleaned = CONVERSATIONAL_PREFIX_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"^\*{1,2}", "", cleaned).strip()
    cleaned = re.sub(r"\*{1,2}$", "", cleaned).strip()
    cleaned = re.sub(r"^\d+\.\s*\*{0,2}", "", cleaned).strip()
    cleaned = re.sub(r"^option\s+\d+:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def looks_artifact(text: str) -> bool:
    cleaned = clean_candidate_text(text)
    low = cleaned.lower()
    if len(low) < 40 or len(low) > 700:
        return False
    if QUESTION_LEAD_RE.match(low):
        return False
    if cleaned.endswith("?") and not any(signal in low for signal in ("rule", "policy", "threshold", "guideline")):
        return False
    if CHANGELOG_RE.match(low) and not any(signal in low for signal in ("default", "prefer", "avoid", "should", "must")):
        return False
    if low.startswith(("you should now see", "expected output", "good thinking", "got it")):
        return False
    if low.startswith(("the implementation shouldn't", "the chunking logic looks solid", "removing from gui", "remove \"decode chk s\"")):
        return False
    return any(signal in low for signal in RULE_SIGNALS)


def classify(text: str) -> tuple[str, bool, bool]:
    low = text.lower()
    trait_hit = any(signal in low for signal in TRAIT_SIGNALS)
    failure_hit = any(signal in low for signal in FAILURE_SIGNALS)
    prevention_hit = any(signal in low for signal in PREVENTION_SIGNALS)

    if failure_hit and trait_hit:
        return "paired", True, prevention_hit
    if failure_hit and prevention_hit:
        return "failure", False, True
    return "trait", True, False


def confidence_for(text: str, artifact_class: str) -> str:
    length = len(text)
    if artifact_class == "paired":
        return "medium"
    if 60 <= length <= 280:
        return "high"
    return "medium"


def title_for(text: str) -> str:
    line = clean_candidate_text(text)
    line = re.sub(r"^[^a-zA-Z]+", "", line)
    line = line.split(". ", 1)[0]
    line = line.split(" - ", 1)[0]
    if len(line) > 96:
        return line[:93].rstrip() + "..."
    return line


def make_record(path: Path, source_anchor: str, text: str, artifact_index: int) -> ArtifactRecord:
    summary = clean_candidate_text(text)
    artifact_class, promote, enforce = classify(summary)
    stat = path.stat()
    ts = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return ArtifactRecord(
        artifact_id=f"artifact-{artifact_index:05d}",
        artifact_class=artifact_class,
        title=title_for(summary),
        summary=summary,
        source_path=str(path.resolve()),
        source_timestamp=ts,
        source_anchor=source_anchor,
        confidence=confidence_for(summary, artifact_class),
        repeat_count=1,
        context_span="local_chunk",
        causal_confidence="medium" if artifact_class == "paired" else "low",
        promotion_candidate=promote,
        enforcement_candidate=enforce,
        paired_artifact_id=None,
        transformation_history=["collected_from_local_source", "classified_by_stub_rules"],
    )


def harvest(roots: List[Path], output_path: Path, limit: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        artifact_index = 1
        for path in iter_files(roots):
            text = read_text(path)
            for chunk, anchor in split_candidates(text):
                if not looks_artifact(chunk):
                    continue
                record = make_record(path, anchor, chunk, artifact_index)
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
                artifact_index += 1
                written += 1
                if limit > 0 and written >= limit:
                    return written
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded ECOmpile artifact harvester stub.")
    parser.add_argument("--roots-file", default=str(DEFAULT_ROOTS_FILE), help="JSON file listing bounded source roots.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL output path.")
    parser.add_argument("--limit", type=int, default=200, help="Max artifacts to emit. Use 0 for no limit.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    roots = load_roots(Path(args.roots_file))
    written = harvest(roots, Path(args.output), args.limit)
    print(f"[harvest] roots={len(roots)} artifacts={written} output={args.output} generated={utc_now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())