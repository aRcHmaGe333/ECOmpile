from __future__ import annotations

import argparse
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path


TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".rs",
    ".txt",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".sh",
    ".ps1",
    ".tsv",
}

ALLOWED_DERIVED_PREFIXES = [
    "docs/archive/",
]


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def normalize(text: str) -> str:
    # Normalize whitespace-only differences so audit catches structural copying.
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_allowed_derived(v2_rel: str) -> bool:
    return any(v2_rel.startswith(prefix) for prefix in ALLOWED_DERIVED_PREFIXES)


def collect_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".private", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
    return files


def build_experiment_index(experiment_root: Path) -> dict[str, list[tuple[Path, str, str, int]]]:
    index: dict[str, list[tuple[Path, str, str, int]]] = {}
    for path in collect_text_files(experiment_root):
        raw = read_text(path)
        if raw is None:
            continue
        norm = normalize(raw)
        digest = sha(norm)
        key = path.suffix.lower()
        index.setdefault(key, []).append((path, digest, norm, len(norm)))
    return index


def audit(v2_root: Path, experiment_root: Path, near_threshold: float) -> dict:
    index = build_experiment_index(experiment_root)
    exact_matches: list[dict] = []
    near_matches: list[dict] = []

    for path in collect_text_files(v2_root):
        raw = read_text(path)
        if raw is None:
            continue
        v2_rel = rel(path, v2_root)
        norm = normalize(raw)
        if not norm:
            continue
        digest = sha(norm)
        candidates = index.get(path.suffix.lower(), [])

        for exp_path, exp_digest, exp_norm, exp_len in candidates:
            if digest == exp_digest:
                exact_matches.append(
                    {
                        "v2_path": v2_rel,
                        "experiment_path": str(exp_path),
                        "allowed_derived": is_allowed_derived(v2_rel),
                    }
                )
                break

        # Near-match check (size gate for performance and relevance)
        v2_len = len(norm)
        if v2_len < 120:
            continue
        for exp_path, _exp_digest, exp_norm, exp_len in candidates:
            if exp_len < 120:
                continue
            ratio_size = min(v2_len, exp_len) / max(v2_len, exp_len)
            if ratio_size < 0.7:
                continue
            ratio = SequenceMatcher(None, norm, exp_norm).ratio()
            if ratio >= near_threshold:
                near_matches.append(
                    {
                        "v2_path": v2_rel,
                        "experiment_path": str(exp_path),
                        "similarity": round(ratio, 4),
                        "allowed_derived": is_allowed_derived(v2_rel),
                    }
                )
                break

    disallowed_exact = [item for item in exact_matches if not item["allowed_derived"]]
    disallowed_near = [item for item in near_matches if not item["allowed_derived"]]
    status = "pass"
    if disallowed_exact:
        status = "fail"
    elif disallowed_near:
        status = "review"

    return {
        "status": status,
        "v2_root": "ECOmpile-v2",
        "experiment_root": "QUARANTINED_EXPERIMENT_ROOT",
        "exact_matches": exact_matches,
        "near_matches": near_matches,
        "disallowed_exact_count": len(disallowed_exact),
        "disallowed_near_count": len(disallowed_near),
        "near_threshold": near_threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V2 against Experiment for copy-risk.")
    parser.add_argument("--v2-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--experiment-root",
        default=str(Path.home() / "code" / "ECOmpile_sensitive_hold" / "Experiment"),
    )
    parser.add_argument("--near-threshold", type=float, default=0.92)
    args = parser.parse_args()

    report = audit(Path(args.v2_root), Path(args.experiment_root), args.near_threshold)
    output = Path(args.v2_root) / "artifacts" / "public" / "provenance_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
