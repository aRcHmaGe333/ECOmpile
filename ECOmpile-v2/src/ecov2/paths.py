from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE_ROOT = (
    Path.home()
    / "code"
    / "TruMate"
    / "tools"
    / "chat_logs"
    / "state"
    / "local_chat_hub"
    / "linked_sessions"
)


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    private_dir: Path
    normalized_path: Path
    outcomes_path: Path
    candidates_path: Path
    review_records_dir: Path
    kernels_dir: Path
    kernels_staging_dir: Path
    kernels_active_dir: Path
    kernels_index_path: Path
    staging_index_path: Path
    public_artifacts_dir: Path
    public_manifest_path: Path


def resolve_root(output_root: str | Path | None) -> Path:
    if output_root is None:
        return Path(__file__).resolve().parents[2]
    return Path(output_root).resolve()


def build_paths(output_root: str | Path | None = None) -> PipelinePaths:
    root = resolve_root(output_root)
    private_dir = root / ".private"
    normalized_path = private_dir / "normalized" / "normalized_sessions.jsonl"
    outcomes_path = private_dir / "outcomes" / "extracted_outcomes.jsonl"
    candidates_path = private_dir / "candidates" / "ranked_candidates.json"
    review_records_dir = private_dir / "review"
    kernels_dir = root / "kernels"
    staging_dir = kernels_dir / "staging"
    active_dir = kernels_dir / "active"
    public_artifacts_dir = root / "artifacts" / "public"
    return PipelinePaths(
        root=root,
        private_dir=private_dir,
        normalized_path=normalized_path,
        outcomes_path=outcomes_path,
        candidates_path=candidates_path,
        review_records_dir=review_records_dir,
        kernels_dir=kernels_dir,
        kernels_staging_dir=staging_dir,
        kernels_active_dir=active_dir,
        kernels_index_path=kernels_dir / "index.tsv",
        staging_index_path=staging_dir / "index.tsv",
        public_artifacts_dir=public_artifacts_dir,
        public_manifest_path=public_artifacts_dir / "public_artifact_manifest.json",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
