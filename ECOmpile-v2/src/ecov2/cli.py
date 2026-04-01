from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compile_kernels import compile_staging_kernels
from .extract import extract_outcomes
from .ingest import ingest_linked_sessions
from .mine import mine_candidates
from .paths import DEFAULT_SOURCE_ROOT, build_paths
from .review import apply_review_decisions, build_review_dilemmas
from .router import dump_route_json, route_prompt


def _bool_parser(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--public-safe",
        type=_bool_parser,
        default=True,
        help="Public-safe redaction mode (default: true).",
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Linked sessions source root.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root (default: current ECOmpile-v2 root).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECOmpile v2 pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest clean linked transcripts into normalized sessions.")
    _add_common_options(ingest)

    extract = subparsers.add_parser("extract", help="Extract convergent actionable outcomes from normalized sessions.")
    _add_common_options(extract)

    mine = subparsers.add_parser("mine", help="Score convergence and emit ranked kernel candidates.")
    _add_common_options(mine)

    compile_cmd = subparsers.add_parser("compile", help="Compile ranked candidates into staging kernels.")
    _add_common_options(compile_cmd)

    review = subparsers.add_parser("review", help="Apply HITL review decisions and promote approved kernels.")
    _add_common_options(review)
    review.add_argument("--decisions", required=False, help="Path to review decisions JSON.")
    review.add_argument(
        "--include-non-eligible",
        type=_bool_parser,
        default=False,
        help="Include non-promotion-eligible candidates in generated dilemma pack (default: false).",
    )
    review.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Token similarity threshold for dilemma grouping (default: 0.6).",
    )

    route = subparsers.add_parser("route", help="Route prompt against active deterministic kernels.")
    _add_common_options(route)
    route.add_argument("prompt", help="Prompt text to route.")
    route.add_argument("--platform", default="windows")
    route.add_argument("--surface", default="cli")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = build_paths(args.output_root)
    source_root = Path(args.source_root).resolve()

    if args.command == "ingest":
        result = ingest_linked_sessions(paths, source_root=source_root, public_safe=args.public_safe)
        print(
            json.dumps(
                {
                    "command": "ingest",
                    "source_root": str(result.source_root),
                    "records_count": result.records_count,
                    "skipped_files": result.skipped_files,
                    "duplicate_sessions_skipped": result.duplicate_sessions_skipped,
                    "normalized_path": str(paths.normalized_path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "extract":
        count = extract_outcomes(paths, public_safe=args.public_safe)
        print(
            json.dumps(
                {
                    "command": "extract",
                    "outcomes_count": count,
                    "outcomes_path": str(paths.outcomes_path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "mine":
        count = mine_candidates(paths, public_safe=args.public_safe)
        print(
            json.dumps(
                {
                    "command": "mine",
                    "candidate_count": count,
                    "candidates_path": str(paths.candidates_path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "compile":
        count = compile_staging_kernels(paths, public_safe=args.public_safe)
        print(
            json.dumps(
                {
                    "command": "compile",
                    "staging_kernel_count": count,
                    "staging_dir": str(paths.kernels_staging_dir),
                    "staging_index": str(paths.staging_index_path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "review":
        if not args.decisions:
            pack = build_review_dilemmas(
                paths,
                public_safe=args.public_safe,
                include_non_eligible=args.include_non_eligible,
                similarity_threshold=args.similarity_threshold,
            )
            print(json.dumps({"command": "review", "mode": "prepare", **pack}, indent=2))
            return 0
        report = apply_review_decisions(
            paths,
            decisions_file=Path(args.decisions).resolve(),
            public_safe=args.public_safe,
        )
        print(json.dumps({"command": "review", "mode": "apply", **report}, indent=2))
        return 0

    if args.command == "route":
        result = route_prompt(
            paths,
            prompt=args.prompt,
            platform=args.platform,
            surface=args.surface,
        )
        print(dump_route_json(result))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
