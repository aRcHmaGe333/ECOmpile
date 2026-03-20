from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass
class Pattern:
    pid: str
    category: str
    severity: str
    block_target: bool
    regex: re.Pattern[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_patterns(path: Path) -> tuple[list[Pattern], list[str]]:
    data = load_json(path)
    items: List[Pattern] = []
    for raw in data.get("patterns", []):
        items.append(
            Pattern(
                pid=str(raw["id"]),
                category=str(raw["category"]),
                severity=str(raw["severity"]),
                block_target=bool(raw.get("block_target", False)),
                regex=re.compile(str(raw["regex"]), flags=re.IGNORECASE),
            )
        )
    ignore_prefixes = [x.lower() for x in data.get("ignore_email_prefixes", [])]
    return items, ignore_prefixes


def parse_redactions(path: Path) -> List[tuple[re.Pattern[str], str]]:
    data = load_json(path)
    out: List[tuple[re.Pattern[str], str]] = []
    for raw in data.get("redactions", []):
        out.append((re.compile(str(raw["regex"]), flags=re.IGNORECASE), str(raw["replace_with"])))
    return out


def iter_scan_files(scan_dirs: Sequence[Path], scan_files: Sequence[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for path in scan_files:
        if path.exists() and path.is_file():
            rp = str(path.resolve())
            if rp not in seen:
                seen.add(rp)
                yield path
    for root in scan_dirs:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".log", ".txt", ".md", ".json"}:
                continue
            rp = str(path.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            yield path


def redact(text: str, redactions: Sequence[tuple[re.Pattern[str], str]]) -> str:
    out = text
    for pattern, replacement in redactions:
        out = pattern.sub(replacement, out)
    return out


def extract_emails(text: str, ignore_prefixes: Sequence[str]) -> List[str]:
    emails = [e.lower() for e in EMAIL_RE.findall(text)]
    out: List[str] = []
    for e in emails:
        local = e.split("@", 1)[0]
        if any(local.startswith(prefix) for prefix in ignore_prefixes):
            continue
        out.append(e)
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze failure logs and build prevention blocklist.")
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=["local_failure_prevention/logs"],
        help="Directory to scan for failure artifacts (repeatable).",
    )
    parser.add_argument(
        "--scan-file",
        action="append",
        default=["C:\\Agent\\agent.log"],
        help="Explicit file to scan (repeatable).",
    )
    parser.add_argument(
        "--patterns-file",
        default="local_failure_prevention/patterns/failure_patterns.json",
        help="Failure pattern definition file.",
    )
    parser.add_argument(
        "--redaction-file",
        default="local_failure_prevention/patterns/ucas_redaction_patterns.json",
        help="UCAS redaction pattern file.",
    )
    parser.add_argument(
        "--state-dir",
        default="local_failure_prevention/state",
        help="State output directory.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    scan_dirs = [Path(p) if Path(p).is_absolute() else (root / p) for p in args.scan_dir]
    scan_files = [Path(p) if Path(p).is_absolute() else (root / p) for p in args.scan_file]
    patterns_file = Path(args.patterns_file) if Path(args.patterns_file).is_absolute() else (root / args.patterns_file)
    redaction_file = Path(args.redaction_file) if Path(args.redaction_file).is_absolute() else (root / args.redaction_file)
    state_dir = Path(args.state_dir) if Path(args.state_dir).is_absolute() else (root / args.state_dir)

    patterns, ignore_prefixes = parse_patterns(patterns_file)
    redactions = parse_redactions(redaction_file)

    events: List[Dict] = []
    counts = Counter()
    category_counts = Counter()
    blocked: Dict[str, Dict] = {}

    for file_path in iter_scan_files(scan_dirs, scan_files):
        try:
            lines = file_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        except OSError:
            continue

        for lineno, line in enumerate(lines, start=1):
            text = line.strip()
            if not text:
                continue
            for p in patterns:
                if not p.regex.search(text):
                    continue
                emails = extract_emails(text, ignore_prefixes)
                event = {
                    "timestamp_utc": utc_now(),
                    "file": str(file_path),
                    "line": lineno,
                    "pattern_id": p.pid,
                    "category": p.category,
                    "severity": p.severity,
                    "text": text,
                    "emails": emails,
                }
                events.append(event)
                counts[p.pid] += 1
                category_counts[p.category] += 1
                if p.block_target:
                    for email in emails:
                        rec = blocked.setdefault(
                            email,
                            {
                                "email": email,
                                "reasons": set(),
                                "categories": set(),
                                "hit_count": 0,
                                "last_seen_utc": None,
                                "sources": set(),
                            },
                        )
                        rec["reasons"].add(p.pid)
                        rec["categories"].add(p.category)
                        rec["hit_count"] += 1
                        rec["last_seen_utc"] = utc_now()
                        rec["sources"].add(str(file_path))

    blocked_list: List[Dict] = []
    for email, rec in sorted(blocked.items()):
        blocked_list.append(
            {
                "email": email,
                "reasons": sorted(rec["reasons"]),
                "categories": sorted(rec["categories"]),
                "hit_count": rec["hit_count"],
                "last_seen_utc": rec["last_seen_utc"],
                "sources": sorted(rec["sources"]),
            }
        )

    internal_events = {"generated_utc": utc_now(), "events": events}
    public_events = {
        "generated_utc": utc_now(),
        "events": [
            {
                **{k: v for k, v in event.items() if k != "text"},
                "text": redact(str(event["text"]), redactions),
                "emails": ["[REDACTED_EMAIL]" for _ in event.get("emails", [])],
            }
            for event in events
        ],
    }
    summary = {
        "generated_utc": utc_now(),
        "event_count": len(events),
        "pattern_counts": dict(counts),
        "category_counts": dict(category_counts),
        "blocked_targets_count": len(blocked_list),
    }

    write_json(state_dir / "failure_events.internal.json", internal_events)
    write_json(state_dir / "failure_events.public.json", public_events)
    write_json(state_dir / "failure_summary.json", summary)
    write_json(state_dir / "blocked_targets.json", {"generated_utc": utc_now(), "targets": blocked_list})

    print(f"[analyze] events={len(events)} blocked_targets={len(blocked_list)}")
    print(f"[analyze] wrote {state_dir / 'failure_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
