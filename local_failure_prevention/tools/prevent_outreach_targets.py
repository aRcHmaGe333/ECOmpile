from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_recipients(value: str) -> List[str]:
    return sorted(set(e.lower() for e in EMAIL_RE.findall(value or "")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prevent repeated outreach failures by filtering blocked targets.")
    parser.add_argument("--rows-file", required=True, help="Input outreach rows JSON (list of objects).")
    parser.add_argument("--blocklist-file", required=True, help="Blocked targets JSON from analyzer.")
    parser.add_argument("--out-file", required=True, help="Filtered rows output JSON.")
    parser.add_argument("--report-file", required=True, help="Prevention report output JSON.")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Exit code 2 if any blocked targets found.")
    args = parser.parse_args()

    rows_path = Path(args.rows_file)
    blocklist_path = Path(args.blocklist_file)
    out_path = Path(args.out_file)
    report_path = Path(args.report_file)

    rows = load_json(rows_path)
    if not isinstance(rows, list):
        raise ValueError("rows-file must contain a JSON list.")
    blocklist = load_json(blocklist_path)
    targets = blocklist.get("targets", []) if isinstance(blocklist, dict) else []

    blocked_map: Dict[str, Dict] = {str(item["email"]).lower(): item for item in targets}
    kept: List[Dict] = []
    removed: List[Dict] = []

    for row in rows:
        to_field = str(row.get("to", ""))
        recipients = extract_recipients(to_field)
        hits = [r for r in recipients if r in blocked_map]
        if not hits:
            kept.append(row)
            continue
        removed.append(
            {
                "row": row,
                "blocked_recipients": [
                    {"email": h, "reasons": blocked_map[h].get("reasons", [])}
                    for h in hits
                ],
            }
        )

    write_json(out_path, kept)
    report = {
        "generated_utc": utc_now(),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed_detail": removed,
    }
    write_json(report_path, report)

    print(f"[prevent] kept={len(kept)} removed={len(removed)}")
    print(f"[prevent] wrote {out_path}")
    print(f"[prevent] wrote {report_path}")
    if args.fail_on_blocked and removed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
