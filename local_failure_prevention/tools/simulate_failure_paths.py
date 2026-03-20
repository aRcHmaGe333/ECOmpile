from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_playwright_version() -> str:
    try:
        import importlib.metadata as meta  # type: ignore
        return meta.version("playwright")
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Side introspection simulation using current environment and local failure events.")
    parser.add_argument(
        "--events-file",
        default="local_failure_prevention/state/failure_events.internal.json",
        help="Internal failure events file from analyzer.",
    )
    parser.add_argument("--task-file", required=True, help="Current automation task JSON.")
    parser.add_argument("--rows-file", required=True, help="Current outreach rows JSON.")
    parser.add_argument(
        "--out-file",
        default="local_failure_prevention/state/side_introspection_simulation.json",
        help="Simulation report output path.",
    )
    args = parser.parse_args()

    events_path = Path(args.events_file)
    task_path = Path(args.task_file)
    rows_path = Path(args.rows_file)
    out_path = Path(args.out_file)

    events_data = load_json(events_path)
    task = load_json(task_path)
    rows = load_json(rows_path)

    events = events_data.get("events", []) if isinstance(events_data, dict) else []
    pattern_counts: Dict[str, int] = {}
    for event in events:
        pid = str(event.get("pattern_id", "unknown"))
        pattern_counts[pid] = pattern_counts.get(pid, 0) + 1

    has_cdp = bool(task.get("cdp_url")) if isinstance(task, dict) else False
    has_auto_launch = bool(task.get("cdp_auto_launch")) if isinstance(task, dict) else False
    row_count = len(rows) if isinstance(rows, list) else 0

    predicted_failures: List[Dict] = []
    controls: List[Dict] = []

    if pattern_counts.get("secure_browser_block", 0) > 0 and not has_cdp:
        predicted_failures.append(
            {
                "failure": "auth_secure_browser_block_recurrence",
                "risk": "high",
                "reason": "Prior secure-browser failures + no CDP channel in current task.",
            }
        )

    if pattern_counts.get("delivery_failure", 0) > 0:
        predicted_failures.append(
            {
                "failure": "delivery_failure_recurrence",
                "risk": "high",
                "reason": "Prior DSN failures exist in current evidence set.",
            }
        )
        controls.append(
            {
                "control": "blocklist_gate",
                "action": "Run prevent_outreach_targets.py before every send batch.",
            }
        )

    if pattern_counts.get("unmonitored_inbox", 0) > 0:
        predicted_failures.append(
            {
                "failure": "dead_channel_recurrence",
                "risk": "high",
                "reason": "Unmonitored inbox detected in prior evidence.",
            }
        )
        controls.append(
            {
                "control": "channel_upgrade",
                "action": "Route outreach to official intake forms/channels when no-reply addresses are detected.",
            }
        )

    if pattern_counts.get("selector_timeout", 0) > 0:
        controls.append(
            {
                "control": "selector_strategy",
                "action": "Use direct compose URLs and broad selectors with readiness gates.",
            }
        )

    if has_cdp:
        controls.append(
            {
                "control": "auth_mode",
                "action": "Keep CDP mode for Gmail flows to avoid unsupported browser login path.",
            }
        )

    report = {
        "generated_utc": utc_now(),
        "side_introspection_level": "behavioral_simulation_only",
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "playwright_version": get_playwright_version(),
        },
        "current_task": {
            "task_file": str(task_path.resolve()),
            "has_cdp": has_cdp,
            "cdp_auto_launch": has_auto_launch,
        },
        "current_rows_count": row_count,
        "observed_pattern_counts": pattern_counts,
        "predicted_failures": predicted_failures,
        "recommended_controls": controls,
    }

    write_json(out_path, report)
    print(f"[simulate] predicted_failures={len(predicted_failures)} controls={len(controls)}")
    print(f"[simulate] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
