from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    return int(proc.returncode)


def scan_runtime_for_source_dependencies() -> dict[str, list[str]]:
    src = ROOT / "src" / "ecov2"
    findings = {"experiment_refs": [], "repo_refs": []}
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "Experiment" in text or "\\Experiment\\" in text:
            findings["experiment_refs"].append(str(path))
        if "\\repo\\" in text or "/repo/" in text:
            findings["repo_refs"].append(str(path))
    return findings


def main() -> int:
    findings = scan_runtime_for_source_dependencies()
    has_dependency_ref = bool(findings["experiment_refs"] or findings["repo_refs"])

    test_rc = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
    provenance_rc = run([sys.executable, "scripts/provenance_audit.py"])
    payload = {
        "runtime_source_ref_scan": findings,
        "tests_passed": test_rc == 0,
        "provenance_audit_passed": provenance_rc == 0,
        "status": "pass" if (test_rc == 0 and provenance_rc == 0 and not has_dependency_ref) else "fail",
    }
    report = ROOT / "artifacts" / "public" / "acceptance_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
