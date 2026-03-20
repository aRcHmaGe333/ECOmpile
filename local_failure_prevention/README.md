# Local Failure Prevention (External-Only)

This module runs locally and uses observable logs/artifacts to:

1. detect recurring failure patterns,
2. build/update a target blocklist,
3. prevent repeated failed outreach attempts.

It is designed for operation without internal model telemetry.

## Folder layout

- `tools/analyze_failures.py` - scans logs and updates failure state/blocklist.
- `tools/prevent_outreach_targets.py` - filters outgoing recipient rows using blocklist.
- `patterns/failure_patterns.json` - regex pattern definitions.
- `logs/` - local evidence notes and exported failure snippets.
- `state/` - generated reports and blocklists.
- `NO_INTERNAL_ACCESS_BOUNDARY.md` - explicit capability boundary statement.

## Quick start

1. Analyze logs and update blocklist:

```powershell
python local_failure_prevention/tools/analyze_failures.py `
  --scan-dir local_failure_prevention/logs `
  --scan-file C:\Agent\agent.log
```

2. Prevent repeat failures in outreach rows:

```powershell
python local_failure_prevention/tools/prevent_outreach_targets.py `
  --rows-file C:\Agent\data\outreach_openai\openai_email_rows.json `
  --blocklist-file local_failure_prevention/state/blocked_targets.json `
  --out-file C:\Agent\data\outreach_openai\openai_email_rows.filtered.json `
  --report-file local_failure_prevention/state/prevention_report.json `
  --fail-on-blocked
```

If blocked recipients are found, the script exits non-zero with a report.
