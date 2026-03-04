# ECOmpile: implementation-ready architecture and commercial terms

Case: `2026-03-04_unknown-contact-sid-removal`
Kernel candidate: `SID_REMOVE_SYSTEM_WIDE_KNOWN`

## What Is Already In Place (External)
- Behavioral failure/correction evidence is packaged in `behavioral_evidence.json`.
- Deterministic kernel candidate is packaged in `SID_REMOVE_SYSTEM_WIDE_KNOWN.kernel.candidate.md`.
- Routing intent and forbid lists are explicit and testable.

## What Is Internal To OpenAI
- True model-internal introspection requires internal telemetry.
- Required event contract is packaged in `internal_trace_contract.json`.
- Internal replay/trace validation is the remaining validation step.

## Commercial Guardrail
Compensation and attribution terms are agreed in writing before internal execution, integration, derivative use, or deployment.

## Package Files
- `behavioral_evidence.json`
- `internal_trace_contract.json`
- `SID_REMOVE_SYSTEM_WIDE_KNOWN.kernel.candidate.md`
