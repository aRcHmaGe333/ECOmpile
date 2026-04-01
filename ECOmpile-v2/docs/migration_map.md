# Curated Migration Map (V1/Experiment -> V2)

This map records selective migration into V2. It is not a wholesale copy.

## Source Policy

- `repo/` and `Experiment/` treated as read-only source material.
- Only required documents are copied, and only as corrected V2 surfaces.
- Originals remain in place.

## Mappings

1. `repo/CONTRIBUTING_KERNELS.md` -> `docs/archive/v1_contributing_kernels.corrected.md`
Reason: preserve useful kernel review constraints while aligning fields to V2 schema and CLI.

2. `repo/docs/public_release.md` -> `docs/archive/v1_public_release.corrected.md`
Reason: preserve external narrative while shifting to convergent outcomes and deterministic execution framing.

3. `Experiment/ChatGPT-Partner Program Interpretation.md` -> `docs/archive/intent_direction_2026-04-01.corrected.md`
Reason: preserve latest intent direction and convert it into implementation guidance for V2 without runtime dependency.

## Not Migrated

Large exploratory trees (full `repo/` and `Experiment/`) are intentionally not copied into V2.

## Runtime Dependency Check

V2 runtime modules under `src/ecov2/` must contain no `Experiment` path dependencies.

## Ownership Check

Before publication, run `scripts/provenance_audit.py` to verify V2 output is adapted and owned, not direct copied source.
