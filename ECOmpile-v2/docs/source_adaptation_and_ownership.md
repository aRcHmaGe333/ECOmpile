# Source Adaptation and Ownership Assurance

This document records how material from the quarantined `Experiment` folder is used in ECOmpile V2.

## Principle

ECOmpile V2 uses `Experiment` as a method reference source, not as a code donor.

Rules:

1. Transfer concepts and proven pathways only.
2. Re-implement for ECOmpile purpose and contracts.
3. Keep outputs structurally and semantically owned by ECOmpile V2.
4. Keep a machine-checkable provenance audit before publication.

## Adapted Practical Methods

1. Method: content deduplication of repeated instruction/session material.
Source pattern: dedupe strategy in `Experiment/rust/.../prompt.rs` (`dedupe_instruction_files`).
ECOmpile adaptation: `stable_session_hash` and duplicate-session filtering in `src/ecov2/ingest.py`.
Purpose fit: prevent duplicate evidence from skewing convergence scoring.

2. Method: bounded execution loop with explicit stop/fallback behavior.
Source pattern: runtime turn loop and explicit outcomes in `Experiment/rust/.../conversation.rs`.
ECOmpile adaptation: deterministic router outputs only `kernel_hit` or explicit fallback in `src/ecov2/router.py`.
Purpose fit: transparent routing without hidden outcomes.

3. Method: explicit permission/escalation policy at decision boundaries.
Source pattern: permission policy model in `Experiment/rust/.../permissions.rs`.
ECOmpile adaptation: mandatory final human decision gate and grouped dilemmas in `src/ecov2/review.py`.
Purpose fit: automate surplus work while preserving final controlled activation.

4. Method: registry-like modular composition.
Source pattern: execution/command/tool registries in `Experiment/src/execution_registry.py` and `Experiment/src/command_graph.py`.
ECOmpile adaptation: schema-locked stage modules and CLI command surface in `src/ecov2/*.py` and `src/ecov2/cli.py`.
Purpose fit: modular, composable pipeline with explicit stage boundaries.

## Ownership Guard

Publication gate includes `scripts/provenance_audit.py`.

Audit behavior:

1. scans V2 textual sources
2. compares against quarantined `Experiment` textual sources
3. flags exact and near-copy risks
4. writes report to `artifacts/public/provenance_audit.json`

A failing exact-copy result blocks readiness.

## Status Use

Before GitHub update, run:

```powershell
python scripts\provenance_audit.py
python scripts\acceptance_check.py
```

Both reports must pass for publication readiness.
