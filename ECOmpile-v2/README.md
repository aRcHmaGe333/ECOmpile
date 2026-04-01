# ECOmpile V2

ECOmpile V2 is the canonical provider-ready build for deterministic outcome compilation.

It ingests convergent success paths from linked local transcripts, extracts reusable outcomes, ranks deterministic kernel candidates, enforces human review before activation, and routes known intents to transparent executable templates.

It is user-centered by design: users generate, amend, and adopt options; convergence from that usage becomes deterministic infrastructure.

## Release Boundary

- Public-safe mode is default for every CLI command.
- Internal evidence is stored only under `.private/` and is gitignored.
- Public artifacts are written to `artifacts/public/`.
- `repo/` and `Experiment/` are source material only and are not runtime dependencies.

## CLI

Entrypoint:

```powershell
python -m ecov2 <command> [options]
```

Commands:

1. `ingest` - normalize supported linked transcript families (`codex`, `vscode`, `claude`).
2. `extract` - emit actionable convergent outcomes.
3. `mine` - score and rank kernel candidates.
4. `compile` - write staged kernel artifacts.
5. `review` - prepare grouped review dilemmas to minimize manual workload.
6. `review --decisions <file>` - apply explicit HITL decisions and promote approved kernels.
7. `route <prompt>` - deterministic router hit/fallback execution.

Common options on all commands:

- `--public-safe` default `true`
- `--source-root` default `%USERPROFILE%\code\TruMate\tools\chat_logs\state\local_chat_hub\linked_sessions`
- `--output-root` default current V2 root

## Pipeline Contracts

All stage I/O is validated against JSON schemas:

- `schemas/normalized_session.schema.json`
- `schemas/extracted_outcome.schema.json`
- `schemas/kernel_candidate.schema.json`
- `schemas/review_decision.schema.json`
- `schemas/public_artifact_manifest.schema.json`

Invalid records fail explicitly.

## Deterministic Mining Rule

Fixed score formula:

```text
score = 0.45*recurrence + 0.25*acceptance_signal + 0.20*cross_session_reuse + 0.10*determinism_shape
```

Promotion-eligible when:

- `score >= 0.72`
- `occurrences >= 3`

Activation remains HITL-only in V2 baseline.

## Low-Labor Review Flow

1. Run `review` with no decisions file to auto-generate grouped dilemmas from similar candidates.
2. Manually decide each dilemma once (approve/reject).
3. Apply final decisions with `review --decisions <file>`.

This keeps surplus classification and grouping automated while preserving manual control of activation.

## Review Gate

No candidate auto-activates.

Only approved `review_decision` records promote staged kernels from:

- `kernels/staging/` -> `kernels/active/`
- and merge into `kernels/index.tsv`

## Provider Packet

Provider-facing release docs are in:

- `docs/provider_packet.md`
- `docs/public_internal_boundary.md`
- `docs/migration_map.md`
- `docs/source_adaptation_and_ownership.md`
- `docs/method_switch_profiles.md`
- `docs/external_value_proof_kit.md`
- `docs/disclosure_inventory.md`

## Modules and Philosophy

- `docs/module_layout.md` (modular architecture and compatibility mapping)
- `docs/intent_operating_philosophy.md` (intent contract and execution instructions)

## Validation

Run acceptance tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The suite covers ingestion, extraction, deterministic scoring, HITL gating, routing hit/fallback, redaction, Experiment independence simulation, and provider-packet boundary checks.
