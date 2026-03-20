# ECOmpile Artifact Harvester Spec

Status: first implementation slice

## Purpose

Define the first bounded ECOmpile harvester for chat-derived and local retrospection artifacts.

This is not a universal ingestion engine.
It is a selective artifact harvester designed to turn local history into reusable inputs for later routing, compilation, and governance.

## Design rule

- traits are the default harvest target
- failures are harvested when needed
- paired artifacts capture the relation between a useful trait and its missing, inverted, overextended, or misapplied form

## First-pass source roots

- known chat export locations already used by UCAS and TruMate
- curated local case folders under `cases/`
- local notes or logs that already function as explicit artifact surfaces

The first pass should prefer known high-yield roots over broad filesystem discovery.

## Artifact classes

### `trait`

Use when the artifact captures a reusable positive operating move.

Examples:
- requirement locking
- anti-boilerplate cleanup behavior
- concise claim-bounding discipline
- useful task derivation pattern

### `failure`

Use when the artifact must be captured to prevent repetition or protect value.

Examples:
- misleading community boilerplate
- distortion-prone framing
- bad default assumptions that recur

### `paired`

Use when a failure is better understood as the absence, inversion, overextension, or misuse of a trait.

Examples:
- positive trait: truth-surface discipline
- paired failure: overstated public framing

## Required ledger fields

- `artifact_id`
- `artifact_class`
- `title`
- `summary`
- `source_path`
- `source_timestamp`
- `source_anchor`
- `confidence`
- `repeat_count`
- `context_span`
- `causal_confidence`
- `promotion_candidate`
- `enforcement_candidate`
- `paired_artifact_id` when applicable
- `transformation_history`

## Routing thresholds

### Trait-default threshold

If the same practical value can be preserved by naming a positive operating trait, prefer that over an explicit failure record.

### Failure-needed threshold

Create or expand a `failure` artifact when at least one of these is true:
- naming the failure materially improves prevention
- the failure causes repeated loss or distortion
- the failure cannot be represented clearly as a trait alone
- downstream controls need an explicit anti-pattern reference

### Kernel-escalation threshold

Stay in user space unless all three are true:
- required evidence is not observable from known user-space roots
- the missing evidence blocks the intended artifact class materially
- lower-level capture has a concrete bounded target and safety rationale

## First output format

Use a JSONL ledger such as `artifacts.jsonl`.

Each line should represent one harvested artifact with provenance and routing metadata.

## First implementation sequence

1. define the bounded source roots
2. implement simple collector for those roots
3. emit raw candidate artifact records
4. classify into `trait`, `failure`, or `paired`
5. apply the trait-default and failure-needed thresholds
6. write auditable ledger output
7. review harvested records against `REQUIREMENTS.lock`

## Non-goals for v0

- complete semantic understanding of every chat
- automatic doctrine promotion without review
- low-level system hooks by default
- giant negative inventories that bury the useful signal

## Expected relation to UCAS

UCAS should evaluate the harvested artifacts for:
- provenance sufficiency
- promotion readiness
- task derivation quality
- whether a failure should remain explicit or collapse into a stronger trait rule