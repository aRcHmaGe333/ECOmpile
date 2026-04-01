# Provider Packet (V2)

## Public Release Shape

ECOmpile V2 is presented as an execution-layer upgrade that converts convergent validated outcomes into deterministic templates.

Positioning:

- all-positive, win-win framing
- stripped implementation surface
- deterministic behavior where convergence is established
- clear boundary between public artifacts and internal evidence

## Human-System Contract

1. ECOmpile provides executable options, not locked doctrine.
2. Users decide what to adopt, amend, merge, approve, or discard.
3. The system compiles outcomes that repeatedly hold under reuse.
4. Final activation remains explicit and review-controlled.

## What Is Public

Public-safe defaults produce and expose:

- schema-defined contracts (`schemas/`)
- staged and active kernel artifacts (`kernels/`)
- public manifest and redacted outputs (`artifacts/public/`)
- provider-facing docs (`docs/`)

## What Stays Internal

Internal artifacts remain under `.private/` only:

- normalized raw sessions
- extracted raw outcomes
- full candidate payloads and review records with sensitive anchors

`.private/` is excluded from repository publication by `.gitignore`.

## Core System Guarantee

When intent and context match an approved active kernel:

1. exploration stops
2. deterministic emit template is returned
3. fallback is used only when no active kernel match exists

No synthetic deterministic output is fabricated on misses.

## CLI Surface

```text
ingest | extract | mine | compile | review | route
```

All commands support:

- `--public-safe` (default `true`)
- `--source-root`
- `--output-root`

## Method Modularity

Public method profile disclosure is documented in `docs/method_switch_profiles.md`.

## Release Notes for Reviewers

- V2 runtime does not import or depend on `Experiment/`.
- `repo/` and `Experiment/` are treated as read-only source inputs for curated migration only.
- Documents corrected from V1 are copied into V2 archive lane; originals are untouched.
- Source adaptation and ownership guard are documented in `docs/source_adaptation_and_ownership.md`.

## Build Verification Steps

1. Run pipeline with a linked-session source root.
2. Generate grouped review dilemmas (`review`) to reduce manual review volume.
3. Apply explicit review decisions for promotion.
4. Route known and unknown intents.
5. Verify public artifacts are redacted.
6. Run acceptance suite from V2 only.

## Public Value Proof

External verification bundle is defined in `docs/external_value_proof_kit.md`.

## Integrity Statement

ECOmpile V2 optimizes for deterministic, transparent outcomes that reduce repeated interaction cost and increase consistency for both users and providers.
