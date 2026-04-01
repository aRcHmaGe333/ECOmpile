# Public vs Internal Boundary

## Default Mode

Public-safe mode is the default execution mode in ECOmpile V2.

## Redaction Rules (Public Output)

Public artifacts redact:

- email addresses
- phone numbers
- account identifiers (SID-like)
- local absolute Windows paths

## Internal Scope

Internal scope is limited to `.private/` and is excluded from source control.

Internal scope may include:

- complete source path anchors
- unredacted candidate payload history
- reviewer audit traces

## Publication Rule

A file is publishable when:

1. it is not under `.private/`
2. it has no raw personal identifiers
3. it has no local absolute user-machine paths
4. it does not require `Experiment/` or `repo/` at runtime

## Promotion Rule

Kernel activation is HITL-only.

- staged candidates are never auto-promoted
- automated mining and dilemma grouping are advisory-only
- explicit approve decisions are mandatory
- rejected/unreviewed candidates remain inactive

## Transparency Rule

Routing has two outcomes only:

- `kernel_hit`: deterministic template emitted
- `fallback`: explicit reason returned

No hidden deterministic claims are permitted on fallback.
