# Contributing Kernels

ECOmpile uses compact kernel artifacts to convert repeated failure-detours into deterministic, context-safe primitives.

## Required Submission Set

Every new kernel contribution must include:
1. case log file,
2. kernel file,
3. index line in `kernels/index.tsv`.

## File Placement

1. Case log:
- `cases/<yyyy-mm-dd_case-name>/conversation.md`

2. Kernel file:
- `kernels/<platform>/<domain>/<KERNEL_ID>.kernel.md`

3. Index update:
- append one TSV row to `kernels/index.tsv`

## Kernel Format (Minimum)

- `KERNEL_ID`
- `PLATFORM`
- `CONTEXT`
- `Intent Signature`
- `Stop Condition`
- `Primitive`
- `Emit Template`
- `Forbid`
- `Source Case`

## Rules

1. No essays in kernel files; keep operational and testable.
2. Include explicit `Forbid` list to block common detours.
3. Context must be concrete (OS/shell/surface where relevant).
4. Emit template must be directly runnable.
5. Do not claim "no built-in method" without proof.

## Review Gate

A kernel is accepted when:
1. intent signature is specific,
2. primitive is valid on stated platform,
3. stop-condition is deterministic,
4. source case demonstrates real detour/failure pattern.
