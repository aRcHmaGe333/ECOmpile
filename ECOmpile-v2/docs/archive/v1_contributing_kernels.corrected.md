# V1 Contributing Kernels (Corrected for V2)

Source: `repo/CONTRIBUTING_KERNELS.md`

## Required Submission Set

Each kernel promotion in V2 requires:

1. staged kernel artifact in `kernels/staging/`
2. explicit `review_decision` record (`approve` or `reject`)
3. active index merge in `kernels/index.tsv` only after approval

## Kernel Minimum Fields

- `KERNEL_ID`
- `PLATFORM`
- `CONTEXT`
- `Intent Signature` (required tokens)
- `Stop Condition`
- `Emit Template`
- `Forbid`
- `Provenance`
- `Metrics`

## Operational Rules

1. No narrative filler in kernel body.
2. Emit template must be deterministic and directly executable or directly reusable.
3. Stop condition must block extra exploration after a valid match.
4. Forbid list must prevent known detours.
5. Provenance must point to convergent evidence anchors.

## Review Gate

A staged candidate is promoted only when:

1. `review_decision.decision = approve`
2. reviewer and timestamp fields are present
3. corresponding staged kernel file exists

No auto-activation is allowed in V2 baseline.
