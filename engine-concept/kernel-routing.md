# Kernel Routing Concept

## Objective

Route repeated prompt patterns to deterministic primitives when a proven kernel exists, instead of re-running expensive exploratory reasoning.

## Routing Flow

1. Context detection
- OS
- shell
- privilege
- surface (CLI/IDE/app)

2. Intent extraction
- identify action, scope, object, constraints

3. Kernel shortlist
- token index prefilter from `kernels/index.tsv`

4. Kernel match
- evaluate intent signature + context requirements

5. Stop-condition
- if matched, stop exploration and emit primitive template

6. Fallback
- if no match, use normal reasoning path

## Core Benefit

- fewer detours
- lower token burn
- improved first-response correctness
- reduced contradiction rate

## Current Kernel Seed

- `SID_REMOVE_SYSTEM_WIDE_KNOWN`
