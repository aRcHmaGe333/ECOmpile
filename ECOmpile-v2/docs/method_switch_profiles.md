# Method Switch Profiles (Public)

ECOmpile V2 supports modular method switching by profile while keeping stable contracts.

## Profile Families

1. Ingestion profiles
- `strict_linked_sessions`: accepts only clean linked session families.
- `expanded_logs`: adds more local sources with stricter normalization checks.

2. Extraction profiles
- `command_first`: prioritizes deterministic command/template outcomes.
- `procedure_first`: prioritizes repeatable multi-step procedures.

3. Mining profiles
- `conservative`: high certainty before promotion eligibility.
- `balanced`: default score behavior.
- `aggressive`: faster candidate surfacing with unchanged HITL activation gate.

4. Review profiles
- `dilemma_grouped`: merges similar candidates into one review decision.
- `single_candidate`: one-by-one review where risk tolerance is low.

5. Routing profiles
- `strict_full_token_match`: full required token match.
- `weighted_match`: high-confidence partial match with stricter fallback checks.

## Invariance Rules

All profiles must preserve:

1. schema-validated stage inputs/outputs
2. explicit fallback on routing miss
3. no automatic activation of active kernels
4. public-safe default artifact behavior

## Why this matters

Profile switching allows ECOmpile deployments to adapt to domain requirements without fragmenting core pipeline semantics.
