# ECOmpile V2 Module Layout

## Core Flow

ECOmpile is a user-centered convergence system:

1. users produce or amend outcomes
2. the system captures and normalizes evidence
3. convergent outcomes are mined and compiled
4. grouped human decisions approve final activation
5. routing executes known kernels deterministically

## Modules

1. `M1: Source Intake`
Purpose: collect real interaction artifacts from linked session sources.
Implementation: `src/ecov2/ingest.py`
Input: linked transcript families (`codex`, `vscode`, `claude`)
Output: normalized sessions

2. `M2: Outcome Extraction`
Purpose: convert normalized exchanges into actionable candidate outcomes.
Implementation: `src/ecov2/extract.py`
Input: normalized sessions
Output: extracted outcomes with anchors, acceptance and determinism signals

3. `M3: Convergence Mining`
Purpose: score recurrence and cross-session reuse to rank candidates.
Implementation: `src/ecov2/mine.py`
Input: extracted outcomes
Output: ranked kernel candidates with promotion eligibility

4. `M4: Kernel Compiler`
Purpose: compile ranked candidates into transparent staging kernels.
Implementation: `src/ecov2/compile_kernels.py`
Input: ranked candidates
Output: `kernels/staging/*.kernel.md` and staging index

5. `M5: Review Load Reducer`
Purpose: merge similar candidates into single review dilemmas.
Implementation: `src/ecov2/review.py` (`build_review_dilemmas`)
Input: ranked candidates
Output: grouped dilemma pack for minimal manual workload

6. `M6: Human Decision Gate`
Purpose: apply final approve/reject decisions and activate kernels.
Implementation: `src/ecov2/review.py` (`apply_review_decisions`)
Input: decisions (`decisions` or grouped `dilemmas`)
Output: promoted active kernels and merged active index

7. `M7: Deterministic Router`
Purpose: stop exploration on match and emit known-good template.
Implementation: `src/ecov2/router.py`
Input: prompt + active kernels
Output: `kernel_hit` with emit template, or explicit fallback reason

8. `M8: Safety/Exposure Boundary`
Purpose: keep public outputs clean while retaining internal evidence privately.
Implementation: `src/ecov2/redaction.py`, `src/ecov2/manifest.py`
Input: stage outputs
Output: redacted public artifacts + internal `.private` records

## Compatibility Rules

1. Every stage validates schema-defined I/O before passing data forward.
2. Public-safe mode is default across all commands.
3. Activation is never automatic; mining is advisory until approved.
4. Similar instances are grouped to reduce manual review volume.
5. Runtime never depends on `Experiment/` or `repo/`.

## CLI Mapping

- `ingest` -> M1
- `extract` -> M2
- `mine` -> M3
- `compile` -> M4
- `review` -> M5 (prepare dilemmas)
- `review --decisions` -> M6 (apply final decisions)
- `route` -> M7
