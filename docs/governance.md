# Governance, Ethics, and Environmental Framework

Summarizes the controls described around `1.md:639-660`, `1.md:1904-1935`, and the expanded governance notes scattered through `2.md`.

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
| --- | --- | --- | --- |
| Over-generalization (loss of nuance) | Medium | High | Maintain neural fallback, enforce Δ < 1e-3 equivalence tests. |
| Code drift vs. model drift | High | Medium | Versioned module registry + scheduled revalidation. |
| Bias hardening | Low | High | Bias audits before locking modules, dataset diversity ≥ 80%. |
| Security of generated code | Medium | High | Sandboxed compilation, static analysis, signed artifacts. |
| Adoption lag | High | Medium | Low-risk chat/productivity pilots with ROI dashboards. |
| Regulatory non-compliance | Low | High | Align with ISO/IEC 23894, ISO/IEC 42001, EU AI Act transparency logs. |

## Compliance Checklist

- [ ] Immutable audit logs (SHA-256) for each neural→code promotion.
- [ ] Provenance file (`.nfrmeta`) capturing inputs, reviewers, and approvals.
- [ ] Explainability summaries (symbolic function description + test vectors).
- [ ] Human-in-the-loop sign-off before deployment to production workloads.
- [ ] Environmental report referencing AI Index 2025 methodology (0.4 kWh GPU vs. 0.004 kWh CPU inference).

## Environmental Footnote

> Energy comparison assumes 1 GPU hour on an NVIDIA A100 ≈ 0.4 kWh, while the compiled CPU path consumes ≈ 0.004 kWh per equivalent task, delivering ~100× savings.

## Regulatory Alignment

- **EU AI Act** — classify self-modifying NeSy systems as high-risk; maintain transparency artifacts for notified bodies.
- **US Executive Order on Safe AI (2023, updated 2025)** — document safety tests and bias monitoring.
- **ISO/IEC 42001 (AI management)** — map ECOmpile processes (trace capture, validation) to required controls.

Keep this document synchronized with any new risk assessments or compliance evidence added elsewhere in the repo.
