# ECOmpile Roadmap

Phases consolidated from the source material (see `1.md:482-520`, `1.md:607-632`, and the investor tables in `2.md`).

| Phase | Timeline | Highlights | Exit Criteria |
| --- | --- | --- | --- |
| Prototype | 0–6 months | Instrument open models (LLaMA/Mistral), capture activation traces, early clustering dashboard. | Stability score reports (≥10 high-confidence subgraphs), reproducibility tests, initial energy baselines. |
| Reconstruction | 6–12 months | Build symbolic regression service, integrate code-gen LLMs, publish SDK alpha (`trace_capture_stub.py`). | ≥5 compiled modules matching neural outputs (Δ < 1e-3), public demo notebook. |
| Validation | 12–18 months | Benchmark vs. GLUE/SuperGLUE, add RLHF-based fallback policy, document compliance hooks. | Hallucination <5%, 60–80% cost reduction on target workloads, auditable logs. |
| Deployment | 18–24 months | Release cloud / on-prem orchestrator, launch federated pilots (edge devices). | Paying design partners, enterprise-ready governance pack, reference architecture. |
| Commercial Scale | 24+ months | Monetize SaaS optimizer, pursue patents + certifications, expand to quantum/neurosymbolic R&D. | ≥$5M ARR pipeline, ISO/IEC 42001 audit readiness, multimodal support. |

## Milestone Checklist

- [ ] Activation tracer open-sourced.
- [ ] SDK sample suite published (`sdk/examples`).
- [ ] Carbon accounting sheet validated against AI Index methodology.
- [ ] Governance review completed (EU AI Act classification, ISO controls).
- [ ] Investor packet synchronized with roadmap (see `docs/overview.md`).

## Dependencies

- Access to profiled models (open weights or proprietary with instrumentation rights).
- Compute budget for multi-run tracing + symbolic regression training.
- Legal review for All-Rights-Reserved release packaging.

Maintain this roadmap via PRs that cite line references from the original exports when adjusting assumptions.
