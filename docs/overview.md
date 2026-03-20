# ECOmpile Overview

ECOmpile envisions self-refining AI systems that observe their own inference traces, extract stable reasoning motifs, and promote them into deterministic code. (The raw exports occasionally reference an earlier working title; treat every instance as meaning ECOmpile.) This document condenses the high-level narrative from `1.md` (public overview at lines 285-352) and `2.md` (integration commentary around lines 595-720).

ECOmpile embodies the idea of AI models reinforcing their own soft logic into stable, reliable, and sustainable structure—creativity wrapped in dependable scaffolding.

It should also harvest useful retrospection artifacts from its own working environment, especially chat-derived rules, decisions, heuristics, and correction patterns that can be compiled into more stable operating behavior.

## Core Narrative

- **Problem** – Pure neural inference is expensive, opaque, and prone to hallucinations (20–75% in reasoning tasks per Vectara 2025 leaderboard).
- **Vision** – Teach models to analyze throughput, detect deterministic subgraphs, and rewrite them as compiled libraries.
- **Outcome** – Hybrid architectures where creativity remains neural, while reliability and efficiency come from code.

## Key Value Pillars

| Pillar | Description |
| --- | --- |
| Reliability | Hardened code paths remove stochastic drift and simplify audits. |
| Efficiency | Stable functions run on commodity CPUs, cutting inference cost by 60–80%. |
| Transparency | Extracted modules are human-readable, testable, and certifiable. |
| Sustainability | Roughly 100× energy savings vs. GPU-only inference (~0.004 kWh/task vs. 0.4 kWh GPU hour). |
| Evolution | Models become self-compilers, continuously crystallizing their best reasoning. |
| Self-harvesting | The system captures reusable traits and only names failures explicitly when that improves prevention or correction. |

## Retrospection Harvesting Rule

- Traits are the default harvest target.
- Failures are harvested when needed, not as the primary narrative.
- Trait and failure artifacts remain linked when one explains the absence, inversion, or misuse of the other.
- Chat-location and local artifact harvesting should start in user space and escalate lower only when justified by missing visibility.

## Audience Layers

1. **Public / Popular** – Analogies (jazz riffs → sheet music) to explain the concept without jargon.
2. **Technical** – Architecture, algorithms, SDK plans, and validation tables.
3. **Investor** – Market sizing, roadmap, funding ask, and IP posture.

## Reading Tips

- Use this overview as a map, then jump to `docs/architecture.md` for system details.
- Refer back to `1.md` or `2.md` for verbatim language when needed; line anchors are noted in each curated doc.
