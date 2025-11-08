# ECOmpile

ECOmpile is a research and productization effort that turns **self-assessing neural systems** into **hybrid neural‑code stacks**. The repository curates the source exports (`1.md`, `2.md`), adds structured documentation, and outlines SDK prototypes so the material can be shared as a cohesive public repo without touching the original files.

Think of it as an AI refinery: models improvise, measure what works best, and then forge those reliable motifs into lean code that anyone can inspect. The result is software that stays creative when it needs to be, yet behaves like dependable infrastructure where it counts.

Public stakeholders—from curious readers to CTOs—can explore this repo to understand why ECOmpile matters for cost, safety, transparency, and sustainability, all without parsing multi-thousand-line transcripts. Every curated document points back to the original exports, so nothing is lost and everything stays verifiable.

> **All Rights Reserved — © 2025 Slavko Stojnić**

---

## Why this repo exists

- Preserve the long-form source documents exactly as authored.
- Offer a navigable, citation-friendly summary for stakeholders who do not want to parse multi-thousand-line transcripts.
- Capture governance, roadmap, and SDK guidance so collaborators can extend ECOmpile without rewriting the original work.

---

## Repository layout

```
.
├── 1.md / 2.md         # Original exports (untouched)
├── README.md           # You are here
├── docs/               # Curated documentation layers
│   ├── overview.md
│   ├── architecture.md
│   ├── roadmap.md
│   ├── governance.md
│   └── references.md
├── sdk/                # Early SDK notes & executable examples
│   ├── README.md
│   └── examples/
│       ├── trace_capture_stub.py
│       ├── federated_pilot.py
│       └── nesy_benchmark.py
├── notes/              # Internal critique (gitignored)
└── .gitignore
```

---

## Reading order

1. **docs/overview.md** – Executive summary + terminology.
2. **docs/architecture.md** – Deep dive into the trace→code pipeline, with Mermaid diagrams and algorithm callouts.
3. **docs/roadmap.md** – Delivery phases, milestones, and KPIs distilled from the source material.
4. **docs/governance.md** – Risk, compliance, and environmental framing.
5. **docs/references.md** – Citation list for every numeric claim surfaced in the curated docs.
6. **docs/public_release.md** – Single-file dossier ready for publication.
7. **sdk/examples/** – Lightweight Python references that illustrate trace capture, symbolic distillation, and benchmarking flows.

> **Diagram note:** GitHub renders the Mermaid diagrams inline, but if you need static images run `npx @mermaid-js/mermaid-cli -i docs/architecture.md -o diagrams/architecture.png` (or a similar `mmdc` command) and attach the PNGs to releases.

---

## Document provenance

- `1.md` and `2.md` remain byte-for-byte identical to the supplied material (they stay in the working tree but are gitignored so they never leave your machine unless you explicitly add them). Any legacy references to earlier working titles inside those exports should be read as “ECOmpile.”
- All derivative documents quote, summarize, or reorganize concepts but **never modify** the originals. When referencing a section, the curated docs call out the line anchors (e.g., `1.md:285` for the public narrative).

---

## License

This repository is distributed for archival and review purposes only.

```
© 2025 Slavko Stojnić — All Rights Reserved.
No rights are granted to copy, modify, or distribute without explicit permission.
```
