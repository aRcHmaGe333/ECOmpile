# ECOmpile

ECOmpile is a project in the R&D phase that aims to turn **self-assessing neural systems** (with a fair bit of HITL) into **hybrid neural‑code stacks**. The repository contains structured documentation and outlines SDK prototypes so the material can be shared as a cohesive public repository.

It's an AI refinery: models improvise, gather best outcomes and then forge those beneficial elements into lean code that can be inspected, accurately, swiftly and (cost-)effectively reproduced. The result is software that stays creative when it needs to be, yet behaves like dependable infrastructure where it counts, not to mention the gains in speed, reliability and the ability to educate, to define, display and reproduce certain actions in a clear, transparent manner.

Public stakeholders — from curious readers to CTOs — can explore this repository to understand why ECOmpile matters for cost, safety, transparency, sustainability, robust and reliable outcomes. 

> **All Rights Reserved — © 2025 Slavko Stojnić**

---

## Why this repo exists

- To inform stakeholders and provide a citation-friendly summary.
- To enlist backing for a high-potential, indispensible concept. 
- To document governance, a roadmap and SDK guidance so that the value of ECOmpile can be actualized.

---

## Repository layout

```
.
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

- All derivative documents quote, summarize, or reorganize concepts but **never modify** the originals.
When referencing a section, the curated docs call out the line anchors (e.g., `1.md:285` for the public narrative).

---

## License

This repository is distributed for archival and review purposes only. 
All offers of collaboration and subsidy are more than welcome. 
e-mail: stojnic.slavko@gmail.com 

```
© 2025 Slavko Stojnić — All Rights Reserved.
No rights are granted to copy, modify, or distribute without explicit permission.
```
