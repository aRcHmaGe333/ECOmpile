# ECOmpile SDK Notes

The SDK concept sketches how to operationalize the ECOmpile pipeline. These files are **illustrative stubs** that developers can adapt when implementing the full stack described in `docs/architecture.md`.

## Modules

| Component | Purpose | Status |
| --- | --- | --- |
| `trace_capture_stub.py` | Demonstrates attaching profiling hooks and generating stability manifests. | Prototype |
| `federated_pilot.py` | Simulates a federated NeSy loop (~10 clients) with symbolic distillation at the edge. | Prototype |
| `nesy_benchmark.py` | Shows how to blend neural logits with symbolic rules on a parity task. | Prototype |
| `openai_handoff_compiler.py` | Packages external behavioral evidence + internal telemetry contract for OpenAI-side validation. | Prototype |

## Running Examples

```bash
python sdk/examples/trace_capture_stub.py
python sdk/examples/federated_pilot.py
python sdk/examples/nesy_benchmark.py
python sdk/examples/openai_handoff_compiler.py --case-file cases/2026-03-04_unknown-contact-sid-removal/conversation.md --spec-file cases/2026-03-04_unknown-contact-sid-removal/openai_handoff_spec.json --strict
```

> Tested on Python 3.10 with PyTorch 2.2, SymPy 1.12, NumPy 1.26, and scikit-learn 1.5.

## Extending

- Replace the random data generators with actual activation captures from your model.
- Export real stability manifests (`.nfrsg`) and feed them into downstream compilation steps.
- Contribute additional examples (e.g., governance checks, audit log writers) in the same folder.

Remember to cite the original source lines from `1.md`/`2.md` when adding new logic.

## OpenAI Handoff Prerequisites

For an email-aligned handoff package:

1. Case trace must be in the repository case format:
- alternating `## Prompt:` and `## Response:` blocks.

2. Case handoff spec JSON must be explicit:
- `kernel_id`, `platform`, `context`,
- `intent_required`,
- `failure_patterns`,
- `correction_patterns`,
- `internal_trace_events_required`,
- `forbid_patterns`.

3. Run with `--strict`:
- fails fast if behavioral evidence chain is incomplete.

4. Run it after a corrected response appears in the case:
- compile the package after case closure,
- attach generated summary/contract artifacts to outreach.

## Scope Boundary

1. Chat logs can produce behavioral evidence and kernel candidates.
2. True model-internal introspection requires internal telemetry APIs and replay-grade event logs in the host environment.
