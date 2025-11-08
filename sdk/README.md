# ECOmpile SDK Notes

The SDK concept sketches how to operationalize the ECOmpile pipeline. These files are **illustrative stubs** that developers can adapt when implementing the full stack described in `docs/architecture.md`.

## Modules

| Component | Purpose | Status |
| --- | --- | --- |
| `trace_capture_stub.py` | Demonstrates attaching profiling hooks and generating stability manifests. | Prototype |
| `federated_pilot.py` | Simulates a federated NeSy loop (~10 clients) with symbolic distillation at the edge. | Prototype |
| `nesy_benchmark.py` | Shows how to blend neural logits with symbolic rules on a parity task. | Prototype |

## Running Examples

```bash
python sdk/examples/trace_capture_stub.py
python sdk/examples/federated_pilot.py
python sdk/examples/nesy_benchmark.py
```

> Tested on Python 3.10 with PyTorch 2.2, SymPy 1.12, NumPy 1.26, and scikit-learn 1.5.

## Extending

- Replace the random data generators with actual activation captures from your model.
- Export real stability manifests (`.nfrsg`) and feed them into downstream compilation steps.
- Contribute additional examples (e.g., governance checks, audit log writers) in the same folder.

Remember to cite the original source lines from `1.md`/`2.md` when adding new logic.

