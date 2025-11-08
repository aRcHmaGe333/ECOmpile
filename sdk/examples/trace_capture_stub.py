"""
Prototype trace capture + stability scoring stub.

This script emulates the instrumentation loop described in `docs/architecture.md`
without requiring a heavyweight model. Replace the synthetic model with a real
transformer to collect actionable activation traces.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


TRACE_DIR = Path("artifacts/traces")
TRACE_DIR.mkdir(parents=True, exist_ok=True)


class TinyBlock(nn.Module):
    """Minimal MLP block used solely for demonstrating hooks."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ToyModel(nn.Module):
    """Stack of blocks to mimic a transformer encoder."""

    def __init__(self, layers: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(dim, dim * 2) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = x + block(x)  # residual
        return x


@dataclass
class TraceRecord:
    layer: int
    mean: float
    std: float
    token_id: int


def capture_traces(
    model: nn.Module, inputs: Sequence[torch.Tensor], runs: int = 10
) -> List[TraceRecord]:
    """Registers forward hooks and aggregates activation statistics."""

    activations: Dict[int, List[torch.Tensor]] = {}

    def hook(layer_idx: int):
        def _inner(_module, _inp, output):
            activations.setdefault(layer_idx, []).append(output.detach().cpu())

        return _inner

    handles = [layer.register_forward_hook(hook(idx)) for idx, layer in enumerate(model.layers)]
    try:
        for token_id in range(runs):
            x = random.choice(inputs)
            model(x)
            # store token metadata for reproducibility
            activations.setdefault(-1, []).append(torch.tensor([token_id], dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()

    traces: List[TraceRecord] = []
    for layer_idx, tensors in activations.items():
        if layer_idx < 0:
            continue
        stacked = torch.stack(tensors)
        traces.append(
            TraceRecord(
                layer=layer_idx,
                mean=float(stacked.mean().item()),
                std=float(stacked.std().item()),
                token_id=int(token_id),
            )
        )
    return traces


def stability_score(mean: float, std: float, eps: float = 1e-6) -> float:
    """Implements S = 1 - σ(A)/(μ(A)+ε)."""

    return 1.0 - (std / (mean + eps))


def persist_manifest(traces: List[TraceRecord], threshold: float = 0.95) -> Path:
    manifest = []
    for record in traces:
        score = stability_score(record.mean, record.std)
        if math.isnan(score) or score < threshold:
            continue
        manifest.append({"layer": record.layer, "score": score, **asdict(record)})

    path = TRACE_DIR / "stability_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    torch.manual_seed(42)
    model = ToyModel(layers=4, dim=16)
    inputs = [torch.randn(1, 16) for _ in range(20)]

    traces = capture_traces(model, inputs, runs=25)
    manifest_path = persist_manifest(traces)
    print(f"[trace-capture] wrote {manifest_path} ({len(traces)} traces captured)")


if __name__ == "__main__":
    main()

