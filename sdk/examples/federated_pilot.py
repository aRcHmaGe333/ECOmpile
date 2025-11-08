"""
Federated neurosymbolic learning (FedNSL) pilot.

Simulates 10 clients that train locally on synthetic data, share weights via
FedAvg, and distill symbolic linear rules after each round. This mirrors the
edge deployment concept described throughout `2.md`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim


ARTIFACT_DIR = Path("artifacts/federated")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class NeSyClient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neural = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
        self.symbolic_func = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neural_out = self.neural(x)
        if self.symbolic_func is None:
            return neural_out

        sym_vals = torch.tensor(
            self.symbolic_func(x.detach().numpy().flatten()),
            dtype=torch.float32,
        ).unsqueeze(1)
        return 0.5 * (neural_out + sym_vals)

    def distill_symbolic(self, samples: np.ndarray) -> None:
        """Fit a simple line y = ax + b on local samples."""

        x_sym = sp.symbols("x")
        slope, intercept = np.polyfit(samples[:, 0], samples[:, 1], 1)
        self.symbolic_func = sp.lambdify(x_sym, slope * x_sym + intercept, "numpy")


def synthetic_dataset(size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(size, 1)
    noise = torch.randn(size, 1) * 0.05
    y = 2 * x + 1 + noise
    return x, y


def client_step(client: NeSyClient, data: Tuple[torch.Tensor, torch.Tensor], epochs: int = 2) -> None:
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=8)
    opt = optim.SGD(client.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            pred = client(batch_x)
            loss = loss_fn(pred, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()


def federated_round(clients: List[NeSyClient]) -> Dict[str, torch.Tensor]:
    global_state: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for idx, client in enumerate(clients):
            data = synthetic_dataset()
            client_step(client, data)

            # Promote symbolic rule using last few samples
            sample = torch.cat((data[0][:5], data[1][:5]), dim=1).numpy()
            client.distill_symbolic(sample)

            state = client.state_dict()
            if not global_state:
                global_state = {k: v.clone() for k, v in state.items()}
            else:
                for name, tensor in state.items():
                    global_state[name] += tensor

        for name in global_state:
            global_state[name] /= len(clients)

    for client in clients:
        client.load_state_dict(global_state, strict=False)
    return global_state


def evaluate(model: NeSyClient) -> float:
    x_test = torch.linspace(0, 1, steps=32).unsqueeze(1)
    y_test = 2 * x_test + 1
    loss = nn.MSELoss()(model(x_test), y_test).item()
    return loss


def plot_losses(loss_history: List[float]) -> Path:
    path = ARTIFACT_DIR / "fed_loss.png"
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color="#2c7fb8")
    plt.xlabel("Round")
    plt.ylabel("MSE Loss")
    plt.title("FedNSL Convergence (Synthetic data)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)
    clients = [NeSyClient() for _ in range(10)]

    losses: List[float] = []
    for round_idx in range(3):
        federated_round(clients)
        loss = evaluate(clients[0])
        losses.append(loss)
        print(f"[fed] round {round_idx + 1} loss={loss:.4f}")

    artifact_path = plot_losses(losses)
    print(f"[fed] loss plot saved to {artifact_path}")


if __name__ == "__main__":
    main()

