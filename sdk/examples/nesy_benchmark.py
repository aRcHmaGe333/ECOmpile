"""
Neurosymbolic parity benchmark.

Combines a simple neural classifier with a symbolic rule that enforces parity,
mirroring the hybrid inference strategy outlined in the ECOmpile whitepaper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def generate_data(samples: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (samples, 4), dtype=torch.float32)
    labels = (bits.sum(dim=1) % 2).long()  # 0 = even, 1 = odd
    return bits, labels


class NeSyParity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neural = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.neural(x)
        parity = (x.sum(dim=1) % 2).long()
        sym_logits = torch.zeros_like(logits)
        sym_logits.scatter_(1, parity.unsqueeze(1), 2.0)  # symbolic boost
        return torch.softmax(logits + sym_logits, dim=1)


def train(model: NeSyParity, x: torch.Tensor, y: torch.Tensor, epochs: int = 200) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate(model: NeSyParity, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        probs = model(x)
        preds = probs.argmax(dim=1).cpu().numpy()
    return accuracy_score(y.cpu().numpy(), preds)


def main() -> None:
    torch.manual_seed(0)
    train_x, train_y = generate_data(512)
    test_x, test_y = generate_data(128)

    model = NeSyParity()
    train(model, train_x, train_y)
    acc = evaluate(model, test_x, test_y)
    print(f"[nesy-benchmark] accuracy={acc:.4f} (symbolic boost should approach 1.0)")


if __name__ == "__main__":
    main()

