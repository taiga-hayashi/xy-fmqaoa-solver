"""Factorization Machine (FM) surrogate, QUBO extraction, and Ising scale S."""
from __future__ import annotations

import hashlib
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class TorchFM(nn.Module):
    """Rank-k Factorization Machine matching the exact parameter RNG order of `fm.py`."""

    def __init__(self, d: int, k: int = 2) -> None:
        super().__init__()
        self.V = nn.Parameter(torch.randn(d, k))
        self.lin = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1_inner = x @ self.V
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)
        term2_inner = x.pow(2) @ self.V.pow(2)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)
        interaction = 0.5 * (term1 - term2)
        return (interaction + self.lin(x)).view(-1)


def train_factorization_machine(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    d: int,
    *,
    k: int = 2,
    epochs: int = 120,
    learning_rate: float = 0.1,
    optimizer_name: str = "AdamW",
    weight_decay: float = 0.01,
) -> TorchFM:
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    model = TorchFM(d=d, k=k)
    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model


def fm_to_qubo(model: TorchFM) -> tuple[dict[tuple[int, int], float], float]:
    linear_weights = model.lin.weight.detach().cpu().numpy().flatten()
    bias = float(model.lin.bias.detach().cpu().numpy().item())
    v_matrix = model.V.detach().cpu().numpy()
    d = v_matrix.shape[0]

    qubo: dict[tuple[int, int], float] = {}
    for i in range(d):
        qubo[(i, i)] = float(linear_weights[i])
    for i in range(d):
        for j in range(i + 1, d):
            qubo[(i, j)] = float(np.dot(v_matrix[i], v_matrix[j]))
    return qubo, bias


def qubo_dict_to_upper_matrix(qdict: dict[tuple[int, int], float], n: int) -> np.ndarray:
    q = np.zeros((n, n), dtype=np.float64)
    for (i, j), v in qdict.items():
        q[min(i, j), max(i, j)] += v
    return q


def qubo_to_ising_coefficients(
    qubo: np.ndarray,
    block_sizes: Sequence[int] | None = None,
    lambda_penalty: float = 0.0,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    mat = np.asarray(qubo, dtype=np.float64).copy()
    if block_sizes and lambda_penalty:
        start = 0
        for size in block_sizes:
            for idx in range(start, start + size):
                mat[idx, idx] -= lambda_penalty
            for left in range(start, start + size):
                for right in range(left + 1, start + size):
                    mat[left, right] += 2.0 * lambda_penalty
            start += size
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    for left in range(mat.shape[0]):
        linear[left] = linear.get(left, 0.0) - float(mat[left, left]) / 2.0
        for right in range(left + 1, mat.shape[0]):
            coeff = float(mat[left, right])
            if coeff:
                linear[left] -= coeff / 4.0
                linear[right] = linear.get(right, 0.0) - coeff / 4.0
                quadratic[(left, right)] = coeff / 4.0
    return linear, quadratic


def model_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit_fm(
    x: np.ndarray,
    y: Sequence[float],
    seed: int,
    *,
    rank: int = 2,
    epochs: int = 120,
    learning_rate: float = 0.1,
    optimizer_name: str = "AdamW",
    weight_decay: float = 0.01,
) -> tuple[TorchFM, dict[tuple[int, int], float], np.ndarray, float, float]:
    n = x.shape[1]
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    m = train_factorization_machine(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        n,
        k=rank,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
    )
    m.eval()
    qd, b = fm_to_qubo(m)
    q = qubo_dict_to_upper_matrix(qd, n)
    h, j = qubo_to_ising_coefficients(q)
    s = max([abs(v) for v in h.values()] + [abs(v) for v in j.values()] + [0.0])
    return m, qd, q, b, s


def predict_all(x_all: np.ndarray, q: np.ndarray, b: float, chunk: int = 1 << 15) -> np.ndarray:
    out = np.empty(len(x_all), dtype=np.float64)
    for first in range(0, len(x_all), chunk):
        xx = x_all[first : first + chunk].astype(np.float64)
        out[first : first + chunk] = np.einsum("bi,ij,bj->b", xx, q, xx) + b
    return out


def initial_ids(seed: int, layers: int, count: int = 40) -> list[int]:
    return random.Random(seed).sample(range(4**layers), count)


def bits_to_ids(raw: np.ndarray, layers: int) -> list[int | None]:
    g = raw.reshape(len(raw), layers, 4)
    valid = (g.sum(2) == 1).all(1)
    idx = (g.argmax(2).astype(np.int64) * (4 ** np.arange(layers - 1, -1, -1))).sum(1)
    return [int(i) if v else None for i, v in zip(idx, valid)]


def select_candidate(
    raw: np.ndarray,
    pred: np.ndarray,
    x_all: np.ndarray,
    seen: set[int],
    layers: int,
) -> tuple[int | None, dict]:
    ids = bits_to_ids(raw, layers)
    unseen = {i for i in ids if i is not None and i not in seen}
    chosen = min(unseen, key=lambda i: (pred[i], tuple(x_all[i]))) if unseen else None
    return chosen, {
        "raw_feasible_rate": sum(i is not None for i in ids) / len(ids),
        "invalid_count": sum(i is None for i in ids),
        "evaluated_duplicate_count": sum(i in seen for i in ids if i is not None),
        "unique_unseen_count": len(unseen),
        "within_unseen_duplicate_count": sum(i is not None and i not in seen for i in ids) - len(unseen),
    }

