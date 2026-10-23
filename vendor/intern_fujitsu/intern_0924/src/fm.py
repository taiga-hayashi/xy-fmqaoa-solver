"""
Factorization Machine (FM) surrogate model using PyTorch.
Computes 2nd-order feature interactions in O(kd) time complexity.
Includes training routine and comprehensive accuracy metrics calculation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, kendalltau
from typing import Dict, Any, Tuple


class TorchFM(nn.Module):
    def __init__(self, d: int, k: int = 2):
        super().__init__()
        self.d = d
        self.k = k
        self.V = nn.Parameter(torch.randn(d, k))
        self.lin = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d)
        term1_inner = x @ self.V  # (B, k)
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)
        term2_inner = (x ** 2) @ (self.V ** 2)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)
        interaction = 0.5 * (term1 - term2)
        return (interaction + self.lin(x)).view(-1)


def train_fm(
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    k: int = 2,
    epochs: int = 120,
    lr: float = 0.1,
    seed: int = 42
) -> TorchFM:
    torch.manual_seed(seed)
    model = TorchFM(d=d, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def evaluate_fm_accuracy(
    model: TorchFM,
    test_X: np.ndarray,
    test_y: np.ndarray
) -> Dict[str, float]:
    """
    Calculate RMSE, MAE, Spearman rho, Kendall tau, and Top-k recall.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(test_X, dtype=torch.float32)
        preds = model(x_tensor).cpu().numpy()

    err = preds - test_y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    # Rank correlations
    rho, _ = spearmanr(preds, test_y)
    tau, _ = kendalltau(preds, test_y)
    rho = float(rho) if not np.isnan(rho) else 0.0
    tau = float(tau) if not np.isnan(tau) else 0.0

    # Top-k recall (minimization problem: lowest values are best)
    n_samples = len(test_y)
    sorted_true_indices = np.argsort(test_y)
    sorted_pred_indices = np.argsort(preds)

    def calc_recall(pct: float) -> float:
        k_count = max(1, int(np.ceil(n_samples * pct)))
        top_true = set(sorted_true_indices[:k_count])
        top_pred = set(sorted_pred_indices[:k_count])
        intersection = len(top_true.intersection(top_pred))
        return float(intersection / k_count)

    recall_top1 = calc_recall(0.01)
    recall_top5 = calc_recall(0.05)
    recall_top10 = calc_recall(0.10)

    return {
        "rmse": rmse,
        "mae": mae,
        "spearman_rho": rho,
        "kendall_tau": tau,
        "top1_recall": recall_top1,
        "top5_recall": recall_top5,
        "top10_recall": recall_top10
    }
