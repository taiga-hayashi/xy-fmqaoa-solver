"""
One-Hot penalty matrix construction and Adaptive penalty weight management.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List


def build_one_hot_penalty_qubo(G: int) -> Tuple[np.ndarray, float]:
    """
    Construct the QUBO matrix and constant offset for the One-Hot penalty:
    P(x) = sum_{g=0}^{G-1} (sum_{j=0}^2 x_{3g+j} - 1)^2
    Since x_i^2 = x_i for binary variables:
    (sum x_i - 1)^2 = sum x_i + 2 sum_{i < j} x_i x_j - 2 sum x_i + 1
                    = - sum x_i + 2 sum_{i < j} x_i x_j + 1
    Returns:
        Q_pen: shape (3*G, 3*G) upper-triangular matrix
        offset_pen: float = G * 1.0
    """
    N = 3 * G
    Q_pen = np.zeros((N, N), dtype=np.float64)
    for g in range(G):
        start = 3 * g
        # Diagonal terms: -1.0
        for i in range(start, start + 3):
            Q_pen[i, i] = -1.0
        # Off-diagonal terms within block: +2.0
        for i in range(start, start + 3):
            for j in range(i + 1, start + 3):
                Q_pen[i, j] = 2.0
    offset_pen = float(G)
    return Q_pen, offset_pen


def apply_penalty_to_qubo(
    Q: np.ndarray,
    offset: float,
    G: int,
    lambda_val: float
) -> Tuple[np.ndarray, float]:
    """
    Add lambda * Penalty to the given QUBO.
    """
    Q_pen, offset_pen = build_one_hot_penalty_qubo(G)
    Q_total = Q + lambda_val * Q_pen
    offset_total = offset + lambda_val * offset_pen
    return Q_total, offset_total


class AdaptivePenaltyTracker:
    """
    Manages and updates alpha_t dynamically across FMQA iterations.
    """
    def __init__(self, initial_alpha: float = 1.0, min_alpha: float = 0.01, max_alpha: float = 100.0):
        self.alpha_t = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.history: List[float] = []

    def get_current_alpha(self) -> float:
        return self.alpha_t

    def update(self, raw_feasible_rate: float) -> float:
        """
        Adaptively update alpha_t based on constraint feasibility.
        If feasible rate is low, increase alpha_t to enforce constraint.
        If feasible rate is 100%, slightly decrease alpha_t to favor objective exploration.
        """
        self.history.append(self.alpha_t)
        if raw_feasible_rate < 0.7:
            self.alpha_t = min(self.max_alpha, self.alpha_t * 2.0)
        elif raw_feasible_rate < 1.0:
            self.alpha_t = min(self.max_alpha, self.alpha_t * 1.25)
        else:
            self.alpha_t = max(self.min_alpha, self.alpha_t * 0.85)
        return self.alpha_t
