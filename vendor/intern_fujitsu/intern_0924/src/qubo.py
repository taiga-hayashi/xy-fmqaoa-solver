"""
QUBO and Ising transformation utilities, and coefficient scale S_t calculation.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Tuple, Any


def fm_to_qubo_matrix(model: torch.nn.Module) -> Tuple[np.ndarray, float]:
    """
    Extract upper-triangular QUBO matrix Q and scalar offset from trained TorchFM.
    Q[i, i] = w_i
    Q[i, j] = v_i . v_j for i < j
    """
    w = model.lin.weight.detach().cpu().numpy().flatten()
    bias = float(model.lin.bias.detach().cpu().numpy().item())
    V = model.V.detach().cpu().numpy()
    d = len(w)

    Q = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        Q[i, i] = float(w[i])
        for j in range(i + 1, d):
            Q[i, j] = float(np.dot(V[i], V[j]))

    return Q, bias


def qubo_to_ising(
    qubo: np.ndarray,
    offset: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert upper-triangular QUBO matrix to Ising Hamiltonian:
    x_i = (1 - Z_i)/2
    H = sum_i h_i Z_i + sum_{i < j} J_ij Z_i Z_j + ising_offset
    Returns:
        h (array of size d)
        J (upper-triangular matrix of size (d, d), J[i, j] for i < j)
        ising_offset (float)
    """
    d = qubo.shape[0]
    h = np.zeros(d, dtype=np.float64)
    J = np.zeros((d, d), dtype=np.float64)
    ising_offset = float(offset)

    for i in range(d):
        q_ii = qubo[i, i]
        h[i] -= q_ii / 2.0
        ising_offset += q_ii / 2.0

        for j in range(i + 1, d):
            q_ij = qubo[i, j]
            if q_ij != 0.0:
                h[i] -= q_ij / 4.0
                h[j] -= q_ij / 4.0
                J[i, j] = q_ij / 4.0
                ising_offset += q_ij / 4.0

    return h, J, ising_offset


def calculate_scale_St(h: np.ndarray, J: np.ndarray) -> float:
    """
    Calculate coefficient scale S_t:
    S_t = max( max_i |h_i|, max_{i < j} |J_ij| )
    """
    max_h = float(np.max(np.abs(h))) if len(h) > 0 else 0.0
    upper_J = J[np.triu_indices(J.shape[0], k=1)]
    max_J = float(np.max(np.abs(upper_J))) if len(upper_J) > 0 else 0.0
    St = max(max_h, max_J)
    return max(St, 1e-8)  # prevent division by zero
