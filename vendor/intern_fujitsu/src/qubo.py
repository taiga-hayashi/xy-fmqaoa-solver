"""Small, backend-independent QUBO representation helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


QuboDict = Dict[Tuple[int, int], float]


def normalize_upper_triangular_qubo(matrix: np.ndarray, method: str = "none") -> tuple[np.ndarray, float]:
    """Return an upper-triangular QUBO normalized by a positive coefficient scale.

    ``method='max_abs'`` bounds every stored QUBO coefficient by one, while
    ``method='rms'`` makes the RMS of its non-zero stored coefficients one.
    The returned scale satisfies ``normalized = matrix / scale``.  A positive
    global scale leaves the discrete optimum unchanged, but it can change a
    finite-range QAOA angle search and is therefore retained for reporting.
    """
    qubo = np.asarray(matrix, dtype=np.float64)
    if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
        raise ValueError("A QUBO matrix must be square.")
    if method not in {"none", "max_abs", "rms"}:
        raise ValueError("method must be one of: none, max_abs, rms.")
    if method == "none":
        return qubo.copy(), 1.0

    values = qubo[np.triu_indices(qubo.shape[0])]
    nonzero = values[np.nonzero(values)]
    if not len(nonzero):
        return qubo.copy(), 1.0
    scale = float(np.max(np.abs(nonzero))) if method == "max_abs" else float(np.sqrt(np.mean(nonzero**2)))
    return qubo / scale, scale


def matrix_to_qubo_dict(matrix: np.ndarray, tolerance: float = 1e-8) -> QuboDict:
    """Convert an upper-triangular QUBO matrix into the project dictionary form.

    The project convention stores every linear term as ``(i, i)`` and every
    quadratic term once as ``(i, j)`` with ``i < j``.  Values with absolute
    magnitude at or below ``tolerance`` are omitted.
    """
    qubo = np.asarray(matrix, dtype=np.float64)
    if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
        raise ValueError("A QUBO matrix must be square.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    rows, cols = np.triu_indices(qubo.shape[0])
    values = qubo[rows, cols]
    return {
        (int(row), int(col)): float(value)
        for row, col, value in zip(rows, cols, values)
        if abs(value) > tolerance
    }
