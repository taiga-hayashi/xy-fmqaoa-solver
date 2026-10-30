"""Penalty-free Categorical Simulated Annealing directly on {0, 1, 2, 3}^L."""
from __future__ import annotations

import numpy as np

from ..optics import labels_for, onehot


def _strides(layers: int) -> np.ndarray:
    return 4 ** np.arange(layers - 1, -1, -1, dtype=np.int64)


def move_energy_range(
    pred: np.ndarray,
    layers: int,
    rng: np.random.Generator,
    n_samples: int = 4096,
) -> tuple[float, float]:
    idx = rng.integers(0, 4**layers, n_samples)
    lab = labels_for(layers)[idx].astype(np.int64)
    st = _strides(layers)
    diffs = []
    for g in range(layers):
        for c in range(4):
            keep = lab[:, g] != c
            nb = idx[keep] + (c - lab[keep, g]) * st[g]
            diffs.append(np.abs(pred[nb] - pred[idx[keep]]))
    d = np.concatenate(diffs)
    pos = d[d > 1e-12]
    return float(d.max()), float(pos.min()) if len(pos) else 1.0


def sample_categorical_sa(
    pred: np.ndarray,
    layers: int,
    seed: int,
    *,
    num_reads: int = 100,
    num_sweeps: int = 1000,
) -> tuple[np.ndarray, dict]:
    """100 independent chains x `num_sweeps` sweeps with single-layer material moves."""
    rng = np.random.default_rng(seed)
    st = _strides(layers)
    dmax, dmin = move_energy_range(pred, layers, rng)
    betas = np.geomspace(
        np.log(2.0) / max(dmax, 1e-300),
        np.log(100.0) / max(dmin, 1e-300),
        num_sweeps,
    )
    lab = rng.integers(0, 4, (num_reads, layers))
    idx = (lab * st).sum(1)
    for beta in betas:
        for g in range(layers):
            new = (lab[:, g] + rng.integers(1, 4, num_reads)) % 4
            nidx = idx + (new - lab[:, g]) * st[g]
            acc = np.log(np.maximum(rng.random(num_reads), np.finfo(float).tiny)) < -beta * (
                pred[nidx] - pred[idx]
            )
            lab[acc, g] = new[acc]
            idx = np.where(acc, nidx, idx)
    return onehot(lab), {
        "sampler": "categorical_sa",
        "beta_range": [float(betas[0]), float(betas[-1])],
        "num_sweeps": num_sweeps,
        "sampler_seed": seed,
    }

