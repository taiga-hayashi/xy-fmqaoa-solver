"""One-Hot penalised QUBO construction and D-Wave Neal Simulated Annealing."""
from __future__ import annotations

from typing import Sequence

import dimod
import numpy as np

try:
    from neal import SimulatedAnnealingSampler as _SASampler
    from neal.sampler import default_beta_range as _default_beta_range
except ImportError:
    from dwave.samplers import SimulatedAnnealingSampler as _SASampler
    from dwave.samplers.sa.sampler import default_beta_range as _default_beta_range


class AdaptivePenaltyTracker:
    """Dynamic penalty controller matching `intern_0924/src/penalty.py`."""

    def __init__(
        self,
        initial_alpha: float = 1.0,
        min_alpha: float = 0.01,
        max_alpha: float = 100.0,
    ) -> None:
        self.alpha_t = float(initial_alpha)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.history: list[float] = []

    def get_current_alpha(self) -> float:
        return self.alpha_t

    def update(self, raw_feasible_rate: float) -> float:
        self.history.append(self.alpha_t)
        if raw_feasible_rate < 0.7:
            self.alpha_t = min(self.max_alpha, self.alpha_t * 2.0)
        elif raw_feasible_rate < 1.0:
            self.alpha_t = min(self.max_alpha, self.alpha_t * 1.25)
        else:
            self.alpha_t = max(self.min_alpha, self.alpha_t * 0.85)
        return self.alpha_t


def build_penalty_bqm(
    qubo_dict: dict[tuple[int, int], float],
    block_sizes: Sequence[int],
    lambda_penalty: float = 5.0,
    offset: float = 0.0,
) -> dimod.BinaryQuadraticModel:
    """Build One-Hot penalised BQM matching `fmqa_solver.py` variable ordering."""
    total_vars = sum(block_sizes)
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    bqm.offset = offset
    for i in range(total_vars):
        bqm.add_variable(i)
    for (i, j), w in qubo_dict.items():
        if i == j:
            bqm.add_linear(i, w)
        else:
            bqm.add_quadratic(i, j, w)
    idx = 0
    for sz in block_sizes:
        terms = [(idx + k, 1.0) for k in range(sz)]
        bqm.add_linear_equality_constraint(
            terms,
            lagrange_multiplier=lambda_penalty,
            constant=-1.0,
        )
        idx += sz
    return bqm


def beta_ranges(
    qd: dict[tuple[int, int], float],
    b: float,
    s: float,
    alpha: float,
    layers: int,
) -> tuple[list[float], list[float]]:
    pen = _default_beta_range(build_penalty_bqm(qd, [4] * layers, s * alpha, b))
    fm = _default_beta_range(build_penalty_bqm(qd, [4] * layers, 0.0, b))
    return [float(pen[0]), float(pen[1])], [float(fm[0]), float(fm[1])]


def sample_sa_auto(
    qd: dict[tuple[int, int], float],
    b: float,
    s: float,
    alpha: float,
    layers: int,
    seed: int,
    *,
    num_reads: int = 100,
    num_sweeps: int = 1000,
) -> tuple[np.ndarray, dict]:
    n = 4 * layers
    lam = s * alpha
    bqm = build_penalty_bqm(qd, [4] * layers, lam, b)
    res = _SASampler().sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_schedule_type="geometric",
        seed=seed,
    )
    var_list = list(res.variables)
    col_order = [var_list.index(i) for i in range(n)]
    arr = res.record.sample[:, col_order]
    raw = np.repeat(arr, res.record.num_occurrences, axis=0).astype(np.int8)
    assert raw.shape == (num_reads, n)
    return raw, {
        "alpha": alpha,
        "lambda": lam,
        "S": s,
        "schedule": "auto",
        "beta_range": [float(x) for x in res.info["beta_range"]],
        "num_sweeps": num_sweeps,
        "sampler_seed": seed,
    }


def sample_sa_schedule(
    qd: dict[tuple[int, int], float],
    b: float,
    s: float,
    alpha: float,
    layers: int,
    seed: int,
    schedule: str = "auto",
    *,
    num_reads: int = 100,
    num_sweeps: int = 1000,
) -> tuple[np.ndarray, dict]:
    """Sample Neal SA under 'auto', 'fmscale', 'wide10x', or 'fmscale10x'."""
    if schedule == "auto":
        return sample_sa_auto(
            qd, b, s, alpha, layers, seed, num_reads=num_reads, num_sweeps=num_sweeps
        )
    n = 4 * layers
    lam = s * alpha
    pen, fm = beta_ranges(qd, b, s, alpha, layers)
    if schedule == "fmscale":
        br, sweeps = fm, num_sweeps
    elif schedule == "wide10x":
        br, sweeps = [pen[0], fm[1]], 10 * num_sweeps
    elif schedule == "fmscale10x":
        br, sweeps = fm, 10 * num_sweeps
    else:
        raise ValueError(f"Unknown SA schedule: {schedule}")

    res = _SASampler().sample(
        build_penalty_bqm(qd, [4] * layers, lam, b),
        num_reads=num_reads,
        num_sweeps=sweeps,
        beta_range=br,
        beta_schedule_type="geometric",
        seed=seed,
    )
    var_list = list(res.variables)
    col_order = [var_list.index(i) for i in range(n)]
    arr = res.record.sample[:, col_order]
    raw = np.repeat(arr, res.record.num_occurrences, axis=0).astype(np.int8)
    assert raw.shape == (num_reads, n)
    return raw, {
        "alpha": alpha,
        "lambda": lam,
        "S": s,
        "schedule": schedule,
        "beta_range": [float(x) for x in res.info["beta_range"]],
        "num_sweeps": sweeps,
        "beta_range_penalised_auto": pen,
        "beta_range_fm_only_auto": fm,
        "sampler_seed": seed,
    }

