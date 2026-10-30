"""Dual-backend Compact Pair-Decomposed XY-FMQAOA (`qarpx` + pure `numpy`).

Both backends simulate the exact same quantum circuit on the One-Hot encoded subspace:
  - Initial state: uniform superposition over all 4^L feasible states (product of 4-qubit W states).
  - Cost layer: exp(-i * gamma * E(z)), decomposed into single-layer and layer-pair diagonals.
  - Mixer layer: ordered ring XY edges (0,1)->(1,2)->(2,3)->(3,0) applied independently to each layer.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from ..optics import labels_for, onehot

GRID = {
    "gammas": [0.0, 0.05, 0.2, 0.4, 0.8, 1.6, 3.2],
    "betas": [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6],
}


class _BudgetReached(Exception):
    pass


def _ordered_ring_xy_unitary(size: int, width: int, beta: float) -> np.ndarray:
    """Exact (2^width x 2^width) unitary for ordered ring XY edges (0,1)->...->(size-1,0)."""
    unitary = np.eye(1 << width, dtype=np.complex128)
    edges = [(0, 1)] if size == 2 else [(k, (k + 1) % size) for k in range(size)]
    cb = np.cos(beta)
    sb = np.sin(beta)
    for a, b in edges:
        r0 = unitary[a].copy()
        r1 = unitary[b].copy()
        unitary[a] = cb * r0 - 1j * sb * r1
        unitary[b] = cb * r1 - 1j * sb * r0
    return unitary


class NumpyPairXYQAOA:
    """Fast, zero-dependency exact statevector simulator via tensor contractions."""

    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int]) -> None:
        self.qubo = np.asarray(qubo, dtype=np.float64)
        self.block_sizes = tuple(int(s) for s in block_sizes)
        if np.any(np.tril(self.qubo, -1) != 0):
            raise ValueError("XY backend expects an upper-triangular QUBO")
        if sum(self.block_sizes) != self.qubo.shape[0]:
            raise ValueError("sum(block_sizes) must equal QUBO dimension")

        # Assemble exact feasible energies via single-group + group-pair decomposition
        self.physical_starts = np.cumsum([0, *self.block_sizes[:-1]]).tolist()
        n_groups = len(self.block_sizes)
        shape = self.block_sizes
        energy_tensor = np.zeros(shape, dtype=np.float64)

        for g, size in enumerate(self.block_sizes):
            idx = self.physical_starts[g] + np.arange(size)
            diag_g = self.qubo[idx, idx]
            bshape = [1] * n_groups
            bshape[g] = size
            energy_tensor = energy_tensor + diag_g.reshape(bshape)

        for g in range(n_groups):
            ig = self.physical_starts[g] + np.arange(self.block_sizes[g])
            for h in range(g + 1, n_groups):
                ih = self.physical_starts[h] + np.arange(self.block_sizes[h])
                pair_block = self.qubo[np.ix_(ig, ih)]
                bshape = [1] * n_groups
                bshape[g] = self.block_sizes[g]
                bshape[h] = self.block_sizes[h]
                energy_tensor = energy_tensor + pair_block.reshape(bshape)

        self.energies = energy_tensor.ravel()
        self.n_feasible = int(np.prod(self.block_sizes))
        self._mixer_mats: dict[tuple[int, float], np.ndarray] = {}

    def _mixer_matrix(self, size: int, beta: float) -> np.ndarray:
        key = (size, float(beta))
        if key not in self._mixer_mats:
            width = max(1, (size - 1).bit_length())
            self._mixer_mats[key] = _ordered_ring_xy_unitary(size, width, float(beta))[:size, :size]
            if len(self._mixer_mats) > 128:
                self._mixer_mats.clear()
                self._mixer_mats[key] = _ordered_ring_xy_unitary(size, width, float(beta))[:size, :size]
        return self._mixer_mats[key]

    def feasible_probabilities(
        self, gammas: Sequence[float], betas: Sequence[float]
    ) -> np.ndarray:
        if len(gammas) != len(betas) or len(gammas) == 0:
            raise ValueError("gammas and betas must be non-empty sequences of equal length")
        gammas_arr = np.asarray(gammas, dtype=np.float64)
        betas_arr = np.asarray(betas, dtype=np.float64)
        if not (np.isfinite(gammas_arr).all() and np.isfinite(betas_arr).all()):
            raise ValueError("Angles must be finite")

        state = np.full(self.n_feasible, 1.0 / np.sqrt(self.n_feasible), dtype=np.complex128)
        n_groups = len(self.block_sizes)

        for gamma, beta in zip(gammas_arr, betas_arr):
            # Cost unitary on feasible basis
            state *= np.exp(-1j * float(gamma) * self.energies)
            # Independent ordered-ring XY mixer on each group axis
            left_dim = 1
            for g, size in enumerate(self.block_sizes):
                right_dim = self.n_feasible // (left_dim * size)
                u_mat = self._mixer_matrix(size, float(beta))
                tensor_3d = state.reshape(left_dim, size, right_dim)
                # Contract along middle axis: out[l, a, r] = sum_b u_mat[a, b] * tensor_3d[l, b, r]
                state = np.matmul(u_mat, tensor_3d).reshape(self.n_feasible)
                left_dim *= size

        probs = np.abs(state) ** 2
        norm = float(probs.sum())
        if not np.isfinite(norm) or abs(1.0 - norm) >= 1e-10:
            raise RuntimeError(f"Statevector norm drift: {norm}")
        return probs / norm


class QarpxPairXYQAOA:
    """Self-contained OpenQARP (`qarpx`) pair-decomposed compact XY-QAOA backend."""

    _mixer_cache: dict = {}

    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int]) -> None:
        self.qubo = np.asarray(qubo, dtype=np.float64)
        self.block_sizes = tuple(int(s) for s in block_sizes)
        if np.any(np.tril(self.qubo, -1) != 0):
            raise ValueError("XY backend expects an upper-triangular QUBO")
        self.register_widths = [max(1, (s - 1).bit_length()) for s in self.block_sizes]
        self.simulator_qubits = sum(self.register_widths)
        self.physical_starts = np.cumsum([0, *self.block_sizes[:-1]]).tolist()
        self.encoded_starts = np.cumsum([0, *self.register_widths[:-1]]).tolist()

        choices_grid = np.array(
            list(itertools.product(*(range(s) for s in self.block_sizes))), dtype=np.uint64
        )
        indices = np.zeros(len(choices_grid), dtype=np.uint64)
        for g, estart in enumerate(self.encoded_starts):
            indices += choices_grid[:, g] << np.uint64(estart)
        self.encoded_indices = indices
        self.initial_state = np.zeros(1 << self.simulator_qubits, dtype=np.complex128)
        self.initial_state[self.encoded_indices] = 1.0 / np.sqrt(len(self.encoded_indices))

        self.single = []
        for g, (size, width) in enumerate(zip(self.block_sizes, self.register_widths)):
            e = np.zeros(1 << width, dtype=np.float64)
            idx = self.physical_starts[g] + np.arange(size)
            e[:size] = self.qubo[idx, idx]
            self.single.append(e)

        self.pairs = []
        for g in range(len(self.block_sizes)):
            for h in range(g + 1, len(self.block_sizes)):
                wg, wh = self.register_widths[g], self.register_widths[h]
                table = np.zeros((1 << wh, 1 << wg), dtype=np.float64)
                ig = self.physical_starts[g] + np.arange(self.block_sizes[g])
                ih = self.physical_starts[h] + np.arange(self.block_sizes[h])
                table[: len(ih), : len(ig)] = self.qubo[np.ix_(ig, ih)].T
                self.pairs.append((g, h, table.ravel()))

    @staticmethod
    def _diagonal(qx, phases: np.ndarray, qubits: Sequence[int]) -> list:
        block = qx.SimpleBlock(len(qubits), "pair cost diagonal")
        block.diagonal_unitary(np.exp(-1j * phases).tolist())
        return [c.remap_qubits(list(qubits)) for c in block.commands()]

    def feasible_probabilities(
        self, gammas: Sequence[float], betas: Sequence[float]
    ) -> np.ndarray:
        import qarpx as qx

        if len(gammas) != len(betas) or not len(gammas):
            raise ValueError("finite, nonempty equal-length angles required")
        valid = [np.arange(1 << w) < s for s, w in zip(self.block_sizes, self.register_widths)]
        commands = []
        for gamma, beta in zip(gammas, betas):
            for g, w in enumerate(self.register_widths):
                qubits = range(self.encoded_starts[g], self.encoded_starts[g] + w)
                commands.extend(self._diagonal(qx, float(gamma) * np.where(valid[g], self.single[g], 0.0), qubits))
            for g, h, table in self.pairs:
                wg, wh = self.register_widths[g], self.register_widths[h]
                mask = (valid[h][:, None] & valid[g][None, :]).ravel()
                qubits = [
                    *range(self.encoded_starts[g], self.encoded_starts[g] + wg),
                    *range(self.encoded_starts[h], self.encoded_starts[h] + wh),
                ]
                commands.extend(self._diagonal(qx, float(gamma) * np.where(mask, table, 0.0), qubits))
            for g, (size, width) in enumerate(zip(self.block_sizes, self.register_widths)):
                key = (size, float(beta))
                if key not in self._mixer_cache:
                    unitary = _ordered_ring_xy_unitary(size, width, float(beta))
                    mixer = qx.SimpleBlock(width, "encoded ordered XY edges")
                    mixer.unitary_synthesis(np.asfortranarray(unitary))
                    self._mixer_cache[key] = list(mixer.commands())
                remap = list(range(self.encoded_starts[g], self.encoded_starts[g] + width))
                commands.extend(c.remap_qubits(remap) for c in self._mixer_cache[key])

        circuit = qx.SimpleBlock(self.simulator_qubits, "OpenQARP pair-decomposed compact XY")
        circuit.set_commands(commands)
        circuit.build()
        state = qx.QarpSimulator().statevector(
            circuit.commands(), self.simulator_qubits, initial_state=self.initial_state
        )
        probs = np.abs(state[self.encoded_indices]) ** 2
        if abs(1.0 - np.vdot(state, state).real) >= 1e-10 or abs(1.0 - probs.sum()) >= 1e-10:
            raise RuntimeError("compact XY state norm or One-Hot mass drift")
        return probs


def create_xy_backend(
    scaled_qubo: np.ndarray,
    block_sizes: Sequence[int],
    backend_name: str = "auto",
) -> NumpyPairXYQAOA | QarpxPairXYQAOA:
    if backend_name == "numpy":
        return NumpyPairXYQAOA(scaled_qubo, block_sizes)
    if backend_name == "qarpx":
        return QarpxPairXYQAOA(scaled_qubo, block_sizes)
    if backend_name == "auto":
        try:
            import qarpx  # noqa: F401

            return QarpxPairXYQAOA(scaled_qubo, block_sizes)
        except ImportError:
            return NumpyPairXYQAOA(scaled_qubo, block_sizes)
    raise ValueError(f"Unknown XY backend: {backend_name}")


def cvar(energy: np.ndarray, prob: np.ndarray, alpha: float = 0.1) -> float:
    order = np.argsort(energy, kind="stable")
    p = prob[order] / prob.sum()
    before = np.cumsum(p) - p
    w = np.maximum(0.0, np.minimum(p, alpha - before))
    return float((w @ energy[order]) / alpha)


def interp(angles: Sequence[float], p: int) -> np.ndarray:
    """INTERP (Zhou et al., PRX 2020): lift depth p angles to p + 1."""
    a = np.asarray(angles, dtype=float)

    def _up(x: np.ndarray) -> np.ndarray:
        xp = np.concatenate([[0.0], x, [0.0]])
        return np.array([(i / p) * xp[i] + ((p - i) / p) * xp[i + 1] for i in range(p + 1)])

    return np.concatenate([_up(a[:p]), _up(a[p:])])


def angle_search(
    backend: NumpyPairXYQAOA | QarpxPairXYQAOA,
    pred: np.ndarray,
    scale: float,
    depth: int,
    objective: str,
    starts: Sequence[Sequence[float]],
    budget: int,
    rng: np.random.Generator,
    *,
    n_global: int,
    n_local: int,
    local_maxfev: int,
    grid: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    trace: list[tuple[np.ndarray, float]] = []
    best: list = [None, None]
    maxima = np.array([2.0 * np.pi] * depth + [np.pi] * depth)
    bounds = [(0.0, float(v)) for v in maxima]
    pred_order = np.argsort(pred, kind="stable")
    pred_sorted = pred[pred_order]

    def _value(prob: np.ndarray) -> float:
        if objective == "mean":
            return float(prob @ pred)
        p = prob[pred_order] / prob.sum()
        before = np.cumsum(p) - p
        w = np.maximum(0.0, np.minimum(p, 0.1 - before))
        return float((w @ pred_sorted) / 0.1)

    def _evaluate(a: Sequence[float], stop: int = budget) -> float:
        if len(trace) >= stop:
            raise _BudgetReached()
        if isinstance(backend, QarpxPairXYQAOA) and len(trace) % 32 == 0:
            QarpxPairXYQAOA._mixer_cache.clear()
        arr = np.clip(np.asarray(a, dtype=float), 0.0, maxima)
        prob = backend.feasible_probabilities(arr[:depth], arr[depth:])
        val = _value(prob)
        trace.append((arr, val))
        if best[0] is None or val < best[0][1]:
            best[0], best[1] = (arr, val), prob
        return val / max(scale, 1e-30)

    try:
        for a in starts:
            _evaluate(a)
        if grid:
            for g, bt in itertools.product(GRID["gammas"], GRID["betas"]):
                _evaluate([g, bt])
        while len(trace) < n_global:
            _evaluate(rng.random(2 * depth) * maxima)
        for a in [t[0] for t in sorted(trace, key=lambda t: t[1])[:n_local]]:
            stop = min(budget, len(trace) + local_maxfev)
            try:
                minimize(
                    lambda v: _evaluate(v, stop),
                    a,
                    method="Powell",
                    bounds=bounds,
                    options={"maxfev": local_maxfev, "xtol": 1e-6, "ftol": 1e-9},
                )
            except _BudgetReached:
                pass
        while len(trace) < budget:
            _evaluate(rng.random(2 * depth) * maxima)
    except _BudgetReached:
        pass

    return best[0][0], float(best[0][1]), best[1], len(trace)


def interp_ladder_search(
    backend: NumpyPairXYQAOA | QarpxPairXYQAOA,
    pred: np.ndarray,
    scale: float,
    max_depth: int,
    objective: str,
    evals_per_depth: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Run INTERP p = 1..max_depth ladder matching `trajectory_xy` cycle 1."""
    a_prev: np.ndarray | None = None
    obj = float("inf")
    prob = np.empty(0)
    total_evals = 0
    # Match exact production settings (512 / 8 / 384) when budget >= 1024; scale down gracefully for --smoke
    n_global = 512 if evals_per_depth >= 1024 else min(evals_per_depth, max(8, evals_per_depth // 2))
    n_local = 8 if evals_per_depth >= 1024 else 2
    local_maxfev = 384 if evals_per_depth >= 1024 else max(12, evals_per_depth // 4)

    for d in range(1, max_depth + 1):
        starts = (
            []
            if a_prev is None
            else [
                interp(a_prev, d - 1),
                np.concatenate([a_prev[: d - 1], [0.0], a_prev[d - 1 :], [0.0]]),
            ]
        )
        a_prev, obj, prob, n = angle_search(
            backend,
            pred,
            scale,
            d,
            objective,
            starts,
            evals_per_depth,
            rng,
            n_global=n_global,
            n_local=n_local,
            local_maxfev=local_maxfev,
            grid=(d == 1),
        )
        total_evals += n
    assert a_prev is not None
    return a_prev, obj, prob, total_evals

