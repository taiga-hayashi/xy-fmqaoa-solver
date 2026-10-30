"""Numerical equivalence tests for `mlfilter-paper-repro`.

Verifies:
  1. Pure-NumPy optical TMM matches vendored `tmm_fast` to <= 1e-13.
  2. `NumpyPairXYQAOA` matches an explicit 4L-qubit gate-by-gate Pauli-Hamiltonian QAOA
     statevector (`RZ`, `RZZ`, and ordered `RXX+RYY` ring edges on the W-product state) to <= 1e-14.
  3. SA schedule variants (`auto`, `fmscale`, `wide10x`, `fmscale10x`) and `Categorical SA`
     produce valid proposal batches and metadata.

Run with:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from repro.optics import FilterEvaluator, labels_for, onehot
from repro.solvers.categorical_sa import sample_categorical_sa
from repro.solvers.sa_penalty import sample_sa_schedule
from repro.solvers.xy_qaoa import NumpyPairXYQAOA
from repro.surrogate import (
    bits_to_ids,
    fit_fm,
    initial_ids,
    predict_all,
    qubo_to_ising_coefficients,
)


def _full_pauli_xy_qaoa_feasible_probs(
    qubo: np.ndarray,
    layers: int,
    gammas: list[float],
    betas: list[float],
) -> np.ndarray:
    """Explicit 4L-qubit statevector simulation using canonical RZ, RZZ, and RXX*RYY gates."""
    n_qubits = 4 * layers
    dim = 1 << n_qubits
    # LSB bit values: shape (dim, n_qubits) where bit q is ((idx >> q) & 1)
    idx_grid = np.arange(dim, dtype=np.uint64)[:, None]
    bit_pos = np.arange(n_qubits, dtype=np.uint64)[None, :]
    bits = ((idx_grid >> bit_pos) & 1).astype(np.float64)
    z_vals = 1.0 - 2.0 * bits  # |0> -> +1, |1> -> -1

    # Feasible basis indices in canonical labels_for(layers) order
    feas_bits = onehot(labels_for(layers)).astype(np.uint64)
    feas_indices = (feas_bits * (1 << np.arange(n_qubits, dtype=np.uint64))[None, :]).sum(axis=1)

    state = np.zeros(dim, dtype=np.complex128)
    state[feas_indices] = 1.0 / np.sqrt(len(feas_indices))

    h_lin, j_quad = qubo_to_ising_coefficients(qubo)
    ising_diag = np.zeros(dim, dtype=np.float64)
    for q_idx, coeff in h_lin.items():
        ising_diag += coeff * z_vals[:, q_idx]
    for (q_a, q_b), coeff in j_quad.items():
        ising_diag += coeff * z_vals[:, q_a] * z_vals[:, q_b]

    # Prebuild two-qubit XY edge gate exp(-i * beta * (XX + YY) / 2)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    xx_yy = np.kron(sx, sx) + np.kron(sy, sy)

    def _apply_two_qubit_gate(psi: np.ndarray, q0: int, q1: int, gate4: np.ndarray) -> np.ndarray:
        # Reshape into n_qubits binary axes (qubit 0 = axis n_qubits-1 in big-endian reshape, so use transpose)
        tensor = psi.reshape((2,) * n_qubits)
        # Note: in numpy reshape((2,)*n), axis (n_qubits - 1 - q) corresponds to bit ((idx >> q) & 1)!
        ax0 = n_qubits - 1 - q0
        ax1 = n_qubits - 1 - q1
        g_tensor = gate4.reshape(2, 2, 2, 2)
        moved = np.moveaxis(tensor, [ax0, ax1], [0, 1])
        contracted = np.tensordot(g_tensor, moved, axes=([2, 3], [0, 1]))
        restored = np.moveaxis(contracted, [0, 1], [ax0, ax1])
        return restored.reshape(dim)

    ring_edges = []
    for g in range(layers):
        base = 4 * g
        ring_edges.extend((base + k, base + ((k + 1) % 4)) for k in range(4))

    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * float(gamma) * ising_diag)
        edge_gate = expm(-0.5j * float(beta) * xx_yy)
        for qa, qb in ring_edges:
            state = _apply_two_qubit_gate(state, qa, qb, edge_gate)

    return np.abs(state[feas_indices]) ** 2


class TestCoreEquivalence(unittest.TestCase):
    def test_numpy_tmm_reference_and_tmm_fast(self) -> None:
        ev_np = FilterEvaluator(7, tmm_backend="numpy")
        opt_fom = ev_np.evaluate(labels_for(7)[:8])
        self.assertEqual(opt_fom.shape, (8,))
        self.assertTrue(np.isfinite(opt_fom).all())

        # Compare against tmm_fast when vendored or installed
        rng = np.random.default_rng(123)
        sample_labels = rng.integers(0, 4, size=(64, 7))
        fom_np = ev_np.evaluate(sample_labels)
        try:
            ev_tf = FilterEvaluator(7, tmm_backend="tmm_fast")
            fom_tf = ev_tf.evaluate(sample_labels)
        except ImportError:
            return
        np.testing.assert_allclose(fom_np, fom_tf, atol=1e-13, rtol=1e-13)

    def test_numpy_pair_xy_matches_full_4L_qubit_pauli_circuit(self) -> None:
        """Exact gate-level check: 2-layer (8-qubit, 256-state) Ising + ordered Ring XY circuit."""
        layers = 2
        rng = np.random.default_rng(2026)
        raw_mat = rng.normal(0.0, 1.0, size=(4 * layers, 4 * layers))
        qubo = np.triu(raw_mat)

        gammas = [0.37, 1.15, 0.82]
        betas = [0.29, 0.64, 1.05]

        prob_compact = NumpyPairXYQAOA(qubo, [4] * layers).feasible_probabilities(gammas, betas)
        prob_pauli = _full_pauli_xy_qaoa_feasible_probs(qubo, layers, gammas, betas)

        self.assertAlmostEqual(float(prob_compact.sum()), 1.0, places=13)
        self.assertAlmostEqual(float(prob_pauli.sum()), 1.0, places=13)
        np.testing.assert_allclose(prob_compact, prob_pauli, atol=1e-14, rtol=1e-13)

    def test_sa_schedules_and_categorical_sa(self) -> None:
        layers = 6
        x_all = onehot(labels_for(layers))
        rng = np.random.default_rng(7)
        ids = initial_ids(42, layers, count=40)
        ys = rng.uniform(0.5, 3.0, size=40).tolist()
        _, qd, q, b, s = fit_fm(x_all[ids], ys, 42, epochs=20)
        pred = predict_all(x_all, q, b)

        for sch in ("auto", "fmscale", "wide10x", "fmscale10x"):
            raw, meta = sample_sa_schedule(
                qd, b, s, 10.0, layers, seed=101, schedule=sch, num_reads=20, num_sweeps=50
            )
            self.assertEqual(raw.shape, (20, 4 * layers))
            self.assertEqual(meta["schedule"], sch)

        cat_raw, cat_meta = sample_categorical_sa(pred, layers, seed=101, num_reads=25, num_sweeps=50)
        self.assertEqual(cat_raw.shape, (25, 4 * layers))
        valid_ids = bits_to_ids(cat_raw, layers)
        self.assertTrue(all(v is not None for v in valid_ids))
        self.assertEqual(cat_meta["sampler"], "categorical_sa")


if __name__ == "__main__":
    unittest.main()
