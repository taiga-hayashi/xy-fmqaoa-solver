"""Compact XY-QAOA whose cost layer is split into group and group-pair diagonals.

Same circuit as qarp_backend.OpenQARPCompactXYQAOA (encoded One-Hot basis,
product W start, ordered Ring XY edges), but the cost phase exp(-i*gamma*E) is
applied as one small diagonal per group and per group pair instead of one
2^n-entry diagonal. For a One-Hot state with choices c_g, an upper-triangular
QUBO gives E = sum_g Q[i_g, i_g] + sum_{g<h} Q[i_g, i_h] exactly, so the
product of these diagonals equals the original diagonal on every feasible
encoding (unused encodings get phase 1, as before). qarpx synthesizes a k-qubit
diagonal into 2^(k+1)-2 gates, so the full diagonal costs O(4^n) while this
costs O(G^2 * 2^n). Every gate is still executed by OpenQARP.

Requires qarp_backend (Intern_Fujitsu/src) and qarpx on sys.path.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qarp_backend import OpenQARPCompactXYQAOA


class OpenQARPPairXYQAOA(OpenQARPCompactXYQAOA):
    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int]):
        super().__init__(qubo, block_sizes)
        self.physical_starts = np.cumsum([0, *self.block_sizes[:-1]]).tolist()
        self.encoded_starts = np.cumsum([0, *self.register_widths[:-1]]).tolist()
        # Per-group linear energies and per-pair coupling tables over local encodings.
        self.single = []
        for g, (size, width) in enumerate(zip(self.block_sizes, self.register_widths)):
            e = np.zeros(1 << width)
            idx = self.physical_starts[g] + np.arange(size)
            e[:size] = self.qubo[idx, idx]
            self.single.append(e)
        self.pairs = []
        for g in range(len(self.block_sizes)):
            for h in range(g + 1, len(self.block_sizes)):
                wg, wh = self.register_widths[g], self.register_widths[h]
                table = np.zeros((1 << wh, 1 << wg))  # [d, c] -> local index c + (d << wg)
                ig = self.physical_starts[g] + np.arange(self.block_sizes[g])
                ih = self.physical_starts[h] + np.arange(self.block_sizes[h])
                table[:len(ih), :len(ig)] = self.qubo[np.ix_(ig, ih)].T
                self.pairs.append((g, h, table.ravel()))
        self._check_decomposition()

    def _check_decomposition(self):
        # Reassemble E on every feasible encoding from the parts; must equal the direct energies.
        widths = np.array(self.register_widths)
        choices = [(self.encoded_indices >> np.uint64(s)) & np.uint64((1 << w) - 1)
                   for s, w in zip(self.encoded_starts, widths)]
        total = sum(self.single[g][c.astype(np.int64)] for g, c in enumerate(choices))
        for g, h, table in self.pairs:
            total = total + table[(choices[g] + (choices[h] << np.uint64(widths[g]))).astype(np.int64)]
        np.testing.assert_allclose(total, self.energies, atol=1e-12 * max(1., float(np.max(abs(self.energies)))), rtol=0)

    def _diagonal(self, qx, phases, qubits):
        block = qx.SimpleBlock(len(qubits), 'pair cost diagonal')
        block.diagonal_unitary(np.exp(-1j * phases).tolist())
        return [c.remap_qubits(list(qubits)) for c in block.commands()]

    def statevector(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        import qarpx as qx
        if len(gammas) != len(betas) or not len(gammas) or not np.isfinite(gammas).all() or not np.isfinite(betas).all():
            raise ValueError('finite, nonempty equal-length angles required')
        valid = [np.arange(1 << w) < s for s, w in zip(self.block_sizes, self.register_widths)]
        commands = []
        for gamma, beta in zip(gammas, betas):
            for g, w in enumerate(self.register_widths):
                qubits = range(self.encoded_starts[g], self.encoded_starts[g] + w)
                commands.extend(self._diagonal(qx, gamma * np.where(valid[g], self.single[g], 0.), qubits))
            for g, h, table in self.pairs:
                wg, wh = self.register_widths[g], self.register_widths[h]
                mask = (valid[h][:, None] & valid[g][None, :]).ravel()
                qubits = [*range(self.encoded_starts[g], self.encoded_starts[g] + wg),
                          *range(self.encoded_starts[h], self.encoded_starts[h] + wh)]
                commands.extend(self._diagonal(qx, gamma * np.where(mask, table, 0.), qubits))
            for g, (size, width) in enumerate(zip(self.block_sizes, self.register_widths)):
                key = (size, float(beta))
                if key not in self._mixer_cache:
                    physical_start = self.physical_starts[g]
                    unitary = np.eye(1 << width, dtype=np.complex128)
                    for left, right in self.mixer_edges:
                        if not physical_start <= left < physical_start + size:
                            continue
                        a, b = left - physical_start, right - physical_start
                        rows = unitary[[a, b]].copy()
                        unitary[a] = np.cos(beta) * rows[0] - 1j * np.sin(beta) * rows[1]
                        unitary[b] = np.cos(beta) * rows[1] - 1j * np.sin(beta) * rows[0]
                    mixer = qx.SimpleBlock(width, 'encoded ordered XY edges')
                    mixer.unitary_synthesis(np.asfortranarray(unitary))
                    self._mixer_cache[key] = list(mixer.commands())
                remap = list(range(self.encoded_starts[g], self.encoded_starts[g] + width))
                commands.extend(c.remap_qubits(remap) for c in self._mixer_cache[key])
        circuit = qx.SimpleBlock(self.simulator_qubits, 'OpenQARP pair-decomposed compact XY')
        circuit.set_commands(commands)
        circuit.build()
        return qx.QarpSimulator().statevector(circuit.commands(), self.simulator_qubits, initial_state=self.initial_state)
