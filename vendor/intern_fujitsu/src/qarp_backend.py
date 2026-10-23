"""Canonical OpenQARP state-vector circuits for the project QAOA solvers."""

from __future__ import annotations

from itertools import product
from typing import Sequence
from numbers import Integral

import numpy as np


def all_bitstrings_lsb(n_qubits: int) -> np.ndarray:
    values = np.arange(1 << n_qubits, dtype=np.uint64)[:, None]
    positions = np.arange(n_qubits, dtype=np.uint64)[None, :]
    return ((values >> positions) & 1).astype(np.float64)


def one_hot_penalty(bits: np.ndarray, block_sizes: Sequence[int]) -> np.ndarray:
    penalty = np.zeros(len(bits), dtype=np.float64)
    start = 0
    for size in block_sizes:
        penalty += (bits[:, start : start + size].sum(axis=1) - 1.0) ** 2
        start += size
    return penalty


def one_hot_patterns(block_sizes: Sequence[int]) -> np.ndarray:
    return np.asarray([np.concatenate(choice) for choice in product(*(np.eye(s) for s in block_sizes))], dtype=np.float64)


def qubo_to_ising_coefficients(
    qubo: np.ndarray, block_sizes: Sequence[int] | None = None, lambda_penalty: float = 0.0
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    matrix = np.asarray(qubo, dtype=np.float64).copy()
    if block_sizes and lambda_penalty:
        start = 0
        for size in block_sizes:
            for index in range(start, start + size):
                matrix[index, index] -= lambda_penalty
            for left in range(start, start + size):
                for right in range(left + 1, start + size):
                    matrix[left, right] += 2.0 * lambda_penalty
            start += size
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    for left in range(matrix.shape[0]):
        linear[left] = linear.get(left, 0.0) - float(matrix[left, left]) / 2.0
        for right in range(left + 1, matrix.shape[0]):
            coefficient = float(matrix[left, right])
            if coefficient:
                linear[left] -= coefficient / 4.0
                linear[right] = linear.get(right, 0.0) - coefficient / 4.0
                quadratic[(left, right)] = coefficient / 4.0
    return linear, quadratic


class OpenQARPStandardQAOA:
    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int], lambda_penalty: float = 0.0):
        self.qubo = np.asarray(qubo, dtype=np.float64)
        self.block_sizes = tuple(block_sizes)
        self.n_qubits = self.qubo.shape[0]
        self.linear, self.quadratic = qubo_to_ising_coefficients(self.qubo, self.block_sizes, lambda_penalty)

    def build_circuit(self, gammas: Sequence[float], betas: Sequence[float]):
        if len(gammas) != len(betas) or len(gammas) == 0:
            raise ValueError("gammas and betas must be non-empty sequences of equal length.")
        import qarpx as qx

        circuit = qx.SimpleBlock(self.n_qubits, "OpenQARP Standard QAOA")
        for qubit in range(self.n_qubits):
            circuit.h(qubit)
        for gamma, beta in zip(gammas, betas):
            for qubit, coefficient in self.linear.items():
                circuit.rz(qubit, 2.0 * coefficient * float(gamma))
            for (left, right), coefficient in self.quadratic.items():
                circuit.rzz(left, right, 2.0 * coefficient * float(gamma))
            for qubit in range(self.n_qubits):
                circuit.rx(qubit, 2.0 * float(beta))
        circuit.build()
        return circuit

    def statevector(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        import qarpx as qx
        circuit = self.build_circuit(gammas, betas)
        return qx.QarpSimulator().statevector(circuit.commands(), self.n_qubits)

    def probabilities(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        probabilities = np.abs(self.statevector(gammas, betas)) ** 2
        norm = float(probabilities.sum())
        if not np.isfinite(norm) or abs(norm - 1.0) >= 1e-10:
            raise RuntimeError("OpenQARP state norm drift exceeds tolerance")
        return probabilities / norm


class OpenQARPXYQAOA:
    """Product-W initial state and ordered edge-XY mixer (Ring by default).

    Each edge applies RXX(beta) then RYY(beta), i.e.
    exp[-i beta (XX+YY)/2]. Adjacent edge terms need not commute:
    this ordered product is not an exact exponential of their sum.
    Initial W states are supplied as an ideal statevector, not prep gates.
    """
    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int], *, mixer_topology: str = "ring"):
        self.qubo = np.asarray(qubo, dtype=np.float64)
        if self.qubo.ndim != 2 or self.qubo.shape[0] != self.qubo.shape[1] or not np.isfinite(self.qubo).all():
            raise ValueError("qubo must be a finite square matrix")
        self.block_sizes = tuple(block_sizes)
        if not self.block_sizes or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 1 for s in self.block_sizes):
            raise ValueError("block_sizes must contain positive integers")
        self.n_qubits = self.qubo.shape[0]
        if sum(self.block_sizes) != self.n_qubits:
            raise ValueError("sum(block_sizes) must match the QUBO dimension")
        if np.any(np.tril(self.qubo, -1) != 0):
            raise ValueError("XY backend expects an upper-triangular QUBO")
        if mixer_topology not in ("ring", "complete"):
            raise ValueError("mixer_topology must be ring or complete")
        self.mixer_topology = mixer_topology
        self.onehot_penalty_lambda = 0.0
        self.linear, self.quadratic = qubo_to_ising_coefficients(self.qubo)
        self.blocks: list[list[int]] = []
        self.mixer_edges: list[tuple[int, int]] = []
        start = 0
        for size in self.block_sizes:
            block = list(range(start, start + size))
            self.blocks.append(block)
            if mixer_topology == "complete":
                self.mixer_edges.extend((left, right) for k, left in enumerate(block) for right in block[k + 1:])
            elif size == 2:
                self.mixer_edges.append((block[0], block[1]))
            elif size > 2:
                self.mixer_edges.extend((block[k], block[(k + 1) % size]) for k in range(size))
            start += size
        self.feasible_patterns = one_hot_patterns(self.block_sizes)
        self.feasible_indices = np.asarray(self.feasible_patterns, dtype=np.uint64) @ (1 << np.arange(self.n_qubits, dtype=np.uint64))
        self.initial_state = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        self.initial_state[self.feasible_indices] = 1.0 / np.sqrt(len(self.feasible_patterns))

    def build_circuit(self, gammas: Sequence[float], betas: Sequence[float]):
        if len(gammas) != len(betas) or len(gammas) == 0:
            raise ValueError("gammas and betas must be non-empty sequences of equal length")
        if not np.isfinite(gammas).all() or not np.isfinite(betas).all():
            raise ValueError("angles must be finite")
        import qarpx as qx
        circuit = qx.SimpleBlock(self.n_qubits, "OpenQARP One-Hot XY-QAOA " + self.mixer_topology)
        for gamma, beta in zip(gammas, betas):
            for qubit, coefficient in self.linear.items():
                circuit.rz(qubit, 2.0 * coefficient * float(gamma))
            for (left, right), coefficient in self.quadratic.items():
                circuit.rzz(left, right, 2.0 * coefficient * float(gamma))
            for left, right in self.mixer_edges:
                circuit.rxx(left, right, float(beta))
                circuit.ryy(left, right, float(beta))
        circuit.build()
        return circuit

    def statevector(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        """Return the unrenormalized OpenQARP state for independent checks."""
        import qarpx as qx
        circuit = self.build_circuit(gammas, betas)
        return qx.QarpSimulator().statevector(circuit.commands(), self.n_qubits, initial_state=self.initial_state)

    def probabilities(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        probabilities = np.abs(self.statevector(gammas, betas)) ** 2
        norm = float(probabilities.sum())
        if not np.isfinite(norm) or abs(norm - 1.0) >= 1e-10:
            raise RuntimeError("OpenQARP state norm drift exceeds tolerance")
        return probabilities / norm


class OpenQARPCompactXYQAOA:
    """Exact simulation of the ordered XY circuit in an encoded One-Hot basis.

    The problem still has sum(block_sizes) One-Hot variables. The simulator
    uses ceil(log2(size)) register bits per group; unused encodings stay empty.
    Both diagonal cost and ordered edge rotations are executed by OpenQARP.
    This is a simulation optimization, not a resource estimate of the original
    circuit or the historical exponential-of-summed-mixer approximation.
    """
    _mixer_cache: dict = {}

    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int]):
        # Reuse the canonical input, edge order and basis definitions.
        original = OpenQARPXYQAOA(qubo, block_sizes)
        self.qubo = original.qubo
        self.block_sizes = original.block_sizes
        self.n_qubits = original.n_qubits
        self.mixer_edges = original.mixer_edges
        self.feasible_patterns = original.feasible_patterns
        self.onehot_penalty_lambda = 0.0
        self.register_widths = [max(1, (size - 1).bit_length()) for size in self.block_sizes]
        self.simulator_qubits = sum(self.register_widths)
        indices = np.zeros(len(self.feasible_patterns), dtype=np.uint64)
        physical_start = encoded_start = 0
        for size, width in zip(self.block_sizes, self.register_widths):
            choices = self.feasible_patterns[:, physical_start:physical_start+size].argmax(axis=1)
            indices += choices.astype(np.uint64) << encoded_start
            physical_start += size
            encoded_start += width
        self.encoded_indices = indices
        self.initial_state = np.zeros(1 << self.simulator_qubits, dtype=np.complex128)
        self.initial_state[indices] = 1.0 / np.sqrt(len(indices))
        self.energies = np.einsum('bi,ij,bj->b', self.feasible_patterns, self.qubo, self.feasible_patterns)

    def statevector(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        import qarpx as qx
        if len(gammas) != len(betas) or not len(gammas) or not np.isfinite(gammas).all() or not np.isfinite(betas).all():
            raise ValueError('finite, nonempty equal-length angles required')
        commands = []
        for gamma, beta in zip(gammas, betas):
            diagonal = np.ones(1 << self.simulator_qubits, dtype=np.complex128)
            diagonal[self.encoded_indices] = np.exp(-1j * gamma * self.energies)
            cost = qx.SimpleBlock(self.simulator_qubits, 'encoded exact XY cost')
            cost.diagonal_unitary(diagonal.tolist())
            commands.extend(cost.commands())
            physical_start = encoded_start = 0
            for size, width in zip(self.block_sizes, self.register_widths):
                key = (size, float(beta))
                if key not in self._mixer_cache:
                    unitary = np.eye(1 << width, dtype=np.complex128)
                    for left, right in self.mixer_edges:
                        if not physical_start <= left < physical_start + size:
                            continue
                        a, b = left-physical_start, right-physical_start
                        rows = unitary[[a, b]].copy()
                        unitary[a] = np.cos(beta)*rows[0]-1j*np.sin(beta)*rows[1]
                        unitary[b] = np.cos(beta)*rows[1]-1j*np.sin(beta)*rows[0]
                    mixer = qx.SimpleBlock(width, 'encoded ordered XY edges')
                    mixer.unitary_synthesis(np.asfortranarray(unitary))
                    self._mixer_cache[key] = list(mixer.commands())
                remap = list(range(encoded_start, encoded_start+width))
                commands.extend(c.remap_qubits(remap) for c in self._mixer_cache[key])
                physical_start += size
                encoded_start += width
        circuit = qx.SimpleBlock(self.simulator_qubits, 'OpenQARP exact compact XY')
        circuit.set_commands(commands)
        circuit.build()
        return qx.QarpSimulator().statevector(circuit.commands(), self.simulator_qubits, initial_state=self.initial_state)

    def feasible_probabilities(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        state = self.statevector(gammas, betas)
        probabilities = np.abs(state[self.encoded_indices])**2
        if abs(1.0 - np.vdot(state, state).real) >= 1e-10 or abs(1.0-probabilities.sum()) >= 1e-10:
            raise RuntimeError('compact XY state norm or One-Hot mass drift')
        return probabilities
