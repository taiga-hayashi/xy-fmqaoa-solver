"""
Four solver implementations for BBO loop:
1. Adaptive-FMQA (Classical SA with adaptive penalty alpha_t)
2. LargePenalty-FMQA (Classical SA with large fixed penalty alpha=100)
3. Penalty-FMQAOA (Standard X-Mixer QAOA with penalty in Cost Hamiltonian)
4. XY-FMQAOA (W-state init + Ring XY-Mixer, zero penalty lambda=0)
"""

from __future__ import annotations
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .problem import is_feasible_one_hot, one_hot_penalty, generate_all_one_hot_states
from .penalty import build_one_hot_penalty_qubo, apply_penalty_to_qubo


def _resolve_statevector_dtype(N: int, requested: str | np.dtype = "auto") -> np.dtype:
    """Choose an explicitly recorded statevector dtype without changing the algorithm."""
    if requested == "auto":
        # Preserve the former complex128 numerics where they are inexpensive. At N=27,
        # complex64 halves the two live statevectors from 4 GiB to 2 GiB in total.
        return np.dtype(np.complex64 if N >= 27 else np.complex128)
    dtype = np.dtype(requested)
    if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("statevector_dtype must be 'auto', complex64, or complex128")
    return dtype


def estimate_penalty_qaoa_memory(
    N: int,
    statevector_dtype: str | np.dtype = "auto",
    workspace_mb: int = 64,
) -> Dict[str, Any]:
    """Conservative resident-array estimate for the memory-aware p=1 simulator."""
    complex_dtype = _resolve_statevector_dtype(N, statevector_dtype)
    real_dtype = np.dtype(np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64)
    dim = 1 << N
    energy_bytes = dim * real_dtype.itemsize
    statevector_bytes = dim * complex_dtype.itemsize
    workspace_bytes = int(workspace_mb) * 1024**2
    estimated_peak = energy_bytes + 2 * statevector_bytes + workspace_bytes
    return {
        "N": N,
        "dimension": dim,
        "statevector_dtype": complex_dtype.name,
        "energy_dtype": real_dtype.name,
        "energy_bytes": energy_bytes,
        "statevector_bytes_each": statevector_bytes,
        "live_statevector_count": 2,
        "workspace_bytes": workspace_bytes,
        "estimated_peak_bytes": estimated_peak,
        "estimated_peak_gib": estimated_peak / 1024**3,
    }


def _build_qubo_energies(
    Q: np.ndarray,
    offset: float,
    real_dtype: np.dtype,
) -> np.ndarray:
    """Build x.T Q x + offset in LSB order without a (2**N, N) bit matrix."""
    N = Q.shape[0]
    if Q.shape != (N, N):
        raise ValueError("Q must be square")
    energies = np.empty(1 << N, dtype=real_dtype)
    energies[0] = offset
    populated = 1
    for j in range(N):
        upper = energies[populated : 2 * populated]
        np.add(energies[:populated], Q[j, j], out=upper, casting="unsafe")
        for i in range(j):
            coefficient = float(Q[i, j] + Q[j, i])
            if coefficient == 0.0:
                continue
            step = 1 << i
            upper.reshape(-1, 2 * step)[:, step:] += coefficient
        populated *= 2
    return energies


def _prepare_cost_state(
    destination: np.ndarray,
    energies: np.ndarray,
    gamma: float,
    chunk_elements: int,
) -> None:
    amplitude = 1.0 / np.sqrt(len(energies))
    complex_dtype = destination.dtype
    for start in range(0, len(energies), chunk_elements):
        stop = min(start + chunk_elements, len(energies))
        phase_arg = np.asarray(-gamma * energies[start:stop], dtype=energies.dtype)
        destination[start:stop] = (
            np.cos(phase_arg) + 1j * np.sin(phase_arg)
        ).astype(complex_dtype, copy=False)
        destination[start:stop] *= amplitude


def _apply_x_mixer_inplace(
    state: np.ndarray,
    beta: float,
    N: int,
    workspace_bytes: int,
) -> None:
    """Apply product RX(2 beta) using bounded temporary storage."""
    c = float(np.cos(beta))
    s = np.asarray(-1j * np.sin(beta), dtype=state.dtype).item()
    # Assignment expressions create temporaries too, so reserve roughly one quarter
    # of the requested workspace for the explicit old-|0> copy.
    chunk_elements = max(1, workspace_bytes // (4 * state.dtype.itemsize))
    for q in range(N):
        step = 1 << q
        view = state.reshape(-1, 2, step)
        if step <= chunk_elements:
            rows_per_chunk = max(1, chunk_elements // step)
            for row_start in range(0, view.shape[0], rows_per_chunk):
                row_stop = min(row_start + rows_per_chunk, view.shape[0])
                left = view[row_start:row_stop, 0, :]
                right = view[row_start:row_stop, 1, :]
                old_left = left.copy()
                left[...] = c * old_left + s * right
                right[...] = s * old_left + c * right
        else:
            for col_start in range(0, step, chunk_elements):
                col_stop = min(col_start + chunk_elements, step)
                left = view[:, 0, col_start:col_stop]
                right = view[:, 1, col_start:col_stop]
                old_left = left.copy()
                left[...] = c * old_left + s * right
                right[...] = s * old_left + c * right


def _state_energy_expectation(
    state: np.ndarray,
    energies: np.ndarray,
    chunk_elements: int,
) -> Tuple[float, float]:
    weighted_sum = 0.0
    norm = 0.0
    for start in range(0, len(state), chunk_elements):
        stop = min(start + chunk_elements, len(state))
        probs = np.abs(state[start:stop]) ** 2
        norm += float(np.sum(probs, dtype=np.float64))
        weighted_sum += float(
            np.dot(probs.astype(np.float64, copy=False), energies[start:stop].astype(np.float64, copy=False))
        )
    if norm <= 0.0:
        raise RuntimeError("statevector norm became non-positive")
    return weighted_sum / norm, norm


def _sample_statevector_streaming(
    state: np.ndarray,
    num_samples: int,
    seed: int,
    chunk_elements: int,
) -> np.ndarray:
    """Sample basis indices without allocating a full probability/CDF array."""
    norm = 0.0
    for start in range(0, len(state), chunk_elements):
        stop = min(start + chunk_elements, len(state))
        norm += float(np.sum(np.abs(state[start:stop]) ** 2, dtype=np.float64))

    rng = np.random.default_rng(seed)
    unsorted_targets = rng.random(num_samples) * norm
    target_order = np.argsort(unsorted_targets)
    targets = unsorted_targets[target_order]
    sampled_sorted = np.empty(num_samples, dtype=np.uint64)
    target_pos = 0
    cumulative = 0.0
    for start in range(0, len(state), chunk_elements):
        if target_pos == num_samples:
            break
        stop = min(start + chunk_elements, len(state))
        probs = np.abs(state[start:stop]) ** 2
        cdf = np.cumsum(probs, dtype=np.float64)
        chunk_total = float(cdf[-1]) if len(cdf) else 0.0
        upper = cumulative + chunk_total
        next_pos = int(np.searchsorted(targets, upper, side="right"))
        for k in range(target_pos, next_pos):
            local = int(np.searchsorted(cdf, targets[k] - cumulative, side="left"))
            sampled_sorted[k] = start + min(local, len(cdf) - 1)
        target_pos = next_pos
        cumulative = upper
    if target_pos < num_samples:
        sampled_sorted[target_pos:] = len(state) - 1
    sampled = np.empty(num_samples, dtype=np.uint64)
    sampled[target_order] = sampled_sorted
    return sampled


def _index_to_bits(index: int, N: int) -> np.ndarray:
    return ((np.uint64(index) >> np.arange(N, dtype=np.uint64)) & 1).astype(np.int8)


def solve_classical_sa(
    Q: np.ndarray,
    offset: float,
    G: int,
    lambda_val: float,
    num_reads: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Classical Simulated Annealing for penalized QUBO.
    """
    t0 = time.perf_counter()
    N = 3 * G
    Q_pen, offset_pen = build_one_hot_penalty_qubo(G)
    Q_tot = Q + lambda_val * Q_pen
    offset_tot = offset + lambda_val * offset_pen

    rng = np.random.default_rng(seed)
    samples = []
    feasible_count = 0

    # Fast Vectorized / Multiple Markov-chain SA simulation
    beta_schedule = np.geomspace(0.1, 10.0, num=30)
    for _ in range(num_reads):
        # Random initial bitstring
        x = rng.integers(0, 2, size=N).astype(np.float64)
        current_energy = float(x @ Q_tot @ x + offset_tot)

        for beta in beta_schedule:
            # Single bit-flip proposal
            flip_idx = rng.integers(0, N)
            x_cand = x.copy()
            x_cand[flip_idx] = 1.0 - x_cand[flip_idx]
            cand_energy = float(x_cand @ Q_tot @ x_cand + offset_tot)
            delta_e = cand_energy - current_energy

            if delta_e <= 0.0 or rng.random() < np.exp(-beta * delta_e):
                x = x_cand
                current_energy = cand_energy

        feas = is_feasible_one_hot(x, G)
        if feas:
            feasible_count += 1
        pure_fval = float(x @ Q @ x + offset)
        samples.append({
            "x": tuple(x.astype(int)),
            "fval": pure_fval,
            "penalized_val": current_energy,
            "feasible": feas,
            "sample_source": "classical_simulated_annealing",
        })

    elapsed = time.perf_counter() - t0
    raw_feas_rate = feasible_count / num_reads

    # Select best feasible if exists, else best overall
    feasible_samples = [s for s in samples if s["feasible"]]
    if feasible_samples:
        best_sample = min(feasible_samples, key=lambda s: s["fval"])
    else:
        best_sample = min(samples, key=lambda s: s["penalized_val"])

    return {
        "solver": "FMQA (SA)",
        "backend": "classical_simulated_annealing",
        "fallback_used": False,
        "statevector_dtype": None,
        "samples": samples,
        "raw_feasible_rate": raw_feas_rate,
        "best_sample": best_sample,
        "runtime": elapsed
    }


def build_ring_xy_mixer_subspace(G: int) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the Ring XY-Mixer Hamiltonian matrix within the 3^G One-Hot subspace.
    For each 3-choice group, ring edges are (0,1), (1,2), (2,0).
    Returns:
        H_M: (3^G, 3^G) sparse CSR matrix
        basis: (3^G, 3*G) array of feasible basis bitstrings
    """
    basis = generate_all_one_hot_states(G)
    dim_sub = len(basis)

    # Map each tuple of choice indices to basis index
    basis_tuples = []
    tuple_to_idx = {}
    for i, row in enumerate(basis):
        tup = tuple(int(np.where(row[3 * g : 3 * g + 3] == 1)[0][0]) for g in range(G))
        basis_tuples.append(tup)
        tuple_to_idx[tup] = i

    rows, cols, data = [], [], []
    for i, tup in enumerate(basis_tuples):
        tup_list = list(tup)
        for g in range(G):
            curr = tup_list[g]
            # Ring neighbors: +1, -1 modulo 3
            neighbors = [(curr + 1) % 3, (curr + 2) % 3]
            for nxt in neighbors:
                new_tup = list(tup_list)
                new_tup[g] = nxt
                j = tuple_to_idx[tuple(new_tup)]
                rows.append(i)
                cols.append(j)
                data.append(0.5)

    H_M = sp.csr_matrix((data, (rows, cols)), shape=(dim_sub, dim_sub), dtype=np.float64)
    return H_M, basis


def solve_xy_fmqa(
    Q: np.ndarray,
    offset: float,
    G: int,
    num_samples: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    XY-Mixer QAOA (p=1) within the strict One-Hot subspace.
    No penalty applied (lambda=0). Feasibility rate is strictly 1.0.
    """
    t0 = time.perf_counter()
    H_M, basis = build_ring_xy_mixer_subspace(G)
    dim_sub = len(basis)

    # 1. Cost energies on basis
    energies = np.einsum('ni,ij,nj->n', basis.astype(np.float64), Q, basis.astype(np.float64)) + offset

    # 2. Initial state: uniform superposition over all 3^G feasible states (Product of W-states)
    psi_0 = np.full(dim_sub, 1.0 / np.sqrt(dim_sub), dtype=np.complex128)

    # 3. Deterministic p=1 grid search (7x7 = 49 candidates)
    side = 7
    gammas = np.linspace(0.05, 0.8, side)
    betas = np.linspace(0.05, 0.8, side)

    best_exp = float("inf")
    best_probs = None
    best_params = (0.0, 0.0)

    for g in gammas:
        phase = np.exp(-1j * g * energies)
        psi_phase = psi_0 * phase
        for b in betas:
            psi_out = expm_multiply(-1j * b * H_M, psi_phase)
            probs = np.abs(psi_out) ** 2
            probs /= np.sum(probs)
            exp_val = float(np.sum(probs * energies))
            if exp_val < best_exp:
                best_exp = exp_val
                best_probs = probs
                best_params = (g, b)

    assert best_probs is not None
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(dim_sub, size=num_samples, p=best_probs)

    samples = []
    for idx in sampled_indices:
        samples.append({
            "x": tuple(basis[idx].astype(int)),
            "fval": float(energies[idx]),
            "penalized_val": float(energies[idx]),
            "feasible": True,
            "sample_source": "ideal_one_hot_subspace_statevector_measurement",
        })

    elapsed = time.perf_counter() - t0
    best_sample = min(samples, key=lambda s: s["fval"])

    return {
        "solver": "XY-FMQAOA",
        "backend": "ideal_one_hot_subspace_statevector",
        "fallback_used": False,
        "statevector_dtype": psi_0.dtype.name,
        "statevector_dimension": dim_sub,
        "samples": samples,
        "raw_feasible_rate": 1.0,
        "best_sample": best_sample,
        "runtime": elapsed,
        "best_params": best_params,
        "probabilities": best_probs,
        "basis": basis
    }


def solve_xy_fmqa_p_layers(
    Q: np.ndarray,
    offset: float,
    G: int,
    p: int = 1,
    num_samples: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    import time
    from scipy.sparse.linalg import expm_multiply
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    H_M, basis = build_ring_xy_mixer_subspace(G)
    dim_sub = len(basis)

    energies = np.einsum('ni,ij,nj->n', basis.astype(np.float64), Q, basis.astype(np.float64)) + offset
    psi_0 = np.full(dim_sub, 1.0 / np.sqrt(dim_sub), dtype=np.complex128)

    def qaoa_energy(params):
        gammas = params[:p]
        betas = params[p:]
        psi = psi_0.copy()
        for g, b in zip(gammas, betas):
            phase = np.exp(-1j * g * energies)
            psi = psi * phase
            psi = expm_multiply(-1j * b * H_M, psi)
        probs = np.abs(psi) ** 2
        probs /= np.sum(probs)
        return float(np.sum(probs * energies))

    rng = np.random.default_rng(seed)
    best_exp = float("inf")
    best_params = None
    
    num_starts = max(3, 10 - p)
    for _ in range(num_starts):
        init_params = rng.uniform(0.01, 0.8, size=2 * p)
        res = minimize(qaoa_energy, init_params, method='COBYLA', options={'maxiter': 200})
        if res.fun < best_exp:
            best_exp = res.fun
            best_params = res.x

    assert best_params is not None
    gammas = best_params[:p]
    betas = best_params[p:]
    psi = psi_0.copy()
    for g, b in zip(gammas, betas):
        phase = np.exp(-1j * g * energies)
        psi = psi * phase
        psi = expm_multiply(-1j * b * H_M, psi)
    
    best_probs = np.abs(psi) ** 2
    best_probs /= np.sum(best_probs)

    sampled_indices = rng.choice(dim_sub, size=num_samples, p=best_probs)
    samples = []
    for idx in sampled_indices:
        samples.append({
            "x": tuple(basis[idx].astype(int)),
            "fval": float(energies[idx]),
            "penalized_val": float(energies[idx]),
            "feasible": True,
            "sample_source": f"xy_qaoa_p{p}",
        })

    elapsed = time.perf_counter() - t0
    best_sample = min(samples, key=lambda s: s["fval"]) if samples else None

    return {
        "solver": f"XY-FMQAOA (p={p})",
        "backend": f"xy_qaoa_subspace_p{p}_cobyla",
        "fallback_used": False,
        "statevector_dtype": psi_0.dtype.name,
        "statevector_dimension": dim_sub,
        "samples": samples,
        "raw_feasible_rate": 1.0,
        "best_sample": best_sample,
        "runtime": elapsed,
        "best_params": best_params.tolist(),
        "probabilities": best_probs,
        "basis": basis
    }


def solve_penalty_qaoa(
    Q: np.ndarray,
    offset: float,
    G: int,
    lambda_val: float,
    num_samples: int = 500,
    seed: int = 42,
    statevector_dtype: str | np.dtype = "auto",
    workspace_mb: int = 64,
    grid_side: int = 7,
) -> Dict[str, Any]:
    """
    Standard X-Mixer QAOA (p=1) with One-Hot penalty in Cost Hamiltonian.

    Every N follows the same ideal full-statevector algorithm.  In particular,
    there is no classical-SA fallback.  The implementation avoids materializing
    the (2**N, N) bit matrix, bounds mixer workspace, and streams measurement.
    """
    t0 = time.perf_counter()
    N = 3 * G
    Q_tot, offset_tot = apply_penalty_to_qubo(Q, offset, G, lambda_val)
    if grid_side < 1:
        raise ValueError("grid_side must be positive")

    memory = estimate_penalty_qaoa_memory(N, statevector_dtype, workspace_mb)
    complex_dtype = np.dtype(memory["statevector_dtype"])
    real_dtype = np.dtype(memory["energy_dtype"])
    dim_full = memory["dimension"]
    workspace_bytes = memory["workspace_bytes"]
    chunk_elements = max(1, workspace_bytes // (4 * complex_dtype.itemsize))

    phase_start = time.perf_counter()
    penalized_energies = _build_qubo_energies(Q_tot, offset_tot, real_dtype)
    energy_build_seconds = time.perf_counter() - phase_start

    gammas = np.linspace(0.05, 0.8, grid_side)
    betas = np.linspace(0.05, 0.8, grid_side)
    cost_state = np.empty(dim_full, dtype=complex_dtype)
    work_state = np.empty(dim_full, dtype=complex_dtype)
    best_exp = float("inf")
    best_params = (0.0, 0.0)
    best_norm = 0.0

    search_start = time.perf_counter()
    for gamma in gammas:
        _prepare_cost_state(cost_state, penalized_energies, float(gamma), chunk_elements)
        for beta in betas:
            np.copyto(work_state, cost_state)
            _apply_x_mixer_inplace(work_state, float(beta), N, workspace_bytes)
            exp_val, norm = _state_energy_expectation(work_state, penalized_energies, chunk_elements)
            if exp_val < best_exp:
                best_exp = exp_val
                best_params = (float(gamma), float(beta))
                best_norm = norm
    grid_search_seconds = time.perf_counter() - search_start

    # Recreate only the winning state. No full probability or CDF vector is retained.
    sampling_start = time.perf_counter()
    _prepare_cost_state(cost_state, penalized_energies, best_params[0], chunk_elements)
    np.copyto(work_state, cost_state)
    _apply_x_mixer_inplace(work_state, best_params[1], N, workspace_bytes)
    sampled_indices = _sample_statevector_streaming(work_state, num_samples, seed, chunk_elements)

    samples = []
    feasible_count = 0
    for idx in sampled_indices:
        b_vec = _index_to_bits(int(idx), N)
        penalty = float(one_hot_penalty(b_vec, G))
        feasible = penalty == 0.0
        feasible_count += int(feasible)
        b_float = b_vec.astype(np.float64)
        pure_fval = float(b_float @ Q @ b_float + offset)
        samples.append({
            "x": tuple(int(v) for v in b_vec),
            "fval": pure_fval,
            "penalized_val": pure_fval + lambda_val * penalty,
            "feasible": feasible,
            "sample_source": "ideal_full_statevector_measurement",
        })
    raw_feas_rate = feasible_count / num_samples
    sampling_seconds = time.perf_counter() - sampling_start

    elapsed = time.perf_counter() - t0
    feasible_samples = [s for s in samples if s["feasible"]]
    if feasible_samples:
        best_sample = min(feasible_samples, key=lambda s: s["fval"])
    else:
        best_sample = min(samples, key=lambda s: s["penalized_val"])

    return {
        "solver": "Penalty-FMQAOA",
        "backend": "ideal_full_statevector",
        "fallback_used": False,
        "statevector_dtype": complex_dtype.name,
        "energy_dtype": real_dtype.name,
        "statevector_dimension": dim_full,
        "memory_estimate": memory,
        "samples": samples,
        "raw_feasible_rate": raw_feas_rate,
        "best_sample": best_sample,
        "runtime": elapsed,
        "best_params": best_params,
        "best_expectation": best_exp,
        "best_state_norm": best_norm,
        "grid_side": grid_side,
        "timings": {
            "energy_build_seconds": energy_build_seconds,
            "grid_search_seconds": grid_search_seconds,
            "sampling_seconds": sampling_seconds,
        },
    }
