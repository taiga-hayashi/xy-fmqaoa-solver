"""
Problem definition for Multiple One-Hot Constrained BBO.
Each group has 3 choices, so N = 3 * G.
Bit ordering: group g occupies bits [3*g, 3*g+1, 3*g+2].
Constraint: sum_{j=0}^2 x_{3*g + j} = 1 for all g in {0, ..., G-1}.
"""

from __future__ import annotations
import itertools
from typing import List, Tuple, Sequence, Optional
import numpy as np


def is_feasible_one_hot(x: Sequence[int | float], G: int) -> bool:
    """Return True if x satisfies the 3-choice One-Hot constraint for all G groups."""
    arr = np.asarray(x, dtype=np.int8)
    if len(arr) != 3 * G:
        return False
    for g in range(G):
        if arr[3 * g : 3 * g + 3].sum() != 1:
            return False
    return True


def one_hot_penalty(x: np.ndarray, G: int) -> float:
    """
    Calculate the total one-hot penalty:
    P(x) = sum_{g=0}^{G-1} (sum_{j=0}^2 x_{3*g + j} - 1)^2
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        penalty = 0.0
        for g in range(G):
            penalty += (arr[3 * g : 3 * g + 3].sum() - 1.0) ** 2
        return float(penalty)
    else:
        # Batch (M, N)
        penalties = np.zeros(len(arr), dtype=np.float64)
        for g in range(G):
            penalties += (arr[:, 3 * g : 3 * g + 3].sum(axis=1) - 1.0) ** 2
        return penalties


def generate_all_one_hot_states(G: int) -> np.ndarray:
    """
    Generate all 3^G feasible One-Hot bitstrings.
    Returns:
        np.ndarray of shape (3^G, 3*G) with int8 entries.
    """
    eye3 = np.eye(3, dtype=np.int8)
    combos = [c for c in itertools.product(eye3, repeat=G)]
    basis = np.array([np.concatenate(c) for c in combos], dtype=np.int8)
    return basis


def create_materials_qubo_bb(N: int, seed: int = 123) -> np.ndarray:
    """
    Scalable BB-1: 多元触媒・機能性材料設計モデル (N = 3 * G)
    G = N // 3 ブロック。各ブロック3選択肢。
    特定サイト間 (隣接ペア (2k, 2k+1) 等) に協同触媒効果・強シナジー相互作用 (-6.0 〜 2.0)。
    その他のブロック間は通常相互作用 (-1.0 〜 1.0)。
    対角項 (化学ポテンシャル) は -2.0 〜 1.0。
    """
    rng = np.random.RandomState(seed)
    G = N // 3
    Q = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                Q[i, i] = rng.uniform(-2.0, 1.0)
            else:
                b_i = i // 3
                b_j = j // 3
                if b_i != b_j:
                    is_synergy = (min(b_i, b_j) % 2 == 0) and (abs(b_i - b_j) == 1)
                    if is_synergy:
                        Q[i, j] = rng.uniform(-6.0, 2.0)
                    else:
                        Q[i, j] = rng.uniform(-1.0, 1.0)
    return Q


def create_random_qubo_bb(N: int, seed: int = 42) -> np.ndarray:
    """
    Scalable BB-2: 全結合ランダム QUBO モデル (N = 3 * G)
    対角項: -1.5 〜 1.5
    非対角項: -1.0 〜 1.0
    """
    rng = np.random.RandomState(seed)
    Q = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                Q[i, i] = rng.uniform(-1.5, 1.5)
            else:
                Q[i, j] = rng.uniform(-1.0, 1.0)
    return Q


def get_bb_qubo(problem_type: str, N: int, seed: int = 42) -> np.ndarray:
    """
    Retrieve the upper-triangular QUBO matrix for the specified BB problem.
    problem_type: 'bb1' or 'materials' for BB-1; 'bb2' or 'random' for BB-2.
    """
    pt = problem_type.lower()
    if pt in ("bb1", "materials", "catalyst"):
        return create_materials_qubo_bb(N, seed=seed)
    elif pt in ("bb2", "random"):
        return create_random_qubo_bb(N, seed=seed)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}. Must be 'bb1' or 'bb2'.")


def evaluate_bb(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluate f(x) = x^T Q x."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return float(arr @ Q @ arr)
    return np.einsum('ni,ij,nj->n', arr, Q, arr)


def get_exact_ground_truth(Q: np.ndarray, G: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate all 3^G feasible states to find exact f_opt (min) and f_worst (max).
    Returns:
        (f_opt, f_worst, best_x, worst_x)
    """
    feasible_states = generate_all_one_hot_states(G)
    values = evaluate_bb(Q, feasible_states)
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))
    return float(values[min_idx]), float(values[max_idx]), feasible_states[min_idx], feasible_states[max_idx]


def generate_initial_dataset(
    Q: np.ndarray, G: int, num_samples: int = 20, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate initial feasible One-Hot training samples for FM.
    """
    rng = np.random.default_rng(seed)
    N = 3 * G
    # Generate random feasible samples
    choices = rng.integers(0, 3, size=(num_samples, G))
    X = np.zeros((num_samples, N), dtype=np.float32)
    for i in range(num_samples):
        for g in range(G):
            X[i, 3 * g + choices[i, g]] = 1.0
    y = evaluate_bb(Q, X).astype(np.float32)
    return X, y
