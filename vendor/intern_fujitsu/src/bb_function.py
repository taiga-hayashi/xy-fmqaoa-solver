import numpy as np
import torch
import itertools

def create_random_qubo_bb(d: int, seed: int = 42) -> np.ndarray:
    """d次元のランダムQUBO行列（上三角）を生成する"""
    np.random.seed(seed)
    Q = np.random.uniform(-1, 1, size=(d, d))
    # 上三角行列にする
    Q = np.triu(Q)
    return Q

def evaluate_bb(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """バイナリ入力 x に対してBB関数 (x^T Q x) の値を計算する"""
    # x: (N, d), Q: (d, d)
    return np.einsum('ni,ij,nj->n', x, Q, x)

def generate_dataset(Q: np.ndarray, num_samples: int, d: int, seed: int = 42):
    """ランダムなバイナリ入力を生成し、BB関数でラベル付けしたデータセットを作成"""
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(num_samples, d)).astype(np.float32)
    y = evaluate_bb(X, Q).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

def get_exact_minimum_bb(Q: np.ndarray, d: int):
    """BB関数の真の最小値とその入力を全探索で求める（非制約）"""
    all_patterns = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.float32)
    y_all = evaluate_bb(all_patterns, Q)
    min_idx = np.argmin(y_all)
    return all_patterns[min_idx], y_all[min_idx]

def get_feasible_patterns(block_sizes: list) -> np.ndarray:
    """各ブロックでちょうど1つのビットが1である実行可能パターンをすべて生成する"""
    block_patterns = []
    for sz in block_sizes:
        # sz次元の単位行列の各行がOne-Hotパターン
        eye = np.eye(sz, dtype=np.float32)
        block_patterns.append(eye)
    
    # 全ブロックの直積
    feasible = []
    for combo in itertools.product(*block_patterns):
        feasible.append(np.concatenate(combo))
    return np.array(feasible, dtype=np.float32)

def is_feasible_one_hot(x: np.ndarray, block_sizes: list) -> bool:
    """入力ビット列 x が全ブロックでOne-Hot制約を満たしているか判定する"""
    x = np.array(x).flatten()
    idx = 0
    for sz in block_sizes:
        if np.sum(x[idx:idx+sz]) != 1:
            return False
        idx += sz
    return True

def get_exact_minimum_categorical_bb(Q: np.ndarray, block_sizes: list):
    """One-Hot制約を満たす実行可能解の中でのBB関数の真の最小値と解を求める"""
    feasible_patterns = get_feasible_patterns(block_sizes)
    y_feas = evaluate_bb(feasible_patterns, Q)
    min_idx = np.argmin(y_feas)
    return feasible_patterns[min_idx], y_feas[min_idx]

def generate_categorical_dataset(Q: np.ndarray, num_samples: int, block_sizes: list, seed: int = 42):
    """One-Hot制約を満たす解のみからランダムにサンプリングしてデータセットを生成"""
    np.random.seed(seed)
    feasible_patterns = get_feasible_patterns(block_sizes)
    indices = np.random.choice(len(feasible_patterns), size=num_samples, replace=True)
    X = feasible_patterns[indices]
    y = evaluate_bb(X, Q).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

