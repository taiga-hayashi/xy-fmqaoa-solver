import dimod
import neal
import numpy as np
from typing import Dict, Tuple, List, Any
from bb_function import is_feasible_one_hot

def build_penalty_bqm(qubo_dict: Dict[Tuple[int, int], float], block_sizes: List[int], lambda_penalty: float = 5.0, offset: float = 0.0) -> dimod.BinaryQuadraticModel:
    """
    QUBO辞書にOne-Hot制約ペナルティ項を追加したBinaryQuadraticModelを構築する。
    """
    total_vars = sum(block_sizes)
    
    # ベースのBQM
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    bqm.offset = offset
    
    # 変数の追加
    for i in range(total_vars):
        bqm.add_variable(i)
        
    for (i, j), w in qubo_dict.items():
        if i == j:
            bqm.add_linear(i, w)
        else:
            bqm.add_quadratic(i, j, w)
            
    # 各ブロックにOne-Hot制約を追加: sum(x_j) = 1
    # ペナルティ: lambda * (sum(x_j) - 1)^2
    idx = 0
    for sz in block_sizes:
        terms = [(idx + k, 1.0) for k in range(sz)]
        bqm.add_linear_equality_constraint(
            terms,
            lagrange_multiplier=lambda_penalty,
            constant=-1.0
        )
        idx += sz
        
    return bqm

def solve_fmqa(
    qubo_dict: Dict[Tuple[int, int], float],
    block_sizes: List[int],
    lambda_penalty: float = 5.0,
    offset: float = 0.0,
    num_reads: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    FMQA (Simulated Annealing) を用いてペナルティ付きQUBOを解く。
    
    Returns:
        dict containing:
            - state_probabilities: 各ビット列の出現確率辞書
            - feasibility_rate: 制約を満たしたサンプルの割合
            - best_feasible_sample: 制約を満たしたサンプル中の最良解
            - best_overall_sample: 全サンプル中の最良解
    """
    bqm = build_penalty_bqm(qubo_dict, block_sizes, lambda_penalty, offset)
    
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    
    total_vars = sum(block_sizes)
    state_counts = {}
    
    feasible_count = 0
    best_feasible = None
    min_feasible_energy = float("inf")
    
    for sample_record in sampleset.data():
        sample = sample_record.sample
        energy = sample_record.energy
        num_occ = sample_record.num_occurrences
        
        bit_tuple = tuple(int(sample[i]) for i in range(total_vars))
        bit_str = "".join(str(b) for b in bit_tuple)
        state_counts[bit_str] = state_counts.get(bit_str, 0) + num_occ
        
        feasible = is_feasible_one_hot(list(bit_tuple), block_sizes)
        if feasible:
            feasible_count += num_occ
            # 純粋なQUBO目的関数値（ペナルティなし）を計算
            pure_fval = offset
            for (i, j), w in qubo_dict.items():
                if bit_tuple[i] == 1 and bit_tuple[j] == 1:
                    pure_fval += w
            if pure_fval < min_feasible_energy:
                min_feasible_energy = pure_fval
                best_feasible = {
                    "bitstring": bit_str,
                    "x": bit_tuple,
                    "fval": pure_fval,
                    "energy": energy
                }
        
    state_probs = {k: v / num_reads for k, v in state_counts.items()}
    feasibility_rate = feasible_count / num_reads
    
    first_record = sampleset.first
    best_overall = {
        "bitstring": "".join(str(first_record.sample[i]) for i in range(total_vars)),
        "x": tuple(int(first_record.sample[i]) for i in range(total_vars)),
        "energy": first_record.energy,
        "feasible": is_feasible_one_hot([first_record.sample[i] for i in range(total_vars)], block_sizes)
    }
    
    return {
        "solver": "FMQA (SA)",
        "lambda_penalty": lambda_penalty,
        "state_probabilities": state_probs,
        "feasibility_rate": feasibility_rate,
        "best_feasible_sample": best_feasible,
        "best_overall_sample": best_overall
    }
