import torch
import numpy as np
from typing import Tuple, Dict

def fm_to_qubo(model: torch.nn.Module) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    TorchFMの重みパラメータからQUBO辞書とオフセットを生成する。
    
    Args:
        model (TorchFM): 学習済み（または初期化済み）のTorchFMモデル
        
    Returns:
        qubo (dict): {(i, j): weight} の形式のQUBO辞書。
                     i == j の場合は1次項、i < j の場合は2次項。
        offset (float): バイアス（定数項）
    """
    # パラメータの取得（CPU上のnumpy配列に変換）
    # model.lin.weight は (1, d) の形状を持つ
    linear_weights = model.lin.weight.detach().cpu().numpy().flatten()
    bias = model.lin.bias.detach().cpu().numpy().item()
    
    # model.V は (d, k) の形状を持つ
    v_matrix = model.V.detach().cpu().numpy()
    d = v_matrix.shape[0]
    
    qubo = {}
    
    # 1次項 (i == j)
    for i in range(d):
        qubo[(i, i)] = float(linear_weights[i])
        
    # 2次項 (i < j)
    # v_i と v_j の内積が相互作用の重みとなる
    for i in range(d):
        for j in range(i + 1, d):
            # v_i = v_matrix[i], v_j = v_matrix[j]
            interaction = np.dot(v_matrix[i], v_matrix[j])
            qubo[(i, j)] = float(interaction)
            
    return qubo, bias
