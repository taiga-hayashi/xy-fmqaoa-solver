"""
Evaluation metrics, regret calculation, and statistical analysis tools.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon
from typing import Dict, Any, List, Tuple


def calculate_normalized_regret(f_val: float, f_opt: float, f_worst: float) -> float:
    """
    Normalized Regret:
    R = (f - f_opt) / (f_worst - f_opt)
    R = 0 is the theoretical best (global optimum), R = 1 is the worst.
    """
    denom = f_worst - f_opt
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip((f_val - f_opt) / denom, 0.0, 1.0))


def compute_bootstrap_ci(data: np.ndarray | List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Calculate 95% bootstrap confidence interval of the median."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot_medians = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_medians.append(np.median(sample))
    lower = float(np.percentile(boot_medians, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(boot_medians, (1.0 + ci) / 2.0 * 100))
    return lower, upper


def perform_paired_wilcoxon(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Perform Wilcoxon signed-rank test for paired samples."""
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0.0) or len(diff) < 3:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}
