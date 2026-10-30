"""Uniform feasible proposal (`RandomFM`) and pure random search (`Random`) baselines."""
from __future__ import annotations

import random

import numpy as np

from ..optics import onehot
from ..surrogate import initial_ids


def sample_random_feasible(
    layers: int,
    seed: int,
    *,
    num_reads: int = 100,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, (num_reads, layers))
    return onehot(labels), {
        "alpha": None,
        "lambda": None,
        "S": None,
        "sampler_seed": seed,
        "sampler": "uniform feasible labels",
    }


def random_search_trajectory(
    seed: int,
    layers: int,
    truth: np.ndarray,
    budget: int,
    *,
    initial_points: int = 40,
    random_stream_offset: int = 73101,
) -> dict:
    ids = initial_ids(seed, layers, count=initial_points)
    seen = set(ids)
    remaining = [i for i in range(4**layers) if i not in seen]
    choices = random.Random(seed + random_stream_offset).sample(remaining, budget - len(ids))
    final_ids = ids + choices
    ys = [float(truth[i]) for i in final_ids]
    return {
        "seed": seed,
        "condition": {"id": "Random", "kind": "random"},
        "layers": layers,
        "budget": budget,
        "initial_ids": ids,
        "final_ids": final_ids,
        "final_values": ys,
        "actual_evaluations": len(final_ids),
        "final_best": float(min(ys)),
        "events": [],
        "total_seconds": 0.0,
        "stopping_reason": "evaluation_cap",
    }

