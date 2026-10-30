"""Proposal solvers for closed-loop and fixed-FM optical filter benchmarks."""
from .baselines import random_search_trajectory, sample_random_feasible
from .categorical_sa import sample_categorical_sa
from .sa_penalty import AdaptivePenaltyTracker, build_penalty_bqm, sample_sa_auto, sample_sa_schedule
from .xy_qaoa import create_xy_backend, cvar, interp, interp_ladder_search, angle_search

__all__ = [
    "AdaptivePenaltyTracker",
    "angle_search",
    "build_penalty_bqm",
    "create_xy_backend",
    "cvar",
    "interp",
    "interp_ladder_search",
    "random_search_trajectory",
    "sample_categorical_sa",
    "sample_random_feasible",
    "sample_sa_auto",
    "sample_sa_schedule",
]

