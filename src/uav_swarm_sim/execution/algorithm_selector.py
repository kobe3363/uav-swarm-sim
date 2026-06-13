"""Three-tier scale-dependent algorithm selection (guideline 3.3).

The ~15-drone threshold is J. Liu's context-specific finding, so the middle band
measures rather than assumes: it returns both decomposers for paired comparison.
"""
from __future__ import annotations

import numpy as np

from ..infrastructure.enums import TierStrategy
from ..physical_model.motion_model import MotionModel
from ..planning.decomposition_base import Decomposer
from ..planning.kmeans_heuristic import KMeansHeuristicDecomposer
from ..planning.weighted_decomposition import WeightedTgcDecomposer


def select(n_drones: int, thresholds: tuple[int, int] = (15, 50)) -> TierStrategy:
    t0, t1 = thresholds
    if n_drones <= t0:
        return TierStrategy.HEURISTIC
    if n_drones >= t1:
        return TierStrategy.TGC
    return TierStrategy.COMPARE_BOTH


def make_decomposers(
    strategy: TierStrategy, motion: MotionModel, rng: np.random.Generator
) -> list[Decomposer]:
    heuristic = KMeansHeuristicDecomposer(motion, weighted=True, rng=rng)
    tgc = WeightedTgcDecomposer()
    if strategy is TierStrategy.HEURISTIC:
        return [heuristic]
    if strategy is TierStrategy.TGC:
        return [tgc]
    return [heuristic, tgc]
