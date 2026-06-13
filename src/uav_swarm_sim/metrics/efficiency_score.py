"""The scalar verdict: useful work over flight-plus-service overhead.

efficiency = pi_time[S2] / (pi_time[S3] + pi_time[S_OBS] + pi_time[S_SWAP])

The denominator includes pi(S_SWAP) (Decision 2): swap time is mission overhead
under a throughput-oriented metric, so strategies that economize swaps via the
weighted decomposition are rewarded exactly where the thesis claims. S0 and
S_FAIL appear in neither numerator nor denominator: idle ground time is not
flight overhead, and failure time is accounted by the replacement device.

Must be computed on the TIME-WEIGHTED pi (not the embedded visit frequencies).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from ..infrastructure.enums import AgentState

_LOG = logging.getLogger(__name__)

_DENOM_STATES = (AgentState.S3_RTH, AgentState.S_OBS, AgentState.S_SWAP)


def efficiency(pi_time: np.ndarray, states: list[AgentState]) -> float:
    idx = {s: i for i, s in enumerate(states)}

    def get(s: AgentState) -> float:
        i = idx.get(s)
        return float(pi_time[i]) if i is not None else 0.0

    num = get(AgentState.S2_MISSION)
    denom = sum(get(s) for s in _DENOM_STATES)
    if denom < 1e-9:
        _LOG.warning("efficiency denominator ~0 (no flight/service overhead); returning inf")
        return math.inf
    return num / denom
