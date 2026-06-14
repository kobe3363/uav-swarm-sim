"""The scalar verdict: useful work over flight-plus-service overhead.

efficiency = pi_time[S2] / (pi_time[S1] + pi_time[S3] + pi_time[S_OBS] + pi_time[S_SWAP])

The denominator is the time the drone spends NOT doing useful mission work but
still airborne or in service: outbound transit (S1), return-to-home (S3),
obstacle avoidance (S_OBS), and battery swap (S_SWAP). S0 and S_FAIL appear in
neither numerator nor denominator: idle ground time is not flight overhead, and
failure time is accounted by the replacement device. Swap time is included
(original Decision 2): swap is mission overhead under a throughput-oriented
metric, so strategies that economize swaps via the weighted decomposition are
rewarded exactly where the thesis claims.

S1_TRANSIT in the denominator (2.5D re-baseline, Decision 2 continued)
---------------------------------------------------------------------
Outbound transit was previously excluded while return (S3) was included. That was
symmetric enough in pure 2D, but once vertical energy is modelled the asymmetry
bites: the launch climb to the coverage altitude is spent inside S1, while the
RTH descent is spent inside S3. Counting S3 but not S1 would charge the descent
overhead and not the (symmetric) climb overhead, biasing the verdict in favour of
configurations with expensive climbs. Including S1 makes transit overhead -- climb
included -- count the same as return overhead, which is the throughput meaning of
the score: only S2 is productive; everything else airborne is overhead.

This changes the efficiency baseline for EVERY scenario (S1 is nonzero even in 2D
horizontal transit), so it is a deliberate re-baseline, not a byte-identical
change: the efficiency fixtures are expected to shift.

Must be computed on the TIME-WEIGHTED pi (not the embedded visit frequencies).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from ..infrastructure.enums import AgentState

_LOG = logging.getLogger(__name__)

_DENOM_STATES = (
    AgentState.S1_TRANSIT,
    AgentState.S3_RTH,
    AgentState.S_OBS,
    AgentState.S_SWAP,
)


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
