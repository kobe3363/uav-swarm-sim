"""The scalar verdict: useful work over flight-plus-service overhead.

efficiency = pi_time[S2] / (pi_time[S1] + pi_time[S_FERRY] + pi_time[S3] + pi_time[S_OBS] + pi_time[S_SWAP])

The denominator is the time the drone spends NOT doing useful mission work but
still airborne or in service: outbound transit (S1), camera-off repositioning
between coverage strips (S_FERRY), return-to-home (S3), obstacle avoidance
(S_OBS), and battery swap (S_SWAP). S0 and S_FAIL appear in neither numerator nor
denominator: idle ground time is not flight overhead, and failure time is
accounted by the replacement device. Swap time is included (original Decision 2):
swap is mission overhead under a throughput-oriented metric, so strategies that
economize swaps via the weighted decomposition are rewarded exactly where the
thesis claims.

S_FERRY in the denominator (camera ON only in S2 is productive)
---------------------------------------------------------------
Only the COVERAGE strips (camera on, S2_MISSION) photograph the surface and count
as productive. The inter-strip connectors and any out-of-area repositioning are
flown with the camera off (S_FERRY): real flight energy, zero coverage benefit.
Counting S_FERRY as overhead makes pi(S_FERRY) the direct "cost of the survey
shape" -- concave/elongated shapes spend more time ferrying -- and keeps the
score's meaning exact: only S2 is productive, everything else airborne is
overhead. Histories that never ferry are unaffected (pi(S_FERRY)=0)."""
from __future__ import annotations

import logging
import math

import numpy as np

from ..infrastructure.enums import AgentState

_LOG = logging.getLogger(__name__)

_DENOM_STATES = (
    AgentState.S1_TRANSIT,
    AgentState.S_FERRY,
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
