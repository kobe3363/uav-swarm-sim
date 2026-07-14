"""Empirically-derived Semi-Markov Decision Process estimator.

Why semi-Markov (not Markov): battery level is hidden memory -- the probability
of leaving S2 for S3 grows with time spent in S2, violating memorylessness. The
state is therefore two-dimensional (state, sojourn time), with sojourn time as
backward recurrence time. The standard estimator for this structure is the
embedded jump chain plus mean sojourn times, which this module produces.

S_FAIL DUAL VIEW (Decision 3a) -- documented here verbatim:
  * The PHYSICAL layer recorded S_FAIL as terminal (agent removed, zone
    redistributed -- thesis-faithful, irreversible).
  * For the ANALYSIS layer, with close_failure_loop=True (default), each terminal
    S_FAIL sojourn is closed at a configurable mean repair/replacement duration
    and given a synthetic transition S_FAIL -> S0: the generic agent-slot is
    re-occupied by a replacement drone. This is an ergodicity device of the
    slot model, NOT a claim about the physical swarm. With close_failure_loop=
    False the estimator returns ergodic=False and the stationary distribution is
    refused.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..infrastructure.enums import AgentState
from .state_history import Sojourn, StateHistory

# canonical ordering used wherever a full state vector is needed
STATE_ORDER: list[AgentState] = [
    AgentState.S0_IDLE,
    AgentState.S1_TRANSIT,
    AgentState.S2_MISSION,
    AgentState.S_FERRY,
    AgentState.S3_RTH,
    AgentState.S_SWAP,
    AgentState.S_OBS,
    AgentState.S_FAIL,
]


@dataclass(frozen=True)
class SmdpEstimate:
    states: list[AgentState]      # visited states, in canonical order
    P: np.ndarray                 # embedded-chain transition matrix (row-stochastic)
    mean_sojourn_s: np.ndarray    # mean sojourn time per state (same index as states)
    n_transitions: np.ndarray     # raw transition counts
    ergodic: bool
    unreachable: list[AgentState]
    closed_failure_loop: bool
    # raw visit counts per state (same index as states), INCLUDING truncated
    # terminal sojourns that produced no outgoing transition -- so
    # visits[i] >= n_transitions[i].sum(), with equality unless state i ended
    # an agent's history unclosed. Default None keeps the frozen dataclass
    # backward-compatible; estimate() always fills it.
    visits: np.ndarray | None = None


def estimate(
    history: StateHistory,
    close_failure_loop: bool = True,
    failure_repair_s: float = 600.0,
) -> SmdpEstimate:
    by_agent: dict[int, list[Sojourn]] = defaultdict(list)
    for s in history.sojourns():
        by_agent[s.agent_id].append(s)

    full = len(STATE_ORDER)
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    N = np.zeros((full, full), dtype=float)
    dur = np.zeros(full, dtype=float)
    visits = np.zeros(full, dtype=float)

    for aid, sjs in by_agent.items():
        sjs = sorted(sjs, key=lambda s: s.t_in)
        for k in range(len(sjs) - 1):
            i, j = idx[sjs[k].state], idx[sjs[k + 1].state]
            N[i, j] += 1
            dur[i] += sjs[k].duration
            visits[i] += 1
        last = sjs[-1]
        li = idx[last.state]
        if close_failure_loop and last.state is AgentState.S_FAIL:
            # synthetic replacement closure: S_FAIL -> S0 at the repair duration
            N[li, idx[AgentState.S0_IDLE]] += 1
            dur[li] += failure_repair_s
            visits[li] += 1
        else:
            # legitimate terminal sojourn: account duration, drop truncated transition
            dur[li] += last.duration
            visits[li] += 1

    visited = [s for s in STATE_ORDER if visits[idx[s]] > 0]
    vmap = {s: i for i, s in enumerate(visited)}
    m = len(visited)
    Psub = np.zeros((m, m))
    Nsub = np.zeros((m, m))
    mean = np.zeros(m)
    unreachable: list[AgentState] = []

    for s in visited:
        si = idx[s]
        row = np.array([N[si, idx[t]] for t in visited], dtype=float)
        total = row.sum()
        ri = vmap[s]
        Nsub[ri] = row
        mean[ri] = dur[si] / visits[si] if visits[si] > 0 else 0.0
        if total > 0:
            Psub[ri] = row / total
        else:
            unreachable.append(s)  # visited but no outgoing transition (truncated terminal)

    ergodic = _is_ergodic(Psub) and not unreachable
    return SmdpEstimate(
        states=visited,
        P=Psub,
        mean_sojourn_s=mean,
        n_transitions=Nsub,
        ergodic=ergodic,
        unreachable=unreachable,
        closed_failure_loop=close_failure_loop,
        visits=np.array([visits[idx[s]] for s in visited], dtype=float),
    )


def _is_ergodic(P: np.ndarray) -> bool:
    if P.shape[0] == 0:
        return False
    if P.shape[0] == 1:
        return P[0, 0] > 0  # single state must self-loop
    g = nx.DiGraph()
    n = P.shape[0]
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                g.add_edge(i, j)
    return nx.is_strongly_connected(g)
