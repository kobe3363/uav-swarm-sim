"""smdp_convergence tests: synthetic histories, hand-computed Wilson CIs.

All histories are built directly (no simulation) -- millisecond tests. The
Wilson expectations are computed INDEPENDENTLY inside the tests (closed forms
for k == n, the raw score formula for the branching case), never by calling
the production wilson_ci on the same inputs.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.enums import AgentState
from uav_swarm_sim.metrics.convergence import Z_95, wilson_ci
from uav_swarm_sim.metrics.run_output import build_results_single
from uav_swarm_sim.metrics.smdp_convergence import (
    convergence_report,
    format_table,
    pool_counts,
    report_from_counts,
    report_to_json,
)
from uav_swarm_sim.metrics.smdp_estimator import STATE_ORDER, estimate
from uav_swarm_sim.metrics.state_history import StateHistory

S = AgentState


def _cyclic_history(cycles: int, order: list[AgentState], agent_id: int = 0,
                    h: StateHistory | None = None, t0: float = 0.0) -> StateHistory:
    """cycles x (order) for one agent; each sojourn lasts 1 s."""
    h = h if h is not None else StateHistory()
    t = t0
    for _ in range(cycles):
        for st in order:
            h.open(agent_id, st, t)
            t += 1.0
            h.close(agent_id, t, "next")
    h.finalize(t)
    return h

_CYCLE = [S.S0_IDLE, S.S1_TRANSIT, S.S2_MISSION, S.S3_RTH, S.S_SWAP]


# --------------------------------------------------------------------------- #
# T1: deterministic cycle, k == n -> closed-form Wilson lower bound            #
# --------------------------------------------------------------------------- #
def test_deterministic_cycle_k_equals_n():
    est = estimate(_cyclic_history(20, _CYCLE), close_failure_loop=True)
    rep = convergence_report(est)
    by_state = {sc.state: sc for sc in rep.per_state}

    s0 = by_state[S.S0_IDLE]
    assert s0.visits == 20 and s0.transitions_out == 20
    cell = s0.transitions[S.S1_TRANSIT]
    assert cell.count == 20 and cell.row_total == 20
    assert cell.p_hat == pytest.approx(1.0)
    # closed form for k == n: lo = n / (n + z^2), hi = 1
    assert cell.wilson95_lo == pytest.approx(20.0 / (20.0 + Z_95 ** 2), abs=1e-12)
    assert cell.wilson95_hi == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# T2: branching node -> Wilson CI recomputed from the raw score formula        #
# --------------------------------------------------------------------------- #
def test_branching_node_wilson_ci_hand_computed():
    # S2 -> S3 three times, S2 -> S_SWAP once (agent keeps flying after each)
    h = StateHistory()
    t = 0.0
    for branch in (S.S3_RTH, S.S3_RTH, S.S3_RTH, S.S_SWAP):
        for st in (S.S0_IDLE, S.S1_TRANSIT, S.S2_MISSION, branch):
            h.open(0, st, t)
            t += 1.0
            h.close(0, t, "next")
    h.finalize(t)
    rep = convergence_report(estimate(h, close_failure_loop=True))
    s2 = {sc.state: sc for sc in rep.per_state}[S.S2_MISSION]
    assert s2.transitions_out == 4
    c3, cs = s2.transitions[S.S3_RTH], s2.transitions[S.S_SWAP]
    assert (c3.count, cs.count) == (3, 1)
    assert c3.p_hat == pytest.approx(0.75) and cs.p_hat == pytest.approx(0.25)

    # independent recomputation of the score interval for k=3, n=4
    z, n, p = Z_95, 4.0, 0.75
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    assert c3.wilson95_lo == pytest.approx(center - half, abs=1e-12)
    assert c3.wilson95_hi == pytest.approx(center + half, abs=1e-12)
    assert c3.wilson95_lo == pytest.approx(0.3006, abs=1e-3)   # sanity anchors
    assert c3.wilson95_hi == pytest.approx(0.9544, abs=1e-3)
    # complementary cell mirrors it: CI(k) = 1 - reversed CI(n-k)
    assert cs.wilson95_lo == pytest.approx(1.0 - c3.wilson95_hi, abs=1e-12)
    assert cs.wilson95_hi == pytest.approx(1.0 - c3.wilson95_lo, abs=1e-12)


# --------------------------------------------------------------------------- #
# T3: a state visited ONCE -> legitimately wide CI                             #
# --------------------------------------------------------------------------- #
def test_single_visit_state_has_wide_ci():
    h = _cyclic_history(10, _CYCLE)
    # one extra S_OBS excursion appended by the same agent
    t = 50.0
    for st in (S.S1_TRANSIT, S.S_OBS, S.S3_RTH):
        h.open(0, st, t)
        t += 1.0
        h.close(0, t, "next")
    h.finalize(t)
    rep = convergence_report(estimate(h, close_failure_loop=True))
    obs = {sc.state: sc for sc in rep.per_state}[S.S_OBS]
    assert obs.visits == 1 and obs.transitions_out == 1
    cell = obs.transitions[S.S3_RTH]
    # k = n = 1: lo = 1 / (1 + z^2) ~ 0.2065, hi = 1 -> width ~ 0.79
    assert cell.wilson95_lo == pytest.approx(1.0 / (1.0 + Z_95 ** 2), abs=1e-12)
    assert cell.wilson95_hi == pytest.approx(1.0)
    assert cell.width > 0.75
    assert rep.weakest_by_visits == (S.S_OBS, 1)
    f, tto, wc = rep.widest_ci
    assert (f, tto) == (S.S_OBS, S.S3_RTH) and wc.width == cell.width


# --------------------------------------------------------------------------- #
# T4: 0 outgoing transitions (truncated terminal) -> graceful, no div by zero  #
# --------------------------------------------------------------------------- #
def test_terminal_state_zero_outgoing_is_graceful():
    h = StateHistory()
    t = 0.0
    for st, d in [(S.S0_IDLE, 1.0), (S.S1_TRANSIT, 2.0), (S.S2_MISSION, 5.0), (S.S_FAIL, 1.0)]:
        h.open(0, st, t)
        t += d
        h.close(0, t, "next")
    h.finalize(t)
    rep = convergence_report(estimate(h, close_failure_loop=False))
    fail = {sc.state: sc for sc in rep.per_state}[S.S_FAIL]
    assert fail.visits == 1
    assert fail.transitions_out == 0 and fail.transitions == {}
    assert set(rep.unvisited) == {S.S_FERRY, S.S3_RTH, S.S_SWAP, S.S_OBS}
    # zero-visit row on the full base is also graceful (pooled-call shape)
    full = report_from_counts(list(STATE_ORDER),
                              np.zeros((8, 8)), np.zeros(8))
    assert full.per_state == [] and full.weakest_by_visits is None
    assert full.widest_ci is None and len(full.unvisited) == 8


# --------------------------------------------------------------------------- #
# T5: weakest-state summary picks min visits and the widest cell               #
# --------------------------------------------------------------------------- #
def test_weakest_state_summary():
    est = estimate(_cyclic_history(20, _CYCLE), close_failure_loop=True)
    rep = convergence_report(est)
    # all states visited 20x; S_SWAP row has n = 19 (truncated tail) -> its
    # k == n interval is the widest (width 1 - n/(n+z^2) grows as n shrinks)
    assert rep.weakest_by_visits[1] == 20
    f, tto, cell = rep.widest_ci
    assert (f, tto) == (S.S_SWAP, S.S0_IDLE)
    assert cell.row_total == 19


# --------------------------------------------------------------------------- #
# T6: visits semantics -- visits == transitions_out + truncated terminals      #
# --------------------------------------------------------------------------- #
def test_visits_field_alignment_and_semantics():
    est = estimate(_cyclic_history(20, _CYCLE), close_failure_loop=True)
    assert est.visits is not None and len(est.visits) == len(est.states)
    row_sums = est.n_transitions.sum(axis=1)
    for i, s in enumerate(est.states):
        truncated = 1 if s is S.S_SWAP else 0  # the last sojourn is S_SWAP
        assert est.visits[i] == row_sums[i] + truncated


# --------------------------------------------------------------------------- #
# T7: pool_counts == counts of the concatenated history                        #
# --------------------------------------------------------------------------- #
def test_pool_counts_matches_concatenated_history():
    est_a = estimate(_cyclic_history(5, _CYCLE, agent_id=0), close_failure_loop=True)
    est_b = estimate(_cyclic_history(7, _CYCLE, agent_id=1), close_failure_loop=True)
    states, counts, visits = pool_counts([est_a, est_b])
    assert states == STATE_ORDER

    both = _cyclic_history(5, _CYCLE, agent_id=0)
    both = _cyclic_history(7, _CYCLE, agent_id=1, h=both)
    est_ab = estimate(both, close_failure_loop=True)
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    for a, sa in enumerate(est_ab.states):
        assert visits[idx[sa]] == est_ab.visits[a]
        for b, sb in enumerate(est_ab.states):
            assert counts[idx[sa], idx[sb]] == est_ab.n_transitions[a, b]
    # pooled report goes through the same path without error
    rep = report_from_counts(states, counts, visits)
    assert {sc.state for sc in rep.per_state} == set(est_ab.states)


# --------------------------------------------------------------------------- #
# T8: ADDITIVE ONLY -- build_results_single old keys/values are untouched      #
# --------------------------------------------------------------------------- #
def _stub_result():
    outcome = SimpleNamespace(value="MISSION_SUCCESS")
    metrics = SimpleNamespace(
        total_energy_j=123.0, duration_s=45.0, workload_std_m=6.0,
        n_swaps=2, n_failures=0, planning_time_s=0.5,
        per_agent_length_m={0: 10.0, 1: 12.0},
    )
    return SimpleNamespace(outcome=outcome, coverage_frac=0.987, aborted=False,
                           metrics=metrics)


def test_build_results_single_is_additive_only():
    est = estimate(_cyclic_history(20, _CYCLE), close_failure_loop=True)
    result = _stub_result()
    identity = {"config_hash": "deadbeef"}

    old = build_results_single(result, est, identity=identity, wall_time_s=1.234)
    # the pre-change schema, frozen: top-level and smdp key sets + spot values
    assert set(old) == {"schema", "kind", "mode", "identity", "status",
                        "outcome", "smdp", "metrics", "timing"}
    assert set(old["smdp"]) == {"ergodic", "stationary_pi_time", "efficiency"}
    assert old["schema"] == "uav-swarm-sim/results/v1"
    assert old["status"] == "MISSION_SUCCESS"
    assert old["metrics"]["total_energy_j"] == 123.0
    assert old["timing"]["wall_time_total_s"] == 1.234

    conv = report_to_json(convergence_report(est))
    new = build_results_single(result, est, identity=identity, wall_time_s=1.234,
                               convergence=conv)
    assert new["smdp"]["convergence"] is conv          # the only addition
    stripped = {**new, "smdp": {k: v for k, v in new["smdp"].items()
                                if k != "convergence"}}
    assert stripped == old                             # everything else identical


# --------------------------------------------------------------------------- #
# T9: wilson_ci re-export + JSON/table smoke                                   #
# --------------------------------------------------------------------------- #
def test_wilson_ci_reexport_is_same_object():
    from uav_swarm_sim.experiments.spare_sizing import wilson_ci as reexported
    assert reexported is wilson_ci


def test_report_json_serializable_and_table_readable():
    est = estimate(_cyclic_history(3, _CYCLE), close_failure_loop=True)
    rep = convergence_report(est)
    blob = json.dumps(report_to_json(rep))            # strict-JSON valid
    assert "wilson95_lo" in blob and "multinomial" in blob
    table = format_table(rep)
    assert "SMDP convergence per state" in table
    assert "S2_MISSION" in table and "weakest by visits" in table
