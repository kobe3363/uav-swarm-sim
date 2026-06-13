"""Batch 5 tests: metrics / SMDP analysis layer (isolated + hand-computable)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.config import MCConfig
from uav_swarm_sim.infrastructure.enums import AgentState
from uav_swarm_sim.metrics.convergence import ci_half_width, converged
from uav_swarm_sim.metrics.efficiency_score import efficiency
from uav_swarm_sim.metrics.monte_carlo import SingleRunResult, run, single_run_from_history
from uav_swarm_sim.metrics.smdp_estimator import STATE_ORDER, estimate
from uav_swarm_sim.metrics.state_history import StateHistory
from uav_swarm_sim.metrics.stationary_distribution import (
    embedded_pi,
    stationary,
    time_weighted_pi,
)
from uav_swarm_sim.metrics.validation import validate_all

S = AgentState


# --------------------------------------------------------------------------- #
# state_history                                                               #
# --------------------------------------------------------------------------- #
def test_history_records_sojourns_and_is_tolerant():
    h = StateHistory()
    # first close with no open -> tolerated
    h.close(0, 1.0, "x")
    h.open(0, S.S1_TRANSIT, 0.0)
    h.close(0, 5.0, "zone_entry")
    h.open(0, S.S2_MISSION, 5.0)
    h.finalize(20.0)
    sj = h.sojourns()
    assert len(sj) == 2
    assert sj[0].state is S.S1_TRANSIT and sj[0].duration == 5.0
    assert sj[1].state is S.S2_MISSION and sj[1].duration == 15.0


# --------------------------------------------------------------------------- #
# a closed, hand-computable mission for one agent-slot                         #
# Cycle: S0 ->(1) S1 ->(2) S2 ->(2) S3 ->(2) S_SWAP ->(2) S0 -> ...            #
# durations chosen so the embedded vs time-weighted distinction is sharp.      #
# --------------------------------------------------------------------------- #
def _cyclic_history(cycles: int, durations: dict) -> StateHistory:
    h = StateHistory()
    order = [S.S0_IDLE, S.S1_TRANSIT, S.S2_MISSION, S.S3_RTH, S.S_SWAP]
    t = 0.0
    for _ in range(cycles):
        for st in order:
            h.open(0, st, t)
            t += durations[st]
            h.close(0, t, "next")
    h.finalize(t)
    return h


def test_embedded_pi_uniform_for_simple_cycle():
    # equal visit counts -> embedded pi uniform across the 5 states
    h = _cyclic_history(20, {S.S0_IDLE: 1, S.S1_TRANSIT: 1, S.S2_MISSION: 1, S.S3_RTH: 1, S.S_SWAP: 1})
    est = estimate(h, close_failure_loop=True)
    assert est.ergodic
    pi_emb, pi_time = stationary(est)
    # 5 states each visited equally -> embedded ~ 0.2 each
    for v in pi_emb:
        assert v == pytest.approx(0.2, abs=1e-6)


def test_time_weighting_changes_distribution():
    # S2 held long, S_OBS-like brief states: time-weighted must up-weight S2
    durs = {S.S0_IDLE: 1.0, S.S1_TRANSIT: 1.0, S.S2_MISSION: 16.0, S.S3_RTH: 1.0, S.S_SWAP: 1.0}
    h = _cyclic_history(30, durs)
    est = estimate(h, close_failure_loop=True)
    pi_emb, pi_time = stationary(est)
    i2 = est.states.index(S.S2_MISSION)
    # embedded gives S2 ~0.2 (equal visits); time-weighted gives 16/20 = 0.8
    assert pi_emb[i2] == pytest.approx(0.2, abs=1e-6)
    assert pi_time[i2] == pytest.approx(16.0 / 20.0, abs=1e-6)
    assert pi_time[i2] > pi_emb[i2]  # the correction matters


def test_time_weighted_pi_formula_directly():
    pi_emb = np.array([0.5, 0.5])
    m = np.array([1.0, 9.0])
    pw = time_weighted_pi(pi_emb, m)
    assert pw[0] == pytest.approx(0.1) and pw[1] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# S_FAIL dual view + ergodicity refusal                                       #
# --------------------------------------------------------------------------- #
def _history_with_terminal_failure() -> StateHistory:
    h = StateHistory()
    # one full cycle then a failure that ends the slot
    seq = [(S.S0_IDLE, 1), (S.S1_TRANSIT, 2), (S.S2_MISSION, 5), (S.S3_RTH, 2),
           (S.S_SWAP, 3), (S.S0_IDLE, 1), (S.S1_TRANSIT, 2), (S.S2_MISSION, 4), (S.S_FAIL, 1)]
    t = 0.0
    for st, d in seq:
        h.open(0, st, t)
        t += d
        h.close(0, t, "next")
    h.finalize(t)
    return h


def test_failure_loop_closed_is_ergodic():
    est = estimate(_history_with_terminal_failure(), close_failure_loop=True)
    assert S.S_FAIL in est.states
    assert est.ergodic  # synthetic S_FAIL -> S0 closes the loop
    pi_emb, pi_time = stationary(est)  # should not raise
    assert abs(pi_time.sum() - 1.0) < 1e-9


def test_failure_loop_open_is_not_ergodic_and_refuses():
    est = estimate(_history_with_terminal_failure(), close_failure_loop=False)
    assert not est.ergodic  # S_FAIL absorbing -> chain not strongly connected
    with pytest.raises(ValueError):
        stationary(est)


# --------------------------------------------------------------------------- #
# efficiency score                                                            #
# --------------------------------------------------------------------------- #
def test_efficiency_includes_swap_in_denominator():
    states = [S.S2_MISSION, S.S3_RTH, S.S_OBS, S.S_SWAP]
    pi = np.array([0.6, 0.1, 0.1, 0.2])
    # 0.6 / (0.1 + 0.1 + 0.2) = 1.5
    assert efficiency(pi, states) == pytest.approx(1.5)


def test_efficiency_ignores_idle_and_fail():
    states = [S.S0_IDLE, S.S2_MISSION, S.S3_RTH, S.S_OBS, S.S_SWAP, S.S_FAIL]
    pi = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
    # idle (0.3) and fail (0.1) excluded: 0.3 / (0.1+0.1+0.1) = 1.0
    assert efficiency(pi, states) == pytest.approx(1.0)


def test_efficiency_infinite_when_no_overhead():
    states = [S.S2_MISSION]
    assert math.isinf(efficiency(np.array([1.0]), states))


def test_swap_heavy_lowers_efficiency():
    states = [S.S2_MISSION, S.S3_RTH, S.S_OBS, S.S_SWAP]
    low_swap = efficiency(np.array([0.6, 0.1, 0.1, 0.05]), states)
    high_swap = efficiency(np.array([0.6, 0.1, 0.1, 0.40]), states)
    assert high_swap < low_swap  # swap time is overhead (throughput view)


# --------------------------------------------------------------------------- #
# convergence + monte carlo                                                   #
# --------------------------------------------------------------------------- #
def test_ci_half_width_and_converged():
    assert math.isinf(ci_half_width([0.5]))
    samples = [0.5, 0.5, 0.5, 0.5]
    assert ci_half_width(samples) == pytest.approx(0.0)
    assert converged(samples, tol=0.01, n_min=3)
    assert not converged([0.5, 0.5], tol=0.01, n_min=3)  # below n_min


def test_monte_carlo_stops_on_convergence():
    # deterministic runs -> zero variance -> converge exactly at n_min
    def run_once(k):
        return SingleRunResult(
            states=[S.S2_MISSION, S.S3_RTH, S.S_OBS, S.S_SWAP],
            pi_time={S.S2_MISSION: 0.6, S.S3_RTH: 0.1, S.S_OBS: 0.1, S.S_SWAP: 0.2},
            efficiency=1.5,
        )
    cfg = MCConfig(n_max=1000, n_min=30, ci_tolerance=0.01)
    res = run(run_once, cfg)
    assert res.converged
    assert res.n_runs == 30  # stops at n_min once CI half-width is 0
    assert res.pi_time_mean[S.S2_MISSION] == pytest.approx(0.6)
    assert res.efficiency_mean == pytest.approx(1.5)


def test_monte_carlo_caps_at_n_max_when_noisy():
    rng = np.random.default_rng(0)

    def run_once(k):
        v = float(rng.uniform(0.0, 1.0))  # high variance -> never converges tightly
        return SingleRunResult([S.S2_MISSION, S.S3_RTH], {S.S2_MISSION: v, S.S3_RTH: 1 - v}, 1.0)

    cfg = MCConfig(n_max=40, n_min=10, ci_tolerance=1e-6)
    res = run(run_once, cfg)
    assert not res.converged and res.n_runs == 40


def test_single_run_from_history_roundtrip():
    h = _cyclic_history(25, {S.S0_IDLE: 1, S.S1_TRANSIT: 1, S.S2_MISSION: 10, S.S3_RTH: 1, S.S_SWAP: 1})
    r = single_run_from_history(h)
    assert not r.aborted
    assert r.pi_time[S.S2_MISSION] > r.pi_time[S.S3_RTH]
    assert r.efficiency > 0


# --------------------------------------------------------------------------- #
# validation verdicts                                                         #
# --------------------------------------------------------------------------- #
def test_validation_directional_and_magnitude():
    rows = validate_all({
        "workload_classic": 200.0, "workload_weighted": 60.0,
        "duration_classic": 219.8, "duration_weighted": 157.8,
        "plan_time_heuristic": 6.0, "plan_time_tgc": 0.1,
        "replan_per_task_s": 5e-4,
    })
    verdicts = {r.claim.split()[0]: r.verdict for r in rows}
    assert all("FAIL" not in r.verdict for r in rows)
    # a failing direction is caught
    bad = validate_all({"workload_classic": 60.0, "workload_weighted": 200.0})
    assert bad[0].verdict == "FAIL"
