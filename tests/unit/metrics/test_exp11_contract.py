"""EXP-11 raw mission data contract (build_mission_contract + serializer wiring).

The builder is exercised with synthetic MissionResult-shaped objects and a real
StateHistory so the schema contract is tested without running the engine. The
engine-sourced fields (coverage areas, final battery) and the final_soc<->trace
cross-check live in the integration test.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from uav_swarm_sim.infrastructure.enums import AgentState, Outcome
from uav_swarm_sim.metrics.monte_carlo import MCResult, SingleRunResult
from uav_swarm_sim.metrics.run_output import (
    CONTRACT_SCHEMA,
    RESULTS_SCHEMA,
    _jsonable,
    build_mission_contract,
    build_results_mc,
    build_results_single,
)
from uav_swarm_sim.metrics.state_history import StateHistory

S = AgentState
CAP = 1000.0


# --------------------------------------------------------------------------- #
# fakes                                                                        #
# --------------------------------------------------------------------------- #
def _history(spans):
    """spans: {agent_id: [(state, t_in, t_out), ...]} in chronological order."""
    h = StateHistory()
    end = 0.0
    for aid, segs in spans.items():
        for st, t_in, t_out in segs:
            h.open(aid, st, t_in)
            h.close(aid, t_out, "x")
            end = max(end, t_out)
    h.finalize(end)
    return h


def _result(*, per_agent_energy, per_agent_length, initial_soc, final_battery,
            spans, photo_agents=(), safety_minima=None, safety_violations=(),
            coverage_measurements=None, coverage_frac=1.0, target_coverage_frac=1.0,
            outcome=Outcome.MISSION_SUCCESS, terminal_reason="coverage_complete",
            airborne_at_end=(), losses=(), retired_agents=(), duration_s=100.0):
    metrics = SimpleNamespace(
        total_energy_j=float(sum(per_agent_energy.values())),
        duration_s=duration_s,
        per_agent_energy_j=per_agent_energy,
        per_agent_length_m=per_agent_length,
    )
    return SimpleNamespace(
        metrics=metrics,
        history=_history(spans),
        photo_events=[SimpleNamespace(agent_id=a) for a in photo_agents],
        safety_minima=safety_minima,
        safety_violations=safety_violations,
        coverage_measurements=coverage_measurements,
        coverage_frac=coverage_frac,
        target_coverage_frac=target_coverage_frac,
        initial_soc_by_drone=initial_soc,
        final_battery_by_drone=final_battery,
        outcome=outcome,
        terminal_reason=terminal_reason,
        airborne_at_end=airborne_at_end,
        losses=losses,
        retired_agents=retired_agents,
    )


def _build(result):
    return build_mission_contract(result, capacity_j=CAP, decomposer_class="TgcDecomposer")


# --------------------------------------------------------------------------- #
# energy reconciliation (B1)                                                   #
# --------------------------------------------------------------------------- #
def test_no_swap_reconciles_with_zero_residual():
    # L0 - Lf == consumed exactly, no swap, no floor clamp.
    r = _result(
        per_agent_energy={0: 300.0, 1: 400.0},
        per_agent_length={0: 10.0, 1: 20.0},
        initial_soc=(1.0, 1.0),
        final_battery=((0, 700.0, 0.7), (1, 600.0, 0.6)),
        spans={0: [(S.S2_MISSION, 0, 10), (S.S_LANDED, 10, 10)],
               1: [(S.S2_MISSION, 0, 20), (S.S_LANDED, 20, 20)]},
    )
    rec = _build(r)["energy"]["reconciliation"]
    assert rec["reconcilable"] is True
    assert rec["sum_per_agent_j"] == pytest.approx(700.0)
    assert rec["clamp_residual_j"] == pytest.approx(0.0, abs=1e-9)
    assert rec["clamp_residual_j"] >= -1e-9  # never negative
    assert rec["swap_recharge_j"] == 0.0
    assert rec["n_swaps_total"] == 0  # key present in BOTH branches (stable schema)


def test_floored_battery_reports_positive_clamp_residual():
    # drone 0 ordered 1200 J but only had 1000 J -> 200 J clamp loss, final 0.
    r = _result(
        per_agent_energy={0: 1200.0},
        per_agent_length={0: 30.0},
        initial_soc=(1.0,),
        final_battery=((0, 0.0, 0.0),),
        spans={0: [(S.S2_MISSION, 0, 30), (S.S_FAIL, 30, 30)]},
        outcome=Outcome.MISSION_FAILED, losses=((0, 30.0, "battery_depleted"),),
    )
    rec = _build(r)["energy"]["reconciliation"]
    assert rec["reconcilable"] is True
    assert rec["clamp_residual_j"] == pytest.approx(200.0)
    assert rec["clamp_residual_j"] >= 0.0


def test_swap_makes_reconciliation_refuse_a_residual():
    r = _result(
        per_agent_energy={0: 900.0},
        per_agent_length={0: 40.0},
        initial_soc=(1.0,),
        final_battery=((0, 500.0, 0.5),),
        spans={0: [(S.S2_MISSION, 0, 20), (S.S_SWAP, 20, 25),
                   (S.S2_MISSION, 25, 40), (S.S_LANDED, 40, 40)]},
    )
    c = _build(r)
    rec = c["energy"]["reconciliation"]
    assert rec["reconcilable"] is False
    assert rec["swap_recharge_j"] is None
    assert rec["clamp_residual_j"] is None
    assert rec["n_swaps_total"] == 1
    assert c["per_agent"]["0"]["n_swaps"] == 1


# --------------------------------------------------------------------------- #
# cut_moment / min final SoC (M2)                                              #
# --------------------------------------------------------------------------- #
def test_cut_moment_from_last_sojourn_state():
    r = _result(
        per_agent_energy={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        per_agent_length={i: 1.0 for i in range(5)},
        initial_soc=(1.0,) * 5,
        final_battery=tuple((i, 500.0, 0.5) for i in range(5)),
        spans={
            0: [(S.S2_MISSION, 0, 5), (S.S_LANDED, 5, 5)],
            1: [(S.S2_MISSION, 0, 5), (S.S_FAIL, 5, 5)],
            2: [(S.S2_MISSION, 0, 5), (S.S_SWAP, 5, 5)],
            3: [(S.S0_IDLE, 0, 5)],
            4: [(S.S1_TRANSIT, 0, 5)],  # still airborne at end
        },
    )
    pa = _build(r)["per_agent"]
    assert pa["0"]["cut_moment"] == "landed"
    assert pa["1"]["cut_moment"] == "failed"
    assert pa["2"]["cut_moment"] == "swapping"
    assert pa["3"]["cut_moment"] == "idle_on_ground"
    assert pa["4"]["cut_moment"] == "airborne_at_end"


def test_min_final_soc_can_belong_to_failed_or_airborne_drone():
    # the fleet minimum SoC is a lost (failed) drone; consumer can see the label.
    r = _result(
        per_agent_energy={0: 1.0, 1: 1.0},
        per_agent_length={0: 1.0, 1: 1.0},
        initial_soc=(1.0, 1.0),
        final_battery=((0, 800.0, 0.8), (1, 50.0, 0.05)),
        spans={0: [(S.S2_MISSION, 0, 5), (S.S_LANDED, 5, 5)],
               1: [(S.S2_MISSION, 0, 5), (S.S_FAIL, 5, 5)]},
        outcome=Outcome.MISSION_FAILED, losses=((1, 5.0, "battery_depleted"),),
    )
    pa = _build(r)["per_agent"]
    lo = min(pa.values(), key=lambda d: d["final_soc"])
    assert lo["final_soc"] == pytest.approx(0.05)
    assert lo["cut_moment"] == "failed"


# --------------------------------------------------------------------------- #
# coverage: raster vs proxy                                                    #
# --------------------------------------------------------------------------- #
def test_coverage_raster_areas_and_fracs():
    meas = {"source": "raster", "a_target_m2": 1000.0, "a_plannable_m2": 900.0,
            "target_covered_area_m2": 950.0, "plannable_covered_area_m2": 890.0}
    r = _result(
        per_agent_energy={0: 1.0}, per_agent_length={0: 1.0}, initial_soc=(1.0,),
        final_battery=((0, 500.0, 0.5),), spans={0: [(S.S_LANDED, 0, 1)]},
        coverage_measurements=meas, coverage_frac=0.988, target_coverage_frac=0.95,
    )
    cov = _build(r)["coverage"]
    assert cov["source"] == "raster"
    assert cov["a_target_m2"] == 1000.0 and cov["a_plannable_m2"] == 900.0
    assert cov["target_covered_area_m2"] == 950.0
    assert cov["plannable_coverage_frac"] == pytest.approx(0.988)
    assert cov["target_coverage_frac"] == pytest.approx(0.95)


def test_coverage_proxy_has_none_areas_and_labeled_source():
    meas = {"source": "segment_proxy", "a_target_m2": None, "a_plannable_m2": None,
            "target_covered_area_m2": None, "plannable_covered_area_m2": None}
    r = _result(
        per_agent_energy={0: 1.0}, per_agent_length={0: 1.0}, initial_soc=(1.0,),
        final_battery=((0, 500.0, 0.5),), spans={0: [(S.S_LANDED, 0, 1)]},
        coverage_measurements=meas, coverage_frac=0.5, target_coverage_frac=None,
    )
    cov = _build(r)["coverage"]
    assert cov["source"] == "segment_proxy"
    assert cov["a_plannable_m2"] is None
    assert cov["plannable_coverage_frac"] == pytest.approx(0.5)
    assert cov["target_coverage_frac"] is None


def test_coverage_none_measurements_still_complete_schema():
    # a result with no measurement object at all must still expose all area keys
    # (as None), so a key-based consumer never hits a KeyError.
    r = _result(
        per_agent_energy={0: 1.0}, per_agent_length={0: 1.0}, initial_soc=(1.0,),
        final_battery=((0, 500.0, 0.5),), spans={0: [(S.S_LANDED, 0, 1)]},
        coverage_measurements=None, coverage_frac=0.5, target_coverage_frac=None,
    )
    cov = _build(r)["coverage"]
    assert cov["source"] == "segment_proxy"
    for k in ("a_target_m2", "a_plannable_m2", "target_covered_area_m2",
              "plannable_covered_area_m2"):
        assert cov[k] is None


# --------------------------------------------------------------------------- #
# photos                                                                       #
# --------------------------------------------------------------------------- #
def test_photo_counts_fleet_and_per_agent():
    r = _result(
        per_agent_energy={0: 1.0, 1: 1.0}, per_agent_length={0: 1.0, 1: 1.0},
        initial_soc=(1.0, 1.0), final_battery=((0, 5.0, 0.5), (1, 5.0, 0.5)),
        spans={0: [(S.S_LANDED, 0, 1)], 1: [(S.S_LANDED, 0, 1)]},
        photo_agents=(0, 0, 0, 1),
    )
    c = _build(r)
    assert c["outcome"]["n_photos_fleet"] == 4
    assert c["per_agent"]["0"]["photo_count"] == 3
    assert c["per_agent"]["1"]["photo_count"] == 1


# --------------------------------------------------------------------------- #
# safety (B3): "not recorded" != "clean"                                      #
# --------------------------------------------------------------------------- #
def test_safety_recorded_keeps_hard_and_soft_separate():
    sm = {"n_hard": 2, "n_soft": 3, "min_separation_m": 1.5,
          "min_obstacle_clearance_m": 0.4}
    viols = (
        SimpleNamespace(kind="separation", severity="hard"),
        SimpleNamespace(kind="obstacle", severity="soft"),
    )
    r = _result(
        per_agent_energy={0: 1.0}, per_agent_length={0: 1.0}, initial_soc=(1.0,),
        final_battery=((0, 5.0, 0.5),), spans={0: [(S.S_LANDED, 0, 1)]},
        safety_minima=sm, safety_violations=viols,
    )
    saf = _build(r)["safety"]
    assert saf["recorded"] is True
    assert saf["n_hard"] == 2 and saf["n_soft"] == 3
    assert len(saf["violations"]) == 2


def test_safety_not_recorded_is_not_zero():
    r = _result(
        per_agent_energy={0: 1.0}, per_agent_length={0: 1.0}, initial_soc=(1.0,),
        final_battery=((0, 5.0, 0.5),), spans={0: [(S.S_LANDED, 0, 1)]},
        safety_minima=None,
    )
    saf = _build(r)["safety"]
    assert saf["recorded"] is False
    assert saf["n_hard"] is None and saf["n_soft"] is None
    assert saf["violations"] is None


# --------------------------------------------------------------------------- #
# structure: no cross-block duplication (M1) + alignment assertion            #
# --------------------------------------------------------------------------- #
def test_no_cross_block_duplication_only_per_agent_holds_values():
    r = _result(
        per_agent_energy={0: 1.0}, per_agent_length={0: 1.0}, initial_soc=(1.0,),
        final_battery=((0, 5.0, 0.5),), spans={0: [(S.S_LANDED, 0, 1)]},
    )
    c = _build(r)
    # the derived blocks name columns, they do not copy per-drone values
    for block in ("energy", "duration", "min_final_soc", "workload"):
        assert "fields" in c[block]
        assert not any(k == "per_agent" for k in c[block])
    assert set(c["min_final_soc"].keys()) == {"fields"}


def test_alignment_assertion_rejects_non_contiguous_ids():
    r = _result(
        per_agent_energy={0: 1.0, 2: 1.0}, per_agent_length={0: 1.0, 2: 1.0},
        initial_soc=(1.0, 1.0), final_battery=((0, 5.0, 0.5), (2, 5.0, 0.5)),
        spans={0: [(S.S_LANDED, 0, 1)], 2: [(S.S_LANDED, 0, 1)]},
    )
    with pytest.raises(ValueError):
        _build(r)


# --------------------------------------------------------------------------- #
# roundtrip + edges                                                            #
# --------------------------------------------------------------------------- #
def test_roundtrip_and_stable_types():
    r = _result(
        per_agent_energy={0: 300.0}, per_agent_length={0: 10.0}, initial_soc=(1.0,),
        final_battery=((0, 700.0, 0.7),), spans={0: [(S.S2_MISSION, 0, 10),
                                                     (S.S_LANDED, 10, 10)]},
    )
    c = _build(r)
    back = json.loads(json.dumps(_jsonable(c)))
    assert back["contract_schema"] == CONTRACT_SCHEMA
    pa = back["per_agent"]["0"]
    assert isinstance(pa["consumed_j"], float)
    assert isinstance(pa["n_swaps"], int) and isinstance(pa["photo_count"], int)
    assert pa["airborne_s"] == pytest.approx(10.0)


def test_zero_work_and_single_drone_edges():
    r = _result(
        per_agent_energy={0: 0.0}, per_agent_length={0: 0.0}, initial_soc=(1.0,),
        final_battery=((0, 1000.0, 1.0),), spans={0: [(S.S0_IDLE, 0, 0)]},
    )
    c = _build(r)
    pa = c["per_agent"]["0"]
    assert pa["length_m"] == 0.0 and pa["consumed_j"] == 0.0
    assert pa["airborne_s"] == 0.0 and pa["photo_count"] == 0
    assert c["energy"]["reconciliation"]["reconcilable"] is True


# --------------------------------------------------------------------------- #
# serializer wiring + flag-off byte identity                                  #
# --------------------------------------------------------------------------- #
def _fake_single():
    metrics = SimpleNamespace(total_energy_j=1.0, duration_s=2.0, workload_std_m=3.0,
                              n_swaps=0, n_failures=0, planning_time_s=0.1,
                              per_agent_length_m={0: 4.0})
    return SimpleNamespace(metrics=metrics, outcome=Outcome.MISSION_SUCCESS,
                           coverage_frac=1.0, aborted=False, initial_soc_by_drone=(1.0,))


def test_single_flag_off_omits_contract_key():
    est = SimpleNamespace(ergodic=False, states=[])
    out = build_results_single(_fake_single(), est, identity={}, wall_time_s=1.0)
    assert "mission_contract" not in out
    assert out["schema"] == RESULTS_SCHEMA


def test_single_flag_on_attaches_contract():
    est = SimpleNamespace(ergodic=False, states=[])
    contract = {"contract_schema": CONTRACT_SCHEMA, "marker": 1}
    out = build_results_single(_fake_single(), est, identity={}, wall_time_s=1.0,
                               mission_contract=contract)
    assert out["mission_contract"]["marker"] == 1


def test_mc_emits_per_replication_contract_when_present():
    runs = [
        SingleRunResult([], {}, 0.0, contract={"contract_schema": CONTRACT_SCHEMA, "r": 1}),
        SingleRunResult([], {}, 0.0, contract={"contract_schema": CONTRACT_SCHEMA, "r": 2}),
    ]
    mc = MCResult(n_runs=2, converged=False, pi_time_mean={}, pi_time_ci={},
                  efficiency_mean=0.0, efficiency_ci=0.0, aborted_frac=0.0,
                  convergence_trace=[], runs=runs)
    out = build_results_mc(mc, identity={}, wall_time_s=0.2)
    assert [x["replication"] for x in out["mission_contract"]] == [1, 2]
    assert out["mission_contract"][0]["r"] == 1


def test_mc_flag_off_omits_contract_key():
    runs = [SingleRunResult([], {}, 0.0), SingleRunResult([], {}, 0.0)]
    mc = MCResult(n_runs=2, converged=False, pi_time_mean={}, pi_time_ci={},
                  efficiency_mean=0.0, efficiency_ci=0.0, aborted_frac=0.0,
                  convergence_trace=[], runs=runs)
    out = build_results_mc(mc, identity={}, wall_time_s=0.2)
    assert "mission_contract" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
