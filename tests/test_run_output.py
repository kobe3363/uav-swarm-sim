"""Tests for the structured run-output module (metrics/run_output.py).

Cover the directory/manifest plumbing, the JSON-safe conversion, and the plan /
results schema builders. The builders are exercised with synthetic MCResult /
MissionResult objects so the schema contract is tested instantly, without
running the engine (the engine path is covered by the live experiment and by an
end-to-end smoke elsewhere).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import config_path

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo, Outcome, PlannerKind
from uav_swarm_sim.metrics.monte_carlo import MCResult, SingleRunResult
from uav_swarm_sim.metrics.run_output import (
    PLAN_SCHEMA,
    RESULTS_SCHEMA,
    RUN_SCHEMA,
    RunContext,
    _jsonable,
    _outcome_counts,
    build_plan,
    build_results_mc,
    build_results_single,
)


@pytest.fixture
def cfg():
    return load_config(config_path(), overrides={
        "fleet.n_drones": 5,
        "env.geojson_path": "data/areas/smoke_area.geojson",
    })


# --------------------------------------------------------------------------- #
# JSON-safe conversion                                                         #
# --------------------------------------------------------------------------- #
def test_jsonable_non_finite_floats_become_null():
    import math
    j = _jsonable({"ci": math.inf, "x": float("nan"), "ok": 1.5, "neg": -math.inf})
    assert j["ci"] is None and j["x"] is None and j["neg"] is None
    assert j["ok"] == 1.5
    # strict JSON: parses without allow_nan, i.e. no Infinity/NaN tokens emitted
    assert json.loads(json.dumps(j, allow_nan=False))["ci"] is None


def test_jsonable_handles_enums_tuples_paths_and_configs(cfg):
    j = _jsonable({
        "algo": DecompositionAlgo.WEIGHTED_VORONOI,   # Enum -> value
        "dims": (1.0, 2.0, 3.0),                      # tuple -> list
        "p": Path("a/b"),                             # Path -> str
        "cfg_fleet": cfg.fleet,                       # dataclass -> dict
    })
    assert j["algo"] == "weighted_voronoi"
    assert j["dims"] == [1.0, 2.0, 3.0]
    assert j["p"] == "a/b"
    assert j["cfg_fleet"]["n_drones"] == 5
    # round-trips through json
    assert json.loads(json.dumps(j))["algo"] == "weighted_voronoi"


# --------------------------------------------------------------------------- #
# run / simulation directory structure + manifest                              #
# --------------------------------------------------------------------------- #
def test_run_creates_structured_dirs_and_manifest(tmp_path):
    run = RunContext(base_dir=str(tmp_path), name="run-FIXED")
    assert run.dir == tmp_path / "run-FIXED"
    assert run.dir.is_dir()
    assert len(run.run_id) == 12  # short GUID

    a = run.simulation("weighted")
    b = run.simulation("kmeans")
    assert a.dir == tmp_path / "run-FIXED" / "simulation-weighted"
    assert b.dir.is_dir()
    assert a.id != b.id  # distinct simulation GUIDs
    # artifacts resolve inside the simulation folder
    assert a.path("tracks.gpx") == a.dir / "tracks.gpx"

    a.write_plan({"identity": {"config_hash": "deadbeef"}})
    a.write_results({"status": "MISSION_SUCCESS"})
    b.write_plan({"identity": {"config_hash": "feedface"}})
    b.write_results({"status": "no_success"})

    manifest_path = run.finalize(summary={"experiment": "demo"})
    assert manifest_path == run.dir / "run.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema"] == RUN_SCHEMA
    assert manifest["run_name"] == "run-FIXED"
    assert manifest["run_id"] == run.run_id
    assert manifest["n_simulations"] == 2
    assert "wall_time_s" in manifest and "software" in manifest
    assert manifest["summary"]["experiment"] == "demo"
    sims = {s["name"]: s for s in manifest["simulations"]}
    assert sims["weighted"]["config_hash"] == "deadbeef"
    assert sims["weighted"]["status"] == "MISSION_SUCCESS"
    assert sims["kmeans"]["dir"] == "simulation-kmeans"
    # the two json files are physically present
    assert (a.dir / "plan.json").exists() and (a.dir / "results.json").exists()


# --------------------------------------------------------------------------- #
# plan.json                                                                    #
# --------------------------------------------------------------------------- #
def test_plan_has_setup_highlights_and_full_config(cfg):
    identity = {"run_id": "r", "simulation_id": "s", "config_hash": cfg.config_hash}
    plan = build_plan(cfg, identity=identity, algo=DecompositionAlgo.WEIGHTED_VORONOI,
                      planner=PlannerKind.DUBINS, engine=None)
    assert plan["schema"] == PLAN_SCHEMA
    s = plan["setup"]
    assert s["n_drones"] == 5
    assert s["drone_type"] in {"FIXED_WING", "MULTIROTOR", "VTOL"}
    assert s["battery_zones"]["critical"] == pytest.approx(0.20)
    assert s["decomposition_algorithm"] == "weighted_voronoi"
    assert s["energy_weighting_enabled"] is True
    # full config dump is present and JSON-safe
    assert plan["config"]["fleet"]["n_drones"] == 5
    json.dumps(plan)  # must not raise


def test_plan_energy_weighting_flag_tracks_algorithm(cfg):
    identity = {"config_hash": cfg.config_hash}
    for algo, expected in [
        (DecompositionAlgo.WEIGHTED_VORONOI, True),
        (DecompositionAlgo.KMEANS, False),
        (DecompositionAlgo.CLASSIC_VORONOI, False),
        (DecompositionAlgo.TGC_BASIC, False),
    ]:
        plan = build_plan(cfg, identity=identity, algo=algo, planner=PlannerKind.DUBINS)
        assert plan["setup"]["energy_weighting_enabled"] is expected


# --------------------------------------------------------------------------- #
# results.json (Monte-Carlo) + outcome counting                                #
# --------------------------------------------------------------------------- #
def _fake_mc():
    runs = [
        SingleRunResult([], {}, 0.5, outcome=Outcome.MISSION_SUCCESS),
        SingleRunResult([], {}, 0.6, outcome=Outcome.MISSION_SUCCESS),
        SingleRunResult([], {}, float("nan"), aborted=True, outcome=Outcome.MISSION_FAILED),
    ]
    return MCResult(
        n_runs=3, converged=True,
        pi_time_mean={AgentState.S2_MISSION: 0.5, AgentState.S3_RTH: 0.2},
        pi_time_ci={AgentState.S2_MISSION: 0.01},
        efficiency_mean=0.55, efficiency_ci=0.02, aborted_frac=1 / 3,
        convergence_trace=[(2, 0.5, 0.05), (3, 0.5, 0.008)], runs=runs,
    )


def test_outcome_counts_by_terminal_outcome():
    counts = _outcome_counts(_fake_mc().runs)
    assert counts["MISSION_SUCCESS"] == 2
    assert counts["MISSION_FAILED"] == 1
    assert counts["MISSION_INCOMPLETE"] == 0


def test_results_mc_reports_outcomes_smdp_stop_reason_and_timing():
    mc = _fake_mc()
    identity = {"run_id": "r", "config_hash": "abc"}
    res = build_results_mc(mc, identity=identity, wall_time_s=9.0)
    assert res["schema"] == RESULTS_SCHEMA and res["mode"] == "monte_carlo"

    assert res["monte_carlo"]["n_runs"] == 3
    assert res["monte_carlo"]["stop_reason"] == "ci_converged"
    assert res["monte_carlo"]["final_ci95_half_width"] == pytest.approx(0.008)

    o = res["outcomes"]
    assert o["n_success"] == 2 and o["n_failed"] == 1
    assert o["success_frac"] == pytest.approx(2 / 3)
    assert o["smdp_aborted_frac"] == pytest.approx(1 / 3)  # distinct from mission failure

    assert res["smdp"]["stationary_pi_time"]["S2_MISSION"] == pytest.approx(0.5)
    assert res["smdp"]["efficiency_mean"] == pytest.approx(0.55)

    assert res["timing"]["wall_time_total_s"] == pytest.approx(9.0)
    assert res["timing"]["wall_time_mean_per_run_s"] == pytest.approx(3.0)
    json.dumps(res)


def test_results_mc_stop_reason_when_not_converged():
    mc = _fake_mc()
    mc.converged = False
    res = build_results_mc(mc, identity={}, wall_time_s=1.0)
    assert res["monte_carlo"]["stop_reason"] == "reached_n_max"


# --------------------------------------------------------------------------- #
# results.json (single mission)                                                #
# --------------------------------------------------------------------------- #
class _FakeMetrics:
    total_energy_j = 1000.0
    duration_s = 200.0
    workload_std_m = 50.0
    n_swaps = 1
    n_failures = 0
    planning_time_s = 0.5
    per_agent_length_m = {0: 100.0, 1: 120.0}


class _FakeResult:
    metrics = _FakeMetrics()
    outcome = Outcome.MISSION_SUCCESS
    coverage_frac = 1.0
    aborted = False


class _FakeEst:
    ergodic = False
    states: list = []


def test_results_single_reports_outcome_metrics_and_timing():
    res = build_results_single(_FakeResult(), _FakeEst(),
                               identity={"config_hash": "x"}, wall_time_s=1.25)
    assert res["mode"] == "single_mission"
    assert res["status"] == "MISSION_SUCCESS"
    assert res["outcome"]["coverage_frac"] == 1.0
    assert res["metrics"]["total_energy_j"] == 1000.0
    assert res["metrics"]["per_agent_length_m"] == {"0": 100.0, "1": 120.0}
    assert res["smdp"]["ergodic"] is False
    assert res["timing"]["wall_time_total_s"] == pytest.approx(1.25)
    json.dumps(res)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
