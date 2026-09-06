"""EXP-05 structured-output provenance tests."""
from __future__ import annotations

from types import SimpleNamespace

from uav_swarm_sim.infrastructure.enums import Outcome
from uav_swarm_sim.metrics.monte_carlo import MCResult, SingleRunResult
from uav_swarm_sim.metrics.run_output import (
    PLAN_SCHEMA,
    RESULTS_SCHEMA,
    build_plan,
    build_results_mc,
    build_results_single,
)


def test_plan_schema_v2_records_initial_soc_config(config_path):
    from uav_swarm_sim.infrastructure.config import load_config

    cfg = load_config(config_path, overrides={
        "battery.initial_soc.mode": "uniform",
        "battery.initial_soc.low": 0.8,
        "battery.initial_soc.high": 1.0,
    })
    plan = build_plan(cfg, identity={}, algo="tgc_basic", planner="dubins")
    assert PLAN_SCHEMA == "uav-swarm-sim/plan/v2"
    assert plan["config"]["battery"]["initial_soc"] == {
        "mode": "uniform",
        "value": None,
        "low": 0.8,
        "high": 1.0,
        "mean": None,
        "std": None,
    }


def test_single_results_v2_records_actual_vector():
    metrics = SimpleNamespace(
        total_energy_j=1.0,
        duration_s=2.0,
        workload_std_m=3.0,
        n_swaps=0,
        n_failures=0,
        planning_time_s=0.1,
        per_agent_length_m={0: 4.0, 1: 5.0},
    )
    result = SimpleNamespace(
        metrics=metrics,
        outcome=Outcome.MISSION_SUCCESS,
        coverage_frac=1.0,
        aborted=False,
        initial_soc_by_drone=(0.81, 0.93),
    )
    est = SimpleNamespace(ergodic=False, states=[])
    output = build_results_single(result, est, identity={}, wall_time_s=0.2)
    assert RESULTS_SCHEMA == "uav-swarm-sim/results/v2"
    assert output["initial_soc_by_drone"] == [0.81, 0.93]


def test_mc_results_v2_records_each_replication_vector():
    runs = [
        SingleRunResult([], {}, 0.0, initial_soc_by_drone=(0.8, 0.9)),
        SingleRunResult([], {}, 0.0, initial_soc_by_drone=(0.85, 0.95)),
    ]
    mc = MCResult(
        n_runs=2,
        converged=False,
        pi_time_mean={},
        pi_time_ci={},
        efficiency_mean=0.0,
        efficiency_ci=0.0,
        aborted_frac=0.0,
        convergence_trace=[],
        runs=runs,
    )
    output = build_results_mc(mc, identity={}, wall_time_s=0.2)
    assert output["initial_soc_by_replication"] == [
        {"replication": 1, "initial_soc_by_drone": [0.8, 0.9]},
        {"replication": 2, "initial_soc_by_drone": [0.85, 0.95]},
    ]
