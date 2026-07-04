"""Batch 6 tests: engine integration, comparison harness, visualization, CLIs."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import config_path

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.infrastructure import visualization as viz
from uav_swarm_sim.metrics.comparison import DECOMPOSITION_PEERS, compare_decomposition
from uav_swarm_sim.metrics.smdp_estimator import estimate
from uav_swarm_sim.metrics.stationary_distribution import stationary


def _smoke_cfg(**extra):
    ov = {
        "fleet.n_drones": 3,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "env.obstacle_size_range_m": [10.0, 30.0],
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "mc.n_max": 2,
        "mc.n_min": 2,
    }
    ov.update(extra)
    return load_config(config_path(), overrides=ov)


@pytest.fixture(scope="module")
def mission():
    cfg = _smoke_cfg()
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    return cfg, eng, result


# --------------------------------------------------------------------------- #
# engine integration                                                          #
# --------------------------------------------------------------------------- #
def test_engine_completes_with_efficiency(mission):
    cfg, eng, result = mission
    assert not result.aborted and result.coverage_frac > 0.99
    est = estimate(result.history, close_failure_loop=True)
    assert est.ergodic
    _, pi_time = stationary(est)
    assert abs(pi_time.sum() - 1.0) < 1e-9


def test_launch_site_scores_ranked(mission):
    cfg, eng, result = mission
    Js = [s.J for s in eng.site_scores]
    assert Js == sorted(Js)  # ascending J (best first)
    assert eng.launch_pose is not None


def test_failure_run_does_not_crash():
    # elevated lambda exercises kill + redistribution wiring
    cfg = _smoke_cfg(**{"failure.hazard_rate_per_hour": 50.0, "sim.max_timesteps": 4000})
    eng = SimulationEngine(cfg, RngFactory(7), 0, algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()  # should not raise even if some agents fail
    assert result.metrics.total_energy_j >= 0


# --------------------------------------------------------------------------- #
# comparison harness                                                          #
# --------------------------------------------------------------------------- #
def test_compare_decomposition_runs():
    cfg = _smoke_cfg()
    variants = compare_decomposition(cfg, RngFactory(cfg.sim.master_seed))
    labels = {v.label for v in variants}
    # the comparison must run exactly the declared peer set (now four: the three
    # position-based baselines + the battery-weighted contribution)
    assert labels == {algo.value for algo in DECOMPOSITION_PEERS}
    assert "kmeans" in labels
    for v in variants:
        assert v.mc.n_runs >= 2


def test_weighted_equals_tgc_basic_with_full_batteries():
    # with full initial batteries and no failures, weighted == tgc_basic (documented)
    cfg = _smoke_cfg()
    variants = {v.label: v for v in compare_decomposition(cfg, RngFactory(cfg.sim.master_seed))}
    w = variants["weighted_voronoi"].mean("total_energy_j")
    t = variants["tgc_basic"].mean("total_energy_j")
    assert w == pytest.approx(t, rel=1e-6)


# --------------------------------------------------------------------------- #
# visualization                                                               #
# --------------------------------------------------------------------------- #
def test_visualization_outputs(mission, tmp_path):
    cfg, eng, result = mission
    p1 = viz.plot_partition(eng.env, result.partition, eng.launch_pose, tmp_path / "part.png")
    p2 = viz.plot_state_gantt(result.history, tmp_path / "gantt.png")
    p3 = viz.plot_battery_traces(result.history, cfg.battery_zones, tmp_path / "batt.png")
    est = estimate(result.history)
    pe, pt = stationary(est)
    emb = {s: float(pe[i]) for i, s in enumerate(est.states)}
    tim = {s: float(pt[i]) for i, s in enumerate(est.states)}
    p4 = viz.plot_pi_bars(emb, tim, tmp_path / "pi.png")
    for p in (p1, p2, p3, p4):
        assert Path(p).exists() and Path(p).stat().st_size > 0


# --------------------------------------------------------------------------- #
# experiment CLIs                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_single_mission_cli(tmp_path):
    from uav_swarm_sim.experiments import run_single_mission
    # write a tiny complete config the CLI can load
    import yaml
    raw = yaml.safe_load(Path(config_path()).read_text())
    raw["fleet"]["n_drones"] = 3
    raw["fleet"]["battery_capacity_wh"] = 400.0
    raw["failure"]["hazard_rate_per_hour"] = 0.0
    raw["env"]["geojson_path"] = "data/areas/smoke_area.geojson"
    raw["env"]["obstacle_density_per_km2"] = 4.0
    raw["sim"]["dt_s"] = 1.0
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(raw))
    rc = run_single_mission.main([
        "--config", str(cfgp), "--base", str(tmp_path / "runs"),
        "--run-name", "run-test", "--name", "weighted",
    ])
    assert rc == 0
    sim_dir = tmp_path / "runs" / "run-test" / "simulation-weighted"
    # structured run/simulation layout: artifacts + both JSON logs + manifest
    assert (sim_dir / "partition.png").exists()
    assert (sim_dir / "plan.json").exists()
    assert (sim_dir / "results.json").exists()
    assert (tmp_path / "runs" / "run-test" / "run.json").exists()


def test_launch_site_cli(tmp_path):
    from uav_swarm_sim.experiments import run_launch_site_study
    import yaml
    raw = yaml.safe_load(Path(config_path()).read_text())
    raw["fleet"]["n_drones"] = 3
    raw["env"]["geojson_path"] = "data/areas/smoke_area.geojson"
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(raw))
    rc = run_launch_site_study.main(["--config", str(cfgp)])
    assert rc == 0
