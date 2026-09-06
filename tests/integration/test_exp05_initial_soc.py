"""EXP-05 engine wiring, paired-seed, and legacy-identity tests."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
import uav_swarm_sim.infrastructure.simulation_engine as engine_module
from uav_swarm_sim.planning.launch_site_optimizer import InfeasibleMissionError


def _cfg(config_path, **extra):
    overrides = {
        "fleet.n_drones": 3,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "env.obstacle_size_range_m": [10.0, 30.0],
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
    }
    overrides.update(extra)
    return load_config(config_path, overrides=overrides)


def _engine(cfg, algo=DecompositionAlgo.WEIGHTED_VORONOI, replication=4):
    return SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), replication=replication, algo=algo
    )


def _run_signature(result):
    metrics = result.metrics
    return (
        metrics.total_energy_j,
        metrics.duration_s,
        metrics.workload_std_m,
        metrics.n_swaps,
        metrics.n_failures,
        tuple(sorted(metrics.per_agent_energy_j.items())),
        tuple(sorted(metrics.per_agent_length_m.items())),
        tuple(result.history.sojourns()),
        result.outcome,
        result.coverage_frac,
        result.stalled_agents,
        result.skipped_legs,
    )


def test_same_vector_reaches_planner_views_and_physical_batteries(config_path, monkeypatch):
    cfg = _cfg(
        config_path,
        **{
            "battery.initial_soc.mode": "uniform",
            "battery.initial_soc.low": 0.72,
            "battery.initial_soc.high": 0.94,
        },
    )
    real_view = engine_module.DroneStateView
    real_optimize_launch = engine_module.optimize_launch
    planning_soc = {}
    launch_soc = []

    def capture_view(drone_id, battery_frac, pose, layer=0):
        planning_soc[drone_id] = battery_frac
        return real_view(drone_id, battery_frac, pose, layer)

    def capture_optimize_launch(*args, **kwargs):
        launch_soc.extend(kwargs["initial_soc_by_drone"])
        return real_optimize_launch(*args, **kwargs)

    monkeypatch.setattr(engine_module, "DroneStateView", capture_view)
    monkeypatch.setattr(engine_module, "optimize_launch", capture_optimize_launch)
    eng = _engine(cfg)
    eng._build()

    expected = eng.initial_soc_by_drone
    assert tuple(launch_soc) == pytest.approx(expected, abs=0.0)
    assert tuple(planning_soc[i] for i in range(3)) == pytest.approx(expected, abs=0.0)
    physical = tuple(eng.fleet.agents[i].battery.frac for i in range(3))
    assert physical == pytest.approx(expected, abs=1e-15)


def test_explicit_fixed_full_charge_matches_legacy_run_exactly(config_path):
    legacy = _cfg(config_path)
    explicit = _cfg(
        config_path,
        **{"battery.initial_soc.mode": "fixed", "battery.initial_soc.value": 1.0},
    )

    legacy_result = _engine(legacy, replication=0).run()
    explicit_result = _engine(explicit, replication=0).run()

    assert legacy_result.initial_soc_by_drone == (1.0, 1.0, 1.0)
    assert explicit_result.initial_soc_by_drone == legacy_result.initial_soc_by_drone
    assert _run_signature(explicit_result) == _run_signature(legacy_result)


def test_random_soc_is_paired_across_algorithms_without_world_drift(config_path):
    cfg = _cfg(
        config_path,
        **{
            "battery.initial_soc.mode": "truncated_normal",
            "battery.initial_soc.low": 0.75,
            "battery.initial_soc.high": 1.0,
            "battery.initial_soc.mean": 0.9,
            "battery.initial_soc.std": 0.06,
        },
    )
    weighted = _engine(cfg, DecompositionAlgo.WEIGHTED_VORONOI)
    classic = _engine(cfg, DecompositionAlgo.CLASSIC_VORONOI)
    weighted._build()
    classic._build()

    assert weighted.initial_soc_by_drone == classic.initial_soc_by_drone
    assert [o.polygon.wkt for o in weighted.env.obstacles] == [
        o.polygon.wkt for o in classic.env.obstacles
    ]
    assert weighted.launch_pose == classic.launch_pose


def test_another_fixed_value_is_used_by_all_batteries(config_path):
    cfg = _cfg(
        config_path,
        **{"battery.initial_soc.mode": "fixed", "battery.initial_soc.value": 0.63},
    )
    eng = _engine(cfg)
    eng._build()
    assert eng.initial_soc_by_drone == pytest.approx((0.63, 0.63, 0.63), abs=0.0)
    assert tuple(eng.fleet.agents[i].battery.frac for i in range(3)) == pytest.approx(
        (0.63, 0.63, 0.63), abs=1e-15
    )


def test_zero_initial_soc_is_valid_config_but_cannot_launch(config_path):
    cfg = _cfg(
        config_path,
        **{"battery.initial_soc.mode": "fixed", "battery.initial_soc.value": 0.0},
    )
    with pytest.raises(InfeasibleMissionError, match="usable 0 J"):
        _engine(cfg)._build()
