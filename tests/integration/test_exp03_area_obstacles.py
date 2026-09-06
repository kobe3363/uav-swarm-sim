from __future__ import annotations

import pytest
from shapely.geometry import Point
from shapely.ops import unary_union

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.planning.geojson_parser import load_area


_AREA_PATH = "data/areas/exp03_1000x750.geojson"


def test_exp03_geojson_has_exact_metric_area():
    area = load_area(_AREA_PATH)

    assert area.bounds == pytest.approx((0.0, 0.0, 1000.0, 750.0))
    assert area.area == pytest.approx(1000.0 * 750.0, abs=1e-9)


def test_target_mode_builds_engine_with_feasible_launch(config_path):
    cfg = load_config(
        config_path,
        overrides={
            "env.geojson_path": _AREA_PATH,
            "env.obstacle_generation_mode": "target",
            "env.obstacle_target_count": 10,
            "env.obstacle_area_fraction": 0.05,
            "env.obstacle_area_fraction_tolerance": 0.005,
            "fleet.n_drones": 2,
            "fleet.battery_capacity_wh": 2000.0,
            "failure.hazard_rate_per_hour": 0.0,
            "launch.candidate_sites": 4,
        },
    )
    engine = SimulationEngine(
        cfg,
        RngFactory(cfg.sim.master_seed),
        replication=0,
        algo=DecompositionAlgo.WEIGHTED_VORONOI,
    )
    engine._build()

    raw = unary_union([o.polygon for o in engine.env.obstacles])
    launch = Point(engine.launch_pose.x, engine.launch_pose.y)
    assert len(engine.env.obstacles) == 10
    assert raw.area == pytest.approx(37_500.0, abs=3_750.0)
    assert engine.env.free_space.is_valid and not engine.env.free_space.is_empty
    assert not engine.env.area.covers(launch)
    assert not raw.covers(launch)
    assert len(engine.deploy_poses) == cfg.fleet.n_drones
