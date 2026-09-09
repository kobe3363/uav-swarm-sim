"""REV-01 real M4E diagnostic: pass the exact former planning failure seam."""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Point

import uav_swarm_sim.planning.energy_balance as balance
from uav_swarm_sim.experiments.run_lloyd_protocol import build_cfg
from uav_swarm_sim.infrastructure.core_types import DecompositionAlgo
from uav_swarm_sim.infrastructure.enums import PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.planning.visibility_router import _path_clear, flyable_region


DIAGNOSTIC_OVERRIDES = {
    "env.geojson_path": "data/areas/exp03_1000x750.geojson",
    "env.coverage_altitude_m": 100.0,
    "platforms.MULTIROTOR.v_cruise": 10.0,
    "platforms.MULTIROTOR.v_coverage": 10.0,
    "sensor.photogrammetry.enabled": True,
    "sensor.photogrammetry.side_overlap": 0.70,
    "sensor.photogrammetry.forward_overlap": 0.80,
    "env.obstacle_generation_mode": "target",
    "env.obstacle_target_count": 10,
    "env.obstacle_area_fraction": 0.05,
    "coverage.raster_enabled": True,
    "coverage.raster_cell_m": 10.0,
    "planning.energy_balance.enabled": True,
    "sim.max_timesteps": 4000,
    "mission.no_swap_mode": True,
    "mission.repartition_enabled": False,
    "coverage.transit_free_space": True,
    "coverage.ferry_free_space": True,
    "rth.execution_coherent": True,
    "rth.energy_map.zone_demotion": True,
    "rth.emergency_frac": None,
    "safety.record_violations": True,
    "failure.hazard_rate_per_hour": 0.0,
}


class _PassedFormerFailure(RuntimeError):
    pass


def test_m4e_seed42_planning_passes_the_former_blocked_centroid(monkeypatch):
    cfg = build_cfg("config/djimatrice4e.yaml", 5, DIAGNOSTIC_OVERRIDES)
    engine = SimulationEngine(
        cfg,
        RngFactory(cfg.sim.master_seed),
        replication=1,
        algo=DecompositionAlgo.LLOYD_ENERGY,
        planner=PlannerKind.DUBINS,
    )
    real_estimate = balance.estimate_fast_from_area
    evidence = {}

    def observe(context, drone, **kwargs):
        centroid_xy = tuple(map(float, kwargs["centroid_xy"]))
        blocked_centroid = (
            kwargs["area_m2"] > 0.0
            and engine.env.in_obstacle(centroid_xy)
        )
        estimate = real_estimate(context, drone, **kwargs)
        if blocked_centroid:
            resolved = balance.resolve_reachable_zone_entry(
                context,
                drone,
                kwargs["work_geometry"],
                centroid_xy,
                kwargs["fallback_pose"],
            )
            region = flyable_region(
                engine.env.area,
                engine.env.buffered_obstacles,
                cfg.coverage.operating_area,
                cfg.coverage.operating_margin_m,
            )
            evidence.update(
                centroid_xy=centroid_xy,
                estimate=estimate,
                work_geometry=kwargs["work_geometry"],
                resolved=resolved,
                region=region,
            )
            raise _PassedFormerFailure
        return estimate

    monkeypatch.setattr(balance, "estimate_fast_from_area", observe)
    with pytest.raises(_PassedFormerFailure):
        engine._build()

    centroid_xy = evidence["centroid_xy"]
    estimate = evidence["estimate"]
    work_geometry = evidence["work_geometry"]
    resolved = evidence["resolved"]
    region = evidence["region"]

    assert centroid_xy == pytest.approx((871.9305418, 673.1083417), abs=1e-6)
    assert engine.env.in_obstacle(centroid_xy)
    assert not region.buffer(1e-8).covers(Point(centroid_xy))
    assert estimate.anchor_pose.as_xy() != pytest.approx(centroid_xy)
    assert work_geometry.covers(Point(estimate.anchor_pose.as_xy()))
    assert region.buffer(1e-8).covers(Point(estimate.anchor_pose.as_xy()))
    assert estimate.anchor_pose == resolved.anchor_pose
    assert _path_clear(resolved.ferry_path, engine.env, region=region)
    assert math.isfinite(estimate.e_ferry_j)
    assert estimate.e_ferry_j == pytest.approx(
        context_energy := engine.em.path_energy(resolved.ferry_path), rel=1e-12,
    )
    assert math.isfinite(context_energy)
