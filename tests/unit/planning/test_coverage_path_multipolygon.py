"""EXP-02 plans every disjoint zone component with camera-off connectors."""
from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon, Point, box

import uav_swarm_sim.planning.coverage_path as coverage_path
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose, Zone
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.coverage_raster import CoverageRaster
from uav_swarm_sim.planning.environment_map import EnvironmentMap


def test_boustrophedon_plans_all_components_and_routes_the_gap(config_path, monkeypatch):
    cfg = load_config(config_path)
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    energy = EnergyModel(spec)
    left = box(0.0, 0.0, 100.0, 10.0)
    right = box(200.0, 0.0, 300.0, 10.0)
    zone = Zone(0, [], MultiPolygon([left, right]), Pose(0.0, 0.0, 0.0))
    calls = []

    def route(a, b, motion_model, env, **kwargs):
        calls.append(kwargs)
        return motion_model.plan(a, b, ManeuverType.TURN)

    monkeypatch.setattr(coverage_path, "route_connector", route)
    plan = coverage_path.boustrophedon(
        zone, spec, motion, energy, env=object(), coverage=cfg.coverage
    )

    assert len(plan.waypoints) == 4
    assert left.covers(Point(plan.waypoints[0].pose.as_xy()))
    assert left.covers(Point(plan.waypoints[1].pose.as_xy()))
    assert right.covers(Point(plan.waypoints[2].pose.as_xy()))
    assert right.covers(Point(plan.waypoints[3].pose.as_xy()))
    assert len(plan.connectors) == 1
    assert calls == [{
        "enabled": True,
        "operating_area": cfg.coverage.operating_area,
        "margin_m": cfg.coverage.operating_margin_m,
    }]


def test_disjoint_components_are_credited_only_after_each_strip_is_flown(config_path):
    cfg = load_config(config_path)
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    energy = EnergyModel(spec)
    left = box(0.0, 0.0, 100.0, 10.0)
    right = box(200.0, 0.0, 300.0, 10.0)
    area = MultiPolygon([left, right])
    plan = coverage_path.boustrophedon(
        Zone(0, [], area, Pose(0.0, 0.0, 0.0)),
        spec,
        motion,
        energy,
        env=EnvironmentMap(area, [], cfg.env.clearance_buffer_m),
        coverage=cfg.coverage,
    )
    raster = CoverageRaster(area, area, 10.0)

    raster.record_segment(plan.waypoints[0].pose, plan.waypoints[1].pose, 10.0, 10.0)
    assert raster.target_coverage_frac == pytest.approx(0.5)

    raster.record_segment(plan.waypoints[2].pose, plan.waypoints[3].pose, 10.0, 10.0)
    assert raster.target_coverage_frac == pytest.approx(1.0)
