"""EXP-02 plans every disjoint zone component with camera-off connectors."""
from __future__ import annotations

from dataclasses import replace

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


def test_ensure_one_moves_boundary_scanline_inside_component():
    rows = coverage_path._strip_intervals(
        box(0.0, 0.0, 100.0, 25.0), 50.0, ensure_one=True
    )

    assert len(rows) == 1
    assert rows[0][0] == pytest.approx((0.0, 100.0, 12.5))


def test_raster_mode_plans_a_thin_single_polygon(config_path):
    cfg = load_config(config_path)
    spec = build_spec(cfg)
    area = box(0.0, 0.0, 100.0, 10.0)

    plan = coverage_path.boustrophedon(
        Zone(0, [], area, Pose(0.0, 0.0, 0.0)),
        spec,
        make_motion_model(spec),
        EnergyModel(spec),
        coverage=replace(cfg.coverage, raster_enabled=True),
    )

    assert len(plan.waypoints) == 2
    assert plan.waypoints[0].pose.y == pytest.approx(5.0)
    assert plan.waypoints[1].pose.y == pytest.approx(5.0)


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
    legacy_single = coverage_path.boustrophedon(
        Zone(0, [], left, Pose(0.0, 0.0, 0.0)), spec, motion, energy
    )
    legacy_multi = coverage_path.boustrophedon(
        zone, spec, motion, energy, env=object(), coverage=cfg.coverage
    )
    raster_coverage = replace(cfg.coverage, raster_enabled=True)
    plan = coverage_path.boustrophedon(
        zone, spec, motion, energy, env=object(), coverage=raster_coverage
    )

    assert legacy_single.waypoints == []
    assert legacy_multi.waypoints == legacy_single.waypoints
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
        coverage=replace(cfg.coverage, raster_enabled=True),
    )
    raster = CoverageRaster(area, area, 10.0)

    raster.record_segment(plan.waypoints[0].pose, plan.waypoints[1].pose, 10.0, 10.0)
    assert raster.target_coverage_frac == pytest.approx(0.5)

    raster.record_segment(plan.waypoints[2].pose, plan.waypoints[3].pose, 10.0, 10.0)
    assert raster.target_coverage_frac == pytest.approx(1.0)
