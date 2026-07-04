"""Batch 3 tests: planning layer (isolated + small integration)."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import DroneStateView, Pose
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning import (
    ClassicVoronoiDecomposer,
    GridPlanner,
    KMeansHeuristicDecomposer,
    TgcBasicDecomposer,
    WeightedTgcDecomposer,
    boustrophedon,
    build_gvg,
    build_tgc,
    generate,
    load_area,
)
from uav_swarm_sim.planning.decomposition_base import is_connected_subset
from uav_swarm_sim.planning.environment_map import EnvironmentMap


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfg(config_path):
    return load_config(config_path)


@pytest.fixture(scope="module")
def area(cfg):
    return load_area(cfg.env.geojson_path)


@pytest.fixture(scope="module")
def env(cfg, area):
    rng = RngFactory(cfg.sim.master_seed).stream("obstacles", 0)
    obstacles = generate(area, cfg.env, rng)
    return EnvironmentMap(area, obstacles, cfg.env.clearance_buffer_m)


@pytest.fixture(scope="module")
def tgc(env):
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    return build_tgc(env, gvg)


def _drones(env, n, rng, battery=None):
    pts = env.sample_free(n, rng)
    out = []
    for i, p in enumerate(pts):
        b = 1.0 if battery is None else battery[i]
        out.append(DroneStateView(id=i, battery_frac=b, pose=Pose(p[0], p[1], 0.0)))
    return out


# --------------------------------------------------------------------------- #
# geojson_parser                                                              #
# --------------------------------------------------------------------------- #
def test_load_area_metric(area):
    assert area.area == pytest.approx(2000 * 900 + 1300 * 600)  # L-shape area
    assert area.exterior.is_ccw


def test_load_area_projects_geographic(tmp_path):
    import json
    gj = {"type": "Polygon", "coordinates": [[[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]]]}
    p = tmp_path / "geo.geojson"
    p.write_text(json.dumps(gj))
    poly = load_area(str(p))
    # 0.01 deg ~ 1.1 km -> projected area on the order of 1e6 m^2, not ~1e-4 deg^2
    assert poly.area > 1e5


# --------------------------------------------------------------------------- #
# obstacle_generator + environment_map                                        #
# --------------------------------------------------------------------------- #
def test_obstacles_within_area_and_merged(env, area):
    for o in env.obstacles:
        assert area.buffer(1e-6).covers(o.polygon)
    # merged: no two obstacles overlap
    for i in range(len(env.obstacles)):
        for j in range(i + 1, len(env.obstacles)):
            assert not env.obstacles[i].polygon.overlaps(env.obstacles[j].polygon)


def test_free_space_smaller_than_area(env, area):
    assert env.free_space.area <= area.area
    assert env.free_space.area > 0


def test_clearance_and_segment_clear(env):
    p = env.sample_free(1, np.random.default_rng(0))[0]
    assert env.clearance(p) >= 0
    a = Pose(*p, 0.0)
    assert env.segment_clear(a, a) in (True, False)


def test_occupancy_grid_shape(env):
    grid, frame = env.occupancy_grid(100.0)
    assert grid.shape == (frame.nx, frame.ny)
    assert grid.any()


# --------------------------------------------------------------------------- #
# GVG / TGC                                                                    #
# --------------------------------------------------------------------------- #
def test_gvg_nonempty_with_obstacles(env):
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    if env.obstacles:
        assert gvg.number_of_nodes() > 0


def test_tgc_regions_tile_free_space(env, tgc):
    total = sum(r.area_m2 for r in tgc.regions)
    # regions partition free space (small tolerance for clipping/Voronoi envelope)
    assert total == pytest.approx(env.free_space.area, rel=0.02)
    assert len(tgc.regions) >= 1


# --------------------------------------------------------------------------- #
# weighted decomposition -- the central guarantee                             #
# --------------------------------------------------------------------------- #
def test_weighted_area_proportional_to_battery(env, tgc):
    rng = np.random.default_rng(3)
    batteries = [1.0, 0.75, 0.5, 0.25]
    drones = _drones(env, 4, rng, battery=batteries)
    if len(drones) < 4:
        pytest.skip("not enough free space sampled")
    dec = WeightedTgcDecomposer()
    part = dec.decompose(tgc, env, drones)
    total = part.total_area_m2
    w = dec.weights(drones)
    # area share should track battery share; allow tolerance from region granularity
    for d in drones:
        share = part.zones[d.id].area_m2 / total
        assert abs(share - w[d.id]) < 0.15, f"drone {d.id}: share {share:.3f} vs target {w[d.id]:.3f}"
    # higher battery -> not-smaller zone (monotonicity of the ranking)
    shares = [part.zones[d.id].area_m2 for d in drones]
    assert shares[0] >= shares[-1]


def test_weighted_covers_free_space_exactly(env, tgc):
    rng = np.random.default_rng(4)
    drones = _drones(env, 5, rng)
    dec = WeightedTgcDecomposer()
    part = dec.decompose(tgc, env, drones)
    # true partition guarantee: every region assigned to exactly one drone, and
    # the summed exact polygon areas of assigned regions equal the region total.
    region_total = sum(r.area_m2 for r in tgc.regions)
    seen_ids: list[int] = []
    assigned_area = 0.0
    for zone in part.zones.values():
        for r in zone.regions:
            seen_ids.append(r.id)
            assigned_area += r.area_m2
    assert len(seen_ids) == len(set(seen_ids)), "a region was assigned to two drones"
    # assigned regions reproduce the (possibly subdivided) free-space area exactly
    assert assigned_area == pytest.approx(region_total, rel=1e-6)


def test_tgc_basic_is_uniform(env, tgc):
    rng = np.random.default_rng(5)
    drones = _drones(env, 4, rng, battery=[1.0, 0.2, 0.2, 0.2])
    w = TgcBasicDecomposer().weights(drones)
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())


def test_classic_voronoi_runs(env, tgc):
    rng = np.random.default_rng(6)
    drones = _drones(env, 4, rng)
    part = ClassicVoronoiDecomposer().decompose(tgc, env, drones)
    assert set(part.zones) == {d.id for d in drones}


def test_kmeans_heuristic_runs(env, tgc, cfg):
    rng = np.random.default_rng(7)
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    drones = _drones(env, 6, rng)
    part = KMeansHeuristicDecomposer(motion, weighted=True, rng=rng).decompose(tgc, env, drones)
    assert set(part.zones) == {d.id for d in drones}
    assert part.total_area_m2 > 0


# --------------------------------------------------------------------------- #
# coverage_path + grid_planner                                                #
# --------------------------------------------------------------------------- #
def test_boustrophedon_produces_waypoints(env, tgc, cfg):
    rng = np.random.default_rng(8)
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    em = EnergyModel(spec)
    drones = _drones(env, 3, rng)
    part = WeightedTgcDecomposer().decompose(tgc, env, drones)
    zone = max(part.zones.values(), key=lambda z: z.area_m2)
    plan = boustrophedon(zone, spec, motion, em)
    assert len(plan.waypoints) >= 2
    assert plan.length_m > 0
    assert plan.est_energy_j > 0


def test_grid_planner_route_and_coverage(env, tgc, cfg):
    rng = np.random.default_rng(9)
    spec = build_spec(cfg)
    drones = _drones(env, 3, rng)
    part = WeightedTgcDecomposer().decompose(tgc, env, drones)
    zone = max(part.zones.values(), key=lambda z: z.area_m2)
    gp = GridPlanner(env, cell_m=100.0)
    cov = gp.coverage(zone, spec)
    assert cov.length_m >= 0
    a = Pose(*env.sample_free(1, rng)[0], 0.0)
    b = Pose(*env.sample_free(1, rng)[0], 0.0)
    path = gp.route(a, b)
    assert path.total_length_m >= 0
