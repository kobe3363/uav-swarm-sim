"""EM-01 Stage 1 -- the energy cost-to-go map builder (planning/energy_map.py).

Pure-builder tests on tiny synthetic environments (no full mission):

  * empty env: E_home is EXACTLY the octile (8-connected) cruise cost, bounded
    below by the straight-line cruise cost; parent chains reach the base;
  * a wall between base and target forces route_home AROUND it (never through
    a red cell) and E_home above the straight-line cost;
  * hand-checked occupancy fractions -> free / yellow / red classification;
  * an obstacle-ringed pocket is unreachable (E_home=inf, parent=-1);
  * a >=90%-covered base cell still builds (warning, never an error) with
    E_home[base]=0 and neighbors converging to it -- the author's Stage-1 rule
    replacing the original base-in-obstacle ValueError;
  * bitwise-deterministic rebuilds;
  * the energy-unit guard: edges are CRUISE, never COVERAGE;
  * base-anchored grid origin: the launch pose sits at its cell CENTER.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.planning.energy_map import (
    battery_tied_cell_m,
    build_energy_map,
    route_home,
)
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle

AREA = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
CELL = 20.0


@pytest.fixture(scope="module")
def kit(config_path):
    cfg = load_config(config_path)
    spec = build_spec(cfg)
    return spec, EnergyModel(spec)


def _build(env, base, kit, **kw):
    spec, em = kit
    return build_energy_map(env, base, CELL, em, spec.v_cruise, **kw)


def _cruise(kit, dist_m):
    spec, em = kit
    return em.distance_energy(dist_m, ManeuverType.CRUISE, spec.v_cruise)


# --------------------------------------------------------------------------- #
# 1. empty env: octile-exact cruise cost, straight-line lower bound            #
# --------------------------------------------------------------------------- #
def test_empty_env_e_home_is_octile_cruise_cost(kit):
    env = EnvironmentMap(AREA, [], 5.0)
    base = Pose(500.0, 500.0, 0.0)
    emap = _build(env, base, kit)
    bi, bj = emap.frame.world_to_cell(base.x, base.y)

    # exact along a principal axis and along a diagonal
    assert emap.e_home[bi, bj] == 0.0
    assert emap.e_home[bi + 10, bj] == pytest.approx(_cruise(kit, 10 * CELL), rel=1e-12)
    assert emap.e_home[bi + 5, bj + 5] == pytest.approx(
        _cruise(kit, 5 * math.sqrt(2.0) * CELL), rel=1e-12
    )

    # every cell: E_home == octile distance x cruise J/m, and >= euclid cost
    nx, ny = emap.frame.nx, emap.frame.ny
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx, dy = np.abs(ii - bi), np.abs(jj - bj)
    octile_m = (np.maximum(dx, dy) - np.minimum(dx, dy) + np.minimum(dx, dy) * math.sqrt(2.0)) * CELL
    euclid_m = np.hypot(dx, dy) * CELL
    assert np.allclose(emap.e_home, _cruise(kit, 1.0) * octile_m, rtol=1e-9)
    assert np.all(emap.e_home >= _cruise(kit, 1.0) * euclid_m - 1e-9)

    # parent chain from the far corner walks all the way to the base center
    waypoints = route_home(emap, Pose(995.0, 995.0, 0.0))
    assert len(waypoints) >= 2
    assert waypoints[-1].as_xy() == pytest.approx((base.x, base.y), abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. wall between base and target -> around, never through                     #
# --------------------------------------------------------------------------- #
def test_wall_forces_route_around_and_raises_cost(kit):
    wall = Obstacle(id=0, cls=0, polygon=box(480.0, 0.0, 520.0, 800.0))
    env = EnvironmentMap(AREA, [wall], 5.0)
    base, target = Pose(250.0, 500.0, 0.0), Pose(750.0, 500.0, 0.0)
    emap = _build(env, base, kit)

    ti, tj = emap.frame.world_to_cell(target.x, target.y)
    straight = _cruise(kit, math.dist(base.as_xy(), target.as_xy()))
    assert np.isfinite(emap.e_home[ti, tj])
    assert emap.e_home[ti, tj] > straight  # the detour is strictly costlier

    waypoints = route_home(emap, target)
    assert waypoints[-1].as_xy() == pytest.approx((base.x, base.y), abs=1e-6)
    for wp in waypoints:  # never enters a red (blocked) cell
        i, j = emap.frame.world_to_cell(wp.x, wp.y)
        assert np.isfinite(emap.penalty[i, j])
    # it actually went around: some waypoint clears the wall's top (y > 800)
    assert max(wp.y for wp in waypoints) > 800.0


# --------------------------------------------------------------------------- #
# 3. occupancy fraction -> class (hand-checked; buffer 0 for exact fractions)  #
# --------------------------------------------------------------------------- #
def test_occupancy_fraction_classes(kit):
    half = Obstacle(id=0, cls=0, polygon=box(240.0, 200.0, 260.0, 210.0))   # 200 m2
    quarter = Obstacle(id=1, cls=0, polygon=box(340.0, 200.0, 350.0, 210.0))  # 100 m2
    env = EnvironmentMap(AREA, [half, quarter], 0.0)
    # base at (510, 510) anchors the lattice on multiples of 20 m (base cell
    # lower-left = 500 => origin = -20), so the hand-placed cells below are
    # exactly [240,260)x[200,220) and [340,360)x[200,220).
    base = Pose(510.0, 510.0, 0.0)
    emap = _build(env, base, kit, yellow_penalty=1.5, red_threshold=0.5)
    frame = emap.frame

    # cell [240, 260) x [200, 220): 20x10 covered -> frac 0.5 -> red (blocked)
    i, j = frame.world_to_cell(250.0, 210.0)
    assert not np.isfinite(emap.penalty[i, j])
    # cell [340, 360) x [200, 220): 10x10 covered -> frac 0.25 -> yellow
    i, j = frame.world_to_cell(350.0, 210.0)
    assert emap.penalty[i, j] == 1.5
    # untouched cell -> free
    i, j = frame.world_to_cell(700.0, 700.0)
    assert emap.penalty[i, j] == 1.0


# --------------------------------------------------------------------------- #
# 4. obstacle-ringed pocket -> unreachable; route_home refuses                 #
# --------------------------------------------------------------------------- #
def test_unreachable_pocket_inf_and_route_home_raises(kit):
    # 100x100 block with a one-cell (20x20) hole at [340,360)^2: the ring of
    # fully covered cells around the hole is red, the hole itself is free.
    ring = Obstacle(
        id=0, cls=0,
        polygon=Polygon(
            box(300.0, 300.0, 400.0, 400.0).exterior.coords,
            [box(340.0, 340.0, 360.0, 360.0).exterior.coords],
        ),
    )
    env = EnvironmentMap(AREA, [ring], 0.0)
    base = Pose(50.0, 50.0, 0.0)
    emap = _build(env, base, kit)

    i, j = emap.frame.world_to_cell(350.0, 350.0)
    assert emap.penalty[i, j] == 1.0            # the pocket itself is free...
    assert emap.e_home[i, j] == np.inf          # ...but no route to base
    assert emap.parent[i, j] == -1
    with pytest.raises(ValueError, match="unreachable"):
        route_home(emap, Pose(350.0, 350.0, 0.0))
    with pytest.raises(ValueError, match="outside"):
        route_home(emap, Pose(1e6, 1e6, 0.0))


# --------------------------------------------------------------------------- #
# 5. base cell 90% covered: builds with a warning, E_home[base]=0, converges   #
#    (author's Stage-1 rule -- replaces the original ValueError)               #
# --------------------------------------------------------------------------- #
def test_obstructed_base_cell_forced_traversable(kit, caplog):
    base = Pose(500.0, 500.0, 0.0)  # base cell = [490, 510)^2
    blocker = Obstacle(id=0, cls=0, polygon=box(490.0, 490.0, 508.0, 510.0))  # 90%
    env = EnvironmentMap(AREA, [blocker], 0.0)
    with caplog.at_level(logging.WARNING, logger="uav_swarm_sim.planning.energy_map"):
        emap = _build(env, base, kit)
    assert any("base" in r.message for r in caplog.records)

    bi, bj = emap.frame.world_to_cell(base.x, base.y)
    assert emap.penalty[bi, bj] == 1.0
    assert emap.e_home[bi, bj] == 0.0
    assert emap.parent[bi, bj] == -1
    # a free neighbor converges to the base
    waypoints = route_home(emap, Pose(560.0, 500.0, 0.0))
    assert waypoints[-1].as_xy() == pytest.approx((base.x, base.y), abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. determinism: identical inputs -> bitwise-identical arrays                 #
# --------------------------------------------------------------------------- #
def test_rebuild_is_bitwise_identical(kit):
    obs = [
        Obstacle(id=0, cls=0, polygon=box(300.0, 300.0, 380.0, 420.0)),
        Obstacle(id=1, cls=0, polygon=box(600.0, 100.0, 660.0, 700.0)),
    ]
    env = EnvironmentMap(AREA, obs, 5.0)
    base = Pose(120.0, 880.0, 0.0)
    a = _build(env, base, kit)
    b = _build(env, base, kit)
    assert np.array_equal(a.e_home, b.e_home)     # bitwise (inf-safe)
    assert np.array_equal(a.parent, b.parent)
    assert np.array_equal(a.penalty, b.penalty)
    assert a.frame == b.frame


# --------------------------------------------------------------------------- #
# 7. energy-unit guard: edges are CRUISE, never COVERAGE                       #
# --------------------------------------------------------------------------- #
def test_edge_energy_is_cruise_not_coverage(kit):
    spec, em = kit
    env = EnvironmentMap(AREA, [], 5.0)
    base = Pose(500.0, 500.0, 0.0)
    emap = _build(env, base, kit)
    bi, bj = emap.frame.world_to_cell(base.x, base.y)
    one_hop = emap.e_home[bi + 1, bj]

    assert one_hop == pytest.approx(
        em.distance_energy(CELL, ManeuverType.CRUISE, spec.v_cruise), rel=1e-12
    )
    for wrong in (
        em.distance_energy(CELL, ManeuverType.COVERAGE, spec.v_coverage),
        em.distance_energy(CELL, ManeuverType.COVERAGE, spec.v_cruise),
    ):
        assert one_hop != pytest.approx(wrong, rel=1e-6)


# --------------------------------------------------------------------------- #
# 8. base-anchored origin: the launch pose sits at its cell CENTER             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base_xy", [(500.0, 500.0), (3.0, 997.0), (997.0, 3.0)])
def test_grid_origin_centers_base_in_its_cell(kit, base_xy):
    env = EnvironmentMap(AREA, [], 5.0)
    base = Pose(base_xy[0], base_xy[1], 0.0)
    emap = _build(env, base, kit)
    frame = emap.frame

    bi, bj = frame.world_to_cell(base.x, base.y)
    assert frame.cell_center(bi, bj) == pytest.approx((base.x, base.y), abs=1e-9)
    assert emap.e_home[bi, bj] == 0.0
    assert (bi, bj) == emap.base_cell
    # the grid covers the survey area's bounds
    minx, miny, maxx, maxy = env.area.bounds
    assert frame.origin_x <= minx and frame.origin_y <= miny
    assert frame.origin_x + frame.nx * frame.cell_m >= maxx
    assert frame.origin_y + frame.ny * frame.cell_m >= maxy


# --------------------------------------------------------------------------- #
# 9. battery-tied cell size (design doc section 3)                             #
# --------------------------------------------------------------------------- #
def test_battery_tied_cell_m_default_arithmetic():
    # 360 kJ x 12 m/s / 220 W / 1000 = 19.6363... m (~20 m, doc section 3)
    assert battery_tied_cell_m(360_000.0, 220.0, 12.0) == pytest.approx(19.6363, abs=1e-3)
    with pytest.raises(ValueError):
        battery_tied_cell_m(0.0, 220.0, 12.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
