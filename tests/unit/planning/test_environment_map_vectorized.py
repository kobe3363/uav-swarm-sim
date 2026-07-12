"""Byte-identity guard for the vectorised EnvironmentMap.first_obstruction.

first_obstruction is the RTH-lookahead hot path (~63% of mission runtime): it was
rewritten from a per-pose Python loop to Shapely 2.x batch predicates. Because
its boolean/arc-length result feeds return_energy -> the RTH decision -> every
metric, the vectorised result MUST equal the original scalar loop exactly. This
test pins that: a reference scalar implementation (the pre-change loop, inlined)
vs the shipped vectorised method, over many random straight paths against
obstacle-ful and obstacle-free maps, plus the edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from uav_swarm_sim.infrastructure.core_types import Path, Pose, straight_segment
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle


def _ref_first_obstruction(env, path, step_m=2.0):
    """The ORIGINAL scalar loop (pre-vectorisation), verbatim -- the reference."""
    poses = path.sample(step_m)
    if not poses:
        return None
    if not env.free_space.covers(Point(poses[0].as_xy())):
        return 0.0
    for i in range(1, len(poses)):
        a, b = poses[i - 1], poses[i]
        if not env.free_space.covers(Point(b.as_xy())) or not env.segment_clear(a, b):
            return i * step_m
    return None


def _rect(cx, cy, hw, hh):
    return Polygon([(cx - hw, cy - hh), (cx + hw, cy - hh),
                    (cx + hw, cy + hh), (cx - hw, cy + hh)])


def _straight_path(x, y, heading, length):
    seg = straight_segment(Pose(x, y, heading), length, ManeuverType.CRUISE, 10.0)
    return Path.from_segments([seg])


@pytest.mark.parametrize("with_obstacles", [True, False])
def test_first_obstruction_vectorised_matches_scalar(with_obstacles):
    area = _rect(100, 100, 100, 100)  # 200x200 square centred at (100,100)
    obstacles = []
    if with_obstacles:
        obstacles = [
            Obstacle(id=0, cls=0, polygon=_rect(70, 70, 20, 15)),
            Obstacle(id=1, cls=0, polygon=_rect(140, 120, 15, 25)),
            Obstacle(id=2, cls=0, polygon=_rect(100, 160, 30, 10)),
        ]
    env = EnvironmentMap(area, obstacles, buffer_m=5.0)

    rng = np.random.default_rng(1234)
    for _ in range(600):
        x, y = rng.uniform(-20, 220, size=2)
        heading = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(5.0, 300.0)
        path = _straight_path(float(x), float(y), float(heading), float(length))
        assert env.first_obstruction(path) == _ref_first_obstruction(env, path), (
            f"drift at start=({x:.1f},{y:.1f}) h={heading:.2f} L={length:.1f}")


def test_first_obstruction_edge_cases():
    area = _rect(100, 100, 100, 100)
    env = EnvironmentMap(area, [Obstacle(id=0, cls=0, polygon=_rect(100, 100, 20, 20))],
                         buffer_m=5.0)
    # empty path -> None
    assert env.first_obstruction(Path.from_segments([])) is None
    # zero-length path (single sample) starting inside -> None
    zero = _straight_path(30.0, 30.0, 0.0, 0.0)
    assert env.first_obstruction(zero) == _ref_first_obstruction(env, zero)
    # first pose OUTSIDE the area -> 0.0
    outside = _straight_path(-50.0, -50.0, 0.0, 40.0)
    assert env.first_obstruction(outside) == 0.0
    assert _ref_first_obstruction(env, outside) == 0.0
    # fully clear path in the corner -> None (both agree)
    clear = _straight_path(20.0, 20.0, 0.0, 30.0)
    assert env.first_obstruction(clear) is None
    assert _ref_first_obstruction(env, clear) is None
