"""physical_model/dubins geometry tests (isolated)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.core_types import Pose, normalize_angle
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model import dubins


def _pose_close(p, q, tol=1e-6):
    return (
        abs(p.x - q.x) <= tol
        and abs(p.y - q.y) <= tol
        and abs(normalize_angle(p.heading - q.heading)) <= tol
    )


def test_dubins_straight_line():
    start, goal = Pose(0, 0, 0.0), Pose(10, 0, 0.0)
    path = dubins.shortest_path(start, goal, r_min=1.0, v=2.0, maneuver=ManeuverType.CRUISE)
    assert path.total_length_m == pytest.approx(10.0, abs=1e-6)
    assert _pose_close(path.end_pose, goal)


def test_dubins_endpoint_matches_goal_random():
    rng = np.random.default_rng(0)
    r_min = 2.0
    for _ in range(200):
        start = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        goal = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        path = dubins.shortest_path(start, goal, r_min, v=1.0, maneuver=ManeuverType.CRUISE)
        assert _pose_close(path.end_pose, goal, tol=1e-4), f"{start} -> {goal}"


def test_dubins_path_length_matches_shortest_path():
    rng = np.random.default_rng(1)
    for _ in range(100):
        start = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        goal = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        L = dubins.path_length(start, goal, 2.0)
        P = dubins.shortest_path(start, goal, 2.0, 1.0, ManeuverType.CRUISE).total_length_m
        assert L == pytest.approx(P, abs=1e-6)


def test_dubins_respects_min_radius():
    path = dubins.shortest_path(Pose(0, 0, 0.0), Pose(0, 1, math.pi), r_min=3.0, v=1.0, maneuver=ManeuverType.CRUISE)
    for seg in path.segments:
        if seg.curvature != 0.0:
            assert abs(1.0 / seg.curvature) == pytest.approx(3.0, abs=1e-9)


def test_dubins_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        dubins.shortest_path(Pose(0, 0, 0), Pose(1, 1, 0), r_min=0.0, v=1.0, maneuver=ManeuverType.CRUISE)
