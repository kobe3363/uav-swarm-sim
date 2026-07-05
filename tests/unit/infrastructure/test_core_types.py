"""infrastructure/core_types geometry tests (isolated)."""
from __future__ import annotations

import math

import pytest

from uav_swarm_sim.infrastructure.core_types import (
    Path,
    Pose,
    arc_segment,
    inplace_turn_segment,
    normalize_angle,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import ManeuverType


def _pose_close(p, q, tol=1e-6):
    return (
        abs(p.x - q.x) <= tol
        and abs(p.y - q.y) <= tol
        and abs(normalize_angle(p.heading - q.heading)) <= tol
    )


def test_straight_segment_endpoint_and_length():
    s = straight_segment(Pose(0, 0, 0.0), 10.0, ManeuverType.CRUISE, 5.0)
    assert _pose_close(s.end, Pose(10, 0, 0))
    assert s.length_m == pytest.approx(10.0)
    assert s.duration_s == pytest.approx(2.0)


def test_arc_quarter_circle_left():
    # left quarter turn, radius 1, from origin heading +x -> ends at (1,1) heading +y
    s = arc_segment(Pose(0, 0, 0.0), curvature=1.0, arc_length=math.pi / 2, maneuver=ManeuverType.TURN, speed=1.0)
    assert _pose_close(s.end, Pose(1.0, 1.0, math.pi / 2), tol=1e-9)


def test_path_pose_at_length_and_time():
    s = straight_segment(Pose(0, 0, 0.0), 10.0, ManeuverType.CRUISE, 5.0)
    p = Path.from_segments([s])
    assert _pose_close(p.pose_at_length(5.0), Pose(5, 0, 0))
    # duration 2.0 s; halfway in time = 1.0 s -> x=5
    assert _pose_close(p.pose_at_time(1.0), Pose(5, 0, 0))


def test_inplace_turn_progresses_by_time_not_length():
    t = inplace_turn_segment(Pose(0, 0, 0.0), math.pi / 2, omega_max=1.0, maneuver=ManeuverType.TURN)
    assert t.length_m == 0.0
    assert t.duration_s == pytest.approx(math.pi / 2)
    p = Path.from_segments([t])
    mid = p.pose_at_time(t.duration_s / 2)
    assert mid.heading == pytest.approx(math.pi / 4)
