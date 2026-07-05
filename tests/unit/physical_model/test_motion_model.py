"""physical_model/motion_model tests (isolated)."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose, normalize_angle
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.motion_model import (
    DubinsModel,
    HolonomicModel,
    make_motion_model,
)


def _spec(config_path, platform="MULTIROTOR"):
    cfg = load_config(config_path, overrides={"platform_type": platform})
    return build_spec(cfg), cfg


def _pose_close(p, q, tol=1e-6):
    return (
        abs(p.x - q.x) <= tol
        and abs(p.y - q.y) <= tol
        and abs(normalize_angle(p.heading - q.heading)) <= tol
    )


def test_make_motion_model_per_platform(config_path):
    assert isinstance(make_motion_model(_spec(config_path, "MULTIROTOR")[0]), HolonomicModel)
    assert isinstance(make_motion_model(_spec(config_path, "FIXED_WING")[0]), DubinsModel)
    assert isinstance(make_motion_model(_spec(config_path, "VTOL")[0]), DubinsModel)


def test_dubins_model_plan_and_cost(config_path):
    spec, _ = _spec(config_path, "FIXED_WING")
    m = DubinsModel(spec)
    start, goal = Pose(0, 0, 0.0), Pose(100, 50, 1.0)
    path = m.plan(start, goal, ManeuverType.CRUISE)
    assert _pose_close(path.end_pose, goal, tol=1e-3)
    assert m.leg_cost(start, goal) == pytest.approx(path.total_length_m, abs=1e-6)


def test_holonomic_plan_reaches_goal_and_euclidean_cost(config_path):
    spec, _ = _spec(config_path, "MULTIROTOR")
    m = HolonomicModel(spec)
    start, goal = Pose(0, 0, 0.0), Pose(30, 40, 1.0)
    path = m.plan(start, goal, ManeuverType.CRUISE)
    assert _pose_close(path.end_pose, goal, tol=1e-9)
    assert m.leg_cost(start, goal) == pytest.approx(50.0)  # 3-4-5
    # advance by time reaches the end pose
    pose, t = path.pose_at_time(path.total_duration_s), path.total_duration_s
    assert _pose_close(pose, goal, tol=1e-9)
