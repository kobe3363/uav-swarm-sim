"""The platform motion abstraction (Phase-1 decision: platform as config).

One interface, two kinematics. Everything downstream (transit planning, coverage
connectors, RTH, avoidance micro-plans) is platform-agnostic and goes through a
MotionModel, so the FW/VTOL-vs-multirotor switch lives in exactly one place.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ..infrastructure.core_types import (
    Path,
    Pose,
    inplace_turn_segment,
    straight_segment,
)
from ..infrastructure.enums import ManeuverType, PlatformType
from . import dubins
from .drone_specs import PlatformSpec


class MotionModel(ABC):
    def __init__(self, spec: PlatformSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> PlatformSpec:
        return self._spec

    @abstractmethod
    def plan(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        ...

    @abstractmethod
    def leg_cost(self, start: Pose, goal: Pose) -> float:
        """Flyable length (m) of the cheapest leg between poses."""

    def advance(self, path: Path, t_elapsed: float, dt: float) -> tuple[Pose | None, float]:
        """Advance a path by time. Returns (new pose, new elapsed time).

        Time-based (not arc-length) so holonomic in-place rotations progress
        correctly. The agent owns ``t_elapsed`` per active path.
        """
        new_t = min(t_elapsed + dt, path.total_duration_s)
        return path.pose_at_time(new_t), new_t


class DubinsModel(MotionModel):
    """Fixed-wing and VTOL cruise: minimum-turn-radius Dubins kinematics."""

    def plan(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        v = self._spec.speed_for(maneuver)
        return dubins.shortest_path(start, goal, self._spec.r_min_m, v, maneuver)

    def leg_cost(self, start: Pose, goal: Pose) -> float:
        return dubins.path_length(start, goal, self._spec.r_min_m)


class HolonomicModel(MotionModel):
    """Multirotor: in-place yaw + straight legs; Euclidean leg cost."""

    def plan(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        v = self._spec.speed_for(maneuver)
        omega = self._spec.omega_max
        dx, dy = goal.x - start.x, goal.y - start.y
        dist = math.hypot(dx, dy)
        segs = []
        if dist > 1e-9:
            bearing = math.atan2(dy, dx)
            turn1 = inplace_turn_segment(start, bearing, omega, ManeuverType.TURN)
            if turn1.duration_s > 0:
                segs.append(turn1)
            after_turn1 = segs[-1].end if segs else start
            straight = straight_segment(after_turn1, dist, maneuver, v)
            segs.append(straight)
            cur = straight.end
        else:
            cur = start
        # final heading alignment
        turn2 = inplace_turn_segment(cur, goal.heading, omega, ManeuverType.TURN)
        if turn2.duration_s > 0:
            segs.append(turn2)
        return Path.from_segments(segs)

    def leg_cost(self, start: Pose, goal: Pose) -> float:
        return math.hypot(goal.x - start.x, goal.y - start.y)


def make_motion_model(spec: PlatformSpec) -> MotionModel:
    if spec.platform is PlatformType.MULTIROTOR:
        return HolonomicModel(spec)
    return DubinsModel(spec)
