"""The platform motion abstraction (Phase-1 decision: platform as config).

One interface, two kinematics. Everything downstream (transit planning, coverage
connectors, RTH, avoidance micro-plans) is platform-agnostic and goes through a
MotionModel, so the FW/VTOL-vs-multirotor switch lives in exactly one place.

2.5D (Batch 3): ``plan`` gains a thin vertical mode -- CLIMB / DESCENT delegate
to ``vertical_segments`` (which owns the climb/descent geometry and, via the
EnergyModel, the mass-coupled energy). Horizontal maneuvers are untouched, and
CLIMB/DESCENT never occur in the single-layer case, so this branch is never taken
there and the 2D behaviour is byte-identical.
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
from .vertical_segments import climb_path, descent_path


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

    def _vertical_path_if_vertical(
        self, start: Pose, goal: Pose, maneuver: ManeuverType
    ) -> Path | None:
        """Thin vertical wrapper. Genuine inter-layer climb/descent are owned by
        ``vertical_segments`` (geometry here; energy via the EnergyModel). The
        altitude change is taken from the poses' ``z``. Returns None for every
        horizontal maneuver so the 2D kinematics handle it unchanged."""
        if maneuver is ManeuverType.CLIMB:
            dz = goal.z - start.z
            return climb_path(self._spec, dz, start) if dz > 0 else Path(())
        if maneuver is ManeuverType.DESCENT:
            dz = start.z - goal.z
            return descent_path(self._spec, dz, start) if dz > 0 else Path(())
        return None

    def advance(self, path: Path, t_elapsed: float, dt: float) -> tuple[Pose | None, float]:
        """Advance a path by time. Returns (new pose, new elapsed time).

        Time-based (not arc-length) so holonomic in-place rotations progress
        correctly. The agent owns ``t_elapsed`` per active path.
        """
        new_t = min(t_elapsed + dt, path.total_duration_s)
        return path.pose_at_time(new_t), new_t

    def straight_leg(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        """A pure straight chord ``start -> goal`` at the maneuver speed, IGNORING
        the turn-radius constraint. This is the linear-corridor fallback used by
        ``planning.trajectory_validation`` when Dubins smoothing bulges a turn arc
        outside the corridor and clips an obstacle buffer: the GVG/boustrophedon
        skeleton is clear by construction, so the chord is collision-safe even
        though it is kinematically abrupt (arrival heading = chord bearing; the
        next leg re-plans from there, so any Dubins infeasibility stays local to
        this one leg). Identical for both kinematics -- it is THE chord -- and it
        matches a holonomic leg's straight phase. Altitude is held (horizontal).
        """
        v = self._spec.speed_for(maneuver)
        dx, dy = goal.x - start.x, goal.y - start.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            return Path(())
        bearing = math.atan2(dy, dx)
        p0 = Pose(start.x, start.y, bearing, start.z)
        return Path.from_segments([straight_segment(p0, dist, maneuver, v)])


class DubinsModel(MotionModel):
    """Fixed-wing and VTOL cruise: minimum-turn-radius Dubins kinematics."""

    def plan(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        vp = self._vertical_path_if_vertical(start, goal, maneuver)
        if vp is not None:
            return vp
        v = self._spec.speed_for(maneuver)
        return dubins.shortest_path(start, goal, self._spec.r_min_m, v, maneuver)

    def leg_cost(self, start: Pose, goal: Pose) -> float:
        return dubins.path_length(start, goal, self._spec.r_min_m)


class HolonomicModel(MotionModel):
    """Multirotor: in-place yaw + straight legs; Euclidean leg cost."""

    def plan(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        vp = self._vertical_path_if_vertical(start, goal, maneuver)
        if vp is not None:
            return vp
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
