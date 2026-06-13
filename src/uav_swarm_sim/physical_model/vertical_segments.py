"""1-D vertical flight model (thesis §2.5).

Takeoff/landing are handled as separate segments outside the 2-D configuration
space. Multirotor/VTOL climb vertically; fixed-wing performs a sloped
climb-out/approach plus a ground-roll energy charge. Altitude is a parameter,
never a planning variable (constant-altitude coverage assumption).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..infrastructure.core_types import Path, PathSegment, Pose, straight_segment
from ..infrastructure.enums import ManeuverType, PlatformType
from .drone_specs import PlatformSpec
from .energy_model import EnergyModel


@dataclass(frozen=True)
class VerticalProfile:
    duration_s: float
    energy_j: float
    segments: tuple[PathSegment, ...]

    def as_path(self) -> Path:
        return Path(self.segments)


def _vertical_profile(
    spec: PlatformSpec,
    em: EnergyModel,
    altitude_m: float,
    maneuver: ManeuverType,
    speed: float,
    at: Pose,
) -> VerticalProfile:
    """Pure vertical phase for multirotor/VTOL (climb or descent)."""
    if altitude_m <= 0:
        return VerticalProfile(0.0, 0.0, ())
    if speed <= 0:
        raise ValueError("vertical speed must be > 0")
    duration = altitude_m / speed
    energy = em.segment_energy(maneuver, duration)
    # represent as a zero-horizontal-length segment carrying the duration
    seg = PathSegment(maneuver, 0.0, duration, at, at, 0.0)
    return VerticalProfile(duration, energy, (seg,))


def takeoff_profile(spec: PlatformSpec, em: EnergyModel, altitude_m: float, at: Pose | None = None) -> VerticalProfile:
    at = at or Pose(0.0, 0.0, 0.0)
    if spec.platform in (PlatformType.MULTIROTOR, PlatformType.VTOL):
        return _vertical_profile(spec, em, altitude_m, ManeuverType.TAKEOFF, spec.v_climb, at)
    # FIXED_WING: sloped climb-out + ground roll
    if altitude_m <= 0:
        return VerticalProfile(0.0, spec.ground_roll_energy_j, ())
    ground_dist = altitude_m / math.tan(spec.climb_angle_rad)
    slant = math.hypot(altitude_m, ground_dist)
    duration = slant / spec.v_climb
    energy = em.segment_energy(ManeuverType.CLIMB, duration) + spec.ground_roll_energy_j
    seg = straight_segment(at, slant, ManeuverType.CLIMB, spec.v_climb)
    return VerticalProfile(duration, energy, (seg,))


def landing_profile(spec: PlatformSpec, em: EnergyModel, altitude_m: float, at: Pose | None = None) -> VerticalProfile:
    at = at or Pose(0.0, 0.0, 0.0)
    if spec.platform in (PlatformType.MULTIROTOR, PlatformType.VTOL):
        return _vertical_profile(spec, em, altitude_m, ManeuverType.LAND, spec.v_descent, at)
    # FIXED_WING: powered sloped approach (no ground-roll energy charged on landing)
    if altitude_m <= 0:
        return VerticalProfile(0.0, 0.0, ())
    ground_dist = altitude_m / math.tan(spec.climb_angle_rad)
    slant = math.hypot(altitude_m, ground_dist)
    duration = slant / spec.v_descent
    energy = em.segment_energy(ManeuverType.LAND, duration)
    seg = straight_segment(at, slant, ManeuverType.LAND, spec.v_descent)
    return VerticalProfile(duration, energy, (seg,))
