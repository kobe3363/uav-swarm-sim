"""1-D vertical flight model (thesis §2.5).

Takeoff/landing are handled as separate segments outside the 2-D configuration
space. Multirotor/VTOL climb vertically; fixed-wing performs a sloped
climb-out/approach plus a ground-roll energy charge. Altitude is a parameter,
never a planning variable (constant-altitude coverage assumption).

2.5D (Batch 3): in addition to the ground<->coverage-altitude takeoff/landing
profiles (UNCHANGED, so the validated 2D model stays byte-identical), this module
gains genuine inter-layer ``climb_path``/``descent_path`` (and matching
profiles). Those move altitude between layers and set ``end.z`` accordingly, so
the gravitational potential term ``m*g*dz`` charged by ``EnergyModel.path_energy``
applies on climb. Descent is dissipative (no regeneration). Mid-mission vertical
transitions are airborne, so no ground-roll energy is charged on them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..infrastructure.core_types import Path, PathSegment, Pose, straight_segment
from ..infrastructure.enums import ManeuverType, PlatformType
from .drone_specs import PlatformSpec
from .energy_model import EnergyModel

# Standard gravity (kept local for documentation; the mass*g*dz term itself is
# applied once, inside EnergyModel.path_energy).
_G = 9.80665


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


# --------------------------------------------------------------------------- #
# 2.5D inter-layer vertical moves (Batch 3)                                   #
# Genuine altitude change between coverage layers. ``end.z`` reflects the gain/  #
# loss so EnergyModel.path_energy charges m*g*dz on climb (and nothing on        #
# descent). Airborne transitions: no ground-roll energy.                         #
# --------------------------------------------------------------------------- #
def _inter_layer_path(spec: PlatformSpec, dz: float, at: Pose, *, climbing: bool) -> Path:
    """Geometry-only inter-layer move of magnitude ``dz`` (> 0). MR/VTOL move
    purely vertically; FW flies a sloped climb/approach whose duration reflects
    the slant. ``end.z`` carries the altitude change (the energy model reads it).
    """
    if dz <= 0:
        return Path(())
    maneuver = ManeuverType.CLIMB if climbing else ManeuverType.DESCENT
    speed = spec.v_climb if climbing else spec.v_descent
    if speed <= 0:
        raise ValueError("vertical speed must be > 0")
    z_end = at.z + dz if climbing else at.z - dz
    if spec.platform in (PlatformType.MULTIROTOR, PlatformType.VTOL):
        duration = dz / speed
        end = Pose(at.x, at.y, at.heading, z_end)
        seg = PathSegment(maneuver, 0.0, duration, at, end, 0.0)
        return Path((seg,))
    # FIXED_WING: sloped path; horizontal run = dz / tan(climb_angle)
    ground = dz / math.tan(spec.climb_angle_rad)
    slant = math.hypot(dz, ground)
    duration = slant / speed
    end = Pose(at.x + ground * math.cos(at.heading), at.y + ground * math.sin(at.heading), at.heading, z_end)
    seg = PathSegment(maneuver, slant, duration, at, end, 0.0)
    return Path((seg,))


def climb_path(spec: PlatformSpec, dz: float, at: Pose | None = None) -> Path:
    """Geometry of an inter-layer climb of height ``dz`` (> 0)."""
    return _inter_layer_path(spec, dz, at or Pose(0.0, 0.0, 0.0), climbing=True)


def descent_path(spec: PlatformSpec, dz: float, at: Pose | None = None) -> Path:
    """Geometry of an inter-layer descent of height ``dz`` (> 0)."""
    return _inter_layer_path(spec, dz, at or Pose(0.0, 0.0, 0.0), climbing=False)


def climb_profile(spec: PlatformSpec, em: EnergyModel, dz: float, at: Pose | None = None) -> VerticalProfile:
    """Inter-layer climb. ``energy_j`` = table CLIMB propulsion + ``m*g*dz``
    potential, the potential supplied by EnergyModel.path_energy (the single home
    of the mass term) from the segment's altitude gain."""
    path = climb_path(spec, dz, at)
    return VerticalProfile(path.total_duration_s, em.path_energy(path), path.segments)


def descent_profile(spec: PlatformSpec, em: EnergyModel, dz: float, at: Pose | None = None) -> VerticalProfile:
    """Inter-layer descent. Dissipative: table DESCENT propulsion only -- the
    negative altitude change contributes nothing to the mass term (no regen)."""
    path = descent_path(spec, dz, at)
    return VerticalProfile(path.total_duration_s, em.path_energy(path), path.segments)
