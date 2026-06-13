"""Scope-limited aerodynamic interaction (after Guo et al. 2025).

Formation drag benefit for FW/VTOL (energy); downwash wake geometry for
multirotor (safety constraint, NOT energy). The benefit applies only in
launch / transit / RTH (flown as CRUISE), never during dispersed coverage.

Whether a drone is actually in formation is decided by the FormationManager
(execution layer); this module only converts that membership into physics.
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon

from ..infrastructure.config import AeroConfig
from ..infrastructure.core_types import Pose
from ..infrastructure.enums import ManeuverType, PlatformType

_BENEFIT_PLATFORMS = {PlatformType.FIXED_WING, PlatformType.VTOL}
_BENEFIT_MANEUVERS = {ManeuverType.CRUISE}


class AeroCorrection:
    def __init__(self, cfg: AeroConfig, platform: PlatformType) -> None:
        self._cfg = cfg
        self._platform = platform

    def power_factor(self, in_formation: bool, maneuver: ManeuverType) -> float:
        """Multiplicative power factor for the energy model.

        Returns (1 - drag_reduction) only when the drone is in formation, the
        maneuver is CRUISE, and the platform is FW/VTOL. Returns 1.0 otherwise
        -- always during COVERAGE and always for multirotor.
        """
        if (
            in_formation
            and maneuver in _BENEFIT_MANEUVERS
            and self._platform in _BENEFIT_PLATFORMS
        ):
            return 1.0 - self._cfg.formation_drag_reduction
        return 1.0

    def wake_zones(self, leader_poses: list[Pose]) -> list[Polygon]:
        """Wake polygons trailing each leader pose.

        For multirotor (and VTOL vertical phases) these are downwash hazard
        zones fed to the safety monitor as 'invisible obstacles'. For FW they
        describe the wingtip-vortex corridor (advisory: a formation follower
        seeks the beneficial station; crossing traffic avoids it).
        """
        half_w = self._cfg.downwash_radius_m
        length = self._cfg.downwash_length_m
        zones: list[Polygon] = []
        for p in leader_poses:
            h = p.heading
            # unit vectors: forward (heading), left normal
            fx, fy = math.cos(h), math.sin(h)
            lx, ly = -math.sin(h), math.cos(h)
            front_x, front_y = p.x, p.y
            back_x, back_y = p.x - length * fx, p.y - length * fy
            corners = [
                (front_x + half_w * lx, front_y + half_w * ly),
                (front_x - half_w * lx, front_y - half_w * ly),
                (back_x - half_w * lx, back_y - half_w * ly),
                (back_x + half_w * lx, back_y + half_w * ly),
            ]
            zones.append(Polygon(corners))
        return zones
