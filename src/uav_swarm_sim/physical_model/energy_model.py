"""Grey-box component energy model (after Steup et al. 2020).

Instantaneous power per maneuver, integrated over time. Foundation for ALL
planning, launch-site, RTH, and redistribution costs. Energy is never a static
distance-average: every value is power x duration, so summing per-tick segment
energies reproduces the continuous integral E = sum_t P(maneuver(t)) * dt.

Formation aero benefit (thesis guideline 1.3) enters as a multiplicative power
factor and is restricted to CRUISE on FW/VTOL only -- see ``power`` below.

2.5D (Batch 3): the ONLY coupling of mass into energy lives in ``path_energy``
as the gravitational potential term ``m*g*dz`` charged on a genuine altitude GAIN
(inter-layer climb). Descent contributes nothing (no regeneration on these
platforms), and every horizontal / 2D / single-layer segment holds altitude
constant (dz == 0), so the term is exactly 0 there and the 2D results remain
byte-identical. Horizontal and hover power stay 100% table-driven.
"""
from __future__ import annotations

from typing import Callable

from ..infrastructure.core_types import Path, PathSegment
from ..infrastructure.enums import ManeuverType, PlatformType
from .drone_specs import PlatformSpec

# Maneuvers eligible for the formation drag benefit. Deliberately CRUISE only:
# the thesis restricts the benefit to launch/transit/RTH (all flown as CRUISE)
# and forbids it during COVERAGE. Restricting the eligibility set here -- rather
# than relying solely on the aero module returning 1.0 -- guarantees the
# invariant structurally: no caller can ever reduce COVERAGE energy.
_FORMATION_ELIGIBLE = {ManeuverType.CRUISE}
_FORMATION_PLATFORMS = {PlatformType.FIXED_WING, PlatformType.VTOL}

# Standard gravity for the 2.5D vertical potential term (m*g*dz on climb). This
# is the SINGLE place mass couples into energy; see ``path_energy``.
_G = 9.80665


class EnergyModel:
    def __init__(self, spec: PlatformSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> PlatformSpec:
        return self._spec

    def power(self, m: ManeuverType, formation_factor: float = 1.0) -> float:
        """Instantaneous power (W) for a maneuver.

        The formation factor is applied only for CRUISE on FW/VTOL; in every
        other case it is ignored (multirotors receive no energy benefit -- their
        downwash is a safety constraint, handled elsewhere -- and COVERAGE is
        never discounted).
        """
        p = self._spec.power_w[m]
        if m in _FORMATION_ELIGIBLE and self._spec.platform in _FORMATION_PLATFORMS:
            p *= formation_factor
        return p

    def segment_energy(
        self, m: ManeuverType, duration_s: float, formation_factor: float = 1.0
    ) -> float:
        """Energy (J) for holding a maneuver for a duration: P * dt.

        Propulsion term only. The vertical potential term (mass) is added in
        ``path_energy`` from per-segment altitude change, since it is a function
        of geometry (dz), not of maneuver/duration alone.
        """
        return self.power(m, formation_factor) * duration_s

    def path_energy(
        self, path: Path, factor_fn: Callable[[PathSegment], float] | None = None
    ) -> float:
        """Energy (J) to fly a Path. Sums per-segment P*duration -- the exact
        discrete integral, shared by predictive planners and the executor so the
        two never drift.

        2.5D: for any segment with a positive altitude change it also charges the
        gravitational potential ``m*g*dz`` (climb work). Descent (dz < 0) and all
        constant-altitude segments (dz == 0; every 2D / single-layer segment)
        contribute zero, keeping the 2D path energies byte-identical.
        """
        total = 0.0
        for seg in path.segments:
            f = factor_fn(seg) if factor_fn is not None else 1.0
            total += self.segment_energy(seg.maneuver, seg.duration_s, f)
            dz = seg.end.z - seg.start.z
            if dz > 0.0:
                total += self._spec.mass_kg * _G * dz
        return total

    def distance_energy(
        self, dist_m: float, m: ManeuverType, speed: float, formation_factor: float = 1.0
    ) -> float:
        """Energy (J) to cover a distance at a speed under a maneuver.

        Equivalent to P * (dist / speed) = P * duration. Convenience for graph
        edge costs (horizontal, constant-altitude legs); still a time integral,
        not a per-distance constant. Vertical potential is not part of a
        horizontal edge cost -- inter-layer climbs are costed via path_energy.
        """
        if speed <= 0:
            raise ValueError("distance_energy requires speed > 0")
        return self.power(m, formation_factor) * dist_m / speed
