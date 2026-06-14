"""Dynamic return-to-home (guideline 3.1).

Continuously compares the remaining-plan cost against the return-from-here cost,
replacing any static reserve (e.g. a fixed 30%). The reserve here is a small
epsilon safety margin only; the decision quantity is the live route-vs-return
comparison, which tightens automatically as avoidance/replanning consume energy.

2.5D (Batch 4): the descent term of the return cost is the landing from the
drone's OWN coverage layer altitude (taken from ``agent.coverage_altitude_m``),
not a single fixed altitude -- a drone on a higher layer must reserve more to get
down. With one layer (altitude == coverage_altitude_m) the per-agent altitude
equals the configured altitude, so the precomputed landing profile is reused and
the RTH decisions are byte-identical to the 2D model.
"""
from __future__ import annotations

from ..infrastructure.config import RTHConfig
from ..infrastructure.core_types import Pose
from ..infrastructure.enums import ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel
from ..physical_model.vertical_segments import landing_profile


class RthCalculator:
    def __init__(
        self,
        em: EnergyModel,
        motion: MotionModel,
        spec: PlatformSpec,
        cfg: RTHConfig,
        base: Pose,
        altitude_m: float = 100.0,
        env=None,
    ) -> None:
        self._em = em
        self._motion = motion
        self._spec = spec
        self._cfg = cfg
        self._base = base
        self._alt = altitude_m
        self._env = env
        self._land = landing_profile(spec, em, altitude_m)
        # per-altitude landing cache for multi-layer (layer altitude -> profile)
        self._land_cache: dict[float, object] = {}

    @property
    def check_interval_s(self) -> float:
        return self._cfg.check_interval_s

    def _landing_for(self, altitude_m: float | None):
        """Landing profile for the drone's layer altitude. None or the configured
        altitude returns the precomputed profile (single-layer byte-identity)."""
        if altitude_m is None or altitude_m == self._alt:
            return self._land
        prof = self._land_cache.get(altitude_m)
        if prof is None:
            prof = landing_profile(self._spec, self._em, altitude_m)
            self._land_cache[altitude_m] = prof
        return prof

    def return_energy(
        self, from_pose: Pose, in_formation: bool = False, altitude_m: float | None = None
    ) -> float:
        route = self._motion.plan(from_pose, self._base, ManeuverType.CRUISE)
        e = self._em.path_energy(route)
        if self._env is not None and not self._env.path_clear(route):
            e *= 1.5  # obstacle detour penalty (CostDB folded into a factor here)
        return e + self._landing_for(altitude_m).energy_j

    def should_return(
        self,
        level_j: float,
        e_next_j: float,
        next_pose: Pose,
        in_formation: bool = False,
        altitude_m: float | None = None,
    ) -> bool:
        reserve = self._cfg.reserve_frac * self._spec.battery_capacity_j
        budget = e_next_j + self.return_energy(next_pose, in_formation, altitude_m) + reserve
        return level_j < budget

    def decide(self, agent) -> str:
        e_next, p_next = agent.lookahead()
        altitude_m = getattr(agent, "coverage_altitude_m", None)
        if self.should_return(agent.battery.level_j, e_next, p_next, altitude_m=altitude_m):
            return "RETURN_NOW"
        return "CONTINUE"
