"""Dynamic return-to-home (guideline 3.1).

Continuously compares the remaining-plan cost against the return-from-here cost,
replacing any static reserve (e.g. a fixed 30%). The reserve here is a small
epsilon safety margin only; the decision quantity is the live route-vs-return
comparison, which tightens automatically as avoidance/replanning consume energy.
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

    @property
    def check_interval_s(self) -> float:
        return self._cfg.check_interval_s

    def return_energy(self, from_pose: Pose, in_formation: bool = False) -> float:
        route = self._motion.plan(from_pose, self._base, ManeuverType.CRUISE)
        e = self._em.path_energy(route)
        if self._env is not None and not self._env.path_clear(route):
            e *= 1.5  # obstacle detour penalty (CostDB folded into a factor here)
        return e + self._land.energy_j

    def should_return(
        self, level_j: float, e_next_j: float, next_pose: Pose, in_formation: bool = False
    ) -> bool:
        reserve = self._cfg.reserve_frac * self._spec.battery_capacity_j
        budget = e_next_j + self.return_energy(next_pose, in_formation) + reserve
        return level_j < budget

    def decide(self, agent) -> str:
        e_next, p_next = agent.lookahead()
        if self.should_return(agent.battery.level_j, e_next, p_next):
            return "RETURN_NOW"
        return "CONTINUE"
