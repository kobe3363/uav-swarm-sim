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

import math
from typing import TYPE_CHECKING

from ..infrastructure.config import RTHConfig
from ..infrastructure.core_types import Pose
from ..infrastructure.enums import ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel
from ..physical_model.vertical_segments import landing_profile

if TYPE_CHECKING:  # import-time cycle guard only; runtime access is duck-typed
    from ..planning.energy_map import EnergyMap

# EM-01 Stage 2 (design doc section 7): below the per-sortie arming threshold
# the energy decision is re-evaluated every 1% of battery capacity consumed
# (1% cap = 3600 J ~ 10 cell hops at the battery-tied resolution). A design
# quantum, not a tunable -- deliberately NOT a config knob.
_DECIDE_STEP_FRAC = 0.01


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
        *,
        energy_map: "EnergyMap | None" = None,
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
        # EM-01 Stage 2 (seam 7a): map-based decide, gated on BOTH the built map
        # and rth.energy_map.decide -- enabled=True alone keeps the Stage-1
        # build-and-attach-only contract (its gate test asserts no metric moves).
        self._map = energy_map
        self._map_decide = energy_map is not None and cfg.energy_map.decide
        # Observability (Stage 5 A/B will report fallback rate as a boxing
        # proxy). Plain ints, never serialized on the flag-off path.
        self.n_map_hits = 0
        self.n_map_fallbacks = 0

    @property
    def check_interval_s(self) -> float:
        return self._cfg.check_interval_s

    @property
    def map_decide_on(self) -> bool:
        """True when the Stage-2 map-based decide arm is active."""
        return self._map_decide

    @property
    def decide_step_j(self) -> float:
        """Battery-quantized cadence step (J): re-decide every 1% of capacity."""
        return _DECIDE_STEP_FRAC * self._spec.battery_capacity_j

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

    def _map_return_energy(self, from_pose: Pose, altitude_m: float | None) -> float | None:
        """Seam 7a: E_home lookup at ``from_pose``'s cell + the landing term.

        The map is CRUISE-horizontal by the Stage-1 contract (energy_map.py
        module docstring), so the descent from the drone's own layer altitude
        is still added here -- exactly mirroring the analytic arm's line. No
        x1.5 fudge: occupancy penalties are already in the edge weights, the
        fudge would double-count. Returns None (-> analytic fallback) for a
        pose outside the grid (defensive; the universal extent makes this
        geometrically unreachable) or an E_home=inf cell (true obstacle
        boxing -- a geometry dead-end, not an energy decision)."""
        frame = self._map.frame
        i, j = frame.world_to_cell(from_pose.x, from_pose.y)
        if not (0 <= i < frame.nx and 0 <= j < frame.ny):
            return None
        e = float(self._map.e_home[i, j])
        if not math.isfinite(e):
            return None
        return e + self._landing_for(altitude_m).energy_j

    def return_energy(
        self, from_pose: Pose, in_formation: bool = False, altitude_m: float | None = None
    ) -> float:
        if self._map_decide:
            e_map = self._map_return_energy(from_pose, altitude_m)
            if e_map is not None:
                self.n_map_hits += 1
                return e_map
            self.n_map_fallbacks += 1
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

    def sortie_arm_j(
        self, bundles: list[tuple[float, Pose]], altitude_m: float | None = None
    ) -> float:
        """Seam 7b: per-sortie arming threshold (J). Above it the decide is
        provably CONTINUE and is skipped entirely.

        ``bundles`` holds, for every remaining plan index k, exactly what
        ``decide()`` would see there: ``(e_bundle_k, end_pose_k)`` from
        ``agent.lookahead(k)``. The decide at k evaluates
        ``e_k + return_energy(p_k) + reserve``, and ``max_ret`` below uses the
        SAME deterministic ``return_energy`` (map value where finite, analytic
        fallback otherwise -- a pure max-E_home bound would not cover a fudged
        fallback), so ``level > arm`` implies CONTINUE for every k. ``delta``
        -- one orthogonal cell hop at the map's actual resolution (~360 J at
        defaults) -- is pure extra margin for the in-cell pose quantization.
        Empty ``bundles`` => inf: never skip (coverage_complete fires anyway).
        """
        if not bundles:
            return float("inf")
        max_ret = max(self.return_energy(p, altitude_m=altitude_m) for _, p in bundles)
        max_bundle = max(e for e, _ in bundles)
        reserve = self._cfg.reserve_frac * self._spec.battery_capacity_j
        delta = 0.0
        if self._map is not None:
            delta = self._em.distance_energy(
                self._map.frame.cell_m, ManeuverType.CRUISE, self._spec.v_cruise
            )
        return max_ret + max_bundle + reserve + delta
