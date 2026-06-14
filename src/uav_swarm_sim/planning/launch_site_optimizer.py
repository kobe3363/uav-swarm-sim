"""Launch site as an optimization VARIABLE (guideline 2.1 / §2.4).

Scored by the thesis's three criteria:
  1. mean kinematic distance to zone entries;
  2. initial-trajectory energy including the formation aero correction
     (launch + transit out + RTH back, all in formation);
  3. expected number of battery swaps over the whole mission.

Provisional zone entries are obtained from one uniform-weight decomposition over
the whole free space (site-independent work layout -- a documented approximation
that decouples site choice from partition shape). Distances use the motion
model's flyable leg cost; energies use the shared EnergyModel so prediction and
simulation never drift.

Energy feasibility (Batch 6.1)
------------------------------
The optimizer is now energy-AWARE: a candidate site is feasible only if a single
drone can reach the furthest navigable (free-space) point from it and return on
one usable battery, including vertical takeoff/landing. Infeasible sites are
never selected, and if NO candidate is feasible the optimizer raises
``InfeasibleMissionError`` rather than silently returning an impossible site.

The feasibility test is applied at SELECTION, not before scoring: every candidate
is still scored and min-max-normalized over the full set exactly as before, then
the best-J *feasible* site is chosen. So when the current min-J winner is feasible
(the regression case -- those missions complete, hence the site reaches and
returns), the chosen site is unchanged and the result is byte-identical; the gate
only changes the outcome when the best site is itself infeasible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point

from ..infrastructure.config import LaunchConfig
from ..infrastructure.core_types import DroneStateView, Pose
from ..infrastructure.enums import ManeuverType
from ..physical_model.aero_correction import AeroCorrection
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel
from ..physical_model.vertical_segments import landing_profile, takeoff_profile
from .environment_map import EnvironmentMap
from .tgc import TGCGraph
from .weighted_decomposition import WeightedTgcDecomposer

# Usable-capacity reserve fraction (matches the launch energy budget and the RTH
# reserve elsewhere -- the tool and the simulation agree on "usable").
_RESERVE_FRAC = 0.05


class InfeasibleMissionError(RuntimeError):
    """No candidate launch site can reach the furthest navigable point and return
    on a single battery: the mission is energetically impossible from anywhere in
    the search region, for this platform and battery capacity. Raised by
    ``optimize`` instead of returning an unreachable site."""


@dataclass(frozen=True)
class SiteScore:
    site: tuple[float, float]
    mean_dist_m: float
    initial_energy_j: float
    expected_swaps: float
    J: float


# --------------------------------------------------------------------------- #
# energy feasibility (Batch 6.1)                                              #
# --------------------------------------------------------------------------- #
def furthest_point_feasible(
    em: EnergyModel,
    spec: PlatformSpec,
    furthest_dist_m: float,
    altitude_m: float = 100.0,
    reserve_frac: float = _RESERVE_FRAC,
    min_work_j: float = 0.0,
) -> bool:
    """True iff one drone can fly from base to a point ``furthest_dist_m`` away at
    cruise, perform ``min_work_j`` of work there, and return -- all on one usable
    battery, including the vertical takeoff and landing energy.

    Cruise is charged SOLO (no formation benefit): the long-range sortie to the
    boundary is not a formation flight, so this is the conservative, physically
    correct gate. ``min_work_j`` defaults to 0 (pure reach-and-return floor); a
    caller may pass a coverage allowance for a stricter test.
    """
    if furthest_dist_m < 0:
        raise ValueError("furthest_dist_m must be >= 0")
    out_and_back = 2.0 * em.distance_energy(furthest_dist_m, ManeuverType.CRUISE, spec.v_cruise)
    vertical = (
        takeoff_profile(spec, em, altitude_m).energy_j
        + landing_profile(spec, em, altitude_m).energy_j
    )
    needed = out_and_back + vertical + max(0.0, min_work_j)
    usable = spec.battery_capacity_j * (1.0 - reserve_frac)
    return needed <= usable


def _furthest_free_vertex_dist(env: EnvironmentMap, from_xy: tuple[float, float]) -> float:
    """Distance to the furthest vertex of the free-space boundary from ``from_xy``.

    The furthest such vertex is the binding constraint: a site that can reach it
    and return can reach every nearer navigable point too, so testing it is
    equivalent to testing every free-space boundary vertex (and far cheaper).
    """
    fx, fy = from_xy
    geom = env.free_space
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    best = 0.0
    for poly in polys:
        if poly.is_empty:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            for x, y in ring.coords:
                d = math.hypot(x - fx, y - fy)
                if d > best:
                    best = d
    return best


def _candidate_sites(cfg: LaunchConfig, env: EnvironmentMap, rng: np.random.Generator) -> list[tuple[float, float]]:
    if not isinstance(cfg.candidate_sites, int):
        return [tuple(s) for s in cfg.candidate_sites]
    # sample free points biased toward the boundary (launch pads sit at the periphery)
    pool = env.sample_free(cfg.candidate_sites * 6, rng)
    if not pool:
        return []
    boundary = env.area.exterior
    pool.sort(key=lambda p: boundary.distance(Point(p)))  # closest to boundary first
    return pool[: cfg.candidate_sites]


def _provisional_entries(
    tgc: TGCGraph, env: EnvironmentMap, n_drones: int, altitude_centroid: Pose
) -> list[Pose]:
    drones = [
        DroneStateView(id=i, battery_frac=1.0, pose=altitude_centroid) for i in range(n_drones)
    ]
    part = WeightedTgcDecomposer().decompose(tgc, env, drones, target_area=None)
    return [z.entry_pose for z in part.zones.values()]


def optimize(
    cfg: LaunchConfig,
    tgc: TGCGraph,
    env: EnvironmentMap,
    motion: MotionModel,
    em: EnergyModel,
    aero: AeroCorrection,
    spec: PlatformSpec,
    n_drones: int,
    rng: np.random.Generator,
    altitude_m: float = 100.0,
) -> tuple[Pose, list[SiteScore]]:
    cx, cy = env.area.centroid.x, env.area.centroid.y
    centroid_pose = Pose(cx, cy, 0.0)
    entries = _provisional_entries(tgc, env, n_drones, centroid_pose)
    candidates = _candidate_sites(cfg, env, rng)
    if not candidates:
        raise RuntimeError("no feasible launch candidate sites found")

    f_form = aero.power_factor(in_formation=True, maneuver=ManeuverType.CRUISE)
    take = takeoff_profile(spec, em, altitude_m)
    land = landing_profile(spec, em, altitude_m)
    e_usable = spec.battery_capacity_j * (1.0 - _RESERVE_FRAC)  # small reserve eps

    rows: list[dict] = []
    for s in candidates:
        site_pose = Pose(s[0], s[1], math.atan2(cy - s[1], cx - s[0]))
        dists = [motion.leg_cost(site_pose, e) for e in entries]
        mean_dist = float(np.mean(dists)) if dists else math.inf

        initial_energy = 0.0
        expected_swaps = 0.0
        for e, dist in zip(entries, dists):
            out_e = em.distance_energy(dist, ManeuverType.CRUISE, spec.v_cruise, f_form)
            back_e = em.distance_energy(dist, ManeuverType.CRUISE, spec.v_cruise, f_form)
            initial_energy += out_e + back_e + take.energy_j + land.energy_j
            # crude per-drone coverage estimate: zone area not known here, so use
            # the transit+return energy as the recurring cost between swaps.
            cycle_e = out_e + back_e + take.energy_j + land.energy_j
            expected_swaps += max(0.0, math.ceil(cycle_e / e_usable) - 1)

        # Batch 6.1 energy feasibility: can a lone drone reach the furthest
        # free-space vertex from this site and return on one usable battery?
        furthest_m = _furthest_free_vertex_dist(env, s)
        feasible = furthest_point_feasible(em, spec, furthest_m, altitude_m)

        rows.append(
            {"site": s, "mean_dist": mean_dist, "energy": initial_energy, "swaps": expected_swaps,
             "pose": site_pose, "feasible": feasible}
        )

    # min-max normalize each criterion, then weighted sum J (over ALL candidates,
    # unchanged -- so feasible-site scores are identical to before)
    def norm(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        lo, hi = vals.min(), vals.max()
        return (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

    dn, en, sn = norm("mean_dist"), norm("energy"), norm("swaps")
    scores: list[SiteScore] = []
    for r, d_, e_, s_ in zip(rows, dn, en, sn):
        J = cfg.w_distance * d_ + cfg.w_energy * e_ + cfg.w_swaps * s_
        scores.append(SiteScore(r["site"], r["mean_dist"], r["energy"], r["swaps"], float(J)))

    scores.sort(key=lambda x: x.J)

    # Batch 6.1: select the best-J FEASIBLE site; refuse if none can fly the
    # mission at all. When the min-J site is feasible this is exactly scores[0],
    # so the chosen site (and thus the whole run) is byte-identical.
    feasible_sites = {r["site"] for r in rows if r["feasible"]}
    if not feasible_sites:
        raise InfeasibleMissionError(
            "No launch site can reach all navigable bounds and return on one "
            f"battery (usable {e_usable:.0f} J at altitude {altitude_m:.0f} m; "
            f"{len(candidates)} candidate sites evaluated). Increase battery "
            "capacity, lower the coverage altitude, or shrink the area."
        )
    best_score = next(sc for sc in scores if sc.site in feasible_sites)
    best = next(r for r in rows if r["site"] == best_score.site)
    return best["pose"], scores
