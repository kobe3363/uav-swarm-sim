"""Launch site as an optimization VARIABLE (guideline 2.1 / §2.4).

Scored by the thesis's three criteria:
  1. mean kinematic distance to zone entries;
  2. initial-trajectory energy including the formation aero correction
     (launch + transit out + RTH back, all in formation);
  3. EXACT number of battery swaps the fleet must perform over the whole
     mission, derived from the same workload math as the B6.2 fleet-sizing
     analyzer (total coverage length / per-sortie coverage budget / total
     sorties / swap count for ``n_drones``).

Provisional zone entries are obtained from one uniform-weight decomposition over
the whole free space (site-independent work layout -- a documented approximation
that decouples site choice from partition shape). Distances use the motion
model's flyable leg cost; energies use the shared EnergyModel so prediction and
simulation never drift.

Operational staging constraint
------------------------------
A Ground Control Station cannot sit inside the survey polygon (the swamp / field
/ hazard area being covered). Launch candidates are therefore drawn from a
flyable STAGING RING just OUTSIDE ``env.area`` -- ``area.buffer(standoff) minus
area`` -- never from inside it. Because synthetic obstacles live only inside
``area``, every point in that ring is clear, so the ring is the realizable form
of "clear of obstacles AND outside the target." (Note: ``env.free_space`` is
``area.difference(obstacles)`` and is a SUBSET of ``area``; requiring a site to be
in ``free_space`` *and* outside ``area`` is unsatisfiable, so the staging ring --
outside ``area``, where everything is clear -- is what realizes the intent.)

Only candidate GENERATION encodes this constraint; the scoring loop is unchanged.

Energy feasibility + exact fatigue (Batch 6.3)
----------------------------------------------
The optimizer is energy-AWARE and fatigue-EXACT:

  * A candidate site is FEASIBLE only if a single drone can reach the furthest
    navigable (free-space) point from it and return on one usable battery,
    including vertical takeoff/landing.
  * Infeasible candidates are DISCARDED BEFORE SCORING.
  * For every feasible candidate the optimizer applies the exact B6.2 workload
    math from this site (site-specific transit overhead) to compute the EXACT
    number of fleet battery swaps, and that exact integer is criterion 3.

Regression fixtures are expected to be re-baselined: filtering before scoring,
the exact swap term, and the staging-ring candidate domain all move the J
landscape relative to earlier batches.

The canonical swap/workload helpers (``coverage_path_length``,
``per_sortie_coverage_budget_j``, ``required_sorties``, ``fleet_swaps``) live in
this core module so the optimizer AND the standalone suitability plotter share
ONE implementation. ``experiments/fleet_sizing.py`` currently keeps its own
arithmetically-identical copy; unifying it onto these helpers is a queued
follow-up.
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

# Boustrophedon turn-around overhead multiplier on the ideal area/swath sweep
# length. MUST match experiments/fleet_sizing.py (TURN_FACTOR_DEFAULT) so the
# optimizer's swap count agrees with the fleet-sizing analyzer's.
TURN_FACTOR_DEFAULT = 1.15

# How far outside the survey area the flyable STAGING ring extends (metres). The
# Ground Control Station / launch pad is placed in this peripheral band, never
# inside the target polygon. Single source of truth: the suitability plotter
# imports this so its ring matches the optimizer's candidate domain exactly.
# Tune here (or lift into LaunchConfig later) to widen/narrow the staging band.
_STAGING_STANDOFF_M = 250.0


class InfeasibleMissionError(RuntimeError):
    """No candidate launch site can reach the furthest navigable point and return
    on a single battery (or no site leaves any per-sortie coverage budget after
    transit overhead): the mission is energetically impossible from anywhere in
    the search region, for this platform and battery capacity. Raised by
    ``optimize`` instead of returning an unreachable site."""


@dataclass(frozen=True)
class SiteScore:
    site: tuple[float, float]
    mean_dist_m: float
    initial_energy_j: float
    expected_swaps: float  # B6.3: now the EXACT fleet swap count (kept name for API compat)
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


# --------------------------------------------------------------------------- #
# canonical workload / swap math (Batch 6.3)                                  #
#                                                                             #
# Pure functions, no I/O, no plotting. Single source of truth for the core:   #
# optimize() (this module) and experiments/plot_launch_suitability.py both    #
# import these so the suitability map and the chosen site agree by            #
# construction. Formulas are IDENTICAL to experiments/fleet_sizing.py.        #
# --------------------------------------------------------------------------- #
def coverage_path_length(
    area_m2: float,
    effective_swath_m: float,
    turn_factor: float = TURN_FACTOR_DEFAULT,
) -> float:
    """Total boustrophedon sweep length to cover ``area_m2`` with a sensor of
    effective swath ``effective_swath_m``, inflated by ``turn_factor`` for the
    turn-around overhead between adjacent passes."""
    if effective_swath_m <= 0.0:
        raise ValueError("effective_swath_m must be > 0")
    if area_m2 < 0.0:
        raise ValueError("area_m2 must be >= 0")
    if turn_factor < 1.0:
        raise ValueError("turn_factor must be >= 1.0")
    return (area_m2 / effective_swath_m) * turn_factor


def per_sortie_coverage_budget_j(
    em: EnergyModel,
    spec: PlatformSpec,
    transit_dist_m: float,
    altitude_m: float = 100.0,
    reserve_frac: float = _RESERVE_FRAC,
) -> float:
    """Energy left for actual coverage work on ONE sortie launched from a site
    whose work area is ``transit_dist_m`` away, after paying vertical
    takeoff+landing and the round-trip cruise transit out of the usable battery.

    Raises ``InfeasibleMissionError`` if the fixed overhead already meets or
    exceeds the usable battery (no coverage progress is possible from this site).
    """
    if transit_dist_m < 0.0:
        raise ValueError("transit_dist_m must be >= 0")
    take = takeoff_profile(spec, em, altitude_m).energy_j
    land = landing_profile(spec, em, altitude_m).energy_j
    transit_round_j = 2.0 * em.distance_energy(transit_dist_m, ManeuverType.CRUISE, spec.v_cruise)
    overhead_j = take + land + transit_round_j
    usable_j = spec.battery_capacity_j * (1.0 - reserve_frac)
    budget_j = usable_j - overhead_j
    if budget_j <= 0.0:
        raise InfeasibleMissionError(
            f"per-sortie overhead {overhead_j:.0f} J (takeoff+landing+round-trip "
            f"transit over {transit_dist_m:.0f} m) >= usable battery {usable_j:.0f} J: "
            "no coverage budget remains from this site."
        )
    return budget_j


def required_sorties(
    em: EnergyModel,
    spec: PlatformSpec,
    area_m2: float,
    transit_dist_m: float,
    altitude_m: float,
    effective_swath_m: float,
    turn_factor: float = TURN_FACTOR_DEFAULT,
    reserve_frac: float = _RESERVE_FRAC,
) -> int:
    """Integer number of coverage sorties needed to sweep ``area_m2`` from a site
    ``transit_dist_m`` from the work centroid. ``ceil(total coverage energy /
    per-sortie coverage budget)``, floored at 1. Propagates
    ``InfeasibleMissionError`` from the budget when overhead eats the battery."""
    cov_len_m = coverage_path_length(area_m2, effective_swath_m, turn_factor)
    cov_energy_j = em.distance_energy(cov_len_m, ManeuverType.COVERAGE, spec.v_coverage)
    budget_j = per_sortie_coverage_budget_j(em, spec, transit_dist_m, altitude_m, reserve_frac)
    return max(1, math.ceil(cov_energy_j / budget_j))


def fleet_swaps(total_sorties_int: int, n_drones: int) -> int:
    """Number of battery swaps the fleet performs: every sortie beyond the first
    one per drone is a swap. ``total_sorties - min(n_drones, total_sorties)``."""
    if n_drones < 1:
        raise ValueError("n_drones must be >= 1")
    if total_sorties_int < 0:
        raise ValueError("total_sorties_int must be >= 0")
    n_active = min(n_drones, total_sorties_int)
    return total_sorties_int - n_active


# --------------------------------------------------------------------------- #
# staging region + candidate generation                                       #
# --------------------------------------------------------------------------- #
def _staging_region(env: EnvironmentMap):
    """The flyable STAGING band just OUTSIDE the survey area:
    ``area.buffer(standoff) minus area``.

    Obstacles are generated only inside ``area``, so the entire ring is clear
    ground -- the realizable "staging periphery" where the GCS/launch pad lives.
    Returns a (Multi)Polygon; may be used for ``.bounds`` / ``.covers``.
    """
    return env.area.buffer(_STAGING_STANDOFF_M).difference(env.area)


def _sample_staging(region, n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Rejection-sample ``n`` points uniformly within the staging ring."""
    if region.is_empty or n <= 0:
        return []
    minx, miny, maxx, maxy = region.bounds
    out: list[tuple[float, float]] = []
    guard = 0
    while len(out) < n and guard < 2000 * max(n, 1):
        guard += 1
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if region.covers(Point(x, y)):
            out.append((x, y))
    return out


def _candidate_sites(cfg: LaunchConfig, env: EnvironmentMap, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Candidate launch sites, STRICTLY in the staging periphery (outside the
    survey area). Any point inside ``env.area`` is discarded before scoring.

    * explicit ``candidate_sites`` list -> keep only those outside ``area``;
    * integer ``candidate_sites`` -> sample the staging ring, preferring points
      nearest the area boundary (shortest transit into the work region).
    """
    area = env.area
    if not isinstance(cfg.candidate_sites, int):
        # author-specified pads: drop any that fall inside the target polygon
        return [tuple(s) for s in cfg.candidate_sites if not area.contains(Point(s))]

    region = _staging_region(env)
    pool = _sample_staging(region, cfg.candidate_sites * 6, rng)
    if not pool:
        return []
    boundary = area.exterior
    pool.sort(key=lambda p: boundary.distance(Point(p)))  # closest to the area edge first
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

    # Workload constants shared by every candidate's exact-swap computation.
    # The work centroid is the navigable free-space centroid; transit overhead is
    # measured from each candidate to THIS point (site-specific).
    area_m2 = float(env.free_space.area)
    work_centroid = env.free_space.centroid
    wcx, wcy = work_centroid.x, work_centroid.y
    swath_m = spec.swath_width_m

    rows: list[dict] = []
    for s in candidates:
        # --- Batch 6.3 Part 1.1: feasibility gate FIRST; discard if infeasible ---
        furthest_m = _furthest_free_vertex_dist(env, s)
        if not furthest_point_feasible(em, spec, furthest_m, altitude_m):
            continue

        site_pose = Pose(s[0], s[1], math.atan2(cy - s[1], cx - s[0]))
        dists = [motion.leg_cost(site_pose, e) for e in entries]
        mean_dist = float(np.mean(dists)) if dists else math.inf

        # criterion 2 (unchanged): formation launch+transit-out+RTH-back energy
        initial_energy = 0.0
        for e, dist in zip(entries, dists):
            out_e = em.distance_energy(dist, ManeuverType.CRUISE, spec.v_cruise, f_form)
            back_e = em.distance_energy(dist, ManeuverType.CRUISE, spec.v_cruise, f_form)
            initial_energy += out_e + back_e + take.energy_j + land.energy_j

        # --- Batch 6.3 Part 1.2: EXACT fleet swaps via B6.2 workload math ---
        transit_site_m = math.hypot(s[0] - wcx, s[1] - wcy)
        try:
            sorties_int = required_sorties(
                em, spec, area_m2, transit_site_m, altitude_m, swath_m,
                TURN_FACTOR_DEFAULT, _RESERVE_FRAC,
            )
        except InfeasibleMissionError:
            # passes the reach-and-return gate but leaves no coverage budget from
            # here -> not a viable launch site for this workload.
            continue
        exact_swaps = fleet_swaps(sorties_int, n_drones)

        rows.append(
            {"site": s, "mean_dist": mean_dist, "energy": initial_energy,
             "swaps": float(exact_swaps), "pose": site_pose}
        )

    if not rows:
        usable = spec.battery_capacity_j * (1.0 - _RESERVE_FRAC)
        raise InfeasibleMissionError(
            "No staging-periphery launch site can both reach all navigable bounds "
            f"and retain a per-sortie coverage budget (usable {usable:.0f} J at "
            f"altitude {altitude_m:.0f} m; {len(candidates)} candidate sites "
            "evaluated). Increase battery capacity, lower the coverage altitude, "
            "shrink the area, or reduce the staging standoff."
        )

    # --- Batch 6.3 Part 1.3: min-max normalize over the FEASIBLE set only,    ---
    # --- with the exact swap count as the fatigue penalty term.               ---
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
    best_score = scores[0]
    best = next(r for r in rows if r["site"] == best_score.site)
    return best["pose"], scores
