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


@dataclass(frozen=True)
class SiteScore:
    site: tuple[float, float]
    mean_dist_m: float
    initial_energy_j: float
    expected_swaps: float
    J: float


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
    e_usable = spec.battery_capacity_j * (1.0 - 0.05)  # small reserve eps

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

        rows.append(
            {"site": s, "mean_dist": mean_dist, "energy": initial_energy, "swaps": expected_swaps,
             "pose": site_pose}
        )

    # min-max normalize each criterion, then weighted sum J
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
    best = next(r for r in rows if r["site"] == scores[0].site)
    return best["pose"], scores
