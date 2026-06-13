"""Target-visit mission planning: the discrete-point counterpart of area coverage.

Where area coverage GROWS connected polygon zones (area proportional to battery)
and SWEEPS each with a boustrophedon path, a target-visit mission CLUSTERS a set
of discrete target points into per-drone subsets (COUNT proportional to battery
-- the same central contribution, applied to a different work unit) and ROUTES
each subset into a tour. The output is a per-drone ``CoveragePlan`` with
``leg_mode="tour"``, after which the agent, state machine, energy model, RTH,
swaps, failures and the whole SMDP/efficiency analysis are reused unchanged.

Allocation is deterministic and spatially coherent: targets are placed in a
global nearest-neighbour visiting order from the launch site, then sliced into
contiguous chunks whose sizes are the battery-weighted capacities. Each chunk is
then locally improved with a bounded 2-opt. A convex-hull "zone" per drone is
produced only so existing partition-based visualization/metrics keep working --
the real work unit is the target list.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from ..infrastructure.config import MissionConfig
from ..infrastructure.core_types import (
    CoveragePlan,
    DroneStateView,
    Partition,
    Pose,
    Waypoint,
    Zone,
)
from ..infrastructure.enums import DecompositionAlgo, ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# generation                                                                  #
# --------------------------------------------------------------------------- #
def generate_targets(env, mission_cfg: MissionConfig, rng: np.random.Generator) -> list[Point]:
    """Explicit coordinates if given (reproducible fixed map), else random free
    points (reproducible via the seeded 'targets' RNG stream)."""
    if mission_cfg.target_coordinates:
        return [(float(x), float(y)) for (x, y) in mission_cfg.target_coordinates]
    return [tuple(p) for p in env.sample_free(mission_cfg.n_targets, rng)]


# --------------------------------------------------------------------------- #
# allocation (battery-weighted, spatially coherent)                           #
# --------------------------------------------------------------------------- #
def capacities(views: list[DroneStateView], n_targets: int, weight_by_battery: bool) -> list[int]:
    """Per-drone target counts, aligned to ``views``, summing exactly to n_targets.
    Battery-weighted (count proportional to battery) when requested, else even."""
    k = len(views)
    if k == 0 or n_targets <= 0:
        return [0] * k
    if weight_by_battery:
        w = [max(0.0, v.battery_frac) for v in views]
        s = sum(w) or float(k)
        raw = [n_targets * (wi / s if s else 1.0 / k) for wi in w]
    else:
        raw = [n_targets / k] * k
    floors = [int(math.floor(r)) for r in raw]
    remainder = n_targets - sum(floors)
    # largest-remainder method, deterministic
    order = sorted(range(k), key=lambda i: (raw[i] - floors[i], -i), reverse=True)
    for j in range(remainder):
        floors[order[j % k]] += 1
    return floors


def _nn_order(points: list[Point], start: Point) -> list[int]:
    """Greedy nearest-neighbour visiting order of indices, from `start`."""
    n = len(points)
    if n == 0:
        return []
    unvisited = set(range(n))
    order: list[int] = []
    cur = start
    while unvisited:
        i = min(unvisited, key=lambda k: (points[k][0] - cur[0]) ** 2 + (points[k][1] - cur[1]) ** 2)
        order.append(i)
        unvisited.discard(i)
        cur = points[i]
    return order


def _two_opt(seq: list[Point], max_passes: int = 6, node_cap: int = 120) -> list[Point]:
    """Bounded 2-opt improvement of an open tour. Skipped for very large tours
    to keep planning fast and deterministic."""
    n = len(seq)
    if n < 4 or n > node_cap:
        return seq

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    seq = list(seq)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = seq[i], seq[i + 1]
                c = seq[j]
                e = seq[j + 1] if j + 1 < n else None
                before = d(a, b) + (d(c, e) if e else 0.0)
                after = d(a, c) + (d(b, e) if e else 0.0)
                if after + 1e-9 < before:
                    seq[i + 1:j + 1] = reversed(seq[i + 1:j + 1])
                    improved = True
        if not improved:
            break
    return seq


def assign_targets(
    targets: list[Point], views: list[DroneStateView], launch: Point, weight_by_battery: bool
) -> dict[int, list[Point]]:
    """Slice the global NN order into battery-weighted contiguous chunks -> one
    spatially-coherent target subset per drone. Every target is assigned exactly
    once; chunk sizes equal the battery-weighted capacities."""
    n = len(targets)
    caps = capacities(views, n, weight_by_battery)
    order = _nn_order(targets, launch)
    ordered = [targets[i] for i in order]
    out: dict[int, list[Point]] = {}
    pos = 0
    for v, c in zip(views, caps):
        out[v.id] = ordered[pos: pos + c]
        pos += c
    return out


# --------------------------------------------------------------------------- #
# routing (per-drone tour -> CoveragePlan with leg_mode="tour")               #
# --------------------------------------------------------------------------- #
def route_tour(
    drone_id: int, entry: Point, pts: list[Point], spec: PlatformSpec, em: EnergyModel
) -> CoveragePlan:
    if not pts:
        return CoveragePlan(drone_id, [], 0.0, 0.0, leg_mode="tour")
    seq = [pts[i] for i in _nn_order(pts, entry)]
    seq = _two_opt(seq)

    waypoints: list[Waypoint] = []
    for j, (x, y) in enumerate(seq):
        nxt = seq[j + 1] if j + 1 < len(seq) else seq[j - 1] if len(seq) > 1 else (x + 1.0, y)
        heading = math.atan2(nxt[1] - y, nxt[0] - x)
        waypoints.append(Waypoint(Pose(x, y, heading), ManeuverType.CRUISE, spec.v_cruise))

    # length/energy of the tour LEGS the agent flies (entry->first is transit, separate)
    length = 0.0
    energy = 0.0
    for a, b in zip(seq, seq[1:]):
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        length += dist
        energy += em.distance_energy(dist, ManeuverType.CRUISE, spec.v_cruise)
    return CoveragePlan(drone_id, waypoints, length, energy, leg_mode="tour")


def _viz_partition(assignment: dict[int, list[Point]], spec: PlatformSpec) -> Partition:
    """A convex-hull 'zone' per drone, only so partition-based plotting/metrics
    keep working. NOT the work unit -- the target list is."""
    zones: dict[int, Zone] = {}
    pad = max(spec.swath_width_m, 10.0)
    for aid, pts in assignment.items():
        if not pts:
            continue
        hull = MultiPoint(pts).convex_hull.buffer(pad)
        if not isinstance(hull, Polygon):
            hull = hull.convex_hull
        entry = Pose(pts[0][0], pts[0][1], 0.0)
        zones[aid] = Zone(drone_id=aid, regions=[], polygon=hull, entry_pose=entry)
    return Partition(DecompositionAlgo.WEIGHTED_VORONOI, zones, 0.0)


def plan_target_mission(
    targets: list[Point],
    views: list[DroneStateView],
    launch_pose: Pose,
    motion: MotionModel,
    spec: PlatformSpec,
    em: EnergyModel,
    weight_by_battery: bool = True,
) -> tuple[Partition, dict[int, CoveragePlan], dict[int, list[Point]]]:
    """Full target-visit plan: (viz partition, per-drone tour plans, assignment)."""
    launch = (launch_pose.x, launch_pose.y)
    assignment = assign_targets(targets, views, launch, weight_by_battery)
    plans = {v.id: route_tour(v.id, launch, assignment.get(v.id, []), spec, em) for v in views}
    partition = _viz_partition(assignment, spec)
    return partition, plans, assignment
