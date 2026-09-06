"""EXP-07: shared Lloyd / power-diagram partitioner over the EXP-02 coverage grid.

One partitioning code path, one pluggable weight source. ``LLOYD_CVT`` pins the
weights to zero -- that is the Voronoi/Lloyd reference arm. ``LLOYD_ENERGY``
(EXP-07b) will reuse this file unchanged and supply a weight vector instead.
Initial conditions, tie-breaking, the eligible-cell set, the centroid update, the
conservation asserts and the convergence logic are shared *by construction*, so
the two identities can differ ONLY in their weight source.

Assignment rule -- an additive power diagram::

    owner(c) = argmin_i ( ||p_c - s_i||^2 - w_i )      over the ALLOWED i

``w_i`` is in **m^2**: it is subtracted from a squared distance. It is not a
battery percentage, not a ratio, and not a normalised share.

Which point is which
--------------------
Each cell carries TWO points and they answer different questions.

*Area centroid* (``shapely.centroid``, computed here) is the cell's MASS
position. Lloyd's fixed point is the area-weighted centroid, so the assignment
and the centroid update use it. It is the right answer even when it falls
outside its own cell -- the centroid of a union is ``sum(a*c)/sum(a)`` however
the individual centroids lie.

*Surface point* (``shapely.point_on_surface``, from the raster) is guaranteed to
be INSIDE the cell. Every MEMBERSHIP question -- which free-space component is
this cell in, is its return cost defined -- uses it, because a clipped cell can
be concave (an obstacle notching one edge) and then its area centroid sits in the
obstacle. Asking "where is this cell?" with a point that is not in it drops
flyable work.

Determinism
-----------
No RNG anywhere. Sites are held in ascending drone-id order and ``np.argmin``
returns the FIRST minimum, so every distance tie -- coincident sites, equal
weights, a cell exactly on a bisector -- resolves to the lowest drone id.
Weights are keyed by drone id, never by array position, so they survive any
later cluster re-indexing.

Explicit failure
----------------
The iteration count is bounded and non-convergence is REPORTED
(``PartitionDiagnostics.converged``), never repaired by falling back to another
algorithm. Cells that no drone may own are dropped and counted, never handed
silently to the nearest drone.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Polygon

from ..infrastructure.config import PartitionConfig
from ..infrastructure.core_types import (
    DecompositionAlgo,
    DroneStateView,
    Partition,
    Pose,
    Zone,
)
from .coverage_raster import CoverageRaster
from .decomposition_base import Decomposer, build_zone
from .environment_map import EnvironmentMap
from .tgc import TGCGraph

INIT_DEPLOY_POSES = "deploy_poses"
INIT_MAXIMIN = "maximin"

# Cells processed per assignment block. The cost matrix is (n_cells x n_drones)
# float64; the default survey holds ~645k plannable cells at raster_cell_m = 2.0,
# so an unchunked 8-drone matrix would allocate ~41 MB every iteration. Chunking
# caps the peak without changing a single assignment.
_CELL_BLOCK = 200_000

# Area conservation is exact up to float summation order.
_AREA_REL_TOL = 1e-9


@dataclass
class PartitionDiagnostics:
    """Explicit outcome record, persisted into the run output."""
    algorithm: str
    decomposer_class: str
    converged: bool
    iterations: int
    max_site_shift_m: float
    settings: dict
    cells: dict
    per_drone: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "decomposer_class": self.decomposer_class,
            "converged": self.converged,
            "iterations": self.iterations,
            "max_site_shift_m": self.max_site_shift_m,
            "settings": dict(self.settings),
            "cells": dict(self.cells),
            "per_drone": {str(k): dict(v) for k, v in self.per_drone.items()},
        }


@dataclass(frozen=True)
class EligibleCells:
    """The work atoms: clipped plannable cells at least one drone may own."""
    geometries: np.ndarray      # clipped (cell n plannable) polygons
    centroids_xy: np.ndarray    # (N, 2) AREA centroids, computed here
    areas_m2: np.ndarray        # (N,) the raster's own clipped areas
    component: np.ndarray       # (N,) free-space connected-component id
    n_rth_unreachable: int      # dropped: E_home non-finite at the cell
    n_no_eligible_owner: int    # dropped: no drone shares the cell's component

    @property
    def count(self) -> int:
        return len(self.areas_m2)

    @property
    def total_area_m2(self) -> float:
        return float(self.areas_m2.sum())


# --------------------------------------------------------------------------- #
# eligible-cell construction (the two HARD reachability constraints)           #
# --------------------------------------------------------------------------- #
def _free_space_parts(env: EnvironmentMap | None) -> list[Polygon]:
    """Connected components of the flyable space, in a fixed order.

    The ordinary case -- obstacles strictly inside the survey -- leaves
    ``plannable_space`` a single ``Polygon``, one component, which makes the
    reachability constraint below INERT: no cell and no drone can be excluded.
    """
    if env is None:
        return []
    space = getattr(env, "plannable_space", None)
    if space is None or space.is_empty:
        return []
    if isinstance(space, Polygon):
        return [space]
    return [g for g in getattr(space, "geoms", []) if isinstance(g, Polygon) and g.area > 0.0]


def _component_of_points(parts: list[Polygon], xy: np.ndarray) -> np.ndarray:
    """Component id per point; -1 when the point lies in none of them.

    With no component information at all -- no env, or an empty flyable space --
    every point is labelled 0, which makes the constraint INERT. Labelling them
    -1 instead would make it maximally exclusive and silently strand the whole
    survey, which is the opposite of the intended failure mode.
    """
    if not parts or len(parts) == 1 or len(xy) == 0:
        return np.zeros(len(xy), dtype=np.int32)
    labels = np.full(len(xy), -1, dtype=np.int32)
    tree = STRtree(np.array(parts, dtype=object))
    hit_point, hit_part = tree.query(shapely.points(xy), predicate="intersects")
    # Lowest component index wins when a point sits exactly on a shared edge.
    order = np.lexsort((hit_part, hit_point))
    hit_point, hit_part = hit_point[order], hit_part[order]
    first = np.ones(len(hit_point), dtype=bool)
    first[1:] = hit_point[1:] != hit_point[:-1]
    labels[hit_point[first]] = hit_part[first].astype(np.int32)
    return labels


def _drone_components(parts: list[Polygon], poses: np.ndarray) -> np.ndarray:
    """Component per drone, falling back to the NEAREST component when a staging
    pose is inside none of them (the deploy ring can graze an obstacle buffer).
    Deterministic: ties take the lowest component index."""
    labels = _component_of_points(parts, poses)
    if len(parts) <= 1:
        return labels
    for i in np.flatnonzero(labels < 0):
        point = shapely.points(poses[i])
        labels[i] = int(np.argmin([float(shapely.distance(point, p)) for p in parts]))
    return labels


def _rth_reachable_mask(energy_map, inside_xy: np.ndarray) -> tuple[np.ndarray, int]:
    """Cells whose return cost is defined.

    Mirrors the EXP-06 test exactly (``energy_balance._estimate``): a cell
    OUTSIDE the grid is not judged, a cell inside it must have a finite
    ``e_home``. With no map built, nothing is filtered.

    ``inside_xy`` must be points KNOWN to lie in their cells (the raster's
    surface points), not area centroids: a concave clipped cell's centroid can
    sit in the obstacle, whose ``e_home`` is non-finite, which would drop
    perfectly flyable work.
    """
    keep = np.ones(len(inside_xy), dtype=bool)
    if energy_map is None or len(inside_xy) == 0:
        return keep, 0
    frame = energy_map.frame
    i = np.floor((inside_xy[:, 0] - frame.origin_x) / frame.cell_m).astype(np.int64)
    j = np.floor((inside_xy[:, 1] - frame.origin_y) / frame.cell_m).astype(np.int64)
    inside = (i >= 0) & (i < frame.nx) & (j >= 0) & (j < frame.ny)
    if inside.any():
        keep[inside] = np.isfinite(np.asarray(energy_map.e_home)[i[inside], j[inside]])
    return keep, int((~keep).sum())


def build_eligible_cells(
    raster: CoverageRaster,
    env: EnvironmentMap | None,
    drone_poses: np.ndarray,
    energy_map=None,
) -> tuple[EligibleCells, np.ndarray]:
    """Uncovered plannable cells minus the two hard reachability constraints.

    Returns the cell set plus the per-drone component labels. Nothing is dropped
    silently: both exclusion counts travel into the diagnostics.
    """
    parts = _free_space_parts(env)
    drone_comp = _drone_components(parts, drone_poses)

    cells = raster.uncovered_plannable_cells()
    geoms = cells.geometries
    if len(geoms) == 0:
        return (
            EligibleCells(geoms, np.zeros((0, 2)), cells.areas_m2,
                          np.zeros(0, dtype=np.int32), 0, 0),
            drone_comp,
        )

    # Two points per cell, each for its own question (see the module docstring):
    # the AREA centroid carries the mass, the raster's surface point is the one
    # guaranteed to lie inside the cell and so answers "where is this cell?".
    centroid = shapely.centroid(geoms)
    xy = np.column_stack((shapely.get_x(centroid), shapely.get_y(centroid)))
    inside = np.column_stack(
        (shapely.get_x(cells.surface_points), shapely.get_y(cells.surface_points))
    )

    keep, n_rth = _rth_reachable_mask(energy_map, inside)
    geoms, xy, inside, areas = geoms[keep], xy[keep], inside[keep], cells.areas_m2[keep]

    cell_comp = _component_of_points(parts, inside)
    reachable_components = np.unique(drone_comp[drone_comp >= 0])
    owned = np.isin(cell_comp, reachable_components)
    n_orphan = int((~owned).sum())
    return (
        EligibleCells(geoms[owned], xy[owned], areas[owned],
                      cell_comp[owned].astype(np.int32), n_rth, n_orphan),
        drone_comp,
    )


# --------------------------------------------------------------------------- #
# initial sites                                                                #
# --------------------------------------------------------------------------- #
def maximin_site_order(xy: np.ndarray, launch_xy: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point sampling over the eligible cell centroids.

    ``s_1`` is the cell nearest the launch pose; each later site maximises its
    minimum distance to those already chosen. ``np.argmin``/``np.argmax`` return
    the FIRST extremum, so every tie resolves to the lowest cell index. No RNG:
    the only input is the cell set itself.

    This is NOT a "correct" initialisation. Farthest-point sampling has its own
    artifact -- sites are pulled toward the domain corners -- and it does not
    yield a global CVT optimum. It is only far better spread than the <18 m
    staging ring. The scientific defence is the init-sites ablation, not this
    choice.
    """
    chosen = np.empty(min(k, len(xy)), dtype=np.int64)
    if len(chosen) == 0:
        return chosen
    chosen[0] = int(np.argmin(((xy - launch_xy) ** 2).sum(axis=1)))
    best = ((xy - xy[chosen[0]]) ** 2).sum(axis=1)
    for m in range(1, len(chosen)):
        chosen[m] = int(np.argmax(best))
        np.minimum(best, ((xy - xy[chosen[m]]) ** 2).sum(axis=1), out=best)
    return chosen


def initial_sites(
    policy: str, drone_poses: np.ndarray, cells: EligibleCells, launch_pose: Pose
) -> np.ndarray:
    """Initial site per drone, rows in ascending drone-id order.

    ``deploy_poses`` (the default) reproduces the drone-pose seeding convention
    the existing position-based decomposers use, so switching the partitioner on
    does not silently change where the sweep starts.

    ``maximin`` spreads the sites over the work itself and then matches sites to
    drones by a deterministic OPTIMAL assignment -- minimum total squared
    distance to the staging poses -- rather than greedily in id order. At t=0 the
    difference is negligible (every drone sits on a <18 m ring), but once the
    drones are spread out (EXP-08 re-partition) a greedy id-order match would
    inject an ordering artifact straight into the ferry energies. The rule is
    defined once here and is identical for LLOYD_CVT and LLOYD_ENERGY.
    """
    if policy == INIT_DEPLOY_POSES:
        return drone_poses.copy()
    if policy != INIT_MAXIMIN:
        raise ValueError(f"unknown planning.partition.init_sites: {policy!r}")

    sites = drone_poses.copy()
    order = maximin_site_order(
        cells.centroids_xy, np.asarray(launch_pose.as_xy(), dtype=float), len(drone_poses)
    )
    if len(order) == 0:
        return sites          # no eligible work at all: staging poses stand in
    from scipy.optimize import linear_sum_assignment

    picked = cells.centroids_xy[order]
    cost = ((drone_poses[:, None, :] - picked[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    # Fewer eligible cells than drones: the surplus drones keep their staging
    # pose. Deterministic, and conservation-safe -- an empty zone is legal.
    sites[rows] = picked[cols]
    return sites


# --------------------------------------------------------------------------- #
# the shared partitioning core                                                 #
# --------------------------------------------------------------------------- #
def assign_cells(
    centroids_xy: np.ndarray, sites: np.ndarray, weights: np.ndarray, allowed: np.ndarray
) -> np.ndarray:
    """``argmin_i(||p_c - s_i||^2 - w_i)`` over the allowed drones, per cell.

    ``sites``/``weights`` are ordered by ascending drone id and ``np.argmin``
    returns the first minimum, so a tie breaks toward the LOWEST drone id.
    ``allowed`` is the (n_cells, n_drones) reachability mask; a disallowed pair
    is pushed to +inf and can never win, because ``build_eligible_cells``
    guarantees every surviving cell at least one allowed drone.
    """
    n_cells = len(centroids_xy)
    labels = np.empty(n_cells, dtype=np.int64)
    for start in range(0, n_cells, _CELL_BLOCK):
        stop = min(start + _CELL_BLOCK, n_cells)
        block = centroids_xy[start:stop]
        cost = ((block[:, None, :] - sites[None, :, :]) ** 2).sum(axis=2) - weights[None, :]
        cost[~allowed[start:stop]] = np.inf
        labels[start:stop] = np.argmin(cost, axis=1)
    return labels


def aggregate(
    labels: np.ndarray, cells: EligibleCells, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-drone cell count, owned area and AREA-WEIGHTED centroid.

    ``sum(a_c * p_c) / sum(a_c)`` over a drone's cells is exactly the centroid of
    their union -- the Lloyd fixed point. An empty zone yields a non-finite
    centroid, which the caller reads as "this site does not move".
    """
    counts = np.bincount(labels, minlength=n).astype(np.int64)
    area = np.bincount(labels, weights=cells.areas_m2, minlength=n)
    wx = np.bincount(labels, weights=cells.areas_m2 * cells.centroids_xy[:, 0], minlength=n)
    wy = np.bincount(labels, weights=cells.areas_m2 * cells.centroids_xy[:, 1], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroids = np.column_stack((wx / area, wy / area))
    return counts, area, centroids


class UniformWeightPolicy:
    """LLOYD_CVT: weights pinned to zero, forever.

    With this policy the partitioner degenerates to plain Lloyd/CVT over the
    coverage grid. It performs NO energy evaluation at all, so the reference arm
    never pays for machinery it does not use -- and, being the same code with
    ``w == 0``, it is bit-identical to LLOYD_ENERGY whenever that arm's weights
    are pinned uniform. Note the converse does NOT follow: equal initial SoC does
    not imply equal energy weights, because predicted demand varies with zone
    geometry (ferry, RTH, area).
    """
    name = "uniform"

    def initial(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=float)

    def update(self, weights: np.ndarray, counts, area, centroids, sites) -> np.ndarray:
        return weights

    def balanced(self) -> bool:
        return True

    def per_drone(self, drone_ids: list[int]) -> dict:
        return {}


class LloydPartitioner:
    """The shared code path. The identity lives entirely in ``weight_policy``."""

    def __init__(self, settings: PartitionConfig, weight_policy) -> None:
        self.settings = settings
        self.weight_policy = weight_policy

    def run(
        self, cells: EligibleCells, drone_poses: np.ndarray, drone_comp: np.ndarray,
        launch_pose: Pose,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
        n = len(drone_poses)
        sites = initial_sites(self.settings.init_sites, drone_poses, cells, launch_pose)
        weights = self.weight_policy.initial(n)
        if cells.count == 0 or n == 0:
            return np.empty(0, dtype=np.int64), sites, weights, True, 0, 0.0

        allowed = cells.component[:, None] == drone_comp[None, :]
        assert allowed.any(axis=1).all(), "build_eligible_cells must drop unowned cells"

        converged, shift, iterations = False, 0.0, 0
        for iterations in range(1, self.settings.max_iterations + 1):
            labels = assign_cells(cells.centroids_xy, sites, weights, allowed)
            counts, area, centroids = aggregate(labels, cells, n)
            moved = np.isfinite(centroids).all(axis=1)     # an empty zone stays put
            previous = sites
            sites = np.where(moved[:, None], centroids, sites)
            weights = self.weight_policy.update(weights, counts, area, centroids, sites)
            shift = float(np.max(np.hypot(*(sites - previous).T)))
            if shift <= self.settings.site_tolerance_m and self.weight_policy.balanced():
                converged = True
                break
        # One final assignment so the returned zones are consistent with the
        # returned sites and weights. The last partition is USED either way: a
        # non-convergence is REPORTED, never repaired by another algorithm.
        labels = assign_cells(cells.centroids_xy, sites, weights, allowed)
        return labels, sites, weights, converged, iterations, shift


# --------------------------------------------------------------------------- #
# Decomposer identities                                                        #
# --------------------------------------------------------------------------- #
class _LloydDecomposer(Decomposer):
    """Adapter onto the frozen ``Decomposer`` contract.

    The ABC signature ``decompose(tgc, env, drones, target_area)`` is NOT
    changed. Everything this partitioner needs beyond it -- the raster, the
    staging poses, the launch pose, the energy map -- arrives through the
    constructor, exactly the way ``KMeansHeuristicDecomposer`` takes its motion
    model and RNG. ``tgc`` is deliberately unused: the work atoms are coverage
    grid cells, not TGC regions.
    """

    def __init__(
        self, *, raster: CoverageRaster, deploy_poses: list[Pose], launch_pose: Pose,
        settings: PartitionConfig, energy_map=None,
    ) -> None:
        self._raster = raster
        self._deploy_poses = list(deploy_poses)
        self._launch_pose = launch_pose
        self._settings = settings
        self._energy_map = energy_map
        self.diagnostics: PartitionDiagnostics | None = None

    def _weight_policy(self):
        raise NotImplementedError

    def decompose(
        self, tgc: TGCGraph, env: EnvironmentMap, drones: list[DroneStateView],
        target_area: Polygon | None = None,
    ) -> Partition:
        t0 = time.perf_counter()
        ordered = sorted(drones, key=lambda d: d.id)          # sites in id order
        ids = [d.id for d in ordered]
        poses = np.array(
            [[self._deploy_poses[d.id].x, self._deploy_poses[d.id].y] for d in ordered],
            dtype=float,
        ).reshape(len(ordered), 2)

        cells, drone_comp = build_eligible_cells(self._raster, env, poses, self._energy_map)
        policy = self._weight_policy()
        labels, sites, weights, converged, iterations, shift = LloydPartitioner(
            self._settings, policy
        ).run(cells, poses, drone_comp, self._launch_pose)

        counts, area, _ = aggregate(labels, cells, len(ordered))

        # Conservation: every eligible cell has exactly one owner and no area is
        # lost. Nothing is dropped for being empty or disconnected.
        assert int(counts.sum()) == cells.count, "cell count not conserved"
        total = cells.total_area_m2
        assert math.isclose(float(area.sum()), total, rel_tol=_AREA_REL_TOL, abs_tol=1e-9), (
            "cell area not conserved"
        )

        zones: dict[int, Zone] = {}
        for position, drone in enumerate(ordered):
            owned = cells.geometries[labels == position] if cells.count else []
            merged = shapely.union_all(owned) if len(owned) else Polygon()
            zones[drone.id] = build_zone(drone.id, [], [merged], drone.pose)

        extra = policy.per_drone(ids)
        self.diagnostics = PartitionDiagnostics(
            algorithm=self.name.value,
            decomposer_class=type(self).__name__,
            converged=converged,
            iterations=iterations,
            max_site_shift_m=shift,
            settings={
                "init_sites": self._settings.init_sites,
                "max_iterations": self._settings.max_iterations,
                "site_tolerance_m": self._settings.site_tolerance_m,
                "weight_policy": policy.name,
            },
            cells={
                "eligible": cells.count,
                "assigned": int(counts.sum()),
                "rth_unreachable": cells.n_rth_unreachable,
                "no_eligible_owner": cells.n_no_eligible_owner,
                "area_m2": total,
            },
            per_drone={
                drone_id: {
                    "n_cells": int(counts[position]),
                    "area_m2": float(area[position]),
                    "weight_m2": float(weights[position]),
                    "site_xy": [float(sites[position][0]), float(sites[position][1])],
                    **extra.get(drone_id, {}),
                }
                for position, drone_id in enumerate(ids)
            },
        )
        return Partition(self.name, zones, time.perf_counter() - t0)


class LloydCvtDecomposer(_LloydDecomposer):
    """``lloyd_cvt`` -- uniform weights: the Lloyd/CVT reference arm."""
    name = DecompositionAlgo.LLOYD_CVT

    def _weight_policy(self):
        return UniformWeightPolicy()
