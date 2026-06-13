"""The <=15-drone tier -- C. Liu's coupled heuristic.

K-means clustering of the work (area-weighted) with greedy drone-to-cluster
assignment on a motion cost matrix (Dubins flyable length for FW/VTOL, Euclidean
for multirotor). Trajectory shape is therefore considered already at allocation
time. ``weighted=True`` uses capacity-constrained clustering whose target mass is
proportional to battery level (the heuristic-tier mirror of the central weighting).
"""
from __future__ import annotations

import time

import numpy as np
from shapely.geometry import Polygon

from ..infrastructure.core_types import (
    DecompositionAlgo,
    DroneStateView,
    Partition,
    Region,
    Zone,
)
from ..physical_model.motion_model import MotionModel
from .decomposition_base import Decomposer, build_zone, clip_regions, ensure_enough_regions
from .environment_map import EnvironmentMap
from .tgc import TGCGraph


class KMeansHeuristicDecomposer(Decomposer):
    name = DecompositionAlgo.WEIGHTED_VORONOI  # heuristic realization of the weighted partition

    def __init__(self, motion: MotionModel, weighted: bool, rng: np.random.Generator, iters: int = 50) -> None:
        self._motion = motion
        self._weighted = weighted
        self._rng = rng
        self._iters = iters

    def decompose(
        self,
        tgc: TGCGraph,
        env: EnvironmentMap,
        drones: list[DroneStateView],
        target_area: Polygon | None = None,
    ) -> Partition:
        t0 = time.perf_counter()
        n = len(drones)
        work = clip_regions(tgc, target_area)
        work = ensure_enough_regions(work, n)
        tgc_by_id = {r.id: r for r in tgc.regions}

        pts = np.array([[w.anchor.x, w.anchor.y] for w in work])
        mass = np.array([w.area for w in work])
        labels = self._kmeans(pts, mass, n, drones)

        # cluster -> entry point (mass centroid), then greedy drone<->cluster on motion cost
        centroids = []
        for c in range(n):
            m = labels == c
            if m.any():
                centroids.append(np.average(pts[m], axis=0, weights=mass[m]))
            else:
                centroids.append(pts[self._rng.integers(0, len(pts))])
        cluster_to_drone = self._greedy_assign(drones, centroids)

        assigned: dict[int, list] = {d.id: [] for d in drones}
        for w, c in zip(work, labels):
            assigned[cluster_to_drone[int(c)]].append(w)

        zones: dict[int, Zone] = {}
        for d in drones:
            geoms = [w.geom for w in assigned[d.id]]
            regions = [tgc_by_id.get(w.id) or Region(w.id, w.geom, w.area, w.anchor) for w in assigned[d.id]]
            zones[d.id] = build_zone(d.id, regions, geoms, d.pose)
        return Partition(DecompositionAlgo.WEIGHTED_VORONOI, zones, time.perf_counter() - t0)

    # ---- k-means ---------------------------------------------------------- #
    def _kmeans(self, pts, mass, k, drones):
        centers = self._kmeanspp_init(pts, mass, k)
        targets = self._mass_targets(mass.sum(), drones) if self._weighted else None
        labels = np.zeros(len(pts), dtype=int)
        for _ in range(self._iters):
            d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = d2.argmin(axis=1)
            if self._weighted and targets is not None:
                new_labels = self._rebalance(pts, mass, centers, new_labels, targets)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = np.average(pts[m], axis=0, weights=mass[m])
        return labels

    def _kmeanspp_init(self, pts, mass, k):
        idx = [int(self._rng.integers(0, len(pts)))]
        for _ in range(1, k):
            d2 = np.min(((pts[:, None, :] - pts[idx][None, :, :]) ** 2).sum(axis=2), axis=1)
            probs = (d2 * mass)
            s = probs.sum()
            if s <= 0:
                idx.append(int(self._rng.integers(0, len(pts))))
            else:
                idx.append(int(self._rng.choice(len(pts), p=probs / s)))
        return pts[idx].astype(float)

    def _mass_targets(self, total_mass, drones):
        bs = np.array([max(0.0, d.battery_frac) for d in drones])
        if bs.sum() <= 0:
            bs = np.ones(len(drones))
        return total_mass * bs / bs.sum()

    def _rebalance(self, pts, mass, centers, labels, targets):
        k = len(centers)
        for _ in range(k):
            cluster_mass = np.array([mass[labels == c].sum() for c in range(k)])
            over = np.where(cluster_mass > targets * 1.05)[0]
            under = np.where(cluster_mass < targets * 0.95)[0]
            if len(over) == 0 or len(under) == 0:
                break
            for c in over:
                members = np.where(labels == c)[0]
                if len(members) <= 1:
                    continue
                # move the member nearest to an under-target center
                for u in under:
                    dists = ((pts[members] - centers[u]) ** 2).sum(axis=1)
                    mv = members[int(dists.argmin())]
                    labels[mv] = u
                    break
        return labels

    def _greedy_assign(self, drones, centroids):
        n = len(drones)
        cost = np.zeros((n, n))
        for ci, c in enumerate(centroids):
            from ..infrastructure.core_types import Pose
            goal = Pose(float(c[0]), float(c[1]), 0.0)
            for di, d in enumerate(drones):
                cost[ci, di] = self._motion.leg_cost(d.pose, goal)
        cluster_to_drone: dict[int, int] = {}
        taken_c, taken_d = set(), set()
        pairs = sorted(((cost[ci, di], ci, di) for ci in range(n) for di in range(n)))
        for _, ci, di in pairs:
            if ci in taken_c or drones[di].id in taken_d:
                continue
            cluster_to_drone[ci] = drones[di].id
            taken_c.add(ci)
            taken_d.add(drones[di].id)
        return cluster_to_drone
