"""CENTRAL CONTRIBUTION: energy-weighted spatial decomposition.

Each drone's region area is proportional to its *momentary battery level*,
realized on the TGC region graph so it works in obstacle environments and stays
informative for identical drones (where an unweighted partition would carry no
fleet-state information). The area-proportionality target is met against exact
Shapely polygon areas; region *shapes* are the approximation, areas are exact.

``TgcBasicDecomposer`` is the switched-off twin (uniform weights) used as the
clean ablation baseline ``tgc_basic`` -- same machinery, weighting disabled, so
any measured difference vs. ``weighted_voronoi`` is attributable to the battery
weighting alone.
"""
from __future__ import annotations

import time

import networkx as nx
from shapely.geometry import Point, Polygon

from ..infrastructure.core_types import (
    DecompositionAlgo,
    DroneStateView,
    Partition,
    Region,
    Zone,
)
from .decomposition_base import (
    Decomposer,
    _WorkRegion,
    build_work_adjacency,
    build_zone,
    clip_regions,
    ensure_enough_regions,
    is_connected_subset,
)
from .environment_map import EnvironmentMap
from .tgc import TGCGraph


class WeightedTgcDecomposer(Decomposer):
    name = DecompositionAlgo.WEIGHTED_VORONOI

    def __init__(self, refine_iters: int = 50) -> None:
        self._refine_iters = refine_iters

    # weighting -- the one method TgcBasic overrides
    def weights(self, drones: list[DroneStateView]) -> dict[int, float]:
        total = sum(max(0.0, d.battery_frac) for d in drones)
        if total <= 0:
            return {d.id: 1.0 / len(drones) for d in drones}
        return {d.id: max(0.0, d.battery_frac) / total for d in drones}

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
        adj = build_work_adjacency(work)
        by_id = {w.id: w for w in work}
        a_total = sum(w.area for w in work)
        weights = self.weights(drones)
        targets = {d.id: weights[d.id] * a_total for d in drones}

        assigned, owner = self._seed(drones, work, adj)
        area = {d.id: sum(by_id[r].area for r in assigned[d.id]) for d in drones}
        self._grow(drones, assigned, owner, area, targets, by_id, adj)
        self._refine(drones, assigned, owner, area, targets, by_id, adj)

        # map work-region ids back to TGC Region objects where possible
        tgc_by_id = {r.id: r for r in tgc.regions}
        zones: dict[int, Zone] = {}
        for d in drones:
            geoms = [by_id[r].geom for r in assigned[d.id]]
            regions: list[Region] = [
                tgc_by_id.get(r) or Region(r, by_id[r].geom, by_id[r].area, by_id[r].anchor)
                for r in assigned[d.id]
            ]
            zones[d.id] = build_zone(d.id, regions, geoms, d.pose)
        return Partition(self.name, zones, time.perf_counter() - t0)

    # ---- internals -------------------------------------------------------- #
    def _seed(self, drones, work, adj):
        assigned: dict[int, set[int]] = {d.id: set() for d in drones}
        owner: dict[int, int] = {}
        used: set[int] = set()
        for d in drones:
            seed = self._nearest_free_region(d.pose, work, used)
            if seed is None:
                continue
            assigned[d.id].add(seed)
            owner[seed] = d.id
            used.add(seed)
        return assigned, owner

    @staticmethod
    def _nearest_free_region(pose, work, used):
        best, best_d = None, float("inf")
        px, py = pose.x, pose.y
        for w in work:
            if w.id in used:
                continue
            # prefer containment, else nearest anchor
            if w.geom.covers(Point(px, py)):
                return w.id
            d = (w.anchor.x - px) ** 2 + (w.anchor.y - py) ** 2
            if d < best_d:
                best_d, best = d, w.id
        return best

    def _grow(self, drones, assigned, owner, area, targets, by_id, adj):
        all_ids = set(by_id)
        unassigned = all_ids - set(owner)
        drone_ids = [d.id for d in drones]
        while unassigned:
            # drone with the largest remaining deficit that has a frontier
            cand = sorted(drone_ids, key=lambda i: targets[i] - area[i], reverse=True)
            moved = False
            for i in cand:
                if targets[i] - area[i] <= 0 and any(targets[j] - area[j] > 0 for j in drone_ids):
                    continue
                frontier = self._frontier(assigned[i], unassigned, adj)
                if not frontier:
                    continue
                # nearest frontier region by adjacency weight to the current zone
                r_star = min(frontier, key=lambda r: self._edge_dist(adj, assigned[i], r))
                assigned[i].add(r_star)
                owner[r_star] = i
                used_area = by_id[r_star].area
                area[i] += used_area
                unassigned.discard(r_star)
                moved = True
                break
            if not moved:
                # connectivity fallback: give the neediest drone the nearest leftover
                i = max(drone_ids, key=lambda i: targets[i] - area[i])
                r_star = min(
                    unassigned,
                    key=lambda r: min(
                        (by_id[r].anchor.x - by_id[s].anchor.x) ** 2
                        + (by_id[r].anchor.y - by_id[s].anchor.y) ** 2
                        for s in (assigned[i] or {next(iter(unassigned))})
                    ),
                )
                assigned[i].add(r_star)
                owner[r_star] = i
                area[i] += by_id[r_star].area
                unassigned.discard(r_star)

    @staticmethod
    def _frontier(zone_ids, unassigned, adj):
        out = set()
        for r in zone_ids:
            for nb in adj.neighbors(r):
                if nb in unassigned:
                    out.add(nb)
        return out

    @staticmethod
    def _edge_dist(adj, zone_ids, r):
        best = float("inf")
        for z in zone_ids:
            if adj.has_edge(z, r):
                best = min(best, adj.edges[z, r]["weight"])
        return best

    def _refine(self, drones, assigned, owner, area, targets, by_id, adj):
        def deviation():
            return sum(abs(area[d.id] - targets[d.id]) for d in drones)

        for _ in range(self._refine_iters):
            improved = False
            for r, i in list(owner.items()):
                # try to move boundary region r from i to an adjacent owner j
                for nb in adj.neighbors(r):
                    j = owner.get(nb)
                    if j is None or j == i:
                        continue
                    if len(assigned[i]) <= 1:
                        continue
                    before = abs(area[i] - targets[i]) + abs(area[j] - targets[j])
                    a = by_id[r].area
                    after = abs((area[i] - a) - targets[i]) + abs((area[j] + a) - targets[j])
                    if after + 1e-9 >= before:
                        continue
                    # connectivity must hold for both zones after the move
                    new_i = assigned[i] - {r}
                    new_j = assigned[j] | {r}
                    if not is_connected_subset(adj, new_i) or not is_connected_subset(adj, new_j):
                        continue
                    assigned[i], assigned[j] = new_i, new_j
                    owner[r] = j
                    area[i] -= a
                    area[j] += a
                    improved = True
                    break
            if not improved:
                break


class TgcBasicDecomposer(WeightedTgcDecomposer):
    """Ablation baseline: identical machinery, uniform (battery-independent)
    weights. Any difference vs. the weighted decomposer isolates the effect of
    the battery weighting."""

    name = DecompositionAlgo.TGC_BASIC

    def weights(self, drones: list[DroneStateView]) -> dict[int, float]:
        return {d.id: 1.0 / len(drones) for d in drones}
