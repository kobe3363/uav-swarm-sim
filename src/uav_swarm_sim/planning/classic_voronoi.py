"""Baseline 1 -- ``classic_voronoi``.

Plain Euclidean Voronoi partition by drone seed points: ignores battery and the
obstacle topology. This is the foil whose workload imbalance the thesis
quantifies; connectivity of zones is not guaranteed (by design).
"""
from __future__ import annotations

import time

from shapely.geometry import Point, Polygon

from ..infrastructure.core_types import (
    DecompositionAlgo,
    DroneStateView,
    Partition,
    Region,
    Zone,
)
from .decomposition_base import Decomposer, build_zone, clip_regions, ensure_enough_regions
from .environment_map import EnvironmentMap
from .tgc import TGCGraph


class ClassicVoronoiDecomposer(Decomposer):
    name = DecompositionAlgo.CLASSIC_VORONOI

    def decompose(
        self,
        tgc: TGCGraph,
        env: EnvironmentMap,
        drones: list[DroneStateView],
        target_area: Polygon | None = None,
    ) -> Partition:
        t0 = time.perf_counter()
        work = clip_regions(tgc, target_area)
        work = ensure_enough_regions(work, len(drones))
        tgc_by_id = {r.id: r for r in tgc.regions}

        assigned: dict[int, list] = {d.id: [] for d in drones}
        for w in work:
            # nearest drone seed (Euclidean) to the region anchor
            d = min(drones, key=lambda dr: (dr.pose.x - w.anchor.x) ** 2 + (dr.pose.y - w.anchor.y) ** 2)
            assigned[d.id].append(w)

        zones: dict[int, Zone] = {}
        for d in drones:
            geoms = [w.geom for w in assigned[d.id]]
            regions = [
                tgc_by_id.get(w.id) or Region(w.id, w.geom, w.area, w.anchor)
                for w in assigned[d.id]
            ]
            zones[d.id] = build_zone(d.id, regions, geoms, d.pose)
        return Partition(self.name, zones, time.perf_counter() - t0)
