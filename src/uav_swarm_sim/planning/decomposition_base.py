"""The contract every decomposition algorithm fulfills, plus shared helpers.

A Decomposer maps (TGC regions, environment, drone views[, target area]) to a
Partition that (a) covers the target area, (b) keeps each drone's zone connected,
(c) records its own planning time, (d) sets each zone's entry pose. The shared
helpers below build a working adjacency from clipped geometries, optionally
subdivide regions when there are fewer regions than drones (the granularity
guard), assemble zones, and compute the imbalance diagnostic.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union

from ..infrastructure.core_types import (
    DecompositionAlgo,
    DroneStateView,
    Partition,
    Pose,
    Region,
    Zone,
)
from ..infrastructure.enums import DecompositionAlgo as _Algo  # noqa: F401  (re-export friendliness)
from .environment_map import EnvironmentMap
from .tgc import TGCGraph

_TOUCH_TOL = 1e-6


def _polygon_parts(geom) -> list[Polygon]:
    """The polygonal content of any geometry, dropping zero-area line/point
    artifacts.

    A boolean cut (``polygon.intersection(box)``) on a CONCAVE region can return a
    ``GeometryCollection`` = a real ``Polygon`` plus a dangling ``LineString``
    where the cut grazes the concavity. In Shapely 2.x ``GeometryCollection``
    (and bare ``LineString``) have ``.boundary is None``, which crashes
    ``build_work_adjacency``. Such line/point parts carry ZERO area and ZERO
    coverage, so they are discarded; every polygonal part is kept in full (no
    survey area is lost, no coverage hole is created). For a plain ``Polygon`` or
    ``MultiPolygon`` this returns exactly the same polygon(s) in the same order,
    so callers stay byte-identical on the non-degenerate inputs that work today.
    """
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom] if geom.area > 0 else []
    if gt in ("MultiPolygon", "GeometryCollection"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_polygon_parts(g))
        return out
    return []  # LineString / Point / empty -> no polygonal content


@dataclass
class _WorkRegion:
    id: int
    geom: Polygon
    anchor: Pose
    area: float


class Decomposer(ABC):
    name: DecompositionAlgo

    @abstractmethod
    def decompose(
        self,
        tgc: TGCGraph,
        env: EnvironmentMap,
        drones: list[DroneStateView],
        target_area: Polygon | None = None,
    ) -> Partition:
        ...


# --------------------------------------------------------------------------- #
# shared geometry helpers                                                      #
# --------------------------------------------------------------------------- #
def clip_regions(tgc: TGCGraph, target_area: Polygon | None) -> list[_WorkRegion]:
    out: list[_WorkRegion] = []
    for r in tgc.regions:
        geom = r.polygon if target_area is None else r.polygon.intersection(target_area)
        if geom.is_empty or geom.area <= 1e-6:
            continue
        if not isinstance(geom, Polygon):
            # MultiPolygon or (on a concave clip) a GeometryCollection of a polygon
            # plus a zero-area line artifact -> take the largest polygonal part,
            # matching the existing MultiPolygon -> largest behaviour.
            parts = _polygon_parts(geom)
            if not parts:
                continue
            geom = max(parts, key=lambda g: g.area)
        out.append(_WorkRegion(r.id, geom, r.anchor, float(geom.area)))
    return out


def _split_longest_axis(poly: Polygon) -> list[Polygon]:
    minx, miny, maxx, maxy = poly.bounds
    if (maxx - minx) >= (maxy - miny):
        mid = (minx + maxx) / 2
        left = Polygon([(minx, miny), (mid, miny), (mid, maxy), (minx, maxy)])
        right = Polygon([(mid, miny), (maxx, miny), (maxx, maxy), (mid, maxy)])
        halves = [left, right]
    else:
        mid = (miny + maxy) / 2
        bot = Polygon([(minx, miny), (maxx, miny), (maxx, mid), (minx, mid)])
        top = Polygon([(minx, mid), (maxx, mid), (maxx, maxy), (minx, maxy)])
        halves = [bot, top]
    pieces = []
    for h in halves:
        c = poly.intersection(h)
        # A concave intersection can yield a MultiPolygon OR a GeometryCollection
        # (polygon(s) + a zero-area line artifact). Keep every polygonal part as
        # its own piece (area-preserving, each part a connected Polygon) and drop
        # the line/point artifacts -- identical to the prior MultiPolygon path for
        # non-degenerate inputs.
        for g in _polygon_parts(c):
            if g.area > 1e-9:
                pieces.append(g)
    return pieces or [poly]


def ensure_enough_regions(work: list[_WorkRegion], n_needed: int) -> list[_WorkRegion]:
    """Granularity guard: subdivide the largest regions until count >= n_needed."""
    work = list(work)
    next_id = max((w.id for w in work), default=-1) + 1
    while len(work) < n_needed:
        work.sort(key=lambda w: w.area, reverse=True)
        biggest = work.pop(0)
        pieces = _split_longest_axis(biggest.geom)
        if len(pieces) < 2:
            work.append(biggest)
            break
        for g in pieces:
            work.append(_WorkRegion(next_id, g, Pose(g.centroid.x, g.centroid.y, 0.0), float(g.area)))
            next_id += 1
    return work


def build_work_adjacency(work: list[_WorkRegion]) -> nx.Graph:
    g = nx.Graph()
    for w in work:
        g.add_node(w.id)
    for i in range(len(work)):
        for j in range(i + 1, len(work)):
            a, b = work[i], work[j]
            if not a.geom.envelope.intersects(b.geom.envelope):
                continue
            inter = a.geom.boundary.intersection(b.geom.boundary)
            if not inter.is_empty and inter.length > _TOUCH_TOL:
                w_dist = math.hypot(a.anchor.x - b.anchor.x, a.anchor.y - b.anchor.y)
                g.add_edge(a.id, b.id, weight=w_dist)
    return g


def is_connected_subset(adj: nx.Graph, ids: set[int]) -> bool:
    if len(ids) <= 1:
        return True
    sub = adj.subgraph(ids)
    return nx.is_connected(sub) if sub.number_of_nodes() == len(ids) else False


def build_zone(drone_id: int, regions: list[Region], geoms: list[Polygon], drone_pose: Pose) -> Zone:
    merged = unary_union(geoms) if geoms else Polygon()
    if merged.is_empty:
        entry = drone_pose
    else:
        dp = Point(drone_pose.as_xy())
        near = nearest_points(merged.boundary, dp)[0]
        cx, cy = merged.centroid.x, merged.centroid.y
        heading = math.atan2(cy - near.y, cx - near.x)
        entry = Pose(near.x, near.y, heading)
    return Zone(drone_id=drone_id, regions=regions, polygon=merged, entry_pose=entry)


def imbalance(partition: Partition, weights: dict[int, float]) -> float:
    total = partition.total_area_m2
    if total <= 0:
        return 0.0
    worst = 0.0
    for did, zone in partition.zones.items():
        worst = max(worst, abs(zone.area_m2 / total - weights.get(did, 0.0)))
    return worst
