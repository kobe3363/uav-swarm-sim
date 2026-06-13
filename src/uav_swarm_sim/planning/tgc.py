"""Topological Graph Construction (after J. Liu et al. 2026).

Condense the GVG into junction-and-corridor topology and extract free-space
*regions* (the atomic units of decomposition). Region polygons come from a
nearest-anchor Voronoi partition of free space clipped exactly with Shapely, so
region *areas are exact* even though region *shapes* are a constructive
approximation of the topological cells.

Robustness: when the GVG is empty (no/sparse obstacles) the whole free space
becomes a single region, so downstream decomposition always has something to
work with.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import networkx as nx
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import voronoi_diagram

from ..infrastructure.core_types import Pose, Region
from .environment_map import EnvironmentMap

_LOG = logging.getLogger(__name__)


@dataclass
class TGCGraph:
    graph: nx.Graph                 # nodes = junctions/leaves; edges = corridors
    regions: list[Region]
    region_adjacency: nx.Graph      # nodes = region ids; edges where regions touch
    planning_time_s: float = 0.0
    _dist_cache: dict = field(default_factory=dict, repr=False)

    def region_of(self, p: tuple[float, float]) -> int:
        pt = Point(p)
        for r in self.regions:
            if r.polygon.covers(pt):
                return r.id
        # fallback: nearest anchor
        return min(self.regions, key=lambda r: (r.anchor.x - p[0]) ** 2 + (r.anchor.y - p[1]) ** 2).id

    def corridor_distance(self, r1: int, r2: int) -> float:
        if r1 == r2:
            return 0.0
        if r1 not in self._dist_cache:
            self._dist_cache[r1] = nx.single_source_dijkstra_path_length(
                self.region_adjacency, r1, weight="weight"
            )
        return self._dist_cache[r1].get(r2, math.inf)


def _corridors(gvg: nx.Graph) -> list[tuple[tuple, tuple, float, list]]:
    """Contract degree-2 chains between junction/leaf nodes into corridors."""
    junctions = {n for n in gvg.nodes if gvg.degree(n) != 2}
    corridors: list[tuple[tuple, tuple, float, list]] = []
    visited_edges: set = set()
    if not junctions:
        # a pure loop or simple path: treat every node as a potential anchor source
        if gvg.number_of_edges() > 0:
            for u, v, d in gvg.edges(data=True):
                corridors.append((u, v, d["length"], [u, v]))
        return corridors
    for j in junctions:
        for nbr in gvg.neighbors(j):
            e = frozenset((j, nbr))
            if e in visited_edges:
                continue
            chain = [j, nbr]
            length = gvg.edges[j, nbr]["length"]
            visited_edges.add(e)
            prev, cur = j, nbr
            while cur not in junctions:
                nxts = [x for x in gvg.neighbors(cur) if x != prev]
                if not nxts:
                    break
                nx_node = nxts[0]
                e2 = frozenset((cur, nx_node))
                if e2 in visited_edges:
                    break
                visited_edges.add(e2)
                length += gvg.edges[cur, nx_node]["length"]
                chain.append(nx_node)
                prev, cur = cur, nx_node
            corridors.append((chain[0], chain[-1], length, chain))
    return corridors


def _anchor_pose(chain: list, free: Polygon) -> Pose:
    line_pts = chain
    mid = line_pts[len(line_pts) // 2]
    p = Point(mid)
    if not free.covers(p):
        p = free.representative_point() if free.covers(free.representative_point()) else p
    # heading along the corridor
    a = line_pts[0]
    b = line_pts[-1]
    heading = math.atan2(b[1] - a[1], b[0] - a[0])
    return Pose(p.x, p.y, heading)


def _regions_from_anchors(free: Polygon, anchors: list[Pose]) -> list[Region]:
    if len(anchors) <= 1:
        geom = free if isinstance(free, Polygon) else max(free.geoms, key=lambda g: g.area)
        a = anchors[0] if anchors else Pose(geom.centroid.x, geom.centroid.y, 0.0)
        return [Region(0, geom, float(geom.area), a)]

    pts = MultiPoint([(a.x, a.y) for a in anchors])
    env_poly = free.envelope
    cells = voronoi_diagram(pts, envelope=env_poly)
    regions: list[Region] = []
    rid = 0
    for a in anchors:
        ap = Point(a.x, a.y)
        # find the Voronoi cell containing this anchor
        cell = next((c for c in cells.geoms if c.covers(ap)), None)
        if cell is None:
            continue
        clipped = cell.intersection(free)
        if clipped.is_empty or clipped.area <= 1e-6:
            continue
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        regions.append(Region(rid, clipped, float(clipped.area), a))
        rid += 1
    if not regions:
        geom = free if isinstance(free, Polygon) else max(free.geoms, key=lambda g: g.area)
        regions = [Region(0, geom, float(geom.area), Pose(geom.centroid.x, geom.centroid.y, 0.0))]
    return regions


def _build_adjacency(regions: list[Region]) -> nx.Graph:
    g = nx.Graph()
    for r in regions:
        g.add_node(r.id)
    for r1, r2 in itertools.combinations(regions, 2):
        if not r1.polygon.envelope.intersects(r2.polygon.envelope):
            continue
        inter = r1.polygon.boundary.intersection(r2.polygon.boundary)
        if not inter.is_empty and inter.length > 1e-6:
            w = math.hypot(r1.anchor.x - r2.anchor.x, r1.anchor.y - r2.anchor.y)
            g.add_edge(r1.id, r2.id, weight=w)
    return g


def build_tgc(env: EnvironmentMap, gvg: nx.Graph) -> TGCGraph:
    import time

    t0 = time.perf_counter()
    free = env.free_space

    corridors = _corridors(gvg)
    tgraph = nx.Graph()
    anchors: list[Pose] = []
    for (u, v, length, chain) in corridors:
        tgraph.add_edge(u, v, length=length)
        anchors.append(_anchor_pose(chain, free))

    regions = _regions_from_anchors(free, anchors)
    adjacency = _build_adjacency(regions)
    dt = time.perf_counter() - t0
    _LOG.info("TGC built: %d corridors, %d regions in %.3fs", len(corridors), len(regions), dt)
    return TGCGraph(graph=tgraph, regions=regions, region_adjacency=adjacency, planning_time_s=dt)
