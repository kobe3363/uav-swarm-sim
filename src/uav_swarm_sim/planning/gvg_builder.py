"""Generalized Voronoi Graph of free space.

The skeleton equidistant between *distinct obstacles* (extended objects, not
point sites) -- which is what makes the TGC framework work in obstacle
environments. Built by densely sampling obstacle and area boundaries, computing
a Voronoi diagram of all samples, and keeping only ridges whose two generating
samples belong to *different owners* (different obstacle, or obstacle vs. the
area boundary). Owner labels are obstacle ids; the area boundary is owner -1.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString

from .environment_map import BOUNDARY_ID, EnvironmentMap

_LOG = logging.getLogger(__name__)


def _densify(coords: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    line = LineString(coords)
    n = max(1, int(line.length / step))
    return [tuple(line.interpolate(k / n, normalized=True).coords[0]) for k in range(n + 1)]


def _boundary_samples(env: EnvironmentMap, step: float):
    pts: list[tuple[float, float]] = []
    owners: list[int] = []
    # obstacles
    for o in env.obstacles:
        ring = list(o.polygon.exterior.coords)
        for p in _densify(ring, step):
            pts.append(p)
            owners.append(o.id)
    # area boundary (owner -1)
    for p in _densify(list(env.area.exterior.coords), step):
        pts.append(p)
        owners.append(BOUNDARY_ID)
    return np.asarray(pts), np.asarray(owners)


def _snap(x: float, y: float, eps: float) -> tuple[float, float]:
    return (round(x / eps) * eps, round(y / eps) * eps)


def build_gvg(env: EnvironmentMap, sample_step_m: float = 3.0, spur_min_m: float = 8.0) -> nx.Graph:
    g = nx.Graph()
    if not env.obstacles:
        _LOG.info("no obstacles: GVG is empty (TGC will use a single region)")
        return g

    pts, owners = _boundary_samples(env, sample_step_m)
    if len(pts) < 4:
        return g
    vor = Voronoi(pts)
    eps = sample_step_m / 2.0
    fs = env.free_space
    buf = env.buffer_m

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if owners[p1] == owners[p2]:
            continue
        if v1 < 0 or v2 < 0:
            continue  # ridge runs to infinity
        a = tuple(vor.vertices[v1])
        b = tuple(vor.vertices[v2])
        seg = LineString([a, b])
        if not fs.covers(seg):
            continue
        na, nb = _snap(*a, eps), _snap(*b, eps)
        if na == nb:
            continue
        if env.clearance(na) < buf or env.clearance(nb) < buf:
            continue
        length = float(np.hypot(na[0] - nb[0], na[1] - nb[1]))
        g.add_node(na, clearance=env.clearance(na))
        g.add_node(nb, clearance=env.clearance(nb))
        g.add_edge(na, nb, length=length)

    _prune_spurs(g, spur_min_m)
    _keep_largest_cc(g)
    return g


def _prune_spurs(g: nx.Graph, spur_min_m: float) -> None:
    changed = True
    while changed:
        changed = False
        for node in list(g.nodes):
            if g.degree(node) == 1:
                nbr = next(iter(g.neighbors(node)))
                if g.edges[node, nbr]["length"] < spur_min_m:
                    g.remove_node(node)
                    changed = True


def _keep_largest_cc(g: nx.Graph) -> None:
    if g.number_of_nodes() == 0:
        return
    comps = list(nx.connected_components(g))
    if len(comps) <= 1:
        return
    largest = max(comps, key=len)
    total_len = sum(d["length"] for *_e, d in g.edges(data=True))
    for comp in comps:
        if comp is largest:
            continue
        g.remove_nodes_from(comp)
    kept_len = sum(d["length"] for *_e, d in g.edges(data=True))
    if total_len > 0 and (total_len - kept_len) / total_len > 0.02:
        _LOG.warning("GVG pruning discarded %.1f%% of edge length", 100 * (1 - kept_len / total_len))
