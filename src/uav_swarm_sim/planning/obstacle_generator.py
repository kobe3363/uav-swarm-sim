"""Configurable synthetic obstacle fields.

Counts via a spatial Poisson process; sizes/shapes from config; each obstacle
carries a class label used later by the GVG. Overlapping obstacles are merged
(so 'distinct obstacle' is well defined) and configurations that disconnect free
space are rejected and resampled.

2.5D (Batch 1)
--------------
Each obstacle is an extruded PRISM: a 2D footprint plus a vertical band
[z_floor, z_ceil]. The band defaults to [obstacle_floor_m, +inf) -- an unbounded
ceiling means the prism is present on every coverage layer, so the
single-layer-z0 case is byte-identical to the 2D field. A finite
obstacle_ceil_range_m makes higher layers clear shorter prisms (altitude as a
tactic).

CRITICAL (byte-identity): ceilings are sampled from the RNG ONLY when a finite
range is configured. With the default (None) NO extra random draws occur, so the
footprint-generation sequence -- and therefore every obstacle FOOTPRINT -- is
bit-for-bit identical to the 2D baseline. The floor is a constant config scalar
and never touches the RNG.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from ..infrastructure.config import EnvConfig

_LOG = logging.getLogger(__name__)
_MAX_RESAMPLE = 20


@dataclass(frozen=True)
class Obstacle:
    id: int
    cls: int
    polygon: Polygon
    z_floor: float = 0.0
    z_ceil: float = math.inf   # +inf => unbounded: active on every layer (2D-identical default)

    @property
    def height_m(self) -> float:
        return self.z_ceil - self.z_floor

    def spans(self, z: float) -> bool:
        """True iff altitude ``z`` falls within this prism's vertical band, i.e.
        the footprint is an active 2D obstacle on the layer at altitude ``z``."""
        return self.z_floor <= z <= self.z_ceil


def _unit_shape(kind: str, size: float, rng: np.random.Generator) -> Polygon:
    if kind == "circle":
        return Point(0.0, 0.0).buffer(size / 2.0, quad_segs=4)  # 16-gon
    if kind == "rectangle":
        aspect = float(rng.uniform(1.0, 3.0))
        w, h = size, size / aspect
        rect = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        return rotate(rect, float(rng.uniform(0, 180)), origin=(0, 0))
    # generic convex polygon (hull of random points)
    k = int(rng.integers(5, 9))
    pts = rng.uniform(-size / 2, size / 2, size=(k, 2))
    hull = MultiPolygon([]) if k < 3 else Polygon(pts).convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty:
        return Point(0, 0).buffer(size / 2.0, quad_segs=4)
    return hull


def _merge_overlaps(raw: list[tuple[int, Polygon]]) -> list[tuple[int, Polygon]]:
    """Union overlapping obstacles, keeping the lower class id."""
    merged: list[tuple[int, Polygon]] = []
    for cls, poly in raw:
        hit = None
        for i, (mcls, mpoly) in enumerate(merged):
            if poly.intersects(mpoly):
                hit = i
                break
        if hit is None:
            merged.append((cls, poly))
        else:
            mcls, mpoly = merged[hit]
            merged[hit] = (min(cls, mcls), unary_union([mpoly, poly]))
    return merged


def _free_connected(area: Polygon, obstacles: list[Polygon], buffer_m: float) -> bool:
    if not obstacles:
        return True
    blocked = unary_union([o.buffer(buffer_m) for o in obstacles])
    free = area.difference(blocked)
    if free.is_empty:
        return False
    if isinstance(free, MultiPolygon):
        big = [g for g in free.geoms if g.area > 0.01 * area.area]
        return len(big) <= 1
    return True


def generate(area: Polygon, cfg: EnvConfig, rng: np.random.Generator) -> list[Obstacle]:
    minx, miny, maxx, maxy = area.bounds
    area_km2 = area.area / 1e6
    lam = cfg.obstacle_density_per_km2 * area_km2

    # Prism vertical band. The floor is a constant config scalar (no RNG). The
    # ceiling is unbounded by default (None); a finite range is the ONLY source
    # of extra RNG draws, and only on the successful attempt below -- see the
    # module docstring on byte-identity.
    floor = float(cfg.obstacle_floor_m)
    ceil_range = cfg.obstacle_ceil_range_m  # None => unbounded; else (lo, hi)

    for attempt in range(_MAX_RESAMPLE):
        n = int(rng.poisson(lam))
        raw: list[tuple[int, Polygon]] = []
        guard = 0
        while len(raw) < n and guard < 50 * max(n, 1):
            guard += 1
            cx = float(rng.uniform(minx, maxx))
            cy = float(rng.uniform(miny, maxy))
            if not area.contains(Point(cx, cy)):
                continue
            kind = str(rng.choice(cfg.obstacle_shapes))
            size = float(rng.uniform(*cfg.obstacle_size_range_m))
            cls = int(rng.integers(0, cfg.n_obstacle_classes))
            poly = translate(_unit_shape(kind, size, rng), cx, cy)
            poly = poly.intersection(area)
            if poly.is_empty or poly.area <= 0 or not isinstance(poly, Polygon):
                continue
            raw.append((cls, poly))

        merged = _merge_overlaps(raw)
        polys = [p for _, p in merged]
        # Connectivity is checked at maximal density (all footprints present ==
        # the lowest/densest layer). Sparser higher layers only gain free space,
        # so this single check bounds every layer.
        if _free_connected(area, polys, cfg.clearance_buffer_m):
            return [
                Obstacle(
                    id=i,
                    cls=cls,
                    polygon=poly,
                    z_floor=floor,
                    z_ceil=(
                        math.inf
                        if ceil_range is None
                        else float(rng.uniform(ceil_range[0], ceil_range[1]))
                    ),
                )
                for i, (cls, poly) in enumerate(merged)
            ]
        _LOG.debug("obstacle config disconnected free space; resample %d", attempt + 1)

    raise RuntimeError(
        f"could not generate a connected obstacle field in {_MAX_RESAMPLE} attempts; "
        "reduce density or clearance_buffer_m"
    )
