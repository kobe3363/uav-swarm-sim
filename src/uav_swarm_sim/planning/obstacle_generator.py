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
    if kind == "square":
        # Axis-aligned fixed square, side == size. Deterministic: NO aspect draw
        # and NO rotation draw, so this branch consumes ZERO RNG draws (vs the
        # rectangle branch's two). Placed BEFORE the generic-polygon fallthrough
        # so a "square" config never silently degrades into a random polygon.
        s = size
        return Polygon([(-s / 2, -s / 2), (s / 2, -s / 2), (s / 2, s / 2), (-s / 2, s / 2)])
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


def _free_connected(
    area: Polygon,
    obstacles: list[Polygon],
    buffer_m: float,
    min_component_fraction: float = 0.01,
) -> bool:
    if not obstacles:
        return True
    blocked = unary_union([o.buffer(buffer_m) for o in obstacles])
    free = area.difference(blocked)
    if free.is_empty:
        return False
    if isinstance(free, MultiPolygon):
        big = [g for g in free.geoms if g.area > min_component_fraction * area.area]
        return len(big) <= 1
    return True


def _generate_target(
    area: Polygon,
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> list[Obstacle]:
    """Generate a bounded, seeded field with fixed count and union-area bounds."""
    if area.is_empty or area.area <= 0.0:
        raise RuntimeError("could not generate target obstacles: survey area must have positive area")

    count = cfg.obstacle_target_count
    target_fraction = cfg.obstacle_area_fraction
    tolerance = cfg.obstacle_area_fraction_tolerance
    max_attempts = cfg.obstacle_generation_max_attempts
    side_m = math.sqrt(area.area * target_fraction / count)
    minx, miny, maxx, maxy = area.bounds
    floor = float(cfg.obstacle_floor_m)
    ceil_range = cfg.obstacle_ceil_range_m
    attempts = 0
    last_reason = f"placed 0 of {count} non-overlapping obstacles"

    while attempts < max_attempts:
        raw: list[tuple[int, Polygon]] = []
        while len(raw) < count and attempts < max_attempts:
            attempts += 1
            cx = float(rng.uniform(minx, maxx))
            cy = float(rng.uniform(miny, maxy))
            candidate = translate(_unit_shape("square", side_m, rng), cx, cy)
            clipped = candidate.intersection(area)
            if clipped.is_empty or clipped.area <= 0.0 or not isinstance(clipped, Polygon):
                continue
            # A boundary clip would no longer be the equal-area square selected
            # for EXP-03. Reject it, but still validate the final clipped/union
            # geometry below as a separate invariant.
            clip_tolerance = 1e-9 * max(1.0, candidate.area)
            if not math.isclose(clipped.area, candidate.area, rel_tol=0.0, abs_tol=clip_tolerance):
                continue
            # ``intersects`` rejects both overlap and boundary contact, because
            # touching footprints would collapse into fewer union components.
            if any(clipped.intersects(existing) for _, existing in raw):
                continue
            cls = int(rng.integers(0, cfg.n_obstacle_classes))
            raw.append((cls, clipped))

        if len(raw) != count:
            last_reason = f"placed {len(raw)} of {count} non-overlapping obstacles"
            continue

        polys = [poly for _, poly in raw]
        final_geometry = unary_union(polys)
        component_count = (
            len(final_geometry.geoms)
            if isinstance(final_geometry, MultiPolygon)
            else 1 if isinstance(final_geometry, Polygon) and not final_geometry.is_empty else 0
        )
        if component_count != count:
            last_reason = f"final union has {component_count} components, expected {count}"
            continue

        final_fraction = final_geometry.area / area.area
        lower = target_fraction - tolerance
        upper = target_fraction + tolerance
        # The configured tolerance is scientific and absolute. The fixed
        # dimensionless epsilon only absorbs arithmetic/GEOS roundoff at an
        # inclusive boundary; it is not a relative relaxation of that band.
        geometry_epsilon = 1e-12
        if final_fraction < lower - geometry_epsilon or final_fraction > upper + geometry_epsilon:
            last_reason = (
                f"final area fraction {final_fraction:.6f} outside "
                f"[{lower:.6f}, {upper:.6f}] after clipping/union"
            )
            continue
        if not final_geometry.is_valid:
            last_reason = "final clipped/union geometry is invalid"
            continue
        if not _free_connected(
            area,
            polys,
            cfg.clearance_buffer_m,
            min_component_fraction=0.0,
        ):
            last_reason = "clearance-buffered free space is disconnected"
            continue

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
            for i, (cls, poly) in enumerate(raw)
        ]

    raise RuntimeError(
        f"could not generate target obstacle field in {max_attempts} attempts: {last_reason}"
    )


def generate(area: Polygon, cfg: EnvConfig, rng: np.random.Generator) -> list[Obstacle]:
    if cfg.obstacle_generation_mode == "target":
        return _generate_target(area, cfg, rng)

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
