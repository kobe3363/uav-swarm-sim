"""The single spatial-truth object: area, obstacles, free space, clearance and
collision queries, occupancy-grid export.

2.5D (Batch 1): ``LayerStack`` (bottom of file) is a vertical stack of per-layer
2D ``EnvironmentMap``s, obtained by slicing the extruded obstacle prisms at each
coverage altitude. ``EnvironmentMap`` itself is unchanged and is reused verbatim
per layer -- 3D never enters the horizontal queries; it lives only in *which*
footprints are present on each layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from ..infrastructure.core_types import Path, Pose
from .obstacle_generator import Obstacle

BOUNDARY_ID = -1


@dataclass(frozen=True)
class GridFrame:
    origin_x: float
    origin_y: float
    cell_m: float
    nx: int
    ny: int

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return (self.origin_x + (i + 0.5) * self.cell_m, self.origin_y + (j + 0.5) * self.cell_m)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int((x - self.origin_x) // self.cell_m), int((y - self.origin_y) // self.cell_m))


class EnvironmentMap:
    def __init__(self, area: Polygon, obstacles: list[Obstacle], buffer_m: float) -> None:
        self.area = area
        self.obstacles = obstacles
        self.buffer_m = buffer_m
        self._buffered = [o.polygon.buffer(buffer_m) for o in obstacles]
        self._obstacles_union = unary_union(self._buffered) if self._buffered else None
        # Raw (unbuffered) obstacle union -- used by in_obstacle() for the
        # "genuine penetration" test. Kept distinct from the buffered union
        # above, which defines clearance and free space.
        self._raw_union = unary_union([o.polygon for o in obstacles]) if obstacles else None
        if self._obstacles_union is not None:
            self.free_space: Polygon = area.difference(self._obstacles_union)
        else:
            self.free_space = area

    # --- queries ----------------------------------------------------------- #
    def clearance(self, p: tuple[float, float]) -> float:
        pt = Point(p)
        d_bound = pt.distance(self.area.exterior)
        if self._obstacles_union is None:
            return d_bound
        return min(d_bound, pt.distance(self._obstacles_union))

    def nearest_obstacle_id(self, p: tuple[float, float]) -> int:
        pt = Point(p)
        best_id, best_d = BOUNDARY_ID, pt.distance(self.area.exterior)
        for o in self.obstacles:
            d = pt.distance(o.polygon)
            if d < best_d:
                best_d, best_id = d, o.id
        return best_id

    def contains(self, p: tuple[float, float]) -> bool:
        return self.free_space.covers(Point(p))

    def in_obstacle(self, p: tuple[float, float]) -> bool:
        """True iff p lies inside a RAW obstacle polygon (unbuffered).

        This is the 'genuine penetration' predicate the SafetyMonitor uses to
        raise S_OBS only on *real* obstacle entry. It is deliberately distinct
        from `contains`, which tests free-space membership against the
        *buffered* obstacles and the area boundary: `not contains` would also be
        True inside the clearance buffer and outside the area -- neither of which
        is a collision -- and would corrupt the S_OBS / SMDP statistics.

        (2.5D note: when obstacles become extruded prisms, this generalizes to a
        per-altitude check -- a pose at height z penetrates only prisms whose
        [floor, ceil] band contains z. That lands with the LayerStack work; the
        2D footprint test here is the single-layer-z0 behaviour.)
        """
        if self._raw_union is None:
            return False
        return self._raw_union.covers(Point(p))

    def segment_in_obstacle(self, a: Pose, b: Pose) -> bool:
        """True iff the straight segment a->b crosses any RAW obstacle polygon
        (unbuffered). Exact Shapely-segment intersection -- the segment analogue
        of ``in_obstacle`` -- replacing point-sampling that can tunnel straight
        through an obstacle thinner than the sample spacing.

        Deliberately tested against the RAW union (like ``in_obstacle``), NOT the
        buffered union: this is the 'genuine penetration' predicate, used by the
        SafetyMonitor to validate that an evasive detour does not itself fly into
        an obstacle. ``segment_clear`` (buffered + must-stay-in-area) is the
        strictly stronger clearance predicate used for trajectory validation;
        the two are intentionally distinct, mirroring the ``in_obstacle`` /
        ``contains`` split above.
        """
        if self._raw_union is None:
            return False
        return LineString([a.as_xy(), b.as_xy()]).intersects(self._raw_union)

    def segment_clear(self, a: Pose, b: Pose) -> bool:
        line = LineString([a.as_xy(), b.as_xy()])
        if not self.area.covers(line):
            return False
        return self._obstacles_union is None or not line.intersects(self._obstacles_union)

    def first_obstruction(self, path: Path, step_m: float = 2.0) -> float | None:
        """Approximate arc-length (m) of the first point on ``path`` that leaves
        free space or whose chord to the previous sample crosses a buffered
        obstacle; ``None`` if the whole path is clear. Same buffered-clearance
        test as ``path_clear`` (every pose in ``free_space`` AND every chord
        ``segment_clear``), but it reports WHERE the first violation is so the
        trajectory validator can log the clip and the runtime S_OBS recovery can
        pick a downstream rejoin point. The location is the sample index times
        ``step_m`` -- exact enough for diagnostics, not a precise contact point.
        """
        poses = path.sample(step_m)
        if not poses:
            return None
        if not self.free_space.covers(Point(poses[0].as_xy())):
            return 0.0
        for i in range(1, len(poses)):
            a, b = poses[i - 1], poses[i]
            if not self.free_space.covers(Point(b.as_xy())) or not self.segment_clear(a, b):
                return i * step_m
        return None

    def path_clear(self, path: Path, step_m: float = 2.0) -> bool:
        """True iff every point of the smoothed ``path`` stays in free space and
        no chord crosses a buffered obstacle. Validated against the BUFFERED
        union (``free_space`` / ``_obstacles_union``), which is strictly stronger
        than the raw-penetration ``in_obstacle`` trigger the SafetyMonitor uses --
        so a path that passes here can never raise S_OBS. Thin wrapper over
        ``first_obstruction`` (single source of truth for the sampling)."""
        return self.first_obstruction(path, step_m) is None

    def occupancy_grid(self, cell_m: float) -> tuple[np.ndarray, GridFrame]:
        minx, miny, maxx, maxy = self.area.bounds
        nx = max(1, int((maxx - minx) / cell_m) + 1)
        ny = max(1, int((maxy - miny) / cell_m) + 1)
        frame = GridFrame(minx, miny, cell_m, nx, ny)
        grid = np.zeros((nx, ny), dtype=bool)
        fs = self.free_space
        for i in range(nx):
            for j in range(ny):
                grid[i, j] = fs.covers(Point(frame.cell_center(i, j)))
        return grid, frame

    def sample_free(self, n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        minx, miny, maxx, maxy = self.area.bounds
        out: list[tuple[float, float]] = []
        guard = 0
        while len(out) < n and guard < 1000 * max(n, 1):
            guard += 1
            x = float(rng.uniform(minx, maxx))
            y = float(rng.uniform(miny, maxy))
            if self.free_space.covers(Point(x, y)):
                out.append((x, y))
        return out


# --------------------------------------------------------------------------- #
# 2.5D: vertical stack of per-layer 2D maps (Batch 1)                         #
# --------------------------------------------------------------------------- #
class LayerStack:
    """A vertical stack of per-layer 2D ``EnvironmentMap``s, obtained by slicing
    the extruded obstacle prisms at each coverage altitude.

    At altitude ``z_i`` the active obstacle set is ``{o for o in prisms if
    o.spans(z_i)}`` -- higher layers clear shorter prisms. Each layer is a full
    ``EnvironmentMap`` whose internals (free space, clearance, penetration,
    occupancy) are reused unchanged, so every existing 2D planner runs per layer
    with no edits. 3D never enters the horizontal planners; it lives only in the
    choice of which footprints are present per layer.

    Single-layer-z0 invariant: one altitude together with unbounded prisms
    (``z_ceil = inf``) means the lone layer contains every footprint, so
    ``layer(0)`` is byte-identical to the 2D ``EnvironmentMap``. The caller
    supplies the same ``buffer_m`` it already passes to the 2D map, so that
    identity is exact.
    """

    def __init__(
        self,
        area: Polygon,
        prisms: list[Obstacle],
        altitudes,
        buffer_m: float,
    ) -> None:
        self.area = area
        self.prisms = list(prisms)
        self.altitudes: tuple[float, ...] = tuple(float(z) for z in altitudes)
        self.buffer_m = buffer_m
        # Build one per-layer 2D map by slicing the prisms at each altitude.
        self._layers: list[EnvironmentMap] = [
            EnvironmentMap(area, [o for o in self.prisms if o.spans(z)], buffer_m)
            for z in self.altitudes
        ]

    @property
    def n_layers(self) -> int:
        return len(self.altitudes)

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, idx: int) -> EnvironmentMap:
        return self._layers[idx]

    def layer(self, idx: int) -> EnvironmentMap:
        """The per-layer 2D ``EnvironmentMap`` at layer index ``idx``."""
        return self._layers[idx]

    def altitude(self, idx: int) -> float:
        return self.altitudes[idx]

    def active_prisms(self, idx: int) -> list[Obstacle]:
        """Prisms whose vertical band intersects layer ``idx`` (the footprints
        that are real obstacles on that layer)."""
        z = self.altitudes[idx]
        return [o for o in self.prisms if o.spans(z)]

    def __iter__(self):
        """Iterate the per-layer ``EnvironmentMap``s in altitude order."""
        return iter(self._layers)

    def items(self):
        """Yield ``(layer_index, EnvironmentMap)`` pairs in altitude order."""
        return enumerate(self._layers)
