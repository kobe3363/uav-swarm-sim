"""Discretized-grid comparative planner (after Majd et al. 2020).

Fast and kinematically naive -- the empirical counterpart in the planning-speed
vs. flyable-length trade-off (guideline 1.2). Coverage is a row-major serpentine
over free zone cells with rectilinear corners (no curvature constraint -- that is
the point); routing is octile A* on the occupancy grid.
"""
from __future__ import annotations

import heapq
import math
import time

from shapely.geometry import Point, Polygon

from ..infrastructure.core_types import (
    CoveragePlan,
    Path,
    Pose,
    Waypoint,
    straight_segment,
)
from ..infrastructure.enums import ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from .environment_map import EnvironmentMap


class GridPlanner:
    def __init__(self, env: EnvironmentMap, cell_m: float) -> None:
        self.env = env
        self.cell_m = cell_m
        self.grid, self.frame = env.occupancy_grid(cell_m)
        self.last_plan_time_s = 0.0

    def coverage(self, zone: Zone_like, spec: PlatformSpec) -> CoveragePlan:  # type: ignore[name-defined]
        t0 = time.perf_counter()
        poly = zone.polygon
        f = self.frame
        # mark zone cells that are free
        rows: list[list[tuple[int, int]]] = []
        for j in range(f.ny):
            row: list[tuple[int, int]] = []
            for i in range(f.nx):
                if not self.grid[i, j]:
                    continue
                cx, cy = f.cell_center(i, j)
                if poly.covers(Point(cx, cy)):
                    row.append((i, j))
            if row:
                rows.append(row)

        waypoints: list[Waypoint] = []
        length = 0.0
        flip = False
        prev = None
        for row in rows:
            seq = row if not flip else list(reversed(row))
            # split row into contiguous runs
            runs: list[list[tuple[int, int]]] = []
            cur = [seq[0]]
            for c in seq[1:]:
                if abs(c[0] - cur[-1][0]) == 1:
                    cur.append(c)
                else:
                    runs.append(cur)
                    cur = [c]
            runs.append(cur)
            for run in runs:
                s = f.cell_center(*run[0])
                e = f.cell_center(*run[-1])
                heading = math.atan2(e[1] - s[1], e[0] - s[0]) if s != e else 0.0
                waypoints.append(Waypoint(Pose(s[0], s[1], heading), ManeuverType.COVERAGE, spec.v_coverage))
                waypoints.append(Waypoint(Pose(e[0], e[1], heading), ManeuverType.COVERAGE, spec.v_coverage))
                length += math.dist(s, e)
                if prev is not None:
                    length += math.dist(prev, s)  # rectilinear row hop
                prev = e
            flip = not flip

        # crude energy: coverage power over the path time (no formation, no curvature)
        energy = (spec.power_w[ManeuverType.COVERAGE] * length / spec.v_coverage) if length > 0 else 0.0
        self.last_plan_time_s = time.perf_counter() - t0
        return CoveragePlan(zone.drone_id, waypoints, length, energy)

    def route(self, a: Pose, b: Pose) -> Path:
        f = self.frame
        start = f.world_to_cell(a.x, a.y)
        goal = f.world_to_cell(b.x, b.y)
        came = self._astar(start, goal)
        if came is None:
            # fall back to a straight segment if no grid path
            dist = math.dist(a.as_xy(), b.as_xy())
            return Path.from_segments([straight_segment(a, dist, ManeuverType.CRUISE, 1.0)])
        # reconstruct cell path -> world polyline -> straight segments
        cells = [goal]
        while cells[-1] != start:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [f.cell_center(i, j) for i, j in cells]
        segs = []
        for p, q in zip(pts, pts[1:]):
            h = math.atan2(q[1] - p[1], q[0] - p[0])
            segs.append(straight_segment(Pose(p[0], p[1], h), math.dist(p, q), ManeuverType.CRUISE, 1.0))
        return Path.from_segments(segs)

    def _astar(self, start, goal):
        if not (self._free(*start) and self._free(*goal)):
            return None
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        openq = [(0.0, start)]
        g = {start: 0.0}
        came: dict = {}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                return came
            for di, dj, w in nbrs:
                nb = (cur[0] + di, cur[1] + dj)
                if not self._free(*nb):
                    continue
                ng = g[cur] + w
                if nb not in g or ng < g[nb]:
                    g[nb] = ng
                    came[nb] = cur
                    h = math.hypot(nb[0] - goal[0], nb[1] - goal[1])
                    heapq.heappush(openq, (ng + h, nb))
        return None

    def _free(self, i, j) -> bool:
        return 0 <= i < self.frame.nx and 0 <= j < self.frame.ny and bool(self.grid[i, j])


# late import to avoid a hard module-level dependency cycle in annotations
from ..infrastructure.core_types import Zone as Zone_like  # noqa: E402
