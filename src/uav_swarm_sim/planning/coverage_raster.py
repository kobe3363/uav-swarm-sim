"""Persistent raster truth for physically flown camera coverage."""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Polygon

from ..infrastructure.core_types import Pose

_AREA_EPS = 1e-12
_MAX_RASTER_CELLS = 2_000_000


class CoverageRaster:
    """Fixed target/plannable masks with monotone, union-only occupancy."""

    def __init__(self, target: Polygon, plannable: Polygon, cell_m: float) -> None:
        if not math.isfinite(cell_m) or cell_m <= 0.0:
            raise ValueError("coverage raster cell_m must be finite and > 0")
        self.target_geometry = target
        self.plannable_geometry = plannable
        self.cell_m = float(cell_m)

        if target.is_empty:
            empty = np.empty(0, dtype=object)
            self._target_parts = self._target_points = empty
            self._plannable_parts = self._plannable_points = empty
            self._target_weights = self._plannable_weights = np.empty(0, dtype=float)
        else:
            minx, miny, maxx, maxy = target.bounds
            nx = max(1, math.ceil((maxx - minx) / self.cell_m))
            ny = max(1, math.ceil((maxy - miny) / self.cell_m))
            n_cells = nx * ny
            if n_cells > _MAX_RASTER_CELLS:
                raise ValueError(
                    f"coverage raster requires {n_cells:,} bounding-box cells; "
                    f"limit is {_MAX_RASTER_CELLS:,}; increase coverage.raster_cell_m"
                )
            ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            x0 = minx + ix.ravel() * self.cell_m
            y0 = miny + iy.ravel() * self.cell_m
            cells = shapely.box(x0, y0, x0 + self.cell_m, y0 + self.cell_m)

            (
                self._target_parts,
                self._target_points,
                self._target_weights,
            ) = self._mask_cells(cells, target)
            (
                self._plannable_parts,
                self._plannable_points,
                self._plannable_weights,
            ) = self._mask_cells(cells, plannable)
        self._target_covered = np.zeros(len(self._target_weights), dtype=bool)
        self._plannable_covered = np.zeros(len(self._plannable_weights), dtype=bool)
        self._target_tree = STRtree(self._target_points)
        self._plannable_tree = STRtree(self._plannable_points)

    @staticmethod
    def _mask_cells(cells, geometry):
        parts = shapely.intersection(cells, geometry)
        weights = np.asarray(shapely.area(parts), dtype=float)
        keep = weights > _AREA_EPS
        parts = parts[keep]
        return parts, shapely.point_on_surface(parts), weights[keep]

    @property
    def target_area_m2(self) -> float:
        return float(self.target_geometry.area)

    @property
    def plannable_area_m2(self) -> float:
        return float(self.plannable_geometry.area)

    @property
    def target_cell_count(self) -> int:
        return len(self._target_weights)

    @property
    def plannable_cell_count(self) -> int:
        return len(self._plannable_weights)

    @staticmethod
    def _fraction(covered: np.ndarray, weights: np.ndarray, area_m2: float) -> float:
        if area_m2 <= _AREA_EPS:
            return 1.0
        return min(1.0, float(weights[covered].sum()) / area_m2)

    @property
    def target_coverage_frac(self) -> float:
        return self._fraction(
            self._target_covered, self._target_weights, self.target_area_m2
        )

    @property
    def target_covered_area_m2(self) -> float:
        return float(self._target_weights[self._target_covered].sum())

    @property
    def plannable_coverage_frac(self) -> float:
        return self._fraction(
            self._plannable_covered,
            self._plannable_weights,
            self.plannable_area_m2,
        )

    @property
    def plannable_covered_area_m2(self) -> float:
        return float(self._plannable_weights[self._plannable_covered].sum())

    @property
    def uncovered_plannable_geometry(self):
        """Clipped raster work geometry for later redistribution consumers."""
        remaining = self._plannable_parts[~self._plannable_covered]
        return shapely.union_all(remaining) if len(remaining) else Polygon()

    def record_segment(
        self,
        old_pose: Pose,
        new_pose: Pose,
        footprint_width_m: float,
        footprint_length_m: float,
    ) -> None:
        """Credit one actually travelled camera-on segment's continuous footprint."""
        if (
            not math.isfinite(footprint_width_m)
            or not math.isfinite(footprint_length_m)
            or footprint_width_m <= 0.0
            or footprint_length_m <= 0.0
        ):
            raise ValueError("camera footprint dimensions must be finite and > 0")
        dx, dy = new_pose.x - old_pose.x, new_pose.y - old_pose.y
        distance = math.hypot(dx, dy)
        if distance <= _AREA_EPS:
            return
        heading = math.atan2(dy, dx)
        ux, uy = math.cos(heading), math.sin(heading)
        vx, vy = -uy, ux
        half_l = footprint_length_m / 2.0
        half_w = footprint_width_m / 2.0
        ax = old_pose.x - half_l * ux
        ay = old_pose.y - half_l * uy
        bx = new_pose.x + half_l * ux
        by = new_pose.y + half_l * uy
        sweep = Polygon([
            (ax + half_w * vx, ay + half_w * vy),
            (bx + half_w * vx, by + half_w * vy),
            (bx - half_w * vx, by - half_w * vy),
            (ax - half_w * vx, ay - half_w * vy),
        ])
        target_hits = self._target_tree.query(sweep, predicate="covers")
        plannable_hits = self._plannable_tree.query(sweep, predicate="covers")
        self._target_covered[target_hits] = True
        self._plannable_covered[plannable_hits] = True
