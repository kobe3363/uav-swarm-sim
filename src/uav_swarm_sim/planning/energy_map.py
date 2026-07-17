"""EM-01 Stage 1: per-replication energy cost-to-go map ("energy map").

One Dijkstra from the base cell over an occupancy-penalised 8-connected grid
produces, for every cell, ``E_home`` -- the minimum RETURN energy (J) to reach
the base -- plus a parent pointer (the base-ward neighbor). Pure, standalone
computation: nothing in the mission path consumes it yet (RTH decide, return /
resume routing are later stages of docs/proposals/energy_map_rth.md).

ENERGY UNIT -- CRITICAL (design doc section 6.2): every edge is costed at
CRUISE via ``EnergyModel.distance_energy(dist, ManeuverType.CRUISE, v_cruise)``
because returns fly cruise. This map is the RETURN cost only. It deliberately
contains NO COVERAGE power, NO camera (``sensor.sensor_power_w``) term and NO
landing profile -- those belong to the consumer's ``e_next_bundle`` /
``return_energy`` (rth_calculator.py) at the decide stage, exactly as the
execution-time physics charges them.

Grid conventions (author's Stage-1 decisions):
  * The grid ORIGIN is lattice-aligned to the base: ``base_pose`` coincides
    with the CENTER of its cell (the RTH target is the reference point, not the
    area's bbox corner). The lattice is extended by whole cells to cover the
    survey area's bounds plus a one-cell pad.
  * The base cell is ALWAYS traversable (penalty 1.0) and ``E_home[base] = 0``
    by definition -- it is the return target; if the base pose physically lies
    inside a buffered obstacle that is logged as a warning, never an error.
  * Cells outside the survey polygon are traversable at weight 1.0 (camera-off
    flight outside the plot is allowed); ONLY the buffered-obstacle occupancy
    fraction can block a cell.

Occupancy is measured against ``env.buffered_obstacles`` -- the same
clearance-buffered union ``path_clear`` and the connector/transit routers
operate on (environment_map.py, visibility_router.py) -- so a map-derived path
respects the regulatory clearance. One documented residue (Stage-3 review):
a sub-cell RAW intrusion inside a YELLOW (partially occupied) cell can still
sit on a center-to-center hop and trip S_OBS -- the yellow x1.5 penalty steers
Dijkstra around such cells whenever a comparable green corridor exists, and
the runtime S_OBS sidestep remains the net for the rest.

Edge cost = cruise hop energy x max(penalty of the two cells it connects).
The max is deliberate and conservative: a hop into OR out of a yellow cell
pays the full yellow price, so a route can never hide an expensive cell behind
a cheap neighbor. Known limit (documented, accepted): a diagonal between two
free cells is allowed even when the two shared orthogonal neighbors are red
(corner-cutting); the clearance buffer plus the >=50% red threshold keep the
practical corridor loss small.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra

from ..infrastructure.core_types import Pose
from ..infrastructure.enums import ManeuverType
from ..physical_model.energy_model import EnergyModel
from .environment_map import EnvironmentMap, GridFrame

log = logging.getLogger(__name__)

# A cell is "free" (weight 1.0) below this occupancy fraction; at or above it
# (and below red_threshold) it is "yellow". Guards float noise from the exact
# Shapely area intersection, not a tunable.
_FREE_FRAC_EPS = 1e-9

# scipy.sparse.csgraph.dijkstra's sentinel for "no predecessor".
_SCIPY_NO_PRED = -9999


@dataclass(frozen=True)
class EnergyMap:
    """Per-replication return-cost grid: E_home (J) + base-ward parent pointers.

    ``e_home``  -- float64 (nx, ny): minimum CRUISE return energy from the cell
                   center to the base cell; ``np.inf`` where unreachable.
    ``parent``  -- int32 (nx, ny): flat index (``i * ny + j``) of the next cell
                   toward the base; ``-1`` at the base cell and wherever
                   unreachable.
    ``penalty`` -- float64 (nx, ny): the occupancy penalty grid the build used
                   (1.0 free/base, ``yellow_penalty`` yellow, ``np.inf``
                   blocked).
    """

    frame: GridFrame
    e_home: np.ndarray
    parent: np.ndarray
    penalty: np.ndarray

    @property
    def base_cell(self) -> tuple[int, int]:
        """(i, j) of the cell with E_home == 0 (the parent-chain root)."""
        flat = int(np.argmin(self.e_home))
        return (flat // self.frame.ny, flat % self.frame.ny)


def battery_tied_cell_m(
    capacity_j: float, cruise_power_w: float, v_cruise: float, frac: float = 1e-3
) -> float:
    """Battery-tied grid resolution (design doc section 3): the cell edge is the
    distance whose CRUISE cost is exactly ``frac`` of the battery capacity --
    ``frac * capacity / (P_cruise / v_cruise)``. Defaults (360 kJ, 220 W,
    12 m/s) give ~19.64 m. Strictly battery-tied, never obstacle-tied."""
    if not all(math.isfinite(v) and v > 0 for v in (capacity_j, cruise_power_w, v_cruise, frac)):
        raise ValueError("battery_tied_cell_m requires finite positive inputs")
    return frac * capacity_j * v_cruise / cruise_power_w


def _base_aligned_frame(
    env: EnvironmentMap, base_pose: Pose, cell_m: float, margin_m: float = 0.0
) -> GridFrame:
    """A GridFrame whose lattice puts ``base_pose`` at the CENTER of its cell,
    extended by whole cells to cover ``env.area.bounds`` inflated by
    ``margin_m`` plus a one-cell pad.

    ``margin_m`` (Stage 2, author's universal-extent rule): the engine passes
    ``coverage.operating_margin_m`` so that every pose a drone can physically
    occupy is IN the grid by construction -- the bbox is convex, so straight
    RTH/transit chords between in-bbox points stay in-bbox, and the
    ferry/transit routers detour at most ``operating_margin_m`` outside the
    hull (S_OBS sidesteps are 15 m). Base-alignment is unaffected: only the
    extent grows, the origin logic is identical (default 0.0 => the Stage-1
    frame, byte-identical)."""
    minx, miny, maxx, maxy = env.area.bounds
    minx -= margin_m
    miny -= margin_m
    maxx += margin_m
    maxy += margin_m
    # Lower-left corner of the base cell on the base-anchored lattice.
    bx = base_pose.x - 0.5 * cell_m
    by = base_pose.y - 0.5 * cell_m
    # Shift down by whole cells until the grid also covers (min bound - 1 cell).
    i0 = max(0, math.ceil((bx - (minx - cell_m)) / cell_m))
    j0 = max(0, math.ceil((by - (miny - cell_m)) / cell_m))
    origin_x = bx - i0 * cell_m
    origin_y = by - j0 * cell_m
    nx = max(1, math.ceil((max(maxx, base_pose.x) + cell_m - origin_x) / cell_m))
    ny = max(1, math.ceil((max(maxy, base_pose.y) + cell_m - origin_y) / cell_m))
    return GridFrame(origin_x, origin_y, cell_m, nx, ny)


def _occupancy_fraction(env: EnvironmentMap, frame: GridFrame) -> np.ndarray:
    """Exact per-cell occupancy fraction against the BUFFERED obstacle union:
    ``area(cell intersect env.buffered_obstacles) / area(cell)``, shape (nx, ny).

    Vectorised Shapely 2.x: one prepared-predicate ``intersects`` prefilter
    (the union is already ``shapely.prepare``d by EnvironmentMap), then the
    exact area intersection only on the touched cells."""
    nx, ny, cell = frame.nx, frame.ny, frame.cell_m
    union = env.buffered_obstacles
    frac = np.zeros((nx, ny), dtype=np.float64)
    if union is None:
        return frac
    xs = frame.origin_x + np.arange(nx, dtype=np.float64) * cell
    ys = frame.origin_y + np.arange(ny, dtype=np.float64) * cell
    x0, y0 = np.meshgrid(xs, ys, indexing="ij")
    boxes = shapely.box(x0.ravel(), y0.ravel(), x0.ravel() + cell, y0.ravel() + cell)
    hit = shapely.intersects(union, boxes)  # prepared receiver -> cheap prune
    if hit.any():
        inter = shapely.intersection(boxes[hit], union)
        flat = frac.ravel()
        flat[hit] = shapely.area(inter) / (cell * cell)
    return frac


def build_energy_map(
    env: EnvironmentMap,
    base_pose: Pose,
    cell_m: float,
    em: EnergyModel,
    v_cruise: float,
    *,
    yellow_penalty: float = 1.5,
    red_threshold: float = 0.5,
    margin_m: float = 0.0,
) -> EnergyMap:
    """Build the per-replication return-cost map: occupancy penalty grid +
    one Dijkstra from the base cell. Deterministic (no RNG); pure function of
    (env geometry, base_pose, cell_m, energy model, parameters).

    Cell classes from the buffered-obstacle occupancy fraction ``f``:
    free ``f < eps`` -> x1.0; yellow ``eps <= f < red_threshold`` ->
    x``yellow_penalty``; red ``f >= red_threshold`` -> blocked (no incident
    edges). The base cell is forced traversable at x1.0 regardless (it is the
    return target); a base inside an obstacle logs a warning, never raises.
    """
    # Guard non-finite inputs for direct callers (config load already validates
    # the gated path): NaN slips through ordinary comparisons and would corrupt
    # grid sizing or turn yellow cells into blocked ones.
    if not math.isfinite(cell_m) or cell_m <= 0:
        raise ValueError("build_energy_map requires a finite cell_m > 0")
    if not math.isfinite(yellow_penalty) or yellow_penalty < 1.0:
        raise ValueError("build_energy_map requires a finite yellow_penalty >= 1.0")
    if not math.isfinite(red_threshold) or not (0.0 < red_threshold <= 1.0):
        raise ValueError("build_energy_map requires a finite red_threshold in (0, 1]")
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("build_energy_map requires a finite margin_m >= 0")
    frame = _base_aligned_frame(env, base_pose, cell_m, margin_m)
    nx, ny = frame.nx, frame.ny
    n = nx * ny

    frac = _occupancy_fraction(env, frame)
    penalty = np.ones((nx, ny), dtype=np.float64)
    penalty[frac >= _FREE_FRAC_EPS] = yellow_penalty
    penalty[frac >= red_threshold] = np.inf

    # Base cell: ALWAYS traversable at x1.0 -- it is the return target and
    # E_home[base] = 0 by definition. A physically obstructed base is a logged
    # (logical) condition, not a build failure.
    bi, bj = frame.world_to_cell(base_pose.x, base_pose.y)
    if not (0 <= bi < nx and 0 <= bj < ny):  # impossible by frame construction
        raise ValueError(f"base cell {(bi, bj)} outside the grid {(nx, ny)}")
    if not np.isfinite(penalty[bi, bj]) or (
        env.buffered_obstacles is not None
        and env.buffered_obstacles.covers(shapely.Point(base_pose.as_xy()))
    ):
        log.warning(
            "energy_map: base pose (%.1f, %.1f) lies in/under a buffered obstacle "
            "(cell occupancy %.2f); base cell forced traversable.",
            base_pose.x, base_pose.y, frac[bi, bj],
        )
    penalty[bi, bj] = 1.0
    base_flat = bi * ny + bj

    # 8-connected edges between traversable cells; undirected, built once per
    # unique direction. Edge cost = cruise hop energy x max(cell penalties).
    traversable = np.isfinite(penalty)
    flat_idx = np.arange(n, dtype=np.int64).reshape(nx, ny)
    hop_orth = em.distance_energy(cell_m, ManeuverType.CRUISE, v_cruise)
    hop_diag = em.distance_energy(math.sqrt(2.0) * cell_m, ManeuverType.CRUISE, v_cruise)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def _shift(d: int, size: int) -> tuple[slice, slice]:
        if d >= 0:
            return slice(0, size - d), slice(d, size)
        return slice(-d, size), slice(0, size + d)

    for di, dj, hop in ((1, 0, hop_orth), (0, 1, hop_orth),
                        (1, 1, hop_diag), (1, -1, hop_diag)):
        si, ti = _shift(di, nx)
        sj, tj = _shift(dj, ny)
        ok = traversable[si, sj] & traversable[ti, tj]
        if not ok.any():
            continue
        w = hop * np.maximum(penalty[si, sj][ok], penalty[ti, tj][ok])
        rows.append(flat_idx[si, sj][ok])
        cols.append(flat_idx[ti, tj][ok])
        data.append(w)

    if data:
        graph = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()
    else:
        graph = coo_matrix((n, n)).tocsr()

    dist, pred = _csgraph_dijkstra(
        graph, directed=False, indices=base_flat, return_predecessors=True
    )
    # scipy's predecessor of cell c on the base->c shortest path IS the next
    # base-ward hop from c. Sentinel -9999 (base itself + unreachable) -> -1.
    parent = pred.astype(np.int32)
    parent[parent == _SCIPY_NO_PRED] = -1

    return EnergyMap(
        frame=frame,
        e_home=dist.reshape(nx, ny),
        parent=parent.reshape(nx, ny),
        penalty=penalty,
    )


def route_home(emap: EnergyMap, from_pose: Pose) -> list[Pose]:
    """Follow the parent pointers from ``from_pose``'s cell to the base ->
    waypoint polyline of cell centers (heading 0; the motion-model smoothing
    pass is the Stage-3 consumer's job). Read-only helper -- nothing in the
    mission path calls this yet.

    Raises ``ValueError`` when ``from_pose`` falls outside the grid or in a
    cell with no route to the base (``E_home == inf``)."""
    frame = emap.frame
    nx, ny = frame.nx, frame.ny
    i, j = frame.world_to_cell(from_pose.x, from_pose.y)
    if not (0 <= i < nx and 0 <= j < ny):
        raise ValueError(
            f"route_home: pose ({from_pose.x:.1f}, {from_pose.y:.1f}) is outside "
            f"the energy-map grid ({nx}x{ny} cells)"
        )
    if not np.isfinite(emap.e_home[i, j]):
        raise ValueError(
            f"route_home: cell {(i, j)} is unreachable from the base (E_home=inf)"
        )
    parent_flat = emap.parent.ravel()
    poses: list[Pose] = []
    flat = i * ny + j
    for _ in range(nx * ny + 1):
        ci, cj = divmod(flat, ny)
        cx, cy = frame.cell_center(ci, cj)
        poses.append(Pose(cx, cy, 0.0))
        nxt = int(parent_flat[flat])
        if nxt < 0:
            return poses
        flat = nxt
    raise RuntimeError("route_home: parent chain did not terminate (corrupt map)")
