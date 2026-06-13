"""Boustrophedon (lawnmower) coverage path within an assigned zone.

Strips run along the zone's long axis (fewest turns => least energy, the §1.5.1
argument made operational). For non-holonomic platforms the inter-strip
connectors are planned by the motion model so U-turns respect the minimum turn
radius; a tight-strip guard switches to interleaved strip order when
2*r_min > swath. Energy is computed via the shared EnergyModel (P*dt).
"""
from __future__ import annotations

import math

from shapely.affinity import rotate
from shapely.geometry import LineString, MultiLineString, Polygon

from ..infrastructure.core_types import (
    CoveragePlan,
    Path,
    Pose,
    Waypoint,
    Zone,
    straight_segment,
)
from ..infrastructure.enums import ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel


def _long_axis_angle(poly: Polygon) -> float:
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    longest = max(edges, key=lambda e: math.dist(e[0], e[1]))
    (x0, y0), (x1, y1) = longest
    return math.atan2(y1 - y0, x1 - x0)


def _strip_intervals(rot_poly: Polygon, swath: float) -> list[list[tuple[float, float, float]]]:
    minx, miny, maxx, maxy = rot_poly.bounds
    rows: list[list[tuple[float, float, float]]] = []
    y = miny + swath / 2.0
    while y <= maxy:
        scan = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        inter = rot_poly.intersection(scan)
        segs: list[tuple[float, float, float]] = []
        if isinstance(inter, LineString) and not inter.is_empty:
            xs = [c[0] for c in inter.coords]
            segs.append((min(xs), max(xs), y))
        elif isinstance(inter, MultiLineString):
            for ls in inter.geoms:
                xs = [c[0] for c in ls.coords]
                segs.append((min(xs), max(xs), y))
        if segs:
            rows.append(sorted(segs))
        y += swath
    return rows


def boustrophedon(
    zone: Zone, spec: PlatformSpec, motion: MotionModel, em: EnergyModel
) -> CoveragePlan:
    poly = zone.polygon
    if poly.is_empty or poly.area <= 0:
        return CoveragePlan(zone.drone_id, [], 0.0, 0.0)
    if not isinstance(poly, Polygon):
        poly = max(poly.geoms, key=lambda g: g.area)

    theta = _long_axis_angle(poly)
    cx, cy = poly.centroid.x, poly.centroid.y
    rot = rotate(poly, -math.degrees(theta), origin=(cx, cy))
    swath = spec.swath_width_m

    rows = _strip_intervals(rot, swath)

    # tight-strip guard: interleave strip order when a simple U-turn is infeasible
    order = list(range(len(rows)))
    if spec.r_min_m > 0 and 2 * spec.r_min_m > swath:
        order = list(range(0, len(rows), 2)) + list(range(1, len(rows), 2))

    # build ordered endpoints in the rotated frame, serpentine within each row
    endpoints: list[tuple[float, float]] = []
    flip = False
    for ridx in order:
        segs = rows[ridx]
        seq = segs if not flip else list(reversed(segs))
        for (x0, x1, yy) in seq:
            a, b = (x0, yy), (x1, yy)
            if flip:
                a, b = b, a
            endpoints.append(a)
            endpoints.append(b)
        flip = not flip

    # rotate endpoints back to world frame
    def unrot(p):
        ca, sa = math.cos(theta), math.sin(theta)
        dx, dy = p[0] - cx, p[1] - cy
        return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)

    world = [unrot(p) for p in endpoints]

    waypoints: list[Waypoint] = []
    length = 0.0
    energy = 0.0
    # iterate strip by strip: even index = strip start, odd = strip end
    for k in range(0, len(world) - 1, 2):
        s = world[k]
        e = world[k + 1]
        heading = math.atan2(e[1] - s[1], e[0] - s[0])
        strip_len = math.dist(s, e)
        waypoints.append(Waypoint(Pose(s[0], s[1], heading), ManeuverType.COVERAGE, spec.v_coverage))
        waypoints.append(Waypoint(Pose(e[0], e[1], heading), ManeuverType.COVERAGE, spec.v_coverage))
        length += strip_len
        energy += em.distance_energy(strip_len, ManeuverType.COVERAGE, spec.v_coverage)
        # connector to next strip start
        if k + 2 < len(world):
            nxt = world[k + 2]
            nh = math.atan2(world[k + 3][1] - nxt[1], world[k + 3][0] - nxt[0]) if k + 3 < len(world) else heading
            conn = motion.plan(Pose(e[0], e[1], heading), Pose(nxt[0], nxt[1], nh), ManeuverType.TURN)
            length += conn.total_length_m
            energy += em.path_energy(conn)

    return CoveragePlan(zone.drone_id, waypoints, length, energy)
