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
from .visibility_router import route_connector


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
    if y > maxy:
        y = (miny + maxy) / 2.0
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


def _component_strips(
    poly: Polygon, swath: float, spec: PlatformSpec
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the ordered world-frame coverage strips for one component."""
    theta = _long_axis_angle(poly)
    cx, cy = poly.centroid.x, poly.centroid.y
    rot = rotate(poly, -math.degrees(theta), origin=(cx, cy))
    rows = _strip_intervals(rot, swath)

    order = list(range(len(rows)))
    if spec.r_min_m > 0 and 2 * spec.r_min_m > swath:
        order = list(range(0, len(rows), 2)) + list(range(1, len(rows), 2))

    endpoints: list[tuple[float, float]] = []
    flip = False
    for ridx in order:
        segs = rows[ridx]
        seq = segs if not flip else list(reversed(segs))
        for (x0, x1, yy) in seq:
            a, b = (x0, yy), (x1, yy)
            if flip:
                a, b = b, a
            endpoints.extend((a, b))
        flip = not flip

    ca, sa = math.cos(theta), math.sin(theta)

    def unrot(p):
        dx, dy = p[0] - cx, p[1] - cy
        return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)

    world = [unrot(p) for p in endpoints]
    return [(world[k], world[k + 1]) for k in range(0, len(world) - 1, 2)]


def boustrophedon(
    zone: Zone, spec: PlatformSpec, motion: MotionModel, em: EnergyModel,
    env=None, coverage=None, altitude_m: float | None = None,
) -> CoveragePlan:
    poly = zone.polygon
    if poly.is_empty or poly.area <= 0:
        return CoveragePlan(zone.drone_id, [], 0.0, 0.0)
    # EXP-01: the enabled nadir-camera model derives cross-track spacing from
    # this layer's AGL.  Legacy specs return their precomputed effective swath.
    swath = spec.coverage_line_spacing_m(altitude_m)
    components = (
        [poly]
        if isinstance(poly, Polygon)
        else [g for g in poly.geoms if isinstance(g, Polygon) and g.area > 0.0]
    )
    strips = [
        (start, end, component_index)
        for component_index, component in enumerate(components)
        for start, end in _component_strips(component, swath, spec)
    ]

    # S_FERRY Step 2: route camera-off connectors around obstacles when enabled.
    # Default (env is None or flag off) => straight chord, byte-identical.
    ferry_on = bool(coverage is not None and getattr(coverage, "ferry_free_space", False)
                    and env is not None)
    store_connectors = ferry_on or (len(components) > 1 and env is not None)

    waypoints: list[Waypoint] = []
    connectors: list[Path] = []
    length = 0.0
    energy = 0.0
    # iterate strip by strip: even index = strip start, odd = strip end
    for k, (s, e, component_index) in enumerate(strips):
        heading = math.atan2(e[1] - s[1], e[0] - s[0])
        strip_len = math.dist(s, e)
        waypoints.append(Waypoint(Pose(s[0], s[1], heading), ManeuverType.COVERAGE, spec.v_coverage))
        waypoints.append(Waypoint(Pose(e[0], e[1], heading), ManeuverType.COVERAGE, spec.v_coverage))
        length += strip_len
        energy += em.distance_energy(strip_len, ManeuverType.COVERAGE, spec.v_coverage)
        # connector to next strip start
        if k + 1 < len(strips):
            nxt, nxt_end, next_component_index = strips[k + 1]
            nh = math.atan2(nxt_end[1] - nxt[1], nxt_end[0] - nxt[0])
            a_pose = Pose(e[0], e[1], heading)
            b_pose = Pose(nxt[0], nxt[1], nh)
            if store_connectors:
                conn = route_connector(
                    a_pose, b_pose, motion, env,
                    enabled=ferry_on or component_index != next_component_index,
                    operating_area=getattr(coverage, "operating_area", "convex_hull"),
                    margin_m=getattr(coverage, "operating_margin_m", 50.0),
                )
                connectors.append(conn)
            else:
                conn = motion.plan(a_pose, b_pose, ManeuverType.TURN)
            length += conn.total_length_m
            energy += em.path_energy(conn)

    return CoveragePlan(zone.drone_id, waypoints, length, energy, connectors=connectors)
