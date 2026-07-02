"""S_FERRY Step 2 -- obstacle-aware routing of the camera-off inter-strip
connectors over the FLYABLE region (operating-area-minus-obstacles), NOT the
survey polygon.

The camera-off connector between two coverage strips is, by the free-flight
mission premise, allowed to leave the survey plot and traverse any obstacle-free
space. Today it is a blind straight chord ``motion.plan(a, b, TURN)``: fine when
unobstructed (it IS the geodesic), but when an obstacle prism sits on the chord
the runtime SafetyMonitor raises S_OBS and detours, while the analytical
``E_cover`` still charges the straight chord -- so analytical != execution on a
blocked chord. This module closes that gap at *plan time*: if the chord is
unobstructed the straight chord is returned unchanged (byte-identical); only a
BLOCKED chord is rerouted, via a reduced visibility graph over the buffered
obstacle vertices, and never through an obstacle.

Why a visibility graph and not the GVG: the GVG roadmap is the clearance
skeleton equidistant between *distinct obstacles* -- it carries no survey-shape
structure and is empty when there are no obstacle pairs, so it cannot express a
shortest obstacle-avoiding chord between two arbitrary strip endpoints. A
vertex-visibility graph + Dijkstra is the right (and minimal) tool here.

The returned connector is a SINGLE ``Path`` (one leg) built by chaining
``motion.plan(v_i, v_{i+1}, TURN)`` along the polyline, so the executor's
structural connector-parity detection (``_cov_idx`` odd) and the S2<->S_FERRY
camera semantics are unchanged, and the per-segment energy is exactly today's
TURN cost integrated over the (possibly longer) routed length.
"""
from __future__ import annotations

import math

import networkx as nx
from shapely.geometry import LineString, box

from ..infrastructure.core_types import Path, Pose
from ..infrastructure.enums import ManeuverType

# The buffered obstacle union already carries the regulatory clearance; we strip
# a hair off it so that a graph vertex lying exactly ON that boundary (and an
# edge running along it) counts as clear rather than as a self-intersection.
_SKIN_EPS_M = 1e-3


def flyable_region(survey_poly, buffered_obstacles, operating_area: str, margin_m: float):
    """The region a camera-off connector may fly in: the operating area minus the
    buffered obstacles. The operating area is deliberately LARGER than the survey
    polygon (``convex_hull`` dilated by ``margin_m`` by default) so the notch of a
    concave shape, the hole of an annulus, and the near exterior are all flyable.
    """
    if operating_area == "survey":
        base = survey_poly
    elif operating_area == "bbox":
        minx, miny, maxx, maxy = survey_poly.bounds
        base = box(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)
    else:  # "convex_hull" (default)
        base = survey_poly.convex_hull.buffer(margin_m)
    if buffered_obstacles is not None:
        return base.difference(buffered_obstacles)
    return base


def _obstacle_vertices(buffered_obstacles, operating_area_poly):
    """Exterior+interior ring vertices of the buffered obstacle union that lie
    within the operating area -- the candidate turn points for a detour."""
    if buffered_obstacles is None:
        return []
    geoms = getattr(buffered_obstacles, "geoms", [buffered_obstacles])
    pts: list[tuple[float, float]] = []
    for g in geoms:
        rings = [g.exterior] + list(g.interiors)
        for ring in rings:
            for x, y in ring.coords:
                if operating_area_poly.covers(LineString([(x, y), (x, y)]).centroid):
                    pts.append((x, y))
    return pts


def _shortest_polyline(a_xy, b_xy, buffered_obstacles, operating_area_poly):
    """Reduced-visibility-graph shortest obstacle-avoiding polyline a->b, or None
    if the endpoints cannot be connected inside the flyable region."""
    nav_core = None
    if buffered_obstacles is not None:
        nav_core = buffered_obstacles.buffer(-_SKIN_EPS_M)
        if nav_core.is_empty:
            nav_core = None

    def edge_ok(p, q) -> bool:
        seg = LineString([p, q])
        if not operating_area_poly.covers(seg):
            return False
        if nav_core is not None and nav_core.intersects(seg):
            return False
        return True

    nodes = [a_xy, b_xy] + _obstacle_vertices(buffered_obstacles, operating_area_poly)
    # de-duplicate while keeping a and b at indices 0 and 1
    seen: dict[tuple[float, float], int] = {}
    uniq: list[tuple[float, float]] = []
    for p in nodes:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen[key] = len(uniq)
            uniq.append(p)

    g = nx.Graph()
    for i, p in enumerate(uniq):
        g.add_node(i, xy=p)
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if edge_ok(uniq[i], uniq[j]):
                g.add_edge(i, j, w=math.dist(uniq[i], uniq[j]))

    src = seen[(round(a_xy[0], 6), round(a_xy[1], 6))]
    dst = seen[(round(b_xy[0], 6), round(b_xy[1], 6))]
    try:
        idx = nx.shortest_path(g, src, dst, weight="w")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return [uniq[i] for i in idx]


def _chain_turn_legs(polyline, start: Pose, end: Pose, motion) -> Path:
    """Chain ``motion.plan(v_i, v_{i+1}, TURN)`` along the polyline into ONE Path.

    Each intermediate vertex is entered heading toward the next vertex, so the
    in-place yaw that ``motion.plan`` inserts at a vertex is cancelled by the
    following leg's entry yaw (near-zero dtheta) -- no spurious rotation energy.
    The final pose keeps ``end.heading`` (the next strip's heading), matching the
    straight-chord connector's arrival heading exactly.
    """
    # headings: each interior vertex faces the next; the last keeps end.heading
    poses: list[Pose] = [start]
    for i in range(1, len(polyline) - 1):
        nx_pt = polyline[i + 1]
        h = math.atan2(nx_pt[1] - polyline[i][1], nx_pt[0] - polyline[i][0])
        poses.append(Pose(polyline[i][0], polyline[i][1], h, start.z))
    poses.append(Pose(end.x, end.y, end.heading, start.z))

    segs = []
    cur = start
    for nxt in poses[1:]:
        leg = motion.plan(cur, nxt, ManeuverType.TURN)
        segs.extend(leg.segments)
        cur = leg.end_pose or nxt
    return Path.from_segments(segs)


def route_connector(
    a: Pose,
    b: Pose,
    motion,
    env,
    *,
    enabled: bool,
    operating_area: str = "convex_hull",
    margin_m: float = 50.0,
) -> Path:
    """The single source of truth for a camera-off connector's geometry.

    Returns the straight chord ``motion.plan(a, b, TURN)`` when routing is off,
    when there are no obstacles, or when the chord is unobstructed (all
    byte-identical to today). Only a chord blocked by a buffered obstacle is
    rerouted around it; if no obstacle-free polyline exists the straight chord is
    returned unchanged (a detour never makes a connector *worse*, and the runtime
    S_OBS recovery remains the safety net exactly as before).
    """
    chord = motion.plan(a, b, ManeuverType.TURN)
    if not enabled:
        return chord
    obs = env.buffered_obstacles
    if obs is None:
        return chord
    seg = LineString([a.as_xy(), b.as_xy()])
    if not obs.intersects(seg):
        return chord  # unobstructed -> straight chord (byte-identical)

    region = flyable_region(env.area, obs, operating_area, margin_m)
    polyline = _shortest_polyline(a.as_xy(), b.as_xy(), obs, region)
    if polyline is None or len(polyline) < 2:
        return chord  # boxed in -> fall back (never worse than today)
    return _chain_turn_legs(polyline, a, b, motion)
