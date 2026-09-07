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

import hashlib
import math

import networkx as nx
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry

from ..infrastructure.core_types import Path, Pose
from ..infrastructure.enums import ManeuverType

# The buffered obstacle union already carries the regulatory clearance; we strip
# a hair off it so that a graph vertex lying exactly ON that boundary (and an
# edge running along it) counts as clear rather than as a self-intersection.
_SKIN_EPS_M = 1e-3


class RouteUnavailable(ValueError):
    """No validated route; callers must not execute an unchecked chord."""


def geometry_key(env):
    """Identity of the static geometry, including clearance (not object identity)."""
    obs = env.buffered_obstacles
    return (hashlib.sha256(env.area.wkb).digest(),
            None if obs is None else hashlib.sha256(obs.wkb).digest())


def _require_endpoints(a, b, region):
    if not all(region.buffer(1e-8).covers(Point(p.as_xy())) for p in (a, b)):
        raise RouteUnavailable("endpoint_outside_free_space")


def flyable_region(survey_poly, buffered_obstacles, operating_area: str, margin_m: float) -> BaseGeometry:
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


def _obstacle_vertices(buffered_obstacles, operating_area_poly) -> list[tuple[float, float]]:
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


def _shortest_polyline(a_xy, b_xy, buffered_obstacles, operating_area_poly) -> list[tuple[float, float]] | None:
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


# --------------------------------------------------------------------------- #
# E3: visibility-graph caching for route_transit.
#
# The only expensive part of _shortest_polyline is the O(V**2) obstacle-vertex
# edge_ok loop (two shapely predicates per pair); it is ENDPOINT-INDEPENDENT --
# it depends solely on (buffered_obstacles, operating_area_poly), never on a/b.
# _build_ok_pairs memoises exactly that result (once per obstacle field), and
# _shortest_polyline_cached splices it back into a per-(a, b) query so the built
# graph -- nodes, edge SET, and edge INSERTION ORDER (which Dijkstra tie-breaks
# on) -- is byte-identical to _shortest_polyline. a/b enter only as two fresh
# nodes; the cache is consulted purely by coordinate-key pair, never by node
# index, and only for genuine vertex-vertex pairs, so the dedup index-shift when
# a/b coincides with an obstacle vertex is irrelevant (see route_transit and the
# forced-collision seam test).
# --------------------------------------------------------------------------- #


def _round_key(p) -> tuple[float, float]:
    return (round(p[0], 6), round(p[1], 6))


def _pair_key(ki, kj):
    """Order-normalised unordered key pair, so (ki, kj) and (kj, ki) collide."""
    return (ki, kj) if ki <= kj else (kj, ki)


def _build_ok_pairs(buffered_obstacles, operating_area_poly) -> set:
    """The endpoint-independent O(V**2) visibility result: the SET of unordered
    obstacle-vertex pairs (by 6-decimal coordinate key) for which ``edge_ok``
    holds. Mirrors the ``edge_ok`` / vertex-dedup logic of ``_shortest_polyline``
    exactly (a/b excluded). Only the True (visible) pairs are stored -- a pair
    absent from the set is not visible -- so memory is O(visible edges), not
    O(V**2): the two representations are byte-equivalent because the caller only
    ever asks whether a pair is visible."""
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

    verts = _obstacle_vertices(buffered_obstacles, operating_area_poly)
    seen: dict[tuple[float, float], int] = {}
    vuniq: list[tuple[float, float]] = []
    vkey: list[tuple[float, float]] = []
    for p in verts:
        k = _round_key(p)
        if k not in seen:
            seen[k] = len(vuniq)
            vuniq.append(p)
            vkey.append(k)

    ok_pairs: set = set()
    for i in range(len(vuniq)):
        for j in range(i + 1, len(vuniq)):
            if edge_ok(vuniq[i], vuniq[j]):
                ok_pairs.add(_pair_key(vkey[i], vkey[j]))
    return ok_pairs


def _shortest_polyline_cached(
    a_xy, b_xy, buffered_obstacles, operating_area_poly, ok_pairs
) -> list[tuple[float, float]] | None:
    """Byte-identical twin of ``_shortest_polyline`` that reuses a prebuilt
    ``ok_pairs`` for the obstacle-vertex edges. The ONLY difference from
    ``_shortest_polyline`` is the source of the vertex-vertex ``edge_ok`` decision
    (cache lookup instead of a fresh shapely call); nodes, dedup, loop order,
    weights, and Dijkstra are unchanged."""
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

    verts = _obstacle_vertices(buffered_obstacles, operating_area_poly)
    # combined dedup -- IDENTICAL order to _shortest_polyline ([a, b] first, then
    # verts) -- while tagging each surviving node's origin. A node is a VERTEX iff
    # a/b did NOT claim its key first; its raw coords are then the first verts
    # occurrence of that key, exactly what _build_ok_pairs keyed on.
    seen: dict[tuple[float, float], int] = {}
    uniq: list[tuple[float, float]] = []
    is_vertex: list[bool] = []
    for p in (a_xy, b_xy):
        k = _round_key(p)
        if k not in seen:
            seen[k] = len(uniq)
            uniq.append(p)
            is_vertex.append(False)
    for p in verts:
        k = _round_key(p)
        if k not in seen:
            seen[k] = len(uniq)
            uniq.append(p)
            is_vertex.append(True)

    g = nx.Graph()
    for i, p in enumerate(uniq):
        g.add_node(i, xy=p)
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if is_vertex[i] and is_vertex[j]:
                ok = _pair_key(_round_key(uniq[i]), _round_key(uniq[j])) in ok_pairs
            else:
                ok = edge_ok(uniq[i], uniq[j])
            if ok:
                g.add_edge(i, j, w=math.dist(uniq[i], uniq[j]))

    src = seen[_round_key(a_xy)]
    dst = seen[_round_key(b_xy)]
    try:
        idx = nx.shortest_path(g, src, dst, weight="w")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return [uniq[i] for i in idx]


def _obstacle_cache_key(buffered_obstacles, operating_area_poly):
    """Value-based cache key for every dependency of ``ok_pairs``.

    ``ok_pairs`` depends on the obstacle geometry and on the already-built
    flyable region.  Hashing ``region.wkb`` keeps the key correct even if a
    caller deliberately shares a cache across environments with different
    survey areas; it is not merely an engine-lifetime assumption.
    """
    return (
        hashlib.sha1(buffered_obstacles.wkb).digest(),
        hashlib.sha1(operating_area_poly.wkb).digest(),
    )


def _path_clear(path: Path, env, ds: float = 1.0, *, region=None) -> bool:
    """Planner acceptance against buffered obstacles; NOT a safety event.

    Each maneuver is checked separately, so sampling cannot cut across a yaw
    vertex. Touching the buffer boundary is permitted (the full clearance is
    attained); penetrating its interior is not. The 1e-8 m erosion only absorbs
    floating point coordinate noise. Region may extend outside the survey.
    """
    obs = env.buffered_obstacles
    core = None if obs is None else obs.buffer(-1e-8)
    accepted_region = None if region is None else region.buffer(1e-8)
    for seg in path.segments:
        pts = Path((seg,)).sample(ds)
        coords = [p.as_xy() for p in pts]
        if not coords:
            continue
        shape = Point(coords[0]) if seg.length_m == 0 else LineString(coords)
        if accepted_region is not None and not accepted_region.covers(shape):
            return False
        if core is not None and core.intersects(shape):
            return False
    return True


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

    Routing off retains the legacy chord. Routing on returns only a validated
    path, or raises RouteUnavailable. No runtime skip is assumed to repair it.
    """
    chord = motion.plan(a, b, ManeuverType.TURN)
    if not enabled:
        return chord
    obs = env.buffered_obstacles
    region = flyable_region(env.area, obs, operating_area, margin_m)
    _require_endpoints(a, b, region)
    if _path_clear(chord, env, region=region):
        return chord
    polyline = _shortest_polyline(a.as_xy(), b.as_xy(), obs, region)
    if polyline is None or len(polyline) < 2:
        raise RouteUnavailable("connector_blocked")
    routed = _chain_turn_legs(polyline, a, b, motion)
    # validate the realized motion (arcs may bulge where the polyline was straight)
    if not _path_clear(routed, env, region=region):
        raise RouteUnavailable("connector_invalid")
    return routed


def _chain_cruise_legs(polyline, start: Pose, end: Pose, motion) -> Path:
    """The CRUISE twin of ``_chain_turn_legs``: chain ``motion.plan(v_i, v_{i+1},
    CRUISE)`` along the polyline into ONE Path, with the same heading treatment
    (each interior vertex faces the next; the final pose keeps ``end.heading``).
    Kept separate so the connector chain stays byte-untouched."""
    poses: list[Pose] = [start]
    for i in range(1, len(polyline) - 1):
        nx_pt = polyline[i + 1]
        h = math.atan2(nx_pt[1] - polyline[i][1], nx_pt[0] - polyline[i][0])
        poses.append(Pose(polyline[i][0], polyline[i][1], h, start.z))
    poses.append(Pose(end.x, end.y, end.heading, start.z))

    segs = []
    cur = start
    for nxt in poses[1:]:
        leg = motion.plan(cur, nxt, ManeuverType.CRUISE)
        segs.extend(leg.segments)
        cur = leg.end_pose or nxt
    return Path.from_segments(segs)


def route_transit(
    a: Pose,
    b: Pose,
    motion,
    env,
    *,
    enabled: bool,
    operating_area: str = "convex_hull",
    margin_m: float = 50.0,
    graph_cache: dict | None = None,
) -> Path:
    """FIX-B1: the single source of truth for an S1 TRANSIT leg's geometry --
    the CRUISE twin of ``route_connector`` (which stays connector-only and
    byte-untouched, since its chord and chained legs are TURN-typed).

    A blind straight transit chord that crosses an obstacle prism is the root
    of the swap livelock: the runtime S_OBS recovery cannot make lateral
    progress on a transit leg (the resume replays the same chord), the boxed-in
    escalation sends the drone home, the landing swaps unconditionally, and the
    relaunch replays the identical blocked chord forever. Routing the chord
    around the buffered obstacles at plan time removes the collision course
    before the SafetyMonitor ever sees it.

    Semantics mirror route_connector: an enabled router returns only a
    buffer-clear realized path, or raises RouteUnavailable. Disabled routing
    retains the legacy chord.

    ``graph_cache`` (E3): an optional per-replication dict memoising the
    endpoint-independent O(V**2) obstacle-vertex visibility result. When ``None``
    (default) the visibility graph is rebuilt from scratch, byte-identical to the
    original behaviour; when supplied, the shared result is spliced into each
    query, byte-identically (see ``_shortest_polyline_cached``).
    """
    chord = motion.plan(a, b, ManeuverType.CRUISE)
    if not enabled:
        return chord
    obs = env.buffered_obstacles
    region = flyable_region(env.area, obs, operating_area, margin_m)
    _require_endpoints(a, b, region)
    if _path_clear(chord, env, region=region):
        return chord
    if graph_cache is None:
        polyline = _shortest_polyline(a.as_xy(), b.as_xy(), obs, region)
    else:
        key = _obstacle_cache_key(obs, region)
        ok_pairs = graph_cache.get(key)
        if ok_pairs is None:
            ok_pairs = _build_ok_pairs(obs, region)
            graph_cache[key] = ok_pairs
        polyline = _shortest_polyline_cached(a.as_xy(), b.as_xy(), obs, region, ok_pairs)
    if polyline is None or len(polyline) < 2:
        raise RouteUnavailable("transit_blocked")
    routed = _chain_cruise_legs(polyline, a, b, motion)
    # validate the realized motion (arcs may bulge where the polyline was straight)
    if not _path_clear(routed, env, region=region):
        raise RouteUnavailable("transit_invalid")
    return routed
