"""E3 seam tests: the cached visibility graph is byte-identical to the uncached
build, on every input, always.

``route_transit`` memoises the endpoint-independent O(V**2) obstacle-vertex
visibility result (``_build_ok_pairs``) and splices it into each (a, b) query
(``_shortest_polyline_cached``). These fast unit tests are the load-bearing guard
for that seam -- in particular the FORCED-COLLISION case (a/b coinciding with an
obstacle vertex's 6-decimal key), which the slow full-mission golden may never
exercise but which is exactly where an index-based cache would corrupt routing.
"""
from __future__ import annotations

import random

from shapely.geometry import box
from shapely.ops import unary_union

import uav_swarm_sim.planning.visibility_router as visibility_router
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.planning.visibility_router import (
    _build_ok_pairs,
    _obstacle_cache_key,
    _obstacle_vertices,
    _round_key,
    _shortest_polyline,
    _shortest_polyline_cached,
    flyable_region,
)


def _field():
    """A small obstacle field with a nontrivial visibility graph: two buffered
    prisms that block a band of chords, over a convex-hull operating area."""
    survey = box(0.0, 0.0, 100.0, 100.0)
    obs = unary_union([box(30.0, 30.0, 50.0, 55.0), box(60.0, 20.0, 78.0, 62.0)])
    region = flyable_region(survey, obs, "convex_hull", 50.0)
    return obs, region


def _assert_identical(a_xy, b_xy, obs, region, ok_pairs):
    uncached = _shortest_polyline(a_xy, b_xy, obs, region)
    cached = _shortest_polyline_cached(a_xy, b_xy, obs, region, ok_pairs)
    assert cached == uncached
    return uncached


def test_seam_byte_identical_over_random_queries():
    """Over many random (a, b), the cached splice reproduces the uncached polyline
    exactly (node-for-node), including the None/degenerate cases."""
    obs, region = _field()
    ok_pairs = _build_ok_pairs(obs, region)
    rng = random.Random(20260802)
    hits = 0  # queries that actually route around an obstacle (>2 nodes)
    for _ in range(400):
        a = (rng.uniform(2.0, 98.0), rng.uniform(2.0, 98.0))
        b = (rng.uniform(2.0, 98.0), rng.uniform(2.0, 98.0))
        poly = _assert_identical(a, b, obs, region, ok_pairs)
        if poly is not None and len(poly) > 2:
            hits += 1
    # the field must actually force detours, or the test proves nothing
    assert hits > 0


def test_forced_collision_a_on_obstacle_vertex():
    """LOAD-BEARING (author P1): a coincides with an obstacle vertex's 6-dec key.
    The vertex loses its dedup slot to a and every edge from that node is computed
    fresh -- the cache (coordinate-keyed, vertex-vertex only) must never substitute
    the vertex's result for a's. Bit-identity must hold."""
    obs, region = _field()
    ok_pairs = _build_ok_pairs(obs, region)
    verts = _obstacle_vertices(obs, region)
    assert verts, "fixture must expose obstacle vertices"
    vkeys = {_round_key(v) for v in verts}

    a_xy = verts[0]  # exact collision with an obstacle vertex
    assert _round_key(a_xy) in vkeys  # the collision is real
    rng = random.Random(4242)
    for _ in range(60):
        b = (rng.uniform(2.0, 98.0), rng.uniform(2.0, 98.0))
        _assert_identical(a_xy, b, obs, region, ok_pairs)


def test_forced_collision_both_endpoints_on_vertices():
    """Both a and b coincide with (distinct) obstacle vertices -- two dropped
    slots, maximal index shift. Still bit-identical."""
    obs, region = _field()
    ok_pairs = _build_ok_pairs(obs, region)
    verts = _obstacle_vertices(obs, region)
    # pick two vertices with distinct keys
    seen, distinct = set(), []
    for v in verts:
        k = _round_key(v)
        if k not in seen:
            seen.add(k)
            distinct.append(v)
        if len(distinct) == 2:
            break
    assert len(distinct) == 2
    _assert_identical(distinct[0], distinct[1], obs, region, ok_pairs)


def test_cache_key_discriminates_obstacle_field():
    """The value key contains every ``ok_pairs`` dependency, including region.

    The same obstacle field may be queried through a different survey area when
    a caller deliberately shares a cache.  That must not reuse a stale graph.
    """
    obs, region = _field()
    key = _obstacle_cache_key(obs, region)

    # identical geometry rebuilt independently -> identical wkb -> identical key
    obs_same = unary_union([box(30.0, 30.0, 50.0, 55.0), box(60.0, 20.0, 78.0, 62.0)])
    assert _obstacle_cache_key(obs_same, region) == key

    # a different obstacle field -> different key (the discriminator)
    obs_diff = unary_union([box(31.0, 30.0, 50.0, 55.0), box(60.0, 20.0, 78.0, 62.0)])
    assert _obstacle_cache_key(obs_diff, region) != key

    # Same obstacles, but a different survey area: region is a direct cache
    # dependency and must discriminate even with identical mode/margin values.
    region_diff = flyable_region(box(-25.0, 0.0, 100.0, 100.0), obs,
                                 "convex_hull", 50.0)
    assert _obstacle_cache_key(obs, region_diff) != key


def test_route_transit_builds_once_for_repeated_field(monkeypatch, config_path):
    """Exercise the production get-or-build path, not a hand-built dict.

    Repeated blocked chords through one environment must build ``ok_pairs`` once
    and reuse it on the second query.
    """
    motion = make_motion_model(build_spec(load_config(config_path)))
    env = EnvironmentMap(
        box(0.0, 0.0, 1000.0, 1000.0),
        [Obstacle(id=0, cls=0, polygon=box(450.0, 400.0, 550.0, 600.0))],
        5.0,
    )
    a = Pose(200.0, 500.0, 0.0)
    b = Pose(800.0, 500.0, 0.0)
    cache: dict = {}
    calls = 0
    real_build = visibility_router._build_ok_pairs

    def _counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(visibility_router, "_build_ok_pairs", _counted_build)
    for _ in range(2):
        routed = visibility_router.route_transit(
            a, b, motion, env, enabled=True, graph_cache=cache,
        )
        assert routed.total_length_m > motion.plan(a, b, ManeuverType.CRUISE).total_length_m

    assert calls == 1
    assert len(cache) == 1
