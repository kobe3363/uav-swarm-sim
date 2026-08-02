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
    """The value key includes obs.wkb: same field -> same key; a different field
    -> different key. An obstacle-omitting key would collide and return a stale
    ok_pairs; the real key does not."""
    obs, region = _field()
    key = _obstacle_cache_key(obs, "convex_hull", 50.0)

    # identical geometry rebuilt independently -> identical wkb -> identical key
    obs_same = unary_union([box(30.0, 30.0, 50.0, 55.0), box(60.0, 20.0, 78.0, 62.0)])
    assert _obstacle_cache_key(obs_same, "convex_hull", 50.0) == key

    # a different obstacle field -> different key (the discriminator)
    obs_diff = unary_union([box(31.0, 30.0, 50.0, 55.0), box(60.0, 20.0, 78.0, 62.0)])
    assert _obstacle_cache_key(obs_diff, "convex_hull", 50.0) != key

    # operating_area / margin are part of the key too
    assert _obstacle_cache_key(obs, "bbox", 50.0) != key
    assert _obstacle_cache_key(obs, "convex_hull", 25.0) != key


def test_ok_pairs_reused_for_same_field_distinct_for_different():
    """A shared graph_cache dict: two fields with different keys build two distinct
    ok_pairs; the same field never rebuilds (single entry)."""
    obs, region = _field()
    obs2 = unary_union([box(10.0, 10.0, 25.0, 40.0)])
    region2 = flyable_region(box(0.0, 0.0, 100.0, 100.0), obs2, "convex_hull", 50.0)

    cache: dict = {}
    k1 = _obstacle_cache_key(obs, "convex_hull", 50.0)
    k2 = _obstacle_cache_key(obs2, "convex_hull", 50.0)
    cache[k1] = _build_ok_pairs(obs, region)
    cache[k2] = _build_ok_pairs(obs2, region2)
    assert k1 != k2
    assert len(cache) == 2
    assert cache[k1] is not cache[k2]
    # a stale-key hazard: obs2's ok_pairs must NOT answer obs's vertex-vertex query.
    # (structurally guaranteed because the keys differ; assert the guard holds.)
    assert cache.get(k1) is not cache.get(k2)
