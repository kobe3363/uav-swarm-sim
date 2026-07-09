"""load_area is memoized (perf): same path -> same cached Polygon, distinct
paths stay distinct. Shapely geometries are immutable and callers use the result
read-only, so sharing the cached object is byte-identical to re-parsing."""
from __future__ import annotations

from uav_swarm_sim.planning.geojson_parser import load_area

_SQUARE = "data/areas/shapes/square.geojson"
_CSHAPE = "data/areas/shapes/c_shape.geojson"


def test_same_path_returns_cached_object():
    a = load_area(_SQUARE)
    b = load_area(_SQUARE)
    assert a is b               # memoized: identical object, no re-parse
    assert a.equals(b)


def test_distinct_paths_are_distinct_geometries():
    sq = load_area(_SQUARE)
    cs = load_area(_CSHAPE)
    assert sq is not cs
    assert not sq.equals(cs)
    assert sq.area > 0 and cs.area > 0
