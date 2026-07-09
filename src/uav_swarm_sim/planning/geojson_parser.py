"""Turn a GeoJSON exploration-area boundary into a Shapely polygon in a local
metric frame.

Geographic coordinates (lon/lat) are detected heuristically and projected with
a local equirectangular projection around the polygon centroid -- adequate for
areas up to tens of km and dependency-free. Coordinates that are already metric
(magnitudes beyond lon/lat bounds) are used as-is.
"""
from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path as FsPath

from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

_LOG = logging.getLogger(__name__)
_EARTH_R = 6_371_000.0


def _looks_geographic(poly_coords: list[tuple[float, float]]) -> bool:
    return all(abs(x) <= 180.0 and abs(y) <= 90.0 for x, y in poly_coords)


def _project_equirectangular(poly: Polygon) -> Polygon:
    cx, cy = poly.centroid.x, poly.centroid.y  # lon0, lat0
    lat0 = math.radians(cy)

    def tf(x: float, y: float) -> tuple[float, float]:
        return (
            _EARTH_R * math.radians(x - cx) * math.cos(lat0),
            _EARTH_R * math.radians(y - cy),
        )

    ext = [tf(x, y) for x, y in poly.exterior.coords]
    holes = [[tf(x, y) for x, y in ring.coords] for ring in poly.interiors]
    return Polygon(ext, holes)


def _first_polygon(geom) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda g: g.area)
        _LOG.warning("MultiPolygon area: using largest of %d parts", len(geom.geoms))
        return largest
    raise ValueError(f"unsupported geometry type for area boundary: {geom.geom_type}")


@lru_cache(maxsize=None)
def load_area(geojson_path: str | FsPath) -> Polygon:
    """Parse a survey-area GeoJSON into a Shapely ``Polygon`` (metric frame).

    Memoized: this is a deterministic pure function of the path, and the sweeps
    load the same handful of shape files hundreds of times (once per analytical
    pass and once per SimulationEngine build). Shapely geometries are immutable
    and every caller uses the result read-only, so sharing the cached polygon is
    byte-identical to re-parsing. (The area set is tiny -- the 9 shapes plus a
    couple of test areas -- so an unbounded cache is safe.)
    """
    data = json.loads(FsPath(geojson_path).read_text())

    gtype = data.get("type")
    if gtype == "FeatureCollection":
        feats = data.get("features", [])
        polys = [f for f in feats if f.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon")]
        if not polys:
            raise ValueError("FeatureCollection has no Polygon/MultiPolygon feature")
        geom = shape(polys[0]["geometry"])
    elif gtype == "Feature":
        geom = shape(data["geometry"])
    elif gtype in ("Polygon", "MultiPolygon"):
        geom = shape(data)
    else:
        raise ValueError(f"unsupported top-level GeoJSON type: {gtype}")

    poly = _first_polygon(geom)
    poly = make_valid(poly)
    poly = _first_polygon(poly) if poly.geom_type in ("Polygon", "MultiPolygon") else poly
    if not isinstance(poly, Polygon):
        raise ValueError("area did not resolve to a single Polygon after validation")

    if _looks_geographic(list(poly.exterior.coords)):
        _LOG.info("area coordinates look geographic; projecting equirectangular")
        poly = _project_equirectangular(poly)

    poly = orient(poly, sign=1.0)  # CCW exterior
    if poly.area <= 0:
        raise ValueError("area polygon has non-positive area")
    return poly
