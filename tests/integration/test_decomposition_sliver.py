"""Regression: degenerate-geometry (sliver) guard in the decomposition core.

Splitting a CONCAVE region with ``_split_longest_axis`` can produce a
``GeometryCollection`` (a real polygon plus a zero-area line artifact); in
Shapely 2.x such a collection has ``.boundary is None``, which used to crash
``build_work_adjacency``. The guard extracts polygonal content and drops the
line/point artifacts, preserving area. These tests pin both the unit behaviour
and the full-mission path that first surfaced the crash (C-shape, n=4).
"""
from __future__ import annotations

import json

import pytest
from shapely.affinity import scale
from shapely.geometry import Polygon

from uav_swarm_sim.experiments.generate_shapes import _c_shape, _l_shape
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.planning.decomposition_base import (
    _WorkRegion,
    _polygon_parts,
    ensure_enough_regions,
)


def _write_geojson(path, poly: Polygon) -> str:
    path.write_text(json.dumps({
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in poly.exterior.coords]]},
    }))
    return str(path)


# --------------------------------------------------------------------------- #
# unit: the split path yields only polygons and never loses area              #
# --------------------------------------------------------------------------- #
def test_subdividing_concave_region_yields_only_polygons_and_preserves_area():
    poly = scale(_c_shape(), xfact=300.0, yfact=300.0, origin=(0, 0))  # concave
    seed = _WorkRegion(0, poly, Pose(poly.centroid.x, poly.centroid.y, 0.0), poly.area)

    work = ensure_enough_regions([seed], 4)

    assert len(work) >= 4
    # no degenerate geometry survives (this is what crashed build_work_adjacency)
    assert all(w.geom.geom_type == "Polygon" for w in work)
    assert all(w.geom.boundary is not None for w in work)
    # area is preserved -- the guard drops only zero-area line/point artifacts
    assert sum(w.area for w in work) == pytest.approx(poly.area, rel=1e-9)


def test_polygon_parts_drops_lines_keeps_polygons():
    from shapely.geometry import GeometryCollection, LineString
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gc = GeometryCollection([poly, LineString([(0, 0), (5, 5)])])
    parts = _polygon_parts(gc)
    assert [p.geom_type for p in parts] == ["Polygon"]
    assert parts[0].area == pytest.approx(100.0)
    assert _polygon_parts(LineString([(0, 0), (1, 1)])) == []


# --------------------------------------------------------------------------- #
# full mission: C-shape at n=4 (the reported crash) now decomposes and covers  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape_fn,width", [(_c_shape, 600.0), (_l_shape, 600.0)])
def test_concave_shape_n4_full_mission_no_crash(config_path, tmp_path, shape_fn, width):
    poly = scale(shape_fn(), xfact=width / 2.0, yfact=width / 2.0, origin=(0, 0))
    gj = _write_geojson(tmp_path / "concave.geojson", poly)
    cfg = load_config(config_path, overrides={
        "env.geojson_path": gj,
        "env.obstacle_density_per_km2": 0.0,
        "failure.hazard_rate_per_hour": 0.0,
        "fleet.n_drones": 4,
        "fleet.battery_capacity_wh": 400.0,
        "sim.max_timesteps": 300000,
        "telemetry.enabled": False,
    })
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()  # used to raise AttributeError in build_work_adjacency

    assert not result.aborted
    assert len(eng.partition.zones) == 4
    assert result.coverage_frac > 0.99
    # every zone is a clean polygon and the partition tiles the full survey area
    assert all(z.polygon.geom_type in ("Polygon", "MultiPolygon") for z in eng.partition.zones.values())
    zone_sum = sum(z.polygon.area for z in eng.partition.zones.values())
    assert zone_sum == pytest.approx(poly.area, rel=1e-6)
