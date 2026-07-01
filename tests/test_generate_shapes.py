"""Tests for the A3 equal-area shape generator (experiments/generate_shapes.py).

The shape family feeds the shape study: every member must have EXACTLY the same
area (so coverage energy differs only by shape, not size), the concavity ordering
(solidity = area / hull_area) must be the one the hypotheses assume, and a written
shape must load back through the real GeoJSON parser and fly a real mission.
"""
from __future__ import annotations

import math

import pytest
from conftest import config_path

from uav_swarm_sim.experiments.generate_shapes import (
    build_all,
    describe,
    normalize_to_area,
    shape_builders,
    write_shape,
)
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.planning.geojson_parser import load_area

_TARGET = 1_000_000.0
_SWATH = 50.0


def test_all_shapes_hit_target_area_exactly():
    # build_all itself asserts < 1e-6; re-check independently on the returned polys.
    for name, poly, _ in build_all(_TARGET, _SWATH, disk_sides=128):
        rel = abs(poly.area - _TARGET) / _TARGET
        assert rel < 1e-6, f"{name}: area rel-err {rel:.2e}"


def test_shapes_are_positive_quadrant_and_valid():
    for name, poly, _ in build_all(_TARGET, _SWATH, disk_sides=128):
        assert poly.is_valid, f"{name} invalid"
        minx, miny, _, _ = poly.bounds
        # normalisation translates the bbox min to the origin (loader heuristic).
        assert minx == pytest.approx(0.0, abs=1e-6)
        assert miny == pytest.approx(0.0, abs=1e-6)


def test_solidity_ordering_matches_hypotheses():
    d = {name: desc for name, _, desc in build_all(_TARGET, _SWATH, disk_sides=128)}
    # convex members: solidity == 1 (area == hull area).
    for convex in ("square", "rect_2_1", "rect_4_1", "rect_8_1", "disk"):
        assert d[convex]["solidity"] == pytest.approx(1.0, abs=1e-3), convex
    # concave members strictly below 1, in the intended severity order.
    assert d["l_shape"]["solidity"] < 1.0
    assert d["star_5"]["solidity"] < d["l_shape"]["solidity"]
    assert d["pinwheel"]["solidity"] < d["star_5"]["solidity"]


def test_disk_is_the_isoperimetric_floor():
    d = {name: desc for name, _, desc in build_all(_TARGET, _SWATH, disk_sides=256)}
    # P^2 / (4 pi A) == 1 for a perfect circle; a 256-gon is essentially there.
    assert d["disk"]["isoperimetric"] == pytest.approx(1.0, abs=5e-3)
    # every other shape is less compact (higher ratio).
    for name, desc in d.items():
        if name != "disk":
            assert desc["isoperimetric"] >= d["disk"]["isoperimetric"]


def test_normalize_rejects_degenerate():
    from shapely.geometry import Polygon

    with pytest.raises(ValueError):
        normalize_to_area(Polygon(), _TARGET)


def test_written_shape_loads_and_flies(tmp_path):
    # square, normalised, written to disk, read back through the real parser.
    poly = normalize_to_area(shape_builders(128)["square"], _TARGET)
    desc = describe("square", poly, _SWATH)
    path = write_shape(tmp_path, "square", poly, desc)
    assert path.exists()

    reloaded = load_area(str(path))
    assert reloaded.area == pytest.approx(_TARGET, rel=1e-6)

    # a real mission on the generated area completes and covers it (obstacles off,
    # ample fleet so the 1 km^2 square is swap-tolerant).
    cfg = load_config(
        config_path(),
        overrides={
            "env.geojson_path": str(path),
            "env.obstacle_density_per_km2": 0.0,
            "failure.hazard_rate_per_hour": 0.0,
            "fleet.n_drones": 5,
            "sim.max_timesteps": 200000,
        },
    )
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert not result.aborted
    assert result.coverage_frac > 0.99
    for a in eng.fleet.agents.values():
        assert a.state is AgentState.S0_IDLE
