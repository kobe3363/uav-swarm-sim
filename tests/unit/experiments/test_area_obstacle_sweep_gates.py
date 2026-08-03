"""Fast unit gates for the area x obstacle-count sweep driver (no simulation).

The slow byte-identity gates that run the real engine (equivalence cell,
serial<->spawn, rep-prefix) live in tests/e2e/test_area_obstacle_sweep.py.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.planning.geojson_parser import load_area
from uav_swarm_sim.experiments.generate_shapes import build_all
from uav_swarm_sim.experiments.run_shape_regime_table import spec_effective_swath
from uav_swarm_sim.experiments.run_shape_sweep import build_cell_cfg
from uav_swarm_sim.experiments import run_area_obstacle_sweep as mod
from uav_swarm_sim.experiments.run_area_obstacle_sweep import (
    ALL_VARIANT_LABELS,
    DEFAULT_VARIANTS,
    build_area_cell_cfg,
    _n_grid,
    _resolve_shapes,
    _resolve_variants,
)


@pytest.fixture(scope="module")
def base():
    return load_config("config/default.yaml")


# --------------------------------------------------------------------------- #
# Gate (i)-a: fixture parity -- HARD STOP condition (author condition 1)       #
# --------------------------------------------------------------------------- #
def test_fixture_parity_l_shape_bitwise(base):
    """The driver regenerates shapes per area; at 1 km^2 the L-shape MUST be
    bitwise-identical to the on-disk S5 fixture, or the new driver would silently
    swap the S5 anchor for a different (even if self-consistent) polygon and
    break every cross-comparison with the S5 / NOW-03 numbers. This is the hard
    STOP: a mismatch is a REJECT, not a soft fallback onto the regenerated shape."""
    eff = spec_effective_swath(base)
    regen = {name: poly for name, poly, _ in build_all(1_000_000.0, eff, 128)}
    disk = load_area("data/areas/shapes/l_shape.geojson")
    gen = regen["l_shape"]
    a = np.asarray(disk.exterior.coords)
    b = np.asarray(gen.exterior.coords)
    assert a.shape == b.shape and np.array_equal(a, b), (
        "regenerated l_shape @1 km^2 != on-disk S5 fixture -- STOP, do not "
        "re-anchor equivalence onto the self-regenerated shape")


# --------------------------------------------------------------------------- #
# Gate (i)-b: cfg equality vs run_shape_sweep shipped (pure, no simulation)     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("n_runs", [2, 5])
def test_cell_cfg_equals_shipped_build_cell_cfg(base, n, n_runs):
    """With density = the config default, no size override and no launch
    override, build_area_cell_cfg produces a Config EQUAL to the shipped
    build_cell_cfg -- the new per-cell helper perturbs nothing."""
    path = "data/areas/shapes/l_shape.geojson"
    new = build_area_cell_cfg(base, path, n, n_runs,
                              density=base.env.obstacle_density_per_km2,
                              size_range=None)
    ref = build_cell_cfg(base, path, n, "shipped", n_runs)
    assert new == ref


def test_cell_cfg_density_and_fixed_size_axes(base):
    """The two new axes land where expected: density sets the obstacle density,
    a fixed S pins the size range to [S, S], density 0 = clean."""
    path = "data/areas/shapes/l_shape.geojson"
    clean = build_area_cell_cfg(base, path, 3, 4, density=0.0)
    assert clean.env.obstacle_density_per_km2 == 0.0
    fixed = build_area_cell_cfg(base, path, 3, 4, density=12.0, size_range=(30.0, 30.0))
    assert fixed.env.obstacle_density_per_km2 == 12.0
    assert fixed.env.obstacle_size_range_m == (30.0, 30.0)
    assert clean.failure.hazard_rate_per_hour == 0.0      # scope: lambda = 0
    assert clean.telemetry.enabled is False               # perf: telemetry off
    assert clean.mc.n_min == clean.mc.n_max == 4           # fixed N, paired


# --------------------------------------------------------------------------- #
# CLI resolvers                                                               #
# --------------------------------------------------------------------------- #
def test_default_variants_are_the_four_peers():
    assert DEFAULT_VARIANTS == ("weighted_voronoi", "tgc_basic",
                                "classic_voronoi", "kmeans")


def test_resolve_variants_canonical_order_and_dedup():
    # arbitrary input order + a duplicate -> canonical order, de-duplicated
    got = _resolve_variants("kmeans,tgc_basic,tgc_basic,weighted_voronoi")
    assert got == ["weighted_voronoi", "tgc_basic", "kmeans"]


def test_resolve_variants_rejects_unknown():
    with pytest.raises(SystemExit):
        _resolve_variants("tgc_basic,not_a_variant")


def test_resolve_shapes_rejects_unknown():
    assert _resolve_shapes("l_shape,square") == ["l_shape", "square"]
    with pytest.raises(SystemExit):
        _resolve_shapes("l_shape,triangle_of_doom")


def test_n_grid_comma_and_range():
    class A:
        n = "2,4,6"
        n_range = None
    assert _n_grid(A) == [2, 4, 6]

    class B:
        n = "2,4,6"
        n_range = (2, 12, 2)          # --n-range overrides --n
    assert _n_grid(B) == [2, 4, 6, 8, 10, 12]

    class C:
        n = "2"
        n_range = (4, 2, -1)          # bad step
    with pytest.raises(SystemExit):
        _n_grid(C)


def test_all_variant_labels_cover_peers_and_naive():
    assert set(ALL_VARIANT_LABELS) == {
        "weighted_voronoi", "tgc_basic", "classic_voronoi", "kmeans",
        "tgc_naive_launch", "classic_naive_launch", "kmeans_naive_launch"}


def test_auto_jobs_reused_from_parallel():
    """--jobs auto resolves via the shared _parallel helper (>=1, bounded)."""
    j = mod._parallel.resolve_jobs("auto")
    assert isinstance(j, int) and j >= 1 and j <= (os.cpu_count() or 1)
    assert mod._parallel.resolve_jobs("1") == 1
    with pytest.raises(ValueError):
        mod._parallel.resolve_jobs("0")
