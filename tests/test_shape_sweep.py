"""End-to-end guard for the S5 shape-sweep harness (run_shape_sweep).

Runs a minimal grid (2 shapes x n in {2,4} x variants, fixed N=2) and asserts
the harness's load-bearing invariants:

  * AC-2 paired seeds: every variant in a cell runs the SAME fixed N.
  * AC-3 per-cell metrics: mean + CI present for all metrics.
  * AC-4 paired contrasts: paired differences carry a CI and an n_pairs.
  * AC-5 regime overlay: each cell tagged, n=4 flagged as the reference row.
  * Scoped null: weighted_voronoi - tgc_basic is exactly zero everywhere.
  * naive-centroid launch site is outside the survey area and feasible.
  * honest read-out returns finite structure.

Kept tiny (N=2, 2 shapes, 2 fleet sizes) so it stays in the regression suite.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.planning.geojson_parser import load_area
from uav_swarm_sim.metrics.run_output import RunContext
from uav_swarm_sim.experiments.run_shape_sweep import (
    METRICS,
    NAIVE_LAUNCH_LABEL,
    REFERENCE_N,
    hypothesis_readout,
    naive_centroid_site,
    sweep,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    base = load_config("config/default.yaml")
    ctx = RunContext(base_dir=str(tmp_path_factory.mktemp("s5")), name="e2e")
    cells, contrasts, problems = sweep(
        base, "data/areas/shapes", ["square", "c_shape"], [2, 4],
        "clean", 2, ctx, quiet=True)
    return cells, contrasts, problems


def test_no_problem_cells(swept):
    _, _, problems = swept
    assert problems == [], f"cells crashed/skipped: {problems}"


def test_paired_seed_equal_run_counts(swept):
    """AC-2: within each (shape, n) every variant used the same fixed N."""
    cells, _, _ = swept
    by_cell: dict[tuple[str, int], set[int]] = {}
    for r in cells:
        by_cell.setdefault((r["shape"], r["n"]), set()).add(r["n_runs"])
    for key, counts in by_cell.items():
        assert counts == {2}, f"{key}: unequal/other run counts {counts}"


def test_five_variants_per_cell(swept):
    cells, _, _ = swept
    by_cell: dict[tuple[str, int], set[str]] = {}
    for r in cells:
        by_cell.setdefault((r["shape"], r["n"]), set()).add(r["variant"])
    for key, variants in by_cell.items():
        assert NAIVE_LAUNCH_LABEL in variants
        assert len(variants) == 5, f"{key}: {variants}"


def test_per_cell_metrics_have_mean_and_ci(swept):
    """AC-3: every metric reports a mean and a CI field per cell."""
    cells, _, _ = swept
    for r in cells:
        for m in METRICS:
            assert f"{m}_mean" in r and f"{m}_ci" in r and f"{m}_n" in r


def test_regime_overlay_and_reference_flag(swept):
    """AC-5: each cell tagged; n=4 flagged as reference, n=2 not."""
    cells, _, _ = swept
    valid = {"BATTERY-LIMITED", "BORDERLINE", "FUEL-SURPLUS"}
    for r in cells:
        assert r["regime"] in valid
        assert r["reference_cell"] == (r["n"] == REFERENCE_N)


def test_paired_contrasts_have_ci_and_counts(swept):
    """AC-4: paired differences carry a diff mean, a CI, and n_pairs."""
    _, contrasts, _ = swept
    assert contrasts
    for c in contrasts:
        assert "diff_mean" in c and "diff_ci" in c and "n_pairs" in c
        assert c["n_pairs"] <= 2


def test_scoped_null_weighted_equals_tgc(swept):
    """Documented finding: weighted_voronoi - tgc_basic == 0 in every cell and
    metric for an identical full-battery fleet."""
    _, contrasts, _ = swept
    null = [c for c in contrasts
            if c["contrast"] == "weighted_voronoi - tgc_basic"]
    assert null, "the scoped-null contrast must be computed and reported"
    for c in null:
        assert c["exact_zero"] is True, (c["metric"], c["diff_mean"])
        assert c["diff_mean"] == 0.0


def test_naive_launch_site_outside_area():
    for shape in ("square", "c_shape", "pinwheel", "rect_8_1"):
        area = load_area(f"data/areas/shapes/{shape}.geojson")
        x, y = naive_centroid_site(area)
        assert not area.contains(Point(x, y)), shape


def test_readout_structure(swept):
    _, contrasts, _ = swept
    ro = hypothesis_readout(contrasts)
    assert ro["null_all_exact"] is True
    assert isinstance(ro["tgc_adv_vs_best_baseline_by_n"], dict)
    # correlations are finite or explicitly nan (few shapes) -- never a crash
    for k in ("h2_corr_solidity", "h2_corr_isoperimetric"):
        assert isinstance(ro[k], float)
