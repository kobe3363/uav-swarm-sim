"""End-to-end + byte-identity gates for the area x obstacle-count sweep.

Runs the real engine (slow). Three determinism gates plus a smoke:

  (i)  Equivalence cell: at L-shape / 1 km^2 / config-default density & size the
       new driver reproduces run_shape_sweep's SHIPPED l_shape cell bitwise --
       proof it perturbs no physics. (The fixture-parity + cfg-equality layers
       are fast and live in tests/unit/experiments/test_area_obstacle_sweep.py.)
  (ii) Serial (--jobs 1) vs spawn (--jobs 2) produce byte-identical CSVs.
  (iii)A reps=2 prefix of a reps=4 cell is byte-identical (rep incrementality).
  smoke: a tiny (shape x area x density x n) grid produces well-formed rows,
       tagged regimes, and paired contrasts with CIs.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.metrics.comparison import VariantResult
from uav_swarm_sim.metrics.run_output import RunContext
from uav_swarm_sim.planning.geojson_parser import load_area
from uav_swarm_sim.experiments.generate_shapes import describe
from uav_swarm_sim.experiments.run_shape_regime_table import spec_effective_swath
from uav_swarm_sim.experiments.run_shape_sweep import (
    METRICS, metric_vectors, run_cell, write_csv)
from uav_swarm_sim.experiments.run_area_obstacle_sweep import (
    _PEER_LABELS, regenerate_shapes, run_area_cell, sweep)

pytestmark = pytest.mark.slow


def _paths_and_descs(base, root, areas, shapes):
    """Mirror main()'s setup: regenerate the family per area, describe once."""
    shape_paths = regenerate_shapes(base, root, areas, shapes, 128)
    eff = spec_effective_swath(base)
    descs = {s: describe(s, load_area(shape_paths[(s, areas[0])]), eff)
             for s in shapes}
    return shape_paths, descs


# --------------------------------------------------------------------------- #
# (i) equivalence cell: new driver == run_shape_sweep SHIPPED l_shape          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2])
def test_equivalence_cell_matches_shipped_shape_sweep(n):
    """The new driver's l_shape cell at 1 km^2, config-default density/size
    reproduces run_shape_sweep.run_cell(..., 'shipped', ...) byte-for-byte on
    every physics field, for every shared (peer) variant. The reference reads
    the ON-DISK S5 shapes dir (the real anchor); fixture parity (unit test)
    guarantees the regenerated 1 km^2 l_shape is the identical polygon."""
    base = load_config("config/default.yaml")
    n_runs = 2
    ref = run_cell(base, "data/areas/shapes", "l_shape", n, "shipped", n_runs)
    new = run_area_cell(base, "data/areas/shapes/l_shape.geojson", n, n_runs,
                        density=base.env.obstacle_density_per_km2,
                        size_range=None, variant_labels=list(_PEER_LABELS))
    for lbl in _PEER_LABELS:
        assert isinstance(ref[lbl], VariantResult) and isinstance(new[lbl], VariantResult)
        assert metric_vectors(ref[lbl]) == metric_vectors(new[lbl]), lbl


# --------------------------------------------------------------------------- #
# (ii) serial vs spawn byte-identical                                          #
# --------------------------------------------------------------------------- #
def test_serial_parallel_bitwise_identical(tmp_path):
    """SACRED determinism gate: --jobs 1 vs --jobs 2 must produce identical
    area_obstacle_sweep.csv and contrasts.csv. 1<->2 straddles the serial->spawn
    boundary; the grid spans two areas x two densities so completion order
    differs from ordinal order (proving the k-index reassembly is jobs-invariant)."""
    base = load_config("config/default.yaml")
    shapes, areas, densities, ns, reps = ["square"], [1.0], [0.0, 8.0], [2], 2
    variants = list(_PEER_LABELS)
    shape_paths, descs = _paths_and_descs(base, tmp_path / "shapes", areas, shapes)

    args = (base, shape_paths, shapes, areas, densities, None, ns, reps,
            variants, descs)
    cells_s, contr_s, prob_s = sweep(*args, jobs=1, quiet=True)
    cells_p, contr_p, prob_p = sweep(*args, jobs=2, quiet=True)

    assert prob_s == prob_p == []
    assert len(cells_s) == len(cells_p) and len(contr_s) == len(contr_p)
    for name, rs, rp in (("area_obstacle_sweep.csv", cells_s, cells_p),
                         ("contrasts.csv", contr_s, contr_p)):
        ps, pp = tmp_path / f"s_{name}", tmp_path / f"p_{name}"
        write_csv(ps, rs)
        write_csv(pp, rp)
        assert ps.read_bytes() == pp.read_bytes(), f"{name} drifted serial vs jobs=2"


# --------------------------------------------------------------------------- #
# (iii) rep incrementality: reps=2 prefix of reps=4                            #
# --------------------------------------------------------------------------- #
def test_rep_prefix_byte_identical():
    """The first-2 per-replication samples of a reps=4 cell equal the reps=2
    cell's samples (same shape/area/density/n/seed): N is incremental and paired
    seeds are stable across the reps axis. One variant (tgc_basic) suffices."""
    base = load_config("config/default.yaml")
    path = "data/areas/shapes/square.geojson"
    kw = dict(density=0.0, size_range=None, variant_labels=["tgc_basic"])
    two = run_area_cell(base, path, 2, 2, **kw)["tgc_basic"]
    four = run_area_cell(base, path, 2, 4, **kw)["tgc_basic"]
    v2, v4 = metric_vectors(two), metric_vectors(four)
    for m in METRICS:
        assert v4[m][:2] == v2[m], m


# --------------------------------------------------------------------------- #
# smoke: tiny 4-axis grid                                                      #
# --------------------------------------------------------------------------- #
def test_smoke_grid_wellformed(tmp_path):
    base = load_config("config/default.yaml")
    shapes, areas, densities, ns, reps = ["l_shape"], [1.0, 2.0], [0.0], [2], 2
    variants = list(_PEER_LABELS)
    shape_paths, descs = _paths_and_descs(base, tmp_path / "shapes", areas, shapes)
    cells, contrasts, problems = sweep(
        base, shape_paths, shapes, areas, densities, None, ns, reps, variants,
        descs, jobs=1, quiet=True)

    assert problems == []
    # one row per (area x variant); regime tagged; area/density columns present
    assert len(cells) == len(areas) * len(variants)
    valid = {"BATTERY-LIMITED", "BORDERLINE", "FUEL-SURPLUS"}
    for r in cells:
        assert r["regime"] in valid
        assert r["area_km2"] in areas and r["density_per_km2"] == 0.0
        for m in METRICS:
            assert f"{m}_mean" in r and f"{m}_ci" in r and f"{m}_n" in r
    # paired contrasts carry a CI + n_pairs; scoped null exact where both ran
    assert contrasts
    for c in contrasts:
        assert "diff_mean" in c and "diff_ci" in c and "n_pairs" in c
        assert c["n_pairs"] <= reps
    null = [c for c in contrasts if c["contrast"] == "weighted_voronoi - tgc_basic"]
    assert null and all(c["exact_zero"] for c in null)


def test_run_context_finalize_shallow(tmp_path):
    """RunContext is used shallowly (dir + finalize), no deep ENG-13 schema."""
    ctx = RunContext(base_dir=str(tmp_path), name="ao_smoke")
    assert ctx.dir.exists()
    p = ctx.finalize(summary={"experiment": "area_obstacle_sweep"})
    assert p.exists() and p.name == "run.json"
