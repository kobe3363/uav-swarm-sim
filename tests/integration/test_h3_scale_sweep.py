"""Regression tests for H3: fine-grained scale sweep break-even detection.

The adaptive CI-based Monte-Carlo stopping rule already existed (convergence.py +
monte_carlo.run); H3 made the tier sweep *usable*: it now shows both methods per
fleet size and locates the empirical break-even where the battery-weighted TGC
overtakes the position k-means baseline. These tests pin down the pure crossover
estimator (the thesis-deliverable number) and check the sweep harness is wired to
return both methods per fleet size.

`tier_crossover` is a pure function: exhaustive, instant. The single end-to-end
wiring check runs a deliberately tiny Monte-Carlo (n_min=n_max=2) over a tiny
area so it stays fast.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.metrics.comparison import compare_tiers, tier_crossover


# --------------------------------------------------------------------------- #
# tier_crossover: the empirical break-even estimator (lower-is-better metric)  #
# --------------------------------------------------------------------------- #
def test_crossover_interpolates_the_zero_crossing():
    # a: 10 -> 5, b: 6 -> 6, over n: 2 -> 4. diff = [4, -1]; zero crossing at
    # frac = 4/(4-(-1)) = 0.8 -> n = 2 + 0.8*(4-2) = 3.6
    assert tier_crossover([2, 4], [10.0, 5.0], [6.0, 6.0]) == pytest.approx(3.6)


def test_crossover_returns_first_sign_flip_only():
    ns = [2, 4, 6, 8]
    a = [10.0, 4.0, 8.0, 2.0]   # crosses below b between 2 and 4, back above, below again
    b = [6.0, 6.0, 6.0, 6.0]
    # first overtake is between n=2 (diff +4) and n=4 (diff -2): frac=4/6 -> n=2+2/3*2
    assert tier_crossover(ns, a, b) == pytest.approx(2 + (4.0 / 6.0) * 2)


def test_crossover_exactly_at_a_grid_point():
    # a == b at n0 then strictly below at n1 -> crossing sits exactly on n0
    assert tier_crossover([10, 20], [6.0, 5.0], [6.0, 6.0]) == pytest.approx(10.0)


def test_no_crossing_when_a_is_always_better():
    # a strictly below b throughout -> never "overtakes" (no >=-to-< event)
    assert tier_crossover([2, 4, 6], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]) is None


def test_no_crossing_when_a_is_always_worse():
    assert tier_crossover([2, 4, 6], [9.0, 9.0, 9.0], [1.0, 1.0, 1.0]) is None


def test_crossover_guards_degenerate_input():
    assert tier_crossover([5], [1.0], [2.0]) is None          # < 2 points
    assert tier_crossover([2, 4], [1.0], [2.0, 3.0]) is None  # length mismatch


# --------------------------------------------------------------------------- #
# compare_tiers wiring: both methods per fleet size, correctly labelled        #
# --------------------------------------------------------------------------- #
def _tiny_sweep_cfg(config_path):
    return load_config(
        config_path,
        overrides={
            "fleet.n_drones": 2,
            "fleet.battery_capacity_wh": 400.0,
            "failure.hazard_rate_per_hour": 0.0,
            "env.geojson_path": "data/areas/smoke_area.geojson",
            "env.obstacle_density_per_km2": 4.0,
            "env.obstacle_size_range_m": [10.0, 30.0],
            "sim.dt_s": 1.0,
            "sim.max_timesteps": 20000,
            "mc.n_min": 2,
            "mc.n_max": 2,
        },
    )


def test_compare_tiers_runs_both_methods_per_fleet_size(config_path):
    cfg = _tiny_sweep_cfg(config_path)
    res = compare_tiers(cfg, [2, 3], RngFactory(cfg.sim.master_seed))
    assert set(res.keys()) == {2, 3}
    for n, variants in res.items():
        labels = {v.label for v in variants}
        assert labels == {f"n={n} weighted", f"n={n} kmeans"}
        # paired design: both methods used the same replication count
        assert all(v.mc.n_runs == 2 for v in variants)


# --------------------------------------------------------------------------- #
# CLI knobs: --budget grid, --n/--n-range override, --mode obstacle density    #
# --------------------------------------------------------------------------- #
def test_n_grid_budget_and_overrides():
    import argparse
    from uav_swarm_sim.experiments.run_scale_tiers import _n_grid

    def ns(n=None, n_range=None, budget="quick"):
        return _n_grid(argparse.Namespace(n=n, n_range=n_range, budget=budget))

    assert ns(budget="quick") == [4, 8, 16, 24]
    assert ns(budget="full") == list(range(2, 101, 2))
    assert ns(n=[6, 2, 6], budget="full") == [2, 6]        # --n overrides, sorted+dedup
    assert ns(n_range=[2, 8, 2], budget="full") == [2, 4, 6, 8]  # --n-range overrides


def test_apply_mode_obstacle_density(config_path):
    from uav_swarm_sim.experiments.run_scale_tiers import _apply_mode
    cfg = load_config(config_path)
    assert _apply_mode(cfg, "clean").env.obstacle_density_per_km2 == 0.0
    shipped = _apply_mode(cfg, "shipped")
    assert shipped.env.obstacle_density_per_km2 == cfg.env.obstacle_density_per_km2


@pytest.mark.slow
def test_sweep_tiers_serial_parallel_deterministic(config_path, tmp_path):
    """Paired-seed determinism gate: RngFactory.stream is stateless, so each
    fleet-size tier is a pure function of (master_seed, n) and the per-tier
    parallel workers must reproduce the serial result exactly.

    Every DETERMINISTIC field is compared. planning_time_s is deliberately
    excluded: it is a wall-clock timing that varies by nature (it differs even
    between two serial runs) and is not a simulation result."""
    import csv as _csv
    from uav_swarm_sim.experiments.run_scale_tiers import sweep_tiers, _write_csv
    cfg = _tiny_sweep_cfg(config_path)
    ns = [2, 3]
    res_s, prob_s = sweep_tiers(cfg, ns, jobs=1, quiet=True)
    res_p, prob_p = sweep_tiers(cfg, ns, jobs=2, quiet=True)
    assert prob_s == prob_p == []
    p_s, p_p = tmp_path / "serial.csv", tmp_path / "parallel.csv"
    _write_csv(str(p_s), ns, res_s)
    _write_csv(str(p_p), ns, res_p)

    def rows_sans_timing(path):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        for r in rows:
            r.pop("planning_time_s", None)  # wall-clock timing, non-deterministic
        return rows

    serial, parallel = rows_sans_timing(p_s), rows_sans_timing(p_p)
    assert serial == parallel, "serial vs parallel scale-sweep metrics drifted"


def test_main_writes_unique_run_dirs_and_never_overwrites(config_path, tmp_path,
                                                          monkeypatch):
    """Repeat runs must land in DISTINCT, experiment-tagged folders (no silent
    overwrite). Stubs the heavy sweep so the CLI/output plumbing is exercised
    without running any mission."""
    import uav_swarm_sim.experiments.run_scale_tiers as rst

    monkeypatch.setattr(rst, "sweep_tiers", lambda *a, **k: ({}, []))
    argv = ["--config", str(config_path), "--n", "2", "--out", str(tmp_path)]
    assert rst.main(argv) == 0
    assert rst.main(argv) == 0

    run_dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(run_dirs) == 2, f"expected two distinct run dirs, got {run_dirs}"
    for d in run_dirs:
        assert d.name.startswith("scale_tiers_")       # names the experiment
        assert (d / "scale_sweep.csv").exists()         # artifact present
        assert (d / "run.json").exists()                # RunContext manifest


@pytest.mark.slow
def test_sweep_tiers_reports_bad_tier_without_crashing(config_path, monkeypatch):
    """RESILIENCE: a tier whose mission raises must be recorded in ``problems``
    and skipped -- the other tiers still complete (a 22h run must not die on one
    bad fleet size). Mirrors run_shape_sweep's per-cell 'report, never skip'."""
    import uav_swarm_sim.experiments.run_scale_tiers as rst
    real = rst._process_tier

    def flaky(cfg, n):
        if n == 3:
            raise RuntimeError("boom at n=3")
        return real(cfg, n)

    monkeypatch.setattr(rst, "_process_tier", flaky)
    cfg = _tiny_sweep_cfg(config_path)
    res, problems = rst.sweep_tiers(cfg, [2, 3, 4], jobs=1, quiet=True)
    assert set(res.keys()) == {2, 4}                 # good tiers survived
    assert [p["n"] for p in problems] == [3]
    assert "boom at n=3" in problems[0]["error"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
