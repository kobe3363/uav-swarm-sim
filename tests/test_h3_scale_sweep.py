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
from conftest import config_path

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
def _tiny_sweep_cfg():
    return load_config(
        config_path(),
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


def test_compare_tiers_runs_both_methods_per_fleet_size():
    cfg = _tiny_sweep_cfg()
    res = compare_tiers(cfg, [2, 3], RngFactory(cfg.sim.master_seed))
    assert set(res.keys()) == {2, 3}
    for n, variants in res.items():
        labels = {v.label for v in variants}
        assert labels == {f"n={n} weighted", f"n={n} kmeans"}
        # paired design: both methods used the same replication count
        assert all(v.mc.n_runs == 2 for v in variants)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
