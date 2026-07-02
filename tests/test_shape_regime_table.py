"""Tests for the shape-regime table driver (experiments/run_shape_regime_table.py).

Fast subset (2 shapes, small n grid, few perms) — the physics it calls is already
covered by the A2/A3 suites; here we pin the table's own logic: the divergence
profile, the correlation helper, and the two load-bearing claims (max-zone falls
as the fleet grows; battery weighting lowers the busiest drone's work-per-battery).
"""
from __future__ import annotations

import math

import pytest
from conftest import config_path

from uav_swarm_sim.experiments.run_shape_regime_table import (
    _divergent_fracs,
    _mean,
    _pearson,
    build_tables,
)
from uav_swarm_sim.infrastructure.config import load_config


def test_divergent_fracs():
    assert _divergent_fracs(1, 0.4, 1.0) == [1.0]
    f = _divergent_fracs(4, 0.4, 1.0)
    assert f[0] == pytest.approx(0.4) and f[-1] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(f, f[1:]))          # strictly increasing
    assert f[1] - f[0] == pytest.approx(f[2] - f[1])     # evenly spaced


def test_pearson_and_mean():
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert _mean([1.0, 3.0, float("nan")]) == pytest.approx(2.0)


def test_table_core_claims():
    cfg = load_config(config_path())
    desc, cells, cap = build_tables(
        cfg, "data/areas/shapes", n_min=2, n_max=4, f_min=0.40, f_max=1.00,
        perms=3, usable_floor="terminal", sensor_power_w=0.0,
        shapes=["square", "pinwheel"])

    # descriptors carried through
    assert desc["square"]["solidity"] == pytest.approx(1.0, abs=1e-3)
    assert desc["pinwheel"]["solidity"] < 0.5

    for name in ("square", "pinwheel"):
        # max-zone ratio is a positive, decreasing function of fleet size
        ratios = [cells[(name, n)]["max_ratio"] for n in (2, 3, 4)]
        assert all(r > 0 for r in ratios)
        assert ratios[0] > ratios[-1]                    # more drones → smaller zones
        # pooled ratio is a lower bound on the per-drone max-zone
        for n in (2, 3, 4):
            assert cells[(name, n)]["pooled"] <= cells[(name, n)]["max_ratio"] + 1e-6

    # the thesis claim: battery weighting lowers the busiest drone's work-per-battery
    # ratio vs the unweighted partition at a divergence moment.
    for name in ("square", "pinwheel"):
        for n in (2, 3, 4):
            c = cells[(name, n)]
            assert c["w_ratio"] <= c["u_ratio"] + 1e-9
