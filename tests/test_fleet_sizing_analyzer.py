"""Regression tests for the B6.2 standalone fleet-sizing analyzer CLI.

Part 2.3 (commit 8bed800) removed ``TURN_FACTOR_DEFAULT`` and dropped the
``turn_factor`` parameter from ``fleet_sizing.sweep`` -- charging U-turn overhead
as real TURN-rate energy instead -- but left ``run_fleet_sizing_analyzer`` still
importing the constant, exposing a ``--turn-factor`` flag, and passing
``turn_factor=`` into ``sweep``. The module therefore raised ImportError on
import (and would have raised TypeError at the ``sweep`` call even past that).

No test drove this CLI, so the breakage slipped through green suites. These
tests close that gap: they exercise the analyzer END-TO-END on a tiny smoke
config and assert the result is not merely exception-free but MEANINGFUL -- a
non-empty Pareto table with one row per swept fleet size and finite,
non-negative sortie/duration values -- so a future "runs, but empty" regression
is caught too.
"""
from __future__ import annotations

import math

import pytest

from uav_swarm_sim.experiments.run_fleet_sizing_analyzer import (
    _build_planning_layer,
    _inputs_from,
    main,
)
from uav_swarm_sim.experiments.fleet_sizing import sweep
from uav_swarm_sim.infrastructure.config import load_config


def _smoke_config_path(config_path) -> str:
    # the small, fast scenario area (sibling of default.yaml under config/)
    return str(config_path.parent / "scenarios" / "smoke.yaml")


def test_analyzer_main_runs_end_to_end(config_path, capsys):
    """main() imports cleanly AND completes (exit 0): guards both the removed
    import and the dropped sweep() kwarg -- either would crash before returning."""
    rc = main(["--config", _smoke_config_path(config_path), "--n-min", "1", "--n-max", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Fleet-Sizing Analysis" in out
    assert "turn factor" not in out  # the dead turn-factor plumbing is gone


def test_analyzer_sweep_result_is_meaningful(config_path):
    """Beyond 'no exception': the sweep must return a non-empty table with one
    row per swept fleet size and finite, non-negative sortie/duration values --
    otherwise a 'works, but empty' regression would pass a smoke check."""
    n_min, n_max = 1, 5
    cfg = load_config(_smoke_config_path(config_path))
    env, spec, em, base_pose = _build_planning_layer(cfg)
    inputs = _inputs_from(env, base_pose, cfg.env.coverage_altitude_m)
    report = sweep(
        inputs, em, spec,
        effective_swath_m=spec.swath_width_m,
        service_time_s=cfg.swap.service_time_s,
        n_bays=cfg.swap.n_bays,
        n_min=n_min, n_max=n_max,
        reserve_frac=cfg.rth.reserve_frac,
    )

    # exactly one row per swept fleet size, in ascending order
    assert len(report.rows) == n_max - n_min + 1
    assert [r.n for r in report.rows] == list(range(n_min, n_max + 1))

    # sortie demand is a finite, non-negative real that rounds up to >= 1 cycle
    assert math.isfinite(report.total_sorties)
    assert report.total_sorties >= 0.0
    assert report.total_sorties_int >= 1

    # coverage energy and every per-N duration are finite and strictly positive
    assert math.isfinite(report.total_coverage_j) and report.total_coverage_j > 0.0
    for r in report.rows:
        assert math.isfinite(r.est_duration_s) and r.est_duration_s > 0.0
        assert r.est_total_swaps >= 0


def test_sweep_has_no_turn_factor_parameter():
    """Pin the post-2.3 API: reintroducing a turn_factor kwarg (the shape of the
    original bug) must fail loudly here rather than silently in the CLI."""
    import inspect
    assert "turn_factor" not in inspect.signature(sweep).parameters
