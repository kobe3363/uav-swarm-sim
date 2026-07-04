"""Regression tests for the spare-sizing study.

Two layers, mirroring the H3 scale-sweep tests:

  * the PURE core (``spare_sizing``) -- the Wilson CI on the success fraction,
    the analytical-prior formula, the knee crossing (point vs Wilson-lower rules),
    the default sweep range, and the honest validate/refute read-out -- tested
    exhaustively and instantly;
  * ONE end-to-end wiring check that the paired-seed Monte-Carlo sweep
    (``run_spare_sizing.run_sweep``) drives the engine, tallies outcomes into
    SparePoints, and is reproducible on the same seeds -- run on a deliberately
    tiny smoke config so it stays fast.
"""
from __future__ import annotations

import pytest
from conftest import config_path

from uav_swarm_sim.experiments.spare_sizing import (
    KNEE_RULE_POINT,
    KNEE_RULE_WILSON,
    TARGETS,
    SparePoint,
    SpareSizingReport,
    analytical_spare_prior,
    default_spare_range,
    find_knee,
    knees_at_targets,
    min_reps_for_target,
    validate_formula,
    wilson_ci,
)


# --------------------------------------------------------------------------- #
# wilson_ci: the success-fraction CI                                          #
# --------------------------------------------------------------------------- #
def test_wilson_matches_known_value():
    lo, hi, phat = wilson_ci(50, 100)
    assert phat == 0.5
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_stays_in_unit_interval_at_extremes():
    lo0, hi0, p0 = wilson_ci(0, 20)      # all failures
    assert p0 == 0.0 and lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1, p1 = wilson_ci(20, 20)     # all successes
    assert p1 == 1.0 and hi1 == 1.0 and 0.0 < lo1 < 1.0


def test_wilson_brackets_point_estimate_and_tightens_with_n():
    lo_s, hi_s, _ = wilson_ci(9, 10)
    lo_l, hi_l, _ = wilson_ci(900, 1000)  # same phat=0.9, 100x the data
    assert lo_s < 0.9 < hi_s
    assert (hi_l - lo_l) < (hi_s - lo_s)   # more data -> narrower interval


def test_wilson_zero_n_is_vacuous():
    lo, hi, phat = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0) and phat != phat  # nan


# --------------------------------------------------------------------------- #
# min_reps_for_target: the Wilson-lower reps floor                             #
# --------------------------------------------------------------------------- #
def test_min_reps_floor_matches_wilson_perfect_batch():
    # certifying 99 %/95 % on a flawless batch needs ~381 / ~73 reps.
    assert min_reps_for_target(0.99) == 381
    assert min_reps_for_target(0.95) == 73
    # and the floor is exactly the point where a perfect k==n Wilson lower clears
    n99 = min_reps_for_target(0.99)
    assert wilson_ci(n99, n99)[0] >= 0.99
    assert wilson_ci(n99 - 1, n99 - 1)[0] < 0.99


def test_min_reps_rejects_degenerate_target():
    with pytest.raises(ValueError):
        min_reps_for_target(1.0)


# --------------------------------------------------------------------------- #
# analytical_spare_prior: spares ~= E_cover/B_usable - n + margin              #
# --------------------------------------------------------------------------- #
def test_prior_subtracts_the_first_sortie_per_active_drone():
    # 20 battery cycles, 5 drones: 5 first-sorties run on onboard packs, so the
    # shared pool must supply 20 - 5 = 15 swaps; +2 margin -> 17.
    prior = analytical_spare_prior(total_sorties_int=20, n_drones=5, margin=2)
    assert prior.base_spares == 15
    assert prior.prior_spares == 17
    assert prior.margin == 2


def test_prior_never_negative_when_fleet_outnumbers_sorties():
    # more drones than sorties: every needed sortie is a first sortie, zero swaps.
    prior = analytical_spare_prior(total_sorties_int=3, n_drones=10, margin=0)
    assert prior.base_spares == 0
    assert prior.prior_spares == 0


# --------------------------------------------------------------------------- #
# find_knee: smallest spare count clearing a target                            #
# --------------------------------------------------------------------------- #
def _pt(spares, k, n=100):
    return SparePoint(spares=spares, n_reps=n, n_success=k,
                      n_failed=n - k, n_incomplete=0)


def test_knee_point_rule_is_first_crossing():
    pts = [_pt(0, 40), _pt(2, 80), _pt(4, 96), _pt(6, 99)]
    # point estimates: 0.40, 0.80, 0.96, 0.99
    assert find_knee(pts, 0.95, KNEE_RULE_POINT) == 4   # first >= 0.95
    assert find_knee(pts, 0.99, KNEE_RULE_POINT) == 6


def test_knee_wilson_rule_is_stricter_than_point():
    # 96/100: point 0.96 clears 0.95, but Wilson lower (~0.90) does not.
    pts = [_pt(2, 96), _pt(4, 99), _pt(6, 100)]
    assert find_knee(pts, 0.95, KNEE_RULE_POINT) == 2
    assert find_knee(pts, 0.95, KNEE_RULE_WILSON) > 2   # needs more spares to be sure


def test_knee_none_when_target_never_reached():
    pts = [_pt(0, 10), _pt(2, 20), _pt(4, 30)]
    assert find_knee(pts, 0.95, KNEE_RULE_POINT) is None


def test_knee_reads_in_ascending_spare_order_regardless_of_input_order():
    pts = [_pt(6, 99), _pt(0, 40), _pt(4, 96), _pt(2, 80)]
    assert find_knee(pts, 0.95, KNEE_RULE_POINT) == 4


def test_knees_at_targets_covers_both_thesis_targets():
    pts = [_pt(0, 40), _pt(2, 80), _pt(4, 96), _pt(6, 100)]
    knees = knees_at_targets(pts, TARGETS)
    assert {k.target for k in knees} == set(TARGETS)


# --------------------------------------------------------------------------- #
# default_spare_range: brackets the prior                                     #
# --------------------------------------------------------------------------- #
def test_default_range_brackets_prior_and_clips_at_zero():
    prior = analytical_spare_prior(total_sorties_int=12, n_drones=4, margin=0)  # base 8
    rng = default_spare_range(prior, span=3)
    assert rng == [5, 6, 7, 8, 9, 10, 11]
    # a small base clips the lower end at zero rather than going negative
    small = analytical_spare_prior(total_sorties_int=6, n_drones=4, margin=0)   # base 2
    assert default_spare_range(small, span=5)[0] == 0


# --------------------------------------------------------------------------- #
# validate_formula / report: honest read-out either way                       #
# --------------------------------------------------------------------------- #
def test_formula_validated_when_prior_lands_on_knee():
    prior = analytical_spare_prior(total_sorties_int=20, n_drones=5, margin=0)  # base/prior 15
    pts = [_pt(13, 90), _pt(14, 97), _pt(15, 100), _pt(16, 100)]
    report = SpareSizingReport.build(pts, prior)
    assert report.verdict.verdict == "validated"
    assert report.verdict.empirical_knee == 15
    assert report.verdict.delta == 0


def test_formula_refuted_when_knee_is_far_from_prior():
    prior = analytical_spare_prior(total_sorties_int=20, n_drones=5, margin=0)  # prior 15
    # knee only at 19 -> off by 4 packs
    pts = [_pt(15, 50), _pt(17, 70), _pt(19, 100), _pt(21, 100)]
    v = validate_formula(prior, knees_at_targets(pts, [0.99])[0])
    assert v.verdict == "refuted"
    assert v.delta == 4
    assert v.measured_margin == 4   # data demanded 4 packs over the zero-margin base


def test_formula_inconclusive_when_range_misses_the_knee():
    prior = analytical_spare_prior(total_sorties_int=20, n_drones=5, margin=0)
    pts = [_pt(10, 30), _pt(12, 45), _pt(14, 60)]  # never reaches 0.99
    report = SpareSizingReport.build(pts, prior)
    assert report.verdict.verdict == "inconclusive"
    assert report.verdict.empirical_knee is None


# --------------------------------------------------------------------------- #
# end-to-end wiring: paired-seed sweep drives the engine (tiny + fast)        #
# --------------------------------------------------------------------------- #
def _tiny_cfg():
    from uav_swarm_sim.infrastructure.config import load_config
    return load_config(
        config_path(),
        overrides={
            "fleet.n_drones": 3,
            "fleet.battery_capacity_wh": 400.0,
            "failure.hazard_rate_per_hour": 0.0,
            "env.geojson_path": "data/areas/smoke_area.geojson",
            "env.obstacle_density_per_km2": 4.0,
            "env.obstacle_size_range_m": [10.0, 30.0],
            "sim.dt_s": 1.0,
            "sim.max_timesteps": 20000,
        },
    )


def test_run_sweep_tallies_outcomes_and_is_paired_reproducible():
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import run_sweep

    cfg = _tiny_cfg()
    reps = 2
    counts = [0, 50]

    pts_a = run_sweep(cfg, counts, reps, RngFactory(cfg.sim.master_seed))
    # every replication lands in exactly one terminal bucket
    for pt in pts_a:
        assert pt.n_success + pt.n_failed + pt.n_incomplete == reps
        assert 0.0 <= pt.success_frac <= 1.0

    # paired & deterministic: a fresh factory on the same master_seed reproduces
    # the identical tallies (same env & failure draws per replication index)
    pts_b = run_sweep(cfg, counts, reps, RngFactory(cfg.sim.master_seed))
    for a, b in zip(pts_a, pts_b):
        assert (a.spares, a.n_success, a.n_failed, a.n_incomplete) == \
               (b.spares, b.n_success, b.n_failed, b.n_incomplete)


def test_report_build_produces_knees_and_verdict_from_sweep():
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import run_sweep

    cfg = _tiny_cfg()
    points = run_sweep(cfg, [0, 50], 2, RngFactory(cfg.sim.master_seed))
    prior = analytical_spare_prior(total_sorties_int=1, n_drones=3, margin=0)
    report = SpareSizingReport.build(points, prior)
    assert {k.target for k in report.knees} == set(TARGETS)
    assert report.verdict is not None  # a read-out is always produced
