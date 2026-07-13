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

from uav_swarm_sim.experiments.spare_sizing import (
    KNEE_RULE_POINT,
    KNEE_RULE_WILSON,
    TARGETS,
    DemandRecord,
    SparePoint,
    SpareSizingReport,
    analytical_spare_prior,
    default_spare_range,
    demand_cdf,
    demand_knees,
    demand_success_count,
    find_knee,
    knees_at_targets,
    max_finite_demand,
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
def _tiny_cfg(config_path):
    from uav_swarm_sim.infrastructure.config import load_config
    return load_config(
        config_path,
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


@pytest.mark.slow
def test_run_sweep_tallies_outcomes_and_is_paired_reproducible(config_path):
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import run_sweep

    cfg = _tiny_cfg(config_path)
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


@pytest.mark.slow
def test_report_build_produces_knees_and_verdict_from_sweep(config_path):
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import run_sweep

    cfg = _tiny_cfg(config_path)
    points = run_sweep(cfg, [0, 50], 2, RngFactory(cfg.sim.master_seed))
    prior = analytical_spare_prior(total_sorties_int=1, n_drones=3, margin=0)
    report = SpareSizingReport.build(points, prior)
    assert {k.target for k in report.knees} == set(TARGETS)
    assert report.verdict is not None  # a read-out is always produced


# --------------------------------------------------------------------------- #
# crash-safe partial log + --resume (ENG: incremental writing)                 #
# --------------------------------------------------------------------------- #
def _write_partial_line(path, identity, spares, n_reps, n_success):
    """Handcraft one results_partial.jsonl record (the pure-parsing tests)."""
    import json
    rec = {
        "schema": "uav-swarm-sim/spare-sizing-partial/v1",
        **identity,
        "spares": spares, "n_reps": n_reps, "n_success": n_success,
        "n_failed": n_reps - n_success, "n_incomplete": 0,
        "success_frac": n_success / n_reps, "wilson95_lo": 0.0, "wilson95_hi": 1.0,
        "written_at": "2026-07-13T00:00:00+00:00",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def test_resume_rejected_on_identity_mismatch(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import _validated_resume

    p = tmp_path / "results_partial.jsonl"
    _write_partial_line(p, {"master_seed": 111, "config_hash": "abc",
                            "reps_per_point": 2}, spares=0, n_reps=2, n_success=2)
    # a different master_seed breaks the paired-seed design -> refuse loudly
    expected = {"master_seed": 999, "config_hash": "abc", "reps_per_point": 2}
    with pytest.raises(SystemExit, match="identity mismatch.*master_seed"):
        _validated_resume(p, expected, [0, 1, 2])
    # same seed but a different config (hash) is just as invalid
    expected = {"master_seed": 111, "config_hash": "OTHER", "reps_per_point": 2}
    with pytest.raises(SystemExit, match="identity mismatch.*config_hash"):
        _validated_resume(p, expected, [0, 1, 2])
    # and so is a different replication count
    expected = {"master_seed": 111, "config_hash": "abc", "reps_per_point": 5}
    with pytest.raises(SystemExit, match="identity mismatch.*reps_per_point"):
        _validated_resume(p, expected, [0, 1, 2])


def test_resume_missing_file_and_empty_log_are_rejected(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import _validated_resume

    ident = {"master_seed": 1, "config_hash": "x", "reps_per_point": 2}
    with pytest.raises(SystemExit, match="no such file"):
        _validated_resume(tmp_path / "nope.jsonl", ident, [0])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no completed points"):
        _validated_resume(empty, ident, [0])


def test_load_partial_tolerates_truncated_final_line(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import load_partial_points

    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 2}
    p = tmp_path / "results_partial.jsonl"
    _write_partial_line(p, ident, spares=0, n_reps=2, n_success=1)
    _write_partial_line(p, ident, spares=1, n_reps=2, n_success=2)
    # a crash mid-append leaves a half-written final line: earlier points survive
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"schema": "uav-swarm-sim/spare-sizing-partial/v1", "spares": 2')
    identity, points = load_partial_points(p)
    assert identity == ident
    assert sorted(points) == [0, 1]
    assert points[1].n_success == 2


def test_load_partial_rejects_unknown_schema_and_malformed_record(tmp_path):
    import json
    from uav_swarm_sim.experiments.run_spare_sizing import load_partial_points

    # a record from some FUTURE/foreign schema (identity fields happen to exist)
    # must be refused up front, not silently accepted as resume state
    wrong = tmp_path / "wrong_schema.jsonl"
    wrong.write_text(json.dumps({
        "schema": "uav-swarm-sim/spare-sizing-partial/v99",
        "master_seed": 7, "config_hash": "h", "reps_per_point": 2,
        "spares": 0, "n_reps": 2, "n_success": 2, "n_failed": 0, "n_incomplete": 0,
    }) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unsupported schema"):
        load_partial_points(wrong)

    # structurally-valid JSON missing a required count field: a clear SystemExit,
    # not a raw KeyError
    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 2}
    broken = tmp_path / "missing_key.jsonl"
    broken.write_text(json.dumps({
        "schema": "uav-swarm-sim/spare-sizing-partial/v1", **ident,
        "spares": 0, "n_reps": 2, "n_failed": 0, "n_incomplete": 0,  # no n_success
    }) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="malformed record"):
        load_partial_points(broken)


def test_resume_ignores_counts_outside_current_grid(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import _validated_resume

    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 2}
    p = tmp_path / "results_partial.jsonl"
    _write_partial_line(p, ident, spares=0, n_reps=2, n_success=1)
    _write_partial_line(p, ident, spares=99, n_reps=2, n_success=2)  # not in grid
    done = _validated_resume(p, ident, [0, 1, 2])
    assert sorted(done) == [0]  # 99 dropped, or the resumed results.json would
    # differ from an uninterrupted run of the same CLI arguments


@pytest.mark.slow
def test_partial_jsonl_appended_and_flushed_after_each_point(tmp_path, config_path):
    import json
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import (
        _partial_identity, sweep_with_partials)

    cfg = _tiny_cfg(config_path)
    reps, counts = 2, [0, 25, 50]
    partial = tmp_path / "results_partial.jsonl"

    seen: list[int] = []

    def _spy(pt):
        # the point's line must already be on disk when progress fires
        lines = partial.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(seen) + 1
        rec = json.loads(lines[-1])
        assert rec["spares"] == pt.spares
        assert rec["n_success"] == pt.n_success
        assert "written_at" in rec
        seen.append(pt.spares)

    pts = sweep_with_partials(cfg, counts, reps, RngFactory(cfg.sim.master_seed),
                              partial, progress=_spy)
    assert seen == counts  # one incremental append per swept point, in order
    # every line carries the run identity the resume safeguard checks
    ident = _partial_identity(cfg, reps)
    for line in partial.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert {k: rec[k] for k in ident} == ident
    assert [p.spares for p in pts] == counts


@pytest.mark.slow
def test_interrupted_plus_resumed_results_match_uninterrupted(tmp_path, config_path):
    import json
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import (
        _results_dict, sweep_with_partials)

    cfg = _tiny_cfg(config_path)
    reps, counts = 2, [0, 25, 50]
    seed = cfg.sim.master_seed

    # the uninterrupted reference run
    full = sweep_with_partials(cfg, counts, reps, RngFactory(seed),
                               tmp_path / "full.jsonl")
    # a run 'crashed' after two of the three points...
    sweep_with_partials(cfg, counts[:2], reps, RngFactory(seed),
                        tmp_path / "crashed.jsonl")
    # ...and resumed: only the remaining count is actually simulated
    resumed = sweep_with_partials(cfg, counts, reps, RngFactory(seed),
                                  tmp_path / "resumed.jsonl",
                                  resume_path=tmp_path / "crashed.jsonl")

    # results.json content (identity held constant; it carries the only
    # timestamp/uuid fields) is byte-identical full vs interrupted+resumed
    ident = {"pinned": "identity"}
    prior = analytical_spare_prior(total_sorties_int=1, n_drones=3, margin=0)
    d_full = _results_dict(SpareSizingReport.build(full, prior), reps, ident)
    d_res = _results_dict(SpareSizingReport.build(resumed, prior), reps, ident)
    assert json.dumps(d_full, sort_keys=True) == json.dumps(d_res, sort_keys=True)

    # the resumed run's own partial log is self-contained (replayed + new point)
    logged = [json.loads(line)["spares"] for line in
              (tmp_path / "resumed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(logged) == counts


# --------------------------------------------------------------------------- #
# demand mode: pure core (AC4 -- hand-computed Wilson knees)                    #
# --------------------------------------------------------------------------- #
def _drec(k, d):
    """A DemandRecord: finite d = successful unbounded run with demand d;
    d=None = non-success (D = infinity)."""
    outcome = "MISSION_SUCCESS" if d is not None else "MISSION_FAILED"
    return DemandRecord(replication=k, outcome=outcome, demand=d)


def _demand_batch(*counts_at):
    """Build records from (demand, how_many) pairs, e.g. (3, 90), (5, 6)."""
    records, k = [], 1
    for d, cnt in counts_at:
        for _ in range(cnt):
            records.append(_drec(k, d))
            k += 1
    return records


def test_demand_success_count_and_max_finite_demand():
    recs = _demand_batch((1, 2), (4, 1), (None, 1))
    assert demand_success_count(recs, 0) == 0
    assert demand_success_count(recs, 1) == 2
    assert demand_success_count(recs, 3) == 2
    assert demand_success_count(recs, 4) == 3    # the D=inf record never counts
    assert demand_success_count(recs, 100) == 3
    assert max_finite_demand(recs) == 4
    assert max_finite_demand([_drec(1, None)]) is None


def test_demand_knees_hand_computed_both_targets():
    # n=100: 90 reps demand 3, 6 reps demand 5, 4 reps demand 8.
    # CDF: P(D<=b) = 0 (b<3), 0.90 (3<=b<5), 0.96 (5<=b<8), 1.00 (b>=8).
    recs = _demand_batch((3, 90), (5, 6), (8, 4))
    knees = {k.target: k for k in demand_knees(recs, TARGETS)}
    # 95 % target: point knee at first b with frac >= 0.95 -> b=5 (0.96);
    # Wilson lower of 96/100 is ~0.90 < 0.95, of 100/100 is ~0.963 -> b=8.
    assert knees[0.95].knee_point == 5
    assert wilson_ci(96, 100)[0] < 0.95 < wilson_ci(100, 100)[0]
    assert knees[0.95].knee_wilson == 8
    # 99 % target: point knee at b=8 (1.00); Wilson lower of a PERFECT 100/100
    # is ~0.963 < 0.99 (needs >= 381 reps) -> no certified knee at any b.
    assert knees[0.99].knee_point == 8
    assert wilson_ci(100, 100)[0] < 0.99
    assert knees[0.99].knee_wilson is None
    assert min_reps_for_target(0.99) > 100  # exactly why it cannot certify


def test_demand_knees_all_demands_equal():
    # n=80, every rep demands exactly 4 packs: the CDF is a single step 0 -> 1.
    recs = _demand_batch((4, 80))
    knees = {k.target: k for k in demand_knees(recs, TARGETS)}
    # point knees for both targets sit AT the step
    assert knees[0.95].knee_point == 4
    assert knees[0.99].knee_point == 4
    # 80 reps clear the 95 % Wilson floor (73) but not the 99 % one (381)
    assert wilson_ci(80, 80)[0] >= 0.95
    assert knees[0.95].knee_wilson == 4
    assert knees[0.99].knee_wilson is None


def test_demand_knees_below_wilson_floor_yield_no_certified_knee():
    # 10 flawless reps: point knee resolves, Wilson-certified knee cannot
    # (10 < 73 [95 %] < 381 [99 %]) -- the CLI's existing [warn] covers this.
    recs = _demand_batch((2, 10))
    knees = {k.target: k for k in demand_knees(recs, TARGETS)}
    assert knees[0.95].knee_point == 2 and knees[0.99].knee_point == 2
    assert knees[0.95].knee_wilson is None
    assert knees[0.99].knee_wilson is None


def test_demand_knees_nonsuccess_blocks_targets():
    # 3 successes + 1 non-success (D=inf): frac never exceeds 0.75.
    recs = _demand_batch((1, 3), (None, 1))
    for k in demand_knees(recs, TARGETS):
        assert k.knee_point is None and k.knee_wilson is None
    # and with NO successful rep at all the scan domain is empty
    for k in demand_knees([_drec(1, None)], TARGETS):
        assert k.knee_point is None and k.knee_wilson is None


def test_demand_cdf_rows_match_wilson_ci():
    recs = _demand_batch((1, 2), (3, 1), (None, 1))  # n=4, max finite D=3
    rows = demand_cdf(recs)
    assert [r["spares"] for r in rows] == [0, 1, 2, 3]
    for r in rows:
        k = demand_success_count(recs, r["spares"])
        lo, hi, phat = wilson_ci(k, 4)
        assert (r["n_le"], r["success_frac"]) == (k, phat)
        assert (r["wilson95_lo"], r["wilson95_hi"]) == (lo, hi)
    assert demand_cdf([_drec(1, None)]) == []  # no success -> empty CDF


# --------------------------------------------------------------------------- #
# demand mode: partial jsonl + resume (AC5)                                    #
# --------------------------------------------------------------------------- #
def _write_demand_line(path, identity, k, outcome, demand, per_drone=None):
    import json
    rec = {
        "schema": "uav-swarm-sim/spare-sizing-demand-partial/v1",
        "kind": "demand",
        **identity,
        "replication": k, "outcome": outcome, "demand": demand,
        "per_drone_swaps": per_drone or {},
        "written_at": "2026-07-13T00:00:00+00:00",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def test_demand_load_parses_records_and_tolerates_truncated_final_line(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import load_demand_records

    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 3}
    p = tmp_path / "results_partial.jsonl"
    _write_demand_line(p, ident, 1, "MISSION_SUCCESS", 2, {"0": 1, "1": 1})
    _write_demand_line(p, ident, 2, "MISSION_FAILED", None, {"0": 3})
    with open(p, "a", encoding="utf-8") as f:  # crash mid-append
        f.write('{"schema": "uav-swarm-sim/spare-sizing-demand-partial/v1", "repl')
    identity, records = load_demand_records(p)
    assert identity == ident
    assert sorted(records) == [1, 2]
    assert records[1].demand == 2
    assert records[1].per_drone_swaps == {0: 1, 1: 1}  # keys back to int
    assert records[2].demand is None and records[2].outcome == "MISSION_FAILED"


def test_demand_and_grid_partial_logs_reject_each_other(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import (
        load_demand_records, load_partial_points)

    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 2}
    demand_log = tmp_path / "demand.jsonl"
    _write_demand_line(demand_log, ident, 1, "MISSION_SUCCESS", 1)
    grid_log = tmp_path / "grid.jsonl"
    _write_partial_line(grid_log, ident, spares=0, n_reps=2, n_success=2)
    # a grid --resume must refuse a demand log, and vice versa: the schema
    # string is the cross-mode firewall
    with pytest.raises(SystemExit, match="unsupported schema"):
        load_partial_points(demand_log)
    with pytest.raises(SystemExit, match="unsupported schema"):
        load_demand_records(grid_log)


def test_demand_load_rejects_malformed_record_and_empty_log(tmp_path):
    import json
    from uav_swarm_sim.experiments.run_spare_sizing import load_demand_records

    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 2}
    broken = tmp_path / "missing_key.jsonl"
    broken.write_text(json.dumps({
        "schema": "uav-swarm-sim/spare-sizing-demand-partial/v1", "kind": "demand",
        **ident, "replication": 1, "outcome": "MISSION_SUCCESS",
        "per_drone_swaps": {},  # no "demand" key
    }) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="malformed record"):
        load_demand_records(broken)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no completed replications"):
        load_demand_records(empty)


def test_demand_resume_identity_mismatch_and_out_of_range_reps(tmp_path):
    from uav_swarm_sim.experiments.run_spare_sizing import _validated_demand_resume

    ident = {"master_seed": 7, "config_hash": "h", "reps_per_point": 3}
    p = tmp_path / "results_partial.jsonl"
    _write_demand_line(p, ident, 1, "MISSION_SUCCESS", 1)
    _write_demand_line(p, ident, 99, "MISSION_SUCCESS", 1)  # outside 1..3
    # identity mismatch -> refuse loudly (same wording as the grid path)
    other = {"master_seed": 8, "config_hash": "h", "reps_per_point": 3}
    with pytest.raises(SystemExit, match="identity mismatch.*master_seed"):
        _validated_demand_resume(p, other, 3)
    with pytest.raises(SystemExit, match="no such file"):
        _validated_demand_resume(tmp_path / "nope.jsonl", ident, 3)
    # matching identity: replication 99 is ignored, replication 1 resumed
    done = _validated_demand_resume(p, ident, 3)
    assert sorted(done) == [1]


def test_demand_mode_cli_rejects_spare_grid_flags():
    from uav_swarm_sim.experiments.run_spare_sizing import main

    with pytest.raises(SystemExit, match="demand-mode replaces the spare grid"):
        main(["--demand-mode", "--spares", "0", "1"])
    with pytest.raises(SystemExit, match="demand-mode replaces the spare grid"):
        main(["--demand-mode", "--spare-range", "0", "4", "1"])


# --------------------------------------------------------------------------- #
# demand mode: the count-constraint equivalence, locked empirically (AC3)      #
# --------------------------------------------------------------------------- #
def _tiny_demand_cfg(config_path):
    """Small enough to swap: 2 drones on ~20 Wh packs over the smoke area give
    demands D in {1, 2} with MISSION_SUCCESS in ~a second per replication."""
    from uav_swarm_sim.infrastructure.config import load_config
    return load_config(
        config_path,
        overrides={
            "fleet.n_drones": 2,
            "fleet.battery_capacity_wh": 20.0,
            "failure.hazard_rate_per_hour": 0.0,
            "env.geojson_path": "data/areas/smoke_area.geojson",
            "env.obstacle_density_per_km2": 4.0,
            "env.obstacle_size_range_m": [10.0, 30.0],
            "sim.dt_s": 1.0,
            "sim.max_timesteps": 20000,
        },
    )


@pytest.mark.slow
def test_demand_equals_grid_success_for_every_replication_and_pool_size(config_path):
    """THE equivalence lock: success(k, B) == (D_k <= B) for EVERY (k, B) pair.

    Demand mode measures D_k once under an unbounded pool; the classic grid
    then re-runs the SAME replications at every pool size 0..max(D)+1 and the
    engine's own terminal outcome must match the count-constraint prediction
    exactly -- the code-level argument (single decrement site, no other pool
    reader) made empirical.
    """
    from uav_swarm_sim.experiments.run_spare_sizing import _with_reserve, run_demand
    from uav_swarm_sim.infrastructure.enums import Outcome, PlannerKind
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine

    cfg = _tiny_demand_cfg(config_path)
    reps = 3
    records = run_demand(cfg, reps, RngFactory(cfg.sim.master_seed))
    assert [r.replication for r in records] == [1, 2, 3]
    finite = [r.demand for r in records if r.demand is not None]
    assert finite, "tiny config must yield at least one successful unbounded run"
    assert max(finite) >= 1, "tiny config must actually demand swap packs"
    for r in records:  # per-drone array sums to the fleet demand (same source)
        if r.demand is not None:
            assert sum(r.per_drone_swaps.values()) == r.demand

    for b in range(0, max(finite) + 2):
        cfg_b = _with_reserve(cfg, b)
        for rec in records:
            out = SimulationEngine(cfg_b, RngFactory(cfg.sim.master_seed),
                                   replication=rec.replication,
                                   planner=PlannerKind.DUBINS).run().outcome
            expected = rec.demand is not None and rec.demand <= b
            assert (out is Outcome.MISSION_SUCCESS) == expected, \
                f"equivalence broken at (k={rec.replication}, B={b}): " \
                f"D_k={rec.demand}, engine outcome={out}"


@pytest.mark.slow
def test_demand_partial_log_appended_per_replication_and_resume_matches(tmp_path, config_path):
    """Crash-safety + resume in demand mode: one fsync'd line per replication,
    and interrupted+resumed records identical to an uninterrupted batch."""
    import json
    from uav_swarm_sim.experiments.run_spare_sizing import (
        _partial_identity, append_demand_record, demand_with_partials, run_demand)
    from uav_swarm_sim.infrastructure.rng import RngFactory

    cfg = _tiny_demand_cfg(config_path)
    reps = 3
    seed = cfg.sim.master_seed
    ident = _partial_identity(cfg, reps)

    # uninterrupted reference batch, with the per-replication append spied on
    full_log = tmp_path / "full.jsonl"
    seen: list[int] = []

    def _spy(rec):
        lines = full_log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(seen) + 1  # already on disk when progress fires
        on_disk = json.loads(lines[-1])
        assert on_disk["kind"] == "demand"
        assert on_disk["replication"] == rec.replication
        assert on_disk["demand"] == rec.demand
        seen.append(rec.replication)

    full = demand_with_partials(cfg, reps, RngFactory(seed), full_log, progress=_spy)
    assert seen == [1, 2, 3]

    # a batch 'crashed' after replication 2...
    crashed_log = tmp_path / "crashed.jsonl"
    run_demand(cfg, reps, RngFactory(seed), replications=[1, 2],
               progress=lambda rec: append_demand_record(crashed_log, rec, ident))
    # ...and resumed: only replication 3 is actually simulated
    resumed = demand_with_partials(cfg, reps, RngFactory(seed),
                                   tmp_path / "resumed.jsonl",
                                   resume_path=crashed_log)
    as_tuple = lambda r: (r.replication, r.outcome, r.demand,
                          sorted(r.per_drone_swaps.items()))
    assert [as_tuple(r) for r in resumed] == [as_tuple(r) for r in full]
    # the resumed run's own log is self-contained (replayed + new replication)
    logged = [json.loads(line)["replication"] for line in
              (tmp_path / "resumed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(logged) == [1, 2, 3]


@pytest.mark.slow
def test_without_resume_sweep_behavior_is_unchanged(tmp_path, config_path):
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.experiments.run_spare_sizing import (
        run_sweep, sweep_with_partials)

    cfg = _tiny_cfg(config_path)
    reps, counts = 2, [0, 50]

    baseline = run_sweep(cfg, counts, reps, RngFactory(cfg.sim.master_seed))
    wrapped = sweep_with_partials(cfg, counts, reps, RngFactory(cfg.sim.master_seed),
                                  tmp_path / "results_partial.jsonl")
    # default-off: no --resume => the wrapper only ADDS the jsonl side file;
    # the swept points (and hence results.json) are exactly run_sweep's
    assert [(p.spares, p.n_success, p.n_failed, p.n_incomplete) for p in wrapped] == \
           [(p.spares, p.n_success, p.n_failed, p.n_incomplete) for p in baseline]
    assert (tmp_path / "results_partial.jsonl").exists()
