"""EXP-12 unit gates: the paired fixed-replication Lloyd protocol runner.

Engine-free -- a stub replication drives the loop -- so these are fast. What is
pinned: fixed N with no early stop (all samples saved, incl. zero variance), the
production 30-rep floor + smoke stamp, fail-fast config validation, the
per-replication error trap, the physical projection, duplicate/resume content
rules, manifest parity (ok / unverifiable / mismatch), and pad dispersion.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.experiments import run_lloyd_protocol as rp
from uav_swarm_sim.experiments.run_lloyd_protocol import (
    ARMS,
    ProtocolRecord,
    append_record,
    build_cfg,
    check_reps,
    load_records,
    pad_dispersion,
    parity_report,
    projection,
    run_arm,
    validate_config,
)
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _mk(algo, n, k, *, fp="fp", outcome="MISSION_SUCCESS", manifest="auto",
        wall=0.1, planning=0.2, smoke=False, error=None):
    man = manifest
    if manifest == "auto":
        man = {"launch_pose_xy": [float(k), 0.0], "fingerprint_sha256": fp}
    return ProtocolRecord(
        algo=algo.value if hasattr(algo, "value") else algo, n_drones=n,
        replication=k, outcome=outcome, coverage_frac=1.0,
        target_coverage_frac=1.0, total_energy_j=100.0, duration_s=10.0,
        planning_time_s=planning, wall_time_s=wall, smoke=smoke,
        manifest=man, contract={"c": 1}, error=error)


# --------------------------------------------------------------------------- #
# 1. fixed N, no early stop, all samples saved (incl. zero variance)           #
# --------------------------------------------------------------------------- #
def test_fixed_n_runs_every_replication_with_zero_variance():
    algo = DecompositionAlgo.LLOYD_CVT

    def stub(cfg, a, n, k, rng, smoke):
        # identical result every rep => zero sample variance: a convergence
        # predicate WOULD stop here, but this runner has no such path.
        return _mk(a, n, k, outcome="MISSION_SUCCESS")

    recs = run_arm(None, algo, 3, None, range(1, 31), False, unit_fn=stub)
    assert len(recs) == 30
    assert sorted(r.replication for r in recs) == list(range(1, 31))


# --------------------------------------------------------------------------- #
# 2. production 30-rep floor + smoke stamp                                     #
# --------------------------------------------------------------------------- #
def test_check_reps_floor_and_smoke():
    with pytest.raises(SystemExit):
        check_reps(29, smoke=False)
    check_reps(29, smoke=True)      # smoke lifts the floor
    check_reps(30, smoke=False)     # exactly at the floor is fine


def test_smoke_flag_lands_in_the_record():
    def stub(cfg, a, n, k, rng, smoke):
        return _mk(a, n, k, smoke=smoke)

    recs = run_arm(None, ARMS[0], 3, None, [1, 2], True, unit_fn=stub)
    assert all(r.smoke is True for r in recs)


# --------------------------------------------------------------------------- #
# 3. fail-fast config validation                                              #
# --------------------------------------------------------------------------- #
def test_validate_config_fails_fast_without_raster():
    # default.yaml is mission.type=coverage with no raster override, so validation
    # refuses specifically on the raster branch (matched, so the branch is truly
    # exercised -- a bare SystemExit could also come from another check).
    cfg = build_cfg("config/study01_demand.yaml", 3)
    with pytest.raises(SystemExit, match=r"coverage\.raster_enabled"):
        validate_config(cfg)


# --------------------------------------------------------------------------- #
# 4. physical projection strips timing                                        #
# --------------------------------------------------------------------------- #
def test_projection_ignores_wall_and_planning_time():
    a = _mk(ARMS[0], 3, 1, wall=0.1, planning=0.2)
    b = _mk(ARMS[0], 3, 1, wall=9.9, planning=8.8)
    assert projection(a) == projection(b)
    assert a.wall_time_s != b.wall_time_s


# --------------------------------------------------------------------------- #
# 5. per-replication error trap (real _run_protocol_unit, fake engine)         #
# --------------------------------------------------------------------------- #
def test_error_trap_records_crash_without_manifest(monkeypatch):
    class _BoomEngine:
        def __init__(self, *a, **k):
            self.launch_pose = None  # _build() never got to siting

        def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(rp, "SimulationEngine", _BoomEngine)
    rec = rp._run_protocol_unit(None, DecompositionAlgo.LLOYD_CVT, 3, 7, None, False)
    assert rec.outcome is None
    assert rec.manifest is None
    assert rec.error["type"] == "RuntimeError"
    assert "boom" in rec.error["message"]


def test_batch_continues_past_a_failed_replication():
    def stub(cfg, a, n, k, rng, smoke):
        if k == 2:
            return _mk(a, n, k, outcome=None, manifest=None,
                       error={"type": "RuntimeError", "message": "x"})
        return _mk(a, n, k)

    recs = run_arm(None, ARMS[0], 3, None, [1, 2, 3], False, unit_fn=stub)
    assert len(recs) == 3
    assert sum(1 for r in recs if r.error is not None) == 1


# --------------------------------------------------------------------------- #
# 6. duplicate / resume content rules                                          #
# --------------------------------------------------------------------------- #
def test_load_records_dedups_identical_duplicate(tmp_path):
    p = tmp_path / rp.PARTIAL_FILENAME
    ident = rp._identity(42, 30, False)
    rec = _mk(ARMS[0], 3, 1)
    append_record(p, rec, "hash-a", ident)
    append_record(p, rec, "hash-a", ident)  # identical dup
    _, out = load_records(p)
    assert len(out) == 1


def test_load_records_refuses_conflicting_duplicate(tmp_path):
    p = tmp_path / rp.PARTIAL_FILENAME
    ident = rp._identity(42, 30, False)
    append_record(p, _mk(ARMS[0], 3, 1, outcome="MISSION_SUCCESS"), "hash-a", ident)
    append_record(p, _mk(ARMS[0], 3, 1, outcome="MISSION_FAILED"), "hash-a", ident)
    with pytest.raises(SystemExit):
        load_records(p)


# --------------------------------------------------------------------------- #
# 7. manifest parity                                                          #
# --------------------------------------------------------------------------- #
def test_parity_ok_when_both_arms_share_fingerprint():
    recs = [_mk(ARMS[0], 3, 1, fp="same"), _mk(ARMS[1], 3, 1, fp="same")]
    rep = parity_report(recs)
    assert rep["ok"] == 1 and rep["mismatch"] == 0 and rep["unverifiable"] == 0


def test_parity_unverifiable_when_one_arm_crashed():
    recs = [_mk(ARMS[0], 3, 1, fp="same"),
            _mk(ARMS[1], 3, 1, manifest=None, outcome=None)]
    rep = parity_report(recs)
    assert rep["unverifiable"] == 1 and rep["ok"] == 0


def test_parity_raises_on_real_mismatch():
    recs = [_mk(ARMS[0], 3, 1, fp="A"), _mk(ARMS[1], 3, 1, fp="B")]
    with pytest.raises(SystemExit):
        parity_report(recs)


# --------------------------------------------------------------------------- #
# 8. pad dispersion                                                           #
# --------------------------------------------------------------------------- #
def test_pad_dispersion_counts_distinct_pads():
    recs = [_mk(ARMS[0], 3, 1), _mk(ARMS[0], 3, 2), _mk(ARMS[0], 3, 3)]
    # launch_pose_xy = [k, 0] for k=1,2,3 -> three distinct pads
    disp = pad_dispersion(recs)
    assert disp["n_manifests"] == 3
    assert disp["n_distinct_pads"] == 3
    assert disp["pad_spread_m"] > 0.0


def test_pad_dispersion_none_when_no_manifest():
    recs = [_mk(ARMS[0], 3, 1, manifest=None, outcome=None)]
    disp = pad_dispersion(recs)
    assert disp["n_manifests"] == 0
    assert disp["n_distinct_pads"] is None
