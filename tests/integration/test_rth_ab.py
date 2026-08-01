"""C1 gate tests: the Stage-5 RTH A/B harness (experiments/run_rth_ab.py).

What is pinned here, per the C1 plan:

1. ARM CONSTRUCTION -- the four arms carry exactly their defining flags, the
   experiment constants are identical across arms, the base Config is never
   mutated, and arm 1 is field-equal to plain-load + the constants (including
   the carried-over config_hash).
2. BYTE-IDENTITY GATE -- the harness's arm-1 per-replication loop reproduces
   the EXISTING demand path (run_spare_sizing.run_demand) bit for bit on the
   same config: full run-signature equality of the underlying engine runs plus
   record-level equality of the shared fields. Existing files are untouched by
   C1 (structural flag-off guarantee); this test pins the behavioral half.
3. PAIRED-SEED SMOKE (arm 1 vs arm 4) -- same obstacle layout across arms,
   reason attribution differing in the expected direction, and the R1
   exclusivity assertion: every sojourn closed with one of the three RTH
   reasons is followed, in that agent's own chronological sequence, by an
   S3_RTH sojourn (runtime pin of the grep-verified fact that
   execution/state_machine.py:112-116 are the only producers).
4. PARTIAL TABULATION -- synthetic records: MISSION_PARTIAL gets its own row
   in the aggregate, never folded into success or failure (the A1 seam).
5. RESUME IDENTITY -- the partial-log loader refuses foreign schemas and
   mismatched identities, tolerates only a truncated final line.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from uav_swarm_sim.experiments.run_rth_ab import (
    ARM4_TERMINAL_FLOOR,
    ARM_SLUGS,
    PARTIAL_SCHEMA,
    RTH_REASONS,
    AbRecord,
    append_ab_record,
    arm_config,
    arm_flags,
    arm_summary,
    experiment_constants,
    load_ab_records,
    load_resume_dir,
    run_arm,
)
from uav_swarm_sim.experiments.run_spare_sizing import _with_reserve, run_demand
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


# --------------------------------------------------------------------------- #
# shared fixtures / helpers                                                    #
# --------------------------------------------------------------------------- #
def _tiny_cfg(config_path, battery_wh=400.0):
    """The small fast world the stage-2/3 gates use (smoke area, 2 drones),
    with the C1 base requirement (stall_detector) satisfied. The default big
    battery keeps the byte-identity run swap-free (the stage-3 convention);
    the reason-attribution smoke passes a small one so returns actually occur."""
    return load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": battery_wh,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
        "safety.stall_detector": True,
    })


def _metrics_tuple(m):
    return (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
            dict(m.per_agent_energy_j), dict(m.per_agent_length_m))


def _run_signature(res) -> tuple:
    """Full byte-identity artifact (the stage-3 gate's convention): summary
    metrics + the complete FSM sojourn trajectory + outcome/coverage/stalled."""
    return (
        _metrics_tuple(res.metrics),
        tuple(res.history.sojourns()),
        res.outcome,
        res.coverage_frac,
        res.stalled_agents,
    )


def _record(arm=1, replication=1, outcome="MISSION_SUCCESS", demand=5,
            reasons=None, **kw) -> AbRecord:
    """A synthetic AbRecord for pure-function tests."""
    defaults = dict(
        per_drone_swaps={0: 3, 1: 2},
        return_depths=[0.41, 0.40],
        total_energy_j=1.0e6,
        duration_s=1000.0,
        coverage_frac=1.0,
        n_skipped_legs=0,
        stalled_agents=(),
        n_map_hits=0,
        n_map_fallbacks=0,
        n_route_fallbacks=0,
    )
    defaults.update(kw)
    return AbRecord(arm=arm, replication=replication, outcome=outcome,
                    demand=demand,
                    reasons=reasons or {r: 0 for r in RTH_REASONS},
                    **defaults)


# --------------------------------------------------------------------------- #
# 1. arm construction                                                          #
# --------------------------------------------------------------------------- #
def test_arm_flag_matrix(config_path):
    base = experiment_constants(_tiny_cfg(config_path))
    expected = {
        1: (False, False, False, False),
        2: (True, False, True, False),
        3: (True, True, True, False),
        4: (True, True, True, True),
    }
    for arm, (en, de, ro, zd) in expected.items():
        cfg = arm_config(base, arm)
        em = cfg.rth.energy_map
        assert (em.enabled, em.decide, em.route, em.zone_demotion) == (en, de, ro, zd), arm
        # the arm-4 terminal floor is in-config and arm-4 only
        want_floor = ARM4_TERMINAL_FLOOR if arm == 4 else base.battery_zones.critical
        assert cfg.battery_zones.critical == want_floor, arm
        flags = arm_flags(cfg, arm)
        assert flags["arm_slug"] == ARM_SLUGS[arm]
        assert flags["terminal_floor"] == want_floor


def test_constants_identical_across_arms_and_base_unmutated(config_path):
    """The author's fifth-variable rule: everything except the four energy_map
    flags (+ arm-4 floor) is pinned identically across arms."""
    loaded = _tiny_cfg(config_path)
    base = experiment_constants(loaded)
    # replace() built copies -- the loaded object still carries its YAML values
    assert loaded.fleet.total_reserve_batteries is not None  # default.yaml: 50
    assert not loaded.safety.stall_skip
    arms = {a: arm_config(base, a) for a in (1, 2, 3, 4)}
    for a, cfg in arms.items():
        assert cfg.fleet.total_reserve_batteries is None, a
        assert cfg.safety.stall_skip is True, a
        assert cfg.safety.stall_detector is True, a
        assert cfg.coverage == base.coverage, a
        assert cfg.sim == base.sim, a
        assert cfg.rth.reserve_frac == base.rth.reserve_frac, a
        assert cfg.config_hash == base.config_hash, a  # carried, not recomputed


def test_arm1_equals_plain_load_plus_constants(config_path):
    """Arm 1 is EXACTLY the constants-pinned base -- no energy_map block, field
    equality including the carried config_hash."""
    base = experiment_constants(_tiny_cfg(config_path))
    manual = _with_reserve(_tiny_cfg(config_path), None)
    manual = dataclasses.replace(
        manual, safety=dataclasses.replace(manual.safety, stall_skip=True))
    assert arm_config(base, 1) == manual


def test_experiment_constants_requires_stall_detector(config_path):
    cfg = load_config(str(config_path), overrides={
        "safety.stall_detector": False,
    })
    with pytest.raises(SystemExit, match="stall_detector"):
        experiment_constants(cfg)


# --------------------------------------------------------------------------- #
# 2. byte-identity gate: harness arm 1 == existing demand path                 #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_arm1_byte_identical_to_run_demand(config_path):
    """The harness's own engine loop reproduces run_demand bit for bit on the
    same config: identical run signature underneath, identical shared record
    fields on top."""
    base = experiment_constants(_tiny_cfg(config_path))
    cfg1 = arm_config(base, 1)

    # record level: the harness record carries exactly the demand record's data
    ab = run_arm(cfg1, 1, 1, RngFactory(cfg1.sim.master_seed))[0]
    # run_demand forces the unbounded pool itself (idempotent on cfg1)
    dem = run_demand(cfg1, 1, RngFactory(cfg1.sim.master_seed))[0]
    assert (ab.replication, ab.outcome, ab.demand, ab.per_drone_swaps) == \
        (dem.replication, dem.outcome, dem.demand, dem.per_drone_swaps)

    # run level: the engine construction the two loops perform is identical
    sig_ab = _run_signature(SimulationEngine(
        cfg1, RngFactory(cfg1.sim.master_seed), replication=1, algo=None,
        planner=PlannerKind.DUBINS).run())
    sig_dem = _run_signature(SimulationEngine(
        _with_reserve(cfg1, None), RngFactory(cfg1.sim.master_seed),
        replication=1, algo=None, planner=PlannerKind.DUBINS).run())
    assert sig_ab == sig_dem


# --------------------------------------------------------------------------- #
# 3. paired-seed smoke: arm 1 vs arm 4                                         #
# --------------------------------------------------------------------------- #
def _obstacle_signature(env) -> list[str]:
    return sorted(o.polygon.wkt for o in env.obstacles)


@pytest.mark.slow
def test_paired_seeds_and_reason_attribution(config_path):
    """Arms share the world (identical obstacle layout at the same replication)
    and the reason attribution obeys the STRUCTURAL arm facts; plus the R1
    runtime assertion that the three RTH reasons occur ONLY on transitions
    into S3_RTH.

    Deliberately asserted here are only the world-independent facts. The full
    arm1-vs-arm4 attribution inversion (critical_battery dominant vs rth_energy
    dominant) is a 1 km^2-regime behavior: in this tiny world one coverage
    leg's lookahead bundle is a large fraction of the small battery, so the
    ANALYTIC route-vs-return reserve (the PRIMARY trigger, guideline 3.1;
    rth_calculator.should_return) fires above the 0.40 zone boundary and even
    arm 1 attributes its returns to ``rth_energy``. At study01 scale the same
    trigger arms below 0.40, the static CRITICAL net pre-empts, and arm 1
    flips to ``critical_battery`` -- which is exactly the regime the Azure run
    measures (C1 probe, rep 1-5: arm 1 = 100% critical_battery). A test at
    that scale costs ~18 min/run and belongs to the experiment, not the net."""
    base = experiment_constants(_tiny_cfg(config_path, battery_wh=20.0))
    rng = RngFactory(base.sim.master_seed)

    results = {}
    for arm in (1, 4):
        eng = SimulationEngine(arm_config(base, arm), rng, replication=1,
                               algo=None, planner=PlannerKind.DUBINS)
        res = eng.run()
        results[arm] = (eng, res)

    # (i) paired seeds: byte-identical environment across arms
    assert _obstacle_signature(results[1][0].env) == \
        _obstacle_signature(results[4][0].env)

    # (ii) structural attribution facts (hold in ANY world)
    def reasons(res):
        counts = {r: 0 for r in RTH_REASONS}
        for s in res.history.sojourns():
            if s.reason_out in counts:
                counts[s.reason_out] += 1
        return counts

    r1, r4 = reasons(results[1][1]), reasons(results[4][1])
    assert sum(r1.values()) > 0            # returns actually happened
    assert sum(r4.values()) > 0
    assert results[1][0].rth.n_map_hits == 0   # arm 1: no map involvement
    assert results[4][0].rth.n_map_hits > 0    # arm 4: the map is consulted
    assert r4["critical_battery"] == 0     # zone_demotion removed the branch
                                           # structurally (state_machine.py:113)

    # (iii) R1 exclusivity: every RTH-reason sojourn is followed by S3_RTH in
    # that agent's own chronological sequence
    for _, res in results.values():
        by_agent: dict[int, list] = {}
        for s in res.history.sojourns():
            by_agent.setdefault(s.agent_id, []).append(s)
        for seq in by_agent.values():
            seq.sort(key=lambda s: s.t_in)
            for i, s in enumerate(seq):
                if s.reason_out in RTH_REASONS:
                    assert i + 1 < len(seq), "RTH-reason sojourn must have a successor"
                    assert seq[i + 1].state is AgentState.S3_RTH, \
                        (s.reason_out, seq[i + 1].state)


# --------------------------------------------------------------------------- #
# 4. PARTIAL tabulation (pure function)                                        #
# --------------------------------------------------------------------------- #
def test_partial_gets_its_own_row_in_the_aggregate():
    records = [
        _record(replication=1, outcome="MISSION_SUCCESS", demand=5),
        _record(replication=2, outcome="MISSION_PARTIAL", demand=None,
                n_skipped_legs=2, coverage_frac=0.96),
        _record(replication=3, outcome="MISSION_INCOMPLETE", demand=None),
        _record(replication=4, outcome="MISSION_SUCCESS", demand=7),
    ]
    s = arm_summary(records)
    oc = s["outcome_counts"]
    assert oc["MISSION_SUCCESS"] == 2
    assert oc["MISSION_PARTIAL"] == 1       # its own row -- the A1 seam
    assert oc["MISSION_INCOMPLETE"] == 1
    assert oc["MISSION_FAILED"] == 0
    # strict predicate: PARTIAL contributes no finite demand
    assert s["demand_median"] == 6.0
    assert s["success_frac"] == pytest.approx(0.5)
    assert s["n_reps_with_skipped_legs"] == 1
    assert s["total_skipped_legs"] == 2


def test_reason_counts_aggregate():
    records = [
        _record(replication=1, reasons={"rth_energy": 0, "critical_battery": 8,
                                        "terminal_battery": 0}),
        _record(replication=2, reasons={"rth_energy": 0, "critical_battery": 6,
                                        "terminal_battery": 1}),
    ]
    rc = arm_summary(records)["reason_counts"]
    assert rc == {"rth_energy": 0, "critical_battery": 14, "terminal_battery": 1}


# --------------------------------------------------------------------------- #
# 5. resume identity                                                           #
# --------------------------------------------------------------------------- #
IDENT = {"master_seed": 42, "config_hash": "abc", "reps": 4}


def _write_partial(tmp_path, records, identity=None, subdir="simulation-arm1-static40"):
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / "results_partial.jsonl"
    for rec in records:
        append_ab_record(p, rec, identity or IDENT)
    return p


def test_partial_roundtrip(tmp_path):
    recs = [_record(replication=1), _record(replication=2, outcome="MISSION_PARTIAL",
                                            demand=None)]
    p = _write_partial(tmp_path, recs)
    identity, loaded = load_ab_records(p)
    assert identity == IDENT
    assert set(loaded) == {(1, 1), (1, 2)}
    assert loaded[(1, 1)] == recs[0]
    assert loaded[(1, 2)] == recs[1]


def test_resume_dir_merges_and_checks_identity(tmp_path):
    _write_partial(tmp_path, [_record(replication=1)])
    _write_partial(tmp_path, [_record(arm=4, replication=1)],
                   subdir="simulation-arm4-full-map")
    done = load_resume_dir(tmp_path, IDENT)
    assert set(done) == {(1, 1), (4, 1)}
    with pytest.raises(SystemExit, match="identity mismatch"):
        load_resume_dir(tmp_path, {**IDENT, "master_seed": 43})


def test_resume_refuses_foreign_schema(tmp_path):
    p = _write_partial(tmp_path, [_record(replication=1)])
    line = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    line["schema"] = "uav-swarm-sim/spare-sizing-demand-partial/v1"
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unsupported schema"):
        load_ab_records(p)


def test_resume_tolerates_truncated_final_line_only(tmp_path):
    p = _write_partial(tmp_path, [_record(replication=1), _record(replication=2)])
    text = p.read_text(encoding="utf-8")
    p.write_text(text + '{"schema": "uav-swarm-sim/rth-ab-partial/v1", "trunc',
                 encoding="utf-8")
    _, loaded = load_ab_records(p)   # truncated FINAL line: skipped
    assert set(loaded) == {(1, 1), (1, 2)}

    lines = text.splitlines()
    corrupted = lines[0][:40] + "\n" + lines[1] + "\n"
    p.write_text(corrupted, encoding="utf-8")
    with pytest.raises(SystemExit, match="corrupt line"):
        load_ab_records(p)           # corrupt NON-final line: refused


def test_resume_missing_dir_and_empty_dir(tmp_path):
    with pytest.raises(SystemExit, match="no such run directory"):
        load_resume_dir(tmp_path / "nope", IDENT)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no results_partial"):
        load_resume_dir(empty, IDENT)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
