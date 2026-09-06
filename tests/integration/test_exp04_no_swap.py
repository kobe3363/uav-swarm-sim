"""EXP-04: finite-battery no-swap lifecycle and explicit mission outcomes.

Engine-level checks of author decision C-4 under ``mission.no_swap_mode``:
a returning drone lands in the terminal S_LANDED state on its own battery
(zero swaps / resets / relaunches), the reserve-pool size has no effect,
SUCCESS needs coverage AND touchdown, all-landed-below-gate is PARTIAL, an
airborne depletion is FAILED, a time cap with airborne drones stays
INCOMPLETE, and the flag-off run is byte-identical to the pre-EXP-04 binary
(cross-commit golden captured on a7ee5d7).
"""
from __future__ import annotations

import hashlib

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    BatteryZone,
    DecompositionAlgo,
    Outcome,
    PlannerKind,
)
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.monte_carlo import single_run_from_history
from uav_swarm_sim.physical_model.battery import Battery

S = AgentState


# --------------------------------------------------------------------------- #
# fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #
def _no_swap_cfg(config_path, *, n_drones=1, reserve=0, max_timesteps=10000, **extra):
    """The EXP-02 raster photogrammetry mission on the smoke area, no-swap on."""
    overrides = {
        "fleet.n_drones": n_drones,
        "fleet.battery_capacity_wh": 1000.0,
        "fleet.total_reserve_batteries": reserve,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "sim.dt_s": 0.5,
        "sim.max_timesteps": max_timesteps,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 17.3,
        "sensor.photogrammetry.sensor_height_mm": 13.0,
        "sensor.photogrammetry.focal_length_mm": 12.0,
        "sensor.photogrammetry.image_width_px": 5280,
        "sensor.photogrammetry.image_height_px": 3956,
        "sensor.photogrammetry.side_overlap": 0.70,
        "sensor.photogrammetry.forward_overlap": 0.80,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "coverage.raster_enabled": True,
        "coverage.raster_cell_m": 10.0,
        "mission.no_swap_mode": True,
    }
    overrides.update(extra)
    return load_config(config_path, overrides=overrides)


def _engine(cfg, replication=0):
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=replication,
                            algo=DecompositionAlgo.WEIGHTED_VORONOI)


def _run_with_post_build(engine, hook):
    """Run the engine, applying ``hook(engine)`` right after ``_build`` (the
    only seam at which a live agent can be given a different battery/RTH
    before the first tick)."""
    orig_build = engine._build

    def _build_then_hook():
        orig_build()
        hook(engine)

    engine._build = _build_then_hook
    return engine.run()


def _gate(cfg):
    return 1.0 - cfg.coverage.raster_completion_tolerance_frac


def _sojourns_of(res, aid):
    return [s for s in res.history.sojourns() if s.agent_id == aid]


def _no_swap_invariants(res):
    """Zero swaps / resets / relaunches: no S_SWAP sojourn, no swap_done exit,
    every agent's battery trace is monotone non-increasing, and every agent
    (each has a zone in these fixtures) launches exactly once."""
    sj = res.history.sojourns()
    assert all(s.state is not S.S_SWAP for s in sj)
    assert all(s.reason_out != "swap_done" for s in sj)
    assert res.metrics.n_swaps == 0
    for aid in {s.agent_id for s in sj}:
        trace = res.history.battery_trace(aid)
        assert all(b <= a + 1e-12 for (_, a), (_, b) in zip(trace, trace[1:])), aid
        launches = [s for s in _sojourns_of(res, aid) if s.reason_out == "launch"]
        assert len(launches) == 1


class _AlwaysHigh(Battery):
    """Battery whose static zone nets never fire (HIGH regardless of level)."""
    @property
    def zone(self) -> BatteryZone:
        return BatteryZone.HIGH


class _RthStub:
    """Minimal RTH surface the agent reads (no energy-map, fixed verdict)."""
    map_decide_on = False
    map_route_on = False
    check_interval_s = 5.0

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    def decide(self, agent) -> str:
        return self._verdict


# --------------------------------------------------------------------------- #
# 1. one drone retires early, the other keeps working; work released once    #
# --------------------------------------------------------------------------- #
def test_early_return_one_uav_other_continues(config_path):
    cfg = _no_swap_cfg(config_path, n_drones=2, reserve=0)

    def drain_agent_0(engine):
        bat = engine.fleet.agents[0].battery
        bat.drain(0.65 * bat.capacity_j)          # 0.35 < nominal 0.40 -> CRITICAL net

    res = _run_with_post_build(_engine(cfg), drain_agent_0)
    _no_swap_invariants(res)

    landed = {aid: [s for s in _sojourns_of(res, aid) if s.state is S.S_LANDED]
              for aid in (0, 1)}
    assert len(landed[0]) == 1 and len(landed[1]) == 1
    t_land_0 = landed[0][0].t_in
    assert _sojourns_of(res, 0)[-1].state is S.S_LANDED
    # the healthy drone keeps covering after drone 0 has touched down
    later_cov = [s for s in _sojourns_of(res, 1)
                 if s.state is S.S2_MISSION and s.t_out > t_land_0]
    assert later_cov, "drone 1 must keep working after drone 0 retired"
    assert landed[1][0].t_in > t_land_0

    # drone 0 released its uncovered work exactly once; drone 1 finished its own
    assert res.retired_agents == (0, 1)
    assert res.work_releases == ((0, t_land_0),)
    # coverage is an independent metric: below the gate -> safe PARTIAL
    assert res.outcome is Outcome.MISSION_PARTIAL
    assert res.terminal_reason == "all_landed_below_gate"
    assert 0.0 < res.coverage_frac < _gate(cfg)
    assert res.losses == () and res.airborne_at_end == ()


# --------------------------------------------------------------------------- #
# 2. reserve-pool size has no effect; pool_exhausted never consulted         #
# --------------------------------------------------------------------------- #
def _signature(res):
    m = res.metrics
    return (m.total_energy_j, m.duration_s, m.n_swaps, dict(m.per_agent_energy_j),
            tuple(res.history.sojourns()), res.outcome, res.coverage_frac,
            res.terminal_reason, res.retired_agents, res.work_releases)


def test_reserve_pool_size_has_no_effect(config_path):
    sigs = []
    for reserve in (0, None, 50):
        cfg = _no_swap_cfg(config_path, n_drones=1, reserve=reserve)
        eng = _engine(cfg)
        res = eng.run()
        _no_swap_invariants(res)
        assert eng.swap_station.pool_exhausted is False
        assert eng.swap_station.reserve_remaining == reserve   # never decremented
        assert res.outcome is not Outcome.MISSION_FAILED
        sigs.append(_signature(res))
    assert sigs[0] == sigs[1] == sigs[2]


# --------------------------------------------------------------------------- #
# 3. reaching the coverage gate airborne is not yet SUCCESS                   #
# --------------------------------------------------------------------------- #
def test_success_only_after_touchdown(config_path):
    cfg = _no_swap_cfg(config_path, n_drones=1)
    eng = _engine(cfg)
    probes = []                       # (t, coverage, outcome, airborne_ids, energy_j)
    orig = eng._evaluate_terminal_no_swap

    def probing(t):
        out = orig(t)
        probes.append((t, eng._coverage_frac(), out,
                       tuple(a.id for a in eng.fleet.airborne()),
                       eng.fleet.agents[0].energy_consumed_j))
        return out

    eng._evaluate_terminal_no_swap = probing
    res = eng.run()
    _no_swap_invariants(res)

    assert res.outcome is Outcome.MISSION_SUCCESS
    assert res.terminal_reason == "coverage_complete_all_landed"
    assert res.coverage_frac >= _gate(cfg)
    gate_hits = [p for p in probes if p[1] >= _gate(cfg)]
    first = gate_hits[0]
    # gate first reached while still airborne -> engine kept running
    assert first[3] == (0,) and first[2] is None
    last = probes[-1]
    assert last[2] is Outcome.MISSION_SUCCESS and last[3] == ()
    # the return leg and touchdown were flown and paid for after the gate
    assert last[0] > first[0]
    assert last[4] > first[4]
    sj = _sojourns_of(res, 0)
    assert sj[-1].state is S.S_LANDED
    rth = [s for s in sj if s.state is S.S3_RTH]
    assert len(rth) == 1 and rth[0].t_out - rth[0].t_in > 0.0
    assert rth[0].reason_out == "landed"
    assert res.retired_agents == (0,) and res.work_releases == ()

    # legacy adapters handle the new state explicitly: ergodic via the
    # S_LANDED -> S0 closure, finite efficiency, S_LANDED excluded from overhead
    single = single_run_from_history(res.history, outcome=res.outcome)
    assert single.aborted is False and S.S_LANDED in single.pi_time
    assert single.efficiency > 0.0 and single.efficiency < float("inf")


# --------------------------------------------------------------------------- #
# 4. all landed below the gate -> PARTIAL                                     #
# --------------------------------------------------------------------------- #
def test_all_landed_below_gate_is_partial(config_path):
    cfg = _no_swap_cfg(config_path, n_drones=1)

    def low_battery(engine):
        # 6 % of 1000 Wh = 216 kJ against the real calculator's 5 % reserve
        # (180 kJ): the DYNAMIC route-vs-return decision (not a static zone
        # net -- those are disabled here) fires after a few strips.
        a = engine.fleet.agents[0]
        a.battery = _AlwaysHigh(a.battery.capacity_j, cfg.battery_zones, initial_frac=0.06)

    res = _run_with_post_build(_engine(cfg), low_battery)
    _no_swap_invariants(res)
    assert res.outcome is Outcome.MISSION_PARTIAL
    assert res.terminal_reason == "all_landed_below_gate"
    assert 0.0 <= res.coverage_frac < _gate(cfg)
    sj = _sojourns_of(res, 0)
    assert sj[-1].state is S.S_LANDED
    assert res.retired_agents == (0,)
    assert res.work_releases == ((0, sj[-1].t_in),)
    assert res.airborne_at_end == () and res.losses == ()


# --------------------------------------------------------------------------- #
# 5. airborne depletion -> FAILED, survivors still land, never SUCCESS        #
# --------------------------------------------------------------------------- #
def test_airborne_depletion_is_failed_never_success(config_path):
    cfg = _no_swap_cfg(config_path, n_drones=2)

    def doom_agent_0(engine):
        a = engine.fleet.agents[0]
        # 0.5 % of 1000 Wh = 18 kJ: well below the ~48 kJ its half of the area
        # costs, and no early return of any kind -> runs dry mid-strip.
        a.battery = _AlwaysHigh(a.battery.capacity_j, cfg.battery_zones, initial_frac=0.005)
        a.rth = _RthStub("CONTINUE")

    res = _run_with_post_build(_engine(cfg), doom_agent_0)

    assert res.outcome is Outcome.MISSION_FAILED
    assert res.terminal_reason == "uav_lost"
    assert len(res.losses) == 1
    aid, t_loss, cause = res.losses[0]
    assert (aid, cause) == (0, "battery_depleted")
    assert res.metrics.n_failures >= 1
    # drone 1 kept working and landed safely AFTER the loss; it is not lost
    sj1 = _sojourns_of(res, 1)
    assert sj1[-1].state is S.S_LANDED and sj1[-1].t_in > t_loss
    assert res.retired_agents == (1,)
    assert res.metrics.n_swaps == 0
    assert res.airborne_at_end == ()
    # coverage is still reported as its own metric, never turned into SUCCESS
    assert 0.0 <= res.coverage_frac <= 1.0


# --------------------------------------------------------------------------- #
# 6. time cap with airborne drones -> INCOMPLETE, no fictitious landing       #
# --------------------------------------------------------------------------- #
def test_time_cap_with_airborne_is_incomplete_not_partial(config_path):
    cfg = _no_swap_cfg(config_path, n_drones=1, max_timesteps=60)   # 30 s of flight
    res = _engine(cfg).run()
    assert res.outcome is Outcome.MISSION_INCOMPLETE
    assert res.terminal_reason == "max_timesteps"
    assert res.airborne_at_end == (0,)
    sj = _sojourns_of(res, 0)
    assert sj[-1].state.is_airborne and sj[-1].reason_out == "mission_end"
    assert all(s.state is not S.S_LANDED for s in sj)
    assert res.retired_agents == () and res.work_releases == ()
    assert res.aborted is True


# --------------------------------------------------------------------------- #
# 7. work the RTH calculator deems unreachable -> immediate return, PARTIAL   #
# --------------------------------------------------------------------------- #
def test_unreachable_work_lands_partial_without_livelock(config_path):
    cfg = _no_swap_cfg(config_path, n_drones=1)

    def unreachable(engine):
        engine.fleet.agents[0].rth = _RthStub("RETURN_NOW")

    res = _run_with_post_build(_engine(cfg), unreachable)
    _no_swap_invariants(res)
    assert res.outcome is Outcome.MISSION_PARTIAL
    assert res.terminal_reason == "all_landed_below_gate"
    assert res.coverage_frac < _gate(cfg)
    sj = _sojourns_of(res, 0)
    assert sj[-1].state is S.S_LANDED
    rth = [s for s in sj if s.state is S.S3_RTH]
    assert len(rth) == 1 and rth[0].reason_out == "landed"
    assert [s.state for s in sj].count(S.S1_TRANSIT) == 1     # no relaunch attempt
    assert res.work_releases == ((0, sj[-1].t_in),)
    # terminates well inside the cap: no livelock waiting for a swap
    assert res.metrics.duration_s < cfg.sim.max_timesteps * cfg.sim.dt_s / 10


# --------------------------------------------------------------------------- #
# 8. flag-off byte-identity: cross-commit golden captured on a7ee5d7          #
# --------------------------------------------------------------------------- #
def _canonical_sojourn(s) -> str:
    return f"{s.agent_id}|{s.state.value}|{s.t_in!r}|{s.t_out!r}|{s.reason_out}"


def _canonical_signature_string(res) -> str:
    m = res.metrics
    parts = [
        f"total_energy_j={m.total_energy_j!r}",
        f"duration_s={m.duration_s!r}",
        f"workload_std_m={m.workload_std_m!r}",
        f"n_swaps={m.n_swaps!r}",
        "per_agent_energy_j=" + ",".join(f"{k}:{v!r}" for k, v in sorted(m.per_agent_energy_j.items())),
        "per_agent_length_m=" + ",".join(f"{k}:{v!r}" for k, v in sorted(m.per_agent_length_m.items())),
        "sojourns=" + ";".join(_canonical_sojourn(s) for s in res.history.sojourns()),
        f"outcome={res.outcome.value}",
        f"coverage_frac={res.coverage_frac!r}",
        f"stalled_agents={getattr(res, 'stalled_agents', ())!r}",
        f"skipped_legs={getattr(res, 'skipped_legs', ())!r}",
    ]
    return "\n".join(parts)


# Captured on the unmodified base commit a7ee5d7 (pre-EXP-04 binary) with the
# scratch script equivalent of _legacy_swap_cfg + _canonical_signature_string.
# 16 Wh forces four S3 -> S_SWAP -> S0 -> S1 cycles, so the golden exercises
# exactly the path EXP-04 gates. Regenerate ONLY from a pre-EXP-04 commit.
_LEGACY_GOLDEN_SHA256 = "cc0950ecc75caff13b61ec9ebd49a5e42a5c6db15d1661f0f8b39e0513b36dc4"
_LEGACY_GOLDEN_SUMMARY = dict(
    total_energy_j=180590.0, duration_s=615.0, n_swaps=4,
    per_agent_energy_j={0: 89164.0, 1: 91426.0},
    per_agent_length_m={0: 2889.1274016844395, 1: 3230.8475302435286},
    workload_std_m=170.86006427954453,
)


def _legacy_swap_cfg(config_path):
    return load_config(config_path, overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 16.0,
        "fleet.total_reserve_batteries": 50,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })


def test_flag_off_swap_cycle_byte_identical_to_pre_exp04(config_path):
    cfg = _legacy_swap_cfg(config_path)
    assert cfg.mission.no_swap_mode is False
    res = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0,
                           algo=DecompositionAlgo.TGC_BASIC,
                           planner=PlannerKind.DUBINS).run()
    m = res.metrics
    assert m.n_swaps == _LEGACY_GOLDEN_SUMMARY["n_swaps"]
    assert m.total_energy_j == _LEGACY_GOLDEN_SUMMARY["total_energy_j"]
    assert m.duration_s == _LEGACY_GOLDEN_SUMMARY["duration_s"]
    assert dict(m.per_agent_energy_j) == _LEGACY_GOLDEN_SUMMARY["per_agent_energy_j"]
    assert dict(m.per_agent_length_m) == pytest.approx(_LEGACY_GOLDEN_SUMMARY["per_agent_length_m"])
    assert m.workload_std_m == pytest.approx(_LEGACY_GOLDEN_SUMMARY["workload_std_m"])
    assert res.outcome is Outcome.MISSION_SUCCESS
    assert hashlib.sha256(_canonical_signature_string(res).encode()).hexdigest() \
        == _LEGACY_GOLDEN_SHA256
    # the new surface is inert with the flag off
    sj = res.history.sojourns()
    assert all(s.state is not S.S_LANDED for s in sj)
    assert any(s.state is S.S_SWAP for s in sj)
    assert res.retired_agents == () and res.work_releases == () and res.losses == ()
    assert res.terminal_reason == "coverage_complete"
    assert res.airborne_at_end == ()
    single = single_run_from_history(res.history)
    assert S.S_LANDED not in single.states and single.aborted is False
