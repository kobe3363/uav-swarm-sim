"""E3 full-mission byte-identity gate: the visibility-graph cache is bitwise
identical to the uncached build on the obstacle-dense FIX-B1 replication.

Uses an IN-PROCESS A/B rather than a pinned cross-commit hash: unlike the
energy-map flag gates (where flag-OFF does nothing, so an in-process compare would
be a self-comparison), the E3 ``graph_cache=None`` branch is the LITERAL pre-E3
code path -- rebuilding the visibility graph from scratch -- so running the dense
mission WITH the cache vs WITHOUT it (forcing ``graph_cache=None`` through the
module-level ``route_transit`` symbol the engine closure resolves) is a genuine
two-path comparison. It is also PLATFORM-INDEPENDENT: both arms run identical
physics on the same box, so the gate holds on any OS / shapely-GEOS version,
whereas a signature hash pinned on one platform can drift on another. The fast
seam unit tests (tests/unit/planning/test_visibility_router_cache.py) -- including
the forced a/b-on-vertex collision -- are the primitive-level guard.
"""
from __future__ import annotations

import dataclasses

import pytest

import uav_swarm_sim.infrastructure.simulation_engine as se
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import Outcome, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine

CONFIG = "config/study01_demand.yaml"
REPLICATION = 1


def _dense_cfg():
    """The obstacle-dense FIX-B1 replication (study01_demand rep 1), transit
    routing ON, unbounded pool -- the same pathological case as
    test_transit_livelock, where route_transit is exercised heavily."""
    cfg = load_config(CONFIG)
    return dataclasses.replace(
        cfg,
        fleet=dataclasses.replace(cfg.fleet, total_reserve_batteries=None),
        coverage=dataclasses.replace(cfg.coverage, transit_free_space=True),
        safety=dataclasses.replace(cfg.safety, stall_detector=True),
    )


def _engine(cfg):
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                            replication=REPLICATION, planner=PlannerKind.DUBINS)


def _metrics_tuple(m):
    return (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
            dict(m.per_agent_energy_j), dict(m.per_agent_length_m))


def _run_signature(res) -> tuple:
    return (
        _metrics_tuple(res.metrics),
        tuple(res.history.sojourns()),
        res.outcome,
        res.coverage_frac,
        getattr(res, "stalled_agents", ()),
        getattr(res, "skipped_legs", ()),
    )


@pytest.mark.slow
def test_dense_mission_cache_byte_identical(monkeypatch):
    """CORE E3 GATE: on the dense fixture, the cached run reproduces the uncached
    run bit-for-bit -- full metrics tuple, complete FSM sojourn trajectory,
    outcome, coverage, the stalled set and skipped legs. Also confirms the cache
    was actually exercised (blocked chords occurred)."""
    cfg = _dense_cfg()

    # uncached reference: force graph_cache=None through the symbol the engine
    # closure resolves at call time (simulation_engine.route_transit).
    real_route_transit = se.route_transit

    def _uncached(*args, **kwargs):
        kwargs["graph_cache"] = None
        return real_route_transit(*args, **kwargs)

    monkeypatch.setattr(se, "route_transit", _uncached)
    res_uncached = _engine(cfg).run()
    monkeypatch.undo()

    eng = _engine(cfg)
    res_cached = eng.run()

    assert eng._transit_graph_cache, "no ok_pairs built -> route_transit never routed"
    assert res_cached.outcome is Outcome.MISSION_SUCCESS  # sanity: FIX-B1 case succeeds
    assert _run_signature(res_cached) == _run_signature(res_uncached)


@pytest.mark.slow
def test_transit_cache_per_replication_isolated():
    """Scope: each engine gets its OWN cache dict from _build (no module-level
    state), so a cache built in one replication cannot bleed into another. Built
    on a tiny obstacle config for speed; the dict-identity property is what
    matters, independent of mission scale."""
    cfg = load_config(CONFIG, overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 20.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })
    cfg = dataclasses.replace(cfg, coverage=dataclasses.replace(
        cfg.coverage, transit_free_space=True))

    e1 = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=1,
                          planner=PlannerKind.DUBINS)
    e2 = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=2,
                          planner=PlannerKind.DUBINS)
    e1._build()
    e2._build()
    assert e1._transit_graph_cache is not e2._transit_graph_cache
    e1._transit_graph_cache["sentinel"] = object()
    assert "sentinel" not in e2._transit_graph_cache  # no cross-replication bleed
