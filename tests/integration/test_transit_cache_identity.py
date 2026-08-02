"""E3 full-mission byte-identity gate: the visibility-graph cache is bitwise
identical to the uncached build on the obstacle-dense FIX-B1 replication.

A CROSS-COMMIT golden (mirrors test_energy_map_zone_demotion / _stage4): the run
signature of study01_demand replication 1 (transit_free_space ON, unbounded pool)
is pinned from the PRE-E3 base commit 0ed3dad -- which had NO cache, so the pin is
the uncached behaviour -- and the post-E3 CACHED run must reproduce it byte-for-
byte. This is the "WITH cache == WITHOUT cache" gate on the mandated dense fixture;
because the cache is exercised heavily here (V=1349 obstacle vertices, ~10 routed
transit legs), a single hash mismatch IS the regression this gate exists to catch:
STOP and diff, do not repin. The forced a/b-on-vertex collision -- which this
mission may never hit -- is covered fast in
tests/unit/planning/test_visibility_router_cache.py.

The signature machinery is REPLICATED (not imported) so a refactor elsewhere
cannot silently move this gate's goalposts.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import Outcome, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine

CONFIG = "config/study01_demand.yaml"
REPLICATION = 1

# Pinned on PRE-E3 base commit 0ed3dad (no cache): study01_demand rep 1,
# transit_free_space ON, unbounded pool, DUBINS -> MISSION_SUCCESS, n_swaps=8,
# duration_s=2823.0, coverage~1.0. If this ever mismatches, that IS the regression.
_GOLDEN_HASH = "87c55427e348a6afc40c7942321ca0a211f2114a18fa906e654390f8c1835a51"


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


def _golden_hash(res) -> str:
    return hashlib.sha256(_canonical_signature_string(res).encode()).hexdigest()


@pytest.mark.slow
def test_dense_mission_cache_byte_identical_to_pre_e3():
    """CORE E3 GATE: the cached run reproduces the pre-E3 (uncached, 0ed3dad) run
    bit-for-bit on the dense fixture. The cache must have been exercised, or the
    gate proves nothing about caching."""
    cfg = _dense_cfg()
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                           replication=REPLICATION, planner=PlannerKind.DUBINS)
    res = eng.run()
    assert eng._transit_graph_cache, "no ok_pairs built -> route_transit never routed"
    assert res.outcome is Outcome.MISSION_SUCCESS  # sanity: the FIX-B1 case succeeds
    assert _golden_hash(res) == _GOLDEN_HASH


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
