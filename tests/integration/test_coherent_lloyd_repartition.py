"""REV-02: coherent flight and event-driven Lloyd repartitioning together."""
from __future__ import annotations

import json

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Event
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo, EventType
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.run_output import build_mission_contract


def _settings(area, algo):
    return {
        "fleet.n_drones": 2,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "launch.candidate_sites": [[0, 0], [0, 160]],
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "platforms.MULTIROTOR.v_cruise": 10.0,
        "sensor.photogrammetry.enabled": True, "sensor.sensor_power_w": 20.0,
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 10.0,
        "coverage.transit_free_space": True, "coverage.ferry_free_space": True,
        "mission.no_swap_mode": True, "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 10.0,
        "rth.execution_coherent": True, "rth.emergency_frac": 0.05,
        "rth.energy_map.zone_demotion": True,
        "planning.energy_balance.enabled": algo is DecompositionAlgo.LLOYD_ENERGY,
        "sim.dt_s": 0.5, "sim.max_timesteps": 3000,
    }


@pytest.fixture
def area(tmp_path):
    path = tmp_path / "area.geojson"
    path.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(box(0, 0, 600, 160))}))
    return path


def _engine(area, algo):
    cfg = load_config("config/djimatrice4e.yaml", overrides=_settings(area, algo))
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), algo=algo)


@pytest.mark.parametrize("algo", [DecompositionAlgo.LLOYD_CVT, DecompositionAlgo.LLOYD_ENERGY])
def test_coherent_revision_preserves_live_aircraft_and_exports_algorithm(area, algo, monkeypatch):
    """A live revision is physical continuity, not a replacement Agent object."""
    observed = []
    original = Agent.commit_retask

    def commit_and_observe(agent, prepared, t, bus):
        before = (agent.pose, agent._coherent.altitude_m, agent.battery.level_j,
                  agent.energy_consumed_j, agent.flown_m, tuple(agent.photo_events))
        original(agent, prepared, t, bus)
        after = (agent.pose, agent._coherent.altitude_m, agent.battery.level_j,
                 agent.energy_consumed_j, agent.flown_m, tuple(agent.photo_events))
        observed.append((before, after, agent.plan_revision))

    monkeypatch.setattr(Agent, "commit_retask", commit_and_observe)
    engine = _engine(area, algo)
    result = engine.run()

    applied = [r for r in result.repartitions if r["applied"]]
    assert applied and observed
    assert all(before == after and revision > 0 for before, after, revision in observed)
    assert {r["algorithm"] for r in result.repartitions} == {algo.value}
    assert all("WeightedTgc" not in r["decomposer_class"] for r in result.repartitions)

    contract = build_mission_contract(
        result, capacity_j=engine.spec.battery_capacity_j,
        decomposer_class=type(engine.decomposer).__name__,
    )
    assert contract["repartitions"] == list(result.repartitions)
    assert all({"t_s", "causes", "revision", "algorithm"} <= set(r)
               for r in contract["repartitions"])
    json.dumps(contract, allow_nan=False)


@pytest.mark.parametrize("algo", [DecompositionAlgo.LLOYD_CVT, DecompositionAlgo.LLOYD_ENERGY])
def test_landed_rth_and_failed_agents_are_excluded_without_algorithm_fallback(area, algo):
    engine = _engine(area, algo)
    engine._build()
    agents = engine.fleet.agents
    agents[0].state, agents[0]._retired = AgentState.S_LANDED, True
    agents[1].state = AgentState.S3_RTH
    engine._run_repartition(1.0, (("uav_retired", 0), ("failure", 1)))

    record = engine._repartition_records[-1]
    assert record.applied is False
    assert record.reason == "no_eligible_executor"
    assert record.excluded["retired"] == (0,)
    assert record.excluded["rth_committed"] == (1,)
    assert record.cells_unassigned == record.cells_before > 0
    assert record.algorithm == algo.value
    assert "WeightedTgc" not in record.decomposer_class


@pytest.mark.parametrize("algo", [DecompositionAlgo.LLOYD_CVT, DecompositionAlgo.LLOYD_ENERGY])
def test_failure_repartitions_to_survivor_with_the_selected_lloyd_method(area, algo):
    engine = _engine(area, algo)
    engine._build()
    engine.fleet.agents[0].state = AgentState.S_FAIL
    engine._run_repartition(1.0, (("failure", 0),))

    record = engine._repartition_records[-1]
    assert record.applied is True
    assert record.excluded["failed"] == (0,)
    assert record.executors == (1,)
    assert record.algorithm == algo.value
    assert "WeightedTgc" not in record.decomposer_class


def test_infeasible_candidate_is_rejected_before_any_coherent_agent_changes(area):
    engine = _engine(area, DecompositionAlgo.LLOYD_CVT)
    engine._build()
    for agent in engine.fleet.agents.values():
        agent.battery._level = 1.0
    before = {aid: (a.plan, a.plan_revision, a.pose, a.state, tuple(a._legs),
                    a.battery.level_j, a.energy_consumed_j)
              for aid, a in engine.fleet.agents.items()}

    engine._run_repartition(1.0, (("interval", None),))

    record = engine._repartition_records[-1]
    assert record.applied is False
    assert record.reason == "candidate_rejected"
    assert record.rejection == "retask_energy_budget"
    after = {aid: (a.plan, a.plan_revision, a.pose, a.state, tuple(a._legs),
                   a.battery.level_j, a.energy_consumed_j)
             for aid, a in engine.fleet.agents.items()}
    assert after == before


def test_subtick_zone_complete_uses_physical_completion_time_for_revision(area):
    engine = _engine(area, DecompositionAlgo.LLOYD_CVT)
    engine._build()
    observed = []
    engine._run_repartition = lambda t, causes: observed.append((t, causes))
    engine.bus.publish(Event(EventType.ZONE_COMPLETE, 7.25, {"agent_id": 0}))
    engine._route_events(7.0)
    engine._drain_repartition(7.0, 1)
    assert observed == [(7.25, (("zone_complete", 0),))]
