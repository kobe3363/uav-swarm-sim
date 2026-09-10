"""Executable EXP-09 acceptance scenarios, not a universal flight guarantee."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.infrastructure.config import ConfigError, load_config
from uav_swarm_sim.infrastructure.enums import AgentState as S, DecompositionAlgo, Outcome
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


def overrides(area):
    return {
        "fleet.n_drones": 1, "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        # EXP-13 leak shield: the base M4E config now sets obstacle_generation_mode=target
        # (density ignored) and planning.energy_balance.enabled=true. This test's scenarios
        # (and the cross-commit legacy golden below) assume a clean, obstacle-free area with
        # the EXP-06 proxy off, so pin both back to the pre-change world here.
        #   was  -> inherited target mode + energy_balance on
        #   now  -> poisson (0 obstacles at density 0.0) + energy_balance off
        #   why  -> this file tests coherent-flight energy, not obstacle generation or the
        #           t=0 proxy; the one test that needs the proxy overrides it True explicitly.
        "env.obstacle_generation_mode": "poisson",
        "planning.energy_balance.enabled": False,
        "env.coverage_altitude_m": 100.0, "launch.candidate_sites": [[0, 0]],
        "platforms.MULTIROTOR.v_coverage": 10.0, "platforms.MULTIROTOR.v_cruise": 10.0,
        "sensor.photogrammetry.enabled": True,
        "sensor.sensor_power_w": 20.0,  # ASSUMPTION: synthetic test payload draw
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 5.0,
        "coverage.transit_free_space": True, "coverage.ferry_free_space": True,
        "mission.no_swap_mode": True, "rth.execution_coherent": True,
        "rth.emergency_frac": 0.05, "rth.energy_map.zone_demotion": True,
        "sim.dt_s": 0.5, "sim.max_timesteps": 3000,
    }


@pytest.fixture
def mission(tmp_path):
    area = tmp_path / "area.geojson"
    area.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(box(0, 0, 300, 120))}))
    return overrides(area)


def engine(settings, algo=DecompositionAlgo.TGC_BASIC):
    cfg = load_config("config/djimatrice4e.yaml", overrides=settings)
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), algo=algo)


@pytest.mark.parametrize("map_on", [False, True])
def test_feasible_mission_pays_takeoff_and_landing_without_depletion(mission, map_on):
    settings = dict(mission, **{"rth.energy_map.enabled": map_on,
                               "rth.energy_map.route": map_on, "rth.energy_map.decide": map_on})
    eng = engine(settings)
    result = eng.run()
    a = eng.fleet.agents[0]
    assert result.outcome is Outcome.MISSION_SUCCESS
    assert a.state is S.S_LANDED
    assert a._coherent.altitude_m == 0
    assert a.battery.level_j >= a.rth.reserve_j - 1e-6
    assert not result.losses
    assert result.photo_events
    assert result.metrics.n_swaps == 0
    assert not a.rth_infeasible_events


def test_rejected_prelaunch_plan_settles_as_normal_partial(mission, monkeypatch):
    from uav_swarm_sim.execution.coherent_flight import CoherentFlight
    original = CoherentFlight.can_launch
    def low_energy_before_launch(flight):
        flight.a.battery._level = 1000
        return original(flight)
    monkeypatch.setattr(CoherentFlight, "can_launch", low_energy_before_launch)
    eng = engine(mission)
    result = eng.run()
    a = eng.fleet.agents[0]
    assert a.state is S.S0_IDLE
    assert not a._launch_ready and a.plan is None
    assert eng._fleet_settled()
    assert result.outcome is Outcome.MISSION_PARTIAL
    assert result.coverage_frac == 0
    assert not result.airborne_at_end


def physical_signature(result):
    """Deterministic physical output; omit wall timing and config provenance hash."""
    m = result.metrics
    value = dict(outcome=result.outcome.value, coverage=result.coverage_frac,
                 energy=m.total_energy_j, duration=m.duration_s, swaps=m.n_swaps,
                 per_agent_energy=m.per_agent_energy_j, per_agent_length=m.per_agent_length_m,
                 retired=result.retired_agents, losses=result.losses,
                 sojourns=[asdict(s) for s in result.history.sojourns()],
                 photos=[asdict(p) for p in result.photo_events])
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda v: v.value)


def test_execution_coherent_must_be_a_real_boolean(mission):
    """A quoted YAML "false" is truthy in Python; the mission mode flags all
    reject non-booleans, and this one must not be the exception that silently
    switches the executor on."""
    with pytest.raises(ConfigError, match=r"rth\.execution_coherent must be a boolean"):
        engine(dict(mission, **{"rth.execution_coherent": "false"}))


@pytest.mark.parametrize("algo", [DecompositionAlgo.LLOYD_CVT, DecompositionAlgo.LLOYD_ENERGY])
def test_coherent_execution_repartitions_with_each_lloyd_method(mission, algo):
    """The formerly refused flag combination must execute a real revision.

    A short fixed trigger makes this independent of which drone completes its
    initial zone first.  It proves more than config acceptance: the resulting
    record exposes the chosen Lloyd implementation rather than a TGC fallback.
    """
    settings = dict(mission, **{
        "fleet.n_drones": 2,
        "launch.candidate_sites": [[0, 0], [0, 120]],
        "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 10.0,
        "planning.energy_balance.enabled": algo is DecompositionAlgo.LLOYD_ENERGY,
    })
    eng = engine(settings, algo=algo)
    result = eng.run()
    assert result.repartitions
    assert any(record["applied"] for record in result.repartitions)
    assert {record["algorithm"] for record in result.repartitions} == {algo.value}
    assert all("WeightedTgc" not in record["decomposer_class"]
               for record in result.repartitions)


def test_flag_off_byte_identity(mission):
    # EXP-13: the base M4E config now defaults execution_coherent/zone_demotion ON,
    # so reach the pre-EXP-09 legacy world by pinning them OFF explicitly.
    #   was  -> pop the keys, inheriting the pre-change base default (False)
    #   now  -> pin execution_coherent=False, zone_demotion=False, emergency_frac=None
    #   why  -> the golden is the legacy (flag-off) physics; popping would now inherit
    #           the protocol's ON values and reach the coherent path instead. The
    #           "omit == default False" property is a loader property tested elsewhere.
    # (The shielded overrides already pin obstacle poisson + energy_balance off, so the
    #  resolved inputs match the golden's exactly.)
    legacy = dict(mission, **{
        "rth.execution_coherent": False,
        "rth.energy_map.zone_demotion": False,
        "rth.emergency_frac": None,
    })
    absent = physical_signature(engine(legacy).run())
    explicit = physical_signature(engine(dict(legacy, **{
        "rth.execution_coherent": False, "rth.emergency_frac": None,
    })).run())
    assert explicit == absent
    golden = Path("tests/fixtures/exp09_legacy_7eca307.json").read_text().strip()
    assert absent == golden


def test_geometrically_rejected_launch_settles_without_deadlock(mission, monkeypatch):
    from uav_swarm_sim.planning.visibility_router import RouteUnavailable
    def blocked(*args):
        raise RouteUnavailable("transit_blocked")
    monkeypatch.setattr(SimulationEngine, "_plan_transit", blocked)
    eng = engine(mission)
    result = eng.run()
    assert eng._fleet_settled()
    assert result.outcome is Outcome.MISSION_PARTIAL
    assert result.coverage_frac == 0
    assert eng.fleet.agents[0].state is S.S0_IDLE


def test_unavailable_exp06_proxy_does_not_abort_valid_flight(mission, monkeypatch, caplog):
    import uav_swarm_sim.planning.energy_balance as eb
    from uav_swarm_sim.planning.visibility_router import RouteUnavailable

    def blocked_proxy(*args):
        raise RouteUnavailable("proxy_anchor_blocked")

    monkeypatch.setattr(eb, "estimate_fast", blocked_proxy)
    eng = engine(dict(mission, **{"planning.energy_balance.enabled": True}))
    result = eng.run()
    assert result.outcome is Outcome.MISSION_SUCCESS
    assert set(result.energy_balance_t0[0]) == {"path"}
    assert "energy_balance_unavailable drone_id=0 method=fast reason=proxy_anchor_blocked" in caplog.text
    assert not result.rth_infeasible_events  # proxy failure is not physical RTH infeasibility


def test_one_infeasible_drone_does_not_stop_survivor(mission, monkeypatch):
    from uav_swarm_sim.execution.agent import Agent
    from uav_swarm_sim.execution.state_machine import Transition
    from uav_swarm_sim.infrastructure.core_types import Pose
    original = Agent.step
    injected = False
    def flight_fault(a, dt, t, bus):
        nonlocal injected
        if a.id == 0 and t >= 20 and a.state.is_airborne and not injected:
            injected = True
            # Deliberate unreachable-pose fault after takeoff, beyond the
            # operating domain. Tests falsifiability and the existing lifecycle.
            a.pose = Pose(5000, 5000, 0)
            a.battery._level = 100
            a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), t, bus)
        original(a, dt, t, bus)
    monkeypatch.setattr(Agent, "step", flight_fault)
    eng = engine(dict(mission, **{"fleet.n_drones": 2}))
    result = eng.run()
    assert injected
    assert result.outcome is Outcome.MISSION_FAILED
    assert eng.fleet.agents[0].state is S.S_FAIL
    assert eng.fleet.agents[1].state is S.S_LANDED
    assert result.metrics.duration_s > 20
    assert result.coverage_frac > 0
    assert result.rth_infeasible_events
    assert result.rth_infeasible_events[0].payload["agent_id"] == 0
    assert result.rth_infeasible_events[0].payload["deficit_j"] is None
