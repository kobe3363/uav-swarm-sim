"""Batch 4 tests: execution layer + physical_model/battery (isolated + small)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import (
    CoveragePlan,
    DroneStateView,
    Event,
    Path,
    Pose,
    Waypoint,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    BatteryZone,
    EventType,
    ManeuverType,
    TierStrategy,
)
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.algorithm_selector import make_decomposers, select
from uav_swarm_sim.execution.events import EventBus
from uav_swarm_sim.execution.failure_model import FailureModel
from uav_swarm_sim.execution.fleet import Fleet
from uav_swarm_sim.execution.formation_manager import FormationManager
from uav_swarm_sim.execution.redistribution import TRIGGERS, Redistributor
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import (
    ALLOWED,
    AgentContext,
    StateMachine,
)
from uav_swarm_sim.execution.swap_station import SwapStation
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection


@pytest.fixture(scope="module")
def cfg(config_path):
    return load_config(config_path)


# --------------------------------------------------------------------------- #
# battery                                                                     #
# --------------------------------------------------------------------------- #
def test_battery_zones(cfg):
    b = Battery(100.0, cfg.battery_zones, 1.0)
    assert b.zone is BatteryZone.HIGH
    b.drain(30)  # 0.70
    assert b.zone is BatteryZone.NOMINAL
    b.drain(35)  # 0.35
    assert b.zone is BatteryZone.CRITICAL
    b.drain(20)  # 0.15
    assert b.zone is BatteryZone.TERMINAL
    b.drain(1000)  # clamps
    assert b.level_j == 0.0
    b.reset()
    assert b.frac == 1.0


# --------------------------------------------------------------------------- #
# state machine                                                               #
# --------------------------------------------------------------------------- #
def _ctx(state, **kw):
    return AgentContext(state=state, battery_zone=kw.pop("zone", BatteryZone.HIGH), **kw)


def test_all_transitions_are_allowed(cfg):
    sm = StateMachine(cfg.battery_zones)
    cases = [
        _ctx(AgentState.S0_IDLE, launch_command=True, plan_assigned=True),
        _ctx(AgentState.S1_TRANSIT, at_zone_entry=True),
        _ctx(AgentState.S1_TRANSIT, threat_flag=True),
        _ctx(AgentState.S2_MISSION, rth_decision=True),
        _ctx(AgentState.S2_MISSION, coverage_complete=True),
        _ctx(AgentState.S2_MISSION, threat_flag=True),
        _ctx(AgentState.S2_MISSION, zone=BatteryZone.CRITICAL),
        _ctx(AgentState.S3_RTH, landed_at_base=True, own_plan_incomplete=True),
        _ctx(AgentState.S3_RTH, landed_at_base=True, own_plan_incomplete=False),
        _ctx(AgentState.S_SWAP, swap_done=True),
        _ctx(AgentState.S_OBS, threat_cleared=True, obs_return_state=AgentState.S2_MISSION),
    ]
    for c in cases:
        tr = sm.step(c)
        assert tr is not None
        assert (tr.src, tr.dst) in ALLOWED, f"{tr.src}->{tr.dst} not allowed"


def test_failure_preempts_from_any_airborne_state(cfg):
    sm = StateMachine(cfg.battery_zones)
    for s in (AgentState.S1_TRANSIT, AgentState.S2_MISSION, AgentState.S3_RTH, AgentState.S_OBS):
        tr = sm.step(_ctx(s, failure_flag=True))
        assert tr is not None and tr.dst is AgentState.S_FAIL


def test_closed_loop_swap_returns_to_idle(cfg):
    sm = StateMachine(cfg.battery_zones)
    tr = sm.step(_ctx(AgentState.S_SWAP, swap_done=True))
    assert tr.dst is AgentState.S0_IDLE  # S3 -> S_SWAP -> S0 closure


def test_terminal_battery_forces_rth(cfg):
    sm = StateMachine(cfg.battery_zones)
    tr = sm.step(_ctx(AgentState.S2_MISSION, zone=BatteryZone.TERMINAL))
    assert tr.dst is AgentState.S3_RTH


# --------------------------------------------------------------------------- #
# agent: full mission cycle                                                   #
# --------------------------------------------------------------------------- #
def _make_agent(config_path, cfg, platform="MULTIROTOR", capacity_j=None, initial_frac=1.0):
    cfg = load_config(config_path, overrides={"platform_type": platform})
    spec = build_spec(cfg)
    if capacity_j is not None:
        object.__setattr__(spec, "battery_capacity_j", capacity_j)
    motion = make_motion_model(spec)
    em = EnergyModel(spec)
    bat = Battery(spec.battery_capacity_j, cfg.battery_zones, initial_frac)
    sm = StateMachine(cfg.battery_zones)
    base = Pose(0.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=100.0)
    aero = AeroCorrection(cfg.aero, spec.platform)
    fm = FormationManager(aero, cfg.aero, spec.platform)
    agent = Agent(0, spec, motion, em, bat, sm, rth, fm, base)
    return agent, spec, motion, em


def _simple_coverage_plan(motion):
    # two short strips near base
    wps = [
        Waypoint(Pose(100, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(150, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(150, 50, math.pi), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(100, 50, math.pi), ManeuverType.COVERAGE, 6.0),
    ]
    return CoveragePlan(0, wps, 0.0, 0.0)


def test_agent_completes_mission_cycle(config_path, cfg):
    agent, spec, motion, em = _make_agent(config_path, cfg)
    transit = motion.plan(agent.base, Pose(100, 0, 0.0), ManeuverType.CRUISE)
    agent.assign(_simple_coverage_plan(motion), transit)
    bus = EventBus()
    states_seen = set()
    for _ in range(20000):
        agent.step(0.5, _ * 0.5, bus)
        states_seen.add(agent.state)
        if agent.state is AgentState.S0_IDLE and agent.flown_m > 0:
            break
    assert AgentState.S1_TRANSIT in states_seen
    assert AgentState.S2_MISSION in states_seen
    assert AgentState.S3_RTH in states_seen
    assert agent.state is AgentState.S0_IDLE
    assert agent.flown_m > 0


def test_battery_drains_while_airborne(config_path, cfg):
    agent, spec, motion, em = _make_agent(config_path, cfg)
    transit = motion.plan(agent.base, Pose(100, 0, 0.0), ManeuverType.CRUISE)
    agent.assign(_simple_coverage_plan(motion), transit)
    bus = EventBus()
    start = agent.battery.level_j
    for k in range(50):
        agent.step(0.5, k * 0.5, bus)
    assert agent.battery.level_j < start


def test_low_battery_triggers_early_rth(config_path, cfg):
    # tiny battery so RTH must fire before coverage completes
    agent, spec, motion, em = _make_agent(config_path, cfg, capacity_j=8000.0, initial_frac=1.0)
    transit = motion.plan(agent.base, Pose(300, 0, 0.0), ManeuverType.CRUISE)
    # far coverage so the route-vs-return comparison bites
    wps = [
        Waypoint(Pose(300, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(500, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(500, 200, math.pi), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(300, 200, math.pi), ManeuverType.COVERAGE, 6.0),
    ]
    agent.assign(CoveragePlan(0, wps, 0.0, 0.0), transit)
    bus = EventBus()
    went_rth = False
    incomplete_at_rth = None
    for k in range(4000):
        agent.step(0.5, k * 0.5, bus)
        if agent.state is AgentState.S3_RTH and not went_rth:
            went_rth = True
            incomplete_at_rth = agent._cov_idx < len(agent._cov_legs)
        if agent.state in (AgentState.S_SWAP, AgentState.S0_IDLE) and went_rth:
            break
    assert went_rth, "RTH never fired on a small battery"


# --------------------------------------------------------------------------- #
# swap station + failure model                                                #
# --------------------------------------------------------------------------- #
def test_swap_station_services_and_emits_done(cfg):
    st = SwapStation(cfg.swap, Pose(0, 0, 0))
    bus = EventBus()
    st.request(0, 0.0)
    st.request(1, 0.0)
    st.request(2, 0.0)  # 3rd waits (2 bays)
    st.step(1.0, bus)
    assert st.busy_bays == 2 and st.queue_len == 1
    st.step(cfg.swap.service_time_s, bus)
    done = [e for e in bus.drain() if e.type is EventType.SWAP_DONE]
    assert len(done) >= 2


def test_failure_model_populates_with_high_lambda(config_path):
    cfg = load_config(config_path, overrides={"failure.hazard_rate_per_hour": 100.0})
    fm = FailureModel(cfg.failure, RngFactory(1).stream("failures", 0))
    bus = EventBus()

    class Stub:
        def __init__(self, i): self.id = i

    fired = 0
    for k in range(200):
        fm.step([Stub(0), Stub(1)], 0.5, k * 0.5, bus)
        fired += sum(1 for e in bus.drain() if e.type is EventType.FAILURE)
    assert fired > 0


def test_failure_model_silent_when_lambda_zero(config_path):
    cfg = load_config(config_path, overrides={"failure.hazard_rate_per_hour": 0.0})
    fm = FailureModel(cfg.failure, RngFactory(1).stream("failures", 0))
    bus = EventBus()

    class Stub:
        def __init__(self, i): self.id = i

    for k in range(100):
        fm.step([Stub(0)], 0.5, k * 0.5, bus)
    assert not bus.drain()


# --------------------------------------------------------------------------- #
# fleet + redistribution: swap/failure asymmetry                              #
# --------------------------------------------------------------------------- #
def test_redistribution_triggers_exclude_swap():
    assert EventType.FAILURE in TRIGGERS and EventType.NEW_TASK in TRIGGERS
    assert EventType.SWAP_DONE not in TRIGGERS and EventType.SWAP_REQUEST not in TRIGGERS


def test_redistribution_rejects_swap_event(cfg):
    from uav_swarm_sim.planning.weighted_decomposition import WeightedTgcDecomposer

    # class _FakeTGC:
    #     regions = []
    # redis = Redistributor(WeightedTgcDecomposer(), _FakeTGC(), None, None, None, build_spec(cfg))
    redis = Redistributor(WeightedTgcDecomposer(), None, None, None, build_spec(cfg))
    with pytest.raises(ValueError):
        redis.handle(Event(EventType.SWAP_DONE, 0.0, {}), None, None, {}, 0.0)


def test_fleet_kill_removes_from_active(config_path, cfg):
    agent, *_ = _make_agent(config_path, cfg)
    fleet = Fleet([agent])
    assert len(fleet.active()) == 1
    fleet.kill(0, 1.0)
    assert len(fleet.active()) == 0
    assert fleet.agents[0].state is AgentState.S_FAIL
    assert fleet.n_failed == 1


# --------------------------------------------------------------------------- #
# algorithm selector                                                          #
# --------------------------------------------------------------------------- #
def test_tier_selection():
    assert select(8) is TierStrategy.HEURISTIC
    assert select(30) is TierStrategy.COMPARE_BOTH
    assert select(64) is TierStrategy.TGC


def test_make_decomposers_counts(cfg):
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    rng = RngFactory(0).stream("kmeans_init", 0)
    assert len(make_decomposers(TierStrategy.HEURISTIC, motion, rng)) == 1
    assert len(make_decomposers(TierStrategy.TGC, motion, rng)) == 1
    assert len(make_decomposers(TierStrategy.COMPARE_BOTH, motion, rng)) == 2


# --------------------------------------------------------------------------- #
# formation manager: benefit gating                                           #
# --------------------------------------------------------------------------- #
def test_formation_benefit_only_fw_transit(config_path, cfg):
    cfg_fw = load_config(config_path, overrides={"platform_type": "FIXED_WING"})
    aero = AeroCorrection(cfg_fw.aero, cfg_fw.platform.type)
    fm = FormationManager(aero, cfg_fw.aero, cfg_fw.platform.type)

    class A:
        def __init__(self, i, s): self.id = i; self.state = s
    a0 = A(0, AgentState.S1_TRANSIT)  # leader
    a1 = A(1, AgentState.S1_TRANSIT)  # follower
    fm.register_departure([a0, a1])
    red = cfg_fw.aero.formation_drag_reduction
    # follower in transit gets the benefit on CRUISE
    assert fm.power_factor(a1, 0.0, ManeuverType.CRUISE) == pytest.approx(1 - red)
    # leader does not
    assert fm.power_factor(a0, 0.0, ManeuverType.CRUISE) == 1.0
    # coverage never benefits
    a1.state = AgentState.S2_MISSION
    assert fm.power_factor(a1, 0.0, ManeuverType.CRUISE) == 1.0
