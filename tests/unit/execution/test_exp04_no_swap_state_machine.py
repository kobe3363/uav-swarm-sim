"""EXP-04 (mission.no_swap_mode): state-machine, agent and adapter seams.

Same-battery lifecycle: touchdown in S3_RTH enters the terminal S_LANDED state
(no swap request, no Battery.reset, no relaunch), UAV_RETIRED is published
exactly once, and the legacy S3 -> S_SWAP -> S0 cycle is untouched when the
flag is off.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.events import EventBus
from uav_swarm_sim.execution.fleet import Fleet
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import ALLOWED, AgentContext, StateMachine
from uav_swarm_sim.infrastructure.config import ConfigError, load_config
from uav_swarm_sim.infrastructure.core_types import CoveragePlan, Pose, Waypoint
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    BatteryZone,
    EventType,
    ManeuverType,
)
from uav_swarm_sim.metrics.efficiency_score import _DENOM_STATES, efficiency
from uav_swarm_sim.metrics.smdp_estimator import STATE_ORDER, estimate
from uav_swarm_sim.metrics.state_history import StateHistory
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model

S = AgentState

_LEGACY_ORDER = [S.S0_IDLE, S.S1_TRANSIT, S.S2_MISSION, S.S_FERRY, S.S3_RTH,
                 S.S_SWAP, S.S_OBS, S.S_FAIL]


@pytest.fixture(scope="module")
def cfg(config_path):
    return load_config(config_path)


def _ctx(state, **kw):
    return AgentContext(state=state, battery_zone=kw.pop("zone", BatteryZone.HIGH), **kw)


def _make_agent(cfg, *, no_swap: bool, initial_frac: float = 0.5):
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    em = EnergyModel(spec)
    bat = Battery(spec.battery_capacity_j, cfg.battery_zones, initial_frac=initial_frac)
    sm = StateMachine(cfg.battery_zones, no_swap_mode=no_swap)
    base = Pose(0.0, 0.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=100.0)
    agent = Agent(0, spec, motion, em, bat, sm, rth, None, base)
    return agent, motion


def _two_strip_plan(motion, base):
    wps = [
        Waypoint(Pose(100, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(300, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(300, 200, math.pi), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(100, 200, math.pi), ManeuverType.COVERAGE, 6.0),
    ]
    plan = CoveragePlan(0, wps, 0.0, 0.0)
    transit = motion.plan(base, wps[0].pose, ManeuverType.CRUISE)
    return plan, transit


# --------------------------------------------------------------------------- #
# state machine                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("own_plan_incomplete", [True, False])
def test_no_swap_landing_goes_to_s_landed(cfg, own_plan_incomplete):
    sm = StateMachine(cfg.battery_zones, no_swap_mode=True)
    tr = sm.step(_ctx(S.S3_RTH, landed_at_base=True, own_plan_incomplete=own_plan_incomplete))
    assert tr is not None
    assert tr.dst is S.S_LANDED and tr.reason == "landed"
    assert (tr.src, tr.dst) in ALLOWED


def test_flag_off_landing_unchanged(cfg):
    for sm in (StateMachine(cfg.battery_zones), StateMachine(cfg.battery_zones, no_swap_mode=False)):
        tr = sm.step(_ctx(S.S3_RTH, landed_at_base=True, own_plan_incomplete=True))
        assert (tr.dst, tr.reason) == (S.S_SWAP, "swap")
        tr = sm.step(_ctx(S.S3_RTH, landed_at_base=True, own_plan_incomplete=False))
        assert (tr.dst, tr.reason) == (S.S0_IDLE, "mission_done")


def test_legacy_swap_cycle_still_closes(cfg):
    sm = StateMachine(cfg.battery_zones)
    tr = sm.step(_ctx(S.S_SWAP, swap_done=True))
    assert (tr.dst, tr.reason) == (S.S0_IDLE, "swap_done")
    assert (S.S3_RTH, S.S_SWAP) in ALLOWED and (S.S_SWAP, S.S0_IDLE) in ALLOWED


def test_s_landed_is_absorbing_in_both_modes(cfg):
    probes = [
        _ctx(S.S_LANDED, swap_done=True),
        _ctx(S.S_LANDED, launch_command=True, plan_assigned=True),
        _ctx(S.S_LANDED, failure_flag=True),
        _ctx(S.S_LANDED, landed_at_base=True, own_plan_incomplete=True),
        _ctx(S.S_LANDED, rth_decision=True, zone=BatteryZone.TERMINAL),
        _ctx(S.S_LANDED, threat_flag=True),
    ]
    for sm in (StateMachine(cfg.battery_zones), StateMachine(cfg.battery_zones, no_swap_mode=True)):
        for c in probes:
            assert sm.step(c) is None


# --------------------------------------------------------------------------- #
# adapters: airborne set, STATE_ORDER, efficiency, SMDP closure               #
# --------------------------------------------------------------------------- #
def test_s_landed_wiring_in_adapters():
    assert S.S_LANDED.is_airborne is False
    # appended LAST: the legacy eight indices are unchanged
    assert STATE_ORDER[:8] == _LEGACY_ORDER and STATE_ORDER[8] is S.S_LANDED
    assert len(STATE_ORDER) == 9
    # parked ground time is neither useful work nor overhead
    assert S.S_LANDED not in _DENOM_STATES
    states = [S.S2_MISSION, S.S3_RTH, S.S_LANDED]
    assert efficiency(np.array([0.5, 0.25, 0.25]), states) == pytest.approx(2.0)


def _landed_history() -> StateHistory:
    h = StateHistory()
    t = 0.0
    for st, d in [(S.S0_IDLE, 1.0), (S.S1_TRANSIT, 2.0), (S.S2_MISSION, 5.0),
                  (S.S3_RTH, 3.0), (S.S_LANDED, 4.0)]:
        h.open(0, st, t)
        t += d
        h.close(0, t, "next")
    h.finalize(t)
    return h


def test_smdp_landed_closure_keeps_chain_ergodic():
    est = estimate(_landed_history())               # close_landed_loop default True
    assert est.ergodic is True and est.unreachable == []
    assert S.S_LANDED in est.states
    li, s0 = est.states.index(S.S_LANDED), est.states.index(S.S0_IDLE)
    assert est.P[li, s0] == pytest.approx(1.0)
    # its own duration only: no repair time is added for a parked drone
    assert est.mean_sojourn_s[li] == pytest.approx(4.0)
    # absorbing when the closure is refused
    est_open = estimate(_landed_history(), close_landed_loop=False)
    assert est_open.ergodic is False and S.S_LANDED in est_open.unreachable


# --------------------------------------------------------------------------- #
# agent                                                                       #
# --------------------------------------------------------------------------- #
def test_agent_in_s_landed_zero_energy_no_reset_no_relaunch(cfg):
    agent, motion = _make_agent(cfg, no_swap=True, initial_frac=0.3)
    plan, transit = _two_strip_plan(motion, agent.base)
    agent.state = S.S_LANDED
    agent._retired = True
    bus = EventBus()
    level0, used0 = agent.battery.level_j, agent.energy_consumed_j

    for k in range(40):
        agent.step(0.5, k * 0.5, bus)
    assert agent.state is S.S_LANDED
    assert agent.battery.level_j == level0          # zero energy on the ground
    assert agent.energy_consumed_j == used0
    assert agent.battery.frac == pytest.approx(0.3)  # never reset to full

    agent.signal_swap_done()                        # no swap cycle exists
    assert agent._swap_done is False
    agent.adopt_plan(plan, transit)                 # re-tasking refused
    assert agent.plan is None and agent._launch_ready is False
    agent.step(0.5, 20.0, bus)
    assert agent.state is S.S_LANDED and bus.drain() == []


@pytest.mark.parametrize("plan_incomplete", [True, False])
def test_uav_retired_published_exactly_once(cfg, plan_incomplete):
    agent, motion = _make_agent(cfg, no_swap=True, initial_frac=0.4)
    plan, transit = _two_strip_plan(motion, agent.base)
    agent.assign(plan, transit)
    n_legs = len(agent._cov_legs)
    assert n_legs > 0
    agent._cov_idx = 0 if plan_incomplete else n_legs
    # returning drone that has just reached base: S3_RTH with an empty leg queue
    agent.state = S.S3_RTH
    agent._set_legs([])
    bus = EventBus()

    agent.step(0.5, 10.0, bus)
    assert agent.state is S.S_LANDED and agent.retired is True
    events = bus.drain()
    assert [e.type for e in events] == [EventType.UAV_RETIRED]
    assert events[0].t == 10.0
    assert events[0].payload == {
        "agent_id": 0,
        "work_released": plan_incomplete,
        "cov_idx": 0 if plan_incomplete else n_legs,
        "n_cov_legs": n_legs,
    }
    assert agent._launch_ready is False
    assert agent.battery.frac == pytest.approx(0.4)   # no Battery.reset

    for k in range(20):
        agent.step(0.5, 10.5 + k * 0.5, bus)
    assert bus.drain() == []                          # published exactly once
    assert agent.state is S.S_LANDED

    fleet = Fleet([agent])
    assert fleet.active() == [agent] and fleet.workers() == []


def test_flag_off_agent_landing_still_requests_swap(cfg):
    agent, motion = _make_agent(cfg, no_swap=False, initial_frac=0.4)
    plan, transit = _two_strip_plan(motion, agent.base)
    agent.assign(plan, transit)
    agent.state = S.S3_RTH
    agent._set_legs([])
    bus = EventBus()
    agent.step(0.5, 10.0, bus)
    assert agent.state is S.S_SWAP and agent.retired is False
    assert [e.type for e in bus.drain()] == [EventType.SWAP_REQUEST]
    assert Fleet([agent]).workers() == [agent]


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #
def test_config_no_swap_default_off_and_validation(config_path):
    assert load_config(config_path).mission.no_swap_mode is False
    with pytest.raises(ConfigError, match="raster_enabled"):
        load_config(config_path, overrides={"mission.no_swap_mode": True})
    # strict boolean: a quoted "false" must not silently enable the mode
    for bad in ("false", "true", 0, 1, [True]):
        with pytest.raises(ConfigError, match="must be a boolean"):
            load_config(config_path, overrides={
                "mission.no_swap_mode": bad, "coverage.raster_enabled": True,
            })
    with pytest.raises(ConfigError, match="mission.type"):
        load_config(config_path, overrides={
            "mission.no_swap_mode": True,
            "coverage.raster_enabled": True,
            "mission.type": "target_visit",
        })
    ok = load_config(config_path, overrides={
        "mission.no_swap_mode": True,
        "coverage.raster_enabled": True,
    })
    assert ok.mission.no_swap_mode is True
