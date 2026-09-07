"""EXP-08: ZONE_COMPLETE, the one-tick re-task hold, and Agent.retask.

Three things are pinned here that the legacy ``adopt_plan`` path gets wrong and
keeps getting wrong (it is deliberately left untouched):

* the state dispatch is exhaustive and RAISES on a state it will not act on,
  instead of replacing the plan and leaving the leg queue running;
* the move is a RECORDED transition, so the transit after a re-task stops being
  charged to the coverage sojourn it interrupted;
* the RTH arm is recomputed exactly once, never twice -- ``sortie_arms`` is a
  reported metric and is asserted elsewhere to hold one entry per S1 sojourn.

Expected energies are computed from the power table here, never read back from
the function under test.
"""
from __future__ import annotations

import math

import pytest

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.events import EventBus
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import ALLOWED, AgentContext, StateMachine
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import CoveragePlan, Pose, Waypoint
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    BatteryZone,
    EventType,
    ManeuverType,
)
from uav_swarm_sim.metrics.state_history import StateHistory
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model

S = AgentState
DT = 1.0


@pytest.fixture(scope="module")
def cfg(config_path):
    return load_config(config_path)


def _ctx(state, **kw):
    return AgentContext(state=state, battery_zone=kw.pop("zone", BatteryZone.HIGH), **kw)


def _agent(cfg, *, repartition: bool, recorder=None, initial_frac: float = 0.9):
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    em = EnergyModel(spec)
    bat = Battery(spec.battery_capacity_j, cfg.battery_zones, initial_frac=initial_frac)
    sm = StateMachine(cfg.battery_zones)
    base = Pose(0.0, 0.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=100.0)
    agent = Agent(0, spec, motion, em, bat, sm, rth, None, base, recorder=recorder,
                  repartition_enabled=repartition)
    return agent, motion, spec


def _plan(motion, base, n_strips=1, x0=100.0, drone_id=0):
    """A short boustrophedon plan: 2 waypoints per strip."""
    wps = []
    for k in range(n_strips):
        y = 200.0 * k
        wps.append(Waypoint(Pose(x0, y, 0.0), ManeuverType.COVERAGE, 6.0))
        wps.append(Waypoint(Pose(x0 + 60.0, y, 0.0), ManeuverType.COVERAGE, 6.0))
    plan = CoveragePlan(drone_id, wps, 0.0, 0.0)
    transit = motion.plan(base, wps[0].pose, ManeuverType.CRUISE)
    return plan, transit


def _fly_to_zone_complete(agent, bus, *, t0=0.0, limit=4000):
    """Step until the agent has finished every coverage leg. Returns the time of
    the tick on which the last leg completed."""
    t = t0
    for _ in range(limit):
        agent.step(DT, t, bus)
        if (agent.state in (S.S2_MISSION, S.S_FERRY) and agent._legs
                and agent._leg_idx >= len(agent._legs)):
            return t
        if agent.state in (S.S3_RTH, S.S_LANDED, S.S_FAIL):
            return t
        t += DT
    raise AssertionError("agent never finished its plan")


# --------------------------------------------------------------------------- #
# A. flag off: nothing new happens                                             #
# --------------------------------------------------------------------------- #
def test_flag_off_publishes_no_zone_complete_and_returns_immediately(cfg):
    agent, motion, _ = _agent(cfg, repartition=False)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()

    t = _fly_to_zone_complete(agent, bus)
    assert agent.state is S.S3_RTH
    assert not [e for e in bus._queue if e.type is EventType.ZONE_COMPLETE]
    # and the return was ordered by finishing the zone, not by a guard
    assert agent._rth_reason == "coverage_complete"
    assert t >= 0.0


def test_the_hold_branch_is_unreachable_with_the_flag_off(cfg):
    """The FSM defers only when the context says a re-task is pending, and only
    the flag can set that. Asserted directly so the gate cannot rot."""
    sm = StateMachine(cfg.battery_zones)
    held = sm.step(_ctx(S.S2_MISSION, coverage_complete=True, repartition_pending=True))
    fired = sm.step(_ctx(S.S2_MISSION, coverage_complete=True))
    assert held is None
    assert (fired.dst, fired.reason) == (S.S3_RTH, "coverage_complete")


# --------------------------------------------------------------------------- #
# B. flag on: announce, hold exactly one tick, then return                     #
# --------------------------------------------------------------------------- #
def test_zone_complete_is_published_once_and_the_return_is_held_one_tick(cfg):
    agent, motion, spec = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()

    t = _fly_to_zone_complete(agent, bus)
    events = [e for e in bus._queue if e.type is EventType.ZONE_COMPLETE]
    assert len(events) == 1
    assert events[0].payload["agent_id"] == 0
    assert events[0].payload["n_cov_legs"] == len(agent._cov_legs)
    assert events[0].payload["cov_idx"] == agent._cov_idx
    # held: still covering, NOT returning
    assert agent.state in (S.S2_MISSION, S.S_FERRY)
    assert agent._repartition_hold is True

    # one more tick with no re-task -> the ordinary return fires, and only once
    bus.drain()
    agent.step(DT, t + DT, bus)
    assert agent.state is S.S3_RTH
    assert agent._rth_reason == "coverage_complete"
    assert not [e for e in bus._queue if e.type is EventType.ZONE_COMPLETE]


def test_the_held_tick_is_charged_as_a_real_hover(cfg):
    """The extra tick is a drone hovering at the end of its last strip, so it
    costs P_HOVER * dt. Expected value computed from the power table."""
    agent, motion, spec = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()

    t = _fly_to_zone_complete(agent, bus)
    energy_at_hold = agent.energy_consumed_j
    level_at_hold = agent.battery.level_j
    flown_at_hold = agent.flown_m
    pose_at_hold = agent.pose

    agent.step(DT, t + DT, bus)          # the held tick: no legs left, hovering

    expected = spec.power_w[ManeuverType.HOVER] * DT
    assert expected > 0.0
    assert agent.energy_consumed_j - energy_at_hold == pytest.approx(expected, rel=1e-12)
    assert level_at_hold - agent.battery.level_j == pytest.approx(expected, rel=1e-12)
    # hovering, so it burned energy and went nowhere
    assert agent.flown_m == pytest.approx(flown_at_hold, abs=0.0)
    assert agent.pose == pose_at_hold


def test_an_empty_plan_announces_nothing(cfg):
    """D-8: an empty plan credits no work. Announcing one would also spin, since
    the drone would be re-tasked, finish instantly and announce again."""
    agent, motion, _ = _agent(cfg, repartition=True)
    agent.assign(CoveragePlan(0, [], 0.0, 0.0), motion.plan(
        agent.base, Pose(10.0, 0.0, 0.0), ManeuverType.CRUISE))
    bus = EventBus()
    for k in range(40):
        agent.step(DT, k * DT, bus)
        if agent.state is S.S3_RTH:
            break
    assert not [e for e in bus._queue if e.type is EventType.ZONE_COMPLETE]


def test_the_energy_and_safety_guards_still_pre_empt_the_hold(cfg):
    """The hold sits BELOW every guard in the coverage branch, so a threat, the
    dynamic RTH decision and both battery nets all still win on the same tick.
    A drone that must go home can never be held."""
    sm = StateMachine(cfg.battery_zones)
    both = dict(coverage_complete=True, repartition_pending=True)
    assert sm.step(_ctx(S.S2_MISSION, threat_flag=True, **both)).dst is S.S_OBS
    assert sm.step(_ctx(S.S2_MISSION, rth_decision=True, **both)).reason == "rth_energy"
    assert sm.step(_ctx(S.S2_MISSION, zone=BatteryZone.CRITICAL, **both)).reason \
        == "critical_battery"
    assert sm.step(_ctx(S.S2_MISSION, zone=BatteryZone.TERMINAL, **both)).reason \
        == "terminal_battery"


# --------------------------------------------------------------------------- #
# C. a zone completion must not disturb the reported RTH surface               #
#    (the EXP-07 B-1 defect class: mechanics polluting a reported metric)       #
# --------------------------------------------------------------------------- #
def test_announcing_a_finished_zone_moves_no_rth_counter_and_no_decision(cfg):
    """Entering S3_RTH computes a return path, which on the map arm increments
    n_map_hits / n_map_fallbacks / n_route_fallbacks -- all three reported. The
    hold means the announcement happens BEFORE any of that, so there is nothing
    to compensate for: the counters cannot move, rather than moving and being
    corrected. _rth_decision is likewise untouched, and provably False, because
    the FSM tests rth_decision ahead of coverage_complete."""
    agent, motion, _ = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()

    before = (agent.rth.n_map_hits, agent.rth.n_map_fallbacks,
              agent.rth.n_route_fallbacks)
    _fly_to_zone_complete(agent, bus)

    assert [e.type for e in bus._queue].count(EventType.ZONE_COMPLETE) == 1
    assert (agent.rth.n_map_hits, agent.rth.n_map_fallbacks,
            agent.rth.n_route_fallbacks) == before
    assert agent._rth_decision is False
    assert agent.photo_events == ()          # no photo pass was closed by this


# --------------------------------------------------------------------------- #
# D. retask: exhaustive dispatch, recorded transition, single arm              #
# --------------------------------------------------------------------------- #
def test_retask_from_coverage_is_a_recorded_transition(cfg):
    history = StateHistory()
    agent, motion, _ = _agent(cfg, repartition=True, recorder=history)
    history.open(0, S.S0_IDLE, 0.0)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()

    t = _fly_to_zone_complete(agent, bus)
    assert agent.state in (S.S2_MISSION, S.S_FERRY)
    source = agent.state

    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    new_transit = motion.plan(agent.pose, new_plan.waypoints[0].pose,
                              ManeuverType.CRUISE)
    agent.retask(new_plan, new_transit, t, bus)

    assert agent.state is S.S1_TRANSIT
    assert (source, S.S1_TRANSIT) in ALLOWED

    # Fly the transit for a known number of ticks, so the split between the two
    # sojourns has an expected value derived here rather than read back.
    n_ticks = 6
    for k in range(1, n_ticks + 1):
        agent.step(DT, t + k * DT, bus)
    end = t + n_ticks * DT
    history.finalize(end)

    sojourns = [s for s in history.sojourns() if s.agent_id == 0]
    closed = [s for s in sojourns if s.reason_out == "retask"]
    assert len(closed) == 1
    # the coverage sojourn ends exactly at the re-task ...
    assert closed[0].state is source
    assert closed[0].t_out == pytest.approx(t)
    # ... and every second after it belongs to the transit, not to coverage.
    # Before EXP-08 the re-task assigned agent.state directly, so this whole
    # stretch was charged to the S2_MISSION sojourn it interrupted.
    transit = [s for s in sojourns if s.state is S.S1_TRANSIT and s.t_in >= t]
    assert transit and transit[0].t_in == pytest.approx(t)
    assert sum(s.t_out - s.t_in for s in sojourns
               if s.state is source and s.t_in >= t) == pytest.approx(0.0)


def test_retask_while_already_transiting_records_no_self_loop(cfg):
    history = StateHistory()
    agent, motion, _ = _agent(cfg, repartition=True, recorder=history)
    history.open(0, S.S0_IDLE, 0.0)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()
    agent.step(DT, 0.0, bus)                       # S0 -> S1
    assert agent.state is S.S1_TRANSIT

    history.finalize(1.0)
    before = len([s for s in history.sojourns() if s.agent_id == 0])

    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    agent.retask(new_plan, motion.plan(agent.pose, new_plan.waypoints[0].pose,
                                       ManeuverType.CRUISE), 1.0, bus)

    assert agent.state is S.S1_TRANSIT
    history.finalize(2.0)
    after = len([s for s in history.sojourns() if s.agent_id == 0])
    assert after == before, "a self-loop would invent a transition that never happened"
    # the destination really did change
    assert agent.plan is new_plan
    assert agent._legs and agent._leg_idx == 0


def test_a_retask_arms_the_rth_threshold_exactly_once(cfg, monkeypatch):
    """_apply_transition arms on entry to S1_TRANSIT, so retask must not arm as
    well. sortie_arms is reported, and a second entry for one sortie is a silent
    metric corruption -- the class of defect this ticket exists to catch."""
    agent, motion, _ = _agent(cfg, repartition=True)
    monkeypatch.setattr(type(agent.rth), "map_decide_on", property(lambda self: True))
    monkeypatch.setattr(agent.rth, "sortie_arm_j", lambda bundles, alt: 1234.0,
                        raising=False)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()

    t = _fly_to_zone_complete(agent, bus)
    arms_before = len(agent.sortie_arms)
    sortie_before = agent._sortie_idx

    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    agent.retask(new_plan, motion.plan(agent.pose, new_plan.waypoints[0].pose,
                                       ManeuverType.CRUISE), t, bus)

    assert len(agent.sortie_arms) == arms_before + 1
    assert agent._sortie_idx == sortie_before + 1


def test_a_retask_while_transiting_amends_the_arm_instead_of_appending(cfg, monkeypatch):
    agent, motion, _ = _agent(cfg, repartition=True)
    monkeypatch.setattr(type(agent.rth), "map_decide_on", property(lambda self: True))
    arms = iter([1000.0, 2000.0, 3000.0])
    monkeypatch.setattr(agent.rth, "sortie_arm_j",
                        lambda bundles, alt: next(arms), raising=False)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()
    agent.step(DT, 0.0, bus)                       # S0 -> S1, arms once
    assert agent.sortie_arms == [(1, 1000.0)]

    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    agent.retask(new_plan, motion.plan(agent.pose, new_plan.waypoints[0].pose,
                                       ManeuverType.CRUISE), 1.0, bus)

    # one sortie, one entry -- but the entry now describes the NEW plan
    assert agent.sortie_arms == [(1, 2000.0)]
    assert agent._sortie_idx == 1


@pytest.mark.parametrize("state", [S.S3_RTH, S.S_FAIL, S.S_LANDED])
def test_retask_refuses_a_state_it_would_not_act_on(cfg, state):
    """adopt_plan silently swaps the plan of a returning or swapping drone and
    leaves its legs alone. This path raises instead: a re-partition must never
    cancel a return, and it must never half-apply."""
    agent, motion, _ = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    agent.state = state
    with pytest.raises(ValueError, match="cannot be re-tasked"):
        agent.retask(plan, transit, 0.0, EventBus())


def test_retask_refuses_a_retired_drone(cfg):
    agent, motion, _ = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    agent._retired = True
    agent.state = S.S2_MISSION
    with pytest.raises(ValueError, match="retired"):
        agent.retask(plan, transit, 0.0, EventBus())


def test_retask_touches_no_history_of_what_the_drone_already_did(cfg):
    agent, motion, _ = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()
    t = _fly_to_zone_complete(agent, bus)

    before = (agent.energy_consumed_j, agent.flown_m, agent.battery.level_j,
              agent.pose)
    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    agent.retask(new_plan, motion.plan(agent.pose, new_plan.waypoints[0].pose,
                                       ManeuverType.CRUISE), t, bus)

    assert (agent.energy_consumed_j, agent.flown_m, agent.battery.level_j,
            agent.pose) == before


def test_retask_from_avoidance_leaves_through_the_existing_edge(cfg):
    """An agent in S_OBS is committed to a validated micro-plan. It keeps flying
    it and then leaves through S_OBS -> S1_TRANSIT, which already exists -- and
    the pre-avoidance leg queue, which belongs to a plan it no longer has, is
    dropped rather than restored on top of the new one."""
    agent, motion, _ = _agent(cfg, repartition=True)
    plan, transit = _plan(motion, agent.base)
    agent.assign(plan, transit)
    bus = EventBus()
    agent.step(DT, 0.0, bus)
    agent.signal_threat(True)
    agent.step(DT, 1.0, bus)
    assert agent.state is S.S_OBS
    assert agent._obs_legs_saved is not None

    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    new_transit = motion.plan(agent.pose, new_plan.waypoints[0].pose,
                              ManeuverType.CRUISE)
    agent.retask(new_plan, new_transit, 2.0, bus)

    assert agent.state is S.S_OBS                  # still avoiding
    assert agent._obs_return is S.S1_TRANSIT
    assert agent._obs_legs_saved is None
    assert (S.S_OBS, S.S1_TRANSIT) in ALLOWED

    agent.signal_threat_cleared()
    agent.step(DT, 3.0, bus)
    assert agent.state is S.S1_TRANSIT
    assert agent._legs == [new_transit]


# --------------------------------------------------------------------------- #
# E. the missing half of the ALLOWED check                                     #
# --------------------------------------------------------------------------- #
def test_the_observed_chain_is_checked_against_allowed(cfg):
    """ALLOWED is read by no code in ``src`` -- the estimator adds the synthetic
    S_FAIL closure but never validates against this table, and the docstring
    that claimed otherwise is corrected in this commit. So this test INTRODUCES
    the missing half of the check rather than satisfying an existing one: it
    walks an observed sojourn chain that includes a re-task and asserts every
    consecutive pair is a designed edge. Before EXP-08 the re-task was applied
    by direct assignment behind the recorder, so an out-of-design pair could not
    even be seen here."""
    history = StateHistory()
    agent, motion, _ = _agent(cfg, repartition=True, recorder=history)
    history.open(0, S.S0_IDLE, 0.0)
    plan, transit = _plan(motion, agent.base, n_strips=2)
    agent.assign(plan, transit)
    bus = EventBus()

    t = _fly_to_zone_complete(agent, bus)
    new_plan, _ = _plan(motion, agent.base, x0=400.0)
    agent.retask(new_plan, motion.plan(agent.pose, new_plan.waypoints[0].pose,
                                       ManeuverType.CRUISE), t, bus)
    for k in range(1, 400):
        agent.step(DT, t + k * DT, bus)
        if agent.state is S.S3_RTH:
            break
    history.finalize(t + 400 * DT)

    chain = [s.state for s in history.sojourns() if s.agent_id == 0]
    assert S.S1_TRANSIT in chain and chain.count(S.S1_TRANSIT) >= 2
    for src, dst in zip(chain, chain[1:]):
        assert (src, dst) in ALLOWED, f"{src} -> {dst} is not a designed edge"
