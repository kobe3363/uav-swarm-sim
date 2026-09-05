"""Agent wiring keeps shutter events off connectors, avoidance and RTH legs."""
from __future__ import annotations

import pytest

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.formation_manager import FormationManager
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import StateMachine
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import AgentState, ManeuverType
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model


def _agent(config_path, coverage_observer=None):
    cfg = load_config(config_path)
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    energy = EnergyModel(spec)
    base = Pose(0.0, 0.0, 0.0)
    battery = Battery(spec.battery_capacity_j, cfg.battery_zones, 1.0)
    state_machine = StateMachine(cfg.battery_zones)
    rth = RthCalculator(energy, motion, spec, cfg.rth, base, altitude_m=100.0)
    aero = AeroCorrection(cfg.aero, spec.platform)
    formation = FormationManager(aero, cfg.aero, spec.platform)
    agent = Agent(
        0, spec, motion, energy, battery, state_machine, rth, formation, base,
        photo_spacing_m=10.0,
        coverage_observer=coverage_observer,
    )
    return agent, motion


def test_only_productive_strip_distance_triggers_photos(config_path):
    agent, motion = _agent(config_path)
    strip = motion.plan(Pose(0.0, 0.0, 0.0), Pose(30.0, 0.0, 0.0), ManeuverType.COVERAGE)
    connector = motion.plan(Pose(30.0, 0.0, 0.0), Pose(30.0, 30.0, 0.0), ManeuverType.TURN)
    rth = motion.plan(Pose(30.0, 30.0, 0.0), Pose(0.0, 0.0, 0.0), ManeuverType.CRUISE)
    agent._leg_mode = "boustrophedon"
    agent._cov_legs = [strip, connector]

    agent.state = AgentState.S2_MISSION
    agent.pose = strip.start_pose
    agent._cov_idx = 0
    agent._set_legs([strip])
    agent._tick_dynamics(2.0, 0.0)
    assert [event.distance_on_strip_m for event in agent.photo_events] == [0.0, 10.0]

    before = len(agent.photo_events)
    for state, index, leg in [
        (AgentState.S_FERRY, 1, connector),
        (AgentState.S_OBS, 0, connector),
        (AgentState.S3_RTH, 0, rth),
    ]:
        agent.state = state
        agent._cov_idx = index
        agent.pose = leg.start_pose
        agent._set_legs([leg])
        agent._tick_dynamics(1.0, 10.0)
    assert len(agent.photo_events) == before


def test_coverage_observer_receives_only_actual_productive_motion(config_path):
    observed = []
    agent, motion = _agent(config_path, lambda old, new: observed.append((old, new)))
    strip = motion.plan(Pose(0.0, 0.0, 0.0), Pose(30.0, 0.0, 0.0), ManeuverType.COVERAGE)
    connector = motion.plan(Pose(30.0, 0.0, 0.0), Pose(30.0, 30.0, 0.0), ManeuverType.TURN)

    agent._leg_mode = "boustrophedon"
    agent._cov_legs = [strip, connector]
    agent.state = AgentState.S2_MISSION
    agent.pose = strip.start_pose
    agent._cov_idx = 0
    agent._set_legs([strip])
    agent._tick_dynamics(1.0, 0.0)
    assert len(observed) == 1
    assert observed[0][0].as_xy() == pytest.approx((0.0, 0.0))
    assert observed[0][1].as_xy() == pytest.approx((agent.spec.v_coverage, 0.0))

    agent.state = AgentState.S_FERRY
    agent.pose = connector.start_pose
    agent._cov_idx = 1
    agent._set_legs([connector])
    agent._tick_dynamics(1.0, 1.0)
    assert len(observed) == 1

    rth = motion.plan(Pose(30.0, 30.0, 0.0), Pose(0.0, 0.0, 0.0), ManeuverType.CRUISE)
    agent.state = AgentState.S3_RTH
    agent.pose = rth.start_pose
    agent._cov_idx = 0
    agent._set_legs([rth])
    agent._tick_dynamics(1.0, 2.0)
    assert len(observed) == 1
