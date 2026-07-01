"""Tests for S_FERRY -- camera-off repositioning between coverage strips.

Covers the wiring (enum/airborne/state-order/efficiency), the FSM S2<->S_FERRY
toggle driven by the active connector leg, the fact that S_FERRY shares every
coverage-interrupt guard with S2_MISSION (a ferrying drone can still hit an
obstacle, fail, or be told to return), and the camera payload energy term that
is charged only while filming a strip.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from uav_swarm_sim.execution.state_machine import AgentContext, StateMachine
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState as S
from uav_swarm_sim.infrastructure.enums import BatteryZone, ManeuverType
from uav_swarm_sim.metrics.efficiency_score import efficiency
from uav_swarm_sim.metrics.smdp_estimator import STATE_ORDER

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


def _sm() -> StateMachine:
    return StateMachine(load_config(CONFIG_PATH).battery_zones)


def _ctx(**over) -> AgentContext:
    base = dict(state=S.S2_MISSION, battery_zone=BatteryZone.NOMINAL)
    base.update(over)
    return AgentContext(**base)


# --------------------------------------------------------------------------- #
# wiring                                                                       #
# --------------------------------------------------------------------------- #
def test_s_ferry_is_airborne_and_in_state_order():
    assert S.S_FERRY.is_airborne is True
    assert S.S_FERRY in STATE_ORDER


def test_efficiency_counts_ferry_as_overhead():
    import numpy as np
    # equal time productive (S2) and ferrying (S_FERRY) -> efficiency 1.0
    states = [S.S2_MISSION, S.S_FERRY]
    assert efficiency(np.array([0.5, 0.5]), states) == pytest.approx(1.0)
    # ferrying is in the denominator: more ferry -> lower efficiency
    assert efficiency(np.array([0.5, 1.5]), [S.S2_MISSION, S.S_FERRY]) == pytest.approx(1 / 3)


def test_efficiency_unaffected_when_no_ferry():
    import numpy as np
    # a history that never ferries: S_FERRY absent -> denominator unchanged
    states = [S.S2_MISSION, S.S3_RTH, S.S_OBS, S.S_SWAP]
    assert efficiency(np.array([3.0, 1.0, 0.5, 0.5]), states) == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# FSM toggle                                                                   #
# --------------------------------------------------------------------------- #
def test_entering_a_connector_toggles_s2_to_ferry():
    tr = _sm().step(_ctx(state=S.S2_MISSION, on_connector=True))
    assert tr is not None and tr.dst is S.S_FERRY and tr.reason == "ferry_start"


def test_leaving_a_connector_toggles_ferry_back_to_s2():
    tr = _sm().step(_ctx(state=S.S_FERRY, on_connector=False))
    assert tr is not None and tr.dst is S.S2_MISSION and tr.reason == "ferry_end"


def test_on_a_strip_stays_in_s2():
    assert _sm().step(_ctx(state=S.S2_MISSION, on_connector=False)) is None


def test_on_a_connector_stays_in_ferry():
    assert _sm().step(_ctx(state=S.S_FERRY, on_connector=True)) is None


# --------------------------------------------------------------------------- #
# S_FERRY shares the coverage-interrupt guards                                 #
# --------------------------------------------------------------------------- #
def test_ferry_returns_home_on_rth_decision():
    tr = _sm().step(_ctx(state=S.S_FERRY, on_connector=True, rth_decision=True))
    assert tr.dst is S.S3_RTH and tr.reason == "rth_energy"


def test_ferry_returns_home_on_critical_battery():
    tr = _sm().step(_ctx(state=S.S_FERRY, on_connector=True, battery_zone=BatteryZone.CRITICAL))
    assert tr.dst is S.S3_RTH and tr.reason == "critical_battery"


def test_ferry_diverts_to_obstacle_avoidance():
    tr = _sm().step(_ctx(state=S.S_FERRY, on_connector=True, threat_flag=True))
    assert tr.dst is S.S_OBS and tr.reason == "obstacle_threat"


def test_ferry_can_fail_midair():
    tr = _sm().step(_ctx(state=S.S_FERRY, on_connector=True, failure_flag=True))
    assert tr.dst is S.S_FAIL and tr.reason == "failure"


def test_threat_preempts_the_ferry_toggle():
    # a threat while entering a connector goes to S_OBS, not S_FERRY
    tr = _sm().step(_ctx(state=S.S2_MISSION, on_connector=True, threat_flag=True))
    assert tr.dst is S.S_OBS


# --------------------------------------------------------------------------- #
# camera energy                                                                #
# --------------------------------------------------------------------------- #
def test_sensor_energy_is_power_times_time():
    from uav_swarm_sim.physical_model.drone_specs import build_spec
    from uav_swarm_sim.physical_model.energy_model import EnergyModel
    em = EnergyModel(build_spec(load_config(CONFIG_PATH)))
    assert em.sensor_energy(10.0, 15.0) == pytest.approx(150.0)
    assert em.sensor_energy(10.0, 0.0) == 0.0          # camera off -> no payload energy
    assert em.sensor_energy(10.0, -5.0) == 0.0         # guard against negative


# --------------------------------------------------------------------------- #
# connector detection: parity, not per-segment maneuver                        #
# --------------------------------------------------------------------------- #
def test_holonomic_strip_yaw_does_not_toggle_to_ferry():
    """On multirotor (holonomic) paths a productive strip leg still begins with an
    in-place-yaw TURN segment. _on_connector must key off leg-index parity, not the
    current segment's maneuver, so a strip (even _cov_idx) never reports a connector
    even while yawing -- otherwise it would flip to S_FERRY mid-strip and corrupt the
    state history / efficiency / camera semantics."""
    from uav_swarm_sim.execution.agent import Agent
    from uav_swarm_sim.infrastructure.core_types import Pose
    from uav_swarm_sim.physical_model.drone_specs import build_spec
    from uav_swarm_sim.physical_model.motion_model import make_motion_model

    spec = build_spec(load_config(CONFIG_PATH))          # default platform = MULTIROTOR
    motion = make_motion_model(spec)
    # start heading != travel bearing => the strip leg carries an initial yaw TURN
    strip = motion.plan(Pose(0.0, 0.0, 1.5), Pose(100.0, 0.0, 0.0), ManeuverType.COVERAGE)
    kinds = {seg.maneuver for seg in strip.segments}
    assert ManeuverType.TURN in kinds and ManeuverType.COVERAGE in kinds, \
        "a holonomic strip must contain both the yaw TURN and the COVERAGE scan"

    # minimal stand-in that exercises only the attribute reads in _on_connector
    class _Stub:
        pass
    a = _Stub()
    a.state = S.S2_MISSION
    a._leg_mode = "boustrophedon"
    a._cov_legs = [strip, strip]                          # index 0 = strip, 1 = connector
    a._cov_idx = 0                                        # even -> strip
    assert Agent._on_connector(a) is False, "a strip must not report a connector while yawing"
    a._cov_idx = 1                                        # odd -> connector
    assert Agent._on_connector(a) is True
    a._leg_mode = "tour"                                  # target-visit plans never ferry
    assert Agent._on_connector(a) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
