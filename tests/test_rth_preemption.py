"""Regression tests for the S2_MISSION return-trigger PRIORITY (H1, second half).

The dynamic route-vs-return reserve (guideline 3.1) is the thesis's primary
early-return mechanism. Before this fix the state machine evaluated the crude
`terminal_battery` (<20%) guard ABOVE the dynamic `rth_energy` guard, and ABOVE
the 20-40% `critical_battery` guard -- an inverted severity order. The original
guard even carried the comment "anomaly: RTH should pre-empt". A return that the
dynamic reserve triggered could therefore be mis-attributed to a battery
threshold, and the <20% net sat above the >=40% net it could never legitimately
beat under normal draining.

These tests lock in the corrected priority within S2_MISSION:

    threat  >  rth_energy (dynamic)  >  critical (>=40% net)  >  terminal (<20% net)
            >  coverage_complete

They are pure StateMachine unit tests: deterministic, no agent/physics, fast.
The transition DESTINATION for every battery/energy trigger is S3_RTH either way
(so trajectories are unchanged by the reorder); what these guard is the REASON
attribution that flows into the telemetry event log and the SMDP transition
labels, plus the documented invariant that a covering drone returns at the
CRITICAL boundary rather than reaching TERMINAL while still in S2.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.execution.state_machine import AgentContext, StateMachine
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, BatteryZone

S = AgentState
CONFIG_PATH = "config/default.yaml"


@pytest.fixture
def sm():
    return StateMachine(load_config(CONFIG_PATH).battery_zones)


def _ctx(**kw):
    """An S2_MISSION context with a healthy (NOMINAL) battery and no flags set,
    overridden by keyword for the trigger(s) under test."""
    base = dict(state=S.S2_MISSION, battery_zone=BatteryZone.NOMINAL)
    base.update(kw)
    return AgentContext(**base)


def test_dynamic_rth_preempts_terminal_battery(sm):
    """If the dynamic reserve says return AND the battery has already reached
    TERMINAL, the return must be attributed to the dynamic mechanism (the thesis
    claim, guideline 3.1) -- not to the crude <20% net it should have
    anticipated."""
    tr = sm.step(_ctx(rth_decision=True, battery_zone=BatteryZone.TERMINAL))
    assert tr.dst is S.S3_RTH
    assert tr.reason == "rth_energy"


def test_dynamic_rth_preempts_critical_battery(sm):
    """Same priority over the 20-40% CRITICAL net: the dynamic reserve, not the
    threshold, owns the return."""
    tr = sm.step(_ctx(rth_decision=True, battery_zone=BatteryZone.CRITICAL))
    assert tr.dst is S.S3_RTH
    assert tr.reason == "rth_energy"


def test_critical_is_evaluated_before_terminal(sm):
    """Severity order among the crude nets: a CRITICAL-zone battery (the higher
    threshold) returns as 'critical_battery'. Because CRITICAL is tested first
    and a normally-draining battery passes through CRITICAL before TERMINAL, a
    covering drone returns at the CRITICAL boundary and never reaches TERMINAL
    while still in S2."""
    tr = sm.step(_ctx(battery_zone=BatteryZone.CRITICAL))
    assert tr.dst is S.S3_RTH
    assert tr.reason == "critical_battery"


def test_terminal_battery_remains_a_last_resort_net(sm):
    """The TERMINAL guard still forces RTH if somehow reached (preserving the
    behaviour test_batch4.test_terminal_battery_forces_rth depends on), but only
    when no higher-priority trigger fired."""
    tr = sm.step(_ctx(battery_zone=BatteryZone.TERMINAL))
    assert tr.dst is S.S3_RTH
    assert tr.reason == "terminal_battery"


def test_threat_preempts_every_return_trigger(sm):
    """Obstacle threat is strictly highest priority in S2: even with the dynamic
    reserve tripped and the battery in TERMINAL, the drone evades first."""
    tr = sm.step(_ctx(threat_flag=True, rth_decision=True, battery_zone=BatteryZone.TERMINAL))
    assert tr.dst is S.S_OBS
    assert tr.reason == "obstacle_threat"


def test_coverage_complete_returns_when_no_energy_trigger_fires(sm):
    """A drone that finishes its zone on a healthy battery returns via the clean
    'coverage_complete' reason (the happy-path completion), unchanged by the
    reorder."""
    tr = sm.step(_ctx(coverage_complete=True, battery_zone=BatteryZone.NOMINAL))
    assert tr.dst is S.S3_RTH
    assert tr.reason == "coverage_complete"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
