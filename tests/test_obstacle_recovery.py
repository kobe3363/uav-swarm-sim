"""Regression tests for the Task 2.5 Q2 obstacle-recovery activation (Batch 2.5).

These lock in the fix for the obstacle-avoidance limit cycle: when a coverage
waypoint sits in / behind a *known static* obstacle, the SafetyMonitor correctly
predicts penetration and the agent enters ``S_OBS`` -- but with the legacy blind
sidestep it returns to the SAME obstructed leg and re-threatens forever, draining
to depletion (drone-0 failure mode) or freezing in ``S_OBS`` (drone-3 failure
mode). The cure already exists in ``Agent`` (validated detour + obstructed-leg
skip + boxed-in escalation, Task 2.5 Q2); it was simply dormant because
``SafetyConfig`` carried no ``obstacle_recovery`` field for the SafetyMonitor to
consult. This batch wires that field through (defaulting OFF, so every config
that does not set it stays byte-identical to the pre-Q2 baseline).

What is real vs. doubled
------------------------
REAL (the code under test): ``Agent`` (FSM + S_OBS recovery), ``StateMachine``,
``SafetyMonitor`` (threat prediction + recovery signalling), ``EnvironmentMap``
(exact Shapely obstacle geometry), ``SafetyConfig``.

DOUBLED (deterministic test doubles implementing exactly the interface slice the
agent / monitor invoke): the platform physics -- motion, energy, battery, RTH,
aero. This isolates the *control* logic, which is what the fix lives in, from the
physics model; the behaviour locked here holds for any platform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import BatteryZonesConfig, SafetyConfig
from uav_swarm_sim.infrastructure.core_types import (
    CoveragePlan,
    Path,
    Pose,
    Waypoint,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import AgentState, BatteryZone, ManeuverType
from uav_swarm_sim.execution.agent import Agent, _OBS_REENTRY_BUDGET
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.execution.safety_monitor import SafetyMonitor
from uav_swarm_sim.execution.state_machine import StateMachine


# --------------------------------------------------------------------------- #
# deterministic test doubles (only the slice Agent / SafetyMonitor call)       #
# --------------------------------------------------------------------------- #
_POWER = {
    ManeuverType.IDLE: 12.0, ManeuverType.TAKEOFF: 280.0, ManeuverType.CLIMB: 280.0,
    ManeuverType.CRUISE: 220.0, ManeuverType.COVERAGE: 210.0, ManeuverType.TURN: 210.0,
    ManeuverType.DESCENT: 140.0, ManeuverType.LAND: 150.0, ManeuverType.HOVER: 200.0,
}
_SPEED = {ManeuverType.CRUISE: 12.0, ManeuverType.COVERAGE: 6.0, ManeuverType.TURN: 6.0}


class _Motion:
    """Holonomic straight-line planner (the MULTIROTOR r_min=0 motion model)."""

    def plan(self, a: Pose, b: Pose, maneuver: ManeuverType) -> Path:
        d = math.dist((a.x, a.y), (b.x, b.y))
        if d <= 1e-9:
            return Path()
        h = math.atan2(b.y - a.y, b.x - a.x)
        v = _SPEED.get(maneuver, 12.0) or 12.0
        return Path.from_segments([straight_segment(Pose(a.x, a.y, h, a.z), d, maneuver, v)])

    def advance(self, leg: Path, t: float, dt: float, current_pose: Pose | None = None):
        # Mirrors the real MotionModel.advance fallback (current_pose=None): pure
        # path-following. The smooth-convergence branch added in production is
        # real-motion physics, deliberately out of scope for these control-logic
        # doubles -- and a no-op here anyway, since this double never drifts off
        # the ideal path, so current_pose would always equal the ideal pose.
        new_t = min(t + dt, leg.total_duration_s)
        return leg.pose_at_time(new_t), new_t


class _Energy:
    def segment_energy(self, man, dt, f=1.0):
        return _POWER.get(man, 200.0) * dt * f

    def sensor_energy(self, dt, sensor_power_w):
        return max(0.0, sensor_power_w) * dt

    def path_energy(self, path: Path):
        return sum(_POWER.get(s.maneuver, 200.0) * s.duration_s for s in path.segments)

    def distance_energy(self, dist, man, speed):
        return _POWER.get(man, 200.0) * (dist / speed) if speed > 0 else 0.0


class _Battery:
    def __init__(self, capacity_j, zones):
        self.capacity_j = capacity_j
        self.charge_j = capacity_j
        self._z = zones

    def drain(self, e):
        self.charge_j = max(0.0, self.charge_j - e)

    @property
    def frac(self):
        return self.charge_j / self.capacity_j if self.capacity_j > 0 else 0.0

    @property
    def zone(self):
        f = self.frac
        if f >= self._z.high:
            return BatteryZone.HIGH
        if f >= self._z.nominal:
            return BatteryZone.NOMINAL
        if f >= self._z.critical:
            return BatteryZone.CRITICAL
        return BatteryZone.TERMINAL

    def reset(self):
        self.charge_j = self.capacity_j


class _Rth:
    check_interval_s = 5.0

    def decide(self, agent):  # keep the drone in coverage unless genuinely low
        return "RETURN_NOW" if agent.battery.frac < 0.15 else "CONTINUE"


class _Aero:
    def wake_zones(self, leaders):
        return []


class _Recorder:
    def __init__(self):
        self.opens: list[str] = []

    def open(self, aid, state, t):
        self.opens.append(state.name)

    def close(self, aid, t, reason):
        pass


class _Bus:
    def publish(self, e):
        pass


@dataclass(frozen=True)
class _Obstacle:
    polygon: Polygon
    id: int = 0


_AREA = Polygon([(0, 0), (2000, 0), (2000, 1500), (0, 1500)])


def _make_agent(recorder=None):
    zones = BatteryZonesConfig()
    return (
        Agent(
            0, object(), _Motion(), _Energy(), _Battery(4.0e6, zones),
            StateMachine(zones), _Rth(), None, Pose(100, 500, 0.0), recorder=recorder,
        ),
        zones,
    )


def _run(agent, monitor, bus, ticks, dt=0.5, stop_states=()):
    t = 0.0
    seen = set()
    for _ in range(ticks):
        monitor.step([agent], t, bus)
        agent.step(dt, t, bus)
        seen.add(agent.state)
        if agent.state in stop_states:
            break
        t += dt
    return seen


# --------------------------------------------------------------------------- #
# config wiring                                                                #
# --------------------------------------------------------------------------- #
def test_safety_config_obstacle_recovery_defaults_off():
    """Existing configs constructed without the new field stay byte-identical:
    recovery defaults OFF (the pre-Q2 behaviour)."""
    cfg = SafetyConfig(min_separation_m=10.0, obstacle_buffer_m=5.0, predict_horizon_s=5.0)
    assert cfg.obstacle_recovery is False


def test_safety_config_obstacle_recovery_can_enable():
    cfg = SafetyConfig(10.0, 5.0, 5.0, obstacle_recovery=True)
    assert cfg.obstacle_recovery is True


# --------------------------------------------------------------------------- #
# behaviour: the limit cycle and its cure                                      #
# --------------------------------------------------------------------------- #
def _obstructed_coverage_agent(recorder):
    """Agent in S2 flying a coverage strip (y=630) blocked by a static obstacle,
    with a second clear strip (y=750) reachable once the blocked leg is skipped."""
    agent, _ = _make_agent(recorder)
    motion = agent.motion
    wps = [
        Waypoint(Pose(150, 630, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(450, 630, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(450, 750, math.pi), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(150, 750, math.pi), ManeuverType.COVERAGE, 6.0),
    ]
    plan = CoveragePlan(0, wps, 0.0, 0.0)
    agent.assign(plan, motion.plan(agent.base, Pose(150, 630, 0.0), ManeuverType.CRUISE))
    # drop the agent straight into coverage on the blocked leg
    agent.state = AgentState.S2_MISSION
    agent._legs = list(agent._cov_legs)
    agent._leg_idx = 0
    agent._t = 0.0
    agent._cov_idx = 0
    return agent


def test_recovery_on_breaks_the_cycle_and_progresses():
    """Fix (recovery ON): the obstructed leg is skipped, the agent covers the
    remaining (clear) leg and returns home -- bounded S_OBS, real progress."""
    rec = _Recorder()
    agent = _obstructed_coverage_agent(rec)
    obstacle = _Obstacle(Polygon([(230, 600), (290, 600), (290, 660), (230, 660)]))
    env = EnvironmentMap(_AREA, [obstacle], buffer_m=5.0)
    cfg = SafetyConfig(10.0, 5.0, 5.0, obstacle_recovery=True)
    mon = SafetyMonitor(env, _Aero(), cfg, agent.motion)

    seen = _run(agent, mon, _Bus(), ticks=600, stop_states=(AgentState.S0_IDLE,))

    obs_entries = sum(1 for s in rec.opens if s == "S_OBS")
    assert obs_entries <= 5, f"recovery ON must not thrash, saw {obs_entries} S_OBS entries"
    assert agent._cov_idx > 0, "recovery ON must advance past the blocked leg"
    assert AgentState.S3_RTH in seen, "agent should complete coverage and return home"


def test_recovery_escalates_to_rth_when_boxed_in():
    """Boxed-in safety net: when every detour re-threatens and no coverage leg
    can complete, the re-entry budget escalates S_OBS -> S3_RTH (the agent
    abandons the region and returns home) rather than freezing or thrashing."""
    rec = _Recorder()
    agent, _ = _make_agent(rec)
    motion = agent.motion
    # several long strips, all fully inside a large obstacle -> nothing clears
    bigbox = _Obstacle(Polygon([(100, 400), (1500, 400), (1500, 1000), (100, 1000)]))
    wps = []
    for i, y in enumerate([450, 500, 550, 600, 650, 700]):
        if i % 2 == 0:
            wps += [Waypoint(Pose(200, y, 0.0), ManeuverType.COVERAGE, 6.0),
                    Waypoint(Pose(1400, y, 0.0), ManeuverType.COVERAGE, 6.0)]
        else:
            wps += [Waypoint(Pose(1400, y, math.pi), ManeuverType.COVERAGE, 6.0),
                    Waypoint(Pose(200, y, math.pi), ManeuverType.COVERAGE, 6.0)]
    plan = CoveragePlan(0, wps, 0.0, 0.0)
    agent.assign(plan, motion.plan(agent.base, Pose(200, 450, 0.0), ManeuverType.CRUISE))
    agent.state = AgentState.S2_MISSION
    agent._legs = list(agent._cov_legs)
    agent._leg_idx = 0
    agent._t = 0.0
    agent._cov_idx = 0

    env = EnvironmentMap(_AREA, [bigbox], buffer_m=5.0)
    cfg = SafetyConfig(10.0, 5.0, 5.0, obstacle_recovery=True)
    mon = SafetyMonitor(env, _Aero(), cfg, agent.motion)

    seen = _run(agent, mon, _Bus(), ticks=800, stop_states=(AgentState.S3_RTH,))

    obs_entries = sum(1 for s in rec.opens if s == "S_OBS")
    assert AgentState.S3_RTH in seen, "boxed-in agent must escalate S_OBS -> S3_RTH"
    assert agent._cov_idx < len(agent._cov_legs), "escalation via budget, not skip-to-completion"
    assert obs_entries <= _OBS_REENTRY_BUDGET + 1, f"escalation too slow: {obs_entries} entries"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
