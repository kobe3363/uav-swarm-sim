"""One drone: pose, battery, plan progress, per-tick stepping.

Flight is modeled as a queue of 'legs' (Path objects) per phase: a transit leg
for S1, alternating strip/connector legs for S2, a return leg for S3, an
avoidance micro-plan for S_OBS. Each tick advances the current leg by time and
drains energy via the EnergyModel for the active maneuver (P*dt), so total
mission energy is the exact time integral. The motion model is the single source
of kinematics; the agent never integrates poses itself.

2.5D (Batch 4): the agent carries its assigned ``layer`` index and the
``coverage_altitude_m`` of that layer. The altitude feeds the RTH descent term
only (a budget quantity); horizontal flight is unchanged and stays in the z=0
plane, so per-layer obstacle slicing and the per-layer return reserve are the
only multi-layer effects -- and with one layer (altitude == coverage_altitude_m)
the behaviour is byte-identical to the 2D model.
"""
from __future__ import annotations

import math
from typing import Protocol

from ..infrastructure.core_types import (
    CoveragePlan,
    DroneStateView,
    Path,
    Pose,
    Waypoint,
)
from ..infrastructure.enums import AgentState, EventType, ManeuverType
from ..infrastructure.core_types import Event
from ..physical_model.battery import Battery
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel
from .rth_calculator import RthCalculator
from .state_machine import AgentContext, StateMachine


class Recorder(Protocol):
    def open(self, agent_id: int, state: AgentState, t: float) -> None: ...
    def close(self, agent_id: int, t: float, reason: str) -> None: ...


class Agent:
    def __init__(
        self,
        id: int,
        spec: PlatformSpec,
        motion: MotionModel,
        em: EnergyModel,
        battery: Battery,
        sm: StateMachine,
        rth: RthCalculator,
        formation,
        base: Pose,
        recorder: Recorder | None = None,
        layer: int = 0,
        coverage_altitude_m: float | None = None,
    ) -> None:
        self.id = id
        self.spec = spec
        self.motion = motion
        self.em = em
        self.battery = battery
        self.sm = sm
        self.rth = rth
        self.formation = formation
        self.base = base
        self.recorder = recorder
        # 2.5D assignment: which coverage layer this drone flies and its altitude
        # (altitude feeds only the RTH descent reserve; flight stays horizontal).
        self.layer = layer
        self.coverage_altitude_m = coverage_altitude_m

        self.state: AgentState = AgentState.S0_IDLE
        self.pose: Pose = base
        self.plan: CoveragePlan | None = None
        self.flown_m: float = 0.0
        self.energy_consumed_j: float = 0.0  # cumulative; survives battery swaps

        # leg queue for the current phase
        self._legs: list[Path] = []
        self._leg_idx: int = 0
        self._t: float = 0.0

        # coverage progress (leg index into the coverage leg list)
        self._cov_legs: list[Path] = []
        self._cov_idx: int = 0
        self._transit: Path | None = None
        self._leg_mode: str = "boustrophedon"

        # flags
        self._launch_ready = False
        self._failure = False
        self._threat = False
        self._threat_cleared = False
        self._coverage_complete = False
        self._obs_return = AgentState.S1_TRANSIT
        self._obs_legs_saved: tuple[list[Path], int] | None = None
        self._last_rth_t = -1e9
        self._swap_done = False

    # ------------------------------------------------------------------ #
    # setup                                                              #
    # ------------------------------------------------------------------ #
    def assign(self, plan: CoveragePlan, transit: Path) -> None:
        self.plan = plan
        self._transit = transit
        self._leg_mode = getattr(plan, "leg_mode", "boustrophedon")
        self._cov_legs = self._build_coverage_legs(plan.waypoints)
        self._cov_idx = 0
        self._coverage_complete = False
        self._launch_ready = True

    def _build_coverage_legs(self, waypoints: list[Waypoint]) -> list[Path]:
        legs: list[Path] = []
        if getattr(self, "_leg_mode", "boustrophedon") == "tour":
            # target-visit: cruise straight between consecutive target points
            for i in range(len(waypoints) - 1):
                a, b = waypoints[i].pose, waypoints[i + 1].pose
                legs.append(self.motion.plan(a, b, ManeuverType.CRUISE))
            return legs
        # area coverage: even legs are strips (COVERAGE), odd legs are U-turn connectors (TURN)
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i].pose, waypoints[i + 1].pose
            maneuver = ManeuverType.COVERAGE if i % 2 == 0 else ManeuverType.TURN
            legs.append(self.motion.plan(a, b, maneuver))
        return legs

    def adopt_plan(self, plan: CoveragePlan, transit: Path) -> None:
        """Re-task a live agent after redistribution: take the new coverage
        plan and re-transit toward its new zone from the current pose.
        (Simplification: in-progress coverage of the old zone is dropped.)"""
        self.plan = plan
        self._leg_mode = getattr(plan, "leg_mode", "boustrophedon")
        self._cov_legs = self._build_coverage_legs(plan.waypoints)
        self._cov_idx = 0
        self._coverage_complete = False
        self._transit = transit
        if self.state in (AgentState.S1_TRANSIT, AgentState.S2_MISSION, AgentState.S_OBS):
            self._set_legs([transit])
            self.state = AgentState.S1_TRANSIT
        elif self.state is AgentState.S0_IDLE:
            self._launch_ready = True

    def view(self) -> DroneStateView:
        return DroneStateView(self.id, self.battery.frac, self.pose, self.layer)

    # ------------------------------------------------------------------ #
    # external signals                                                   #
    # ------------------------------------------------------------------ #
    def signal_failure(self) -> None:
        self._failure = True

    def signal_threat(self, on: bool) -> None:
        if on and not self._threat and self.state.is_airborne and self.state is not AgentState.S_OBS:
            self._threat = True
        elif not on:
            self._threat = False

    def signal_swap_done(self) -> None:
        self._swap_done = True

    # ------------------------------------------------------------------ #
    # per-tick                                                           #
    # ------------------------------------------------------------------ #
    def step(self, dt: float, t: float, bus) -> None:
        if self.state is AgentState.S_FAIL:
            return

        self._tick_dynamics(dt, t)

        # avoidance micro-plan finished -> clear the threat so S_OBS can resume
        if self.state is AgentState.S_OBS and self._phase_done():
            self._threat_cleared = True

        # periodic RTH check while in coverage
        if self.state is AgentState.S2_MISSION and (t - self._last_rth_t) >= self.rth.check_interval_s:
            self._last_rth_t = t
            self._rth_decision = self.rth.decide(self) == "RETURN_NOW"
        else:
            self._rth_decision = getattr(self, "_rth_decision", False)

        ctx = self._make_ctx()
        tr = self.sm.step(ctx)
        if tr is not None:
            self._apply_transition(tr, t, bus)

    def _tick_dynamics(self, dt: float, t: float) -> None:
        if self.state in (AgentState.S0_IDLE, AgentState.S_SWAP, AgentState.S_FAIL):
            if self.state is AgentState.S0_IDLE:
                e = self.em.segment_energy(ManeuverType.IDLE, dt)
                self.battery.drain(e)
                self.energy_consumed_j += e
            return
        if self._leg_idx >= len(self._legs):
            return
        leg = self._legs[self._leg_idx]
        man = leg.maneuver_at_time(self._t) or ManeuverType.CRUISE
        f = self.formation.power_factor(self, t, man) if self.formation else 1.0
        e = self.em.segment_energy(man, dt, f)
        self.battery.drain(e)
        self.energy_consumed_j += e
        new_pose, new_t = self.motion.advance(leg, self._t, dt)
        if new_pose is not None:
            self.flown_m += math.dist(self.pose.as_xy(), new_pose.as_xy())
            self.pose = new_pose
        self._t = new_t
        if new_t >= leg.total_duration_s - 1e-9:
            self._leg_idx += 1
            self._t = 0.0
            if self.state is AgentState.S2_MISSION:
                self._cov_idx = self._leg_idx

    def _phase_done(self) -> bool:
        return self._leg_idx >= len(self._legs)

    def _make_ctx(self) -> AgentContext:
        return AgentContext(
            state=self.state,
            battery_zone=self.battery.zone,
            failure_flag=self._failure,
            threat_flag=self._threat,
            threat_cleared=self._threat_cleared,
            launch_command=self._launch_ready,
            plan_assigned=self.plan is not None,
            at_zone_entry=(self.state is AgentState.S1_TRANSIT and self._phase_done()),
            rth_decision=getattr(self, "_rth_decision", False),
            coverage_complete=(self.state is AgentState.S2_MISSION and self._phase_done()),
            landed_at_base=(self.state is AgentState.S3_RTH and self._phase_done()),
            own_plan_incomplete=(self._cov_idx < len(self._cov_legs)),
            swap_done=self._swap_done,
            obs_return_state=self._obs_return,
        )

    def _apply_transition(self, tr, t: float, bus) -> None:
        if self.recorder is not None:
            self.recorder.close(self.id, t, tr.reason)
        dst = tr.dst
        if dst is AgentState.S1_TRANSIT:
            self._launch_ready = False
            self._set_legs([self._transit] if self._transit is not None else [])
        elif dst is AgentState.S2_MISSION:
            self._set_legs(self._cov_legs[self._cov_idx:])
        elif dst is AgentState.S3_RTH:
            ret = self.motion.plan(self.pose, self.base, ManeuverType.CRUISE)
            self._set_legs([ret])
        elif dst is AgentState.S_SWAP:
            bus.publish(Event(EventType.SWAP_REQUEST, t, {"agent_id": self.id}))
            self._set_legs([])
        elif dst is AgentState.S0_IDLE:
            if tr.reason == "swap_done":
                self.battery.reset()
                self._swap_done = False
                # resume remaining coverage: transit from base to resume entry
                self._transit = self._resume_transit()
                self._launch_ready = True
            self._set_legs([])
        elif dst is AgentState.S_OBS:
            self._obs_return = self.state
            self._obs_legs_saved = (self._legs, self._leg_idx)
            self._threat = False
            self._set_legs([self._avoidance_plan()])
        elif dst is AgentState.S_FAIL:
            self._set_legs([])

        self.state = dst
        if self.recorder is not None:
            self.recorder.open(self.id, dst, t)

        if dst in (AgentState.S1_TRANSIT, AgentState.S2_MISSION, AgentState.S3_RTH) and \
                tr.src is AgentState.S_OBS and self._obs_legs_saved is not None:
            # resume the interrupted leg queue after avoidance
            self._legs, self._leg_idx = self._obs_legs_saved
            self._obs_legs_saved = None
            self._threat_cleared = False

    def _set_legs(self, legs: list[Path]) -> None:
        self._legs = [p for p in legs if p is not None]
        self._leg_idx = 0
        self._t = 0.0

    def _resume_transit(self) -> Path:
        if self._cov_idx < len(self._cov_legs):
            nxt = self._cov_legs[self._cov_idx]
            entry = nxt.start_pose or self.pose
        else:
            entry = self.pose
        return self.motion.plan(self.base, entry, ManeuverType.CRUISE)

    def _avoidance_plan(self) -> Path:
        # lateral sidestep then continue; effective at separating drones
        h = self.pose.heading
        lx, ly = -math.sin(h), math.cos(h)
        side = Pose(self.pose.x + 15 * lx + 10 * math.cos(h),
                    self.pose.y + 15 * ly + 10 * math.sin(h), h)
        return self.motion.plan(self.pose, side, ManeuverType.CRUISE)

    # ------------------------------------------------------------------ #
    # RTH lookahead                                                      #
    # ------------------------------------------------------------------ #
    def lookahead(self) -> tuple[float, Pose]:
        """Energy of the next coverage leg(s) and the pose at its end."""
        if self._cov_idx >= len(self._cov_legs):
            return 0.0, self.pose
        leg = self._cov_legs[self._cov_idx]
        e_next = self.em.path_energy(leg)
        p_next = leg.end_pose or self.pose
        # include the following connector if present
        if self._cov_idx + 1 < len(self._cov_legs):
            conn = self._cov_legs[self._cov_idx + 1]
            e_next += self.em.path_energy(conn)
            p_next = conn.end_pose or p_next
        return e_next, p_next

    def signal_threat_cleared(self) -> None:
        self._threat_cleared = True
