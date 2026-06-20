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

2.5D (Batch 5a): flown distance is accumulated in 3D (it includes any altitude
change of the executed segment). Because flight is in the z=0 plane today, the 3D
and 2D path lengths coincide and the length metric stays byte-identical; the
moment inter-layer climbs are flown as legs, their vertical extent is counted.
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

# Task 2.5 Q2 (stateful S_OBS recovery): if the validated avoidance + leg-skip
# still cannot clear the corridor after this many CONSECUTIVE obstacle re-entries
# (no clean coverage progress in between), the drone is boxed in -- it abandons
# the region and returns home (S_OBS -> S3_RTH) instead of thrashing to depletion.
# Only consulted when the SafetyMonitor sends recovery-mode threat signals
# (safety.obstacle_recovery enabled); otherwise it is inert and the agent's S_OBS
# behaviour is byte-identical to the pre-Q2 baseline.
_OBS_REENTRY_BUDGET = 6


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

        # Task 2.5 Q2 (stateful S_OBS recovery). Dormant unless the SafetyMonitor
        # sends recovery-mode threat signals; all three default to the inert state
        # so the pre-Q2 S_OBS behaviour is byte-identical when the flag is off.
        self._obs_avoidance: Path | None = None   # validated EVADE plan from the monitor
        self._obs_skip_leg: bool = False          # REJOIN: skip the obstructed coverage leg
        self._obs_reentries: int = 0              # consecutive obstacle re-entries (escalation)

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

    def signal_threat(self, on: bool, avoidance: Path | None = None,
                      skip_leg: bool = False) -> None:
        # The optional ``avoidance``/``skip_leg`` carry the SafetyMonitor's
        # recovery decision (Task 2.5 Q2). With the default arguments this is the
        # original signal exactly, so the off path is unchanged.
        if on and not self._threat and self.state.is_airborne and self.state is not AgentState.S_OBS:
            self._threat = True
            self._obs_avoidance = avoidance
            self._obs_skip_leg = skip_leg
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
            # 2.5D path length: measure displacement in 3D so an executed
            # altitude change counts. Flight is in the z=0 plane today, so this
            # equals the 2D distance and the length metric stays byte-identical.
            self.flown_m += math.dist(self.pose.as_xyz(), new_pose.as_xyz())
            self.pose = new_pose
        self._t = new_t
        if new_t >= leg.total_duration_s - 1e-9:
            self._leg_idx += 1
            self._t = 0.0
            if self.state is AgentState.S2_MISSION:
                self._cov_idx = self._leg_idx
                # clean coverage progress -> reset the Q2 thrash counter (guarded so
                # it is a no-op, and thus byte-identical, when recovery is off).
                if self._obs_reentries:
                    self._obs_reentries = 0

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
            self._threat = False
            # EVADE: prefer the monitor's obstacle-aware validated detour; fall back
            # to the blind lateral sidestep (the only behaviour when recovery is off).
            plan = self._obs_avoidance if self._obs_avoidance is not None else self._avoidance_plan()
            if self._obs_skip_leg:
                self._obs_reentries += 1
                if self._obs_reentries > _OBS_REENTRY_BUDGET:
                    # boxed in: abandon the region and return home (reuses the
                    # existing S_OBS -> S3_RTH transition via obs_return_state).
                    self._obs_return = AgentState.S3_RTH
                    self._obs_legs_saved = None
                    self._obs_reentries = 0
                else:
                    self._obs_legs_saved = (self._legs, self._leg_idx)
            else:
                self._obs_legs_saved = (self._legs, self._leg_idx)
            self._set_legs([plan])
        elif dst is AgentState.S_FAIL:
            self._set_legs([])

        self.state = dst
        if self.recorder is not None:
            self.recorder.open(self.id, dst, t)

        if dst in (AgentState.S1_TRANSIT, AgentState.S2_MISSION, AgentState.S3_RTH) and \
                tr.src is AgentState.S_OBS and self._obs_legs_saved is not None:
            saved_legs, saved_idx = self._obs_legs_saved
            if self._obs_skip_leg and dst is AgentState.S2_MISSION:
                # REJOIN: do NOT re-fly the obstructed coverage leg (it caused the
                # threat and cannot be covered anyway) -- advance past it. This is
                # what structurally kills the S_OBS ping-pong: coverage legs are
                # consumed monotonically, so a blocked strip is never retried.
                self._legs = saved_legs
                self._leg_idx = min(saved_idx + 1, len(saved_legs))
                self._t = 0.0
                self._cov_idx = self._leg_idx
            else:
                # original behaviour: resume the interrupted leg queue after avoidance
                self._legs, self._leg_idx = saved_legs, saved_idx
            self._obs_legs_saved = None
            self._threat_cleared = False
        # leaving avoidance: clear the Q2 sub-state (idempotent -> a no-op, hence
        # byte-identical, when recovery is off since these are already False/None).
        if tr.src is AgentState.S_OBS:
            self._threat_cleared = False
            self._obs_skip_leg = False
            self._obs_avoidance = None

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
