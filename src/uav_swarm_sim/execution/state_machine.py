"""The eight-state behavioral automaton.

Base set (C. Liu et al.): S0_IDLE, S1_TRANSIT, S2_MISSION, S_FAIL.
Author's extensions: S3_RTH, S_OBS, S_SWAP, S_FERRY.

Guards are evaluated top-down (first match wins). The closed loop is structural:
S3 -> S_SWAP -> S0 is realized here. S_FAIL is terminal in the *physical*
simulation (the agent is removed, its zone redistributed); the S_FAIL -> S0
replacement closure exists only in the SMDP analysis layer (Batch 5), by design.

EXP-04 (``no_swap_mode``): with the flag on, touchdown in S3_RTH goes to the
terminal S_LANDED state instead of S_SWAP / S0_IDLE -- the drone keeps its own
battery and never relaunches. S_LANDED has no outgoing edge in the physical
layer; like S_FAIL, its S0 closure exists only in the SMDP analysis layer.

EXP-08 (``repartition_enabled``): an in-flight re-partition re-tasks a covering
drone, which then transits to its new zone. That adds two edges to the designed
structure -- S2_MISSION -> S1_TRANSIT and S_FERRY -> S1_TRANSIT -- and, unlike
the pre-EXP-08 re-task, the move is applied as a real recorded transition rather
than by assigning ``agent.state`` behind the recorder's back. The guards in
``step`` do not produce these edges: they are driven by the engine
(``Agent.retask``), which is why they are listed in ALLOWED but appear in no
branch below. The thesis FSM figure needs both.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..infrastructure.config import BatteryZonesConfig
from ..infrastructure.enums import AgentState, BatteryZone

S = AgentState


@dataclass
class AgentContext:
    state: AgentState
    battery_zone: BatteryZone
    failure_flag: bool = False
    threat_flag: bool = False
    threat_cleared: bool = False
    launch_command: bool = False
    plan_assigned: bool = False
    at_zone_entry: bool = False
    rth_decision: bool = False
    coverage_complete: bool = False
    landed_at_base: bool = False
    own_plan_incomplete: bool = False
    swap_done: bool = False
    obs_return_state: AgentState = AgentState.S1_TRANSIT
    on_connector: bool = False  # active coverage leg is a camera-off connector (S2<->S_FERRY toggle)
    # EXP-08: the agent has announced ZONE_COMPLETE and the engine has one tick
    # to hand it a new zone. Defers ONLY the coverage_complete -> S3_RTH edge;
    # every safety and energy guard is evaluated ahead of it and is unaffected.
    repartition_pending: bool = False
    emergency_battery: bool | None = None  # None preserves the legacy reporting-zone guard


@dataclass(frozen=True)
class Transition:
    src: AgentState
    dst: AgentState
    reason: str


# The designed transition structure (physical layer).
#
# NOTE on what does and does not consume this set. The SMDP estimator adds the
# synthetic S_FAIL -> S0 closure (metrics/smdp_estimator.py), but it does NOT
# check the observed chain against ALLOWED -- nothing in ``src`` reads this set
# at all; only the unit tests do, and they check the transitions the machine
# PRODUCES. Until EXP-08 that gap was invisible and load-bearing: re-tasking a
# live agent replaced its state by direct assignment, bypassing both the
# recorder and this table, so an observed chain that this table forbids was
# never noticed. The comment used to claim the estimator enforced it; a comment
# asserting a mechanism that does not exist is the same "the output lies" defect
# class this table is meant to guard against, so it is corrected here rather
# than left standing.
ALLOWED: set[tuple[AgentState, AgentState]] = {
    (S.S0_IDLE, S.S1_TRANSIT),
    (S.S1_TRANSIT, S.S2_MISSION),
    (S.S1_TRANSIT, S.S_OBS),
    (S.S1_TRANSIT, S.S3_RTH),
    (S.S1_TRANSIT, S.S_FAIL),
    (S.S2_MISSION, S.S3_RTH),
    (S.S2_MISSION, S.S_OBS),
    (S.S2_MISSION, S.S_FAIL),
    (S.S2_MISSION, S.S_FERRY),
    # EXP-08 (mission.repartition_enabled): a re-partition hands a covering
    # drone a DIFFERENT zone, and it transits to the new zone before covering
    # it. Both halves of the coverage phase can be re-tasked -- keying this on
    # S2_MISSION alone would make eligibility depend on _cov_idx parity, i.e. on
    # whether the drone happens to be on a strip or on the connector between
    # two. These two edges exist only under the flag; a legacy run never
    # produces them.
    (S.S2_MISSION, S.S1_TRANSIT),
    (S.S_FERRY, S.S2_MISSION),
    (S.S_FERRY, S.S3_RTH),
    (S.S_FERRY, S.S_OBS),
    (S.S_FERRY, S.S_FAIL),
    (S.S_FERRY, S.S1_TRANSIT),   # EXP-08 re-task, see above
    (S.S3_RTH, S.S_OBS),
    (S.S3_RTH, S.S_SWAP),
    (S.S3_RTH, S.S0_IDLE),
    (S.S3_RTH, S.S_LANDED),  # EXP-04 no_swap_mode only
    (S.S3_RTH, S.S_FAIL),
    (S.S_OBS, S.S1_TRANSIT),
    (S.S_OBS, S.S2_MISSION),
    (S.S_OBS, S.S_FERRY),
    (S.S_OBS, S.S3_RTH),
    (S.S_OBS, S.S_FAIL),
    (S.S_SWAP, S.S0_IDLE),
}


class StateMachine:
    ALLOWED = ALLOWED

    def __init__(
        self,
        zones_cfg: BatteryZonesConfig,
        *,
        zone_demotion: bool = False,
        no_swap_mode: bool = False,
    ) -> None:
        self._zones = zones_cfg
        # EM-01 B1: when True (rth.energy_map.zone_demotion, requires decide) the
        # static CRITICAL net is removed from _coverage_guards so the dynamic map
        # governs the normal energy return. Default False => the guard is byte-
        # identical to pre-B1 and every existing call site is unchanged.
        self._zone_demotion = zone_demotion
        # EXP-04 (mission.no_swap_mode): touchdown is terminal (S_LANDED).
        # Default False => the S3_RTH branch is byte-identical to pre-EXP-04.
        self._no_swap = no_swap_mode

    def _coverage_guards(self, s: AgentState, ctx: AgentContext) -> Transition | None:
        """Triggers that interrupt coverage from EITHER S2_MISSION or S_FERRY.

        The dynamic route-vs-return reserve (guideline 3.1) is the PRIMARY early-
        return trigger and pre-empts the crude battery-zone nets, so a return it
        triggers is attributed to the live energy calculation.

        Two regimes for the battery-zone nets, by ``zone_demotion`` (EM-01 B1):
          * DEFAULT (``zone_demotion=False``): the nets are progressively-severe
            last-resort catches. CRITICAL (the higher threshold, 0.40) is tested
            before TERMINAL (0.20), so a normally-draining drone returns at the
            CRITICAL boundary and never reaches TERMINAL while still covering.
          * DEMOTED (``zone_demotion=True``, map deciding): the CRITICAL net is
            removed, so the dynamic map (``rth_energy``) governs the normal return
            and the usable sortie deepens to the 0.20 floor. TERMINAL remains the
            last-resort failsafe and CAN now legitimately fire while covering
            (on cheap near-base legs where the map's reserve sits below 0.20).
        Guard ORDER is otherwise identical in both regimes. (Irreversible failure
        is handled for all airborne states in ``step``.)
        """
        if ctx.threat_flag:
            return Transition(s, S.S_OBS, "obstacle_threat")
        if ctx.rth_decision:
            return Transition(s, S.S3_RTH, "rth_energy")
        if not self._zone_demotion and ctx.battery_zone is BatteryZone.CRITICAL:
            return Transition(s, S.S3_RTH, "critical_battery")
        if (ctx.battery_zone is BatteryZone.TERMINAL if ctx.emergency_battery is None
                else ctx.emergency_battery):
            return Transition(s, S.S3_RTH, "terminal_battery")
        return None

    def step(self, ctx: AgentContext) -> Transition | None:
        s = ctx.state

        # highest priority: irreversible failure from any airborne state
        if ctx.failure_flag and s.is_airborne:
            return Transition(s, S.S_FAIL, "failure")

        if s is S.S0_IDLE:
            if ctx.launch_command and ctx.plan_assigned:
                return Transition(s, S.S1_TRANSIT, "launch")
            return None

        if s is S.S1_TRANSIT:
            if ctx.threat_flag:
                return Transition(s, S.S_OBS, "obstacle_threat")
            if ctx.at_zone_entry:
                return Transition(s, S.S2_MISSION, "zone_entry")
            return None

        if s in (S.S2_MISSION, S.S_FERRY):
            # guards that interrupt coverage apply identically whether the drone is
            # filming a strip (S2_MISSION) or ferrying between strips (S_FERRY)
            g = self._coverage_guards(s, ctx)
            if g is not None:
                return g
            # camera on/off toggle, driven by the active leg: a COVERAGE strip is
            # productive (camera on, S2_MISSION); a connector is camera-off
            # repositioning (S_FERRY) -- real flight energy, no coverage benefit.
            if s is S.S2_MISSION and ctx.on_connector:
                return Transition(s, S.S_FERRY, "ferry_start")
            if s is S.S_FERRY and not ctx.on_connector:
                return Transition(s, S.S2_MISSION, "ferry_end")
            if ctx.coverage_complete:
                # EXP-08: hold the automatic return for exactly one tick while a
                # re-partition decides whether there is more work for this drone.
                # Reached only under mission.repartition_enabled, and only AFTER
                # the guards above -- a threat, the dynamic RTH decision and both
                # battery nets all still fire immediately, so the hold can never
                # keep a drone airborne that should be going home. If no re-task
                # arrives, the flag is cleared and this edge fires next tick.
                if ctx.repartition_pending:
                    return None
                return Transition(s, S.S3_RTH, "coverage_complete")
            return None

        if s is S.S3_RTH:
            if ctx.threat_flag:
                return Transition(s, S.S_OBS, "obstacle_threat")
            if ctx.landed_at_base:
                if self._no_swap:
                    # EXP-04: same-battery lifecycle -- touchdown is terminal
                    # whether or not the plan was finished (no swap, no relaunch).
                    return Transition(s, S.S_LANDED, "landed")
                if ctx.own_plan_incomplete:
                    return Transition(s, S.S_SWAP, "swap")
                return Transition(s, S.S0_IDLE, "mission_done")
            return None

        if s is S.S_OBS:
            if ctx.threat_cleared:
                ret = ctx.obs_return_state
                return Transition(s, ret, "threat_cleared")
            return None

        if s is S.S_SWAP:
            if ctx.swap_done:
                return Transition(s, S.S0_IDLE, "swap_done")  # closes S3 -> S_SWAP -> S0
            return None

        # S_FAIL and S_LANDED (EXP-04): terminal in the physical layer -- no
        # guard (swap_done, launch_command, failure_flag) ever leaves them.
        return None
