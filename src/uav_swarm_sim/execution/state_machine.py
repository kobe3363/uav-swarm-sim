"""The seven-state behavioral automaton.

Base set (C. Liu et al.): S0_IDLE, S1_TRANSIT, S2_MISSION, S_FAIL.
Author's extensions: S3_RTH, S_OBS, S_SWAP.

Guards are evaluated top-down (first match wins). The closed loop is structural:
S3 -> S_SWAP -> S0 is realized here. S_FAIL is terminal in the *physical*
simulation (the agent is removed, its zone redistributed); the S_FAIL -> S0
replacement closure exists only in the SMDP analysis layer (Batch 5), by design.
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


@dataclass(frozen=True)
class Transition:
    src: AgentState
    dst: AgentState
    reason: str


# The designed transition structure (physical layer). The SMDP estimator checks
# the observed chain against this and adds the synthetic S_FAIL -> S0 closure.
ALLOWED: set[tuple[AgentState, AgentState]] = {
    (S.S0_IDLE, S.S1_TRANSIT),
    (S.S1_TRANSIT, S.S2_MISSION),
    (S.S1_TRANSIT, S.S_OBS),
    (S.S1_TRANSIT, S.S3_RTH),
    (S.S1_TRANSIT, S.S_FAIL),
    (S.S2_MISSION, S.S3_RTH),
    (S.S2_MISSION, S.S_OBS),
    (S.S2_MISSION, S.S_FAIL),
    (S.S3_RTH, S.S_OBS),
    (S.S3_RTH, S.S_SWAP),
    (S.S3_RTH, S.S0_IDLE),
    (S.S3_RTH, S.S_FAIL),
    (S.S_OBS, S.S1_TRANSIT),
    (S.S_OBS, S.S2_MISSION),
    (S.S_OBS, S.S3_RTH),
    (S.S_OBS, S.S_FAIL),
    (S.S_SWAP, S.S0_IDLE),
}


class StateMachine:
    ALLOWED = ALLOWED

    def __init__(self, zones_cfg: BatteryZonesConfig) -> None:
        self._zones = zones_cfg

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

        if s is S.S2_MISSION:
            if ctx.threat_flag:
                return Transition(s, S.S_OBS, "obstacle_threat")
            if ctx.battery_zone is BatteryZone.TERMINAL:
                return Transition(s, S.S3_RTH, "terminal_battery")  # anomaly: RTH should pre-empt
            if ctx.rth_decision:
                return Transition(s, S.S3_RTH, "rth_energy")
            if ctx.battery_zone is BatteryZone.CRITICAL:
                return Transition(s, S.S3_RTH, "critical_battery")
            if ctx.coverage_complete:
                return Transition(s, S.S3_RTH, "coverage_complete")
            return None

        if s is S.S3_RTH:
            if ctx.threat_flag:
                return Transition(s, S.S_OBS, "obstacle_threat")
            if ctx.landed_at_base:
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

        # S_FAIL: terminal in the physical layer
        return None
