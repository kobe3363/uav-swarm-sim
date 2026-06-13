"""Owns formation membership through mission phases.

Decides who flies in formation when, and therefore when the FW/VTOL energy
benefit applies (the scope limitation of guideline 1.3 enforced in one place):
formation during launch/transit and episodic RTH, never during coverage. For
multirotor the manager still tracks grouping (for wake geometry) but the power
factor is always 1.0.
"""
from __future__ import annotations

import math

from ..infrastructure.config import AeroConfig
from ..infrastructure.core_types import Pose
from ..infrastructure.enums import AgentState, ManeuverType, PlatformType
from ..physical_model.aero_correction import AeroCorrection


class FormationManager:
    def __init__(self, aero: AeroCorrection, cfg: AeroConfig, platform: PlatformType) -> None:
        self._aero = aero
        self._cfg = cfg
        self._platform = platform
        self._departing: set[int] = set()
        self._leader: int | None = None

    def register_departure(self, agents, shared_routes: dict | None = None) -> None:
        """Mark a group of agents as departing together (a transit formation).
        The lowest-id agent is the leader; followers ride the wake benefit."""
        ids = sorted(a.id for a in agents)
        self._departing = set(ids)
        self._leader = ids[0] if ids else None

    def in_formation(self, agent, t: float) -> bool:
        # coverage is always dispersed
        if agent.state is AgentState.S2_MISSION:
            return False
        # transit/RTH with >=2 grouped agents: followers are in formation
        if agent.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH):
            return (
                len(self._departing) >= 2
                and agent.id in self._departing
                and agent.id != self._leader
            )
        return False

    def power_factor(self, agent, t: float, maneuver: ManeuverType) -> float:
        return self._aero.power_factor(self.in_formation(agent, t), maneuver)

    def stations(self, t: float) -> dict[int, Pose]:
        """Beneficial wake stations (FW/VTOL). Placeholder echelon offsets; the
        engine can refine with live leader poses."""
        return {}
