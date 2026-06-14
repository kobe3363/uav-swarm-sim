"""Owns formation membership through mission phases.

Decides who flies in formation when, and therefore when the FW/VTOL energy
benefit applies (the scope limitation of guideline 1.3 enforced in one place):
formation during launch/transit and episodic RTH, never during coverage. For
multirotor the manager still tracks grouping (for wake geometry) but the power
factor is always 1.0.

2.5D (Batch 4): formations are intra-layer. Drones at different coverage
altitudes cannot share a wake, so departing agents are grouped BY LAYER, each
group with its own (lowest-id) leader. With a single layer this is one group with
the lowest-id leader -- identical to the 2D behaviour.
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
        # per-layer departing groups and their leaders
        self._departing_by_layer: dict[int, set[int]] = {}
        self._leader_by_layer: dict[int, int] = {}

    def register_departure(self, agents, shared_routes: dict | None = None) -> None:
        """Mark agents as departing together, grouped by layer. The lowest-id
        agent in each layer is that layer's leader; followers ride the wake."""
        self._departing_by_layer = {}
        self._leader_by_layer = {}
        by_layer: dict[int, list[int]] = {}
        for a in agents:
            by_layer.setdefault(getattr(a, "layer", 0), []).append(a.id)
        for layer, ids in by_layer.items():
            ids_sorted = sorted(ids)
            self._departing_by_layer[layer] = set(ids_sorted)
            self._leader_by_layer[layer] = ids_sorted[0] if ids_sorted else None

    def in_formation(self, agent, t: float) -> bool:
        # coverage is always dispersed
        if agent.state is AgentState.S2_MISSION:
            return False
        # transit/RTH with >=2 grouped agents on the SAME layer: followers benefit
        if agent.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH):
            layer = getattr(agent, "layer", 0)
            group = self._departing_by_layer.get(layer, set())
            leader = self._leader_by_layer.get(layer)
            return (
                len(group) >= 2
                and agent.id in group
                and agent.id != leader
            )
        return False

    def power_factor(self, agent, t: float, maneuver: ManeuverType) -> float:
        return self._aero.power_factor(self.in_formation(agent, t), maneuver)

    def stations(self, t: float) -> dict[int, Pose]:
        """Beneficial wake stations (FW/VTOL). Placeholder echelon offsets; the
        engine can refine with live leader poses."""
        return {}
