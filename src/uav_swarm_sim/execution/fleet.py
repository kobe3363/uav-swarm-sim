"""Homogeneous fleet container and active/failed registry.

The only place agents are created or removed. ``kill`` freezes a failed agent in
S_FAIL and removes it from ``active()`` -- it does NOT respawn it: in the physical
simulation failure is irreversible. The SMDP slot-replacement closure happens in
the metrics layer only.
"""
from __future__ import annotations

from ..infrastructure.core_types import DroneStateView
from ..infrastructure.enums import AgentState
from .agent import Agent


class Fleet:
    def __init__(self, agents: list[Agent]) -> None:
        self.agents: dict[int, Agent] = {a.id: a for a in agents}
        self._failed: set[int] = set()

    def active(self) -> list[Agent]:
        return [a for aid, a in self.agents.items() if aid not in self._failed]

    def airborne(self) -> list[Agent]:
        return [a for a in self.active() if a.state.is_airborne]

    def kill(self, agent_id: int, t: float) -> None:
        agent = self.agents.get(agent_id)
        if agent is None:
            return
        agent.state = AgentState.S_FAIL
        self._failed.add(agent_id)

    def views(self) -> list[DroneStateView]:
        return [a.view() for a in self.active()]

    @property
    def n_failed(self) -> int:
        return len(self._failed)
