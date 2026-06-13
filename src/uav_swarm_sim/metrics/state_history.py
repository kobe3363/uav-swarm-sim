"""The raw evidence for the SMDP layer: per agent-slot sequences of
(state, t_enter, t_exit) plus battery/position traces.

Implements the Recorder protocol the Agent calls (open/close). The sojourn time
recorded here is the backward recurrence time -- the second coordinate of the
SMDP's two-dimensional state (state, time-in-state). Storing full intervals lets
the estimator compute exact mean sojourns rather than binning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..infrastructure.enums import AgentState

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sojourn:
    agent_id: int
    state: AgentState
    t_in: float
    t_out: float
    reason_out: str

    @property
    def duration(self) -> float:
        return self.t_out - self.t_in


class StateHistory:
    def __init__(self) -> None:
        self._closed: list[Sojourn] = []
        self._open: dict[int, tuple[AgentState, float]] = {}
        self._battery: dict[int, list[tuple[float, float]]] = {}
        self._position: dict[int, list[tuple[float, float, float, AgentState]]] = {}
        self._dynobs: list[tuple[float, str, list[tuple[int, float, float]]]] = []

    # --- Recorder protocol ------------------------------------------------- #
    def open(self, agent_id: int, state: AgentState, t: float) -> None:
        if agent_id in self._open:
            # invariant: one open sojourn per agent -- close the previous defensively
            prev_state, t_in = self._open[agent_id]
            self._closed.append(Sojourn(agent_id, prev_state, t_in, t, "reopen"))
        self._open[agent_id] = (state, t)

    def close(self, agent_id: int, t: float, reason: str) -> None:
        entry = self._open.pop(agent_id, None)
        if entry is None:
            return  # tolerant: first transition may close an unopened initial state
        state, t_in = entry
        self._closed.append(Sojourn(agent_id, state, t_in, max(t, t_in), reason))

    def finalize(self, t_end: float) -> None:
        for agent_id, (state, t_in) in list(self._open.items()):
            self._closed.append(Sojourn(agent_id, state, t_in, max(t_end, t_in), "mission_end"))
        self._open.clear()

    # --- battery trace ----------------------------------------------------- #
    def record_battery(self, agent_id: int, t: float, frac: float) -> None:
        self._battery.setdefault(agent_id, []).append((t, frac))

    def battery_trace(self, agent_id: int) -> list[tuple[float, float]]:
        return self._battery.get(agent_id, [])

    # --- position trace (for 2D replay) ------------------------------------ #
    def record_position(self, agent_id: int, t: float, x: float, y: float, state: AgentState) -> None:
        """Log one (time, x, y, state) sample for an agent. Recorded every tick
        for every agent by the engine, so all agents' traces are time-aligned
        (same length, same timestamps) -- which is what makes frame-by-frame
        replay straightforward."""
        self._position.setdefault(agent_id, []).append((t, x, y, state))

    def position_trace(self, agent_id: int) -> list[tuple[float, float, float, AgentState]]:
        return self._position.get(agent_id, [])

    # --- dynamic-obstacle trace (for replay) ------------------------------- #
    def record_dynamic_obstacles(self, t: float, snapshot, mode) -> None:
        """One frame: (time, swarm mode name, [(obstacle_id, x, y), ...])."""
        self._dynobs.append((t, getattr(mode, "name", str(mode)), list(snapshot)))

    def dynamic_obstacle_frames(self):
        return self._dynobs

    # --- access ------------------------------------------------------------ #
    def sojourns(self) -> list[Sojourn]:
        return list(self._closed)
