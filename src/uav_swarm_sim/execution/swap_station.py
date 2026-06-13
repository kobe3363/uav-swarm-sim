"""Battery-swap service at the launch site -- the 'reincarnation' mechanism.

Depleted drone returns, swaps, rejoins as the same agent with the same remaining
plan. Queue waiting time counts as S_SWAP sojourn (the drone is at the station
either way), so station capacity pressure is visible in pi(S_SWAP) and hence in
the efficiency score -- the throughput interpretation chosen in Decision 2.
"""
from __future__ import annotations

from collections import deque

from ..infrastructure.config import SwapConfig
from ..infrastructure.core_types import Event, Pose
from ..infrastructure.enums import EventType


class SwapStation:
    def __init__(self, cfg: SwapConfig, base: Pose) -> None:
        self._service_time = cfg.service_time_s
        self._n_bays = cfg.n_bays
        self._base = base
        self._queue: deque[int] = deque()
        self._in_service: dict[int, float] = {}  # agent_id -> remaining service time

    def request(self, agent_id: int, t: float) -> None:
        if agent_id in self._in_service or agent_id in self._queue:
            return
        self._queue.append(agent_id)

    def step(self, dt: float, bus) -> None:
        # progress active services
        done = []
        for aid in list(self._in_service):
            self._in_service[aid] -= dt
            if self._in_service[aid] <= 0:
                done.append(aid)
        for aid in done:
            del self._in_service[aid]
            bus.publish(Event(EventType.SWAP_DONE, 0.0, {"agent_id": aid}))
        # admit from queue into free bays
        while self._queue and len(self._in_service) < self._n_bays:
            aid = self._queue.popleft()
            self._in_service[aid] = self._service_time

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    @property
    def busy_bays(self) -> int:
        return len(self._in_service)
