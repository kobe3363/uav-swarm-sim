"""Battery-swap service at the launch site -- the 'reincarnation' mechanism.

Depleted drone returns, swaps, rejoins as the same agent with the same remaining
plan. Queue waiting time counts as S_SWAP sojourn (the drone is at the station
either way), so station capacity pressure is visible in pi(S_SWAP) and hence in
the efficiency score -- the throughput interpretation chosen in Decision 2.

Shared battery pool (Phase 2, Task 2.1)
---------------------------------------
The station draws fresh packs from a finite shared reserve,
``total_reserve_batteries``, common to the whole fleet.

  * DECREMENT OWNER: the reserve is decremented in exactly ONE place -- the admit
    loop inside ``step()`` -- at the instant a queued drone is committed to a free
    bay (one reserve pack consumed per swap). No other method ever changes it.
  * EXHAUSTION: when a drone needs a pack but the reserve is empty, the station
    does NOT admit it and raises the monotonic ``pool_exhausted`` flag. The
    SimulationEngine polls that flag and is the component that actually records
    MISSION_FAILED and halts -- the station owns *detection*, never the
    mission-level terminal decision (that wiring lives in the run loop alongside
    Task 2.2's terminal-state evaluation).
  * COST MODEL UNCHANGED: swaps remain zero-energy and cost only service time
    (the drone has landed). The pool is a COUNT constraint, never an energy one.
  * ``total_reserve_batteries=None`` means an unbounded reserve and reproduces the
    pre-Phase-2 infinite-reserve behaviour byte-for-byte.
"""
from __future__ import annotations

from collections import deque

from ..infrastructure.config import SwapConfig
from ..infrastructure.core_types import Event, Pose
from ..infrastructure.enums import EventType


class SwapStation:
    def __init__(
        self,
        cfg: SwapConfig,
        base: Pose,
        total_reserve_batteries: int | None = None,
    ) -> None:
        self._service_time = cfg.service_time_s
        self._n_bays = cfg.n_bays
        self._base = base
        self._queue: deque[int] = deque()
        self._in_service: dict[int, float] = {}  # agent_id -> remaining service time
        # Finite shared reserve of fresh packs for the whole fleet. None =>
        # unbounded (pre-Phase-2 behaviour). Decremented ONLY in step()'s admit
        # loop; see the module docstring. ``_pool_exhausted`` is monotonic.
        self._reserve: int | None = total_reserve_batteries
        self._pool_exhausted: bool = False

    def request(self, agent_id: int, t: float) -> None:
        if agent_id in self._in_service or agent_id in self._queue:
            return
        # Immediate exhaustion signal: a drone is asking for a pack the shared
        # reserve can no longer provide. It is still enqueued (it IS at the
        # station, in S_SWAP) so state stays consistent until the engine halts.
        if self._reserve is not None and self._reserve <= 0:
            self._pool_exhausted = True
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
            # Cannot serve without a fresh pack: raise the exhaustion flag and
            # stop admitting. The queued drone(s) stay in S_SWAP until the engine
            # observes pool_exhausted and halts the mission with MISSION_FAILED.
            if self._reserve is not None and self._reserve <= 0:
                self._pool_exhausted = True
                break
            aid = self._queue.popleft()
            # ---- SOLE OWNER of the reserve decrement: one pack per admitted swap.
            if self._reserve is not None:
                self._reserve -= 1
            self._in_service[aid] = self._service_time

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    @property
    def busy_bays(self) -> int:
        return len(self._in_service)

    @property
    def pool_exhausted(self) -> bool:
        """True once a drone has needed a fresh pack the shared reserve could not
        provide. Monotonic. The SimulationEngine polls this each tick to record
        MISSION_FAILED and halt; the station itself never ends the mission."""
        return self._pool_exhausted

    @property
    def reserve_remaining(self) -> int | None:
        """Fresh packs left in the shared reserve; None when the reserve is
        unbounded. Read-only view for metrics/observability."""
        return self._reserve
