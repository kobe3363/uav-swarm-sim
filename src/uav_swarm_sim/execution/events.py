"""Minimal synchronous event bus -- the spine of event-driven adaptivity.

Processing order is publication order (deterministic). Redistribution triggers
are exactly {FAILURE, NEW_TASK}; SWAP_* events never reach the redistributor, so
the reversibility of battery swaps is enforced by wiring, not convention.
"""
from __future__ import annotations

import logging

from ..infrastructure.core_types import Event
from ..infrastructure.enums import EventType


class EventBus:
    def __init__(self) -> None:
        self._queue: list[Event] = []
        self._logger: logging.Logger | None = None

    def publish(self, e: Event) -> None:
        if self._logger is not None:
            self._logger.debug("event %s t=%.2f %s", e.type.value, e.t, e.payload)
        self._queue.append(e)

    def drain(self) -> list[Event]:
        out = self._queue
        self._queue = []
        return out

    def subscribe_log(self, logger: logging.Logger) -> None:
        self._logger = logger
