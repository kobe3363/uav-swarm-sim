"""Stochastic agent failure -- the hazard process that populates S_FAIL.

Memoryless per-tick hazard: per airborne agent, fail with p = 1 - exp(-lambda*dt/3600)
(lambda per flight-hour; ground states immune). The constant rate keeps the
failure-arrival process Markov; the semi-Markov character of the system comes
from the sojourn-time/battery memory of the *other* states.
"""
from __future__ import annotations

import math

import numpy as np

from ..infrastructure.config import FailureConfig
from ..infrastructure.core_types import Event
from ..infrastructure.enums import EventType


class FailureModel:
    def __init__(self, cfg: FailureConfig, rng: np.random.Generator) -> None:
        self._lambda = cfg.hazard_rate_per_hour
        self._rng = rng

    def step(self, airborne, dt: float, t: float, bus) -> None:
        if self._lambda <= 0:
            return
        p = 1.0 - math.exp(-self._lambda * dt / 3600.0)
        for agent in airborne:
            if self._rng.random() < p:
                bus.publish(Event(EventType.FAILURE, t, {"agent_id": agent.id}))
