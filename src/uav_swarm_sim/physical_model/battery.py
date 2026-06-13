"""Per-drone energy store: continuous drain plus four-zone discretization.

The continuous battery level is the hidden memory that breaks the memoryless
property -- which is why the analysis layer (Batch 5) is semi-Markov, not a plain
Markov chain. The four zones are coarse guards / reporting bins; the fine
decision instrument is the dynamic RTH calculator.
"""
from __future__ import annotations

from ..infrastructure.config import BatteryZonesConfig
from ..infrastructure.enums import BatteryZone


class Battery:
    def __init__(self, capacity_j: float, zones: BatteryZonesConfig, initial_frac: float = 1.0) -> None:
        if capacity_j <= 0:
            raise ValueError("capacity_j must be > 0")
        self._cap = capacity_j
        self._zones = zones
        self._level = max(0.0, min(1.0, initial_frac)) * capacity_j

    def drain(self, energy_j: float) -> None:
        self._level = max(0.0, self._level - max(0.0, energy_j))

    def reset(self) -> None:
        """Battery swap: full charge restored (same drone resumes)."""
        self._level = self._cap

    @property
    def capacity_j(self) -> float:
        return self._cap

    @property
    def level_j(self) -> float:
        return self._level

    @property
    def frac(self) -> float:
        return self._level / self._cap

    @property
    def zone(self) -> BatteryZone:
        f = self.frac
        if f >= self._zones.high:
            return BatteryZone.HIGH
        if f >= self._zones.nominal:
            return BatteryZone.NOMINAL
        if f >= self._zones.critical:
            return BatteryZone.CRITICAL
        return BatteryZone.TERMINAL
