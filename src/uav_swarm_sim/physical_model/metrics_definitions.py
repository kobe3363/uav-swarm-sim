"""Platform-independent definitions of the unified metric set (guideline 1.4).

These live at the physical-model level because the thesis places the evaluation
metrics there and they are platform-independent, which is what makes algorithm
comparisons across scale tiers commensurable. ``metrics/mission_metrics.py``
(Batch 5) computes them from recorded simulation state using these definitions.

The three metrics:
  * total mission energy (J)        -- sum of energy consumed by all drones
  * mission duration (s)            -- wall-clock of the mission
  * workload distribution std (m)   -- std of per-drone trajectory length
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

METRIC_NAMES: tuple[str, str, str] = ("total_energy_j", "duration_s", "workload_std_m")


def total_energy(per_agent_energy_j: Iterable[float]) -> float:
    """Total mission energy = sum over drones of consumed energy."""
    return float(sum(per_agent_energy_j))


def workload_std(per_agent_length_m: Mapping[int, float], ddof: int = 0) -> float:
    """Workload-distribution metric: standard deviation of per-drone trajectory
    lengths. Population std (ddof=0) by default -- it describes the dispersion of
    the realized assignment, not an inference about a larger population.
    """
    vals = list(per_agent_length_m.values())
    n = len(vals)
    if n == 0:
        return 0.0
    if n - ddof <= 0:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - ddof)
    return var ** 0.5
