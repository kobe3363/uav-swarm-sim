"""The unified deterministic metric set (guideline 1.4), computed from recorded
simulation state. Identical across all scales and algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..infrastructure.enums import AgentState
from ..physical_model.metrics_definitions import total_energy, workload_std
from .state_history import StateHistory


@dataclass(frozen=True)
class MissionMetrics:
    total_energy_j: float
    duration_s: float
    workload_std_m: float
    per_agent_length_m: dict[int, float]
    n_swaps: int
    n_failures: int
    coverage_frac: float
    planning_time_s: float
    replan_times_s: tuple[float, ...] = ()
    # Per-drone consumed energy (J). Appended with a default so every existing
    # constructor call site stays valid; sums to ``total_energy_j`` by
    # construction. Primary input for the S5 executed energy-imbalance metric
    # (max/mean) -- the thesis measures ENERGY, flown length is only a proxy.
    per_agent_energy_j: dict[int, float] = field(default_factory=dict)


def compute(
    history: StateHistory,
    fleet,
    partition,
    t_end: float,
    planning_time_s: float = 0.0,
    replan_times_s: tuple[float, ...] = (),
    coverage_frac: float = 1.0,
) -> MissionMetrics:
    agents = list(fleet.agents.values())
    per_agent_length = {a.id: a.flown_m for a in agents}
    per_agent_energy = {a.id: a.energy_consumed_j for a in agents}

    sojourns = history.sojourns()
    n_swaps = sum(1 for s in sojourns if s.state is AgentState.S_SWAP)
    n_failures = sum(1 for s in sojourns if s.state is AgentState.S_FAIL)
    # fall back to the fleet registry if history has no S_FAIL sojourn yet
    n_failures = max(n_failures, getattr(fleet, "n_failed", 0))

    return MissionMetrics(
        total_energy_j=total_energy(per_agent_energy.values()),
        duration_s=t_end,
        workload_std_m=workload_std(per_agent_length),
        per_agent_length_m=per_agent_length,
        n_swaps=n_swaps,
        n_failures=n_failures,
        coverage_frac=coverage_frac,
        planning_time_s=planning_time_s,
        replan_times_s=tuple(replan_times_s),
        per_agent_energy_j=per_agent_energy,
    )
