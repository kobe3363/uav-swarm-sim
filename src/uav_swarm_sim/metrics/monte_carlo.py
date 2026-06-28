"""Monte-Carlo experiment driver: N≈1000 replications with the CI-based
convergence criterion (Decision 3c).

Decoupled from the engine: ``run`` takes a per-replication callable
``run_once(replication) -> SingleRunResult``. The Batch-6 engine provides a thin
adapter that builds a SimulationEngine, estimates the SMDP, computes the
stationary distribution and efficiency, and returns a SingleRunResult. This
keeps the convergence/aggregation logic testable in isolation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..infrastructure.config import MCConfig
from ..infrastructure.enums import AgentState
from .convergence import ci_half_width, converged
from .smdp_estimator import STATE_ORDER, estimate
from .stationary_distribution import stationary
from .efficiency_score import efficiency


@dataclass
class SingleRunResult:
    states: list[AgentState]
    pi_time: dict[AgentState, float]
    efficiency: float
    metrics: object | None = None
    aborted: bool = False
    outcome: object | None = None  # Outcome enum (mission terminal outcome), distinct from `aborted`


@dataclass
class MCResult:
    n_runs: int
    converged: bool
    pi_time_mean: dict[AgentState, float]
    pi_time_ci: dict[AgentState, float]
    efficiency_mean: float
    efficiency_ci: float
    aborted_frac: float
    convergence_trace: list[tuple[int, float, float]]
    runs: list[SingleRunResult] = field(default_factory=list)


def run(run_once: Callable[[int], SingleRunResult], mc_cfg: MCConfig) -> MCResult:
    s2_samples: list[float] = []
    eff_samples: list[float] = []
    pi_accum: dict[AgentState, list[float]] = defaultdict(list)
    trace: list[tuple[int, float, float]] = []
    runs: list[SingleRunResult] = []
    did_converge = False

    for k in range(1, mc_cfg.n_max + 1):
        r = run_once(k)
        runs.append(r)
        s2 = r.pi_time.get(AgentState.S2_MISSION, 0.0)
        s2_samples.append(s2)
        if np.isfinite(r.efficiency):
            eff_samples.append(r.efficiency)
        for st in STATE_ORDER:
            pi_accum[st].append(r.pi_time.get(st, 0.0))

        hw = ci_half_width(s2_samples)
        trace.append((k, float(np.mean(s2_samples)), hw))
        if converged(s2_samples, mc_cfg.ci_tolerance, mc_cfg.n_min):
            did_converge = True
            break

    pi_mean = {st: float(np.mean(v)) for st, v in pi_accum.items()}
    pi_ci = {st: ci_half_width(v) for st, v in pi_accum.items()}
    eff_mean = float(np.mean(eff_samples)) if eff_samples else float("nan")
    eff_ci = ci_half_width(eff_samples) if len(eff_samples) >= 2 else float("inf")
    aborted_frac = sum(1 for r in runs if r.aborted) / len(runs) if runs else 0.0

    return MCResult(
        n_runs=len(runs),
        converged=did_converge,
        pi_time_mean=pi_mean,
        pi_time_ci=pi_ci,
        efficiency_mean=eff_mean,
        efficiency_ci=eff_ci,
        aborted_frac=aborted_frac,
        convergence_trace=trace,
        runs=runs,
    )


def single_run_from_history(
    history, close_failure_loop: bool = True, failure_repair_s: float = 600.0
) -> SingleRunResult:
    """Adapter: history -> SMDP estimate -> stationary pi -> efficiency.

    Used by the Batch-6 engine wrapper and by tests. Returns an aborted result
    (efficiency NaN) if the chain is not ergodic.
    """
    est = estimate(history, close_failure_loop, failure_repair_s)
    try:
        _, pi_time = stationary(est)
    except ValueError:
        return SingleRunResult(est.states, {}, float("nan"), aborted=True)
    pi_map = {s: float(pi_time[i]) for i, s in enumerate(est.states)}
    eff = efficiency(pi_time, est.states)
    return SingleRunResult(est.states, pi_map, eff)
