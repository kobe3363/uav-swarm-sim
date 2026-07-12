"""Head-to-head comparison harness (binds the engine to Monte Carlo).

Every comparison runs Monte Carlo per variant on the SAME replication seeds
(paired design): environment and failure draws are identical across compared
variants at each replication, so differences are attributable to the variant, not
the noise. Produces simple result tables consumable by the experiment CLIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..infrastructure.config import Config
from ..infrastructure.enums import AgentState, DecompositionAlgo, PlannerKind
from ..infrastructure.profiling import phase
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from .monte_carlo import MCResult, SingleRunResult, run, single_run_from_history


@dataclass
class VariantResult:
    label: str
    mc: MCResult
    workload_std_m: list[float] = field(default_factory=list)
    duration_s: list[float] = field(default_factory=list)
    total_energy_j: list[float] = field(default_factory=list)
    planning_time_s: list[float] = field(default_factory=list)
    replan_time_s: list[float] = field(default_factory=list)

    def mean(self, attr: str) -> float:
        vals = getattr(self, attr)
        return float(np.mean(vals)) if vals else float("nan")


def _runner(cfg: Config, rng: RngFactory, algo: DecompositionAlgo | None, planner: PlannerKind, sink: VariantResult):
    def run_once(replication: int) -> SingleRunResult:
        eng = SimulationEngine(cfg, rng, replication=replication, algo=algo, planner=planner)
        result = eng.run()
        m = result.metrics
        sink.workload_std_m.append(m.workload_std_m)
        sink.duration_s.append(m.duration_s)
        sink.total_energy_j.append(m.total_energy_j)
        sink.planning_time_s.append(m.planning_time_s)
        sink.replan_time_s.extend(m.replan_times_s)
        # single source of truth for the history -> SingleRunResult reduction
        with phase("smdp_reduce"):
            return single_run_from_history(result.history, metrics=m, outcome=result.outcome,
                                           aborted=result.aborted)
    return run_once


def _variant(cfg, rng, label, algo, planner, on_rep=None) -> VariantResult:
    vr = VariantResult(label=label, mc=None)  # type: ignore[arg-type]
    vr.mc = run(_runner(cfg, rng, algo, planner, vr), cfg.mc, on_rep=on_rep)
    return vr


# public alias: run ONE variant's Monte-Carlo batch (same paired-seed RngFactory
# must be shared across variants to keep the comparison paired).
run_variant = _variant


# The headline decomposition comparison peers (paired-seed, identical pipeline):
# three position-based baselines vs. the battery-weighted contribution. Named
# here as the single source of truth for "what gets compared".
DECOMPOSITION_PEERS: tuple[DecompositionAlgo, ...] = (
    DecompositionAlgo.CLASSIC_VORONOI,   # Euclidean Voronoi    (position-based)
    DecompositionAlgo.KMEANS,            # position k-means     (position-based)
    DecompositionAlgo.TGC_BASIC,         # unweighted TGC       (position-based)
    DecompositionAlgo.WEIGHTED_VORONOI,  # battery-weighted TGC (the contribution)
)


def compare_decomposition(cfg: Config, rng: RngFactory) -> list[VariantResult]:
    out = []
    for algo in DECOMPOSITION_PEERS:
        out.append(_variant(cfg, rng, algo.value, algo, PlannerKind.DUBINS))
    return out


def compare_kinematics(cfg: Config, rng: RngFactory) -> list[VariantResult]:
    out = []
    for planner in (PlannerKind.DUBINS, PlannerKind.GRID):
        out.append(_variant(cfg, rng, planner.value, DecompositionAlgo.WEIGHTED_VORONOI, planner))
    return out


def compare_tiers(cfg: Config, n_values: list[int], rng: RngFactory) -> dict[int, list[VariantResult]]:
    out: dict[int, list[VariantResult]] = {}
    for n in n_values:
        cfg_n = _with_override(cfg, n)
        # the thesis tier question is heuristic (k-means) vs. the weighted TGC
        # contribution; run both per n on paired seeds so the per-scale gap is
        # attributable to the algorithm.
        out[n] = [
            _variant(cfg_n, rng, f"n={n} weighted", DecompositionAlgo.WEIGHTED_VORONOI, PlannerKind.DUBINS),
            _variant(cfg_n, rng, f"n={n} kmeans", DecompositionAlgo.KMEANS, PlannerKind.DUBINS),
        ]
    return out


def tier_crossover(ns: list[int], series_a: list[float], series_b: list[float]) -> float | None:
    """Fleet size at which ``series_a`` overtakes ``series_b`` for a
    LOWER-IS-BETTER metric (a goes from >= b to < b as n increases).

    Returns a fractional fleet size via linear interpolation of the zero
    crossing of (a - b) between the two bracketing fleet sizes, or ``None`` if a
    never overtakes b across the swept range (no such sign change). ``ns`` is
    assumed sorted ascending; the three sequences must be the same length.

    This is the empirical break-even the fine-grained sweep exists to locate
    (e.g. the n at which the battery-weighted TGC starts beating position
    k-means), turning the thesis's "16-49: measure rather than assume" into a
    concrete number.
    """
    if not (len(ns) == len(series_a) == len(series_b)) or len(ns) < 2:
        return None
    diff = [a - b for a, b in zip(series_a, series_b)]
    for i in range(1, len(ns)):
        d0, d1 = diff[i - 1], diff[i]
        if d0 >= 0.0 and d1 < 0.0:  # a was >= b, becomes < b: a overtakes b here
            frac = d0 / (d0 - d1)   # zero crossing of the line through (n0,d0),(n1,d1)
            return ns[i - 1] + frac * (ns[i] - ns[i - 1])
    return None


def _with_override(cfg: Config, n: int) -> Config:
    # rebuild from the same hash inputs is not trivial; instead mutate a copy via dataclasses.replace
    import dataclasses
    fleet = dataclasses.replace(cfg.fleet, n_drones=n)
    return dataclasses.replace(cfg, fleet=fleet)
