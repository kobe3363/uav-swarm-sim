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
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from .monte_carlo import MCResult, SingleRunResult, run
from .smdp_estimator import estimate
from .stationary_distribution import stationary
from .efficiency_score import efficiency


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
        est = estimate(result.history, close_failure_loop=True)
        try:
            _, pi_time = stationary(est)
        except ValueError:
            return SingleRunResult(est.states, {}, float("nan"), metrics=m, aborted=True)
        pi_map = {s: float(pi_time[i]) for i, s in enumerate(est.states)}
        return SingleRunResult(est.states, pi_map, efficiency(pi_time, est.states), metrics=m,
                               aborted=result.aborted)
    return run_once


def _variant(cfg, rng, label, algo, planner) -> VariantResult:
    vr = VariantResult(label=label, mc=None)  # type: ignore[arg-type]
    vr.mc = run(_runner(cfg, rng, algo, planner, vr), cfg.mc)
    return vr


def compare_decomposition(cfg: Config, rng: RngFactory) -> list[VariantResult]:
    out = []
    for algo in (DecompositionAlgo.CLASSIC_VORONOI, DecompositionAlgo.TGC_BASIC,
                 DecompositionAlgo.WEIGHTED_VORONOI):
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
        out[n] = [
            _variant(cfg_n, rng, f"n={n} weighted", DecompositionAlgo.WEIGHTED_VORONOI, PlannerKind.DUBINS),
        ]
    return out


def _with_override(cfg: Config, n: int) -> Config:
    from ..infrastructure.config import load_config
    # rebuild from the same hash inputs is not trivial; instead mutate a copy via dataclasses.replace
    import dataclasses
    fleet = dataclasses.replace(cfg.fleet, n_drones=n)
    return dataclasses.replace(cfg, fleet=fleet)
