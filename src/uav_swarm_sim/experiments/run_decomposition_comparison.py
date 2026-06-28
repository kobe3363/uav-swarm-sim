"""Headline experiment: classic_voronoi vs k-means vs tgc_basic vs weighted_voronoi.

One RUN with one SIMULATION per decomposition algorithm. Each simulation folder
gets plan.json (the setup) and results.json (Monte-Carlo aggregate: success rate,
SMDP, stopping logic, timing). The run folder gets the comparison box plot and a
run.json manifest summarizing all simulations.
"""
from __future__ import annotations

import argparse
from time import perf_counter

from ..infrastructure.config import load_config
from ..infrastructure.enums import DecompositionAlgo, PlannerKind
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..infrastructure import visualization as viz
from ..metrics.comparison import DECOMPOSITION_PEERS, run_variant
from ..metrics.run_output import RunContext, build_plan, build_results_mc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--base", default="runs")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    run = RunContext(base_dir=args.base, name=args.run_name)
    # one reference engine build (env + launch are decomposition-algo-independent)
    # to populate the spatial 'derived' block of each plan.
    ref_eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                               algo=DecompositionAlgo.WEIGHTED_VORONOI, planner=PlannerKind.DUBINS)
    ref_eng._build()

    # ONE shared RngFactory across all variants keeps the comparison paired.
    rng = RngFactory(cfg.sim.master_seed)
    variants = {}
    print(f"{'variant':<18}{'runs':>6}{'conv':>6}{'workload_std':>14}{'duration':>12}"
          f"{'energy':>14}{'efficiency':>12}")
    for algo in DECOMPOSITION_PEERS:
        sim = run.simulation(algo.value)
        identity = sim.identity(config_hash=cfg.config_hash)
        sim.write_plan(build_plan(cfg, identity=identity, algo=algo,
                                  planner=PlannerKind.DUBINS, engine=ref_eng))

        t0 = perf_counter()
        v = run_variant(cfg, rng, algo.value, algo, PlannerKind.DUBINS)
        wall = perf_counter() - t0
        variants[algo.value] = v

        sim.write_results(build_results_mc(v.mc, identity=identity, wall_time_s=wall, variant=v))
        print(f"{algo.value:<18}{v.mc.n_runs:>6}{('Y' if v.mc.converged else 'n'):>6}"
              f"{v.mean('workload_std_m'):>14.1f}{v.mean('duration_s'):>12.0f}"
              f"{v.mean('total_energy_j'):>14.0f}{v.mc.efficiency_mean:>12.3f}")

    # run-level comparison figure
    viz.plot_comparison_box({lbl: v.workload_std_m for lbl, v in variants.items()},
                            "workload_std_m", run.dir / "workload_box.png")

    run.finalize(summary={
        "experiment": "decomposition_comparison",
        "peers": [a.value for a in DECOMPOSITION_PEERS],
        "efficiency_by_variant": {lbl: v.mc.efficiency_mean for lbl, v in variants.items()},
    })
    print(f"run -> {run.dir}/  ({len(variants)} simulations + workload_box.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
