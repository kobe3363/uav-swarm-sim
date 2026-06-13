"""One mission, one config, full visual dump -- the defense demo.

Usage:
  python -m uav_swarm_sim.experiments.run_single_mission \
      --config config/default.yaml [--algo weighted_voronoi] [--planner dubins|grid] \
      [--seed N] [--out runs/demo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..infrastructure.config import load_config
from ..infrastructure.enums import DecompositionAlgo, PlannerKind
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..infrastructure import visualization as viz
from ..metrics.smdp_estimator import estimate
from ..metrics.stationary_distribution import stationary
from ..metrics.efficiency_score import efficiency


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--algo", default="weighted_voronoi")
    ap.add_argument("--planner", default="dubins", choices=["dubins", "grid"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args(argv)

    overrides = {"sim.master_seed": args.seed} if args.seed is not None else None
    cfg = load_config(args.config, overrides)
    algo = DecompositionAlgo(args.algo)
    planner = PlannerKind.GRID if args.planner == "grid" else PlannerKind.DUBINS
    out = Path(args.out)

    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0, algo=algo, planner=planner)
    result = eng.run()

    est = estimate(result.history, close_failure_loop=True)
    pi_line = ""
    if est.ergodic:
        pi_emb, pi_time = stationary(est)
        emb = {s: float(pi_emb[i]) for i, s in enumerate(est.states)}
        tim = {s: float(pi_time[i]) for i, s in enumerate(est.states)}
        viz.plot_pi_bars(emb, tim, out / "pi_bars.png")
        eff = efficiency(pi_time, est.states)
        pi_line = f" | efficiency {eff:.3f}"

    viz.plot_environment(eng.env, None, out / "environment.png")
    viz.plot_partition(eng.env, result.partition, eng.launch_pose, out / "partition.png")
    viz.plot_state_colored_paths(result.history, eng.env, out / "paths.png", partition=result.partition, viz=cfg.viz)
    viz.animate_mission(result.history, eng.env, out / "replay.gif", partition=result.partition, viz=cfg.viz)
    viz.plot_state_gantt(result.history, out / "state_gantt.png")
    viz.plot_battery_traces(result.history, cfg.battery_zones, out / "battery.png")

    m = result.metrics
    print(f"[single] aborted={result.aborted} coverage={result.coverage_frac:.3f} "
          f"energy={m.total_energy_j:.0f} J duration={m.duration_s:.0f} s "
          f"workload_std={m.workload_std_m:.1f} m{pi_line}")
    print(f"figures -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
