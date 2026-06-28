"""One mission, one config, full visual dump -- the defense demo, written into a
structured run/simulation folder with plan.json + results.json + all artifacts.

Usage:
  python -m uav_swarm_sim.experiments.run_single_mission \
      --config config/default.yaml [--algo weighted_voronoi] [--planner dubins|grid] \
      [--seed N] [--name demo] [--base runs]

Produces:
  runs/run-<timestamp>/
    run.json
    simulation-<name>/
      plan.json results.json
      environment.png partition.png paths.png replay.gif state_gantt.png battery.png
      pi_bars.png tracks.gpx
"""
from __future__ import annotations

import argparse
from time import perf_counter

from ..infrastructure.config import load_config
from ..infrastructure.enums import DecompositionAlgo, PlannerKind
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..infrastructure import visualization as viz
from ..metrics.smdp_estimator import estimate
from ..metrics.stationary_distribution import stationary
from ..metrics.efficiency_score import efficiency
from ..metrics.gpx_exporter import write_gpx
from ..metrics.run_output import RunContext, build_plan, build_results_single


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--algo", default="weighted_voronoi")
    ap.add_argument("--planner", default="dubins", choices=["dubins", "grid"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="simulation folder name (default: the algo)")
    ap.add_argument("--base", default="runs", help="base directory for runs")
    ap.add_argument("--run-name", default=None, help="override run folder name")
    args = ap.parse_args(argv)

    # telemetry on => GPX tracks written for this demo; overrides are applied
    # before computing cfg.config_hash, so the hash reflects these values.
    overrides = {"telemetry.enabled": True}
    if args.seed is not None:
        overrides["sim.master_seed"] = args.seed
    cfg = load_config(args.config, overrides)
    algo = DecompositionAlgo(args.algo)
    planner = PlannerKind.GRID if args.planner == "grid" else PlannerKind.DUBINS

    run = RunContext(base_dir=args.base, name=args.run_name)
    sim = run.simulation(args.name or algo.value)

    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0, algo=algo, planner=planner)
    t0 = perf_counter()
    result = eng.run()
    wall = perf_counter() - t0

    # plan.json (built with the live engine so spatial derived quantities land in it)
    identity = sim.identity(config_hash=cfg.config_hash)
    sim.write_plan(build_plan(cfg, identity=identity, algo=algo, planner=planner, engine=eng))

    # SMDP for results + the pi-bars figure
    est = estimate(result.history, close_failure_loop=True)
    pi_line = ""
    if est.ergodic:
        pi_emb, pi_time = stationary(est)
        emb = {s: float(pi_emb[i]) for i, s in enumerate(est.states)}
        tim = {s: float(pi_time[i]) for i, s in enumerate(est.states)}
        viz.plot_pi_bars(emb, tim, sim.path("pi_bars.png"))
        pi_line = f" | efficiency {efficiency(pi_time, est.states):.3f}"

    # results.json
    sim.write_results(build_results_single(result, est, identity=identity, wall_time_s=wall))

    # artifacts (all into the simulation folder)
    viz.plot_environment(eng.env, None, sim.path("environment.png"))
    viz.plot_partition(eng.env, result.partition, eng.launch_pose, sim.path("partition.png"))
    viz.plot_state_colored_paths(result.history, eng.env, sim.path("paths.png"),
                                 partition=result.partition, viz=cfg.viz)
    viz.animate_mission(result.history, eng.env, sim.path("replay.gif"),
                        partition=result.partition, viz=cfg.viz)
    viz.plot_state_gantt(result.history, sim.path("state_gantt.png"))
    viz.plot_battery_traces(result.history, cfg.battery_zones, sim.path("battery.png"))
    if eng.telemetry is not None:
        write_gpx(eng.telemetry, str(sim.path("tracks.gpx")))

    run.finalize(summary={"outcome": result.outcome.value, "coverage_frac": result.coverage_frac})

    m = result.metrics
    print(f"[single] outcome={result.outcome.value} coverage={result.coverage_frac:.3f} "
          f"energy={m.total_energy_j:.0f} J duration={m.duration_s:.0f} s "
          f"workload_std={m.workload_std_m:.1f} m{pi_line}")
    print(f"run -> {run.dir}/  (simulation: {sim.dir.name}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
