"""Replay one specific replication as a 2D animation colored by mission state.

Because the engine is deterministic in (config, master_seed, replication, algo,
planner), any replication from a Monte-Carlo batch can be reproduced EXACTLY by
re-running it with the same replication index -- no stored traces needed. This
script does that, then writes:
  * paths.png   -- static flight paths colored by state (no extra dependencies)
  * replay.gif  -- animated 2D replay, drones colored by state over time

Usage:
  python -m uav_swarm_sim.experiments.run_replay \
      --config config/scenarios/smoke.yaml --replication 60 \
      [--seed N] [--algo weighted_voronoi] [--planner dubins|grid] \
      [--fps 12] [--max-frames 200] [--out runs/replay60]

Note: --replication picks which replication to reproduce; if you ran a 1000-run
Monte-Carlo and want "simulation #60", pass --replication 60 with the same
--config/--seed/--algo/--planner you used for the batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..infrastructure.config import load_config
from ..infrastructure.enums import DecompositionAlgo, PlannerKind
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..infrastructure import visualization as viz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/scenarios/smoke.yaml")
    ap.add_argument("--replication", type=int, default=0, help="which replication to reproduce")
    ap.add_argument("--seed", type=int, default=None, help="master seed (defaults to the config's)")
    ap.add_argument("--algo", default="weighted_voronoi")
    ap.add_argument("--planner", default="dubins", choices=["dubins", "grid"])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", default="runs/replay")
    args = ap.parse_args(argv)

    overrides = {"sim.master_seed": args.seed} if args.seed is not None else None
    cfg = load_config(args.config, overrides)
    algo = DecompositionAlgo(args.algo)
    planner = PlannerKind.GRID if args.planner == "grid" else PlannerKind.DUBINS
    out = Path(args.out)

    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                           replication=args.replication, algo=algo, planner=planner)
    result = eng.run()

    png = viz.plot_state_colored_paths(result.history, eng.env, out / "paths.png", viz=cfg.viz,
                                       partition=result.partition)
    gif = viz.animate_mission(result.history, eng.env, out / "replay.gif", viz=cfg.viz,
                              fps=args.fps, max_frames=args.max_frames,
                              partition=result.partition)

    print(f"[replay] replication={args.replication} seed={cfg.sim.master_seed} "
          f"algo={args.algo} planner={args.planner}")
    print(f"  aborted={result.aborted} coverage={result.coverage_frac:.3f} "
          f"duration={result.metrics.duration_s:.0f} s")
    print(f"  static paths -> {png}")
    print(f"  replay gif   -> {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
