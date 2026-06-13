"""§2.4: launch-site optimization study -- ranked candidate table."""
from __future__ import annotations

import argparse

from ..infrastructure.config import load_config
from ..infrastructure.rng import RngFactory, STREAM_LAUNCH_SAMPLING
from ..infrastructure.simulation_engine import SimulationEngine
from ..infrastructure.enums import DecompositionAlgo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="runs/launch")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    eng._build()  # build phase computes launch site + scores
    print(f"chosen launch site: ({eng.launch_pose.x:.0f}, {eng.launch_pose.y:.0f})")
    print(f"{'site':<22}{'mean_dist':>12}{'energy':>14}{'exp_swaps':>12}{'J':>8}")
    for s in eng.site_scores[:10]:
        print(f"({s.site[0]:>7.0f},{s.site[1]:>7.0f})    {s.mean_dist_m:>12.0f}"
              f"{s.initial_energy_j:>14.0f}{s.expected_swaps:>12.1f}{s.J:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
