"""Guideline 3.3: sweep fleet sizes; report planning time across the tiers."""
from __future__ import annotations

import argparse

from ..infrastructure.config import load_config
from ..infrastructure.rng import RngFactory
from ..metrics.comparison import compare_tiers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 24])
    ap.add_argument("--out", default="runs/tiers")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    res = compare_tiers(cfg, args.n, RngFactory(cfg.sim.master_seed))
    print(f"{'n':>4}{'plan_time':>12}{'duration':>12}{'workload_std':>14}")
    for n, variants in res.items():
        v = variants[0]
        print(f"{n:>4}{v.mean('planning_time_s'):>12.4f}{v.mean('duration_s'):>12.0f}"
              f"{v.mean('workload_std_m'):>14.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
