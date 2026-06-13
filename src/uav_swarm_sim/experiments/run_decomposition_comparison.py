"""Headline experiment: classic_voronoi vs tgc_basic vs weighted_voronoi."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..infrastructure.config import load_config
from ..infrastructure.rng import RngFactory
from ..infrastructure import visualization as viz
from ..metrics.comparison import compare_decomposition


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="runs/decomp")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out)

    variants = compare_decomposition(cfg, RngFactory(cfg.sim.master_seed))
    print(f"{'variant':<18}{'workload_std':>14}{'duration':>12}{'energy':>14}{'efficiency':>12}")
    for v in variants:
        print(f"{v.label:<18}{v.mean('workload_std_m'):>14.1f}{v.mean('duration_s'):>12.0f}"
              f"{v.mean('total_energy_j'):>14.0f}{v.mc.efficiency_mean:>12.3f}")
    viz.plot_comparison_box({v.label: v.workload_std_m for v in variants}, "workload_std_m",
                            out / "workload_box.png")
    print(f"figures -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
