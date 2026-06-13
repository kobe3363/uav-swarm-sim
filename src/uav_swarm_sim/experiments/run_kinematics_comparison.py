"""Guideline 1.2: Dubins vs discretized grid (FIXED_WING / VTOL only)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..infrastructure.config import load_config
from ..infrastructure.enums import PlatformType
from ..infrastructure.rng import RngFactory
from ..metrics.comparison import compare_kinematics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="runs/kinematics")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.platform.type is PlatformType.MULTIROTOR:
        print("kinematics comparison is vacuous for MULTIROTOR (holonomic). "
              "Set platform_type to FIXED_WING or VTOL.")
        return 1
    variants = compare_kinematics(cfg, RngFactory(cfg.sim.master_seed))
    print(f"{'planner':<10}{'plan_time':>12}{'duration':>12}{'energy':>14}")
    for v in variants:
        print(f"{v.label:<10}{v.mean('planning_time_s'):>12.4f}{v.mean('duration_s'):>12.0f}"
              f"{v.mean('total_energy_j'):>14.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
