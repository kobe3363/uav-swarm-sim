"""Guideline 3.3: fine-grained fleet-size sweep with empirical break-even.

Runs the battery-weighted TGC contribution against the position k-means baseline
across a range of fleet sizes -- on paired seeds, each variant Monte-Carlo'd with
the CI-based adaptive stopping rule (so each fleet size uses only as many
replications as it needs). Reports, per fleet size, both methods' mean metrics
with 95% CIs and the actual replication count, then locates the empirical
break-even fleet size at which the weighted contribution overtakes k-means on
each lower-is-better metric. Writes a CSV and (if matplotlib is present) a plot.

Examples:
  # explicit sizes
  python -m uav_swarm_sim.experiments.run_scale_tiers --n 4 8 16 24 48 --out runs/tiers
  # fine grid 2,4,...,100 in one flag
  python -m uav_swarm_sim.experiments.run_scale_tiers --n-range 2 100 2 --out runs/tiers
"""
from __future__ import annotations

import argparse
import csv
import os

from ..infrastructure.config import load_config
from ..infrastructure.rng import RngFactory
from ..metrics.comparison import compare_tiers, tier_crossover

# lower-is-better metrics swept for the break-even analysis
_METRICS = (
    ("total_energy_j", "energy (J)"),
    ("duration_s", "duration (s)"),
    ("workload_std_m", "workload std (m)"),
)


def _by_algo(variants):
    """Split a fleet-size's variant pair into (weighted, kmeans) by label."""
    weighted = next(v for v in variants if v.label.endswith("weighted"))
    kmeans = next(v for v in variants if v.label.endswith("kmeans"))
    return weighted, kmeans


def _n_grid(args) -> list[int]:
    if args.n_range is not None:
        start, stop, step = args.n_range
        if step <= 0:
            raise SystemExit("--n-range STEP must be positive")
        return list(range(start, stop + 1, step))
    return sorted(set(args.n))


def _write_csv(path: str, ns: list[int], res: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "algo", "n_runs", "converged",
            "total_energy_j", "duration_s", "workload_std_m",
            "planning_time_s", "efficiency_mean", "efficiency_ci",
        ])
        for n in ns:
            for v in res[n]:
                algo = "weighted" if v.label.endswith("weighted") else "kmeans"
                w.writerow([
                    n, algo, v.mc.n_runs, int(v.mc.converged),
                    f"{v.mean('total_energy_j'):.6g}", f"{v.mean('duration_s'):.6g}",
                    f"{v.mean('workload_std_m'):.6g}", f"{v.mean('planning_time_s'):.6g}",
                    f"{v.mc.efficiency_mean:.6g}", f"{v.mc.efficiency_ci:.6g}",
                ])


def _plot(path: str, ns: list[int], res: dict, crossovers: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, len(_METRICS), figsize=(5 * len(_METRICS), 4))
    for ax, (attr, label) in zip(axes, _METRICS):
        w_series = [_by_algo(res[n])[0].mean(attr) for n in ns]
        k_series = [_by_algo(res[n])[1].mean(attr) for n in ns]
        ax.plot(ns, w_series, "-o", label="weighted TGC", markersize=3)
        ax.plot(ns, k_series, "-s", label="k-means", markersize=3)
        xc = crossovers.get(attr)
        if xc is not None:
            ax.axvline(xc, color="gray", linestyle="--", linewidth=1)
            ax.annotate(f"n*={xc:.1f}", (xc, ax.get_ylim()[1]), fontsize=8,
                        ha="left", va="top", color="gray")
        ax.set_xlabel("fleet size n")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 24],
                    help="explicit fleet sizes")
    ap.add_argument("--n-range", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                    default=None, help="generate a fleet-size grid (inclusive of STOP)")
    ap.add_argument("--out", default="runs/tiers")
    args = ap.parse_args(argv)

    ns = _n_grid(args)
    cfg = load_config(args.config)
    res = compare_tiers(cfg, ns, RngFactory(cfg.sim.master_seed))
    os.makedirs(args.out, exist_ok=True)

    # per-fleet-size table (both methods)
    hdr = (f"{'n':>4} {'algo':>9} {'runs':>5} {'conv':>5} "
           f"{'energy_J':>12} {'dur_s':>9} {'wl_std_m':>10} {'eff':>8}")
    print(hdr)
    print("-" * len(hdr))
    for n in ns:
        for v in sorted(res[n], key=lambda x: 0 if x.label.endswith("weighted") else 1):
            algo = "weighted" if v.label.endswith("weighted") else "kmeans"
            print(f"{n:>4} {algo:>9} {v.mc.n_runs:>5} {('Y' if v.mc.converged else 'n'):>5} "
                  f"{v.mean('total_energy_j'):>12.0f} {v.mean('duration_s'):>9.0f} "
                  f"{v.mean('workload_std_m'):>10.1f} {v.mc.efficiency_mean:>8.3f}")

    # empirical break-even: where weighted overtakes k-means (lower is better)
    crossovers: dict[str, float | None] = {}
    print("\nempirical break-even (weighted overtakes k-means; None = no crossing in range):")
    for attr, label in _METRICS:
        w_series = [_by_algo(res[n])[0].mean(attr) for n in ns]
        k_series = [_by_algo(res[n])[1].mean(attr) for n in ns]
        xc = tier_crossover(ns, w_series, k_series)
        crossovers[attr] = xc
        print(f"  {label:>16}: " + (f"n* = {xc:.1f}" if xc is not None else "no crossing"))

    csv_path = os.path.join(args.out, "scale_sweep.csv")
    _write_csv(csv_path, ns, res)
    print(f"\nwrote {csv_path}")
    plot_path = os.path.join(args.out, "scale_sweep.png")
    if _plot(plot_path, ns, res, crossovers):
        print(f"wrote {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
