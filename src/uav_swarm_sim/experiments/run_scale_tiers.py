"""Guideline 3.3: fine-grained fleet-size sweep with empirical break-even.

Runs the battery-weighted TGC contribution against the position k-means baseline
across a range of fleet sizes -- on paired seeds, each variant Monte-Carlo'd with
the CI-based adaptive stopping rule (so each fleet size uses only as many
replications as it needs). Reports, per fleet size, both methods' mean metrics
with 95% CIs and the actual replication count, then locates the empirical
break-even fleet size at which the weighted contribution overtakes k-means on
each lower-is-better metric. Writes a CSV and (if matplotlib is present) a plot.

Parallelism (analogous to run_shape_sweep): each fleet-size TIER is a pure
deterministic function of (master_seed, n) -- RngFactory.stream is stateless, so
a worker that rebuilds RngFactory(master_seed) reproduces the exact paired-seed
result. --jobs therefore parallelises over tiers byte-identically to serial.

Each run lands in its own unique folder (``<--out>/scale_tiers_<timestamp>_<guid>/``)
so repeat runs never overwrite each other; ``--out`` is the BASE dir (default
``runs``), and ``--run-name`` pins a fixed name when a stable path is needed.

Examples:
  # explicit sizes (-> runs/scale_tiers_<...>/)
  python -m uav_swarm_sim.experiments.run_scale_tiers --n 4 8 16 24
  # fine grid 2,4,...,100 in one flag, under a custom base dir
  python -m uav_swarm_sim.experiments.run_scale_tiers --n-range 2 100 2 --out runs
  # the full fine grid on 4 workers, clean (no obstacles)
  python -m uav_swarm_sim.experiments.run_scale_tiers --budget full --mode clean --jobs 4
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy loads (transitively, below): keeps
# N worker processes from oversubscribing cores with N*threads at --jobs>1, and
# keeps the FP reduction order identical across serial/parallel. setdefault
# leaves an explicit user override intact; spawn workers re-import this module so
# the pin also applies in each child. (Mirrors run_shape_sweep.)
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import argparse
import csv
import dataclasses
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..infrastructure import profiling
from ..infrastructure.config import Config, load_config
from ..infrastructure.enums import DecompositionAlgo, PlannerKind
from ..infrastructure.rng import RngFactory
from ..metrics.comparison import (
    VariantResult, _with_override, run_variant, tier_crossover)
from ..metrics.run_output import RunContext, unique_run_name

_PROGRESS_EVERY_S = 60.0  # throttle interval for in-tier per-replication lines

# lower-is-better metrics swept for the break-even analysis
_METRICS = (
    ("total_energy_j", "energy (J)"),
    ("duration_s", "duration (s)"),
    ("workload_std_m", "workload std (m)"),
)

# --budget -> fleet-size GRID (MC replications stay adaptive from config).
# --n / --n-range override the budget grid. "full" mirrors the docstring's fine
# grid (n = 2..100 step 2) and the roadmap's N~100 shipped-grid intent.
_BUDGET_GRIDS = {"quick": [4, 8, 16, 24], "full": list(range(2, 101, 2))}


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
    if args.n is not None:               # explicit sizes override the budget grid
        return sorted(set(args.n))
    return list(_BUDGET_GRIDS[args.budget])


def _apply_mode(cfg: Config, mode: str) -> Config:
    """--mode clean zeroes the static obstacle density (isolates the pure
    scale effect); --mode shipped keeps the config density (the prior default).
    Mirrors run_shape_sweep's clean/shipped modes.

    Telemetry is forced OFF here too: it is a read-only per-run diagnostic
    (GPX + JSONL to a fixed, overwritten path), so disabling it is byte-identical
    for every metric while removing pure per-mission waste across the sweep."""
    cfg = dataclasses.replace(
        cfg, telemetry=dataclasses.replace(cfg.telemetry, enabled=False))
    if mode == "clean":
        env = dataclasses.replace(cfg.env, obstacle_density_per_km2=0.0)
        return dataclasses.replace(cfg, env=env)
    return cfg


def _auto_jobs() -> int:
    """Worker count for ``--jobs auto``: PHYSICAL cores minus one (leave a core
    for the OS), floored at 1. Cross-platform (Windows/Linux/macOS/Azure) via
    psutil; degrades safely to logical CPUs then 1 if unavailable. (Kept in sync
    with run_shape_sweep._auto_jobs.)"""
    n = None
    try:
        import psutil
        n = psutil.cpu_count(logical=False)  # None in some restricted sandboxes
    except Exception:  # noqa: BLE001 -- psutil missing/broken -> logical fallback
        n = None
    if not n:
        n = os.cpu_count()
    return max(1, (n or 1) - 1)


# --------------------------------------------------------------------------- #
# the per-tier worker + the (serial | parallel) sweep                         #
# --------------------------------------------------------------------------- #
def _make_progress_cb(n: int, label: str):
    """A throttled per-replication progress printer for one variant. Prints a
    self-contained line every ``_PROGRESS_EVERY_S`` (and on the first rep), with
    flush -- spawn workers inherit the parent's stdout, so these lines show up in
    the same log without any multiprocessing plumbing (interleaving across
    workers is fine; each line is self-contained)."""
    state = {"last": 0.0, "t0": time.perf_counter()}

    def cb(k: int, n_max: int) -> None:
        now = time.perf_counter()
        if k == 1 or now - state["last"] >= _PROGRESS_EVERY_S:
            state["last"] = now
            print(f"[n={n:>3} {label:>8}: rep {k:>4}/{n_max} | "
                  f"{now - state['t0']:6.0f}s]", flush=True)

    return cb


def _process_tier(cfg: Config, n: int, progress: bool = False,
                  ) -> list[VariantResult]:
    """One fleet-size tier -> [weighted, kmeans] on paired seeds. Rebuilds
    RngFactory(master_seed) internally, so the tier is a pure deterministic
    function of (master_seed, n) -- byte-identical whether run serially or in a
    worker process. (Body mirrors comparison.compare_tiers exactly: same rng
    shared across both variants, same labels/algos/planner, same order -- but
    calls run_variant directly so a per-replication progress hook can be
    injected.) The heavy per-run history (MCResult.runs) is dropped: nothing
    downstream reads it, and clearing it keeps the payload light and picklable."""
    rng = RngFactory(cfg.sim.master_seed)  # ONE factory -> paired seeds
    cfg_n = _with_override(cfg, n)
    variants = []
    for label, algo in (("weighted", DecompositionAlgo.WEIGHTED_VORONOI),
                        ("kmeans", DecompositionAlgo.KMEANS)):
        cb = _make_progress_cb(n, label) if progress else None
        variants.append(run_variant(cfg_n, rng, f"n={n} {label}", algo,
                                    PlannerKind.DUBINS, on_rep=cb))
    for v in variants:
        v.mc.runs = []
    profiling.flush_worker()  # persist this worker's phase timers (no-op if OFF)
    return variants


_CSV_HEADER = ["n", "algo", "n_runs", "converged", "total_energy_j", "duration_s",
               "workload_std_m", "planning_time_s", "efficiency_mean",
               "efficiency_ci"]


def _variant_row(n: int, v: VariantResult) -> list:
    algo = "weighted" if v.label.endswith("weighted") else "kmeans"
    return [n, algo, v.mc.n_runs, int(v.mc.converged),
            f"{v.mean('total_energy_j'):.6g}", f"{v.mean('duration_s'):.6g}",
            f"{v.mean('workload_std_m'):.6g}", f"{v.mean('planning_time_s'):.6g}",
            f"{v.mc.efficiency_mean:.6g}", f"{v.mc.efficiency_ci:.6g}"]


def sweep_tiers(cfg: Config, ns: list[int], jobs: int = 1, quiet: bool = False,
                out_csv: str | None = None, progress: bool = False,
                ) -> tuple[dict[int, list[VariantResult]], list[dict]]:
    """Run every fleet-size tier; return ``(res, problems)`` where res is
    ``{n: [weighted, kmeans]}`` for the tiers that SUCCEEDED and problems lists
    ``{n, error}`` for any that raised.

    RESILIENCE (a long overnight run must survive a single bad tier):
    * a tier that raises is recorded in ``problems`` and skipped, NOT propagated
      -- the remaining tiers still run and are written (mirrors run_shape_sweep's
      run_cell "report, never skip silently"). Only n_runs=1000-style hard
      process crashes (OOM) can still abort the pool.
    * if ``out_csv`` is given, each completed tier's rows are appended and flushed
      to disk immediately, so an external kill (OOM / spot eviction) mid-run
      preserves every tier finished so far. main() rewrites out_csv in canonical
      ``ns`` order at the end for the clean final artifact.

    jobs<=1 runs serially (the determinism baseline / revert path). jobs>1 uses a
    spawn ProcessPoolExecutor over tiers (spawn, not fork, avoids the deadlock
    risk of forking a multi-threaded parent on Linux/Azure -- ENG-09)."""
    res: dict[int, list[VariantResult]] = {}
    problems: list[dict] = []
    fh = writer = None
    if out_csv is not None:
        fh = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        fh.flush()

    def _record(n: int, tier: list[VariantResult]) -> None:
        res[n] = tier
        if writer is not None:
            for v in tier:
                writer.writerow(_variant_row(n, v))
            fh.flush()

    def _fail(n: int, exc: Exception) -> None:
        problems.append({"n": n, "error": f"{type(exc).__name__}: {exc}"})
        if not quiet:
            print(f"[tier n={n}] FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)

    total = len(ns)
    t_grid = time.perf_counter()
    try:
        if jobs <= 1:
            for i, n in enumerate(ns, start=1):
                if not quiet:
                    print(f"[tier {i:>3}/{total} n={n:>3}] START "
                          f"(elapsed {(time.perf_counter()-t_grid)/3600:.2f}h)",
                          flush=True)
                t0 = time.perf_counter()
                try:
                    _record(n, _process_tier(cfg, n, progress=progress))
                    if not quiet:
                        print(f"[tier {i:>3}/{total} n={n:>3}] done in "
                              f"{time.perf_counter()-t0:.0f}s", flush=True)
                except Exception as exc:  # noqa: BLE001 -- report, never skip
                    _fail(n, exc)
        else:
            if not quiet:
                print(f"[submitted {total} tiers over {jobs} workers]", flush=True)
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
                fut_to_n = {ex.submit(_process_tier, cfg, n, progress): n
                            for n in ns}
                done = 0
                for fut in as_completed(fut_to_n):
                    n = fut_to_n[fut]
                    done += 1
                    try:
                        _record(n, fut.result())
                        if not quiet:
                            print(f"[tier {done:>3}/{total} n={n:>3}] done "
                                  f"(elapsed "
                                  f"{(time.perf_counter()-t_grid)/3600:.2f}h)",
                                  flush=True)
                    except Exception as exc:  # noqa: BLE001 -- report, never skip
                        _fail(n, exc)
    finally:
        if fh is not None:
            fh.close()
    return res, problems


def _write_csv(path: str, ns: list[int], res: dict) -> None:
    """Final ordered rewrite -- canonical ``ns`` order; tiers absent from res
    (failed) are skipped rather than raising KeyError."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        for n in ns:
            if n not in res:
                continue
            for v in res[n]:
                w.writerow(_variant_row(n, v))


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


def _write_profiling(run_dir) -> None:
    """Merge every worker's phase timers (and this process's own) into
    profiling.md + profiling.csv. Only called when UAV_SWARM_PROFILE is set."""
    snap = profiling.collect(run_dir)
    (run_dir / "profiling.md").write_text(
        "# Phase profiling (aggregated wall time across the run)\n\n"
        + profiling.format_report(snap) + "\n", encoding="utf-8")
    with (run_dir / "profiling.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(profiling.to_csv_rows(snap))
    print(f"phase profiling -> {run_dir}/profiling.md", flush=True)


def _run_profile_subset(cfg: Config, ctx: RunContext) -> None:
    """cProfile one representative tier (n=16, both variants, reps capped to 3)
    for a function-level hot-spot view; writes profile.prof + profile.txt and
    runs no grid. n=16 is large enough to surface the O(n^2) per-step costs."""
    import cProfile
    import io
    import pstats

    n = 16
    cfgp = dataclasses.replace(
        cfg, mc=dataclasses.replace(cfg.mc, n_min=3, n_max=3, ci_tolerance=1.0))
    pr = cProfile.Profile()
    pr.enable()
    _process_tier(cfgp, n)
    pr.disable()
    pr.dump_stats(str(ctx.dir / "profile.prof"))
    buf = io.StringIO()
    buf.write(f"cProfile: tier n={n} (weighted + kmeans) n_runs=3\n\n"
              f"=== by cumulative time (top 40) ===\n")
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("cumulative").print_stats(40)
    buf.write("\n=== by total/self time (top 40) ===\n")
    st.sort_stats("tottime").print_stats(40)
    (ctx.dir / "profile.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(f"cProfile -> {ctx.dir}/profile.txt (+ profile.prof)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n", type=int, nargs="+", default=None,
                    help="explicit fleet sizes (override the --budget grid)")
    ap.add_argument("--n-range", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                    default=None, help="generate a fleet-size grid (inclusive of STOP)")
    ap.add_argument("--mode", choices=["clean", "shipped"], default="shipped",
                    help="clean zeroes the static obstacle density; shipped keeps "
                         "the config density (default, prior behaviour)")
    ap.add_argument("--budget", choices=["quick", "full"], default="quick",
                    help="fleet-size grid when neither --n nor --n-range is given: "
                         "quick=[4,8,16,24], full=n 2..100 step 2 (MC stays adaptive)")
    ap.add_argument("--jobs", default="auto",
                    help="parallel worker processes over tiers (default 'auto' = "
                         "physical cores minus 1; '1' = serial). Output is "
                         "byte-identical to serial at any --jobs.")
    ap.add_argument("--out", default="runs",
                    help="BASE output dir; each run lands in its own "
                         "'scale_tiers_<timestamp>_<guid>' subfolder under it "
                         "(so repeat runs never overwrite).")
    ap.add_argument("--run-name", default=None,
                    help="force a fixed run-dir name (default: unique per run). "
                         "Pass a name to pin a stable path.")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile one representative tier (n=16, both variants, "
                         "few reps) to <run>/profile.txt + profile.prof, then "
                         "EXIT without running the grid (function-level hot "
                         "spots). For the coarse per-PHASE wall-time breakdown "
                         "across a real run instead, set env UAV_SWARM_PROFILE=1.")
    args = ap.parse_args(argv)

    ns = _n_grid(args)
    cfg = _apply_mode(load_config(args.config), args.mode)
    jobs = _auto_jobs() if args.jobs == "auto" else int(args.jobs)
    # RunContext.__init__ mkdir's the run dir immediately -> the crash-safe
    # incremental out_csv (opened before the sweep) still works unchanged.
    ctx = RunContext(base_dir=args.out,
                     name=args.run_name or unique_run_name("scale_tiers"))
    if args.profile:
        _run_profile_subset(cfg, ctx)
        ctx.finalize(summary={"experiment": "scale_tiers", "mode": "profile"})
        return 0
    if profiling.enabled():
        os.environ[profiling._ENV_DIR] = str(ctx.dir)  # workers flush phase timers here
    csv_path = str(ctx.dir / "scale_sweep.csv")
    print(f"scale tiers: mode={args.mode} budget={args.budget} "
          f"n={ns} jobs={jobs} -> {ctx.dir}", flush=True)
    # out_csv gives crash-safety: each finished tier is flushed to disk as it
    # completes, so an OOM / eviction mid-run keeps the tiers done so far.
    res, problems = sweep_tiers(cfg, ns, jobs, out_csv=csv_path, progress=True)
    present = [n for n in ns if n in res]  # skip tiers that failed

    # per-fleet-size table (both methods)
    hdr = (f"{'n':>4} {'algo':>9} {'runs':>5} {'conv':>5} "
           f"{'energy_J':>12} {'dur_s':>9} {'wl_std_m':>10} {'eff':>8}")
    print(hdr)
    print("-" * len(hdr))
    for n in present:
        for v in sorted(res[n], key=lambda x: 0 if x.label.endswith("weighted") else 1):
            algo = "weighted" if v.label.endswith("weighted") else "kmeans"
            print(f"{n:>4} {algo:>9} {v.mc.n_runs:>5} {('Y' if v.mc.converged else 'n'):>5} "
                  f"{v.mean('total_energy_j'):>12.0f} {v.mean('duration_s'):>9.0f} "
                  f"{v.mean('workload_std_m'):>10.1f} {v.mc.efficiency_mean:>8.3f}")

    # empirical break-even: where weighted overtakes k-means (lower is better)
    crossovers: dict[str, float | None] = {}
    print("\nempirical break-even (weighted overtakes k-means; None = no crossing in range):")
    for attr, label in _METRICS:
        w_series = [_by_algo(res[n])[0].mean(attr) for n in present]
        k_series = [_by_algo(res[n])[1].mean(attr) for n in present]
        xc = tier_crossover(present, w_series, k_series)
        crossovers[attr] = xc
        print(f"  {label:>16}: " + (f"n* = {xc:.1f}" if xc is not None else "no crossing"))

    # final ordered rewrite (canonical ns order; the incremental file was
    # completion-ordered) + plot over the tiers that succeeded
    _write_csv(csv_path, ns, res)
    print(f"\nwrote {csv_path}")
    plot_path = str(ctx.dir / "scale_sweep.png")
    if present and _plot(plot_path, present, res, crossovers):
        print(f"wrote {plot_path}")
    if profiling.enabled():
        _write_profiling(ctx.dir)

    ctx.finalize(summary={"experiment": "scale_tiers", "mode": args.mode,
                          "budget": args.budget, "ns": ns, "jobs": jobs,
                          "crossovers": {a: crossovers.get(a) for a, _ in _METRICS},
                          "problems": problems})

    if problems:
        print(f"\nPROBLEM tiers: {len(problems)} (results incomplete)",
              file=sys.stderr)
        for p in problems:
            print(f"  n={p['n']}: {p['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
