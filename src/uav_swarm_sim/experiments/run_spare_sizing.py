"""Spare-sizing study: how many shared swap-battery packs to stock.

Sweeps the spare count (``fleet.total_reserve_batteries`` -- the finite shared
pool the whole fleet draws swap packs from) and, at each count, runs a fixed
batch of Monte-Carlo missions on PAIRED seeds. A mission is a SUCCESS when its
coverage completes before the pool runs dry (``Outcome.MISSION_SUCCESS``); it is
a FAILURE when the pool exhausts first (``pool_exhausted`` -> ``MISSION_FAILED``).
The success FRACTION across replications, with a Wilson CI, is the success
probability at that spare count.

Deliverables (thesis AC):
  * the success-probability knee at BOTH targets (99 %, 95 %), with a CI on the
    fraction -- the smallest spare count whose Wilson lower bound clears the
    target (the robust reading; the point-estimate crossing is reported too);
  * paired seeds across spare counts, so the spare effect is not seed noise -- a
    single shared ``RngFactory`` and identical replication indices mean the
    environment and failure draws at replication k are byte-identical across all
    spare counts (the count only decides whether the pool exhausts);
  * an honest validate/refute of the analytical prior
    ``spares ~= E_cover/B_usable - n + margin`` against the empirical 99 % knee;
  * structured ``runs/`` output (plan.json + results.json + knee PNG) under a
    timestamped RunContext.

The analytical prior reuses the B6.2 fleet-sizing planning layer + core to get
``total_sorties_int`` (the integer battery-cycle demand) from the SAME real base
pose the heavy simulation would use, so prior and measurement share one geometry.

Examples:
  # default range from the analytical prior, 200 paired reps per spare count
  python -m uav_swarm_sim.experiments.run_spare_sizing --out runs/spares
  # explicit sweep, fewer reps for a quick look
  python -m uav_swarm_sim.experiments.run_spare_sizing --spares 0 2 4 6 8 --reps 50
  # a fixed grid
  python -m uav_swarm_sim.experiments.run_spare_sizing --spare-range 0 20 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..infrastructure.config import Config, load_config
from ..infrastructure.enums import Outcome
from ..infrastructure.rng import STREAM_LAUNCH_SAMPLING, STREAM_OBSTACLES, RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..metrics.run_output import RunContext
from .fleet_sizing import FleetSizingInputs, sweep as fleet_sweep
from .spare_sizing import (
    TARGETS,
    SparePoint,
    SpareSizingReport,
    analytical_spare_prior,
    default_spare_range,
    min_reps_for_target,
)


# --------------------------------------------------------------------------- #
# config override for the sweep variable                                       #
# --------------------------------------------------------------------------- #
def _with_reserve(cfg: Config, spares: int) -> Config:
    """A config copy with the shared swap-pool size set to ``spares`` -- the ONLY
    thing that varies across the sweep (mirrors comparison._with_override)."""
    fleet = dataclasses.replace(cfg.fleet, total_reserve_batteries=spares)
    return dataclasses.replace(cfg, fleet=fleet)


# --------------------------------------------------------------------------- #
# analytical prior from the B6.2 planning layer                                #
# --------------------------------------------------------------------------- #
# The planning-layer reproduction below is a self-contained copy of the B6.2
# analyzer's setup (run_fleet_sizing_analyzer._build_planning_layer /
# _inputs_from): the same seeded streams the simulation uses, so the base pose
# and geometry feeding the analytical prior match the heavy run's. It is inlined
# rather than imported to avoid coupling to that CLI's module-level state.
def _build_planning_layer(cfg):
    """Reproduce the engine's layer-0 world + real launch site, deterministically.
    Returns (env, spec, em, base_pose)."""
    from ..physical_model.aero_correction import AeroCorrection
    from ..physical_model.drone_specs import build_spec
    from ..physical_model.energy_model import EnergyModel
    from ..physical_model.motion_model import make_motion_model
    from ..planning.environment_map import EnvironmentMap
    from ..planning.geojson_parser import load_area
    from ..planning.gvg_builder import build_gvg
    from ..planning.launch_site_optimizer import optimize
    from ..planning.obstacle_generator import generate as generate_obstacles
    from ..planning.tgc import build_tgc

    spec = build_spec(cfg)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    aero = AeroCorrection(cfg.aero, spec.platform)

    rngf = RngFactory(cfg.sim.master_seed)
    obs_rng = rngf.stream(STREAM_OBSTACLES, 0)
    launch_rng = rngf.stream(STREAM_LAUNCH_SAMPLING, 0)

    area = load_area(cfg.env.geojson_path)
    obstacles = generate_obstacles(area, cfg.env, obs_rng)
    env = EnvironmentMap(area, obstacles, cfg.env.clearance_buffer_m)
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    tgc = build_tgc(env, gvg)

    base_pose, _scores = optimize(
        cfg.launch, tgc, env, motion, em, aero, spec,
        cfg.fleet.n_drones, launch_rng, cfg.env.coverage_altitude_m,
    )
    return env, spec, em, base_pose


def _inputs_from(env, base_pose, altitude_m: float) -> FleetSizingInputs:
    from ..planning.launch_site_optimizer import _furthest_free_vertex_dist
    centroid = env.free_space.centroid
    transit = math.hypot(centroid.x - base_pose.x, centroid.y - base_pose.y)
    return FleetSizingInputs(
        area_m2=float(env.free_space.area),
        furthest_dist_m=_furthest_free_vertex_dist(env, base_pose.as_xy()),
        transit_dist_m=transit,
        altitude_m=altitude_m,
    )


def _compute_prior(cfg: Config, margin: int):
    """Run the B6.2 planning layer + fleet-sizing core at the configured fleet
    size to obtain ``total_sorties_int``, then apply the DoR spare formula.
    Returns ``(SparePrior, total_coverage_j)`` -- coverage energy is echoed for
    the plan.json setup block."""
    env, spec, em, base_pose = _build_planning_layer(cfg)
    inputs = _inputs_from(env, base_pose, cfg.env.coverage_altitude_m)
    n = cfg.fleet.n_drones
    report = fleet_sweep(
        inputs, em, spec,
        effective_swath_m=spec.swath_width_m,
        service_time_s=cfg.swap.service_time_s,
        n_bays=cfg.swap.n_bays,
        n_min=n, n_max=n,
        reserve_frac=cfg.rth.reserve_frac,
    )
    prior = analytical_spare_prior(report.total_sorties_int, n, margin)
    return prior, report.total_coverage_j


# --------------------------------------------------------------------------- #
# the paired Monte-Carlo sweep                                                 #
# --------------------------------------------------------------------------- #
def run_sweep(cfg: Config, spare_counts, reps: int, rng: RngFactory,
              algo=None, planner=None, progress=None) -> list[SparePoint]:
    """For each spare count, run ``reps`` missions on paired seeds and tally the
    terminal outcomes into a ``SparePoint``.

    Pairing: the SAME ``rng`` and the SAME replication indices ``1..reps`` are
    used at every spare count. ``RngFactory.stream(name, k)`` is a pure function
    of ``(master_seed, name, k)``, so the environment and failure draws at
    replication k are identical across counts -- only the pool size differs.
    """
    from ..infrastructure.enums import PlannerKind
    if planner is None:
        planner = PlannerKind.DUBINS

    points: list[SparePoint] = []
    for s in spare_counts:
        cfg_s = _with_reserve(cfg, s)
        n_succ = n_fail = n_inc = 0
        for k in range(1, reps + 1):
            eng = SimulationEngine(cfg_s, rng, replication=k, algo=algo, planner=planner)
            outcome = eng.run().outcome
            if outcome is Outcome.MISSION_SUCCESS:
                n_succ += 1
            elif outcome is Outcome.MISSION_FAILED:
                n_fail += 1
            else:
                n_inc += 1
        pt = SparePoint(spares=s, n_reps=reps, n_success=n_succ,
                        n_failed=n_fail, n_incomplete=n_inc)
        points.append(pt)
        if progress is not None:
            progress(pt)
    return points


# --------------------------------------------------------------------------- #
# crash-safe incremental partial log + resume                                  #
# --------------------------------------------------------------------------- #
# STUDY-01's final sweep (--reps 500) runs for many hours serially; writing
# results only at the end loses everything on a crash. Each completed
# SparePoint is therefore appended to <run_dir>/results_partial.jsonl the
# moment run_sweep's progress callback delivers it, and --resume replays a
# previous run's partial log, skipping the spare counts it already finished.
PARTIAL_SCHEMA = "uav-swarm-sim/spare-sizing-partial/v1"
PARTIAL_FILENAME = "results_partial.jsonl"


def _partial_identity(cfg: Config, reps: int) -> dict:
    """The run-identity fields a --resume candidate must match EXACTLY.

    ``master_seed`` + ``config_hash`` pin the paired-seed design and every
    config input; ``reps_per_point`` pins the replication indices 1..reps. If
    any of these differ, the previous run's points belong to a DIFFERENT
    experiment and merging them would silently corrupt the report."""
    return {
        "master_seed": cfg.sim.master_seed,
        "config_hash": cfg.config_hash,
        "reps_per_point": reps,
    }


def append_partial_point(path, pt: SparePoint, identity: dict) -> None:
    """Append one completed SparePoint as a single JSON line, flushed and
    fsync'd so a crash right after loses at most the point in flight."""
    rec = {
        "schema": PARTIAL_SCHEMA,
        **identity,
        **_point_dict(pt),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_partial_points(path) -> tuple[dict, dict[int, SparePoint]]:
    """Parse a results_partial.jsonl into ``(identity, {spares: SparePoint})``.

    A crash mid-append can leave a truncated FINAL line -- it is skipped with a
    warning (every earlier line was flushed whole). A malformed line anywhere
    else means the file is not a partial log we understand: refuse it."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    identity: dict | None = None
    points: dict[int, SparePoint] = {}
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"[resume] ignoring truncated final line in {path}",
                      file=sys.stderr)
                continue
            raise SystemExit(f"--resume: corrupt line {i + 1} in {path}") from None
        if rec.get("schema") != PARTIAL_SCHEMA:
            raise SystemExit(f"--resume: unsupported schema {rec.get('schema')!r} "
                             f"at line {i + 1} in {path} (expected {PARTIAL_SCHEMA!r})")
        ident = {k: rec.get(k) for k in ("master_seed", "config_hash", "reps_per_point")}
        if identity is None:
            identity = ident
        elif ident != identity:
            raise SystemExit(f"--resume: inconsistent run identity at line {i + 1} "
                             f"in {path} (mixed runs in one file?)")
        try:
            pt = SparePoint(spares=int(rec["spares"]), n_reps=int(rec["n_reps"]),
                            n_success=int(rec["n_success"]), n_failed=int(rec["n_failed"]),
                            n_incomplete=int(rec["n_incomplete"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"--resume: malformed record at line {i + 1} "
                             f"in {path}: {exc}") from exc
        points[pt.spares] = pt
    if identity is None:
        raise SystemExit(f"--resume: no completed points in {path}")
    return identity, points


def _validated_resume(resume_path, expected: dict, counts) -> dict[int, SparePoint]:
    """Load a previous partial log and REFUSE it unless its identity matches
    the current run exactly. Completed counts outside the current sweep grid
    are ignored (merging them would make the resumed results.json differ from
    an uninterrupted run of the same CLI arguments)."""
    if not Path(resume_path).is_file():
        raise SystemExit(f"--resume: no such file: {resume_path}")
    identity, points = load_partial_points(resume_path)
    if identity != expected:
        diffs = "; ".join(
            f"{k}: partial={identity.get(k)!r} vs current={expected[k]!r}"
            for k in expected if identity.get(k) != expected[k]
        )
        raise SystemExit(f"--resume rejected: run identity mismatch ({diffs}). "
                         f"A partial log can only resume the run that wrote it.")
    extra = sorted(set(points) - set(counts))
    if extra:
        print(f"[resume] ignoring completed spare counts outside the current "
              f"sweep grid: {extra}", file=sys.stderr)
    return {s: points[s] for s in counts if s in points}


def sweep_with_partials(cfg: Config, spare_counts, reps: int, rng: RngFactory,
                        partial_path, resume_path=None, algo=None, planner=None,
                        progress=None) -> list[SparePoint]:
    """Crash-safe wrapper around the UNTOUCHED ``run_sweep``: every completed
    SparePoint is appended to ``partial_path`` via the existing progress seam,
    and ``resume_path`` (a previous run's partial log) skips finished counts.

    Resume safety: ``RngFactory.stream(name, k)`` is a PURE function of
    ``(master_seed, name, k)`` (see infrastructure/rng.py -- a fresh Generator
    is derived per call, no factory state is advanced), and each replication of
    each spare count draws only from streams keyed by its own replication
    index. Skipping already-completed counts therefore CANNOT shift any draw of
    the remaining counts: the resumed points are byte-identical to an
    uninterrupted sweep's.
    """
    identity = _partial_identity(cfg, reps)
    done: dict[int, SparePoint] = {}
    if resume_path is not None:
        done = _validated_resume(resume_path, identity, spare_counts)
        # replay the resumed points into THIS run's partial log, so the new log
        # is itself a complete resume point if this run is also interrupted
        for s in sorted(done):
            append_partial_point(partial_path, done[s], identity)

    def _progress(pt: SparePoint) -> None:
        append_partial_point(partial_path, pt, identity)
        if progress is not None:
            progress(pt)

    todo = [s for s in spare_counts if s not in done]
    new_pts = run_sweep(cfg, todo, reps, rng, algo=algo, planner=planner,
                        progress=_progress)
    return sorted(list(done.values()) + new_pts, key=lambda p: p.spares)


# --------------------------------------------------------------------------- #
# structured output                                                            #
# --------------------------------------------------------------------------- #
def _point_dict(pt: SparePoint) -> dict:
    lo, hi, phat = pt.wilson
    return {
        "spares": pt.spares,
        "n_reps": pt.n_reps,
        "n_success": pt.n_success,
        "n_failed": pt.n_failed,
        "n_incomplete": pt.n_incomplete,
        "success_frac": phat,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
    }


def _results_dict(report: SpareSizingReport, reps: int, identity: dict) -> dict:
    knees = {
        f"{k.target:.2f}": {"knee_point": k.knee_point, "knee_wilson": k.knee_wilson}
        for k in report.knees
    }
    p = report.prior
    v = report.verdict
    return {
        "schema": "uav-swarm-sim/spare-sizing/v1",
        "kind": "results",
        "mode": "spare_sizing_sweep",
        "identity": identity,
        "status": "ok",
        "paired_design": {
            "sweep_variable": "fleet.total_reserve_batteries",
            "reps_per_point": reps,
            "note": "shared RngFactory + identical replication indices => env & "
                    "failure draws paired across spare counts",
        },
        "targets": list(TARGETS),
        "knees": knees,
        "analytical_prior": {
            "formula": "spares ~= E_cover/B_usable - n + margin",
            "total_sorties_int": p.total_sorties_int,
            "n_drones": p.n_drones,
            "margin_packs": p.margin,
            "base_spares": p.base_spares,
            "prior_spares": p.prior_spares,
        },
        "reps_sufficiency": {
            f"{t:.2f}": {
                "reps_used": reps,
                "min_reps_to_certify_wilson": min_reps_for_target(t),
                "can_certify": reps >= min_reps_for_target(t),
            }
            for t in TARGETS
        },
        "formula_validation": None if v is None else {
            "target": v.target,
            "prior_spares": v.prior_spares,
            "empirical_knee_point": v.empirical_knee,
            "certified_knee_wilson": v.certified_knee,
            "delta_packs": v.delta,
            "measured_margin_packs": v.measured_margin,
            "verdict": v.verdict,
        },
        "points": [_point_dict(pt) for pt in report.points],
    }


def _plot(path: str, report: SpareSizingReport) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional
        print(f"[plot skipped: {exc}]", file=sys.stderr)
        return False

    pts = report.points
    xs = [p.spares for p in pts]
    ys = [p.success_frac for p in pts]
    los = [p.success_frac - p.wilson_lo for p in pts]
    his = [p.wilson_hi - p.success_frac for p in pts]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(xs, ys, yerr=[los, his], marker="o", color="#1f77b4",
                capsize=3, label="success fraction (95 % Wilson CI)")

    colors = {0.99: "#d62728", 0.95: "#2ca02c"}
    for k in report.knees:
        c = colors.get(k.target, "#7f7f7f")
        ax.axhline(k.target, color=c, linestyle=":", linewidth=1, alpha=0.7)
        if k.knee_wilson is not None:
            ax.axvline(k.knee_wilson, color=c, linestyle="--", linewidth=1.2)
            ax.annotate(f"{k.target:.0%} knee\nN={k.knee_wilson}",
                        (k.knee_wilson, k.target), color=c, fontsize=8,
                        ha="left", va="bottom")

    prior_x = report.prior.prior_spares
    ax.axvline(prior_x, color="#9467bd", linestyle="-.", linewidth=1.2,
               label=f"analytical prior (spares={prior_x})")

    ax.set_xlabel("spare battery packs (shared pool size)")
    ax.set_ylabel("mission success probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Spare sizing — success-probability knee")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# rendering (stdout)                                                           #
# --------------------------------------------------------------------------- #
def _render(report: SpareSizingReport, reps: int) -> str:
    lines = [
        "# Spare-Sizing Study (paired-seed Monte-Carlo)\n",
        f"- Replications per spare count (paired): {reps}",
        f"- Analytical prior: total_sorties_int={report.prior.total_sorties_int}, "
        f"n={report.prior.n_drones}, base={report.prior.base_spares}, "
        f"margin={report.prior.margin} -> **prior_spares={report.prior.prior_spares}**\n",
        "| spares | success | 95% Wilson CI | fail | incomplete |",
        "|-------:|--------:|:--------------|-----:|-----------:|",
    ]
    for p in report.points:
        lo, hi, phat = p.wilson
        lines.append(
            f"| {p.spares} | {phat:.1%} | [{lo:.1%}, {hi:.1%}] | "
            f"{p.n_failed} | {p.n_incomplete} |"
        )
    lines.append("")
    for k in report.knees:
        kp = "—" if k.knee_point is None else str(k.knee_point)
        kw = "—" if k.knee_wilson is None else str(k.knee_wilson)
        need = min_reps_for_target(k.target)
        note = "" if reps >= need else f"  _(needs ≥{need} reps to certify; have {reps})_"
        lines.append(f"- **{k.target:.0%} target**: point-estimate knee = **{kp}**  |  "
                     f"certified (Wilson lower ≥ target) = {kw}{note}")
    v = report.verdict
    if v is not None:
        lines.append("")
        if v.verdict == "inconclusive":
            lines.append(f"- _Formula check ({v.target:.0%}): **inconclusive** — the "
                         f"sweep range did not bracket the knee (raise --span or "
                         f"widen --spare-range)._")
        else:
            sign = "" if v.delta == 0 else (f"+{v.delta}" if v.delta > 0 else str(v.delta))
            lines.append(
                f"- _Formula check ({v.target:.0%}): prior={v.prior_spares}, "
                f"empirical (point) knee={v.empirical_knee} (Δ={sign} packs) → "
                f"**{v.verdict.upper()}**. Data-demanded margin over the zero-margin "
                f"base = {v.measured_margin} packs._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def _spare_counts(args, prior) -> list[int]:
    if args.spares is not None:
        return sorted(set(args.spares))
    if args.spare_range is not None:
        start, stop, step = args.spare_range
        if step <= 0:
            raise SystemExit("--spare-range STEP must be positive")
        return list(range(start, stop + 1, step))
    return default_spare_range(prior, span=args.span)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spare-sizing success-probability study (paired-seed MC).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--reps", type=int, default=200,
                    help="paired replications per spare count")
    ap.add_argument("--margin", type=int, default=0,
                    help="additive spare-pack margin in the analytical prior")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--spares", type=int, nargs="+", default=None,
                     help="explicit spare counts to sweep")
    grp.add_argument("--spare-range", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                     default=None, help="spare-count grid (inclusive of STOP)")
    ap.add_argument("--span", type=int, default=8,
                    help="default half-width of the sweep bracket around the prior")
    ap.add_argument("--out", default="runs", help="runs/ base directory")
    ap.add_argument("--resume", default=None, metavar="PARTIAL_JSONL",
                    help="path to a previous run's results_partial.jsonl; its "
                         "completed spare counts are verified against this run's "
                         "identity (master_seed/config/reps), skipped in the "
                         "sweep, and merged into the final report")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    # Surface the Wilson-lower reps floor up front: certifying the 99 % target
    # needs ~381 reps (73 for 95 %). Below it the point-estimate knee still
    # resolves, but the strict certified knee cannot -- flag it, don't hide it.
    for t in TARGETS:
        need = min_reps_for_target(t)
        if args.reps < need:
            print(f"[warn] --reps {args.reps} < {need}: the {t:.0%} target cannot be "
                  f"*certified* under the Wilson-lower rule (point-estimate knee "
                  f"still reported).", file=sys.stderr)

    prior, coverage_j = _compute_prior(cfg, args.margin)
    counts = _spare_counts(args, prior)

    # one shared factory => paired seeds across every spare count
    rng = RngFactory(cfg.sim.master_seed)

    def _progress(pt: SparePoint):
        lo, hi, phat = pt.wilson
        print(f"  spares={pt.spares:>3}: success {phat:.1%} "
              f"[{lo:.1%}, {hi:.1%}]  (fail {pt.n_failed}, inc {pt.n_incomplete})",
              file=sys.stderr)

    # ---- structured runs/ output (created BEFORE the sweep so the crash-safe
    # partial log has a home from the first completed point) ------------------ #
    run = RunContext(base_dir=args.out)
    sim = run.simulation("spare-sizing")
    sim.write_plan({
        "schema": "uav-swarm-sim/plan/v1",
        "kind": "plan",
        "identity": sim.identity(config_hash=cfg.config_hash),
        "setup": {
            "sweep_variable": "fleet.total_reserve_batteries",
            "spare_counts": counts,
            "reps_per_point": args.reps,
            "targets": list(TARGETS),
            "margin_packs": args.margin,
            "n_drones": cfg.fleet.n_drones,
            "n_bays": cfg.swap.n_bays,
            "service_time_s": cfg.swap.service_time_s,
            "master_seed": cfg.sim.master_seed,
            "total_coverage_j": coverage_j,
        },
    })
    partial_path = sim.path(PARTIAL_FILENAME)

    print(f"Sweeping spare counts {counts[0]}..{counts[-1]} "
          f"({len(counts)} points × {args.reps} paired reps)…", file=sys.stderr)
    if args.resume:
        print(f"[resume] continuing from {args.resume}", file=sys.stderr)
    points = sweep_with_partials(cfg, counts, args.reps, rng, partial_path,
                                 resume_path=args.resume, progress=_progress)
    report = SpareSizingReport.build(points, prior)

    sim.write_results(_results_dict(report, args.reps, sim.identity()))
    plot_path = sim.path("spare_sizing_knee.png")
    plotted = _plot(str(plot_path), report)
    run.finalize(summary={
        "knees": {f"{k.target:.2f}": k.knee_wilson for k in report.knees},
        "analytical_prior_spares": report.prior.prior_spares,
        "formula_verdict": None if report.verdict is None else report.verdict.verdict,
    })

    print(_render(report, args.reps))
    print(f"\n[structured output: {run.dir}]")
    if plotted:
        print(f"[knee plot: {plot_path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
