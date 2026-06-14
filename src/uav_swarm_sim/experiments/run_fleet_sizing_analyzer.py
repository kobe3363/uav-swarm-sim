"""B6.2 standalone fleet-sizing decision-support tool.

Builds the planning layer (EnvironmentMap + the energy-aware LaunchSiteOptimizer)
to obtain the REAL base pose and navigable extent, then hands that geometry to
the pure ``fleet_sizing`` core and prints a "law of diminishing returns" Pareto
table the operator reads to pick N (and then sets manually in default.yaml).

It deliberately does NOT import or run the SimulationEngine, Agent, or any
dt-stepped loop -- the heavy simulation is a separate, manual step afterwards.

Reproducibility: the environment and launch site are built from the SAME seeded
streams the simulation uses (master_seed, "obstacles", "launch_sampling",
replication 0), so the base pose here matches the one the heavy run would use.
"""
from __future__ import annotations

import argparse
import math
import sys

from ..infrastructure.config import load_config
from ..infrastructure.rng import STREAM_LAUNCH_SAMPLING, STREAM_OBSTACLES, RngFactory
from ..physical_model.aero_correction import AeroCorrection
from ..physical_model.drone_specs import build_spec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import make_motion_model
from ..planning.environment_map import EnvironmentMap
from ..planning.geojson_parser import load_area
from ..planning.gvg_builder import build_gvg
from ..planning.launch_site_optimizer import (
    InfeasibleMissionError,
    optimize,
    _furthest_free_vertex_dist,
)
from ..planning.obstacle_generator import generate as generate_obstacles
from ..planning.tgc import build_tgc
from .fleet_sizing import TURN_FACTOR_DEFAULT, FleetSizingInputs, sweep


# --------------------------------------------------------------------------- #
# formatting                                                                  #
# --------------------------------------------------------------------------- #
def _fmt_dur(seconds: float) -> str:
    s = int(round(abs(seconds)))
    days, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if days > 0:
        return f"{days}d {h:02d}:{m:02d}:{sec:02d}"
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _fmt_saved(marginal_s: float, is_first: bool) -> str:
    if is_first:
        return "—"
    if marginal_s < -1e-6:
        return f"(+{_fmt_dur(marginal_s)} worse)"
    return _fmt_dur(marginal_s)


# --------------------------------------------------------------------------- #
# planning-layer setup (the "use the real base_pose" step)                    #
# --------------------------------------------------------------------------- #
def _build_planning_layer(cfg):
    """Reproduce the engine's layer-0 world + real launch site, deterministically.
    Returns (env, spec, em, base_pose)."""
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

    # the energy-aware optimizer raises InfeasibleMissionError if no site works
    base_pose, _scores = optimize(
        cfg.launch, tgc, env, motion, em, aero, spec,
        cfg.fleet.n_drones, launch_rng, cfg.env.coverage_altitude_m,
    )
    return env, spec, em, base_pose


def _inputs_from(env, base_pose, altitude_m: float) -> FleetSizingInputs:
    centroid = env.free_space.centroid
    transit = math.hypot(centroid.x - base_pose.x, centroid.y - base_pose.y)
    return FleetSizingInputs(
        area_m2=float(env.free_space.area),
        furthest_dist_m=_furthest_free_vertex_dist(env, base_pose.as_xy()),
        transit_dist_m=transit,
        altitude_m=altitude_m,
    )


# --------------------------------------------------------------------------- #
# rendering                                                                   #
# --------------------------------------------------------------------------- #
def _render_table(report, knee_n) -> str:
    lines = [
        "| N | Est. Mission Time | Est. Total Swaps | Time Saved vs N-1 |",
        "|--:|:------------------|-----------------:|:------------------|",
    ]
    first_n = report.rows[0].n
    for row in report.rows:
        knee = "  ← knee" if (knee_n is not None and row.n == knee_n) else ""
        lines.append(
            f"| {row.n} | {_fmt_dur(row.est_duration_s)} | {row.est_total_swaps} | "
            f"{_fmt_saved(row.marginal_time_saved_s, row.n == first_n)}{knee} |"
        )
    return "\n".join(lines)


def _find_knee(report, knee_frac: float):
    """First N (after the smallest) whose marginal time saved falls below
    ``knee_frac`` of the single-drone duration -- the diminishing-returns point."""
    if not report.rows:
        return None
    base_duration = report.rows[0].est_duration_s
    threshold = knee_frac * base_duration
    for row in report.rows[1:]:
        if row.marginal_time_saved_s < threshold:
            return row.n
    return None


def _save_pareto(report, path: str, knee_n) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional
        print(f"[plot skipped: {exc}]", file=sys.stderr)
        return False
    ns = [r.n for r in report.rows]
    durs = [r.est_duration_s / 3600.0 for r in report.rows]  # hours
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, durs, marker="o", color="#1f77b4", label="est. mission time")
    if knee_n is not None:
        ky = next(r.est_duration_s / 3600.0 for r in report.rows if r.n == knee_n)
        ax.scatter([knee_n], [ky], color="#d62728", zorder=5, label=f"knee (N={knee_n})")
    ax.set_xlabel("fleet size N")
    ax.set_ylabel("estimated mission time (h)")
    ax.set_title("Fleet sizing — law of diminishing returns")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Standalone fleet-sizing Pareto analyzer (no simulation).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n-max", type=int, default=20, help="largest fleet size to evaluate")
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--turn-factor", type=float, default=TURN_FACTOR_DEFAULT)
    ap.add_argument("--knee-frac", type=float, default=0.05,
                    help="diminishing-returns threshold as a fraction of the N=1 duration")
    ap.add_argument("--plot", default=None, help="optional path to save a Pareto PNG")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    try:
        env, spec, em, base_pose = _build_planning_layer(cfg)
        inputs = _inputs_from(env, base_pose, cfg.env.coverage_altitude_m)
        report = sweep(
            inputs, em, spec,
            effective_swath_m=spec.swath_width_m,
            service_time_s=cfg.swap.service_time_s,
            n_bays=cfg.swap.n_bays,
            n_min=args.n_min,
            n_max=args.n_max,
            turn_factor=args.turn_factor,
            reserve_frac=cfg.rth.reserve_frac,
        )
    except InfeasibleMissionError as exc:
        print("MISSION IMPOSSIBLE")
        print(str(exc))
        return 2

    knee_n = _find_knee(report, args.knee_frac)

    # --- header / assumptions ------------------------------------------------ #
    b = report.budget
    print("# Fleet-Sizing Analysis (analytical — no simulation run)\n")
    print(f"- Platform: {spec.platform.value}  |  usable battery: {b.usable_j:,.0f} J "
          f"({spec.battery_capacity_j:,.0f} J × {1 - cfg.rth.reserve_frac:.2f})")
    print(f"- Launch site (real, from optimizer): ({base_pose.x:,.1f}, {base_pose.y:,.1f})")
    print(f"- Navigable area: {inputs.area_m2:,.0f} m²  |  furthest navigable point: "
          f"{inputs.furthest_dist_m:,.0f} m  |  representative transit: {inputs.transit_dist_m:,.0f} m")
    print(f"- Effective swath: {spec.swath_width_m:,.1f} m  |  turn factor: {args.turn_factor}")
    print(f"- Total coverage path: {report.total_coverage_length_m:,.0f} m "
          f"({report.total_coverage_j:,.0f} J)")
    print(f"- Per-sortie: overhead {b.overhead_j:,.0f} J (takeoff+transit+landing), "
          f"coverage budget {b.coverage_budget_j:,.0f} J, full-sortie time {_fmt_dur(b.t_sortie_s)}")
    print(f"- **Total required sorties (battery cycles): {report.total_sorties:.2f} "
          f"→ {report.total_sorties_int}**")
    print(f"- Swap station: {cfg.swap.n_bays} bays × {cfg.swap.service_time_s:.0f} s "
          f"(ground queue = TIME only, zero energy)\n")

    print(_render_table(report, knee_n))

    if knee_n is not None:
        print(f"\n_Diminishing returns past **N = {knee_n}**: each further drone saves "
              f"less than {args.knee_frac:.0%} of the single-drone mission time._")
    else:
        print(f"\n_No clear knee within N ≤ {args.n_max} at the {args.knee_frac:.0%} threshold._")

    print("\n_Note: duration via (÷N) is an optimistic bound (perfectly divisible "
          "coverage, one representative transit). Feasibility is gated separately "
          "by the furthest-point reach-and-return check; coverage length is a floor "
          "(obstacle detours add more)._")

    if args.plot:
        if _save_pareto(report, args.plot, knee_n):
            print(f"\n[Pareto plot saved to {args.plot}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
