"""Shape-regime table: the free-flight partition/regime picture for the shape study.

(Previously mislabeled the "contour-following BEFORE" table. That label was wrong:
the connectors it costs are already straight free-space chords, not contour-
following in-polygon paths -- see the framing note below.)

This is a pure-analysis driver over the *existing* A2/A3 tools (no new architecture,
no Monte Carlo, no long simulation). It builds the equal-area shape family and, for
each (shape, fleet-size) cell, reports the two regime metrics established in A2 --
the pooled lower-bound ratio and the assignment-aware per-drone **max-zone** ratio
against the real weighted-Voronoi partition -- and, from those partitions, the
shape-induced zone imbalance. It answers four questions before any MC sweep:

  1. n* table -- for every shape and n = 1..N, both metrics per cell, and the n at
     which each shape crosses into battery-limited by max-zone (the PRIMARY
     indicator; batteries are not pooled, so the busiest drone binds).
  2. Weighted vs. unweighted -- does battery-weighted decomposition cut the busiest
     drone's load (and the zone imbalance) MORE for concave shapes than compact
     ones? Weighted == unweighted at full battery by construction, so this is
     evaluated at a representative REDISTRIBUTION moment with a battery-divergence
     profile (see the flag below).
  3. H5 signal -- does zone imbalance track SOLIDITY (area / convex-hull area) or the
     "raw" perimeter/elongation measure (isoperimetric ratio)? Reported as the
     across-shape correlation of each against the balanced-partition imbalance.
  4. Sweep-grid design -- the spec (not the run) for the eventual MC study.

*** CORRECTED framing (H5-as-connector-routing is MOOT). *** This table was first
labeled the "contour-following BEFORE" of an H5 connector-routing story. That story
does not apply: under the free-flight mission premise (a drone may leave the survey
plot whenever the camera is off and no obstacle is there), the camera-off
connectors are ALREADY straight free-space chords -- the geodesic between strip
endpoints, already hull-level, already cutting across a concave notch. There is no
in-polygon contour cost for S_FERRY Step 2 to erode, so re-running this grid with
Step 2 ON does not move the partition numbers. Accordingly this is not a "BEFORE"
of anything; it is the free-flight partition/regime picture, full stop. What Step 2
*does* fix is a correctness gap (an obstacle blocking a chord: runtime S_OBS
detours, analytical E_cover did not), not the shape imbalance.

Consistent with that, the partition imbalance below tracks the ISOPERIMETRIC ratio
(elongation), NOT solidity: the thin CONVEX rect_8_1 partitions worse than the
concave star_5/pinwheel, because both thin-convex and concave pieces pay the same
elongation cost and free-flight connectors already neutralize any concavity-vs-hull
routing difference. The genuine, decomposition-driven shape result stands and is
reported here: battery-weighted decomposition cuts the busiest drone's load (and the
zone imbalance) MORE for concave shapes than for compact ones (deliverable 2).

*** THESIS-AFFECTING modeling choice (deliverable 2). *** The battery-divergence
profile is a model of one mid-mission redistribution instant, not a measured state.
Default: fleet battery fractions linearly spaced in [--f-min 0.40 (nominal), --f-max
1.00 (full)], each drone judged against its OWN remaining usable energy
(frac - critical)*capacity, and averaged over --perms slot-assignments (fixed seed)
so the result does not hinge on which drone happens to be the drained one. Vary
--f-min/--f-max/--perms and state the choice in the thesis.
"""
from __future__ import annotations

import argparse
import math
import random
import sys

from ..infrastructure.config import load_config
from ..infrastructure.core_types import DroneStateView
from ..planning.geojson_parser import load_area
from ..planning.launch_site_optimizer import InfeasibleMissionError
from ..planning.weighted_decomposition import TgcBasicDecomposer, WeightedTgcDecomposer
from .generate_shapes import describe
from .run_regime_calculator import (
    _build_planning_layer,
    _usable_fraction,
    coverage_energy,
    per_zone_energy,
    transit_and_vertical,
)

# canonical order (compact -> concave), used for every table
_SHAPE_ORDER = ["square", "rect_2_1", "rect_4_1", "rect_8_1", "disk",
                "l_shape", "star_5", "pinwheel"]


def _zone_energies(env, tgc, spec, em, motion, base_pose, drones, decomposer,
                   sensor_power_w, altitude_m):
    """Per-drone E_zone (core + that zone's transit + vertical) for an arbitrary
    decomposer and drone list -- the generalisation of run_regime_calculator's
    per_zone_energy to divergent battery fractions and either decomposer."""
    part = decomposer.decompose(tgc, env, drones, target_area=None)
    out = {}
    for did, zone in part.zones.items():
        core = coverage_energy(zone.polygon, spec, em, motion,
                               sensor_power_w)["coverage_total_j"]
        td = math.hypot(zone.entry_pose.x - base_pose.x,
                        zone.entry_pose.y - base_pose.y)
        tv = transit_and_vertical(em, spec, td, altitude_m)
        out[did] = {"area_m2": zone.polygon.area,
                    "e_zone_j": core + tv["transit_round_j"] + tv["vertical_j"]}
    return out


def _divergent_fracs(n: int, f_min: float, f_max: float) -> list[float]:
    if n == 1:
        return [f_max]
    step = (f_max - f_min) / (n - 1)
    return [f_min + i * step for i in range(n)]


def build_tables(cfg, shapes_dir, n_min, n_max, f_min, f_max, perms,
                 usable_floor, sensor_power_w, shapes=None):
    order = shapes or _SHAPE_ORDER
    frac, _ = _usable_fraction(cfg, usable_floor)
    cap = None  # set from first spec
    critical = cfg.battery_zones.critical
    altitude = cfg.env.coverage_altitude_m

    # shape descriptors (solidity, isoperimetric) from the generator, target area
    # inferred from the loaded polygon so it matches whatever was generated.
    desc = {}
    for name in order:
        poly = load_area(f"{shapes_dir}/{name}.geojson")
        d = describe(name, poly, spec_effective_swath(cfg))
        desc[name] = d

    cells = {}          # (shape, n) -> metrics
    for name in order:
        geojson = f"{shapes_dir}/{name}.geojson"
        for n in range(n_min, n_max + 1):
            env, tgc, area, spec, em, motion, base = _build_planning_layer(
                cfg, geojson, n)
            if cap is None:
                cap = spec.battery_capacity_j
            b_usable = spec.battery_capacity_j * frac

            # whole-area E_cover (for the pooled lower bound)
            cov = coverage_energy(area, spec, em, motion, sensor_power_w)
            centroid = area.centroid
            td = math.hypot(centroid.x - base.x, centroid.y - base.y)
            tv = transit_and_vertical(em, spec, td, altitude)
            e_cover = cov["coverage_total_j"] + tv["transit_round_j"] + tv["vertical_j"]
            pooled = e_cover / (n * b_usable)

            # balanced (equal-battery) partition -> deliverable 1 + 3
            bal = per_zone_energy(env, tgc, spec, em, motion, base, n,
                                  sensor_power_w, altitude, coverage=cfg.coverage)
            ez = [r["e_zone_j"] for r in bal]
            max_zone = max(ez)
            min_zone = min(ez)
            max_ratio = max_zone / b_usable
            imbalance = max_zone / min_zone if min_zone > 0 else float("nan")

            # divergent-battery redistribution moment -> deliverable 2 (n >= 2)
            w_ratio = u_ratio = w_imb = u_imb = float("nan")
            if n >= 2:
                fracs = _divergent_fracs(n, f_min, f_max)
                rng = random.Random(1234 + n)
                wr, ur, wi, ui = [], [], [], []
                for _ in range(perms):
                    perm = fracs[:]
                    rng.shuffle(perm)
                    drones = [DroneStateView(id=i, battery_frac=perm[i], pose=base)
                              for i in range(n)]
                    rem = {i: max(0.0, (perm[i] - critical)) * spec.battery_capacity_j
                           for i in range(n)}
                    w = _zone_energies(env, tgc, spec, em, motion, base, drones,
                                       WeightedTgcDecomposer(), sensor_power_w, altitude)
                    u = _zone_energies(env, tgc, spec, em, motion, base, drones,
                                       TgcBasicDecomposer(), sensor_power_w, altitude)
                    w_ratios = [w[i]["e_zone_j"] / rem[i] if rem[i] > 0 else float("inf")
                                for i in w]
                    u_ratios = [u[i]["e_zone_j"] / rem[i] if rem[i] > 0 else float("inf")
                                for i in u]
                    wr.append(max(w_ratios))
                    ur.append(max(u_ratios))
                    # "balance" = spread of work-per-remaining-battery across drones;
                    # the weighting's JOB is to equalise this (it deliberately makes
                    # zone *areas* unequal, so raw energy imbalance is the wrong lens).
                    wi.append(max(w_ratios) / min(w_ratios)
                              if min(w_ratios) > 0 and math.isfinite(min(w_ratios)) else float("nan"))
                    ui.append(max(u_ratios) / min(u_ratios)
                              if min(u_ratios) > 0 and math.isfinite(min(u_ratios)) else float("nan"))
                w_ratio, u_ratio = _mean(wr), _mean(ur)
                w_imb, u_imb = _mean(wi), _mean(ui)

            cells[(name, n)] = {
                "pooled": pooled, "max_ratio": max_ratio, "imbalance": imbalance,
                "n_strips": cov["n_strips"], "e_cover": e_cover,
                "w_ratio": w_ratio, "u_ratio": u_ratio,
                "w_imb": w_imb, "u_imb": u_imb,
            }
    return desc, cells, cap


def spec_effective_swath(cfg) -> float:
    from ..physical_model.drone_specs import build_spec
    return build_spec(cfg).swath_width_m


def _mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    syy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def _fmt_grid(cells, desc, n_min, n_max, key, fmt="{:.2f}"):
    ns = list(range(n_min, n_max + 1))
    head = "| shape | solidity | " + " | ".join(f"n={n}" for n in ns) + " |"
    sep = "|:--|--:|" + "".join("--:|" for _ in ns)
    lines = [head, sep]
    for name in _SHAPE_ORDER:
        cellvals = []
        for n in ns:
            v = cells[(name, n)][key]
            cellvals.append(fmt.format(v) if math.isfinite(v) else "—")
        lines.append(f"| {name} | {desc[name]['solidity']:.3f} | "
                     + " | ".join(cellvals) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--shapes-dir", default="data/areas/shapes")
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--f-min", type=float, default=0.40,
                    help="most-drained drone battery frac for the divergence model "
                         "(THESIS-AFFECTING; default 0.40 = nominal)")
    ap.add_argument("--f-max", type=float, default=1.00,
                    help="freshest drone battery frac (default 1.00 = full)")
    ap.add_argument("--perms", type=int, default=8,
                    help="battery↔slot permutations averaged (default 8)")
    ap.add_argument("--usable-floor", choices=["terminal", "return", "rth"],
                    default="terminal")
    ap.add_argument("--sensor-power-w", type=float, default=None)
    ap.add_argument("--csv", default=None, help="also write the full per-cell grid to this CSV")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sensor_power_w = (args.sensor_power_w if args.sensor_power_w is not None
                      else cfg.sensor.sensor_power_w)
    try:
        desc, cells, cap = build_tables(
            cfg, args.shapes_dir, args.n_min, args.n_max, args.f_min, args.f_max,
            args.perms, args.usable_floor, sensor_power_w)
    except InfeasibleMissionError as exc:
        print("MISSION IMPOSSIBLE (launch siting)"); print(str(exc)); return 2

    frac, frac_label = _usable_fraction(cfg, args.usable_floor)
    b_usable = cap * frac
    ns = list(range(args.n_min, args.n_max + 1))

    print("# Shape-regime table — free-flight partition/regime picture\n")
    print("> Camera-off connectors are already straight free-space chords (free-flight "
          "premise), so this is NOT a contour-following 'BEFORE' and S_FERRY Step 2 does "
          "not move these partition numbers — H5-as-connector-routing is moot. Imbalance "
          "tracks the isoperimetric (elongation) ratio, not solidity; the real shape "
          "result is decomposition-driven (weighted cuts concave imbalance more).\n")
    print(f"- Fleet budget floor: --usable-floor {args.usable_floor} → {frac_label} "
          f"→ B_usable {b_usable:,.0f} J per drone (THESIS-AFFECTING)")
    print(f"- Regime classified on the per-drone **max-zone** ratio (primary); "
          f"pooled ratio is a lower bound only.\n")

    # ---- Deliverable 1 ------------------------------------------------------ #
    print("## 1. n\\* table — max-zone ratio (pooled ratio in the read-out below)\n")
    print("Per-drone max-zone / B_usable at full battery (>1 ⇒ busiest drone must "
          "swap ⇒ battery-limited):\n")
    print(_fmt_grid(cells, desc, args.n_min, args.n_max, "max_ratio"))
    print("\nPooled E_cover / (n·B_usable) — lower bound, for comparison:\n")
    print(_fmt_grid(cells, desc, args.n_min, args.n_max, "pooled"))
    print("\nBattery-limited-by-max-zone crossover (largest n with max-zone > 1; "
          "battery-limited for n ≤ this):\n")
    for name in _SHAPE_ORDER:
        bl = [n for n in ns if cells[(name, n)]["max_ratio"] > 1.0]
        n_cross = max(bl) if bl else 0
        note = f"battery-limited for n ≤ {n_cross}" if n_cross else "surplus across the grid"
        print(f"- {name:9s} (solidity {desc[name]['solidity']:.3f}): {note}")

    # ---- Deliverable 2 ------------------------------------------------------ #
    print("\n## 2. Weighted vs. unweighted at a redistribution moment "
          f"(battery ∈ [{args.f_min:.2f}, {args.f_max:.2f}], {args.perms} perms)\n")
    print("Busiest-drone ratio = max_i (E_zone_i / remaining_usable_i). Lower is "
          "better; 'reduction' is what the battery weighting buys. 'work/batt spread' "
          "= max/min of that ratio across drones (weighting should EQUALISE it → "
          "lower spread):\n")
    print("| shape | solidity | n | unweighted max | weighted max | reduction | "
          "unwt spread | wtd spread |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|")
    peak = {}
    for name in _SHAPE_ORDER:
        for n in ns:
            if n < 2:
                continue
            c = cells[(name, n)]
            red = c["u_ratio"] - c["w_ratio"]
            if math.isfinite(red):
                peak.setdefault(name, []).append((n, red, c))
            print(f"| {name} | {desc[name]['solidity']:.3f} | {n} | "
                  f"{c['u_ratio']:.3f} | {c['w_ratio']:.3f} | {red:+.3f} | "
                  f"{c['u_imb']:.2f} | {c['w_imb']:.2f} |")
    # per-shape peak reduction and its correlation with solidity
    sol, redmax = [], []
    print("\nPeak weighting benefit per shape (max reduction over n) vs solidity:\n")
    for name in _SHAPE_ORDER:
        if name in peak and peak[name]:
            n_at, r_at, _ = max(peak[name], key=lambda t: t[1])
            sol.append(desc[name]["solidity"]); redmax.append(r_at)
            print(f"- {name:9s} (solidity {desc[name]['solidity']:.3f}): "
                  f"peak reduction {r_at:+.3f} at n={n_at}")
    if len(sol) >= 3:
        r = _pearson(sol, redmax)
        print(f"\n→ correlation(peak weighting benefit, solidity) = {r:+.3f} "
              f"(negative ⇒ MORE benefit for concave/low-solidity shapes, as H expects)")

    # ---- Deliverable 3 ------------------------------------------------------ #
    print("\n## 3. H5 signal — does balanced-partition imbalance track solidity or "
          "the raw isoperimetric ratio?\n")
    print("Balanced (full-battery) zone imbalance, averaged over "
          f"n={max(2, args.n_min)}..{args.n_max}, vs the two shape measures:\n")
    print("| shape | solidity | isoperimetric | mean imbalance |")
    print("|:--|--:|--:|--:|")
    sol2, iso2, imb2 = [], [], []
    for name in _SHAPE_ORDER:
        vals = [cells[(name, n)]["imbalance"] for n in ns
                if n >= 2 and math.isfinite(cells[(name, n)]["imbalance"])]
        mimb = _mean(vals)
        sol2.append(desc[name]["solidity"])
        iso2.append(desc[name]["isoperimetric"])
        imb2.append(mimb)
        print(f"| {name} | {desc[name]['solidity']:.3f} | "
              f"{desc[name]['isoperimetric']:.3f} | {mimb:.2f} |")
    r_sol = _pearson(sol2, imb2)
    r_iso = _pearson(iso2, imb2)
    n_convex = sum(1 for s in sol2 if s > 0.999)
    print(f"\n- correlation(imbalance, solidity)      = {r_sol:+.3f}")
    print(f"- correlation(imbalance, isoperimetric) = {r_iso:+.3f}")
    print(f"\nRead (data-driven): solidity is ~constant (= 1.0) for {n_convex} of "
          f"{len(sol2)} shapes in this family, so it has little variance to correlate "
          "on — any solidity signal must come from the 3 concave shapes alone.")
    if abs(r_iso) > abs(r_sol) + 0.15:
        print("At the partition level imbalance tracks the ISOPERIMETRIC "
              "(elongation/perimeter) measure MORE than solidity: the thin CONVEX "
              "rectangles partition as badly as — or worse than — the concave shapes. "
              "H5's convex-hull-of-the-survey-shape framing does NOT govern partition "
              "imbalance; concavity per se is not the driver. And since connectors are "
              "already free-space chords, S_FERRY Step 2 does not change this — the "
              "genuine shape effect is decomposition-driven (weighted cuts concave "
              "imbalance more), not connector-routing.")
    elif abs(r_sol) > abs(r_iso) + 0.15:
        print("Imbalance tracks SOLIDITY more than the isoperimetric measure, so "
              "H5's convex-hull framing already holds at the partition level.")
    else:
        print("Neither measure clearly dominates in this 8-shape family; a family "
              "with more graded solidity values is needed to separate them cleanly.")

    # ---- Deliverable 4 ------------------------------------------------------ #
    print("\n## 4. Sweep-grid design for the eventual MC study (spec, not a run)\n")
    print("- Fleet: n = 2..6 (spans the transition: small n all battery-limited → "
          "n≈4 shape-divergent → large n all surplus).")
    print("- Shapes: all 8 (square, rect_2_1/4_1/8_1, disk, l_shape, star_5, pinwheel).")
    print("- Decomposition algos: all 4 — weighted_voronoi (contribution), tgc_basic "
          "(unweighted ablation twin), classic_voronoi, kmeans.")
    print("- Platform: MULTIROTOR only; dynamic obstacles OFF; hazard λ = 0.")
    print("- Battery/area/launch: the shipped baseline (1 km², 100 Wh, optimizer-sited "
          "launch) — NOT retuned; the baseline already straddles the transition.")
    print("- Baseline reference cell: **n = 4** (the divergence peak — square "
          "borderline-surplus while pinwheel is battery-limited).")
    print("- Metrics per cell: SMDP efficiency, workload/energy balance, swap count, "
          "makespan; compared weighted vs the three baselines on paired seeds.")
    print("- No BEFORE/AFTER over connector routing: connectors are already free-space "
          "chords, so S_FERRY Step 2 leaves these partition numbers unchanged (it only "
          "fixes analytical=execution on obstacle-blocked chords). The shape effect to "
          "sweep is decomposition-driven (weighted vs baselines), not routing.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shape", "solidity", "isoperimetric", "n", "pooled_ratio",
                        "max_zone_ratio", "balanced_imbalance", "unweighted_max",
                        "weighted_max", "unweighted_imb", "weighted_imb"])
            for name in _SHAPE_ORDER:
                for n in ns:
                    c = cells[(name, n)]
                    w.writerow([name, f"{desc[name]['solidity']:.4f}",
                                f"{desc[name]['isoperimetric']:.4f}", n,
                                f"{c['pooled']:.4f}", f"{c['max_ratio']:.4f}",
                                f"{c['imbalance']:.4f}", f"{c['u_ratio']:.4f}",
                                f"{c['w_ratio']:.4f}", f"{c['u_imb']:.4f}",
                                f"{c['w_imb']:.4f}"])
        print(f"\n_Full per-cell grid written to {args.csv}._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
