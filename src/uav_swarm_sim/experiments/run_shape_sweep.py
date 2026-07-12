"""S5 -- the shape-sweep Monte-Carlo study (the thesis's central empirical grid).

Sweeps survey SHAPE (9 equal-area 1 km^2 polygons) x fleet size n x
decomposition variant on PAIRED SEEDS and reports per-cell metrics, paired
contrasts with CIs, an analytical A2 regime overlay, and an honest H1/H2
read-out.

Design decisions (all THESIS-AFFECTING, agreed with the author)
---------------------------------------------------------------
* HOMOGENEOUS fleet, lambda = 0 (no hazard), dynamic obstacles OFF, MULTIROTOR,
  the shipped 1 km^2 / 100 Wh / optimizer-sited baseline -- NOT retuned.
* SCOPED NULL (empirically verified before this harness was written): for an
  identical full-battery fleet the battery-weighted decomposition is inactive:
    - initial partition uses equal battery_frac=1.0 for every drone, and at
      equal fractions WeightedTgcDecomposer == TgcBasicDecomposer exactly;
    - mid-mission redistribution fires only on FAILURE (lambda=0: never) and
      always routes through WeightedTgcDecomposer regardless of mission algo.
  Therefore weighted_voronoi - tgc_basic == 0 across the whole grid. The sweep
  RUNS weighted_voronoi anyway to document that null empirically (a scoped
  finding: battery-weighting's role is the diverged-battery / redistribution
  regime, outside this study).
* HEADLINE contrast: TGC (represented by tgc_basic) vs classic_voronoi and vs
  kmeans, per shape x n.
* SECONDARY AXIS: optimal launch (shipped optimizer, 8 staging-ring candidates)
  vs NAIVE-CENTROID launch for tgc_basic. Launch pads must sit OUTSIDE the
  survey area (staging periphery), so "centroid launch" is defined as the
  single feasible site nearest the survey-area centroid: the centroid projected
  to the area boundary and pushed a few metres outward into the staging band.
  Deterministic per shape; passed as an explicit one-element candidate list so
  the optimizer machinery (feasibility check included) is reused unchanged.
* MODES: --mode clean (obstacle_density = 0; isolates the pure shape effect and
  matches the analytical A2 tags exactly) is PRIMARY; --mode shipped keeps the
  config's static obstacle density as a robustness check on the n in {2, 4}
  subset (reference row + battery-limited row), obstacles paired per seed.
* FIXED N per cell (n_min = n_max = N), NOT CI-convergence stopping:
  early-stopping would let variants converge at different run counts and break
  exact seed pairing. Budgets: --budget quick -> N=5 clean / N=10 shipped;
  --budget full -> N=20 clean / N=100 shipped; --n-runs overrides. In clean
  mode the residual across-replication variance is launch-site sampling only
  (deterministic algos share the paired launch site per replication, so e.g.
  the weighted-vs-tgc contrast is exact at any N).
* Metrics per cell, mean +/- 95% CI over the N paired runs:
    - SMDP efficiency (throughput headline),
    - EXECUTED energy imbalance max/mean over drones (PRIMARY balance metric,
      from MissionMetrics.per_agent_energy_j),
    - flown-length imbalance max/mean (secondary proxy),
    - swap count, makespan (mission duration), mission-success fraction.
  Alongside: the ANALYTICAL planned-zone imbalance max/min E_zone (the BEFORE
  table's framing, for continuity) computed per algo on the clean planning
  layer with equal battery fractions.
* Regime overlay: every (shape, n) cell tagged with the A2 regime using the
  exact run_regime_calculator rule (pooled ratio + per-drone max-zone ratio,
  usable floor "terminal", borderline band 1 +/- 0.10) on the OBSTACLE-FREE
  planning layer. n = 4 is the highlighted reference row.

Usage
-----
python -m uav_swarm_sim.experiments.run_shape_sweep \
    [--config config/default.yaml] [--shapes-dir data/areas/shapes] \
    [--mode clean|shipped] [--budget quick|full] [--n-runs N] \
    [--shapes square,c_shape,...] [--n 2,3,4,5,6] \
    [--base runs] [--run-name NAME]

Outputs (under runs/<run>/): shape_sweep.csv (per cell x variant),
contrasts.csv (paired differences with CIs), summary.md (tables + the honest
H1/H2 read-out), manifest.
"""
from __future__ import annotations

import os

# ENG-09 (B5): pin BLAS/OpenMP to a single thread BEFORE numpy is imported so
# (a) N worker processes do not oversubscribe cores with N*threads, and (b) the
# floating-point reduction order is identical in the serial and parallel paths
# -> bitwise-identical CSVs. setdefault leaves an explicit user override intact.
# In spawn mode every worker re-imports this module first, so the pin also takes
# effect in each child before its numpy loads.
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import argparse
import csv
import dataclasses
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from ..infrastructure import profiling
from ..infrastructure.config import Config, MCConfig, load_config
from ..infrastructure.enums import DecompositionAlgo, Outcome, PlannerKind
from ..infrastructure.rng import STREAM_KMEANS_INIT, RngFactory
from ..metrics.comparison import VariantResult, run_variant
from ..metrics.convergence import ci_half_width
from ..metrics.run_output import RunContext, unique_run_name
from ..planning.classic_voronoi import ClassicVoronoiDecomposer
from ..planning.geojson_parser import load_area
from ..planning.kmeans_heuristic import KMeansHeuristicDecomposer
from ..planning.launch_site_optimizer import (
    _STAGING_STANDOFF_M,
    InfeasibleMissionError,
)
from ..planning.weighted_decomposition import TgcBasicDecomposer, WeightedTgcDecomposer
from ..physical_model.motion_model import make_motion_model
from ..physical_model.drone_specs import build_spec
from .generate_shapes import describe
from .run_regime_calculator import (
    _BORDERLINE_MARGIN,
    _build_planning_layer,
    _usable_fraction,
    classify,
    coverage_energy,
    per_zone_energy,
    transit_and_vertical,
)
from .run_shape_regime_table import spec_effective_swath

# canonical order (compact -> concave), C-shape appended as the primary
# concave test vehicle
SHAPE_ORDER = ["square", "rect_2_1", "rect_4_1", "rect_8_1", "disk",
               "l_shape", "star_5", "pinwheel", "c_shape"]

REFERENCE_N = 4  # the divergence-peak reference row (A2 finding)

# variant labels. The four decomposition peers run with the shipped optimized
# launch; the naive-launch twins below re-run three of them on the launch-axis
# NAIVE-centroid pad (the launch-confound ablation).
ALGO_VARIANTS: tuple[DecompositionAlgo, ...] = (
    DecompositionAlgo.WEIGHTED_VORONOI,
    DecompositionAlgo.TGC_BASIC,
    DecompositionAlgo.CLASSIC_VORONOI,
    DecompositionAlgo.KMEANS,
)
# NAIVE-launch twins: each label re-runs its decomposition algo on the SAME
# naive-centroid pad (one pad + one deploy ring per replication, shared across
# every algo -> launch is neutralised while the algorithm axis varies). This
# isolates whether the TGC~kmeans verdict survives without TGC's optimizer-sited
# "home pad" (Problem B). weighted_naive is INTENTIONALLY absent: for an
# identical full-battery fleet weighted_voronoi == tgc_basic byte-identically
# (the documented scoped null), so a weighted twin would only duplicate
# tgc_naive_launch.
NAIVE_LAUNCH_VARIANTS: dict[str, DecompositionAlgo] = {
    "tgc_naive_launch": DecompositionAlgo.TGC_BASIC,
    "classic_naive_launch": DecompositionAlgo.CLASSIC_VORONOI,
    "kmeans_naive_launch": DecompositionAlgo.KMEANS,
}
NAIVE_LAUNCH_LABEL = "tgc_naive_launch"  # back-compat alias (regime tag, tests)

# metric extractors: label -> (higher_is_better, fn(SingleRunResult) -> float)
_NAIVE_OFFSET_M = 5.0  # outward push of the centroid-projected pad (< standoff)


# --------------------------------------------------------------------------- #
# naive-centroid launch site                                                  #
# --------------------------------------------------------------------------- #
def naive_centroid_site(area) -> tuple[float, float]:
    """The launch-axis NAIVE baseline: the feasible pad nearest the survey-area
    centroid. Pads must live in the staging periphery OUTSIDE the area, so the
    centroid is projected to the nearest boundary point and pushed
    ``_NAIVE_OFFSET_M`` outward (away from the centroid). If that direction
    re-enters the polygon (concave pockets), 16 compass directions around the
    boundary point are scanned at increasing radii for the first point that is
    outside the area but inside the staging band. Deterministic per shape."""
    c = area.centroid
    p, _ = nearest_points(area.exterior, c)
    band = area.buffer(_STAGING_STANDOFF_M).difference(area)

    dx, dy = p.x - c.x, p.y - c.y
    d = math.hypot(dx, dy)
    if d > 0.0:
        q = Point(p.x + dx / d * _NAIVE_OFFSET_M, p.y + dy / d * _NAIVE_OFFSET_M)
        if band.covers(q):
            return (q.x, q.y)
    for r in (_NAIVE_OFFSET_M, 5 * _NAIVE_OFFSET_M, 10 * _NAIVE_OFFSET_M):
        for k in range(16):
            a = 2.0 * math.pi * k / 16.0
            q = Point(p.x + r * math.cos(a), p.y + r * math.sin(a))
            if band.covers(q):
                return (q.x, q.y)
    raise RuntimeError("no naive-centroid staging point found (degenerate area)")


# --------------------------------------------------------------------------- #
# per-cell config                                                             #
# --------------------------------------------------------------------------- #
def build_cell_cfg(base: Config, shape_path: str, n: int, mode: str,
                   n_runs: int, launch_site: tuple[float, float] | None = None,
                   ) -> Config:
    """The (shape, n) cell config. Clean mode zeroes the static obstacle
    density (the shipped default is 8/km^2) to isolate the shape effect;
    hazard lambda is forced to 0 in BOTH modes (agreed scope). Fixed-N MC via
    n_min = n_max = N so every variant runs the identical replication set."""
    env = dataclasses.replace(
        base.env, geojson_path=shape_path,
        obstacle_density_per_km2=(0.0 if mode == "clean"
                                  else base.env.obstacle_density_per_km2))
    fleet = dataclasses.replace(base.fleet, n_drones=n)
    failure = dataclasses.replace(base.failure, hazard_rate_per_hour=0.0)
    mc = MCConfig(n_max=n_runs, n_min=n_runs, ci_tolerance=0.0)
    # Telemetry is a per-run diagnostic (GPX + JSONL to a FIXED path), not a
    # Monte-Carlo output: with it on, every one of the thousands of sweep missions
    # rebuilds a TelemetryLog and re-exports to the same overwritten file (pure
    # waste + concurrent-write contention under --jobs). It is a read-only probe,
    # so forcing it OFF is byte-identical for every metric (mirrors hazard_rate=0).
    telemetry = dataclasses.replace(base.telemetry, enabled=False)
    cfg = dataclasses.replace(base, env=env, fleet=fleet, failure=failure, mc=mc,
                              telemetry=telemetry)
    if launch_site is not None:
        launch = dataclasses.replace(cfg.launch, candidate_sites=(launch_site,))
        cfg = dataclasses.replace(cfg, launch=launch)
    return cfg


# --------------------------------------------------------------------------- #
# A2 regime tag (analytical, obstacle-free -- mirrors run_regime_calculator)  #
# --------------------------------------------------------------------------- #
def regime_tag(clean_cfg: Config, shape_path: str, n: int,
               usable_floor: str = "terminal") -> dict:
    """The exact A2 classification for one (shape, n) cell: pooled lower bound
    E_cover/(n*B_usable) plus the assignment-aware per-drone max-zone ratio;
    battery-limited if either exceeds 1, borderline band 1 +/- 0.10."""
    env, tgc, area, spec, em, motion, base_pose = _build_planning_layer(
        clean_cfg, shape_path, n)
    sensor_power_w = clean_cfg.sensor.sensor_power_w
    cov = coverage_energy(area, spec, em, motion, sensor_power_w)
    centroid = area.centroid
    td = math.hypot(centroid.x - base_pose.x, centroid.y - base_pose.y)
    tv = transit_and_vertical(em, spec, td, clean_cfg.env.coverage_altitude_m)
    e_cover = cov["coverage_total_j"] + tv["transit_round_j"] + tv["vertical_j"]

    frac, _ = _usable_fraction(clean_cfg, usable_floor)
    b_usable = spec.battery_capacity_j * frac
    pooled_ratio = e_cover / (n * b_usable) if b_usable > 0 else float("inf")
    zone_rows = per_zone_energy(env, tgc, spec, em, motion, base_pose, n,
                                sensor_power_w, clean_cfg.env.coverage_altitude_m,
                                coverage=clean_cfg.coverage)
    max_zone_j = max(r["e_zone_j"] for r in zone_rows)
    min_zone_j = min(r["e_zone_j"] for r in zone_rows)
    max_zone_ratio = max_zone_j / b_usable
    if pooled_ratio > 1.0 or max_zone_ratio > 1.0:
        regime = "BATTERY-LIMITED"
    elif (classify(pooled_ratio) == "BORDERLINE"
          or max_zone_ratio > 1.0 - _BORDERLINE_MARGIN):
        regime = "BORDERLINE"
    else:
        regime = "FUEL-SURPLUS"
    return {"regime": regime, "pooled_ratio": pooled_ratio,
            "max_zone_ratio": max_zone_ratio,
            "planned_imbalance_weighted": (max_zone_j / min_zone_j
                                           if min_zone_j > 0 else float("nan"))}


def planned_imbalance(clean_cfg: Config, shape_path: str, n: int,
                      algo: DecompositionAlgo) -> float:
    """ANALYTICAL planned-zone imbalance max/min E_zone for one peer algo on
    the clean planning layer with EQUAL battery fractions (the BEFORE table's
    framing, kept for continuity next to the executed max/mean metric)."""
    from ..infrastructure.core_types import DroneStateView
    from ..execution.fleet import deploy_ring_poses

    env, tgc, area, spec, em, motion, base_pose = _build_planning_layer(
        clean_cfg, shape_path, n)
    if algo is DecompositionAlgo.CLASSIC_VORONOI:
        dec = ClassicVoronoiDecomposer()
    elif algo is DecompositionAlgo.TGC_BASIC:
        dec = TgcBasicDecomposer()
    elif algo is DecompositionAlgo.WEIGHTED_VORONOI:
        dec = WeightedTgcDecomposer()
    else:
        rng = RngFactory(clean_cfg.sim.master_seed).stream(STREAM_KMEANS_INIT, 0)
        dec = KMeansHeuristicDecomposer(make_motion_model(build_spec(clean_cfg)),
                                        weighted=False, rng=rng)
    # Seed drones on the SAME deploy ring the engine uses (position-based peers
    # classic/kmeans partition by seed location; all-at-base collapses Voronoi).
    poses = deploy_ring_poses(base_pose, n, spec.dims_m,
                              clean_cfg.safety.min_separation_m)
    drones = [DroneStateView(id=i, battery_frac=1.0, pose=poses[i])
              for i in range(n)]
    part = dec.decompose(tgc, env, drones, target_area=None)
    e = []
    for zone in part.zones.values():
        if zone.polygon.is_empty:  # degenerate zone -> imbalance undefined
            return float("nan")
        core = coverage_energy(zone.polygon, spec, em, motion,
                               clean_cfg.sensor.sensor_power_w)["coverage_total_j"]
        td = math.hypot(zone.entry_pose.x - base_pose.x,
                        zone.entry_pose.y - base_pose.y)
        tv = transit_and_vertical(em, spec, td, clean_cfg.env.coverage_altitude_m)
        e.append(core + tv["transit_round_j"] + tv["vertical_j"])
    return max(e) / min(e) if e and min(e) > 0 else float("nan")


# --------------------------------------------------------------------------- #
# per-run metric extraction + paired contrasts                                #
# --------------------------------------------------------------------------- #
METRICS = ("efficiency", "total_energy", "energy_imbalance", "length_imbalance",
           "swaps", "makespan", "success")
HIGHER_IS_BETTER = {"efficiency": True, "total_energy": False,
                    "energy_imbalance": False, "length_imbalance": False,
                    "swaps": False, "makespan": False, "success": True}


def _imb(values) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    mean = sum(vals) / len(vals)
    return max(vals) / mean if mean > 0 else float("nan")


def metric_vectors(vr: VariantResult) -> dict[str, list[float]]:
    """Per-replication metric samples, index-aligned to the replication order
    (the pairing axis). NaNs are kept in place so paired differences can drop
    the SAME replications on both sides."""
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for r in vr.mc.runs:
        m = r.metrics
        out["efficiency"].append(float(r.efficiency))
        out["total_energy"].append(float(m.total_energy_j))
        out["energy_imbalance"].append(_imb(m.per_agent_energy_j.values()))
        out["length_imbalance"].append(_imb(m.per_agent_length_m.values()))
        out["swaps"].append(float(m.n_swaps))
        out["makespan"].append(float(m.duration_s))
        out["success"].append(1.0 if r.outcome is Outcome.MISSION_SUCCESS else 0.0)
    return out


def mean_ci(samples: list[float]) -> tuple[float, float, int]:
    """(mean, 95% CI half-width, n) over the finite samples."""
    finite = [s for s in samples if np.isfinite(s)]
    if not finite:
        return float("nan"), float("nan"), 0
    return (float(np.mean(finite)),
            ci_half_width(finite) if len(finite) >= 2 else float("inf"),
            len(finite))


def paired_contrast(a: list[float], b: list[float]) -> dict:
    """Blocked/paired contrast a - b: per-replication differences, mean + CI on
    the DIFFERENCE (not two independent means). Replications where either side
    is NaN are dropped pairwise; the count is reported."""
    if len(a) != len(b):
        raise ValueError(f"pairing broken: {len(a)} vs {len(b)} replications")
    d = [x - y for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
    dropped = len(a) - len(d)
    if not d:
        return {"mean": float("nan"), "ci": float("nan"), "n": 0,
                "dropped": dropped, "exact_zero": False}
    return {"mean": float(np.mean(d)),
            "ci": ci_half_width(d) if len(d) >= 2 else float("inf"),
            "n": len(d), "dropped": dropped,
            "exact_zero": all(x == 0.0 for x in d)}


# --------------------------------------------------------------------------- #
# the sweep                                                                   #
# --------------------------------------------------------------------------- #
def run_cell(base: Config, shapes_dir: str, shape: str, n: int, mode: str,
             n_runs: int) -> dict[str, VariantResult | Exception]:
    """One (shape, n) cell: the 4 decomposition peers on the optimized launch
    plus the naive-centroid-launch twins (tgc/classic/kmeans), ALL sharing one
    RngFactory so every stream draw is paired per replication. The naive twins
    also share ONE cfg_naive -> one pad + one deploy ring per replication, so the
    launch axis is neutral across algos. A variant that raises (e.g. infeasible
    naive pad) is recorded as its exception -- reported, not silently skipped
    (AC-1)."""
    shape_path = f"{shapes_dir}/{shape}.geojson"
    cfg = build_cell_cfg(base, shape_path, n, mode, n_runs)
    rng = RngFactory(cfg.sim.master_seed)  # ONE factory -> paired seeds
    out: dict[str, VariantResult | Exception] = {}
    for algo in ALGO_VARIANTS:
        try:
            out[algo.value] = run_variant(cfg, rng, algo.value, algo,
                                          PlannerKind.DUBINS)
        except Exception as exc:  # noqa: BLE001 -- report, never skip silently
            out[algo.value] = exc
    # NAIVE-launch twins: build the pad + cfg once (deterministic per shape), then
    # re-run each algo on it. Sharing rng and cfg_naive keeps seeds paired and the
    # pad identical across algos. Streams are pure functions of (seed, name, rep),
    # so appending these leaves the four optimizer variants byte-identical.
    try:
        site = naive_centroid_site(load_area(shape_path))
        cfg_naive = build_cell_cfg(base, shape_path, n, mode, n_runs,
                                   launch_site=site)
    except Exception as exc:  # noqa: BLE001 -- pad build failed: fail every twin
        for label in NAIVE_LAUNCH_VARIANTS:
            out[label] = exc
    else:
        for label, algo in NAIVE_LAUNCH_VARIANTS.items():
            try:
                out[label] = run_variant(cfg_naive, rng, label, algo,
                                         PlannerKind.DUBINS)
            except Exception as exc:  # noqa: BLE001
                out[label] = exc
    # AC-2: assert the pairing across every variant that ran
    counts = {lbl: v.mc.n_runs for lbl, v in out.items()
              if isinstance(v, VariantResult)}
    if len(set(counts.values())) > 1:
        raise AssertionError(f"paired-seed violation in ({shape}, n={n}): "
                             f"unequal run counts {counts}")
    return out


CONTRASTS = (
    ("tgc_basic", "classic_voronoi"),        # headline 1
    ("tgc_basic", "kmeans"),                 # headline 2
    ("weighted_voronoi", "tgc_basic"),       # scoped-null verification
    ("tgc_basic", "tgc_naive_launch"),       # launch axis: tgc (optimized - naive)
    ("classic_voronoi", "classic_naive_launch"),  # launch axis: classic
    ("kmeans", "kmeans_naive_launch"),       # launch axis: kmeans
    # KEY (Problem B): TGC vs kmeans with BOTH on the neutral naive pad -- does
    # the headline-2 verdict survive without TGC's optimizer-sited home pad?
    ("tgc_naive_launch", "kmeans_naive_launch"),
)


def _process_cell(base: Config, shapes_dir: str, shape: str, n: int, mode: str,
                  n_runs: int, clean_cfg: Config, desc_shape: dict,
                  ) -> tuple[list[dict], list[dict], list[dict], str, float]:
    """One (shape, n) cell -> (cell_rows, contrast_rows, problems, regime, secs).

    ENG-09: this is the picklable per-cell worker. It runs the identical body
    the serial loop used to run inline; the serial and parallel paths both call
    it, so per-cell output is byte-identical by construction. Returns only plain
    dict rows (never VariantResult) so the process boundary stays light. The
    regime tag and wall time are returned for the parent's progress line."""
    shape_path = f"{shapes_dir}/{shape}.geojson"
    t0 = time.perf_counter()
    tag = regime_tag(clean_cfg, shape_path, n)
    variants = run_cell(base, shapes_dir, shape, n, mode, n_runs)
    vecs = {lbl: metric_vectors(v) for lbl, v in variants.items()
            if isinstance(v, VariantResult)}
    cell_rows: list[dict] = []
    contrast_rows: list[dict] = []
    problems: list[dict] = []
    for lbl, v in variants.items():
        if isinstance(v, Exception):
            problems.append({"shape": shape, "n": n, "variant": lbl,
                             "error": f"{type(v).__name__}: {v}"})
            continue
        algo = NAIVE_LAUNCH_VARIANTS.get(lbl) or DecompositionAlgo(lbl)
        row = {"shape": shape, "n": n, "variant": lbl,
               "n_runs": v.mc.n_runs, "regime": tag["regime"],
               "pooled_ratio": round(tag["pooled_ratio"], 4),
               "max_zone_ratio": round(tag["max_zone_ratio"], 4),
               "solidity": round(desc_shape["solidity"], 4),
               "isoperimetric": round(desc_shape["isoperimetric"], 4),
               "planned_imbalance_maxmin": round(
                   planned_imbalance(clean_cfg, shape_path, n, algo), 4),
               "reference_cell": (n == REFERENCE_N)}
        for m in METRICS:
            mu, ci, k = mean_ci(vecs[lbl][m])
            row[f"{m}_mean"] = round(mu, 6)
            row[f"{m}_ci"] = round(ci, 6) if np.isfinite(ci) else ci
            row[f"{m}_n"] = k
        cell_rows.append(row)
    for a, b in CONTRASTS:
        if a not in vecs or b not in vecs:
            continue
        for m in METRICS:
            c = paired_contrast(vecs[a][m], vecs[b][m])
            contrast_rows.append({
                "shape": shape, "n": n, "contrast": f"{a} - {b}",
                "metric": m, "diff_mean": round(c["mean"], 6),
                "diff_ci": (round(c["ci"], 6) if np.isfinite(c["ci"])
                            else c["ci"]),
                "n_pairs": c["n"], "dropped_pairs": c["dropped"],
                "exact_zero": c["exact_zero"],
                "regime": tag["regime"],
                "solidity": round(desc_shape["solidity"], 4),
                "isoperimetric": round(desc_shape["isoperimetric"], 4),
                "reference_cell": (n == REFERENCE_N)})
    profiling.flush_worker()  # persist this worker's phase timers (no-op if OFF)
    return (cell_rows, contrast_rows, problems, tag["regime"],
            time.perf_counter() - t0)


def sweep(base: Config, shapes_dir: str, shapes: list[str], ns: list[int],
          mode: str, n_runs: int, ctx: RunContext, quiet: bool = False,
          jobs: int = 1) -> tuple[list[dict], list[dict], list[dict]]:
    """Runs the grid; returns (cell_rows, contrast_rows, problem_rows).

    ENG-09: with ``jobs == 1`` cells run serially in this process (the
    determinism baseline and revert path). With ``jobs > 1`` cells run in a
    ProcessPoolExecutor; results are reassembled in the ORIGINAL cell ordinal
    order (not completion order), so the concatenated rows -- and therefore the
    CSVs -- are byte-identical to the serial run regardless of which worker
    finishes first."""
    clean_cfg = build_cell_cfg(base, f"{shapes_dir}/{shapes[0]}.geojson",
                               max(ns), "clean", 1)  # tags are obstacle-free
    desc = {}
    for s in shapes:
        poly = load_area(f"{shapes_dir}/{s}.geojson")
        desc[s] = describe(s, poly, spec_effective_swath(base))

    # cells in canonical (shape, n) order -> stable ordinal for reassembly
    cells = [(shape, n) for shape in shapes for n in ns]
    total = len(cells)
    results: list[tuple[list[dict], list[dict], list[dict]] | None] = (
        [None] * total)
    t_grid = time.perf_counter()

    done = [0]

    def _record(k: int, shape: str, n: int, res) -> None:
        cr, contr, prob, regime, secs = res
        results[k] = (cr, contr, prob)
        done[0] += 1
        if not quiet:
            print(f"[{done[0]:>3d}/{total} cell {shape:>9s} n={n}] "
                  f"{regime:<15s} {secs:6.1f}s "
                  f"(elapsed {(time.perf_counter()-t_grid)/3600:.2f}h)", flush=True)

    if jobs <= 1:
        for k, (shape, n) in enumerate(cells):
            _record(k, shape, n, _process_cell(
                base, shapes_dir, shape, n, mode, n_runs, clean_cfg,
                desc[shape]))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut_to_k = {
                ex.submit(_process_cell, base, shapes_dir, shape, n, mode,
                          n_runs, clean_cfg, desc[shape]): (k, shape, n)
                for k, (shape, n) in enumerate(cells)}
            for fut in as_completed(fut_to_k):
                k, shape, n = fut_to_k[fut]
                _record(k, shape, n, fut.result())

    cell_rows: list[dict] = []
    contrast_rows: list[dict] = []
    problems: list[dict] = []
    for res in results:  # reassemble in ordinal order -> byte-identical output
        assert res is not None  # every cell must have produced a result
        cr, contr, prob = res
        cell_rows.extend(cr)
        contrast_rows.extend(contr)
        problems.extend(prob)
    if not quiet:
        print(f"grid wall time: {time.perf_counter() - t_grid:.1f}s")
    return cell_rows, contrast_rows, problems


# --------------------------------------------------------------------------- #
# analysis / read-out                                                         #
# --------------------------------------------------------------------------- #
def _corr(xs: list[float], ys: list[float]) -> float:
    finite = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(finite) < 3:
        return float("nan")
    x, y = zip(*finite)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def hypothesis_readout(contrast_rows: list[dict]) -> dict:
    """The honest H1/H2 read-out, REFRAMED for the scoped null: the 'advantage'
    is the TGC-vs-best-position-baseline gain (efficiency diff of tgc_basic
    minus the better of classic/kmeans per cell), since the original weighted
    advantage is identically zero for a homogeneous full-battery fleet. Also
    verifies that null. Numbers are reported whichever way they fall."""
    null_rows = [r for r in contrast_rows
                 if r["contrast"] == "weighted_voronoi - tgc_basic"]
    null_max_abs = max((abs(r["diff_mean"]) for r in null_rows
                        if np.isfinite(r["diff_mean"])), default=float("nan"))
    null_all_exact = bool(null_rows) and all(r["exact_zero"] for r in null_rows)

    # TGC advantage per (shape, n): min over the two headline contrasts of the
    # efficiency diff would overstate; take the diff vs the BEST baseline
    # (i.e. the smaller of the two efficiency gains -- conservative).
    adv: dict[tuple[str, int], dict] = {}
    for r in contrast_rows:
        if r["metric"] != "efficiency":
            continue
        if r["contrast"] not in ("tgc_basic - classic_voronoi",
                                 "tgc_basic - kmeans"):
            continue
        key = (r["shape"], r["n"])
        cur = adv.get(key)
        if cur is None or r["diff_mean"] < cur["diff_mean"]:
            adv[key] = r
    by_n: dict[int, list[float]] = {}
    regime_of: dict[tuple[str, int], str] = {}
    for (shape, n), r in adv.items():
        by_n.setdefault(n, []).append(r["diff_mean"])
        regime_of[(shape, n)] = r["regime"]
    h1 = {n: float(np.nanmean(v)) for n, v in sorted(by_n.items())}

    # H2: per shape, the PEAK advantage over n, correlated with the descriptors
    peak: dict[str, dict] = {}
    for (shape, n), r in adv.items():
        cur = peak.get(shape)
        if cur is None or r["diff_mean"] > cur["diff_mean"]:
            peak[shape] = r
    shapes = sorted(peak)
    peaks = [peak[s]["diff_mean"] for s in shapes]
    sol = [peak[s]["solidity"] for s in shapes]
    iso = [peak[s]["isoperimetric"] for s in shapes]
    return {"null_max_abs": null_max_abs, "null_all_exact": null_all_exact,
            "tgc_adv_vs_best_baseline_by_n": h1,
            "regime_by_cell": {f"{s}|n={n}": v
                               for (s, n), v in sorted(regime_of.items())},
            "h2_corr_solidity": _corr(sol, peaks),
            "h2_corr_isoperimetric": _corr(iso, peaks),
            "h2_peak_by_shape": {s: round(peak[s]["diff_mean"], 4)
                                 for s in shapes}}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_summary(path: Path, mode: str, n_runs: int, shapes: list[str],
                  ns: list[int], cell_rows: list[dict],
                  contrast_rows: list[dict], problems: list[dict],
                  readout: dict) -> None:
    L: list[str] = []
    L.append("# S5 shape-sweep -- summary\n")
    L.append(f"- mode: **{mode}** (lambda = 0 both modes; clean zeroes static "
             f"obstacles, shipped keeps the config density, obstacles paired "
             f"per seed)")
    L.append(f"- fixed N per cell: **{n_runs}** (paired seeds; no early stop)")
    L.append(f"- grid: {len(shapes)} shapes x n in {ns} x "
             f"{len(ALGO_VARIANTS)} optimizer peers + "
             f"{len(NAIVE_LAUNCH_VARIANTS)} naive-launch twins "
             f"({', '.join(NAIVE_LAUNCH_VARIANTS)})")
    L.append(f"- reference row: n = {REFERENCE_N}\n")
    if problems:
        L.append("## PROBLEM cells (reported, not skipped)\n")
        for p in problems:
            L.append(f"- {p['shape']} n={p['n']} {p['variant']}: {p['error']}")
        L.append("")

    L.append("## Scoped null: weighted_voronoi - tgc_basic\n")
    L.append(f"- max |diff| over every cell and metric: "
             f"**{readout['null_max_abs']:.3g}**; every paired difference "
             f"exactly zero: **{readout['null_all_exact']}**")
    L.append("- Reading: battery-weighting is INACTIVE for an identical "
             "full-battery fleet (equal fractions -> identical partition; "
             "redistribution fires only on failure and lambda = 0). Its role "
             "is the diverged-battery / redistribution regime -- outside this "
             "study's scope.\n")

    L.append("## H1 (reframed): TGC advantage vs best position baseline, by n\n")
    L.append("| n | mean efficiency advantage (tgc - best of classic/kmeans) |")
    L.append("|---|---|")
    for n, v in readout["tgc_adv_vs_best_baseline_by_n"].items():
        mark = " **<- reference**" if n == REFERENCE_N else ""
        L.append(f"| {n} | {v:+.4f}{mark} |")
    L.append("")
    L.append("## H2 (reframed): peak TGC advantage vs shape descriptors\n")
    L.append(f"- corr(peak advantage, solidity) = "
             f"**{readout['h2_corr_solidity']:+.3f}**")
    L.append(f"- corr(peak advantage, isoperimetric) = "
             f"**{readout['h2_corr_isoperimetric']:+.3f}**")
    L.append(f"- peak advantage per shape: {readout['h2_peak_by_shape']}\n")

    # launch axis: optimized (tgc_basic) - naive-centroid, read on ENERGY
    L.append("## Secondary axis: optimized launch - naive-centroid (tgc_basic)\n")
    L.append("Read on ENERGY (the optimizer's objective is energy/swap-waste "
             "prevention, not SMDP throughput). diff = optimized - naive; a "
             "NEGATIVE total_energy diff means the optimizer saves energy. The "
             "naive pad is the survey-centroid projected to the nearest legal "
             "staging point, so it minimises raw transit; the optimizer's "
             "swap-aware value is expected to surface in battery-limited cells.\n")
    L.append("| shape | n | regime | d total_energy (J) | d efficiency |")
    L.append("|---|---|---|---|---|")
    lax = {}
    for c in contrast_rows:
        if c["contrast"] != f"tgc_basic - {NAIVE_LAUNCH_LABEL}":
            continue
        lax.setdefault((c["shape"], c["n"], c["regime"]), {})[c["metric"]] = c
    for (shape, n, regime), by_m in lax.items():
        e = by_m.get("total_energy", {}).get("diff_mean", float("nan"))
        f = by_m.get("efficiency", {}).get("diff_mean", float("nan"))
        L.append(f"| {shape} | {n} | {regime} | {e:+.0f} | {f:+.3f} |")
    L.append("")

    # KEY (Problem B): TGC vs kmeans, BOTH on the neutral naive pad. If the
    # headline-2 (optimizer-launch) TGC>=kmeans verdict is really TGC's, it must
    # persist here; if it collapses to ~0, the headline gap was a launch-siting
    # confound, not a decomposition effect.
    L.append("## KEY: TGC - kmeans on NEUTRAL (naive) launch -- Problem B\n")
    L.append("Both variants share the identical naive-centroid pad and deploy "
             "ring per replication, so this contrast isolates the DECOMPOSITION "
             "axis with the launch advantage removed. Read on efficiency (the "
             "headline metric); a diff that stays >0 means the TGC verdict "
             "survives the launch-confound ablation.\n")
    L.append("| shape | n | regime | d efficiency (tgc - kmeans, neutral) |")
    L.append("|---|---|---|---|")
    keyc = {}
    for c in contrast_rows:
        if (c["contrast"] == "tgc_naive_launch - kmeans_naive_launch"
                and c["metric"] == "efficiency"):
            keyc[(c["shape"], c["n"], c["regime"])] = c["diff_mean"]
    for (shape, n, regime), diff in keyc.items():
        mark = "**" if n == REFERENCE_N else ""
        L.append(f"| {shape} | {mark}{n}{mark} | {regime} | {diff:+.4f} |")
    L.append("")

    L.append("## Per-cell efficiency (mean +/- CI) by variant\n")
    variants = sorted({r["variant"] for r in cell_rows})
    L.append("| shape | n | regime | " + " | ".join(variants) + " |")
    L.append("|---|---|---|" + "---|" * len(variants))
    seen = {}
    for r in cell_rows:
        seen.setdefault((r["shape"], r["n"], r["regime"]), {})[r["variant"]] = r
    for (shape, n, regime), by_v in seen.items():
        cells = []
        for v in variants:
            r = by_v.get(v)
            cells.append("--" if r is None
                         else f"{r['efficiency_mean']:.3f}±{r['efficiency_ci']:.3f}")
        mark = "**" if n == REFERENCE_N else ""
        L.append(f"| {shape} | {mark}{n}{mark} | {regime} | "
                 + " | ".join(cells) + " |")
    L.append("\n(Full metric set in shape_sweep.csv; paired differences with "
             "CIs in contrasts.csv.)\n")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
BUDGETS = {"quick": {"clean": 5, "shipped": 10},
           "full": {"clean": 20, "shipped": 100}}
SHIPPED_DEFAULT_NS = [2, 4]  # robustness subset: battery-limited + reference


def _auto_jobs() -> int:
    """Worker count for ``--jobs auto``: PHYSICAL cores minus one (leave a core
    for the OS / interactive work), floored at 1.

    Cross-platform (Windows / Linux / macOS / Azure) via psutil's physical-core
    count. Degrades safely -- if the physical count is unavailable (restricted
    container, exotic platform) it falls back to logical CPUs, then to 1 -- so
    the CLI never crashes on core detection. On a 1-core host this returns 1
    (serial). CPU-bound geometry work barely benefits from hyperthreads, so
    pinning to physical cores avoids oversubscription (ENG-09)."""
    n = None
    try:
        import psutil
        n = psutil.cpu_count(logical=False)  # None in some restricted sandboxes
    except Exception:  # noqa: BLE001 -- psutil missing/broken -> logical fallback
        n = None
    if not n:
        n = os.cpu_count()
    return max(1, (n or 1) - 1)


def _write_profiling(run_dir) -> None:
    """Merge every worker's phase timers (and this process's own) and write
    profiling.md + profiling.csv into the run dir. Only called when
    UAV_SWARM_PROFILE is set, so it never touches a normal run."""
    snap = profiling.collect(run_dir)
    (run_dir / "profiling.md").write_text(
        "# Phase profiling (aggregated wall time across the run)\n\n"
        + profiling.format_report(snap) + "\n", encoding="utf-8")
    with (run_dir / "profiling.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(profiling.to_csv_rows(snap))
    print(f"phase profiling -> {run_dir}/profiling.md", flush=True)


def _run_profile_subset(base: Config, shapes_dir: str, ctx: RunContext) -> None:
    """cProfile one representative variant on a small SHIPPED slice (with
    obstacles, so the geometry hot paths show) for a function-level view; writes
    profile.prof + profile.txt and runs no grid."""
    import cProfile
    import io
    import pstats

    shape, n, n_runs = SHAPE_ORDER[0], REFERENCE_N, 3
    cfg = build_cell_cfg(base, f"{shapes_dir}/{shape}.geojson", n, "shipped", n_runs)
    rng = RngFactory(cfg.sim.master_seed)
    pr = cProfile.Profile()
    pr.enable()
    run_variant(cfg, rng, "tgc_basic", DecompositionAlgo.TGC_BASIC, PlannerKind.DUBINS)
    pr.disable()
    pr.dump_stats(str(ctx.dir / "profile.prof"))
    buf = io.StringIO()
    buf.write(f"cProfile: shape={shape} n={n} variant=tgc_basic mode=shipped "
              f"n_runs={n_runs}\n\n=== by cumulative time (top 40) ===\n")
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("cumulative").print_stats(40)
    buf.write("\n=== by total/self time (top 40) ===\n")
    st.sort_stats("tottime").print_stats(40)
    (ctx.dir / "profile.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(f"cProfile -> {ctx.dir}/profile.txt (+ profile.prof)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--shapes-dir", default="data/areas/shapes")
    ap.add_argument("--mode", choices=["clean", "shipped"], default="clean")
    ap.add_argument("--budget", choices=["quick", "full"], default="quick")
    ap.add_argument("--n-runs", type=int, default=None,
                    help="override the budget's fixed N per cell")
    ap.add_argument("--shapes", default=None,
                    help="comma list (default: the canonical 9)")
    ap.add_argument("--n", default=None,
                    help="comma list of fleet sizes (default: 2..6 clean, "
                         "2,4 shipped)")
    ap.add_argument("--base", default="runs")
    ap.add_argument("--run-name", default=None,
                    help="force a fixed run-dir name (default: unique per run, "
                         "'shape_sweep_<mode>_<timestamp>_<guid>', so repeat "
                         "runs never overwrite). Pass e.g. 'shape_sweep_clean' "
                         "to pin the canonical path.")
    ap.add_argument("--jobs", default="auto",
                    help="parallel worker processes over cells (default 'auto' "
                         "= physical cores minus 1, floored at 1; pass '1' for "
                         "serial, or an explicit N). Output is byte-identical to "
                         "serial at any --jobs (ENG-09).")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile a small representative slice (1 shape x "
                         "reference n x 1 variant, shipped, few reps) to "
                         "<run>/profile.txt + profile.prof, then EXIT without "
                         "running the grid (function-level hot spots). For the "
                         "coarse per-PHASE wall-time breakdown across a real run "
                         "instead, set env var UAV_SWARM_PROFILE=1.")
    args = ap.parse_args(argv)

    base = load_config(args.config)
    shapes = (args.shapes.split(",") if args.shapes else list(SHAPE_ORDER))
    ns = ([int(x) for x in args.n.split(",")] if args.n
          else (list(range(2, 7)) if args.mode == "clean"
                else list(SHIPPED_DEFAULT_NS)))
    n_runs = args.n_runs or BUDGETS[args.budget][args.mode]
    jobs = _auto_jobs() if args.jobs == "auto" else int(args.jobs)

    ctx = RunContext(base_dir=args.base,
                     name=args.run_name or unique_run_name("shape_sweep", args.mode))
    if args.profile:
        _run_profile_subset(base, args.shapes_dir, ctx)
        ctx.finalize(summary={"experiment": "shape_sweep", "mode": "profile"})
        return 0
    if profiling.enabled():
        # workers inherit this env var (spawn re-reads it) and flush their phase
        # timers here; the parent merges them after the grid.
        os.environ[profiling._ENV_DIR] = str(ctx.dir)
    print(f"S5 shape sweep: mode={args.mode} N={n_runs} shapes={len(shapes)} "
          f"n={ns} jobs={jobs} -> {ctx.dir}", flush=True)
    cell_rows, contrast_rows, problems = sweep(
        base, args.shapes_dir, shapes, ns, args.mode, n_runs, ctx, jobs=jobs)

    readout = hypothesis_readout(contrast_rows)
    write_csv(ctx.dir / "shape_sweep.csv", cell_rows)
    write_csv(ctx.dir / "contrasts.csv", contrast_rows)
    write_summary(ctx.dir / "summary.md", args.mode, n_runs, shapes, ns,
                  cell_rows, contrast_rows, problems, readout)
    if profiling.enabled():
        _write_profiling(ctx.dir)
    ctx.finalize(summary={"experiment": "shape_sweep", "mode": args.mode,
                          "n_runs": n_runs, "shapes": shapes, "ns": ns,
                          "problems": problems, "readout": readout})
    print(f"run -> {ctx.dir}/ (shape_sweep.csv, contrasts.csv, summary.md)")
    if problems:
        print(f"PROBLEM cells: {len(problems)} (see summary.md)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
