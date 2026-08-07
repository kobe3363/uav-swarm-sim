"""D3 -- the AREA x OBSTACLE-COUNT sweep (a thin driver composing S5 primitives).

Sweeps two axes the existing runners cannot: target survey **AREA** (the L-shape
family regenerated at each area with fixed edge proportions) and **static
obstacle COUNT** (a fixed obstacle size with a swept spatial density -- the
obstacle count is Poisson(density * area_km2), so "fixed size, varying count" is
``obstacle_size_range_m = [S, S]`` + a density grid). Every cell is
``(shape, area_km2, density_per_km2, n)``; per cell it runs the selected
decomposition variants on PAIRED SEEDS and reports the identical S5 metric set,
paired contrasts with 95% CIs, and the analytical regime tag.

Design (parameter-agnostic: every K1-K4 quantity is a CLI input, nothing baked)
------------------------------------------------------------------------------
* AREA axis ``--areas`` (K1): shapes are regenerated per area with
  ``generate_shapes.build_all`` (scale-invariant descriptors; solidity /
  isoperimetric are ratios), so a growing area holds edge proportions fixed.
* OBSTACLE-COUNT axis ``--densities`` (K4): density 0 => clean (pure
  shape+scale effect); density d > 0 => Poisson(d * area_km2) obstacles of a
  FIXED size when ``--obstacle-size-m S`` pins ``size_range = [S, S]``.
* FLEET grid ``--n`` / ``--n-range`` (K2); REPS ``--reps`` (K3), fixed and
  paired -- adaptive stopping is OFF (it would let variants converge at
  different run counts and break exact seed pairing).
* VARIANTS ``--variants`` (default: the 4 decomposition peers; the 3
  naive-launch twins are opt-in). The launch-confound (naive) axis was S5's
  secondary question; the D3 headline is shape/area/density x decomposition.

Determinism (the methodological cornerstone -- inherited, not reinvented)
------------------------------------------------------------------------
``RngFactory.stream(name, rep)`` is a pure function of
``(master_seed, name, replication)`` with NO fleet-size / area / density
dependence, so paired seeds are preserved across every tier and N is
incremental. Every variant in a cell shares ONE ``RngFactory`` -> identical
environment/failure draws per replication. Cells run serially (``--jobs 1``, the
determinism baseline) or over a spawn ``ProcessPoolExecutor`` (``--jobs > 1``);
rows are reassembled in ordinal cell order, so the CSVs are byte-identical to
serial at any ``--jobs`` (the ENG-09 / E2 / E3 seam, reused via
``experiments/_parallel``).

This driver REUSES (does not copy) the S5 machinery -- ``metric_vectors``,
``mean_ci``, ``paired_contrast``, ``METRICS``, ``regime_tag``,
``planned_imbalance``, ``naive_centroid_site``, ``write_csv`` from
``run_shape_sweep`` -- so cell/metric/contrast semantics (including the strict
``MISSION_SUCCESS`` predicate) are frozen by import, never redefined here.

Usage
-----
    python -m uav_swarm_sim.experiments.run_area_obstacle_sweep \
        [--config config/default.yaml] [--areas 1,2,4,8,16] \
        [--densities 0,8] [--obstacle-size-m S] \
        [--n 2,4,6 | --n-range 2 12 2] [--reps 20] \
        [--shapes l_shape] [--variants weighted_voronoi,tgc_basic,...] \
        [--jobs auto] [--out runs] [--run-name NAME]

Outputs (under runs/<run>/): area_obstacle_sweep.csv (per cell x variant),
contrasts.csv (paired differences with CIs), summary.md, run.json manifest, and
the regenerated shape family under shapes/.
"""
from __future__ import annotations

import os

# ENG-09 (B5): pin BLAS/OpenMP to a single thread BEFORE numpy is imported so
# (a) N workers do not oversubscribe cores with N*threads, and (b) the FP
# reduction order is identical serial<->parallel -> bitwise-identical CSVs.
# setdefault leaves an explicit user override intact; spawn workers re-import
# this module first, so the pin also takes effect in each child before numpy
# loads. (Verbatim from run_shape_sweep / run_scale_tiers -- load-bearing.)
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from ..infrastructure.config import Config, MCConfig, load_config
from ..infrastructure.enums import DecompositionAlgo, PlannerKind
from ..infrastructure.rng import RngFactory
from ..metrics.comparison import VariantResult, run_variant
from ..metrics.run_output import RunContext, unique_run_name
from ..planning.geojson_parser import load_area
from . import _parallel
from .generate_shapes import build_all, describe, shape_builders, write_shape
from .run_shape_regime_table import spec_effective_swath
from .run_shape_sweep import (
    ALGO_VARIANTS,
    CONTRASTS,
    HIGHER_IS_BETTER,
    METRICS,
    NAIVE_LAUNCH_VARIANTS,
    REFERENCE_N,
    mean_ci,
    metric_vectors,
    naive_centroid_site,
    paired_contrast,
    planned_imbalance,
    regime_tag,
    write_csv,
)

# Canonical variant order: the 4 decomposition peers (ALGO_VARIANTS) then the 3
# naive-launch twins. Iterating in this order makes the per-cell output
# deterministic regardless of the --variants CSV order.
_PEER_LABELS: tuple[str, ...] = tuple(a.value for a in ALGO_VARIANTS)
_NAIVE_LABELS: tuple[str, ...] = tuple(NAIVE_LAUNCH_VARIANTS)
ALL_VARIANT_LABELS: tuple[str, ...] = _PEER_LABELS + _NAIVE_LABELS
DEFAULT_VARIANTS: tuple[str, ...] = _PEER_LABELS  # author-confirmed default


# --------------------------------------------------------------------------- #
# per-cell config (NEW helper -- mirrors run_shape_sweep.build_cell_cfg with    #
# two added axes: an explicit obstacle DENSITY and an optional fixed SIZE)      #
# --------------------------------------------------------------------------- #
def build_area_cell_cfg(base: Config, shape_path: str, n: int, n_runs: int,
                        density: float, size_range: tuple[float, float] | None = None,
                        launch_site: tuple[float, float] | None = None) -> Config:
    """The ``(shape, area, density, n)`` cell config.

    Mirrors ``run_shape_sweep.build_cell_cfg`` field-for-field (``dataclasses.
    replace`` on env/fleet/failure/mc/telemetry; hazard forced to 0 in every
    mode; telemetry forced OFF -- a read-only per-run probe, byte-identical
    to drop; fixed-N MC via ``n_min = n_max = N`` so every variant runs the
    identical replication set) with two added axes:

    * ``density`` sets ``env.obstacle_density_per_km2`` directly (0.0 => clean,
      the analytic-tag-exact case; d > 0 => the obstacle-count axis, since the
      count is ``Poisson(d * area_km2)``);
    * ``size_range`` (when not None) pins ``env.obstacle_size_range_m`` to the
      fixed ``[S, S]`` band -- "fixed size, varying count".

    BYTE-IDENTITY (equivalence gate (i)-b): with ``density =
    base.env.obstacle_density_per_km2``, ``size_range = None`` and
    ``launch_site = None`` this returns a ``Config`` EQUAL (frozen-dataclass
    ``==``) to ``build_cell_cfg(base, shape_path, n, "shipped", n_runs)`` -- the
    two helpers construct identical objects, so the new driver perturbs nothing.
    """
    env_kw: dict = {"geojson_path": shape_path,
                    "obstacle_density_per_km2": float(density)}
    if size_range is not None:
        env_kw["obstacle_size_range_m"] = (float(size_range[0]), float(size_range[1]))
    env = dataclasses.replace(base.env, **env_kw)
    fleet = dataclasses.replace(base.fleet, n_drones=n)
    failure = dataclasses.replace(base.failure, hazard_rate_per_hour=0.0)
    mc = MCConfig(n_max=n_runs, n_min=n_runs, ci_tolerance=0.0)
    telemetry = dataclasses.replace(base.telemetry, enabled=False)
    cfg = dataclasses.replace(base, env=env, fleet=fleet, failure=failure, mc=mc,
                              telemetry=telemetry)
    if launch_site is not None:
        launch = dataclasses.replace(cfg.launch, candidate_sites=(launch_site,))
        cfg = dataclasses.replace(cfg, launch=launch)
    return cfg


# --------------------------------------------------------------------------- #
# variant dispatch                                                            #
# --------------------------------------------------------------------------- #
def _algo_of(label: str) -> DecompositionAlgo:
    """The decomposition algo behind a variant label (peer or naive twin)."""
    naive = NAIVE_LAUNCH_VARIANTS.get(label)
    return naive if naive is not None else DecompositionAlgo(label)


def run_area_cell(base: Config, shape_path: str, n: int, n_runs: int,
                  density: float, size_range: tuple[float, float] | None,
                  variant_labels: list[str],
                  ) -> dict[str, VariantResult | Exception]:
    """One cell: the selected variants on ONE shared ``RngFactory`` (paired
    seeds). Peers run on the shipped optimized launch; naive twins share ONE
    ``cfg_naive`` (one pad + one deploy ring per replication -> launch neutral
    across algos). A variant that raises is recorded as its exception --
    reported, never skipped silently (mirrors run_shape_sweep.run_cell).

    Because ``stream(name, rep)`` is a pure function of ``(seed, name, rep)``,
    the variant RUN ORDER is irrelevant to the draws, so selecting a subset of
    variants leaves each selected variant byte-identical to the full-set run."""
    cfg = build_area_cell_cfg(base, shape_path, n, n_runs, density, size_range)
    rng = RngFactory(cfg.sim.master_seed)  # ONE factory -> paired seeds
    out: dict[str, VariantResult | Exception] = {}

    peers = [lbl for lbl in _PEER_LABELS if lbl in variant_labels]
    naives = [lbl for lbl in _NAIVE_LABELS if lbl in variant_labels]

    for lbl in peers:
        try:
            out[lbl] = run_variant(cfg, rng, lbl, DecompositionAlgo(lbl),
                                   PlannerKind.DUBINS)
        except Exception as exc:  # noqa: BLE001 -- report, never skip silently
            out[lbl] = exc

    if naives:
        try:
            site = naive_centroid_site(load_area(shape_path))
            cfg_naive = build_area_cell_cfg(base, shape_path, n, n_runs, density,
                                            size_range, launch_site=site)
        except Exception as exc:  # noqa: BLE001 -- pad build failed: fail every twin
            for lbl in naives:
                out[lbl] = exc
        else:
            for lbl in naives:
                try:
                    out[lbl] = run_variant(cfg_naive, rng, lbl,
                                           NAIVE_LAUNCH_VARIANTS[lbl],
                                           PlannerKind.DUBINS)
                except Exception as exc:  # noqa: BLE001
                    out[lbl] = exc

    # paired-seed assertion across every variant that ran
    counts = {lbl: v.mc.n_runs for lbl, v in out.items()
              if isinstance(v, VariantResult)}
    if len(set(counts.values())) > 1:
        raise AssertionError(
            f"paired-seed violation at {shape_path} n={n} density={density}: "
            f"unequal run counts {counts}")
    return out


# --------------------------------------------------------------------------- #
# per-cell worker (picklable; serial and spawn paths both call it)             #
# --------------------------------------------------------------------------- #
def _process_area_cell(k: int, base: Config, shape: str, shape_path: str,
                       area_km2: float, density: float,
                       size_range: tuple[float, float] | None, n: int,
                       n_runs: int, variant_labels: list[str], desc_shape: dict,
                       ) -> dict:
    """One ``(shape, area, density, n)`` cell -> a record carrying its ordinal
    ``k`` (for stable reassembly), the cell rows, the contrast rows, any
    problems, and the regime/wall-time for the progress line. Returns only plain
    dict rows (never a ``VariantResult``) so the process boundary stays light."""
    t0 = time.perf_counter()
    # Regime tag: analytical, obstacle-free planning layer at THIS area. The tag
    # is a pure function of the shape geometry + config scalars + n, so a clean
    # (density-0) carrier cfg is built here and regime_tag reads the area from
    # shape_path (mirrors run_shape_sweep's clean_cfg usage).
    clean_cfg = build_area_cell_cfg(base, shape_path, n, 1, density=0.0)
    tag = regime_tag(clean_cfg, shape_path, n)

    variants = run_area_cell(base, shape_path, n, n_runs, density, size_range,
                             variant_labels)
    vecs = {lbl: metric_vectors(v) for lbl, v in variants.items()
            if isinstance(v, VariantResult)}

    cell_rows: list[dict] = []
    contrast_rows: list[dict] = []
    problems: list[dict] = []
    for lbl, v in variants.items():
        if isinstance(v, Exception):
            problems.append({"shape": shape, "area_km2": area_km2,
                             "density_per_km2": density, "n": n, "variant": lbl,
                             "error": f"{type(v).__name__}: {v}"})
            continue
        row = {"shape": shape, "area_km2": area_km2,
               "density_per_km2": density, "n": n, "variant": lbl,
               "n_runs": v.mc.n_runs, "regime": tag["regime"],
               "pooled_ratio": round(tag["pooled_ratio"], 4),
               "max_zone_ratio": round(tag["max_zone_ratio"], 4),
               "solidity": round(desc_shape["solidity"], 4),
               "isoperimetric": round(desc_shape["isoperimetric"], 4),
               "planned_imbalance_maxmin": round(
                   planned_imbalance(clean_cfg, shape_path, n, _algo_of(lbl)), 4),
               "reference_cell": (n == REFERENCE_N)}
        for m in METRICS:
            mu, ci, kk = mean_ci(vecs[lbl][m])
            row[f"{m}_mean"] = round(mu, 6)
            row[f"{m}_ci"] = round(ci, 6) if np.isfinite(ci) else ci
            row[f"{m}_n"] = kk
        cell_rows.append(row)

    for a, b in CONTRASTS:
        if a not in vecs or b not in vecs:
            continue
        for m in METRICS:
            c = paired_contrast(vecs[a][m], vecs[b][m])
            contrast_rows.append({
                "shape": shape, "area_km2": area_km2,
                "density_per_km2": density, "n": n,
                "contrast": f"{a} - {b}", "metric": m,
                "diff_mean": round(c["mean"], 6),
                "diff_ci": (round(c["ci"], 6) if np.isfinite(c["ci"])
                            else c["ci"]),
                "n_pairs": c["n"], "dropped_pairs": c["dropped"],
                "exact_zero": c["exact_zero"], "regime": tag["regime"],
                "solidity": round(desc_shape["solidity"], 4),
                "isoperimetric": round(desc_shape["isoperimetric"], 4),
                "reference_cell": (n == REFERENCE_N)})

    return {"k": k, "cell_rows": cell_rows, "contrast_rows": contrast_rows,
            "problems": problems, "regime": tag["regime"], "shape": shape,
            "area_km2": area_km2, "density": density, "n": n,
            "secs": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# shape regeneration + the sweep                                              #
# --------------------------------------------------------------------------- #
def regenerate_shapes(base: Config, shapes_root: Path, areas_km2: list[float],
                      shapes: list[str], disk_sides: int) -> dict[tuple[str, float], str]:
    """Regenerate the requested shapes at each area into
    ``shapes_root/a<area>/<shape>.geojson`` and return a
    ``{(shape, area_km2): path}`` map. Done ONCE in the parent (deterministic,
    no per-worker race). ``build_all`` asserts each shape hits its target area
    within 1e-6 relative error; ``describe``/``build_all`` use the drone-spec
    effective swath (matches run_shape_sweep's descriptor computation)."""
    eff_swath = spec_effective_swath(base)
    want = set(shapes)
    paths: dict[tuple[str, float], str] = {}
    for area_km2 in areas_km2:
        target_m2 = area_km2 * 1e6
        area_dir = shapes_root / f"a{_area_slug(area_km2)}"
        for name, poly, desc in build_all(target_m2, eff_swath, disk_sides):
            if name not in want:
                continue
            paths[(name, area_km2)] = str(write_shape(area_dir, name, poly, desc))
    return paths


def _area_slug(area_km2: float) -> str:
    """Filesystem-safe area tag: '1', '2p5', '16' (no dots in dir names)."""
    s = ("%g" % area_km2).replace(".", "p")
    return s


def sweep(base: Config, shape_paths: dict[tuple[str, float], str],
          shapes: list[str], areas_km2: list[float], densities: list[float],
          size_range: tuple[float, float] | None, ns: list[int], n_runs: int,
          variant_labels: list[str], descs: dict[str, dict], jobs: int = 1,
          quiet: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Run the grid; return ``(cell_rows, contrast_rows, problem_rows)``.

    Cells are enumerated in canonical ``(shape, area, density, n)`` order and
    given a stable ordinal ``k``. ``_parallel.run_units`` runs them serially
    (``jobs <= 1``, the byte-identical revert path) or over a spawn pool
    (``jobs > 1``); records arrive in completion order, so they are sorted by
    ``k`` before concatenation -> the CSVs are byte-identical at any ``jobs``."""
    cells = [(shape, area_km2, density, n)
             for shape in shapes
             for area_km2 in areas_km2
             for density in densities
             for n in ns]
    total = len(cells)
    unit_args = [
        (k, base, shape, shape_paths[(shape, area_km2)], area_km2, density,
         size_range, n, n_runs, variant_labels, descs[shape])
        for k, (shape, area_km2, density, n) in enumerate(cells)]

    t_grid = time.perf_counter()
    done = [0]

    def _progress(rec: dict) -> None:
        done[0] += 1
        if not quiet:
            print(f"[{done[0]:>3d}/{total} {rec['shape']:>9s} "
                  f"A={rec['area_km2']:g} d={rec['density']:g} n={rec['n']}] "
                  f"{rec['regime']:<15s} {rec['secs']:6.1f}s "
                  f"(elapsed {(time.perf_counter()-t_grid)/3600:.2f}h)", flush=True)

    records = _parallel.run_units(_process_area_cell, unit_args, jobs,
                                  on_result=_progress)
    records.sort(key=lambda r: r["k"])  # stable ordinal -> byte-identical output

    cell_rows: list[dict] = []
    contrast_rows: list[dict] = []
    problems: list[dict] = []
    for rec in records:
        cell_rows.extend(rec["cell_rows"])
        contrast_rows.extend(rec["contrast_rows"])
        problems.extend(rec["problems"])
    if not quiet:
        print(f"grid wall time: {time.perf_counter() - t_grid:.1f}s")
    return cell_rows, contrast_rows, problems


# --------------------------------------------------------------------------- #
# summary (descriptive -- no invented hypothesis read-out; the CSVs carry the  #
# full data for the D3 analysis)                                              #
# --------------------------------------------------------------------------- #
def write_summary(path: Path, shapes: list[str], areas_km2: list[float],
                  densities: list[float], ns: list[int], n_runs: int,
                  size_range: tuple[float, float] | None,
                  variant_labels: list[str], cell_rows: list[dict],
                  contrast_rows: list[dict], problems: list[dict]) -> None:
    L: list[str] = []
    L.append("# Area x obstacle-count sweep -- summary\n")
    L.append(f"- shapes: {shapes}")
    L.append(f"- areas (km^2): {areas_km2}")
    L.append(f"- densities (per km^2; 0 = clean): {densities}")
    L.append(f"- obstacle size: "
             f"{'fixed [%g, %g] m' % size_range if size_range else 'config default range'}")
    L.append(f"- fleet sizes n: {ns}")
    L.append(f"- fixed N per cell: **{n_runs}** (paired seeds; no early stop)")
    L.append(f"- variants: {variant_labels}")
    L.append(f"- cells: {len(shapes)*len(areas_km2)*len(densities)*len(ns)} "
             f"(shape x area x density x n)\n")
    if problems:
        L.append("## PROBLEM cells (reported, not skipped)\n")
        for p in problems:
            L.append(f"- {p['shape']} A={p['area_km2']:g} d={p['density_per_km2']:g} "
                     f"n={p['n']} {p['variant']}: {p['error']}")
        L.append("")

    # scoped null: weighted_voronoi - tgc_basic (only if both ran)
    null_rows = [r for r in contrast_rows
                 if r["contrast"] == "weighted_voronoi - tgc_basic"]
    if null_rows:
        null_max_abs = max((abs(r["diff_mean"]) for r in null_rows
                            if np.isfinite(r["diff_mean"])), default=float("nan"))
        null_all_exact = all(r["exact_zero"] for r in null_rows)
        L.append("## Scoped null: weighted_voronoi - tgc_basic\n")
        L.append(f"- max |diff| over every cell and metric: **{null_max_abs:.3g}**; "
                 f"every paired difference exactly zero: **{null_all_exact}**")
        L.append("- Reading: battery-weighting is INACTIVE for an identical "
                 "full-battery fleet at every area/density (equal fractions -> "
                 "identical partition; redistribution fires only on failure, "
                 "lambda = 0).\n")

    # per-cell efficiency table (mean +/- CI) by variant
    L.append("## Per-cell efficiency (mean +/- CI) by variant\n")
    variants = sorted({r["variant"] for r in cell_rows})
    L.append("| shape | area | density | n | regime | " + " | ".join(variants) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(variants))
    seen: dict[tuple, dict] = {}
    order: list[tuple] = []
    for r in cell_rows:
        key = (r["shape"], r["area_km2"], r["density_per_km2"], r["n"], r["regime"])
        if key not in seen:
            seen[key] = {}
            order.append(key)
        seen[key][r["variant"]] = r
    for key in order:
        shape, area_km2, density, n, regime = key
        by_v = seen[key]
        cells = []
        for v in variants:
            r = by_v.get(v)
            cells.append("--" if r is None
                         else f"{r['efficiency_mean']:.3f}±{r['efficiency_ci']:.3f}")
        L.append(f"| {shape} | {area_km2:g} | {density:g} | {n} | {regime} | "
                 + " | ".join(cells) + " |")
    L.append("\n(Full metric set in area_obstacle_sweep.csv; paired differences "
             "with CIs in contrasts.csv.)\n")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _n_grid(args) -> list[int]:
    """Fleet grid from --n (comma list) or --n-range START STOP STEP (inclusive
    of STOP). --n-range takes precedence when both are given."""
    if args.n_range is not None:
        start, stop, step = args.n_range
        if step <= 0:
            raise SystemExit("--n-range STEP must be positive")
        return list(range(start, stop + 1, step))
    return [int(x) for x in args.n.split(",") if x.strip() != ""]


def _resolve_variants(spec: str) -> list[str]:
    labels = [x.strip() for x in spec.split(",") if x.strip() != ""]
    unknown = [x for x in labels if x not in ALL_VARIANT_LABELS]
    if unknown:
        raise SystemExit(
            f"unknown --variants {unknown}; choose from {list(ALL_VARIANT_LABELS)}")
    # canonical order, de-duplicated
    return [lbl for lbl in ALL_VARIANT_LABELS if lbl in labels]


def _resolve_shapes(spec: str) -> list[str]:
    known = set(shape_builders(128))  # keys are disk-sides independent
    labels = [x.strip() for x in spec.split(",") if x.strip() != ""]
    unknown = [x for x in labels if x not in known]
    if unknown:
        raise SystemExit(f"unknown --shapes {unknown}; choose from {sorted(known)}")
    return labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/djimatrice4e.yaml")
    ap.add_argument("--shapes-dir", default="data/areas/shapes",
                    help="(unused for generation -- shapes are regenerated per "
                         "area into the run dir; kept for parity/tools)")
    ap.add_argument("--areas", default="1,2,4,8,16",
                    help="comma list of target areas in km^2 (K1)")
    ap.add_argument("--densities", default="0,8",
                    help="comma list of obstacle densities per km^2; 0 = clean (K4)")
    ap.add_argument("--obstacle-size-m", type=float, default=None,
                    help="fixed obstacle size S -> size_range=[S,S] (fixed size, "
                         "varying count). Default: the config's size range.")
    ap.add_argument("--n", default="2,4,6",
                    help="comma list of fleet sizes (K2)")
    ap.add_argument("--n-range", type=int, nargs=3, metavar=("START", "STOP", "STEP"),
                    default=None, help="fleet grid (inclusive of STOP); overrides --n")
    ap.add_argument("--reps", type=int, default=20,
                    help="fixed N per cell, paired (K3; adaptive stopping OFF)")
    ap.add_argument("--shapes", default="l_shape",
                    help="comma list of shapes (default l_shape; a LIST so the "
                         "9-shape family stays possible)")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help="comma list of variant labels (default: the 4 "
                         f"decomposition peers). Choices: {list(ALL_VARIANT_LABELS)}")
    ap.add_argument("--disk-sides", type=int, default=128,
                    help="disk polygon sides (matches generate_shapes default)")
    _parallel.add_jobs_arg(ap)
    ap.add_argument("--out", default="runs", help="BASE output dir")
    ap.add_argument("--run-name", default=None,
                    help="force a fixed run-dir name (default: unique per run)")
    args = ap.parse_args(argv)

    base = load_config(args.config)
    areas_km2 = _floats(args.areas)
    densities = _floats(args.densities)
    ns = _n_grid(args)
    shapes = _resolve_shapes(args.shapes)
    variant_labels = _resolve_variants(args.variants)
    size_range = ((args.obstacle_size_m, args.obstacle_size_m)
                  if args.obstacle_size_m is not None else None)
    jobs = _parallel.resolve_jobs(args.jobs)

    ctx = RunContext(base_dir=args.out,
                     name=args.run_name or unique_run_name("area_obstacle_sweep"))
    shape_paths = regenerate_shapes(base, ctx.dir / "shapes", areas_km2, shapes,
                                    args.disk_sides)
    eff_swath = spec_effective_swath(base)
    descs = {s: describe(s, load_area(shape_paths[(s, areas_km2[0])]), eff_swath)
             for s in shapes}

    print(f"area x obstacle sweep: shapes={shapes} areas={areas_km2} "
          f"densities={densities} n={ns} reps={args.reps} "
          f"variants={variant_labels} jobs={jobs} -> {ctx.dir}", flush=True)

    cell_rows, contrast_rows, problems = sweep(
        base, shape_paths, shapes, areas_km2, densities, size_range, ns,
        args.reps, variant_labels, descs, jobs=jobs)

    write_csv(ctx.dir / "area_obstacle_sweep.csv", cell_rows)
    write_csv(ctx.dir / "contrasts.csv", contrast_rows)
    write_summary(ctx.dir / "summary.md", shapes, areas_km2, densities, ns,
                  args.reps, size_range, variant_labels, cell_rows,
                  contrast_rows, problems)
    ctx.finalize(summary={"experiment": "area_obstacle_sweep", "shapes": shapes,
                          "areas_km2": areas_km2, "densities": densities,
                          "ns": ns, "reps": args.reps,
                          "obstacle_size_m": args.obstacle_size_m,
                          "variants": variant_labels, "jobs": jobs,
                          "problems": problems})
    print(f"run -> {ctx.dir}/ (area_obstacle_sweep.csv, contrasts.csv, summary.md)")
    if problems:
        print(f"PROBLEM cells: {len(problems)} (see summary.md)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
