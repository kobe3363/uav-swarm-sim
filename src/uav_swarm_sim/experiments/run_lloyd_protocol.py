"""EXP-12: paired fixed-replication protocol runner for the two Lloyd arms.

Runs a FIXED number of replications (no SMDP convergence early-stop) for
``lloyd_cvt`` vs ``lloyd_energy`` on paired, byte-identical physical inputs per
replication ``k``: one shared ``RngFactory`` and identical replication indices,
so obstacles / initial SoC / launch pose at ``k`` are the same for both arms
(launch siting runs before, and never sees, the decomposition algorithm). Every
replication is recorded -- SUCCESS / PARTIAL / FAILED / INCOMPLETE and crashes;
the EXP-11 mission contract is attached as-is (no second per-drone form). The
runner changes no trajectory, energy, RNG or outcome; it is a protocol driver.

It is opt-in (nothing imports it) and reuses the ENG-09/E2 determinism plumbing
(``_parallel.run_units``: ``--jobs 1`` is the byte-identical serial path, the
parent is the single partial-log writer, results are index-sorted so the written
files are jobs-invariant) and the crash-safe partial-log + ``--resume`` pattern
from ``run_rth_ab``. Fixed N bypasses ``monte_carlo.run`` entirely so no
convergence predicate can stop the batch, and so a per-replication error can be
trapped INSIDE the unit (``run_units`` otherwise propagates a worker exception
and aborts the whole batch).

Example:
  python -m uav_swarm_sim.experiments.run_lloyd_protocol \\
      --config config/study01_demand.yaml --n 3 5 8 --reps 30 --out runs
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy loads (transitively via the engine
# import below): stops N workers oversubscribing cores at --jobs>1 AND keeps the
# FP reduction order identical serial<->parallel so paired determinism holds.
# spawn workers re-import this module, so the pin applies in each child too.
# (Load-bearing cause of ENG-09 determinism, not an optimisation; mirrors
# run_rth_ab.)
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import argparse
import dataclasses
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..infrastructure.config import Config, load_config
from ..infrastructure.enums import DecompositionAlgo, MissionType, Outcome, PlannerKind
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..metrics.run_output import RunContext, build_mission_contract, unique_run_name
from ._parallel import add_jobs_arg, resolve_jobs, run_units
from .run_spare_sizing import _append_jsonl_line

# The two explicit Lloyd arms (never reached by any tier auto-selection; named
# here as the single source of "what the protocol compares").
ARMS: tuple[DecompositionAlgo, ...] = (
    DecompositionAlgo.LLOYD_CVT,
    DecompositionAlgo.LLOYD_ENERGY,
)

MIN_PRODUCTION_REPS = 30

PARTIAL_SCHEMA = "uav-swarm-sim/lloyd-protocol-partial/v1"
RESULTS_SCHEMA = "uav-swarm-sim/lloyd-protocol/v1"
PARTIAL_FILENAME = "results_partial.jsonl"

# Record fields that carry wall-clock timing (nondeterministic); excluded from
# the "physical projection" used for order/jobs invariance and dedup checks.
_TIMING_FIELDS = ("wall_time_s", "planning_time_s")


# --------------------------------------------------------------------------- #
# config construction + fail-fast validation                                  #
# --------------------------------------------------------------------------- #
def build_cfg(config_path: str, n: int,
              extra_overrides: dict | None = None) -> Config:
    """Load the per-n protocol config with a TRUE per-n ``config_hash``.

    ``fleet.n_drones`` is a YAML field, so its hash moves with n (unlike the
    run_rth_ab arm flags, which are not YAML and carry the base hash). The two
    serialization knobs ``experiment_mode`` and ``contract_export`` are set here
    -- they are deliberately absent from every shipped config, so they can only
    be turned on by the runner, and doing so yields a legitimate protocol hash.
    """
    overrides = {
        "fleet.n_drones": n,
        "mission.experiment_mode": True,
        "mission.contract_export": True,
    }
    if extra_overrides:
        overrides.update(extra_overrides)
    return load_config(config_path, overrides=overrides)


def validate_config(cfg: Config) -> None:
    """Fail-fast on the scenario/physics knobs the protocol needs.

    These are NOT flipped silently (that would change the physics): the config
    the user points at must already declare coverage + raster + energy_balance.
    The two serialization knobs are set by ``build_cfg`` and so are asserted, not
    required of the input."""
    if cfg.mission.type is not MissionType.COVERAGE:
        raise SystemExit("lloyd-protocol requires mission.type: coverage")
    if not cfg.coverage.raster_enabled:
        raise SystemExit("lloyd-protocol requires coverage.raster_enabled: true "
                         "(both Lloyd arms consume the EXP-02 raster)")
    if not cfg.planning.energy_balance.enabled:
        raise SystemExit("lloyd-protocol requires planning.energy_balance.enabled: "
                         "true (lloyd_energy needs it; lloyd_cvt pays the same "
                         "symmetric t=0 estimate)")
    assert cfg.mission.experiment_mode and cfg.mission.contract_export


def check_reps(reps: int, smoke: bool) -> None:
    """Production rejects fewer than 30 replications; --smoke lifts the floor and
    is stamped into every record so a non-production run can never masquerade as
    one."""
    if not smoke and reps < MIN_PRODUCTION_REPS:
        raise SystemExit(
            f"--reps {reps} < {MIN_PRODUCTION_REPS}: a production protocol run "
            f"needs at least {MIN_PRODUCTION_REPS} replications. Pass --smoke for "
            "a non-experimental short run (it is recorded in every record).")


# --------------------------------------------------------------------------- #
# input manifest (a RECORD of what the engine drew, not a second source)       #
# --------------------------------------------------------------------------- #
def _obstacles_digest(env) -> str:
    """sha256 over the sorted exterior rings of the obstacle set (coords rounded
    to 1e-6 m) -- a stable geometry fingerprint, not a full WKT dump."""
    rings = []
    for o in env.obstacles:
        coords = tuple((round(float(x), 6), round(float(y), 6))
                       for x, y in o.polygon.exterior.coords)
        rings.append(coords)
    rings.sort()
    blob = json.dumps(rings, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_manifest(eng: SimulationEngine, cfg: Config) -> dict:
    """The per-k physical inputs, read from the built engine. ``fingerprint_sha256``
    covers every physical field EXCEPT the algorithm, so the two arms of a pair
    must produce the same fingerprint at each k. No injection path: every value
    is one the engine's RNG already drew."""
    lp = eng.launch_pose
    physical = {
        "config_hash": cfg.config_hash,
        "master_seed": cfg.sim.master_seed,
        "n_drones": cfg.fleet.n_drones,
        "launch_pose_xy": [round(float(lp.x), 6), round(float(lp.y), 6)],
        "deploy_poses_xy": [[round(float(p.x), 6), round(float(p.y), 6)]
                            for p in eng.deploy_poses],
        "initial_soc_by_drone": [float(s) for s in eng.initial_soc_by_drone],
        "n_obstacles": len(eng.env.obstacles),
        "total_obstacle_area_m2": round(
            float(sum(o.polygon.area for o in eng.env.obstacles)), 3),
        "obstacles_sha256": _obstacles_digest(eng.env),
    }
    blob = json.dumps(physical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    physical["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    return physical


# --------------------------------------------------------------------------- #
# per-replication record                                                       #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class ProtocolRecord:
    algo: str                       # decomposition arm (NOT part of the fingerprint)
    n_drones: int
    replication: int
    outcome: str | None             # Outcome value, or None if the rep crashed
    coverage_frac: float | None     # plannable raster coverage [0,1]
    target_coverage_frac: float | None
    total_energy_j: float | None
    duration_s: float | None
    planning_time_s: float | None
    wall_time_s: float
    smoke: bool
    manifest: dict | None           # None if _build() raised before inputs existed
    contract: dict | None           # EXP-11 mission contract, as-is
    error: dict | None              # {type, message, traceback} on a crash


def _run_protocol_unit(cfg: Config, algo: DecompositionAlgo, n: int, k: int,
                       rng: RngFactory, smoke: bool) -> ProtocolRecord:
    """One (arm, replication) unit -> ProtocolRecord. Top-level and picklable so
    it can run in a spawn worker under --jobs.

    The one documented error boundary (§4.7): any exception from the engine is
    trapped and recorded, so a single bad replication is preserved as data rather
    than aborting the batch (``run_units`` would otherwise re-raise a worker
    exception through ``fut.result()``). A best-effort manifest is still captured
    when the engine reached the point of drawing its inputs."""
    from time import perf_counter

    eng = SimulationEngine(cfg, rng, replication=k, algo=algo,
                           planner=PlannerKind.DUBINS)
    t0 = perf_counter()
    try:
        res = eng.run()
    except Exception as exc:  # noqa: BLE001 -- the documented per-rep boundary
        wall = perf_counter() - t0
        # Inputs exist only if _build() got past launch siting; capture them if so.
        manifest = None
        if getattr(eng, "launch_pose", None) is not None:
            try:
                manifest = build_manifest(eng, cfg)
            except Exception:  # noqa: BLE001 -- manifest is diagnostic, never fatal
                manifest = None
        return ProtocolRecord(
            algo=algo.value, n_drones=n, replication=k, outcome=None,
            coverage_frac=None, target_coverage_frac=None, total_energy_j=None,
            duration_s=None, planning_time_s=None, wall_time_s=wall, smoke=smoke,
            manifest=manifest, contract=None,
            error={"type": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc()},
        )
    wall = perf_counter() - t0
    contract = build_mission_contract(
        res, capacity_j=cfg.fleet.battery_capacity_j,
        decomposer_class=type(eng.decomposer).__name__)
    return ProtocolRecord(
        algo=algo.value, n_drones=n, replication=k,
        outcome=res.outcome.value,
        coverage_frac=float(res.coverage_frac),
        target_coverage_frac=(None if res.target_coverage_frac is None
                              else float(res.target_coverage_frac)),
        total_energy_j=float(res.metrics.total_energy_j),
        duration_s=float(res.metrics.duration_s),
        planning_time_s=(None if eng.planning_time_s is None
                         else float(eng.planning_time_s)),
        wall_time_s=wall, smoke=smoke,
        manifest=build_manifest(eng, cfg), contract=contract, error=None,
    )


def run_arm(cfg: Config, algo: DecompositionAlgo, n: int, rng: RngFactory,
            replications, smoke: bool, progress=None, jobs: int = 1,
            unit_fn=_run_protocol_unit) -> list[ProtocolRecord]:
    """One arm's replications through ``run_units`` (``jobs<=1`` serial, the
    byte-identical revert path; ``jobs>1`` spawn pool). The parent stays the
    single partial-log writer via ``progress``. ``unit_fn`` is injectable so tests
    can drive the loop with a fake replication (no engine)."""
    return run_units(unit_fn, [(cfg, algo, n, k, rng, smoke) for k in replications],
                     jobs, on_result=progress)


# --------------------------------------------------------------------------- #
# record (de)serialization + physical projection                              #
# --------------------------------------------------------------------------- #
def _record_dict(rec: ProtocolRecord) -> dict:
    return dataclasses.asdict(rec)


def _record_from_dict(rec: dict) -> ProtocolRecord:
    return ProtocolRecord(
        algo=str(rec["algo"]),
        n_drones=int(rec["n_drones"]),
        replication=int(rec["replication"]),
        outcome=None if rec["outcome"] is None else str(rec["outcome"]),
        coverage_frac=None if rec["coverage_frac"] is None else float(rec["coverage_frac"]),
        target_coverage_frac=(None if rec["target_coverage_frac"] is None
                              else float(rec["target_coverage_frac"])),
        total_energy_j=None if rec["total_energy_j"] is None else float(rec["total_energy_j"]),
        duration_s=None if rec["duration_s"] is None else float(rec["duration_s"]),
        planning_time_s=(None if rec["planning_time_s"] is None
                         else float(rec["planning_time_s"])),
        wall_time_s=float(rec["wall_time_s"]),
        smoke=bool(rec["smoke"]),
        manifest=None if rec["manifest"] is None else dict(rec["manifest"]),
        contract=None if rec["contract"] is None else dict(rec["contract"]),
        error=None if rec["error"] is None else dict(rec["error"]),
    )


def projection(rec: ProtocolRecord) -> dict:
    """The physical, deterministic view of a record: everything except the
    wall-clock timing fields. Two runs that differ only in scheduling (``--jobs``,
    arm order) must produce equal projections; a genuine determinism break shows
    up here."""
    d = _record_dict(rec)
    for f in _TIMING_FIELDS:
        d.pop(f, None)
    return d


# --------------------------------------------------------------------------- #
# crash-safe partial log + resume                                             #
# --------------------------------------------------------------------------- #
def _identity(master_seed: int, reps: int, smoke: bool) -> dict:
    """Run-level identity a --resume candidate must match exactly. The per-n
    ``config_hash`` is checked separately (each cell has its own)."""
    return {"master_seed": master_seed, "reps": reps, "smoke": smoke}


def append_record(path, rec: ProtocolRecord, config_hash: str,
                  identity: dict) -> None:
    """Append one completed replication as a crash-safe (flushed + fsync'd) JSON
    line, the moment it completes."""
    _append_jsonl_line(path, {
        "schema": PARTIAL_SCHEMA,
        **identity,
        "config_hash": config_hash,
        **_record_dict(rec),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def load_records(path) -> tuple[dict, dict[tuple[int, str, int], tuple[str, ProtocolRecord]]]:
    """Parse one cell's partial log into ``(identity, {(n, algo, k): (config_hash,
    record)})``. A truncated FINAL line (crash mid-append) is skipped with a
    warning; any other malformed or foreign-schema line is refused. A repeated
    ``(n, algo, k)`` is resolved by CONTENT: an identical physical projection is
    deduplicated; a differing one is a corrupt/mixed log and is refused (never
    last-wins)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    identity: dict | None = None
    out: dict[tuple[int, str, int], tuple[str, ProtocolRecord]] = {}
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
        ident = {k: rec.get(k) for k in ("master_seed", "reps", "smoke")}
        if identity is None:
            identity = ident
        elif ident != identity:
            raise SystemExit(f"--resume: inconsistent run identity at line {i + 1} "
                             f"in {path} (mixed runs in one file?)")
        try:
            ab = _record_from_dict(rec)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"--resume: malformed record at line {i + 1} "
                             f"in {path}: {exc}") from exc
        key = (ab.n_drones, ab.algo, ab.replication)
        cfg_hash = str(rec["config_hash"])
        if key in out:
            prev_hash, prev = out[key]
            if prev_hash != cfg_hash or projection(prev) != projection(ab):
                raise SystemExit(
                    f"--resume: conflicting duplicate for {key} in {path} "
                    "(same (n, algo, replication), different content)")
            continue  # identical duplicate -> keep one
        out[key] = (cfg_hash, ab)
    if identity is None:
        raise SystemExit(f"--resume: no completed replications in {path}")
    return identity, out


def load_resume_dir(run_dir, expected: dict, per_n_hashes: dict[int, str],
                    ns: set[int]) -> dict[tuple[int, str, int], ProtocolRecord]:
    """Collect every completed ``(n, algo, k)`` from a previous run's per-cell
    partial logs. Refuses any log whose run identity does not match, or whose
    per-n ``config_hash`` disagrees with this run's. Records for an ``n`` not in
    the current ``--n`` set are dropped with a warning (they are not replayed)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"--resume: no such run directory: {run_dir}")
    partials = sorted(run_dir.glob(f"simulation-*/{PARTIAL_FILENAME}"))
    if not partials:
        raise SystemExit(f"--resume: no {PARTIAL_FILENAME} found under {run_dir}")
    done: dict[tuple[int, str, int], ProtocolRecord] = {}
    dropped: set[int] = set()
    for p in partials:
        identity, records = load_records(p)
        if identity != expected:
            diffs = "; ".join(f"{k}: partial={identity.get(k)!r} vs "
                              f"current={expected[k]!r}"
                              for k in expected if identity.get(k) != expected[k])
            raise SystemExit(f"--resume rejected ({p}): run identity mismatch "
                             f"({diffs}).")
        for key, (cfg_hash, rec) in records.items():
            n = key[0]
            if n not in ns:
                dropped.add(n)
                continue
            if per_n_hashes.get(n) != cfg_hash:
                raise SystemExit(
                    f"--resume rejected ({p}): config_hash for n={n} "
                    f"({cfg_hash}) != current ({per_n_hashes.get(n)})")
            done[key] = rec
    if dropped:
        print(f"[resume] WARNING: dropped completed replications for n={sorted(dropped)} "
              f"(not in --n); resume those from the OLD directory, not this one.",
              file=sys.stderr)
    return done


# --------------------------------------------------------------------------- #
# parity + pad dispersion + per-cell summary                                  #
# --------------------------------------------------------------------------- #
def parity_report(records: list[ProtocolRecord]) -> dict:
    """Group by ``(n, k)`` and check both arms drew identical physical inputs.
    ``ok`` = both manifests present and fingerprints equal; ``unverifiable`` =
    at least one arm's rep crashed before a manifest existed (recorded per AC-4,
    not a parity failure); ``mismatch`` = both present but fingerprints differ,
    which RAISES (a real paired-determinism break, surfaced at report time, never
    mid-batch)."""
    by_pair: dict[tuple[int, int], dict[str, ProtocolRecord]] = {}
    for r in records:
        by_pair.setdefault((r.n_drones, r.replication), {})[r.algo] = r
    statuses: dict[str, int] = {"ok": 0, "unverifiable": 0, "mismatch": 0}
    mismatches: list[dict] = []
    for (n, k), arms in sorted(by_pair.items()):
        fps = []
        have_all = len(arms) == len(ARMS)
        for a in ARMS:
            r = arms.get(a.value)
            fps.append(None if r is None or r.manifest is None
                       else r.manifest["fingerprint_sha256"])
        if not have_all or any(f is None for f in fps):
            statuses["unverifiable"] += 1
        elif len(set(fps)) == 1:
            statuses["ok"] += 1
        else:
            statuses["mismatch"] += 1
            mismatches.append({"n": n, "replication": k,
                               "fingerprints": {a.value: fp
                                                for a, fp in zip(ARMS, fps)}})
    if mismatches:
        raise SystemExit(f"manifest parity FAILED: {len(mismatches)} pair(s) drew "
                         f"different physical inputs across arms: {mismatches}")
    return {"n_pairs": len(by_pair), **statuses}


def pad_dispersion(records: list[ProtocolRecord]) -> dict:
    """How much the launch pad moved across replications (a per-n property; both
    arms share the pad at each k). Quantifies the across-k movement the pad audit
    found. ``None`` fields when no successful manifest exists. Units: metres."""
    pads: dict[int, tuple[float, float]] = {}
    for r in records:
        if r.manifest is not None and r.replication not in pads:
            pads[r.replication] = tuple(r.manifest["launch_pose_xy"])
    if not pads:
        return {"n_manifests": 0, "n_distinct_pads": None, "pad_centroid_xy": None,
                "pad_spread_m": None, "pad_std_x_m": None, "pad_std_y_m": None}
    xy = np.array(list(pads.values()), dtype=float)
    centroid = xy.mean(axis=0)
    spread = float(np.max(np.hypot(*(xy - centroid).T))) if len(xy) else 0.0
    return {
        "n_manifests": len(xy),
        "n_distinct_pads": len({tuple(p) for p in xy.tolist()}),
        "pad_centroid_xy": [round(float(centroid[0]), 6), round(float(centroid[1]), 6)],
        "pad_spread_m": round(spread, 6),
        "pad_std_x_m": round(float(xy[:, 0].std()), 6),
        "pad_std_y_m": round(float(xy[:, 1].std()), 6),
    }


def _cell_summary(records: list[ProtocolRecord], reps: int) -> dict:
    """One (n, arm) cell's factual roll-up. Outcome rows stay separate; crashes
    and missing replications are visible, never folded away."""
    done_k = {r.replication for r in records}
    missing = sorted(set(range(1, reps + 1)) - done_k)
    n_errors = sum(1 for r in records if r.error is not None)
    outcome_counts: dict[str, int] = {}
    for r in records:
        key = r.outcome if r.outcome is not None else "ERROR"
        outcome_counts[key] = outcome_counts.get(key, 0) + 1
    return {
        "n_expected": reps,
        "n_done": len(records),
        "n_missing": len(missing),
        "missing_replications": missing,
        "n_errors": n_errors,
        "outcome_counts": outcome_counts,
    }


def _results_dict(records: list[ProtocolRecord], n: int, algo: DecompositionAlgo,
                  reps: int, config_hash: str, sim_identity: dict, smoke: bool) -> dict:
    return {
        "schema": RESULTS_SCHEMA,
        "kind": "results",
        "mode": "lloyd_protocol",
        "identity": sim_identity,
        "status": "ok",
        "setup": {
            "n_drones": n,
            "arm": algo.value,
            "reps": reps,
            "smoke": smoke,
            "config_hash": config_hash,
            "fixed_replications": True,
            "init_sites": "deploy_poses",
        },
        "summary": _cell_summary(records, reps),
        "pad_dispersion": pad_dispersion(records),
        "records": [_record_dict(r) for r in records],
    }


# --------------------------------------------------------------------------- #
# entrypoint                                                                    #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EXP-12 paired fixed-replication Lloyd protocol runner.")
    ap.add_argument("--config", default="config/study01_demand.yaml")
    ap.add_argument("--n", type=int, nargs="+", default=[5],
                    help="fleet sizes to run (default 5; 3 and 8 supported)")
    ap.add_argument("--reps", type=int, default=MIN_PRODUCTION_REPS,
                    help="fixed paired replications per (n, arm) cell")
    ap.add_argument("--arms", nargs="+", default=[a.value for a in ARMS],
                    choices=[a.value for a in ARMS],
                    help="subset of Lloyd arms to run")
    ap.add_argument("--smoke", action="store_true",
                    help="non-experimental short run: lifts the 30-rep floor and "
                         "is stamped into every record")
    ap.add_argument("--out", default="runs", help="runs/ base directory")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="previous lloyd-protocol run directory to resume from")
    add_jobs_arg(ap)
    args = ap.parse_args(argv)

    ns = sorted(set(args.n))
    arms = [a for a in ARMS if a.value in set(args.arms)]
    jobs = resolve_jobs(args.jobs)
    check_reps(args.reps, args.smoke)

    # per-n configs (each a true per-n config_hash) + fail-fast validation
    cfgs = {n: build_cfg(args.config, n) for n in ns}
    for cfg in cfgs.values():
        validate_config(cfg)
    per_n_hashes = {n: cfg.config_hash for n, cfg in cfgs.items()}
    master_seed = cfgs[ns[0]].sim.master_seed
    if any(cfg.sim.master_seed != master_seed for cfg in cfgs.values()):
        raise SystemExit("all per-n configs must share one master_seed")
    identity = _identity(master_seed, args.reps, args.smoke)
    rng = RngFactory(master_seed)  # one shared factory => paired seeds

    done: dict[tuple[int, str, int], ProtocolRecord] = {}
    if args.resume:
        done = load_resume_dir(args.resume, identity, per_n_hashes, set(ns))
        print(f"[resume] {len(done)} completed (n, arm, rep) triples from "
              f"{args.resume}", file=sys.stderr)

    run = RunContext(base_dir=args.out, name=unique_run_name("lloyd_protocol"))
    all_records: list[ProtocolRecord] = []
    per_cell: dict[tuple[int, str], list[ProtocolRecord]] = {}
    for n in ns:
        cfg = cfgs[n]
        for algo in arms:
            sim = run.simulation(f"n{n}-{algo.value}")
            sim.write_plan({
                "schema": "uav-swarm-sim/plan/v1",
                "kind": "plan",
                "identity": sim.identity(config_hash=cfg.config_hash),
                "setup": {
                    "mode": "lloyd_protocol",
                    "n_drones": n,
                    "arm": algo.value,
                    "reps": args.reps,
                    "smoke": args.smoke,
                    "fixed_replications": True,
                    "master_seed": master_seed,
                    "init_sites": "deploy_poses",
                },
            })
            partial_path = sim.path(PARTIAL_FILENAME)

            resumed = [done[(n, algo.value, k)] for k in range(1, args.reps + 1)
                       if (n, algo.value, k) in done]
            for rec in resumed:
                append_record(partial_path, rec, cfg.config_hash, identity)

            def _progress(rec: ProtocolRecord, _path=partial_path,
                          _hash=cfg.config_hash) -> None:
                append_record(_path, rec, _hash, identity)
                d = "err" if rec.error is not None else rec.outcome
                print(f"  n{rec.n_drones} {rec.algo} rep {rec.replication:>4}/"
                      f"{args.reps}: {d}", file=sys.stderr)

            todo = [k for k in range(1, args.reps + 1)
                    if (n, algo.value, k) not in done]
            print(f"n={n} {algo.value}: {len(todo)} replications "
                  f"({len(resumed)} resumed)...", file=sys.stderr)
            new_recs = run_arm(cfg, algo, n, rng, todo, args.smoke,
                               progress=_progress, jobs=jobs)
            records = sorted(resumed + new_recs, key=lambda r: r.replication)
            per_cell[(n, algo.value)] = records
            all_records.extend(records)
            sim.write_results(_results_dict(records, n, algo, args.reps,
                                            cfg.config_hash, sim.identity(), args.smoke))

    parity = parity_report(all_records)  # raises on a real cross-arm mismatch
    run.finalize(summary={
        "mode": "lloyd_protocol",
        "reps": args.reps,
        "smoke": args.smoke,
        "n_values": ns,
        "arms": [a.value for a in arms],
        "manifest_parity": parity,
        "cells": {f"n{n}-{algo}": _cell_summary(recs, args.reps)
                  for (n, algo), recs in per_cell.items()},
        "pad_dispersion": {f"n{n}": pad_dispersion(per_cell[(n, arms[0].value)])
                           for n in ns},
    })

    print(f"[structured output: {run.dir}]  manifest_parity={parity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
