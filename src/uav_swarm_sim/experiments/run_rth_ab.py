"""Stage 5 A/B: four RTH arms on paired seeds (task C1 -- the thesis result).

The thesis claim under test: a dynamic, distance-aware energy-map RTH
outperforms a static battery-fraction threshold. Four arms isolate each factor,
each adding exactly one thing to the one before, on IDENTICAL paired seeds:

  arm 1  static40      energy_map absent -- the literature baseline: the static
                       CRITICAL guard returns the drone the moment its battery
                       fraction drops below ``nominal`` = 0.40 (the guard tests
                       the CRITICAL *zone* [critical, nominal), so it fires at
                       the 0.40 boundary -- "static-40%", never "static-20%").
  arm 2  route-only    ``enabled + route``: the map ROUTES the S3 return and the
                       post-swap resume around obstacles, but the return
                       *decision* stays the static path (isolation arm).
  arm 3  decide-route  ``enabled + decide + route``: the map decides AND routes,
                       but ``zone_demotion`` is off, so the static 0.40 net
                       still pre-empts the map on every normal return.
  arm 4  full-map      ``+ zone_demotion``: the static CRITICAL branch is
                       removed; the map alone governs the normal return, with
                       the TERMINAL failsafe floor at 0.10 (B2 sweep choice),
                       applied in-config -- default.yaml is never edited.

The HEADLINE observable is the transition-reason attribution
(``Sojourn.reason_out``): ``rth_energy`` vs ``critical_battery`` vs
``terminal_battery`` counts per arm. Arm 1: ``critical_battery`` dominates,
``rth_energy`` ~0. Arm 4: ``rth_energy`` governs. That inversion IS the result.
Reason-string exclusivity is grep-verified: the three strings are produced ONLY
by the ``_coverage_guards`` transitions into S3_RTH (execution/state_machine.py)
and reach ``Sojourn.reason_out`` through the single ``recorder.close`` call in
``Agent.step`` -- no other ``reason_out`` value collides.

Contrast decomposition (read-out frame, pinned here so it cannot misfire):
arm 3's reason attribution looks like arm 1's BY DESIGN (the static net
pre-empts a map that is computing underneath -- map hits without map-attributed
returns). Therefore: **arm3 - arm1 isolates ROUTING** (success/coverage/
INCOMPLETE shift with no attribution change) and **arm4 - arm3 isolates
decide + demotion** (the attribution inversion + sortie depth). Reading
"arm 3 ~= arm 1 by reason" as "the map does nothing" is the misread this frame
exists to prevent.

Experiment constants, pinned IDENTICALLY across all four arms (the arms differ
ONLY in ``energy_map.enabled/decide/route/zone_demotion`` + the arm-4 floor):
  * ``fleet.total_reserve_batteries = None`` -- unbounded pool, demand-mode
    semantics: ``D_k = n_swaps`` on MISSION_SUCCESS else infinity, exactly the
    ``run_spare_sizing`` demand equivalence success(k, B) <=> D_k <= B.
  * ``safety.stall_skip = true`` -- stall handling is ORTHOGONAL to the
    energy-map axis, so it must be a constant, not an arm difference: residual
    obstacle-boxing ends in the same MISSION_PARTIAL mechanism in every arm and
    the arm-to-arm difference stays pure RTH. LIMITATION (goes in plan.json):
    arm 1 is therefore "static-40% net + EM-01 skip-on-stall", not a pristine
    literature baseline; this touches only the secondary success/demand
    columns, never the reason attribution.

Success predicate: strict ``outcome is MISSION_SUCCESS`` (byte-identical to the
demand mode's). PARTIAL and INCOMPLETE both count as infinite demand in the
CDF but remain THREE separate rows in the outcome table -- the
INCOMPLETE -> PARTIAL shift between arms is itself an expected thesis effect.

Pairing: one shared ``RngFactory(cfg.sim.master_seed)`` and identical
replication indices 1..reps across arms. ``RngFactory.stream(name, k)`` is a
pure function of (master_seed, name, k), so environment/failure draws at
replication k are byte-identical across arms; the energy map itself is
deterministic (no RNG draw). Arms run sequentially; stream purity means order
cannot shift any draw.

Serial by design -- no ``--jobs`` (parallelism is the separate E2 follow-up,
which requires its own bitwise serial==parallel verification). Each completed
replication is appended to the arm's ``results_partial.jsonl`` immediately
(fsync'd), and ``--resume <previous run dir>`` skips finished (arm, rep) pairs
after an exact identity check -- crash safety for the multi-day Azure run.

Example (the author's Azure run):
  python -m uav_swarm_sim.experiments.run_rth_ab \\
      --config config/study01_demand.yaml --reps 100 --out runs
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..infrastructure.config import Config, EnergyMapConfig, load_config
from ..infrastructure.enums import Outcome, PlannerKind
from ..infrastructure.rng import RngFactory
from ..infrastructure.simulation_engine import SimulationEngine
from ..metrics.convergence import wilson_ci
from ..metrics.run_output import RunContext, unique_run_name
from .run_spare_sizing import _append_jsonl_line, _per_drone_swaps, _with_reserve

# --------------------------------------------------------------------------- #
# the four arms                                                                #
# --------------------------------------------------------------------------- #
ARM_SLUGS = {1: "static40", 2: "route-only", 3: "decide-route", 4: "full-map"}
ARM4_TERMINAL_FLOOR = 0.10   # B2 sweep choice; in-config only, never default.yaml

# The three (and only three) reason_out values a coverage-guard return can
# carry -- produced exclusively by the Transition(..., S3_RTH, ...) triple in
# execution/state_machine.py (grep-verified; pinned by the smoke test).
RTH_REASONS = ("rth_energy", "critical_battery", "terminal_battery")

PARTIAL_SCHEMA = "uav-swarm-sim/rth-ab-partial/v1"
RESULTS_SCHEMA = "uav-swarm-sim/rth-ab/v1"
PARTIAL_FILENAME = "results_partial.jsonl"


def experiment_constants(cfg: Config) -> Config:
    """The knobs pinned identically across ALL arms: unbounded pool (demand
    semantics) + stall_skip (orthogonal boxing accounting, held constant so the
    arm contrast stays pure RTH). Everything else is left exactly as loaded."""
    if not cfg.safety.stall_detector:
        raise SystemExit("rth-ab requires safety.stall_detector: true in the "
                         "base config (stall_skip depends on it)")
    cfg = _with_reserve(cfg, None)
    safety = dataclasses.replace(cfg.safety, stall_skip=True)
    return dataclasses.replace(cfg, safety=safety)


def arm_config(cfg: Config, arm: int) -> Config:
    """One arm's Config from the constants-pinned base. ``dataclasses.replace``
    only -- the base object is frozen and never mutated. NOTE: ``config_hash``
    is a plain field computed from the raw YAML at load time, so every arm
    carries the BASE hash; the arm identity therefore records the explicit flag
    dict (``arm_flags``), never the hash alone."""
    if arm not in ARM_SLUGS:
        raise ValueError(f"unknown arm {arm!r} (valid: 1..4)")
    if arm == 1:
        return cfg
    em = EnergyMapConfig(
        enabled=True,
        decide=(arm >= 3),
        route=True,
        zone_demotion=(arm == 4),
    )
    # the loader's dependency rules, re-asserted for replace-built configs
    assert not em.decide or em.enabled
    assert not em.zone_demotion or em.decide
    cfg = dataclasses.replace(cfg, rth=dataclasses.replace(cfg.rth, energy_map=em))
    if arm == 4:
        zones = dataclasses.replace(cfg.battery_zones, critical=ARM4_TERMINAL_FLOOR)
        cfg = dataclasses.replace(cfg, battery_zones=zones)
    return cfg


def arm_flags(cfg: Config, arm: int) -> dict:
    """The explicit arm-defining flag dict recorded in every identity/plan
    block (the honest substitute for the stale per-arm config_hash)."""
    em = cfg.rth.energy_map
    return {
        "arm": arm,
        "arm_slug": ARM_SLUGS[arm],
        "energy_map_enabled": em.enabled,
        "energy_map_decide": em.decide,
        "energy_map_route": em.route,
        "energy_map_zone_demotion": em.zone_demotion,
        "terminal_floor": cfg.battery_zones.critical,
    }


# --------------------------------------------------------------------------- #
# per-replication record                                                       #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class AbRecord:
    arm: int
    replication: int
    outcome: str                       # all four Outcome values, PARTIAL distinct
    demand: int | None                 # n_swaps on strict SUCCESS, else None (= inf)
    per_drone_swaps: dict[int, int]
    reasons: dict[str, int]            # the three RTH_REASONS counts
    return_depths: list[float]         # battery frac at each return decision
    total_energy_j: float
    duration_s: float
    coverage_frac: float
    n_skipped_legs: int
    stalled_agents: tuple[int, ...]
    n_map_hits: int
    n_map_fallbacks: int
    n_route_fallbacks: int


def _reason_counts(history) -> dict[str, int]:
    counts = {r: 0 for r in RTH_REASONS}
    for s in history.sojourns():
        if s.reason_out in counts:
            counts[s.reason_out] += 1
    return counts


def _return_depths(history) -> list[float]:
    """Battery fraction at each energy-return decision: the last battery sample
    at or before the closing time of every sojourn that exited on one of the
    three RTH reasons. The trace is appended in time order per agent, so a
    linear scan up to t_out reads the decision-time value."""
    depths: list[float] = []
    for s in history.sojourns():
        if s.reason_out not in RTH_REASONS:
            continue
        frac = None
        for t, f in history.battery_trace(s.agent_id):
            if t <= s.t_out:
                frac = f
            else:
                break
        if frac is not None:
            depths.append(round(float(frac), 6))
    return depths


def run_arm(cfg_arm: Config, arm: int, reps: int, rng: RngFactory,
            progress=None, replications=None) -> list[AbRecord]:
    """One arm's batch: replications 1..reps (or the given subset, for resume)
    on the SHARED factory. Engine construction mirrors run_demand's
    (algo=None -> tier default, planner=DUBINS) so arm 1 is byte-identical to
    the existing demand path on the same config -- the gate test pins this."""
    todo = list(replications) if replications is not None else list(range(1, reps + 1))
    records: list[AbRecord] = []
    for k in todo:
        eng = SimulationEngine(cfg_arm, rng, replication=k, algo=None,
                               planner=PlannerKind.DUBINS)
        res = eng.run()
        success = res.outcome is Outcome.MISSION_SUCCESS
        rec = AbRecord(
            arm=arm,
            replication=k,
            outcome=res.outcome.value,
            demand=int(res.metrics.n_swaps) if success else None,
            per_drone_swaps=_per_drone_swaps(res.history),
            reasons=_reason_counts(res.history),
            return_depths=_return_depths(res.history),
            total_energy_j=float(res.metrics.total_energy_j),
            duration_s=float(res.metrics.duration_s),
            coverage_frac=float(res.coverage_frac),
            n_skipped_legs=len(res.skipped_legs),
            stalled_agents=tuple(res.stalled_agents),
            n_map_hits=int(eng.rth.n_map_hits),
            n_map_fallbacks=int(eng.rth.n_map_fallbacks),
            n_route_fallbacks=int(eng.rth.n_route_fallbacks),
        )
        records.append(rec)
        if progress is not None:
            progress(rec)
    return records


# --------------------------------------------------------------------------- #
# crash-safe partial log + resume (the demand-mode pattern, arm-aware)         #
# --------------------------------------------------------------------------- #
def _identity(cfg: Config, reps: int) -> dict:
    """Run-identity fields a --resume candidate must match EXACTLY. The
    config_hash is the BASE config's (see arm_config); the arm flags are part
    of each record, and the per-arm check happens on (arm, replication)."""
    return {
        "master_seed": cfg.sim.master_seed,
        "config_hash": cfg.config_hash,
        "reps": reps,
    }


def _record_dict(rec: AbRecord) -> dict:
    return {
        "arm": rec.arm,
        "arm_slug": ARM_SLUGS[rec.arm],
        "replication": rec.replication,
        "outcome": rec.outcome,
        "demand": rec.demand,
        "per_drone_swaps": {str(k): v for k, v in sorted(rec.per_drone_swaps.items())},
        "reasons": dict(rec.reasons),
        "return_depths": list(rec.return_depths),
        "total_energy_j": rec.total_energy_j,
        "duration_s": rec.duration_s,
        "coverage_frac": rec.coverage_frac,
        "n_skipped_legs": rec.n_skipped_legs,
        "stalled_agents": list(rec.stalled_agents),
        "n_map_hits": rec.n_map_hits,
        "n_map_fallbacks": rec.n_map_fallbacks,
        "n_route_fallbacks": rec.n_route_fallbacks,
    }


def _record_from_dict(rec: dict) -> AbRecord:
    return AbRecord(
        arm=int(rec["arm"]),
        replication=int(rec["replication"]),
        outcome=str(rec["outcome"]),
        demand=None if rec["demand"] is None else int(rec["demand"]),
        per_drone_swaps={int(a): int(v) for a, v
                         in dict(rec["per_drone_swaps"]).items()},
        reasons={str(k): int(v) for k, v in dict(rec["reasons"]).items()},
        return_depths=[float(d) for d in rec["return_depths"]],
        total_energy_j=float(rec["total_energy_j"]),
        duration_s=float(rec["duration_s"]),
        coverage_frac=float(rec["coverage_frac"]),
        n_skipped_legs=int(rec["n_skipped_legs"]),
        stalled_agents=tuple(int(a) for a in rec["stalled_agents"]),
        n_map_hits=int(rec["n_map_hits"]),
        n_map_fallbacks=int(rec["n_map_fallbacks"]),
        n_route_fallbacks=int(rec["n_route_fallbacks"]),
    )


def append_ab_record(path, rec: AbRecord, identity: dict) -> None:
    """Append one completed replication as a crash-safe (flushed + fsync'd)
    JSON line -- the moment it completes, so a crash loses at most the
    replication in flight."""
    _append_jsonl_line(path, {
        "schema": PARTIAL_SCHEMA,
        **identity,
        **_record_dict(rec),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def load_ab_records(path) -> tuple[dict, dict[tuple[int, int], AbRecord]]:
    """Parse one arm's results_partial.jsonl into
    ``(identity, {(arm, replication): AbRecord})`` -- same tolerance rules as
    the demand loader: a truncated FINAL line (crash mid-append) is skipped
    with a warning; anything else malformed or foreign-schema'd is refused."""
    import json

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    identity: dict | None = None
    records: dict[tuple[int, int], AbRecord] = {}
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
        ident = {k: rec.get(k) for k in ("master_seed", "config_hash", "reps")}
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
        records[(ab.arm, ab.replication)] = ab
    if identity is None:
        raise SystemExit(f"--resume: no completed replications in {path}")
    return identity, records


def load_resume_dir(run_dir, expected: dict) -> dict[tuple[int, int], AbRecord]:
    """Collect every completed (arm, replication) from a previous run
    directory's per-arm partial logs; REFUSE any log whose identity does not
    match this run exactly (a partial log can only resume the run that wrote
    it). A missing/empty arm folder simply contributes nothing."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"--resume: no such run directory: {run_dir}")
    done: dict[tuple[int, int], AbRecord] = {}
    partials = sorted(run_dir.glob(f"simulation-*/{PARTIAL_FILENAME}"))
    if not partials:
        raise SystemExit(f"--resume: no {PARTIAL_FILENAME} found under {run_dir}")
    for p in partials:
        identity, records = load_ab_records(p)
        if identity != expected:
            diffs = "; ".join(
                f"{k}: partial={identity.get(k)!r} vs current={expected[k]!r}"
                for k in expected if identity.get(k) != expected[k]
            )
            raise SystemExit(f"--resume rejected ({p}): run identity mismatch "
                             f"({diffs}).")
        done.update(records)
    return done


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def arm_summary(records: list[AbRecord]) -> dict:
    """One arm's aggregate row set. The outcome breakdown keeps SUCCESS /
    PARTIAL / INCOMPLETE / FAILED as separate rows (never folded): the
    INCOMPLETE -> PARTIAL shift between arms is itself a thesis observable."""
    n = len(records)
    outcomes = Counter(r.outcome for r in records)
    n_succ = outcomes.get(Outcome.MISSION_SUCCESS.value, 0)
    lo, hi, phat = wilson_ci(n_succ, n) if n else (0.0, 0.0, 0.0)
    demands = sorted(r.demand for r in records if r.demand is not None)
    depths = [d for r in records for d in r.return_depths]
    return {
        "n_reps": n,
        "outcome_counts": {o.value: outcomes.get(o.value, 0) for o in Outcome},
        "success_frac": phat if n else None,
        "wilson95_lo": lo if n else None,
        "wilson95_hi": hi if n else None,
        "reason_counts": {
            r: sum(rec.reasons.get(r, 0) for rec in records) for r in RTH_REASONS
        },
        "demand_median": float(statistics.median(demands)) if demands else None,
        "demand_max": demands[-1] if demands else None,
        "return_depth_min": min(depths) if depths else None,
        "return_depth_mean": (sum(depths) / len(depths)) if depths else None,
        "total_energy_mean_j": (sum(r.total_energy_j for r in records) / n) if n else None,
        "duration_mean_s": (sum(r.duration_s for r in records) / n) if n else None,
        "coverage_frac_mean": (sum(r.coverage_frac for r in records) / n) if n else None,
        "n_reps_with_skipped_legs": sum(1 for r in records if r.n_skipped_legs),
        "total_skipped_legs": sum(r.n_skipped_legs for r in records),
        "n_map_hits": sum(r.n_map_hits for r in records),
        "n_map_fallbacks": sum(r.n_map_fallbacks for r in records),
        "n_route_fallbacks": sum(r.n_route_fallbacks for r in records),
    }


LIMITATIONS = (
    "arm 1 runs with safety.stall_skip=true (the experiment constant), so it is "
    "'static-40% net + EM-01 skip-on-stall', not a pristine literature baseline; "
    "this touches only the secondary success/demand columns, never the reason "
    "attribution",
    "contrast decomposition: arm 3's reason attribution matches arm 1's BY "
    "DESIGN (the static 0.40 net pre-empts the computing map), so arm3-arm1 "
    "isolates ROUTING and arm4-arm3 isolates decide+zone_demotion; 'arm 3 ~= "
    "arm 1 by reason' must NOT be read as 'the map does nothing'",
)


def _results_dict(records: list[AbRecord], identity: dict, flags: dict,
                  sim_identity: dict) -> dict:
    return {
        "schema": RESULTS_SCHEMA,
        "kind": "results",
        "mode": "rth_ab",
        "identity": sim_identity,
        "run_identity": identity,
        "arm_flags": flags,
        "status": "ok",
        "paired_design": {
            "note": "shared RngFactory + identical replication indices across "
                    "arms => env & failure draws byte-identical; arms differ "
                    "only in the energy_map flags (+ arm-4 terminal floor)",
            "success_predicate": "outcome is MISSION_SUCCESS (strict; PARTIAL "
                                 "and INCOMPLETE both count as infinite demand "
                                 "but stay separate outcome rows)",
        },
        "limitations": list(LIMITATIONS),
        "summary": arm_summary(records),
        "records": [_record_dict(r) for r in records],
    }


def _render(per_arm: dict[int, list[AbRecord]]) -> str:
    """Headline stdout table (ASCII-safe for piped Windows consoles)."""
    lines = [
        "# RTH A/B (Stage 5): four arms on paired seeds\n",
        "| arm | success | partial | incomplete | failed | rth_energy | "
        "critical_battery | terminal_battery | demand med | depth mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(per_arm):
        s = arm_summary(per_arm[arm])
        oc = s["outcome_counts"]
        rc = s["reason_counts"]
        dm = "-" if s["demand_median"] is None else f"{s['demand_median']:g}"
        dp = "-" if s["return_depth_mean"] is None else f"{s['return_depth_mean']:.3f}"
        lines.append(
            f"| {arm} {ARM_SLUGS[arm]} | {oc['MISSION_SUCCESS']} "
            f"| {oc['MISSION_PARTIAL']} | {oc['MISSION_INCOMPLETE']} "
            f"| {oc['MISSION_FAILED']} | {rc['rth_energy']} "
            f"| {rc['critical_battery']} | {rc['terminal_battery']} "
            f"| {dm} | {dp} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 5 RTH A/B: four arms on paired seeds (task C1).")
    ap.add_argument("--config", default="config/study01_demand.yaml")
    ap.add_argument("--reps", type=int, default=100,
                    help="paired replications per arm")
    ap.add_argument("--arms", type=int, nargs="+", default=[1, 2, 3, 4],
                    choices=[1, 2, 3, 4], help="subset of arms to run")
    ap.add_argument("--out", default="runs", help="runs/ base directory")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="previous rth-ab run directory; its completed "
                         "(arm, replication) pairs are identity-checked, "
                         "skipped, and merged into the final report")
    args = ap.parse_args(argv)
    arms = sorted(set(args.arms))

    base = experiment_constants(load_config(args.config))
    identity = _identity(base, args.reps)
    # one shared factory => paired seeds across every arm
    rng = RngFactory(base.sim.master_seed)

    done: dict[tuple[int, int], AbRecord] = {}
    if args.resume:
        done = load_resume_dir(args.resume, identity)
        print(f"[resume] {len(done)} completed (arm, rep) pairs from "
              f"{args.resume}", file=sys.stderr)

    run = RunContext(base_dir=args.out, name=unique_run_name("rth_ab"))
    per_arm: dict[int, list[AbRecord]] = {}
    for arm in arms:
        cfg_arm = arm_config(base, arm)
        flags = arm_flags(cfg_arm, arm)
        sim = run.simulation(f"arm{arm}-{ARM_SLUGS[arm]}")
        sim.write_plan({
            "schema": "uav-swarm-sim/plan/v1",
            "kind": "plan",
            "identity": sim.identity(config_hash=base.config_hash),
            "setup": {
                "mode": "rth_ab",
                "arm_flags": flags,
                "constants": {
                    "pool": "unbounded (fleet.total_reserve_batteries=None)",
                    "stall_skip": True,
                    "stall_detector": base.safety.stall_detector,
                    "transit_free_space": base.coverage.transit_free_space,
                    "reserve_frac": base.rth.reserve_frac,
                    "master_seed": base.sim.master_seed,
                },
                "reps": args.reps,
                "n_drones": base.fleet.n_drones,
                "limitations": list(LIMITATIONS),
            },
        })
        partial_path = sim.path(PARTIAL_FILENAME)

        resumed = [done[(arm, k)] for k in range(1, args.reps + 1)
                   if (arm, k) in done]
        # replay resumed records into THIS run's partial log so the new log is
        # itself a complete resume point if this run is also interrupted
        for rec in resumed:
            append_ab_record(partial_path, rec, identity)

        def _progress(rec: AbRecord, _arm=arm) -> None:
            append_ab_record(partial_path, rec, identity)
            d = "inf" if rec.demand is None else rec.demand
            r = rec.reasons
            print(f"  arm {_arm} rep {rec.replication:>4}/{args.reps}: "
                  f"{rec.outcome}  D={d}  rth={r['rth_energy']} "
                  f"crit={r['critical_battery']} term={r['terminal_battery']}",
                  file=sys.stderr)

        todo = [k for k in range(1, args.reps + 1) if (arm, k) not in done]
        print(f"Arm {arm} ({ARM_SLUGS[arm]}): {len(todo)} replications "
              f"({len(resumed)} resumed)...", file=sys.stderr)
        new_recs = run_arm(cfg_arm, arm, args.reps, rng,
                           progress=_progress, replications=todo)
        records = sorted(resumed + new_recs, key=lambda r: r.replication)
        per_arm[arm] = records
        sim.write_results(_results_dict(records, identity, flags,
                                        sim.identity()))

    run.finalize(summary={
        "mode": "rth_ab",
        "reps": args.reps,
        "arms": {str(a): arm_summary(per_arm[a]) for a in arms},
    })

    print(_render(per_arm))
    print(f"\n[structured output: {run.dir}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
