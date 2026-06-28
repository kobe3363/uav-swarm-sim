"""Structured, comparable run output.

Organizes simulation artifacts under a single timestamped, GUID-identified run:

    runs/run-2026-06-28-11-59-35/        (a RUN: name = the dir, id = a GUID)
      run.json                           (manifest: identity, software, timing, sims)
      simulation-weighted/               (a SIMULATION within the run)
        plan.json                        (the launch PLAN: every input/setup)
        results.json                     (the OUTCOME: success rate, SMDP, MC logic, timing)
        environment.png  partition.png  paths.png  replay.gif  tracks.gpx  ...
      simulation-kmeans/
        ...

Design intent: one run can hold several simulations (e.g. one per decomposition
algorithm); every artifact of a simulation lives in its own folder; the two JSON
files separate *what was launched* (plan) from *what happened* (results); and a
run is identifiable and comparable by ``run_id`` (GUID) and ``run_name`` (date),
with a ``config_hash`` per simulation for exact-input matching.

Both schemas are deliberately verbose -- it is cheaper to log a field you turn
out not to need than to re-run a study because you didn't.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from ..infrastructure.enums import Outcome
from .convergence import ci_half_width

PLAN_SCHEMA = "uav-swarm-sim/plan/v1"
RESULTS_SCHEMA = "uav-swarm-sim/results/v1"
RUN_SCHEMA = "uav-swarm-sim/run/v1"


# --------------------------------------------------------------------------- #
# JSON-safe conversion                                                         #
# --------------------------------------------------------------------------- #
def _jsonable(o: Any) -> Any:
    """Recursively convert configs/enums/tuples/Paths/numpy into JSON types.

    Non-finite floats (inf/nan, e.g. an undefined CI half-width at n<2) become
    null, so the output is strict-JSON valid (consumable by JS/jq/pandas, which
    reject Infinity)."""
    if isinstance(o, bool):
        return o
    if isinstance(o, Enum):
        return o.value
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: _jsonable(getattr(o, f.name)) for f in fields(o)}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (str, int)) or o is None:
        return o
    return str(o)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_version() -> str | None:
    try:
        from importlib.metadata import version
        for name in ("uav-swarm-sim", "uav_swarm_sim"):
            try:
                return version(name)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _software() -> dict:
    return {
        "package_version": _package_version(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stat(samples) -> dict:
    """mean / sample-std / 95% CI half-width / n, dropping NaNs."""
    arr = [float(x) for x in samples if x is not None and x == x]
    if not arr:
        return {"mean": None, "std": None, "ci95_half_width": None, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci = ci_half_width(arr) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_half_width": ci, "n": len(arr)}


# --------------------------------------------------------------------------- #
# run / simulation containers                                                  #
# --------------------------------------------------------------------------- #
class SimulationOutput:
    """One simulation's folder within a run; owns plan.json, results.json, and
    all of that simulation's artifacts (png/gif/gpx)."""

    def __init__(self, name: str, directory: Path, run_id: str, run_name: str) -> None:
        self.name = name
        self.id = uuid4().hex[:12]
        self.created_at = _now_utc_iso()
        self.dir = directory
        self.run_id = run_id
        self.run_name = run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_hash: str | None = None
        self.status: str = "created"

    def path(self, filename: str) -> Path:
        """Absolute path for an artifact (figure/gif/gpx) inside this folder."""
        return self.dir / filename

    def identity(self, config_hash: str | None = None) -> dict:
        if config_hash is not None:
            self.config_hash = config_hash
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "simulation_id": self.id,
            "simulation_name": self.name,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
        }

    def write_plan(self, plan: dict) -> Path:
        self.config_hash = plan.get("identity", {}).get("config_hash", self.config_hash)
        p = self.dir / "plan.json"
        p.write_text(json.dumps(_jsonable(plan), indent=2), encoding="utf-8")
        return p

    def write_results(self, results: dict) -> Path:
        self.status = results.get("status", self.status)
        p = self.dir / "results.json"
        p.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
        return p


class RunContext:
    """A timestamped, GUID-identified run directory holding 1+ simulations."""

    def __init__(self, base_dir: str = "runs", name: str | None = None,
                 run_id: str | None = None, command: list[str] | None = None) -> None:
        self.run_id = run_id or uuid4().hex[:12]
        self.created_at = _now_utc_iso()
        self.name = name or ("run-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.dir = Path(base_dir) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.command = command if command is not None else sys.argv
        self._sims: list[SimulationOutput] = []
        self._t0 = perf_counter()

    def simulation(self, name: str) -> SimulationOutput:
        sim = SimulationOutput(name, self.dir / f"simulation-{name}", self.run_id, self.name)
        self._sims.append(sim)
        return sim

    def finalize(self, summary: dict | None = None) -> Path:
        manifest = {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "run_name": self.name,
            "created_at": self.created_at,
            "finished_at": _now_utc_iso(),
            "wall_time_s": round(perf_counter() - self._t0, 3),
            "software": _software(),
            "command": list(self.command),
            "n_simulations": len(self._sims),
            "simulations": [
                {
                    "name": s.name,
                    "id": s.id,
                    "dir": s.dir.name,
                    "plan": "plan.json",
                    "results": "results.json",
                    "config_hash": s.config_hash,
                    "status": s.status,
                }
                for s in self._sims
            ],
        }
        if summary:
            manifest["summary"] = summary
        p = self.dir / "run.json"
        p.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        return p


# --------------------------------------------------------------------------- #
# plan.json builder                                                            #
# --------------------------------------------------------------------------- #
def build_plan(cfg, *, identity: dict, algo, planner, engine=None) -> dict:
    """The launch PLAN: a curated 'setup' highlight block (the inputs an analyst
    reaches for first), the full config dump, and engine-derived spatial
    quantities when an already-built engine is supplied."""
    energy_weighting = getattr(algo, "value", str(algo)) == "weighted_voronoi"

    setup = {
        "n_drones": cfg.fleet.n_drones,
        "drone_type": cfg.platform.type.value,
        "battery_capacity_wh": cfg.fleet.battery_capacity_wh,
        "battery_capacity_j": cfg.fleet.battery_capacity_j,
        "drone_dims_m": list(cfg.fleet.drone_dims_m),
        "total_reserve_batteries": cfg.fleet.total_reserve_batteries,  # None => unbounded pool
        "coverage_altitude_m": cfg.env.coverage_altitude_m,
        "obstacle_density_per_km2": cfg.env.obstacle_density_per_km2,
        "obstacle_size_range_m": list(cfg.env.obstacle_size_range_m),
        "clearance_buffer_m": cfg.env.clearance_buffer_m,
        "dynamic_obstacles_enabled": cfg.dynamic_obstacles.enabled,
        "battery_zones": {
            "high": cfg.battery_zones.high,
            "nominal": cfg.battery_zones.nominal,
            "critical": cfg.battery_zones.critical,
            "note": "fractions of capacity; TERMINAL is < critical",
        },
        "decomposition_algorithm": getattr(algo, "value", str(algo)),
        "flight_planner": getattr(planner, "value", str(planner)),
        "energy_weighting_enabled": energy_weighting,
        "mission_type": cfg.mission.type.value if hasattr(cfg.mission.type, "value") else str(cfg.mission.type),
        "master_seed": cfg.sim.master_seed,
    }

    derived: dict = {}
    if engine is not None and getattr(engine, "env", None) is not None:
        env = engine.env
        survey_area = float(env.area.area)
        obstacle_area = float(sum(o.polygon.area for o in env.obstacles))
        free_area = float(env.free_space.area) if getattr(env, "free_space", None) is not None else survey_area
        derived = {
            "survey_area_m2": survey_area,
            "free_area_m2": free_area,
            "obstacle_count": len(env.obstacles),
            "total_obstacle_area_m2": obstacle_area,
            "obstacle_to_survey_area_ratio": (obstacle_area / survey_area) if survey_area > 0 else None,
            "free_to_survey_area_ratio": (free_area / survey_area) if survey_area > 0 else None,
            "planning_time_s": getattr(engine, "planning_time_s", None),
        }
        lp = getattr(engine, "launch_pose", None)
        if lp is not None:
            derived["launch_pose_xy"] = [float(lp.x), float(lp.y)]
            derived["launch_pose_z"] = float(getattr(lp, "z", 0.0))
        scores = getattr(engine, "site_scores", None)
        if scores is not None:
            derived["n_launch_candidates_evaluated"] = len(scores)

    return {
        "schema": PLAN_SCHEMA,
        "kind": "plan",
        "identity": identity,
        "software": _software(),
        "setup": setup,
        "derived": derived,
        "config": _jsonable(cfg),
    }


# --------------------------------------------------------------------------- #
# results.json builders                                                        #
# --------------------------------------------------------------------------- #
_OUTCOMES = (Outcome.MISSION_SUCCESS, Outcome.MISSION_FAILED, Outcome.MISSION_INCOMPLETE)


def _outcome_counts(runs) -> dict[str, int]:
    counts = {o.value: 0 for o in _OUTCOMES}
    for r in runs:
        oc = getattr(r, "outcome", None)
        key = oc.value if isinstance(oc, Outcome) else (oc if isinstance(oc, str) else None)
        if key in counts:
            counts[key] += 1
    return counts


def build_results_mc(mc, *, identity: dict, wall_time_s: float,
                     variant=None) -> dict:
    """The OUTCOME of a Monte-Carlo simulation batch: how many replications ran
    and why it stopped, how many missions succeeded, the SMDP stationary
    distribution and efficiency, aggregated metrics, and timing."""
    counts = _outcome_counts(mc.runs)
    n_total = len(mc.runs)
    n_success = counts.get(Outcome.MISSION_SUCCESS.value, 0)
    final_ci = mc.convergence_trace[-1][2] if mc.convergence_trace else None
    # the MC loop stops for exactly two reasons: the CI converged, or it ran out
    # of the n_max budget without converging.
    stop_reason = "ci_converged" if mc.converged else "reached_n_max"

    # per-metric aggregates: prefer the variant's per-run lists, else the runs'
    metric_block: dict = {}
    if variant is not None:
        metric_block = {
            "total_energy_j": _stat(variant.total_energy_j),
            "duration_s": _stat(variant.duration_s),
            "workload_std_m": _stat(variant.workload_std_m),
            "planning_time_s": _stat(variant.planning_time_s),
        }

    return {
        "schema": RESULTS_SCHEMA,
        "kind": "results",
        "mode": "monte_carlo",
        "identity": identity,
        "status": "ok" if n_success > 0 else ("no_success" if n_total else "empty"),
        "monte_carlo": {
            "n_runs": mc.n_runs,
            "converged": bool(mc.converged),
            "stop_reason": stop_reason,
            "monitored_quantity": "pi_time(S2_MISSION)",
            "final_ci95_half_width": final_ci,
            "convergence_trace": [
                {"n": n, "mean": mean, "ci95_half_width": hw}
                for (n, mean, hw) in mc.convergence_trace
            ],
        },
        "outcomes": {
            "n_runs": n_total,
            "n_success": n_success,
            "n_failed": counts.get(Outcome.MISSION_FAILED.value, 0),
            "n_incomplete": counts.get(Outcome.MISSION_INCOMPLETE.value, 0),
            "success_frac": (n_success / n_total) if n_total else None,
            "outcome_counts": counts,
            "smdp_aborted_frac": mc.aborted_frac,  # SMDP non-ergodic share (distinct from mission failure)
        },
        "smdp": {
            "stationary_pi_time": {s.value: mc.pi_time_mean.get(s, 0.0) for s in mc.pi_time_mean},
            "pi_time_ci95_half_width": {s.value: mc.pi_time_ci.get(s, 0.0) for s in mc.pi_time_ci},
            "efficiency_mean": mc.efficiency_mean,
            "efficiency_ci95_half_width": mc.efficiency_ci,
            "efficiency_note": "pi(S2) / (pi(S3)+pi(S_OBS)+pi(S_SWAP))",
        },
        "metrics": metric_block,
        "timing": {
            "wall_time_total_s": round(wall_time_s, 3),
            "wall_time_mean_per_run_s": round(wall_time_s / mc.n_runs, 4) if mc.n_runs else None,
        },
    }


def build_results_single(result, est, *, identity: dict, wall_time_s: float) -> dict:
    """The OUTCOME of a single mission (the visual-demo case): terminal outcome,
    its SMDP, and the single-run metrics."""
    m = result.metrics
    smdp: dict = {"ergodic": bool(getattr(est, "ergodic", False))}
    if getattr(est, "ergodic", False):
        from .stationary_distribution import stationary
        from .efficiency_score import efficiency
        _, pi_time = stationary(est)
        smdp["stationary_pi_time"] = {s.value: float(pi_time[i]) for i, s in enumerate(est.states)}
        smdp["efficiency"] = float(efficiency(pi_time, est.states))

    return {
        "schema": RESULTS_SCHEMA,
        "kind": "results",
        "mode": "single_mission",
        "identity": identity,
        "status": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
        "outcome": {
            "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
            "coverage_frac": result.coverage_frac,
            "aborted": result.aborted,
        },
        "smdp": smdp,
        "metrics": {
            "total_energy_j": m.total_energy_j,
            "duration_s": m.duration_s,
            "workload_std_m": m.workload_std_m,
            "n_swaps": m.n_swaps,
            "n_failures": m.n_failures,
            "planning_time_s": m.planning_time_s,
            "per_agent_length_m": {str(k): v for k, v in m.per_agent_length_m.items()},
        },
        "timing": {"wall_time_total_s": round(wall_time_s, 3)},
    }
