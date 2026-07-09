"""Opt-in wall-time PHASE profiling for the experiment sweeps.

DEFAULT-OFF and byte-identical when off. Gated on the ``UAV_SWARM_PROFILE``
environment variable, read once at import: when disabled, :func:`phase` is a
zero-overhead no-op context manager (it yields immediately, takes no
``perf_counter`` reading and touches no state), so a profiled-off run is
byte-identical to the pre-instrumentation baseline. The timers only ever
*measure* -- they never alter control flow or data.

When enabled, ``phase(name)`` accumulates ``(calls, total_wall_s)`` into a
process-local dict. In the parallel sweeps each worker process owns its own
accumulator; a worker flushes :func:`flush_worker` to
``<UAV_SWARM_PROFILE_DIR>/_profiles/phases_<pid>.json`` and the parent
:func:`collect`\\ s every worker file into one report. Environment variables are
inherited by ``ProcessPoolExecutor`` workers (spawn re-imports this module and
re-reads them), so no flag has to be threaded through the call sites.

This is the "which PROCESSES take longest" coarse view; the ``--profile``
cProfile mode in the experiment CLIs gives the complementary per-FUNCTION view.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

_ENV_ENABLE = "UAV_SWARM_PROFILE"
_ENV_DIR = "UAV_SWARM_PROFILE_DIR"


def enabled() -> bool:
    """True iff phase profiling is switched on for this process (env-driven)."""
    return bool(os.environ.get(_ENV_ENABLE))


# Cached once at import so the hot path pays a single attribute read, not an
# os.environ lookup, per phase. Tests toggle this attribute directly.
_ENABLED = enabled()

# name -> [calls, total_s]
_ACC: dict[str, list] = {}


@contextmanager
def phase(name: str):
    """Accumulate wall time spent in the ``with`` block under ``name``.

    No-op (zero measurement, zero state change) when profiling is disabled, so
    the instrumented code path stays byte-identical to the un-instrumented one.
    """
    if not _ENABLED:
        yield
        return
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        rec = _ACC.get(name)
        if rec is None:
            _ACC[name] = [1, dt]
        else:
            rec[0] += 1
            rec[1] += dt


def record(name: str, seconds: float) -> None:
    """Manually add a measured span to ``name`` (one call = one increment).

    For sites where a ``with`` block is awkward -- e.g. timing a large loop
    without re-indenting its whole body. No-op when profiling is disabled, so
    the surrounding code stays byte-identical.
    """
    if not _ENABLED:
        return
    rec = _ACC.get(name)
    if rec is None:
        _ACC[name] = [1, seconds]
    else:
        rec[0] += 1
        rec[1] += seconds


def snapshot() -> dict[str, list]:
    """A copy of this process's accumulated ``{name: [calls, total_s]}``."""
    return {k: [v[0], v[1]] for k, v in _ACC.items()}


def reset() -> None:
    """Clear this process's accumulator (used by tests and cProfile runs)."""
    _ACC.clear()


def merge(into: dict[str, list], other: dict[str, list]) -> dict[str, list]:
    """Sum ``other`` into ``into`` bucket-wise; returns ``into``."""
    for k, rec in other.items():
        calls, secs = rec[0], rec[1]
        cur = into.get(k)
        if cur is None:
            into[k] = [calls, secs]
        else:
            cur[0] += calls
            cur[1] += secs
    return into


def flush_worker() -> None:
    """Persist this process's snapshot to ``<PROFILE_DIR>/_profiles/phases_<pid>.json``.

    Called by a worker at the end of each cell/tier when profiling is on;
    overwriting per-pid means the last write holds the full accumulation for
    that (possibly reused) worker process. No-op if profiling is off or no dir
    was set.
    """
    if not _ENABLED:
        return
    d = os.environ.get(_ENV_DIR)
    if not d:
        return
    pdir = Path(d) / "_profiles"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"phases_{os.getpid()}.json").write_text(
        json.dumps(snapshot()), encoding="utf-8")


def collect(profile_dir: str | Path) -> dict[str, list]:
    """Merge every worker's ``phases_*.json`` under ``<profile_dir>/_profiles``
    with this process's own in-memory snapshot into one accumulator, so both the
    serial (``--jobs 1``, no files) and parallel paths report the same way."""
    merged: dict[str, list] = {}
    merge(merged, snapshot())
    pdir = Path(profile_dir) / "_profiles"
    if pdir.is_dir():
        for f in sorted(pdir.glob("phases_*.json")):
            try:
                merge(merged, json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue  # a half-written/corrupt worker file never aborts the report
    return merged


# canonical pipeline order for the report (setup phases, then the dt-loop, then
# post-processing); any unrecognised bucket is appended by descending cost.
_ORDER = ("build.load_area", "build.env_obstacles", "build.gvg_tgc",
          "build.launch_opt", "build.decompose", "build.coverage_plan",
          "dt_loop", "telemetry_export", "metrics_compute", "smdp_reduce")


def _sorted_items(snap: dict[str, list]) -> list[tuple[str, list]]:
    known = [(k, snap[k]) for k in _ORDER if k in snap]
    extra = sorted(((k, v) for k, v in snap.items() if k not in _ORDER),
                   key=lambda kv: kv[1][1], reverse=True)
    return known + extra


def format_report(snap: dict[str, list]) -> str:
    """Markdown table: phase, total_s, %, calls, mean_ms.

    Percentages are of the SUMMED measured phase time (an attribution of the
    instrumented wall time, not the run's wall clock), so they add to 100%.
    """
    items = _sorted_items(snap)
    total = sum(v[1] for _, v in items) or 1.0
    lines = ["| phase | total_s | % | calls | mean_ms |",
             "|---|---:|---:|---:|---:|"]
    for name, (calls, secs) in items:
        mean_ms = 1000.0 * secs / calls if calls else 0.0
        lines.append(f"| {name} | {secs:.3f} | {100.0 * secs / total:.1f} | "
                     f"{calls} | {mean_ms:.3f} |")
    lines.append(f"| **sum** | **{total:.3f}** | 100.0 | | |")
    return "\n".join(lines)


def to_csv_rows(snap: dict[str, list]) -> list[list]:
    """``[[header], [phase, total_s, calls, mean_ms], ...]`` for a CSV writer."""
    rows: list[list] = [["phase", "total_s", "calls", "mean_ms"]]
    for name, (calls, secs) in _sorted_items(snap):
        mean_ms = 1000.0 * secs / calls if calls else 0.0
        rows.append([name, f"{secs:.6f}", calls, f"{mean_ms:.6f}"])
    return rows
