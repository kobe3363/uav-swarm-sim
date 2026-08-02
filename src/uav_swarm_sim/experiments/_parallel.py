"""Shared ``--jobs`` plumbing for the demand-mode experiment scripts (E2).

The determinism-preserving parallelism pattern proven by ENG-09
(``run_shape_sweep`` / ``run_scale_tiers``): a **spawn** ``ProcessPoolExecutor``
whose worker is a *pure* function of ``(master_seed, ...)`` -- so a worker
reconstructing (or being handed) an immutable ``RngFactory`` draws
byte-identically regardless of which worker finishes first -- with the **parent
process the single writer** of any crash-safe partial log (no cross-process
write contention). This module holds the two pieces the ``run_rth_ab`` and
``run_spare_sizing`` demand call sites share, so the core is not copied a third
time (``run_shape_sweep`` / ``run_scale_tiers`` keep their own in-sync copies;
unifying all four is deferred out of E2 scope).

IMPORTANT: the BLAS/OpenMP single-thread pin that keeps FP reduction order
identical serial<->parallel (and stops N workers oversubscribing N*threads)
lives at the TOP of each *entry-point* module, BEFORE numpy loads -- NOT here.
Importing this helper must not pull numpy, and the pin must be set before the
first numpy import in the spawning process. The pin is a load-bearing cause of
ENG-09 determinism, not an optimisation.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

R = TypeVar("R")


def _auto_jobs() -> int:
    """Worker count for ``--jobs auto``: PHYSICAL cores minus one (leave a core
    for the OS), floored at 1. Cross-platform (Windows/Linux/macOS/Azure) via
    psutil; degrades safely to logical CPUs then 1 if psutil is unavailable.
    (Same body as ``run_shape_sweep`` / ``run_scale_tiers._auto_jobs``.)"""
    n = None
    try:
        import psutil
        n = psutil.cpu_count(logical=False)  # None in some restricted sandboxes
    except Exception:  # noqa: BLE001 -- psutil missing/broken -> logical fallback
        n = None
    if not n:
        n = os.cpu_count()
    return max(1, (n or 1) - 1)


def resolve_jobs(arg: str) -> int:
    """Map the ``--jobs`` CLI string (``"auto"`` | integer >= 1) to a worker
    count. A non-positive count is rejected here rather than silently treated as
    serial by ``run_units`` (which runs ``jobs <= 1`` in-process), keeping the
    documented ``auto|1|N`` contract honest."""
    if arg == "auto":
        return _auto_jobs()
    jobs = int(arg)
    if jobs < 1:
        raise ValueError(f"--jobs must be 'auto' or an integer >= 1, got {arg!r}")
    return jobs


def add_jobs_arg(ap) -> None:
    """Add the standard ENG-09 ``--jobs`` argument to an ``ArgumentParser``.

    ``auto`` (default) = physical cores - 1; ``1`` = serial (the byte-identical
    revert path); or an explicit worker count. Output is byte-identical to
    serial at any value. Two-scripts rule: running BOTH demand scripts on one
    box must NOT leave each at ``auto`` (that double-subscribes the cores) --
    give each an explicit split instead."""
    ap.add_argument(
        "--jobs", default="auto", metavar="N",
        help="parallel worker processes: 'auto' (physical cores - 1), '1' "
             "(serial, the byte-identical revert path), or an explicit N. "
             "Results are byte-identical to serial at any N. Running two demand "
             "scripts at once? Do NOT leave both at 'auto' -- split the cores.")


def spawn_pool(jobs: int) -> ProcessPoolExecutor:
    """A spawn-context ``ProcessPoolExecutor`` with ``jobs`` workers.

    Spawn (not fork) avoids the deadlock of forking a multi-threaded parent on
    Linux/Azure (ENG-09); spawn children re-import the entry module, so the
    BLAS pin set at its top applies inside each worker too."""
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=jobs, mp_context=ctx)


def run_units(unit_fn: Callable[..., R],
              unit_args: Sequence[tuple] | Iterable[tuple],
              jobs: int,
              on_result: Callable[[R], None] | None = None) -> list[R]:
    """Run ``unit_fn(*args)`` for every ``args`` in ``unit_args``.

    Serial (``jobs <= 1``) runs them in ``unit_args`` order -- the byte-identical
    ENG-09 revert path. Parallel (``jobs > 1``) submits every unit to a spawn
    pool and collects via ``as_completed``. In BOTH modes ``on_result(rec)`` is
    called in THIS (parent) process the moment each result arrives, so a
    crash-safe partial-log writer stays a single writer and never contends
    across processes.

    Returns the results in COMPLETION order (serial: submission order; parallel:
    whichever worker finished first). Callers that need a stable order sort by
    their record's index field -- both demand deliverables already do, so the
    written ``results.json`` is byte-identical regardless of ``jobs``.

    Determinism contract: ``unit_fn`` must be a top-level, picklable pure
    function of its args (each picklable); it draws only from an ``RngFactory``
    that is a pure function of ``(master_seed, name, replication)``, so no
    mutable RNG state crosses the pickle boundary and completion order cannot
    change any result. Worker exceptions propagate (matching serial), so a real
    error still aborts the run rather than being silently dropped."""
    results: list[R] = []
    if jobs <= 1:
        for args in unit_args:
            rec = unit_fn(*args)
            results.append(rec)
            if on_result is not None:
                on_result(rec)
        return results
    with spawn_pool(jobs) as ex:
        futures = [ex.submit(unit_fn, *args) for args in unit_args]
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            if on_result is not None:
                on_result(rec)
    return results
