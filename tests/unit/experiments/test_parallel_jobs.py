"""Fast unit guards for the shared E2 --jobs plumbing (experiments/_parallel.py).

No simulation here: the byte-identity gates that actually run the engine under
--jobs live in tests/integration/test_rth_ab.py and test_spare_demand_jobs.py.
This module pins the pure resolution helpers (mirrors test_shape_sweep_jobs.py
for the shared _auto_jobs, so ENG-09's 'works on every OS / Azure' guarantee is
re-asserted for the copy the demand scripts import)."""
from __future__ import annotations

import os

from uav_swarm_sim.experiments._parallel import (
    _auto_jobs, resolve_jobs, run_units)


def test_auto_jobs_at_least_one_and_bounded():
    """auto = physical - 1, floored at 1, and never exceeds logical CPUs."""
    j = _auto_jobs()
    assert isinstance(j, int)
    assert j >= 1
    assert j <= (os.cpu_count() or 1)


def test_auto_jobs_survives_missing_psutil(monkeypatch):
    """If psutil is unavailable, auto degrades to logical CPUs - 1 (>=1) rather
    than crashing -- the 'works on all OS / Azure' guarantee."""
    import builtins
    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("simulated: psutil not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    j = _auto_jobs()
    assert j == max(1, (os.cpu_count() or 1) - 1)


def test_resolve_jobs_auto_and_explicit():
    """'auto' resolves to _auto_jobs(); an integer string resolves to itself."""
    assert resolve_jobs("auto") == _auto_jobs()
    assert resolve_jobs("1") == 1
    assert resolve_jobs("3") == 3


def _double(x):
    return x * 2


def test_run_units_serial_preserves_submission_order_and_callback():
    """jobs<=1 runs in unit_args order and fires on_result once per unit -- the
    byte-identical revert path (no pool involved)."""
    seen: list[int] = []
    out = run_units(_double, [(1,), (2,), (3,)], jobs=1, on_result=seen.append)
    assert out == [2, 4, 6]
    assert seen == [2, 4, 6]
