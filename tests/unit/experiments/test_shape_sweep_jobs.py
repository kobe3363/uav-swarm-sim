"""Fast unit guards for ENG-09 --jobs auto core resolution (no simulation)."""
from __future__ import annotations

import os

from uav_swarm_sim.experiments.run_shape_sweep import _auto_jobs


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
