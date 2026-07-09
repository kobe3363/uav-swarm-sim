"""Guards for the opt-in phase-profiling layer (infrastructure/profiling.py).

The load-bearing property is BYTE-IDENTITY when disabled: ``phase``/``record``
must touch no state and take no measurement, so an un-profiled run is exactly the
pre-instrumentation baseline. When enabled, buckets accumulate and round-trip
through the per-worker JSON flush the parallel sweeps use.
"""
from __future__ import annotations

from uav_swarm_sim.infrastructure import profiling


def test_disabled_phase_and_record_are_noops(monkeypatch):
    """The byte-identity guarantee: OFF -> nothing recorded, nothing mutated."""
    monkeypatch.setattr(profiling, "_ENABLED", False)
    profiling.reset()
    with profiling.phase("build.gvg_tgc"):
        pass
    profiling.record("dt_loop", 1.0)
    assert profiling.snapshot() == {}


def test_enabled_phase_and_record_accumulate(monkeypatch):
    monkeypatch.setattr(profiling, "_ENABLED", True)
    profiling.reset()
    with profiling.phase("a"):
        pass
    with profiling.phase("a"):
        pass
    profiling.record("b", 2.5)
    profiling.record("b", 1.5)
    snap = profiling.snapshot()
    assert snap["a"][0] == 2                 # two calls
    assert snap["b"] == [2, 4.0]             # [calls, total_s]
    profiling.reset()
    assert profiling.snapshot() == {}


def test_merge_sums_buckets():
    into = {"a": [1, 1.0]}
    profiling.merge(into, {"a": [2, 2.0], "b": [1, 0.5]})
    assert into == {"a": [3, 3.0], "b": [1, 0.5]}


def test_format_report_orders_pipeline_then_extras_and_totals():
    snap = {"dt_loop": [10, 5.0], "build.gvg_tgc": [10, 1.0], "custom": [3, 9.0]}
    rep = profiling.format_report(snap)
    # canonical phases appear in pipeline order before any unknown bucket
    assert rep.index("build.gvg_tgc") < rep.index("dt_loop") < rep.index("custom")
    assert "**sum**" in rep


def test_collect_serial_uses_in_memory_snapshot(monkeypatch, tmp_path):
    """--jobs 1: the parent itself ran the missions -> collect folds in the
    in-memory snapshot even with no worker files present."""
    monkeypatch.setattr(profiling, "_ENABLED", True)
    profiling.reset()
    profiling.record("dt_loop", 1.0)
    merged = profiling.collect(tmp_path)      # no _profiles dir
    assert merged["dt_loop"] == [1, 1.0]
    profiling.reset()


def test_flush_worker_then_collect_from_files(monkeypatch, tmp_path):
    """--jobs N: a worker flushes to _profiles/phases_<pid>.json and the parent
    (empty in-memory) merges the files back."""
    monkeypatch.setattr(profiling, "_ENABLED", True)
    monkeypatch.setenv(profiling._ENV_DIR, str(tmp_path))
    profiling.reset()
    with profiling.phase("build.gvg_tgc"):
        pass
    profiling.flush_worker()
    assert list((tmp_path / "_profiles").glob("phases_*.json"))
    profiling.reset()                          # simulate the parent: empty in-memory
    merged = profiling.collect(tmp_path)
    assert merged.get("build.gvg_tgc", [0])[0] == 1


def test_flush_worker_is_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(profiling, "_ENABLED", False)
    monkeypatch.setenv(profiling._ENV_DIR, str(tmp_path))
    profiling.reset()
    profiling.flush_worker()
    assert not (tmp_path / "_profiles").exists()
