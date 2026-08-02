"""E2 byte-identity gate for run_spare_sizing --demand-mode under --jobs.

The determinism seam: parallelising the unbounded demand batch must be bitwise
identical to serial, or the C2 / STUDY-01 re-run this gates would depend on a
worker count. These gates run the real engine (slow); the pure resolution
helpers are covered fast in tests/unit/experiments/test_parallel_jobs.py.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.experiments.run_spare_sizing import (
    _partial_identity, append_demand_record, demand_with_partials, run_demand)
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.rng import RngFactory


def _tiny_demand_cfg(config_path):
    """2 drones on ~20 Wh packs over the smoke area -> demands D in {1, 2} with
    MISSION_SUCCESS in ~a second per replication (mirrors the spare-sizing
    suite's fixture, kept local so this gate is self-contained)."""
    return load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 20.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "env.obstacle_size_range_m": [10.0, 30.0],
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })


def _by_rep(records):
    """DemandRecords sorted by replication -- the full identity comparison. The
    record carries NO wall-clock field, so every field is matched bitwise."""
    return sorted(records, key=lambda r: r.replication)


@pytest.mark.slow
def test_demand_jobs_serial_parallel_byte_identical(config_path):
    """E2 CORE GATE (demand path): --jobs 1 (serial) vs --jobs 2 (spawn) produce
    byte-identical DemandRecords. 1 vs 2 straddles the serial <-> spawn boundary
    (the only place BLAS-threaded FP reduction could diverge); reps=4 over 2
    workers interleaves completion, so passing also proves the index-sort
    reassembly is jobs-invariant."""
    cfg = _tiny_demand_cfg(config_path)
    reps = 4
    serial = run_demand(cfg, reps, RngFactory(cfg.sim.master_seed), jobs=1)
    parallel = run_demand(cfg, reps, RngFactory(cfg.sim.master_seed), jobs=2)
    assert _by_rep(serial) == _by_rep(parallel)


@pytest.mark.slow
def test_demand_resume_under_jobs_matches_uninterrupted_serial(config_path, tmp_path):
    """Resume + --jobs together: a batch that 'crashed' after replication 2,
    resumed with --jobs 2, yields the exact records of an uninterrupted serial
    run. Proves single-writer-in-parent + index-keyed resume survive parallel
    execution byte-for-byte."""
    cfg = _tiny_demand_cfg(config_path)
    reps = 4
    seed = cfg.sim.master_seed

    reference = _by_rep(demand_with_partials(
        cfg, reps, RngFactory(seed), tmp_path / "ref.jsonl", jobs=1))

    # a batch that finished only replications 1..2, logged under the reps=4 identity
    ident = _partial_identity(cfg, reps)
    crashed = tmp_path / "crashed.jsonl"
    run_demand(cfg, reps, RngFactory(seed), replications=[1, 2],
               progress=lambda rec: append_demand_record(crashed, rec, ident))

    resumed = _by_rep(demand_with_partials(
        cfg, reps, RngFactory(seed), tmp_path / "resumed.jsonl",
        resume_path=crashed, jobs=2))
    assert resumed == reference
