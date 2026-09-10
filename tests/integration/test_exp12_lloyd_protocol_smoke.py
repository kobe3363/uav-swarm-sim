"""EXP-12 integration smoke: both Lloyd arms on a small real world.

Uses the test_exp07_lloyd override template (small rectangle, hazard 0, raster +
photogrammetry on) rather than the full-scale default. Pins: a real 2-rep paired
run of both arms with per-k manifest parity (fingerprints equal across arms), and
--jobs / arm-order invariance on the physical projection (never raw JSON, which
carries wall/planning times). No pinned golden hash constants -- every comparison
is between values produced WITHIN the same run.
"""
from __future__ import annotations

import json

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.experiments.run_lloyd_protocol import (
    ARMS,
    build_cfg,
    projection,
    run_arm,
    validate_config,
)
from uav_swarm_sim.infrastructure.rng import RngFactory


@pytest.fixture
def cfg_factory(tmp_path):
    area = tmp_path / "rect.geojson"
    area.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(box(0, 0, 600, 240))}))
    extra = {
        "fleet.battery_capacity_wh": 1000.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "launch.candidate_sites": [[300.0, 0.0]],
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "platforms.MULTIROTOR.r_min_m": 0.0, "platforms.MULTIROTOR.omega_max": 1.0,
        "sensor.sensor_power_w": 15.0,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 8.0,
        "sensor.photogrammetry.sensor_height_mm": 6.0,
        "sensor.photogrammetry.focal_length_mm": 10.0,
        "sensor.photogrammetry.image_width_px": 4000,
        "sensor.photogrammetry.image_height_px": 3000,
        "sensor.photogrammetry.side_overlap": 0.5,
        "sensor.photogrammetry.forward_overlap": 0.5,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 10.0,
        "planning.energy_balance.enabled": True,
        "sim.dt_s": 0.5, "sim.max_timesteps": 2000,
    }

    def make(n=3):
        cfg = build_cfg("config/study01_demand.yaml", n, extra)
        validate_config(cfg)
        return cfg

    return make


def _run_both(cfg, n, reps, jobs=1, arms=ARMS):
    """One shared factory (paired seeds) across both arms; returns {algo: [recs]}."""
    rng = RngFactory(cfg.sim.master_seed)
    return {a.value: run_arm(cfg, a, n, rng, range(1, reps + 1), False, jobs=jobs)
            for a in arms}


def test_smoke_both_arms_run_and_manifests_pair(cfg_factory):
    cfg = cfg_factory(3)
    out = _run_both(cfg, 3, reps=2)
    cvt = {r.replication: r for r in out["lloyd_cvt"]}
    energy = {r.replication: r for r in out["lloyd_energy"]}
    assert set(cvt) == {1, 2} and set(energy) == {1, 2}
    for k in (1, 2):
        assert cvt[k].manifest is not None and energy[k].manifest is not None
        # AC-1: paired physical inputs -> identical fingerprint across arms at k
        assert (cvt[k].manifest["fingerprint_sha256"]
                == energy[k].manifest["fingerprint_sha256"])
    # assert the RESOLVED decomposer identity, not just contract presence:
    # ProtocolRecord.algo only echoes the requested arg, so a presence check could
    # pass even if both arms resolved the same decomposer.
    assert cvt[1].contract is not None
    assert energy[1].contract is not None
    assert cvt[1].contract["outcome"]["decomposer_class"] == "LloydCvtDecomposer"
    assert energy[1].contract["outcome"]["decomposer_class"] == "LloydEnergyDecomposer"


def test_smoke_arm_order_is_invariant(cfg_factory):
    cfg = cfg_factory(3)
    forward = _run_both(cfg, 3, reps=2, arms=ARMS)
    reverse = _run_both(cfg, 3, reps=2, arms=tuple(reversed(ARMS)))
    for a in ARMS:
        fwd = sorted(forward[a.value], key=lambda r: r.replication)
        rev = sorted(reverse[a.value], key=lambda r: r.replication)
        assert [projection(r) for r in fwd] == [projection(r) for r in rev]


@pytest.mark.slow
def test_smoke_jobs_invariant(cfg_factory):
    cfg = cfg_factory(3)
    serial = _run_both(cfg, 3, reps=2, jobs=1)
    parallel = _run_both(cfg, 3, reps=2, jobs=2)
    for a in ARMS:
        s = sorted(serial[a.value], key=lambda r: r.replication)
        p = sorted(parallel[a.value], key=lambda r: r.replication)
        assert [projection(r) for r in s] == [projection(r) for r in p]
