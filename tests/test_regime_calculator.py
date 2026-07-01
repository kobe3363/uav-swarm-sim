"""Tests for the A2 regime calculator (experiments/run_regime_calculator.py).

The calculator's one load-bearing claim is that its *analytical* coverage energy
(boustrophedon legs + discrete P.dt integration) equals what the engine actually
drains during the coverage phase (S2_MISSION + S_FERRY). We pin that agreement on
a small swap-free mission, plus unit-check the breakdown, the camera term, the
regime classifier, and the usable-battery floors.
"""
from __future__ import annotations

import math

import pytest
from conftest import config_path

from uav_swarm_sim.experiments.generate_shapes import normalize_to_area, shape_builders
from uav_swarm_sim.experiments.run_regime_calculator import (
    _VERIFY_TOL,
    _build_planning_layer,
    _positive_int,
    _run_once,
    _usable_fraction,
    classify,
    coverage_energy,
    per_zone_energy,
)
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model

_SMOKE = "data/areas/smoke_area.geojson"


def _planning_bits():
    cfg = load_config(config_path())
    spec = build_spec(cfg)
    return spec, EnergyModel(spec), make_motion_model(spec)


def test_coverage_energy_breakdown_is_consistent():
    spec, em, motion = _planning_bits()
    poly = normalize_to_area(shape_builders(128)["square"], 1_000_000.0)
    cov = coverage_energy(poly, spec, em, motion, sensor_power_w=0.0)

    assert cov["strip_j"] > 0.0
    assert cov["connector_j"] > 0.0
    assert cov["sensor_j"] == 0.0                       # camera off
    assert cov["n_strips"] >= 1
    # boustrophedon: exactly one fewer connector than strips.
    assert cov["n_connectors"] == cov["n_strips"] - 1
    # total is the sum of its parts.
    assert cov["coverage_total_j"] == pytest.approx(
        cov["strip_j"] + cov["connector_j"] + cov["sensor_j"]
    )


def test_camera_payload_only_adds_energy_on_strips():
    spec, em, motion = _planning_bits()
    poly = normalize_to_area(shape_builders(128)["square"], 1_000_000.0)
    off = coverage_energy(poly, spec, em, motion, sensor_power_w=0.0)
    on = coverage_energy(poly, spec, em, motion, sensor_power_w=15.0)

    # propulsion is identical; only the sensor term changes.
    assert on["strip_j"] == pytest.approx(off["strip_j"])
    assert on["connector_j"] == pytest.approx(off["connector_j"])
    assert on["sensor_j"] > 0.0
    assert on["coverage_total_j"] > off["coverage_total_j"]
    assert on["coverage_total_j"] - off["coverage_total_j"] == pytest.approx(on["sensor_j"])


def test_classify_thresholds():
    assert classify(2.0) == "BATTERY-LIMITED"
    assert classify(0.5) == "FUEL-SURPLUS"
    assert classify(1.0) == "BORDERLINE"
    # the band edges resolve to BORDERLINE, just outside flips.
    assert classify(1.10) == "BORDERLINE"
    assert classify(0.90) == "BORDERLINE"
    assert classify(1.11) == "BATTERY-LIMITED"
    assert classify(0.89) == "FUEL-SURPLUS"


def test_usable_fractions_ordered():
    cfg = load_config(config_path())
    term, _ = _usable_fraction(cfg, "terminal")
    ret, _ = _usable_fraction(cfg, "return")
    rth, _ = _usable_fraction(cfg, "rth")
    # rth (only the small RTH reserve carved out) is the largest usable fraction;
    # return (down to nominal) the smallest.
    assert rth > term > ret
    for f in (term, ret, rth):
        assert 0.0 < f < 1.0
    with pytest.raises(ValueError):
        _usable_fraction(cfg, "bogus")


def test_positive_int_rejects_nonpositive():
    import argparse

    assert _positive_int("3") == 3
    for bad in ("0", "-1", "-5"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)


def test_per_zone_energy_is_assignment_aware():
    # build the real planning layer on the small area and partition it for a
    # 2-drone fleet: every drone gets a zone, energies are positive and sorted
    # busiest-first, and the zones tile (roughly) the whole survey area.
    cfg = load_config(config_path())
    env, tgc, area, spec, em, motion, base = _build_planning_layer(
        cfg, "data/areas/smoke_area.geojson", n_drones=2)
    rows = per_zone_energy(env, tgc, spec, em, motion, base, 2,
                           sensor_power_w=0.0, altitude_m=cfg.env.coverage_altitude_m)
    assert len(rows) == 2
    for r in rows:
        assert r["e_zone_j"] > 0.0
        assert r["e_zone_j"] == pytest.approx(
            r["core_j"] + r["transit_j"] + r["vertical_j"])
    # sorted busiest-first.
    assert rows[0]["e_zone_j"] >= rows[-1]["e_zone_j"]
    # partition covers the survey area (allow slack for TGC region granularity).
    covered = sum(r["area_m2"] for r in rows)
    assert covered == pytest.approx(area.area, rel=0.05)


def test_analytical_coverage_matches_engine_swap_free():
    # small area, one drone: swap-free, so no re-flown legs inflate the engine
    # number. Analytical coverage energy must match the engine's drained coverage
    # phase within the dt-quantisation tolerance.
    result, engine_j, analytical_j, n_swaps, dt = _run_once(
        config_path(), _SMOKE, n_drones=1, sensor_power_w=0.0
    )
    assert result.outcome.value == "MISSION_SUCCESS"
    assert n_swaps == 0
    assert engine_j > 0.0
    rel_err = abs(analytical_j - engine_j) / engine_j
    assert rel_err < _VERIFY_TOL, f"rel_err {rel_err:.3e} >= {_VERIFY_TOL}"
