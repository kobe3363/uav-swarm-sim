"""Validation that config/djimatrice4e.yaml loads to the expected DJI Matrice 4E
(M4E) spec.

All HARDWARE values trace to the DJI M4E spec page (live-verified 2026-08-07):
    https://enterprise.dji.com/matrice-4-series/specs
The power table (derived from hover power via the shipped ratio model) and
v_coverage (an operational assumption) are provenance-marked in the config and
are asserted here only for their derived values, not against a datasheet.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
from uav_swarm_sim.physical_model.drone_specs import build_spec

M4E = "config/djimatrice4e.yaml"


def test_m4e_config_loads_as_multirotor():
    cfg = load_config(M4E)
    assert cfg.platform.type is PlatformType.MULTIROTOR


def test_m4e_battery_mass_and_dims():
    cfg = load_config(M4E)
    assert cfg.fleet.battery_capacity_wh == 99.5                 # DJI: 99.5 Wh
    assert cfg.fleet.battery_capacity_j == pytest.approx(99.5 * 3600.0)
    assert cfg.platform.mass_kg == 1.219                         # DJI: 1219 g (standard props)
    assert cfg.fleet.drone_dims_m == (0.307, 0.3875, 0.1495)     # DJI: 307.0x387.5x149.5 mm unfolded


def test_m4e_speeds():
    cfg = load_config(M4E)
    assert cfg.platform.v_cruise == 21.0        # DJI max horizontal 21 m/s
    assert cfg.platform.v_coverage == 15.0      # ASSUMPTION: mapping survey speed
    assert cfg.platform.v_climb == 10.0         # DJI max ascent 10 m/s
    assert cfg.platform.v_descent == 8.0        # DJI max descent 8 m/s


def test_m4e_power_table_complete_and_hover_derived():
    cfg = load_config(M4E)
    for m in ManeuverType:                       # all 9 maneuvers present (else load rejects it)
        assert m in cfg.platform.power_w
    # HOVER = 99.5 Wh x 3600 / (42 min x 60 s) = 142.1 W (derived anchor)
    assert cfg.platform.power_w[ManeuverType.HOVER] == pytest.approx(142.1)
    assert all(v >= 0.0 for v in cfg.platform.power_w.values())


def test_m4e_effective_swath_and_altitude():
    cfg = load_config(M4E)
    assert cfg.sensor.swath_width_m == 132.0     # RAW footprint (2.5 cm GSD x 5280 px)
    assert cfg.sensor.overlap_frac == 0.6
    # build_spec applies (1 - overlap): 132.0 x 0.4 = 52.8 m effective strip
    assert build_spec(cfg).swath_width_m == pytest.approx(52.8)
    assert cfg.env.coverage_altitude_m == 91.5   # DERIVED: 2.5 cm GSD, M4E wide cam


def test_m4e_obstacles_are_fixed_squares():
    cfg = load_config(M4E)
    assert cfg.env.obstacle_shapes == ("square",)
    assert cfg.env.obstacle_size_range_m == (52.8, 52.8)   # side == effective swath (1 strip)
