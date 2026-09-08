"""Validation that config/djimatrice4e.yaml loads to the M4E PROTOCOL config (EXP-13).

Provenance discipline (AC-1): HARDWARE values trace to the DJI M4E spec page
(live-verified 2026-08-07, https://enterprise.dji.com/matrice-4-series/specs);
OPERATIONAL choices (survey speed, camera payload draw) are ASSUMPTIONs and are
asserted as such, kept separable from the spec ceilings. DERIVED geometry
(footprint / GSD / line_spacing) is asserted against the OPTICS FORMULA, not a
pinned literal, so the test cannot drift from photogrammetry.solve.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
from uav_swarm_sim.physical_model.drone_specs import build_spec

M4E = "config/djimatrice4e.yaml"

# DJI M4E hardware spec ceilings (NOT the operational protocol values below).
DJI_MAX_HORIZONTAL_SPEED_M_S = 21.0   # spec ceiling; the protocol flies slower
LEGACY_EFFECTIVE_SWATH_M = 52.8       # 132.0 * (1 - 0.6): the DISABLED legacy path


def test_m4e_config_loads_as_multirotor():
    cfg = load_config(M4E)
    assert cfg.platform.type is PlatformType.MULTIROTOR


def test_m4e_battery_mass_and_dims():
    cfg = load_config(M4E)
    assert cfg.fleet.battery_capacity_wh == 99.5                 # DJI: 99.5 Wh
    assert cfg.fleet.battery_capacity_j == pytest.approx(99.5 * 3600.0)
    assert cfg.platform.mass_kg == 1.219                         # DJI: 1219 g (standard props)
    assert cfg.fleet.drone_dims_m == (0.307, 0.3875, 0.1495)     # DJI: 307.0x387.5x149.5 mm unfolded


def test_m4e_speeds_are_operational_choice_below_the_spec_ceiling():
    """AC-1: the operational choice (10.0) is asserted AND the DJI spec ceiling
    (21.0) is documented separately, so hardware origin stays visible and is not
    silently overwritten by the operational value."""
    cfg = load_config(M4E)
    assert cfg.platform.v_cruise == 10.0        # PROTOCOL operational choice
    assert cfg.platform.v_coverage == 10.0      # PROTOCOL: v_coverage = v_cruise = 10.0
    # hardware origin remains separable from the operational choice:
    assert cfg.platform.v_cruise < DJI_MAX_HORIZONTAL_SPEED_M_S
    assert cfg.platform.v_climb == 10.0         # DJI max ascent 10 m/s
    assert cfg.platform.v_descent == 8.0        # DJI max descent 8 m/s


def test_m4e_power_table_complete_and_hover_derived():
    cfg = load_config(M4E)
    for m in ManeuverType:                       # all 9 maneuvers present (else load rejects it)
        assert m in cfg.platform.power_w
    # HOVER = 99.5 Wh x 3600 / (42 min x 60 s) = 142.1 W (derived anchor)
    assert cfg.platform.power_w[ManeuverType.HOVER] == pytest.approx(142.1)
    assert all(v >= 0.0 for v in cfg.platform.power_w.values())


def test_m4e_photogrammetry_enabled_and_optics():
    cfg = load_config(M4E)
    pg = cfg.sensor.photogrammetry
    assert pg.enabled is True                     # EXP-13 protocol: camera ON
    assert (pg.sensor_width_mm, pg.sensor_height_mm) == (17.3, 13.0)  # model assumption
    assert pg.focal_length_mm == 12.0             # model assumption, not DJI calibration
    assert (pg.image_width_px, pg.image_height_px) == (5280, 3956)    # DJI max image
    assert (pg.side_overlap, pg.forward_overlap) == (0.70, 0.80)
    assert pg.min_photo_interval_s == 0.5         # DJI minimum for 20 MP JPEG
    assert cfg.env.coverage_altitude_m == 100.0   # PROTOCOL AGL


def test_m4e_camera_geometry_is_formula_derived_and_governs_effective_swath():
    """DERIVED geometry asserted against the optics formula (not a literal), and
    the camera path — NOT the legacy 132/0.6 swath — governs build_spec."""
    cfg = load_config(M4E)
    pg = cfg.sensor.photogrammetry
    h = cfg.env.coverage_altitude_m
    # optics formula (photogrammetry.py:47-50)
    footprint_width = h * pg.sensor_width_mm / pg.focal_length_mm
    footprint_length = h * pg.sensor_height_mm / pg.focal_length_mm
    line_spacing = footprint_width * (1.0 - pg.side_overlap)
    photo_spacing = footprint_length * (1.0 - pg.forward_overlap)
    gsd_width = footprint_width / pg.image_width_px
    nominal_interval = photo_spacing / cfg.platform.v_coverage

    sol = build_spec(cfg).photogrammetry_at(h)
    assert sol.footprint_width_m == pytest.approx(footprint_width)
    assert sol.line_spacing_m == pytest.approx(line_spacing)
    assert sol.photo_spacing_m == pytest.approx(photo_spacing)
    assert sol.gsd_width_m_px == pytest.approx(gsd_width)
    # shutter feasibility: nominal interval >= DJI minimum (config accepted it)
    assert nominal_interval + 1e-12 >= pg.min_photo_interval_s

    # the camera path governs the effective swath (line_spacing), NOT legacy 52.8
    assert build_spec(cfg).swath_width_m == pytest.approx(line_spacing)
    assert build_spec(cfg).swath_width_m != pytest.approx(LEGACY_EFFECTIVE_SWATH_M)


def test_m4e_legacy_swath_fields_are_inactive_but_present():
    """The legacy raw-footprint fields remain (DJI provenance) but do not govern
    geometry while photogrammetry is enabled."""
    cfg = load_config(M4E)
    assert cfg.sensor.swath_width_m == 132.0
    assert cfg.sensor.overlap_frac == 0.6
    # legacy fallback value, deliberately NOT what build_spec uses here
    assert cfg.sensor.swath_width_m * (1.0 - cfg.sensor.overlap_frac) == pytest.approx(
        LEGACY_EFFECTIVE_SWATH_M)


def test_m4e_camera_payload_draw_is_a_marked_assumption():
    """#6: sensor_power_w is an ASSUMPTION (not DJI spec), non-zero so the camera
    energy term is actually charged over COVERAGE."""
    cfg = load_config(M4E)
    assert cfg.sensor.sensor_power_w == 20.0
    assert cfg.sensor.sensor_power_w > 0.0        # camera-energy branch is live


def test_m4e_obstacle_target_mode_and_area_derived_side():
    """Target mode: side is area-derived, so obstacle_size_range_m/density are
    ignored (the legacy 'obstacle == 1 strip' invariant no longer holds)."""
    import math

    cfg = load_config(M4E)
    assert cfg.env.obstacle_generation_mode == "target"
    assert cfg.env.obstacle_target_count == 10
    assert cfg.env.obstacle_area_fraction == 0.05
    assert cfg.env.obstacle_area_fraction_tolerance == 0.005
    assert cfg.env.obstacle_shapes == ("square",)
    # side derived from the survey area (obstacle_generator.py:130 side = sqrt(area *
    # frac / count)), NOT from obstacle_size_range_m. On the 1000x750 protocol area:
    survey_area_m2 = 1000.0 * 750.0
    side = math.sqrt(survey_area_m2 * cfg.env.obstacle_area_fraction / cfg.env.obstacle_target_count)
    assert side == pytest.approx(math.sqrt(3750.0))                 # 61.24 m
    assert side != pytest.approx(cfg.env.obstacle_size_range_m[0])  # NOT the ignored 52.8 field


def test_m4e_protocol_scenario_flags():
    """Every new protocol scalar has an asserting test (AC-1)."""
    cfg = load_config(M4E)
    # EXP-04 no-swap + EXP-08 reallocation OFF (author decision §4.1)
    assert cfg.mission.no_swap_mode is True
    assert cfg.mission.repartition_enabled is False
    # EXP-07 experiment mode + EXP-11 contract export
    assert cfg.mission.experiment_mode is True
    assert cfg.mission.contract_export is True
    # EXP-02 raster + EXP-09 free-space routing
    assert cfg.coverage.raster_enabled is True
    assert cfg.coverage.transit_free_space is True
    assert cfg.coverage.ferry_free_space is True
    # EXP-06 energy balance
    assert cfg.planning.energy_balance.enabled is True
    # EXP-09 coherent routing + AC-4 zone demotion, with NO energy map built
    assert cfg.rth.execution_coherent is True
    assert cfg.rth.energy_map.zone_demotion is True
    assert cfg.rth.energy_map.enabled is False
    assert cfg.rth.emergency_frac is None
    # EXP-10 safety recording
    assert cfg.safety.record_violations is True
    # clean paired scenario
    assert cfg.failure.hazard_rate_per_hour == 0.0
    # AC-1: initial state of charge is the resolved default (block not written)
    assert cfg.battery.initial_soc.mode == "fixed"
    assert cfg.battery.initial_soc.value == 1.0


def test_m4e_total_reserve_batteries_inert_under_no_swap():
    """The field is retained but inert; the value itself is not load-bearing."""
    cfg = load_config(M4E)
    assert cfg.mission.no_swap_mode is True        # ... which makes the pool inert
    assert cfg.fleet.total_reserve_batteries == 50


def test_m4e_config_satisfies_lloyd_protocol_fail_fast():
    """The protocol config must pass run_lloyd_protocol's fail-fast (else the
    smoke command cannot even start)."""
    from uav_swarm_sim.experiments.run_lloyd_protocol import validate_config
    validate_config(load_config(M4E))   # raises SystemExit on failure
