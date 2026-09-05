"""physical_model/drone_specs tests (isolated)."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.physical_model.drone_specs import build_spec


def _spec(config_path, platform="MULTIROTOR"):
    cfg = load_config(config_path, overrides={"platform_type": platform})
    return build_spec(cfg), cfg


def test_effective_swath_and_capacity(config_path):
    spec, cfg = _spec(config_path)
    assert spec.swath_width_m == pytest.approx(
        cfg.sensor.swath_width_m * (1 - cfg.sensor.overlap_frac)
    )
    assert spec.battery_capacity_j == pytest.approx(cfg.fleet.battery_capacity_j)


def test_enabled_camera_replaces_legacy_swath_and_exposes_photo_spacing(config_path):
    cfg = load_config(config_path, overrides={
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 17.3,
        "sensor.photogrammetry.sensor_height_mm": 13.0,
        "sensor.photogrammetry.focal_length_mm": 12.0,
        "sensor.photogrammetry.image_width_px": 5280,
        "sensor.photogrammetry.image_height_px": 3956,
        "sensor.photogrammetry.side_overlap": 0.70,
        "sensor.photogrammetry.forward_overlap": 0.80,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "env.coverage_altitude_m": 100.0,
        "platforms.MULTIROTOR.v_coverage": 10.0,
    })
    spec = build_spec(cfg)
    assert spec.swath_width_m == pytest.approx(43.25)
    assert spec.coverage_line_spacing_m(50.0) == pytest.approx(21.625)
    assert spec.coverage_photo_spacing_m() == pytest.approx(21.666666666666664)


def test_disabled_camera_keeps_exact_legacy_swath(config_path):
    legacy = build_spec(load_config(config_path))
    explicit_off = build_spec(load_config(config_path, overrides={
        "sensor.photogrammetry.enabled": False,
        "sensor.photogrammetry.sensor_width_mm": 999.0,
        "sensor.photogrammetry.forward_overlap": 0.99,
    }))
    assert explicit_off.swath_width_m == legacy.swath_width_m
    assert explicit_off.coverage_photo_spacing_m() is None
