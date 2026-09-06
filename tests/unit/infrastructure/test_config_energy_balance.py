"""Strict, default-off EXP-06 configuration and unchanged raw-YAML provenance."""
from pathlib import Path

import pytest

from uav_swarm_sim.infrastructure.config import ConfigError, load_config


@pytest.mark.parametrize("path", sorted(Path("config").glob("*.yaml")))
def test_every_yaml_defaults_off(path):
    assert load_config(path).planning.energy_balance.enabled is False


@pytest.mark.parametrize("block", [{}, {"energy_balance": {}}, None])
def test_optional_planning(block):
    assert load_config("config/default.yaml", {"planning": block}).planning.energy_balance.enabled is False


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_enabled_requires_boolean(value):
    with pytest.raises(ConfigError, match="planning.energy_balance.enabled must be a boolean"):
        load_config("config/default.yaml", {"planning.energy_balance.enabled": value})


@pytest.mark.parametrize("key,value,message", [
    ("planning", [], "planning must be a mapping"),
    ("planning.energy_balance", False, "planning.energy_balance must be a mapping"),
    ("planning.typo", True, "planning has unknown"),
    ("planning.energy_balance.typo", True, "planning.energy_balance has unknown"),
])
def test_mapping_and_unknown_keys(key, value, message):
    with pytest.raises(ConfigError, match=message):
        load_config("config/default.yaml", {key: value})


@pytest.mark.parametrize("raster,photo", [(False, False), (False, True), (True, False)])
def test_enabled_requires_entire_chain(raster, photo):
    with pytest.raises(ConfigError, match="enabled requires coverage.raster_enabled.*sensor.photogrammetry.enabled"):
        load_config("config/default.yaml", {
            "planning.energy_balance.enabled": True,
            "coverage.raster_enabled": raster,
            "sensor.photogrammetry.enabled": photo,
        })


def test_parse_enabled_and_reject_target_mission():
    overrides = {
        "planning.energy_balance.enabled": True,
        "coverage.raster_enabled": True,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 17.3,
        "sensor.photogrammetry.sensor_height_mm": 13.0,
        "sensor.photogrammetry.focal_length_mm": 12.0,
        "sensor.photogrammetry.image_width_px": 5280,
        "sensor.photogrammetry.image_height_px": 3956,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
    }
    assert load_config("config/default.yaml", overrides).planning.energy_balance.enabled is True
    with pytest.raises(ConfigError, match="requires mission.type = coverage"):
        load_config("config/default.yaml", dict(overrides, **{"mission.type": "target_visit"}))


def test_absent_block_preserves_hash():
    # Raw YAML is unchanged; constructing default dataclasses cannot enter its hash.
    baseline = load_config("config/default.yaml")
    assert baseline.config_hash == load_config("config/default.yaml", {}).config_hash
