"""EXP-08: mission.repartition_enabled / repartition_interval_s parsing + validation.

The flag switches the redistribution policy AND the decomposer identity used for
every re-partition, so a mis-typed value must never be quietly coerced (the same
strict-boolean rule EXP-04 and EXP-07 apply to their gates).
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import ConfigError, load_config

# The EXP-02 raster is a hard prerequisite (D-8), so every "on" fixture needs it.
_RASTER_ON = {
    "coverage.raster_enabled": True,
    "sensor.photogrammetry.enabled": True,
    "sensor.photogrammetry.sensor_width_mm": 17.3,
    "sensor.photogrammetry.sensor_height_mm": 13.0,
    "sensor.photogrammetry.focal_length_mm": 12.0,
    "sensor.photogrammetry.image_width_px": 5280,
    "sensor.photogrammetry.image_height_px": 3956,
    "sensor.photogrammetry.side_overlap": 0.70,
    "sensor.photogrammetry.forward_overlap": 0.80,
    "sensor.photogrammetry.min_photo_interval_s": 0.5,
}


def _on(config_path, **extra):
    return load_config(config_path, overrides={
        **_RASTER_ON, "mission.repartition_enabled": True, **extra,
    })


# --------------------------------------------------------------------------- #
# defaults                                                                     #
# --------------------------------------------------------------------------- #
def test_both_knobs_default_off(config_path):
    mission = load_config(config_path).mission
    assert mission.repartition_enabled is False
    assert mission.repartition_interval_s is None


def test_the_new_keys_are_absent_from_every_shipped_yaml():
    """config_hash is taken over the RAW yaml, so these defaults must live in
    code. A repartition key in a shipped config would move every pinned hash and
    every cross-commit golden."""
    import pathlib

    for path in sorted(pathlib.Path("config").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "repartition_enabled" not in text, path
        assert "repartition_interval_s" not in text, path


# --------------------------------------------------------------------------- #
# strict typing                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["true", "yes", 1, 0, None])
def test_repartition_enabled_refuses_a_non_boolean(config_path, bad):
    with pytest.raises(ConfigError, match="repartition_enabled must be a boolean"):
        load_config(config_path, overrides={"mission.repartition_enabled": bad})


@pytest.mark.parametrize("bad", ["30", True, [30]])
def test_interval_refuses_a_non_number(config_path, bad):
    with pytest.raises(ConfigError, match="repartition_interval_s must be a number"):
        _on(config_path, **{"mission.repartition_interval_s": bad})


# --------------------------------------------------------------------------- #
# prerequisites (D-8)                                                          #
# --------------------------------------------------------------------------- #
def test_repartition_requires_the_coverage_raster_it_partitions(config_path):
    with pytest.raises(ConfigError, match="requires coverage.raster_enabled"):
        load_config(config_path, overrides={"mission.repartition_enabled": True})


def test_repartition_requires_an_area_coverage_mission(config_path):
    with pytest.raises(ConfigError, match="requires mission.type = coverage"):
        load_config(config_path, overrides={
            **_RASTER_ON,
            "mission.repartition_enabled": True,
            "mission.type": "target_visit",
        })


def test_interval_alone_is_refused(config_path):
    """A configured cadence with the feature off would be silently inert."""
    with pytest.raises(ConfigError, match="requires mission.repartition_enabled"):
        load_config(config_path, overrides={
            **_RASTER_ON, "mission.repartition_interval_s": 60.0,
        })


# --------------------------------------------------------------------------- #
# the interval is a whole number of ticks, never silently rounded              #
# --------------------------------------------------------------------------- #
def test_a_whole_multiple_of_dt_is_accepted(config_path):
    cfg = _on(config_path, **{"sim.dt_s": 0.5, "mission.repartition_interval_s": 60.0})
    assert cfg.mission.repartition_interval_s == pytest.approx(60.0)
    # 60 s / 0.5 s = 120 ticks exactly -- computed here, not read back from the loader
    assert round(cfg.mission.repartition_interval_s / cfg.sim.dt_s) == 120


@pytest.mark.parametrize("dt,interval", [(0.5, 60.25), (1.0, 0.5), (0.5, 0.25)])
def test_an_interval_that_is_not_a_whole_tick_count_is_refused(config_path, dt, interval):
    with pytest.raises(ConfigError, match="whole"):
        _on(config_path, **{"sim.dt_s": dt, "mission.repartition_interval_s": interval})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_nonpositive_or_non_finite_interval_is_refused(config_path, bad):
    with pytest.raises(ConfigError):
        _on(config_path, **{"mission.repartition_interval_s": bad})


@pytest.mark.parametrize("bad_dt", [0.0, -0.5, float("nan"), float("inf")])
def test_a_broken_timestep_is_a_config_error_not_an_arithmetic_crash(config_path, bad_dt):
    """The interval check divides by sim.dt_s, and the general "dt_s > 0" rule
    runs AFTER it. Without a local guard, dt_s = 0 surfaces as ZeroDivisionError
    and dt_s = nan as ValueError from round() -- both escaping the ConfigError
    contract every other malformed field follows. (Found in review of this PR.)
    """
    with pytest.raises(ConfigError):
        _on(config_path, **{"sim.dt_s": bad_dt,
                            "mission.repartition_interval_s": 60.0})
