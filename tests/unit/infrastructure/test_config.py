"""infrastructure/config tests (isolated)."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import (
    Config,
    ConfigError,
    load_config,
)
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType


# --------------------------------------------------------------------------- #
# config: loading                                                             #
# --------------------------------------------------------------------------- #
def test_default_loads_and_is_frozen(config_path):
    cfg = load_config(config_path)
    assert isinstance(cfg, Config)
    with pytest.raises(Exception):
        cfg.fleet.n_drones = 5  # frozen dataclass


def test_wh_to_joules_conversion(config_path):
    cfg = load_config(config_path)
    assert cfg.fleet.battery_capacity_j == pytest.approx(cfg.fleet.battery_capacity_wh * 3600.0)


def test_active_platform_resolved_and_power_table_complete(config_path):
    cfg = load_config(config_path)
    assert cfg.platform.type is PlatformType.MULTIROTOR
    # every maneuver has a coefficient
    for m in ManeuverType:
        assert m in cfg.platform.power_w
    # multirotor is holonomic -> r_min may be 0
    assert cfg.platform.r_min_m == 0.0


def test_platform_override_resolves_other_table(config_path):
    cfg = load_config(config_path, overrides={"platform_type": "FIXED_WING"})
    assert cfg.platform.type is PlatformType.FIXED_WING
    assert cfg.platform.r_min_m > 0.0
    assert cfg.platform.climb_angle_rad > 0.0  # degrees -> radians happened


def test_candidate_sites_int_default(config_path):
    cfg = load_config(config_path)
    assert isinstance(cfg.launch.candidate_sites, int)
    assert cfg.launch.candidate_sites == 8


def test_battery_zone_defaults(config_path):
    cfg = load_config(config_path)
    assert (cfg.battery_zones.high, cfg.battery_zones.nominal, cfg.battery_zones.critical) == (
        0.75, 0.40, 0.20,
    )


# --------------------------------------------------------------------------- #
# config: validation                                                          #
# --------------------------------------------------------------------------- #
def test_reject_zero_drones(config_path):
    with pytest.raises(ConfigError, match="n_drones"):
        load_config(config_path, overrides={"fleet.n_drones": 0})


def test_reject_out_of_order_battery_zones(config_path):
    with pytest.raises(ConfigError, match="battery_zones"):
        load_config(config_path, overrides={"battery_zones.nominal": 0.80})


def test_reject_fixed_wing_with_zero_turn_radius(config_path):
    with pytest.raises(ConfigError, match="r_min_m"):
        load_config(
            config_path,
            overrides={"platform_type": "FIXED_WING", "platforms.FIXED_WING.r_min_m": 0.0},
        )


def test_reject_bad_platform_type(config_path):
    with pytest.raises(ConfigError, match="platform_type"):
        load_config(config_path, overrides={"platform_type": "BLIMP"})


def test_reject_drag_reduction_out_of_range(config_path):
    with pytest.raises(ConfigError, match="formation_drag_reduction"):
        load_config(config_path, overrides={"aero.formation_drag_reduction": 1.5})


def test_reject_negative_hazard_rate(config_path):
    with pytest.raises(ConfigError, match="hazard_rate"):
        load_config(config_path, overrides={"failure.hazard_rate_per_hour": -1.0})


def test_reject_bad_tier_thresholds(config_path):
    with pytest.raises(ConfigError, match="tier_thresholds"):
        load_config(config_path, overrides={"tier_thresholds": [50, 15]})


# --------------------------------------------------------------------------- #
# config: provenance hash                                                      #
# --------------------------------------------------------------------------- #
def test_hash_is_deterministic(config_path):
    a = load_config(config_path)
    b = load_config(config_path)
    assert a.config_hash == b.config_hash
    assert len(a.config_hash) == 64  # sha256 hex


def test_hash_changes_with_override(config_path):
    a = load_config(config_path)
    b = load_config(config_path, overrides={"fleet.n_drones": 30})
    assert a.config_hash != b.config_hash
