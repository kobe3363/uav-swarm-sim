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


# --------------------------------------------------------------------------- #
# config: EM-01 rth.energy_map (Stage 1, optional-key gating)                  #
# --------------------------------------------------------------------------- #
def test_energy_map_defaults_off_and_absent_from_default_yaml(config_path):
    """The rth.energy_map block is deliberately ABSENT from default.yaml, so
    the provenance hash and every existing fixture stay unchanged (the same
    optional-key rule as telemetry / obstacle_recovery / stall_detector)."""
    import yaml

    raw = yaml.safe_load(config_path.read_text())
    assert "energy_map" not in raw["rth"]

    emc = load_config(config_path).rth.energy_map
    assert emc.enabled is False
    assert emc.cell_m is None          # None => battery-tied derived
    assert emc.yellow_penalty == 1.5
    assert emc.red_threshold == 0.5


def test_energy_map_overrides(config_path):
    cfg = load_config(config_path, overrides={
        "rth.energy_map.enabled": True,
        "rth.energy_map.cell_m": 25.0,
        "rth.energy_map.yellow_penalty": 2.0,
        "rth.energy_map.red_threshold": 0.4,
    })
    emc = cfg.rth.energy_map
    assert emc.enabled is True
    assert emc.cell_m == 25.0
    assert emc.yellow_penalty == 2.0
    assert emc.red_threshold == 0.4


@pytest.mark.parametrize("key, bad", [
    ("rth.energy_map.cell_m", -1.0),
    ("rth.energy_map.cell_m", 0.0),
    ("rth.energy_map.cell_m", float("nan")),
    ("rth.energy_map.cell_m", float("inf")),
    ("rth.energy_map.yellow_penalty", 0.5),
    ("rth.energy_map.yellow_penalty", float("nan")),
    ("rth.energy_map.yellow_penalty", float("inf")),
    ("rth.energy_map.red_threshold", 0.0),
    ("rth.energy_map.red_threshold", 1.5),
    ("rth.energy_map.red_threshold", float("nan")),
])
def test_energy_map_rejects_invalid_values(config_path, key, bad):
    with pytest.raises(ConfigError, match="energy_map"):
        load_config(config_path, overrides={key: bad})


def test_energy_map_rejects_non_mapping(config_path):
    """A scalar/list ``rth.energy_map`` must raise ConfigError, not the
    AttributeError that ``.get`` on a non-dict would otherwise throw."""
    for bad in (True, [1, 2], "on"):
        with pytest.raises(ConfigError, match="energy_map must be a mapping"):
            load_config(config_path, overrides={"rth.energy_map": bad})


# --------------------------------------------------------------------------- #
# config: EM-01 rth.energy_map.decide (Stage 2 sub-flag)                       #
# --------------------------------------------------------------------------- #
def test_energy_map_decide_defaults_off_and_absent(config_path):
    """Same optional-key rule as Stage 1: ``decide`` defaults False and the
    whole block stays absent from default.yaml (hash/fixtures unchanged)."""
    import yaml

    raw = yaml.safe_load(config_path.read_text())
    assert "energy_map" not in raw["rth"]
    assert load_config(config_path).rth.energy_map.decide is False


def test_energy_map_decide_override_parses(config_path):
    cfg = load_config(config_path, overrides={
        "rth.energy_map.enabled": True,
        "rth.energy_map.decide": True,
    })
    assert cfg.rth.energy_map.enabled is True
    assert cfg.rth.energy_map.decide is True


def test_energy_map_decide_requires_enabled(config_path):
    """decide=True without enabled=True is a contradiction (the map would not
    be built) and must fail loudly at load time."""
    with pytest.raises(ConfigError, match="decide"):
        load_config(config_path, overrides={"rth.energy_map.decide": True})


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_operating_margin_rejects_non_finite_and_negative(config_path, bad):
    """NaN passes a bare < 0 check; the margin feeds flyable_region buffering
    and the Stage-2 energy-map extent, so it must die here as ConfigError."""
    with pytest.raises(ConfigError, match="operating_margin_m"):
        load_config(config_path, overrides={"coverage.operating_margin_m": bad})
