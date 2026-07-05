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
