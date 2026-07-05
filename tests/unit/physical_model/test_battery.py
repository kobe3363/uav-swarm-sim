"""physical_model/battery tests (isolated)."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import BatteryZone
from uav_swarm_sim.physical_model.battery import Battery


@pytest.fixture(scope="module")
def cfg(config_path):
    return load_config(config_path)


def test_battery_zones(cfg):
    b = Battery(100.0, cfg.battery_zones, 1.0)
    assert b.zone is BatteryZone.HIGH
    b.drain(30)  # 0.70
    assert b.zone is BatteryZone.NOMINAL
    b.drain(35)  # 0.35
    assert b.zone is BatteryZone.CRITICAL
    b.drain(20)  # 0.15
    assert b.zone is BatteryZone.TERMINAL
    b.drain(1000)  # clamps
    assert b.level_j == 0.0
    b.reset()
    assert b.frac == 1.0
