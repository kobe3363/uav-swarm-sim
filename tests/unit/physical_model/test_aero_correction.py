"""physical_model/aero_correction tests (isolated)."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection


def test_power_factor_rules(config_path):
    cfg = load_config(config_path)
    red = cfg.aero.formation_drag_reduction
    fw = AeroCorrection(cfg.aero, PlatformType.FIXED_WING)
    mr = AeroCorrection(cfg.aero, PlatformType.MULTIROTOR)
    assert fw.power_factor(True, ManeuverType.CRUISE) == pytest.approx(1 - red)
    assert fw.power_factor(True, ManeuverType.COVERAGE) == 1.0
    assert fw.power_factor(False, ManeuverType.CRUISE) == 1.0
    assert mr.power_factor(True, ManeuverType.CRUISE) == 1.0


def test_wake_zones_geometry(config_path):
    cfg = load_config(config_path)
    mr = AeroCorrection(cfg.aero, PlatformType.MULTIROTOR)
    zones = mr.wake_zones([Pose(0, 0, 0.0)])
    assert len(zones) == 1
    z = zones[0]
    assert z.is_valid and z.area > 0
    # wake extends behind (negative x) for heading 0
    minx, _, maxx, _ = z.bounds
    assert minx < 0 <= maxx + 1e-9
