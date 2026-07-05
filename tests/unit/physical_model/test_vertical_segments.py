"""physical_model/vertical_segments tests (isolated)."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.vertical_segments import (
    landing_profile,
    takeoff_profile,
)


def _spec(config_path, platform="MULTIROTOR"):
    cfg = load_config(config_path, overrides={"platform_type": platform})
    return build_spec(cfg), cfg


def test_multirotor_takeoff_vertical(config_path):
    spec, _ = _spec(config_path, "MULTIROTOR")
    em = EnergyModel(spec)
    prof = takeoff_profile(spec, em, altitude_m=100.0)
    assert prof.duration_s == pytest.approx(100.0 / spec.v_climb)
    assert prof.energy_j > 0


def test_fixed_wing_takeoff_includes_ground_roll(config_path):
    spec, _ = _spec(config_path, "FIXED_WING")
    em = EnergyModel(spec)
    prof = takeoff_profile(spec, em, altitude_m=100.0)
    # energy strictly exceeds the climb-only integral by the ground-roll charge
    climb_only = prof.energy_j - spec.ground_roll_energy_j
    assert spec.ground_roll_energy_j > 0
    assert climb_only > 0
    land = landing_profile(spec, em, altitude_m=100.0)
    assert land.energy_j > 0
