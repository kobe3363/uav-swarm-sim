"""physical_model/energy_model tests (isolated): the E = sum P*dt identity
and the formation invariant."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import (
    Path,
    Pose,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel


def _spec(config_path, platform="MULTIROTOR"):
    cfg = load_config(config_path, overrides={"platform_type": platform})
    return build_spec(cfg), cfg


def test_segment_energy_is_power_times_time(config_path):
    spec, _ = _spec(config_path, "MULTIROTOR")
    em = EnergyModel(spec)
    p = em.power(ManeuverType.CRUISE)
    assert em.segment_energy(ManeuverType.CRUISE, 10.0) == pytest.approx(p * 10.0)


def test_path_energy_equals_manual_integral(config_path):
    spec, _ = _spec(config_path, "MULTIROTOR")
    em = EnergyModel(spec)
    s1 = straight_segment(Pose(0, 0, 0), 10.0, ManeuverType.CRUISE, 5.0)
    s2 = straight_segment(s1.end, 6.0, ManeuverType.COVERAGE, 3.0)
    path = Path.from_segments([s1, s2])
    manual = (
        em.power(ManeuverType.CRUISE) * s1.duration_s
        + em.power(ManeuverType.COVERAGE) * s2.duration_s
    )
    assert em.path_energy(path) == pytest.approx(manual)


def test_energy_is_time_integral_not_distance_average(config_path):
    # same distance, different speed -> different energy (proves time-integration)
    spec, _ = _spec(config_path, "MULTIROTOR")
    em = EnergyModel(spec)
    slow = straight_segment(Pose(0, 0, 0), 10.0, ManeuverType.CRUISE, 2.0)  # 5 s
    fast = straight_segment(Pose(0, 0, 0), 10.0, ManeuverType.CRUISE, 10.0)  # 1 s
    e_slow = em.path_energy(Path.from_segments([slow]))
    e_fast = em.path_energy(Path.from_segments([fast]))
    assert e_slow > e_fast


def test_formation_factor_applies_to_cruise_fw_only(config_path):
    fw = EnergyModel(_spec(config_path, "FIXED_WING")[0])
    mr = EnergyModel(_spec(config_path, "MULTIROTOR")[0])
    f = 0.8486
    # FW CRUISE: discounted
    assert fw.power(ManeuverType.CRUISE, f) == pytest.approx(fw.power(ManeuverType.CRUISE) * f)
    # FW COVERAGE: never discounted (thesis invariant)
    assert fw.power(ManeuverType.COVERAGE, f) == pytest.approx(fw.power(ManeuverType.COVERAGE))
    # MULTIROTOR CRUISE: no benefit
    assert mr.power(ManeuverType.CRUISE, f) == pytest.approx(mr.power(ManeuverType.CRUISE))


def test_distance_energy_consistency(config_path):
    spec, _ = _spec(config_path, "FIXED_WING")
    em = EnergyModel(spec)
    d, v = 100.0, spec.v_cruise
    assert em.distance_energy(d, ManeuverType.CRUISE, v) == pytest.approx(
        em.segment_energy(ManeuverType.CRUISE, d / v)
    )
