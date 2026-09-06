"""Independent, fully pinned EXP-06 numeric fixture."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose, Zone
from uav_swarm_sim.infrastructure.enums import ManeuverType as M, PlatformType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import HolonomicModel
from uav_swarm_sim.planning.coverage_raster import CoverageRaster
from uav_swarm_sim.planning.energy_balance import DroneEnergyState, build_energy_balance_context


@pytest.fixture
def energy_case():
    cfg = load_config("config/default.yaml", {
        "env.coverage_altitude_m": 100.0,
        "sensor.sensor_power_w": 15.0,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 8.0,
        "sensor.photogrammetry.sensor_height_mm": 6.0,
        "sensor.photogrammetry.focal_length_mm": 10.0,
        "sensor.photogrammetry.image_width_px": 4000,
        "sensor.photogrammetry.image_height_px": 3000,
        "sensor.photogrammetry.side_overlap": 0.5,
        "sensor.photogrammetry.forward_overlap": 0.5,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "coverage.raster_enabled": True,
        "rth.reserve_frac": 0.05,
    })
    spec = replace(build_spec(cfg), platform=PlatformType.MULTIROTOR,
                   mass_kg=4.0, battery_capacity_j=360000.0, r_min_m=0.0, omega_max=1.0,
                   v_cruise=12.0, v_coverage=10.0, v_climb=4.0, v_descent=3.0,
                   swath_width_m=40.0,
                   power_w={**cfg.platform.power_w, M.CRUISE: 220.0,
                            M.COVERAGE: 250.0, M.TURN: 240.0,
                            M.TAKEOFF: 400.0, M.LAND: 300.0})
    em = EnergyModel(spec)
    motion = HolonomicModel(spec)
    base = Pose(0.0, 100.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, 100.0)
    ctx = build_energy_balance_context(
        cfg, em, spec, motion, None,
        lambda pose, alt: rth.return_energy(pose, altitude_m=alt),
    )
    zone = Zone(0, [], box(300, 0, 500, 120), Pose(0.0, 20.0, 0.0))
    return SimpleNamespace(cfg=cfg, spec=spec, em=em, motion=motion, base=base,
                           rth=rth, ctx=ctx, zone=zone,
                           drone=DroneEnergyState(0, zone.entry_pose, 360000.0, False),
                           raster=CoverageRaster(zone.polygon, zone.polygon, 10.0))
