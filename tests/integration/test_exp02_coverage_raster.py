"""EXP-02 engine wiring for persistent physical raster coverage."""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, Outcome
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


def test_raster_engine_reports_target_and_plannable_physical_coverage(config_path):
    cfg = load_config(config_path, overrides={
        "fleet.n_drones": 1,
        "fleet.battery_capacity_wh": 1000.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "sim.dt_s": 0.5,
        "sim.max_timesteps": 10000,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 17.3,
        "sensor.photogrammetry.sensor_height_mm": 13.0,
        "sensor.photogrammetry.focal_length_mm": 12.0,
        "sensor.photogrammetry.image_width_px": 5280,
        "sensor.photogrammetry.image_height_px": 3956,
        "sensor.photogrammetry.side_overlap": 0.70,
        "sensor.photogrammetry.forward_overlap": 0.80,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "coverage.raster_enabled": True,
        "coverage.raster_cell_m": 10.0,
    })
    engine = SimulationEngine(
        cfg,
        RngFactory(cfg.sim.master_seed),
        replication=0,
        algo=DecompositionAlgo.WEIGHTED_VORONOI,
    )

    result = engine.run()

    assert result.outcome is Outcome.MISSION_SUCCESS
    assert result.coverage_frac == pytest.approx(
        engine.coverage_raster.plannable_coverage_frac
    )
    assert result.target_coverage_frac == pytest.approx(
        engine.coverage_raster.target_coverage_frac
    )
    assert result.coverage_frac >= 1.0 - cfg.coverage.raster_completion_tolerance_frac
