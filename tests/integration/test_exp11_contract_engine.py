"""EXP-11: engine-sourced contract fields (coverage areas, final battery) and the
final_soc <-> battery_trace cross-check. One real raster mission, then the raw
contract is built from its result -- no re-run.
"""
from __future__ import annotations

import json

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.run_output import _jsonable, build_mission_contract

S = AgentState


def _cfg(config_path, **extra):
    return load_config(config_path, overrides={
        "fleet.n_drones": 2,
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
        "mission.no_swap_mode": True,
        **extra,
    })


@pytest.fixture(scope="module")
def result(config_path):
    cfg = _cfg(config_path)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    res = eng.run()
    return cfg, eng, res


def test_final_battery_matches_live_and_trace_for_survivors(result):
    cfg, eng, res = result
    final = {aid: (lvl, soc) for aid, lvl, soc in res.final_battery_by_drone}
    for a in eng.fleet.agents.values():
        # engine field is the exact live battery reading
        assert final[a.id][1] == pytest.approx(a.battery.frac)
        # independent cross-check: survivors' last recorded trace sample equals it
        if a.id not in eng.fleet._failed:
            trace = res.history.battery_trace(a.id)
            assert trace and trace[-1][1] == pytest.approx(final[a.id][1], abs=1e-9)


def test_coverage_measurements_come_from_the_raster(result):
    cfg, eng, res = result
    r = eng.coverage_raster
    m = res.coverage_measurements
    assert m["source"] == "raster"
    assert m["a_target_m2"] == pytest.approx(r.target_area_m2)
    assert m["a_plannable_m2"] == pytest.approx(r.plannable_area_m2)
    assert m["plannable_covered_area_m2"] == pytest.approx(r.plannable_covered_area_m2)


def test_contract_is_self_contained_and_reconciles(result):
    cfg, eng, res = result
    c = build_mission_contract(res, capacity_j=cfg.fleet.battery_capacity_j,
                               decomposer_class=type(eng.decomposer).__name__)
    # roundtrips as strict JSON
    json.loads(json.dumps(_jsonable(c)))
    # no swaps in no_swap_mode => reconciliation is authoritative and non-negative
    rec = c["energy"]["reconciliation"]
    assert rec["reconcilable"] is True
    assert rec["clamp_residual_j"] >= -1e-6
    # fleet total energy equals the per-agent sum
    assert rec["sum_per_agent_j"] == pytest.approx(c["energy"]["total_consumed_j"])
    # coverage denominators are both present and positive
    assert c["coverage"]["a_target_m2"] > 0 and c["coverage"]["a_plannable_m2"] > 0
    assert c["outcome"]["decomposer_class"]
