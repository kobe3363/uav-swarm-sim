"""EXP-06 diagnostics are observational; physical drain bounds use the executor."""
from collections import Counter
from dataclasses import fields, replace
import json
import math
from types import SimpleNamespace

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState as S, DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.monte_carlo import MCResult, single_run_from_history
from uav_swarm_sim.metrics.run_output import build_results_mc, build_results_single
from uav_swarm_sim.planning.energy_balance import EnergyBalanceStatus, ZoneEnergyEstimate


@pytest.fixture
def mission_overrides(tmp_path):
    area = tmp_path / "rectangle.geojson"
    area.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(box(0, 0, 600, 120))}))
    return {
        "fleet.n_drones": 1, "fleet.battery_capacity_wh": 1000.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        # Both half-zones remain horizontal sweeps. A tall half-zone has a
        # rotated endpoint at y=-1e-15, rejected by the existing strict map.
        "launch.candidate_sites": [[300.0, 0.0]],
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "platforms.MULTIROTOR.r_min_m": 0.0,
        "platforms.MULTIROTOR.omega_max": 1.0,
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
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 10.0,
        "mission.no_swap_mode": True,
        "sim.dt_s": 0.5, "sim.max_timesteps": 2000,
    }


def _run(overrides):
    cfg = load_config("config/default.yaml", overrides=overrides)
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                              algo=DecompositionAlgo.TGC_BASIC)
    return engine, engine.run()


def _physical_signature(result):
    m = result.metrics
    return (result.outcome, result.coverage_frac, result.target_coverage_frac,
            m.total_energy_j, m.duration_s, m.n_swaps, m.n_failures,
            dict(m.per_agent_energy_j), dict(m.per_agent_length_m),
            tuple(result.history.sojourns()), Counter(e.agent_id for e in result.photo_events),
            result.retired_agents, result.work_releases, result.losses)


def _output(result):
    return build_results_single(result, SimpleNamespace(ergodic=False), identity={}, wall_time_s=0)


def test_t16_absent_and_explicit_off_are_byte_identical(mission_overrides, monkeypatch):
    import uav_swarm_sim.planning.energy_balance as balance

    def forbidden(*args, **kwargs):
        pytest.fail("flag off must not construct an energy balance context")

    monkeypatch.setattr(balance, "build_energy_balance_context", forbidden)
    absent_engine, absent = _run(mission_overrides)
    off_engine, off = _run(dict(mission_overrides, **{"planning.energy_balance.enabled": False}))
    # Exact tuple equality is deliberate here: this is the required byte-identity gate.
    assert _physical_signature(absent) == _physical_signature(off)
    assert not hasattr(absent_engine, "energy_balance_t0")
    assert not hasattr(off_engine, "energy_balance_t0")
    assert "energy_balance" not in _output(absent)
    assert "energy_balance" not in _output(off)


@pytest.mark.parametrize("map_on", [False, True])
def test_t17_on_preserves_metrics_and_exports_every_component(mission_overrides, map_on):
    overrides = dict(mission_overrides, **{
        "fleet.n_drones": 2,
        "rth.energy_map.enabled": map_on, "rth.energy_map.decide": map_on,
        "rth.energy_map.cell_m": 20.0,
    })
    off_engine, off = _run(overrides)
    engine, on = _run(dict(overrides, **{"planning.energy_balance.enabled": True}))
    assert _physical_signature(off) == _physical_signature(on)
    assert (engine.rth.n_map_hits, engine.rth.n_map_fallbacks) == (
        off_engine.rth.n_map_hits, off_engine.rth.n_map_fallbacks)
    assert set(engine.energy_balance_t0) == {0, 1}
    out = _output(on)
    assert set(out["energy_balance"]) == {"0", "1"}
    for aid, methods in engine.energy_balance_t0.items():
        assert set(methods) == {"fast", "path"}
        for method, estimate in methods.items():
            assert estimate.status is EnergyBalanceStatus.FEASIBLE
            exported = out["energy_balance"][str(aid)][method]
            assert set(exported) == {f.name for f in fields(ZoneEnergyEstimate)}
            assert exported["status"] == estimate.status.value
            for pose_key in ("anchor_pose", "exit_pose"):
                assert exported[pose_key] == pytest.approx(getattr(estimate, pose_key).as_xy())
            for f in fields(estimate):
                value = getattr(estimate, f.name)
                if isinstance(value, float):
                    assert math.isfinite(value)
                    assert exported[f.name] == pytest.approx(value, rel=1e-9)
    json.dumps(out["energy_balance"], allow_nan=False)


def test_t18_realised_drain_has_signed_tick_bound(mission_overrides, monkeypatch):
    """Agent._tick_dynamics charges full dt on a leg's last partial tick (V2).

    Measure from entry into S1_TRANSIT to completion of the last strip, excluding
    idle and RTH. The current executor has no separate takeoff leg: S0->S1 is
    its post-launch boundary. The estimate's takeoff budget is not part of this
    horizontal drain comparison. Realised drain must be >= the exact integral,
    with excess <= n_legs * max(power_w) * dt; n_legs comes from the real plan.
    """
    marks = {}
    original = Agent._tick_dynamics

    def observe(agent, dt, t):
        if agent.id == 0 and agent.state is S.S1_TRANSIT and "start" not in marks:
            marks["start"] = agent.battery.level_j
        original(agent, dt, t)
        if (agent.id == 0 and "start" in marks and "end" not in marks
                and agent.state in (S.S2_MISSION, S.S_FERRY)
                and agent._cov_idx == len(agent._cov_legs)):
            marks["end"] = agent.battery.level_j

    monkeypatch.setattr(Agent, "_tick_dynamics", observe)
    engine, result = _run(dict(mission_overrides, **{"planning.energy_balance.enabled": True}))
    assert set(marks) == {"start", "end"}
    assert len(engine.partition.zones) == 1 and not engine.env.obstacles
    assert result.coverage_frac == pytest.approx(1.0)
    plan = engine.plans[0]
    assert len(plan.waypoints) == 6
    n_legs = 1 + len(plan.waypoints) - 1  # one ferry + alternating strips/connectors
    estimate = engine.energy_balance_t0[0]["path"]
    expected = estimate.e_coverage_j + estimate.e_ferry_j
    realised = marks["start"] - marks["end"]
    assert realised >= expected - 1e-9
    assert realised - expected <= n_legs * max(engine.spec.power_w.values()) * engine.cfg.sim.dt_s


def test_mc_adapter_and_optional_output(mission_overrides):
    _, result = _run(dict(mission_overrides, **{"planning.energy_balance.enabled": True}))
    run = single_run_from_history(result.history, metrics=result.metrics, outcome=result.outcome,
                                  energy_balance_t0=result.energy_balance_t0)
    assert run.energy_balance_t0 is result.energy_balance_t0
    mc = MCResult(1, False, {}, {}, math.nan, math.nan, 0.0, [], [run])
    output = build_results_mc(mc, identity={}, wall_time_s=0)
    assert output["energy_balance"][0]["replication"] == 1
    assert output["energy_balance"][0]["drones"] == _output(result)["energy_balance"]
    off = replace(mc, runs=[replace(run, energy_balance_t0=None)])
    assert "energy_balance" not in build_results_mc(off, identity={}, wall_time_s=0)
