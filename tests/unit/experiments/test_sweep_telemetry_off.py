"""The sweeps force telemetry OFF (perf), and doing so is byte-identical.

Telemetry is a read-only observability probe (GPX + JSONL to a fixed, overwritten
path). Rebuilding and re-exporting it on every one of the thousands of sweep
missions is pure waste, so ``build_cell_cfg`` / ``_apply_mode`` disable it. This
test pins BOTH facts: (1) the cell/tier configs come out with telemetry off, and
(2) toggling telemetry changes NO mission metric -- the acceptance gate for the
optimization being byte-identical.
"""
from __future__ import annotations

import dataclasses

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.experiments.run_shape_sweep import build_cell_cfg
from uav_swarm_sim.experiments.run_scale_tiers import _apply_mode


def test_shape_sweep_cell_cfg_disables_telemetry(config_path):
    base = load_config(str(config_path))
    assert base.telemetry.enabled is True     # guards the premise (shipped default ON)
    cfg = build_cell_cfg(base, "data/areas/shapes/square.geojson", 2, "shipped", 3)
    assert cfg.telemetry.enabled is False


def test_scale_tiers_apply_mode_disables_telemetry(config_path):
    base = load_config(str(config_path))
    for mode in ("clean", "shipped"):
        assert _apply_mode(base, mode).telemetry.enabled is False


@pytest.mark.slow
def test_telemetry_on_off_metrics_byte_identical(config_path, tmp_path):
    """Toggling telemetry must not perturb a single metric: the FanoutRecorder
    feeds StateHistory the identical open/close stream either way, and the probe
    never writes back into physics/history."""
    from uav_swarm_sim.infrastructure.rng import RngFactory
    from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
    from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, PlannerKind

    base = load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
    })
    tel_on = dataclasses.replace(base.telemetry, enabled=True,
                                 gpx_path=str(tmp_path / "t.gpx"),
                                 llm_log_path=str(tmp_path / "t.jsonl"))
    cfg_on = dataclasses.replace(base, telemetry=tel_on)
    cfg_off = dataclasses.replace(
        base, telemetry=dataclasses.replace(base.telemetry, enabled=False))

    def run(cfg):
        eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0,
                               algo=DecompositionAlgo.TGC_BASIC,
                               planner=PlannerKind.DUBINS)
        m = eng.run().metrics
        return (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
                dict(m.per_agent_energy_j), dict(m.per_agent_length_m))

    assert run(cfg_on) == run(cfg_off)
