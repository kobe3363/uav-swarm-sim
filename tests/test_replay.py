"""Tests for the 2D positional replay feature: position logging, determinism of
traces, and GIF/PNG generation."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import config_path

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.infrastructure import visualization as viz


def _smoke_cfg():
    return load_config(
        config_path(),
        overrides={
            "fleet.n_drones": 3,
            "fleet.battery_capacity_wh": 400.0,
            "failure.hazard_rate_per_hour": 0.0,
            "env.geojson_path": "data/areas/smoke_area.geojson",
            "env.obstacle_density_per_km2": 4.0,
            "env.obstacle_size_range_m": [10.0, 30.0],
            "sim.dt_s": 1.0,
            "sim.max_timesteps": 20000,
        },
    )


def _run(replication=0):
    cfg = _smoke_cfg()
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=replication,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    return cfg, eng, result


@pytest.fixture(scope="module")
def run0():
    return _run(0)


# --------------------------------------------------------------------------- #
# position logging                                                            #
# --------------------------------------------------------------------------- #
def test_position_traces_exist_for_every_agent(run0):
    cfg, eng, result = run0
    for aid in eng.fleet.agents:
        tr = result.history.position_trace(aid)
        assert len(tr) > 0, f"agent {aid} has no position trace"
        # each sample is (t, x, y, state) with a valid state and non-decreasing time
        ts = [s[0] for s in tr]
        assert ts == sorted(ts)
        assert all(isinstance(s[3], AgentState) for s in tr)


def test_all_traces_time_aligned(run0):
    cfg, eng, result = run0
    lengths = {aid: len(result.history.position_trace(aid)) for aid in eng.fleet.agents}
    # every agent logged once per tick -> identical lengths and timestamps
    assert len(set(lengths.values())) == 1
    aids = list(eng.fleet.agents)
    t0 = [s[0] for s in result.history.position_trace(aids[0])]
    for aid in aids[1:]:
        assert [s[0] for s in result.history.position_trace(aid)] == t0


def test_traces_capture_multiple_states(run0):
    cfg, eng, result = run0
    seen = set()
    for aid in eng.fleet.agents:
        seen.update(s[3] for s in result.history.position_trace(aid))
    # a normal mission visits at least transit, mission, RTH, idle
    assert AgentState.S2_MISSION in seen
    assert AgentState.S1_TRANSIT in seen


# --------------------------------------------------------------------------- #
# determinism of the replay traces                                            #
# --------------------------------------------------------------------------- #
def test_replication_replay_is_deterministic():
    _, _, r_a = _run(replication=60)
    _, _, r_b = _run(replication=60)
    for aid in r_a.history._position:  # same agents
        ta = r_a.history.position_trace(aid)
        tb = r_b.history.position_trace(aid)
        assert ta == tb, f"replication 60 not reproducible for agent {aid}"


# --------------------------------------------------------------------------- #
# output generation                                                           #
# --------------------------------------------------------------------------- #
def test_static_state_colored_paths_png(run0, tmp_path):
    cfg, eng, result = run0
    p = viz.plot_state_colored_paths(result.history, eng.env, tmp_path / "paths.png",
                                     partition=result.partition)
    assert Path(p).exists() and Path(p).stat().st_size > 0


@pytest.mark.slow
def test_replay_gif_is_produced(run0, tmp_path):
    cfg, eng, result = run0
    g = viz.animate_mission(result.history, eng.env, tmp_path / "replay.gif",
                            fps=8, max_frames=40, partition=result.partition)
    assert Path(g).exists() and Path(g).stat().st_size > 0
    # GIF magic header
    assert Path(g).read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_animate_requires_position_traces(tmp_path):
    from uav_swarm_sim.metrics.state_history import StateHistory

    class _Env:  # minimal stand-in; should never be reached
        pass

    empty = StateHistory()
    with pytest.raises(ValueError):
        viz.animate_mission(empty, _Env(), tmp_path / "x.gif")
