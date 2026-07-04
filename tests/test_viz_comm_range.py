"""Tests for the optional communication-range overlay (VizConfig)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
from matplotlib.patches import Circle

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.infrastructure import visualization as viz


def _engine(config_path):
    cfg = load_config(config_path, overrides={
        "fleet.n_drones": 3, "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0, "env.geojson_path": "data/areas/smoke_area.geojson",
        "sim.dt_s": 1.0, "sim.max_timesteps": 20000,
    })
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    eng.run()
    return eng


def _count_circles(ax):
    return sum(1 for p in ax.patches if isinstance(p, Circle))


# --------------------------------------------------------------------------- #
# config defaults                                                             #
# --------------------------------------------------------------------------- #
def test_comm_range_defaults_off(config_path):
    v = load_config(config_path).viz
    assert v.show_comm_range is False
    assert v.comm_range_m > 0 and 0.0 < v.comm_range_alpha <= 1.0


# --------------------------------------------------------------------------- #
# helper: circles created only when enabled                                  #
# --------------------------------------------------------------------------- #
def test_helper_off_creates_no_patches(config_path):
    fig, ax = matplotlib.pyplot.subplots()
    off = load_config(config_path).viz
    assert viz._make_comm_circles(ax, [0, 1, 2], off) == {}
    assert _count_circles(ax) == 0
    assert viz._make_comm_circles(ax, [0, 1, 2], None) == {}   # None == off
    matplotlib.pyplot.close(fig)


def test_helper_on_creates_one_circle_per_drone(config_path):
    fig, ax = matplotlib.pyplot.subplots()
    on = load_config(config_path, overrides={"viz.show_comm_range": True,
                                               "viz.comm_range_m": 100.0}).viz
    circles = viz._make_comm_circles(ax, [0, 1, 2], on)
    assert set(circles) == {0, 1, 2}
    assert _count_circles(ax) == 3
    for c in circles.values():
        assert c.radius == 100.0
        assert c.get_zorder() == 1          # low z-order: behind drones/obstacles
    matplotlib.pyplot.close(fig)


def test_dashed_vs_filled_styles(config_path):
    on_dashed = load_config(config_path, overrides={"viz.show_comm_range": True,
                                                      "viz.comm_range_dashed": True}).viz
    on_fill = load_config(config_path, overrides={"viz.show_comm_range": True,
                                                    "viz.comm_range_dashed": False}).viz
    fig, ax = matplotlib.pyplot.subplots()
    d = next(iter(viz._make_comm_circles(ax, [0], on_dashed).values()))
    f = next(iter(viz._make_comm_circles(ax, [1], on_fill).values()))
    assert d.get_fill() is False and f.get_fill() is True
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# end-to-end rendering: off vs on, both produce valid files                   #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_static_plot_off_vs_on(config_path, tmp_path):
    eng = _engine(config_path)
    off = load_config(config_path).viz
    on = load_config(config_path, overrides={"viz.show_comm_range": True}).viz
    p_off = viz.plot_state_colored_paths(eng.history, eng.env, tmp_path / "off.png", viz=off)
    p_on = viz.plot_state_colored_paths(eng.history, eng.env, tmp_path / "on.png", viz=on)
    assert Path(p_off).stat().st_size > 0 and Path(p_on).stat().st_size > 0


@pytest.mark.slow
def test_animation_off_vs_on(config_path, tmp_path):
    eng = _engine(config_path)
    on = load_config(config_path, overrides={"viz.show_comm_range": True,
                                               "viz.comm_range_m": 120.0}).viz
    g_off = viz.animate_mission(eng.history, eng.env, tmp_path / "off.gif", max_frames=25)  # viz=None
    g_on = viz.animate_mission(eng.history, eng.env, tmp_path / "on.gif", max_frames=25, viz=on)
    for g in (g_off, g_on):
        assert Path(g).read_bytes()[:6] in (b"GIF87a", b"GIF89a")
