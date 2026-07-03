"""S_FERRY Step 2 -- obstacle-aware connector routing over the flyable region.

Covers the acceptance criteria:
  AC-1  C-shape survey shape at exactly 1 km^2, loads, a run covers the surveyed
        region (not the notch).
  AC-2  flyable region: with routing ON and no obstacle, a connector crossing the
        empty notch takes the straight chord (no spurious avoidance).
  AC-3  obstacle-aware: an NFZ on a connector chord forces a detour that never
        intersects the obstacle; a connector's cost is never reduced by routing.
  AC-4  analytical == execution: with an obstacle present and routing ON, the
        engine's drained coverage energy matches the analytical E_cover (single
        source). The test PRINTS the chord-vs-detour connector energies.
  AC-5  byte-identity: routing OFF, or ON with nothing blocking, reproduces
        today's straight-chord connectors exactly.
"""
from __future__ import annotations

import json
import math

import pytest
from conftest import config_path
from shapely.affinity import scale
from shapely.geometry import LineString, Polygon, box

from uav_swarm_sim.experiments.generate_shapes import _c_shape as _canonical_c_shape
from uav_swarm_sim.experiments.run_regime_calculator import (
    _COVERAGE_STATES,
    build_spec,
    coverage_energy,
)
from uav_swarm_sim.infrastructure import simulation_engine as _se
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose, Zone
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo, ManeuverType
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.coverage_path import boustrophedon
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.geojson_parser import load_area
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.planning.visibility_router import route_connector


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
# the canonical unit C-shape (2x2, solidity 0.8) shipped as data/areas/shapes/c_shape.geojson
_UNIT_C = _canonical_c_shape()


def _c_shape(width_m: float) -> Polygon:
    """The canonical C-shape scaled to a ``width_m`` bounding box (metric)."""
    return scale(_UNIT_C, xfact=width_m / 2.0, yfact=width_m / 2.0, origin=(0, 0))


def _write_geojson(path, poly: Polygon) -> str:
    feat = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in poly.exterior.coords]]},
    }
    path.write_text(json.dumps(feat))
    return str(path)


def _kit(cfg):
    spec = build_spec(cfg)
    return spec, make_motion_model(spec), EnergyModel(spec)


def _first_notch_connector(poly, spec, motion, em):
    """Return (a_pose, b_pose, chord) for the first inter-strip connector whose
    straight chord leaves the survey polygon (i.e. crosses the notch)."""
    zone = Zone(0, [], poly, Pose(poly.centroid.x, poly.centroid.y, 0.0))
    plan = boustrophedon(zone, spec, motion, em)
    wp = plan.waypoints
    for i in range(1, len(wp) - 1, 2):
        a, b = wp[i].pose, wp[i + 1].pose
        seg = LineString([a.as_xy(), b.as_xy()])
        if seg.difference(poly.buffer(1e-6)).length > 1.0:
            return a, b, seg
    raise AssertionError("no notch-crossing connector found on the C-shape")


def _path_hits(path, obstacle) -> bool:
    ps = path.sample(1.0)
    return any(
        obstacle.intersects(LineString([ps[i].as_xy(), ps[i + 1].as_xy()]))
        for i in range(len(ps) - 1)
    )


# --------------------------------------------------------------------------- #
# AC-1 -- C-shape shipped at 1 km^2, loads, a mission covers it (not the notch) #
# --------------------------------------------------------------------------- #
def test_ac1_c_shape_shipped_is_one_km2_and_loads():
    poly = load_area("data/areas/shapes/c_shape.geojson")
    assert poly.area == pytest.approx(1_000_000.0, rel=1e-6)
    assert poly.is_valid
    assert len(list(poly.interiors)) == 0  # simple polygon (donut deferred)
    assert poly.area / poly.convex_hull.area == pytest.approx(0.8, abs=1e-3)


def test_ac1_mission_covers_c_shape_not_the_notch(tmp_path):
    # a small C-shape swept by a 4-drone fleet (weighted-Voronoi partition over a
    # concave polygon -- exercises the decomposition sliver guard) on a generous
    # battery => swap-free and fast; the surveyed polygon is covered and the notch
    # (outside the polygon) is not.
    poly = _c_shape(500.0)
    gj = _write_geojson(tmp_path / "c_small.geojson", poly)
    cfg = load_config(
        config_path(),
        overrides={
            "env.geojson_path": gj,
            "env.obstacle_density_per_km2": 0.0,
            "failure.hazard_rate_per_hour": 0.0,
            "fleet.n_drones": 4,
            "fleet.battery_capacity_wh": 400.0,
            "sim.max_timesteps": 200000,
            "telemetry.enabled": False,
        },
    )
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert not result.aborted
    assert len(eng.partition.zones) == 4
    assert result.coverage_frac > 0.99
    # the notch centre is outside the survey polygon => never a coverage target
    notch_centre = box(250.0, 150.0, 500.0, 350.0).centroid
    assert not poly.covers(notch_centre)


# --------------------------------------------------------------------------- #
# AC-5 -- byte-identity: OFF, or ON with nothing blocking == today's chords    #
# --------------------------------------------------------------------------- #
def test_ac5_flag_off_is_byte_identical():
    cfg = load_config(config_path())
    spec, motion, em = _kit(cfg)
    poly = _c_shape(800.0)
    zone = Zone(0, [], poly, Pose(poly.centroid.x, poly.centroid.y, 0.0))
    env = EnvironmentMap(poly, [], cfg.env.clearance_buffer_m)

    legacy = boustrophedon(zone, spec, motion, em)                       # no env/flag
    off = boustrophedon(zone, spec, motion, em, env=env, coverage=_cov(False))
    assert off.connectors == []                                         # nothing stored
    assert off.length_m == pytest.approx(legacy.length_m, abs=1e-9)
    assert off.est_energy_j == pytest.approx(legacy.est_energy_j, abs=1e-9)


def test_ac5_flag_on_no_obstacle_matches_chords():
    cfg = load_config(config_path())
    spec, motion, em = _kit(cfg)
    poly = _c_shape(800.0)
    zone = Zone(0, [], poly, Pose(poly.centroid.x, poly.centroid.y, 0.0))
    env = EnvironmentMap(poly, [], cfg.env.clearance_buffer_m)   # obstacle-free

    off = boustrophedon(zone, spec, motion, em, env=env, coverage=_cov(False))
    on = boustrophedon(zone, spec, motion, em, env=env, coverage=_cov(True))
    # ON populates one connector per odd leg, but each equals the straight chord
    assert len(on.connectors) == (len(on.waypoints) // 2) - 1
    assert on.est_energy_j == pytest.approx(off.est_energy_j, abs=1e-9)
    # and at least one of those chords crosses the notch (outside the polygon) --
    # proving the flyable region admits the exterior with no spurious avoidance
    crossed = any(
        LineString([c.start_pose.as_xy(), c.end_pose.as_xy()]).difference(poly.buffer(1e-6)).length > 1.0
        for c in on.connectors
    )
    assert crossed


# --------------------------------------------------------------------------- #
# AC-2 / AC-3 -- notch chord is kept when clear; detoured (never through) when  #
# an NFZ blocks it                                                              #
# --------------------------------------------------------------------------- #
def test_ac2_empty_notch_takes_straight_chord():
    cfg = load_config(config_path())
    spec, motion, em = _kit(cfg)
    poly = _c_shape(800.0)
    a, b, chord_seg = _first_notch_connector(poly, spec, motion, em)
    env = EnvironmentMap(poly, [], cfg.env.clearance_buffer_m)  # notch empty

    routed = route_connector(a, b, motion, env, enabled=True,
                             operating_area="convex_hull", margin_m=50.0)
    chord = motion.plan(a, b, ManeuverType.TURN)
    # identical geometry: same segment tuple => byte-identical connector
    assert routed.segments == chord.segments
    assert routed.total_length_m == pytest.approx(chord.total_length_m, abs=1e-9)


def test_ac3_obstacle_forces_detour_around_never_through():
    cfg = load_config(config_path())
    spec, motion, em = _kit(cfg)
    poly = _c_shape(800.0)
    a, b, chord_seg = _first_notch_connector(poly, spec, motion, em)
    # NFZ squarely on the chord, inside the notch (never touches a coverage strip)
    mid = chord_seg.interpolate(0.5, normalized=True)
    nfz = box(mid.x - 20, mid.y - 20, mid.x + 20, mid.y + 20)
    env = EnvironmentMap(poly, [Obstacle(id=0, cls=0, polygon=nfz)], cfg.env.clearance_buffer_m)

    chord = motion.plan(a, b, ManeuverType.TURN)
    detour = route_connector(a, b, motion, env, enabled=True,
                             operating_area="convex_hull", margin_m=50.0)
    assert _path_hits(chord, nfz)            # the straight chord WOULD hit it
    assert not _path_hits(detour, nfz)       # the routed connector does not
    # a connector's cost is never reduced by routing (detour >= chord)
    assert detour.total_length_m >= chord.total_length_m - 1e-9
    assert em.path_energy(detour) >= em.path_energy(chord) - 1e-9


# --------------------------------------------------------------------------- #
# AC-4 -- analytical == execution with an obstacle present (prints numbers)     #
# --------------------------------------------------------------------------- #
def test_ac4_analytical_equals_execution_with_obstacle(tmp_path, monkeypatch, capsys):
    poly = _c_shape(400.0)
    gj = _write_geojson(tmp_path / "c_ac4.geojson", poly)

    cfg0 = load_config(config_path())
    spec, motion, em = _kit(cfg0)
    a, b, chord_seg = _first_notch_connector(poly, spec, motion, em)
    mid = chord_seg.interpolate(0.5, normalized=True)
    nfz = box(mid.x - 15, mid.y - 15, mid.x + 15, mid.y + 15)

    # inject exactly this one NFZ into the engine's obstacle field
    monkeypatch.setattr(_se, "generate_obstacles", lambda area, env, rng: [Obstacle(id=0, cls=0, polygon=nfz)])

    cfg = load_config(
        config_path(),
        overrides={
            "env.geojson_path": gj,
            "fleet.n_drones": 1,
            "fleet.battery_capacity_wh": 1500.0,   # swap-free single sortie
            "failure.hazard_rate_per_hour": 0.0,
            "sim.max_timesteps": 200000,
            "coverage.ferry_free_space": True,
            "telemetry.enabled": False,
        },
    )
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert not result.aborted
    assert result.coverage_frac > 0.99
    n_swaps = sum(1 for s in eng.history.sojourns() if s.state is AgentState.S_SWAP)
    assert n_swaps == 0  # a swap re-flies a leg and would break the comparison

    cap = eng.spec.battery_capacity_j
    engine_cov_j = 0.0
    for aid in eng.fleet.agents:
        pos = eng.history.position_trace(aid)
        bat = eng.history.battery_trace(aid)
        for i in range(min(len(pos), len(bat)) - 1):
            if pos[i][3] in _COVERAGE_STATES:
                engine_cov_j += cap * (bat[i][1] - bat[i + 1][1])

    analytical_cov_j = sum(
        coverage_energy(z.polygon, eng.spec, eng.em, eng.motion, 0.0,
                        env=eng.env, coverage=cfg.coverage)["coverage_total_j"]
        for z in eng.partition.zones.values()
    )
    rel = abs(engine_cov_j - analytical_cov_j) / analytical_cov_j

    # the concrete chord-vs-detour numbers for the blocked connector
    chord = motion.plan(a, b, ManeuverType.TURN)
    detour = route_connector(a, b, motion, eng.env, enabled=True,
                             operating_area=cfg.coverage.operating_area,
                             margin_m=cfg.coverage.operating_margin_m)
    with capsys.disabled():
        print("\n[AC-4] analytical vs execution (routing ON, one NFZ on a notch connector)")
        print(f"       engine drained coverage E = {engine_cov_j:10.1f} J")
        print(f"       analytical  E_cover       = {analytical_cov_j:10.1f} J")
        print(f"       relative error            = {rel * 100:.3f} %  (tol 2.0 %)")
        print(f"       blocked connector: CHORD  len={chord.total_length_m:6.1f} m  "
              f"E={em.path_energy(chord):8.1f} J  hits_obstacle={_path_hits(chord, nfz)}")
        print(f"                          DETOUR len={detour.total_length_m:6.1f} m  "
              f"E={em.path_energy(detour):8.1f} J  hits_obstacle={_path_hits(detour, nfz)}")
        print(f"                          detour cost premium = "
              f"{(em.path_energy(detour) / em.path_energy(chord) - 1) * 100:+.2f} %")

    assert rel < 0.02
    assert not _path_hits(detour, nfz)
    assert em.path_energy(detour) > em.path_energy(chord)  # detour is strictly longer here


# --------------------------------------------------------------------------- #
def _cov(on: bool):
    """A minimal CoverageConfig stand-in for the boustrophedon call."""
    from uav_swarm_sim.infrastructure.config import CoverageConfig
    return CoverageConfig(ferry_free_space=on, operating_area="convex_hull", operating_margin_m=50.0)
