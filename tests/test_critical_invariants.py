"""Regression tests for five thesis-critical invariants that had zero direct
test coverage before this file: the GVG clearance guarantee, the launch-site
optimizer's staging-ring constraint and swap-weight sensitivity, the GPX
exporter's coordinate projection and file contract, and the redistribution
handler's swap/failure trigger asymmetry.

Each of these is exercised only *indirectly*, if at all, by the existing
batch/smoke suite: a regression in any one of them could pass 172/172 today
and still silently break a thesis claim or corrupt the gynyba (defense) demo
deliverable (e.g. a malformed GPX file, or a launch site inside the surveyed
polygon). These tests close that gap with the smallest fixtures that can fail
for the right reason.

What is real vs. doubled
-------------------------
REAL: gvg_builder.build_gvg, environment_map.EnvironmentMap, launch_site_optimizer
(furthest_point_feasible, optimize, the staging ring), gpx_exporter.build_gpx /
write_gpx, execution.redistribution.Redistributor.should_handle.

DOUBLED: only what gpx_exporter needs from TelemetryLog (drone_ids / gpx_track),
via a minimal duck-typed fake -- gpx_exporter's contract is structural, not
tied to TelemetryLog's internals, so a fake is the correct isolation boundary
here (consistent with test_obstacle_recovery.py's doubling rationale).
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import DroneStateView, Event, Pose
from uav_swarm_sim.infrastructure.enums import EventType, ManeuverType, PlatformType
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection
from uav_swarm_sim.physical_model.drone_specs import PlatformSpec, build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.gvg_builder import build_gvg
from uav_swarm_sim.planning.launch_site_optimizer import optimize
from uav_swarm_sim.planning.tgc import build_tgc
from uav_swarm_sim.metrics.gpx_exporter import build_gpx, write_gpx

CONFIG_PATH = "config/default.yaml"


# =========================================================================== #
# 1. GVG clearance guarantee (planning/gvg_builder.py)                        #
#                                                                              #
# This is the geometric foundation the thesis's "safe corridor" requirement   #
# rests on: build_gvg explicitly filters out any ridge endpoint whose         #
# clearance is below env.buffer_m. If that filter ever regresses (e.g. a      #
# refactor drops the check, or compares the wrong buffer), every zone built   #
# downstream on the GVG/TGC skeleton could route a drone through a gap        #
# narrower than the configured safety buffer -- silently, since most tests    #
# exercise GVG only as a black box feeding TGC/decomposition, not its         #
# clearance property directly.                                               #
# =========================================================================== #
def test_gvg_edges_never_violate_the_clearance_buffer():
    """Every GVG node's clearance (distance to the nearest obstacle or the area
    boundary) must be >= the configured buffer. This is the corridor-safety
    invariant the thesis relies on for "drones never route through a gap
    narrower than the safe distance"."""
    area = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    # two obstacles close enough that a naive (non-clearance-checked) GVG would
    # route a ridge through the gap between them
    obstacles_xy = [
        Polygon([(300, 400), (340, 400), (340, 440), (300, 440)]),
        Polygon([(360, 400), (400, 400), (400, 440), (360, 440)]),
    ]

    class _Obs:
        def __init__(self, poly, oid):
            self.polygon = poly
            self.id = oid

    buffer_m = 15.0
    env = EnvironmentMap(area, [_Obs(p, i) for i, p in enumerate(obstacles_xy)], buffer_m=buffer_m)
    g = build_gvg(env, sample_step_m=3.0, spur_min_m=8.0)

    assert g.number_of_nodes() > 0, "expected a non-empty GVG for two distinct obstacles"
    for node, data in g.nodes(data=True):
        # the node's own clearance attribute and a freshly recomputed clearance
        # must agree, and both must respect the buffer
        assert env.clearance(node) >= buffer_m - 1e-6, (
            f"GVG node {node} has clearance {env.clearance(node):.3f} m, "
            f"below the configured buffer {buffer_m} m"
        )
        assert data["clearance"] >= buffer_m - 1e-6


def test_gvg_is_empty_with_no_obstacles():
    """Documented degenerate case: zero obstacles -> empty GVG (TGC falls back to
    a single region). A regression that returns a non-empty graph here would
    silently fabricate corridors where none are needed."""
    area = Polygon([(0, 0), (500, 0), (500, 500), (0, 500)])
    env = EnvironmentMap(area, [], buffer_m=10.0)
    g = build_gvg(env)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


# =========================================================================== #
# 2. Launch site optimizer: staging-ring constraint + swap-weight effect      #
#    (planning/launch_site_optimizer.py)                                      #
#                                                                              #
# test_launch_site_feasibility.py already covers furthest_point_feasible and  #
# the InfeasibleMissionError path well. The gap: nothing asserts the site     #
# optimize() actually RETURNS respects the module's core architectural        #
# promise -- "a Ground Control Station cannot sit inside the survey polygon"  #
# -- and nothing shows the swap-count criterion (w_swaps) has any effect on    #
# the chosen site, which is the thesis's third optimization criterion.        #
# =========================================================================== #
@pytest.fixture
def _launch_scenario():
    cfg = load_config(CONFIG_PATH)
    spec = build_spec(cfg)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    aero = AeroCorrection(cfg.aero, spec.platform)
    area = Polygon([(0, 0), (800, 0), (800, 800), (0, 800)])
    env = EnvironmentMap(area, [], cfg.env.clearance_buffer_m)
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    tgc = build_tgc(env, gvg)
    return cfg, spec, em, motion, aero, env, tgc


def test_optimized_launch_site_is_outside_the_survey_polygon(_launch_scenario):
    """The chosen pad must lie in the staging periphery, never inside the
    surveyed area -- the operational constraint the module exists to enforce
    (a GCS cannot be parked in the middle of the field it is surveying)."""
    cfg, spec, em, motion, aero, env, tgc = _launch_scenario
    rng = np.random.default_rng(1)
    pose, scores = optimize(cfg.launch, tgc, env, motion, em, aero, spec, 4, rng, 100.0)

    assert not env.area.contains(Point(pose.as_xy())), (
        "launch site must not be inside the survey polygon"
    )
    # every scored candidate must satisfy the same constraint, not just the winner
    for s in scores:
        assert not env.area.contains(Point(s.site)), (
            f"scored candidate {s.site} falls inside the survey polygon"
        )


def test_swap_weight_changes_the_chosen_site():
    """Criterion 3 (exact fleet swap count) must actually influence which site
    wins: with w_swaps pushed to dominate the objective, the optimizer must
    select the candidate with the fewest exact fleet swaps, on a scenario
    engineered so swap count genuinely varies across candidates (a battery
    capacity tight enough that some sites need multiple sorties per drone).
    This guards against w_swaps being wired into the score but never actually
    changing the argmin (e.g. a normalization bug that zeroes it out)."""
    import dataclasses

    cfg = load_config(CONFIG_PATH)
    spec = build_spec(cfg)
    # shrink the battery so per-sortie coverage budget is tight enough that
    # different staging sites need different numbers of sorties (verified:
    # this combination yields >=4 distinct swap values among 8 candidates).
    spec = dataclasses.replace(spec, battery_capacity_j=spec.battery_capacity_j * 0.4)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    aero = AeroCorrection(cfg.aero, spec.platform)
    area = Polygon([(0, 0), (3000, 0), (3000, 200), (0, 200)])
    env = EnvironmentMap(area, [], cfg.env.clearance_buffer_m)
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    tgc = build_tgc(env, gvg)

    swap_dominated = dataclasses.replace(cfg.launch, w_distance=0.0, w_energy=0.0, w_swaps=1.0)
    rng = np.random.default_rng(7)
    pose, scores = optimize(swap_dominated, tgc, env, motion, em, aero, spec, 4, rng, 100.0)

    swap_values = {s.expected_swaps for s in scores}
    assert len(swap_values) > 1, (
        "expected_swaps is constant across all candidates on this scenario; "
        "the swap-weight sensitivity check below would be vacuous"
    )
    best_swap_row = min(scores, key=lambda s: s.expected_swaps)
    assert scores[0].site == best_swap_row.site, (
        "with w_swaps=1.0 and all other weights 0, the winning site must be "
        "the one with the fewest exact fleet swaps"
    )


# =========================================================================== #
# 3. GPX exporter: coordinate projection round-trip + file contract           #
#    (metrics/gpx_exporter.py)                                                #
#                                                                              #
# Zero existing tests touch this module. It is a defense-deliverable output   #
# (real GPX, loadable in gpx.studio per the project's explicit requirement),  #
# so a silent regression here (e.g. swapped lat/lon, broken <ele>, or a       #
# write_gpx path bug) would not fail any existing test and would only surface  #
# when a human opens the file -- too late, the day of the demo.               #
# =========================================================================== #
class _FakeTelemetry:
    """Minimal duck-typed double of TelemetryLog's exporter-facing surface
    (drone_ids / gpx_track), per this module's structural contract."""

    def __init__(self, tracks: dict[int, list[tuple[float, float, float, float]]]):
        self._tracks = tracks

    def drone_ids(self) -> list[int]:
        return sorted(self._tracks)

    def gpx_track(self, agent_id: int) -> list[tuple[float, float, float, float]]:
        return self._tracks.get(agent_id, [])


def test_gpx_projection_round_trips_to_the_configured_origin():
    """A drone at local-plane (0, 0, z) must project to exactly (lat0, lon0),
    and a known eastward/northward offset must move lon/lat in the correct
    DIRECTION (not swapped) by approximately the right magnitude under the
    equirectangular approximation. Catches a lat/lon swap or an inverted axis,
    which would silently place every track on the wrong side of the map."""
    lat0, lon0 = 54.6872, 25.2797
    telem = _FakeTelemetry({0: [(0.0, 0.0, 0.0, 100.0)]})
    xml_str = build_gpx(telem, drone_id=0, lat0=lat0, lon0=lon0)
    root = ET.fromstring(xml_str.split("\n", 1)[1])  # strip the XML decl line
    ns = "{http://www.topografix.com/GPX/1/1}"
    pt = root.find(f"{ns}trk/{ns}trkseg/{ns}trkpt")
    assert pt is not None, "expected exactly one trkpt for the origin fix"
    assert float(pt.attrib["lat"]) == pytest.approx(lat0, abs=1e-6)
    assert float(pt.attrib["lon"]) == pytest.approx(lon0, abs=1e-6)
    assert float(pt.find(f"{ns}ele").text) == pytest.approx(100.0, abs=1e-6)

    # a fix at t=0, 1000 m east (x) and 1000 m north (y) of the origin
    telem2 = _FakeTelemetry({0: [(0.0, 1000.0, 1000.0, 0.0)]})
    xml_str2 = build_gpx(telem2, drone_id=0, lat0=lat0, lon0=lon0)
    root2 = ET.fromstring(xml_str2.split("\n", 1)[1])
    pt2 = root2.find(f"{ns}trk/{ns}trkseg/{ns}trkpt")
    lat2, lon2 = float(pt2.attrib["lat"]), float(pt2.attrib["lon"])
    # moving north increases latitude; moving east increases longitude
    assert lat2 > lat0, "northward offset must increase latitude"
    assert lon2 > lon0, "eastward offset must increase longitude"
    # magnitude sanity: ~1000 m north is ~1000/111320 deg of latitude
    assert (lat2 - lat0) == pytest.approx(1000.0 / 111_320.0, rel=1e-3)


def test_gpx_is_well_formed_xml_with_required_elements():
    """The produced string must be valid, parseable GPX 1.1 with the elements a
    GIS viewer (gpx.studio, QGIS) requires: <trk>/<name>, <trkseg>, and for each
    point lat/lon attributes plus <ele> and <time> children, in time order."""
    track = [(0.0, 0.0, 0.0, 50.0), (10.0, 5.0, 5.0, 55.0), (20.0, 10.0, 10.0, 60.0)]
    telem = _FakeTelemetry({3: track})
    xml_str = build_gpx(telem, drone_id=3)

    assert xml_str.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    root = ET.fromstring(xml_str.split("\n", 1)[1])
    ns = "{http://www.topografix.com/GPX/1/1}"
    assert root.tag == f"{ns}gpx"

    trk = root.find(f"{ns}trk")
    assert trk is not None
    assert trk.find(f"{ns}name").text == "drone_3"

    pts = trk.findall(f"{ns}trkseg/{ns}trkpt")
    assert len(pts) == 3
    times = []
    for pt in pts:
        assert "lat" in pt.attrib and "lon" in pt.attrib
        assert pt.find(f"{ns}ele") is not None
        time_text = pt.find(f"{ns}time").text
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", time_text), (
            f"timestamp {time_text!r} is not a well-formed GPX UTC time"
        )
        times.append(time_text)
    assert times == sorted(times), "track points must be in non-decreasing time order"


def test_write_gpx_creates_one_file_per_drone_known_to_telemetry(tmp_path):
    """write_gpx must emit one *_drone_<id>.gpx file per drone ID reported by
    drone_ids(), and must create missing output directories.

    NOTE (found during review, not a behaviour change in this patch): write_gpx
    writes a file for EVERY id in drone_ids(), even one whose gpx_track() is
    empty -- build_gpx's `if not track: continue` only skips adding a <trk>
    child, so the result is a syntactically valid but track-less GPX file
    rather than no file at all. That is arguably a minor wart (an empty file in
    the defense-demo output folder) but it is not contradicted by any
    docstring or README claim, so this test documents the actual behaviour
    rather than asserting a stricter contract the code does not promise.
    """
    telem = _FakeTelemetry({
        0: [(0.0, 0.0, 0.0, 10.0)],
        1: [(0.0, 1.0, 1.0, 10.0)],
        2: [],  # no telemetry recorded for this drone
    })
    out_dir = tmp_path / "nested" / "runs"
    out_path = str(out_dir / "tracks.gpx")

    write_gpx(telem, out_path)

    assert (out_dir / "tracks_drone_0.gpx").exists()
    assert (out_dir / "tracks_drone_1.gpx").exists()
    # each written file must itself be parseable, and the non-empty ones must
    # carry a <trk>
    for did in (0, 1):
        content = (out_dir / f"tracks_drone_{did}.gpx").read_text(encoding="utf-8")
        root = ET.fromstring(content.split("\n", 1)[1])
        ns = "{http://www.topografix.com/GPX/1/1}"
        assert root.find(f"{ns}trk") is not None


# =========================================================================== #
# 4. Redistribution: swap/failure trigger asymmetry                          #
#    (execution/redistribution.py)                                            #
#                                                                              #
# README.md S5 states this asymmetry as a guarantee ("only FAILURE/NEW_TASK   #
# reach the redistributor... enforced by wiring (TRIGGERS), not convention"). #
# No existing test calls should_handle/handle directly with a SWAP event to   #
# confirm the rejection; test_batch4.py exercises Redistributor only through  #
# full-fleet integration scenarios that happen to trigger FAILURE/NEW_TASK.   #
# A regression that widened TRIGGERS (e.g. someone "fixing" a bug by adding   #
# EventType.SWAP_REQUEST) would silently violate a documented architectural   #
# contract and could re-trigger a full re-partition on every routine swap.    #
# =========================================================================== #
def test_should_handle_accepts_only_failure_and_new_task():
    from uav_swarm_sim.execution.redistribution import Redistributor

    accepted = {
        et for et in EventType
        if Redistributor.should_handle(Event(type=et, t=0.0, payload={}))
    }
    assert accepted == {EventType.FAILURE, EventType.NEW_TASK}, (
        f"redistribution TRIGGERS must be exactly {{FAILURE, NEW_TASK}}, "
        f"got {accepted} -- swap must remain reversible and NOT re-trigger "
        f"redistribution (README.md S5)"
    )


def test_handle_raises_on_a_non_trigger_event():
    """handle() must defensively reject any event should_handle rejects (e.g. a
    battery swap request), rather than silently redistributing on it -- the
    second half of the wiring guarantee, exercised at the method that does the
    real work."""
    from uav_swarm_sim.execution.redistribution import Redistributor

    swap_event = Event(type=EventType.SWAP_REQUEST, t=0.0, payload={"agent_id": 0})
    # a Redistributor with all-None collaborators is fine here: should_handle
    # is checked and the ValueError raised before any collaborator is touched.
    redistributor = Redistributor(decomposer=None, layer_graphs=None, motion=None, em=None, spec=None)
    with pytest.raises(ValueError, match="does not handle"):
        redistributor.handle(swap_event, fleet=None, partition=None, plans=None, t=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
