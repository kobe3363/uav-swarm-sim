"""FIX-B1 -- obstacle-aware S1 transit routing (``coverage.transit_free_space``).

The livelock root cause (see docs / the demand-probe diagnosis): S1 transit legs
are blind straight CRUISE chords; a chord blocked by an obstacle prism traps the
drone in an S_OBS -> boxed-in -> RTH -> swap -> identical-chord macro-loop.
``route_transit`` is the plan-time cure: the CRUISE twin of ``route_connector``
(same visibility-graph internals, same fallback semantics), used for the initial
assign transit, ``Agent._resume_transit`` and redistribution re-transit when the
flag is on.

Covers:
  * flag OFF -> exactly the straight ``motion.plan(a, b, CRUISE)`` chord
    (byte-identical geometry, the default path);
  * flag ON + clear chord -> still exactly the straight chord;
  * flag ON + blocked chord -> a detour that never crosses the raw obstacle,
    is never shorter than the chord, and ends at the requested pose;
  * config schema: ``transit_free_space`` absent from default.yaml -> False.
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon, box

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.planning.visibility_router import route_transit

AREA = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
BUFFER_M = 5.0
A = Pose(200.0, 500.0, 0.0)
B = Pose(800.0, 500.0, 0.0)


def _motion(config_path):
    return make_motion_model(build_spec(load_config(config_path)))


def _env(obstacles):
    return EnvironmentMap(AREA, obstacles, BUFFER_M)


def _same_geometry(p1, p2) -> bool:
    """Two Paths realize the same motion: identical sampled polylines."""
    s1, s2 = p1.sample(5.0), p2.sample(5.0)
    if len(s1) != len(s2):
        return False
    return all(math.dist(q1.as_xy(), q2.as_xy()) < 1e-9 for q1, q2 in zip(s1, s2))


def test_flag_off_is_the_straight_cruise_chord(config_path):
    motion = _motion(config_path)
    env = _env([Obstacle(id=0, cls=0, polygon=box(450, 400, 550, 600))])
    chord = motion.plan(A, B, ManeuverType.CRUISE)
    off = route_transit(A, B, motion, env, enabled=False)
    assert off.total_length_m == chord.total_length_m
    assert off.total_duration_s == chord.total_duration_s
    assert _same_geometry(off, chord)


def test_flag_on_clear_chord_unchanged(config_path):
    motion = _motion(config_path)
    env = _env([Obstacle(id=0, cls=0, polygon=box(450, 700, 550, 900))])  # off-chord
    chord = motion.plan(A, B, ManeuverType.CRUISE)
    on = route_transit(A, B, motion, env, enabled=True)
    assert on.total_length_m == chord.total_length_m
    assert _same_geometry(on, chord)


def test_flag_on_blocked_chord_detours_clear_of_obstacle(config_path):
    motion = _motion(config_path)
    env = _env([Obstacle(id=0, cls=0, polygon=box(450, 400, 550, 600))])
    chord = motion.plan(A, B, ManeuverType.CRUISE)
    assert env.segment_in_obstacle(A, B)  # the premise: the chord is blocked
    routed = route_transit(A, B, motion, env, enabled=True)
    # a detour is a real reroute: longer than the chord, same endpoints
    assert routed.total_length_m > chord.total_length_m
    end = routed.end_pose
    assert math.dist(end.as_xy(), B.as_xy()) < 1e-6
    # and it never crosses the raw obstacle (the SafetyMonitor S_OBS predicate)
    pts = routed.sample(1.0)
    assert all(
        not env.segment_in_obstacle(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
    )


def test_transit_free_space_defaults_off(config_path):
    cfg = load_config(config_path)
    assert cfg.coverage.transit_free_space is False
