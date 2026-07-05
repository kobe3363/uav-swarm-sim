"""Regression tests for the Batch-2 exact segment-vs-obstacle geometry.

The fix replaced point-sampling in ``SafetyMonitor._chord_clear`` (which could
tunnel straight through an obstacle thinner than the sample spacing) with an
exact Shapely-segment test, exposed as ``EnvironmentMap.segment_in_obstacle``.

The core invariant: ``segment_in_obstacle`` is *exact* -- it agrees with an
arbitrarily dense point reference and never misses a crossing -- and the raw
(unbuffered) semantics of the former ``in_obstacle`` sampling are preserved
(distinct from the stronger buffered ``segment_clear`` used for trajectory
validation).

These feed ``EnvironmentMap`` a duck-typed obstacle stand-in (the map reads only
``.polygon`` and ``.id``) so the test does not depend on the concrete
``obstacle_generator.Obstacle`` constructor.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from shapely.geometry import Polygon

from uav_swarm_sim.execution.safety_monitor import SafetyMonitor
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.planning.environment_map import EnvironmentMap


@dataclass(frozen=True)
class _Obs:
    """Minimal obstacle stand-in: EnvironmentMap reads only .polygon and .id."""
    polygon: Polygon
    id: int = 0


_AREA = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


def _rect(cx, cy, w, h):
    return Polygon([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                    (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)])


def _dense_blocked(a: Pose, b: Pose, env: EnvironmentMap, n: int = 2000) -> bool:
    """Independent reference: dense point sampling of the raw-penetration test."""
    for k in range(n + 1):
        x = a.x + (b.x - a.x) * k / n
        y = a.y + (b.y - a.y) * k / n
        if env.in_obstacle((x, y)):
            return True
    return False


def test_segment_in_obstacle_catches_thin_wall():
    """A 1 m wall placed between two 12.5 m-spaced samples is tunneled through by
    point sampling but caught exactly. (buffer 0 isolates the sampling effect.)"""
    wall = _Obs(_rect(55.5, 50, 1.0, 100), id=1)
    env = EnvironmentMap(_AREA, [wall], buffer_m=0.0)
    through = (Pose(0.0, 50.0, 0.0), Pose(100.0, 50.0, 0.0))
    around = (Pose(0.0, 10.0, 0.0), Pose(40.0, 10.0, 0.0))   # stays left of the wall
    assert env.segment_in_obstacle(*through) is True
    assert env.segment_in_obstacle(*around) is False
    # confirm the old 9 samples really would have missed it
    xs = [through[0].x + (through[1].x - through[0].x) * k / 8 for k in range(9)]
    assert not any(55 <= x <= 56 for x in xs)


def test_segment_in_obstacle_matches_dense_reference():
    """Exactness: over a random battery, segment_in_obstacle equals a dense
    point reference -- no false alarms, and critically no misses."""
    rng = random.Random(20260621)
    misses = false_alarms = 0
    for _ in range(150):
        obs = []
        for oid in range(rng.randint(1, 4)):
            cx, cy = rng.uniform(15, 85), rng.uniform(15, 85)
            obs.append(_Obs(_rect(cx, cy, rng.uniform(1.0, 12.0),
                                  rng.uniform(1.0, 12.0)), oid))
        env = EnvironmentMap(_AREA, obs, buffer_m=0.0)
        a = Pose(rng.uniform(0, 100), rng.uniform(0, 100), 0.0)
        b = Pose(rng.uniform(0, 100), rng.uniform(0, 100), 0.0)
        if math.hypot(b.x - a.x, b.y - a.y) < 1.0:
            continue
        exact = env.segment_in_obstacle(a, b)
        dense = _dense_blocked(a, b, env, n=1000)
        if exact and not dense:
            false_alarms += 1
        if dense and not exact:
            misses += 1
    assert misses == 0, f"segment_in_obstacle missed {misses} crossings the dense ref found"
    assert false_alarms == 0, f"segment_in_obstacle false-alarmed {false_alarms} times"


def test_segment_in_obstacle_empty_field_is_never_blocked():
    env = EnvironmentMap(_AREA, [], buffer_m=5.0)
    assert env.segment_in_obstacle(Pose(1, 1, 0.0), Pose(99, 99, 0.0)) is False


def test_segment_in_obstacle_is_raw_not_buffered():
    """Intentional semantics: segment_in_obstacle tests the RAW polygon, while
    segment_clear tests the buffered union. A chord skimming the buffer but not
    the raw obstacle is 'in obstacle' = False yet 'clear' = False."""
    obs = _Obs(_rect(50, 50, 4, 4), id=1)                 # raw square y in [48,52]
    env = EnvironmentMap(_AREA, [obs], buffer_m=5.0)
    a, b = Pose(1.0, 45.0, 0.0), Pose(99.0, 45.0, 0.0)    # 3 m below raw -> inside 5 m buffer
    assert env.segment_in_obstacle(a, b) is False          # does not penetrate raw
    assert env.segment_clear(a, b) is False                # but violates buffered clearance


def test_chord_clear_delegates_to_exact_test():
    """SafetyMonitor._chord_clear now returns the exact result. Constructed with
    None dependencies because _chord_clear consumes only its ``env`` argument."""
    mon = SafetyMonitor(None, None, None, None)
    wall = _Obs(_rect(55.5, 50, 1.0, 100), id=1)
    env = EnvironmentMap(_AREA, [wall], buffer_m=0.0)
    assert mon._chord_clear(Pose(0.0, 50.0, 0.0), Pose(100.0, 50.0, 0.0), env) is False
    assert mon._chord_clear(Pose(0.0, 10.0, 0.0), Pose(40.0, 10.0, 0.0), env) is True
