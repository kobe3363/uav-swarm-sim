"""Regression tests for the Batch-3 KDTree separation refactor.

``SafetyMonitor`` used to detect predicted drone-vs-drone separation conflicts
with a per-agent O(n^2) pairwise loop. Batch 3 replaces it with a single per-tick
``scipy.spatial.KDTree.query_pairs`` neighbour query (``_separation_yielders``),
computed once and consulted by membership. The refactor is a pure performance
change: the set of yielding drones must be **byte-identical** to the old loop.

The core invariant locked here: ``_separation_yielders`` equals an independent
brute-force evaluation of the exact former predicate -- only NON-formation
(S1/S3-excluded) drones participate, only same-LAYER pairs are compared, poses
are compared at the SAME predicted timestep (the old ``zip``), the strict
``< min_separation`` boundary is preserved, and the LOWER-id drone of each
conflicting pair yields. Verified on crafted edge cases plus heavy random fuzz,
and at the ``step`` level on concrete converging-drone scenarios.

Self-contained: ``SafetyMonitor`` reads only ``.id/.state/.layer/.pose`` (and
``._legs/._leg_idx/._t`` for prediction) off each agent and calls
``signal_threat`` on it, so duck-typed stand-ins suffice -- no full Agent / engine
fixture is needed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import SafetyConfig
from uav_swarm_sim.infrastructure.core_types import Path, Pose, straight_segment
from uav_swarm_sim.infrastructure.enums import AgentState, ManeuverType
from uav_swarm_sim.execution.safety_monitor import SafetyMonitor
from uav_swarm_sim.planning.environment_map import EnvironmentMap

_SEP = 15.0
_FORMATION = (AgentState.S1_TRANSIT, AgentState.S3_RTH)
_AREA = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])


@dataclass
class _Agent:
    """Stand-in carrying only what SafetyMonitor reads."""
    id: int
    state: AgentState
    layer: int = 0
    pose: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0))
    _legs: list = field(default_factory=list)
    _leg_idx: int = 0
    _t: float = 0.0
    calls: list = field(default_factory=list)

    def signal_threat(self, on, avoidance=None, skip_leg=False):
        self.calls.append((on, avoidance is not None, skip_leg))


class _Aero:
    def wake_zones(self, leaders):
        return []


class _Motion:
    def plan(self, a, b, m):
        d = math.dist((a.x, a.y), (b.x, b.y))
        if d <= 1e-9:
            return Path()
        h = math.atan2(b.y - a.y, b.x - a.x)
        return Path.from_segments([straight_segment(Pose(a.x, a.y, h, a.z), d, m, 12.0)])


class _Bus:
    def __init__(self):
        self.ids = []

    def publish(self, e):
        self.ids.append(e.payload.get("agent_id"))


def _monitor(obstacles=(), recovery=False):
    env = EnvironmentMap(_AREA, list(obstacles), buffer_m=5.0)
    cfg = SafetyConfig(min_separation_m=_SEP, obstacle_buffer_m=5.0,
                       predict_horizon_s=5.0, obstacle_recovery=recovery)
    return SafetyMonitor(env, _Aero(), cfg, _Motion())


def _bruteforce_yielders(agents, preds, sep):
    """Independent reference: the exact former O(n^2) separation predicate."""
    out = set()
    for a in agents:
        if a.state in _FORMATION:
            continue
        a_layer = getattr(a, "layer", 0)
        for b in agents:
            if b.id <= a.id:
                continue
            if getattr(b, "layer", 0) != a_layer:
                continue
            if b.state in _FORMATION:
                continue
            for pa, pb in zip(preds[a.id], preds[b.id]):
                if math.dist(pa.as_xy(), pb.as_xy()) < sep:
                    out.add(a.id)  # lower-id yields
                    break
    return out


# --------------------------------------------------------------------------- #
# crafted edge cases                                                          #
# --------------------------------------------------------------------------- #
def test_separation_yielders_crafted_edges():
    mon = _monitor()
    S2 = AgentState.S2_MISSION

    def check(agents, preds, expected):
        assert mon._separation_yielders(agents, preds) == expected
        assert _bruteforce_yielders(agents, preds, _SEP) == expected

    # exact min_sep apart -> EXCLUDED (strict <)
    a = [_Agent(0, S2), _Agent(1, S2)]
    check(a, {0: [Pose(100, 100, 0)], 1: [Pose(100 + _SEP, 100, 0)]}, set())
    # just inside -> the lower id (0) yields
    check(a, {0: [Pose(100, 100, 0)], 1: [Pose(100 + _SEP - 1e-6, 100, 0)]}, {0})
    # different layers, coincident -> EXCLUDED
    a2 = [_Agent(0, S2, layer=0), _Agent(1, S2, layer=1)]
    check(a2, {0: [Pose(50, 50, 0)], 1: [Pose(50, 50, 0)]}, set())
    # both in formation phase -> EXCLUDED
    a3 = [_Agent(0, AgentState.S1_TRANSIT), _Agent(1, AgentState.S3_RTH)]
    check(a3, {0: [Pose(50, 50, 0)], 1: [Pose(51, 50, 0)]}, set())
    # one S2 one formation -> EXCLUDED (formation peer skipped)
    a4 = [_Agent(0, S2), _Agent(1, AgentState.S1_TRANSIT)]
    check(a4, {0: [Pose(50, 50, 0)], 1: [Pose(51, 50, 0)]}, set())
    # time-aligned: conflict only at k=1 but agent 0 has a single-pose prediction
    # (spent leg) -> zip truncates, no conflict seen
    a5 = [_Agent(0, S2), _Agent(1, S2)]
    check(a5, {0: [Pose(0, 0, 0)], 1: [Pose(500, 500, 0), Pose(0, 0, 0)]}, set())
    # lower id yields even with non-contiguous ids
    a6 = [_Agent(3, S2), _Agent(7, S2)]
    check(a6, {3: [Pose(200, 200, 0)], 7: [Pose(202, 200, 0)]}, {3})


# --------------------------------------------------------------------------- #
# random equivalence fuzz                                                     #
# --------------------------------------------------------------------------- #
def test_separation_yielders_matches_bruteforce_random():
    mon = _monitor()
    states = [AgentState.S1_TRANSIT, AgentState.S2_MISSION, AgentState.S3_RTH]
    rng = random.Random(20260621)
    for _ in range(4000):
        n = rng.randint(2, 14)
        agents = [_Agent(i, rng.choice(states), layer=rng.choice([0, 0, 0, 1, 2]))
                  for i in range(n)]
        cx, cy = rng.uniform(50, 350), rng.uniform(50, 350)
        preds = {}
        for a in agents:
            ln = rng.choice([1, 5, 5, 5])
            base = (cx, cy) if rng.random() < 0.5 else (rng.uniform(0, 400), rng.uniform(0, 400))
            preds[a.id] = [Pose(base[0] + rng.uniform(-15, 15),
                                base[1] + rng.uniform(-15, 15), 0.0) for _ in range(ln)]
        assert mon._separation_yielders(agents, preds) == _bruteforce_yielders(agents, preds, _SEP)


def test_separation_yielders_is_deterministic():
    mon = _monitor()
    agents = [_Agent(i, AgentState.S2_MISSION) for i in range(6)]
    preds = {i: [Pose(100 + i, 100, 0), Pose(100 + i, 101, 0)] for i in range(6)}
    first = mon._separation_yielders(agents, preds)
    for _ in range(5):
        assert mon._separation_yielders(agents, preds) == first


# --------------------------------------------------------------------------- #
# step-level integration                                                       #
# --------------------------------------------------------------------------- #
def _straight_leg(pose, length):
    return Path.from_segments([straight_segment(pose, length, ManeuverType.CRUISE, 12.0)])


def test_step_two_converging_drones_lower_id_yields():
    """Two drones predicted to pass within min_sep: the lower-id drone is the one
    signalled (yields); the higher-id drone is not."""
    mon = _monitor()
    a0 = _Agent(0, AgentState.S2_MISSION, pose=Pose(0, 0, 0.0))
    a1 = _Agent(1, AgentState.S2_MISSION, pose=Pose(30, 0, math.pi))
    a0._legs = [_straight_leg(a0.pose, 60.0)]   # heads +x
    a1._legs = [_straight_leg(a1.pose, 60.0)]   # heads -x, so they cross
    mon.step([a0, a1], 0.0, _Bus())
    assert a0.calls and a0.calls[-1][0] is True, "lower-id drone should yield"
    assert not a1.calls, "higher-id drone should not be signalled"


def test_step_cross_layer_no_separation_threat():
    """Same converging geometry but on different layers -> vertically separated,
    so neither drone is signalled for separation."""
    mon = _monitor()
    a0 = _Agent(0, AgentState.S2_MISSION, layer=0, pose=Pose(0, 0, 0.0))
    a1 = _Agent(1, AgentState.S2_MISSION, layer=1, pose=Pose(30, 0, math.pi))
    a0._legs = [_straight_leg(a0.pose, 60.0)]
    a1._legs = [_straight_leg(a1.pose, 60.0)]
    mon.step([a0, a1], 0.0, _Bus())
    assert not a0.calls and not a1.calls


def test_step_formation_phase_drones_ignored():
    """Two drones in transit (formation phase) on a collision course are spaced by
    the FormationManager, not the collision monitor -> not signalled."""
    mon = _monitor()
    a0 = _Agent(0, AgentState.S1_TRANSIT, pose=Pose(0, 0, 0.0))
    a1 = _Agent(1, AgentState.S1_TRANSIT, pose=Pose(30, 0, math.pi))
    a0._legs = [_straight_leg(a0.pose, 60.0)]
    a1._legs = [_straight_leg(a1.pose, 60.0)]
    mon.step([a0, a1], 0.0, _Bus())
    assert not a0.calls and not a1.calls
