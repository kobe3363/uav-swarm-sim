"""Micro-benchmark for the Batch-3 separation refactor.

Times the former O(n^2) pairwise separation scan against the new per-tick
``SafetyMonitor._separation_yielders`` (KDTree.query_pairs) at increasing fleet
sizes, and asserts the two produce the identical yield set at every size. The
former loop is reproduced inline here as the baseline (it no longer exists in the
monitor). Run:  ``python -m uav_swarm_sim.experiments.bench_separation``
(or ``python bench_separation.py`` from a checkout root).
"""
from __future__ import annotations

import math
import random
import time

from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import SafetyConfig
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import AgentState
from uav_swarm_sim.execution.safety_monitor import SafetyMonitor
from uav_swarm_sim.planning.environment_map import EnvironmentMap

_SEP = 15.0
_AREA = Polygon([(0, 0), (4000, 0), (4000, 4000), (0, 4000)])
_FORMATION = (AgentState.S1_TRANSIT, AgentState.S3_RTH)


class _Agent:
    __slots__ = ("id", "state", "layer", "pose")

    def __init__(self, id, pose):
        self.id = id
        self.state = AgentState.S2_MISSION
        self.layer = 0
        self.pose = pose


class _Aero:
    def wake_zones(self, leaders):
        return []


def _bruteforce_yielders(agents, preds, sep):
    """The former O(n^2) pairwise separation predicate (baseline)."""
    out = set()
    for a in agents:
        if a.state in _FORMATION:
            continue
        a_layer = a.layer
        for b in agents:
            if b.id <= a.id:
                continue
            if b.layer != a_layer:
                continue
            if b.state in _FORMATION:
                continue
            for pa, pb in zip(preds[a.id], preds[b.id]):
                if math.dist(pa.as_xy(), pb.as_xy()) < sep:
                    out.add(a.id)
                    break
    return out


def _scene(n, rng):
    """n drones, single layer, 5 predicted poses each; a realistic mix of a few
    tight clusters (where conflicts actually occur) plus spread-out traffic."""
    clusters = [(rng.uniform(200, 3800), rng.uniform(200, 3800)) for _ in range(max(1, n // 12))]
    agents, preds = [], {}
    for i in range(n):
        if rng.random() < 0.4:
            cx, cy = rng.choice(clusters)
            x, y = cx + rng.uniform(-25, 25), cy + rng.uniform(-25, 25)
        else:
            x, y = rng.uniform(0, 4000), rng.uniform(0, 4000)
        agents.append(_Agent(i, Pose(x, y, 0.0)))
        preds[i] = [Pose(x + rng.uniform(-20, 20), y + rng.uniform(-20, 20), 0.0) for _ in range(5)]
    return agents, preds


def _time(fn, scenes, repeat):
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for agents, preds in scenes:
            fn(agents, preds)
        best = min(best, time.perf_counter() - t0)
    return best / len(scenes)


def main():
    env = EnvironmentMap(_AREA, [], buffer_m=5.0)
    cfg = SafetyConfig(min_separation_m=_SEP, obstacle_buffer_m=5.0, predict_horizon_s=5.0)
    mon = SafetyMonitor(env, _Aero(), cfg, None)
    rng = random.Random(7)

    print(f"{'n':>6} {'O(n^2) ms/tick':>16} {'KDTree ms/tick':>16} {'speedup':>9}  identical")
    print("-" * 62)
    for n in (10, 25, 50, 100, 200, 400, 800):
        scenes = [_scene(n, rng) for _ in range(25)]
        # correctness: identical yield set on every scene
        ok = all(_bruteforce_yielders(a, p, _SEP) == mon._separation_yielders(a, p)
                 for a, p in scenes)
        rep = 5 if n <= 200 else 2
        t_bf = _time(lambda a, p: _bruteforce_yielders(a, p, _SEP), scenes, rep) * 1e3
        t_kd = _time(lambda a, p: mon._separation_yielders(a, p), scenes, rep) * 1e3
        print(f"{n:>6} {t_bf:>16.3f} {t_kd:>16.3f} {t_bf / t_kd:>8.1f}x  {ok}")


if __name__ == "__main__":
    main()
