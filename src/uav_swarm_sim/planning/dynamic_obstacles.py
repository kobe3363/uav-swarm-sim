"""Drone-sized moving obstacles (e.g. birds).

A configurable number spawn at random free positions and travel in a straight
line at a fixed speed in a random direction, reflecting off the area boundary so
they stay in play. Seeded RNG -> reproducible. Position is advanced once per tick
by the engine; the SensingCoordinator decides who (if anyone) notices them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DynamicObstacle:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float


class DynamicObstacleField:
    def __init__(self, env, count: int, speed_m_s: float, size_m: float, rng: np.random.Generator) -> None:
        self.env = env
        self.radius = size_m / 2.0
        self._bounds = env.area.bounds  # (minx, miny, maxx, maxy)
        self.obstacles: list[DynamicObstacle] = []
        seeds = env.sample_free(count, rng) if count > 0 else []
        for i, (x, y) in enumerate(seeds):
            theta = float(rng.uniform(0.0, 2 * math.pi))
            self.obstacles.append(
                DynamicObstacle(i, float(x), float(y),
                                speed_m_s * math.cos(theta), speed_m_s * math.sin(theta),
                                self.radius)
            )

    def step(self, dt: float) -> None:
        minx, miny, maxx, maxy = self._bounds
        for o in self.obstacles:
            o.x += o.vx * dt
            o.y += o.vy * dt
            # reflect at the area bounding box (keeps obstacles within the theatre)
            if o.x < minx:
                o.x = minx + (minx - o.x); o.vx = -o.vx
            elif o.x > maxx:
                o.x = maxx - (o.x - maxx); o.vx = -o.vx
            if o.y < miny:
                o.y = miny + (miny - o.y); o.vy = -o.vy
            elif o.y > maxy:
                o.y = maxy - (o.y - maxy); o.vy = -o.vy

    def snapshot(self) -> list[tuple[int, float, float]]:
        return [(o.id, o.x, o.y) for o in self.obstacles]
