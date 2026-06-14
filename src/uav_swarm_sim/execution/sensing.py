"""Swarm-level sensing coordinator: the passive/active mode logic.

Each tick (only when the feature is enabled):
  * Compute the detection range from the CURRENT swarm mode (passive = short,
    active = long/LIDAR).
  * If ANY airborne drone has a dynamic obstacle within that range, the WHOLE
    swarm goes ACTIVE (or stays active and refreshes its hold timer). If nothing
    is detected for ``dynamic_hold_s`` while active, it reverts to PASSIVE.
  * Any drone with an obstacle inside the near-miss distance is signalled a
    threat, which drives the existing S_OBS graceful avoidance maneuver.

Energy: while ACTIVE the coordinator reports ``scan_power_w`` > 0, which the
engine drains from every airborne drone -- the high cost of proactive scanning.
If an obstacle appears but never enters any drone's passive range, it is never
detected and the swarm ignores it (an accepted outcome).

2.5D (Batch 4): detection is per layer -- a drone only senses obstacles in its
own coverage-layer band, since obstacles at other altitudes are vertically
separated. With one layer the filter is a no-op (everything on layer 0), so the
detection set, threats and mode transitions are byte-identical.
"""
from __future__ import annotations

import math

from ..infrastructure.config import DynamicObstacleConfig, SafetyConfig
from ..infrastructure.core_types import Event
from ..infrastructure.enums import EventType, SensingMode


class SensingCoordinator:
    def __init__(self, cfg: DynamicObstacleConfig, safety_cfg: SafetyConfig) -> None:
        self.cfg = cfg
        self.mode: SensingMode = SensingMode.PASSIVE
        self._near_miss_m = safety_cfg.min_separation_m
        self._last_detect_t = -1e18
        self._cooldown: dict[int, float] = {}
        self._cooldown_s = max(2.0, safety_cfg.predict_horizon_s)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def scan_power_w(self) -> float:
        return self.cfg.active_scan_power_w if self.mode is SensingMode.ACTIVE else 0.0

    def detection_range_m(self) -> float:
        return (self.cfg.active_sense_range_m if self.mode is SensingMode.ACTIVE
                else self.cfg.passive_sense_range_m)

    def step(self, agents, field, t: float, bus) -> None:
        if not self.cfg.enabled or field is None:
            return
        airborne = [a for a in agents if a.state.is_airborne and a.state.name != "S_OBS"]
        sense_r = self.detection_range_m()

        detected_any = False
        for a in airborne:
            a_layer = getattr(a, "layer", 0)
            nearest = None
            for o in field.obstacles:
                if getattr(o, "layer", 0) != a_layer:
                    continue  # obstacle in another altitude band -> not visible here
                d = math.hypot(a.pose.x - o.x, a.pose.y - o.y)
                if d <= sense_r:
                    detected_any = True
                if nearest is None or d < nearest[0]:
                    nearest = (d, o)
            # near-miss -> graceful avoidance (re-uses S_OBS), with a per-agent cooldown
            if nearest is not None:
                d, o = nearest
                if d <= self._near_miss_m + o.radius and t >= self._cooldown.get(a.id, -1.0):
                    a.signal_threat(True)
                    self._cooldown[a.id] = t + self._cooldown_s
                    bus.publish(Event(EventType.OBSTACLE_THREAT, t,
                                      {"agent_id": a.id, "dynamic_obstacle_id": o.id}))

        # swarm-wide mode transition
        if detected_any:
            self._last_detect_t = t
            if self.mode is SensingMode.PASSIVE:
                self.mode = SensingMode.ACTIVE   # provoked: the whole swarm starts scanning
        elif self.mode is SensingMode.ACTIVE and (t - self._last_detect_t) > self.cfg.dynamic_hold_s:
            self.mode = SensingMode.PASSIVE      # quiet long enough -> save power
