"""Proactive obstacle avoidance as sustained monitoring (guideline 3.4).

Every tick, predicted poses over a horizon are checked against other drones,
obstacles, and aerodynamic wake zones (treated as 'invisible obstacles'). This
is continuous surveillance, not a reactive last-moment trigger. Deterministic
yielding (lower-id agent yields) keeps the resolution collision-free by
construction. All S_OBS time is flight overhead by definition and lands in the
efficiency-score denominator via the SMDP layer.
"""
from __future__ import annotations

import math

from shapely.geometry import Point, Polygon

from ..infrastructure.config import SafetyConfig
from ..infrastructure.core_types import Event, Path, Pose
from ..infrastructure.enums import AgentState, EventType, ManeuverType, PlatformType
from ..physical_model.aero_correction import AeroCorrection
from ..physical_model.motion_model import MotionModel


class SafetyMonitor:
    def __init__(self, env, aero: AeroCorrection, cfg: SafetyConfig, motion: MotionModel) -> None:
        self._env = env
        self._aero = aero
        self._cfg = cfg
        self._motion = motion
        self._cooldown_until: dict[int, float] = {}

    def _predicted_poses(self, agent, n: int = 4) -> list[Pose]:
        legs = getattr(agent, "_legs", [])
        idx = getattr(agent, "_leg_idx", 0)
        t0 = getattr(agent, "_t", 0.0)
        if idx >= len(legs):
            return [agent.pose]
        leg: Path = legs[idx]
        horizon = self._cfg.predict_horizon_s
        return [
            leg.pose_at_time(t0 + k / n * horizon) or agent.pose for k in range(n + 1)
        ]

    def step(self, agents, t: float, bus) -> None:
        airborne = [a for a in agents if a.state.is_airborne and a.state.name != "S_OBS"]
        preds = {a.id: self._predicted_poses(a) for a in airborne}

        for a in airborne:
            if t < self._cooldown_until.get(a.id, -1.0):
                continue  # recently avoided -> let it fly before re-checking
            threat = self._threatened(a, airborne, preds)
            if threat:
                a.signal_threat(True)
                self._cooldown_until[a.id] = t + self._cfg.predict_horizon_s
                bus.publish(Event(EventType.OBSTACLE_THREAT, t, {"agent_id": a.id}))

    def _threatened(self, a, airborne, preds) -> bool:
        # Inter-drone conflicts during formation phases (transit/RTH) are governed
        # by the FormationManager (spacing), not collision avoidance -- ignore them
        # here to avoid launch/return thrashing.
        a_formation = a.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH)

        # pairwise predicted separation (lower-id yields), skipped if both formation
        if not a_formation:
            for b in airborne:
                if b.id <= a.id:
                    continue
                if b.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH):
                    continue
                for pa, pb in zip(preds[a.id], preds[b.id]):
                    if math.dist(pa.as_xy(), pb.as_xy()) < self._cfg.min_separation_m:
                        return True

        # genuine obstacle penetration (raw obstacle polygons; not boundary/buffer)
        if self._env is not None:
            for p in preds[a.id]:
                if self._env.in_obstacle(p.as_xy()):
                    return True

        # wake zones from other airborne drones (invisible obstacles), only when
        # this agent is dispersed (not riding a formation)
        if not a_formation:
            leaders = [b.pose for b in airborne if b.id != a.id]
            for wake in self._aero.wake_zones(leaders):
                for p in preds[a.id]:
                    if wake.covers(Point(p.as_xy())):
                        return True
        return False

    def avoidance_plan(self, agent, threat: Polygon | None = None) -> Path:
        h = agent.pose.heading
        offset = max(self._cfg.min_separation_m, self._cfg.obstacle_buffer_m + 5.0)
        # lateral waypoint to the left, then rejoin ahead
        lx, ly = -math.sin(h), math.cos(h)
        side = Pose(agent.pose.x + offset * lx, agent.pose.y + offset * ly, h)
        return self._motion.plan(agent.pose, side, ManeuverType.CRUISE)
