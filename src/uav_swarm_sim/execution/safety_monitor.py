"""Proactive obstacle avoidance as sustained monitoring (guideline 3.4).

Every tick, predicted poses over a horizon are checked against other drones,
obstacles, and aerodynamic wake zones (treated as 'invisible obstacles'). This
is continuous surveillance, not a reactive last-moment trigger. Deterministic
yielding (lower-id agent yields) keeps the resolution collision-free by
construction. All S_OBS time is flight overhead by definition and lands in the
efficiency-score denominator via the SMDP layer.

2.5D (Batch 4): separation is intra-layer. Drones on different coverage layers
are vertically separated, so pairwise separation and wake checks skip cross-layer
pairs, and each drone's obstacle-penetration test uses its OWN layer's sliced map
(higher layers clear short obstacles). The monitor takes the LayerStack and
selects the per-agent map; a plain EnvironmentMap is still accepted (back-compat).
With one layer every drone is on layer 0 whose map is the 2D world, so the
threats raised are byte-identical.
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
    def __init__(self, world, aero: AeroCorrection, cfg: SafetyConfig, motion: MotionModel) -> None:
        # ``world`` is a LayerStack (per-layer maps) or a single EnvironmentMap.
        self._world = world
        self._aero = aero
        self._cfg = cfg
        self._motion = motion
        self._cooldown_until: dict[int, float] = {}

    def _env_for(self, a):
        """The obstacle map this agent is checked against: its layer's sliced map
        if a LayerStack was supplied, else the single map (back-compat)."""
        w = self._world
        if w is not None and hasattr(w, "layer"):
            return w.layer(getattr(a, "layer", 0))
        return w

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
        recovery = bool(getattr(self._cfg, "obstacle_recovery", False))

        for a in airborne:
            if t < self._cooldown_until.get(a.id, -1.0):
                continue  # recently avoided -> let it fly before re-checking
            threat = self._threatened(a, airborne, preds)
            if threat:
                # Task 2.5 Q2 (gated): for a genuine OBSTACLE threat, hand the agent
                # an obstacle-validated detour and tell it to skip the obstructed
                # coverage leg on rejoin. Separation/wake threats (and the whole
                # off path) keep the original blind-sidestep signal.
                if recovery and self._is_obstacle_threat(a, preds):
                    a.signal_threat(True, avoidance=self.avoidance_plan(a), skip_leg=True)
                else:
                    a.signal_threat(True)
                self._cooldown_until[a.id] = t + self._cfg.predict_horizon_s
                bus.publish(Event(EventType.OBSTACLE_THREAT, t, {"agent_id": a.id}))

    def _is_obstacle_threat(self, a, preds) -> bool:
        """True iff a predicted pose penetrates an obstacle on the agent's layer --
        the skip-eligible threat kind. Separation and wake threats resume normally.
        Read-only; used only when recovery mode is enabled."""
        env = self._env_for(a)
        if env is None:
            return False
        for p in preds[a.id]:
            if env.in_obstacle(p.as_xy()):
                return True
        return False

    def _threatened(self, a, airborne, preds) -> bool:
        # Inter-drone conflicts during formation phases (transit/RTH) are governed
        # by the FormationManager (spacing), not collision avoidance -- ignore them
        # here to avoid launch/return thrashing.
        a_formation = a.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH)
        a_layer = getattr(a, "layer", 0)

        # pairwise predicted separation (lower-id yields), skipped if both formation
        # or on different layers (vertically separated -> cannot collide).
        if not a_formation:
            for b in airborne:
                if b.id <= a.id:
                    continue
                if getattr(b, "layer", 0) != a_layer:
                    continue
                if b.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH):
                    continue
                for pa, pb in zip(preds[a.id], preds[b.id]):
                    if math.dist(pa.as_xy(), pb.as_xy()) < self._cfg.min_separation_m:
                        return True

        # genuine obstacle penetration on THIS agent's layer (raw obstacle
        # polygons; not boundary/buffer)
        env = self._env_for(a)
        if env is not None:
            for p in preds[a.id]:
                if env.in_obstacle(p.as_xy()):
                    return True

        # wake zones from other airborne drones ON THE SAME LAYER (invisible
        # obstacles), only when this agent is dispersed (not riding a formation)
        if not a_formation:
            leaders = [b.pose for b in airborne if b.id != a.id and getattr(b, "layer", 0) == a_layer]
            for wake in self._aero.wake_zones(leaders):
                for p in preds[a.id]:
                    if wake.covers(Point(p.as_xy())):
                        return True
        return False

    def avoidance_plan(self, agent, env=None) -> Path:
        """An obstacle-VALIDATED lateral detour (Task 2.5 Q2). Tries increasing
        offsets on each side and returns the first whose straight chord is clear of
        obstacles on the agent's layer, so -- unlike the agent's blind sidestep --
        the evasive maneuver does not itself fly into the obstacle. Falls back to a
        plain offset if nothing clears; a genuinely boxed-in drone is then handled
        by the agent's re-entry escalation budget. Used only in recovery mode."""
        if env is None:
            env = self._env_for(agent)
        h = agent.pose.heading
        lx, ly = -math.sin(h), math.cos(h)     # unit left-normal
        fx, fy = math.cos(h), math.sin(h)      # unit forward
        base = max(self._cfg.min_separation_m, self._cfg.obstacle_buffer_m + 5.0)
        for mult in (1.0, 1.5, 2.0, 3.0):
            off = base * mult
            for sign in (1.0, -1.0):           # prefer left, then right
                side = Pose(agent.pose.x + sign * off * lx + base * fx,
                            agent.pose.y + sign * off * ly + base * fy, h)
                if env is None or self._chord_clear(agent.pose, side, env):
                    return self._motion.plan(agent.pose, side, ManeuverType.CRUISE)
        # nothing clear: degenerate single step (escalation returns it home if this persists)
        side = Pose(agent.pose.x + base * lx + base * fx,
                    agent.pose.y + base * ly + base * fy, h)
        return self._motion.plan(agent.pose, side, ManeuverType.CRUISE)

    def _chord_clear(self, a: Pose, b: Pose, env, n: int = 8) -> bool:
        for k in range(n + 1):
            x = a.x + (b.x - a.x) * k / n
            y = a.y + (b.y - a.y) * k / n
            if env.in_obstacle((x, y)):
                return False
        return True
