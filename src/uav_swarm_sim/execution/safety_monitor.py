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

from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon

from ..infrastructure.config import SafetyConfig
from ..infrastructure.core_types import Event, Path, Pose, SafetyViolation
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
        # Predicted intra-layer separation conflicts, resolved ONCE per tick with
        # a KDTree neighbour query instead of the per-agent O(n^2) pairwise scan
        # (Batch 3). Byte-identical to the former loop -- see _separation_yielders.
        sep_yielders = self._separation_yielders(airborne, preds)

        for a in airborne:
            if t < self._cooldown_until.get(a.id, -1.0):
                continue  # recently avoided -> let it fly before re-checking
            threat = self._threatened(a, airborne, preds, sep_yielders)
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

    def _threatened(self, a, airborne, preds, sep_yielders) -> bool:
        # Inter-drone conflicts during formation phases (transit/RTH) are governed
        # by the FormationManager (spacing), not collision avoidance -- ignore them
        # here to avoid launch/return thrashing.
        a_formation = a.state in (AgentState.S1_TRANSIT, AgentState.S3_RTH)
        a_layer = getattr(a, "layer", 0)

        # pairwise predicted separation (lower-id yields). The conflict set is
        # computed once per tick in _separation_yielders (KDTree); membership here
        # is exactly equivalent to the former inline O(n^2) pairwise scan -- the
        # non-formation set, same-layer, time-aligned, strict-min_sep rule are all
        # encoded in that set, so a formation-phase drone is never a member.
        if a.id in sep_yielders:
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

    def _separation_yielders(self, airborne, preds) -> set[int]:
        """Drones that must yield this tick for predicted intra-layer separation.

        Computed once per tick with a per-layer, per-predicted-timestep KDTree
        neighbour query, replacing the former per-agent O(n^2) pairwise loop. The
        result is byte-identical to that loop, by construction:

          * only NON-formation drones participate (formation/transit drones are
            spaced by the FormationManager); ``airborne`` already excludes S_OBS,
            so the participant set is exactly the S1/S3-excluded remainder;
          * only same-LAYER drones are compared (different layers are vertically
            separated and cannot collide);
          * poses are compared at the SAME predicted timestep (the old ``zip`` of
            the two pose lists) -- an agent whose prediction is shorter than k (a
            spent leg collapses to a single pose) drops out at step k, matching
            ``zip``'s truncation;
          * ``query_pairs(min_sep)`` returns candidate pairs within the radius
            (distance <= min_sep); each is then confirmed with the EXACT original
            strict ``math.dist < min_sep`` test, so the boundary is identical;
          * for each confirmed pair the LOWER-id drone yields (the old
            ``b.id <= a.id: continue`` direction).
        """
        sep = self._cfg.min_separation_m
        participants = [
            a for a in airborne
            if a.state not in (AgentState.S1_TRANSIT, AgentState.S3_RTH)
        ]
        by_layer: dict[int, list] = {}
        for a in participants:
            by_layer.setdefault(getattr(a, "layer", 0), []).append(a)

        yielders: set[int] = set()
        for group in by_layer.values():
            if len(group) < 2:
                continue
            max_k = max(len(preds[a.id]) for a in group)
            for k in range(max_k):
                pts: list[tuple[float, float]] = []
                ids: list[int] = []
                for a in group:
                    pa = preds[a.id]
                    if len(pa) > k:
                        pts.append(pa[k].as_xy())
                        ids.append(a.id)
                if len(pts) < 2:
                    continue
                tree = KDTree(pts)
                for i, j in tree.query_pairs(sep):
                    if math.dist(pts[i], pts[j]) < sep:
                        yielders.add(ids[i] if ids[i] < ids[j] else ids[j])
        return yielders

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

    def _chord_clear(self, a: Pose, b: Pose, env) -> bool:
        """True iff the straight chord a->b does not cross any RAW obstacle on
        the agent's layer. Exact Shapely-segment test (``segment_in_obstacle``)
        replacing the former 9-point sampling, which could tunnel straight
        through a wall thinner than the sample spacing -- accepting a detour that
        in fact flies into the obstacle. Raw (unbuffered) to stay consistent with
        the ``in_obstacle`` penetration trigger this detour resolves. Reached only
        via ``avoidance_plan`` under recovery mode, so the default path is
        unaffected."""
        return not env.segment_in_obstacle(a, b)


# --------------------------------------------------------------------------- #
# EXP-10: factual safety-violation recorder                                   #
# --------------------------------------------------------------------------- #
# Methodological constants (module-level, documented -- NOT config knobs, to keep
# the diff small). _HYST_FRAC is the open->close hysteresis band on the
# separation/speed thresholds so a metric hovering exactly at the boundary does
# not chatter into many one-tick episodes; it is metric-shaping and is reported
# as a methodological choice. _SPEED_TOL_FRAC absorbs float noise on the speed
# comparison. Obstacle episodes need no hysteresis: the clearance buffer already
# supplies a margin between the raw-penetration (hard) trigger and the buffer
# edge where the episode closes.
_HYST_FRAC = 0.05
_SPEED_TOL_FRAC = 1e-3
_EPS_ABS = 1e-6
# Vertical maneuvers -- excluded from the HORIZONTAL speed peak (their speed is
# an ascent/descent rate governed by v_climb/v_descent, a separate envelope that
# the current model never lets execution exceed; see the recorder docstring).
_VERTICAL = frozenset({ManeuverType.CLIMB, ManeuverType.DESCENT,
                       ManeuverType.TAKEOFF, ManeuverType.LAND})


class _Episode:
    """One open factual-violation episode (mutable while accumulating).

    ``extremum`` is the closest separation (min), the deepest raw penetration
    (max), or the peak horizontal speed (max) depending on ``kind``; it is
    updated ONLY on a genuinely-violating sample. ``n_samples`` counts violating
    samples -> the reported duration is ``n_samples * dt`` (NOT ``t_end -
    t_start``, which is the episode span across any hysteresis-band samples).
    """
    __slots__ = ("kind", "severity", "agents", "t_start", "t_end",
                 "n_samples", "extremum", "limit")

    def __init__(self, kind, severity, agents, t, extremum, limit=None):
        self.kind = kind
        self.severity = severity
        self.agents = agents
        self.t_start = t
        self.t_end = t
        self.n_samples = 1
        self.extremum = extremum
        self.limit = limit

    def finish(self, dt, ended_reason, truncated_at_end=False):
        sep = self.extremum if self.kind == "separation" else None
        pen = self.extremum if self.kind == "obstacle" else None
        spd = self.extremum if self.kind == "speed" else None
        return SafetyViolation(
            kind=self.kind, severity=self.severity, agents=self.agents,
            t_start=self.t_start, t_end=self.t_end,
            duration_s=self.n_samples * dt, ended_reason=ended_reason,
            truncated_at_end=truncated_at_end,
            min_separation_m=sep, max_penetration_m=pen,
            peak_speed_m_s=spd, limit_m_s=(self.limit if self.kind == "speed" else None),
        )


class ViolationRecorder:
    """EXP-10: records FACTUAL safety breaches from the EXECUTED fleet state.

    Independent of ``SafetyMonitor`` -- it shares no cooldown, no formation-phase
    exclusion and no threat state, so a breach that happened during avoidance,
    RTH, a formation phase, takeoff or landing is recorded, not hidden by a
    prediction exception. It observes at ``dt`` granularity: a crossing that falls
    entirely between two samples is NOT seen (a documented limitation -- this is
    sampled monitoring, not continuous collision detection). It changes no
    trajectory, energy, RNG stream or outcome; it only reads pose/state/layer and
    the pre-tick leg (for the speed peak).

    Three kinds:
      * separation -- horizontal distance between two airborne drones on the same
        layer < ``min_separation_m`` (severity hard: an actual loss of the
        required separation bubble; there is no softer tier). In a multi-layer
        run different-layer pairs are ASSUMED vertically separated (a known gap
        near ground during climb); with a single layer the filter is inert.
      * obstacle -- a pose inside a RAW obstacle (hard) or inside the clearance
        buffer only (soft), on the drone's own layer slice (so finite-height
        prisms are handled). ONE escalating episode per continuous encounter;
        ``max_penetration_m`` is the deepest raw penetration (0.0 for pure-soft).
      * speed -- executed HORIZONTAL speed above the platform envelope
        (``v_cruise``; hard) or above the commanded segment speed (soft). HARD is
        an invariant GUARD: ``MotionModel.advance`` caps executed displacement at
        ``max(ideal, 6 m/s)`` and every commanded speed is <= v_cruise, so
        execution can never exceed the envelope unless a planner emits an
        over-envelope segment -- none does. Vertical speed is fully bounded by the
        commanded climb/descent profiles and is not tracked here.
    """

    def __init__(self, world, cfg: SafetyConfig, spec, dt: float) -> None:
        self._world = world
        self._min_sep_m = cfg.min_separation_m
        self._v_env_h = spec.v_cruise          # horizontal hardware envelope
        self._dt = dt
        self._n_layers = int(getattr(world, "n_layers", 1))
        self._open: dict[tuple, _Episode] = {}
        self._done: list[SafetyViolation] = []
        self._snap: dict[int, tuple] = {}
        self._min_separation: float | None = None
        self._min_clearance: float | None = None

    def _env_for(self, a):
        w = self._world
        if w is not None and hasattr(w, "layer"):
            return w.layer(getattr(a, "layer", 0))
        return w

    # ------------------------------------------------------------------ #
    def snapshot(self, agents) -> None:
        """Pre-tick: remember each agent's pose and the leg it is about to fly,
        so the post-tick speed check can reconstruct the within-tick peak."""
        self._snap = {}
        for a in agents:
            legs = getattr(a, "_legs", [])
            idx = getattr(a, "_leg_idx", 0)
            leg = legs[idx] if 0 <= idx < len(legs) else None
            self._snap[a.id] = (a.pose, leg, getattr(a, "_t", 0.0))

    def observe(self, agents, t: float) -> None:
        """Post-tick: evaluate the executed state of the airborne fleet."""
        observed = {a.id: a for a in agents}
        self._close_absent(observed)
        self._observe_separation(observed, t)
        self._observe_obstacle(observed, t)
        self._observe_speed(observed, t)

    def finalize(self, t_end: float) -> None:
        """Close every still-open episode at mission end."""
        for key in list(self._open):
            self._finish(key, "mission_end", truncated_at_end=True)

    def result(self):
        n_hard = sum(1 for v in self._done if v.severity == "hard")
        n_soft = sum(1 for v in self._done if v.severity == "soft")
        minima = {
            "min_separation_m": self._min_separation,
            "min_obstacle_clearance_m": self._min_clearance,
            "n_hard": n_hard,
            "n_soft": n_soft,
        }
        return tuple(self._done), minima

    # ------------------------------------------------------------------ #
    def _finish(self, key, ended_reason, truncated_at_end=False) -> None:
        ep = self._open.pop(key, None)
        if ep is not None:
            self._done.append(ep.finish(self._dt, ended_reason, truncated_at_end))

    def _close_absent(self, observed) -> None:
        """A breach whose agent (or either paired agent) left the observed set
        this tick -- it failed (Fleet.kill) or landed (S_LANDED), both of which
        drop it from airborne(). Close it at its own last violating sample so the
        breach survives the transition and is NOT relabelled a mission-end
        truncation (acceptance check 7.7)."""
        for key in list(self._open):
            ids = key[1:] if key[0] == "separation" else (key[1],)
            if any(i not in observed for i in ids):
                self._finish(key, "left_observed_set")

    # ---- separation --------------------------------------------------- #
    def _observe_separation(self, observed, t) -> None:
        ids = sorted(observed)
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = observed[ids[a_i]], observed[ids[b_i]]
                # Different layers are vertically separated (as the predictive
                # monitor treats them). Inert when there is only one layer.
                if self._n_layers > 1 and getattr(a, "layer", 0) != getattr(b, "layer", 0):
                    continue
                d = math.dist(a.pose.as_xy(), b.pose.as_xy())
                if self._min_separation is None or d < self._min_separation:
                    self._min_separation = d
                key = ("separation", ids[a_i], ids[b_i])
                if d < self._min_sep_m:
                    self._extend(key, "separation", "hard", (ids[a_i], ids[b_i]),
                                 t, d, reduce=min)
                elif key in self._open and d >= self._min_sep_m * (1.0 + _HYST_FRAC):
                    self._finish(key, "recovered")

    # ---- obstacle ----------------------------------------------------- #
    def _observe_obstacle(self, observed, t) -> None:
        for aid, a in observed.items():
            env = self._env_for(a)
            if env is None:
                continue
            p = a.pose.as_xy()
            raw = env.raw_obstacles
            if raw is not None:
                clr = Point(p).distance(raw)   # 0.0 when inside the raw obstacle
                if self._min_clearance is None or clr < self._min_clearance:
                    self._min_clearance = clr
            in_raw = env.in_obstacle(p)
            buffered = env.buffered_obstacles
            in_buffer = in_raw or (buffered is not None and buffered.covers(Point(p)))
            key = ("obstacle", aid)
            if not in_buffer:
                if key in self._open:
                    self._finish(key, "recovered")
                continue
            severity = "hard" if in_raw else "soft"
            # penetration depth only inside the raw obstacle (review #6): a
            # pure-soft/buffer episode keeps 0.0, never a fictitious outside gap.
            depth = Point(p).distance(raw.boundary) if (in_raw and raw is not None) else 0.0
            self._extend(key, "obstacle", severity, (aid,), t, depth, reduce=max)

    # ---- speed -------------------------------------------------------- #
    def _observe_speed(self, observed, t) -> None:
        for aid, a in observed.items():
            snap = self._snap.get(aid)
            if snap is None:
                continue                       # first airborne tick: no prev pose
            prev_pose, leg, t0 = snap
            if leg is None or leg.is_empty:
                continue                       # no commanded leg => no speed reference
                                               # (a real airborne drone never moves
                                               # without an active leg)
            v_disp = math.dist(prev_pose.as_xy(), a.pose.as_xy()) / self._dt if self._dt > 0 else 0.0
            v_peak = self._peak_speed(leg, t0)   # commanded peak = on-path actual
            v_act = max(v_disp, v_peak)
            sev, limit = self._classify_speed(v_act, v_peak)
            for severity in ("hard", "soft"):
                key = ("speed", aid, severity)
                if sev == severity:
                    self._extend(key, "speed", severity, (aid,), t, v_act,
                                 reduce=max, limit=limit)
                elif key in self._open:
                    # Recovery is judged against the CURRENT tick's threshold, not
                    # the stored open-limit: the episode closes as soon as the
                    # executed speed no longer exceeds what THIS tick commands
                    # (i.e. the drone is compliant), so a later breach opens a
                    # fresh record. Keying recovery off the stored limit stranded a
                    # zero-command in-place-turn soft episode open forever -- its
                    # stored limit is 0, so it only closed at a standstill -- even
                    # after the drone resumed compliant flight at a positive
                    # command. No hysteresis gap is applied here (unlike
                    # separation): a per-tick commanded speed that jumps between
                    # legs would let a below-limit band trap the episode, and
                    # recovery speed ramps smoothly so threshold chatter does not
                    # arise in practice.
                    cur_limit = self._v_env_h if severity == "hard" else v_peak
                    thr = max(cur_limit * (1.0 + _SPEED_TOL_FRAC), _EPS_ABS)
                    if v_act <= thr:
                        self._finish(key, "recovered")

    def _classify_speed(self, v_act, v_cmd):
        """(severity, limit) for a violating tick, else (None, None). HARD when
        the executed peak exceeds the platform envelope; SOFT when it exceeds the
        commanded segment speed (v_cmd) but not the envelope."""
        if v_act > self._v_env_h * (1.0 + _SPEED_TOL_FRAC):
            return "hard", self._v_env_h
        thr = v_cmd * (1.0 + _SPEED_TOL_FRAC)
        if v_act > thr and v_act > _EPS_ABS:
            return "soft", v_cmd
        return None, None

    def _peak_speed(self, leg, t0):
        """Max HORIZONTAL commanded speed among the segments of ``leg`` crossed in
        [t0, min(t0+dt, dur)] -- the within-tick peak that the tick-average
        displacement speed would otherwise hide (acceptance check 7.6). In-place
        turns (length 0) and vertical segments do not contribute."""
        if leg is None or leg.is_empty:
            return 0.0
        lo, hi = t0, min(t0 + self._dt, leg.total_duration_s)
        if hi <= lo:
            return 0.0
        acc, best = 0.0, 0.0
        for seg in leg.segments:
            seg_lo, seg_hi = acc, acc + seg.duration_s
            if seg_hi > lo and seg_lo < hi and seg.length_m > 0.0 \
                    and seg.maneuver not in _VERTICAL:
                best = max(best, seg.speed)
            acc = seg_hi
        return best

    def _extend(self, key, kind, severity, agents, t, value, *, reduce, limit=None):
        ep = self._open.get(key)
        if ep is None:
            self._open[key] = _Episode(kind, severity, agents, t, value, limit)
            return
        # escalate soft -> hard within one continuous obstacle encounter (the
        # worst severity reached is the episode's severity).
        if severity == "hard" and ep.severity == "soft":
            ep.severity = "hard"
        ep.t_end = t
        ep.n_samples += 1
        # speed: keep the limit paired with the recorded PEAK (value is the new
        # peak when it is at least the running max). Evaluated before the extremum
        # is reassigned.
        if limit is not None and value >= ep.extremum:
            ep.limit = limit
        ep.extremum = reduce(ep.extremum, value)
