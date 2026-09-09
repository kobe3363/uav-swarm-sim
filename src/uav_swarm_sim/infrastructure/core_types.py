"""Shared, strictly typed value objects passed between layers.

This batch implements the geometry-light vocabulary needed by the physical
model: Pose, Waypoint, PathSegment, Path. The Shapely-dependent types
(Region, Zone, Partition, CoveragePlan, Event, MissionResult) are added when
the planning layer first needs them (Batch 3).

Design refinement over the blueprint
------------------------------------
The blueprint's PathSegment carried only (maneuver, length, duration), which is
insufficient to reconstruct poses, and a pure arc-length cursor cannot represent
a holonomic in-place rotation (time elapses while length stays 0). Therefore:

  * each PathSegment carries its ``start`` pose, ``end`` pose, and signed
    ``curvature`` (1/radius; 0 = straight; the geometry needed to interpolate);
  * Path traversal is **time-based** (``pose_at_time``), which is exact for
    energy (E = P * dt) and handles in-place turns; ``pose_at_length`` /
    ``sample`` remain available for geometric/collision sampling by distance.

2.5D refinement (Batch 0)
-------------------------
``Pose`` gains an altitude ``z`` (default 0.0 == ground / 2D plane), so the
single-layer-z0 case is byte-identical to the 2D model. ``z`` lives on the
kinematic primitive; the discrete *layer index* lives on the assignment-bearing
types (DroneStateView, Zone, CoveragePlan), NOT on Pose -- during an inter-layer
climb a drone's ``z`` is well-defined while its "layer" is ambiguous. The
horizontal segment constructors / interpolators below hold ``z`` constant
(== ``start.z``); genuine vertical segments are owned by vertical_segments.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from .enums import DecompositionAlgo, EventType, ManeuverType, Outcome

TWO_PI = 2.0 * math.pi


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % TWO_PI - math.pi


def mod2pi(a: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    return a - TWO_PI * math.floor(a / TWO_PI)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # radians
    z: float = 0.0  # altitude; 0.0 == ground / 2D plane (single-layer-z0 default)

    def as_xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Waypoint:
    pose: Pose
    maneuver: ManeuverType
    speed: float


@dataclass(frozen=True)
class PathSegment:
    maneuver: ManeuverType
    length_m: float
    duration_s: float
    start: Pose
    end: Pose
    curvature: float = 0.0  # signed 1/radius; 0 => straight

    @property
    def speed(self) -> float:
        return self.length_m / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def is_inplace(self) -> bool:
        return self.length_m == 0.0 and self.duration_s > 0.0


# --------------------------------------------------------------------------- #
# Segment constructors (centralize the arc geometry)                          #
# Horizontal moves preserve altitude (end.z == start.z); a no-op at z=0.      #
# --------------------------------------------------------------------------- #
def straight_segment(
    start: Pose, length: float, maneuver: ManeuverType, speed: float
) -> PathSegment:
    h = start.heading
    end = Pose(start.x + length * math.cos(h), start.y + length * math.sin(h), h, start.z)
    duration = length / speed if speed > 0 else 0.0
    return PathSegment(maneuver, length, duration, start, end, 0.0)


def arc_segment(
    start: Pose, curvature: float, arc_length: float, maneuver: ManeuverType, speed: float
) -> PathSegment:
    """Arc of signed curvature (k>0 = left/CCW). arc_length >= 0."""
    if curvature == 0.0:
        return straight_segment(start, arc_length, maneuver, speed)
    r = 1.0 / curvature
    h0 = start.heading
    dtheta = curvature * arc_length  # signed heading change
    cx = start.x - r * math.sin(h0)
    cy = start.y + r * math.cos(h0)
    ex = cx + r * math.sin(h0 + dtheta)
    ey = cy - r * math.cos(h0 + dtheta)
    end = Pose(ex, ey, normalize_angle(h0 + dtheta), start.z)
    duration = arc_length / speed if speed > 0 else 0.0
    return PathSegment(maneuver, arc_length, duration, start, end, curvature)


def inplace_turn_segment(
    start: Pose, target_heading: float, omega_max: float, maneuver: ManeuverType
) -> PathSegment:
    """Holonomic in-place rotation: zero length, nonzero duration."""
    dtheta = abs(normalize_angle(target_heading - start.heading))
    duration = dtheta / omega_max if omega_max > 0 else 0.0
    end = Pose(start.x, start.y, normalize_angle(target_heading), start.z)
    return PathSegment(maneuver, 0.0, duration, start, end, 0.0)


def _pose_in_segment_by_length(seg: PathSegment, local_s: float) -> Pose:
    # Horizontal interpolation; z held constant (== seg.start.z). Vertical
    # segments are interpolated by vertical_segments.py, not here.
    if seg.curvature == 0.0:
        h = seg.start.heading
        return Pose(seg.start.x + local_s * math.cos(h), seg.start.y + local_s * math.sin(h), h, seg.start.z)
    r = 1.0 / seg.curvature
    h0 = seg.start.heading
    dtheta = seg.curvature * local_s
    cx = seg.start.x - r * math.sin(h0)
    cy = seg.start.y + r * math.cos(h0)
    x = cx + r * math.sin(h0 + dtheta)
    y = cy - r * math.cos(h0 + dtheta)
    return Pose(x, y, normalize_angle(h0 + dtheta), seg.start.z)


def _pose_in_segment_by_fraction(seg: PathSegment, frac: float) -> Pose:
    """Interpolate within a segment by time fraction in [0, 1]."""
    if seg.is_inplace:
        h0 = seg.start.heading
        dh = normalize_angle(seg.end.heading - h0)
        return Pose(seg.start.x, seg.start.y, normalize_angle(h0 + frac * dh), seg.start.z)
    return _pose_in_segment_by_length(seg, frac * seg.length_m)


@dataclass(frozen=True)
class Path:
    segments: tuple[PathSegment, ...] = ()

    @classmethod
    def from_segments(cls, segments: list[PathSegment]) -> "Path":
        return cls(tuple(segments))

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    @property
    def is_empty(self) -> bool:
        return len(self.segments) == 0

    @property
    def start_pose(self) -> Pose | None:
        return self.segments[0].start if self.segments else None

    @property
    def end_pose(self) -> Pose | None:
        return self.segments[-1].end if self.segments else None

    def pose_at_length(self, s: float) -> Pose | None:
        """Geometric sampling by arc length. In-place turns (length 0) are
        transparent to this traversal (they occupy a single point)."""
        if not self.segments:
            return None
        s = max(0.0, min(s, self.total_length_m))
        acc = 0.0
        for seg in self.segments:
            if seg.length_m == 0.0:
                continue
            if s <= acc + seg.length_m:
                return _pose_in_segment_by_length(seg, s - acc)
            acc += seg.length_m
        return self.segments[-1].end

    def pose_at_time(self, t: float) -> Pose | None:
        """Traversal by elapsed time. Correct for both moving and in-place
        segments; this is what the agent uses each tick."""
        if not self.segments:
            return None
        t = max(0.0, min(t, self.total_duration_s))
        acc = 0.0
        for seg in self.segments:
            if seg.duration_s == 0.0:
                continue
            if t <= acc + seg.duration_s:
                return _pose_in_segment_by_fraction(seg, (t - acc) / seg.duration_s)
            acc += seg.duration_s
        return self.segments[-1].end

    def maneuver_at_time(self, t: float) -> ManeuverType | None:
        if not self.segments:
            return None
        t = max(0.0, min(t, self.total_duration_s))
        acc = 0.0
        for seg in self.segments:
            if t <= acc + seg.duration_s:
                return seg.maneuver
            acc += seg.duration_s
        return self.segments[-1].maneuver

    def sample(self, ds: float) -> list[Pose]:
        """Sample poses every ``ds`` meters of arc length (plus the endpoint)."""
        out: list[Pose] = []
        total = self.total_length_m
        if total == 0.0:
            sp = self.start_pose
            return [sp] if sp is not None else []
        n = int(total // ds)
        for i in range(n + 1):
            p = self.pose_at_length(i * ds)
            if p is not None:
                out.append(p)
        end = self.pose_at_length(total)
        if end is not None and (not out or out[-1] != end):
            out.append(end)
        return out


# --------------------------------------------------------------------------- #
# Shapely-dependent types (added in Batch 3 for the planning layer)           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Region:
    """Atomic free-space cell produced by the TGC. Areas are exact Shapely
    polygon areas -- this is what makes the area-proportionality guarantee of
    the weighted decomposition exact even though region shapes are approximate."""
    id: int
    polygon: Polygon
    area_m2: float
    anchor: Pose


@dataclass(frozen=True)
class DroneStateView:
    """Immutable snapshot of a drone handed to planners (no behavior, just the
    fields a decomposer needs)."""
    id: int
    battery_frac: float
    pose: Pose
    layer: int = 0   # assigned coverage layer index; 0 == single-layer-z0 default
    # EXP-08: is the drone in the air RIGHT NOW? Only the energy-balance budget
    # reads it (``_budget`` deducts takeoff for a grounded drone and must not
    # charge it twice to one already flying), so every t=0 construction keeps
    # the default and stays byte-identical: at t=0 every drone is on the ground.
    airborne: bool = False
    # REV-02: planning a live re-partition must use the same per-drone home and
    # actual AGL as coherent execution.  Both are additive defaults so every
    # pre-REV-02 planner construction remains source-compatible.
    base: Pose | None = None
    # ``None`` retains the legacy meaning of an airborne snapshot: take-off
    # energy has already been charged.  Coherent execution supplies AGL.
    agl_m: float | None = None


@dataclass
class Zone:
    drone_id: int
    regions: list[Region]
    polygon: Polygon            # merged region polygons (may be MultiPolygon)
    entry_pose: Pose
    layer: int = 0              # source coverage layer; 0 == single-layer-z0 default

    @property
    def area_m2(self) -> float:
        return float(self.polygon.area)


@dataclass
class Partition:
    algo: DecompositionAlgo
    zones: dict[int, Zone]      # drone_id -> Zone
    planning_time_s: float

    @property
    def total_area_m2(self) -> float:
        return sum(z.area_m2 for z in self.zones.values())


@dataclass
class CoveragePlan:
    drone_id: int
    waypoints: list[Waypoint]
    length_m: float
    est_energy_j: float
    leg_mode: str = "boustrophedon"   # "boustrophedon" (sweep) | "tour" (target-visit)
    layer: int = 0                    # coverage layer this plan is stamped to; 0 == single-layer-z0
    # S_FERRY Step 2: plan-time routed camera-off connectors, one Path per odd
    # (TURN) leg, in strip order. Empty => the executor/analytical rebuild falls
    # back to the straight motion.plan(a, b, TURN) chord (byte-identical). When
    # populated (routing enabled), it is the SINGLE source both the executor and
    # the analytical E_cover consume, so their connector cost stays in lock-step.
    connectors: list[Path] = field(default_factory=list)
    strips_energy_j: float = 0.0
    connectors_energy_j: float = 0.0


@dataclass(frozen=True)
class Event:
    type: EventType
    t: float
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PhotoEvent:
    """One physical shutter trigger on a productive coverage-strip pass."""

    agent_id: int
    t_s: float
    pose: Pose
    coverage_leg_index: int
    distance_on_strip_m: float


@dataclass(frozen=True)
class SafetyViolation:
    """EXP-10: one FACTUAL safety breach that actually occurred during execution
    (not a predicted-then-avoided threat).

    ``kind`` is "separation" | "obstacle" | "speed"; ``severity`` is "hard" |
    "soft". ``agents`` is ``(id,)`` for obstacle/speed and the NORMALIZED pair
    ``(lo, hi)`` for separation (so A-B and B-A are one record). One continuous
    breach is ONE record; a breach that ends and restarts is a new record.

    Time semantics (deliberately distinct):
      * ``t_start`` / ``t_end`` -- the first and last SAMPLE at which the breach
        was observed, i.e. the episode SPAN. Hysteresis can hold an episode open
        across intervening non-violating samples.
      * ``duration_s`` -- the accumulated VIOLATING time (n_samples * dt), which
        is therefore NOT ``t_end - t_start`` in general.
    ``ended_reason`` is "recovered" (metric returned past the hysteresis band),
    "left_observed_set" (the agent failed/landed mid-breach -- its history is
    preserved, closed at its last violating sample) or "mission_end".
    ``truncated_at_end`` is True ONLY for an episode still open at mission end.

    Exactly one extremum is set per ``kind`` (the rest stay None), so hard/soft
    and the three kinds stay separable for EXP-11 and are never collapsed to one
    score. Every extremum is updated ONLY while actually in violation, never
    during the hysteresis-band non-violating samples:
      * separation -> ``min_separation_m`` (the closest the pair actually came);
      * obstacle   -> ``max_penetration_m`` (deepest inside the RAW obstacle;
                      0.0 for a pure-soft/buffer-only episode);
      * speed      -> ``peak_speed_m_s`` with ``limit_m_s`` (the envelope for a
                      hard record, the commanded segment speed for a soft one).
    """
    kind: str
    severity: str
    agents: tuple[int, ...]
    t_start: float
    t_end: float
    duration_s: float
    ended_reason: str
    truncated_at_end: bool = False
    min_separation_m: float | None = None
    max_penetration_m: float | None = None
    peak_speed_m_s: float | None = None
    limit_m_s: float | None = None


@dataclass
class MissionResult:
    metrics: object              # metrics.mission_metrics.MissionMetrics
    history: object              # metrics.state_history.StateHistory
    partition: "Partition | None"
    aborted: bool
    coverage_frac: float
    config_hash: str
    # Phase 2 (Task 2.2): explicit terminal outcome decided in the run loop.
    # Defaulted so any existing direct MissionResult(...) construction is
    # unaffected; the engine always passes the resolved Outcome.
    outcome: Outcome = Outcome.MISSION_INCOMPLETE
    # FIX-B4 (safety.stall_detector): agents whose swap-livelock cut the run
    # short -- >= 5 consecutive swap requests without coverage progress.
    # Empty tuple always, unless the detector is enabled AND it fired.
    stalled_agents: tuple[int, ...] = ()
    # EM-01 Stage 4 (safety.stall_skip): coverage strips forfeited after a
    # stall, as sorted (agent_id, coverage-leg index) pairs -- the index is the
    # strip's position in the agent's FINAL coverage-leg list (even = strip).
    # Explicit accounting, never a silent drop: any entry here forces the
    # terminal outcome to MISSION_PARTIAL and an honestly reduced
    # coverage_frac. Empty tuple always, unless the flag is on AND a skip fired.
    skipped_legs: tuple[tuple[int, int], ...] = ()
    # EXP-01: actual distance-triggered shutter events.  Empty in legacy mode;
    # intentionally not serialized by the legacy result schema (EXP-11 owns it).
    photo_events: tuple[PhotoEvent, ...] = ()
    # EXP-02: physical camera coverage of A_target (raw obstacles excluded).
    # None only for direct legacy constructions outside SimulationEngine.
    target_coverage_frac: float | None = None
    # EXP-04: explicit terminal diagnostics (additive; not serialized by the
    # legacy result schema). ``terminal_reason`` names the condition that
    # ended the run in BOTH modes -- the pre-existing telemetry strings
    # ("battery_depleted", "pool_exhausted", "coverage_complete_with_gaps",
    # "coverage_complete") plus "stall_livelock" / "max_timesteps" for the two
    # MISSION_INCOMPLETE exits; ``airborne_at_end`` lists drones still flying
    # when the run stopped (non-empty only on a time cap / stall halt).
    # ``retired_agents``, ``work_releases`` (agent_id, t) and ``losses``
    # (agent_id, t, cause) are populated only under mission.no_swap_mode.
    terminal_reason: str | None = None
    airborne_at_end: tuple[int, ...] = ()
    retired_agents: tuple[int, ...] = ()
    work_releases: tuple[tuple[int, float], ...] = ()
    losses: tuple[tuple[int, float, str], ...] = ()
    # EXP-05: actual t=0 capacity fractions, ordered by contiguous drone ID.
    initial_soc_by_drone: tuple[float, ...] = ()
    # EXP-06: absent when disabled; each method retains all joule components.
    energy_balance_t0: dict | None = None
    # EXP-07: how the grid partition was reached (convergence, dropped cells,
    # per-drone weight/area). None for every non-grid decomposition algorithm.
    partition_diagnostics: object | None = None
    # EXP-08: one record per in-flight re-partition ATTEMPT, in order, including
    # the attempts that were deliberately not applied (no eligible executor, no
    # remaining work, no progress). Empty tuple unless
    # mission.repartition_enabled is on -- an attempt that changes nothing is
    # recorded rather than dropped, so a run can never be read as "no
    # re-partition happened" when in fact one was refused.
    repartitions: tuple[dict, ...] = ()
    # EXP-08: what the one-tick re-task hold cost this run. The hold is applied
    # by one rule to both arms, but it is NOT paired -- two arms can finish a
    # different NUMBER of zones, so their totals differ. Reported so the size of
    # that asymmetry is visible instead of assumed negligible. None with the flag
    # off (no drone is ever held).
    repartition_hold: dict | None = None
    # EXP-09 diagnostics only; no change to terminal outcome classification.
    rth_infeasible_events: tuple[Event, ...] = ()
    # EXP-10: factual safety breaches recorded during execution (empty unless
    # safety.record_violations is on). Additive; not serialized by the legacy
    # result schema -- the EXP-11 exporter owns serialization. Hard and soft are
    # kept as separate records, never collapsed to one score.
    safety_violations: tuple[SafetyViolation, ...] = ()
    # EXP-10: run-level safety minima, the FROZEN EXP-11 interface. Keys:
    # "min_separation_m" (closest any airborne same-layer pair came; None if <2
    # were ever airborne together), "min_obstacle_clearance_m" (closest any
    # airborne drone came to a raw obstacle; None if obstacle-free), "n_hard" and
    # "n_soft" (violation-record counts by severity). None with the flag off.
    safety_minima: dict | None = None
    # EXP-11: raw raster coverage AREAS for the data contract (fracs already live
    # in ``coverage_frac``/``target_coverage_frac`` -- not duplicated here). Keys:
    # "source" ("raster" | "segment_proxy"), "a_target_m2", "a_plannable_m2",
    # "target_covered_area_m2", "plannable_covered_area_m2". With no raster the
    # source is "segment_proxy" and the four areas are None -- a proxy frac is
    # never dressed up as a raster measurement. None only for direct legacy
    # constructions outside SimulationEngine.
    coverage_measurements: dict | None = None
    # EXP-11: per-drone END-OF-RUN battery, read from the live Battery at result
    # build (exact, not the sample-resolution history trace): (agent_id,
    # final_level_j, final_soc), ordered by contiguous drone id, parallel to
    # ``initial_soc_by_drone``. Includes lost drones -- the cut moment is labeled
    # in the contract, never filtered.
    final_battery_by_drone: tuple[tuple[int, float, float], ...] = ()
