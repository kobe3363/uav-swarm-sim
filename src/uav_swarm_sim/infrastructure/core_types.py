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


@dataclass(frozen=True)
class Event:
    type: EventType
    t: float
    payload: dict = field(default_factory=dict)


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
