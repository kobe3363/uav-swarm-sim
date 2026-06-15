"""Pre-flight trajectory validation + bounded repair (Task 2.5, Q1).

A linear corridor from the GVG/boustrophedon decomposition is clear by
construction, but the MotionModel's Dubins smoothing can bulge a turn arc
outside that corridor and clip an obstacle buffer. Executing the clipped arc
trips the SafetyMonitor (S_OBS) and -- pre-2.5 -- the drone bounced and snapped
straight back to the SAME invalid arc: an infinite, battery-draining ping-pong.

This module validates every smoothed leg against the EnvironmentMap's BUFFERED
obstacles BEFORE it is flown, and repairs a clipping leg with a bounded,
deterministic ladder (see ``enums.LegRepair`` for the outcome vocabulary):

  1. CLEAN      -- the smoothed leg is already clear; returned unchanged.
  2. RESMOOTHED -- subdivide-and-resmooth: insert a midpoint on the straight
                   chord and re-Dubins each half (shorter arcs bulge less); retry
                   up to ``max_depth`` levels.
  3. LINEAR     -- linear-corridor fallback: fly the straight chord at the same
                   maneuver. Provably clear -- the skeleton came from the
                   obstacle-aware decomposition; only the smoothing broke it.
                   Kinematically abrupt (Dubins feasibility relaxed for this one
                   leg) and marginally cheaper (a chord is shorter than its arc).
  4. BLOCKED    -- even the chord clips: the corridor is genuinely obstructed
                   (a dynamic / newly-discovered obstacle straddles it). NOT a
                   smoothing artifact; the caller escalates (runtime S_OBS
                   recovery, or a TGC reroute + logged coverage gap) rather than
                   flying into it.

Pure geometry: no RNG, no I/O, no SimulationEngine import. Validation goes
through ``env.path_clear`` / ``env.first_obstruction``, so the validator uses the
SAME buffered union the clearance and free-space queries use -- there is no
separate buffer to fall out of sync, and "clear here" is strictly stronger than
the SafetyMonitor's raw-penetration trigger, which is exactly what makes the
resume loop terminate.

Holonomic platforms (r_min == 0) get straight legs from the MotionModel already,
so validation passes on the first try and the returned Path is the identical
object: the multirotor single-layer baseline is byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..infrastructure.core_types import Path, Pose, normalize_angle
from ..infrastructure.enums import LegRepair, ManeuverType

# Path-sampling resolution (m) for clearance checks. Matches EnvironmentMap's
# path_clear default so prediction and the monitor agree on the same geometry.
_STEP_M = 2.0

# Subdivide-and-resmooth recursion bound. 2 levels -> up to 4 sub-legs, which
# clears every realistic single-arc bulge while staying cheap and terminating.
_MAX_DEPTH_DEFAULT = 2


@dataclass(frozen=True)
class ValidatedLeg:
    """A leg guaranteed clear of buffered obstacles (unless ``repair`` is
    ``BLOCKED``), plus how that guarantee was obtained."""
    path: Path
    repair: LegRepair
    start: Pose
    goal: Pose
    maneuver: ManeuverType

    @property
    def ok(self) -> bool:
        """Flyable: every outcome except BLOCKED yields a clear path."""
        return self.repair is not LegRepair.BLOCKED

    @property
    def repaired(self) -> bool:
        """True if the original smoothed leg had to be altered to clear it."""
        return self.repair in (LegRepair.RESMOOTHED, LegRepair.LINEAR)


def _chord_midpoint(a: Pose, b: Pose) -> Pose:
    """Midpoint of the straight chord ``a -> b``, facing along the chord so a
    Dubins plan ``a -> m -> b`` passes through it heading toward ``b``."""
    bearing = math.atan2(b.y - a.y, b.x - a.x)
    return Pose(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), normalize_angle(bearing), 0.5 * (a.z + b.z))


def _resmooth(motion, env, a: Pose, b: Pose, maneuver: ManeuverType, depth: int) -> Path | None:
    """Recursively subdivide ``a -> b`` on the chord and re-Dubins each half,
    recursing into any half that still clips. Returns a concatenated CLEAR Path,
    or ``None`` if no subdivision up to ``depth`` clears both halves."""
    if depth <= 0:
        return None
    m = _chord_midpoint(a, b)
    first = motion.plan(a, m, maneuver)
    second = motion.plan(m, b, maneuver)
    if not env.path_clear(first, _STEP_M):
        first = _resmooth(motion, env, a, m, maneuver, depth - 1)
        if first is None:
            return None
    if not env.path_clear(second, _STEP_M):
        second = _resmooth(motion, env, m, b, maneuver, depth - 1)
        if second is None:
            return None
    return Path.from_segments([*first.segments, *second.segments])


def plan_clear_leg(
    motion,
    env,
    a: Pose,
    b: Pose,
    maneuver: ManeuverType,
    max_depth: int = _MAX_DEPTH_DEFAULT,
) -> ValidatedLeg:
    """Plan a smoothed leg ``a -> b`` and guarantee it is clear of the map's
    buffered obstacles, repairing a clipping arc via the bounded ladder. Always
    returns a ``ValidatedLeg``; ``.ok`` is False only when even the straight
    corridor chord is obstructed (BLOCKED), which the caller must escalate."""
    leg = motion.plan(a, b, maneuver)
    if env.path_clear(leg, _STEP_M):
        return ValidatedLeg(leg, LegRepair.CLEAN, a, b, maneuver)

    # 1. subdivide-and-resmooth (re-validated as a whole for safety)
    repaired = _resmooth(motion, env, a, b, maneuver, max_depth)
    if repaired is not None and env.path_clear(repaired, _STEP_M):
        return ValidatedLeg(repaired, LegRepair.RESMOOTHED, a, b, maneuver)

    # 2. linear-corridor fallback (the chord; clear by construction of the skeleton)
    chord = motion.straight_leg(a, b, maneuver)
    if chord.is_empty or env.path_clear(chord, _STEP_M):
        return ValidatedLeg(chord, LegRepair.LINEAR, a, b, maneuver)

    # 3. genuinely obstructed -> caller escalates
    return ValidatedLeg(chord, LegRepair.BLOCKED, a, b, maneuver)


def validate_plan(
    motion,
    env,
    poses: list[Pose],
    maneuvers: list[ManeuverType],
    max_depth: int = _MAX_DEPTH_DEFAULT,
) -> list[ValidatedLeg]:
    """Validate a whole leg sequence defined by ``len(poses)`` waypoints and
    ``len(poses) - 1`` per-leg maneuvers. Each leg is validated from its own
    waypoint pose (leg-local), so a repaired leg's altered arrival heading does
    not cascade -- the next leg is still validated clear from its waypoint. The
    caller decides what to do with any BLOCKED leg (skip + log a coverage gap,
    reroute, or hand to runtime recovery)."""
    if len(maneuvers) != max(0, len(poses) - 1):
        raise ValueError("need exactly len(poses) - 1 maneuvers")
    return [
        plan_clear_leg(motion, env, poses[i], poses[i + 1], maneuvers[i], max_depth)
        for i in range(len(poses) - 1)
    ]
