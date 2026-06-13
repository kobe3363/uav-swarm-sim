"""Dubins shortest paths between oriented poses under a minimum turn radius
(after C. Liu et al. 2026). Source of 'true flyable length' for FW/VTOL.

Implementation note
-------------------
The six Dubins words (LSL, RSR, LSR, RSL, RLR, LRL) are evaluated in closed
form. Rather than trusting the closed-form word lengths blindly, every
candidate word is reconstructed geometrically and its endpoint is checked
against the goal pose; words whose reconstruction does not land on the goal are
discarded. This makes the planner self-validating: a formula slip cannot return
a wrong path, it can only (at worst) fall back to another feasible word.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..infrastructure.core_types import (
    Path,
    PathSegment,
    Pose,
    arc_segment,
    mod2pi,
    straight_segment,
)
from ..infrastructure.enums import ManeuverType

_EPS = 1e-9


@dataclass(frozen=True)
class _Word:
    t: float
    p: float
    q: float
    kinds: tuple[str, str, str]  # each of 'L','S','R'

    @property
    def length_norm(self) -> float:
        return self.t + self.p + self.q


def _lsl(a: float, b: float, d: float) -> _Word | None:
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p_sq < 0:
        return None
    tmp = math.atan2(cb - ca, d + sa - sb)
    return _Word(mod2pi(tmp - a), math.sqrt(p_sq), mod2pi(b - tmp), ("L", "S", "L"))


def _rsr(a: float, b: float, d: float) -> _Word | None:
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p_sq < 0:
        return None
    tmp = math.atan2(ca - cb, d - sa + sb)
    return _Word(mod2pi(a - tmp), math.sqrt(p_sq), mod2pi(tmp - b), ("R", "S", "R"))


def _lsr(a: float, b: float, d: float) -> _Word | None:
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    return _Word(mod2pi(tmp - a), p, mod2pi(tmp - b), ("L", "S", "R"))


def _rsl(a: float, b: float, d: float) -> _Word | None:
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    p_sq = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    return _Word(mod2pi(a - tmp), p, mod2pi(b - tmp), ("R", "S", "L"))


def _rlr(a: float, b: float, d: float) -> _Word | None:
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
    q = mod2pi(a - b - t + p)
    return _Word(t, p, q, ("R", "L", "R"))


def _lrl(a: float, b: float, d: float) -> _Word | None:
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(-a + math.atan2(-ca + cb, d + sa - sb) + p / 2.0)
    q = mod2pi(mod2pi(b) - a + 2 * math.pi * 0 - t + p)
    return _Word(t, p, q, ("L", "R", "L"))


_WORD_FNS = (_lsl, _rsr, _lsr, _rsl, _rlr, _lrl)


def _normalized_inputs(start: Pose, goal: Pose, r_min: float) -> tuple[float, float, float]:
    dx = goal.x - start.x
    dy = goal.y - start.y
    big_d = math.hypot(dx, dy)
    d = big_d / r_min
    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(start.heading - theta)
    beta = mod2pi(goal.heading - theta)
    return alpha, beta, d


def _segments_for_word(
    start: Pose, word: _Word, r_min: float, v: float, maneuver: ManeuverType
) -> list[PathSegment]:
    params = (word.t, word.p, word.q)
    segs: list[PathSegment] = []
    cur = start
    for kind, param in zip(word.kinds, params):
        if kind == "S":
            length = param * r_min
            if length <= _EPS:
                continue
            seg = straight_segment(cur, length, maneuver, v)
        else:
            arc_len = param * r_min
            if arc_len <= _EPS:
                continue
            k = (1.0 / r_min) if kind == "L" else (-1.0 / r_min)
            seg = arc_segment(cur, k, arc_len, ManeuverType.TURN, v)
        segs.append(seg)
        cur = seg.end
    return segs


def _endpoint_ok(segs: list[PathSegment], goal: Pose, r_min: float) -> bool:
    if not segs:
        # empty path only valid if already at goal pose
        return False
    end = segs[-1].end
    tol = 1e-6 * max(1.0, r_min)
    pos_ok = math.hypot(end.x - goal.x, end.y - goal.y) <= tol * 10
    head_ok = abs((end.heading - goal.heading + math.pi) % (2 * math.pi) - math.pi) <= 1e-6
    return pos_ok and head_ok


def _best_word(start: Pose, goal: Pose, r_min: float) -> _Word | None:
    alpha, beta, d = _normalized_inputs(start, goal, r_min)
    best: _Word | None = None
    for fn in _WORD_FNS:
        w = fn(alpha, beta, d)
        if w is None:
            continue
        if best is None or w.length_norm < best.length_norm:
            # verify by reconstruction before accepting
            segs = _segments_for_word(start, w, r_min, 1.0, ManeuverType.CRUISE)
            if _endpoint_ok(segs, goal, r_min):
                best = w
    return best


def shortest_path(
    start: Pose, goal: Pose, r_min: float, v: float, maneuver: ManeuverType
) -> Path:
    """Shortest Dubins path. Turn arcs are labeled TURN; the straight gets the
    caller's ``maneuver`` (CRUISE or COVERAGE)."""
    if r_min <= 0:
        raise ValueError("Dubins requires r_min > 0 (holonomic platforms use HolonomicModel)")
    if v <= 0:
        raise ValueError("Dubins requires speed > 0")

    # coincident pose -> empty path
    if math.hypot(goal.x - start.x, goal.y - start.y) < _EPS and abs(
        (goal.heading - start.heading + math.pi) % (2 * math.pi) - math.pi
    ) < 1e-9:
        return Path(())

    word = _best_word(start, goal, r_min)
    if word is None:
        raise RuntimeError("no feasible Dubins word found (should not happen)")
    segs = _segments_for_word(start, word, r_min, v, maneuver)
    return Path.from_segments(segs)


def path_length(start: Pose, goal: Pose, r_min: float) -> float:
    """Cheap length-only query (for cost matrices). Consistent with
    shortest_path total length."""
    if r_min <= 0:
        raise ValueError("Dubins requires r_min > 0")
    if math.hypot(goal.x - start.x, goal.y - start.y) < _EPS and abs(
        (goal.heading - start.heading + math.pi) % (2 * math.pi) - math.pi
    ) < 1e-9:
        return 0.0
    word = _best_word(start, goal, r_min)
    if word is None:
        raise RuntimeError("no feasible Dubins word found")
    return word.length_norm * r_min
