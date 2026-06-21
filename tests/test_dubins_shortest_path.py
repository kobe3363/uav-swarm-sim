"""Property-based regression test for the Dubins shortest-path selector.

This guards the Batch-1 fix in ``physical_model/dubins.py``: ``_best_word`` must
return the *shortest* path among all geometrically-validated Dubins words, not
merely a feasible one. The historical defect nested the endpoint-reconstruction
check inside the ``w.length_norm < best.length_norm`` branch, which conflated
"is this shorter than the incumbent" with "should this candidate be validated".

The core invariant asserted here is exactly the property such a bug would
violate: *the returned length is <= the length of every individually-validated
Dubins word* (and, more strongly, equals the minimum over them).

These tests are intentionally white-box -- they reach into the module internals
(`_WORD_FNS`, `_normalized_inputs`, `_segments_for_word`, `_endpoint_ok`) to
build an independent brute-force reference, because the invariant is a property
of the selection logic specifically.
"""
from __future__ import annotations

import itertools
import math

import pytest

from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model import dubins as D

# --------------------------------------------------------------------------- #
# Test grid. Offsets are scaled by r_min so the (alpha, beta, d) geometry is   #
# comparable across radii. Separations stay clear of the coincident regime,    #
# which dubins special-cases to an empty path.                                 #
# --------------------------------------------------------------------------- #
_HEADINGS = [i * math.pi / 4.0 for i in range(8)]          # 45-degree steps
_OFFSETS = [-4.0, -1.5, -0.7, 0.7, 1.5, 4.0]               # x r_min
_RMINS = [1.0, 7.0, 25.0]
_TOL = 1e-6                                                  # length tolerance (meters-ish)


def _grid():
    for h0, gh, dx, dy, r in itertools.product(
        _HEADINGS, _HEADINGS, _OFFSETS, _OFFSETS, _RMINS
    ):
        start = Pose(0.0, 0.0, h0)
        goal = Pose(dx * r, dy * r, gh)
        yield start, goal, r


def _validated_word_lengths(start: Pose, goal: Pose, r_min: float):
    """Independent brute force: normalized length of every Dubins word whose
    geometric reconstruction lands on the goal pose. Mirrors no selection
    logic -- it validates each of the six words in isolation."""
    alpha, beta, d = D._normalized_inputs(start, goal, r_min)
    lengths = []
    for fn in D._WORD_FNS:
        w = fn(alpha, beta, d)
        if w is None:
            continue
        segs = D._segments_for_word(start, w, r_min, 1.0, ManeuverType.CRUISE)
        if D._endpoint_ok(segs, goal, r_min):
            lengths.append((w.kinds, w.length_norm))
    return lengths


def test_returned_length_le_every_validated_word():
    """THE Batch-1 property: the chosen path is no longer than any individually
    valid Dubins word. A selector that can return a feasible-but-not-shortest
    path violates this."""
    checked = 0
    for start, goal, r in _grid():
        valid = _validated_word_lengths(start, goal, r)
        assert valid, f"no validated word for {start} -> {goal}, r={r}"
        chosen = D.path_length(start, goal, r)               # normalized * r
        for kinds, lnorm in valid:
            assert chosen <= lnorm * r + _TOL, (
                f"returned {chosen:.6f} > validated word {''.join(kinds)} "
                f"length {lnorm * r:.6f} for {start} -> {goal}, r={r}"
            )
        checked += 1
    assert checked > 1000                                    # grid actually ran


def test_returns_exact_shortest_among_validated():
    """Stronger than the inequality: the chosen length equals the minimum over
    all validated words (no shorter valid word was skipped)."""
    for start, goal, r in _grid():
        valid = _validated_word_lengths(start, goal, r)
        ref_min = min(lnorm for _, lnorm in valid) * r
        chosen = D.path_length(start, goal, r)
        assert chosen == pytest.approx(ref_min, abs=_TOL), (
            f"chosen {chosen:.6f} != min validated {ref_min:.6f} "
            f"for {start} -> {goal}, r={r}"
        )


def test_returned_path_lands_on_goal():
    """Feasibility guard: the path returned by shortest_path actually reaches
    the goal pose (position and heading), so the selected word is genuinely
    valid, not just closed-form-shortest."""
    for start, goal, r in _grid():
        path = D.shortest_path(start, goal, r, 12.0, ManeuverType.CRUISE)
        end = path.end_pose
        assert end is not None
        pos_err = math.hypot(end.x - goal.x, end.y - goal.y)
        head_err = abs((end.heading - goal.heading + math.pi) % (2 * math.pi) - math.pi)
        assert pos_err <= 1e-4 * max(1.0, r), f"pos_err={pos_err:.2e} {start}->{goal} r={r}"
        assert head_err <= 1e-5, f"head_err={head_err:.2e} {start}->{goal} r={r}"


def test_path_length_matches_shortest_path():
    """The cheap length-only query must agree with the constructed path length:
    cost matrices (which call path_length) must mirror what execution flies."""
    for start, goal, r in _grid():
        length_only = D.path_length(start, goal, r)
        path = D.shortest_path(start, goal, r, 12.0, ManeuverType.CRUISE)
        assert length_only == pytest.approx(path.total_length_m, abs=_TOL)


def test_three_arc_words_are_exercised():
    """Ensure the grid is not vacuous: the rare three-arc words (RLR / LRL) --
    the regime touched by the _lrl cleanup -- are actually selected somewhere,
    so the property tests above are exercising them rather than skipping them."""
    selected = set()
    for start, goal, r in _grid():
        alpha, beta, d = D._normalized_inputs(start, goal, r)
        # recover the selected word's kinds via the public selector
        path = D.shortest_path(start, goal, r, 12.0, ManeuverType.CRUISE)
        # classify by curvature signs across arc segments
        kinds = tuple(
            "S" if s.curvature == 0.0 else ("L" if s.curvature > 0 else "R")
            for s in path.segments
        )
        selected.add(kinds)
    has_rlr = ("R", "L", "R") in selected
    has_lrl = ("L", "R", "L") in selected
    assert has_rlr or has_lrl, f"no three-arc word selected; got {selected}"


def test_coincident_pose_is_empty_path():
    """Boundary: identical start/goal yields an empty path and zero length
    (the special case guarded ahead of _best_word)."""
    p = Pose(10.0, -5.0, 1.1, z=0.0)
    path = D.shortest_path(p, p, 5.0, 12.0, ManeuverType.CRUISE)
    assert path.is_empty
    assert D.path_length(p, p, 5.0) == 0.0
