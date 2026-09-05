"""Unit tests for trajectory validation and bounded repair."""
from __future__ import annotations

import math

import pytest

from uav_swarm_sim.infrastructure.core_types import (
    Path,
    PathSegment,
    Pose,
    normalize_angle,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import LegRepair, ManeuverType
from uav_swarm_sim.planning.trajectory_validation import (
    ValidatedLeg,
    _chord_midpoint,
    _resmooth,
    plan_clear_leg,
    validate_plan,
)


def _path_from_lengths(start: Pose, lengths: list[float], maneuver: ManeuverType) -> Path:
    if not lengths:
        return Path()
    segments: list[PathSegment] = []
    current = start
    for length in lengths:
        segment = straight_segment(current, length, maneuver, speed=1.0)
        segments.append(segment)
        current = segment.end
    return Path.from_segments(segments)


class FakeMotion:
    def __init__(self, plan_fn, straight_fn):
        self._plan_fn = plan_fn
        self._straight_fn = straight_fn
        self.plan_calls: list[tuple[Pose, Pose, ManeuverType]] = []
        self.straight_calls: list[tuple[Pose, Pose, ManeuverType]] = []
        self.plan_results: list[Path] = []
        self.straight_results: list[Path] = []

    def plan(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        self.plan_calls.append((start, goal, maneuver))
        path = self._plan_fn(start, goal, maneuver, len(self.plan_calls))
        self.plan_results.append(path)
        return path

    def straight_leg(self, start: Pose, goal: Pose, maneuver: ManeuverType) -> Path:
        self.straight_calls.append((start, goal, maneuver))
        path = self._straight_fn(start, goal, maneuver, len(self.straight_calls))
        self.straight_results.append(path)
        return path


class FakeEnv:
    def __init__(self, path_clear_fn):
        self._path_clear_fn = path_clear_fn
        self.calls: list[tuple[Path, float]] = []

    def path_clear(self, path: Path, step_m: float = 2.0) -> bool:
        self.calls.append((path, step_m))
        return self._path_clear_fn(path, step_m, len(self.calls))


def _scaled_plan_motion(scale: float, straight_length: float = 10.0, *, empty_chord: bool = False) -> FakeMotion:
    def plan_fn(start: Pose, goal: Pose, maneuver: ManeuverType, call_no: int) -> Path:
        del call_no
        distance = math.dist(start.as_xy(), goal.as_xy())
        return _path_from_lengths(start, [scale * distance], maneuver)

    def straight_fn(start: Pose, goal: Pose, maneuver: ManeuverType, call_no: int) -> Path:
        del goal, call_no
        if empty_chord:
            return Path()
        return _path_from_lengths(start, [straight_length], maneuver)

    return FakeMotion(plan_fn, straight_fn)


def test_plan_clear_leg_clean_returns_motion_path_identity() -> None:
    a = Pose(0.0, 0.0, 0.0)
    b = Pose(10.0, 0.0, 0.0)
    motion = _scaled_plan_motion(scale=10.0)
    env = FakeEnv(lambda path, step_m, call_no: True)

    leg = plan_clear_leg(motion, env, a, b, ManeuverType.CRUISE)

    assert leg.path is motion.plan_results[0]
    assert leg.repair is LegRepair.CLEAN
    assert leg.ok is True
    assert leg.repaired is False


def test_plan_clear_leg_resmoothed_subdivides_and_clears() -> None:
    a = Pose(0.0, 0.0, 0.0)
    b = Pose(10.0, 0.0, 0.0)
    maneuver = ManeuverType.COVERAGE
    midpoint = _chord_midpoint(a, b)
    motion = _scaled_plan_motion(scale=10.0)
    env = FakeEnv(lambda path, step_m, call_no: len(path.segments) >= 2 or path.total_length_m <= 60.0)

    leg = plan_clear_leg(motion, env, a, b, maneuver)

    assert leg.repair is LegRepair.RESMOOTHED
    assert leg.ok is True
    assert leg.repaired is True
    assert motion.plan_calls == [(a, b, maneuver), (a, midpoint, maneuver), (midpoint, b, maneuver)]


def test_plan_clear_leg_linear_falls_back_to_clear_chord() -> None:
    a = Pose(0.0, 0.0, 0.0)
    b = Pose(10.0, 0.0, 0.0)
    motion = _scaled_plan_motion(scale=100.0)
    env = FakeEnv(lambda path, step_m, call_no: path.total_length_m <= 15.0)

    leg = plan_clear_leg(motion, env, a, b, ManeuverType.TURN)

    assert leg.repair is LegRepair.LINEAR
    assert leg.ok is True
    assert leg.repaired is True
    assert leg.path is motion.straight_results[0]


def test_plan_clear_leg_blocked_when_even_chord_clips() -> None:
    a = Pose(0.0, 0.0, 0.0)
    b = Pose(10.0, 0.0, 0.0)
    motion = _scaled_plan_motion(scale=100.0)
    env = FakeEnv(lambda path, step_m, call_no: False)

    leg = plan_clear_leg(motion, env, a, b, ManeuverType.TURN)

    assert leg.repair is LegRepair.BLOCKED
    assert leg.ok is False
    assert leg.repaired is False


def test_plan_clear_leg_linear_short_circuits_empty_chord_before_clearance() -> None:
    a = Pose(0.0, 0.0, 0.0)
    b = Pose(10.0, 0.0, 0.0)
    motion = _scaled_plan_motion(scale=100.0, empty_chord=True)

    def path_clear(path: Path, step_m: float, call_no: int) -> bool:
        del step_m, call_no
        if path.is_empty:
            raise AssertionError("empty chord should short-circuit before path_clear")
        return False

    env = FakeEnv(path_clear)

    leg = plan_clear_leg(motion, env, a, b, ManeuverType.TURN, max_depth=0)

    assert leg.repair is LegRepair.LINEAR
    assert leg.ok is True
    assert leg.repaired is True
    assert leg.path.is_empty is True
    assert len(env.calls) == 1


def test_chord_midpoint_matches_midpoint_coordinates_and_heading() -> None:
    a = Pose(2.0, -3.0, 0.75, 1.0)
    b = Pose(-4.0, 5.0, -2.1, 7.0)

    midpoint = _chord_midpoint(a, b)

    assert midpoint.x == pytest.approx((a.x + b.x) / 2.0)
    assert midpoint.y == pytest.approx((a.y + b.y) / 2.0)
    assert midpoint.z == pytest.approx((a.z + b.z) / 2.0)
    assert midpoint.heading == pytest.approx(normalize_angle(math.atan2(b.y - a.y, b.x - a.x)))


def test_resmooth_depth_bound_returns_none_and_terminates() -> None:
    a = Pose(0.0, 0.0, 0.0)
    b = Pose(4.0, 0.0, 0.0)
    motion = _scaled_plan_motion(scale=1.0)
    env = FakeEnv(lambda path, step_m, call_no: False)

    assert _resmooth(motion, env, a, b, ManeuverType.COVERAGE, depth=0) is None
    assert motion.plan_calls == []
    assert env.calls == []

    assert _resmooth(motion, env, a, b, ManeuverType.COVERAGE, depth=2) is None
    assert len(motion.plan_calls) == 4
    assert len(env.calls) == 2


def test_validate_plan_happy_path_is_leg_local() -> None:
    poses = [
        Pose(0.0, 0.0, 0.0),
        Pose(10.0, 0.0, 0.0),
        Pose(20.0, 0.0, 0.0),
    ]
    maneuvers = [ManeuverType.CRUISE, ManeuverType.TURN]
    motion = _scaled_plan_motion(scale=10.0)
    env = FakeEnv(lambda path, step_m, call_no: True)

    legs = validate_plan(motion, env, poses, maneuvers)

    assert len(legs) == 2
    assert [leg.start for leg in legs] == poses[:2]
    assert [leg.goal for leg in legs] == poses[1:]
    assert motion.plan_calls == [
        (poses[0], poses[1], maneuvers[0]),
        (poses[1], poses[2], maneuvers[1]),
    ]


def test_validate_plan_rejects_maneuver_length_mismatch() -> None:
    motion = _scaled_plan_motion(scale=10.0)
    env = FakeEnv(lambda path, step_m, call_no: True)
    poses = [Pose(0.0, 0.0, 0.0), Pose(10.0, 0.0, 0.0), Pose(20.0, 0.0, 0.0)]

    with pytest.raises(ValueError):
        validate_plan(motion, env, poses, [ManeuverType.CRUISE])


@pytest.mark.parametrize(
    ("poses", "maneuvers"),
    [
        ([], []),
        ([Pose(0.0, 0.0, 0.0)], []),
    ],
)
def test_validate_plan_empty_and_single_pose_cases(poses: list[Pose], maneuvers: list[ManeuverType]) -> None:
    motion = _scaled_plan_motion(scale=10.0)
    env = FakeEnv(lambda path, step_m, call_no: True)

    assert validate_plan(motion, env, poses, maneuvers) == []
    assert motion.plan_calls == []
    assert env.calls == []


@pytest.mark.parametrize(
    ("repair", "expected_ok", "expected_repaired"),
    [
        (LegRepair.CLEAN, True, False),
        (LegRepair.RESMOOTHED, True, True),
        (LegRepair.LINEAR, True, True),
        (LegRepair.BLOCKED, False, False),
    ],
)
def test_validated_leg_truth_table(repair: LegRepair, expected_ok: bool, expected_repaired: bool) -> None:
    leg = ValidatedLeg(
        path=Path(),
        repair=repair,
        start=Pose(0.0, 0.0, 0.0),
        goal=Pose(1.0, 0.0, 0.0),
        maneuver=ManeuverType.IDLE,
    )

    assert leg.ok is expected_ok
    assert leg.repaired is expected_repaired