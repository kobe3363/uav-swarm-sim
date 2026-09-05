"""Distance-triggered photo events are invariant to timestep subdivision."""
from __future__ import annotations

import pytest

from uav_swarm_sim.execution.photo_tracker import PhotoTracker
from uav_swarm_sim.infrastructure.core_types import Pose


def _signature(tracker):
    return [
        (event.coverage_leg_index, event.distance_on_strip_m,
         event.t_s, event.pose.x, event.pose.y)
        for event in tracker.events
    ]


def test_entry_and_crossed_cadence_points_are_recorded_without_forced_endpoint():
    tracker = PhotoTracker(agent_id=3, spacing_m=20.0)
    tracker.start_pass(5.0, Pose(0.0, 0.0, 0.0), coverage_leg_index=4)
    tracker.advance(Pose(0.0, 0.0, 0.0), Pose(50.0, 0.0, 0.0), 5.0, 5.0, 4)
    tracker.finish_pass()

    assert [event.distance_on_strip_m for event in tracker.events] == [0.0, 20.0, 40.0]
    assert [event.t_s for event in tracker.events] == pytest.approx([5.0, 7.0, 9.0])
    assert [event.pose.x for event in tracker.events] == pytest.approx([0.0, 20.0, 40.0])
    assert all(event.agent_id == 3 for event in tracker.events)


def test_subdividing_the_same_motion_does_not_change_events():
    one = PhotoTracker(agent_id=0, spacing_m=12.5)
    one.start_pass(0.0, Pose(0.0, 0.0, 0.0), 0)
    one.advance(Pose(0.0, 0.0, 0.0), Pose(50.0, 0.0, 0.0), 0.0, 5.0, 0)

    split = PhotoTracker(agent_id=0, spacing_m=12.5)
    split.start_pass(0.0, Pose(0.0, 0.0, 0.0), 0)
    for i in range(5):
        split.advance(
            Pose(10.0 * i, 0.0, 0.0), Pose(10.0 * (i + 1), 0.0, 0.0),
            float(i), 1.0, 0,
        )

    assert _signature(split) == pytest.approx(_signature(one))


def test_pause_preserves_distance_but_new_pass_resets_it():
    tracker = PhotoTracker(agent_id=0, spacing_m=10.0)
    tracker.start_pass(0.0, Pose(0.0, 0.0, 0.0), 0)
    tracker.advance(Pose(0.0, 0.0, 0.0), Pose(6.0, 0.0, 0.0), 0.0, 0.6, 0)
    # No advance while paused (S_OBS); the remaining 4 m reaches the same cadence.
    tracker.advance(Pose(6.0, 0.0, 0.0), Pose(10.0, 0.0, 0.0), 2.0, 0.4, 0)
    tracker.finish_pass()
    tracker.start_pass(5.0, Pose(0.0, 5.0, 0.0), 2)

    assert [(e.coverage_leg_index, e.distance_on_strip_m) for e in tracker.events] == [
        (0, 0.0), (0, 10.0), (2, 0.0),
    ]
