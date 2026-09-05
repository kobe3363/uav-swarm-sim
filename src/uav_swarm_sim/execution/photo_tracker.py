"""Distance-based shutter triggering for productive coverage-strip passes."""
from __future__ import annotations

import math

from ..infrastructure.core_types import PhotoEvent, Pose

_EPS_M = 1e-9


class PhotoTracker:
    """Accumulate actual travelled distance independently of simulation ``dt``.

    A pass emits at its entry (distance zero), then at every integer multiple of
    ``spacing_m``.  Its endpoint is not forced unless it is itself a cadence
    point.  ``pause`` is represented by simply not calling ``advance``; callers
    use ``finish_pass`` for strip completion, RTH or plan replacement.
    """

    def __init__(self, agent_id: int, spacing_m: float) -> None:
        if not math.isfinite(spacing_m) or spacing_m <= 0.0:
            raise ValueError("photo spacing must be finite and > 0")
        self.agent_id = agent_id
        self.spacing_m = spacing_m
        self.events: list[PhotoEvent] = []
        self.active = False
        self.distance_m = 0.0
        self._next_distance_m = spacing_m

    def start_pass(self, t_s: float, pose: Pose, coverage_leg_index: int) -> None:
        """Start a fresh strip pass and expose once at its entry pose."""
        self.active = True
        self.distance_m = 0.0
        self._next_distance_m = self.spacing_m
        self.events.append(PhotoEvent(
            agent_id=self.agent_id,
            t_s=t_s,
            pose=pose,
            coverage_leg_index=coverage_leg_index,
            distance_on_strip_m=0.0,
        ))

    def advance(
        self,
        old_pose: Pose,
        new_pose: Pose,
        t_start_s: float,
        elapsed_s: float,
        coverage_leg_index: int,
    ) -> None:
        """Record every cadence point crossed during one actual pose update."""
        if not self.active:
            raise RuntimeError("photo pass must be started before advance")
        step_m = math.dist(old_pose.as_xyz(), new_pose.as_xyz())
        if step_m <= _EPS_M:
            return

        before = self.distance_m
        after = before + step_m
        while self._next_distance_m <= after + _EPS_M:
            offset_m = self._next_distance_m - before
            frac = min(1.0, max(0.0, offset_m / step_m))
            pose = Pose(
                old_pose.x + frac * (new_pose.x - old_pose.x),
                old_pose.y + frac * (new_pose.y - old_pose.y),
                new_pose.heading,
                old_pose.z + frac * (new_pose.z - old_pose.z),
            )
            self.events.append(PhotoEvent(
                agent_id=self.agent_id,
                t_s=t_start_s + frac * elapsed_s,
                pose=pose,
                coverage_leg_index=coverage_leg_index,
                distance_on_strip_m=self._next_distance_m,
            ))
            self._next_distance_m += self.spacing_m
        self.distance_m = after

    def finish_pass(self) -> None:
        """End a pass without forcing an extra endpoint exposure."""
        self.active = False
        self.distance_m = 0.0
        self._next_distance_m = self.spacing_m
