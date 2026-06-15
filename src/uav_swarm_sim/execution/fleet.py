"""Homogeneous fleet container and active/failed registry.

The only place agents are created or removed. ``kill`` freezes a failed agent in
S_FAIL and removes it from ``active()`` -- it does NOT respawn it: in the physical
simulation failure is irreversible. The SMDP slot-replacement closure happens in
the metrics layer only.
"""
from __future__ import annotations

import math

from ..infrastructure.core_types import DroneStateView, Pose, normalize_angle
from ..infrastructure.enums import AgentState
from .agent import Agent


def deploy_ring_poses(
    base: Pose,
    n: int,
    dims_m: tuple[float, float, float],
    min_separation_m: float,
    outward_heading: bool = True,
) -> list[Pose]:
    """Evenly spaced t=0 staging poses on a circle centred on ``base``.

    Launching every drone from one identical (x, y) pose overlaps them, which
    (1) trips the SafetyMonitor the moment they are airborne and (2) hands the
    decomposer N identical seeds, collapsing the whole partition onto a single
    drone. This spaces the N drones at angles ``theta_k = 2*pi*k/N`` on the
    smallest ring that keeps every ADJACENT pair at least ``min_separation_m``
    apart at the surface: their circular footprints (diameter ``hypot(L, W)``)
    never touch, and the centre-to-centre gap exceeds ``min_separation_m`` so the
    SafetyMonitor -- which compares centre distance against ``min_separation_m``
    -- raises no t=0 separation warning.

    Closed form. N drones on a regular N-gon have their CLOSEST pair at adjacent
    vertices, a chord ``2*R*sin(pi/N)`` apart. Requiring that chord to be at
    least ``s := hypot(L, W) + min_separation_m`` gives

        R = s / (2 * sin(pi / N))     for N >= 2
        R = 0                          for N <= 1   (single drone stays on base,
                                                     byte-identical to pre-2.4)

    Headings point radially outward (``outward_heading=True``) so the swarm fans
    out, otherwise they keep ``base.heading``. Altitude ``z`` is inherited from
    ``base`` -- the ring is horizontal at the launch altitude.
    """
    if n <= 1:
        # Single drone (or none): stay exactly on the base pose. R == 0 keeps the
        # single-drone / single-layer case byte-identical to the pre-2.4 model.
        return [base] if n == 1 else []
    footprint_diameter = math.hypot(dims_m[0], dims_m[1])
    required_sep = footprint_diameter + min_separation_m
    radius = required_sep / (2.0 * math.sin(math.pi / n))
    poses: list[Pose] = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        x = base.x + radius * math.cos(theta)
        y = base.y + radius * math.sin(theta)
        heading = normalize_angle(theta) if outward_heading else base.heading
        poses.append(Pose(x, y, heading, base.z))
    return poses


class Fleet:
    def __init__(self, agents: list[Agent]) -> None:
        self.agents: dict[int, Agent] = {a.id: a for a in agents}
        self._failed: set[int] = set()

    def active(self) -> list[Agent]:
        return [a for aid, a in self.agents.items() if aid not in self._failed]

    def airborne(self) -> list[Agent]:
        return [a for a in self.active() if a.state.is_airborne]

    def kill(self, agent_id: int, t: float) -> None:
        agent = self.agents.get(agent_id)
        if agent is None:
            return
        agent.state = AgentState.S_FAIL
        self._failed.add(agent_id)

    def views(self) -> list[DroneStateView]:
        return [a.view() for a in self.active()]

    @property
    def n_failed(self) -> int:
        return len(self._failed)
