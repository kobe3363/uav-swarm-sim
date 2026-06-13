"""Event-driven redistribution (guideline 3.2).

A failure or a new task immediately re-partitions the affected work among the
*active* agents via the (weighted) TGC decomposer, reading momentary battery
levels so the central weighting acts at exactly this moment. Battery swaps never
reach this handler -- the swapped drone resumes its own remaining plan -- so the
swap/failure asymmetry is enforced by wiring (TRIGGERS) rather than convention.
"""
from __future__ import annotations

import time

from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..infrastructure.core_types import CoveragePlan, Event, Partition
from ..infrastructure.enums import EventType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel
from ..planning.coverage_path import boustrophedon
from ..planning.decomposition_base import Decomposer
from ..planning.tgc import TGCGraph

TRIGGERS = {EventType.FAILURE, EventType.NEW_TASK}


class Redistributor:
    def __init__(
        self,
        decomposer: Decomposer,
        tgc: TGCGraph,
        env,
        motion: MotionModel,
        em: EnergyModel,
        spec: PlatformSpec,
    ) -> None:
        self._dec = decomposer
        self._tgc = tgc
        self._env = env
        self._motion = motion
        self._em = em
        self._spec = spec
        self.last_replan_time_s = 0.0

    @staticmethod
    def should_handle(event: Event) -> bool:
        return event.type in TRIGGERS

    def handle(
        self,
        event: Event,
        fleet,
        partition: Partition,
        plans: dict[int, CoveragePlan],
        t: float,
    ) -> tuple[Partition, dict[int, CoveragePlan]]:
        if not self.should_handle(event):
            raise ValueError(f"redistribution does not handle {event.type} (swap is reversible)")

        t0 = time.perf_counter()
        active = fleet.active()
        active_ids = {a.id for a in active}

        # affected area: union of active zones, plus the failed drone's zone or
        # the new-task polygon. (Failure-faithful: the failed zone is pooled and
        # redistributed among survivors.)
        polys = [z.polygon for did, z in partition.zones.items() if did in active_ids and not z.polygon.is_empty]
        if event.type is EventType.FAILURE:
            failed = event.payload.get("agent_id")
            fz = partition.zones.get(failed)
            if fz is not None and not fz.polygon.is_empty:
                polys.append(fz.polygon)
        elif event.type is EventType.NEW_TASK:
            np_poly = event.payload.get("polygon")
            if isinstance(np_poly, Polygon):
                polys.append(np_poly)

        target = unary_union(polys) if polys else None

        views = [a.view() for a in active]
        new_part = self._dec.decompose(self._tgc, self._env, views, target_area=target)

        new_plans: dict[int, CoveragePlan] = dict(plans)
        for a in active:
            zone = new_part.zones.get(a.id)
            if zone is None:
                continue
            new_plans[a.id] = boustrophedon(zone, self._spec, self._motion, self._em)

        self.last_replan_time_s = time.perf_counter() - t0
        return new_part, new_plans
