"""Event-driven redistribution (guideline 3.2).

A failure or a new task immediately re-partitions the affected work among the
*active* agents via the (weighted) TGC decomposer, reading momentary battery
levels so the central weighting acts at exactly this moment. Battery swaps never
reach this handler -- the swapped drone resumes its own remaining plan -- so the
swap/failure asymmetry is enforced by wiring (TRIGGERS) rather than convention.

2.5D (Batch 4): redistribution is per layer. A failure re-partitions only the
failed drone's LAYER, among that layer's surviving drones, on that layer's sliced
graph; drones on other layers keep their zones. New tasks default to the primary
layer (0). With a single layer (every drone on layer 0, whose graph is the 2D
map) this reduces to the original whole-fleet redistribution byte-for-byte.
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

TRIGGERS = {EventType.FAILURE, EventType.NEW_TASK}


class Redistributor:
    def __init__(
        self,
        decomposer: Decomposer,
        layer_graphs,
        motion: MotionModel,
        em: EnergyModel,
        spec: PlatformSpec,
        coverage=None,
    ) -> None:
        self._dec = decomposer
        self._layer_graphs = layer_graphs   # LayerGraphs: by_layer[idx] -> (env, tgc)
        self._motion = motion
        self._em = em
        self._spec = spec
        self._coverage = coverage           # S_FERRY Step 2 routing config (or None)
        self.last_replan_time_s = 0.0

    @staticmethod
    def should_handle(event: Event) -> bool:
        return event.type in TRIGGERS

    def _layer_of(self, fleet, drone_id: int) -> int:
        a = fleet.agents.get(drone_id)
        return getattr(a, "layer", 0) if a is not None else 0

    def _graph_for(self, layer: int):
        by_layer = self._layer_graphs.by_layer
        return by_layer.get(layer) or by_layer[0]

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

        # which layer this event re-partitions
        if event.type is EventType.FAILURE:
            failed = event.payload.get("agent_id")
            layer = self._layer_of(fleet, failed)
        else:  # NEW_TASK -> primary layer
            failed = None
            layer = 0

        env_l, tgc_l = self._graph_for(layer)
        active_l = [a for a in active if getattr(a, "layer", 0) == layer]
        active_l_ids = {a.id for a in active_l}

        # affected area: union of this layer's active zones, plus the failed
        # drone's zone (failure) or the new-task polygon. Faithful: the failed
        # zone is pooled and redistributed among same-layer survivors.
        polys = [
            z.polygon for did, z in partition.zones.items()
            if did in active_l_ids and not z.polygon.is_empty
        ]
        if event.type is EventType.FAILURE:
            fz = partition.zones.get(failed)
            if fz is not None and not fz.polygon.is_empty:
                polys.append(fz.polygon)
        elif event.type is EventType.NEW_TASK:
            np_poly = event.payload.get("polygon")
            if isinstance(np_poly, Polygon):
                polys.append(np_poly)

        target = unary_union(polys) if polys else None

        views = [a.view() for a in active_l]
        new_part_l = self._dec.decompose(tgc_l, env_l, views, target_area=target)
        for did, zone in new_part_l.zones.items():
            zone.layer = layer

        # merge: other-layer zones unchanged; affected layer replaced
        new_zones = {}
        for did, z in partition.zones.items():
            if did not in active_ids:
                continue
            if self._layer_of(fleet, did) != layer:
                new_zones[did] = z
        new_zones.update(new_part_l.zones)
        new_part = Partition(partition.algo, new_zones, new_part_l.planning_time_s)

        new_plans: dict[int, CoveragePlan] = dict(plans)
        for a in active_l:
            zone = new_part_l.zones.get(a.id)
            if zone is None:
                continue
            new_plans[a.id] = boustrophedon(zone, self._spec, self._motion, self._em,
                                            env=env_l, coverage=self._coverage)

        self.last_replan_time_s = time.perf_counter() - t0
        return new_part, new_plans
