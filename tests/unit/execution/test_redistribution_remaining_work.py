"""EXP-02 redistribution consumes only persistent uncovered raster work."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from shapely.geometry import box

from uav_swarm_sim.execution.redistribution import Redistributor
from uav_swarm_sim.infrastructure.core_types import CoveragePlan, Event, Partition, Pose, Zone
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, EventType
from uav_swarm_sim.planning.coverage_raster import CoverageRaster


class _Agent:
    id = 0
    layer = 0

    def view(self):
        return object()


class _Fleet:
    def __init__(self):
        self.agent = _Agent()
        self.agents = {0: self.agent}

    def active(self):
        return [self.agent]


class _CapturingDecomposer:
    def __init__(self):
        self.target = None

    def decompose(self, tgc, env, views, target_area=None):
        self.target = target_area
        return Partition(DecompositionAlgo.WEIGHTED_VORONOI, {}, 0.0)


def test_redistribution_intersects_pooled_zones_with_uncovered_work():
    whole = box(0.0, 0.0, 100.0, 20.0)
    raster = CoverageRaster(whole, whole, 10.0)
    raster.record_segment(Pose(0.0, 10.0, 0.0), Pose(50.0, 10.0, 0.0), 20.0, 0.1)
    remaining = raster.uncovered_plannable_geometry
    zone = Zone(0, [], whole, Pose(0.0, 0.0, 0.0))
    partition = Partition(DecompositionAlgo.WEIGHTED_VORONOI, {0: zone}, 0.0)
    old_plan = CoveragePlan(0, [], 1.0, 1.0)
    decomposer = _CapturingDecomposer()
    redistributor = Redistributor(
        decomposer,
        SimpleNamespace(by_layer={0: (object(), object())}),
        motion=object(),
        em=object(),
        spec=object(),
        remaining_work_provider=lambda: raster.uncovered_plannable_geometry,
    )

    _new_partition, plans = redistributor.handle(
        Event(EventType.FAILURE, 0.0, {"agent_id": 1}),
        _Fleet(),
        partition,
        {0: old_plan},
        0.0,
    )

    assert decomposer.target.equals(remaining)
    assert plans[0].waypoints == []
    assert plans[0].length_m == 0.0
    assert raster.plannable_coverage_frac == pytest.approx(0.5)


def test_new_task_is_not_clipped_to_initial_raster_work():
    whole = box(0.0, 0.0, 100.0, 20.0)
    added = box(200.0, 0.0, 300.0, 20.0)
    raster = CoverageRaster(whole, whole, 10.0)
    zone = Zone(0, [], whole, Pose(0.0, 0.0, 0.0))
    partition = Partition(DecompositionAlgo.WEIGHTED_VORONOI, {0: zone}, 0.0)
    decomposer = _CapturingDecomposer()
    redistributor = Redistributor(
        decomposer,
        SimpleNamespace(by_layer={0: (object(), object())}),
        motion=object(),
        em=object(),
        spec=object(),
        remaining_work_provider=lambda: raster.uncovered_plannable_geometry,
    )

    redistributor.handle(
        Event(EventType.NEW_TASK, 0.0, {"polygon": added}),
        _Fleet(),
        partition,
        {0: CoveragePlan(0, [], 1.0, 1.0)},
        0.0,
    )

    assert decomposer.target.equals(whole.union(added))
