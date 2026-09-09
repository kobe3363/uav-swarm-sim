"""REV-01: mathematical Lloyd centroids are not physical ferry endpoints."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, box

import uav_swarm_sim.planning.energy_balance as balance
import uav_swarm_sim.planning.lloyd_partition as lloyd
from uav_swarm_sim.infrastructure.config import PartitionConfig
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.planning.energy_balance import (
    EnergyBalanceStatus,
    estimate_fast_from_area,
    resolve_reachable_zone_entry,
)
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.lloyd_partition import (
    EligibleCells,
    EnergyWeightPolicy,
    LloydPartitioner,
)
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.planning.visibility_router import (
    RouteUnavailable,
    _path_clear,
    flyable_region,
)


def _obstacle(polygon, obstacle_id: int = 0) -> Obstacle:
    return Obstacle(obstacle_id, 0, polygon)


def _routing_context(case, env: EnvironmentMap, *, operating_area: str = "survey"):
    return replace(
        case.ctx,
        env=env,
        coverage=replace(
            case.ctx.coverage,
            transit_free_space=True,
            operating_area=operating_area,
            operating_margin_m=0.0,
        ),
        transit_graph_cache={},
        execution_coherent=True,
    )


@pytest.mark.parametrize(
    ("centroid_xy", "inside_raw"),
    [((50.0, 50.0), True), ((63.0, 50.0), False)],
    ids=["raw-obstacle", "clearance-only"],
)
def test_blocked_centroid_resolves_to_zone_point_and_real_ferry_energy(
    energy_case, centroid_xy, inside_raw,
):
    env = EnvironmentMap(
        box(0.0, 0.0, 100.0, 100.0),
        [_obstacle(box(40.0, 40.0, 60.0, 60.0))],
        buffer_m=5.0,
    )
    work = env.plannable_space
    drone = replace(energy_case.drone, pose=Pose(10.0, 50.0, 0.0), airborne=True)
    ctx = _routing_context(energy_case, env)

    assert env.in_obstacle(centroid_xy) is inside_raw
    assert not env.contains(centroid_xy)

    estimate = estimate_fast_from_area(
        ctx,
        drone,
        alt=100.0,
        area_m2=float(work.area),
        centroid_xy=centroid_xy,
        fallback_pose=Pose(-999.0, -999.0, 0.0),
        work_geometry=work,
    )
    resolved = resolve_reachable_zone_entry(
        ctx, drone, work, centroid_xy, Pose(-999.0, -999.0, 0.0),
    )
    region = flyable_region(env.area, env.buffered_obstacles, "survey", 0.0)

    assert estimate.status is EnergyBalanceStatus.FEASIBLE
    assert math.isfinite(estimate.e_ferry_j)
    assert estimate.anchor_pose.as_xy() != pytest.approx(centroid_xy)
    assert work.covers(Point(estimate.anchor_pose.as_xy()))
    assert region.buffer(1e-8).covers(Point(estimate.anchor_pose.as_xy()))
    assert _path_clear(resolved.ferry_path, env, region=region)
    assert estimate.e_ferry_j == pytest.approx(
        ctx.em.path_energy(resolved.ferry_path), rel=1e-12,
    )
    assert resolved.ferry_path.segments[0].start.as_xy() == drone.pose.as_xy()


def test_bad_point_candidate_does_not_abort_when_another_is_legal(
    energy_case, monkeypatch,
):
    env = EnvironmentMap(box(0.0, 0.0, 100.0, 100.0), [], buffer_m=0.0)
    ctx = _routing_context(energy_case, env)
    drone = replace(energy_case.drone, pose=Pose(0.0, 20.0, 0.0), airborne=True)
    work = box(20.0, 10.0, 60.0, 50.0)
    real_ferry = balance._ferry
    attempts: list[tuple[float, float]] = []

    def fail_first(context, state, anchor):
        attempts.append(anchor.as_xy())
        if len(attempts) == 1:
            raise RouteUnavailable("candidate_blocked")
        return real_ferry(context, state, anchor)

    monkeypatch.setattr(balance, "_ferry", fail_first)
    resolved = resolve_reachable_zone_entry(
        ctx, drone, work, (40.0, 30.0), drone.pose,
    )

    assert len(attempts) == 2
    assert attempts[0] == (40.0, 30.0)
    assert resolved.anchor_pose.as_xy() == attempts[1]
    assert work.covers(Point(resolved.anchor_pose.as_xy()))


def test_multipolygon_choice_is_deterministic_and_checks_all_components(energy_case):
    env = EnvironmentMap(box(0.0, 0.0, 200.0, 100.0), [], buffer_m=0.0)
    ctx = _routing_context(energy_case, env)
    drone = replace(energy_case.drone, pose=Pose(0.0, 20.0, 0.0), airborne=True)
    work = MultiPolygon([
        box(10.0, 10.0, 30.0, 30.0),
        box(150.0, 10.0, 170.0, 30.0),
    ])

    first = resolve_reachable_zone_entry(ctx, drone, work, (90.0, 20.0), drone.pose)
    second = resolve_reachable_zone_entry(ctx, drone, work, (90.0, 20.0), drone.pose)

    assert first == second
    assert first.anchor_pose.as_xy() == pytest.approx((150.0, 20.0))
    assert any(part.covers(Point(first.anchor_pose.as_xy())) for part in work.geoms)


def test_reachable_component_does_not_hide_unreachable_assigned_work(energy_case):
    survey = box(0.0, 0.0, 200.0, 100.0)
    env = EnvironmentMap(
        survey,
        [_obstacle(box(95.0, 0.0, 105.0, 100.0))],
        buffer_m=0.0,
    )
    ctx = _routing_context(energy_case, env)
    drone = replace(energy_case.drone, pose=Pose(20.0, 20.0, 0.0), airborne=True)
    work = MultiPolygon([
        box(10.0, 10.0, 30.0, 30.0),
        box(150.0, 10.0, 170.0, 30.0),
    ])
    original_area = float(work.area)

    with pytest.raises(
        RouteUnavailable,
        match=r"zone_component_unreachable:1:outside_start_component_area_m2=400",
    ):
        resolve_reachable_zone_entry(ctx, drone, work, (90.0, 20.0), drone.pose)

    assert work.area == pytest.approx(original_area)


def test_geometry_area_mismatch_is_rejected_instead_of_losing_work(energy_case):
    work = box(0.0, 0.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="work_geometry area must match area_m2"):
        estimate_fast_from_area(
            energy_case.ctx,
            energy_case.drone,
            alt=100.0,
            area_m2=99.0,
            centroid_xy=(5.0, 5.0),
            fallback_pose=energy_case.drone.pose,
            work_geometry=work,
        )


def test_energy_policy_skips_zone_unions_when_routing_cannot_use_them(
    energy_case, monkeypatch,
):
    settings = PartitionConfig(
        init_sites="deploy_poses", max_iterations=1, site_tolerance_m=0.0,
    )

    def policy(context):
        return EnergyWeightPolicy(
            context,
            [energy_case.drone],
            100.0,
            settings,
            energy_case.spec.battery_capacity_j,
            [energy_case.drone.pose],
        )

    without_environment = policy(energy_case.ctx)
    assert without_environment.requires_zone_geometry is False

    env = EnvironmentMap(box(0.0, 0.0, 100.0, 100.0), [], buffer_m=0.0)
    disabled_context = replace(
        energy_case.ctx,
        env=env,
        coverage=replace(energy_case.ctx.coverage, transit_free_space=False),
    )
    assert policy(disabled_context).requires_zone_geometry is False
    assert policy(_routing_context(energy_case, env)).requires_zone_geometry is True

    geometries = np.empty(1, dtype=object)
    geometries[0] = box(10.0, 10.0, 20.0, 20.0)
    cells = EligibleCells(
        geometries=geometries,
        centroids_xy=np.array([[15.0, 15.0]]),
        areas_m2=np.array([100.0]),
        component=np.array([0], dtype=np.int32),
        n_rth_unreachable=0,
        n_no_eligible_owner=0,
    )

    def unexpected_union(*_args, **_kwargs):
        pytest.fail("disabled routing must not aggregate zone geometry")

    monkeypatch.setattr(lloyd, "aggregate_geometries", unexpected_union)
    labels, *_ = LloydPartitioner(settings, without_environment).run(
        cells,
        np.array([[energy_case.drone.pose.x, energy_case.drone.pose.y]]),
        np.array([0], dtype=np.int32),
        energy_case.drone.pose,
    )
    assert labels.tolist() == [0]
