"""Auditable remaining-work demand and usable battery budget (joules).

Takeoff is deducted only before launch; airborne levels already paid it. The
reserve is held once in the budget, and the injected return cost includes
landing exactly once. Camera energy belongs only to remaining coverage strips.
Injecting the identical RTH callable is the architectural reuse contract:
planning must not import execution. Bind its altitude argument by keyword.

EXP-07 consumes demand_j, budget_j and status, never the ratio alone. The
estimate is optimistic by up to one tick per leg: the executor charges a full
tick even when the last part of a leg is shorter. No tick correction is applied
here. Fast and path components remain separate; they are never blended.

The fast anchor faces the ferry bearing. Fast ferry energy remains a constant-
time distance surrogate without yaw (at most P_TURN*pi/omega_max omitted,
754 J in the pinned fixture, covered by its fast/path tolerance). Return
energy includes yaw through return_energy, including the base heading.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Callable, Literal

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..infrastructure.config import Config, CoverageConfig
from ..infrastructure.core_types import Pose, Zone
from ..infrastructure.enums import ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import MotionModel
from ..physical_model.vertical_segments import takeoff_profile
from .coverage_raster import CoverageRaster
from .coverage_path import boustrophedon
from .energy_map import EnergyMap
from .environment_map import EnvironmentMap
from .launch_site_optimizer import _coverage_geometry
from .visibility_router import route_transit


class EnergyBalanceStatus(Enum):
    FEASIBLE = "feasible"
    BUDGET_NONPOSITIVE = "budget_nonpositive"
    FERRY_BLOCKED = "ferry_blocked"
    RTH_UNREACHABLE = "rth_unreachable"


@dataclass(frozen=True)
class DroneEnergyState:
    drone_id: int
    pose: Pose
    level_j: float
    airborne: bool


@dataclass(frozen=True)
class EnergyBalanceContext:
    em: EnergyModel
    spec: PlatformSpec
    motion: MotionModel
    env: EnvironmentMap | None
    coverage: CoverageConfig
    sensor_power_w: float
    layer_altitudes_m: tuple[float, ...]
    reserve_j: float
    return_energy: Callable[[Pose, float | None], float]
    emap: EnergyMap | None = None
    transit_graph_cache: dict | None = None


@dataclass(frozen=True)
class ZoneEnergyEstimate:
    drone_id: int
    method: Literal["fast", "path"]
    status: EnergyBalanceStatus
    e_level_j: float
    e_takeoff_deducted_j: float
    e_remaining_j: float
    e_ferry_j: float
    e_strips_j: float
    e_connectors_j: float
    e_camera_j: float
    e_coverage_j: float
    e_rth_j: float
    e_reserve_j: float
    budget_j: float
    demand_j: float
    demand_budget_ratio: float | None
    remaining_area_m2: float
    n_strips: float
    anchor_pose: Pose
    exit_pose: Pose


def build_energy_balance_context(
    cfg: Config, em: EnergyModel, spec: PlatformSpec, motion: MotionModel,
    env: EnvironmentMap | None, return_energy: Callable[[Pose, float | None], float],
    emap: EnergyMap | None = None, graph_cache: dict | None = None,
) -> EnergyBalanceContext:
    return EnergyBalanceContext(
        em, spec, motion, env, cfg.coverage, cfg.sensor.sensor_power_w,
        tuple(cfg.layers.altitudes_m), cfg.rth.reserve_frac * spec.battery_capacity_j,
        return_energy, emap, graph_cache,
    )


def remaining_work_geometry(zone_polygon, raster: CoverageRaster | None) -> BaseGeometry:
    """Intersect the zone with persistent raster work; discard non-area parts."""
    geometry = zone_polygon if raster is None else zone_polygon.intersection(
        raster.uncovered_plannable_geometry
    )

    def polygons(g):
        if isinstance(g, Polygon):
            if g.area > 1e-9:
                yield g
        elif hasattr(g, "geoms"):
            for part in g.geoms:
                yield from polygons(part)

    parts = list(polygons(geometry))
    return unary_union(parts) if parts else Polygon()


def _finite(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")


def _inputs(ctx, drone, zone):
    alt = ctx.layer_altitudes_m[zone.layer]
    _finite(level_j=drone.level_j, reserve_j=ctx.reserve_j,
            sensor_power_w=ctx.sensor_power_w, altitude_m=alt,
            v_cruise=ctx.spec.v_cruise, v_coverage=ctx.spec.v_coverage,
            v_climb=ctx.spec.v_climb, v_descent=ctx.spec.v_descent,
            mass_kg=ctx.spec.mass_kg, capacity_j=ctx.spec.battery_capacity_j,
            omega_max=ctx.spec.omega_max, r_min_m=ctx.spec.r_min_m,
            swath=ctx.spec.coverage_line_spacing_m(alt))
    for pose in (drone.pose, zone.entry_pose):
        _finite(x=pose.x, y=pose.y, heading=pose.heading, z=pose.z)
    for power in ctx.spec.power_w.values():
        _finite(power_w=power)
    _finite(area_m2=zone.polygon.area)
    return alt


def _budget(ctx, drone, altitude_m, e_ferry_j, e_rth_j) -> tuple[float, float, float]:
    """The only assembly point for takeoff, remaining charge and denominator."""
    takeoff = 0.0 if drone.airborne else takeoff_profile(ctx.spec, ctx.em, altitude_m).energy_j
    remaining = drone.level_j - takeoff
    budget = remaining - (e_ferry_j + e_rth_j + ctx.reserve_j)
    _finite(e_takeoff_deducted_j=takeoff, e_remaining_j=remaining, budget_j=budget)
    return takeoff, remaining, budget


def _ferry(ctx, drone, anchor):
    # Horizontal executor paths use z=0; takeoff is accounted for by _budget.
    return route_transit(
        replace(drone.pose, z=anchor.z), anchor, ctx.motion, ctx.env,
        enabled=ctx.coverage.transit_free_space,
        operating_area=ctx.coverage.operating_area,
        margin_m=ctx.coverage.operating_margin_m, graph_cache=ctx.transit_graph_cache,
    )


def _estimate(ctx, drone, method, alt, area, n_strips, anchor, exit_pose,
              ferry, strips, connectors, camera, ferry_path):
    rth = ctx.return_energy(exit_pose, alt)
    demand = strips + connectors + camera
    _finite(e_ferry_j=ferry, e_strips_j=strips, e_connectors_j=connectors,
            e_camera_j=camera, e_rth_j=rth, demand_j=demand,
            remaining_area_m2=area, n_strips=n_strips,
            anchor_x=anchor.x, anchor_y=anchor.y, exit_x=exit_pose.x, exit_y=exit_pose.y)
    takeoff, remaining, budget = _budget(ctx, drone, alt, ferry, rth)
    status = EnergyBalanceStatus.FEASIBLE
    if budget <= 0:
        status = EnergyBalanceStatus.BUDGET_NONPOSITIVE
    # Physical obstructions take precedence over a depleted budget; all terms
    # are still returned, so consumers can inspect both conditions.
    if ctx.env is not None and not ctx.env.path_clear(ferry_path):
        status = EnergyBalanceStatus.FERRY_BLOCKED
    if ctx.emap is not None:
        frame = ctx.emap.frame
        i, j = frame.world_to_cell(exit_pose.x, exit_pose.y)
        if 0 <= i < frame.nx and 0 <= j < frame.ny:
            if not math.isfinite(float(ctx.emap.e_home[i, j])):
                status = EnergyBalanceStatus.RTH_UNREACHABLE
    ratio = demand / budget if status is EnergyBalanceStatus.FEASIBLE else None
    if ratio is not None:
        _finite(demand_budget_ratio=ratio)
    return ZoneEnergyEstimate(
        drone.drone_id, method, status, drone.level_j, takeoff, remaining,
        ferry, strips, connectors, camera, demand, rth, ctx.reserve_j,
        budget, demand, ratio, area, n_strips, anchor, exit_pose,
    )


def estimate_fast(ctx, drone, zone, raster: CoverageRaster | None) -> ZoneEnergyEstimate:
    """Square-footprint work approximation, anchored at the remaining centroid."""
    alt = _inputs(ctx, drone, zone)
    geometry = remaining_work_geometry(zone.polygon, raster)
    area = geometry.area
    swath = ctx.spec.coverage_line_spacing_m(alt)
    strip_length, _, turn_distance = _coverage_geometry(area, swath)
    if area:
        center = geometry.centroid
        anchor = Pose(center.x, center.y,
                      math.atan2(center.y - drone.pose.y, center.x - drone.pose.x))
    else:
        anchor = zone.entry_pose
    return _estimate(
        ctx, drone, "fast", alt, area, math.sqrt(area) / swath, anchor, anchor,
        ctx.em.distance_energy(math.dist(drone.pose.as_xy(), anchor.as_xy()),
                               ManeuverType.CRUISE, ctx.spec.v_cruise),
        ctx.em.distance_energy(strip_length, ManeuverType.COVERAGE, ctx.spec.v_coverage),
        ctx.em.distance_energy(turn_distance, ManeuverType.TURN, ctx.spec.v_cruise),
        ctx.em.sensor_energy(strip_length / ctx.spec.v_coverage, ctx.sensor_power_w),
        _ferry(ctx, drone, anchor),
    )


def estimate_path(ctx, drone, zone, raster: CoverageRaster | None) -> ZoneEnergyEstimate:
    """Authoritative sweep with executor-equivalent connectors and first target."""
    assert ctx.spec.photogrammetry is not None, (
        "estimate_path requires photogrammetry: energy_balance.enabled -> "
        "coverage.raster_enabled -> sensor.photogrammetry.enabled"
    )
    alt = _inputs(ctx, drone, zone)
    geometry = remaining_work_geometry(zone.polygon, raster)
    plan = boustrophedon(
        Zone(drone.drone_id, [], geometry, zone.entry_pose, zone.layer),
        ctx.spec, ctx.motion, ctx.em, ctx.env, ctx.coverage, alt,
    )
    _finite(strips_energy_j=plan.strips_energy_j,
            connectors_energy_j=plan.connectors_energy_j, est_energy_j=plan.est_energy_j)
    assert math.isclose(plan.strips_energy_j + plan.connectors_energy_j,
                        plan.est_energy_j, rel_tol=0.0, abs_tol=1e-6)
    anchor = plan.waypoints[0].pose if plan.waypoints else zone.entry_pose
    exit_pose = plan.waypoints[-1].pose if plan.waypoints else zone.entry_pose
    strip_length = sum(math.dist(start.pose.as_xy(), end.pose.as_xy())
                       for start, end in zip(plan.waypoints[::2], plan.waypoints[1::2]))
    ferry_path = _ferry(ctx, drone, anchor)
    return _estimate(
        ctx, drone, "path", alt, geometry.area, float(len(plan.waypoints) // 2),
        anchor, exit_pose, ctx.em.path_energy(ferry_path),
        plan.strips_energy_j, plan.connectors_energy_j,
        ctx.em.sensor_energy(strip_length / ctx.spec.v_coverage, ctx.sensor_power_w),
        ferry_path,
    )
