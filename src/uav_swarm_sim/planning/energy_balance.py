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

import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from ..infrastructure.config import Config, CoverageConfig
from ..infrastructure.core_types import Path, Pose, Zone
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
from .visibility_router import RouteUnavailable, route_transit, _path_clear, flyable_region


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
    execution_coherent: bool = False


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


@dataclass(frozen=True)
class ResolvedZoneEntry:
    """A physical entry pose and the exact validated ferry path that reaches it."""

    anchor_pose: Pose
    ferry_path: Path


def build_energy_balance_context(
    cfg: Config, em: EnergyModel, spec: PlatformSpec, motion: MotionModel,
    env: EnvironmentMap | None, return_energy: Callable[[Pose, float | None], float],
    emap: EnergyMap | None = None, graph_cache: dict | None = None,
) -> EnergyBalanceContext:
    return EnergyBalanceContext(
        em, spec, motion, env, cfg.coverage, cfg.sensor.sensor_power_w,
        tuple(cfg.layers.altitudes_m), cfg.rth.reserve_frac * spec.battery_capacity_j,
        return_energy, emap, graph_cache, cfg.rth.execution_coherent,
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
        enabled=ctx.coverage.transit_free_space and ctx.env is not None,
        operating_area=ctx.coverage.operating_area,
        margin_m=ctx.coverage.operating_margin_m, graph_cache=ctx.transit_graph_cache,
    )


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """Positive-area polygonal parts in a geometry, in canonical order."""
    if isinstance(geometry, Polygon):
        parts = [geometry] if geometry.area > 1e-9 else []
    else:
        parts = []
        for part in getattr(geometry, "geoms", ()):
            parts.extend(_polygon_parts(part))

    def key(part: Polygon):
        normal = shapely.normalize(part)
        return (*map(float, part.bounds), float(part.area), normal.wkb)

    return sorted(parts, key=key)


def resolve_reachable_zone_entry(
    ctx,
    drone,
    work_geometry: BaseGeometry | None,
    centroid_xy,
    fallback_pose: Pose,
) -> ResolvedZoneEntry:
    """Resolve the mathematical centroid to a physical, reachable zone entry.

    Lloyd's area centroid remains the mass point used by the partitioner. This
    function is only the physical seam: with obstacle-aware transit enabled it
    checks every polygonal work component against the router's exact flyable
    region and requires a validated route from the supplied *current* drone
    pose. A reachable point in one component never certifies another one.

    Candidate order is deterministic. A directly reachable centroid is retained;
    otherwise the current-pose-nearest point and the part's point-on-surface are
    considered. Direct routes are preferred, avoiding an expensive detour when
    the same zone has an immediately reachable legal entry. Within that class,
    the successful candidate nearest the mathematical centroid is selected, with
    canonical component order breaking ties. A visibility-route fallback is
    attempted only when no direct candidate exists.

    No environment, disabled obstacle-aware transit, and empty work retain the
    historical fast-estimate anchor behavior.
    """
    if work_geometry is None or work_geometry.is_empty:
        anchor = fallback_pose if work_geometry is not None else Pose(
            float(centroid_xy[0]),
            float(centroid_xy[1]),
            math.atan2(
                float(centroid_xy[1]) - drone.pose.y,
                float(centroid_xy[0]) - drone.pose.x,
            ),
        )
        return ResolvedZoneEntry(anchor, _ferry(ctx, drone, anchor))

    cx, cy = map(float, centroid_xy)
    if ctx.env is None or not ctx.coverage.transit_free_space:
        anchor = Pose(cx, cy, math.atan2(cy - drone.pose.y, cx - drone.pose.x))
        return ResolvedZoneEntry(anchor, _ferry(ctx, drone, anchor))

    region = flyable_region(
        ctx.env.area,
        ctx.env.buffered_obstacles,
        ctx.coverage.operating_area,
        ctx.coverage.operating_margin_m,
    )
    accepted_region = region.buffer(1e-8)
    centroid = Point(cx, cy)
    current = Point(drone.pose.as_xy())
    region_parts = _polygon_parts(accepted_region)
    start_parts = [part for part in region_parts if part.covers(current)]
    if not start_parts:
        raise RouteUnavailable("endpoint_outside_free_space")
    start_region = start_parts[0]

    options: list[tuple[int, float, int, int, ResolvedZoneEntry]] = []
    parts = _polygon_parts(work_geometry)
    if not parts:
        raise RouteUnavailable("zone_has_no_polygonal_work")

    for component_index, component in enumerate(parts):
        outside_area = float(component.difference(start_region).area)
        tolerance = max(1e-9, float(component.area) * 1e-12)
        if outside_area > tolerance:
            raise RouteUnavailable(
                f"zone_component_unreachable:{component_index}:"
                f"outside_start_component_area_m2={outside_area:.12g}"
            )

        safe_parts = _polygon_parts(component.intersection(start_region))
        if not safe_parts:
            raise RouteUnavailable(
                f"zone_component_unreachable:{component_index}:empty"
            )

        component_candidates: list[tuple[float, int, Pose, Path | None]] = []
        for safe_index, safe in enumerate(safe_parts):
            points: list[Point] = []
            if safe.covers(centroid):
                points.append(centroid)
            near = nearest_points(safe, current)[0]
            if not points or not points[-1].equals(near):
                points.append(near)
            surface = safe.representative_point()
            if not any(point.equals(surface) for point in points):
                points.append(surface)

            for candidate_index, candidate in enumerate(points):
                anchor = Pose(
                    float(candidate.x),
                    float(candidate.y),
                    math.atan2(candidate.y - drone.pose.y, candidate.x - drone.pose.x),
                )
                chord = ctx.motion.plan(
                    replace(drone.pose, z=anchor.z), anchor, ManeuverType.CRUISE,
                )
                direct = chord if _path_clear(chord, ctx.env, region=region) else None
                rank = safe_index * 3 + candidate_index
                distance2 = (anchor.x - cx) ** 2 + (anchor.y - cy) ** 2
                component_candidates.append((distance2, rank, anchor, direct))

        direct = [candidate for candidate in component_candidates
                  if candidate[3] is not None]
        failed: set[tuple[float, float]] = set()
        last_error: RouteUnavailable | None = None
        resolved: ResolvedZoneEntry | None = None
        chosen_distance = math.inf
        chosen_rank = 0
        route_class = 1
        if direct:
            direct.sort(key=lambda item: (item[0], item[1]))
            for distance2, rank, anchor, _ in direct:
                try:
                    resolved = ResolvedZoneEntry(anchor, _ferry(ctx, drone, anchor))
                    chosen_distance, chosen_rank, route_class = distance2, rank, 0
                    break
                except RouteUnavailable as exc:
                    last_error = exc
                    failed.add(anchor.as_xy())

        if resolved is None:
            component_candidates.sort(key=lambda item: (item[0], item[1]))
            for distance2, rank, anchor, _ in component_candidates:
                if anchor.as_xy() in failed:
                    continue
                try:
                    resolved = ResolvedZoneEntry(anchor, _ferry(ctx, drone, anchor))
                    chosen_distance, chosen_rank = distance2, rank
                    break
                except RouteUnavailable as exc:
                    last_error = exc

        if resolved is None:
            reason = str(last_error) if last_error is not None else "no_valid_endpoint"
            raise RouteUnavailable(
                f"zone_component_unreachable:{component_index}:{reason}"
            ) from last_error
        options.append(
            (route_class, chosen_distance, component_index, chosen_rank, resolved)
        )

    options.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return options[0][4]


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
    if ctx.env is not None:
        if ctx.execution_coherent:
            region = flyable_region(ctx.env.area, ctx.env.buffered_obstacles,
                                     ctx.coverage.operating_area, ctx.coverage.operating_margin_m)
            clear = _path_clear(ferry_path, ctx.env, region=region)
        else:
            clear = ctx.env.path_clear(ferry_path)
        if not clear:
            status = EnergyBalanceStatus.FERRY_BLOCKED
    if ctx.emap is not None and not ctx.execution_coherent:
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


def coverage_energy_density_j_per_m2(ctx, altitude_m: float) -> float:
    """Marginal coverage cost, J per m^2 of remaining work.

    Derived by evaluating the SAME executor-mirroring calls this module already
    uses, at A = 1 m^2. Both terms are strictly linear in A, so the evaluation is
    exact rather than a fit::

        strip_length(A) = A / swath
        E_strips(A)     = P_COVERAGE * A / (v_coverage * swath)
        E_camera(A)     = P_sensor   * A / (v_coverage * swath)
        rho             = (P_COVERAGE + P_sensor) / (v_coverage * swath)

    Unit discipline (CLAUDE.md rule 4): the next-bundle term is COVERAGE plus the
    camera; E_home uses CRUISE and is not part of this. The turn/connector term is
    O(sqrt(A)), i.e. NOT linear in area, and is deliberately excluded from a
    MARGINAL density -- the exact demand still comes from the estimators below.

    EXP-07b uses ``1 / rho`` as the J -> m^2 scale in its weight law, so that
    constant is derived from the energy model rather than chosen by hand.
    """
    swath = ctx.spec.coverage_line_spacing_m(altitude_m)
    per_m2 = 1.0 / swath
    return (
        ctx.em.distance_energy(per_m2, ManeuverType.COVERAGE, ctx.spec.v_coverage)
        + ctx.em.sensor_energy(per_m2 / ctx.spec.v_coverage, ctx.sensor_power_w)
    )


def estimate_fast_from_area(
    ctx, drone, *, alt: float, area_m2: float, centroid_xy, fallback_pose: Pose,
    work_geometry: BaseGeometry | None = None,
) -> ZoneEnergyEstimate:
    """The fast estimate's arithmetic, given the remaining area and its centroid.

    Extracted so a caller that ALREADY knows the remaining area -- the EXP-07
    grid partitioner, which owns the cells it summed -- can skip rebuilding the
    work geometry. That matters twice over: ``remaining_work_geometry`` rebuilds
    a union over every uncovered cell on each call, and recomputing the area from
    geometry would give a second, slightly different number for the same zone.

    ``estimate_fast`` below is this function plus the geometry step, so the two
    paths are the same arithmetic on the same inputs.
    """
    swath = ctx.spec.coverage_line_spacing_m(alt)
    strip_length, _, turn_distance = _coverage_geometry(area_m2, swath)
    if work_geometry is not None and not math.isclose(
        float(work_geometry.area), area_m2, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError(
            "work_geometry area must match area_m2: "
            f"{float(work_geometry.area)!r} != {area_m2!r}"
        )
    resolved = resolve_reachable_zone_entry(
        ctx,
        drone,
        work_geometry if area_m2 else Polygon(),
        centroid_xy,
        fallback_pose,
    )
    anchor = resolved.anchor_pose
    routed_ferry = bool(
        area_m2
        and work_geometry is not None
        and ctx.env is not None
        and ctx.coverage.transit_free_space
    )
    ferry_energy = (
        ctx.em.path_energy(resolved.ferry_path)
        if routed_ferry
        else ctx.em.distance_energy(
            math.dist(drone.pose.as_xy(), anchor.as_xy()),
            ManeuverType.CRUISE,
            ctx.spec.v_cruise,
        )
    )
    return _estimate(
        ctx, drone, "fast", alt, area_m2, math.sqrt(area_m2) / swath, anchor, anchor,
        ferry_energy,
        ctx.em.distance_energy(strip_length, ManeuverType.COVERAGE, ctx.spec.v_coverage),
        ctx.em.distance_energy(turn_distance, ManeuverType.TURN, ctx.spec.v_cruise),
        ctx.em.sensor_energy(strip_length / ctx.spec.v_coverage, ctx.sensor_power_w),
        resolved.ferry_path,
    )


def estimate_fast(ctx, drone, zone, raster: CoverageRaster | None) -> ZoneEnergyEstimate:
    """Square-footprint work approximation, anchored at the remaining centroid."""
    alt = _inputs(ctx, drone, zone)
    geometry = remaining_work_geometry(zone.polygon, raster)
    area = geometry.area
    centroid = geometry.centroid if area else None
    return estimate_fast_from_area(
        ctx, drone, alt=alt, area_m2=area,
        centroid_xy=(centroid.x, centroid.y) if area else (0.0, 0.0),
        fallback_pose=zone.entry_pose,
        work_geometry=geometry,
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
