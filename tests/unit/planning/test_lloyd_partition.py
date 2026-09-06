"""EXP-07a: the shared grid partitioner.

Expected values come from hand-computed geometry or from an agreed behavioural
rule -- never from the output of the function under test. Where a number is
pinned, the derivation is written out in the test.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from uav_swarm_sim.execution.fleet import deploy_ring_poses
from uav_swarm_sim.infrastructure.config import PartitionConfig
from uav_swarm_sim.infrastructure.core_types import DroneStateView, Pose
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.planning.coverage_raster import CoverageRaster
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.lloyd_partition import (
    EligibleCells,
    LloydCvtDecomposer,
    aggregate,
    assign_cells,
    build_eligible_cells,
    initial_sites,
    maximin_site_order,
)

SETTINGS = PartitionConfig(init_sites="deploy_poses", max_iterations=50, site_tolerance_m=1.0)


def _decompose(area, poses, settings=SETTINGS, cell_m=10.0, obstacles=(), launch=None):
    env = EnvironmentMap(area, list(obstacles), 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, cell_m)
    dec = LloydCvtDecomposer(
        raster=raster, deploy_poses=poses,
        launch_pose=launch or Pose(area.centroid.x, area.bounds[1], 0.0),
        settings=settings,
    )
    drones = [DroneStateView(i, 1.0, p) for i, p in enumerate(poses)]
    return dec, dec.decompose(None, env, drones)


# --------------------------------------------------------------------------- #
# B. geometry / core                                                           #
# --------------------------------------------------------------------------- #
def test_power_diagram_bisector_sits_where_the_algebra_says():
    """Four unit cells at x = 0.5, 1.5, 2.5, 3.5; sites at x = 0 and x = 4.

    ``d0^2 - w0 < d1^2 - w1``  <=>  ``x^2 - w0 < (x-4)^2 - w1``
                               <=>  ``8x < 16 + w0 - w1``
                               <=>  ``x < 2 + (w0 - w1) / 8``.

    So w equal -> split at 2.0; ``w0 - w1 = 7`` -> split at 2.875; and
    ``w0 - w1 = 4`` puts the boundary EXACTLY on the cell at x = 2.5, which the
    id tie-break must award to drone 0.
    """
    xy = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]])
    sites = np.array([[0.0, 0.5], [4.0, 0.5]])
    allowed = np.ones((4, 2), dtype=bool)

    assert assign_cells(xy, sites, np.zeros(2), allowed).tolist() == [0, 0, 1, 1]
    assert assign_cells(xy, sites, np.array([7.0, 0.0]), allowed).tolist() == [0, 0, 0, 1]
    assert assign_cells(xy, sites, np.array([4.0, 0.0]), allowed).tolist() == [0, 0, 0, 1]


def test_area_weighted_centroid_of_unequal_cells():
    """Cells at x = 0.5, 1.5 with areas 1 and 3 -> centroid at
    (1*0.5 + 3*1.5) / 4 = 1.25, not the midpoint 1.0."""
    cells = EligibleCells(
        geometries=np.empty(4, dtype=object),
        centroids_xy=np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]]),
        areas_m2=np.array([1.0, 3.0, 1.0, 1.0]),
        component=np.zeros(4, dtype=np.int32),
        n_rth_unreachable=0, n_no_eligible_owner=0,
    )
    counts, area, centroids = aggregate(np.array([0, 0, 1, 1]), cells, 2)
    assert counts.tolist() == [2, 2]
    assert area.tolist() == [4.0, 2.0]
    assert centroids[0].tolist() == [1.25, 0.5]
    assert centroids[1].tolist() == [3.0, 0.5]


def test_empty_zone_yields_a_non_finite_centroid_so_its_site_stays_put():
    cells = EligibleCells(
        geometries=np.empty(2, dtype=object),
        centroids_xy=np.array([[0.5, 0.5], [1.5, 0.5]]),
        areas_m2=np.array([1.0, 1.0]),
        component=np.zeros(2, dtype=np.int32),
        n_rth_unreachable=0, n_no_eligible_owner=0,
    )
    _, area, centroids = aggregate(np.array([0, 0]), cells, 2)
    assert area[1] == 0.0
    assert np.isfinite(centroids[0]).all()
    assert not np.isfinite(centroids[1]).all()


def test_every_cell_has_exactly_one_owner_and_no_area_is_lost():
    area = box(0.0, 0.0, 400.0, 200.0)
    dec, partition = _decompose(area, [Pose(100.0, 100.0, 0.0), Pose(300.0, 100.0, 0.0),
                                       Pose(200.0, 50.0, 0.0)])
    diag = dec.diagnostics
    assert diag.cells["assigned"] == diag.cells["eligible"] == 800
    assert sum(v["n_cells"] for v in diag.per_drone.values()) == diag.cells["eligible"]
    assert sum(v["area_m2"] for v in diag.per_drone.values()) == pytest.approx(
        area.area, rel=1e-12
    )
    assert partition.total_area_m2 == pytest.approx(area.area, rel=1e-9)


def test_coincident_sites_give_everything_to_the_lowest_id():
    area = box(0.0, 0.0, 400.0, 200.0)
    shared = Pose(200.0, 100.0, 0.0)
    _, partition = _decompose(area, [shared, shared])
    assert partition.zones[0].area_m2 == pytest.approx(area.area, rel=1e-12)
    assert partition.zones[1].area_m2 == 0.0
    # an empty zone is legal, is not dropped, and entries at the drone's own pose
    assert partition.zones[1].entry_pose == shared


def test_symmetric_world_gives_a_symmetric_partition():
    area = box(0.0, 0.0, 400.0, 200.0)
    _, partition = _decompose(area, [Pose(100.0, 100.0, 0.0), Pose(300.0, 100.0, 0.0)])
    assert partition.zones[0].area_m2 == pytest.approx(partition.zones[1].area_m2, rel=1e-12)
    assert partition.zones[0].area_m2 == pytest.approx(area.area / 2, rel=1e-12)


def test_three_drones_on_a_rectangle_split_it_into_equal_thirds():
    area = box(0.0, 0.0, 600.0, 240.0)
    _, partition = _decompose(area, [Pose(100.0, 120.0, 0.0), Pose(300.0, 120.0, 0.0),
                                     Pose(500.0, 120.0, 0.0)])
    for zone in partition.zones.values():
        assert zone.area_m2 == pytest.approx(area.area / 3, rel=1e-12)


def test_a_disconnected_zone_is_kept_whole_not_reduced_to_its_largest_part():
    """A wall anchored to the left edge leaves the free space connected only
    around its right end. Seeded in the bottom strip and in that right corridor,
    the bottom drone also owns the top-left strip -- it is nearer to it than the
    corridor drone is -- while the only link between the two halves belongs to
    the corridor drone. Its zone is therefore genuinely disconnected, and must be
    kept whole rather than collapsed to its largest part."""
    area = box(0.0, 0.0, 300.0, 120.0)
    wall = box(0.0, 50.0, 240.0, 70.0)
    dec, partition = _decompose(area, [Pose(120.0, 10.0, 0.0), Pose(280.0, 60.0, 0.0)],
                                obstacles=[_obstacle(wall)])

    assert isinstance(partition.zones[0].polygon, MultiPolygon)
    assert len(partition.zones[0].polygon.geoms) >= 2
    # nothing was dropped to achieve it
    assert dec.diagnostics.cells["no_eligible_owner"] == 0
    assert sum(z.area_m2 for z in partition.zones.values()) == pytest.approx(
        dec.diagnostics.cells["area_m2"], rel=1e-12
    )


# --------------------------------------------------------------------------- #
# F. hard reachability constraints                                             #
# --------------------------------------------------------------------------- #
def test_a_chamber_with_no_drone_is_dropped_explicitly_not_given_to_the_nearest():
    """A full-height wall splits the flyable space into two components. With both
    drones on the left, the right chamber has no eligible owner: it is counted
    and excluded, never handed to whichever drone happens to be closest."""
    area = box(0.0, 0.0, 300.0, 120.0)
    wall = box(140.0, 0.0, 160.0, 120.0)
    dec, partition = _decompose(area, [Pose(20.0, 60.0, 0.0), Pose(60.0, 60.0, 0.0)],
                                obstacles=[_obstacle(wall)])

    diag = dec.diagnostics
    assert diag.cells["no_eligible_owner"] == 168        # the whole right chamber
    assert diag.cells["assigned"] == diag.cells["eligible"]
    # the surviving work is exactly the left chamber, and every zone sits in it
    assert diag.cells["area_m2"] == pytest.approx(140.0 * 120.0, rel=1e-12)
    for zone in partition.zones.values():
        if not zone.polygon.is_empty:
            assert zone.polygon.bounds[2] <= 140.0 + 1e-9


def test_a_single_component_makes_the_reachability_constraint_inert():
    """The ordinary case -- obstacles strictly inside the survey -- leaves one
    component, so no cell and no drone can be excluded by connectivity."""
    area = box(0.0, 0.0, 300.0, 120.0)
    island = box(120.0, 40.0, 180.0, 80.0)
    dec, _ = _decompose(area, [Pose(60.0, 60.0, 0.0), Pose(240.0, 60.0, 0.0)],
                        obstacles=[_obstacle(island)])
    assert dec.diagnostics.cells["no_eligible_owner"] == 0
    assert dec.diagnostics.cells["rth_unreachable"] == 0


def test_cells_with_an_undefined_return_cost_are_dropped_for_everyone():
    """RTH reachability is a property of the CELL and the base, not of a drone,
    so an unreachable cell leaves the eligible set entirely. The in-bounds test
    mirrors energy_balance._estimate: a cell outside the grid is not judged."""
    from types import SimpleNamespace

    area = box(0.0, 0.0, 200.0, 100.0)
    env = EnvironmentMap(area, [], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 10.0)
    frame = SimpleNamespace(origin_x=0.0, origin_y=0.0, cell_m=100.0, nx=2, ny=1)
    e_home = np.array([[10.0], [math.inf]])          # the right half is unreachable
    cells, _ = build_eligible_cells(
        raster, env, np.array([[50.0, 50.0]]),
        energy_map=SimpleNamespace(frame=frame, e_home=e_home),
    )
    assert cells.n_rth_unreachable == 100
    assert cells.count == 100
    assert cells.centroids_xy[:, 0].max() < 100.0


def test_non_convergence_is_reported_not_repaired():
    area = box(0.0, 0.0, 400.0, 200.0)
    settings = PartitionConfig(init_sites="deploy_poses", max_iterations=1,
                               site_tolerance_m=0.001)
    dec, partition = _decompose(area, [Pose(10.0, 10.0, 0.0), Pose(390.0, 190.0, 0.0)],
                                settings=settings)
    assert dec.diagnostics.converged is False
    assert dec.diagnostics.iterations == 1
    assert dec.diagnostics.max_site_shift_m > settings.site_tolerance_m
    # the partition is still USED, and still conserves everything
    assert partition.total_area_m2 == pytest.approx(area.area, rel=1e-9)
    assert dec.name is DecompositionAlgo.LLOYD_CVT      # no fallback to another algo


def _obstacle(polygon):
    from uav_swarm_sim.planning.obstacle_generator import Obstacle

    return Obstacle(id=0, cls=0, polygon=polygon)


# --------------------------------------------------------------------------- #
# C. initial sites                                                             #
# --------------------------------------------------------------------------- #
def _cells_from(area, cell_m=10.0):
    env = EnvironmentMap(area, [], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, cell_m)
    cells, _ = build_eligible_cells(raster, env, np.zeros((1, 2)))
    return cells


def test_default_init_reproduces_the_drone_pose_seeding_convention():
    """The code default must leave the seeding where the position-based
    decomposers put it, so turning the partitioner on changes the algorithm and
    nothing else."""
    poses = deploy_ring_poses(Pose(150.0, 60.0, 0.0), 5, (0.307, 0.3875, 0.1495), 10.0)
    xy = np.array([[p.x, p.y] for p in poses])
    cells = _cells_from(box(0.0, 0.0, 300.0, 120.0))

    sites = initial_sites("deploy_poses", xy, cells, Pose(150.0, 0.0, 0.0))
    assert np.array_equal(sites, xy)


def test_the_staging_ring_really_is_degenerate_at_survey_scale():
    """R = (hypot(L, W) + min_separation) / (2 sin(pi/N)); for the M4E
    (0.307 x 0.3875 m, 10 m separation) s = 10.494 m, so N = 5 rings at 8.93 m.
    That is the whole reason the maximin switch exists."""
    base = Pose(500.0, 375.0, 0.0)
    poses = deploy_ring_poses(base, 5, (0.307, 0.3875, 0.1495), 10.0)
    radii = [math.hypot(p.x - base.x, p.y - base.y) for p in poses]
    expected = (math.hypot(0.307, 0.3875) + 10.0) / (2 * math.sin(math.pi / 5))
    assert expected == pytest.approx(8.93, abs=0.01)
    for r in radii:
        assert r == pytest.approx(expected, rel=1e-9)


def test_maximin_is_deterministic_and_breaks_ties_on_the_lowest_cell_index():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    launch = np.array([0.0, 0.0])
    order = maximin_site_order(xy, launch, 3)
    # s1 = nearest the launch (index 0); s2 = farthest from it -- indices 2 and 3
    # are equidistant, so the lowest index wins; s3 = index 1.
    assert order.tolist() == [0, 2, 1]
    assert maximin_site_order(xy, launch, 3).tolist() == order.tolist()


def test_maximin_spreads_sites_that_deploy_poses_would_leave_coincident():
    """The behavioural point of the switch: from a degenerate ring the CVT arm
    would hand everything to one drone; maximin gives both real work."""
    area = box(0.0, 0.0, 400.0, 200.0)
    shared = Pose(200.0, 100.0, 0.0)
    _, ring = _decompose(area, [shared, shared])
    _, spread = _decompose(area, [shared, shared],
                           settings=PartitionConfig("maximin", 50, 1.0))

    assert ring.zones[1].area_m2 == 0.0
    assert min(z.area_m2 for z in spread.zones.values()) > 0.0
    assert sum(z.area_m2 for z in spread.zones.values()) == pytest.approx(area.area, rel=1e-9)


def test_site_to_drone_matching_is_optimal_not_greedy_in_id_order():
    """Greedy id-order matching would let drone 0 claim the site nearest itself
    and strand drone 1 with a far one. The optimal assignment minimises the TOTAL
    squared distance, which matters once the drones are spread out."""
    cells = _cells_from(box(0.0, 0.0, 300.0, 100.0))
    drones = np.array([[150.0, 50.0], [10.0, 50.0]])
    sites = initial_sites("maximin", drones, cells, Pose(5.0, 50.0, 0.0))

    order = maximin_site_order(cells.centroids_xy, np.array([5.0, 50.0]), 2)
    picked = cells.centroids_xy[order]
    greedy = np.array([picked[0], picked[1]])          # drone 0 takes s1, drone 1 takes s2
    def cost(a):
        return float(((drones - a) ** 2).sum())

    # strict: with <= a regression to greedy matching would still pass
    assert cost(sites) < cost(greedy)
    assert {tuple(p) for p in sites} == {tuple(p) for p in picked}


def test_paired_replications_give_both_arms_identical_initial_sites():
    """D-14: the arms must start from the same sites for the comparison to be
    paired. With one shared code path this is a property of the inputs, so the
    test pins the inputs -- same cells, same poses, same launch -> same sites."""
    cells = _cells_from(box(0.0, 0.0, 300.0, 120.0))
    poses = np.array([[20.0, 20.0], [280.0, 100.0], [150.0, 60.0]])
    launch = Pose(150.0, 0.0, 0.0)
    for policy in ("deploy_poses", "maximin"):
        first = initial_sites(policy, poses, cells, launch)
        second = initial_sites(policy, poses, cells, launch)
        assert np.array_equal(first, second)
        assert first.shape == (3, 2)


def test_unknown_init_policy_raises():
    cells = _cells_from(box(0.0, 0.0, 100.0, 100.0))
    with pytest.raises(ValueError, match="init_sites"):
        initial_sites("spiral", np.zeros((2, 2)), cells, Pose(0.0, 0.0, 0.0))


def test_more_drones_than_cells_keeps_the_surplus_on_their_staging_pose():
    cells = _cells_from(box(0.0, 0.0, 20.0, 10.0))     # exactly 2 cells at 10 m
    poses = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    sites = initial_sites("maximin", poses, cells, Pose(0.0, 0.0, 0.0))
    assert len(sites) == 4
    # the two unmatched drones keep their own pose; nothing crashes, no duplicates
    assert sum(1 for s in sites if tuple(s) in {tuple(p) for p in poses}) >= 2


def test_missing_component_information_leaves_the_constraint_inert():
    """No env (or an empty flyable space) means no connectivity information. The
    reachability constraint must then exclude NOTHING -- labelling every point
    unreachable would strand the entire survey instead of failing loudly."""
    area = box(0.0, 0.0, 200.0, 100.0)
    raster = CoverageRaster(area, area, 10.0)
    poses = [Pose(50.0, 50.0, 0.0), Pose(150.0, 50.0, 0.0)]
    dec = LloydCvtDecomposer(raster=raster, deploy_poses=poses,
                             launch_pose=Pose(100.0, 0.0, 0.0), settings=SETTINGS)
    partition = dec.decompose(None, None, [DroneStateView(i, 1.0, p)
                                           for i, p in enumerate(poses)])

    assert dec.diagnostics.cells["no_eligible_owner"] == 0
    assert dec.diagnostics.cells["eligible"] == 200
    assert sum(z.area_m2 for z in partition.zones.values()) == pytest.approx(
        area.area, rel=1e-12
    )


# --------------------------------------------------------------------------- #
# F (cont). the two points answer different questions                          #
# --------------------------------------------------------------------------- #
NOTCH = box(43.0, 40.0, 47.0, 46.0)   # bites into the bottom edge of cell (40,40)-(50,50)


def _notched_cells(extra_obstacles=(), energy_map=None, drone_xy=((5.0, 5.0),)):
    area = box(0.0, 0.0, 100.0, 100.0)
    obstacles = [_obstacle(NOTCH)] + [_obstacle(g) for g in extra_obstacles]
    env = EnvironmentMap(area, obstacles, 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 10.0)
    return build_eligible_cells(raster, env, np.array(drone_xy, dtype=float),
                                energy_map=energy_map)


def test_a_notched_cell_really_does_put_its_area_centroid_in_the_obstacle():
    """The premise of the three tests below, pinned so they cannot rot silently.

    The clipped cell (40,40)-(50,50) minus the notch is concave. Its area
    centroid (45.0, 45.63) lies INSIDE the notch -- i.e. outside both the cell
    and the flyable space -- while point_on_surface (41.5, 43.0) is in the cell.
    """
    area = box(0.0, 0.0, 100.0, 100.0)
    env = EnvironmentMap(area, [_obstacle(NOTCH)], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 10.0)
    raw = raster.uncovered_plannable_cells()

    outside = np.flatnonzero(~shapely.covers(raw.geometries, shapely.centroid(raw.geometries)))
    assert outside.tolist() == [44]
    centroid = shapely.centroid(raw.geometries[44])
    assert (centroid.x, centroid.y) == pytest.approx((45.0, 45.631579), abs=1e-6)
    assert NOTCH.covers(centroid)
    surface = raw.surface_points[44]
    assert (surface.x, surface.y) == pytest.approx((41.5, 43.0), abs=1e-6)
    assert raw.geometries[44].covers(surface) and not NOTCH.covers(surface)


def test_the_rth_filter_judges_a_cell_by_a_point_that_is_actually_in_it():
    """(a) An energy map that marks the notch unreachable must not take the cell
    with it: the cell is flyable, only its area centroid is not."""
    frame = SimpleNamespace(origin_x=0.0, origin_y=0.0, cell_m=1.0, nx=100, ny=100)
    e_home = np.full((100, 100), 10.0)
    e_home[43:47, 40:46] = math.inf                 # exactly the notch footprint
    energy_map = SimpleNamespace(frame=frame, e_home=e_home)

    cells, _ = _notched_cells(energy_map=energy_map)
    assert cells.n_rth_unreachable == 0
    assert cells.count == 100
    # the surviving set still contains the notched cell, at its full clipped area
    assert float(cells.areas_m2.sum()) == pytest.approx(100 * 100 - NOTCH.area, rel=1e-12)
    assert 76.0 in cells.areas_m2.tolist()


def test_the_component_label_of_a_notched_cell_comes_from_inside_it():
    """(b) With the flyable space genuinely split, the notched cell must still be
    labelled with the component it sits in -- not -1, which would orphan it."""
    wall = box(70.0, 0.0, 74.0, 100.0)              # splits left from right
    cells, drone_comp = _notched_cells(extra_obstacles=[wall],
                                       drone_xy=((5.0, 5.0), (90.0, 90.0)))
    assert set(drone_comp.tolist()) == {0, 1}       # one drone per chamber
    assert cells.n_no_eligible_owner == 0
    assert (cells.component >= 0).all()
    # the notched cell is present and sits in the left-hand component
    notched = np.flatnonzero(np.isclose(cells.areas_m2, 76.0))
    assert len(notched) == 1
    assert cells.component[notched[0]] == drone_comp[0]


def test_mass_arithmetic_still_uses_the_area_centroid_not_the_surface_point():
    """(c) The membership fix must not leak into the Lloyd arithmetic: the cell's
    MASS position stays the area centroid, which is the CVT fixed point even when
    it lies outside its own cell."""
    cells, _ = _notched_cells()
    notched = np.flatnonzero(np.isclose(cells.areas_m2, 76.0))
    assert len(notched) == 1
    x, y = cells.centroids_xy[notched[0]]
    assert (x, y) == pytest.approx((45.0, 45.631579), abs=1e-6)     # area centroid
    assert (x, y) != pytest.approx((41.5, 43.0), abs=1e-6)          # NOT the surface point


def test_cells_straddling_two_components_are_counted_not_hidden():
    """A separator thinner than one cell leaves cells in both components. The
    cell stays the atom (splitting it would change the raster's own cell set),
    so the condition is COUNTED rather than silently absorbed."""
    area = box(0.0, 0.0, 100.0, 100.0)
    hairline = box(50.0, 0.0, 51.0, 100.0)          # 1 m wall inside a 10 m cell
    env = EnvironmentMap(area, [_obstacle(hairline)], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 10.0)
    cells, drone_comp = build_eligible_cells(
        raster, env, np.array([[5.0, 50.0], [95.0, 50.0]])
    )
    assert set(drone_comp.tolist()) == {0, 1}
    assert cells.n_spanning_components == 10       # one column of straddling cells
    assert cells.n_no_eligible_owner == 0          # counted, not dropped


def test_a_thick_separator_produces_no_straddling_cells():
    area = box(0.0, 0.0, 100.0, 100.0)
    wall = box(40.0, 0.0, 60.0, 100.0)              # aligned with the 10 m grid
    env = EnvironmentMap(area, [_obstacle(wall)], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 10.0)
    cells, _ = build_eligible_cells(raster, env, np.array([[5.0, 50.0], [95.0, 50.0]]))
    assert cells.n_spanning_components == 0
