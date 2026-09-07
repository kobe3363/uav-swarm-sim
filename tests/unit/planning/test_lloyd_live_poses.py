"""EXP-08: the grid partitioner reads the drone state it is HANDED.

Before this change ``_LloydDecomposer`` seeded every site from a constructor
snapshot of the staging ring, and ``LloydEnergyDecomposer`` weighted from a
snapshot of the t=0 charge -- while using ``drone.pose`` for the zone entry pose
in the very same method. A contract argument honoured in one place and ignored
in another is the same defect class as one ignored outright; it just fails
quietly instead of raising.

t=0 is unaffected. That claim is checked here at its PREMISE -- the views the
engine builds are exactly the snapshot that used to be frozen -- rather than by
a golden captured from this same code.
"""
from __future__ import annotations

import json

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.infrastructure.config import PartitionConfig, load_config
from uav_swarm_sim.infrastructure.core_types import DroneStateView, Pose
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.planning.coverage_raster import CoverageRaster
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.lloyd_partition import LloydCvtDecomposer

SETTINGS = PartitionConfig(init_sites="deploy_poses", max_iterations=50,
                           site_tolerance_m=1.0)


def _obstacle(polygon):
    from uav_swarm_sim.planning.obstacle_generator import Obstacle

    return Obstacle(id=0, cls=0, polygon=polygon)


def _views(poses, fracs=None, airborne=False):
    fracs = [1.0] * len(poses) if fracs is None else fracs
    return [DroneStateView(i, f, p, 0, airborne)
            for i, (p, f) in enumerate(zip(poses, fracs))]


# --------------------------------------------------------------------------- #
# t=0: the premise of the byte-identity claim                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture
def engine_overrides(tmp_path):
    area = tmp_path / "rect.geojson"
    area.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(box(0, 0, 600, 240))}))
    return {
        "fleet.n_drones": 3, "fleet.battery_capacity_wh": 1000.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "launch.candidate_sites": [[300.0, 0.0]],
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "platforms.MULTIROTOR.r_min_m": 0.0, "platforms.MULTIROTOR.omega_max": 1.0,
        "sensor.sensor_power_w": 15.0,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 8.0,
        "sensor.photogrammetry.sensor_height_mm": 6.0,
        "sensor.photogrammetry.focal_length_mm": 10.0,
        "sensor.photogrammetry.image_width_px": 4000,
        "sensor.photogrammetry.image_height_px": 3000,
        "sensor.photogrammetry.side_overlap": 0.5,
        "sensor.photogrammetry.forward_overlap": 0.5,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 10.0,
        "sim.dt_s": 0.5, "sim.max_timesteps": 2000,
        "battery.initial_soc.mode": "uniform",
        "battery.initial_soc.low": 0.35, "battery.initial_soc.high": 0.95,
    }


def test_the_t0_views_carry_exactly_what_the_decomposer_used_to_freeze(engine_overrides):
    """Reading the views instead of a snapshot is byte-identical at t=0 only if
    the two agree there. That is a property of the ENGINE, so it is asserted
    against the engine rather than assumed: every t=0 view must carry the drone's
    staging pose, its drawn initial SoC, and airborne=False.

    Heterogeneous initial SoC on purpose -- on a uniform fleet the battery half
    of this check would pass for the wrong reason.
    """
    cfg = load_config("config/default.yaml", overrides=engine_overrides)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.LLOYD_CVT)
    eng._build()

    socs = eng.initial_soc_by_drone
    assert len({round(s, 9) for s in socs}) > 1, "fixture must not be homogeneous"
    for agent in eng.fleet.agents.values():
        view = agent.view()
        assert view.pose == eng.deploy_poses[agent.id]
        assert view.battery_frac == pytest.approx(socs[agent.id], abs=1e-12)
        assert view.airborne is False


def test_the_airborne_flag_defaults_to_grounded():
    """Every pre-EXP-08 construction omits it, and at t=0 grounded is correct."""
    assert DroneStateView(0, 1.0, Pose(0.0, 0.0, 0.0)).airborne is False


# --------------------------------------------------------------------------- #
# the pose is honoured                                                         #
# --------------------------------------------------------------------------- #
def _flat_case(cell_m=10.0, settings=SETTINGS):
    area = box(0.0, 0.0, 300.0, 120.0)
    env = EnvironmentMap(area, [], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, cell_m)
    dec = LloydCvtDecomposer(raster=raster, launch_pose=Pose(150.0, 0.0, 0.0),
                             settings=settings)
    return env, dec


def test_a_moved_drone_moves_its_own_site():
    """Same decomposer, same raster, one drone relocated: its site must follow.
    With a frozen snapshot both calls would return identical sites.

    Capped at ONE iteration on purpose. Lloyd is a fixed-point iteration, and on
    a convex domain both of these starts converge to the SAME centroidal
    partition -- run to convergence the seeding washes out entirely and this test
    would pass vacuously in both directions. One iteration is where the seeding
    is still visible, which is also how EXP-07a proves its default seeding.
    (The corollary is worth stating: for the CVT arm an in-flight re-partition is
    differentiated mainly by the reduced CELL set and the reduced DRONE set, not
    by the poses; for the ENERGY arm the poses additionally move the weights --
    see the two tests below.)
    """
    from dataclasses import replace as _replace

    one_step = _replace(SETTINGS, max_iterations=1, site_tolerance_m=1e9)
    env, dec = _flat_case(settings=one_step)
    staged = [Pose(20.0, 60.0, 0.0), Pose(280.0, 60.0, 0.0)]
    moved = [Pose(20.0, 60.0, 0.0), Pose(150.0, 20.0, 0.0)]

    dec.decompose(None, env, _views(staged))
    sites_before = {k: tuple(v["site_xy"]) for k, v in dec.diagnostics.per_drone.items()}
    areas_before = {k: v["area_m2"] for k, v in dec.diagnostics.per_drone.items()}

    dec.decompose(None, env, _views(moved))
    sites_after = {k: tuple(v["site_xy"]) for k, v in dec.diagnostics.per_drone.items()}
    areas_after = {k: v["area_m2"] for k, v in dec.diagnostics.per_drone.items()}

    assert sites_after != sites_before
    assert areas_after != areas_before
    # conservation is untouched by the relocation: the whole survey is still owned
    assert sum(areas_after.values()) == pytest.approx(300.0 * 120.0, rel=1e-12)


def test_lloyd_cvt_converges_to_the_same_partition_from_either_start():
    """The counterpart of the test above, stated rather than left implicit: run
    to convergence, the CVT arm's fixed point on this convex domain is the same
    from both starts. Nobody should read the pose fix as changing what CVT
    converges to -- it changes which drone state the iteration STARTS from, and
    (decisively) which free-space component each drone is deemed to be in."""
    env, dec = _flat_case()
    first = dec.decompose(None, env, _views([Pose(20.0, 60.0, 0.0),
                                             Pose(280.0, 60.0, 0.0)]))
    first_wkt = {i: z.polygon.wkt for i, z in first.zones.items()}
    second = dec.decompose(None, env, _views([Pose(20.0, 60.0, 0.0),
                                              Pose(150.0, 20.0, 0.0)]))
    assert {i: z.polygon.wkt for i, z in second.zones.items()} == first_wkt
    assert dec.diagnostics.converged is True


def test_two_calls_with_the_same_views_are_identical():
    """The partitioner carries no RNG and no cross-call state, so re-running it
    on unchanged inputs must reproduce itself exactly. This is what makes a
    re-partition a function of the drone state alone."""
    env, dec = _flat_case()
    poses = [Pose(20.0, 60.0, 0.0), Pose(280.0, 60.0, 0.0)]
    first = dec.decompose(None, env, _views(poses))
    first_wkt = {i: z.polygon.wkt for i, z in first.zones.items()}
    second = dec.decompose(None, env, _views(poses))
    assert {i: z.polygon.wkt for i, z in second.zones.items()} == first_wkt


def test_the_target_area_refusal_still_stands():
    """The pose fix must not be mistaken for making this partitioner able to
    honour a sub-area. It still cannot, and still says so."""
    env, dec = _flat_case()
    with pytest.raises(NotImplementedError, match="target_area"):
        dec.decompose(None, env, _views([Pose(20.0, 60.0, 0.0)]),
                      target_area=box(0.0, 0.0, 100.0, 100.0))


def test_the_capability_is_declared_rather_than_inferred_from_the_class():
    """A caller asks which input this decomposer takes; it does not test its
    class. An isinstance branch is the confound EXP-08 removes."""
    from uav_swarm_sim.planning.classic_voronoi import ClassicVoronoiDecomposer
    from uav_swarm_sim.planning.weighted_decomposition import TgcBasicDecomposer

    _, dec = _flat_case()
    assert dec.partitions_raster_work is True
    assert TgcBasicDecomposer().partitions_raster_work is False
    assert ClassicVoronoiDecomposer().partitions_raster_work is False


# --------------------------------------------------------------------------- #
# LLOYD_ENERGY: the weights follow the LIVE charge and the LIVE pose (D-2a)     #
# --------------------------------------------------------------------------- #
ALT = 100.0
CAPACITY_J = 360000.0


@pytest.fixture
def energy_case():
    """Explicit fixture, no engine and no RNG: an energy decomposer over a flat
    600 x 240 survey, mirroring the pinned EXP-07b unit fixture."""
    from dataclasses import replace as _replace

    from uav_swarm_sim.execution.rth_calculator import RthCalculator
    from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
    from uav_swarm_sim.physical_model.drone_specs import build_spec
    from uav_swarm_sim.physical_model.energy_model import EnergyModel
    from uav_swarm_sim.physical_model.motion_model import HolonomicModel
    from uav_swarm_sim.planning.energy_balance import build_energy_balance_context
    from uav_swarm_sim.planning.lloyd_partition import LloydEnergyDecomposer

    cfg = load_config("config/default.yaml", {
        "env.coverage_altitude_m": ALT,
        "sensor.sensor_power_w": 15.0,
        "sensor.photogrammetry.enabled": True,
        "sensor.photogrammetry.sensor_width_mm": 8.0,
        "sensor.photogrammetry.sensor_height_mm": 6.0,
        "sensor.photogrammetry.focal_length_mm": 10.0,
        "sensor.photogrammetry.image_width_px": 4000,
        "sensor.photogrammetry.image_height_px": 3000,
        "sensor.photogrammetry.side_overlap": 0.5,
        "sensor.photogrammetry.forward_overlap": 0.5,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
        "coverage.raster_enabled": True,
        "rth.reserve_frac": 0.05,
    })
    spec = _replace(
        build_spec(cfg), platform=PlatformType.MULTIROTOR, mass_kg=4.0,
        battery_capacity_j=CAPACITY_J, r_min_m=0.0, omega_max=1.0,
        v_cruise=12.0, v_coverage=10.0, v_climb=4.0, v_descent=3.0,
        power_w={**cfg.platform.power_w, ManeuverType.CRUISE: 220.0,
                 ManeuverType.COVERAGE: 250.0, ManeuverType.TURN: 240.0,
                 ManeuverType.TAKEOFF: 400.0, ManeuverType.LAND: 300.0},
    )
    em, motion = EnergyModel(spec), HolonomicModel(spec)
    base = Pose(300.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, ALT)
    ctx = build_energy_balance_context(
        cfg, em, spec, motion, None,
        lambda pose, alt: rth.return_energy(pose, altitude_m=alt),
    )
    area = box(0.0, 0.0, 600.0, 240.0)
    env = EnvironmentMap(area, [], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 20.0)
    dec = LloydEnergyDecomposer(
        raster=raster, launch_pose=base, settings=SETTINGS,
        energy_context=ctx, altitude_m=ALT, capacity_j=CAPACITY_J,
    )
    return env, dec


_ENERGY_POSES = [Pose(150.0, 120.0, 0.0), Pose(450.0, 120.0, 0.0)]


def test_the_energy_weights_follow_the_live_battery_level(energy_case):
    """The whole point of D-2a: a re-partition must weight from what each drone
    HAS LEFT. A frozen t=0 snapshot would return identical joules for both calls,
    so the two arms would keep computing the same partition forever."""
    env, dec = energy_case

    dec.decompose(None, env, _views(_ENERGY_POSES, fracs=[1.0, 1.0]))
    full = dec.diagnostics.per_drone

    dec.decompose(None, env, _views(_ENERGY_POSES, fracs=[1.0, 0.40]))
    drained = dec.diagnostics.per_drone

    # the charge reached the estimator: e_level is capacity * frac, computed here
    assert full[1]["budget_j"] != drained[1]["budget_j"]
    assert full[1]["slack_j"] != drained[1]["slack_j"]
    # ... and it moved the partition: the emptier drone takes the smaller zone
    assert drained[1]["area_m2"] < drained[0]["area_m2"]
    assert drained[1]["area_m2"] < full[1]["area_m2"]
    # conservation holds in both calls
    for record in (full, drained):
        assert sum(v["area_m2"] for v in record.values()) == pytest.approx(
            600.0 * 240.0, rel=1e-9
        )


def test_the_energy_budget_follows_the_live_pose_through_ferry_and_rth(energy_case):
    """``budget = level - takeoff - (ferry + rth + reserve)`` and both ferry and
    rth are taken from the drone's pose. Moving a drone with an unchanged battery
    must therefore move its budget -- which a pose snapshot could not do."""
    env, dec = energy_case

    dec.decompose(None, env, _views(_ENERGY_POSES))
    near = dec.diagnostics.per_drone[0]["budget_j"]

    far = [Pose(20.0, 230.0, 0.0), _ENERGY_POSES[1]]
    dec.decompose(None, env, _views(far))
    assert dec.diagnostics.per_drone[0]["budget_j"] != near


def test_an_airborne_view_is_not_charged_takeoff_twice(energy_case):
    """``_budget`` deducts takeoff only for a grounded drone. A re-partition sees
    drones that are already flying and must not bill them for a takeoff they
    already paid, so the airborne flag has to reach DroneEnergyState."""
    env, dec = energy_case

    dec.decompose(None, env, _views(_ENERGY_POSES, airborne=False))
    grounded = dec.diagnostics.per_drone[0]["budget_j"]

    dec.decompose(None, env, _views(_ENERGY_POSES, airborne=True))
    flying = dec.diagnostics.per_drone[0]["budget_j"]

    # an airborne drone keeps the takeoff energy in its budget, so it is larger
    assert flying > grounded


# --------------------------------------------------------------------------- #
# the pose also labels the free-space component (author requirement 2)         #
# --------------------------------------------------------------------------- #
# area 300 x 120, a full-height wall over x in [140, 160], cells of 10 m:
# left chamber  x in [0, 140]   -> 14 columns x 12 rows = 168 cells, 16 800 m^2
# right chamber x in [160, 300] -> 14 columns x 12 rows = 168 cells, 16 800 m^2
_CHAMBER_CELLS = 14 * 12
_CHAMBER_AREA = 140.0 * 120.0


def _walled_case():
    area = box(0.0, 0.0, 300.0, 120.0)
    env = EnvironmentMap(area, [_obstacle(box(140.0, 0.0, 160.0, 120.0))], 0.0)
    raster = CoverageRaster(env.target_space, env.plannable_space, 10.0)
    dec = LloydCvtDecomposer(raster=raster, launch_pose=Pose(70.0, 0.0, 0.0),
                             settings=SETTINGS)
    return env, dec


def test_a_drone_that_flew_into_another_component_gets_that_components_cells():
    """``build_eligible_cells`` labels each DRONE's free-space component from the
    same poses that seed the sites. With live poses a drone that has flown across
    a separator is labelled with the component it is now in, so the far chamber
    stops being orphaned. With a staging snapshot it never could be."""
    env, dec = _walled_case()

    both_left = [Pose(20.0, 60.0, 0.0), Pose(60.0, 60.0, 0.0)]
    dec.decompose(None, env, _views(both_left))
    assert dec.diagnostics.cells["no_eligible_owner"] == _CHAMBER_CELLS
    assert dec.diagnostics.cells["area_m2"] == pytest.approx(_CHAMBER_AREA, rel=1e-12)

    one_crossed = [Pose(20.0, 60.0, 0.0), Pose(280.0, 60.0, 0.0)]
    partition = dec.decompose(None, env, _views(one_crossed))

    diag = dec.diagnostics
    assert diag.cells["no_eligible_owner"] == 0
    assert diag.cells["eligible"] == 2 * _CHAMBER_CELLS
    assert diag.cells["assigned"] == diag.cells["eligible"]
    assert diag.cells["area_m2"] == pytest.approx(2 * _CHAMBER_AREA, rel=1e-12)
    # each drone owns exactly its own chamber -- nothing crossed the wall
    assert partition.zones[0].polygon.bounds[2] <= 140.0 + 1e-9
    assert partition.zones[1].polygon.bounds[0] >= 160.0 - 1e-9
    for zone in partition.zones.values():
        assert zone.area_m2 == pytest.approx(_CHAMBER_AREA, rel=1e-12)


def test_the_orphan_count_stays_exact_when_a_component_loses_its_only_drone():
    """The reverse direction: a drone leaving a chamber puts that chamber's cells
    back into no_eligible_owner, exactly and by count -- never silently to the
    nearest drone on the far side of the wall."""
    env, dec = _walled_case()

    dec.decompose(None, env, _views([Pose(20.0, 60.0, 0.0), Pose(280.0, 60.0, 0.0)]))
    assert dec.diagnostics.cells["no_eligible_owner"] == 0

    # drone 1 has crossed back; the right chamber now has nobody who can fly it
    partition = dec.decompose(None, env, _views([Pose(20.0, 60.0, 0.0),
                                                 Pose(60.0, 60.0, 0.0)]))
    assert dec.diagnostics.cells["no_eligible_owner"] == _CHAMBER_CELLS
    assert dec.diagnostics.cells["assigned"] == dec.diagnostics.cells["eligible"]
    for zone in partition.zones.values():
        if not zone.polygon.is_empty:
            assert zone.polygon.bounds[2] <= 140.0 + 1e-9
