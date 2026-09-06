"""EXP-07b: the energy weight source, on the frozen EXP-07a core.

The identity test is the point of the 07a/07b split: it is only meaningful
because the partitioning core it compares against was written, reviewed and
merged before this arm existed. Expected values are hand-derived or taken from
an independent evaluation of the energy model.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import box

from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.infrastructure.config import PartitionConfig, load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import HolonomicModel
from uav_swarm_sim.planning.coverage_raster import CoverageRaster
from uav_swarm_sim.planning.energy_balance import (
    DroneEnergyState,
    build_energy_balance_context,
    coverage_energy_density_j_per_m2,
)
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.lloyd_partition import (
    EnergyWeightPolicy,
    assign_cells,
    LloydPartitioner,
    UniformWeightPolicy,
    build_eligible_cells,
)

ALT = 100.0
CAPACITY_J = 360000.0


@pytest.fixture
def case():
    """A pinned, fully explicit fixture: no engine, no RNG, no config file drift
    beyond the photogrammetry block the estimator requires."""
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
    spec = replace(build_spec(cfg), platform=PlatformType.MULTIROTOR,
                   mass_kg=4.0, battery_capacity_j=CAPACITY_J, r_min_m=0.0, omega_max=1.0,
                   v_cruise=12.0, v_coverage=10.0, v_climb=4.0, v_descent=3.0,
                   power_w={**cfg.platform.power_w, ManeuverType.CRUISE: 220.0,
                            ManeuverType.COVERAGE: 250.0, ManeuverType.TURN: 240.0,
                            ManeuverType.TAKEOFF: 400.0, ManeuverType.LAND: 300.0})
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
    poses = [Pose(150.0, 120.0, 0.0), Pose(450.0, 120.0, 0.0)]
    xy = np.array([[p.x, p.y] for p in poses])
    cells, drone_comp = build_eligible_cells(raster, env, xy)
    return dict(cfg=cfg, spec=spec, em=em, ctx=ctx, base=base, area=area,
                poses=poses, xy=xy, cells=cells, drone_comp=drone_comp)


def _states(case, levels):
    return [DroneEnergyState(i, p, level, False)
            for i, (p, level) in enumerate(zip(case["poses"], levels))]


def _policy(case, levels, settings):
    return EnergyWeightPolicy(case["ctx"], _states(case, levels), ALT, settings,
                              CAPACITY_J, case["poses"])


def _run(case, policy, settings):
    return LloydPartitioner(settings, policy).run(
        case["cells"], case["xy"], case["drone_comp"], Pose(300.0, 0.0, 0.0)
    )


SETTINGS = PartitionConfig(init_sites="deploy_poses", max_iterations=30,
                           site_tolerance_m=1.0)


# --------------------------------------------------------------------------- #
# D. identity against the frozen core                                          #
# --------------------------------------------------------------------------- #
def test_uniform_weights_make_energy_reproduce_cvt_bit_for_bit(case):
    """Weights pinned uniform (step 0) leave the two arms running the SAME code
    on the same numbers, so the partitions must agree exactly -- not to a
    tolerance."""
    pinned = replace(SETTINGS, weight_step=0.0)
    cvt = _run(case, UniformWeightPolicy(), SETTINGS)
    energy = _run(case, _policy(case, [CAPACITY_J, CAPACITY_J], pinned), pinned)

    assert np.array_equal(cvt[0], energy[0])          # identical cell ownership
    assert np.array_equal(cvt[1], energy[1])          # identical final sites
    assert np.array_equal(cvt[2], energy[2])          # identical weights (all zero)
    assert cvt[3] == energy[3] and cvt[4] == energy[4]


def test_the_identity_is_not_trivial_live_weights_do_move_the_partition(case):
    """Guard on the test above: with the shipped weight step and unequal
    batteries the two arms must NOT agree, or the identity proves nothing."""
    live = _run(case, _policy(case, [CAPACITY_J, 0.45 * CAPACITY_J], SETTINGS), SETTINGS)
    cvt = _run(case, UniformWeightPolicy(), SETTINGS)
    assert not np.array_equal(cvt[0], live[0])


def test_equal_soc_does_not_imply_equal_energy_weights(case):
    """Equal batteries are NOT the CVT case. Predicted demand varies with zone
    geometry -- ferry and RTH are taken from the anchor -- so two drones with the
    same level_j still see different slack and the weights separate."""
    policy = _policy(case, [CAPACITY_J, CAPACITY_J], SETTINGS)
    _, _, weights, _, _, _, _ = _run(case, policy, SETTINGS)
    assert len({round(float(w), 9) for w in weights}) > 1
    slacks = {round(float(s), 6) for s in policy._slack}
    assert len(slacks) > 1


# --------------------------------------------------------------------------- #
# E. the slack weight law                                                      #
# --------------------------------------------------------------------------- #
def test_the_energy_density_matches_an_independent_evaluation(case):
    """rho = (P_COVERAGE + P_sensor) / (v_coverage * swath), computed here from
    the raw spec numbers rather than from the function under test."""
    spec, ctx = case["spec"], case["ctx"]
    swath = spec.coverage_line_spacing_m(ALT)
    expected = (spec.power_w[ManeuverType.COVERAGE] + ctx.sensor_power_w) / (
        spec.v_coverage * swath
    )
    assert coverage_energy_density_j_per_m2(ctx, ALT) == pytest.approx(expected, rel=1e-12)
    assert expected > 0.0


def test_the_weight_step_has_units_of_area(case):
    """slack / rho is [J] / [J per m^2] = [m^2]. Sweeping one drone's level by a
    known number of joules must move its weight by that many square metres,
    scaled by the damping constant."""
    settings = replace(SETTINGS, weight_step=1.0, weight_clamp_factor=1e9)
    policy = _policy(case, [CAPACITY_J, CAPACITY_J], settings)
    area = np.array([1000.0, 1000.0])
    centroids = np.array([[150.0, 120.0], [450.0, 120.0]])

    first = policy.update(np.zeros(2), None, area, centroids, None)
    slack = policy._slack.copy()
    expected_step = (slack - slack.mean()) / policy.rho_j_per_m2
    assert first == pytest.approx(expected_step - expected_step.mean(), abs=1e-9)


def test_a_smaller_battery_takes_a_smaller_zone(case):
    """Direction only -- no magnitude is asserted, so the check cannot be tuned
    to flatter the energy arm."""
    labels, *_ = _run(
        case, _policy(case, [CAPACITY_J, 0.45 * CAPACITY_J], SETTINGS), SETTINGS
    )
    areas = np.bincount(labels, weights=case["cells"].areas_m2, minlength=2)
    assert areas[1] < areas[0]


def test_budget_is_recomputed_every_iteration_because_it_follows_the_zone(case):
    """budget_j subtracts ferry and RTH, both taken from the candidate zone's
    anchor. Moving the zone must therefore move the budget -- which is why the
    law uses slack and why the fixed point is not contractive."""
    policy = _policy(case, [CAPACITY_J, CAPACITY_J], SETTINGS)
    centroids = np.array([[150.0, 120.0], [450.0, 120.0]])
    near = policy._estimate_all(np.array([1000.0, 1000.0]), centroids).copy()
    far = policy._estimate_all(np.array([1000.0, 1000.0]),
                               centroids + np.array([0.0, 100.0]))
    assert not np.allclose(near, far)


def test_a_drone_that_cannot_leave_the_ground_is_reported(case):
    """The real failure case: negative slack with an EMPTY zone. Checked after
    the loop, never as an entry condition -- an in-flight negative slack is a
    transient of the current candidate zone."""
    healthy = _policy(case, [CAPACITY_J, CAPACITY_J], SETTINGS)
    assert healthy.cannot_fly() == []

    grounded = _policy(case, [CAPACITY_J, 0.02 * CAPACITY_J], SETTINGS)
    assert grounded.cannot_fly() == [1]
    report = grounded.per_drone([0, 1])
    assert report[1]["cannot_fly"] is True and report[0]["cannot_fly"] is False


def test_reaching_the_clamp_is_reported_not_hidden(case):
    settings = replace(SETTINGS, weight_clamp_factor=1e-9, weight_step=1.0)
    policy = _policy(case, [CAPACITY_J, 0.45 * CAPACITY_J], settings)
    policy.update(np.zeros(2), None, np.array([1000.0, 1000.0]),
                  np.array([[150.0, 120.0], [450.0, 120.0]]), None)
    assert policy._clamped.any()
    assert any(d["clamped"] for d in policy.per_drone([0, 1]).values())


def test_the_ratio_is_still_reported_even_though_it_does_not_drive_the_law(case):
    policy = _policy(case, [CAPACITY_J, CAPACITY_J], SETTINGS)
    _run(case, policy, SETTINGS)
    report = policy.per_drone([0, 1])
    for entry in report.values():
        assert entry["slack_j"] == pytest.approx(entry["budget_j"] - entry["demand_j"])
        assert entry["status"] is not None
        if entry["ratio"] is not None:
            assert entry["ratio"] == pytest.approx(entry["demand_j"] / entry["budget_j"])


def test_slack_stays_defined_where_the_ratio_is_none(case):
    """The reason the law was moved off the ratio: _estimate returns budget_j and
    demand_j whatever the status, so slack survives a non-positive budget while
    demand_budget_ratio becomes None."""
    policy = _policy(case, [CAPACITY_J, 0.02 * CAPACITY_J], SETTINGS)
    policy._estimate_all(np.array([50000.0, 50000.0]),
                         np.array([[150.0, 120.0], [450.0, 120.0]]))
    starved = policy._estimates[1]
    assert starved.demand_budget_ratio is None
    assert math.isfinite(starved.budget_j - starved.demand_j)
    assert math.isfinite(float(policy._slack[1]))


# --------------------------------------------------------------------------- #
# grounded drones and the loop arithmetic                                      #
# --------------------------------------------------------------------------- #
def _three_drone_case(case, levels):
    """Three drones on the same rectangle: one grounded, two flying on unequal
    budgets so the weight law is genuinely doing work around it."""
    poses = [Pose(100.0, 120.0, 0.0), Pose(300.0, 120.0, 0.0), Pose(500.0, 120.0, 0.0)]
    xy = np.array([[p.x, p.y] for p in poses])
    states = [DroneEnergyState(i, p, lvl, False)
              for i, (p, lvl) in enumerate(zip(poses, levels))]
    settings = replace(SETTINGS, max_iterations=6, site_tolerance_m=0.001)
    policy = EnergyWeightPolicy(case["ctx"], states, ALT, settings, CAPACITY_J, poses)
    return policy, settings, xy


def test_a_grounded_drone_never_poisons_the_loop_arithmetic(case):
    """A grounded drone must stay out of the mean and out of the balance target,
    and every weight must stay finite -- grounding is an ELIGIBILITY statement,
    not a weight. A -inf sentinel would assign correctly but would poison the
    mean, could be silently clipped back to a finite value, and would create rows
    with no finite cost at all.

    The CVT arm cannot catch any of this: it has no weights.
    """
    policy, settings, xy = _three_drone_case(
        case, [CAPACITY_J, 0.55 * CAPACITY_J, 0.02 * CAPACITY_J]
    )
    assert policy.cannot_fly() == [2]
    assert not policy.all_grounded
    assert policy.excluded(3).tolist() == [False, False, True]

    labels, sites, weights, converged, iterations, _, kept = LloydPartitioner(
        settings, policy
    ).run(case["cells"], xy, np.zeros(3, dtype=np.int32), Pose(300.0, 0.0, 0.0))

    assert iterations >= 2, "the interaction only shows up across iterations"
    assert np.isfinite(weights).all(), "no sentinel may leak into the weights"
    assert not np.isnan(weights).any()
    assert np.isfinite(policy._slack[:2]).all()               # balance target finite
    assert np.abs(weights[:2]).max() <= policy._clamp + 1e-9  # clamp still binds
    # the grounded drone owns nothing, and every cell still has exactly one owner
    counts = np.bincount(labels, minlength=3)
    assert counts[2] == 0
    assert counts.sum() == kept.count
    areas = np.bincount(labels, weights=kept.areas_m2, minlength=3)
    assert float(areas.sum()) == pytest.approx(kept.total_area_m2, rel=1e-12)
    assert areas[2] == 0.0
    # one component, two live drones -> nothing became uncoverable
    assert kept.count == case["cells"].count
    assert kept.n_no_eligible_owner == case["cells"].n_no_eligible_owner


def test_the_grounded_drone_owns_nothing_in_every_iteration_not_just_the_first(case):
    """Re-running the assignment with each iteration's weights must keep the
    grounded drone empty throughout. Because it is out via the eligibility mask
    rather than via a weight, no amount of weight drift can bring it back."""
    policy, settings, xy = _three_drone_case(
        case, [CAPACITY_J, 0.55 * CAPACITY_J, 0.02 * CAPACITY_J]
    )
    allowed = np.ones((case["cells"].count, 3), dtype=bool) & (~policy.excluded(3))[None, :]
    weights = policy.initial(3)
    area = np.full(3, case["cells"].total_area_m2 / 3.0)
    centroids = xy.astype(float)

    for sweep in range(4):
        labels = assign_cells(case["cells"].centroids_xy, xy, weights, allowed)
        assert np.bincount(labels, minlength=3)[2] == 0, f"sweep {sweep}"
        weights = policy.update(weights, None, area, centroids, xy)
        assert np.isfinite(weights).all(), f"sweep {sweep}"


def test_when_nobody_can_fly_nothing_is_grounded_and_the_fact_is_reported(case):
    """Excluding everyone would leave every cell with no eligible owner, and the
    core would count the entire survey as uncoverable. Report the fact instead
    and let the partition stand, so the output says who cannot fly rather than
    silently emptying the mission."""
    policy, settings, xy = _three_drone_case(
        case, [0.02 * CAPACITY_J, 0.02 * CAPACITY_J, 0.02 * CAPACITY_J]
    )
    assert policy.all_grounded is True
    assert policy.cannot_fly() == [0, 1, 2]
    assert not policy.excluded(3).any()          # nobody is taken out of play

    labels, _, weights, _, _, _, kept = LloydPartitioner(settings, policy).run(
        case["cells"], xy, np.zeros(3, dtype=np.int32), Pose(300.0, 0.0, 0.0)
    )
    assert np.isfinite(weights).all()
    assert kept.count == case["cells"].count
    assert kept.n_no_eligible_owner == case["cells"].n_no_eligible_owner
    assert np.bincount(labels, minlength=3).sum() == kept.count
    assert all(d["cannot_fly"] for d in policy.per_drone([0, 1, 2]).values())


def test_a_grounded_drone_alone_in_its_component_leaves_uncoverable_work(case):
    """The semantics the author fixed: such a drone IS taken out of play, and its
    region is counted as uncoverable. Keeping it in play to hold those cells
    would leave work with a drone that will never fly it -- the same unaccounted
    coverage loss in another shape. Nobody will cover them, and it is said."""
    poses = [Pose(100.0, 120.0, 0.0), Pose(500.0, 120.0, 0.0)]
    xy = np.array([[p.x, p.y] for p in poses])
    states = [DroneEnergyState(0, poses[0], CAPACITY_J, False),
              DroneEnergyState(1, poses[1], 0.02 * CAPACITY_J, False)]
    policy = EnergyWeightPolicy(case["ctx"], states, ALT, SETTINGS, CAPACITY_J, poses)
    assert policy.excluded(2).tolist() == [False, True]

    cells = case["cells"]
    drone_comp = np.array([0, 1], dtype=np.int32)          # one drone per component
    component = np.where(cells.centroids_xy[:, 0] < 300.0, 0, 1).astype(np.int32)
    split = replace(cells, component=component)

    labels, _, _, _, _, _, kept = LloydPartitioner(SETTINGS, policy).run(
        split, xy, drone_comp, Pose(300.0, 0.0, 0.0)
    )
    orphans = int((component == 1).sum())
    assert orphans > 0
    assert kept.n_no_eligible_owner == cells.n_no_eligible_owner + orphans
    assert kept.count == cells.count - orphans
    assert (labels == 1).sum() == 0                        # nothing left with it
    assert kept.centroids_xy[:, 0].max() < 300.0           # and nothing crossed over


def test_an_unusable_energy_density_is_refused_rather_than_dividing_by_it(case):
    """rho is the divisor that turns joules into square metres. A platform with no
    coverage power and no camera makes it 0.0, which would push NaN weights into
    the partition loop and quietly wreck the assignment."""
    # the propulsion term reads its power off the EnergyModel, so the model has to
    # be rebuilt from the zero-power spec -- replacing ctx.spec alone changes only
    # the speed and swath
    dead = replace(case["spec"],
                   power_w={**case["spec"].power_w, ManeuverType.COVERAGE: 0.0})
    ctx = replace(case["ctx"], sensor_power_w=0.0, spec=dead, em=EnergyModel(dead))
    from uav_swarm_sim.planning.energy_balance import coverage_energy_density_j_per_m2

    assert coverage_energy_density_j_per_m2(ctx, ALT) == 0.0
    with pytest.raises(ValueError, match="finite and > 0"):
        EnergyWeightPolicy(ctx, _states(case, [CAPACITY_J, CAPACITY_J]), ALT, SETTINGS,
                           CAPACITY_J, case["poses"])


def test_the_reported_energy_describes_the_zones_that_are_returned(case):
    """update() records demand, budget and slack for the assignment BEFORE it
    moves the weights, and run() then makes one more assignment with the moved
    weights. Without a read-only refresh the diagnostics would describe zones the
    engine never sees -- so every drone's reported energy must have been computed
    against its own final area."""
    policy = _policy(case, [CAPACITY_J, 0.45 * CAPACITY_J], SETTINGS)
    labels, _, _, _, _, _, kept = _run(case, policy, SETTINGS)
    area = np.bincount(labels, weights=kept.areas_m2, minlength=2)

    report = policy.per_drone([0, 1])
    for i in (0, 1):
        assert report[i]["estimate_area_m2"] == pytest.approx(float(area[i]), rel=1e-12)
        assert report[i]["slack_j"] == pytest.approx(
            report[i]["budget_j"] - report[i]["demand_j"]
        )
