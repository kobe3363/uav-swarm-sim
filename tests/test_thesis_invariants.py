"""Thesis-critical invariant locks (H1 hardening).

These tests defend the five claims a thesis examiner is most likely to probe and
that the existing 172-test suite does *not* yet pin down directly. Each test maps
to one defensible statement in the methodology; nothing here duplicates an
existing test (the existing suite already locks: E = sum P*dt vs distance-average,
the formation factor being CRUISE/FW-only, weighted-area proportional-to-battery,
the SMDP embedded->time-weighted correction, launch-site furthest-point
feasibility, swap != failure routing, Dubins shortest-path properties, and
paired-seed reproducibility).

Groups
------
1. 2.5D byte-identity of the energy model -- the vertical extension must be a
   strict no-op on every horizontal / single-layer segment, so the validated 2D
   energies are unchanged.
2. 2.5D vertical-energy correctness -- the ONLY mass coupling is the m*g*dz
   gravitational potential charged on a genuine climb; descent is dissipative
   (no regeneration).
3. Dynamic RTH reserve -- the return trigger is the live route-vs-return
   comparison (distance-dependent), explicitly NOT a fixed battery fraction.
4. Battery swap is energy-free -- a swap consumes wall-clock/sim time but zero
   energy (the drone has landed). Time-vs-energy asymmetry is a locked physics
   invariant.
5. The energy-weighted partition contribution -- battery weighting moves a
   depleted drone's area share strictly below the uniform (position-only)
   baseline, and collapses to the baseline when batteries are equal.
"""
from __future__ import annotations

import pytest
from conftest import config_path

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import (
    DroneStateView,
    Path,
    Pose,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import AgentState, ManeuverType
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.physical_model.vertical_segments import (
    climb_path,
    climb_profile,
    descent_path,
)
from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.events import EventBus
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import StateMachine
from uav_swarm_sim.planning.weighted_decomposition import (
    TgcBasicDecomposer,
    WeightedTgcDecomposer,
)

# The single home of the gravitational constant (mirrors energy_model._G).
_G = 9.80665


def _spec(platform: str = "MULTIROTOR"):
    cfg = load_config(config_path(), overrides={"platform_type": platform})
    return build_spec(cfg), cfg


# --------------------------------------------------------------------------- #
# Group 1 -- 2.5D byte-identity: the vertical machinery is a no-op horizontally #
# --------------------------------------------------------------------------- #
def test_horizontal_leg_charges_no_vertical_potential():
    """A constant-altitude (dz == 0) leg must carry exactly zero mass term, even
    though the platform has positive mass. This is what makes the 2.5D extension
    byte-identical to the validated 2D model on every horizontal segment: the
    energy of a cruise leg equals its pure propulsion integral P*dt and nothing
    else."""
    spec, _ = _spec()
    em = EnergyModel(spec)
    assert spec.mass_kg > 0.0, "mass must exist so the test is non-vacuous"

    leg = Path.from_segments(
        [straight_segment(Pose(0.0, 0.0, 0.0, 0.0), 250.0, ManeuverType.CRUISE, 12.0)]
    )
    propulsion_only = em.segment_energy(ManeuverType.CRUISE, leg.total_duration_s)
    assert em.path_energy(leg) == pytest.approx(propulsion_only)


def test_nonpositive_altitude_change_is_empty_and_free():
    """The boundary that guarantees the no-op: a climb/descent of zero (or
    negative) height produces an empty path and zero energy, so the single-layer
    case never enters the vertical branch at all."""
    spec, _ = _spec()
    em = EnergyModel(spec)
    for dz in (0.0, -25.0):
        assert climb_path(spec, dz).segments == ()
        assert descent_path(spec, dz).segments == ()
        assert em.path_energy(climb_path(spec, dz)) == 0.0
        assert em.path_energy(descent_path(spec, dz)) == 0.0


# --------------------------------------------------------------------------- #
# Group 2 -- 2.5D vertical energy: m*g*dz on climb, no regeneration on descent  #
# --------------------------------------------------------------------------- #
def test_climb_charges_exactly_mgh_potential():
    """A genuine inter-layer climb charges its table propulsion PLUS exactly the
    gravitational potential m*g*dz -- the single point at which mass couples into
    energy. Isolating the potential (total minus propulsion) must equal m*g*dz to
    floating-point tolerance."""
    spec, _ = _spec()
    em = EnergyModel(spec)
    dz = 60.0
    path = climb_path(spec, dz, Pose(0.0, 0.0, 0.0, 0.0))
    propulsion = em.segment_energy(ManeuverType.CLIMB, path.total_duration_s)
    potential = em.path_energy(path) - propulsion
    assert potential == pytest.approx(spec.mass_kg * _G * dz)


def test_descent_is_dissipative_no_regeneration():
    """Descent charges propulsion only: the negative altitude change contributes
    nothing (these platforms have no regenerative recovery). Consequently a climb
    of dz followed by a descent of dz does not return to the propulsion-only
    cost -- the unrecovered surplus is exactly the m*g*dz spent on the way up."""
    spec, _ = _spec()
    em = EnergyModel(spec)
    dz = 60.0
    at = Pose(0.0, 0.0, 0.0, 0.0)

    desc = descent_path(spec, dz, at)
    assert em.path_energy(desc) == pytest.approx(
        em.segment_energy(ManeuverType.DESCENT, desc.total_duration_s)
    ), "descent must carry no mass term (no regen, no negative credit)"

    climb = climb_path(spec, dz, at)
    roundtrip = em.path_energy(climb) + em.path_energy(desc)
    propulsion_only = em.segment_energy(
        ManeuverType.CLIMB, climb.total_duration_s
    ) + em.segment_energy(ManeuverType.DESCENT, desc.total_duration_s)
    assert roundtrip - propulsion_only == pytest.approx(spec.mass_kg * _G * dz)


def test_climb_energy_strictly_increases_with_altitude():
    """Monotonicity of the vertical energy: climbing twice as high strictly costs
    more (the potential term scales with dz). Guards against a regression that
    drops or flattens the altitude dependence."""
    spec, _ = _spec()
    em = EnergyModel(spec)
    e_low = climb_profile(spec, em, 40.0).energy_j
    e_high = climb_profile(spec, em, 80.0).energy_j
    assert e_high > e_low > 0.0


# --------------------------------------------------------------------------- #
# Group 3 -- dynamic RTH reserve is distance-dependent, not a fixed fraction    #
# --------------------------------------------------------------------------- #
def test_return_energy_strictly_increases_with_distance_to_home():
    """The return cost is computed from the drone's live position, so a drone far
    from base needs strictly more energy to get home than a near one. This is the
    quantity the dynamic reserve is built on (vs. a static percentage)."""
    spec, cfg = _spec()
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    base = Pose(0.0, 0.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=100.0)

    near = rth.return_energy(Pose(50.0, 0.0, 0.0, 0.0))
    far = rth.return_energy(Pose(5000.0, 0.0, 0.0, 0.0))
    assert far > near > 0.0


def test_rth_trigger_depends_on_distance_not_fixed_fraction():
    """At ONE fixed battery level (and identical next-step cost), the return
    decision flips with geometry: a far drone must return while a near drone may
    continue. A static-fraction reserve could never produce this distance-keyed
    flip -- it is the essence of the dynamic route-vs-return rule (guideline
    3.1)."""
    spec, cfg = _spec()
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    base = Pose(0.0, 0.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=100.0)

    near = Pose(50.0, 0.0, 0.0, 0.0)
    far = Pose(5000.0, 0.0, 0.0, 0.0)
    e_next = 0.0
    reserve = cfg.rth.reserve_frac * spec.battery_capacity_j
    # a level above the near requirement but below the far requirement
    level = 0.5 * (rth.return_energy(near) + rth.return_energy(far)) + reserve

    assert rth.should_return(level, e_next, near) is False
    assert rth.should_return(level, e_next, far) is True


# --------------------------------------------------------------------------- #
# Group 4 -- battery swap consumes time but zero energy (the drone has landed)  #
# --------------------------------------------------------------------------- #
def test_battery_swap_consumes_time_but_zero_energy():
    """While an agent is in S_SWAP it must burn zero energy and lose no charge --
    swap time is logistics overhead, not flight. The drone stays in S_SWAP until
    an external swap_done signal arrives, so stepping it on its own advances time
    without touching the battery."""
    cfg = load_config(config_path())
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    em = EnergyModel(spec)
    bat = Battery(spec.battery_capacity_j, cfg.battery_zones, initial_frac=0.5)
    sm = StateMachine(cfg.battery_zones)
    base = Pose(0.0, 0.0, 0.0, 0.0)
    rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=100.0)
    # formation=None is exercised by the production code path (S_SWAP returns
    # from _tick_dynamics before any formation lookup).
    agent = Agent(0, spec, motion, em, bat, sm, rth, None, base)

    agent.state = AgentState.S_SWAP
    agent._swap_done = False  # no station servicing -> remains mid-swap
    bus = EventBus()

    level_before = agent.battery.level_j
    energy_before = agent.energy_consumed_j

    t = 0.0
    for _ in range(40):  # 40 * 0.5 s = 20 s of swap time
        agent.step(0.5, t, bus)
        t += 0.5

    assert agent.state is AgentState.S_SWAP, "must stay mid-swap without swap_done"
    assert agent.battery.level_j == level_before, "swap must not drain charge"
    assert agent.energy_consumed_j == energy_before, "swap must cost zero energy"


# --------------------------------------------------------------------------- #
# Group 5 -- the energy-weighted partition contribution (directional vs uniform)#
# --------------------------------------------------------------------------- #
def test_battery_weighting_shrinks_depleted_drone_share_vs_uniform():
    """The central contribution, isolated at the weighting layer: with unequal
    batteries the battery-weighted target share of a depleted drone is strictly
    SMALLER than the position-only (uniform) baseline gives it, and the full
    drone's share is strictly LARGER; the uniform baseline stays 1/n regardless
    of battery. The TgcBasic ablation shares the identical machinery, so this
    difference is attributable to the battery weighting alone."""
    drones = [
        DroneStateView(0, 1.0, Pose(0.0, 0.0, 0.0)),    # full
        DroneStateView(1, 0.2, Pose(100.0, 0.0, 0.0)),  # depleted
        DroneStateView(2, 0.5, Pose(200.0, 0.0, 0.0)),  # mid
    ]
    n = len(drones)
    weighted = WeightedTgcDecomposer().weights(drones)
    uniform = TgcBasicDecomposer().weights(drones)

    # baseline is exactly 1/n for everyone, battery-independent
    for d in drones:
        assert uniform[d.id] == pytest.approx(1.0 / n)

    # weighted shares form a valid distribution
    assert sum(weighted.values()) == pytest.approx(1.0)

    # directional effect of the weighting
    assert weighted[1] < uniform[1], "depleted drone must get less than uniform"
    assert weighted[0] > uniform[0], "full drone must get more than uniform"

    # and the weighted ranking follows battery: full > mid > depleted
    assert weighted[0] > weighted[2] > weighted[1]


def test_weighting_collapses_to_uniform_when_batteries_equal():
    """Boundary that the ablation depends on: when every battery is equal the
    weighted partition must coincide with the uniform baseline (the weighting has
    no imbalance to correct), so any measured difference in an experiment is due
    to genuine battery spread, not algorithm bias."""
    eq = [DroneStateView(i, 0.7, Pose(i * 10.0, 0.0, 0.0)) for i in range(4)]
    weighted = WeightedTgcDecomposer().weights(eq)
    for d in eq:
        assert weighted[d.id] == pytest.approx(1.0 / len(eq))
