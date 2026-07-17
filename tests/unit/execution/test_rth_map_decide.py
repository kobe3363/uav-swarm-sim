"""EM-01 Stage 2 -- map-based RTH decide (rth.energy_map.decide).

Unit tests for the three Stage-2 seams on tiny synthetic environments:

  * 7a map arm: on-lattice the map value EQUALS the analytic straight-chord
    value (both are cruise J/m x distance + landing); off-lattice it lives in
    the provable octile envelope [straight - hop, straight x 1.0824 + hop] --
    exact equality off-lattice is impossible by construction, the envelope IS
    the "map == analytic" assertion;
  * the x1.5 obstacle fudge is NOT applied on the map arm (occupancy penalties
    already live in the edge weights -- it would double-count), while the
    analytic arm keeps it;
  * the landing term is still added on top of E_home (the map is
    cruise-horizontal by the Stage-1 contract);
  * fallback: an E_home=inf cell (true obstacle boxing) or an out-of-grid pose
    (defensive; the universal extent makes it geometrically unreachable) falls
    back to the analytic value bit-for-bit, counted in n_map_fallbacks;
  * 7b arm bound: level > sortie_arm_j implies should_return is False for
    every remaining bundle (per-endpoint decide-time quantities, sound under
    mixed map/fallback endpoints);
  * battery-quantized cadence: no decide above the arm, then re-decides only
    per 1% capacity drop; the flag-off 5 s time cadence is untouched.
"""
from __future__ import annotations

import dataclasses
import math

import pytest
from shapely.geometry import Polygon, box

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.events import EventBus
from uav_swarm_sim.execution.formation_manager import FormationManager
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import StateMachine
from uav_swarm_sim.infrastructure.config import EnergyMapConfig, load_config
from uav_swarm_sim.infrastructure.core_types import CoveragePlan, Pose, Waypoint
from uav_swarm_sim.infrastructure.enums import AgentState, ManeuverType
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.physical_model.vertical_segments import landing_profile
from uav_swarm_sim.planning.energy_map import build_energy_map
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle

AREA = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
CELL = 20.0
ALT = 100.0
# max octile/euclid ratio on an 8-connected grid (at 22.5 deg): sqrt(2)*cos(pi/8)/...
_OCTILE_FACTOR = 1.0824


@pytest.fixture(scope="module")
def kit(config_path):
    cfg = load_config(config_path, overrides={"platform_type": "MULTIROTOR"})
    spec = build_spec(cfg)
    return cfg, spec, make_motion_model(spec), EnergyModel(spec)


def _emap(env, base, kit, **kw):
    _, spec, _, em = kit
    return build_energy_map(env, base, CELL, em, spec.v_cruise, **kw)


def _calc(kit, base, env=None, emap=None):
    """emap=None => the analytic arm (map_decide gated on a built map)."""
    cfg, spec, motion, em = kit
    rth_cfg = dataclasses.replace(
        cfg.rth, energy_map=EnergyMapConfig(enabled=True, decide=True))
    return RthCalculator(em, motion, spec, rth_cfg, base, altitude_m=ALT, env=env,
                         energy_map=emap)


def _land(kit, altitude_m=ALT):
    _, spec, _, em = kit
    return landing_profile(spec, em, altitude_m).energy_j


def _cruise(kit, dist_m):
    _, spec, _, em = kit
    return em.distance_energy(dist_m, ManeuverType.CRUISE, spec.v_cruise)


# --------------------------------------------------------------------------- #
# 1. seam 7a: exact equality on a lattice axis (pure straight, no yaw)         #
# --------------------------------------------------------------------------- #
def test_map_return_energy_exact_on_lattice_axis(kit):
    env = EnvironmentMap(AREA, [], 5.0)
    # base and pose both head pi (pose faces the base): the holonomic plan is a
    # pure straight with zero yaw segments, so analytic == cruise J/m x dist.
    base = Pose(500.0, 500.0, math.pi)
    emap = _emap(env, base, kit)
    map_calc = _calc(kit, base, env=env, emap=emap)
    ana_calc = _calc(kit, base, env=env)

    pose = Pose(500.0 + 10 * CELL, 500.0, math.pi)
    expect = _cruise(kit, 10 * CELL) + _land(kit)
    assert map_calc.return_energy(pose) == pytest.approx(expect, rel=1e-9)
    assert ana_calc.return_energy(pose) == pytest.approx(expect, rel=1e-9)
    assert map_calc.n_map_hits == 1 and map_calc.n_map_fallbacks == 0
    assert map_calc.map_decide_on and not ana_calc.map_decide_on


# --------------------------------------------------------------------------- #
# 2. seam 7a: octile envelope off-lattice + decision-level agreement           #
# --------------------------------------------------------------------------- #
def test_map_return_energy_octile_bounds_obstacle_free(kit):
    cfg, spec, _, _ = kit
    env = EnvironmentMap(AREA, [], 5.0)
    base = Pose(500.0, 500.0, 0.0)
    emap = _emap(env, base, kit)
    map_calc = _calc(kit, base, env=env, emap=emap)
    ana_calc = _calc(kit, base, env=env)

    land = _land(kit)
    hop_diag = _cruise(kit, math.sqrt(2.0) * CELL)
    reserve = cfg.rth.reserve_frac * spec.battery_capacity_j
    for xy in [(837.3, 211.9), (63.1, 402.7), (991.0, 993.5), (531.2, 468.9)]:
        pose = Pose(xy[0], xy[1], 0.0)
        m = map_calc.return_energy(pose) - land
        straight = _cruise(kit, math.dist(xy, (base.x, base.y)))
        # octile >= euclid; <= 8.24% octile overhead; one diagonal hop of slack
        # for the in-cell pose quantization (the base end is exact by frame
        # construction).
        assert m >= straight - hop_diag - 1e-9
        assert m <= straight * _OCTILE_FACTOR + hop_diag + 1e-9

        # outside the envelope both arms make the SAME decision
        e_next = 1000.0
        hi = e_next + straight * _OCTILE_FACTOR + hop_diag + land + reserve + 1.0
        ana_hi = e_next + ana_calc.return_energy(pose) + reserve + 1.0
        level_hi = max(hi, ana_hi)
        assert not map_calc.should_return(level_hi, e_next, pose)
        assert not ana_calc.should_return(level_hi, e_next, pose)
        assert map_calc.should_return(1.0, e_next, pose)
        assert ana_calc.should_return(1.0, e_next, pose)


# --------------------------------------------------------------------------- #
# 3. seam 7a: no x1.5 fudge on the map arm                                     #
# --------------------------------------------------------------------------- #
def test_map_arm_has_no_obstacle_fudge(kit):
    _, _, motion, em = kit
    wall = Obstacle(id=0, cls=0, polygon=box(480.0, 0.0, 520.0, 800.0))
    env = EnvironmentMap(AREA, [wall], 5.0)
    base = Pose(250.0, 500.0, 0.0)
    emap = _emap(env, base, kit)
    map_calc = _calc(kit, base, env=env, emap=emap)
    ana_calc = _calc(kit, base, env=env)

    pose = Pose(750.0, 500.0, math.pi)  # straight chord crosses the wall
    i, j = emap.frame.world_to_cell(pose.x, pose.y)
    land = _land(kit)

    # map arm == E_home + landing EXACTLY: the occupancy penalty lives in the
    # edge weights, no x1.5 on top (that would double-count the obstacle)
    assert map_calc.return_energy(pose) == pytest.approx(
        float(emap.e_home[i, j]) + land, rel=1e-12
    )
    # the analytic arm on the same pose IS the fudged blocked-chord estimate
    route = motion.plan(pose, base, ManeuverType.CRUISE)
    assert not env.path_clear(route)
    assert ana_calc.return_energy(pose) == pytest.approx(
        em.path_energy(route) * 1.5 + land, rel=1e-12
    )


# --------------------------------------------------------------------------- #
# 4. the landing term is still added at decide time (analytical == execution)  #
# --------------------------------------------------------------------------- #
def test_landing_term_still_added_on_map_arm(kit):
    env = EnvironmentMap(AREA, [], 5.0)
    base = Pose(500.0, 500.0, 0.0)
    emap = _emap(env, base, kit)
    map_calc = _calc(kit, base, env=env, emap=emap)

    pose = Pose(700.0, 500.0, 0.0)
    diff = map_calc.return_energy(pose, altitude_m=100.0) - map_calc.return_energy(
        pose, altitude_m=50.0
    )
    expect = _land(kit, 100.0) - _land(kit, 50.0)
    assert expect > 0.0
    assert diff == pytest.approx(expect, rel=1e-9)


# --------------------------------------------------------------------------- #
# 5. fallback: E_home=inf (boxing) and out-of-grid -> analytic, bit-for-bit    #
# --------------------------------------------------------------------------- #
def test_fallback_unreachable_cell_and_outside_grid(kit):
    # obstacle-ringed pocket (Stage-1 fixture): the pocket cell is free but has
    # no route to the base => E_home = inf
    ring = Obstacle(
        id=0, cls=0,
        polygon=Polygon(
            box(300.0, 300.0, 400.0, 400.0).exterior.coords,
            [box(340.0, 340.0, 360.0, 360.0).exterior.coords],
        ),
    )
    env = EnvironmentMap(AREA, [ring], 0.0)
    base = Pose(50.0, 50.0, 0.0)
    emap = _emap(env, base, kit)
    map_calc = _calc(kit, base, env=env, emap=emap)
    ana_calc = _calc(kit, base, env=env)

    pocket = Pose(350.0, 350.0, 0.0)
    assert map_calc.return_energy(pocket) == ana_calc.return_energy(pocket)  # bit-for-bit
    assert map_calc.n_map_fallbacks == 1 and map_calc.n_map_hits == 0

    # defensive out-of-grid check on a deliberately small grid (the engine's
    # universal extent makes this unreachable in a real mission)
    small_area = box(0.0, 0.0, 200.0, 200.0)
    env2 = EnvironmentMap(Polygon(small_area.exterior.coords), [], 0.0)
    base2 = Pose(100.0, 100.0, 0.0)
    calc2 = _calc(kit, base2, env=env2, emap=_emap(env2, base2, kit))
    ana2 = _calc(kit, base2, env=env2)
    outside = Pose(900.0, 900.0, 0.0)
    assert calc2.return_energy(outside) == ana2.return_energy(outside)
    assert calc2.n_map_fallbacks == 1 and calc2.n_map_hits == 0


# --------------------------------------------------------------------------- #
# 6. seam 7b: the arm bounds every remaining decide (mixed map + fallback)     #
# --------------------------------------------------------------------------- #
def test_sortie_arm_bound_soundness(kit):
    cfg, spec, _, _ = kit
    ring = Obstacle(
        id=0, cls=0,
        polygon=Polygon(
            box(300.0, 300.0, 400.0, 400.0).exterior.coords,
            [box(340.0, 340.0, 360.0, 360.0).exterior.coords],
        ),
    )
    env = EnvironmentMap(AREA, [ring], 0.0)
    base = Pose(50.0, 50.0, 0.0)
    calc = _calc(kit, base, env=env, emap=_emap(env, base, kit))

    bundles = [
        (5000.0, Pose(900.0, 900.0, 0.0)),   # map-served endpoint
        (12000.0, Pose(350.0, 350.0, 0.0)),  # E_home=inf -> analytic fallback
        (800.0, Pose(500.0, 120.0, 0.0)),    # map-served endpoint
    ]
    arm = calc.sortie_arm_j(bundles, ALT)
    reserve = cfg.rth.reserve_frac * spec.battery_capacity_j
    assert math.isfinite(arm)
    for e_k, p_k in bundles:
        assert arm >= e_k + calc.return_energy(p_k, altitude_m=ALT) + reserve
        assert not calc.should_return(arm + 1.0, e_k, p_k, altitude_m=ALT)

    assert calc.sortie_arm_j([], ALT) == float("inf")  # empty plan: never skip


# --------------------------------------------------------------------------- #
# agent-level: cadence + arming mechanics                                      #
# --------------------------------------------------------------------------- #
def _simple_plan():
    wps = [
        Waypoint(Pose(100, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(150, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(150, 50, math.pi), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(100, 50, math.pi), ManeuverType.COVERAGE, 6.0),
    ]
    return CoveragePlan(0, wps, 0.0, 0.0)


def _make_agent(config_path, decide: bool, capacity_j: float | None = None):
    cfg = load_config(config_path, overrides={"platform_type": "MULTIROTOR"})
    spec = build_spec(cfg)
    if capacity_j is not None:
        object.__setattr__(spec, "battery_capacity_j", capacity_j)
    motion = make_motion_model(spec)
    em = EnergyModel(spec)
    bat = Battery(spec.battery_capacity_j, cfg.battery_zones, 1.0)
    sm = StateMachine(cfg.battery_zones)
    base = Pose(0.0, 0.0, 0.0)
    if decide:
        env = EnvironmentMap(AREA, [], 5.0)
        emap = build_energy_map(env, base, CELL, em, spec.v_cruise)
        rth_cfg = dataclasses.replace(
            cfg.rth, energy_map=EnergyMapConfig(enabled=True, decide=True))
        rth = RthCalculator(em, motion, spec, rth_cfg, base, altitude_m=ALT,
                            env=env, energy_map=emap)
    else:
        rth = RthCalculator(em, motion, spec, cfg.rth, base, altitude_m=ALT)
    aero = AeroCorrection(cfg.aero, spec.platform)
    fm = FormationManager(aero, cfg.aero, spec.platform)
    agent = Agent(0, spec, motion, em, bat, sm, rth, fm, base)
    transit = motion.plan(base, Pose(100, 0, 0.0), ManeuverType.CRUISE)
    agent.assign(_simple_plan(), transit)
    return agent, cfg


def _drive_to_mission(agent, bus, t=0.0, dt=0.5, max_ticks=10000):
    for _ in range(max_ticks):
        agent.step(dt, t, bus)
        t += dt
        if agent.state is AgentState.S2_MISSION:
            return t
    raise AssertionError(f"agent never reached S2_MISSION (state={agent.state})")


def _count_decides(agent):
    calls: list[float] = []
    orig = agent.rth.decide
    agent.rth.decide = lambda a: (calls.append(a.battery.level_j), "CONTINUE")[1]
    return calls, orig


def test_battery_quantized_cadence(config_path):
    # small capacity so the tiny plan drains several 1% quanta
    agent, _ = _make_agent(config_path, decide=True, capacity_j=100_000.0)
    bus = EventBus()
    t = _drive_to_mission(agent, bus)
    step_j = agent.rth.decide_step_j
    assert step_j == pytest.approx(1000.0)

    calls, _ = _count_decides(agent)

    # phase 1 -- ABOVE the arm: no decide at all while the battery drains
    agent._arm_level_j = agent.battery.level_j - 2 * step_j
    agent._rth_last_check_level_j = float("inf")
    arm = agent._arm_level_j
    while agent.battery.level_j > arm and agent.state in (
        AgentState.S2_MISSION, AgentState.S_FERRY
    ):
        agent.step(0.5, t, bus)
        t += 0.5
    assert agent.state in (AgentState.S2_MISSION, AgentState.S_FERRY)
    assert len(calls) <= 1  # nothing above the arm (first call AT the crossing)

    # phase 2 -- below the arm: first tick evaluates, then only per >=1% drop
    for _ in range(4000):
        if agent.state not in (AgentState.S2_MISSION, AgentState.S_FERRY):
            break
        agent.step(0.5, t, bus)
        t += 0.5
    assert len(calls) >= 3
    assert calls[0] <= arm + 1e-9  # never above the arm
    diffs = [calls[k] - calls[k + 1] for k in range(len(calls) - 1)]
    assert all(d >= step_j - 1e-6 for d in diffs)


def test_time_cadence_unchanged_when_decide_off(config_path):
    agent, cfg = _make_agent(config_path, decide=False)
    assert agent.rth.map_decide_on is False
    bus = EventBus()
    t = _drive_to_mission(agent, bus)

    calls, _ = _count_decides(agent)
    lvl0 = agent.battery.level_j
    n_ticks = int(30.0 / 0.5)
    for _ in range(n_ticks):
        if agent.state not in (AgentState.S2_MISSION, AgentState.S_FERRY):
            break
        agent.step(0.5, t, bus)
        t += 0.5

    # 30 s at check_interval_s=5 s -> ~6 calls: MORE than the battery-quantized
    # cadence could ever produce over the same drain (drain/step + 1), which
    # proves the flag-off arm still runs on time, not on battery quanta
    assert len(calls) >= 4
    drained = lvl0 - agent.battery.level_j
    max_battery_cadence_calls = int(drained / agent.rth.decide_step_j) + 1
    assert len(calls) > max_battery_cadence_calls


def test_arm_recomputed_per_sortie_and_stale_decision_cleared(config_path):
    agent, _ = _make_agent(config_path, decide=True)
    bus = EventBus()
    agent.step(0.5, 0.0, bus)  # S0 -> S1 fires _arm_sortie
    assert agent.state is AgentState.S1_TRANSIT
    assert agent._sortie_idx == 1 and len(agent.sortie_arms) == 1
    assert math.isfinite(agent._arm_level_j)
    assert agent.sortie_arms[0] == (1, agent._arm_level_j)

    # stale RETURN_NOW from a previous sortie is cleared on re-arm
    agent._rth_decision = True
    agent._rth_last_check_level_j = 12345.0
    agent._arm_sortie()
    assert agent._rth_decision is False
    assert agent._sortie_idx == 2 and len(agent.sortie_arms) == 2
    assert agent._rth_last_check_level_j == float("inf")

    # adopt_plan on a live agent (redistribution) re-arms as well
    agent.state = AgentState.S2_MISSION
    transit = agent.motion.plan(agent.pose, Pose(100, 0, 0.0), ManeuverType.CRUISE)
    agent.adopt_plan(_simple_plan(), transit)
    assert agent.state is AgentState.S1_TRANSIT
    assert agent._sortie_idx == 3 and len(agent.sortie_arms) == 3


def test_arm_sortie_noop_when_decide_off(config_path):
    agent, _ = _make_agent(config_path, decide=False)
    bus = EventBus()
    agent.step(0.5, 0.0, bus)  # S0 -> S1
    assert agent.state is AgentState.S1_TRANSIT
    assert agent._sortie_idx == 0 and agent.sortie_arms == []
    assert agent._arm_level_j == float("inf")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
