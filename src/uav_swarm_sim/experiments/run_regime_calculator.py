"""A2 -- regime calculator: E_cover vs n·B_usable (battery-limited vs fuel-surplus).

Standalone decision-support tool, modelled on ``run_fleet_sizing_analyzer.py``
(pure calculation -- it builds the planning layer and the real launch site, but
its headline numbers are analytical, not a dt-stepped run). It answers the one
question the shape study must settle FIRST: for a given area/shape + platform +
fleet, is the baseline **battery-limited** (E_cover > n·B_usable => swaps needed,
shape matters, weighting matters) or **fuel-surplus** (n·B exceeds E_cover =>
no swaps, weighted ≈ unweighted, shape stops modulating)? If the baseline is
fuel-surplus, every shape sweep is flat and uninformative -- so this go/no-go
gates the whole study.

E_cover -- physical truth, not a closed-form shortcut
-----------------------------------------------------
``E_cover`` is the energy to cover the whole survey polygon ONCE. Its dominant,
shape-dependent core (coverage strips + inter-strip connectors) is NOT a fudge
formula: it is derived from the SAME code the simulator executes. We build the
boustrophedon plan for the whole area (``planning/coverage_path.boustrophedon``),
rebuild the agent's coverage legs exactly as ``Agent._build_coverage_legs`` does
(even global leg = COVERAGE strip, odd = TURN connector, each via
``motion.plan``), and integrate each leg with ``EnergyModel.path_energy`` plus the
camera payload term over its COVERAGE segments (mirroring ``Agent._tick_dynamics``
/ ``_leg_sensor_energy``). Transit (out+back, CRUISE) and the takeoff/landing
vertical profiles are added as launch-dependent overhead.

    E_cover = E_coverage(strips) + E_connectors(TURN) + E_sensor(camera)
              + E_transit_round(CRUISE) + E_vertical(takeoff + landing)

The coverage core is verified against a REAL single mission (``--verify``): the
engine's actually-drained coverage-phase (S2_MISSION + S_FERRY) energy must match
the analytical coverage energy within a stated tolerance. The small residual is
the engine's dt-quantisation (it charges a full P·dt on the last partial tick of
each leg) and shrinks linearly with dt -- so the analytical value is the exact
leg integral the discrete sum converges to.

B_usable -- read from the battery-zone config
---------------------------------------------
``B_usable = capacity_j × (1 − reserve_floor)``. THREE floors are reported (each
read from config, none hard-coded); the classification uses the one chosen with
``--usable-floor``:
  * ``terminal`` (default) = ``battery_zones.critical`` (0.20): drain until the
    TERMINAL reserve. This is the "reserve/terminal fraction" the zones define.
  * ``return``  = ``battery_zones.nominal`` (0.40): the fraction at which the
    CRITICAL-zone guard actually forces the drone home in the executed sim
    (the OPERATIONAL usable budget).
  * ``rth``     = ``rth.reserve_frac`` (0.05): the hard epsilon floor only
    (theoretical maximum, matches ``fleet_sizing``).

*** THESIS-AFFECTING CHOICE: the floor changes B_usable, hence the ratio and n*.
    It is FLAGGED here and left selectable; the default ``terminal`` is the
    battery-zone reserve. Pick deliberately and state it in the thesis. ***

Usage
-----
    python -m uav_swarm_sim.experiments.run_regime_calculator \
        [--config config/default.yaml] [--geojson data/areas/shapes/square.geojson] \
        [--n-drones 5] [--usable-floor terminal|return|rth] \
        [--verify] [--verify-n 1] [--sensor-power-w 0]
"""
from __future__ import annotations

import argparse
import math
import sys

from ..infrastructure.config import load_config
from ..infrastructure.core_types import Pose, Zone
from ..infrastructure.enums import AgentState, DecompositionAlgo, ManeuverType, PlannerKind
from ..infrastructure.rng import STREAM_LAUNCH_SAMPLING, STREAM_OBSTACLES, RngFactory
from ..physical_model.aero_correction import AeroCorrection
from ..physical_model.drone_specs import build_spec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import make_motion_model
from ..physical_model.vertical_segments import landing_profile, takeoff_profile
from ..planning.coverage_path import boustrophedon
from ..planning.environment_map import EnvironmentMap
from ..planning.geojson_parser import load_area
from ..planning.gvg_builder import build_gvg
from ..planning.launch_site_optimizer import InfeasibleMissionError, optimize
from ..planning.obstacle_generator import generate as generate_obstacles
from ..planning.tgc import build_tgc

# Coverage-phase states whose drained energy is the verification target.
_COVERAGE_STATES = {AgentState.S2_MISSION, AgentState.S_FERRY}

# Analytical vs engine coverage-energy agreement bound. The residual at dt=0.5 is
# ~0.7% (engine's full-P·dt last-tick quantisation) and → 0 as dt → 0.
_VERIFY_TOL = 0.02

# Regime classification margin around ratio == 1.
_BORDERLINE_MARGIN = 0.10


# --------------------------------------------------------------------------- #
# analytical E_cover core (rebuilds the agent's real coverage legs)            #
# --------------------------------------------------------------------------- #
def _rebuild_coverage_legs(plan_waypoints, motion):
    """Exactly ``Agent._build_coverage_legs`` for boustrophedon plans: even global
    leg index = COVERAGE strip, odd = TURN connector."""
    legs = []
    for i in range(len(plan_waypoints) - 1):
        a, b = plan_waypoints[i].pose, plan_waypoints[i + 1].pose
        maneuver = ManeuverType.COVERAGE if i % 2 == 0 else ManeuverType.TURN
        legs.append(motion.plan(a, b, maneuver))
    return legs


def _leg_sensor_energy(leg, em, sensor_power_w: float) -> float:
    """Camera payload energy the leg draws at execution: sensor power over its
    COVERAGE segments only. Mirrors ``Agent._leg_sensor_energy``."""
    if sensor_power_w <= 0.0:
        return 0.0
    cov_dur = sum(s.duration_s for s in leg.segments if s.maneuver is ManeuverType.COVERAGE)
    return em.sensor_energy(cov_dur, sensor_power_w)


def coverage_energy(polygon, spec, em, motion, sensor_power_w: float) -> dict:
    """Analytical energy to sweep ``polygon`` once, built from the SAME
    boustrophedon + leg construction + P·dt integration the engine runs.

    Returns a breakdown dict: strip propulsion, connector propulsion, camera
    sensor, total coverage energy, and leg/geometry counts.
    """
    zone = Zone(drone_id=0, regions=[], polygon=polygon,
                entry_pose=Pose(polygon.centroid.x, polygon.centroid.y, 0.0))
    plan = boustrophedon(zone, spec, motion, em)
    legs = _rebuild_coverage_legs(plan.waypoints, motion)

    strip_j = connector_j = sensor_j = 0.0
    n_strips = n_connectors = 0
    for i, leg in enumerate(legs):
        e = em.path_energy(leg)
        if i % 2 == 0:
            strip_j += e
            n_strips += 1
            sensor_j += _leg_sensor_energy(leg, em, sensor_power_w)
        else:
            connector_j += e
            n_connectors += 1
    return {
        "strip_j": strip_j,
        "connector_j": connector_j,
        "sensor_j": sensor_j,
        "coverage_total_j": strip_j + connector_j + sensor_j,
        "n_strips": n_strips,
        "n_connectors": n_connectors,
        "path_length_m": plan.length_m,
    }


def transit_and_vertical(em, spec, transit_dist_m: float, altitude_m: float) -> dict:
    """Launch-dependent overhead of E_cover: round-trip transit at CRUISE plus the
    takeoff + landing vertical profiles (climb/descent). NOTE: in the single-layer
    engine the vertical profiles are a *budget/reserve* term (RTH reserve, sortie
    accounting) and are NOT executed as an energy drain; transit IS executed."""
    transit_round_j = 2.0 * em.distance_energy(transit_dist_m, ManeuverType.CRUISE, spec.v_cruise)
    take = takeoff_profile(spec, em, altitude_m)
    land = landing_profile(spec, em, altitude_m)
    return {
        "transit_round_j": transit_round_j,
        "takeoff_j": take.energy_j,
        "landing_j": land.energy_j,
        "vertical_j": take.energy_j + land.energy_j,
    }


# --------------------------------------------------------------------------- #
# planning-layer setup (real launch site, mirrors fleet-sizing analyzer)       #
# --------------------------------------------------------------------------- #
def _build_planning_layer(cfg, geojson_path: str):
    spec = build_spec(cfg)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    aero = AeroCorrection(cfg.aero, spec.platform)

    rngf = RngFactory(cfg.sim.master_seed)
    obs_rng = rngf.stream(STREAM_OBSTACLES, 0)
    launch_rng = rngf.stream(STREAM_LAUNCH_SAMPLING, 0)

    area = load_area(geojson_path)
    obstacles = generate_obstacles(area, cfg.env, obs_rng)
    env = EnvironmentMap(area, obstacles, cfg.env.clearance_buffer_m)
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    tgc = build_tgc(env, gvg)
    base_pose, _ = optimize(
        cfg.launch, tgc, env, motion, em, aero, spec,
        cfg.fleet.n_drones, launch_rng, cfg.env.coverage_altitude_m,
    )
    return env, area, spec, em, motion, base_pose


def _usable_fraction(cfg, floor: str) -> tuple[float, str]:
    """The usable battery fraction and a human label, all read from config."""
    if floor == "terminal":
        return 1.0 - cfg.battery_zones.critical, f"1 − critical({cfg.battery_zones.critical:.2f})"
    if floor == "return":
        return 1.0 - cfg.battery_zones.nominal, f"1 − nominal({cfg.battery_zones.nominal:.2f})"
    if floor == "rth":
        return 1.0 - cfg.rth.reserve_frac, f"1 − reserve_frac({cfg.rth.reserve_frac:.2f})"
    raise ValueError(f"unknown usable-floor {floor!r}")


def classify(ratio: float) -> str:
    if ratio > 1.0 + _BORDERLINE_MARGIN:
        return "BATTERY-LIMITED"
    if ratio < 1.0 - _BORDERLINE_MARGIN:
        return "FUEL-SURPLUS"
    return "BORDERLINE"


# --------------------------------------------------------------------------- #
# verification: analytical coverage energy vs a real single run                #
# --------------------------------------------------------------------------- #
def _run_once(config_path, geojson_path, n_drones, sensor_power_w):
    """One engine mission (obstacles + hazard OFF, MULTIROTOR). Returns
    (result, engine_cov_j, analytical_cov_j, n_swaps, dt_s)."""
    from ..infrastructure.simulation_engine import SimulationEngine

    overrides = {
        "env.geojson_path": geojson_path,
        "env.obstacle_density_per_km2": 0.0,
        "failure.hazard_rate_per_hour": 0.0,
        "fleet.n_drones": n_drones,
        "sensor.sensor_power_w": sensor_power_w,
        "telemetry.enabled": False,
        "safety.obstacle_recovery": False,
    }
    vcfg = load_config(config_path, overrides)
    eng = SimulationEngine(vcfg, RngFactory(vcfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI, planner=PlannerKind.DUBINS)
    result = eng.run()

    cap = eng.spec.battery_capacity_j
    engine_cov_j = 0.0
    for aid in eng.fleet.agents:
        pos = eng.history.position_trace(aid)
        bat = eng.history.battery_trace(aid)
        for i in range(min(len(pos), len(bat)) - 1):
            if pos[i][3] in _COVERAGE_STATES:
                engine_cov_j += cap * (bat[i][1] - bat[i + 1][1])

    analytical_cov_j = sum(
        coverage_energy(z.polygon, eng.spec, eng.em, eng.motion, sensor_power_w)["coverage_total_j"]
        for z in eng.partition.zones.values()
    )
    n_swaps = sum(1 for s in eng.history.sojourns() if s.state is AgentState.S_SWAP)
    return result, engine_cov_j, analytical_cov_j, n_swaps, vcfg.sim.dt_s


def verify_against_engine(config_path: str, geojson_path: str, n_drones: int,
                          sensor_power_w: float, e_cover_core_j: float,
                          usable_rth_j: float) -> dict:
    """Run ONE real mission and compare the engine's actually-drained coverage-phase
    (S2_MISSION + S_FERRY) energy to the analytical coverage energy recomputed on
    the engine's OWN partition zones (apples-to-apples).

    A verification is only clean when NO battery swaps occurred: on a mid-mission
    RTH the resumed drone re-flies the interrupted leg from its start, so a swap
    double-counts partial coverage and inflates the engine number. So we auto-pick
    the smallest fleet whose per-zone coverage fits one battery (each drone one
    sortie) starting from a computed floor, then bump n until swap-free."""
    n0 = max(n_drones, math.ceil(e_cover_core_j / (0.6 * usable_rth_j))) if usable_rth_j > 0 else n_drones
    n = max(1, n0)
    result = engine_cov_j = analytical_cov_j = n_swaps = dt_s = None
    for _ in range(6):
        result, engine_cov_j, analytical_cov_j, n_swaps, dt_s = _run_once(
            config_path, geojson_path, n, sensor_power_w
        )
        if n_swaps == 0:
            break
        n += 1

    rel_err = (abs(analytical_cov_j - engine_cov_j) / engine_cov_j
               if engine_cov_j > 0 else float("nan"))
    return {
        "outcome": result.outcome.value,
        "coverage_frac": result.coverage_frac,
        "engine_cov_j": engine_cov_j,
        "analytical_cov_j": analytical_cov_j,
        "rel_err": rel_err,
        "dt_s": dt_s,
        "n_drones": n,
        "n_swaps": n_swaps,
    }


# --------------------------------------------------------------------------- #
# entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regime calculator: E_cover vs n·B_usable (battery-limited vs fuel-surplus)."
    )
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--geojson", default=None,
                    help="survey polygon (default: env.geojson_path from config)")
    ap.add_argument("--n-drones", type=int, default=None,
                    help="fleet size for the ratio (default: fleet.n_drones from config)")
    ap.add_argument("--usable-floor", choices=["terminal", "return", "rth"], default="terminal",
                    help="battery reserve floor for B_usable (THESIS-AFFECTING; default terminal)")
    ap.add_argument("--sensor-power-w", type=float, default=None,
                    help="camera payload power (default: sensor.sensor_power_w from config)")
    ap.add_argument("--verify", action="store_true",
                    help="run one real single mission and check analytical == engine coverage energy")
    ap.add_argument("--verify-n", type=int, default=1, help="fleet size for --verify (default 1)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    geojson = args.geojson or cfg.env.geojson_path
    n_drones = args.n_drones if args.n_drones is not None else cfg.fleet.n_drones
    sensor_power_w = (args.sensor_power_w if args.sensor_power_w is not None
                      else cfg.sensor.sensor_power_w)

    try:
        env, area, spec, em, motion, base_pose = _build_planning_layer(cfg, geojson)
    except InfeasibleMissionError as exc:
        print("MISSION IMPOSSIBLE (launch siting)")
        print(str(exc))
        return 2

    if spec.platform.value != "MULTIROTOR":
        print(f"[warning] platform is {spec.platform.value}; the shape study assumes "
              "MULTIROTOR (holonomic). Numbers below still use the active platform.",
              file=sys.stderr)

    # --- E_cover components -------------------------------------------------- #
    cov = coverage_energy(area, spec, em, motion, sensor_power_w)
    centroid = area.centroid
    transit_dist_m = math.hypot(centroid.x - base_pose.x, centroid.y - base_pose.y)
    tv = transit_and_vertical(em, spec, transit_dist_m, cfg.env.coverage_altitude_m)
    e_cover = cov["coverage_total_j"] + tv["transit_round_j"] + tv["vertical_j"]

    # --- B_usable + regime --------------------------------------------------- #
    frac, frac_label = _usable_fraction(cfg, args.usable_floor)
    b_usable = spec.battery_capacity_j * frac
    fleet_budget = n_drones * b_usable
    ratio = e_cover / fleet_budget if fleet_budget > 0 else float("inf")
    regime = classify(ratio)
    n_star = max(1, math.ceil(e_cover / b_usable))

    # --- report -------------------------------------------------------------- #
    eff_swath = spec.swath_width_m
    print("# Regime Calculator — E_cover vs n·B_usable (analytical)\n")
    print(f"- Survey polygon: {geojson}")
    print(f"- Area (survey): {area.area:,.0f} m²  ({area.area / 1e6:.3f} km²)  |  "
          f"solidity {area.area / area.convex_hull.area:.4f}")
    print(f"- Platform: {spec.platform.value}  |  effective swath {eff_swath:,.1f} m  |  "
          f"v_cov {spec.v_coverage:.1f} m/s, v_cruise {spec.v_cruise:.1f} m/s")
    print(f"- Launch site (real, from optimizer): ({base_pose.x:,.1f}, {base_pose.y:,.1f})  |  "
          f"transit to centroid {transit_dist_m:,.0f} m")
    print(f"- Camera payload: sensor_power_w = {sensor_power_w:.1f} W "
          f"({'ON' if sensor_power_w > 0 else 'off'})\n")

    print("## E_cover breakdown (energy to cover the whole area ONCE)\n")
    print(f"- Coverage strips (COVERAGE): {cov['strip_j']:,.0f} J  "
          f"({cov['n_strips']} strips)")
    print(f"- Inter-strip connectors (TURN): {cov['connector_j']:,.0f} J  "
          f"({cov['n_connectors']} connectors)")
    if sensor_power_w > 0:
        print(f"- Camera payload (COVERAGE segments only): {cov['sensor_j']:,.0f} J")
    print(f"- **Coverage core (strips+connectors+camera): {cov['coverage_total_j']:,.0f} J** "
          f"[shape-dependent; the term the study modulates]")
    print(f"- Round-trip transit (CRUISE): {tv['transit_round_j']:,.0f} J")
    print(f"- Vertical takeoff+landing: {tv['vertical_j']:,.0f} J  "
          f"[budget/reserve term — NOT drained in the single-layer engine]")
    print(f"- **E_cover (total): {e_cover:,.0f} J**\n")

    print("## B_usable + regime\n")
    print(f"- capacity: {spec.battery_capacity_j:,.0f} J "
          f"({cfg.fleet.battery_capacity_wh:.0f} Wh)")
    print(f"- usable floor (--usable-floor {args.usable_floor}, THESIS-AFFECTING): "
          f"{frac_label} = {frac:.2f}")
    print(f"- **B_usable = {b_usable:,.0f} J**  per drone")
    for name in ("terminal", "return", "rth"):
        f2, lbl = _usable_fraction(cfg, name)
        mark = "  ← used" if name == args.usable_floor else ""
        print(f"    · {name:8s}: {lbl} → {spec.battery_capacity_j * f2:,.0f} J{mark}")
    print(f"- fleet budget n·B_usable (n={n_drones}): {fleet_budget:,.0f} J")
    print(f"- **E_cover / (n·B_usable) = {ratio:.3f}  →  {regime}** "
          f"(borderline band 1 ± {_BORDERLINE_MARGIN:.2f})")
    print(f"- **crossover fleet n\\* = ceil(E_cover / B_usable) = {n_star}** "
          "(smallest fleet that covers the area on one battery each)\n")

    # --- verification -------------------------------------------------------- #
    if args.verify:
        print("## Verification — analytical vs real single run\n")
        frac_rth, _ = _usable_fraction(cfg, "rth")
        usable_rth_j = spec.battery_capacity_j * frac_rth
        v = verify_against_engine(args.config, geojson, args.verify_n, sensor_power_w,
                                  cov["coverage_total_j"], usable_rth_j)
        ok = math.isfinite(v["rel_err"]) and v["rel_err"] < _VERIFY_TOL
        print(f"- single run: n={v['n_drones']}, obstacles+hazard OFF, MULTIROTOR, "
              f"dt={v['dt_s']} s → outcome {v['outcome']}, coverage {v['coverage_frac']:.3f}")
        print(f"- engine drained coverage-phase (S2+S_FERRY): {v['engine_cov_j']:,.1f} J")
        print(f"- analytical coverage (rebuilt legs on engine zones): {v['analytical_cov_j']:,.1f} J")
        print(f"- relative error: {v['rel_err']:.3e}  (tolerance {_VERIFY_TOL:.0%})  "
              f"→ {'PASS ✓' if ok else 'FAIL ✗'}")
        print("- residual is the engine's dt-quantisation (full P·dt on each leg's "
              "last partial tick); it shrinks ∝ dt, so the analytical value is the "
              "exact leg integral the discrete sum converges to.\n")

    # --- go / no-go ---------------------------------------------------------- #
    print("## Go / No-Go for the shape study\n")
    if regime == "BATTERY-LIMITED":
        print(f"- ✅ **GO.** At n={n_drones} the fleet is battery-limited "
              f"(ratio {ratio:.2f} > 1): swaps are needed, shape and weighting both "
              f"modulate the outcome. Sweep fleet sizes straddling n\\*={n_star} "
              f"(e.g. n ∈ [1, {max(2 * n_star, n_drones + 2)}]) so the sweep crosses "
              "the battery-limited → fuel-surplus boundary.")
    elif regime == "FUEL-SURPLUS":
        print(f"- ⚠️ **NO-GO at n={n_drones}.** The fleet is fuel-surplus "
              f"(ratio {ratio:.2f} < 1): a battery covers the area comfortably, so "
              "sweeps will be FLAT and weighting ≈ position-based. Adjustments to "
              "cross the threshold:")
        print(f"    · shrink the sweep fleet to n ≤ n\\*−1 = {max(1, n_star - 1)} "
              "(the battery-limited side), or")
        print("    · reduce battery capacity, or enlarge the area / lower the swath "
              "so E_cover rises, or")
        print("    · enable the camera payload (sensor_power_w > 0) to raise coverage "
              "energy.")
    else:
        print(f"- 🟨 **BORDERLINE at n={n_drones}** (ratio {ratio:.2f}). The sweep "
              f"should center on n\\*={n_star}; verify a couple of points each side "
              "cross the boundary before committing.")
    print(f"\n_n\\* = {n_star}. The /N sortie overhead makes the real crossover a "
          "touch higher; the fleet-sizing analyzer's per-sortie model refines it._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
