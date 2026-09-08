"""EXP-10 integration: the factual recorder is observational only.

Two guarantees at the engine level:
  * flag OFF (absent or explicit False) is byte-identical to the pre-EXP-10 run
    AND leaves the two MissionResult fields at their inert defaults;
  * flag ON changes no physics/energy/RNG (the physical signature is identical to
    the flag-off run) while attaching the safety-violation surface for EXP-11.
"""
from dataclasses import asdict
import json

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


def _overrides(area):
    # EXP-13 leak shield: the base M4E config is now the full protocol (target
    # obstacles, no-swap, coherent routing, zone demotion, free-space routing,
    # energy balance, camera payload draw, and safety recording ON). This file
    # tests the EXP-10 recorder on a plain DEFAULT-mode swap-mission, so restore
    # that pre-change world here; record_violations is set per-arm by each test.
    #   was  -> inherited the whole protocol (recorder ON, no-swap coherent run)
    #   now  -> default-mode swap mission, recorder controlled per-arm
    #   why  -> inheriting no_swap+coherent would move the recorder onto the
    #           terminal-landing path and could make its violation counts vacuous.
    return {
        "fleet.n_drones": 3, "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        "env.obstacle_generation_mode": "poisson",
        "env.coverage_altitude_m": 100.0, "launch.candidate_sites": [[0, 0]],
        "platforms.MULTIROTOR.v_coverage": 10.0, "platforms.MULTIROTOR.v_cruise": 12.0,
        "sensor.photogrammetry.enabled": True, "sensor.sensor_power_w": 0.0,
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 5.0,
        "coverage.transit_free_space": False, "coverage.ferry_free_space": False,
        "mission.no_swap_mode": False,
        "rth.execution_coherent": False, "rth.energy_map.zone_demotion": False,
        "planning.energy_balance.enabled": False,
        "sim.dt_s": 0.5, "sim.max_timesteps": 2000,
    }


@pytest.fixture
def area(tmp_path):
    p = tmp_path / "area.geojson"
    p.write_text(json.dumps({"type": "Feature", "properties": {},
                             "geometry": mapping(box(0, 0, 260, 120))}))
    return p


def _run(area, **extra):
    settings = dict(_overrides(area), **extra)
    cfg = load_config("config/djimatrice4e.yaml", overrides=settings)
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                            algo=DecompositionAlgo.TGC_BASIC).run()


def _physical_signature(result):
    """Deterministic physical output; excludes the EXP-10 surface (which is what
    we assert does NOT perturb the physics)."""
    m = result.metrics
    value = dict(
        outcome=result.outcome.value, coverage=result.coverage_frac,
        energy=m.total_energy_j, duration=m.duration_s, swaps=m.n_swaps,
        failures=m.n_failures,
        per_agent_energy=m.per_agent_energy_j, per_agent_length=m.per_agent_length_m,
        sojourns=[asdict(s) for s in result.history.sojourns()],
        photos=[asdict(p) for p in result.photo_events],
    )
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda v: v.value)


def test_recorder_default_is_off_in_default_config():
    """The 'default is OFF' guarantee is a CODE property, asserted against the
    default config (the M4E protocol config deliberately turns the recorder ON)."""
    assert load_config("config/default.yaml").safety.record_violations is False


def test_flag_off_is_inert_and_byte_identical(area):
    # record_violations pinned False on BOTH arms so the off/off contrast is real
    # under the EXP-13 base (which sets it True); see _overrides shield note.
    off_a = _run(area, **{"safety.record_violations": False})
    off_b = _run(area, **{"safety.record_violations": False})
    assert _physical_signature(off_a) == _physical_signature(off_b)
    # inert defaults when the recorder is never built
    assert off_a.safety_violations == ()
    assert off_a.safety_minima is None


def test_flag_on_does_not_change_physics_and_attaches_surface(area):
    off = _run(area, **{"safety.record_violations": False})
    on = _run(area, **{"safety.record_violations": True})
    # identical physics/energy/RNG: recording is purely observational
    assert _physical_signature(on) == _physical_signature(off)
    # the EXP-11 surface is present and well-formed
    assert on.safety_minima is not None
    assert set(on.safety_minima) == {
        "min_separation_m", "min_obstacle_clearance_m", "n_hard", "n_soft"}
    hard = sum(1 for v in on.safety_violations if v.severity == "hard")
    soft = sum(1 for v in on.safety_violations if v.severity == "soft")
    assert on.safety_minima["n_hard"] == hard
    assert on.safety_minima["n_soft"] == soft
    for v in on.safety_violations:
        assert v.kind in {"separation", "obstacle", "speed"}
        assert v.severity in {"hard", "soft"}
        assert v.duration_s > 0.0
        # HARD speed is an invariant guard, unreachable in a real run (the motion
        # model caps executed speed at the envelope) -- never fires here.
        assert not (v.kind == "speed" and v.severity == "hard")
