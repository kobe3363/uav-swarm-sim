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
    return {
        "fleet.n_drones": 3, "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": str(area), "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0, "launch.candidate_sites": [[0, 0]],
        "platforms.MULTIROTOR.v_coverage": 10.0, "platforms.MULTIROTOR.v_cruise": 12.0,
        "sensor.photogrammetry.enabled": True,
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 5.0,
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


def test_flag_off_is_default_and_byte_identical(area):
    absent = _run(area)
    explicit = _run(area, **{"safety.record_violations": False})
    assert _physical_signature(absent) == _physical_signature(explicit)
    # inert defaults when the recorder is never built
    assert absent.safety_violations == ()
    assert absent.safety_minima is None


def test_flag_on_does_not_change_physics_and_attaches_surface(area):
    off = _run(area)
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
