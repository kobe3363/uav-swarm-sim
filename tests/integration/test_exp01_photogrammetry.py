"""EXP-01 mission regressions: photo events observe but do not alter physics."""
from __future__ import annotations

import hashlib

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Event
from uav_swarm_sim.infrastructure.enums import (
    DecompositionAlgo,
    EventType,
    PlannerKind,
)
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


def _overrides(*, enabled=True, forward=0.80, dt=0.5):
    return {
        "fleet.n_drones": 1,
        "fleet.battery_capacity_wh": 1000.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "sim.dt_s": dt,
        "sim.max_timesteps": 10000,
        "sensor.photogrammetry.enabled": enabled,
        "sensor.photogrammetry.sensor_width_mm": 17.3,
        "sensor.photogrammetry.sensor_height_mm": 13.0,
        "sensor.photogrammetry.focal_length_mm": 12.0,
        "sensor.photogrammetry.image_width_px": 5280,
        "sensor.photogrammetry.image_height_px": 3956,
        "sensor.photogrammetry.side_overlap": 0.70,
        "sensor.photogrammetry.forward_overlap": forward,
        "sensor.photogrammetry.min_photo_interval_s": 0.5,
    }


def _run(config_path, *, profile=True, **kwargs):
    overrides = _overrides(**kwargs)
    if not profile:
        overrides = {
            key: value for key, value in overrides.items()
            if not key.startswith("sensor.photogrammetry.")
        }
    cfg = load_config(config_path, overrides=overrides)
    engine = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), 0,
        algo=DecompositionAlgo.WEIGHTED_VORONOI,
    )
    return engine, engine.run()


def _physics_signature(engine, result):
    metrics = result.metrics
    plans = {
        aid: (plan.waypoints, plan.length_m, plan.est_energy_j, plan.connectors)
        for aid, plan in engine.plans.items()
    }
    return (
        result.outcome, result.aborted, result.coverage_frac,
        metrics.total_energy_j, metrics.duration_s, metrics.workload_std_m,
        metrics.per_agent_length_m, metrics.per_agent_energy_j,
        metrics.n_swaps, metrics.n_failures, plans, result.history.sojourns(),
    )


def _cadence_signature(result):
    return [
        (event.agent_id, event.coverage_leg_index, event.distance_on_strip_m)
        for event in result.photo_events
    ]


def test_forward_overlap_changes_photos_but_not_path_or_energy(config_path):
    lower_engine, lower = _run(config_path, forward=0.70)
    higher_engine, higher = _run(config_path, forward=0.80)

    assert len(higher.photo_events) > len(lower.photo_events)
    assert _physics_signature(higher_engine, higher) == _physics_signature(lower_engine, lower)
    assert all(event.coverage_leg_index % 2 == 0 for event in higher.photo_events)


def test_photo_count_and_distance_cadence_are_invariant_to_dt(config_path):
    _coarse_engine, coarse = _run(config_path, dt=1.0)
    _fine_engine, fine = _run(config_path, dt=0.25)

    # MotionModel's off-path convergence is itself dt-sensitive.  EXP-01 does
    # not change that physics; the shutter guarantee is invariant event count
    # and travelled-distance cadence for the resulting physical trajectory.
    assert _cadence_signature(fine) == _cadence_signature(coarse)


def test_initial_transit_and_first_photo_use_first_strip_start(config_path):
    engine, result = _run(config_path)
    zone = engine.partition.zones[0]
    first_strip_start = engine.plans[0].waypoints[0].pose

    assert zone.entry_pose.as_xyz() != first_strip_start.as_xyz()
    assert engine.fleet.agents[0]._transit.end_pose.as_xyz() == pytest.approx(
        first_strip_start.as_xyz(), abs=1e-12
    )
    assert result.photo_events[0].pose.as_xyz() == pytest.approx(
        first_strip_start.as_xyz(), abs=1e-12
    )


def test_redistribution_transit_uses_replanned_first_strip_start(config_path):
    cfg = load_config(config_path, overrides=_overrides())
    engine = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), 0,
        algo=DecompositionAlgo.WEIGHTED_VORONOI,
    )
    engine._build()
    prior_zone = engine.partition.zones[0]
    engine._redistribute(
        Event(EventType.NEW_TASK, 0.0, {"polygon": prior_zone.polygon}), 0.0
    )

    zone = engine.partition.zones[0]
    first_strip_start = engine.plans[0].waypoints[0].pose
    assert zone.entry_pose.as_xyz() != first_strip_start.as_xyz()
    assert engine.fleet.agents[0]._transit.end_pose.as_xyz() == pytest.approx(
        first_strip_start.as_xyz(), abs=1e-12
    )


def test_disabled_profile_produces_no_events_and_legacy_swath(config_path):
    disabled_engine, disabled = _run(config_path, enabled=False)
    legacy_engine, legacy = _run(config_path, profile=False)
    expected_swath = (
        legacy_engine.cfg.sensor.swath_width_m
        * (1.0 - legacy_engine.cfg.sensor.overlap_frac)
    )
    assert disabled_engine.spec.swath_width_m == expected_swath
    assert disabled.photo_events == legacy.photo_events == ()
    assert _physics_signature(disabled_engine, disabled) == _physics_signature(
        legacy_engine, legacy
    )


def test_enabled_photogrammetry_rejects_fixed_cell_grid_planner(config_path):
    cfg = load_config(config_path, overrides=_overrides())
    engine = SimulationEngine(cfg, RngFactory(1), planner=PlannerKind.GRID)
    with pytest.raises(ValueError, match="photogrammetry.*GRID"):
        engine._build()


# --------------------------------------------------------------------------- #
# Flag-off byte identity (CLAUDE.md rule 3).                                   #
#                                                                              #
# The tests above only prove "explicit-off == profile-absent" inside a single  #
# binary; both arms move together if EXP-01 perturbed the legacy path. This    #
# golden is the cross-commit half: the signature was captured on d6bd6fa, the  #
# commit immediately preceding EXP-01, using config/default.yaml unchanged and #
# no photogrammetry keys at all -- an explicit-off override is deliberately    #
# NOT the baseline, because it materialises a PhotogrammetryConfig that does   #
# not exist pre-change.                                                        #
#                                                                              #
# Regeneration (only if a deliberate, documented physics change lands):        #
#   git worktree add <tmp> <pre-change commit>; run _default_cfg + _engine     #
#   there with this same _canonical_signature_string and re-pin the hash.      #
# --------------------------------------------------------------------------- #
_PRE_EXP01_COMMIT = "d6bd6fa"
_PRE_EXP01_GOLDEN_HASH = (
    "ce9fc17176d5d55f5264f265392d55319a98ec5032e2a1360e6b8d44ea525b43"
)
_PRE_EXP01_N_SOJOURNS = 22
_PRE_EXP01_DURATION_S = 266.0
_PRE_EXP01_TOTAL_ENERGY_J = 110988.0


def _default_cfg(config_path):
    """config/default.yaml (untouched by EXP-01) shrunk to a tiny mission.

    Mirrors test_energy_map_stage4._tiny_cfg so the two cross-commit goldens
    describe the same reference run.
    """
    return load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })


def _canonical_signature_string(res) -> str:
    """Full serialised run signature; string-based so the hash is not coupled
    to dataclass ``__repr__`` churn."""
    m = res.metrics
    parts = [
        f"total_energy_j={m.total_energy_j!r}",
        f"duration_s={m.duration_s!r}",
        f"workload_std_m={m.workload_std_m!r}",
        f"n_swaps={m.n_swaps!r}",
        "per_agent_energy_j=" + ",".join(
            f"{k}:{v!r}" for k, v in sorted(m.per_agent_energy_j.items())),
        "per_agent_length_m=" + ",".join(
            f"{k}:{v!r}" for k, v in sorted(m.per_agent_length_m.items())),
        "sojourns=" + ";".join(
            f"{s.agent_id}|{s.state.value}|{s.t_in!r}|{s.t_out!r}|{s.reason_out}"
            for s in res.history.sojourns()),
        f"outcome={res.outcome.value}",
        f"coverage_frac={res.coverage_frac!r}",
        f"stalled_agents={getattr(res, 'stalled_agents', ())!r}",
        f"skipped_legs={getattr(res, 'skipped_legs', ())!r}",
    ]
    return "\n".join(parts)


def test_default_config_run_byte_identical_to_pre_exp01(config_path):
    """Flag-off (photogrammetry absent) must reproduce the pre-EXP-01 run
    exactly: summary metrics, the whole FSM sojourn trajectory, outcome,
    coverage and the stalled/skipped sets."""
    cfg = _default_cfg(config_path)
    assert cfg.sensor.photogrammetry.enabled is False
    engine = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), replication=0,
        algo=DecompositionAlgo.TGC_BASIC, planner=PlannerKind.DUBINS,
    )
    res = engine.run()
    # No camera profile reaches the spec, so every EXP-01 branch is dormant.
    assert engine.spec.photogrammetry is None

    # Human-readable anchors first: a mismatch here names the drift directly.
    assert len(res.history.sojourns()) == _PRE_EXP01_N_SOJOURNS
    assert res.metrics.duration_s == _PRE_EXP01_DURATION_S
    assert res.metrics.total_energy_j == _PRE_EXP01_TOTAL_ENERGY_J
    digest = hashlib.sha256(
        _canonical_signature_string(res).encode()
    ).hexdigest()
    assert digest == _PRE_EXP01_GOLDEN_HASH, (
        f"flag-off run drifted from {_PRE_EXP01_COMMIT}"
    )
