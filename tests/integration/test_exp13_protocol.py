"""EXP-13 end-to-end acceptance for the M4E protocol config (config/djimatrice4e.yaml).

Small, reproducible checks that the protocol config actually runs end-to-end and
that its load-bearing mechanisms fire at EXECUTION time (not merely that the
config validates). Scope is acceptance, NOT an experiment: no matrix, no stats.

What the paired e2e does NOT check (recorded per AC-2):
  * reallocation of released work — OFF by author decision (§4.1 coherent routing);
    only work-release ACCOUNTING is asserted, so "no reallocation" cannot be
    mistaken for "work vanished".
  * the ~small acceptance box does NOT validate the 1000x750 protocol scenario;
    the 3/5/8 --smoke run of run_lloyd_protocol covers that separately.
"""
from __future__ import annotations

import json
import math

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.execution.state_machine import AgentContext, StateMachine
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import (
    AgentState as S, BatteryZone, DecompositionAlgo, ManeuverType as M, Outcome,
)
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.run_output import build_mission_contract
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.experiments.run_lloyd_protocol import build_manifest

M4E = "config/djimatrice4e.yaml"

# A small, battery-limited acceptance scenario: a box the fleet cannot fully
# cover on one charge under no-swap, so at least one drone retires with unfinished
# cells and releases work (exercising the accounting AC-2/(a) cares about).
_ACCEPT = {
    "fleet.n_drones": 2,
    # Battery sized (probed, not tuned-to-green) so BOTH drones reliably reach the
    # no-swap terminal landing with unfinished cells: coverage ~0.96 (< 1) and
    # work_releases is non-empty, which is exactly what AC-2/(a) must exercise.
    "fleet.battery_capacity_wh": 6.0,
    "launch.candidate_sites": [[0, 0]],
    # Obstacle-free acceptance area: coherent OBSTACLE routing is exercised by
    # test_exp09_routes_energy (blocking box); here we keep the Lloyd partition on
    # a clean field so the paired arms run deterministically end-to-end.
    "env.obstacle_generation_mode": "poisson", "env.obstacle_density_per_km2": 0.0,
    "sim.dt_s": 0.5, "sim.max_timesteps": 6000,
}


def _accept_overrides(area) -> dict:
    return dict(_ACCEPT, **{"env.geojson_path": str(area)})


@pytest.fixture
def area(tmp_path):
    p = tmp_path / "accept.geojson"
    p.write_text(json.dumps({"type": "Feature", "properties": {},
                             "geometry": mapping(box(0, 0, 300, 200))}))
    return p


def _run_arm(area, algo: DecompositionAlgo):
    cfg = load_config(M4E, overrides=_accept_overrides(area))
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0, algo=algo)
    res = eng.run()
    manifest = build_manifest(eng, cfg)
    contract = build_mission_contract(
        res, capacity_j=cfg.fleet.battery_capacity_j,
        decomposer_class=type(eng.decomposer).__name__)
    return eng, res, manifest, contract


# --------------------------------------------------------------------------- #
# AC-2: one small reproducible PAIRED acceptance case                          #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_paired_acceptance_touches_cameras_masks_noswap_energy_and_output_identity(area):
    eng_cvt, cvt, man_cvt, con_cvt = _run_arm(area, DecompositionAlgo.LLOYD_CVT)
    eng_en, en, man_en, con_en = _run_arm(area, DecompositionAlgo.LLOYD_ENERGY)

    # -- cameras: photogrammetry ON => photo events recorded on both arms
    assert cvt.photo_events and en.photo_events

    # -- no-swap lifecycle: touchdown is terminal, no swaps, survivors landed
    def _agents(eng):
        a = eng.fleet.agents
        return list(a.values()) if isinstance(a, dict) else list(a)
    for eng, res in ((eng_cvt, cvt), (eng_en, en)):
        assert res.metrics.n_swaps == 0
        states = {a.state for a in _agents(eng)}
        assert states <= {S.S_LANDED, S.S_FAIL}, states
        assert not res.airborne_at_end
        assert res.outcome in (Outcome.MISSION_SUCCESS, Outcome.MISSION_PARTIAL,
                               Outcome.MISSION_FAILED)

    # -- work-release accounting (AC-2/(a)): released work is recorded, not vanished.
    #    coverage < 1 must be explained by released work or a loss.
    for res in (cvt, en):
        assert isinstance(res.retired_agents, tuple)
        released_ids = {aid for aid, _ in res.work_releases}
        assert released_ids <= set(res.retired_agents)   # a release implies a retirement
        assert 0.0 <= res.coverage_frac < 1.0            # a drone landed with unfinished cells
        # released work is ACCOUNTED, not vanished: coverage < 1 is explained by a release.
        assert res.work_releases, "coverage < 1 but no work released -> work vanished"

    # -- energy reconciliation (#8): under no-swap the EXP-11 contract reconciles
    #    with an exact clamp residual (run_output.py:442-458).
    for con in (con_cvt, con_en):
        rec = con["energy"]["reconciliation"]
        assert rec["reconcilable"] is True
        assert rec["n_swaps_total"] == 0
        assert rec["clamp_residual_j"] is not None
        assert math.isfinite(rec["clamp_residual_j"])

    # -- paired OUTPUT IDENTITY: the two arms share every physical input (the
    #    algorithm is excluded from the fingerprint).
    assert man_cvt["fingerprint_sha256"] == man_en["fingerprint_sha256"]


# --------------------------------------------------------------------------- #
# AC-4 / condition (c): the static 0.40 CRITICAL guard is removed at RUNTIME    #
# --------------------------------------------------------------------------- #
def test_zone_demotion_removes_static_critical_guard_at_decision_time():
    """The decision code (StateMachine.step, called every tick by Agent.step) is
    exercised directly. With the protocol's zone_demotion=True the CRITICAL net is
    gone; the control (zone_demotion=False) still fires it — proving the removal is
    behavioural, not just a config that validates."""
    cfg = load_config(M4E)
    assert cfg.rth.execution_coherent is True and cfg.rth.energy_map.enabled is False
    zones = cfg.battery_zones

    demoted = StateMachine(zones, zone_demotion=True, no_swap_mode=True)
    control = StateMachine(zones, zone_demotion=False, no_swap_mode=True)

    ctx = AgentContext(S.S2_MISSION, BatteryZone.CRITICAL)
    # control: the static net force-returns at the 0.40 CRITICAL boundary
    t_ctrl = control.step(ctx)
    assert t_ctrl is not None and t_ctrl.reason == "critical_battery"
    # protocol: no static critical return — the coherent/dynamic path governs
    assert demoted.step(ctx) is None


# --------------------------------------------------------------------------- #
# #6 / condition (d): the camera-energy branch is live and charges COVERAGE only#
# --------------------------------------------------------------------------- #
def test_camera_energy_charged_over_coverage_only():
    """sensor_power_w=20 makes the previously-dead camera branch execute: the
    payload energy is charged over COVERAGE segment duration only, zero elsewhere."""
    cfg = load_config(M4E)
    spec = build_spec(cfg)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    from uav_swarm_sim.infrastructure.core_types import Pose

    assert spec.photogrammetry is not None
    assert cfg.sensor.sensor_power_w == 20.0

    cover = motion.plan(Pose(0, 0, 0), Pose(100, 0, 0), M.COVERAGE)
    cover_dur = sum(s.duration_s for s in cover.segments if s.maneuver is M.COVERAGE)
    assert cover_dur > 0.0
    # camera term over the coverage leg = sensor_power_w * coverage duration
    assert em.sensor_energy(cover_dur, cfg.sensor.sensor_power_w) == pytest.approx(
        cfg.sensor.sensor_power_w * cover_dur)
    assert em.sensor_energy(cover_dur, cfg.sensor.sensor_power_w) > 0.0
    # a non-coverage (ferry) leg draws zero camera energy
    ferry = motion.plan(Pose(0, 0, 0), Pose(100, 0, 0), M.CRUISE)
    ferry_cov = sum(s.duration_s for s in ferry.segments if s.maneuver is M.COVERAGE)
    assert em.sensor_energy(ferry_cov, cfg.sensor.sensor_power_w) == 0.0


# --------------------------------------------------------------------------- #
# AC-4: experiment_mode blocks tier auto-selection (no alternate tier)          #
# --------------------------------------------------------------------------- #
def test_experiment_mode_blocks_tier_auto_selection(area):
    """With mission.experiment_mode=true a run MUST name its algorithm; the engine
    refuses to auto-select a tier decomposer (simulation_engine.py:218)."""
    cfg = load_config(M4E, overrides=_accept_overrides(area))
    assert cfg.mission.experiment_mode is True
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), algo=None)
    with pytest.raises(ValueError, match="experiment_mode"):
        eng.run()
