"""EXP-08 at the engine boundary: identical rules, the run's own decomposer.

The deliverable is RULE parity, not identical event times -- different
trajectories legitimately fire the same rule at different moments (D-12). What
must hold for every algorithm is: one trigger set, one eligibility rule, one
ordering, the run's own decomposer, and no covered cell ever re-issued.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    DecompositionAlgo,
    ManeuverType,
    Outcome,
    PlannerKind,
)
from uav_swarm_sim.infrastructure.rng import (
    STREAM_KMEANS_INIT,
    STREAM_REPARTITION_INIT,
    RngFactory,
)
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.run_output import build_results_single

_PHOTO = {
    "sensor.photogrammetry.enabled": True,
    "sensor.photogrammetry.sensor_width_mm": 8.0,
    "sensor.photogrammetry.sensor_height_mm": 6.0,
    "sensor.photogrammetry.focal_length_mm": 10.0,
    "sensor.photogrammetry.image_width_px": 4000,
    "sensor.photogrammetry.image_height_px": 3000,
    "sensor.photogrammetry.side_overlap": 0.5,
    "sensor.photogrammetry.forward_overlap": 0.5,
    "sensor.photogrammetry.min_photo_interval_s": 0.5,
}


@pytest.fixture(scope="module")
def area_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("exp08") / "rect.geojson"
    path.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(box(0, 0, 600, 240))}))
    return str(path)


def _overrides(area_file, **extra):
    base = {
        "fleet.n_drones": 3, "fleet.battery_capacity_wh": 40.0,
        "fleet.total_reserve_batteries": 0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": area_file, "env.obstacle_density_per_km2": 0.0,
        "env.coverage_altitude_m": 100.0,
        "launch.candidate_sites": [[300.0, 0.0]],
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "platforms.MULTIROTOR.r_min_m": 0.0, "platforms.MULTIROTOR.omega_max": 1.0,
        "sensor.sensor_power_w": 15.0,
        **_PHOTO,
        "coverage.raster_enabled": True, "coverage.raster_cell_m": 10.0,
        "mission.no_swap_mode": True,
        "sim.dt_s": 0.5, "sim.max_timesteps": 40000,
        "battery.initial_soc.mode": "uniform",
        "battery.initial_soc.low": 0.45, "battery.initial_soc.high": 0.95,
        "planning.energy_balance.enabled": True,
    }
    base.update(extra)
    return base


def _run(area_file, algo, **extra):
    cfg = load_config("config/default.yaml", overrides=_overrides(area_file, **extra))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0, algo=algo)
    return engine, engine.run()


def _applied(result):
    return [r for r in result.repartitions if r["applied"]]


# --------------------------------------------------------------------------- #
# A. the feature does what it exists to do                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", [DecompositionAlgo.LLOYD_CVT,
                                  DecompositionAlgo.LLOYD_ENERGY])
def test_released_work_is_picked_up_instead_of_being_stranded(area_file, algo):
    """A finite-battery fleet strands the work of every drone that lands early
    (D-11). With the flag on a survivor takes it over. Both arms, one rule.

    35 Wh on purpose. This is NOT a claim that re-partitioning always helps: in a
    battery-starved regime (measured here at 25 and 30 Wh) the ferry to a
    re-partitioned zone can cost more than the coverage it buys, and the flag-on
    run finishes BELOW the flag-off one. That is a real property of the
    mechanism, and quantifying it is the main experiment's job -- so the fixture
    is pinned to a regime where work is genuinely stranded AND reachable, rather
    than the assertion being loosened until every regime passes.
    """
    small = {"fleet.battery_capacity_wh": 35.0}
    _, off = _run(area_file, algo, **small, **{"mission.repartition_enabled": False})
    _, on = _run(area_file, algo, **small, **{"mission.repartition_enabled": True})

    assert off.repartitions == () and off.repartition_hold is None
    assert off.outcome is Outcome.MISSION_PARTIAL
    assert off.coverage_frac < 1.0

    assert on.coverage_frac > off.coverage_frac
    assert _applied(on), "at least one revision must have been applied"


@pytest.mark.parametrize("algo,expected", [
    (DecompositionAlgo.LLOYD_CVT, "LloydCvtDecomposer"),
    (DecompositionAlgo.LLOYD_ENERGY, "LloydEnergyDecomposer"),
    (DecompositionAlgo.TGC_BASIC, "TgcBasicDecomposer"),
    (DecompositionAlgo.WEIGHTED_VORONOI, "WeightedTgcDecomposer"),
    (DecompositionAlgo.CLASSIC_VORONOI, "ClassicVoronoiDecomposer"),
    (DecompositionAlgo.KMEANS, "KMeansHeuristicDecomposer"),
])
def test_every_arm_re_partitions_with_its_own_decomposer(area_file, algo, expected):
    """D-12: CVT stays CVT, ENERGY stays ENERGY -- and so does every legacy peer.
    A new-mode run must never fall back to WeightedTgcDecomposer, which today is
    what kmeans, classic_voronoi and both Lloyd arms silently get."""
    engine, result = _run(area_file, algo, **{
        "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 30.0,     # force a revision deterministically
    })
    assert engine.redistributor is None, "the legacy path must not even exist"
    assert type(engine.repartitioner.decomposer).__name__ == expected
    applied = _applied(result)
    assert applied, "the interval must have produced at least one revision"
    for record in applied:
        assert record["decomposer_class"] == expected
        assert record["algorithm"] == algo.value


def test_the_arms_may_fire_the_same_rule_at_different_times(area_file):
    """Rule parity is the deliverable; identical event times are NOT. Different
    trajectories legitimately reach the same trigger at different moments."""
    _, cvt = _run(area_file, DecompositionAlgo.LLOYD_CVT,
                  **{"mission.repartition_enabled": True})
    _, energy = _run(area_file, DecompositionAlgo.LLOYD_ENERGY,
                     **{"mission.repartition_enabled": True})

    causes = {r["reason"] for r in cvt.repartitions} | {
        r["reason"] for r in energy.repartitions}
    assert causes <= {"applied", "no_eligible_executor", "no_remaining_work",
                      "no_progress"}
    # both ran the same machinery; nothing here asserts the times agree
    assert _applied(cvt) and _applied(energy)


# --------------------------------------------------------------------------- #
# B. conservation and non-destructiveness, end to end                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", [DecompositionAlgo.LLOYD_CVT,
                                  DecompositionAlgo.LLOYD_ENERGY])
def test_no_cell_is_lost_or_duplicated_across_every_revision(area_file, algo):
    """The engine's four guards raise on violation, so a clean run is itself the
    assertion. What is checked here is that they were EXERCISED and that the
    reported bookkeeping adds up on every applied revision."""
    _, result = _run(area_file, algo, **{
        "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 20.0,
    })
    applied = _applied(result)
    assert len(applied) >= 2, "want several consecutive revisions"
    for record in applied:
        assert record["cells_assigned"] + record["cells_unassigned"] \
            == record["cells_before"]
        assert record["area_assigned_m2"] <= record["area_before_m2"] + 1e-6
    # the pool only ever shrinks: coverage is monotone and never returns to it
    befores = [r["cells_before"] for r in applied]
    assert befores == sorted(befores, reverse=True)


def test_a_revision_erases_no_coverage_and_no_energy(area_file):
    """A plan swap changes what a drone will do next, never what it has done."""
    cfg = load_config("config/default.yaml", overrides=_overrides(
        area_file, **{"mission.repartition_enabled": True,
                      "mission.repartition_interval_s": 20.0}))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                              algo=DecompositionAlgo.LLOYD_CVT)
    seen = []
    original = engine._run_repartition

    def probe(t, causes):
        before = (
            engine.coverage_raster.plannable_covered_area_m2,
            {a.id: (a.energy_consumed_j, a.flown_m, a.battery.level_j, a.pose)
             for a in engine.fleet.agents.values()},
        )
        original(t, causes)
        after = (
            engine.coverage_raster.plannable_covered_area_m2,
            {a.id: (a.energy_consumed_j, a.flown_m, a.battery.level_j, a.pose)
             for a in engine.fleet.agents.values()},
        )
        seen.append((before, after))

    engine._run_repartition = probe
    engine.run()

    assert seen, "no revision was attempted"
    for before, after in seen:
        assert before == after


def test_a_drone_committed_to_a_return_is_never_handed_work(area_file):
    cfg = load_config("config/default.yaml", overrides=_overrides(
        area_file, **{"mission.repartition_enabled": True,
                      "mission.repartition_interval_s": 15.0}))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                              algo=DecompositionAlgo.LLOYD_CVT)
    offences = []
    original = engine._run_repartition

    def probe(t, causes):
        # Identity, not the serialized value: if AgentState.S3_RTH.value ever
        # changed, a string comparison would leave `returning` empty and this
        # probe would pass while checking nothing. (Found in review of this PR.)
        returning = {a.id for a in engine.fleet.agents.values()
                     if a.state is AgentState.S3_RTH}
        snapshot = {a.id: (a.plan, a._cov_idx, tuple(a._legs))
                    for a in engine.fleet.agents.values() if a.id in returning}
        original(t, causes)
        for aid, before in snapshot.items():
            agent = engine.fleet.agents[aid]
            if (agent.plan, agent._cov_idx, tuple(agent._legs)) != before:
                offences.append((t, aid))
        if returning:
            offences.extend(
                (t, aid) for aid in returning
                for r in engine._repartition_records[-1:]
                if aid in r.executors
            )

    engine._run_repartition = probe
    result = engine.run()
    assert result.repartitions, "no revision was attempted"
    assert offences == []


# --------------------------------------------------------------------------- #
# C. ordering, refusals and the loop bound                                     #
# --------------------------------------------------------------------------- #
def test_simultaneous_causes_produce_exactly_one_revision(area_file):
    cfg = load_config("config/default.yaml", overrides=_overrides(
        area_file, **{"mission.repartition_enabled": True,
                      "mission.repartition_interval_s": 20.0}))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                              algo=DecompositionAlgo.LLOYD_CVT)
    result = engine.run()

    per_tick: dict[float, int] = {}
    for record in result.repartitions:
        per_tick[record["t_s"]] = per_tick.get(record["t_s"], 0) + 1
    assert per_tick and max(per_tick.values()) == 1, (
        "a tick may produce at most one revision, however many causes fired")
    # and every cause that fired is named in the record it belongs to
    for record in result.repartitions:
        assert record["causes"], "a revision must say what caused it"
        for name, _aid in record["causes"]:
            assert name in {"failure", "uav_retired", "zone_complete", "interval"}


def test_repeated_triggers_without_progress_do_not_loop(area_file):
    """The interval keeps firing; the guard must let exactly one revision through
    per real change and refuse the rest, loudly."""
    _, result = _run(area_file, DecompositionAlgo.LLOYD_CVT, **{
        "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 5.0,          # very aggressive
    })
    refused = [r for r in result.repartitions if r["reason"] == "no_progress"]
    assert refused, "the aggressive cadence must have been refused at least once"
    assert len(_applied(result)) < len(result.repartitions)
    for record in refused:
        assert record["applied"] is False
        assert record["cells_unassigned"] == record["cells_before"]


def test_the_periodic_trigger_fires_mid_leg_and_on_whole_ticks(area_file):
    _, result = _run(area_file, DecompositionAlgo.LLOYD_CVT, **{
        "mission.repartition_enabled": True,
        "sim.dt_s": 0.5, "mission.repartition_interval_s": 10.0,
    })
    interval_records = [r for r in result.repartitions
                        if any(c[0] == "interval" for c in r["causes"])]
    assert interval_records
    for record in interval_records:
        # 10 s at dt 0.5 s = every 20th step, so every hit is a multiple of 10 s
        assert record["t_s"] % 10.0 == pytest.approx(0.0, abs=1e-9)


def test_an_injected_task_is_refused_rather_than_dropped(area_file):
    """The raster is built once from the survey area and cannot grow, so an
    injected polygon has no cells. Accepting it would mean taking work and never
    flying it."""
    cfg = load_config("config/default.yaml", overrides=_overrides(
        area_file, **{"mission.repartition_enabled": True}))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                              algo=DecompositionAlgo.LLOYD_CVT)
    engine.inject_task(box(700.0, 0.0, 800.0, 100.0), at_time_s=5.0)
    with pytest.raises(NotImplementedError, match="cannot accept a NEW_TASK"):
        engine.run()


# --------------------------------------------------------------------------- #
# D. what the hold costs (author requirement: visible, not assumed negligible)  #
# --------------------------------------------------------------------------- #
def test_the_hold_cost_is_reported_and_is_a_rounding_error(area_file):
    """One hover tick per zone completion. It is applied by one rule to both
    arms but is NOT paired -- the arms can finish a different NUMBER of zones --
    so the size of the asymmetry is reported rather than assumed away."""
    cfg = load_config("config/default.yaml", overrides=_overrides(
        area_file, **{"mission.repartition_enabled": True}))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                              algo=DecompositionAlgo.LLOYD_CVT)
    result = engine.run()

    hold = result.repartition_hold
    assert hold is not None and hold["ticks"] >= 1
    # expected value from the power table and dt, not read back from the code
    hover_w = engine.spec.power_w[ManeuverType.HOVER]
    assert hover_w > 0.0
    assert hold["energy_j"] == pytest.approx(hold["ticks"] * hover_w * cfg.sim.dt_s,
                                             rel=1e-12)
    # and it is negligible against a battery, which is the point of reporting it
    assert hold["frac_of_one_battery"] < 0.01
    assert sum(hold["per_agent_j"].values()) == pytest.approx(hold["energy_j"])


# --------------------------------------------------------------------------- #
# E. reporting                                                                 #
# --------------------------------------------------------------------------- #
def test_every_attempt_reaches_results_json_and_is_strict_json(area_file):
    _, result = _run(area_file, DecompositionAlgo.LLOYD_ENERGY, **{
        "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 20.0,
    })
    out = build_results_single(result, SimpleNamespace(ergodic=False),
                               identity={}, wall_time_s=0.0)
    assert len(out["repartitions"]) == len(result.repartitions)
    assert out["repartition_hold"]["ticks"] >= 0
    json.dumps(out, allow_nan=False)


def test_a_legacy_run_carries_no_repartition_key(area_file):
    _, result = _run(area_file, DecompositionAlgo.LLOYD_CVT,
                     **{"mission.repartition_enabled": False})
    out = build_results_single(result, SimpleNamespace(ergodic=False),
                               identity={}, wall_time_s=0.0)
    assert "repartitions" not in out
    assert "repartition_hold" not in out


# --------------------------------------------------------------------------- #
# F. flag-off byte identity for ALL FOUR legacy algorithms                      #
# --------------------------------------------------------------------------- #
# Captured on the unmodified base commit 52e433e (EXP-07b merge, PR #69) with the
# scratch equivalent of _legacy_signature below. The config forces hazard
# failures so the legacy Redistributor path -- the one EXP-08 must leave
# untouched -- actually runs; every algorithm asserts a non-empty replan list, so
# the golden cannot pass vacuously.
#
# tgc_basic and weighted_voronoi already re-use their own decomposer today
# (TgcBasicDecomposer subclasses WeightedTgcDecomposer). classic_voronoi and
# kmeans are the two that get SUBSTITUTED, and they are the reason removing the
# isinstance branch is gated on the flag rather than done unconditionally.
# Regenerate ONLY from a pre-EXP-08 commit.
_LEGACY_GOLDEN_SHA256 = "c90ca5e83bde52c28ded46024c721b74189042f498a935cbeb0b143627293c68"

_LEGACY_OVERRIDES = {
    "fleet.n_drones": 4,
    "fleet.battery_capacity_wh": 30.0,
    "fleet.total_reserve_batteries": 50,
    "failure.hazard_rate_per_hour": 900.0,
    "env.geojson_path": "data/areas/smoke_area.geojson",
    "env.obstacle_density_per_km2": 4.0,
    "sim.dt_s": 1.0,
    "sim.max_timesteps": 20000,
}

_LEGACY_ALGOS = [DecompositionAlgo.TGC_BASIC, DecompositionAlgo.WEIGHTED_VORONOI,
                 DecompositionAlgo.CLASSIC_VORONOI, DecompositionAlgo.KMEANS]


def _legacy_sojourn(s):
    return f"{s.agent_id}|{s.state.value}|{s.t_in!r}|{s.t_out!r}|{s.reason_out}"


def _legacy_signature(res):
    m = res.metrics
    return "\n".join([
        f"total_energy_j={m.total_energy_j!r}",
        f"duration_s={m.duration_s!r}",
        f"workload_std_m={m.workload_std_m!r}",
        f"n_swaps={m.n_swaps!r}",
        f"n_failures={m.n_failures!r}",
        "per_agent_energy_j=" + ",".join(
            f"{k}:{v!r}" for k, v in sorted(m.per_agent_energy_j.items())),
        "per_agent_length_m=" + ",".join(
            f"{k}:{v!r}" for k, v in sorted(m.per_agent_length_m.items())),
        "replan_times_n=" + str(len(m.replan_times_s)),
        "sojourns=" + ";".join(_legacy_sojourn(s) for s in res.history.sojourns()),
        f"outcome={res.outcome.value}",
        f"coverage_frac={res.coverage_frac!r}",
        f"terminal_reason={res.terminal_reason}",
        "zones=" + ";".join(
            f"{i}:{z.polygon.wkt}" for i, z in sorted(res.partition.zones.items())),
    ])


@pytest.mark.slow
def test_flag_off_is_byte_identical_for_every_legacy_algorithm():
    signatures = {}
    for algo in _LEGACY_ALGOS:
        for replication in (0, 1):
            cfg = load_config("config/default.yaml", overrides=_LEGACY_OVERRIDES)
            assert cfg.mission.repartition_enabled is False
            result = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                                      replication=replication, algo=algo,
                                      planner=PlannerKind.DUBINS).run()
            # the golden must not be able to pass vacuously
            assert result.metrics.n_failures > 0, algo
            assert len(result.metrics.replan_times_s) > 0, algo
            assert result.repartitions == ()
            assert result.repartition_hold is None
            signatures[f"{algo.value}:{replication}"] = _legacy_signature(result)

    blob = json.dumps(signatures, sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == _LEGACY_GOLDEN_SHA256


# --------------------------------------------------------------------------- #
# G. the k-means re-partition stream is held apart from the t=0 stream          #
# --------------------------------------------------------------------------- #
def test_repartitioning_with_kmeans_does_not_disturb_its_t0_init(area_file):
    """KMeansHeuristicDecomposer draws inside every decompose call and a numpy
    Generator is stateful. Re-partitioning with the run's own instance would
    consume from the stream that seeded t=0 and make the replication-keyed init
    variance -- a characteristic this project keeps and reports -- depend on the
    trigger schedule. A sibling on its own stream cannot."""
    on_engine, on = _run(area_file, DecompositionAlgo.KMEANS, **{
        "mission.repartition_enabled": True,
        "mission.repartition_interval_s": 20.0,
    })
    assert _applied(on), "re-partitions must actually have run"

    # The t=0 partition is identical with the flag on and off, so nothing the
    # re-partition did consumed from the init stream. Taken from _build rather
    # than from the finished run, whose ``partition`` a revision has replaced.
    def _t0_zones(repartition):
        cfg = load_config("config/default.yaml", overrides=_overrides(
            area_file, **{"mission.repartition_enabled": repartition}))
        engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                                  algo=DecompositionAlgo.KMEANS)
        engine._build()
        return {i: z.polygon.wkt for i, z in engine.partition.zones.items()}

    off_t0 = _t0_zones(False)
    assert off_t0 and _t0_zones(True) == off_t0

    # and the sibling really is on the other stream
    assert STREAM_REPARTITION_INIT != STREAM_KMEANS_INIT
    assert on_engine.repartitioner.decomposer is not on_engine.decomposer
    for record in _applied(on):
        assert record["rng_stream"] == STREAM_REPARTITION_INIT


def test_a_deterministic_decomposer_re_partitions_with_the_very_same_object(area_file):
    """with_rng returns self for everything that is deterministic given its
    inputs, so no sibling is manufactured where none is needed."""
    engine, _ = _run(area_file, DecompositionAlgo.LLOYD_CVT,
                     **{"mission.repartition_enabled": True})
    assert engine.repartitioner.decomposer is engine.decomposer


# --------------------------------------------------------------------------- #
# H. the legacy redistribution path is structurally unreachable                 #
# --------------------------------------------------------------------------- #
def test_the_legacy_redistributor_cannot_be_reached_with_the_flag_on(area_file):
    cfg = load_config("config/default.yaml", overrides=_overrides(
        area_file, **{"mission.repartition_enabled": True}))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                              algo=DecompositionAlgo.LLOYD_CVT)
    engine._build()
    assert engine.redistributor is None
    from uav_swarm_sim.infrastructure.core_types import Event
    from uav_swarm_sim.infrastructure.enums import EventType

    with pytest.raises(AssertionError, match="legacy redistribution reached"):
        engine._redistribute(Event(EventType.FAILURE, 1.0, {"agent_id": 0}), 1.0)
