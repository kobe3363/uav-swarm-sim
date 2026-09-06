"""EXP-07a at the engine boundary: algorithm identity, experiment mode, flag-off.

The identity tests are the D-3 work: the enum alone cannot distinguish
``KMeansHeuristicDecomposer(weighted=True)`` from ``WeightedTgcDecomposer``,
because the fleet-size tier path resolves both to ``weighted_voronoi``.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest
from shapely.geometry import box, mapping

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.run_output import build_plan, build_results_single


@pytest.fixture
def overrides(tmp_path):
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
    }


def _engine(overrides, algo=DecompositionAlgo.LLOYD_CVT, **extra):
    cfg = load_config("config/default.yaml", overrides=dict(overrides, **extra))
    engine = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0, algo=algo)
    engine._build()
    return cfg, engine


# --------------------------------------------------------------------------- #
# G. algorithm identity                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_drones", [3, 5, 8])
def test_the_resolved_class_is_recorded_even_when_the_enum_is_ambiguous(overrides, n_drones):
    """With no explicit algo and n <= tier_thresholds[0], the engine builds a
    k-means heuristic that LABELS ITSELF weighted_voronoi. The enum therefore
    cannot identify the algorithm; the resolved class can, and does."""
    cfg, engine = _engine(overrides, algo=None, **{"fleet.n_drones": n_drones})
    plan = build_plan(cfg, identity={}, algo=None, planner=PlannerKind.DUBINS, engine=engine)

    assert engine.partition.algo is DecompositionAlgo.WEIGHTED_VORONOI     # ambiguous
    assert plan["setup"]["decomposer_class"] == "KMeansHeuristicDecomposer"  # unambiguous
    assert type(engine.decomposer).__name__ == "KMeansHeuristicDecomposer"


def test_the_two_implementations_behind_weighted_voronoi_are_now_distinguishable(overrides):
    _, implicit = _engine(overrides, algo=None)
    _, explicit = _engine(overrides, algo=DecompositionAlgo.WEIGHTED_VORONOI)

    assert implicit.partition.algo is DecompositionAlgo.WEIGHTED_VORONOI
    assert explicit.partition.algo is DecompositionAlgo.WEIGHTED_VORONOI
    assert type(implicit.decomposer).__name__ != type(explicit.decomposer).__name__
    assert type(explicit.decomposer).__name__ == "WeightedTgcDecomposer"


def test_lloyd_records_its_own_identity_and_the_settings_actually_used(overrides):
    cfg, engine = _engine(overrides, **{"planning.partition.max_iterations": 7})
    plan = build_plan(cfg, identity={}, algo=DecompositionAlgo.LLOYD_CVT,
                      planner=PlannerKind.DUBINS, engine=engine)

    assert plan["setup"]["decomposition_algorithm"] == "lloyd_cvt"
    assert plan["setup"]["decomposer_class"] == "LloydCvtDecomposer"
    assert plan["setup"]["partition_settings"] == {
        "init_sites": "deploy_poses", "max_iterations": 7,
        "site_tolerance_m": 1.0, "weight_policy": "uniform",
    }


def test_experiment_mode_refuses_to_pick_an_algorithm_by_fleet_size(overrides):
    with pytest.raises(ValueError, match="experiment_mode forbids"):
        _engine(overrides, algo=None, **{"mission.experiment_mode": True})


def test_experiment_mode_is_satisfied_by_naming_the_algorithm(overrides):
    for algo in (DecompositionAlgo.TGC_BASIC, DecompositionAlgo.LLOYD_CVT):
        _, engine = _engine(overrides, algo=algo, **{"mission.experiment_mode": True})
        assert engine.partition.algo is algo


def test_lloyd_requires_the_coverage_raster_it_consumes(overrides):
    with pytest.raises(ValueError, match="raster_enabled"):
        _engine(overrides, **{"coverage.raster_enabled": False})


# --------------------------------------------------------------------------- #
# results plumbing                                                             #
# --------------------------------------------------------------------------- #
def test_the_partition_record_reaches_results_json_and_is_strict_json(overrides):
    _, engine = _engine(overrides)
    result = engine.run()
    out = build_results_single(result, SimpleNamespace(ergodic=False),
                               identity={}, wall_time_s=0.0)

    record = out["partition"]
    assert record["algorithm"] == "lloyd_cvt"
    assert record["decomposer_class"] == "LloydCvtDecomposer"
    assert record["converged"] is True
    assert record["cells"]["assigned"] == record["cells"]["eligible"]
    assert set(record["per_drone"]) == {"0", "1", "2"}
    json.dumps(out, allow_nan=False)


# --------------------------------------------------------------------------- #
# H. flag-off byte identity                                                    #
# --------------------------------------------------------------------------- #
def test_partition_settings_are_inert_for_every_other_algorithm(overrides):
    """Proof 3: the new config block cannot perturb a legacy run. Changing every
    partition knob must leave a non-Lloyd run's partition and output untouched."""
    loud = {"planning.partition.init_sites": "maximin",
            "planning.partition.max_iterations": 3,
            "planning.partition.site_tolerance_m": 25.0}
    for algo in (DecompositionAlgo.TGC_BASIC, DecompositionAlgo.WEIGHTED_VORONOI,
                 DecompositionAlgo.CLASSIC_VORONOI, DecompositionAlgo.KMEANS):
        _, quiet_engine = _engine(overrides, algo=algo)
        _, loud_engine = _engine(overrides, algo=algo, **loud)

        quiet = {i: z.polygon.wkt for i, z in quiet_engine.partition.zones.items()}
        assert quiet == {i: z.polygon.wkt for i, z in loud_engine.partition.zones.items()}
        assert quiet_engine.partition_diagnostics is None
        assert loud_engine.partition_diagnostics is None


def test_no_partition_key_appears_in_a_legacy_runs_output(overrides):
    cfg, engine = _engine(overrides, algo=DecompositionAlgo.TGC_BASIC)
    result = engine.run()
    out = build_results_single(result, SimpleNamespace(ergodic=False),
                               identity={}, wall_time_s=0.0)
    plan = build_plan(cfg, identity={}, algo=DecompositionAlgo.TGC_BASIC,
                      planner=PlannerKind.DUBINS, engine=engine)

    assert "partition" not in out
    assert "partition_settings" not in plan["setup"]
    # the resolved class is still recorded -- that key is additive on every path
    assert plan["setup"]["decomposer_class"] == "TgcBasicDecomposer"


def test_the_new_config_block_is_absent_from_every_shipped_yaml():
    """config_hash is taken over the RAW yaml, so these defaults must live in
    code. If a partition block were added to a shipped config, every pinned hash
    and every cross-commit golden would move."""
    for path in sorted(pathlib.Path("config").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "partition:" not in text, path
        assert "experiment_mode" not in text, path


def test_lloyd_default_init_seeds_where_the_drones_stage(overrides):
    """Proof 1: with the default policy the partitioner seeds from the staging
    poses, so switching the algorithm on does not also move the seeding. One
    iteration with an enormous tolerance stops before the sites have travelled
    far, which exposes the seeding the run actually started from."""
    _, engine = _engine(overrides, **{"planning.partition.max_iterations": 1,
                                      "planning.partition.site_tolerance_m": 1e9})
    diagnostics = engine.partition_diagnostics
    assert diagnostics.iterations == 1
    assert diagnostics.converged is True
    assert diagnostics.settings["init_sites"] == "deploy_poses"

    ring = {(round(p.x, 6), round(p.y, 6)) for p in engine.deploy_poses}
    assert len(ring) == 3
    # every staging pose sits on the launch ring, which is what seeded the sweep
    radii = {round(((p.x - engine.launch_pose.x) ** 2
                    + (p.y - engine.launch_pose.y) ** 2) ** 0.5, 6)
             for p in engine.deploy_poses}
    assert len(radii) == 1


def test_the_cli_default_algo_cannot_stand_in_for_a_named_one_in_experiment_mode():
    """D-3 at the CLI boundary. The engine guard only sees `algo is None`, and a
    CLI default arrives as a real algorithm -- so an experiment-mode run that
    omitted `--algo` would have recorded `weighted_voronoi` as though the user
    had chosen it. Both entry points must refuse instead."""
    from uav_swarm_sim.experiments.run_replay import resolve_algo as replay_resolve
    from uav_swarm_sim.experiments.run_single_mission import resolve_algo

    strict = load_config("config/default.yaml", {"mission.experiment_mode": True})
    for resolver in (resolve_algo, replay_resolve):
        with pytest.raises(SystemExit, match="requires an explicit --algo"):
            resolver(strict, None)
        # naming it explicitly is always accepted
        assert resolver(strict, "tgc_basic") is DecompositionAlgo.TGC_BASIC


def test_the_cli_default_is_unchanged_outside_experiment_mode():
    """Flag-off identity at the CLI: omitting --algo still means weighted_voronoi."""
    from uav_swarm_sim.experiments.run_replay import resolve_algo as replay_resolve
    from uav_swarm_sim.experiments.run_single_mission import resolve_algo

    relaxed = load_config("config/default.yaml")
    assert relaxed.mission.experiment_mode is False
    for resolver in (resolve_algo, replay_resolve):
        assert resolver(relaxed, None) is DecompositionAlgo.WEIGHTED_VORONOI
        assert resolver(relaxed, "lloyd_cvt") is DecompositionAlgo.LLOYD_CVT
