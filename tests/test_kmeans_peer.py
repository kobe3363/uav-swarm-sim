"""Regression tests for H2: k-means promoted to a first-class comparison peer.

Before this change, KMeansHeuristicDecomposer reported its name as
DecompositionAlgo.WEIGHTED_VORONOI and was reachable only via the small-swarm
tier selector -- it was an *implementation* of the weighted partition, never a
distinct algorithm. The headline decomposition comparison ran only three algos.

These tests lock in the new contract:
  * KMEANS is a distinct enum member and a named member of DECOMPOSITION_PEERS;
  * the decomposer's identity follows its `weighted` flag (position-based
    k-means -> KMEANS; battery-weighted heuristic -> WEIGHTED_VORONOI);
  * the engine dispatches algo=KMEANS to the position-based (weighted=False)
    k-means on the paired stream, and a full mission runs through the IDENTICAL
    pipeline and is labelled KMEANS;
  * the tier selector is unaffected by the relabel (HEURISTIC still realizes the
    weighted partition, so no scale-tier behaviour changed).
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.execution.algorithm_selector import make_decomposers
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    DecompositionAlgo,
    PlannerKind,
    TierStrategy,
)
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.comparison import DECOMPOSITION_PEERS
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.kmeans_heuristic import KMeansHeuristicDecomposer
from uav_swarm_sim.planning.weighted_decomposition import WeightedTgcDecomposer


def _tiny_cfg(config_path, algo_overrides=None):
    base = {
        "fleet.n_drones": 3,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "env.obstacle_size_range_m": [10.0, 30.0],
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
    }
    if algo_overrides:
        base.update(algo_overrides)
    return load_config(config_path, overrides=base)


@pytest.fixture
def cfg(config_path):
    return _tiny_cfg(config_path)


# --------------------------------------------------------------------------- #
# 1. KMEANS is a distinct, named first-class algorithm                         #
# --------------------------------------------------------------------------- #
def test_kmeans_is_a_distinct_first_class_algo():
    assert hasattr(DecompositionAlgo, "KMEANS")
    assert DecompositionAlgo.KMEANS.value == "kmeans"
    # it must NOT be an alias of the weighted contribution
    assert DecompositionAlgo.KMEANS is not DecompositionAlgo.WEIGHTED_VORONOI


def test_decomposition_peers_lists_kmeans_alongside_the_baselines():
    """The headline comparison set is the single source of truth for 'what gets
    compared'; it must contain the three position-based baselines and the
    weighted contribution -- four distinct peers."""
    assert DecompositionAlgo.KMEANS in DECOMPOSITION_PEERS
    assert set(DECOMPOSITION_PEERS) == {
        DecompositionAlgo.CLASSIC_VORONOI,
        DecompositionAlgo.KMEANS,
        DecompositionAlgo.TGC_BASIC,
        DecompositionAlgo.WEIGHTED_VORONOI,
    }
    assert len(DECOMPOSITION_PEERS) == len(set(DECOMPOSITION_PEERS))  # no dupes


# --------------------------------------------------------------------------- #
# 2. The decomposer's identity follows its weighting                           #
# --------------------------------------------------------------------------- #
def test_kmeans_decomposer_identity_follows_weighting(cfg):
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    rng = RngFactory(0).stream("kmeans_init", 0)

    position_based = KMeansHeuristicDecomposer(motion, weighted=False, rng=rng)
    battery_weighted = KMeansHeuristicDecomposer(motion, weighted=True, rng=rng)

    assert position_based.name is DecompositionAlgo.KMEANS
    assert battery_weighted.name is DecompositionAlgo.WEIGHTED_VORONOI


# --------------------------------------------------------------------------- #
# 3. The engine dispatches algo=KMEANS to the position-based peer              #
# --------------------------------------------------------------------------- #
def test_engine_dispatches_kmeans_as_position_based_peer(cfg):
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    eng = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), replication=0,
        algo=DecompositionAlgo.KMEANS, planner=PlannerKind.DUBINS,
    )
    dec = eng._make_decomposer(motion)
    assert isinstance(dec, KMeansHeuristicDecomposer)
    assert dec._weighted is False           # the POSITION-based baseline
    assert dec.name is DecompositionAlgo.KMEANS


# --------------------------------------------------------------------------- #
# 4. A full mission runs end-to-end through the identical pipeline as KMEANS   #
# --------------------------------------------------------------------------- #
def test_kmeans_runs_end_to_end_and_is_labelled_kmeans(cfg):
    """Proves k-means is wired through the SAME engine path as the other peers:
    a tiny mission completes, covers its area, ends every agent idle, and the
    resulting partition is labelled KMEANS (not WEIGHTED_VORONOI)."""
    eng = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), replication=0,
        algo=DecompositionAlgo.KMEANS, planner=PlannerKind.DUBINS,
    )
    result = eng.run()
    assert not result.aborted
    assert result.coverage_frac > 0.99
    assert result.metrics.total_energy_j > 0
    assert eng.partition.algo is DecompositionAlgo.KMEANS
    for a in eng.fleet.agents.values():
        assert a.state is AgentState.S0_IDLE


def test_kmeans_is_deterministic_on_a_fixed_seed(cfg):
    """Paired-design requirement: identical seed -> identical result, so any
    difference vs. another peer is attributable to the algorithm, not the noise.
    (k-means uses a seeded k-means++ init from the content-addressed stream.)"""
    r1 = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), 0,
        algo=DecompositionAlgo.KMEANS, planner=PlannerKind.DUBINS,
    ).run()
    r2 = SimulationEngine(
        cfg, RngFactory(cfg.sim.master_seed), 0,
        algo=DecompositionAlgo.KMEANS, planner=PlannerKind.DUBINS,
    ).run()
    assert r1.config_hash == r2.config_hash
    assert r1.metrics.total_energy_j == pytest.approx(r2.metrics.total_energy_j)
    assert r1.metrics.workload_std_m == pytest.approx(r2.metrics.workload_std_m)


# --------------------------------------------------------------------------- #
# 5. The tier selector is unaffected by the relabel (no scale-tier regression) #
# --------------------------------------------------------------------------- #
def test_tier_selector_still_realizes_the_weighted_partition(cfg):
    """The small-swarm HEURISTIC tier must still produce the battery-weighted
    realization (its k-means runs with weighted=True), so promoting the
    position-based form to a peer changed nothing about scale-tier behaviour."""
    spec = build_spec(cfg)
    motion = make_motion_model(spec)
    rng = RngFactory(0).stream("kmeans_init", 0)

    heuristic = make_decomposers(TierStrategy.HEURISTIC, motion, rng)
    assert len(heuristic) == 1
    assert heuristic[0].name is DecompositionAlgo.WEIGHTED_VORONOI

    compare_both = make_decomposers(TierStrategy.COMPARE_BOTH, motion, rng)
    assert len(compare_both) == 2
    names = {d.name for d in compare_both}
    assert names == {DecompositionAlgo.WEIGHTED_VORONOI}  # heuristic(weighted) + WeightedTgc
    assert any(isinstance(d, WeightedTgcDecomposer) for d in compare_both)
    assert any(isinstance(d, KMeansHeuristicDecomposer) for d in compare_both)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
