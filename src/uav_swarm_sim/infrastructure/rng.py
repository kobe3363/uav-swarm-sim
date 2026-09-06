"""Reproducible randomness.

One master seed fans out into independent, named streams via numpy
SeedSequence spawn keys. The key property the rest of the system relies on:

    same (master_seed, name, replication)  ->  identical generator
    different name                          ->  statistically independent stream

This enables the paired Monte-Carlo design: replication ``k`` uses the same
``"obstacles"`` and ``"failures"`` streams regardless of which decomposition
algorithm is being compared, so environment and failure draws are identical
across compared algorithms while differing across replications.
"""
from __future__ import annotations

import hashlib

import numpy as np

# canonical stream names (documentation; any string works)
STREAM_OBSTACLES = "obstacles"
STREAM_FAILURES = "failures"
STREAM_LAUNCH_SAMPLING = "launch_sampling"
STREAM_INITIAL_SOC = "initial_soc"
STREAM_KMEANS_INIT = "kmeans_init"
STREAM_TARGETS = "targets"
STREAM_DYNOBS = "dynamic_obstacles"
# EXP-08 (mission.repartition_enabled): draws made by an IN-FLIGHT re-partition,
# held apart from STREAM_KMEANS_INIT on purpose.
#
# KMeansHeuristicDecomposer draws from its generator inside every decompose()
# call (kmeans_heuristic.py: the k-means++ init and the empty-cluster centroid),
# and a Generator is stateful. Re-partitioning with the run's own k-means
# instance would therefore consume from the SAME stream the t=0 partition used,
# which makes the replication-keyed k-means init variance -- a characteristic
# this project deliberately keeps and reports (CLAUDE.md) -- depend on the
# re-partition trigger schedule. Binding re-partition draws to their own stream
# keeps STREAM_KMEANS_INIT consumed exactly once per run, at t=0.
#
# Paired-seed determinism is unaffected either way: ``stream`` is a pure
# function of (master_seed, name, replication), so a new name cannot move the
# obstacle, failure, launch, SoC or target draws.
STREAM_REPARTITION_INIT = "repartition_init"


def _stable_key(name: str, replication: int) -> int:
    """Deterministic 64-bit key from (name, replication).

    Python's built-in hash() is salted per-process, so we use sha256 to get a
    value stable across runs and machines.
    """
    digest = hashlib.sha256(f"{name}:{replication}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class RngFactory:
    def __init__(self, master_seed: int) -> None:
        self._master_seed = int(master_seed)

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def stream(self, name: str, replication: int = 0) -> np.random.Generator:
        """Return an independent Generator for the given named stream.

        Deterministic in (master_seed, name, replication).
        """
        key = _stable_key(name, replication)
        seq = np.random.SeedSequence(entropy=self._master_seed, spawn_key=(key,))
        return np.random.default_rng(seq)
