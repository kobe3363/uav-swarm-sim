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
STREAM_KMEANS_INIT = "kmeans_init"
STREAM_TARGETS = "targets"
STREAM_DYNOBS = "dynamic_obstacles"


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
