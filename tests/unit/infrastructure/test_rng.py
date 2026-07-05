"""infrastructure/rng tests (isolated): reproducibility & independence."""
from __future__ import annotations

import numpy as np

from uav_swarm_sim.infrastructure.rng import (
    STREAM_FAILURES,
    STREAM_OBSTACLES,
    RngFactory,
)


def test_same_key_same_draws():
    f1 = RngFactory(42)
    f2 = RngFactory(42)
    d1 = f1.stream(STREAM_OBSTACLES, replication=3).random(10)
    d2 = f2.stream(STREAM_OBSTACLES, replication=3).random(10)
    np.testing.assert_array_equal(d1, d2)


def test_different_name_independent():
    f = RngFactory(42)
    a = f.stream(STREAM_OBSTACLES, replication=0).random(100)
    b = f.stream(STREAM_FAILURES, replication=0).random(100)
    assert not np.array_equal(a, b)


def test_different_replication_differs():
    f = RngFactory(42)
    a = f.stream(STREAM_OBSTACLES, replication=0).random(100)
    b = f.stream(STREAM_OBSTACLES, replication=1).random(100)
    assert not np.array_equal(a, b)


def test_paired_design_streams_match_across_factories():
    # two independently constructed factories (same seed) -> identical failure
    # stream at the same replication: the property the paired MC design needs.
    fa = RngFactory(7)
    fb = RngFactory(7)
    for k in range(3):
        ea = fa.stream(STREAM_FAILURES, replication=k).integers(0, 1000, size=50)
        eb = fb.stream(STREAM_FAILURES, replication=k).integers(0, 1000, size=50)
        np.testing.assert_array_equal(ea, eb)


def test_different_master_seed_differs():
    a = RngFactory(1).stream(STREAM_OBSTACLES).random(100)
    b = RngFactory(2).stream(STREAM_OBSTACLES).random(100)
    assert not np.array_equal(a, b)
