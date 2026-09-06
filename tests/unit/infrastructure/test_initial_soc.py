"""EXP-05 initial state-of-charge configuration and sampling tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.config import ConfigError, InitialSocConfig, load_config
from uav_swarm_sim.infrastructure.initial_soc import generate_initial_soc
from uav_swarm_sim.infrastructure.rng import (
    STREAM_INITIAL_SOC,
    STREAM_LAUNCH_SAMPLING,
    STREAM_OBSTACLES,
    RngFactory,
)


def test_missing_block_is_fixed_full_charge_without_rng(config_path):
    cfg = load_config(config_path).battery.initial_soc

    class NoRng:
        def stream(self, *_args):
            raise AssertionError("fixed mode must not request an RNG stream")

    assert cfg == InitialSocConfig()
    assert generate_initial_soc(cfg, 3, NoRng(), 7) == (1.0, 1.0, 1.0)


def test_fixed_supports_another_fraction_without_rng():
    cfg = InitialSocConfig(mode="fixed", value=0.625)

    class NoRng:
        def stream(self, *_args):
            raise AssertionError("fixed mode must not request an RNG stream")

    assert generate_initial_soc(cfg, 2, NoRng(), 1) == (0.625, 0.625)


def test_uniform_uses_named_replication_stream_and_preserves_values():
    calls = []

    class FakeGenerator:
        def uniform(self, low, high, size):
            assert (low, high, size) == (0.2, 0.8, 3)
            return np.array([0.2, 0.35, 0.8])

    factory = SimpleNamespace(
        stream=lambda name, replication: calls.append((name, replication)) or FakeGenerator()
    )
    cfg = InitialSocConfig(mode="uniform", value=None, low=0.2, high=0.8)
    actual = generate_initial_soc(cfg, 3, factory, 11)
    assert actual == pytest.approx((0.2, 0.35, 0.8), abs=0.0)
    assert calls == [(STREAM_INITIAL_SOC, 11)]


def test_truncated_normal_uses_standardized_bounds_without_clipping(monkeypatch):
    sentinel_rng = object()
    factory = SimpleNamespace(stream=lambda _name, _rep: sentinel_rng)
    cfg = InitialSocConfig(
        mode="truncated_normal", value=None, low=0.4, high=0.6, mean=0.5, std=0.1
    )
    calls = []

    def fake_rvs(lower, upper, **kwargs):
        calls.append((lower, upper, kwargs))
        return np.array([0.4, 0.45, 0.6])

    monkeypatch.setattr("uav_swarm_sim.infrastructure.initial_soc.truncnorm.rvs", fake_rvs)
    assert generate_initial_soc(cfg, 3, factory, 2) == pytest.approx(
        (0.4, 0.45, 0.6), abs=0.0
    )
    lower, upper, kwargs = calls[0]
    assert lower == pytest.approx(-1.0, abs=1e-15)
    assert upper == pytest.approx(1.0, abs=1e-15)
    assert kwargs == {
        "loc": 0.5,
        "scale": 0.1,
        "size": 3,
        "random_state": sentinel_rng,
    }


@pytest.mark.parametrize("mode", ["uniform", "truncated_normal"])
def test_random_modes_are_bounded_and_deterministic(mode):
    cfg = (
        InitialSocConfig(mode=mode, value=None, low=0.7, high=0.95)
        if mode == "uniform"
        else InitialSocConfig(
            mode=mode, value=None, low=0.7, high=0.95, mean=0.82, std=0.06
        )
    )
    a = generate_initial_soc(cfg, 128, RngFactory(1234), 5)
    b = generate_initial_soc(cfg, 128, RngFactory(1234), 5)
    c = generate_initial_soc(cfg, 128, RngFactory(1234), 6)
    assert a == b
    assert a != c
    assert all(0.7 <= value <= 0.95 for value in a)


def test_initial_soc_draws_do_not_change_obstacle_or_launch_streams():
    factory = RngFactory(99)
    obstacle_expected = factory.stream(STREAM_OBSTACLES, 3).random(12)
    launch_expected = factory.stream(STREAM_LAUNCH_SAMPLING, 0).random(12)
    cfg = InitialSocConfig(mode="uniform", value=None, low=0.5, high=1.0)

    generate_initial_soc(cfg, 20, factory, 3)

    assert np.array_equal(factory.stream(STREAM_OBSTACLES, 3).random(12), obstacle_expected)
    assert np.array_equal(factory.stream(STREAM_LAUNCH_SAMPLING, 0).random(12), launch_expected)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"battery.initial_soc.mode": "beta"}, "mode"),
        ({"battery.initial_soc.value": float("nan")}, "value"),
        ({"battery.initial_soc.value": float("inf")}, "value"),
        ({"battery.initial_soc.value": -0.01}, "value"),
        ({"battery.initial_soc.value": 1.01}, "value"),
        ({"battery.initial_soc.mode": "uniform", "battery.initial_soc.low": 0.8,
          "battery.initial_soc.high": 0.8}, "bounds"),
        ({"battery.initial_soc.mode": "uniform", "battery.initial_soc.low": float("nan"),
          "battery.initial_soc.high": 1.0}, "bounds"),
        ({"battery.initial_soc.mode": "uniform", "battery.initial_soc.low": 0.8,
          "battery.initial_soc.high": float("inf")}, "bounds"),
        ({"battery.initial_soc.mode": "truncated_normal", "battery.initial_soc.low": 0.5,
          "battery.initial_soc.high": 0.9, "battery.initial_soc.mean": 0.4,
          "battery.initial_soc.std": 0.1}, "mean"),
        ({"battery.initial_soc.mode": "truncated_normal", "battery.initial_soc.low": 0.5,
          "battery.initial_soc.high": 0.9, "battery.initial_soc.mean": 0.7,
          "battery.initial_soc.std": 0.0}, "std"),
        ({"battery.initial_soc.mode": "truncated_normal", "battery.initial_soc.low": 0.5,
          "battery.initial_soc.high": 0.9, "battery.initial_soc.mean": 0.7,
          "battery.initial_soc.std": float("nan")}, "std"),
    ],
)
def test_invalid_initial_soc_config_is_rejected(config_path, overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(config_path, overrides=overrides)


@pytest.mark.parametrize("bad", [None, True, [], "full"])
def test_initial_soc_requires_mapping(config_path, bad):
    with pytest.raises(ConfigError, match="initial_soc must be a mapping"):
        load_config(config_path, overrides={"battery.initial_soc": bad})


def test_mode_specific_fields_are_strict(config_path):
    with pytest.raises(ConfigError, match="invalid field.*low"):
        load_config(config_path, overrides={"battery.initial_soc.low": 0.5})
    with pytest.raises(ConfigError, match="low is required"):
        load_config(config_path, overrides={
            "battery.initial_soc.mode": "uniform",
            "battery.initial_soc.high": 1.0,
        })


def test_soc_fraction_rejects_boolean(config_path):
    with pytest.raises(ConfigError, match="value must be numeric"):
        load_config(config_path, overrides={"battery.initial_soc.value": True})
