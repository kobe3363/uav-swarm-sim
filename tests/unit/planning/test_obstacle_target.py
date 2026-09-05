from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
import yaml
from shapely.geometry import box
from shapely.ops import unary_union

from uav_swarm_sim.infrastructure.config import ConfigError, load_config
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import _free_connected, generate


def _target_env(env, **changes):
    values = {
        "obstacle_generation_mode": "target",
        "obstacle_target_count": 10,
        "obstacle_area_fraction": 0.05,
        "obstacle_area_fraction_tolerance": 0.005,
        "obstacle_generation_max_attempts": 10_000,
    }
    values.update(changes)
    return replace(env, **values)


def _geometry_signature(obstacles):
    return [(o.cls, o.polygon.wkb_hex, o.z_floor, o.z_ceil) for o in obstacles]


def test_target_config_defaults_off_and_parses_explicit_mode(config_path):
    raw = yaml.safe_load(config_path.read_text())
    default = load_config(config_path)
    target = load_config(config_path, overrides={"env.obstacle_generation_mode": "target"})

    assert "obstacle_generation_mode" not in raw["env"]
    assert default.env.obstacle_generation_mode == "poisson"
    assert target.env.obstacle_generation_mode == "target"
    assert target.env.obstacle_target_count == 10
    assert target.env.obstacle_area_fraction == pytest.approx(0.05)
    assert target.env.obstacle_area_fraction_tolerance == pytest.approx(0.005)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("env.obstacle_generation_mode", "random"),
        ("env.obstacle_target_count", 0),
        ("env.obstacle_area_fraction", 0.0),
        ("env.obstacle_area_fraction", 1.0),
        ("env.obstacle_area_fraction_tolerance", -0.001),
        ("env.obstacle_generation_max_attempts", 0),
    ],
)
def test_target_config_rejects_invalid_values(config_path, key, bad):
    with pytest.raises(ConfigError, match="obstacle_"):
        load_config(config_path, overrides={key: bad})


def test_target_config_rejects_attempt_budget_below_count(config_path):
    with pytest.raises(ConfigError, match="max_attempts must be >=.*target_count"):
        load_config(
            config_path,
            overrides={
                "env.obstacle_generation_mode": "target",
                "env.obstacle_target_count": 10,
                "env.obstacle_generation_max_attempts": 9,
            },
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 17, 20260906])
def test_target_field_has_exact_count_area_bounds_and_no_intersections(config_path, seed):
    area = box(0.0, 0.0, 1000.0, 750.0)
    cfg = _target_env(load_config(config_path).env)
    obstacles = generate(area, cfg, np.random.default_rng(seed))
    final_geometry = unary_union([o.polygon for o in obstacles])

    assert len(obstacles) == 10
    assert final_geometry.area == pytest.approx(37_500.0, abs=3_750.0)
    assert len(final_geometry.geoms) == 10
    assert all(o.polygon.area == pytest.approx(3_750.0, abs=1e-8) for o in obstacles)
    assert all(o.z_floor == pytest.approx(0.0) and math.isinf(o.z_ceil) for o in obstacles)
    for left, right in combinations(obstacles, 2):
        assert not left.polygon.intersects(right.polygon)


def test_target_seed_reproduces_geometry_and_other_seeds_move_it(config_path):
    area = box(0.0, 0.0, 1000.0, 750.0)
    cfg = _target_env(load_config(config_path).env)

    first = generate(area, cfg, np.random.default_rng(91))
    repeat = generate(area, cfg, np.random.default_rng(91))
    different = generate(area, cfg, np.random.default_rng(92))

    assert _geometry_signature(first) == _geometry_signature(repeat)
    assert [o.polygon.wkb_hex for o in first] != [o.polygon.wkb_hex for o in different]


def test_target_raw_area_is_separate_from_clearance_buffer(config_path):
    area = box(0.0, 0.0, 1000.0, 750.0)
    cfg = _target_env(load_config(config_path).env)
    obstacles = generate(area, cfg, np.random.default_rng(7))
    env_map = EnvironmentMap(area, obstacles, cfg.clearance_buffer_m)
    raw_area = unary_union([o.polygon for o in obstacles]).area

    assert raw_area == pytest.approx(37_500.0, abs=3_750.0)
    assert env_map.free_space.area < area.area - raw_area
    assert all(env_map.in_obstacle(o.polygon.representative_point().coords[0]) for o in obstacles)


def test_target_rejects_clipped_candidate_and_reports_bounded_failure(config_path):
    class BoundaryRng:
        def uniform(self, *_args):
            return 0.0

        def integers(self, *_args):
            return 0

    area = box(0.0, 0.0, 10.0, 10.0)
    cfg = _target_env(
        load_config(config_path).env,
        obstacle_target_count=1,
        obstacle_area_fraction=0.25,
        obstacle_area_fraction_tolerance=0.01,
        obstacle_generation_max_attempts=1,
    )

    with pytest.raises(RuntimeError, match=r"1 attempts: placed 0 of 1"):
        generate(area, cfg, BoundaryRng())


def test_target_impossible_packing_stops_at_attempt_limit(config_path):
    area = box(0.0, 0.0, 1.0, 1.0)
    cfg = _target_env(
        load_config(config_path).env,
        obstacle_area_fraction=0.9,
        obstacle_area_fraction_tolerance=0.0,
        obstacle_generation_max_attempts=30,
    )

    with pytest.raises(RuntimeError, match=r"30 attempts: (placed|final area fraction)"):
        generate(area, cfg, np.random.default_rng(4))


def test_zero_area_tolerance_accepts_roundoff_only(config_path):
    area = box(0.0, 0.0, 1000.0, 750.0)
    cfg = _target_env(
        load_config(config_path).env,
        obstacle_area_fraction_tolerance=0.0,
    )

    obstacles = generate(area, cfg, np.random.default_rng(17))
    final_fraction = unary_union([o.polygon for o in obstacles]).area / area.area

    assert len(obstacles) == 10
    assert final_fraction == pytest.approx(0.05, abs=1e-12)


def test_target_connectivity_counts_small_isolated_pockets():
    area = box(0.0, 0.0, 100.0, 100.0)
    narrow_barrier = box(0.5, 0.0, 1.0, 100.0)

    assert _free_connected(area, [narrow_barrier], buffer_m=0.0)
    assert not _free_connected(
        area,
        [narrow_barrier],
        buffer_m=0.0,
        min_component_fraction=0.0,
    )


def test_explicit_poisson_mode_preserves_geometry_and_rng_state(config_path):
    area = box(0.0, 0.0, 400.0, 300.0)
    implicit = replace(
        load_config(config_path).env,
        obstacle_density_per_km2=20.0,
        obstacle_size_range_m=(20.0, 20.0),
        obstacle_shapes=("square",),
    )
    explicit = replace(implicit, obstacle_generation_mode="poisson")
    rng_implicit = np.random.default_rng(1234)
    rng_explicit = np.random.default_rng(1234)

    before = generate(area, implicit, rng_implicit)
    after = generate(area, explicit, rng_explicit)

    # Fixed golden captured from the pre-EXP-03 origin/main commit 14bb610.
    expected = [
        (
            2,
            "01030000000100000005000000C5E0657BE2685D403E687A0EB4B6464062F0B23D713461403E687A0E"
            "B4B6464062F0B23D713461407CD0F41C686D3940C5E0657BE2685D407CD0F41C686D3940C5E0657BE268"
            "5D403E687A0EB4B64640",
            0.0,
            math.inf,
        ),
        (
            0,
            "0103000000010000000500000059D66F6F1B7A7740E9DF99841346564059D66F6F1BBA7840E9DF9984"
            "1346564059D66F6F1BBA7840E9DF99841346514059D66F6F1B7A7740E9DF99841346514059D66F6F1B7A"
            "7740E9DF998413465640",
            0.0,
            math.inf,
        ),
        (
            1,
            "0103000000010000000500000065BC16AB583E6D4013D266D961D1704065BC16AB58BE6F4013D266D9"
            "61D1704065BC16AB58BE6F4026A4CDB2C3226F4065BC16AB583E6D4026A4CDB2C3226F4065BC16AB583E"
            "6D4013D266D961D17040",
            0.0,
            math.inf,
        ),
    ]
    expected_rng_bytes = (
        "9105c5acad3280f47786eda87027190cd49d5abcf2a919bd3c6206394868f9dd"
        "8b870c2ca566e9c90084d3de3e969b7b403f650fcbaf8e29813c06af17eaa30a"
    )

    assert _geometry_signature(before) == expected
    assert _geometry_signature(after) == expected
    assert rng_implicit.bytes(64).hex() == expected_rng_bytes
    assert rng_explicit.bytes(64).hex() == expected_rng_bytes
