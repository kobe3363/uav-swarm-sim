"""FIX-B4 -- engine-level swap-livelock stall detector (``safety.stall_detector``).

``StallDetector.observe`` is called once per SWAP_REQUEST with the agent's
current coverage-leg index. An agent that requests ``budget`` consecutive swaps
without any ``_cov_idx`` progress is flagged stalled; the engine then halts the
mission early (outcome stays MISSION_INCOMPLETE) and reports the agents in
``MissionResult.stalled_agents``. Default budget = 5 no-progress cycles.
"""
from __future__ import annotations

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.simulation_engine import (
    _STALL_SWAP_BUDGET,
    StallDetector,
)


def test_budget_default_is_five():
    assert _STALL_SWAP_BUDGET == 5


def test_first_swap_is_never_a_stall_cycle():
    det = StallDetector()
    det.observe(3, 30)
    assert det.stalled == set()


def test_stalls_after_budget_consecutive_noprogress_swaps():
    det = StallDetector()
    det.observe(3, 30)                 # first swap: establishes the baseline
    for k in range(_STALL_SWAP_BUDGET - 1):
        det.observe(3, 30)             # no-progress cycles 1..4
        assert det.stalled == set(), f"stalled too early at cycle {k + 1}"
    det.observe(3, 30)                 # no-progress cycle 5 -> stalled
    assert det.stalled == {3}


def test_progress_resets_the_count():
    det = StallDetector()
    det.observe(1, 10)
    for _ in range(_STALL_SWAP_BUDGET - 1):
        det.observe(1, 10)             # 4 no-progress cycles
    det.observe(1, 12)                 # progress -> reset
    for _ in range(_STALL_SWAP_BUDGET - 1):
        det.observe(1, 12)             # 4 more, still under budget
    assert det.stalled == set()
    det.observe(1, 12)                 # 5th consecutive -> stalled
    assert det.stalled == {1}


def test_agents_are_tracked_independently():
    det = StallDetector()
    for _ in range(_STALL_SWAP_BUDGET + 1):
        det.observe(0, 7)
        det.observe(4, 9)
    det.observe(2, 5)                  # a single swap elsewhere
    assert det.stalled == {0, 4}


def test_stall_detector_defaults_off(config_path):
    cfg = load_config(config_path)
    assert cfg.safety.stall_detector is False
