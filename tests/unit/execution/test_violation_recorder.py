"""EXP-10 factual safety-violation recorder (acceptance checks 7.1-7.9).

Duck-typed stand-ins: ``ViolationRecorder`` reads only ``id/pose/layer`` and the
pre-tick ``_legs/_leg_idx/_t`` off each agent, so a plain object suffices -- no
full Agent/engine fixture. Expected values are hand-computed (distances,
``n_samples*dt``, ``hypot/dt``), never read back from the implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import SafetyConfig
from uav_swarm_sim.infrastructure.core_types import (
    Path, Pose, inplace_turn_segment, straight_segment,
)
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.execution.safety_monitor import ViolationRecorder
from uav_swarm_sim.planning.environment_map import EnvironmentMap, LayerStack
from uav_swarm_sim.planning.obstacle_generator import Obstacle

_AREA = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
_DT = 1.0


@dataclass
class _Agent:
    id: int
    pose: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0))
    layer: int = 0
    _legs: list = field(default_factory=list)
    _leg_idx: int = 0
    _t: float = 0.0


def _rec(world=None, sep=10.0, dt=_DT, v_cruise=21.0):
    world = EnvironmentMap(_AREA, [], buffer_m=5.0) if world is None else world
    cfg = SafetyConfig(min_separation_m=sep, obstacle_buffer_m=5.0, predict_horizon_s=5.0)
    return ViolationRecorder(world, cfg, SimpleNamespace(v_cruise=v_cruise), dt)


def _place(rec, agents, xys, t):
    """One sampled tick with agents at ``xys`` (no motion => speed inert)."""
    for a, (x, y) in zip(agents, xys):
        a.pose = Pose(float(x), float(y), 0.0)
    rec.snapshot(agents)
    rec.observe(agents, t)


def _by_kind(violations, kind):
    return [v for v in violations if v.kind == kind]


# --------------------------------------------------------------------------- #
# 7.1 predicted-but-avoided threat -> no factual violation                    #
# --------------------------------------------------------------------------- #
def test_approach_without_breach_records_nothing():
    rec = _rec(sep=10.0)
    a = [_Agent(0), _Agent(1)]
    _place(rec, a, [(0, 0), (0, 12)], 0.0)   # 12 m apart
    _place(rec, a, [(0, 0), (0, 11)], 1.0)   # 11 m: closer but still clear
    rec.finalize(1.0)
    v, minima = rec.result()
    assert v == ()
    assert minima["min_separation_m"] == 11.0   # tracked even with no breach
    assert minima["n_hard"] == 0 and minima["n_soft"] == 0


# --------------------------------------------------------------------------- #
# 7.2 short / continuous / restarted; pair id normalized, no dup              #
# --------------------------------------------------------------------------- #
def test_continuous_breach_is_one_episode():
    rec = _rec(sep=10.0)
    a = [_Agent(0), _Agent(1)]
    for t in (0.0, 1.0, 2.0):
        _place(rec, a, [(0, 0), (0, 9)], t)   # 9 m < 10 m
    rec.finalize(2.0)
    v = _by_kind(rec.result()[0], "separation")
    assert len(v) == 1
    ep = v[0]
    assert ep.severity == "hard"
    assert ep.agents == (0, 1)
    assert ep.duration_s == 3 * _DT
    assert ep.min_separation_m == 9.0


def test_ended_and_restarted_is_two_episodes():
    rec = _rec(sep=10.0)
    a = [_Agent(0), _Agent(1)]
    _place(rec, a, [(0, 0), (0, 9)], 0.0)    # breach
    _place(rec, a, [(0, 0), (0, 11)], 1.0)   # >= 10.5 => recovered/closed
    _place(rec, a, [(0, 0), (0, 9)], 2.0)    # fresh breach
    rec.finalize(2.0)
    v = _by_kind(rec.result()[0], "separation")
    assert len(v) == 2
    assert v[0].ended_reason == "recovered"


def test_pair_id_normalized_no_duplication():
    rec = _rec(sep=10.0)
    a = [_Agent(7), _Agent(3)]               # deliberately unordered ids
    _place(rec, a, [(0, 0), (0, 9)], 0.0)
    _place(rec, a, [(0, 0), (0, 9)], 1.0)
    rec.finalize(1.0)
    v = _by_kind(rec.result()[0], "separation")
    assert len(v) == 1
    assert v[0].agents == (3, 7)             # normalized (lo, hi)


# --------------------------------------------------------------------------- #
# 7.3 hysteresis; duration != span; extremum only while in violation          #
# --------------------------------------------------------------------------- #
def test_hysteresis_holds_episode_and_duration_differs_from_span():
    rec = _rec(sep=10.0)                      # close threshold = 10 * 1.05 = 10.5
    a = [_Agent(0), _Agent(1)]
    _place(rec, a, [(0, 0), (0, 9.0)], 0.0)   # breach (violating sample)
    _place(rec, a, [(0, 0), (0, 10.2)], 1.0)  # in [10, 10.5) band: HOLD, not counted
    _place(rec, a, [(0, 0), (0, 10.2)], 2.0)  # still holding
    _place(rec, a, [(0, 0), (0, 9.0)], 3.0)   # breach again (same episode)
    _place(rec, a, [(0, 0), (0, 11.0)], 4.0)  # >= 10.5 => close
    rec.finalize(4.0)
    v = _by_kind(rec.result()[0], "separation")
    assert len(v) == 1
    ep = v[0]
    assert ep.duration_s == 2 * _DT           # only the two violating samples
    assert ep.t_start == 0.0 and ep.t_end == 3.0
    assert (ep.t_end - ep.t_start) == 3 * _DT  # span != duration
    assert ep.min_separation_m == 9.0          # holds (10.2) never touched the min


# --------------------------------------------------------------------------- #
# 7.4 obstacle: buffer / interior / transition; one escalating episode        #
# --------------------------------------------------------------------------- #
def _obstacle_env():
    # axis-aligned square [90,110]^2 (centre 100,100, side 20); buffer 5 => [85,115].
    sq = Polygon([(90, 90), (110, 90), (110, 110), (90, 110)])
    return EnvironmentMap(_AREA, [Obstacle(0, 0, sq)], buffer_m=5.0)


def test_soft_then_hard_is_one_escalating_episode():
    rec = _rec(world=_obstacle_env())
    a = [_Agent(0)]
    _place(rec, a, [(112, 100)], 0.0)        # in buffer only -> soft
    _place(rec, a, [(100, 100)], 1.0)        # inside raw obstacle -> hard
    rec.finalize(1.0)
    v = _by_kind(rec.result()[0], "obstacle")
    assert len(v) == 1
    ep = v[0]
    assert ep.severity == "hard"             # escalated, not two records
    assert ep.max_penetration_m == 10.0      # centre to nearest edge
    assert ep.duration_s == 2 * _DT


def test_pure_soft_episode_has_zero_penetration():
    rec = _rec(world=_obstacle_env())
    a = [_Agent(0)]
    _place(rec, a, [(112, 100)], 0.0)        # buffer only, never raw
    _place(rec, a, [(113, 100)], 1.0)
    _place(rec, a, [(200, 200)], 2.0)        # clear -> close
    rec.finalize(2.0)
    v = _by_kind(rec.result()[0], "obstacle")
    assert len(v) == 1
    assert v[0].severity == "soft"
    assert v[0].max_penetration_m == 0.0     # never a fictitious outside distance
    assert v[0].ended_reason == "recovered"


# --------------------------------------------------------------------------- #
# 7.5 formation/RTH-phase states recorded; cross-layer is a documented GAP    #
# --------------------------------------------------------------------------- #
def test_records_regardless_of_flight_phase():
    # The recorder is state-agnostic: unlike the predictive SafetyMonitor (which
    # skips S1_TRANSIT / S3_RTH separation), a factual breach during any airborne
    # phase is recorded. The stand-ins carry no state -> the recorder never
    # consults one, which IS the property under test.
    rec = _rec(sep=10.0)
    a = [_Agent(0), _Agent(1)]
    _place(rec, a, [(0, 0), (0, 8)], 0.0)
    rec.finalize(0.0)
    assert len(_by_kind(rec.result()[0], "separation")) == 1


def test_cross_layer_separation_is_not_recorded_documents_gap():
    # KNOWN GAP (review #4): different-layer agents are ASSUMED vertically
    # separated, so a horizontal near-miss across layers is NOT recorded. This
    # documents the limitation; it is not a claim of cross-layer coverage.
    stack = LayerStack(_AREA, [], altitudes=[50.0, 100.0], buffer_m=5.0)
    rec = _rec(world=stack, sep=10.0)
    a = [_Agent(0, layer=0), _Agent(1, layer=1)]
    _place(rec, a, [(0, 0), (0, 5)], 0.0)    # 5 m apart horizontally, different layers
    rec.finalize(0.0)
    assert _by_kind(rec.result()[0], "separation") == []


# --------------------------------------------------------------------------- #
# 7.6 speed: commanded (soft), in-place turn, multi-segment hidden peak (hard) #
# --------------------------------------------------------------------------- #
def _speed_tick(rec, agent, leg, pre, post, t):
    agent._legs = [leg]
    agent._leg_idx = 0
    agent._t = 0.0
    agent.pose = Pose(pre[0], pre[1], 0.0)
    rec.snapshot([agent])
    agent.pose = Pose(post[0], post[1], 0.0)
    rec.observe([agent], t)


def test_in_place_turn_reconvergence_is_soft():
    rec = _rec(v_cruise=21.0)                 # dt = 1 s
    a = _Agent(0)
    turn = Path.from_segments([inplace_turn_segment(Pose(0, 0, 0), math.pi / 2, 1.0,
                                                    ManeuverType.TURN)])
    # commanded translational speed 0 (in-place), executed displacement 6 m/s
    # (the off-path re-convergence floor) -> SOFT, well under the 21 m/s envelope.
    _speed_tick(rec, a, turn, (0, 0), (6, 0), 0.0)
    rec.finalize(0.0)
    v = _by_kind(rec.result()[0], "speed")
    assert len(v) == 1
    assert v[0].severity == "soft"
    assert v[0].peak_speed_m_s == 6.0
    assert v[0].limit_m_s == 0.0


def test_zero_command_soft_episode_closes_on_compliant_flight():
    # A soft episode opened on an in-place turn (commanded 0) must CLOSE once the
    # drone resumes compliant flight at a positive command -- recovery is judged
    # against the current tick, not the stored limit=0 -- and a later breach must
    # then be a NEW record (SafetyViolation contract).
    rec = _rec(v_cruise=21.0)
    a = _Agent(0)
    turn = Path.from_segments([inplace_turn_segment(Pose(0, 0, 0), math.pi / 2, 1.0,
                                                    ManeuverType.TURN)])
    cruise = Path.from_segments([straight_segment(Pose(6, 0, 0), 12.0,
                                                  ManeuverType.CRUISE, 12.0)])
    _speed_tick(rec, a, turn, (0, 0), (6, 0), 0.0)     # soft opens (limit 0)
    _speed_tick(rec, a, cruise, (6, 0), (18, 0), 1.0)  # compliant 12 m/s -> closes
    _speed_tick(rec, a, turn, (18, 0), (24, 0), 2.0)   # fresh soft breach
    rec.finalize(2.0)
    v = _by_kind(rec.result()[0], "speed")
    assert len(v) == 2                                  # two distinct records, not one stale
    assert v[0].ended_reason == "recovered"
    assert v[0].t_start == 0.0 and v[0].t_end == 0.0
    assert all(ep.severity == "soft" for ep in v)


def test_multi_segment_peak_is_not_hidden_by_the_average():
    rec = _rec(v_cruise=21.0)
    a = _Agent(0)
    # A 25 m/s straight (0.2 s, 5 m) then a 0.8 s in-place yaw: one dt spans both.
    seg1 = straight_segment(Pose(0, 0, 0), 5.0, ManeuverType.CRUISE, 25.0)
    seg2 = inplace_turn_segment(seg1.end, 1.0, 1.25, ManeuverType.TURN)
    leg = Path.from_segments([seg1, seg2])
    # executed displacement over the whole dt is only 5 m -> v_disp = 5 m/s, which
    # would hide the 25 m/s peak; the crossed-segment peak surfaces it as HARD.
    _speed_tick(rec, a, leg, (0, 0), (5, 0), 0.0)
    rec.finalize(0.0)
    v = _by_kind(rec.result()[0], "speed")
    assert len(v) == 1
    assert v[0].severity == "hard"
    assert v[0].peak_speed_m_s == 25.0        # the within-tick peak, not the 5 m/s average
    assert v[0].limit_m_s == 21.0


# --------------------------------------------------------------------------- #
# 7.7 mission-end open episode; agent leaving the observed set                 #
# --------------------------------------------------------------------------- #
def test_open_episode_at_mission_end_is_flagged():
    rec = _rec(sep=10.0)
    a = [_Agent(0), _Agent(1)]
    _place(rec, a, [(0, 0), (0, 9)], 0.0)
    rec.finalize(0.0)
    v = _by_kind(rec.result()[0], "separation")
    assert len(v) == 1
    assert v[0].ended_reason == "mission_end"
    assert v[0].truncated_at_end is True


def test_agent_leaving_observed_set_closes_episode_without_truncation():
    rec = _rec(sep=10.0)
    a0, a1 = _Agent(0), _Agent(1)
    _place(rec, [a0, a1], [(0, 0), (0, 9)], 0.0)   # both airborne: breach opens
    # next tick a1 has failed/landed -> absent from the observed (airborne) set
    a0.pose = Pose(0, 0, 0)
    rec.snapshot([a0])
    rec.observe([a0], 1.0)
    rec.finalize(1.0)
    v = _by_kind(rec.result()[0], "separation")
    assert len(v) == 1
    assert v[0].ended_reason == "left_observed_set"
    assert v[0].truncated_at_end is False
    assert v[0].t_end == 0.0                  # closed at its own last violating sample
    assert v[0].duration_s == 1 * _DT


# --------------------------------------------------------------------------- #
# 7.9 dt granularity: a between-samples crossing is NOT detected (limitation)  #
# --------------------------------------------------------------------------- #
def test_between_samples_crossing_is_missed_documents_dt_limit():
    # Two drones swap sides between two samples, passing through each other at the
    # un-sampled midpoint. Sampled monitoring cannot see it -- asserted as a
    # documented limitation, not sold as continuous collision detection.
    rec = _rec(sep=10.0)
    a = [_Agent(0), _Agent(1)]
    _place(rec, a, [(0, 0), (0, 20)], 0.0)   # 20 m apart
    _place(rec, a, [(0, 20), (0, 0)], 1.0)   # swapped: still 20 m apart at the sample
    rec.finalize(1.0)
    assert _by_kind(rec.result()[0], "separation") == []
