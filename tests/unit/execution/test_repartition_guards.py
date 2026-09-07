"""EXP-08: the eligibility rule, the four conservation guards, and atomicity.

Every guard here is a ``raise``. ``python -O`` strips ``assert``, and a stripped
conservation check is exactly how already-covered ground gets re-issued in
silence -- that was defect 6 of EXP-07, and the last test in this file proves the
guards survive ``-O``.

Expected cell counts and areas are computed from the fixture geometry, never read
back from the code under test.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest
from shapely.geometry import box

from uav_swarm_sim.execution.repartition import (
    NO_ELIGIBLE_EXECUTOR,
    NO_PROGRESS,
    NO_REMAINING_WORK,
    Repartitioner,
    eligible_executors,
)
from uav_swarm_sim.infrastructure.core_types import (
    DecompositionAlgo,
    DroneStateView,
    Partition,
    Pose,
    Zone,
)
from uav_swarm_sim.infrastructure.enums import AgentState
from uav_swarm_sim.planning.coverage_raster import CoverageRaster

S = AgentState

# 200 x 100 survey at 10 m cells -> 20 x 10 = 200 plannable cells, 20 000 m^2.
AREA = box(0.0, 0.0, 200.0, 100.0)
CELL_M = 10.0
TOTAL_CELLS = 20 * 10
TOTAL_AREA = 200.0 * 100.0


# --------------------------------------------------------------------------- #
# doubles                                                                      #
# --------------------------------------------------------------------------- #
class _Agent:
    RETASKABLE = frozenset({S.S0_IDLE, S.S1_TRANSIT, S.S2_MISSION, S.S_FERRY,
                            S.S_OBS, S.S_SWAP})

    def __init__(self, aid, state=S.S2_MISSION, retired=False, obs_return=None,
                 pose=None, layer=0):
        self.id = aid
        self.state = state
        self.retired = retired
        self._obs_return = obs_return
        self.pose = pose or Pose(0.0, 0.0, 0.0)
        self.layer = layer
        self.retasked = []

    def view(self):
        return DroneStateView(self.id, 1.0, self.pose, self.layer,
                              self.state.is_airborne)

    def retask(self, plan, transit, t, bus):
        self.retasked.append((plan, transit, t))


class _Fleet:
    def __init__(self, agents):
        self.agents = {a.id: a for a in agents}

    def active(self):
        return [a for a in self.agents.values() if a.state is not S.S_FAIL]

    def workers(self):
        return [a for a in self.active() if not a.retired]


class _Decomposer:
    """Returns a partition the test dictates, so the guards can be aimed."""

    name = DecompositionAlgo.LLOYD_CVT
    partitions_raster_work = True

    def __init__(self, zones_for):
        self._zones_for = zones_for
        self.calls = []

    def decompose(self, tgc, env, drones, target_area=None):
        self.calls.append((tuple(d.id for d in drones), target_area))
        return Partition(self.name, self._zones_for(drones), 0.0)


def _raster(covered_segments=()):
    raster = CoverageRaster(AREA, AREA, CELL_M)
    for a, b in covered_segments:
        raster.record_segment(a, b, 20.0, 0.1)
    return raster


def _zone(drone_id, geom):
    return Zone(drone_id, [], geom, Pose(0.0, 0.0, 0.0))


def _repartitioner(raster, decomposer, **kw):
    return Repartitioner(
        decomposer=decomposer, raster=raster, env=None, tgc=None,
        motion=None, em=None, spec=None, coverage=None, altitude_m=100.0,
        plan_transit=kw.pop("plan_transit", lambda a, b: "transit"),
        entry_pose=kw.pop("entry_pose", lambda plan, legacy: legacy),
        **kw,
    )


class _StubPlan:
    waypoints: list = []


@pytest.fixture(autouse=True)
def _no_real_boustrophedon(monkeypatch):
    """The guards are the subject here, not the sweep planner."""
    monkeypatch.setattr("uav_swarm_sim.execution.repartition.boustrophedon",
                        lambda *a, **k: _StubPlan())


# --------------------------------------------------------------------------- #
# A. eligibility                                                               #
# --------------------------------------------------------------------------- #
def test_eligibility_is_not_fleet_active():
    """fleet.active() includes drones flying home and drones in avoidance.
    Handing either of them a new zone is the failure mode D-12 names."""
    agents = [
        _Agent(0, S.S2_MISSION),
        _Agent(1, S.S_FERRY),
        _Agent(2, S.S3_RTH),                                  # committed return
        _Agent(3, S.S_FAIL),                                  # lost
        _Agent(4, S.S2_MISSION, retired=True),                # landed terminally
        _Agent(5, S.S_OBS, obs_return=S.S3_RTH),              # boxed-in escalation
        _Agent(6, S.S_OBS, obs_return=S.S2_MISSION),          # ordinary avoidance
        _Agent(7, S.S_SWAP),
        _Agent(8, S.S0_IDLE),
    ]
    fleet = _Fleet(agents)
    assert [a.id for a in eligible_executors(fleet)] == [0, 1, 6, 7, 8]
    # and it really is narrower than active()
    assert {a.id for a in fleet.active()} - {a.id for a in eligible_executors(fleet)} \
        == {2, 4, 5}


def test_an_avoiding_drone_heading_home_is_a_committed_return_in_disguise():
    fleet = _Fleet([_Agent(0, S.S_OBS, obs_return=S.S3_RTH)])
    assert eligible_executors(fleet) == []


# --------------------------------------------------------------------------- #
# B. refusals are recorded, never silent                                       #
# --------------------------------------------------------------------------- #
def test_no_eligible_executor_is_recorded_with_the_work_it_walked_away_from():
    raster = _raster()
    fleet = _Fleet([_Agent(0, S.S3_RTH)])
    attempt = _repartitioner(raster, _Decomposer(lambda d: {})).attempt(fleet, 5.0, ())

    record = attempt.record
    assert attempt.staged is None
    assert record.reason == NO_ELIGIBLE_EXECUTOR and record.applied is False
    assert record.executors == ()
    assert record.excluded["rth_committed"] == (0,)
    # the whole remaining pool is what this attempt left unassigned; reporting 0
    # there would read as "nothing was left over"
    assert record.cells_before == TOTAL_CELLS
    assert record.cells_unassigned == TOTAL_CELLS
    assert record.area_before_m2 == pytest.approx(TOTAL_AREA, rel=1e-12)


def test_no_remaining_work_does_not_reset_everyone_to_serve_zero_cells():
    raster = _raster()
    # cover the whole survey: one sweep per row, 20 m wide footprint over 10 m cells
    for row in range(10):
        y = 5.0 + 10.0 * row
        raster.record_segment(Pose(-20.0, y, 0.0), Pose(220.0, y, 0.0), 20.0, 0.1)
    assert raster.plannable_coverage_frac == pytest.approx(1.0)

    decomposer = _Decomposer(lambda d: {})
    attempt = _repartitioner(raster, decomposer).attempt(
        _Fleet([_Agent(0)]), 9.0, (("zone_complete", 0),))
    assert attempt.record.reason == NO_REMAINING_WORK
    assert attempt.staged is None
    assert decomposer.calls == [], "the decomposer must not even be asked"


def test_a_repeated_trigger_without_progress_produces_one_revision_then_refusals():
    """Both components of the fingerprint change only through a real physical
    event, which is what bounds the loop."""
    raster = _raster()
    agents = [_Agent(0), _Agent(1)]
    fleet = _Fleet(agents)
    def split(drones):
        """Whatever drones turn up, hand the first one everything and the rest
        an empty zone -- the executor set shrinks partway through this test."""
        return {d.id: _zone(d.id, AREA if i == 0 else box(0.0, 0.0, 0.0, 0.0))
                for i, d in enumerate(drones)}

    decomposer = _Decomposer(split)
    rp = _repartitioner(raster, decomposer)

    first = rp.attempt(fleet, 1.0, (("interval", None),))
    assert first.record.applied is True
    for k in range(2, 6):
        again = rp.attempt(fleet, float(k), (("interval", None),))
        assert again.record.reason == NO_PROGRESS
        assert again.staged is None
    assert len(decomposer.calls) == 1, "no decomposition after the first"

    # a real event -- one drone leaves -- moves the fingerprint and unblocks it
    agents[1].retired = True
    after = rp.attempt(fleet, 9.0, (("uav_retired", 1),))
    assert after.record.applied is True
    assert after.record.executors == (0,)


# --------------------------------------------------------------------------- #
# C. the four conservation guards                                              #
# --------------------------------------------------------------------------- #
def _attempt_with_zones(raster, zones_for, agents=None):
    fleet = _Fleet(agents or [_Agent(0), _Agent(1)])
    return _repartitioner(raster, _Decomposer(zones_for)).attempt(fleet, 1.0, ())


def test_guard_1_an_already_covered_cell_is_never_re_issued():
    """The left half is flown, then a zone is proposed over the whole survey."""
    raster = _raster([(Pose(-20.0, y, 0.0), Pose(100.0, y, 0.0))
                      for y in (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0,
                                85.0, 95.0)])
    assert raster.plannable_coverage_frac == pytest.approx(0.5, abs=0.02)

    with pytest.raises(AssertionError, match="already-covered cell"):
        _attempt_with_zones(raster, lambda d: {
            d[0].id: _zone(d[0].id, AREA),
            d[1].id: _zone(d[1].id, box(0.0, 0.0, 0.0, 0.0)),
        })


def test_guard_2_a_cell_outside_the_snapshot_is_refused():
    """Guard 2 is defence in depth, and this test says so rather than pretending.

    Through the ordinary path it is nearly unreachable: the snapshot IS the
    raster's uncovered set at the instant of the call, so every uncovered cell a
    zone claims is in it by construction. What guard 2 catches is a decomposer
    answering about a DIFFERENT cell set than the one this revision read -- so
    that is what is constructed here, by verifying a whole-survey zone against a
    deliberately truncated snapshot.

    The earlier version of this test covered a row and proposed a zone over the
    whole survey. Guard 1 fires first on that input, so it passed on the wrong
    guard while its name promised this one, and a `match` alternation hid the
    substitution. (Found in review of this PR.)
    """
    raster = _raster()
    full = raster.uncovered_plannable_cells()
    assert len(full) == TOTAL_CELLS
    truncated = replace(full, indices=full.indices[:TOTAL_CELLS // 2])

    rp = _repartitioner(raster, _Decomposer(lambda d: {}))
    partition = Partition(DecompositionAlgo.LLOYD_CVT, {0: _zone(0, AREA)}, 0.0)
    covered = raster.plannable_covered_area_m2

    with pytest.raises(AssertionError, match="not in the remaining-work snapshot"):
        rp._verify(partition, truncated, covered, TOTAL_AREA)

    # against the FULL snapshot the very same zone passes, so the failure above
    # is guard 2 and nothing else
    assert rp._verify(partition, full, covered, TOTAL_AREA) == (TOTAL_CELLS,
                                                               TOTAL_AREA)


def test_guard_3_the_same_ground_may_not_go_to_two_drones():
    raster = _raster()
    left = box(0.0, 0.0, 120.0, 100.0)
    right = box(80.0, 0.0, 200.0, 100.0)          # 80..120 is claimed twice
    with pytest.raises(AssertionError, match="to BOTH drone 0 and drone 1"):
        _attempt_with_zones(raster, lambda d: {
            d[0].id: _zone(d[0].id, left),
            d[1].id: _zone(d[1].id, right),
        })
    # the overlap really is 40 m x 100 m, computed here from the fixture
    assert left.intersection(right).area == pytest.approx(4000.0)


def test_guard_3_does_not_fire_on_zones_that_merely_touch():
    """The reason this guard is an AREA test and not a cell-index test.

    A TGC/Voronoi/k-means zone is a clipped REGION whose edge can run through the
    middle of a cell; a cell whose surface point lands exactly on the shared edge
    of two such zones is ``covers``-inside both while no ground is shared at all.
    Two touching zones intersect in a LINE, whose area is zero. An index-based
    check reported this as double-tasking -- it was found by this guard firing on
    a real k-means run, not by inspection."""
    raster = _raster()
    left = box(0.0, 0.0, 100.0, 100.0)
    right = box(100.0, 0.0, 200.0, 100.0)
    assert left.intersection(right).area == 0.0
    assert not left.intersection(right).is_empty, "they really do touch"

    attempt = _attempt_with_zones(raster, lambda d: {
        d[0].id: _zone(d[0].id, left),
        d[1].id: _zone(d[1].id, right),
    })
    assert attempt.record.applied is True
    # a cell on the seam is counted ONCE, not twice
    assert attempt.record.cells_assigned == TOTAL_CELLS


def test_guard_4_area_may_not_appear_from_nowhere():
    """Guards 1-3 are about cell membership; a zone can still be geometrically
    larger than the pool without its surface points betraying it."""
    raster = _raster()
    with pytest.raises(AssertionError, match="assigned .* out of a remaining"):
        _attempt_with_zones(raster, lambda d: {
            d[0].id: _zone(d[0].id, box(-500.0, -500.0, 200.0, 100.0)),
            d[1].id: _zone(d[1].id, box(0.0, 0.0, 0.0, 0.0)),
        })


def test_a_missing_zone_for_an_eligible_executor_is_refused():
    """An EMPTY zone is legal -- it means "no work for you, go home". A MISSING
    one means the decomposer forgot a drone, which would leave it flying an old
    plan over ground now owned by someone else."""
    raster = _raster()
    with pytest.raises(AssertionError, match="no zone for eligible executor"):
        _attempt_with_zones(raster, lambda d: {d[0].id: _zone(d[0].id, AREA)})


def test_planning_may_not_touch_the_raster():
    raster = _raster()

    class _Cheat(_Decomposer):
        def decompose(self, tgc, env, drones, target_area=None):
            raster.record_segment(Pose(-20.0, 5.0, 0.0), Pose(220.0, 5.0, 0.0),
                                  20.0, 0.1)
            return Partition(self.name, {d.id: _zone(d.id, box(0.0, 0.0, 0.0, 0.0))
                                         for d in drones}, 0.0)

    rp = _repartitioner(raster, _Cheat(lambda d: {}))
    with pytest.raises(AssertionError, match="changed raster coverage"):
        rp.attempt(_Fleet([_Agent(0)]), 1.0, ())


# --------------------------------------------------------------------------- #
# D. what the decomposer receives                                              #
# --------------------------------------------------------------------------- #
def test_a_raster_reading_decomposer_is_given_no_target_area():
    raster = _raster()
    decomposer = _Decomposer(lambda d: {d[0].id: _zone(d[0].id, AREA)})
    _repartitioner(raster, decomposer).attempt(_Fleet([_Agent(0)]), 1.0, ())
    ids, target = decomposer.calls[0]
    assert ids == (0,)
    assert target is None


def test_a_polygon_decomposer_is_given_the_remaining_work_as_its_target_area():
    """Same rule, expressed in the other contract's language: partition the
    remaining work only. Chosen by the decomposer's own declaration, never by
    testing its class."""
    raster = _raster()
    raster.record_segment(Pose(-20.0, 5.0, 0.0), Pose(220.0, 5.0, 0.0), 20.0, 0.1)
    remaining = raster.uncovered_plannable_geometry

    decomposer = _Decomposer(lambda d: {d[0].id: _zone(d[0].id, remaining)})
    decomposer.partitions_raster_work = False
    _repartitioner(raster, decomposer).attempt(_Fleet([_Agent(0)]), 1.0, ())
    _ids, target = decomposer.calls[0]
    assert target is not None and target.equals(remaining)


def test_the_views_carry_live_pose_and_the_airborne_bit():
    raster = _raster()
    agent = _Agent(0, S.S2_MISSION, pose=Pose(123.0, 45.0, 0.5))
    captured = {}

    def zones(drones):
        captured["view"] = drones[0]
        return {drones[0].id: _zone(drones[0].id, AREA)}

    _repartitioner(raster, _Decomposer(zones)).attempt(_Fleet([agent]), 1.0, ())
    assert captured["view"].pose == Pose(123.0, 45.0, 0.5)
    assert captured["view"].airborne is True


# --------------------------------------------------------------------------- #
# E. atomicity                                                                 #
# --------------------------------------------------------------------------- #
def test_a_failure_anywhere_in_planning_leaves_every_agent_untouched():
    """attempt() mutates nothing, so a raise cannot half-apply. Proved on the
    transit planner -- the LAST thing staged, after the decomposition and after
    every guard has passed."""
    raster = _raster()
    agents = [_Agent(0), _Agent(1)]
    calls = {"n": 0}

    def exploding_transit(a, b):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("router blew up on the second drone")
        return "transit"

    rp = _repartitioner(raster, _Decomposer(lambda d: {
        d[0].id: _zone(d[0].id, box(0.0, 0.0, 100.0, 100.0)),
        d[1].id: _zone(d[1].id, box(100.0, 0.0, 200.0, 100.0)),
    }), plan_transit=exploding_transit)

    with pytest.raises(RuntimeError, match="second drone"):
        rp.attempt(_Fleet(agents), 1.0, ())
    assert all(a.retasked == [] for a in agents), "no agent may have been re-tasked"


def test_a_successful_attempt_still_re_tasks_nobody_by_itself():
    """Applying is the engine's job. The separation is what makes the apply loop
    computation-free and therefore atomic."""
    raster = _raster()
    agents = [_Agent(0), _Agent(1)]
    attempt = _repartitioner(raster, _Decomposer(lambda d: {
        d[0].id: _zone(d[0].id, box(0.0, 0.0, 100.0, 100.0)),
        d[1].id: _zone(d[1].id, box(100.0, 0.0, 200.0, 100.0)),
    })).attempt(_Fleet(agents), 1.0, ())

    assert attempt.record.applied is True
    assert attempt.staged is not None
    assert all(a.retasked == [] for a in agents)
    # conservation is reported, and the numbers are the fixture's own
    assert attempt.record.cells_before == TOTAL_CELLS
    assert attempt.record.cells_assigned == TOTAL_CELLS
    assert attempt.record.cells_unassigned == 0
    assert attempt.record.area_assigned_m2 == pytest.approx(TOTAL_AREA, rel=1e-12)


def test_an_empty_zone_is_legal_and_reported_as_unassigned_work():
    raster = _raster()
    attempt = _repartitioner(raster, _Decomposer(lambda d: {
        d[0].id: _zone(d[0].id, box(0.0, 0.0, 100.0, 100.0)),
        d[1].id: _zone(d[1].id, box(0.0, 0.0, 0.0, 0.0)),
    })).attempt(_Fleet([_Agent(0), _Agent(1)]), 1.0, ())

    assert attempt.record.applied is True
    assert attempt.record.cells_assigned == TOTAL_CELLS // 2
    assert attempt.record.cells_unassigned == TOTAL_CELLS // 2


# --------------------------------------------------------------------------- #
# F. the guards must survive python -O (EXP-07 defect 6)                       #
# --------------------------------------------------------------------------- #
def test_the_conservation_guards_still_fire_under_dash_O():
    """``assert`` is stripped by -O. Verified by running the real guard in a -O
    subprocess rather than by reading the source and hoping."""
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, "src")
        from shapely.geometry import box
        from uav_swarm_sim.execution.repartition import Repartitioner
        from uav_swarm_sim.infrastructure.core_types import (
            DecompositionAlgo, DroneStateView, Partition, Pose, Zone)
        from uav_swarm_sim.planning.coverage_raster import CoverageRaster

        assert False, "sanity: this line must NOT raise under -O"

        AREA = box(0.0, 0.0, 200.0, 100.0)
        raster = CoverageRaster(AREA, AREA, 10.0)

        class D:
            name = DecompositionAlgo.LLOYD_CVT
            partitions_raster_work = True
            def decompose(self, tgc, env, drones, target_area=None):
                huge = box(-500.0, -500.0, 200.0, 100.0)
                return Partition(self.name,
                                 {0: Zone(0, [], huge, Pose(0.0, 0.0, 0.0))}, 0.0)

        class A:
            RETASKABLE = frozenset()
            id, state, retired, layer = 0, None, False, 0
            pose = Pose(0.0, 0.0, 0.0)
            def view(self):
                return DroneStateView(0, 1.0, self.pose, 0, True)

        class F:
            def __init__(self, a): self.agents = {0: a}
            def active(self): return list(self.agents.values())
            def workers(self): return list(self.agents.values())

        a = A()
        A.RETASKABLE = frozenset({None})          # make the single agent eligible
        rp = Repartitioner(decomposer=D(), raster=raster, env=None, tgc=None,
                           motion=None, em=None, spec=None, coverage=None,
                           altitude_m=100.0, plan_transit=lambda x, y: None,
                           entry_pose=lambda p, l: l)
        try:
            rp.attempt(F(a), 1.0, ())
        except AssertionError as exc:
            print("GUARD_FIRED:", str(exc)[:60])
            sys.exit(0)
        print("GUARD_STRIPPED")
        sys.exit(1)
        """
    )
    result = subprocess.run([sys.executable, "-O", "-c", script],
                            capture_output=True, text=True)
    assert "GUARD_FIRED" in result.stdout, (result.stdout, result.stderr)
    assert result.returncode == 0
