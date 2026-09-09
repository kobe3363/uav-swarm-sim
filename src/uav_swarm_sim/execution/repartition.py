"""EXP-08: in-flight re-partition of the REMAINING work (D-8, D-11, D-12, D-2a).

One trigger set, one eligibility rule, one ordering -- identical for every
decomposition algorithm. Only the partitioning METHOD differs. That is the whole
deliverable: without an in-flight re-partition the two Lloyd arms compute the
identical partition from identical t=0 inputs and the comparison axis measures
nothing, because ``energy_balance.budget_j`` is a function of the candidate zone
and of the drone's live pose and charge (D-2a).

Why this is not ``Redistributor``
---------------------------------
``Redistributor`` is polygon-shaped by construction, not by convenience: it pools
``unary_union`` of the affected drones' zone polygons and hands that down as
``target_area``. Under D-8 the pool is a different set -- the raster's uncovered
cells, whoever owned them -- and the grid partitioners' work atom is a cell, not
a TGC region. ``Redistributor`` cannot produce the right set without being gutted,
so it is left untouched and keeps serving the flag-off path byte for byte. Its
documented trigger contract ({FAILURE, NEW_TASK}) is likewise untouched.

What each decomposer receives here
----------------------------------
``decompose(tgc, env, views, target_area)`` where ``views`` are LIVE
``DroneStateView``s of the eligible executors in ascending id order, and

* ``target_area = None`` when the decomposer declares ``partitions_raster_work``
  -- it reads the uncovered cells itself and refuses a sub-area;
* ``target_area = raster.uncovered_plannable_geometry`` otherwise -- which, in
  the TGC/Voronoi/k-means contract, says exactly the same thing: partition the
  remaining work only.

Which of the two is decided by the decomposer's own declaration, never by testing
its class.

Nothing here is silent
----------------------
Every attempt produces a record, including the attempts deliberately NOT applied
(no eligible executor, no remaining work, no progress), so a run can never be
read as "no re-partition happened" when one was refused. Every invariant is a
``raise``: ``python -O`` strips ``assert``, and a stripped conservation check is
how covered ground gets re-issued in silence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..infrastructure.core_types import CoveragePlan, Partition
from ..infrastructure.enums import AgentState
from ..planning.coverage_path import boustrophedon
from ..planning.visibility_router import RouteUnavailable

# Areas are compared after float summation over many clipped cells.
_AREA_REL_TOL = 1e-9
_AREA_ABS_TOL = 1e-6

# Why an attempt was or was not applied. Recorded verbatim in the run output.
APPLIED = "applied"
NO_ELIGIBLE_EXECUTOR = "no_eligible_executor"
NO_REMAINING_WORK = "no_remaining_work"
NO_PROGRESS = "no_progress"
CANDIDATE_REJECTED = "candidate_rejected"


@dataclass
class RepartitionRecord:
    """One re-partition ATTEMPT, applied or not."""

    revision: int
    t_s: float
    causes: tuple[tuple[str, int | None], ...]
    reason: str
    applied: bool
    algorithm: str | None = None
    decomposer_class: str | None = None
    rng_stream: str | None = None
    executors: tuple[int, ...] = ()
    excluded: dict[str, tuple[int, ...]] = field(default_factory=dict)
    cells_before: int = 0
    cells_assigned: int = 0
    cells_unassigned: int = 0
    area_before_m2: float = 0.0
    area_assigned_m2: float = 0.0
    plan_time_s: float = 0.0
    rejection: str | None = None

    def to_json(self) -> dict:
        return {
            "revision": self.revision,
            "t_s": self.t_s,
            "causes": [list(c) for c in self.causes],
            "reason": self.reason,
            "applied": self.applied,
            "algorithm": self.algorithm,
            "decomposer_class": self.decomposer_class,
            "rng_stream": self.rng_stream,
            "executors": list(self.executors),
            "excluded": {k: list(v) for k, v in self.excluded.items()},
            "cells_before": self.cells_before,
            "cells_assigned": self.cells_assigned,
            # Legal (nobody can reach them) but never silent: this is the work
            # this revision leaves uncovered on purpose.
            "cells_unassigned": self.cells_unassigned,
            "area_before_m2": self.area_before_m2,
            "area_assigned_m2": self.area_assigned_m2,
            "plan_time_s": self.plan_time_s,
            "rejection": self.rejection,
        }


@dataclass(frozen=True)
class RepartitionAttempt:
    """Result of one attempt. ``staged`` is None unless it is to be applied."""

    record: RepartitionRecord
    staged: tuple[Partition, dict[int, CoveragePlan], tuple[tuple[object, object], ...]] | None = None
    fingerprint: tuple[int, frozenset[int]] | None = None


def eligible_executors(fleet) -> list:
    """Drones that may be GIVEN work by a re-partition, in ascending id order.

    This is deliberately NOT ``fleet.active()``. Active includes drones that are
    flying home and drones in avoidance, and handing either of them a new zone is
    the failure mode D-12 names. Three exclusions, applied identically to both
    arms:

    * failed (``S_FAIL``) and retired (``S_LANDED``) -- ``fleet.workers()``;
    * committed to a return (``S3_RTH``) -- a re-partition must never cancel a
      return the energy or safety logic ordered. A drone that has finished its
      zone is NOT in this state when it is offered work: it announced
      ZONE_COMPLETE and the FSM held the return for one tick, precisely so the
      offer can be made from S2_MISSION/S_FERRY instead;
    * escalated out of avoidance toward home (``S_OBS`` whose ``_obs_return`` is
      ``S3_RTH``) -- the boxed-in case, which is a committed return wearing a
      different state.

    Eligibility is purely lifecycle. It carries NO energy predicate on purpose:
    an engine-level energy gate would be a new RTH policy (EXP-09 scope) and it
    would erase the very axis under test -- the ENERGY arm's advantage is that it
    PLANS with slack, and hiding that behind a shared gate would flatter both
    arms equally. Energy feasibility stays with RthCalculator at execution time,
    exactly as it is today.
    """
    out = []
    for agent in fleet.workers():
        if agent.state not in agent.RETASKABLE:
            continue
        if (agent.state is AgentState.S_OBS
                and getattr(agent, "_obs_return", None) is AgentState.S3_RTH):
            continue
        out.append(agent)
    return sorted(out, key=lambda a: a.id)


def _exclusion_report(fleet, chosen) -> dict[str, tuple[int, ...]]:
    """Why every other drone was left out, by name. A count would not let a
    reader tell a retired fleet from a fleet that is all flying home."""
    picked = {a.id for a in chosen}
    buckets: dict[str, list[int]] = {"failed": [], "retired": [], "rth_committed": [],
                                     "other": []}
    for aid, agent in sorted(fleet.agents.items()):
        if aid in picked:
            continue
        if agent.state is AgentState.S_FAIL:
            buckets["failed"].append(aid)
        elif getattr(agent, "retired", False):
            buckets["retired"].append(aid)
        elif agent.state is AgentState.S3_RTH or (
            agent.state is AgentState.S_OBS
            and getattr(agent, "_obs_return", None) is AgentState.S3_RTH
        ):
            buckets["rth_committed"].append(aid)
        else:
            buckets["other"].append(aid)
    return {k: tuple(v) for k, v in buckets.items() if v}


class Repartitioner:
    """Re-partitions the raster's remaining work among the eligible executors."""

    def __init__(
        self, *, decomposer, raster, env, tgc, motion, em, spec, coverage,
        altitude_m: float, plan_transit, entry_pose, rth=None,
        rng_stream: str | None = None,
    ) -> None:
        self._dec = decomposer
        self._raster = raster
        self._env = env
        self._tgc = tgc
        self._motion = motion
        self._em = em
        self._spec = spec
        self._coverage = coverage
        self._altitude_m = altitude_m
        self._plan_transit = plan_transit    # (from, to) -> Path (engine's)
        self._entry_pose = entry_pose        # (plan, legacy_entry) -> Pose (engine's)
        self._rth = rth
        self._rng_stream = rng_stream
        self.revision = 0
        # Progress guard state: the pair that describes what the LAST APPLIED
        # revision saw. A new revision runs only if the pair has moved.
        self._last_applied: tuple[int, frozenset[int]] | None = None

    # ------------------------------------------------------------------ #
    @property
    def decomposer(self):
        """The decomposer this re-partitioner actually runs.

        Public and read-only for the same reason ``Redistributor`` exposes one --
        a consumer reporting which algorithm produced the current zones must be
        able to ask. Unlike that one, this is always the RUN's decomposer: CVT
        stays CVT, ENERGY stays ENERGY, and there is no fallback to weighted TGC.
        """
        return self._dec

    def attempt(self, fleet, t: float, causes) -> RepartitionAttempt:
        """Plan one revision. Mutates NOTHING -- the engine applies the result.

        Everything that can fail happens here, before a single agent is touched,
        so a raise anywhere leaves the old partition, the old plans and every
        agent's progress exactly as they were. There is no half-applied state to
        recover from, by construction rather than by try/except.
        """
        self.revision += 1
        causes = tuple(causes)
        executors = eligible_executors(fleet)
        # Taken before the executor check on purpose: an attempt refused for
        # want of an executor still has to say how much work it walked away
        # from. It is a pure read (coverage_raster.uncovered_plannable_cells).
        snapshot = self._raster.uncovered_plannable_cells()
        n_before = len(snapshot)
        area_before = float(snapshot.areas_m2.sum())

        algorithm = getattr(self._dec.name, "value", str(self._dec.name))

        def _record(reason, *, rejection: str | None = None, plan_time: float = 0.0):
            # A refused attempt assigned nothing, so ALL the remaining work is
            # what it left unassigned. Reporting 0 there would read as "nothing
            # was left over", which is the opposite of what happened.
            return RepartitionAttempt(RepartitionRecord(
                revision=self.revision, t_s=float(t), causes=causes,
                reason=reason, applied=False,
                algorithm=algorithm, decomposer_class=type(self._dec).__name__,
                rng_stream=self._rng_stream, rejection=rejection, plan_time_s=plan_time,
                executors=tuple(a.id for a in executors),
                excluded=_exclusion_report(fleet, executors),
                cells_before=n_before, cells_assigned=0,
                cells_unassigned=n_before, area_before_m2=area_before,
            ))

        if not executors:
            # EXP-04 owns what happens next (the fleet settles and the outcome is
            # decided on raster coverage). Recorded rather than returned silently,
            # which is what the legacy path does today.
            return _record(NO_ELIGIBLE_EXECUTOR)
        if n_before == 0:
            # Nothing left to divide. Re-partitioning anyway would reset every
            # drone's coverage progress to serve zero cells.
            return _record(NO_REMAINING_WORK)

        fingerprint = (n_before, frozenset(a.id for a in executors))
        if fingerprint == self._last_applied:
            # Neither the work nor the executor set has moved since the last
            # applied revision, so this one would reproduce it very nearly while
            # resetting everyone's progress. Both components change only through
            # a real physical event, which is what bounds repeated triggers.
            #
            # Deliberately conservative, and the cost is worth stating: the
            # drones' POSES are not in the fingerprint, so a revision that would
            # differ only because everyone has moved is refused. Including poses
            # would make the fingerprint change every tick and remove the bound
            # entirely. In practice the guard almost never bites while anyone is
            # covering -- cells are being credited every tick, so n_before falls
            # continuously -- and it bites exactly when nothing is being covered
            # (everyone transiting or idle), which is when a re-partition would
            # be pure churn.
            return _record(NO_PROGRESS)

        views = [a.view() for a in executors]
        target = (None if self._dec.partitions_raster_work
                  else self._raster.uncovered_plannable_geometry)

        covered_before = self._raster.plannable_covered_area_m2
        counters = self._save_rth_counters()
        # plan_time_s feeds the SAME replan_times aggregate as the legacy
        # Redistributor.last_replan_time_s, so it must span the same scope or the
        # aggregate compares two different things. Legacy spans decomposition
        # plus boustrophedon plan construction and stops before the engine plans
        # transits, so the sweep planning below is added in and _plan_transit is
        # left out. The conservation guards are excluded too -- legacy has no
        # such step, and timing it would again make the two incomparable.
        t0 = time.perf_counter()
        try:
            partition = self._dec.decompose(self._tgc, self._env, views,
                                            target_area=target)
        finally:
            self._restore_rth_counters(counters)
        plan_time = time.perf_counter() - t0

        n_assigned, area_assigned = self._verify(partition, snapshot, covered_before,
                                                 area_before)

        # Staging: build every plan and every transit BEFORE anything is applied.
        plans: dict[int, CoveragePlan] = {}
        staged = []
        try:
            for agent in executors:
                zone = partition.zones.get(agent.id)
                if zone is None:
                    raise AssertionError(
                        f"re-partition returned no zone for eligible executor "
                        f"{agent.id}; an empty zone is legal, a missing one is not"
                    )
                zone.layer = agent.layer
                sweep_t0 = time.perf_counter()
                plan = boustrophedon(zone, self._spec, self._motion, self._em,
                                     env=self._env, coverage=self._coverage,
                                     altitude_m=self._altitude_m)
                plan_time += time.perf_counter() - sweep_t0
                entry = self._entry_pose(plan, zone.entry_pose)
                origin = (agent._coherent.retask_origin()
                          if getattr(agent, "_coherent", None) is not None
                          else agent.pose)
                transit = self._plan_transit(origin, entry)
                # Validation includes coherent continuity, every productive
                # endpoint's individual RTH, and the immediate energy bundle.
                # It is deliberately complete before any agent is touched.
                prepared = agent.prepare_retask(plan, transit, self.revision)
                plans[agent.id] = plan
                staged.append((agent, prepared))
        except RouteUnavailable as exc:
            return _record(CANDIDATE_REJECTED, rejection=str(exc), plan_time=plan_time)

        record = RepartitionRecord(
            revision=self.revision, t_s=float(t), causes=causes, reason=APPLIED,
            applied=True,
            algorithm=algorithm,
            decomposer_class=type(self._dec).__name__,
            rng_stream=self._rng_stream,
            executors=tuple(a.id for a in executors),
            excluded=_exclusion_report(fleet, executors),
            cells_before=n_before, cells_assigned=n_assigned,
            cells_unassigned=n_before - n_assigned,
            area_before_m2=area_before, area_assigned_m2=area_assigned,
            plan_time_s=plan_time,
        )
        return RepartitionAttempt(record, (partition, plans, tuple(staged)), fingerprint)

    def mark_applied(self, attempt: RepartitionAttempt) -> None:
        """Advance the progress guard only after the engine commits the revision."""
        if attempt.fingerprint is not None:
            self._last_applied = attempt.fingerprint

    # ------------------------------------------------------------------ #
    # invariants -- every one a raise, never an assert                    #
    # ------------------------------------------------------------------ #
    def _verify(self, partition, snapshot, covered_before, area_before):
        """Four guarantees about the proposed zones, checked against the RASTER
        rather than against the decomposer's own report of itself.

        The grid partitioner already asserts internal conservation over the cell
        set it was handed. That is a different claim: this checks the link
        between what the decomposer produced and what the raster actually holds,
        so a decomposer that reports conservation over the wrong set is caught.
        """
        if self._raster.plannable_covered_area_m2 != covered_before:
            raise AssertionError(
                "re-partition changed raster coverage; planning must be a pure "
                f"read ({covered_before!r} -> "
                f"{self._raster.plannable_covered_area_m2!r})"
            )

        in_snapshot = set(snapshot.indices.tolist())
        claimed: set[int] = set()
        total_area = 0.0
        occupied: list[tuple[int, object]] = []
        for drone_id, zone in sorted(partition.zones.items()):
            geometry = zone.polygon
            if geometry.is_empty:
                continue
            total_area += float(geometry.area)
            uncovered_hits, covered_hits = self._raster.plannable_cells_within(geometry)

            # (1) no covered ground is ever re-issued.
            if len(covered_hits):
                raise AssertionError(
                    f"re-partition gave drone {drone_id} {len(covered_hits)} "
                    "already-covered cell(s); completed work never returns to "
                    "the pool"
                )
            # (2) every claimed cell was in the snapshot this revision read.
            hits = set(uncovered_hits.tolist())
            outside = hits - in_snapshot
            if outside:
                raise AssertionError(
                    f"re-partition gave drone {drone_id} {len(outside)} cell(s) "
                    "that were not in the remaining-work snapshot"
                )
            claimed |= hits
            occupied.append((drone_id, geometry))

        # (3) no GROUND is given to two drones.
        #
        # Deliberately an area test, not a cell-index test. Zones produced by the
        # grid partitioners are unions of whole cells, so a cell index identifies
        # its owner unambiguously there -- but a TGC/Voronoi/k-means zone is a
        # clipped REGION whose edge can run through the middle of a cell, and a
        # cell whose surface point lands exactly on the shared edge of two such
        # zones is ``covers``-inside both while no ground is shared at all. Two
        # zones that merely touch intersect in a line, whose area is zero;
        # genuine double-tasking has positive area. The area test says what is
        # actually meant and is right for both decomposer families.
        tolerance = max(_AREA_ABS_TOL, area_before * _AREA_REL_TOL)
        for i, (left_id, left) in enumerate(occupied):
            for right_id, right in occupied[i + 1:]:
                if not left.envelope.intersects(right.envelope):
                    continue
                overlap = float(left.intersection(right).area)
                if overlap > tolerance:
                    raise AssertionError(
                        f"re-partition gave {overlap!r} m2 to BOTH drone "
                        f"{left_id} and drone {right_id}; ground is owned once"
                    )

        # (4) no area is invented. Cells left unassigned are legal and reported;
        # area appearing from nowhere is not.
        if total_area > area_before + tolerance:
            raise AssertionError(
                f"re-partition assigned {total_area!r} m2 out of a remaining "
                f"{area_before!r} m2"
            )
        # A cell straddled by a zone edge is counted ONCE: this is the set of
        # cells some drone will fly over, not a sum of per-zone tallies.
        return len(claimed), total_area

    # ------------------------------------------------------------------ #
    def _save_rth_counters(self):
        """n_map_hits / n_map_fallbacks / n_route_fallbacks are REPORTED metrics
        (experiments/run_rth_ab.py). The energy weight source queries return costs
        many times per re-partition, so planning must not move them -- the same
        save/restore the t=0 diagnostics use."""
        if self._rth is None:
            return None
        return (self._rth.n_map_hits, self._rth.n_map_fallbacks,
                self._rth.n_route_fallbacks)

    def _restore_rth_counters(self, counters) -> None:
        if counters is None or self._rth is None:
            return
        (self._rth.n_map_hits, self._rth.n_map_fallbacks,
         self._rth.n_route_fallbacks) = counters
