"""EXP-09 opt-in flight execution; legacy Agent stepping remains untouched.

Altitude is the existing separate 1-D ground/flight profile, while routes use
the layer's 2-D coordinates. TAKEOFF/LAND use the existing table-only profiles
(no extra m*g*h or descent credit). Every phase consumes its actual duration.
"""
from __future__ import annotations

import math

from ..infrastructure.core_types import Path, normalize_angle
from ..infrastructure.enums import AgentState as S, ManeuverType as M
from ..physical_model.vertical_segments import takeoff_profile
from ..planning.visibility_router import RouteUnavailable
from .state_machine import Transition


class CoherentFlight:
    def __init__(self, agent):
        self.a = agent
        self.altitude_m = 0.0
        self.holding = False
        self.return_plan = None
        self.rejected_reason: str | None = None
        self.assignment_error: str | None = None
        self.plan_revision = 0
        self._pending_retask_legs: tuple = ()
        self._vertical_leg_start_agl: float | None = None

    def energy(self, path, start=0.0, *, camera=False):
        a = self.a
        return a.em.path_interval_energy(path, start, path.total_duration_s,
                                         a._sensor_power_w if camera else 0.0)

    def reject(self, reason):
        """Never-launched agents are unassigned S0, satisfying _fleet_settled."""
        a = self.a
        self.rejected_reason = str(reason)
        a.plan = None
        a._launch_ready = False
        a._transit = None
        a._cov_legs = []
        a._set_legs([])

    def _validate_paths(self, transit, coverage_legs, *, start_pose=None):
        a = self.a
        paths = [transit, *coverage_legs]
        previous = a.pose if start_pose is None else start_pose
        for i, path in enumerate(paths):
            if path is None or not a.rth.path_clear(path, productive=i > 0 and (i - 1) % 2 == 0):
                raise RouteUnavailable("assignment_buffer_blocked")
            if path.start_pose is not None and (
                math.dist(previous.as_xy(), path.start_pose.as_xy()) > 1e-6
                or abs(normalize_angle(previous.heading - path.start_pose.heading)) > 1e-6
            ):
                raise RouteUnavailable("assignment_discontinuous")
            previous = path.end_pose or previous
        # Check connectivity and real return costs at every productive endpoint;
        # a later runtime skip must never be needed to repair the plan.
        for path in coverage_legs[::2]:
            a.rth.return_plan(path.end_pose or a.pose, base=a.base,
                              altitude_m=a.coverage_altitude_m)

    def validate_assignment(self):
        self._validate_paths(self.a._transit, self.a._cov_legs)

    def _retask_vertical_legs(self):
        """Unflown vertical phase for a prospective plan, without mutation."""
        a = self.a
        if a.state.is_airborne:
            remaining_altitude = max(0.0, a.coverage_altitude_m - self.altitude_m)
            # ``as_path`` contains only the airborne CLIMB segment.  The
            # ground-roll term lives in VerticalProfile.energy and is therefore
            # never re-executed for this already-airborne continuation.
            return ([takeoff_profile(a.spec, a.em, remaining_altitude, at=a.pose).as_path()]
                    if remaining_altitude > 1e-8 else [])
        return [takeoff_profile(
                a.spec, a.em, a.coverage_altitude_m, at=a.pose,
            ).as_path()]

    def retask_origin(self):
        """Where the new horizontal transit starts after any unflown climb."""
        legs = self._retask_vertical_legs()
        return (legs[-1].end_pose if legs and legs[-1].end_pose is not None
                else self.a.pose)

    def validate_retask(self, transit, coverage_legs, revision: int):
        """Return executable incoming legs or reject the candidate untouched."""
        a = self.a
        if revision <= self.plan_revision:
            raise RouteUnavailable("stale_plan_revision")
        vertical = self._retask_vertical_legs()
        incoming = [*vertical, transit]
        # A fixed-wing airborne climb moves horizontally, unlike a multirotor.
        self._validate_paths(transit, coverage_legs, start_pose=self.retask_origin())
        first = coverage_legs[0] if coverage_legs else Path()
        endpoint = first.end_pose or transit.end_pose or a.pose
        required = (
            sum(self.energy(path) for path in incoming)
            + self.energy(first, camera=True)
            + a.rth.return_energy(endpoint, altitude_m=a.coverage_altitude_m, base=a.base)
            + a.rth.reserve_j
        )
        # S_SWAP cannot execute until the existing lifecycle installs a full
        # pack. Validate that post-swap capacity without mutating live state.
        available = (a.battery.capacity_j if a.state is S.S_SWAP
                     else a.battery.level_j)
        if available < required:
            raise RouteUnavailable("retask_energy_budget")
        return incoming

    def accept_plan_revision(self, revision: int, incoming_legs) -> None:
        """Invalidate only plan-local execution state after staged acceptance."""
        self.plan_revision = revision
        self._pending_retask_legs = tuple(incoming_legs)
        self.return_plan = None
        self.assignment_error = None
        self.rejected_reason = None
        self.holding = False

    def discard_pending_retask_legs(self) -> None:
        """Drop an airborne connector that cannot apply after ground lifecycle.

        A retask accepted while swapping or still idle resumes through the
        normal ground launch path.  Keeping the staged airborne connector would
        let a later unrelated S_OBS -> S1 transition fly stale geometry.
        """
        self._pending_retask_legs = ()

    def transition_legs(self, source):
        """Legs for S1 entry: launch only from ground, never after a retask."""
        if source is S.S0_IDLE:
            return self.launch_legs()
        return self.retask_transit_legs(source)

    def retask_transit_legs(self, source):
        if self._pending_retask_legs:
            legs = list(self._pending_retask_legs)
            self._pending_retask_legs = ()
            return legs
        return [self.a._transit] if self.a._transit is not None else []

    def bundle(self):
        """Remaining current strip OR connector plus the following strip.

        No completed work is charged again. At a strip boundary the next call
        evaluates its connector and next productive strip before either moves.
        """
        a = self.a
        k = a._cov_idx
        if k >= len(a._cov_legs):
            return 0.0, a.pose
        end = min(len(a._cov_legs), k + (2 if k % 2 else 1))
        cost = 0.0
        pose = a.pose
        for j in range(k, end):
            path = a._cov_legs[j]
            cost += self.energy(path, a._t if j == k else 0.0, camera=j % 2 == 0)
            pose = path.end_pose or pose
        return cost, pose

    def launch_legs(self):
        a = self.a
        takeoff = takeoff_profile(a.spec, a.em, a.coverage_altitude_m, at=a.base)
        self.altitude_m = 0.0
        self.holding = False
        return [takeoff.as_path(), a._transit]

    def can_launch(self):
        a = self.a
        try:
            self.validate_assignment()
            takeoff = takeoff_profile(a.spec, a.em, a.coverage_altitude_m, at=a.base)
            first = a._cov_legs[0] if a._cov_legs else Path()
            endpoint = first.end_pose or a._transit.end_pose or a.pose
            required = (takeoff.energy_j + self.energy(a._transit)
                        + self.energy(first, camera=True)
                        + a.rth.return_energy(endpoint, altitude_m=a.coverage_altitude_m, base=a.base)
                        + a.rth.reserve_j)
            if a.battery.level_j < required:
                self.reject("launch_energy_budget")
                return False
            return True
        except RouteUnavailable as exc:
            self.reject(exc)
            return False

    def prepare_return(self, t, bus):
        a = self.a
        self.holding = False
        try:
            self.return_plan = a.rth.return_plan(a.pose, base=a.base, altitude_m=self.altitude_m)
        except RouteUnavailable as exc:
            self.return_plan = None
            self.holding = True
            a.report_rth_infeasible(t, bus, None, str(exc))
            # Remain airborne and consume HOVER until the existing depletion
            # lifecycle kills this drone. Other drones continue unaffected.
            return []
        deficit = self.return_plan.energy_j - a.battery.level_j
        if deficit > 0:
            a.report_rth_infeasible(t, bus, deficit, "insufficient_return_energy")
        elif deficit + a.rth.reserve_j > 0:
            a.report_rth_infeasible(t, bus, deficit + a.rth.reserve_j, "reserve_shortfall")
        return [self.return_plan.path]

    def step(self, dt, t, bus):
        a = self.a
        # The engine had one full tick to accept the published completion.  If
        # it did not, spend exactly one physical hover tick before the ordinary
        # FSM return.  RTH/threat pre-emption stays above this convenience hold.
        if a._repartition_hold:
            # A failure is terminal and must pre-empt the one-step convenience
            # hold just as it pre-empts ordinary mission movement below.
            if a._failure and a.state.is_airborne:
                a._repartition_hold = False
                a._apply_transition(Transition(a.state, S.S_FAIL, "failure"), t, bus)
                return
            emergency = a.rth.emergency_frac
            if emergency is None:
                emergency = a.battery._zones.critical
            reason = "terminal_battery" if a.battery.frac < emergency else None
            try:
                if reason is None and a.battery.level_j < (
                    a.rth.return_energy(a.pose, altitude_m=self.altitude_m, base=a.base)
                    + a.rth.reserve_j
                ):
                    reason = "rth_energy"
            except RouteUnavailable:
                reason = "rth_energy"
            if reason is not None:
                a._repartition_hold = False
                a._apply_transition(Transition(a.state, S.S3_RTH, reason), t, bus)
                return
            if a._threat:
                a._repartition_hold = False
                a._apply_transition(Transition(a.state, S.S_OBS, "obstacle_threat"), t, bus)
                return
            a._tick_dynamics(dt, t)
            a._repartition_hold = False
            ctx = a._make_ctx()
            ctx.rth_decision = False
            transition = a.sm.step(ctx)
            if transition is not None:
                a._apply_transition(transition, t + dt, bus)
            return
        remaining = dt
        while remaining > 1e-10:
            a = self.a
            before = (a.state, a._leg_idx, a._cov_idx, a._t)
            used = self._step_once(remaining, t + dt - remaining, bus)
            if used is None or not a.state.is_airborne:
                break
            if a._repartition_hold:
                # Do not consume the hold inside this same simulation tick; the
                # engine must drain ZONE_COMPLETE and stage the revision first.
                break
            remaining -= used
            if used == 0 and before == (a.state, a._leg_idx, a._cov_idx, a._t):
                raise RuntimeError("coherent flight made no progress")

    def _step_once(self, dt, t, bus):
        a = self.a
        used = None
        if a.state is S.S_FAIL:
            return
        if a._failure and a.state.is_airborne:
            a._apply_transition(Transition(a.state, S.S_FAIL, "failure"), t, bus)
            return
        if a.state is S.S0_IDLE:
            a._tick_dynamics(dt, t)
            if a._launch_ready and not self.can_launch():
                return
        elif a.state.is_airborne:
            if self.assignment_error is not None:
                self.assignment_error = None
                a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), t, bus)
            # Check before movement, regardless of legacy timer/1%-battery
            # cadence. A long leg or a long timer cannot overspend RTH energy.
            if a.state in (S.S2_MISSION, S.S_FERRY, S.S1_TRANSIT, S.S_OBS):
                emergency = a.rth.emergency_frac
                if emergency is None:
                    emergency = a.battery._zones.critical
                reason = None
                if a.battery.frac < emergency:
                    reason = "terminal_battery"
                try:
                    if a.state in (S.S2_MISSION, S.S_FERRY):
                        cost, end = self.bundle()
                        if a.battery.level_j < cost + a.rth.return_energy(
                            end, altitude_m=self.altitude_m, base=a.base
                        ) + a.rth.reserve_j:
                            reason = "rth_energy"
                    else:
                        cost = sum(self.energy(p, a._t if i == a._leg_idx else 0)
                                   for i, p in enumerate(a._legs) if i >= a._leg_idx)
                        end = (a._legs[-1].end_pose if a._legs else None) or a.pose
                        if a.battery.level_j < cost + a.rth.return_energy(
                            end, altitude_m=(a.coverage_altitude_m if a.state is S.S1_TRANSIT
                                             else self.altitude_m), base=a.base
                        ) + a.rth.reserve_j:
                            reason = "rth_energy"
                except RouteUnavailable:
                    reason = "rth_energy"
                if reason:
                    a._apply_transition(Transition(a.state, S.S3_RTH, reason), t, bus)
            # Revalidate the actual active path, including any monitor detour.
            # If it no longer connects to our real pose, replan from here; do
            # not use motion.advance's unpriced off-path convergence.
            if a._leg_idx < len(a._legs):
                path = a._legs[a._leg_idx]
                expected = path.pose_at_time(a._t)
                invalid = not a.rth.path_clear(path)
                displaced = expected is not None and (
                    math.dist(expected.as_xy(), a.pose.as_xy()) > 1e-6
                    or abs(normalize_angle(expected.heading - a.pose.heading)) > 1e-6
                )
                if invalid or displaced:
                    if a.state is S.S3_RTH:
                        a._set_legs(self.prepare_return(t, bus))
                    else:
                        a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), t, bus)
            used = self.tick(dt, t)
        # Depletion is evaluated by the engine before a zero-energy drone can
        # acquire a fictitious landed state on this same step.
        if a.state.is_airborne and a.battery.level_j <= 0:
            return
        if a.state is S.S_OBS and a._phase_done():
            a._threat_cleared = True
        a._announce_zone_complete(t + (used or 0), bus)
        ctx = a._make_ctx()
        ctx.rth_decision = False  # handled above before consuming any energy
        if self.holding:
            ctx.threat_flag = False
            ctx.landed_at_base = False
        transition = a.sm.step(ctx)
        if transition is not None:
            if transition.dst in (S.S3_RTH, S.S_OBS):
                a._repartition_hold = False
            a._apply_transition(transition, t + (used or 0), bus)
        return used

    def tick(self, dt, t):
        a = self.a
        if self.holding:
            energy = min(a.battery.level_j, a.em.segment_energy(M.HOVER, dt))
            a.battery.drain(energy)
            a.energy_consumed_j += energy
            return dt
        if a._leg_idx >= len(a._legs):
            return 0.0
        path = a._legs[a._leg_idx]
        if a._t <= 1e-12:
            self._vertical_leg_start_agl = self.altitude_m
        end = min(a._t + dt, path.total_duration_s)
        used = end - a._t
        cursor = 0.0
        productive = (a.state in (S.S2_MISSION, S.S_FERRY) and a._cov_idx % 2 == 0)
        # Walk overlaps for motion and photo events too: yaw must never cause
        # a chord to be photographed or camera power to leak into a turn.
        for seg in path.segments:
            lo, hi = max(a._t, cursor), min(end, cursor + seg.duration_s)
            if hi > lo:
                power = a.em.power(seg.maneuver)
                if productive and seg.maneuver is M.COVERAGE:
                    power += a._sensor_power_w
                # The affordable duration must come from the SAME integral that
                # charges it: path_interval_energy also bills the potential term
                # on a climbing segment, so clamping on propulsion power alone
                # would cut at a point the battery cannot pay for (drain floors
                # at 0, energy_consumed_j does not -> reported energy would
                # exceed what the battery supplied).
                rate = power + a.em.potential_power_w(seg)
                if rate > 0 and rate * (hi - lo) > a.battery.level_j:
                    hi = lo + a.battery.level_j / rate
                    end = hi
                    used = end - a._t
                elapsed = hi - lo
                part = Path((seg,))
                old = part.pose_at_time(lo - cursor)
                new = part.pose_at_time(hi - cursor)
                energy = a.em.path_interval_energy(part, lo - cursor, hi - cursor,
                                                   a._sensor_power_w if productive else 0.0)
                a.battery.drain(energy)
                a.energy_consumed_j += energy
                a.flown_m += math.dist(old.as_xyz(), new.as_xyz())
                a.pose = new
                photo_on = productive and seg.maneuver is M.COVERAGE
                if photo_on and a._photo_tracker is not None:
                    at = t + lo - a._t
                    if not a._photo_tracker.active:
                        a._photo_tracker.start_pass(at, old, a._cov_idx)
                    a._photo_tracker.advance(old, new, at, elapsed, a._cov_idx)
                    if a._coverage_observer is not None:
                        a._coverage_observer(old, new)
                if seg.maneuver in (M.TAKEOFF, M.CLIMB):
                    # Fixed-wing climb-out is a sloped CLIMB path, whereas a
                    # multirotor uses TAKEOFF.  Interpolate from this leg's
                    # authoritative starting AGL so a partial re-task climb
                    # reaches exactly coverage altitude without double-counting.
                    start_agl = (self._vertical_leg_start_agl
                                 if self._vertical_leg_start_agl is not None else self.altitude_m)
                    self.altitude_m = min(
                        a.coverage_altitude_m,
                        start_agl + (a.coverage_altitude_m - start_agl)
                        * (end / path.total_duration_s),
                    )
                elif seg.maneuver is M.LAND:
                    self.altitude_m = max(0.0, self.altitude_m - a.spec.v_descent * elapsed)
                if a.battery.level_j <= 1e-9:
                    # Symmetry: whatever residual is drained is also reported.
                    residual = a.battery.level_j
                    a.battery.drain(residual)
                    a.energy_consumed_j += residual
                    break
            cursor += seg.duration_s
        a._t = end
        if end >= path.total_duration_s - 1e-9:
            if productive and a._photo_tracker is not None:
                a._photo_tracker.finish_pass()
            a._leg_idx += 1
            a._t = 0.0
            self._vertical_leg_start_agl = None
            if a.state in (S.S2_MISSION, S.S_FERRY):
                a._cov_idx += 1
        return used
