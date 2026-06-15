"""Phase 3 observability: the event-driven telemetry collector + the fan-out
recorder that feeds it ALONGSIDE the SMDP StateHistory.

Design (Phase 3 architecture): ONE in-memory log of mission DISCONTINUITIES --
state transitions, swaps, obstacle encounters, the terminal verdict -- plus a
coarse periodic position FIX stream purely for GIS legibility. Two dumb
serializers (``gpx_exporter`` for GIS, ``llm_log_exporter`` for the Phase-4
judge) project this single log; the collector never formats anything.

It is a read-only PROBE. It implements the same ``open``/``close`` Recorder
protocol the Agent already calls -- routed through ``FanoutRecorder`` so neither
the Agent nor StateHistory changes -- and pulls each pose/battery/energy snapshot
straight off the bound fleet. The per-phase deltas (dt, denergy, ddistance) it
computes are OBSERVATIONAL: they never feed back into physics, energy, or the
SMDP. When telemetry is disabled the engine never constructs it, so a run is
byte-identical to the pre-Phase-3 baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..infrastructure.enums import AgentState, BatteryZone, TelemetryEventKind


@dataclass(frozen=True)
class AgentSnapshot:
    """Instantaneous, read-only view of one agent at a transition or a fix."""
    x: float
    y: float
    z: float
    batt_frac: float
    batt_zone: BatteryZone
    energy_j: float    # cumulative energy_consumed_j (survives swaps)
    flown_m: float     # cumulative flown_m


@dataclass(frozen=True)
class TelemetryEvent:
    kind: TelemetryEventKind
    t: float
    drone: int | None
    from_state: AgentState | None
    to_state: AgentState | None
    reason: str
    snap: AgentSnapshot | None
    dt_phase: float          # time spent in the state just exited
    dE_phase_j: float        # energy burned during that phase
    dist_phase_m: float      # distance flown during that phase
    context: dict            # event-specific: obstacle_id / outcome / coverage_frac / ...

    def to_record(self) -> dict:
        """Compact, JSON-ready dict. Floats are rounded to hold LLM tokens down,
        and absent/zero fields are omitted so each row stays sparse."""
        rec: dict = {"kind": "event", "t": round(self.t, 1), "event": self.kind.value}
        if self.drone is not None:
            rec["drone"] = self.drone
        if self.from_state is not None:
            rec["from"] = self.from_state.value
        if self.to_state is not None:
            rec["to"] = self.to_state.value
        if self.reason:
            rec["reason"] = self.reason
        if self.snap is not None:
            rec["batt_frac"] = round(self.snap.batt_frac, 3)
            rec["batt_zone"] = self.snap.batt_zone.value
            rec["x"] = round(self.snap.x, 1)
            rec["y"] = round(self.snap.y, 1)
            rec["z"] = round(self.snap.z, 1)
        # phase deltas are meaningful only on a real transition out of a prior phase
        if self.dt_phase > 0.0 or self.dE_phase_j > 0.0 or self.dist_phase_m > 0.0:
            rec["dt_phase"] = round(self.dt_phase, 1)
            rec["dE_phase_j"] = round(self.dE_phase_j, 0)
            rec["dist_phase_m"] = round(self.dist_phase_m, 1)
        for k, v in self.context.items():
            if v is not None:
                rec[k] = round(v, 4) if isinstance(v, float) else v
        return rec


def _classify(from_state, to_state, reason) -> TelemetryEventKind:
    """Map a state transition to a salient telemetry verb."""
    if to_state is AgentState.S_OBS:
        return TelemetryEventKind.OBSTACLE
    if to_state is AgentState.S_SWAP:
        return TelemetryEventKind.SWAP_REQ
    if to_state is AgentState.S_FAIL:
        return TelemetryEventKind.FAIL
    if from_state is AgentState.S_SWAP and reason == "swap_done":
        return TelemetryEventKind.SWAP_DONE
    return TelemetryEventKind.STATE


class TelemetryLog:
    """Collector. Fed transitions via the Recorder protocol and periodic fixes
    via ``record_fix``; pulls each snapshot from the late-bound fleet."""

    def __init__(self, fix_interval_s: float = 30.0, run_header: dict | None = None) -> None:
        self.fix_interval_s = float(fix_interval_s)
        self.header: dict = dict(run_header or {})
        self.summary: dict = {}
        self._events: list[TelemetryEvent] = []
        self._fixes: dict[int, list[tuple[float, float, float, float]]] = {}  # id -> [(t,x,y,z)]
        self._entry: dict[int, tuple[AgentState, float, AgentSnapshot | None]] = {}
        self._pending_reason: dict[int, str] = {}
        self._fleet = None

    # --- binding ---------------------------------------------------------- #
    def bind_fleet(self, fleet) -> None:
        """Give the collector read access to agent pose/battery/energy. Call once
        after Fleet construction and before any open/close/record_fix."""
        self._fleet = fleet

    def set_header(self, header: dict) -> None:
        self.header = dict(header)

    def _snap(self, agent_id: int) -> AgentSnapshot | None:
        if self._fleet is None:
            return None
        a = self._fleet.agents.get(agent_id)
        if a is None:
            return None
        return AgentSnapshot(
            x=a.pose.x, y=a.pose.y, z=a.pose.z,
            batt_frac=a.battery.frac, batt_zone=a.battery.zone,
            energy_j=a.energy_consumed_j, flown_m=a.flown_m,
        )

    # --- Recorder protocol (transition discontinuities) ------------------- #
    def open(self, agent_id, state, t) -> None:
        snap = self._snap(agent_id)
        prev = self._entry.get(agent_id)
        if prev is not None:
            from_state, t_in, entry_snap = prev
            reason = self._pending_reason.pop(agent_id, "")
            dt = max(0.0, t - t_in)
            dE = (snap.energy_j - entry_snap.energy_j) if (snap and entry_snap) else 0.0
            dd = (snap.flown_m - entry_snap.flown_m) if (snap and entry_snap) else 0.0
            self._events.append(TelemetryEvent(
                _classify(from_state, state, reason), t, agent_id, from_state, state,
                reason, snap, dt, max(0.0, dE), max(0.0, dd), {},
            ))
        else:
            # first entry (initial state, e.g. S0 at t=0) -> a START marker
            self._events.append(TelemetryEvent(
                TelemetryEventKind.START, t, agent_id, None, state, "spawn", snap,
                0.0, 0.0, 0.0, {},
            ))
        self._entry[agent_id] = (state, t, snap)

    def close(self, agent_id, t, reason) -> None:
        # the destination is known only at the paired open(); just stash the reason
        self._pending_reason[agent_id] = reason

    def finalize(self, t_end: float) -> None:
        """Emit a closing event for every still-open phase (drones airborne at the
        max-timestep ceiling, or parked in S0 at success)."""
        for agent_id, (from_state, t_in, entry_snap) in list(self._entry.items()):
            snap = self._snap(agent_id) or entry_snap
            dt = max(0.0, t_end - t_in)
            dE = (snap.energy_j - entry_snap.energy_j) if (snap and entry_snap) else 0.0
            dd = (snap.flown_m - entry_snap.flown_m) if (snap and entry_snap) else 0.0
            self._events.append(TelemetryEvent(
                TelemetryEventKind.STATE, t_end, agent_id, from_state, None, "mission_end",
                snap, dt, max(0.0, dE), max(0.0, dd), {"final": True},
            ))
        self._entry.clear()

    # --- periodic position fixes (GPX legibility only) -------------------- #
    def record_fix(self, agent_id, t) -> None:
        snap = self._snap(agent_id)
        if snap is not None:
            self._fixes.setdefault(agent_id, []).append((t, snap.x, snap.y, snap.z))

    # --- run-level events ------------------------------------------------- #
    def record_terminal(self, t, outcome, reason, **context) -> None:
        ctx = {"outcome": getattr(outcome, "value", str(outcome)), "verdict_reason": reason}
        ctx.update({k: v for k, v in context.items() if v is not None})
        self._events.append(TelemetryEvent(
            TelemetryEventKind.TERMINAL, t, None, None, None, reason, None,
            0.0, 0.0, 0.0, ctx,
        ))

    def record_leg_repair(self, agent_id, t, repair_kind, **context) -> None:
        """Forward-compat hook for Task 2.5 Q2: a leg the pre-flight validator
        repaired (resmoothed / linear) or could not (blocked). Not emitted until
        trajectory validation is wired into the agent."""
        ctx = {"repair": getattr(repair_kind, "value", str(repair_kind))}
        ctx.update({k: v for k, v in context.items() if v is not None})
        self._events.append(TelemetryEvent(
            TelemetryEventKind.LEG_REPAIR, t, agent_id, None, None, "leg_repair",
            self._snap(agent_id), 0.0, 0.0, 0.0, ctx,
        ))

    def set_summary(self, **summary) -> None:
        self.summary = {k: v for k, v in summary.items() if v is not None}

    # --- access for exporters --------------------------------------------- #
    def events(self) -> list[TelemetryEvent]:
        """All events, time-sorted (run-level events, drone=None, sort first at a
        tie so a TERMINAL precedes nothing it should follow)."""
        return sorted(self._events, key=lambda e: (e.t, -1 if e.drone is None else e.drone))

    def drone_ids(self) -> list[int]:
        ids = set(self._fixes) | {e.drone for e in self._events if e.drone is not None}
        return sorted(ids)

    def gpx_track(self, agent_id: int) -> list[tuple[float, float, float, float]]:
        """Merged, time-sorted (t,x,y,z) for one drone: transition positions PLUS
        periodic fixes, with consecutive duplicates dropped."""
        pts = list(self._fixes.get(agent_id, []))
        for e in self._events:
            if e.drone == agent_id and e.snap is not None:
                pts.append((e.t, e.snap.x, e.snap.y, e.snap.z))
        pts.sort(key=lambda p: p[0])
        out: list[tuple[float, float, float, float]] = []
        for p in pts:
            if not out or abs(p[0] - out[-1][0]) > 1e-9 or p[1:] != out[-1][1:]:
                out.append(p)
        return out

    def derive_counts(self) -> dict:
        """Per-drone sortie / obstacle counts + fleet swap count, derived from the
        event stream (observational; the engine folds these into the summary)."""
        sorties: dict[int, int] = {}
        obstacles: dict[int, int] = {}
        swaps = 0
        for e in self._events:
            if e.from_state is AgentState.S0_IDLE and e.to_state is AgentState.S1_TRANSIT:
                sorties[e.drone] = sorties.get(e.drone, 0) + 1
            if e.kind is TelemetryEventKind.OBSTACLE:
                obstacles[e.drone] = obstacles.get(e.drone, 0) + 1
            if e.kind is TelemetryEventKind.SWAP_DONE:
                swaps += 1
        return {
            "per_drone_sorties": {str(k): v for k, v in sorted(sorties.items())},
            "per_drone_obstacle_events": {str(k): v for k, v in sorted(obstacles.items())},
            "total_swaps": swaps,
        }


class FanoutRecorder:
    """Fan transition open/close out to several recorders so the Agent keeps a
    single ``recorder`` reference. Lets TelemetryLog run ALONGSIDE the SMDP
    StateHistory without changing the Agent or StateHistory."""

    def __init__(self, recorders) -> None:
        self._rs = list(recorders)

    def open(self, agent_id, state, t) -> None:
        for r in self._rs:
            r.open(agent_id, state, t)

    def close(self, agent_id, t, reason) -> None:
        for r in self._rs:
            r.close(agent_id, t, reason)
