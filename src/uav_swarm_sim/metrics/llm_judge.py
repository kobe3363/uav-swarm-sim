"""Phase 4: LLM-as-a-judge mission diagnosis over the Phase 3 telemetry log.

Reads the JSONL event log written by metrics/llm_log_exporter.py
(run_header -> events -> run_summary) and produces a structured, GROUNDED
``MissionDiagnosis``: an outcome classification, a single root cause from a fixed
taxonomy, a causal narrative, contributing factors, a critical-event timeline,
and recommendations.

Two design commitments make this an evaluation device rather than a chatbot:

1. DETERMINISTIC GROUNDING. Before the model is called, the ground-truth facts
   (outcome, the terminal trigger, which drone died, per-drone minimum battery,
   swap / obstacle counts) are extracted from the log by ``extract_facts`` -- no
   model involved. They are handed to the model AND used afterwards to VERIFY its
   claims (``_ground``): if it names a drone that never failed, or contradicts
   the recorded outcome, the mismatch is flagged in the report.

2. INJECTABLE MODEL. ``diagnose`` takes a ``model`` callable ``(system, user) ->
   str``. ``anthropic_model`` is one implementation; a canned function in the
   tests is another; a five-line adapter wraps any other LLM. The module imports
   with no third-party dependency, and the whole pipeline (facts, prompt, parse,
   ground, render) is unit-testable offline.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable

ModelFn = Callable[[str, str], str]   # (system_prompt, user_prompt) -> raw model text


class RootCause(Enum):
    """Fixed diagnosis taxonomy. The model must choose exactly one; the values map
    onto the simulation's terminal physics so a verdict is comparable across runs
    and checkable against the log."""
    NOMINAL_SUCCESS = "NOMINAL_SUCCESS"                        # MISSION_SUCCESS
    BATTERY_DEPLETION_AIRBORNE = "BATTERY_DEPLETION_AIRBORNE"  # a drone hit 0% in flight
    SHARED_POOL_EXHAUSTION = "SHARED_POOL_EXHAUSTION"          # swap reserve emptied pre-coverage
    COVERAGE_TIMEOUT = "COVERAGE_TIMEOUT"                      # max_timesteps without completion
    HAZARD_ATTRITION = "HAZARD_ATTRITION"                      # hazard S_FAILs degraded the fleet
    OBSTACLE_THRASHING = "OBSTACLE_THRASHING"                 # excessive S_OBS avoidance churn
    SWAP_LOGISTICS = "SWAP_LOGISTICS"                         # swap queueing dominated the timeline
    UNDETERMINED = "UNDETERMINED"                             # log insufficient to attribute


_TAXONOMY = tuple(rc.value for rc in RootCause)


# --------------------------------------------------------------------------- #
# Deterministic facts (no model)                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MissionFacts:
    platform: str
    n_drones: int
    outcome: str
    coverage_frac: float
    t_end_s: float
    n_failed: int
    pool_exhausted: bool
    reserve_remaining: int | None
    total_swaps: int
    obstacle_events: int
    terminal_reason: str | None
    failed_drones: tuple[int, ...]
    per_drone_sorties: dict
    min_batt_by_drone: dict          # drone_id -> lowest batt_frac seen in the trace
    salient_timeline: tuple          # compact (t, event, drone, reason, batt_frac)
    deterministic_root_cause: str    # taxonomy guess from the log alone
    config_hash: str


def load_jsonl(path: str) -> list[dict]:
    """Read a telemetry JSONL file into a list of record dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _by_kind(records: list[dict]) -> tuple[dict, dict, list[dict]]:
    header: dict = {}
    summary: dict = {}
    events: list[dict] = []
    for r in records:
        k = r.get("kind")
        if k == "run_header":
            header = r
        elif k == "run_summary":
            summary = r
        elif k == "event":
            events.append(r)
    return header, summary, events


def _infer_root_cause(outcome, terminal_reason, summary) -> str:
    if outcome == "MISSION_SUCCESS":
        return RootCause.NOMINAL_SUCCESS.value
    if terminal_reason == "battery_depleted":
        return RootCause.BATTERY_DEPLETION_AIRBORNE.value
    if terminal_reason == "pool_exhausted" or summary.get("pool_exhausted"):
        return RootCause.SHARED_POOL_EXHAUSTION.value
    if outcome == "MISSION_INCOMPLETE":
        # incomplete with no hard terminal trigger -> hit the time ceiling
        return RootCause.COVERAGE_TIMEOUT.value
    return RootCause.UNDETERMINED.value


def extract_facts(records: list[dict]) -> MissionFacts:
    """Pull deterministic ground truth from the JSONL records (no model)."""
    header, summary, events = _by_kind(records)
    outcome = str(summary.get("outcome", "MISSION_INCOMPLETE"))
    coverage = float(summary.get("coverage_frac", 0.0) or 0.0)

    failed = tuple(sorted({e["drone"] for e in events
                           if e.get("event") == "FAIL" and e.get("drone") is not None}))
    terminal = next((e for e in events if e.get("event") == "TERMINAL"), {})
    terminal_reason = terminal.get("reason") or terminal.get("verdict_reason")

    min_batt: dict = {}
    for e in events:
        d, bf = e.get("drone"), e.get("batt_frac")
        if d is None or bf is None:
            continue
        min_batt[d] = min(min_batt.get(d, 1.0), float(bf))

    obstacle_events = sum(1 for e in events if e.get("event") == "OBSTACLE")

    salient = tuple(
        (round(float(e.get("t", 0.0)), 1), e.get("event"), e.get("drone"),
         e.get("reason") or e.get("verdict_reason") or "", e.get("batt_frac"))
        for e in events if e.get("event") not in ("STATE", "START")
    )

    return MissionFacts(
        platform=str(header.get("platform", "?")),
        n_drones=int(header.get("n_drones", 0) or 0),
        outcome=outcome,
        coverage_frac=coverage,
        t_end_s=float(summary.get("t_end_s", 0.0) or 0.0),
        n_failed=int(summary.get("n_failed", len(failed)) or 0),
        pool_exhausted=bool(summary.get("pool_exhausted", False)),
        reserve_remaining=summary.get("reserve_remaining"),
        total_swaps=int(summary.get("total_swaps", 0) or 0),
        obstacle_events=obstacle_events,
        terminal_reason=terminal_reason,
        failed_drones=failed,
        per_drone_sorties=dict(summary.get("per_drone_sorties", {}) or {}),
        min_batt_by_drone=min_batt,
        salient_timeline=salient,
        deterministic_root_cause=_infer_root_cause(outcome, terminal_reason, summary),
        config_hash=str(header.get("config_hash", "")),
    )


def facts_to_dict(f: MissionFacts) -> dict:
    """JSON-ready view of the deterministic facts (prompt anchor + CLI display)."""
    return {
        "platform": f.platform,
        "n_drones": f.n_drones,
        "outcome": f.outcome,
        "coverage_frac": f.coverage_frac,
        "t_end_s": f.t_end_s,
        "terminal_reason": f.terminal_reason,
        "failed_drones": list(f.failed_drones),
        "n_failed": f.n_failed,
        "pool_exhausted": f.pool_exhausted,
        "reserve_remaining": f.reserve_remaining,
        "total_swaps": f.total_swaps,
        "obstacle_events": f.obstacle_events,
        "min_battery_by_drone": f.min_batt_by_drone,
        "deterministic_root_cause_hint": f.deterministic_root_cause,
    }


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #
_SYSTEM_TEMPLATE = """You are an expert evaluator of autonomous UAV swarm reconnaissance missions. You are given a machine-generated event log of one simulated mission plus deterministic ground-truth facts extracted from that log. Diagnose the mission: classify its outcome, attribute a SINGLE root cause, explain the causal chain, and recommend changes.

How to read the log (JSON Lines):
- run_header: mission setup (platform, drone count, area, altitudes, the finite shared battery-pool size, planner weights).
- event rows: mission discontinuities. The `event` field is the kind: START (spawn), STATE (state transition), OBSTACLE (entered avoidance, S_OBS), SWAP_REQ / SWAP_DONE (battery swap), FAIL (drone lost, S_FAIL), TERMINAL (the mission verdict). Each row carries the from/to state, the reason, the drone's battery fraction and zone, its position, and per-phase deltas dt_phase / dE_phase_j / dist_phase_m (time, energy, distance spent in the state just exited).
- run_summary: final outcome and aggregate counts.

Simulation physics you must respect:
- MISSION_FAILED is a hard physics halt with exactly two triggers: an AIRBORNE drone's battery reaching 0 (battery_depleted), or the shared swap pool being exhausted before 100% coverage (pool_exhausted).
- Hazard-induced FAIL events (random attrition) do NOT by themselves halt the mission; survivors are re-tasked (redistribution). They can still cause an INCOMPLETE outcome if too few drones remain.
- A battery SWAP costs time but ZERO energy (the drone lands). An airborne HOLD inside S_OBS costs both time and energy.
- MISSION_SUCCESS means 100% area coverage AND every survivor returned to idle.

Choose the root cause from EXACTLY this set: {TAXONOMY}.

Respond with ONLY a single JSON object (no prose, no markdown fences) of the form:
{
  "outcome": "<MISSION_SUCCESS|MISSION_FAILED|MISSION_INCOMPLETE>",
  "root_cause": "<one value from the taxonomy>",
  "confidence": <0.0-1.0>,
  "summary": "<2-4 sentence causal explanation grounded in specific events and times>",
  "contributing_factors": ["<factor>", ...],
  "critical_events": [{"t": <seconds>, "drone": <id or null>, "what": "<short>"}, ...],
  "recommendations": ["<actionable change to config or strategy>", ...]
}
Ground every claim in the log. Cite times and drone ids that actually appear."""


def _system_prompt() -> str:
    return _SYSTEM_TEMPLATE.replace("{TAXONOMY}", ", ".join(_TAXONOMY))


def build_prompt(records: list[dict], facts: MissionFacts, *, max_events: int = 400) -> tuple[str, str]:
    """Return ``(system, user)``. The user message carries the deterministic facts
    (the grounding anchor) followed by the event log, bounded to ``max_events`` by
    downsampling STATE rows while keeping every OBSTACLE/SWAP/FAIL/TERMINAL row."""
    header, summary, events = _by_kind(records)

    salient = [e for e in events if e.get("event") not in ("STATE", "START")]
    states = [e for e in events if e.get("event") in ("STATE", "START")]
    note = ""
    if len(events) > max_events:
        keep = max(0, max_events - len(salient))
        step = max(1, len(states) // keep) if keep else len(states) + 1
        states = states[::step]
        note = (f"NOTE: the log had {len(events)} events; STATE rows were "
                f"downsampled to fit. All OBSTACLE/SWAP/FAIL/TERMINAL events are retained.")
    kept = sorted(salient + states, key=lambda e: float(e.get("t", 0.0)))

    body = [header] + kept + [summary] if (header or summary) else kept
    log_lines = "\n".join(json.dumps(r, separators=(",", ":")) for r in body if r)

    user = (
        "DETERMINISTIC FACTS (ground truth extracted from the log):\n"
        + json.dumps(facts_to_dict(facts), indent=2)
        + (("\n\n" + note) if note else "")
        + "\n\nFULL EVENT LOG (JSONL):\n"
        + log_lines
        + "\n\nDiagnose this mission as instructed. Respond with ONLY the JSON object."
    )
    return _system_prompt(), user


# --------------------------------------------------------------------------- #
# Verdict parsing + diagnosis assembly + grounding                            #
# --------------------------------------------------------------------------- #
def parse_verdict(text: str) -> dict:
    """Parse the model's JSON verdict, tolerating ```fences``` and surrounding prose."""
    s = text.strip()
    if "```" in s:
        for part in s.split("```"):
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError("no JSON object found in model output")
    return json.loads(s[i:j + 1])


@dataclass(frozen=True)
class MissionDiagnosis:
    outcome: str
    root_cause: str
    confidence: float
    summary: str
    contributing_factors: tuple
    critical_events: tuple
    recommendations: tuple
    grounding: dict
    facts: MissionFacts
    raw: dict

    @property
    def grounded(self) -> bool:
        checks = self.grounding.get("checks", [])
        return bool(checks) and all(c.get("ok", False) for c in checks)


def _assemble(verdict: dict, facts: MissionFacts) -> MissionDiagnosis:
    rc = str(verdict.get("root_cause", RootCause.UNDETERMINED.value))
    if rc not in _TAXONOMY:
        rc = RootCause.UNDETERMINED.value
    return MissionDiagnosis(
        outcome=str(verdict.get("outcome", facts.outcome)),
        root_cause=rc,
        confidence=float(verdict.get("confidence", 0.0) or 0.0),
        summary=str(verdict.get("summary", "")).strip(),
        contributing_factors=tuple(verdict.get("contributing_factors", []) or []),
        critical_events=tuple(verdict.get("critical_events", []) or []),
        recommendations=tuple(verdict.get("recommendations", []) or []),
        grounding={},
        facts=facts,
        raw=verdict,
    )


def _cited_drone_ids(critical_events) -> list[int]:
    out: list[int] = []
    for ev in critical_events:
        d = ev.get("drone") if isinstance(ev, dict) else None
        if isinstance(d, bool) or d is None:
            continue
        if isinstance(d, (int, float)):
            out.append(int(d))
        elif isinstance(d, str) and d.lstrip("-").isdigit():
            out.append(int(d))
    return sorted(set(out))


def _ground(diag: MissionDiagnosis, facts: MissionFacts) -> MissionDiagnosis:
    """Verify the model's claims against the deterministic facts."""
    checks = []

    checks.append({
        "name": "outcome_matches_log",
        "ok": diag.outcome == facts.outcome,
        "detail": f"model={diag.outcome} log={facts.outcome}",
    })

    cited = _cited_drone_ids(diag.critical_events)
    if facts.n_drones:
        unknown = [d for d in cited if d < 0 or d >= facts.n_drones]
    else:
        unknown = []
    checks.append({
        "name": "cited_drones_exist",
        "ok": len(unknown) == 0,
        "detail": f"cited={cited} fleet=0..{max(facts.n_drones - 1, 0)} unknown={unknown}",
    })

    # for a HARD physics terminal the model must agree; for interpretive cases
    # (timeout / undetermined) a more specific attribution is allowed.
    interpretive = facts.deterministic_root_cause in (
        RootCause.UNDETERMINED.value, RootCause.COVERAGE_TIMEOUT.value)
    checks.append({
        "name": "root_cause_consistent_with_terminal",
        "ok": diag.root_cause == facts.deterministic_root_cause or interpretive,
        "detail": f"model={diag.root_cause} deterministic_hint={facts.deterministic_root_cause}",
    })

    grounding = {
        "checks": checks,
        "passed": sum(1 for c in checks if c["ok"]),
        "total": len(checks),
    }
    return dataclasses.replace(diag, grounding=grounding)


def diagnose(records: list[dict], model: ModelFn, *, max_events: int = 400) -> MissionDiagnosis:
    """Full pipeline: extract facts -> build prompt -> call model -> parse ->
    assemble -> ground. ``model`` is any ``(system, user) -> str`` callable."""
    facts = extract_facts(records)
    system, user = build_prompt(records, facts, max_events=max_events)
    raw_text = model(system, user)
    verdict = parse_verdict(raw_text)
    return _ground(_assemble(verdict, facts), facts)


# --------------------------------------------------------------------------- #
# Model adapter (Anthropic) -- lazily imported so the module needs no SDK      #
# --------------------------------------------------------------------------- #
def anthropic_model(model: str = "claude-sonnet-4-6", *, max_tokens: int = 2000,
                    api_key: str | None = None, temperature: float = 0.0) -> ModelFn:
    """Return a ModelFn backed by the Anthropic Messages API. The ``anthropic``
    package is imported only when the returned callable is actually invoked, so
    importing this module (and running dry-runs / tests) needs no SDK or key."""
    def _call(system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError("the 'anthropic' package is required for live "
                               "diagnosis: pip install anthropic") from e
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", "") == "text")
    return _call


# --------------------------------------------------------------------------- #
# Report rendering                                                            #
# --------------------------------------------------------------------------- #
def render_markdown(diag: MissionDiagnosis) -> str:
    """Render a Markdown diagnosis report (ASCII markers for cross-platform safety)."""
    f = diag.facts
    L: list[str] = []
    L.append(f"# Mission Diagnosis - {diag.outcome}")
    L.append("")
    L.append(f"**Root cause:** `{diag.root_cause}`  |  "
             f"**Confidence:** {diag.confidence:.2f}  |  "
             f"**Grounded:** {'yes' if diag.grounded else 'NO - see audit'}")
    L.append("")
    L.append("## Summary")
    L.append(diag.summary or "_(none provided)_")
    L.append("")
    L.append("## Mission facts (deterministic)")
    L.append(f"- Platform: {f.platform}, drones: {f.n_drones}")
    L.append(f"- Coverage: {f.coverage_frac:.1%}, ended at t={f.t_end_s:.0f}s")
    L.append(f"- Terminal trigger: {f.terminal_reason or 'none'}")
    L.append(f"- Failed drones: {list(f.failed_drones) or 'none'} (n_failed={f.n_failed})")
    L.append(f"- Shared pool exhausted: {f.pool_exhausted}, reserve remaining: {f.reserve_remaining}")
    L.append(f"- Swaps: {f.total_swaps}, obstacle encounters: {f.obstacle_events}")
    if f.min_batt_by_drone:
        wid, wval = min(f.min_batt_by_drone.items(), key=lambda kv: kv[1])
        L.append(f"- Lowest battery observed: drone {wid} at {wval:.0%}")
    L.append("")
    if diag.contributing_factors:
        L.append("## Contributing factors")
        L.extend(f"- {x}" for x in diag.contributing_factors)
        L.append("")
    if diag.critical_events:
        L.append("## Critical event timeline")
        for ev in diag.critical_events:
            t = ev.get("t") if isinstance(ev, dict) else None
            d = ev.get("drone") if isinstance(ev, dict) else None
            what = ev.get("what", "") if isinstance(ev, dict) else str(ev)
            who = f"drone {d}" if d is not None else "mission"
            ts = f"t={t:.0f}s" if isinstance(t, (int, float)) else "t=?"
            L.append(f"- {ts} - {who}: {what}")
        L.append("")
    if diag.recommendations:
        L.append("## Recommendations")
        L.extend(f"- {x}" for x in diag.recommendations)
        L.append("")
    L.append("## Grounding audit")
    for c in diag.grounding.get("checks", []):
        L.append(f"- [{'OK' if c.get('ok') else 'FLAG'}] **{c['name']}** - {c.get('detail', '')}")
    g = diag.grounding
    L.append("")
    L.append(f"_Verification: {g.get('passed', '?')}/{g.get('total', '?')} checks passed. "
             f"The diagnosis is an LLM judgment; the facts and audit above are computed "
             f"deterministically from the log._")
    return "\n".join(L) + "\n"
