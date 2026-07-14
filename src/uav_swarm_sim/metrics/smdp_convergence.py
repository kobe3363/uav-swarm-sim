"""Per-state convergence diagnostics for the empirical SMDP estimate.

WHAT IT MEASURES. The embedded-chain matrix P-hat of an ``SmdpEstimate`` is a
row-wise maximum-likelihood estimate from raw transition counts. This module
quantifies how well-supported each row is:

  * per-state VISIT COUNT (raw sojourns observed, including truncated terminal
    sojourns that produced no outgoing transition);
  * per-transition Wilson 95 % CI on each p_ij (cell count as a binomial
    against the row total -- the interval stays inside [0, 1] and behaves
    sensibly at 0/1, exactly the regime rare states live in);
  * a WEAKEST-STATE summary: the visited state with the fewest visits, and the
    single widest transition CI across the whole matrix.

HOW TO READ THE WEAKEST STATE. The stationary distribution (and everything
derived from it: pi_time, efficiency) is only as trustworthy as the least
supported row of P-hat. A state visited once has p_hat in {0, 1} with a CI
spanning most of [0, 1]; any pi mass flowing through it is anecdotal. Report
the weakest state next to pi -- if its CI is wide, more replications (or a
longer mission) are needed before the pi values involving it mean anything.

MULTINOMIAL CAVEAT. Each row of the embedded chain is one multinomial sample;
the Wilson interval treats each cell j as a binomial (j vs not-j) against the
row total. The per-cell intervals are therefore marginal, NOT a joint
(simultaneous) confidence region for the row -- do not read the set of
intervals as covering all cells at 95 % at once.

Pure functions, no I/O. ``convergence_report`` serves the single-mission path;
``pool_counts`` + ``report_from_counts`` let a future Monte-Carlo caller pool
raw counts across replications (counts are additive) without touching this
module. Not wired into any sweep script by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..infrastructure.enums import AgentState
from .convergence import wilson_ci
from .smdp_estimator import STATE_ORDER, SmdpEstimate


@dataclass(frozen=True)
class TransitionCI:
    """One observed transition cell i->j with its marginal Wilson 95 % CI."""
    count: int
    row_total: int
    p_hat: float
    wilson95_lo: float
    wilson95_hi: float

    @property
    def width(self) -> float:
        return self.wilson95_hi - self.wilson95_lo


@dataclass(frozen=True)
class StateConvergence:
    """Evidence behind one row of the embedded-chain estimate."""
    state: AgentState
    visits: int            # raw sojourns observed (>= transitions_out)
    transitions_out: int   # observed outgoing transitions (the binomial row total)
    transitions: dict[AgentState, TransitionCI]  # only observed (count > 0) cells


@dataclass(frozen=True)
class ConvergenceReport:
    per_state: list[StateConvergence]                       # canonical STATE_ORDER
    unvisited: list[AgentState]                             # never observed
    weakest_by_visits: tuple[AgentState, int] | None        # min visits among visited
    widest_ci: tuple[AgentState, AgentState, TransitionCI] | None  # (from, to, cell)


def report_from_counts(
    states: Sequence[AgentState],
    counts: np.ndarray,
    visits: np.ndarray,
) -> ConvergenceReport:
    """Build the diagnostics from raw per-state evidence.

    ``counts[i, j]`` = observed i->j transitions, ``visits[i]`` = raw sojourns
    in state i; both indexed like ``states``. States with 0 visits (possible in
    a pooled full-base call) are reported as unvisited. A visited state with a
    zero row (truncated terminal, e.g. an open S_FAIL) gets an empty
    ``transitions`` dict -- no division by zero, no vacuous intervals.
    """
    idx = {s: i for i, s in enumerate(states)}
    per_state: list[StateConvergence] = []
    unvisited = [s for s in STATE_ORDER if s not in idx or visits[idx[s]] <= 0]
    widest: tuple[AgentState, AgentState, TransitionCI] | None = None

    for s in STATE_ORDER:
        if s not in idx or visits[idx[s]] <= 0:
            continue
        si = idx[s]
        row = counts[si]
        row_total = int(round(float(row.sum())))
        cells: dict[AgentState, TransitionCI] = {}
        for t in STATE_ORDER:
            if t not in idx:
                continue
            k = int(round(float(row[idx[t]])))
            if k <= 0:
                continue
            lo, hi, p_hat = wilson_ci(k, row_total)
            cell = TransitionCI(k, row_total, p_hat, lo, hi)
            cells[t] = cell
            if widest is None or cell.width > widest[2].width:
                widest = (s, t, cell)
        per_state.append(StateConvergence(
            state=s, visits=int(round(float(visits[si]))),
            transitions_out=row_total, transitions=cells,
        ))

    weakest = None
    if per_state:
        w = min(per_state, key=lambda sc: sc.visits)  # ties: first in STATE_ORDER
        weakest = (w.state, w.visits)
    return ConvergenceReport(per_state, unvisited, weakest, widest)


def convergence_report(est: SmdpEstimate) -> ConvergenceReport:
    """Diagnostics for one mission's estimate (counts pooled across agents,
    exactly as ``estimate`` pooled them)."""
    visits = est.visits if est.visits is not None else est.n_transitions.sum(axis=1)
    return report_from_counts(est.states, est.n_transitions, np.asarray(visits, dtype=float))


def pool_counts(
    estimates: Sequence[SmdpEstimate],
) -> tuple[list[AgentState], np.ndarray, np.ndarray]:
    """Sum raw counts across replications on the full STATE_ORDER base.

    Counts are additive, so pooling then estimating equals estimating the
    concatenated history. Future hook for Monte-Carlo callers (feed the result
    to ``report_from_counts``); deliberately not wired into any sweep script.
    """
    full = len(STATE_ORDER)
    idx = {s: i for i, s in enumerate(STATE_ORDER)}
    counts = np.zeros((full, full), dtype=float)
    visits = np.zeros(full, dtype=float)
    for est in estimates:
        v = est.visits if est.visits is not None else est.n_transitions.sum(axis=1)
        for a, sa in enumerate(est.states):
            visits[idx[sa]] += float(v[a])
            for b, sb in enumerate(est.states):
                counts[idx[sa], idx[sb]] += float(est.n_transitions[a, b])
    return list(STATE_ORDER), counts, visits


def report_to_json(report: ConvergenceReport) -> dict:
    """JSON-ready block for results.json (additive: new keys only)."""
    widest = None
    if report.widest_ci is not None:
        f, t, c = report.widest_ci
        widest = {"from": f.value, "to": t.value, "count": c.count,
                  "row_total": c.row_total, "p_hat": c.p_hat,
                  "wilson95_lo": c.wilson95_lo, "wilson95_hi": c.wilson95_hi,
                  "width": c.width}
    return {
        "note": ("Wilson 95% CI per transition cell (binomial vs row total); "
                 "rows are multinomial, so the intervals are marginal, not a "
                 "joint confidence region."),
        "per_state": {
            sc.state.value: {
                "visits": sc.visits,
                "transitions_out": sc.transitions_out,
                "transitions": {
                    t.value: {"count": c.count, "p_hat": c.p_hat,
                              "wilson95_lo": c.wilson95_lo,
                              "wilson95_hi": c.wilson95_hi}
                    for t, c in sc.transitions.items()
                },
            }
            for sc in report.per_state
        },
        "unvisited": [s.value for s in report.unvisited],
        "weakest": {
            "by_min_visits": (
                {"state": report.weakest_by_visits[0].value,
                 "visits": report.weakest_by_visits[1]}
                if report.weakest_by_visits is not None else None
            ),
            "widest_ci95": widest,
        },
    }


def format_table(report: ConvergenceReport) -> str:
    """Markdown/console table: 'SMDP convergence per state'."""
    lines = [
        "SMDP convergence per state (Wilson 95% CI per transition; "
        "per-cell binomial, not a joint region)",
        "| state | visits | out | widest outgoing CI |",
        "| --- | ---: | ---: | --- |",
    ]
    for sc in report.per_state:
        if sc.transitions:
            t, c = max(sc.transitions.items(), key=lambda kv: kv[1].width)
            cell = (f"->{t.name} p={c.p_hat:.2f} "
                    f"[{c.wilson95_lo:.2f}, {c.wilson95_hi:.2f}]")
        else:
            cell = "(no outgoing transitions observed)"
        lines.append(f"| {sc.state.name} | {sc.visits} | {sc.transitions_out} | {cell} |")
    if report.unvisited:
        lines.append("unvisited: " + ", ".join(s.name for s in report.unvisited))
    if report.weakest_by_visits is not None:
        s, v = report.weakest_by_visits
        lines.append(f"weakest by visits: {s.name} ({v})")
    if report.widest_ci is not None:
        f, t, c = report.widest_ci
        lines.append(f"widest CI: {f.name}->{t.name} p={c.p_hat:.2f} "
                     f"[{c.wilson95_lo:.2f}, {c.wilson95_hi:.2f}] (width {c.width:.2f})")
    return "\n".join(lines)
