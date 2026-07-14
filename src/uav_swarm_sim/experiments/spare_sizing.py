"""Pure core for the spare-sizing (shared battery-pool) study.

The heavy, engine-driven Monte-Carlo lives in ``run_spare_sizing`` (the CLI); this
module holds only the pure, exhaustively testable pieces:

  * ``wilson_ci`` -- the score interval for a success *proportion*. The right CI
    near p -> 1 (our 99 %/95 % targets), where the normal approximation used by
    ``convergence.ci_half_width`` collapses (it can wander above 1.0).
  * ``analytical_spare_prior`` -- the DoR formula ``spares ~= E_cover/B_usable
    - n + margin`` expressed against the SAME swap accounting the B6.2 fleet
    sizing core uses (``total_sorties_int - min(n, total_sorties_int)``), so the
    analytical claim and the empirical knee are compared on one definition.
  * ``find_knee`` -- the diminishing-uncertainty analogue of B6.2's
    ``_find_knee``: the smallest spare count whose success metric first clears a
    target. Two rules are supported -- the point estimate, and the (thesis-safe)
    Wilson lower bound.
  * ``DemandRecord`` / ``demand_cdf`` / ``demand_knees`` -- the demand-mode
    (B = infinity) core: because the shared pool is a pure COUNT constraint,
    ``success(k, B) <=> D_k <= B`` where ``D_k`` is the swap-pack demand the
    replication exhibits under an unbounded pool (infinity when the unbounded
    run itself fails). One unbounded batch reconstructs the whole success-vs-B
    curve post hoc, knees included, on the SAME Wilson rules as the grid sweep.

The sweep VARIABLE is the spare count (``fleet.total_reserve_batteries``); each
count is Monte-Carlo'd on *paired* replication seeds so a change in the
success fraction is attributable to spares, not to seed noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..metrics.convergence import Z_95, wilson_ci  # noqa: F401  (wilson_ci re-exported; body moved to metrics.convergence)

# The two thesis targets are fixed by the DoR. Named here as the single source of
# truth so the CLI, the plot and the tests agree.
TARGETS: tuple[float, ...] = (0.99, 0.95)

# Knee rules: "point" = smallest spare whose point estimate clears the target;
# "wilson_lower" = smallest spare whose Wilson 95 % lower bound clears it (the
# robust, seed-noise-resistant reading the thesis reports).
KNEE_RULE_POINT = "point"
KNEE_RULE_WILSON = "wilson_lower"


# --------------------------------------------------------------------------- #
# binomial CI on the success fraction                                          #
# --------------------------------------------------------------------------- #
def min_reps_for_target(target: float, z: float = Z_95) -> int:
    """Smallest replication count at which a FLAWLESS batch (k == n) can certify
    ``target`` under the Wilson-lower rule.

    For ``k == n`` the Wilson lower bound is ``n / (n + z^2)``; requiring that to
    reach ``target`` gives ``n >= z^2 * target / (1 - target)``. This is the hard
    reps floor the strict (Wilson-lower) knee imposes: e.g. ~381 for 99 % and ~73
    for 95 %. Below it, no spare count -- however reliable -- can be *certified*
    at that target, only estimated (the point-estimate knee still resolves).
    """
    if not (0.0 < target < 1.0):
        raise ValueError("target must be in (0, 1)")
    return math.ceil(z * z * target / (1.0 - target))


# --------------------------------------------------------------------------- #
# per-spare-count outcome record                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SparePoint:
    """One spare count's Monte-Carlo outcome over ``n_reps`` paired replications."""
    spares: int
    n_reps: int
    n_success: int
    n_failed: int
    n_incomplete: int

    @property
    def success_frac(self) -> float:
        return self.n_success / self.n_reps if self.n_reps else float("nan")

    @property
    def wilson(self) -> tuple[float, float, float]:
        return wilson_ci(self.n_success, self.n_reps)

    @property
    def wilson_lo(self) -> float:
        return self.wilson[0]

    @property
    def wilson_hi(self) -> float:
        return self.wilson[1]


# --------------------------------------------------------------------------- #
# analytical prior (the DoR formula)                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SparePrior:
    """The analytical prediction ``spares ~= E_cover/B_usable - n + margin``.

    ``base_spares`` is the zero-margin prediction (every drone's first sortie
    runs on its initial pack, so only ``total_sorties_int - min(n, sorties)``
    swaps draw from the shared reserve -- the same accounting as
    ``fleet_sizing._station_time``). ``margin`` is an ADDITIVE integer pack
    count (the decision locked in design)."""
    total_sorties_int: int
    n_drones: int
    margin: int
    base_spares: int
    prior_spares: int


def analytical_spare_prior(total_sorties_int: int, n_drones: int, margin: int = 0) -> SparePrior:
    """``E_cover/B_usable - n + margin`` in the pool's own swap-count units.

    ``total_sorties_int`` is the integer battery-cycle demand from the B6.2
    fleet-sizing core (``ceil(E_cover / B_usable)``). The first sortie of each of
    the (at most ``total_sorties_int``) active drones runs on its onboard pack, so
    the shared reserve must supply the remaining swaps; ``margin`` adds a fixed
    number of spare packs on top.
    """
    n_active = min(n_drones, total_sorties_int)
    base = max(0, total_sorties_int - n_active)
    return SparePrior(
        total_sorties_int=total_sorties_int,
        n_drones=n_drones,
        margin=margin,
        base_spares=base,
        prior_spares=base + margin,
    )


# --------------------------------------------------------------------------- #
# knee detection (mirrors B6.2's _find_knee threshold-crossing)                #
# --------------------------------------------------------------------------- #
def _metric(pt: SparePoint, rule: str) -> float:
    if rule == KNEE_RULE_WILSON:
        return pt.wilson_lo
    if rule == KNEE_RULE_POINT:
        return pt.success_frac
    raise ValueError(f"unknown knee rule: {rule!r}")


def find_knee(points: Sequence[SparePoint], target: float,
              rule: str = KNEE_RULE_WILSON) -> int | None:
    """Smallest spare count whose success metric first reaches ``target``.

    Points are read in ascending spare-count order (the sweep variable). Returns
    that spare count, or ``None`` if no swept count clears the target under the
    chosen rule. This is the spare-sizing analogue of the fleet-sizing knee: the
    first place the diminishing-returns curve crosses the operator's threshold.
    """
    for pt in sorted(points, key=lambda p: p.spares):
        if _metric(pt, rule) >= target:
            return pt.spares
    return None


@dataclass(frozen=True)
class KneeResult:
    target: float
    knee_point: int | None       # smallest spare with point estimate  >= target
    knee_wilson: int | None      # smallest spare with Wilson lower bnd >= target


def knees_at_targets(points: Sequence[SparePoint],
                     targets: Sequence[float] = TARGETS) -> list[KneeResult]:
    return [
        KneeResult(
            target=t,
            knee_point=find_knee(points, t, KNEE_RULE_POINT),
            knee_wilson=find_knee(points, t, KNEE_RULE_WILSON),
        )
        for t in targets
    ]


# --------------------------------------------------------------------------- #
# formula validation read-out (honest either way)                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FormulaVerdict:
    prior_spares: int            # analytical prediction (with margin)
    empirical_knee: int | None   # the point-estimate knee at the target, or None
    certified_knee: int | None   # the stricter Wilson-lower knee, or None
    target: float
    delta: int | None            # empirical_knee - prior_spares (None if no knee)
    verdict: str                 # "validated" / "refuted" / "inconclusive"
    measured_margin: int | None  # empirical_knee - base_spares (the margin the data demanded)


def validate_formula(prior: SparePrior, knee: KneeResult,
                     tolerance: int = 1) -> FormulaVerdict:
    """Honest read-out: does the analytical prior land on the empirical knee?

    The comparison uses the POINT-ESTIMATE knee (where the success curve first
    crosses the target) -- the best estimate of the true crossing and, unlike the
    strict Wilson-lower knee, always resolved when the sweep range brackets it.
    ``validated`` when it sits within ``tolerance`` packs of
    ``prior.prior_spares``; ``refuted`` when further off; ``inconclusive`` when no
    swept count reached the target (the range missed the knee). The conservative
    ``certified_knee`` (Wilson-lower) is carried through for context, and
    ``measured_margin`` reports the margin the data actually demanded over the
    zero-margin base, so the formula can be re-centred whichever way it falls.
    """
    emp = knee.knee_point
    if emp is None:
        return FormulaVerdict(prior.prior_spares, None, knee.knee_wilson,
                              knee.target, None, "inconclusive", None)
    delta = emp - prior.prior_spares
    verdict = "validated" if abs(delta) <= tolerance else "refuted"
    return FormulaVerdict(prior.prior_spares, emp, knee.knee_wilson, knee.target,
                          delta, verdict, emp - prior.base_spares)


# --------------------------------------------------------------------------- #
# default sweep range from the analytical prior                               #
# --------------------------------------------------------------------------- #
def default_spare_range(prior: SparePrior, span: int = 8) -> list[int]:
    """A spare-count grid bracketing the analytical prior.

    Centred on the zero-margin base ``prior.base_spares`` so the sweep straddles
    the predicted knee whatever margin the data turns out to need: from
    ``max(0, base - span)`` up to ``base + span`` inclusive, step 1.
    """
    center = prior.base_spares
    lo = max(0, center - span)
    hi = center + span
    return list(range(lo, hi + 1))


# --------------------------------------------------------------------------- #
# demand mode (B = infinity): per-replication swap-pack demand                 #
# --------------------------------------------------------------------------- #
# The pool is a pure COUNT constraint (swap_station.py: single decrement site,
# no other code reads the remaining count), so a replication's success at ANY
# pool size B is determined by its demand D under an unbounded pool:
#   success(k, B)  <=>  D_k <= B,
# with D_k := infinity when the unbounded run itself does not succeed (battery
# depletion / timeout -- no pool size can fix those). One unbounded batch of
# ``reps`` replications therefore reconstructs the WHOLE success-vs-B curve.


@dataclass(frozen=True)
class DemandRecord:
    """One replication's measured swap-pack demand under an UNBOUNDED pool.

    ``demand`` is the number of packs the replication drew from the shared
    reserve (== its S_SWAP sojourn count: with an unbounded pool every swap
    request is admitted, and a mission cannot end in success with a drone
    still queued). ``demand is None`` encodes D = infinity: the unbounded run
    did not end in MISSION_SUCCESS, so it counts as a non-success at EVERY
    finite pool size. ``per_drone_swaps`` is observational (workload spread);
    the knee math never reads it.
    """
    replication: int
    outcome: str                    # infrastructure.enums.Outcome.value
    demand: int | None              # None <=> D = infinity (non-success run)
    per_drone_swaps: dict[int, int] = field(default_factory=dict)


def demand_success_count(records: Sequence[DemandRecord], b: int) -> int:
    """``#{k : D_k <= b}`` -- by the count-constraint equivalence, exactly the
    number of replications that would end in MISSION_SUCCESS at pool size ``b``."""
    return sum(1 for r in records if r.demand is not None and r.demand <= b)


def max_finite_demand(records: Sequence[DemandRecord]) -> int | None:
    """Largest finite demand observed, or None when no replication succeeded."""
    ds = [r.demand for r in records if r.demand is not None]
    return max(ds) if ds else None


def demand_cdf(records: Sequence[DemandRecord]) -> list[dict]:
    """Empirical demand CDF with a Wilson band: one row per pool size
    ``b = 0..max finite D`` carrying ``#{D_k <= b}`` out of ``n = len(records)``
    and its Wilson 95 % interval -- the demand-mode analogue of the grid
    sweep's per-point success fraction. Empty when nothing succeeded."""
    n = len(records)
    bmax = max_finite_demand(records)
    if n == 0 or bmax is None:
        return []
    rows: list[dict] = []
    for b in range(bmax + 1):
        k = demand_success_count(records, b)
        lo, hi, phat = wilson_ci(k, n)
        rows.append({"spares": b, "n_le": k, "success_frac": phat,
                     "wilson95_lo": lo, "wilson95_hi": hi})
    return rows


def demand_knees(records: Sequence[DemandRecord],
                 targets: Sequence[float] = TARGETS) -> list[KneeResult]:
    """Post-hoc knees from measured demand, one ``KneeResult`` per target.

    ``knee_point``  = min b with ``#{D_k <= b}/n >= target``;
    ``knee_wilson`` = min b with ``wilson_ci(#{D_k <= b}, n)`` lower bound
    >= target (the same Wilson-lower rule the grid sweep certifies with).
    Both scan ``b = 0..max finite D`` -- beyond it the counts cannot grow, so
    a target not reached there is unreachable (None): either too many
    non-success replications or ``reps < min_reps_for_target(target)``.
    """
    n = len(records)
    bmax = max_finite_demand(records)
    out: list[KneeResult] = []
    for t in targets:
        kp: int | None = None
        kw: int | None = None
        if n and bmax is not None:
            for b in range(bmax + 1):
                k = demand_success_count(records, b)
                if kp is None and k / n >= t:
                    kp = b
                if kw is None and wilson_ci(k, n)[0] >= t:
                    kw = b
                if kp is not None and kw is not None:
                    break
        out.append(KneeResult(target=t, knee_point=kp, knee_wilson=kw))
    return out


# --------------------------------------------------------------------------- #
# top-level report                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class SpareSizingReport:
    points: list[SparePoint]
    prior: SparePrior
    knees: list[KneeResult] = field(default_factory=list)
    verdict: FormulaVerdict | None = None

    @classmethod
    def build(cls, points: Sequence[SparePoint], prior: SparePrior,
              targets: Sequence[float] = TARGETS,
              validate_target: float = 0.99) -> "SpareSizingReport":
        pts = sorted(points, key=lambda p: p.spares)
        knees = knees_at_targets(pts, targets)
        vknee = next((k for k in knees if k.target == validate_target), knees[0] if knees else None)
        verdict = validate_formula(prior, vknee) if vknee is not None else None
        return cls(points=list(pts), prior=prior, knees=knees, verdict=verdict)
