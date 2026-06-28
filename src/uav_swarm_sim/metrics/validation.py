"""Internal validation of the simulation's own invariants.

External-literature reproduction was dropped: the source datasets of the anchor
papers are unavailable, so this study makes COMPARATIVE claims (method-vs-method
on paired seeds, where systematic modelling error cancels in the difference)
rather than absolute ones. Validation is therefore INTERNAL: it asserts the
properties that must hold for the comparative results to be trustworthy --
Monte-Carlo convergence, a well-formed stationary distribution, deterministic
reproducibility (the paired-design precondition), and the documented degeneracy
of the battery-weighted partition to its unweighted twin at full battery.

``directional`` and ``magnitude`` remain as generic verdict primitives;
``validate_all`` consumes precomputed internal numbers (provided by the
experiment harness / tests) and returns a verdict table. No external reference
values are cited.
"""
from __future__ import annotations

from dataclasses import dataclass

_REL_TOL = 1e-6
_PI_TOL = 1e-6
_PI_NONNEG_TOL = 1e-9


@dataclass(frozen=True)
class ValidationRow:
    source: str
    claim: str
    our_value: float
    reference: str
    verdict: str  # "PASS (internal)" | "PASS (direction)" | "INFO (magnitude)" | "FAIL"


# --------------------------------------------------------------------------- #
# generic verdict primitives                                                   #
# --------------------------------------------------------------------------- #
def boolean(source: str, claim: str, ok: bool, reference: str, value: float | None = None) -> ValidationRow:
    """A pass/fail check on an internal invariant."""
    return ValidationRow(
        source, claim, float(value) if value is not None else (1.0 if ok else 0.0),
        reference, "PASS (internal)" if ok else "FAIL",
    )


def directional(source: str, claim: str, value_a: float, value_b: float, expect: str, reference: str) -> ValidationRow:
    """expect 'less'    -> PASS iff value_a < value_b (a is the proposed method).
       expect 'greater' -> PASS iff value_a > value_b.

    Valid as an INTERNAL claim only when a and b are measured on the SAME seeds
    (paired design), so the direction is attributable to the method.
    """
    if expect == "less":
        ok = value_a < value_b
    elif expect == "greater":
        ok = value_a > value_b
    else:
        raise ValueError("expect must be 'less' or 'greater'")
    return ValidationRow(source, claim, value_a, reference, "PASS (direction)" if ok else "FAIL")


def magnitude(source: str, claim: str, value: float, lo: float, hi: float, reference: str) -> ValidationRow:
    inside = lo <= value <= hi
    return ValidationRow(source, claim, value, reference, "INFO (magnitude)" if inside else "FAIL")


# --------------------------------------------------------------------------- #
# internal validation table                                                    #
# --------------------------------------------------------------------------- #
def validate_all(results: dict) -> list[ValidationRow]:
    """Build the internal validation table from precomputed numbers.

    Recognised keys (all internal; each block is optional and emitted only when
    its keys are present):

      Monte-Carlo convergence:
        mc_converged (bool), mc_ci (float), mc_tol (float)
      Stationary distribution validity:
        pi_sum (float), pi_min (float)
      Deterministic reproducibility (paired-design precondition):
        energy_seed_a (float), energy_seed_b (float)   -- same seed, two runs
      Battery-weighting degeneracy (documented invariant):
        energy_weighted_full (float), energy_tgc_basic_full (float)
      Optional internal paired comparison (informational direction):
        workload_weighted (float), workload_classic (float)  -- same seeds
    """
    rows: list[ValidationRow] = []

    if {"mc_converged", "mc_ci", "mc_tol"} <= results.keys():
        ci, tol = results["mc_ci"], results["mc_tol"]
        rows.append(boolean(
            "internal", "Monte Carlo converged (CI half-width <= tol)",
            bool(results["mc_converged"]) and ci <= tol,
            f"CI={ci:.2e} <= tol={tol:.2e}", value=ci,
        ))

    if "pi_sum" in results:
        rows.append(magnitude(
            "internal", "stationary pi sums to 1",
            results["pi_sum"], 1.0 - _PI_TOL, 1.0 + _PI_TOL, "sum(pi) == 1",
        ))
    if "pi_min" in results:
        rows.append(boolean(
            "internal", "stationary pi is non-negative",
            results["pi_min"] >= -_PI_NONNEG_TOL,
            f"min(pi)={results['pi_min']:.2e} >= 0", value=results["pi_min"],
        ))

    if {"energy_seed_a", "energy_seed_b"} <= results.keys():
        a, b = results["energy_seed_a"], results["energy_seed_b"]
        rel = abs(a - b) / max(1.0, abs(a))
        rows.append(boolean(
            "internal", "deterministic on a fixed seed (paired-design precondition)",
            rel <= _REL_TOL, f"|dE|/E={rel:.2e} <= {_REL_TOL:.0e}", value=rel,
        ))

    if {"energy_weighted_full", "energy_tgc_basic_full"} <= results.keys():
        w, t = results["energy_weighted_full"], results["energy_tgc_basic_full"]
        rel = abs(w - t) / max(1.0, abs(t))
        rows.append(boolean(
            "internal", "battery-weighted degenerates to unweighted TGC at full battery",
            rel <= _REL_TOL, f"|Ew-Et|/Et={rel:.2e} <= {_REL_TOL:.0e}", value=rel,
        ))

    if {"workload_weighted", "workload_classic"} <= results.keys():
        rows.append(directional(
            "internal (paired)", "weighted workload std <= classic (same seeds)",
            results["workload_weighted"], results["workload_classic"],
            "less", "paired same-seed comparison",
        ))

    return rows
