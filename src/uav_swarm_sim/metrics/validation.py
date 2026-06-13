"""Directional reproduction of published aggregated metrics (§2.7.1).

Open-source data is unavailable for most sources, so the standard is direction
plus order-of-magnitude, not exact reproduction. This module provides the
verdict helpers (testable now) and a ``validate_all`` that consumes precomputed
comparison values. The real scenarios are wired by the Batch-6 experiment
harness, which runs the simulation and feeds the numbers in here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationRow:
    source: str
    claim: str
    our_value: float
    reference: str
    verdict: str  # "PASS (direction)" | "INFO (magnitude)" | "FAIL"


def directional(source: str, claim: str, value_a: float, value_b: float, expect: str, reference: str) -> ValidationRow:
    """expect 'less'  -> PASS iff value_a < value_b (a is the proposed method).
       expect 'greater' -> PASS iff value_a > value_b."""
    if expect == "less":
        ok = value_a < value_b
    elif expect == "greater":
        ok = value_a > value_b
    else:
        raise ValueError("expect must be 'less' or 'greater'")
    verdict = "PASS (direction)" if ok else "FAIL"
    return ValidationRow(source, claim, value_a, reference, verdict)


def magnitude(source: str, claim: str, value: float, lo: float, hi: float, reference: str) -> ValidationRow:
    inside = lo <= value <= hi
    verdict = "INFO (magnitude)" if inside else "FAIL"
    return ValidationRow(source, claim, value, reference, verdict)


def ratio_magnitude(source: str, claim: str, ratio: float, expect_order: float, reference: str, tol_decades: float = 1.0) -> ValidationRow:
    """PASS-as-INFO if the ratio is within tol_decades orders of magnitude of
    the expected order (used for the >60x and ~1e-4 s/task claims)."""
    if ratio <= 0 or expect_order <= 0:
        return ValidationRow(source, claim, ratio, reference, "FAIL")
    decades = abs(math.log10(ratio) - math.log10(expect_order))
    verdict = "INFO (magnitude)" if decades <= tol_decades else "FAIL"
    return ValidationRow(source, claim, ratio, reference, verdict)


def validate_all(results: dict) -> list[ValidationRow]:
    """Build the validation table from precomputed numbers.

    Expected keys (provided by the Batch-6 harness):
      workload_classic, workload_weighted        (Ruan-style balance)
      duration_classic, duration_weighted
      plan_time_heuristic, plan_time_tgc, scaling_n   (J. Liu scaling)
      replan_per_task_s                            (C. Liu replanning magnitude)
    """
    rows: list[ValidationRow] = []
    if {"workload_classic", "workload_weighted"} <= results.keys():
        rows.append(directional(
            "Ruan et al.", "weighted/tgc workload std << classic",
            results["workload_weighted"], results["workload_classic"],
            "less", "78-206 dm imbalance reduction",
        ))
    if {"duration_classic", "duration_weighted"} <= results.keys():
        rows.append(directional(
            "Ruan et al.", "mission duration decreases vs classic",
            results["duration_weighted"], results["duration_classic"],
            "less", "219.8 -> 157.8 s",
        ))
    if {"plan_time_heuristic", "plan_time_tgc"} <= results.keys():
        ratio = results["plan_time_heuristic"] / max(results["plan_time_tgc"], 1e-12)
        rows.append(ratio_magnitude(
            "J. Liu et al.", "TGC >60x faster group allocation at scale",
            ratio, 60.0, ">60x at scale", tol_decades=1.0,
        ))
    if "replan_per_task_s" in results:
        rows.append(magnitude(
            "C. Liu et al.", "per-task replanning time ~1e-4..1e-3 s",
            results["replan_per_task_s"], 1e-5, 1e-2, "1e-4..1e-3 s/task",
        ))
    return rows
