"""Metrics / SMDP analysis layer: how the simulation proves anything."""
from .convergence import ci_half_width, converged, wilson_ci
from .efficiency_score import efficiency
from .mission_metrics import MissionMetrics, compute
from .monte_carlo import MCResult, SingleRunResult, run, single_run_from_history
from .smdp_convergence import (
    ConvergenceReport,
    StateConvergence,
    TransitionCI,
    convergence_report,
    format_table,
    pool_counts,
    report_from_counts,
    report_to_json,
)
from .smdp_estimator import STATE_ORDER, SmdpEstimate, estimate
from .state_history import Sojourn, StateHistory
from .stationary_distribution import embedded_pi, stationary, time_weighted_pi
from .validation import ValidationRow, validate_all

__all__ = [
    "StateHistory", "Sojourn", "MissionMetrics", "compute",
    "SmdpEstimate", "estimate", "STATE_ORDER",
    "embedded_pi", "time_weighted_pi", "stationary", "efficiency",
    "ci_half_width", "converged", "wilson_ci", "MCResult", "SingleRunResult", "run",
    "single_run_from_history", "ValidationRow", "validate_all",
    "ConvergenceReport", "StateConvergence", "TransitionCI",
    "convergence_report", "report_from_counts", "pool_counts",
    "report_to_json", "format_table",
]
