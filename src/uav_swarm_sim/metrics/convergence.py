"""Monte-Carlo convergence criterion (Decision 3c).

Stop when the 95% CI half-width of the monitored quantity (pi_time(S2)) drops
below a configurable tolerance, with n >= n_min. Stronger than a fixed-N rule or
a change-window heuristic.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

Z_95 = 1.959963984540054


def ci_half_width(samples: Sequence[float], z: float = Z_95) -> float:
    """95% normal-approx CI half-width of the sample mean.

    Returns +inf for fewer than 2 samples (undefined sample std).
    """
    n = len(samples)
    if n < 2:
        return math.inf
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)  # ddof=1
    return z * math.sqrt(var / n)


def converged(samples: Sequence[float], tol: float, n_min: int) -> bool:
    return len(samples) >= n_min and ci_half_width(samples) <= tol
