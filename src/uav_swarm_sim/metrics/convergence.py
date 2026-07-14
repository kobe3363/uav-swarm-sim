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


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial success proportion.

    Returns ``(lo, hi, phat)`` with ``phat = k/n`` the point estimate and
    ``[lo, hi]`` the two-sided score interval at confidence ``z`` (default 95 %).
    Unlike the normal (Wald) interval it stays inside ``[0, 1]`` and behaves
    sensibly at the extremes ``k == 0`` and ``k == n`` -- exactly the regime the
    99 % target lives in. ``n == 0`` yields the vacuous ``(0.0, 1.0, nan)``.

    Moved here (verbatim) from ``experiments.spare_sizing`` so the metrics layer
    can use it without importing upward; ``spare_sizing`` re-exports it.
    """
    if n <= 0:
        return (0.0, 1.0, float("nan"))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi, phat)
