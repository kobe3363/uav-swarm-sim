"""Stationary distribution pi -- the primary stochastic metric.

embedded_pi solves pi = pi P (left eigenvector for eigenvalue 1): this is the
*visit frequency* of the embedded jump chain, NOT time fractions.

time_weighted_pi applies the CRITICAL correction the code must implement:
    pi_time[i] = pi_emb[i] * m_i / sum_j pi_emb[j] * m_j
A state visited often but briefly (e.g. S_OBS) must not outweigh a state visited
rarely but held long (e.g. S2). Without this renormalization by mean sojourn
times the efficiency score is subtly wrong. Both vectors are returned so the
correction is visible (plotted side by side), not buried.
"""
from __future__ import annotations

import numpy as np

from .smdp_estimator import SmdpEstimate


def embedded_pi(P: np.ndarray) -> np.ndarray:
    """Left eigenvector for eigenvalue 1, normalized to a probability vector.

    Solved as a constrained linear system (P^T - I) pi = 0, sum(pi) = 1 via
    least squares -- robust for small, possibly ill-conditioned chains.
    """
    n = P.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s <= 0:
        raise ValueError("degenerate embedded distribution (sum <= 0)")
    pi = pi / s
    resid = np.max(np.abs(pi @ P - pi))
    if resid > 1e-6:
        raise ValueError(f"embedded pi residual too large: {resid:.2e}")
    return pi


def time_weighted_pi(pi_emb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The mandatory embedded -> time correction."""
    w = pi_emb * m
    s = w.sum()
    if s <= 0:
        raise ValueError("degenerate time-weighted distribution (sum <= 0)")
    return w / s


def stationary(est: SmdpEstimate) -> tuple[np.ndarray, np.ndarray]:
    """Return (pi_embedded, pi_time). Refuses unless the chain is ergodic --
    the stationarity claim is only valid under the closed loop."""
    if not est.ergodic:
        raise ValueError(
            "chain is not ergodic (closed loop broken or truncated terminal "
            f"states {est.unreachable}); stationary distribution is undefined"
        )
    pi_emb = embedded_pi(est.P)
    pi_time = time_weighted_pi(pi_emb, est.mean_sojourn_s)
    return pi_emb, pi_time
