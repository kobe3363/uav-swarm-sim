"""Per-replication initial battery state-of-charge generation."""
from __future__ import annotations

from scipy.stats import truncnorm

from .config import InitialSocConfig
from .rng import STREAM_INITIAL_SOC, RngFactory


def generate_initial_soc(
    cfg: InitialSocConfig,
    n_drones: int,
    rng_factory: RngFactory,
    replication: int,
) -> tuple[float, ...]:
    """Return one SoC fraction per drone ID without coupling other RNG streams."""
    if cfg.mode == "fixed":
        assert cfg.value is not None
        return (cfg.value,) * n_drones

    rng = rng_factory.stream(STREAM_INITIAL_SOC, replication)
    assert cfg.low is not None and cfg.high is not None
    if cfg.mode == "uniform":
        return tuple(float(x) for x in rng.uniform(cfg.low, cfg.high, size=n_drones))

    assert cfg.mode == "truncated_normal"
    assert cfg.mean is not None and cfg.std is not None
    lower = (cfg.low - cfg.mean) / cfg.std
    upper = (cfg.high - cfg.mean) / cfg.std
    samples = truncnorm.rvs(
        lower,
        upper,
        loc=cfg.mean,
        scale=cfg.std,
        size=n_drones,
        random_state=rng,
    )
    return tuple(float(x) for x in samples)
