"""physical_model/metrics_definitions tests (isolated)."""
from __future__ import annotations

import numpy as np
import pytest

from uav_swarm_sim.physical_model.metrics_definitions import workload_std


def test_workload_std_zero_when_equal():
    assert workload_std({0: 100.0, 1: 100.0, 2: 100.0}) == pytest.approx(0.0)


def test_workload_std_matches_numpy():
    d = {0: 10.0, 1: 20.0, 2: 35.0, 3: 5.0}
    assert workload_std(d) == pytest.approx(float(np.std(list(d.values()))))
