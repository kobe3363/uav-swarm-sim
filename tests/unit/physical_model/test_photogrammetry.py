"""Independent analytical fixtures for the EXP-01 camera model."""
from __future__ import annotations

import pytest

from uav_swarm_sim.physical_model.photogrammetry import PhotogrammetryModel


def _m4e() -> PhotogrammetryModel:
    return PhotogrammetryModel(
        sensor_width_mm=17.3,
        sensor_height_mm=13.0,
        focal_length_mm=12.0,
        image_width_px=5280,
        image_height_px=3956,
        side_overlap=0.70,
        forward_overlap=0.80,
        min_photo_interval_s=0.5,
    )


def test_m4e_geometry_at_100_m_and_10_m_s():
    solution = _m4e().solve(100.0, 10.0)

    # Hand-computed from the published formula, not from model output:
    # W=100*17.3/12; L=100*13/12; overlap spacings are 30% and 20%.
    assert solution.footprint_width_m == pytest.approx(144.16666666666666)
    assert solution.footprint_length_m == pytest.approx(108.33333333333333)
    assert solution.gsd_width_m_px == pytest.approx(0.02730429292929293)
    assert solution.gsd_length_m_px == pytest.approx(0.027384563532187394)
    assert solution.line_spacing_m == pytest.approx(43.25)
    assert solution.photo_spacing_m == pytest.approx(21.666666666666664)
    assert solution.nominal_interval_s == pytest.approx(2.1666666666666665)
    assert solution.nominal_interval_s >= _m4e().min_photo_interval_s


def test_footprint_scales_linearly_with_agl_height():
    low = _m4e().solve(50.0, 10.0)
    high = _m4e().solve(100.0, 10.0)
    assert high.footprint_width_m == pytest.approx(2.0 * low.footprint_width_m)
    assert high.footprint_length_m == pytest.approx(2.0 * low.footprint_length_m)
    assert high.line_spacing_m == pytest.approx(2.0 * low.line_spacing_m)


@pytest.mark.parametrize("height,speed", [(0.0, 10.0), (-1.0, 10.0), (100.0, 0.0)])
def test_solver_rejects_nonpositive_operating_inputs(height, speed):
    with pytest.raises(ValueError):
        _m4e().solve(height, speed)
