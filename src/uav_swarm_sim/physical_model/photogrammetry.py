"""Nadir-camera geometry for flat-ground photogrammetry.

All optical dimensions use millimetres.  Their ratio is dimensionless, so an
AGL height in metres produces a ground footprint in metres without a hidden
unit conversion.  The model deliberately excludes lens distortion, terrain
relief and camera tilt; those require calibration/terrain data that EXP-01 does
not provide.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhotogrammetrySolution:
    """Geometry and cadence derived for one height and coverage speed."""

    footprint_width_m: float
    footprint_length_m: float
    gsd_width_m_px: float
    gsd_length_m_px: float
    line_spacing_m: float
    photo_spacing_m: float
    nominal_interval_s: float


@dataclass(frozen=True)
class PhotogrammetryModel:
    """Landscape-oriented camera: width cross-track, height along-track."""

    sensor_width_mm: float
    sensor_height_mm: float
    focal_length_mm: float
    image_width_px: int
    image_height_px: int
    side_overlap: float
    forward_overlap: float
    min_photo_interval_s: float

    def solve(self, height_agl_m: float, coverage_speed_m_s: float) -> PhotogrammetrySolution:
        """Return footprint, GSD, overlap spacing and nominal photo interval."""
        if height_agl_m <= 0.0:
            raise ValueError("height_agl_m must be > 0")
        if coverage_speed_m_s <= 0.0:
            raise ValueError("coverage_speed_m_s must be > 0")

        width = height_agl_m * self.sensor_width_mm / self.focal_length_mm
        length = height_agl_m * self.sensor_height_mm / self.focal_length_mm
        line_spacing = width * (1.0 - self.side_overlap)
        photo_spacing = length * (1.0 - self.forward_overlap)
        return PhotogrammetrySolution(
            footprint_width_m=width,
            footprint_length_m=length,
            gsd_width_m_px=width / self.image_width_px,
            gsd_length_m_px=length / self.image_height_px,
            line_spacing_m=line_spacing,
            photo_spacing_m=photo_spacing,
            nominal_interval_s=photo_spacing / coverage_speed_m_s,
        )
