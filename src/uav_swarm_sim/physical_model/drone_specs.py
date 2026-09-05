"""The single homogeneous platform specification shared by every drone.

Homogeneity is enforced here: one PlatformSpec is built from config and handed
to every agent's energy and motion models. Coefficients are theoretical
approximations from typical specifications (no flight-data regression); the
README flags FW/VTOL tables as coarser than the quadrotor-validated baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..infrastructure.config import Config
from ..infrastructure.enums import ManeuverType, PlatformType
from .photogrammetry import PhotogrammetryModel, PhotogrammetrySolution


@dataclass(frozen=True)
class PlatformSpec:
    platform: PlatformType
    v_cruise: float
    v_coverage: float
    v_climb: float
    v_descent: float
    r_min_m: float
    omega_max: float
    power_w: dict[ManeuverType, float]
    battery_capacity_j: float
    dims_m: tuple[float, float, float]
    swath_width_m: float          # effective footprint after overlap
    climb_angle_rad: float
    ground_roll_energy_j: float
    mass_kg: float = 2.0          # 2.5D vertical-energy term ONLY (see TODO below)
    photogrammetry: PhotogrammetryModel | None = None
    coverage_altitude_m: float = 0.0

    # ------------------------------------------------------------------ #
    # TODO (Future Work): mass is DELIBERATELY confined to the vertical    #
    # energy term (the m*g*v_z climb work added in Batch 3). Horizontal    #
    # and hover power stay 100% table-driven (power_w) so the single-      #
    # layer-z0 case remains byte-identical to the validated 2D model.      #
    # When environmental effects (wind, rain, fog) or speed/altitude-      #
    # dependent aerodynamics are modelled, horizontal power MUST be re-     #
    # derived to couple with mass (and air density). Doing so will          #
    # intentionally break 2D byte-identity and require re-baselining the    #
    # regression fixtures.                                                  #
    # ------------------------------------------------------------------ #

    def speed_for(self, maneuver: ManeuverType) -> float:
        """Nominal speed for a maneuver (used by motion planning)."""
        if maneuver is ManeuverType.COVERAGE:
            return self.v_coverage
        if maneuver is ManeuverType.CLIMB or maneuver is ManeuverType.TAKEOFF:
            return self.v_climb
        if maneuver is ManeuverType.DESCENT or maneuver is ManeuverType.LAND:
            return self.v_descent
        # CRUISE, TURN, HOVER, IDLE default to cruise speed for planning
        return self.v_cruise

    def photogrammetry_at(self, altitude_m: float | None = None) -> PhotogrammetrySolution | None:
        """Solve the enabled camera at one layer's AGL; None is legacy mode."""
        if self.photogrammetry is None:
            return None
        height = self.coverage_altitude_m if altitude_m is None else altitude_m
        return self.photogrammetry.solve(height, self.v_coverage)

    def coverage_line_spacing_m(self, altitude_m: float | None = None) -> float:
        """Cross-track line spacing, camera-derived when EXP-01 is enabled."""
        solution = self.photogrammetry_at(altitude_m)
        return self.swath_width_m if solution is None else solution.line_spacing_m

    def coverage_photo_spacing_m(self, altitude_m: float | None = None) -> float | None:
        """Along-track photo distance, absent in the legacy camera model."""
        solution = self.photogrammetry_at(altitude_m)
        return None if solution is None else solution.photo_spacing_m


def build_spec(cfg: Config) -> PlatformSpec:
    p = cfg.platform
    pg = cfg.sensor.photogrammetry
    camera = None
    if pg.enabled:
        camera = PhotogrammetryModel(
            sensor_width_mm=pg.sensor_width_mm,
            sensor_height_mm=pg.sensor_height_mm,
            focal_length_mm=pg.focal_length_mm,
            image_width_px=pg.image_width_px,
            image_height_px=pg.image_height_px,
            side_overlap=pg.side_overlap,
            forward_overlap=pg.forward_overlap,
            min_photo_interval_s=pg.min_photo_interval_s,
        )
        effective_swath = camera.solve(
            cfg.env.coverage_altitude_m, p.v_coverage
        ).line_spacing_m
    else:
        effective_swath = cfg.sensor.swath_width_m * (1.0 - cfg.sensor.overlap_frac)

    # platform-specific semantic check beyond config's presence check:
    # multirotor / VTOL must have a meaningful (>0) hover power; FW ignores it.
    if p.type in (PlatformType.MULTIROTOR, PlatformType.VTOL):
        if p.power_w.get(ManeuverType.HOVER, 0.0) <= 0.0:
            raise ValueError(
                f"{p.type.value} requires a positive HOVER power coefficient"
            )

    return PlatformSpec(
        platform=p.type,
        v_cruise=p.v_cruise,
        v_coverage=p.v_coverage,
        v_climb=p.v_climb,
        v_descent=p.v_descent,
        r_min_m=p.r_min_m,
        omega_max=p.omega_max,
        power_w=dict(p.power_w),
        battery_capacity_j=cfg.fleet.battery_capacity_j,
        dims_m=cfg.fleet.drone_dims_m,
        swath_width_m=effective_swath,
        climb_angle_rad=p.climb_angle_rad,
        ground_roll_energy_j=p.ground_roll_energy_j,
        mass_kg=p.mass_kg,
        photogrammetry=camera,
        coverage_altitude_m=cfg.env.coverage_altitude_m,
    )
