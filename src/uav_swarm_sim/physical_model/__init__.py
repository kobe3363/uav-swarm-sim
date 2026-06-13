"""Physical-model layer (thesis layer 1): energy, kinematics, aerodynamics."""
from .aero_correction import AeroCorrection
from .drone_specs import PlatformSpec, build_spec
from .energy_model import EnergyModel
from .motion_model import (
    DubinsModel,
    HolonomicModel,
    MotionModel,
    make_motion_model,
)
from .vertical_segments import (
    VerticalProfile,
    landing_profile,
    takeoff_profile,
)

__all__ = [
    "AeroCorrection", "PlatformSpec", "build_spec", "EnergyModel",
    "MotionModel", "DubinsModel", "HolonomicModel", "make_motion_model",
    "VerticalProfile", "takeoff_profile", "landing_profile",
]
