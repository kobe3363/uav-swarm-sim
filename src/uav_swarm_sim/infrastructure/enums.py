"""All enumerations used across the simulation.

Single home for every enum so that state, maneuver, platform, and algorithm
names are identical across the simulation, the metrics layer, the plots, and
the thesis text. Importing from here (rather than redefining locally) is what
keeps those names from drifting.
"""
from __future__ import annotations

from enum import Enum


class PlatformType(Enum):
    FIXED_WING = "FIXED_WING"
    MULTIROTOR = "MULTIROTOR"
    VTOL = "VTOL"


class AgentState(Enum):
    """Seven-state behavioral automaton.

    Base set (C. Liu et al. 2026): S0_IDLE, S1_TRANSIT, S2_MISSION, S_FAIL.
    Author's extensions: S3_RTH, S_OBS, S_SWAP.
    """
    S0_IDLE = "S0_IDLE"
    S1_TRANSIT = "S1_TRANSIT"
    S2_MISSION = "S2_MISSION"
    S3_RTH = "S3_RTH"
    S_SWAP = "S_SWAP"
    S_OBS = "S_OBS"
    S_FAIL = "S_FAIL"

    @property
    def is_airborne(self) -> bool:
        """True for states in which the drone is flying (consumes flight energy
        and is exposed to the failure hazard). Ground/service states are excluded.
        """
        return self in {
            AgentState.S1_TRANSIT,
            AgentState.S2_MISSION,
            AgentState.S3_RTH,
            AgentState.S_OBS,
        }


class ManeuverType(Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    COVERAGE = "COVERAGE"
    TURN = "TURN"
    DESCENT = "DESCENT"
    LAND = "LAND"
    HOVER = "HOVER"


class BatteryZone(Enum):
    HIGH = "HIGH"
    NOMINAL = "NOMINAL"
    CRITICAL = "CRITICAL"
    TERMINAL = "TERMINAL"


class DecompositionAlgo(Enum):
    """Labels match the thesis comparison output exactly.

    WEIGHTED_VORONOI is the thesis-facing name for the battery-weighted TGC
    decomposition (the central contribution).
    """
    CLASSIC_VORONOI = "classic_voronoi"
    TGC_BASIC = "tgc_basic"
    WEIGHTED_VORONOI = "weighted_voronoi"


class PlannerKind(Enum):
    DUBINS = "DUBINS"
    GRID = "GRID"


class EventType(Enum):
    FAILURE = "FAILURE"
    NEW_TASK = "NEW_TASK"
    SWAP_REQUEST = "SWAP_REQUEST"
    SWAP_DONE = "SWAP_DONE"
    OBSTACLE_THREAT = "OBSTACLE_THREAT"
    ZONE_COMPLETE = "ZONE_COMPLETE"
    MISSION_COMPLETE = "MISSION_COMPLETE"


class TierStrategy(Enum):
    HEURISTIC = "HEURISTIC"
    TGC = "TGC"
    COMPARE_BOTH = "COMPARE_BOTH"


class MissionType(Enum):
    """What kind of mission each drone performs.

    COVERAGE     -- sweep an assigned 2D area (boustrophedon); the default.
    TARGET_VISIT -- visit a set of discrete target points (a per-drone tour).
    """
    COVERAGE = "coverage"
    TARGET_VISIT = "target_visit"


class SensingMode(Enum):
    """Swarm-wide obstacle-sensing posture.

    PASSIVE -- low-power, short detection range; default. Drones notice a dynamic
               obstacle only when it is near (late/abrupt reaction).
    ACTIVE  -- high-power LIDAR, long detection range; drains scan power from every
               airborne drone. Entered when ANY agent detects an obstacle, so the
               whole swarm starts scanning together; reverts after a quiet hold.
    """
    PASSIVE = "passive"
    ACTIVE = "active"


class Outcome(Enum):
    """Terminal mission outcome (Phase 2, Task 2.2).

    Decided once per tick inside ``SimulationEngine.run()`` by a mutually-exclusive
    evaluation (failure is tested before success). Carried on ``MissionResult``.

    MISSION_SUCCESS    -- 100% of the partitioned area is covered AND every
                          surviving drone has returned to S0_IDLE.
    MISSION_FAILED     -- a physics-dictated halt: an AIRBORNE drone's battery
                          reached 0 (forced into S_FAIL mid-flight), or the shared
                          swap reserve was exhausted before coverage completed.
                          NOTE: hazard-induced S_FAIL (failure_model) does NOT
                          trigger this -- those failures are meant to populate
                          S_FAIL for the elevated-hazard Monte-Carlo / SMDP
                          statistics and the run continues via redistribution.
    MISSION_INCOMPLETE -- neither terminal condition fired before the run ended
                          (e.g. the sim.max_timesteps ceiling). Default outcome.
    """
    MISSION_SUCCESS = "MISSION_SUCCESS"
    MISSION_FAILED = "MISSION_FAILED"
    MISSION_INCOMPLETE = "MISSION_INCOMPLETE"
