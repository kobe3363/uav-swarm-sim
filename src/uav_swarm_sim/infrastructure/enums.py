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
    S_FERRY = "S_FERRY"  # repositioning between coverage strips, camera OFF (non-productive flight)

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
            AgentState.S_FERRY,
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

    Position-based baselines: CLASSIC_VORONOI (Euclidean Voronoi), KMEANS
    (position k-means), TGC_BASIC (unweighted topological). WEIGHTED_VORONOI is
    the thesis-facing name for the battery-weighted TGC decomposition (the
    central contribution).
    """
    CLASSIC_VORONOI = "classic_voronoi"
    KMEANS = "kmeans"
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


class LegRepair(Enum):
    """Outcome of pre-flight trajectory validation for a single smoothed leg
    (Task 2.5, Q1; see planning/trajectory_validation.py).

    A linear GVG/boustrophedon corridor is clear by construction, but Dubins
    smoothing can bulge a turn arc outside it and clip an obstacle buffer.
    ``plan_clear_leg`` validates every leg against the BUFFERED obstacle union
    before it is flown and repairs a clipping leg with a bounded ladder:

    CLEAN      -- the smoothed leg was already clear; returned unchanged. This is
                  the ONLY outcome for a holonomic platform (r_min == 0 produces
                  straight legs), so the multirotor baseline stays byte-identical.
    RESMOOTHED -- cleared by subdivide-and-resmooth: a midpoint was inserted on
                  the chord and each half re-Dubins'd (shorter arcs bulge less).
    LINEAR     -- fell back to the straight corridor chord at the same maneuver.
                  Provably clear (the skeleton came from the obstacle-aware
                  decomposition); kinematically abrupt, marginally cheaper.
    BLOCKED    -- even the chord clips: the corridor is genuinely obstructed (a
                  dynamic / newly-discovered obstacle straddles it). NOT a
                  smoothing artifact; the caller must escalate (runtime S_OBS
                  recovery, or a TGC reroute with a logged coverage gap) rather
                  than fly into it.
    """
    CLEAN = "clean"
    RESMOOTHED = "resmoothed"
    LINEAR = "linear"
    BLOCKED = "blocked"


class TelemetryEventKind(Enum):
    """Semantic verb for one row of the Phase 3 event-driven telemetry log
    (metrics/telemetry_log.py; consumed by the GPX and LLM-log exporters).

    Derived from the agent's state transition (or a run-level hook) so the
    Phase-4 LLM judge gets a salient label without inferring intent from raw
    state pairs. The transition's full from/to/reason are still carried alongside.
    """
    START = "START"            # initial state entry (t=0)
    STATE = "STATE"            # generic state transition
    OBSTACLE = "OBSTACLE"      # entered S_OBS (obstacle / threat encounter)
    SWAP_REQ = "SWAP_REQ"      # entered S_SWAP (battery swap requested)
    SWAP_DONE = "SWAP_DONE"    # left S_SWAP -> S0_IDLE (swap complete)
    FAIL = "FAIL"              # entered S_FAIL (drone lost)
    LEG_REPAIR = "LEG_REPAIR"  # pre-flight trajectory repair (Task 2.5 Q2 hook)
    TERMINAL = "TERMINAL"      # mission terminal verdict (success / failure)
