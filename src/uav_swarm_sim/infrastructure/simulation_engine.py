"""The orchestrator -- composes all layers into one mission run and owns the dt loop.

Build: GeoJSON -> EnvironmentMap (+ obstacles) -> GVG -> TGC -> launch site ->
decomposer (tier-selected or explicit) -> coverage plan per zone -> agents +
support objects. Loop: failure -> safety -> agents step -> swap station -> drain
events (failure/new-task -> redistribution; swap-done -> resume) -> completion
check. Deterministic in (config, master_seed, replication, algo, planner).

2.5D (Batch 2): obstacles are extruded prisms sliced into one 2D map per
coverage layer (a LayerStack). Drones are assigned to layers (Level 1) and the
existing decomposer runs per layer (Level 2). A single layer at the coverage
altitude is the whole world and reproduces the 2D pipeline byte-for-byte. Layer 0
remains the primary 2D graph for launch siting, RTH and redistribution; making
those per-layer is Batch 4.
"""
from __future__ import annotations

import logging
from time import perf_counter

from shapely.geometry import Polygon

from .profiling import phase, record
from .initial_soc import generate_initial_soc
from ..infrastructure.config import Config
from ..infrastructure.core_types import (
    CoveragePlan,
    DroneStateView,
    Event,
    MissionResult,
    Path,
    Pose,
)
from ..infrastructure.enums import (
    AgentState,
    DecompositionAlgo,
    EventType,
    MissionType,
    Outcome,
    SensingMode,
    ManeuverType,
    PlannerKind,
    TierStrategy,
)
from ..infrastructure.rng import (
    STREAM_FAILURES,
    STREAM_KMEANS_INIT,
    STREAM_LAUNCH_SAMPLING,
    STREAM_DYNOBS,
    STREAM_OBSTACLES,
    STREAM_TARGETS,
    RngFactory,
)
from ..physical_model.aero_correction import AeroCorrection
from ..physical_model.battery import Battery
from ..physical_model.drone_specs import build_spec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import make_motion_model
from ..planning.classic_voronoi import ClassicVoronoiDecomposer
from ..planning.coverage_path import boustrophedon
from ..planning.coverage_raster import CoverageRaster
from ..planning.energy_map import battery_tied_cell_m, build_energy_map
from ..planning.environment_map import LayerStack
from ..planning.geojson_parser import load_area
from ..planning.grid_planner import GridPlanner
from ..planning.kmeans_heuristic import KMeansHeuristicDecomposer
from ..planning.layer_planner import assign_to_layers, build_layer_graphs, decompose_layers
from ..planning.launch_site_optimizer import optimize as optimize_launch
from ..planning.lloyd_partition import LloydCvtDecomposer, LloydEnergyDecomposer
from ..planning.obstacle_generator import generate as generate_obstacles
from ..planning.target_mission import generate_targets, plan_target_mission
from ..planning.dynamic_obstacles import DynamicObstacleField
from ..planning.visibility_router import route_transit
from ..execution.sensing import SensingCoordinator
from ..planning.weighted_decomposition import (
    TgcBasicDecomposer,
    WeightedTgcDecomposer,
)
from ..metrics import mission_metrics
from ..metrics.state_history import StateHistory
from ..metrics.telemetry_log import TelemetryLog, FanoutRecorder
from ..execution.agent import Agent
from ..execution.events import EventBus
from ..execution.failure_model import FailureModel
from ..execution.fleet import Fleet, deploy_ring_poses
from ..execution.formation_manager import FormationManager
from ..execution.redistribution import Redistributor
from ..execution.rth_calculator import RthCalculator
from ..execution.safety_monitor import SafetyMonitor
from ..execution.state_machine import StateMachine
from ..execution.swap_station import SwapStation

_LOG = logging.getLogger(__name__)

# Coverage is treated as 100% complete at/above this fraction. Mirrors the
# existing abort gate (``coverage_frac < 0.999``) so success/abort agree.
_COVERAGE_COMPLETE_FRAC = 0.999

# FIX-B4: consecutive no-progress swap cycles before an agent counts as stalled.
_STALL_SWAP_BUDGET = 5


class StallDetector:
    """FIX-B4 (safety.stall_detector): swap-livelock net.

    ``observe`` is called once per SWAP_REQUEST with the agent's current
    coverage-leg index (``_cov_idx``). A swap request at the SAME index as the
    agent's previous request is one no-progress cycle; ``_STALL_SWAP_BUDGET``
    consecutive such cycles flag the agent stalled (a livelocked drone burns a
    pack + ~2.5 min per cycle forever -- see the demand-probe root-cause
    diagnosis). The engine halts the mission early when ``stalled`` is
    non-empty; the first request and any request after real progress reset the
    count, so a legitimately multi-sortie drone can never trip it.
    """

    def __init__(self, budget: int = _STALL_SWAP_BUDGET) -> None:
        self._budget = budget
        self._last_idx: dict[int, int] = {}
        self._count: dict[int, int] = {}
        self.stalled: set[int] = set()

    def observe(self, agent_id: int, cov_idx: int) -> bool:
        """Record one SWAP_REQUEST. Returns True when this observation hits the
        no-progress budget (the caller may then divert to skip-on-stall, EM-01
        Stage 4); the FIX-B4 halt path ignores the return value, so the flag-off
        behaviour is unchanged."""
        prev = self._last_idx.get(agent_id)
        count = self._count.get(agent_id, 0) + 1 if prev == cov_idx else 0
        self._count[agent_id] = count
        self._last_idx[agent_id] = cov_idx
        if count >= self._budget:
            self.stalled.add(agent_id)
            return True
        return False


class SimulationEngine:
    def __init__(
        self,
        cfg: Config,
        rng: RngFactory,
        replication: int = 0,
        algo: DecompositionAlgo | None = None,
        planner: PlannerKind = PlannerKind.DUBINS,
    ) -> None:
        self.cfg = cfg
        self.rng = rng
        self.replication = replication
        self.algo = algo
        self.planner = planner
        self._pending_tasks: list[tuple[Polygon, float]] = []

    def inject_task(self, polygon: Polygon, at_time_s: float) -> None:
        self._pending_tasks.append((polygon, at_time_s))

    # ------------------------------------------------------------------ #
    def _make_decomposer(self, motion):
        kmeans_rng = self.rng.stream(STREAM_KMEANS_INIT, self.replication)
        if self.algo is DecompositionAlgo.CLASSIC_VORONOI:
            return ClassicVoronoiDecomposer()
        if self.algo is DecompositionAlgo.TGC_BASIC:
            return TgcBasicDecomposer()
        if self.algo is DecompositionAlgo.WEIGHTED_VORONOI:
            return WeightedTgcDecomposer()
        if self.algo is DecompositionAlgo.KMEANS:
            # position-based k-means baseline (weighted=False), a first-class
            # comparison peer; same paired stream so it is reproducible.
            return KMeansHeuristicDecomposer(motion, weighted=False, rng=kmeans_rng)
        if self.algo is DecompositionAlgo.LLOYD_CVT:
            # EXP-07a: Lloyd/CVT over the EXP-02 coverage grid. The raster, the
            # staging poses, the launch pose and the energy map are all built
            # above this call, so they are handed in through the constructor --
            # the Decomposer ABC signature is untouched.
            return LloydCvtDecomposer(
                raster=self._require_raster(DecompositionAlgo.LLOYD_CVT),
                deploy_poses=self.deploy_poses,
                launch_pose=self.launch_pose,
                settings=self.cfg.planning.partition,
                energy_map=self.energy_map,
            )
        if self.algo is DecompositionAlgo.LLOYD_ENERGY:
            # EXP-07b: the same partitioner, an energy weight source. It needs
            # return costs at planning time, which is why the RTH calculator is
            # built above the decomposition (B-1).
            from ..planning.energy_balance import (
                DroneEnergyState, build_energy_balance_context,
            )

            raster = self._require_raster(DecompositionAlgo.LLOYD_ENERGY)
            if not self.cfg.planning.energy_balance.enabled:
                raise ValueError(
                    "--algo lloyd_energy requires planning.energy_balance.enabled = true"
                )
            capacity_j = self.spec.battery_capacity_j
            return LloydEnergyDecomposer(
                raster=raster,
                deploy_poses=self.deploy_poses,
                launch_pose=self.launch_pose,
                settings=self.cfg.planning.partition,
                energy_map=self.energy_map,
                energy_context=build_energy_balance_context(
                    self.cfg, self.em, self.spec, self.motion, self.env,
                    lambda pose, alt: self.rth.return_energy(pose, altitude_m=alt),
                    emap=self.energy_map, graph_cache=self._transit_graph_cache,
                ),
                drone_states=[
                    DroneEnergyState(i, self.deploy_poses[i],
                                     self.initial_soc_by_drone[i] * capacity_j, False)
                    for i in range(self.cfg.fleet.n_drones)
                ],
                altitude_m=self.layers.altitude(0),
                capacity_j=capacity_j,
            )
        # no explicit algo: pick by scale tier
        from ..execution.algorithm_selector import select

        # EXP-07 (D-3): an experiment run must name its algorithm. The tier
        # auto-selection resolves two DIFFERENT implementations
        # (KMeansHeuristicDecomposer(weighted=True) and WeightedTgcDecomposer)
        # and both report as `weighted_voronoi`, so leaving it implicit makes a
        # run's algorithm identity unrecoverable from its output.
        if self.cfg.mission.experiment_mode:
            raise ValueError(
                "mission.experiment_mode forbids fleet-size algorithm auto-selection; "
                "name the decomposition algorithm explicitly (e.g. --algo tgc_basic)"
            )
        strat = select(self.cfg.fleet.n_drones, self.cfg.tier_thresholds)
        if strat is TierStrategy.HEURISTIC:
            return KMeansHeuristicDecomposer(motion, weighted=True, rng=kmeans_rng)
        return WeightedTgcDecomposer()

    def _require_raster(self, algo: DecompositionAlgo):
        """The grid partitioners consume EXP-02 cells, so the raster must exist.

        Validated here rather than in ``load_config`` because the algorithm comes
        from the CLI, which the config loader never sees.
        """
        if self._mission_type is not MissionType.COVERAGE:
            raise ValueError(f"--algo {algo.value} requires mission.type = coverage")
        if self.coverage_raster is None:
            raise ValueError(
                f"--algo {algo.value} requires coverage.raster_enabled = true "
                "(which requires sensor.photogrammetry.enabled = true)"
            )
        if len(self.layers) != 1:
            raise ValueError(f"--algo {algo.value} requires exactly one coverage layer")
        return self.coverage_raster

    def _build(self):
        cfg = self.cfg
        self.spec = build_spec(cfg)
        if self.spec.photogrammetry is not None and self.planner is PlannerKind.GRID:
            raise ValueError(
                "sensor.photogrammetry.enabled is not supported by the fixed-cell GRID "
                "coverage planner; use the boustrophedon/DUBINS planner"
            )
        self.motion = make_motion_model(self.spec)
        self.em = EnergyModel(self.spec)
        self.aero = AeroCorrection(cfg.aero, self.spec.platform)

        with phase("build.load_area"):
            area = load_area(cfg.env.geojson_path)
        obs_rng = self.rng.stream(STREAM_OBSTACLES, self.replication)
        self.initial_soc_by_drone = generate_initial_soc(
            cfg.battery.initial_soc, cfg.fleet.n_drones, self.rng, self.replication
        )
        # 2.5D: slice the extruded prisms into one 2D map per coverage layer.
        # A single layer at the coverage altitude (with unbounded prisms) is the
        # whole world and reproduces the 2D map exactly. Build per-layer GVG+TGC
        # once; layer 0 stays the primary 2D graph for launch siting, RTH and
        # redistribution (those become per-layer in Batch 4).
        with phase("build.env_obstacles"):
            obstacles = generate_obstacles(area, cfg.env, obs_rng)
            self.layers = LayerStack(
                area, obstacles, cfg.layers.altitudes_m, cfg.env.clearance_buffer_m
            )
        with phase("build.gvg_tgc"):
            self.layer_graphs = build_layer_graphs(
                self.layers, gvg_sample_step_m=20.0, gvg_spur_min_m=30.0
            )
        self.env, self.tgc = self.layer_graphs.by_layer[0]
        self.planning_time_s = self.layer_graphs.planning_time_s

        # launch site optimization (on layer 0 / primary graph)
        # Launch siting is a per-scenario decision, NOT a per-replication Monte-
        # Carlo draw: the pad must be identical across replications so paired-seed
        # variance reflects environment/failure draws only (matches the standalone
        # analytical scripts, which all site with stream(STREAM_LAUNCH_SAMPLING, 0)).
        # Pinning to replication 0 makes tgc_basic/weighted/classic byte-
        # deterministic under lambda=0 clean; obstacles (STREAM_OBSTACLES) and
        # failures (STREAM_FAILURES) still vary per replication as intended.
        launch_rng = self.rng.stream(STREAM_LAUNCH_SAMPLING, 0)
        with phase("build.launch_opt"):
            self.launch_pose, self.site_scores = optimize_launch(
                cfg.launch, self.tgc, self.env, self.motion, self.em, self.aero,
                self.spec, cfg.fleet.n_drones, launch_rng, cfg.env.coverage_altitude_m,
                initial_soc_by_drone=self.initial_soc_by_drone,
            )

        # EM-01 Stage 1 (rth.energy_map, default OFF => byte-identical): build
        # the per-replication energy cost-to-go map once, now that env + base
        # are known. The RTH decide / routing consumers shipped in later stages
        # (see execution/rth_calculator.py), each gated by its own flag.
        # Design doc retired -- retrieve with
        # `git show 493e8d2:docs/proposals/energy_map_rth.md`
        # (context: docs/archive/PROJECT_HISTORY.md section 8).
        # Deterministic, no RNG draw.
        self.energy_map = None
        if cfg.rth.energy_map.enabled:
            emc = cfg.rth.energy_map
            cell = emc.cell_m if emc.cell_m is not None else battery_tied_cell_m(
                self.spec.battery_capacity_j,
                self.spec.power_w[ManeuverType.CRUISE], self.spec.v_cruise,
            )
            with phase("build.energy_map"):
                # Stage 2 (author's universal-extent rule): inflate the grid by
                # the ferry margin so every physically reachable pose is IN the
                # grid by construction -- coverage plans do not exist yet at
                # this point (they are built below), so the extent comes from
                # bbox(area U base) + operating_margin_m, which bounds every
                # transit/ferry/RTH excursion the routers can produce.
                self.energy_map = build_energy_map(
                    self.env, self.launch_pose, cell, self.em, self.spec.v_cruise,
                    yellow_penalty=emc.yellow_penalty,
                    red_threshold=emc.red_threshold,
                    margin_m=cfg.coverage.operating_margin_m,
                )

        # 2.5D (Task 2.4): distribute the N drones on a ring around the launch
        # pose instead of stacking them at one (x, y). This is the single source
        # of every drone's t=0 pose -- it feeds BOTH the decomposer seeds
        # (init_views below) and the agents' home/spawn pose, so the partition no
        # longer collapses on identical seeds and the SafetyMonitor sees no
        # overlap. N == 1 yields R == 0 (the base pose), keeping the single-drone
        # single-layer case byte-identical. See fleet.deploy_ring_poses.
        self.deploy_poses = deploy_ring_poses(
            self.launch_pose, cfg.fleet.n_drones, self.spec.dims_m,
            cfg.safety.min_separation_m,
        )

        # FIX-B1: obstacle-aware S1 transit routing (coverage.transit_free_space,
        # default OFF => every transit stays the straight CRUISE chord,
        # byte-identical). Uses the layer-0 map, same as launch siting and RTH.
        if cfg.coverage.transit_free_space:
            _env0, _cov = self.env, cfg.coverage
            # E3: per-replication visibility-graph cache. The engine is built
            # fresh per replication (and per --jobs spawn worker), so this dict is
            # naturally per-replication/per-worker and holds no cross-replication
            # state. The cache is byte-identical to the uncached build.
            self._transit_graph_cache: dict = {}
            self._transit_planner = lambda a, b: route_transit(
                a, b, self.motion, _env0, enabled=True,
                operating_area=_cov.operating_area,
                margin_m=_cov.operating_margin_m,
                graph_cache=self._transit_graph_cache,
            )
        else:
            self._transit_graph_cache = {}
            self._transit_planner = None

        # EXP-07b (B-1): hoisted above the decomposition so planning can reach
        # return_energy. Side-effect free, so every other run is unaffected.
        self.rth = RthCalculator(
            self.em, self.motion, self.spec, cfg.rth, self.launch_pose,
            cfg.env.coverage_altitude_m, self.env,
            # EM-01 Stage 2: None when enabled is off; with enabled=True but
            # decide=False the calculator consumes nothing (Stage-1 gate test
            # stays green) -- the decide sub-flag is checked inside.
            energy_map=self.energy_map,
        )

        # --- mission planning: area coverage OR target visit -------------- #
        self._mission_type = cfg.mission.type
        self._weight_targets = cfg.mission.weight_targets_by_battery
        # EXP-04: same-battery lifecycle (validated in config to require an
        # area-coverage mission with the EXP-02 raster). Default False => every
        # branch keyed on it below is the pre-EXP-04 code path.
        self._no_swap = cfg.mission.no_swap_mode
        self.coverage_raster = None
        if (self._mission_type is MissionType.COVERAGE
                and cfg.coverage.raster_enabled):
            if len(self.layers) != 1:
                raise ValueError(
                    "coverage.raster_enabled currently requires exactly one coverage layer"
                )
            solution = self.spec.photogrammetry_at(self.layers.altitude(0))
            if solution is None:
                raise ValueError(
                    "coverage.raster_enabled requires sensor.photogrammetry.enabled"
                )
            self.coverage_raster = CoverageRaster(
                self.env.target_space,
                self.env.plannable_space,
                cfg.coverage.raster_cell_m,
            )
        init_views = [
            DroneStateView(i, self.initial_soc_by_drone[i], self.deploy_poses[i])
            for i in range(cfg.fleet.n_drones)
        ]
        self.assignment = {}          # drone_id -> list[(x, y)] (target mode only)
        self.layer_of: dict[int, int] = {}   # drone_id -> assigned layer index
        self.decomposer = None
        if self._mission_type is MissionType.TARGET_VISIT:
            tgt_rng = self.rng.stream(STREAM_TARGETS, self.replication)
            self.targets = generate_targets(self.env, cfg.mission, tgt_rng)
            with phase("build.decompose"):
                self.partition, self.plans, self.assignment = plan_target_mission(
                    self.targets, init_views, self.launch_pose, self.motion,
                    self.spec, self.em, weight_by_battery=self._weight_targets,
                )
        else:
            # EXP-07b (B-1): the energy-weighted partitioner needs return costs at
            # PLANNING time, so the RTH calculator is constructed here rather than
            # after the decomposition. Its __init__ is pure -- attribute
            # assignment plus the landing_profile builder, its own empty cache and
            # zeroed counters; it draws no RNG stream and does not mutate the
            # energy map -- so moving it changes nothing for any other run.
            #
            # n_map_hits / n_map_fallbacks / n_route_fallbacks ARE reported
            # metrics (experiments/run_rth_ab.py), and the partitioner can query
            # return energy hundreds of times, so the whole decomposition is
            # wrapped in the same save/restore the t=0 diagnostics use below.
            counters = (self.rth.n_map_hits, self.rth.n_map_fallbacks,
                        self.rth.n_route_fallbacks)
            try:
                self.decomposer = self._make_decomposer(self.motion)
                # Level 1: assign drones to layers (single-layer => all on layer 0).
                # Level 2: the reused decomposer runs per layer over its sliced map.
                layer_assignment = assign_to_layers(
                    init_views, self.layers, cfg.layers.assignment_policy
                )
                self.layer_of = {d.id: idx for idx, ds in layer_assignment.items() for d in ds}
                with phase("build.decompose"):
                    self.partition = decompose_layers(
                        self.layer_graphs, layer_assignment, self.decomposer
                    )
            finally:
                (self.rth.n_map_hits, self.rth.n_map_fallbacks,
                 self.rth.n_route_fallbacks) = counters
            self.plans = {}
        # EXP-07: the grid partitioners record how the partition was reached
        # (convergence, dropped cells, per-drone weights/areas). Absent -- and
        # therefore absent from the run output -- for every other algorithm.
        self.partition_diagnostics = getattr(self.decomposer, "diagnostics", None)

        # support objects
        sm = StateMachine(cfg.battery_zones, zone_demotion=cfg.rth.energy_map.zone_demotion,
                          no_swap_mode=self._no_swap)
        self.bus = EventBus()
        self.history = StateHistory()
        # Phase 3 telemetry: optional, OFF by default, a read-only probe. When on,
        # each agent's single recorder fans out to [StateHistory, TelemetryLog];
        # when off, the recorder IS StateHistory -> byte-identical to before.
        if cfg.telemetry.enabled:
            self.telemetry = TelemetryLog(fix_interval_s=cfg.telemetry.fix_interval_s)
            recorder = FanoutRecorder([self.history, self.telemetry])
        else:
            self.telemetry = None
            recorder = self.history
        self.swap_station = SwapStation(
            cfg.swap, self.launch_pose, cfg.fleet.total_reserve_batteries
        )
        # FIX-B4: swap-livelock net (default OFF => no tracking, byte-identical)
        self._stall = StallDetector() if cfg.safety.stall_detector else None
        # EM-01 Stage 4: skip-on-stall (validated to require the detector)
        self._stall_skip = cfg.safety.stall_skip and self._stall is not None
        self.safety = SafetyMonitor(self.layers, self.aero, cfg.safety, self.motion)
        # dynamic obstacles + swarm sensing (feature is OFF unless enabled in config)
        self.sensing = SensingCoordinator(cfg.dynamic_obstacles, cfg.safety)
        if cfg.dynamic_obstacles.enabled and cfg.dynamic_obstacles.count > 0:
            dyn_rng = self.rng.stream(STREAM_DYNOBS, self.replication)
            self._dynfield = DynamicObstacleField(
                self.env, cfg.dynamic_obstacles.count, cfg.dynamic_obstacles.speed_m_s,
                cfg.dynamic_obstacles.size_m, dyn_rng, self.layers.n_layers,
            )
        else:
            self._dynfield = None
        self.formation = FormationManager(self.aero, cfg.aero, self.spec.platform)
        self.failure = FailureModel(cfg.failure, self.rng.stream(STREAM_FAILURES, self.replication))
        rth = self.rth          # EXP-07b: built before the decomposition (see _build)

        grid = GridPlanner(self.env, cell_m=50.0) if self.planner is PlannerKind.GRID else None

        # agents + plans (per-zone boustrophedon coverage plan + entry transit)
        agents: list[Agent] = []
        with phase("build.coverage_plan"):
            for i in range(cfg.fleet.n_drones):
                battery = Battery(
                    self.spec.battery_capacity_j,
                    cfg.battery_zones,
                    self.initial_soc_by_drone[i],
                )
                i_layer = self.layer_of.get(i, 0)
                agent = Agent(i, self.spec, self.motion, self.em, battery, sm, rth,
                              self.formation, self.deploy_poses[i], recorder=recorder,
                              layer=i_layer, coverage_altitude_m=self.layers.altitude(i_layer),
                              sensor_power_w=cfg.sensor.sensor_power_w,
                              transit_planner=self._transit_planner,
                              photo_spacing_m=(
                                  self.spec.coverage_photo_spacing_m(self.layers.altitude(i_layer))
                                  if self._mission_type is MissionType.COVERAGE else None
                              ),
                              coverage_observer=self._coverage_observer(i_layer))
                if self._mission_type is MissionType.TARGET_VISIT:
                    plan = self.plans.get(i)
                    if plan is not None and plan.waypoints:
                        transit = self.motion.plan(self.deploy_poses[i], plan.waypoints[0].pose,
                                                   ManeuverType.CRUISE)
                        agent.assign(plan, transit)
                else:
                    zone = self.partition.zones.get(i)
                    if zone is not None:
                        plan = (grid.coverage(zone, self.spec) if grid is not None
                                else boustrophedon(zone, self.spec, self.motion, self.em,
                                                   env=self.env, coverage=cfg.coverage,
                                                   altitude_m=self.layers.altitude(i_layer)))
                        entry_pose = self._coverage_entry_pose(plan, zone.entry_pose)
                        transit = self._plan_transit(self.deploy_poses[i], entry_pose)
                        agent.assign(plan, transit)
                        self.plans[i] = plan
                agents.append(agent)

        if cfg.planning.energy_balance.enabled:
            from ..planning.energy_balance import (
                DroneEnergyState, ZoneEnergyEstimate, build_energy_balance_context,
                estimate_fast, estimate_path,
            )

            ctx = build_energy_balance_context(
                cfg, self.em, self.spec, self.motion, self.env,
                lambda pose, alt: rth.return_energy(pose, altitude_m=alt),
                emap=self.energy_map, graph_cache=self._transit_graph_cache,
            )
            self.energy_balance_t0: dict[int, dict[str, ZoneEnergyEstimate]] = {}
            # Return queries increment diagnostics; t=0 estimates must not
            # change the execution's map-hit/fallback observations.
            map_counts = rth.n_map_hits, rth.n_map_fallbacks
            try:
                for agent in agents:
                    zone = self.partition.zones.get(agent.id)
                    if zone is None:
                        continue
                    state = DroneEnergyState(
                        agent.id, self.deploy_poses[agent.id], agent.battery.level_j, False,
                    )
                    self.energy_balance_t0[agent.id] = {
                        "fast": estimate_fast(ctx, state, zone, self.coverage_raster),
                        "path": estimate_path(ctx, state, zone, self.coverage_raster),
                    }
            finally:
                rth.n_map_hits, rth.n_map_fallbacks = map_counts

        self.fleet = Fleet(agents)
        self.formation.register_departure(agents)
        self.redistributor = (
            None if self._mission_type is MissionType.TARGET_VISIT else Redistributor(
                self.decomposer if isinstance(self.decomposer, (WeightedTgcDecomposer,))
                else WeightedTgcDecomposer(),
                self.layer_graphs, self.motion, self.em, self.spec,
                coverage=cfg.coverage,
                layer_altitudes=cfg.layers.altitudes_m,
                remaining_work_provider=(
                    (lambda: self.coverage_raster.uncovered_plannable_geometry)
                    if self.coverage_raster is not None else None
                ),
            )
        )
        self.replan_times: list[float] = []
        # EXP-04 terminal diagnostics. ``_terminal_reason`` is filled in both
        # modes (the pre-existing telemetry reason strings); the other two only
        # ever receive entries under no_swap_mode.
        self._terminal_reason: str | None = None
        self._retirements: list[tuple[int, float, bool]] = []   # (agent_id, t, work_released)
        self._losses: list[tuple[int, float, str]] = []         # (agent_id, t, cause)

        # Phase 3: bind telemetry to the live fleet (so it can read pose/battery/
        # energy) and stamp the run header, before any sojourn opens.
        if self.telemetry is not None:
            self.telemetry.bind_fleet(self.fleet)
            self.telemetry.set_header(self._telemetry_header())

        # open initial S0 sojourns (via the recorder so telemetry, when enabled,
        # sees each drone's t=0 entry; the recorder IS history when disabled)
        for a in agents:
            recorder.open(a.id, AgentState.S0_IDLE, 0.0)

    # ------------------------------------------------------------------ #
    def run(self) -> MissionResult:
        self._build()
        cfg = self.cfg
        dt = cfg.sim.dt_s
        complete = False
        self._outcome = Outcome.MISSION_INCOMPLETE
        t = 0.0
        self._last_fix_t = -1e9   # Phase 3: periodic GPX position-fix clock
        _loop_t0 = perf_counter()  # profiling.record no-ops when disabled (byte-identical)
        for step in range(cfg.sim.max_timesteps):
            t = step * dt
            self.failure.step(self.fleet.airborne(), dt, t, self.bus)
            self.safety.step(self.fleet.active(), t, self.bus)
            if self._dynfield is not None:
                self._dynfield.step(dt)
                self.sensing.step(self.fleet.active(), self._dynfield, t, self.bus)
            for a in self.fleet.active():
                a.step(dt, t, self.bus)
                self.history.record_battery(a.id, t, a.battery.frac)
            # proactive scanning is expensive: drain LIDAR power while active
            scan_w = self.sensing.scan_power_w()
            if scan_w > 0.0:
                for a in self.fleet.airborne():
                    e = scan_w * dt
                    a.battery.drain(e)
                    a.energy_consumed_j += e
            self.swap_station.step(dt, self.bus)
            for poly, at in self._pending_tasks:
                if abs(t - at) < dt / 2:
                    self.bus.publish(Event(EventType.NEW_TASK, t, {"polygon": poly}))
            self._route_events(t)
            # log every agent's (x, y, state) after the tick settles, for 2D replay
            for a in self.fleet.agents.values():
                self.history.record_position(a.id, t, a.pose.x, a.pose.y, a.state)
            if self._dynfield is not None:
                self.history.record_dynamic_obstacles(t, self._dynfield.snapshot(), self.sensing.mode)
            # Phase 3: coarse periodic position fixes so long uniform phases still
            # render as lines in GPX (telemetry off -> this block is skipped).
            if self.telemetry is not None and (t - self._last_fix_t) >= self.telemetry.fix_interval_s:
                self._last_fix_t = t
                for a in self.fleet.active():
                    self.telemetry.record_fix(a.id, t)
            # Phase 2 (Tasks 2.1b + 2.2): mutually-exclusive terminal evaluation,
            # right after event routing + position logging. Failure is tested
            # before success; the first match halts the dt loop.
            outcome = self._evaluate_terminal(t)
            # FIX-B4: swap-livelock early halt. Checked only after the regular
            # terminal evaluation so a same-tick genuine terminal outcome always
            # wins; the outcome stays MISSION_INCOMPLETE (the same label the
            # max_timesteps burn would eventually produce), the stalled agents
            # are reported via MissionResult.stalled_agents.
            if outcome is None and self._stall is not None and self._stall.stalled:
                outcome = Outcome.MISSION_INCOMPLETE
                self._terminal_reason = "stall_livelock"
            if outcome is not None:
                self._outcome = outcome
                complete = outcome is Outcome.MISSION_SUCCESS
                break
        record("dt_loop", perf_counter() - _loop_t0)
        if self._terminal_reason is None:
            # fell out of the dt loop: the sim.max_timesteps cap. Drones still
            # airborne are reported as such -- never landed by fiat (EXP-04).
            self._terminal_reason = "max_timesteps"
        airborne_at_end = tuple(sorted(a.id for a in self.fleet.airborne()))

        t_end = t
        self.history.finalize(t_end)
        coverage_frac = self._coverage_frac()
        if self.telemetry is not None:
            self.telemetry.finalize(t_end)
            self.telemetry.set_summary(
                outcome=self._outcome.value,
                coverage_frac=round(coverage_frac, 4),
                t_end_s=round(t_end, 1),
                n_failed=self.fleet.n_failed,
                pool_exhausted=self.swap_station.pool_exhausted,
                reserve_remaining=self.swap_station.reserve_remaining,
                **self.telemetry.derive_counts(),
            )
            with phase("telemetry_export"):
                self._export_telemetry()
        with phase("metrics_compute"):
            metrics = mission_metrics.compute(
                self.history, self.fleet, self.partition, t_end,
                planning_time_s=self.planning_time_s,
                replan_times_s=tuple(self.replan_times),
                coverage_frac=coverage_frac,
            )
        aborted = (
            (not complete)
            or (
                len(self.fleet.active()) == 0
                and coverage_frac < self._coverage_complete_frac()
            )
        )
        stalled = tuple(sorted(self._stall.stalled)) if self._stall is not None else ()
        return MissionResult(metrics, self.history, self.partition, aborted, coverage_frac,
                             cfg.config_hash, self._outcome, stalled_agents=stalled,
                             skipped_legs=self._skipped_legs(),
                             photo_events=self._photo_events(),
                             target_coverage_frac=self._target_coverage_frac(),
                             terminal_reason=self._terminal_reason,
                             airborne_at_end=airborne_at_end,
                             retired_agents=tuple(aid for aid, _, _ in self._retirements),
                             work_releases=tuple((aid, rt) for aid, rt, rel in self._retirements if rel),
                             losses=tuple(self._losses),
                             initial_soc_by_drone=self.initial_soc_by_drone,
                             energy_balance_t0=getattr(self, "energy_balance_t0", None),
                             partition_diagnostics=getattr(self, "partition_diagnostics", None))

    def _photo_events(self):
        """Stable fleet-wide event order for EXP-01 and the later EXP-11 schema."""
        events = [event for agent in self.fleet.agents.values() for event in agent.photo_events]
        return tuple(sorted(
            events,
            key=lambda event: (
                event.t_s, event.agent_id, event.coverage_leg_index,
                event.distance_on_strip_m,
            ),
        ))

    def _plan_transit(self, a: Pose, b: Pose) -> Path:
        """An S1 transit leg: the straight CRUISE chord, unless FIX-B1
        (coverage.transit_free_space) routes a blocked chord around the
        buffered obstacles at plan time. Default OFF => exactly the chord."""
        if self._transit_planner is not None:
            return self._transit_planner(a, b)
        return self.motion.plan(a, b, ManeuverType.CRUISE)

    def _coverage_observer(self, layer: int):
        if self.coverage_raster is None:
            return None
        solution = self.spec.photogrammetry_at(self.layers.altitude(layer))
        if solution is None:
            return None
        return lambda old, new: self.coverage_raster.record_segment(
            old,
            new,
            solution.footprint_width_m,
            solution.footprint_length_m,
        )

    def _coverage_entry_pose(self, plan: CoveragePlan, legacy_entry: Pose) -> Pose:
        """Target the first strip when photos are enabled; preserve legacy transit."""
        if self.spec.photogrammetry is not None and plan.waypoints:
            return plan.waypoints[0].pose
        return legacy_entry

    # ------------------------------------------------------------------ #
    def _route_events(self, t: float) -> None:
        for e in self.bus.drain():
            if e.type is EventType.FAILURE:
                aid = e.payload.get("agent_id")
                self.fleet.kill(aid, t)
                if self._no_swap and all(l[0] != aid for l in self._losses):
                    # EXP-04: a hazard loss counts as FAILED once the fleet
                    # settles (a depletion re-published as FAILURE was already
                    # recorded with its own cause and is not double-counted).
                    self._losses.append((aid, t, "hazard_failure"))
                self._redistribute(e, t)
            elif e.type is EventType.NEW_TASK:
                self._redistribute(e, t)
            elif e.type is EventType.SWAP_REQUEST:
                aid = e.payload.get("agent_id")
                if self._stall is not None:
                    a = self.fleet.agents.get(aid)
                    if a is not None:
                        hit = self._stall.observe(aid, a._cov_idx)
                        # EM-01 Stage 4 (safety.stall_skip): the budget-hitting
                        # cycle forfeits the unreachable leg instead of halting
                        # the run -- the agent is un-flagged (the mission goes
                        # on), the skip retargets its post-swap resume, and the
                        # gap is accounted in skipped_legs / MISSION_PARTIAL.
                        # Boustrophedon only: tour plans have no strip/connector
                        # structure and their coverage_frac would credit a
                        # skipped target as visited.
                        if (hit and self._stall_skip
                                and self._mission_type is not MissionType.TARGET_VISIT):
                            self._stall.stalled.discard(aid)
                            a.skip_stuck_leg()
                self.swap_station.request(aid, t)
            elif e.type is EventType.SWAP_DONE:
                a = self.fleet.agents.get(e.payload.get("agent_id"))
                if a is not None:
                    a.signal_swap_done()
            elif e.type is EventType.UAV_RETIRED:
                # EXP-04: record the same-battery touchdown and the one-time
                # release of its uncovered work. Deliberately NO redistribution
                # here -- re-partitioning the released cells among the
                # remaining workers is EXP-08's trigger policy.
                self._retirements.append((
                    e.payload.get("agent_id"), e.t, bool(e.payload.get("work_released")),
                ))
            # OBSTACLE_THREAT is informational (signal already set by the monitor)

    def _redistribute(self, e: Event, t: float) -> None:
        # EXP-04: only non-retired survivors can take work; == active() in
        # every legacy run (nobody retires without no_swap_mode).
        active = self.fleet.workers()
        if not active:
            return
        if self._mission_type is MissionType.TARGET_VISIT:
            self._redistribute_targets(active, t)
            return
        new_part, new_plans = self.redistributor.handle(e, self.fleet, self.partition, self.plans, t)
        self.replan_times.append(self.redistributor.last_replan_time_s)
        # EXP-07: redistribution runs its OWN decomposer -- weighted TGC unless
        # the run's decomposer is one of its subclasses -- so from here on the
        # zones are no longer the ones the grid partitioner produced. The
        # partition diagnostics stay (they are the honest record of how planning
        # started) but are stamped with what replaced them, so no reader can take
        # them for a description of the zones actually being flown. Marked once:
        # the first supersession is the one that ends the recorded partition.
        diagnostics = getattr(self, "partition_diagnostics", None)
        if diagnostics is not None and getattr(diagnostics, "superseded_by", None) is None:
            diagnostics.superseded_by = {
                "decomposer": type(self.redistributor.decomposer).__name__,
                "trigger": e.type.value if hasattr(e.type, "value") else str(e.type),
                "t_s": float(t),
            }
        self.partition = new_part
        self.plans = new_plans
        for a in active:
            zone = new_part.zones.get(a.id)
            if zone is None:
                if self.coverage_raster is not None:
                    plan = new_plans[a.id]
                    transit = self.motion.plan(a.pose, a.pose, ManeuverType.CRUISE)
                    a.adopt_plan(plan, transit)
                continue
            plan = new_plans[a.id]
            entry_pose = self._coverage_entry_pose(plan, zone.entry_pose)
            transit = self._plan_transit(a.pose, entry_pose)
            a.adopt_plan(plan, transit)

    def _redistribute_targets(self, active, t: float) -> None:
        import time as _time
        from ..planning.target_mission import plan_target_mission
        t0 = _time.perf_counter()
        active_ids = {a.id for a in active}
        # gather still-unvisited targets across all drones (failed -> all theirs unvisited)
        unvisited = []
        for aid, tgts in self.assignment.items():
            a = self.fleet.agents.get(aid)
            if a is None:
                continue
            if aid not in active_ids:
                unvisited.extend(tgts)
            else:
                visited = min(len(tgts), 1 + a._cov_idx)  # first via transit + cov legs flown
                unvisited.extend(tgts[visited:])
        views = [a.view() for a in active]
        self.partition, self.plans, self.assignment = plan_target_mission(
            unvisited, views, self.launch_pose, self.motion, self.spec, self.em,
            weight_by_battery=self._weight_targets,
        )
        self.replan_times.append(_time.perf_counter() - t0)
        for a in active:
            plan = self.plans.get(a.id)
            if plan is not None and plan.waypoints:
                transit = self.motion.plan(a.pose, plan.waypoints[0].pose, ManeuverType.CRUISE)
                a.adopt_plan(plan, transit)

    def _skipped_legs(self) -> tuple[tuple[int, int], ...]:
        """EM-01 Stage 4: every forfeited coverage strip as sorted
        (agent_id, leg index) pairs. Always empty when safety.stall_skip is off
        (skip_stuck_leg is then never called), so the flag-off path costs one
        empty generator per terminal check and nothing else."""
        return tuple(sorted(
            (aid, k)
            for aid, a in self.fleet.agents.items()
            for k in getattr(a, "_skipped_cov", ())
        ))

    def _mission_complete(self) -> bool:
        active = self.fleet.active()
        if not active:
            return False
        for a in active:
            done = (
                a.state is AgentState.S0_IDLE
                and not a._launch_ready
                and a._cov_idx >= len(a._cov_legs)
            )
            if not done:
                return False
        return True

    def _telemetry_header(self) -> dict:
        """Run-setup object emitted once at the top of the LLM event log."""
        cfg = self.cfg
        try:
            area_m2 = round(float(self.env.area.area), 1)
        except Exception:
            area_m2 = None
        return {
            "config_hash": cfg.config_hash,
            "platform": self.spec.platform.value,
            "n_drones": cfg.fleet.n_drones,
            "area_m2": area_m2,
            "altitudes_m": list(cfg.layers.altitudes_m),
            "reserve_batteries": cfg.fleet.total_reserve_batteries,
            "launch_weights": {"dist": cfg.launch.w_distance,
                               "energy": cfg.launch.w_energy,
                               "swaps": cfg.launch.w_swaps},
            "dt_s": cfg.sim.dt_s,
            "mission_type": cfg.mission.type.value,
        }

    def _export_telemetry(self) -> None:
        """Write the GPX tracks + JSONL event log to the configured paths."""
        from ..metrics.gpx_exporter import write_gpx
        from ..metrics.llm_log_exporter import write_jsonl
        tc = self.cfg.telemetry
        write_gpx(self.telemetry, tc.gpx_path,
                  lat0=tc.origin_lat, lon0=tc.origin_lon, epoch_iso=tc.epoch_iso)
        write_jsonl(self.telemetry, tc.llm_log_path)

    def _evaluate_terminal(self, t: float) -> Outcome | None:
        """Mutually-exclusive terminal check (Phase 2, Tasks 2.1b + 2.2).

        Evaluated once per tick after event routing and position logging. Returns
        the Outcome to halt on, or None to keep running. Failure is checked BEFORE
        success: a drone whose battery dies on the very tick coverage finishes is a
        failure, not a success.

        Only BATTERY DEPLETION mid-flight is a failure here; the detector keys on
        ``battery.frac``, not on S_FAIL membership, so hazard-induced kills (which
        deliberately populate S_FAIL for the elevated-hazard Monte-Carlo / SMDP
        statistics) are naturally excluded and never halt the run.

        EXP-04: under ``mission.no_swap_mode`` the whole evaluation is replaced
        by ``_evaluate_terminal_no_swap`` (same-battery lifecycle semantics).
        """
        if self._no_swap:
            return self._evaluate_terminal_no_swap(t)

        cov = self._coverage_frac()
        coverage_complete = cov >= self._coverage_complete_frac()

        # ---- Condition 1: MISSION_FAILED (fail-fast) ----------------------- #
        # (a) any AIRBORNE drone whose battery has reached 0 -> forced S_FAIL.
        depleted = [a for a in self.fleet.airborne() if a.battery.frac <= 0.0]
        if depleted:
            for a in depleted:
                self.fleet.kill(a.id, t)        # freeze mid-flight in S_FAIL
            self._terminal_reason = "battery_depleted"
            if self.telemetry is not None:
                self.telemetry.record_terminal(
                    t, Outcome.MISSION_FAILED, "battery_depleted",
                    coverage_frac=cov, n_depleted=len(depleted))
            return Outcome.MISSION_FAILED
        # (b) shared swap reserve exhausted before coverage is complete.
        if self.swap_station.pool_exhausted and not coverage_complete:
            self._terminal_reason = "pool_exhausted"
            if self.telemetry is not None:
                self.telemetry.record_terminal(
                    t, Outcome.MISSION_FAILED, "pool_exhausted", coverage_frac=cov)
            return Outcome.MISSION_FAILED

        # ---- Condition 1.5: MISSION_PARTIAL (EM-01 Stage 4) ---------------- #
        # Every surviving drone finished every leg it did NOT forfeit and is
        # idle, but >= 1 coverage strip was skipped as unreachable
        # (safety.stall_skip). Checked BEFORE success so a forfeited strip can
        # never be classified MISSION_SUCCESS even if the zone-area-weighted
        # coverage fraction rounds above the completeness gate. Dead branch
        # whenever the flag is off (skipped is then always empty).
        skipped = self._skipped_legs()
        if skipped and self._mission_complete():
            self._terminal_reason = "coverage_complete_with_gaps"
            if self.telemetry is not None:
                self.telemetry.record_terminal(
                    t, Outcome.MISSION_PARTIAL, "coverage_complete_with_gaps",
                    coverage_frac=cov, n_skipped=len(skipped))
            return Outcome.MISSION_PARTIAL

        # ---- Condition 2: MISSION_SUCCESS ---------------------------------- #
        # 100% area coverage AND every surviving drone parked in S0_IDLE.
        # ``_mission_complete`` already encodes "every survivor finished its
        # assigned legs (=> full partitioned area) and is idle" and additionally
        # guards the t=0 / empty-plan edge; AND-ing the area gate keeps the
        # explicit Task 2.2 coverage condition and never relaxes the timing.
        if coverage_complete and self._mission_complete():
            self._terminal_reason = "coverage_complete"
            if self.telemetry is not None:
                self.telemetry.record_terminal(
                    t, Outcome.MISSION_SUCCESS, "coverage_complete", coverage_frac=cov)
            return Outcome.MISSION_SUCCESS

        return None

    # ------------------------------------------------------------------ #
    # EXP-04: same-battery (no-swap) lifecycle terminal evaluation         #
    # ------------------------------------------------------------------ #
    def _fleet_settled(self) -> bool:
        """True once no surviving drone can still act: every active agent is
        in the terminal S_LANDED state, or parked in S0_IDLE without a plan
        (never launched -- e.g. no zone). A drone in any other state, or an
        S0_IDLE drone still armed to launch, keeps the mission open. An empty
        active set (everyone lost) is settled."""
        for a in self.fleet.active():
            if a.state is AgentState.S_LANDED:
                continue
            if (a.state is AgentState.S0_IDLE and not a._launch_ready
                    and a.plan is None):
                continue
            return False
        return True

    def _evaluate_terminal_no_swap(self, t: float) -> Outcome | None:
        """Author decision C-4 (EXP-04). Evaluated once per tick after event
        routing, like the legacy check, but the outcome is decided ONLY when
        the fleet has settled:

          * FAILED  -- any drone was lost: battery reached 0 while airborne
                       (killed here, cause "battery_depleted") or a hazard
                       failure (S_FAIL via FAILURE). The run does NOT halt on
                       the loss: the survivors keep working and land, so their
                       coverage and safe touchdowns are measured; a depletion
                       is re-published as FAILURE so the existing redistribution
                       path treats it exactly like a hazard loss.
          * SUCCESS -- fleet settled, no loss, A_plannable raster coverage at or
                       above the completion gate. Reaching the gate while any
                       drone is still airborne is NOT success: the return leg and
                       touchdown must be flown (and paid for) first.
          * PARTIAL -- fleet settled, no loss, coverage below the gate.
          * None    -- otherwise (still flying). The sim.max_timesteps cap and a
                       stall halt therefore leave MISSION_INCOMPLETE with the
                       airborne drones reported, never a fictitious landing.

        ``pool_exhausted`` is never consulted: no SWAP_REQUEST is ever issued
        in this mode, so the reserve size cannot influence the outcome.
        """
        cov = self._coverage_frac()
        depleted = [a for a in self.fleet.airborne() if a.battery.frac <= 0.0]
        for a in depleted:
            self.fleet.kill(a.id, t)            # physics: it is down, mid-flight
            self._losses.append((a.id, t, "battery_depleted"))
            self.bus.publish(Event(EventType.FAILURE, t, {"agent_id": a.id}))
        if not self._fleet_settled():
            return None
        if self._losses or self.fleet.n_failed > 0:
            outcome, reason = Outcome.MISSION_FAILED, "uav_lost"
        elif cov >= self._coverage_complete_frac():
            outcome, reason = Outcome.MISSION_SUCCESS, "coverage_complete_all_landed"
        else:
            outcome, reason = Outcome.MISSION_PARTIAL, "all_landed_below_gate"
        self._terminal_reason = reason
        if self.telemetry is not None:
            self.telemetry.record_terminal(
                t, outcome, reason, coverage_frac=cov,
                n_retired=len(self._retirements), n_lost=len(self._losses))
        return outcome

    def _coverage_frac(self) -> float:
        if self._mission_type is MissionType.TARGET_VISIT:
            total = sum(len(v) for v in self.assignment.values())
            if total == 0:
                return 1.0
            visited = 0
            for aid, tgts in self.assignment.items():
                a = self.fleet.agents.get(aid)
                if a is None:
                    continue
                if a._cov_idx >= len(a._cov_legs):
                    visited += len(tgts)              # completed tour -> all visited
                else:
                    visited += min(len(tgts), 1 + a._cov_idx)
            return min(1.0, visited / total)

        if self.coverage_raster is not None:
            return self.coverage_raster.plannable_coverage_frac
            
        # AREA COVERAGE
        total = self.partition.total_area_m2
        if total <= 0:
            return 1.0
        covered = 0.0
        for aid, zone in self.partition.zones.items():
            a = self.fleet.agents.get(aid)
            if a is not None:
                if len(a._cov_legs) > 0:
                    # FIX: Give partial coverage credit based on legs completed.
                    # EM-01 Stage 4: legs forfeited by skip-on-stall are jumped
                    # by _cov_idx but must NOT be credited -- subtract the
                    # deficit (0 whenever safety.stall_skip is off, so the
                    # arithmetic is byte-identical there).
                    done = a._cov_idx - getattr(a, "_cov_frac_deficit", 0)
                    fraction_done = min(1.0, done / len(a._cov_legs))
                    covered += fraction_done * zone.area_m2
                elif a._cov_idx >= len(a._cov_legs):
                    # Fallback for empty leg plans
                    covered += zone.area_m2
                    
        return min(1.0, covered / total)

    def _target_coverage_frac(self) -> float | None:
        if self.coverage_raster is not None:
            return self.coverage_raster.target_coverage_frac
        return None

    def _coverage_complete_frac(self) -> float:
        if self.coverage_raster is None:
            return _COVERAGE_COMPLETE_FRAC
        return 1.0 - self.cfg.coverage.raster_completion_tolerance_frac
