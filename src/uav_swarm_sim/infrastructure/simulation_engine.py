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

from shapely.geometry import Polygon

from ..infrastructure.config import Config
from ..infrastructure.core_types import (
    DroneStateView,
    Event,
    MissionResult,
    Pose,
)
from ..infrastructure.enums import (
    AgentState,
    DecompositionAlgo,
    EventType,
    MissionType,
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
from ..planning.environment_map import LayerStack
from ..planning.geojson_parser import load_area
from ..planning.grid_planner import GridPlanner
from ..planning.kmeans_heuristic import KMeansHeuristicDecomposer
from ..planning.layer_planner import assign_to_layers, build_layer_graphs, decompose_layers
from ..planning.launch_site_optimizer import optimize as optimize_launch
from ..planning.obstacle_generator import generate as generate_obstacles
from ..planning.target_mission import generate_targets, plan_target_mission
from ..planning.dynamic_obstacles import DynamicObstacleField
from ..execution.sensing import SensingCoordinator
from ..planning.weighted_decomposition import (
    TgcBasicDecomposer,
    WeightedTgcDecomposer,
)
from ..metrics import mission_metrics
from ..metrics.state_history import StateHistory
from ..execution.agent import Agent
from ..execution.events import EventBus
from ..execution.failure_model import FailureModel
from ..execution.fleet import Fleet
from ..execution.formation_manager import FormationManager
from ..execution.redistribution import Redistributor
from ..execution.rth_calculator import RthCalculator
from ..execution.safety_monitor import SafetyMonitor
from ..execution.state_machine import StateMachine
from ..execution.swap_station import SwapStation

_LOG = logging.getLogger(__name__)


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
        # no explicit algo: pick by scale tier
        from ..execution.algorithm_selector import select

        strat = select(self.cfg.fleet.n_drones, self.cfg.tier_thresholds)
        if strat is TierStrategy.HEURISTIC:
            return KMeansHeuristicDecomposer(motion, weighted=True, rng=kmeans_rng)
        return WeightedTgcDecomposer()

    def _build(self):
        cfg = self.cfg
        self.spec = build_spec(cfg)
        self.motion = make_motion_model(self.spec)
        self.em = EnergyModel(self.spec)
        self.aero = AeroCorrection(cfg.aero, self.spec.platform)

        area = load_area(cfg.env.geojson_path)
        obs_rng = self.rng.stream(STREAM_OBSTACLES, self.replication)
        obstacles = generate_obstacles(area, cfg.env, obs_rng)
        # 2.5D: slice the extruded prisms into one 2D map per coverage layer.
        # A single layer at the coverage altitude (with unbounded prisms) is the
        # whole world and reproduces the 2D map exactly. Build per-layer GVG+TGC
        # once; layer 0 stays the primary 2D graph for launch siting, RTH and
        # redistribution (those become per-layer in Batch 4).
        self.layers = LayerStack(
            area, obstacles, cfg.layers.altitudes_m, cfg.env.clearance_buffer_m
        )
        self.layer_graphs = build_layer_graphs(
            self.layers, gvg_sample_step_m=20.0, gvg_spur_min_m=30.0
        )
        self.env, self.tgc = self.layer_graphs.by_layer[0]
        self.planning_time_s = self.layer_graphs.planning_time_s

        # launch site optimization (on layer 0 / primary graph)
        launch_rng = self.rng.stream(STREAM_LAUNCH_SAMPLING, self.replication)
        self.launch_pose, self.site_scores = optimize_launch(
            cfg.launch, self.tgc, self.env, self.motion, self.em, self.aero,
            self.spec, cfg.fleet.n_drones, launch_rng, cfg.env.coverage_altitude_m,
        )

        # --- mission planning: area coverage OR target visit -------------- #
        self._mission_type = cfg.mission.type
        self._weight_targets = cfg.mission.weight_targets_by_battery
        init_views = [
            DroneStateView(i, 1.0, self.launch_pose) for i in range(cfg.fleet.n_drones)
        ]
        self.assignment = {}          # drone_id -> list[(x, y)] (target mode only)
        self.layer_of: dict[int, int] = {}   # drone_id -> assigned layer index
        self.decomposer = None
        if self._mission_type is MissionType.TARGET_VISIT:
            tgt_rng = self.rng.stream(STREAM_TARGETS, self.replication)
            self.targets = generate_targets(self.env, cfg.mission, tgt_rng)
            self.partition, self.plans, self.assignment = plan_target_mission(
                self.targets, init_views, self.launch_pose, self.motion,
                self.spec, self.em, weight_by_battery=self._weight_targets,
            )
        else:
            self.decomposer = self._make_decomposer(self.motion)
            # Level 1: assign drones to layers (single-layer => all on layer 0).
            # Level 2: the reused decomposer runs per layer over its sliced map.
            layer_assignment = assign_to_layers(
                init_views, self.layers, cfg.layers.assignment_policy
            )
            self.layer_of = {d.id: idx for idx, ds in layer_assignment.items() for d in ds}
            self.partition = decompose_layers(
                self.layer_graphs, layer_assignment, self.decomposer
            )
            self.plans = {}

        # support objects
        sm = StateMachine(cfg.battery_zones)
        self.bus = EventBus()
        self.history = StateHistory()
        self.swap_station = SwapStation(cfg.swap, self.launch_pose)
        self.safety = SafetyMonitor(self.env, self.aero, cfg.safety, self.motion)
        # dynamic obstacles + swarm sensing (feature is OFF unless enabled in config)
        self.sensing = SensingCoordinator(cfg.dynamic_obstacles, cfg.safety)
        if cfg.dynamic_obstacles.enabled and cfg.dynamic_obstacles.count > 0:
            dyn_rng = self.rng.stream(STREAM_DYNOBS, self.replication)
            self._dynfield = DynamicObstacleField(
                self.env, cfg.dynamic_obstacles.count, cfg.dynamic_obstacles.speed_m_s,
                cfg.dynamic_obstacles.size_m, dyn_rng,
            )
        else:
            self._dynfield = None
        self.formation = FormationManager(self.aero, cfg.aero, self.spec.platform)
        self.failure = FailureModel(cfg.failure, self.rng.stream(STREAM_FAILURES, self.replication))
        rth = RthCalculator(
            self.em, self.motion, self.spec, cfg.rth, self.launch_pose,
            cfg.env.coverage_altitude_m, self.env,
        )
        self.rth = rth

        grid = GridPlanner(self.env, cell_m=50.0) if self.planner is PlannerKind.GRID else None

        # agents + plans
        agents: list[Agent] = []
        for i in range(cfg.fleet.n_drones):
            battery = Battery(self.spec.battery_capacity_j, cfg.battery_zones, 1.0)
            agent = Agent(i, self.spec, self.motion, self.em, battery, sm, rth,
                          self.formation, self.launch_pose, recorder=self.history)
            if self._mission_type is MissionType.TARGET_VISIT:
                plan = self.plans.get(i)
                if plan is not None and plan.waypoints:
                    transit = self.motion.plan(self.launch_pose, plan.waypoints[0].pose,
                                               ManeuverType.CRUISE)
                    agent.assign(plan, transit)
            else:
                zone = self.partition.zones.get(i)
                if zone is not None:
                    plan = (grid.coverage(zone, self.spec) if grid is not None
                            else boustrophedon(zone, self.spec, self.motion, self.em))
                    transit = self.motion.plan(self.launch_pose, zone.entry_pose, ManeuverType.CRUISE)
                    agent.assign(plan, transit)
                    self.plans[i] = plan
            agents.append(agent)

        self.fleet = Fleet(agents)
        self.formation.register_departure(agents)
        self.redistributor = (
            None if self._mission_type is MissionType.TARGET_VISIT else Redistributor(
                self.decomposer if isinstance(self.decomposer, (WeightedTgcDecomposer,))
                else WeightedTgcDecomposer(),
                self.tgc, self.env, self.motion, self.em, self.spec,
            )
        )
        self.replan_times: list[float] = []

        # open initial S0 sojourns
        for a in agents:
            self.history.open(a.id, AgentState.S0_IDLE, 0.0)

    # ------------------------------------------------------------------ #
    def run(self) -> MissionResult:
        self._build()
        cfg = self.cfg
        dt = cfg.sim.dt_s
        complete = False
        t = 0.0
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
            if self._mission_complete():
                complete = True
                break

        t_end = t
        self.history.finalize(t_end)
        coverage_frac = self._coverage_frac()
        metrics = mission_metrics.compute(
            self.history, self.fleet, self.partition, t_end,
            planning_time_s=self.planning_time_s,
            replan_times_s=tuple(self.replan_times),
            coverage_frac=coverage_frac,
        )
        aborted = (not complete) or (len(self.fleet.active()) == 0 and coverage_frac < 0.999)
        return MissionResult(metrics, self.history, self.partition, aborted, coverage_frac,
                             cfg.config_hash)

    # ------------------------------------------------------------------ #
    def _route_events(self, t: float) -> None:
        for e in self.bus.drain():
            if e.type is EventType.FAILURE:
                aid = e.payload.get("agent_id")
                self.fleet.kill(aid, t)
                self._redistribute(e, t)
            elif e.type is EventType.NEW_TASK:
                self._redistribute(e, t)
            elif e.type is EventType.SWAP_REQUEST:
                self.swap_station.request(e.payload.get("agent_id"), t)
            elif e.type is EventType.SWAP_DONE:
                a = self.fleet.agents.get(e.payload.get("agent_id"))
                if a is not None:
                    a.signal_swap_done()
            # OBSTACLE_THREAT is informational (signal already set by the monitor)

    def _redistribute(self, e: Event, t: float) -> None:
        active = self.fleet.active()
        if not active:
            return
        if self._mission_type is MissionType.TARGET_VISIT:
            self._redistribute_targets(active, t)
            return
        new_part, new_plans = self.redistributor.handle(e, self.fleet, self.partition, self.plans, t)
        self.replan_times.append(self.redistributor.last_replan_time_s)
        self.partition = new_part
        self.plans = new_plans
        for a in active:
            zone = new_part.zones.get(a.id)
            if zone is None:
                continue
            transit = self.motion.plan(a.pose, zone.entry_pose, ManeuverType.CRUISE)
            a.adopt_plan(new_plans[a.id], transit)

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
        total = self.partition.total_area_m2
        if total <= 0:
            return 1.0
        covered = 0.0
        for aid, zone in self.partition.zones.items():
            a = self.fleet.agents.get(aid)
            if a is not None and a._cov_idx >= len(a._cov_legs):
                covered += zone.area_m2
        return min(1.0, covered / total)
