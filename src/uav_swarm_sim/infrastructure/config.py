"""Typed configuration schema + YAML loader + validation.

This is the mechanism by which every global input propagates through the
system. No other module reads YAML; everything receives a frozen ``Config``.

Loader responsibilities (blueprint infrastructure/config.py):
  1. parse YAML into a nested dict;
  2. apply dot-key overrides (e.g. {"fleet.n_drones": 30});
  3. resolve the active platform table into a single PlatformConfig;
  4. unit conversion: Wh -> J, degrees -> radians;
  5. validation with field-path error messages;
  6. compute a sha256 provenance hash of the canonical merged config;
  7. return a frozen Config.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from math import isfinite, radians
from pathlib import Path
from typing import Any

import yaml

from .enums import ManeuverType, MissionType, PlatformType

WH_TO_J = 3600.0

# Recognized drone->layer assignment policies (see planning/layer_planner.py).
# "single" = all drones on layer 0 (the single-layer-z0 regression net).
_LAYER_POLICIES = ("single", "area_balanced", "battery_tiered")


class ConfigError(ValueError):
    """Raised on any invalid configuration, with the offending field path."""


# --------------------------------------------------------------------------- #
# Schema (frozen dataclasses)                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FleetConfig:
    n_drones: int
    battery_capacity_wh: float
    battery_capacity_j: float            # derived: wh * 3600
    drone_dims_m: tuple[float, float, float]
    # Phase 2 (Task 2.1): finite shared reserve of swap packs for the WHOLE fleet.
    # The swap station decrements this once per admitted swap; exhaustion before
    # 100% coverage drives MISSION_FAILED. None => unbounded reserve, i.e. the
    # pre-Phase-2 infinite-reserve behaviour (kept so fixtures that omit the key,
    # and any direct FleetConfig(...) construction, are unaffected).
    total_reserve_batteries: int | None = None


@dataclass(frozen=True)
class PlatformConfig:
    type: PlatformType
    v_cruise: float
    v_coverage: float
    v_climb: float
    v_descent: float
    r_min_m: float
    omega_max: float
    climb_angle_rad: float               # derived from climb_angle_deg
    ground_roll_energy_j: float
    power_w: dict[ManeuverType, float]
    mass_kg: float = 2.0                  # 2.5D: vertical-energy term ONLY; never horizontal


@dataclass(frozen=True)
class SensorConfig:
    swath_width_m: float
    overlap_frac: float
    sensor_power_w: float = 0.0  # camera/gimbal payload draw while filming (W); 0 => no camera-energy term


@dataclass(frozen=True)
class CoverageConfig:
    """S_FERRY Step 2: plan-time, obstacle-aware routing of the camera-off
    inter-strip connectors (the odd TURN legs) over the FLYABLE region --
    operating-area-minus-obstacles, NOT the survey polygon -- so the connector
    detours *around* an obstacle that blocks its straight chord instead of flying
    blind (today's connectors ignore obstacles; runtime S_OBS then catches the
    real prism, but the analytical E_cover does not, so analytical != execution
    when a chord is blocked). The routing is a single source shared by
    ``coverage_path.boustrophedon`` (which stores the routed connectors on the
    ``CoveragePlan``), ``Agent._build_coverage_legs`` and
    ``run_regime_calculator``, so analytical and executed connector cost stay in
    lock-step.

    ``ferry_free_space`` defaults **OFF** so every existing run is byte-identical:
    with it off (or when a chord is unobstructed) the connector is exactly today's
    straight ``motion.plan(a, b, TURN)`` chord. Studies opt in.

    ``operating_area`` bounds where a detour may route: ``convex_hull`` (of the
    survey outer ring) dilated by ``operating_margin_m`` is the default and is the
    free-flight premise made finite (the drone may leave the survey plot whenever
    the camera is off and no obstacle is there). ``operating_margin_m`` (default
    50 m == one coverage swath) gives a detour room to pass *outside* an obstacle
    sitting on the hull edge; it is deliberately >> ``env.clearance_buffer_m``
    (5 m) and tied to an existing length scale rather than a magic number.

    ``transit_free_space`` (FIX-B1) applies the same plan-time routing to the S1
    TRANSIT chords -- the initial assign transit, ``Agent._resume_transit`` after
    a swap, and the redistribution re-transit -- via
    ``visibility_router.route_transit`` (the CRUISE twin of ``route_connector``).
    A blocked straight transit chord is the swap-livelock root cause: S_OBS
    cannot make lateral progress on a transit leg, the boxed-in escalation sends
    the drone home, the unconditional swap burns a pack, and the relaunch
    replays the identical chord. Defaults **OFF** so every existing run is
    byte-identical (an unobstructed chord stays the straight chord even when
    on); it shares ``operating_area``/``operating_margin_m`` with the connector
    routing. The S3_RTH return leg is deliberately NOT routed (out of scope).
    """
    ferry_free_space: bool = False
    operating_area: str = "convex_hull"   # "convex_hull" | "bbox" | "survey"
    operating_margin_m: float = 50.0
    transit_free_space: bool = False      # FIX-B1: obstacle-aware S1 transit routing


@dataclass(frozen=True)
class AeroConfig:
    formation_drag_reduction: float
    downwash_radius_m: float
    downwash_length_m: float
    formation_spacing_m: float
    rth_rendezvous_window_s: float


@dataclass(frozen=True)
class EnvConfig:
    geojson_path: str
    coverage_altitude_m: float
    obstacle_density_per_km2: float
    obstacle_size_range_m: tuple[float, float]
    obstacle_shapes: tuple[str, ...]
    n_obstacle_classes: int
    clearance_buffer_m: float
    # 2.5D prism extrusion: footprint over [floor, ceil]. ceil=None => UNBOUNDED,
    # i.e. every layer sees every obstacle -> single-layer case is 2D-identical.
    obstacle_floor_m: float = 0.0
    obstacle_ceil_range_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class LaunchConfig:
    candidate_sites: int | tuple[tuple[float, float], ...]
    w_distance: float
    w_energy: float
    w_swaps: float


@dataclass(frozen=True)
class BatteryZonesConfig:
    high: float = 0.75
    nominal: float = 0.40
    critical: float = 0.20


@dataclass(frozen=True)
class SwapConfig:
    service_time_s: float
    n_bays: int


@dataclass(frozen=True)
class FailureConfig:
    hazard_rate_per_hour: float


@dataclass(frozen=True)
class SafetyConfig:
    min_separation_m: float
    obstacle_buffer_m: float
    predict_horizon_s: float
    # Task 2.5 Q2: when true, the SafetyMonitor sends recovery-mode threat
    # signals (validated detour + obstructed-leg skip + boxed-in escalation),
    # activating the stateful S_OBS recovery in Agent that prevents the
    # obstacle-avoidance limit cycle. Defaults false so any config that does
    # not set it is byte-identical to the pre-Q2 behaviour and its provenance
    # hash is unchanged (the hash is taken over the raw YAML, which is absent
    # this key unless explicitly added).
    obstacle_recovery: bool = False
    # FIX-B4: engine-level swap-livelock net. When true, an agent that requests
    # 5 consecutive swaps without any coverage-leg progress is flagged stalled
    # and the mission halts early (outcome stays MISSION_INCOMPLETE) with the
    # agents reported in MissionResult.stalled_agents -- a fast, honestly
    # labelled diagnostic instead of a max_timesteps burn. Defaults false =>
    # byte-identical (same optional-key hash rule as obstacle_recovery).
    stall_detector: bool = False


@dataclass(frozen=True)
class EnergyMapConfig:
    """EM-01 Stage 1: per-replication energy cost-to-go map (planning/energy_map.py).

    OFF by default and, like telemetry / obstacle_recovery / stall_detector,
    deliberately ABSENT from default.yaml: ``config_hash`` is taken over the raw
    YAML, so omitting the ``rth.energy_map`` block leaves the hash and every
    fixture unchanged, and a flag-off run is byte-identical by construction
    (Stage 1 builds and attaches the map only -- nothing consumes it yet).

    ``cell_m`` None => battery-tied resolution (design doc section 3): the cell
    edge is the distance costing exactly 1/1000 of the battery capacity at
    reference CRUISE (~19.64 m for the default 360 kJ / 220 W / 12 m/s).
    """
    enabled: bool = False
    cell_m: float | None = None      # None => battery-tied derived
    yellow_penalty: float = 1.5      # traversal weight of a partially occupied cell
    red_threshold: float = 0.5       # occupancy fraction at/above which a cell blocks


@dataclass(frozen=True)
class RTHConfig:
    check_interval_s: float
    reserve_frac: float
    energy_map: EnergyMapConfig = field(default_factory=EnergyMapConfig)


@dataclass(frozen=True)
class SimConfig:
    dt_s: float
    max_timesteps: int
    master_seed: int


@dataclass(frozen=True)
class MCConfig:
    n_max: int = 1000
    n_min: int = 30
    ci_tolerance: float = 0.01


@dataclass(frozen=True)
class DynamicObstacleConfig:
    enabled: bool
    count: int
    speed_m_s: float
    size_m: float                  # drone-sized; obstacle modelled as a circle of this diameter
    passive_sense_range_m: float   # short detection range when the swarm is passive
    active_sense_range_m: float    # long LIDAR detection range when the swarm is active
    active_scan_power_w: float     # extra power per airborne drone while scanning (active mode)
    dynamic_hold_s: float          # stay active this long after the last detection, then revert


@dataclass(frozen=True)
class VizConfig:
    show_comm_range: bool       # master OFF switch for the comm-range overlay
    comm_range_m: float         # radius drawn around each drone (metres; view-only)
    comm_range_alpha: float     # opacity of the circle (outline and/or fill)
    comm_range_dashed: bool     # True -> dashed unfilled outline; False -> translucent fill


@dataclass(frozen=True)
class MissionConfig:
    type: MissionType
    n_targets: int
    target_coordinates: tuple[tuple[float, float], ...]
    weight_targets_by_battery: bool


@dataclass(frozen=True)
class LayersConfig:
    """2.5D coverage stack: ordered horizontal layer altitudes + the drone->layer
    assignment policy. Defaults (a single layer at the coverage altitude, see the
    loader) reproduce the 2D model byte-for-byte."""
    altitudes_m: tuple[float, ...]       # strictly ascending; len() == n_layers
    assignment_policy: str = "single"    # see planning/layer_planner.py

    @property
    def n_layers(self) -> int:
        return len(self.altitudes_m)


def _single_layer_default() -> LayersConfig:
    # Used only for direct Config(...) construction; the loader synthesizes a
    # single layer at env.coverage_altitude_m instead. Keeps any fixture that
    # builds Config directly valid with a single-layer-z0 stack.
    return LayersConfig(altitudes_m=(0.0,), assignment_policy="single")


@dataclass(frozen=True)
class TelemetryConfig:
    """Phase 3 observability outputs (GPX tracks + LLM-ready JSONL event log).

    OFF by default and side-effect-free when disabled, so a run with telemetry
    off is byte-identical to the pre-Phase-3 baseline. Deliberately NOT present
    in default.yaml: ``config_hash`` is computed from the raw YAML, so omitting
    the section leaves the hash (and every fixture that asserts it) unchanged.
    Enable it via an override or by adding a ``telemetry:`` block when wanted.
    """
    enabled: bool = False
    gpx_path: str = "telemetry_tracks.gpx"
    llm_log_path: str = "telemetry_events.jsonl"
    fix_interval_s: float = 30.0     # periodic GPX position-fix cadence (s)
    origin_lat: float = 54.6872      # local-tangent-plane false origin (Vilnius)
    origin_lon: float = 25.2797
    epoch_iso: str = "2026-01-01T00:00:00Z"   # GPX <time> = epoch + sim seconds


@dataclass(frozen=True)
class Config:
    fleet: FleetConfig
    platform: PlatformConfig
    sensor: SensorConfig
    coverage: CoverageConfig
    aero: AeroConfig
    env: EnvConfig
    launch: LaunchConfig
    battery_zones: BatteryZonesConfig
    swap: SwapConfig
    failure: FailureConfig
    safety: SafetyConfig
    rth: RTHConfig
    sim: SimConfig
    mc: MCConfig
    mission: MissionConfig
    dynamic_obstacles: DynamicObstacleConfig
    viz: VizConfig
    tier_thresholds: tuple[int, int]
    layers: LayersConfig = field(default_factory=_single_layer_default)
    config_hash: str = field(default="")
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _require(raw: dict, key: str, path: str) -> Any:
    if key not in raw:
        raise ConfigError(f"missing required field: {path}.{key}" if path else key)
    return raw[key]


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set d['a']['b']['c'] = value from dotted_key 'a.b.c'."""
    parts = dotted_key.split(".")
    node = d
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def _canonical_hash(raw: dict) -> str:
    blob = json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #
def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError("top-level YAML must be a mapping")

    raw = copy.deepcopy(raw)
    for dotted, value in (overrides or {}).items():
        _deep_set(raw, dotted, value)

    # hash the merged raw config BEFORE enum/unit transformation (provenance)
    config_hash = _canonical_hash(raw)

    cfg = _build(raw, config_hash)
    _validate(cfg, raw)
    return cfg


def _build(raw: dict, config_hash: str) -> Config:
    # ---- fleet ----
    f = _require(raw, "fleet", "")
    cap_wh = float(_require(f, "battery_capacity_wh", "fleet"))
    dims = tuple(float(x) for x in _require(f, "drone_dims_m", "fleet"))
    if len(dims) != 3:
        raise ConfigError("fleet.drone_dims_m must have exactly 3 values")
    # Optional finite shared swap reserve; absent => None (unbounded).
    trb_raw = f.get("total_reserve_batteries", None)
    total_reserve_batteries = None if trb_raw is None else int(trb_raw)
    fleet = FleetConfig(
        n_drones=int(_require(f, "n_drones", "fleet")),
        battery_capacity_wh=cap_wh,
        battery_capacity_j=cap_wh * WH_TO_J,
        drone_dims_m=dims,  # type: ignore[arg-type]
        total_reserve_batteries=total_reserve_batteries,
    )

    # ---- platform (resolve active table) ----
    ptype_str = _require(raw, "platform_type", "")
    try:
        ptype = PlatformType(ptype_str)
    except ValueError as exc:
        raise ConfigError(
            f"platform_type '{ptype_str}' not one of {[p.value for p in PlatformType]}"
        ) from exc
    tables = _require(raw, "platforms", "")
    if ptype.value not in tables:
        raise ConfigError(f"platforms.{ptype.value} table missing for active platform_type")
    pt = tables[ptype.value]
    raw_power = _require(pt, "power_w", f"platforms.{ptype.value}")
    power: dict[ManeuverType, float] = {}
    for k, v in raw_power.items():
        try:
            power[ManeuverType(k)] = float(v)
        except ValueError as exc:
            raise ConfigError(
                f"platforms.{ptype.value}.power_w: '{k}' is not a ManeuverType"
            ) from exc
    platform = PlatformConfig(
        type=ptype,
        v_cruise=float(_require(pt, "v_cruise", f"platforms.{ptype.value}")),
        v_coverage=float(_require(pt, "v_coverage", f"platforms.{ptype.value}")),
        v_climb=float(_require(pt, "v_climb", f"platforms.{ptype.value}")),
        v_descent=float(_require(pt, "v_descent", f"platforms.{ptype.value}")),
        r_min_m=float(_require(pt, "r_min_m", f"platforms.{ptype.value}")),
        omega_max=float(_require(pt, "omega_max", f"platforms.{ptype.value}")),
        climb_angle_rad=radians(float(pt.get("climb_angle_deg", 90.0))),
        ground_roll_energy_j=float(pt.get("ground_roll_energy_j", 0.0)),
        power_w=power,
        mass_kg=float(pt.get("mass_kg", 2.0)),
    )

    # ---- sensor / aero ----
    s = _require(raw, "sensor", "")
    sensor = SensorConfig(
        swath_width_m=float(_require(s, "swath_width_m", "sensor")),
        overlap_frac=float(_require(s, "overlap_frac", "sensor")),
        sensor_power_w=float(s.get("sensor_power_w", 0.0)),
    )

    # ---- coverage (S_FERRY Step 2 connector routing) ----
    cov = raw.get("coverage", {})
    coverage = CoverageConfig(
        ferry_free_space=bool(cov.get("ferry_free_space", False)),
        operating_area=str(cov.get("operating_area", "convex_hull")),
        operating_margin_m=float(cov.get("operating_margin_m", 50.0)),
        transit_free_space=bool(cov.get("transit_free_space", False)),
    )
    a = _require(raw, "aero", "")
    aero = AeroConfig(
        formation_drag_reduction=float(_require(a, "formation_drag_reduction", "aero")),
        downwash_radius_m=float(_require(a, "downwash_radius_m", "aero")),
        downwash_length_m=float(_require(a, "downwash_length_m", "aero")),
        formation_spacing_m=float(_require(a, "formation_spacing_m", "aero")),
        rth_rendezvous_window_s=float(_require(a, "rth_rendezvous_window_s", "aero")),
    )

    # ---- env ----
    e = _require(raw, "env", "")
    size_range = tuple(float(x) for x in _require(e, "obstacle_size_range_m", "env"))
    if len(size_range) != 2:
        raise ConfigError("env.obstacle_size_range_m must have exactly 2 values")
    ceil_raw = e.get("obstacle_ceil_range_m", None)
    obstacle_ceil_range = None if ceil_raw is None else tuple(float(x) for x in ceil_raw)
    env = EnvConfig(
        geojson_path=str(_require(e, "geojson_path", "env")),
        coverage_altitude_m=float(_require(e, "coverage_altitude_m", "env")),
        obstacle_density_per_km2=float(_require(e, "obstacle_density_per_km2", "env")),
        obstacle_size_range_m=size_range,  # type: ignore[arg-type]
        obstacle_shapes=tuple(str(x) for x in _require(e, "obstacle_shapes", "env")),
        n_obstacle_classes=int(_require(e, "n_obstacle_classes", "env")),
        clearance_buffer_m=float(_require(e, "clearance_buffer_m", "env")),
        obstacle_floor_m=float(e.get("obstacle_floor_m", 0.0)),
        obstacle_ceil_range_m=obstacle_ceil_range,  # type: ignore[arg-type]
    )

    # ---- layers (2.5D coverage stack) ----
    # Default = a single layer at the coverage altitude. With unbounded prisms
    # (above), every obstacle is present in that layer -> byte-identical 2D.
    lz = raw.get("layers", {}) or {}
    alt_raw = lz.get("altitudes_m", None)
    if alt_raw is None:
        layer_altitudes: tuple[float, ...] = (env.coverage_altitude_m,)
    else:
        layer_altitudes = tuple(float(z) for z in alt_raw)
    layers = LayersConfig(
        altitudes_m=layer_altitudes,
        assignment_policy=str(lz.get("assignment_policy", "single")),
    )

    # ---- launch ----
    lc = _require(raw, "launch", "")
    cand = _require(lc, "candidate_sites", "launch")
    if isinstance(cand, int):
        candidate_sites: int | tuple[tuple[float, float], ...] = cand
    else:
        candidate_sites = tuple((float(p[0]), float(p[1])) for p in cand)
    launch = LaunchConfig(
        candidate_sites=candidate_sites,
        w_distance=float(_require(lc, "w_distance", "launch")),
        w_energy=float(_require(lc, "w_energy", "launch")),
        w_swaps=float(_require(lc, "w_swaps", "launch")),
    )

    # ---- battery zones ----
    bz = raw.get("battery_zones", {})
    battery_zones = BatteryZonesConfig(
        high=float(bz.get("high", 0.75)),
        nominal=float(bz.get("nominal", 0.40)),
        critical=float(bz.get("critical", 0.20)),
    )

    # ---- swap / failure / safety / rth / sim / mc ----
    sw = _require(raw, "swap", "")
    swap = SwapConfig(
        service_time_s=float(_require(sw, "service_time_s", "swap")),
        n_bays=int(_require(sw, "n_bays", "swap")),
    )
    fa = _require(raw, "failure", "")
    failure = FailureConfig(
        hazard_rate_per_hour=float(_require(fa, "hazard_rate_per_hour", "failure")),
    )
    sf = _require(raw, "safety", "")
    safety = SafetyConfig(
        min_separation_m=float(_require(sf, "min_separation_m", "safety")),
        obstacle_buffer_m=float(_require(sf, "obstacle_buffer_m", "safety")),
        predict_horizon_s=float(_require(sf, "predict_horizon_s", "safety")),
        obstacle_recovery=bool(sf.get("obstacle_recovery", False)),
        stall_detector=bool(sf.get("stall_detector", False)),
    )
    rt = _require(raw, "rth", "")
    emr = rt.get("energy_map", {}) or {}
    if not isinstance(emr, dict):
        raise ConfigError("rth.energy_map must be a mapping")
    cell_raw = emr.get("cell_m", None)
    energy_map = EnergyMapConfig(
        enabled=bool(emr.get("enabled", False)),
        cell_m=None if cell_raw is None else float(cell_raw),
        yellow_penalty=float(emr.get("yellow_penalty", 1.5)),
        red_threshold=float(emr.get("red_threshold", 0.5)),
    )
    rth = RTHConfig(
        check_interval_s=float(_require(rt, "check_interval_s", "rth")),
        reserve_frac=float(_require(rt, "reserve_frac", "rth")),
        energy_map=energy_map,
    )
    si = _require(raw, "sim", "")
    sim = SimConfig(
        dt_s=float(_require(si, "dt_s", "sim")),
        max_timesteps=int(_require(si, "max_timesteps", "sim")),
        master_seed=int(_require(si, "master_seed", "sim")),
    )
    mc_raw = raw.get("mc", {})
    mc = MCConfig(
        n_max=int(mc_raw.get("n_max", 1000)),
        n_min=int(mc_raw.get("n_min", 30)),
        ci_tolerance=float(mc_raw.get("ci_tolerance", 0.01)),
    )

    m = raw.get("mission", {})
    mission = MissionConfig(
        type=MissionType(str(m.get("type", "coverage"))),
        n_targets=int(m.get("n_targets", 30)),
        target_coordinates=tuple(
            (float(xy[0]), float(xy[1])) for xy in (m.get("target_coordinates") or [])
        ),
        weight_targets_by_battery=bool(m.get("weight_targets_by_battery", True)),
    )

    do = raw.get("dynamic_obstacles", {})
    dynamic_obstacles = DynamicObstacleConfig(
        enabled=bool(do.get("enabled", False)),
        count=int(do.get("count", 0)),
        speed_m_s=float(do.get("speed_m_s", 8.0)),
        size_m=float(do.get("size_m", 1.5)),
        passive_sense_range_m=float(do.get("passive_sense_range_m", 30.0)),
        active_sense_range_m=float(do.get("active_sense_range_m", 150.0)),
        active_scan_power_w=float(do.get("active_scan_power_w", 60.0)),
        dynamic_hold_s=float(do.get("dynamic_hold_s", 20.0)),
    )

    vz = raw.get("viz", {})
    viz = VizConfig(
        show_comm_range=bool(vz.get("show_comm_range", False)),
        comm_range_m=float(vz.get("comm_range_m", 250.0)),
        comm_range_alpha=float(vz.get("comm_range_alpha", 0.12)),
        comm_range_dashed=bool(vz.get("comm_range_dashed", True)),
    )

    tt = tuple(int(x) for x in raw.get("tier_thresholds", [15, 50]))
    if len(tt) != 2:
        raise ConfigError("tier_thresholds must have exactly 2 values")

    tl = raw.get("telemetry", {}) or {}
    telemetry = TelemetryConfig(
        enabled=bool(tl.get("enabled", False)),
        gpx_path=str(tl.get("gpx_path", "telemetry_tracks.gpx")),
        llm_log_path=str(tl.get("llm_log_path", "telemetry_events.jsonl")),
        fix_interval_s=float(tl.get("fix_interval_s", 30.0)),
        origin_lat=float(tl.get("origin_lat", 54.6872)),
        origin_lon=float(tl.get("origin_lon", 25.2797)),
        epoch_iso=str(tl.get("epoch_iso", "2026-01-01T00:00:00Z")),
    )

    return Config(
        fleet=fleet, platform=platform, sensor=sensor, coverage=coverage, aero=aero, env=env,
        launch=launch, battery_zones=battery_zones, swap=swap, failure=failure,
        safety=safety, rth=rth, sim=sim, mc=mc, mission=mission, dynamic_obstacles=dynamic_obstacles, viz=viz, tier_thresholds=tt,  # type: ignore[arg-type]
        layers=layers,
        config_hash=config_hash,
        telemetry=telemetry,
    )


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def _validate(cfg: Config, raw: dict) -> None:
    if not (1 <= cfg.fleet.n_drones <= 100):
        raise ConfigError(f"fleet.n_drones must be in [1, 100], got {cfg.fleet.n_drones}")
    if cfg.fleet.battery_capacity_wh <= 0:
        raise ConfigError("fleet.battery_capacity_wh must be > 0")
    if (
        cfg.fleet.total_reserve_batteries is not None
        and cfg.fleet.total_reserve_batteries < 0
    ):
        raise ConfigError(
            "fleet.total_reserve_batteries must be >= 0 (or omitted for unbounded), "
            f"got {cfg.fleet.total_reserve_batteries}"
        )

    bz = cfg.battery_zones
    if not (1.0 > bz.high > bz.nominal > bz.critical > 0.0):
        raise ConfigError(
            "battery_zones must satisfy 1 > high > nominal > critical > 0, got "
            f"high={bz.high}, nominal={bz.nominal}, critical={bz.critical}"
        )

    # r_min required (> 0) for non-holonomic platforms (FW / VTOL use Dubins)
    if cfg.platform.type is not PlatformType.MULTIROTOR and cfg.platform.r_min_m <= 0:
        raise ConfigError(
            f"platforms.{cfg.platform.type.value}.r_min_m must be > 0 (Dubins platform)"
        )

    # every ManeuverType has a power coefficient for the active platform
    missing = [m.value for m in ManeuverType if m not in cfg.platform.power_w]
    if missing:
        raise ConfigError(
            f"platforms.{cfg.platform.type.value}.power_w missing maneuvers: {missing}"
        )
    if any(v < 0 for v in cfg.platform.power_w.values()):
        raise ConfigError("power_w coefficients must be non-negative")

    # platform mass (used only by the B3 vertical-energy term, but validated now)
    if cfg.platform.mass_kg <= 0:
        raise ConfigError(f"platforms.{cfg.platform.type.value}.mass_kg must be > 0")

    if cfg.sensor.swath_width_m <= 0:
        raise ConfigError("sensor.swath_width_m must be > 0")
    if not (0.0 <= cfg.sensor.overlap_frac < 1.0):
        raise ConfigError("sensor.overlap_frac must be in [0, 1)")
    if cfg.sensor.sensor_power_w < 0:
        raise ConfigError("sensor.sensor_power_w must be >= 0")

    _op_areas = {"convex_hull", "bbox", "survey"}
    if cfg.coverage.operating_area not in _op_areas:
        raise ConfigError(
            f"coverage.operating_area must be one of {sorted(_op_areas)}"
        )
    if cfg.coverage.operating_margin_m < 0:
        raise ConfigError("coverage.operating_margin_m must be >= 0")

    if not (0.0 < cfg.aero.formation_drag_reduction < 1.0):
        raise ConfigError("aero.formation_drag_reduction must be in (0, 1)")

    if cfg.failure.hazard_rate_per_hour < 0:
        raise ConfigError("failure.hazard_rate_per_hour must be >= 0")

    if cfg.sim.dt_s <= 0:
        raise ConfigError("sim.dt_s must be > 0")
    if cfg.sim.max_timesteps <= 0:
        raise ConfigError("sim.max_timesteps must be > 0")

    if not (0.0 <= cfg.rth.reserve_frac < 1.0):
        raise ConfigError("rth.reserve_frac must be in [0, 1)")

    # ---- EM-01 energy map (Stage 1) ----
    # Finiteness is checked first: YAML admits .nan/.inf, and NaN silently
    # passes ordinary comparisons (NaN <= 0 and NaN < 1.0 are both False), so a
    # bare range check would let a non-finite value reach the grid builder.
    emc = cfg.rth.energy_map
    if emc.cell_m is not None and (not isfinite(emc.cell_m) or emc.cell_m <= 0):
        raise ConfigError(
            "rth.energy_map.cell_m must be a finite value > 0 (or omitted for battery-tied)"
        )
    if not isfinite(emc.yellow_penalty) or emc.yellow_penalty < 1.0:
        raise ConfigError("rth.energy_map.yellow_penalty must be finite and >= 1.0")
    if not isfinite(emc.red_threshold) or not (0.0 < emc.red_threshold <= 1.0):
        raise ConfigError("rth.energy_map.red_threshold must be finite and in (0, 1]")

    t0, t1 = cfg.tier_thresholds
    if not (0 < t0 < t1):
        raise ConfigError(f"tier_thresholds must satisfy 0 < t0 < t1, got {cfg.tier_thresholds}")

    if cfg.mc.n_min < 2 or cfg.mc.n_max < cfg.mc.n_min:
        raise ConfigError("mc requires n_max >= n_min >= 2")
    if cfg.mc.ci_tolerance <= 0:
        raise ConfigError("mc.ci_tolerance must be > 0")

    # ---- 2.5D coverage stack ----
    alt = cfg.layers.altitudes_m
    if len(alt) < 1:
        raise ConfigError("layers.altitudes_m must list at least 1 altitude")
    if any(b <= a for a, b in zip(alt, alt[1:])):
        raise ConfigError(f"layers.altitudes_m must be strictly ascending, got {alt}")
    if cfg.layers.assignment_policy not in _LAYER_POLICIES:
        raise ConfigError(
            f"layers.assignment_policy '{cfg.layers.assignment_policy}' "
            f"not one of {list(_LAYER_POLICIES)}"
        )

    # ---- 2.5D obstacle prisms ----
    if cfg.env.obstacle_floor_m < 0:
        raise ConfigError("env.obstacle_floor_m must be >= 0")
    cr = cfg.env.obstacle_ceil_range_m
    if cr is not None and not (len(cr) == 2 and cfg.env.obstacle_floor_m <= cr[0] <= cr[1]):
        raise ConfigError("env.obstacle_ceil_range_m must be [lo, hi] with floor <= lo <= hi")
