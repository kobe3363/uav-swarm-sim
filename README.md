# UAV Swarm Reconnaissance Simulation (2.5D Layered Architecture)

A modular, explainable simulation of **coordinated mission optimization for a homogeneous reconnaissance UAV swarm**. It is the Part III computational artifact for the master's thesis *Optimising flight missions between identical reconnaissance drones* (Konstantin Belena, Vilnius Gediminas Technical University, Antanas Gustaitis Aviation Institute).

The simulation operationalizes the thesis's central methodological contribution — **energy-weighted spatial decomposition by momentary battery level on a topological-graph (TGC/GVG) framework for identical drones in obstacle environments** — and proves its properties through a **Semi-Markov Decision Process (SMDP)** analysis layer.

The system operates in a **2.5D layered environment**: coverage planning stays strictly 2D within configurable horizontal altitude bands, connected by physics-accurate vertical climb and descent segments.

It is written for **clarity at a thesis defense**: one module per idea, strictly typed interfaces, and a direct mapping from every quantitative claim in the thesis to exactly one place in the code (see §11).

> **This is the single authoritative README.** It supersedes two earlier drafts that described different evolutionary snapshots of the same system. Both are preserved for history under `docs/archive/`:
> - `docs/archive/README_v1_single_layer.md` — the earlier "five layers" single-altitude draft.
> - `docs/archive/README_v2_2.5d_draft.md` — the earlier 2.5D draft.
>
> If anything below disagrees with an archived draft, **this document wins**. Where the code, this document, and the written thesis text disagree, those gaps are tracked explicitly in §12.

---

## 1. What this project is

A discrete-time, Monte-Carlo simulation that, for a fleet of identical reconnaissance drones:

1. ingests an exploration-area boundary (GeoJSON) and a synthetic obstacle field (extruded as 3D prisms over `[floor, ceil]` ranges);
2. chooses an optimal, energy-aware launch site strictly from a safe staging ring **outside** the survey polygon (a decision variable, not an assumption);
3. partitions the area among the drones — across altitude layers — so each drone's region is proportional to its **momentary battery level**;
4. enforces strict fleet logistics: a finite **shared battery pool** and exact battery-swap cycle counting;
5. flies kinematically realistic coverage paths under a **seven-state behavioral automaton** with a dynamic return-to-home rule, event-driven redistribution, proactive obstacle avoidance, battery-swap "reincarnation", and rigorous **terminal-state** evaluation (mission success vs. physics-dictated failure);
6. measures the result both deterministically (energy, duration, workload balance, swap metrics) and stochastically (stationary distribution over states, efficiency score), with statistical convergence;
7. optionally exports a structured **telemetry log**: GPX 1.1 tracks for GIS visualization and a JSONL event log for automated mission analysis;
8. optionally runs an **LLM-as-a-judge** over that telemetry log to auto-diagnose the mission — classifying the outcome, attributing a root cause, and **verifying the diagnosis against deterministic ground truth**.

---

## 2. Architecture: five layers mirroring the methodology

The package layout deliberately mirrors the three methodology layers of the thesis (physical model → planning → execution), plus an SMDP analysis layer and a supporting infrastructure layer.

| Package | Thesis layer | Responsibility |
|---|---|---|
| `physical_model/` | **Physical model** (guidelines 1.1–1.4) | Grey-box component energy model; aerodynamic formation correction; Dubins kinematics; the `MotionModel` platform abstraction; 1-D vertical takeoff/landing and inter-layer climb segments; turn aerodynamic penalties. |
| `planning/` | **Planning** (guidelines 2.1–2.2) | GeoJSON parsing; Poisson 3D-prism obstacle generation; layered environment mapping; GVG + TGC construction; the weighted decomposition (central contribution) and its baselines (classic Voronoi, k-means); Dijkstra cost database; launch-site optimizer; boustrophedon coverage paths; the grid comparative planner; pre-flight trajectory validation; target-visit tours; drone-sized moving obstacles. |
| `execution/` | **Execution** (guidelines 3.1–3.4) | The seven-state automaton; the agent and fleet; the event bus; dynamic RTH calculator; event-driven redistribution; proactive safety monitor; formation manager; battery-swap station with finite shared pool; hazard-rate failure model; three-tier scale-dependent selection; swarm passive/active dynamic-obstacle sensing. |
| `metrics/` | **Analysis (SMDP)** | State-history recording; deterministic mission metrics; SMDP estimation (embedded chain + mean sojourns); the stationary distribution with the embedded→time-weighted correction; the efficiency score; the Monte-Carlo runner with CI-based convergence; algorithm comparison; validation; the event-driven telemetry collector with GPX + JSONL exporters; the grounded LLM-as-a-judge mission-diagnosis engine. |
| `infrastructure/` | **(supporting)** | Typed configuration and `config_hash`; reproducible content-addressed RNG; logging; the `SimulationEngine` orchestrator; all visualization, including the optional view-only comm-range overlay. |
| `experiments/` | **(supporting)** | Thin CLI entry points that compose the layers into the thesis experiments. |

### How a single mission flows through the layers

```
config/default.yaml
   │  (infrastructure.config: typed, validated, unit-normalized; config_hash from raw YAML)
   ▼
EnvironmentMap (3D prisms)  →  LayerStack (sliced 2D horizontal planes)     [planning]
   ▼
GVG  →  TGC (corridors + free-space regions + adjacency)                    [planning]
   ▼
LaunchSiteOptimizer  →  launch site                                         [planning]
   │   (3 criteria: mean distance, formation-corrected energy, expected swaps;
   │    constrained to the staging ring outside the survey polygon)
   ▼
Decomposer (tier-selected): area ∝ momentary battery level (per layer)      [planning]
   ▼
CoveragePath per zone (boustrophedon + Dubins smoothing + leg validation)   [planning]
   ▼
SimulationEngine dt-loop (fail-fast):                                       [infrastructure]
   fleet init (circular deployment) → failure → safety → agents step
   (motion + energy + state + RTH) → swap station (shared pool)
   → drain events → redistribution → terminal evaluation
   ▼
StateHistory + MissionMetrics                                              [metrics]
   ▼
SMDP estimate → stationary π (embedded → time-weighted) → efficiency score  [metrics]
   ▼
[if telemetry enabled] GPX tracks + JSONL event log written to disk         [metrics]
   ▼
[offline, optional] LLM-as-a-judge over the JSONL → grounded diagnosis      [metrics]
   ▼
MonteCarlo / Comparison / Validation  →  CSV + Markdown + plots             [metrics + infrastructure]
```

The single most important computational subtlety is in `metrics/stationary_distribution.py`: the left-eigenvector of the embedded transition matrix yields **visit frequencies**, not time fractions. The code multiplies each embedded `π[i]` by the mean sojourn time `m_i` and renormalizes to obtain the **time-weighted** `π` that the efficiency score requires. The two distributions are always plotted side by side so the correction is visible rather than implicit.

---

## 3. Platform support and the energy-coefficient caveat

`platform_type` is a configuration enum; each run simulates **one** homogeneous platform:

- **`FIXED_WING`** — `DubinsModel` kinematics (minimum turn radius); formation flight yields a **drag/energy benefit** (≈15.14% in homogeneous formation, Guo et al. 2025), applied only during launch, transit-to-zone, and episodic RTH — never during dispersed coverage.
- **`MULTIROTOR`** — `HolonomicModel` kinematics (in-place turns, Euclidean legs); formation downwash is treated as a **safety constraint** (wake zones as "invisible obstacles"), **not** an energy benefit.
- **`VTOL`** — `DubinsModel` for the horizontal coverage phase, bracketed by 1-D vertical climb/descent segments; the three-phase mission of thesis §2.5 verbatim. Cruise gets the formation energy benefit; vertical phases treat downwash as a constraint.

> **⚠ Energy-coefficient caveat (read before trusting absolute numbers).**
> No physical flight experiments were conducted (thesis guideline 1.1). Motor power coefficients are **theoretical approximations from typical platform specifications** following Steup et al.'s component method. The **quadrotor** (`MULTIROTOR`) coefficients are closest to the validated Steup baseline. The **`FIXED_WING` and `VTOL`** coefficient tables are **coarser extrapolations** and should be read as *relative*, structurally-consistent estimates — not calibrated absolute energies. All comparative results (algorithm-vs-algorithm, site-vs-site, Dubins-vs-grid) are computed on **paired identical seeds**, so conclusions about *differences* are robust even where absolute energies are approximate.

---

## 4. Terminal states and realistic dynamics

The simulation enforces strict, mutually-exclusive terminal outcomes, evaluated once per tick inside `SimulationEngine.run()` (failure is tested before success), preventing unrealistic "zombie" computations. The verdict is carried on `MissionResult` as an `Outcome`:

- **`MISSION_SUCCESS`** — 100% of the partitioned area is covered AND every surviving drone has returned to `S0_IDLE`.
- **`MISSION_FAILED`** — a physics-dictated halt: an **airborne drone's battery reaches 0** (forced into `S_FAIL` mid-flight), or the **shared battery pool is exhausted** before coverage completes. Note: hazard-induced `S_FAIL` (the `failure_model`) does **not** trigger this — those failures populate `S_FAIL` for the elevated-hazard Monte-Carlo / SMDP statistics, and the run continues via redistribution.
- **`MISSION_INCOMPLETE`** — neither terminal condition fired before the run ended (e.g. the `sim.max_timesteps` ceiling). The default outcome.

**Circular deployment footprint:** drones are instantiated in a mathematically computed ring around the launch pose at `t = 0`, preventing artificial collisions and immediate `S_OBS` deadlocks.

---

## 5. The S_FAIL dual view (physical layer vs. analysis layer)

Failure is modeled **differently and deliberately** in two separate layers. This is a conscious modeling decision, documented here and in `metrics/smdp_estimator.py`:

- **Physical simulation layer (thesis-faithful, irreversible).** When a drone fails, `execution/fleet.py::kill` removes it permanently and `execution/redistribution.py` immediately re-partitions the **uncovered** work among the **surviving active** agents via the weighted TGC decomposer. The failed drone does not return. A battery-depletion failure of an airborne drone halts the run with `MISSION_FAILED` (§4); a hazard-induced failure (`λ > 0`) removes the agent and triggers redistribution without halting.

- **SMDP analysis layer (ergodicity device).** The stationary distribution `π = πP` exists only if the embedded Markov chain is **ergodic** (strongly connected). A terminal `S_FAIL` is an absorbing state and would make `π` undefined. The estimator therefore models a **generic agent-slot**: a failed slot is closed at a configurable mean repair/replacement time and given a synthetic transition `S_FAIL → S0` (a replacement drone occupies the slot). This closes the loop alongside the genuine `S3 → S_SWAP → S0` swap loop.

This is **not** a claim that the physical swarm self-heals. It is the standard renewal-theoretic treatment that lets us speak of long-run time-fractions per state for a continuously-operated slot. The estimator exposes `close_failure_loop` (default `True`); with `False` it refuses to compute `π` and reports `ergodic=False`, which is the correct behavior for a literal absorbing-failure reading.

Contrast with **battery swap**, which is reversible and does *not* trigger redistribution: the swapped drone resumes its own remaining plan. The swap/failure asymmetry is enforced by wiring (only `FAILURE` and `NEW_TASK` reach the redistributor), not by convention.

---

## 6. Configuration — all global inputs in one place

`config/default.yaml` mirrors the typed schema in `infrastructure/config.py` one-to-one. The complete set of global inputs that propagate through the system:

- **Fleet (homogeneous):** `n_drones`, `battery_capacity_wh`, `drone_dims_m`.
- **Platform:** `type`, cruise/coverage/climb/descent speeds, `r_min_m`, `omega_max`, and the per-maneuver grey-box **`power_w` table** (one table per platform type).
- **Sensor:** `swath_width_m`, `overlap_frac` (drives boustrophedon strip spacing).
- **Aerodynamics:** `formation_drag_reduction` (0.1514), downwash geometry, formation spacing, RTH rendezvous window.
- **Environment:** `geojson_path`, obstacle density / size range / shapes / class count, clearance buffer; obstacle prism `[floor, ceil]` ranges; layer band definitions.
- **Launch:** candidate sites (explicit list or sample count), the staging-ring standoff, and the three objective weights `w_distance`, `w_energy`, `w_swaps`.
- **Battery zones:** `high: 0.75`, `nominal: 0.40`, `critical: 0.20` (TERMINAL < 0.20).
- **Swap:** service time, number of bays, shared-pool size. **Failure:** hazard rate `λ` per flight-hour. **Safety:** separation, obstacle buffer, prediction horizon. **RTH:** check interval, small `reserve_frac` ε (≈0.05 — explicitly *not* a static 30%).
- **Simulation:** `dt_s`, `max_timesteps`, `master_seed`.
- **Monte Carlo:** `n_max: 1000`, `n_min`, `ci_tolerance`.
- **Tier thresholds:** `[15, 50]`.
- **Mission type:** `mission.type` = `coverage` (default) or `target_visit`; for target missions, `n_targets` or explicit `target_coordinates`, and `weight_targets_by_battery` (per-drone target count ∝ battery).
- **Dynamic obstacles:** `dynamic_obstacles.enabled` (default **false**), `count`, `speed_m_s`, `size_m`, `passive_sense_range_m`, `active_sense_range_m`, `active_scan_power_w`, `dynamic_hold_s` — drone-sized moving obstacles with swarm-triggered passive/active LIDAR sensing.
- **Telemetry (Phase 3, view-only — see §7):** `telemetry.enabled` (default **false**), `gpx_path`, `llm_log_path`, `fix_interval_s`, `origin_lat`/`origin_lon` (projection false origin, default Vilnius), `epoch_iso`.
- **Visualization (view-only, no physics effect):** `viz.show_comm_range` (default **false**), `comm_range_m`, `comm_range_alpha`, `comm_range_dashed` — draws a communication-range circle around each drone in the replay GIF and the state-colored path PNG.

> **Note on the failure hazard rate.** With `λ = 0`, the `S_FAIL` state is never entered and never populates in Monte Carlo; set `λ > 0` to study resilience and to give the failure-loop closure something to act on.
>
> **Note on `config_hash` stability.** The hash is computed from the **raw YAML before parsing**, so absent optional blocks (telemetry, dynamic obstacles, comm-range) do not change it, and adding a new field with a default preserves the hash for existing configs.

---

## 7. Observability and telemetry (Phase 3)

When enabled, the simulation engine emits a structured event log of every mission **discontinuity** — not raw per-tick dumps, but semantically classified transitions with per-phase deltas and instantaneous agent snapshots. The telemetry layer is architecturally **decoupled** from the simulation core: it shares the agent's existing `Recorder.open/close` protocol via a `FanoutRecorder` and pulls snapshots read-only from the live fleet. It never feeds back into physics, energy, or the SMDP. When disabled (the default), no telemetry objects are constructed and the run is **byte-identical** to the pre-telemetry baseline.

Two output formats:

- **GPX 1.1 tracks** — one `<trk>` per drone, for loading directly into QGIS, Google Earth, gpx.studio, or mission planners. Planar simulation coordinates (metres on a local tangent plane) are projected to geographic WGS-84 via an equirectangular approximation with a configurable false origin (`origin_lat`/`origin_lon`, default Vilnius). Altitude `z` maps to `<ele>` exactly, so 3D viewers render the layer structure correctly. One file per drone is written by `write_gpx`.
- **JSONL event log** — a compact, causal trace of the mission (run header → events → run summary) small enough for an LLM context window. Each event carries a semantic verb (`OBSTACLE`, `SWAP_REQ`, `SWAP_DONE`, `FAIL`, `TERMINAL`, …), the state transition (`from`/`to`/`reason`), battery fraction and zone, position, and per-phase deltas (`Δt`, `ΔE`, `Δdist`). Sparse by design: fields appear only where they apply.

### Enabling telemetry

Add a top-level `telemetry:` block to the scenario YAML you pass with `--config`:

```yaml
telemetry:
  enabled: true
  gpx_path: "runs/my_run/tracks.gpx"
  llm_log_path: "runs/my_run/events.jsonl"
  fix_interval_s: 30.0                # periodic GPX position-fix cadence (seconds)
  origin_lat: 54.6872                 # tangent-plane false origin latitude (default: Vilnius)
  origin_lon: 25.2797                 # tangent-plane false origin longitude
  epoch_iso: "2026-01-01T00:00:00Z"   # GPX <time> = this epoch + sim seconds
```

All fields have defaults, so the minimal enablement is just `telemetry: {enabled: true}`. When the block is absent (or `enabled: false`), the feature does not exist — no objects, no files, and `config_hash` is unchanged.

### Telemetry output schema

**JSONL** (one JSON object per line):

```
{"kind":"run_header","config_hash":"...","platform":"MULTIROTOR","n_drones":5,...}
{"kind":"event","t":10.0,"event":"STATE","drone":0,"from":"S0_IDLE","to":"S1_TRANSIT","reason":"launch","batt_frac":0.99,...}
{"kind":"event","t":300.0,"event":"OBSTACLE","drone":0,"from":"S2_MISSION","to":"S_OBS","reason":"obstacle_threat",...}
{"kind":"event","t":1180.0,"event":"TERMINAL","reason":"coverage_complete","outcome":"MISSION_SUCCESS","coverage_frac":1.0}
{"kind":"run_summary","outcome":"MISSION_SUCCESS","coverage_frac":1.0,"t_end_s":1180.0,"per_drone_sorties":{"0":2},...}
```

**GPX** (standard GPX 1.1, one `<trk>` per drone):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="uav-swarm-sim" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>drone_0</name>
    <trkseg>
      <trkpt lat="54.68720000" lon="25.27970000">
        <ele>100.00</ele>
        <time>2026-01-01T00:00:00Z</time>
      </trkpt>
      ...
    </trkseg>
  </trk>
</gpx>
```

---

## 8. Automated mission diagnosis (Phase 4 — LLM as a judge)

An LLM-as-a-judge reads the Phase 3 JSONL log and produces a structured, **grounded** mission diagnosis: an outcome classification, a single root cause from a fixed taxonomy, a causal narrative, contributing factors, a critical-event timeline, and recommendations. The logic lives in `metrics/llm_judge.py`; the CLI is `experiments/run_llm_diagnosis.py`.

Three design commitments make this an evaluation device rather than a chatbot:

- **Deterministic grounding.** Before any model is called, the ground-truth facts (outcome, the terminal trigger, which drone died, per-drone minimum battery, swap / obstacle counts) are extracted directly from the log — no model involved. These facts are handed to the model AND used afterwards to **verify its claims**. If the model names a drone that never failed or contradicts the recorded outcome, the mismatch is flagged in a "grounding audit" rather than silently trusted.
- **Fixed taxonomy.** The model must attribute the root cause from a closed set aligned to the simulation's terminal physics (`BATTERY_DEPLETION_AIRBORNE`, `SHARED_POOL_EXHAUSTION`, `COVERAGE_TIMEOUT`, `HAZARD_ATTRITION`, `OBSTACLE_THRASHING`, `SWAP_LOGISTICS`, `NOMINAL_SUCCESS`, `UNDETERMINED`), so verdicts are machine-comparable across runs.
- **Injectable, model-agnostic.** The judge takes a `model` callable `(system, user) -> str`. An Anthropic adapter is provided (`anthropic_model`, lazily imported), but any other LLM is wired in with a five-line adapter. The module imports with no third-party dependency, and the whole pipeline (facts → prompt → parse → ground → render) is unit-testable offline.

### Running the diagnosis

The diagnosis runs **offline against a telemetry JSONL file** — independent of `--out` and of the simulation run itself.

```bash
# 1. produce a telemetry log (§7), e.g. runs/demo/events.jsonl

# 2. FREE: inspect the deterministic facts + the exact prompt, no API call, no cost
python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --dry-run

# 3. LIVE diagnosis -> writes a Markdown report (needs an API key)
python -m uav_swarm_sim.experiments.run_llm_diagnosis \
    --log runs/demo/events.jsonl --out runs/demo/diagnosis.md --model claude-sonnet-4-6
```

The live call requires an Anthropic API key (`ANTHROPIC_API_KEY` env var, or `--api-key`) and the `anthropic` package (`pip install anthropic` — only for live diagnosis, not for `--dry-run`).

### Interactive analysis (the mission-analyst prompt)

For quick, **human-readable** interpretation — when you (or an examiner) want a narrative diagnosis and you also have the **figures** in hand — the repository ships a ready-to-use **mission-analyst system prompt** (`mission_analyst_prompt.md`). Paste it as the system prompt in any LLM chat, attach the run's artifacts (lead with `events.jsonl`, add the config YAML and the figures), and ask *"analyze this run."* Unlike the CLI judge, it reads the plots as well as the logs and returns a structured written report (verdict, setup, what happened, per-drone notes, ranked problems, ranked recommendations, anomalies, what would sharpen the analysis). It is grounding-disciplined: it cites the figure / drone / timestamp behind each claim, separates observation from hypothesis, and flags contradictions between sources rather than papering over them.

The two interfaces are complementary: the CLI judge for a scriptable, auditable verdict; the prompt for the readable story. If you run both, agreement raises confidence and disagreement is itself a useful signal.

---

## 9. Installation

Requires **Python 3.12+**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # networkx, shapely, scipy, numpy, matplotlib, pulp, pyyaml, pytest
pip install -e .                        # editable install of the uav_swarm_sim package
```

The `anthropic` package is required **only** for live LLM diagnosis (§8). Everything else — the simulation, telemetry export, and the diagnosis `--dry-run` — runs without it.

---

## 10. Running

All entry points live in `src/uav_swarm_sim/experiments/` and take a `--config` path plus (usually) an output directory. Runs are **deterministic** given `(config, master_seed, replication, algorithm, planner)`.

```bash
# One mission with full visual output (the defense demo)
python -m uav_swarm_sim.experiments.run_single_mission --config config/default.yaml --out runs/demo

# Headline experiment: classic_voronoi vs tgc_basic vs weighted_voronoi
python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/scenarios/tier_mid_comparison.yaml --out runs/decomp

# Guideline 1.2: Dubins vs discretized grid (FIXED_WING / VTOL only)
python -m uav_swarm_sim.experiments.run_kinematics_comparison --config config/default.yaml --out runs/kinematics

# Guideline 3.3: scale-tier sweep, locate the empirical break-even in 16–49
python -m uav_swarm_sim.experiments.run_scale_tiers --config config/default.yaml --n 8 12 24 36 48 64 80 --out runs/tiers

# §2.4: launch-site optimization study + suitability heatmap
python -m uav_swarm_sim.experiments.run_launch_site_study --config config/default.yaml --out runs/launch
python -m uav_swarm_sim.experiments.plot_launch_suitability

# B6.2 Fleet Sizing (analytical Pareto knee)
python -m uav_swarm_sim.experiments.run_fleet_sizing_analyzer --config config/default.yaml --n-max 20

# LLM-as-a-judge mission diagnosis over a telemetry log (see §8)
python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --dry-run
```

Each run writes figures (environment, partition, trajectories, per-agent state Gantt, battery traces, embedded-vs-time-weighted `π` bars, comparison box plots, Monte-Carlo convergence), a `result.json`/CSV, and a structured log. To run any experiment **with telemetry enabled**, add the `telemetry:` block to the YAML you pass via `--config` (§7).

Smoke test:

```bash
pytest tests/test_smoke.py
```

---

## 11. Mapping thesis claims to code (the explainability invariant)

The naming invariant: every quantitative claim in the thesis maps to exactly one place in the code, and the state / maneuver / algorithm names are identical across the simulation, the metrics layer, the plots, and the thesis text (single home: `infrastructure/enums.py`). The table is grouped by methodology level.

### Physical-model level (guidelines 1.1–1.4)

| Thesis claim | Module(s) |
|---|---|
| Continuous component (grey-box) energy, `E = Σ P(maneuver)·dt`; no static distance-averages | `physical_model/energy_model.py` |
| Kinematically realistic trajectories (true flyable length) | `physical_model/dubins.py`, `physical_model/motion_model.py` |
| Grid comparison planner (guideline 1.2: planning-speed vs kinematic-accuracy) | `planning/grid_planner.py` + `experiments/run_kinematics_comparison.py` |
| 15.14% formation drag benefit, scope-limited to cruise on FW/VTOL | `physical_model/aero_correction.py` + `execution/formation_manager.py` |
| 2.5D vertical climb/descent segments; mass couples in only via `m·g·dz` on climb | `physical_model/vertical_segments.py` + `physical_model/energy_model.py` |
| Per-maneuver power table; battery model | `physical_model/drone_specs.py`, `physical_model/battery.py` |
| Three deterministic metrics (energy, duration, workload std) | `metrics/mission_metrics.py` |

### Planning level (guidelines 2.1–2.2)

| Thesis claim | Module(s) |
|---|---|
| GeoJSON ingest of the survey boundary | `planning/geojson_parser.py` |
| Synthetic obstacle field (drone-sized, size classes, safe corridor / clearance) | `planning/obstacle_generator.py`, `planning/environment_map.py` |
| Works in obstacle environments (clearance between distinct obstacle classes) | `planning/gvg_builder.py` |
| Topological free-space regions (the atomic decomposition units) | `planning/tgc.py` |
| **Area ∝ momentary battery — the central contribution** | `planning/weighted_decomposition.py` (with `TgcBasicDecomposer` as its switched-off ablation twin) |
| Position-based baseline: Euclidean Voronoi | `planning/classic_voronoi.py` |
| Position-based baseline: k-means | `planning/kmeans_heuristic.py` (see §12 — currently the tier heuristic, not yet a first-class comparison peer) |
| Launch site as an optimization variable (3 criteria, staging ring outside the polygon) | `planning/launch_site_optimizer.py` |
| Boustrophedon coverage paths by GSD/overlap | `planning/coverage_path.py` |
| Multi-layer (2.5D) planning | `planning/layer_planner.py` |
| Pre-flight trajectory validation / leg repair | `planning/trajectory_validation.py` |
| Target-visit missions (per-drone tours) | `planning/target_mission.py` |
| Dynamic (moving) obstacles | `planning/dynamic_obstacles.py` |

### Execution level (guidelines 3.1–3.4)

| Thesis claim | Module(s) |
|---|---|
| Seven-state behavioral automaton | `execution/state_machine.py` |
| Agent and fleet | `execution/agent.py`, `execution/fleet.py` |
| Dynamic RTH (route-vs-return, not a static reserve) | `execution/rth_calculator.py` |
| Event-driven redistribution; swap ≠ failure (only FAILURE/NEW_TASK reach the redistributor) | `execution/redistribution.py`, `execution/events.py` |
| Three-tier scale selection (~15 threshold) | `execution/algorithm_selector.py` |
| Proactive, continuous obstacle avoidance (wake zones as invisible obstacles) | `execution/safety_monitor.py` |
| Swarm passive/active dynamic-obstacle sensing posture | `execution/sensing.py` |
| Battery swap "reincarnation" with a finite shared pool | `execution/swap_station.py` |
| Hazard-rate failure model | `execution/failure_model.py` |
| Aerodynamic formation manager | `execution/formation_manager.py` |

### SMDP analysis level

| Thesis claim | Module(s) |
|---|---|
| Semi-Markov, not Markov (battery = hidden memory → state is `(state, sojourn time)`) | `metrics/smdp_estimator.py` |
| Stationary `π` with the embedded→time-weighted correction | `metrics/stationary_distribution.py` |
| Efficiency = `π(S2) / (π(S3)+π(S_OBS)+π(S_SWAP))` | `metrics/efficiency_score.py` |
| Monte Carlo with CI-based convergence | `metrics/monte_carlo.py`, `metrics/convergence.py` |
| State-history recording | `metrics/state_history.py` |
| Algorithm comparison; validation | `metrics/comparison.py`, `metrics/validation.py` |

### Infrastructure & observability (supporting)

| Concern | Module(s) |
|---|---|
| Typed configuration + `config_hash` | `infrastructure/config.py` |
| Reproducible content-addressed RNG (paired-seed Monte Carlo) | `infrastructure/rng.py` |
| Simulation engine / orchestrator | `infrastructure/simulation_engine.py` |
| Frozen domain types; enums (single naming home) | `infrastructure/core_types.py`, `infrastructure/enums.py` |
| Visualization (incl. view-only comm-range overlay) | `infrastructure/visualization.py` |
| Telemetry collector; GPX + JSONL exporters | `metrics/telemetry_log.py`, `metrics/gpx_exporter.py`, `metrics/llm_log_exporter.py` |
| Grounded LLM-as-a-judge | `metrics/llm_judge.py` |

---

## 12. Reconciliation: thesis text ⇄ code ⇄ plan, and the hardening roadmap

This section is the heart of "single source of truth": it records exactly where the written thesis text, this code, and the agreed implementation plan **diverge**, plus the active roadmap. Nothing here is silently assumed.

### A. Where the simulation spec is ahead of the written Part II text

These do not affect the code; the author may wish to reconcile the written methodology before the defense.

1. **The entire SMDP / stochastic-analysis layer is not yet described in Part II.** Guideline 1.4 currently promises only the three deterministic metrics (energy, duration, workload std). The simulation additionally produces the stationary distribution `π` and the efficiency score, with Monte-Carlo convergence. *Recommended addition:* a subsection introducing (a) the semi-Markov justification (battery as hidden memory → two-dimensional `(state, sojourn time)`); (b) `π` as the time-fraction-per-state metric, valid only under the closed loop; (c) the embedded→time-weighted correction; (d) the efficiency score and its throughput-oriented denominator.
2. **The closed-loop ergodicity requirement deserves an explicit sentence** distinguishing the *physical* irreversibility of failure from the *analytical* slot-replacement closure (§5).
3. **The efficiency-score denominator choice (including `π(S_SWAP)`) is a defensible but unstated modeling decision** — swap time as mission overhead under a throughput metric, consistent with the thesis leitmotif that minimizing swaps saves both energy and duration. State it as a choice.
4. **The Dubins-vs-grid comparison and the launch-site three-criterion objective** are described in Part II prose but not yet as formal experiment specifications; a one-line forward reference in §2.7.1 ties the text to `run_kinematics_comparison` / `run_launch_site_study`.
5. **The battery-zone thresholds (HIGH ≥75 / NOMINAL ≥40 / CRITICAL ≥20 / TERMINAL <20)** appear only in the simulation, as coarse transition guards and reporting bins. If cited as results, introduce them in Part II as a modeling parameter.

### B. Plan deltas agreed during the rebuild discussion

1. **External-literature validation is dropped.** No access to the source datasets of the six anchor papers. Validation is therefore **internal only**: paired-seed comparative claims (systematic modeling error cancels in differences), the `∫P·dt` ≈ closed-form property check, byte-identity off-path checks, and Monte-Carlo convergence. `metrics/validation.py` is to be refocused accordingly (roadmap H3); any external-reproduction code/claims should be removed.
2. **k-means is to be elevated to a first-class comparison peer** alongside Euclidean Voronoi, classic-TGC, and weighted-TGC, all run on paired seeds through the identical pipeline (roadmap H2). Today `DecompositionAlgo` contains only `{CLASSIC_VORONOI, TGC_BASIC, WEIGHTED_VORONOI}` and k-means is used as the tier heuristic.
3. **A fine-grained adaptive scale sweep (2, 4, …, 100)** is the experimental design for the comparison study, with CI-based adaptive stopping rather than a fixed N (roadmap H3). `metrics/convergence.py` is the seed; the three-tier policy remains the *operational* selector and is *informed* by the sweep's empirical break-even.
4. **GPX is confirmed real and GIS-valid** (WGS-84, equirectangular, Vilnius origin, exact `<ele>`), loadable into gpx.studio. **Done** — `metrics/gpx_exporter.py` (§7).

### C. Known status and active hardening roadmap (H0–H5)

Current test status: **170 of 172 passing.** Two known-red tests in `tests/test_obstacle_recovery.py` fail on a test-mock interface drift (`_Motion.advance()` does not accept the `current_pose` keyword the production `agent.py` now passes) — a localized mock fix, not a core defect. (An earlier draft cited "151/151 green"; the suite has since grown.)

- **H0 — single source of truth (this document).** Merge the two README drafts into one authority; archive both; add the thesis→module table. **Done.**
- **H1 — green the suite.** Fix the two mock-drift tests; fix the one self-flagged FSM correctness anomaly in `execution/state_machine.py` (a `BatteryZone.TERMINAL` entry currently transitions `S2_MISSION → S3_RTH` with the comment *"RTH should pre-empt"* — dynamic RTH must trigger *before* TERMINAL is reached, not after). Restore the all-green net.
- **H2 — align the decomposition axis to the plan.** Promote k-means to a first-class peer (delta B.2).
- **H3 — experiment layer for the study.** Build/extend the adaptive fine-grained scale sweep (delta B.3); refocus `validation.py` to internal validation only (delta B.1).
- **H4 — prune genuine duplication (optional, light touch).** Consolidate any truly overlapping `metrics/` modules; remove dead code.
- **H5 — observability + study runs.** Confirm GPX/telemetry (done); run the altitude-tradeoff and energy-vs-position studies; generate the defense figures.

### D. Minor doc/code path inconsistencies to clean up (H1)

- The mission-analyst prompt is referenced in places as `docs/mission_analyst_prompt.md` but currently lives at the repository root as `mission_analyst_prompt.md`. Pick one location and make the references consistent.

---

## 13. Scope boundaries

Explicitly **out of scope** (named in the thesis as context, not required by the guidelines): wind-field modeling, communication/network *modeling* (note: `viz.show_comm_range` draws a comm-range circle for readability only — it does **not** model link budgets or connectivity), the 3-D Dubins airplane extension (excluded by §2.5's constant-altitude assumption), real flight-data regression (no physical experiments), and learning-based planners (MARL/DRL). The architecture isolates each so that adding one later touches a single module.
