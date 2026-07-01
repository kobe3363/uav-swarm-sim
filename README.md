# UAV Swarm Reconnaissance Simulation (2.5D Layered Architecture)

A modular, explainable simulation of **coordinated mission optimization for a homogeneous reconnaissance UAV swarm**. It is the computational artifact for the master's thesis *Optimising flight missions between identical reconnaissance drones* (Konstantin Belena, Vilnius Gediminas Technical University, Antanas Gustaitis Aviation Institute).

The simulation operationalizes the thesis's central methodological contribution — **energy-weighted spatial decomposition by momentary battery level on a topological-graph (TGC/GVG) framework for identical drones in obstacle environments** — and proves its properties through a **Semi-Markov Decision Process (SMDP)** analysis layer.

The system operates in a **2.5D layered environment**: coverage planning stays strictly 2D at a fixed mission altitude, connected by physics-accurate vertical climb and descent segments (e.g. takeoff to the coverage altitude, and an optional return-to-home echelon above the mission altitude). The mission itself is flown at constant altitude — the drones do not dive between layers during coverage.

It is written for **clarity at a thesis defense**: one module per idea, strictly typed interfaces, and a direct mapping from every quantitative claim in the thesis to exactly one place in the code (see §13).

> **This is the single authoritative README** for the current state of the system. Where the code, this document, and the written thesis text disagree, those gaps are tracked explicitly in §13.

---

## 1. What this project is

A discrete-time, Monte-Carlo simulation that, for a fleet of identical reconnaissance drones:

1. ingests an exploration-area boundary (GeoJSON) and a synthetic obstacle field (full-height prisms — fixed, immovable cylinders taller than the mission altitude, so they block on every metre of altitude);
2. chooses an optimal, energy-aware launch site strictly from a safe staging ring **outside** the survey polygon (a decision variable, not an assumption);
3. partitions the area among the drones so each drone's region is proportional to its **momentary battery level**;
4. enforces strict fleet logistics: a finite **shared battery pool** and exact battery-swap cycle counting;
5. flies kinematically realistic coverage paths under a **seven-state behavioral automaton** with a dynamic return-to-home rule, event-driven redistribution, proactive obstacle avoidance, battery-swap "reincarnation", and rigorous **terminal-state** evaluation (mission success vs. physics-dictated failure);
6. measures the result both deterministically (energy, duration, workload balance, swap metrics) and stochastically (stationary distribution over states, efficiency score), with statistical convergence;
7. writes every run into a **structured, self-describing run folder** (§9): one `plan.json` (the exact inputs) and one `results.json` (the aggregated outcome) per simulation, plus all figures, the replay GIF, and GPX tracks — so any run is reproducible, comparable, and analyzable after the fact;
8. optionally runs an **LLM-as-a-judge** over the telemetry log to auto-diagnose the mission — classifying the outcome, attributing a root cause, and **verifying the diagnosis against deterministic ground truth**.

---

## 2. Architecture: layers mirroring the methodology

The package layout deliberately mirrors the three methodology layers of the thesis (physical model → planning → execution), plus an SMDP analysis layer and a supporting infrastructure layer.

| Package | Thesis layer | Responsibility |
|---|---|---|
| `physical_model/` | **Physical model** (guidelines 1.1–1.4) | Grey-box component energy model; aerodynamic formation correction; Dubins kinematics; the `MotionModel` platform abstraction; 1-D vertical takeoff/landing and inter-layer climb segments; turn aerodynamic penalties. |
| `planning/` | **Planning** (guidelines 2.1–2.2) | GeoJSON parsing; Poisson prism obstacle generation; layered environment mapping; GVG + TGC construction; the weighted decomposition (central contribution) and its three position-based baselines (Euclidean Voronoi, position k-means, unweighted TGC); Dijkstra cost database; launch-site optimizer; boustrophedon coverage paths; the grid comparative planner; pre-flight trajectory validation; target-visit tours; drone-sized moving obstacles. |
| `execution/` | **Execution** (guidelines 3.1–3.4) | The seven-state automaton; the agent and fleet; the event bus; dynamic RTH calculator; event-driven redistribution; proactive safety monitor; formation manager; battery-swap station with finite shared pool; hazard-rate failure model; three-tier scale-dependent selection; swarm passive/active dynamic-obstacle sensing. |
| `metrics/` | **Analysis (SMDP)** | State-history recording; deterministic mission metrics; SMDP estimation (embedded chain + mean sojourns); the stationary distribution with the embedded→time-weighted correction; the efficiency score; the Monte-Carlo runner with CI-based convergence; algorithm comparison and the empirical break-even estimator; internal validation; the structured run-output writer; the telemetry collector with GPX + JSONL exporters; the grounded LLM-as-a-judge. |
| `infrastructure/` | **(supporting)** | Typed configuration and `config_hash`; reproducible content-addressed RNG; logging; the `SimulationEngine` orchestrator; all visualization. |
| `experiments/` | **(supporting)** | Thin CLI entry points that compose the layers into the thesis experiments. |

### How a single mission flows through the layers

```
config/default.yaml
   │  (infrastructure.config: typed, validated, unit-normalized; config_hash from raw YAML)
   ▼
EnvironmentMap (full-height prisms) → LayerStack (the mission-altitude plane)   [planning]
   ▼
GVG  →  TGC (corridors + free-space regions + adjacency)                        [planning]
   ▼
LaunchSiteOptimizer  →  launch site                                             [planning]
   │   (3 criteria: mean distance, formation-corrected energy, expected swaps;
   │    constrained to the staging ring outside the survey polygon)
   ▼
Decomposer (tier-selected): area ∝ momentary battery level                      [planning]
   ▼
CoveragePath per zone (boustrophedon + Dubins smoothing + leg validation)       [planning]
   ▼
SimulationEngine dt-loop (fail-fast):                                           [infrastructure]
   fleet init (circular deployment) → failure → safety → agents step
   (motion + energy + state + RTH) → swap station (shared pool)
   → drain events → redistribution → terminal evaluation
   ▼
StateHistory + MissionMetrics                                                  [metrics]
   ▼
SMDP estimate → stationary π (embedded → time-weighted) → efficiency score      [metrics]
   ▼
Structured run folder: plan.json + results.json + figures + GIF + GPX           [metrics]
   ▼
[offline, optional] LLM-as-a-judge over the JSONL → grounded diagnosis          [metrics]
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

### The seven-state automaton and return-to-home priority

Each drone runs a seven-state FSM (`S0_IDLE, S1_TRANSIT, S2_MISSION, S_OBS, S3_RTH, S_SWAP, S_FAIL`) plus a swarm-level vigilance posture. Inside `S2_MISSION`, the return triggers are evaluated in a deliberate **priority order**:

```
obstacle threat  >  dynamic RTH (route-vs-return)  >  CRITICAL battery (>=40% net)
                 >  TERMINAL battery (<20% net)     >  coverage complete
```

The **dynamic RTH** (route-vs-return energy reserve, not a static 30% rule) is the primary early-return mechanism and pre-empts the crude battery-zone nets: a normally-draining drone returns at the CRITICAL boundary and never reaches TERMINAL while still covering. The battery-zone guards are progressively-severe last-resort safety nets.

---

## 5. The S_FAIL dual view (physical layer vs. analysis layer)

Failure is modeled **differently and deliberately** in two separate layers. This is a conscious modeling decision, documented here and in `metrics/smdp_estimator.py`:

- **Physical simulation layer (thesis-faithful, irreversible).** When a drone fails, `execution/fleet.py::kill` removes it permanently and `execution/redistribution.py` immediately re-partitions the **uncovered** work among the **surviving active** agents via the weighted TGC decomposer. The failed drone does not return. A battery-depletion failure of an airborne drone halts the run with `MISSION_FAILED` (§4); a hazard-induced failure (`λ > 0`) removes the agent and triggers redistribution without halting.

- **SMDP analysis layer (ergodicity device).** The stationary distribution `π = πP` exists only if the embedded Markov chain is **ergodic** (strongly connected). A terminal `S_FAIL` is an absorbing state and would make `π` undefined. The estimator therefore models a **generic agent-slot**: a failed slot is closed at a configurable mean repair/replacement time and given a synthetic transition `S_FAIL → S0` (a replacement drone occupies the slot). This closes the loop alongside the genuine `S3 → S_SWAP → S0` swap loop.

This is **not** a claim that the physical swarm self-heals. It is the standard renewal-theoretic treatment that lets us speak of long-run time-fractions per state for a continuously-operated slot. The estimator exposes `close_failure_loop` (default `True`); with `False` it refuses to compute `π` and reports `ergodic=False`, the correct behavior for a literal absorbing-failure reading.

Contrast with **battery swap**, which is reversible and does *not* trigger redistribution: the swapped drone resumes its own remaining plan. The swap/failure asymmetry is enforced by wiring (only `FAILURE` and `NEW_TASK` reach the redistributor), not by convention.

---

## 6. Configuration — all global inputs in one place

`config/default.yaml` mirrors the typed schema in `infrastructure/config.py` one-to-one. The complete set of global inputs:

- **Fleet (homogeneous):** `n_drones` (default 5), `battery_capacity_wh` (default 100), `drone_dims_m`, `total_reserve_batteries` (the finite shared swap pool; `null` = unbounded).
- **Platform:** `type`, cruise/coverage/climb/descent speeds, `r_min_m`, `omega_max`, and the per-maneuver grey-box **`power_w` table** (one table per platform type).
- **Sensor:** `swath_width_m` (default 100), `overlap_frac` (default 0.5 → effective strip = 50 m); drives boustrophedon strip spacing.
- **Aerodynamics:** `formation_drag_reduction` (0.1514), downwash geometry, formation spacing, RTH rendezvous window.
- **Environment:** `geojson_path`, `coverage_altitude_m` (default 100), obstacle density / size range / shapes / class count, clearance buffer; obstacle prism `[floor, ceil]` range (`obstacle_ceil_range_m: null` ⇒ **full-height** prisms — the default and the thesis model).
- **Launch:** candidate sites (explicit list or sample count), the staging-ring standoff, and the three objective weights `w_distance`, `w_energy`, `w_swaps`.
- **Battery zones:** `high: 0.75`, `nominal: 0.40`, `critical: 0.20` (TERMINAL < 0.20).
- **Swap:** service time, number of bays, shared-pool size. **Failure:** hazard rate `λ` per flight-hour (default 0.3, deliberately elevated so `S_FAIL` populates). **Safety:** separation, obstacle buffer, prediction horizon. **RTH:** check interval, small `reserve_frac` ε (≈0.05 — explicitly *not* a static 30%).
- **Simulation:** `dt_s`, `max_timesteps`, `master_seed` (default 42).
- **Monte Carlo:** `n_max: 1000`, `n_min: 30`, `ci_tolerance: 0.01` (stop when the 95% CI half-width of `π_time(S2)` ≤ this).
- **Tier thresholds:** `[15, 50]` (≤15 heuristic; 16–49 compare both; ≥50 TGC).
- **Mission type:** `mission.type` = `coverage` (default) or `target_visit`.
- **Dynamic obstacles:** `dynamic_obstacles.enabled` (default **false**) and its sensing parameters.
- **Telemetry / Visualization:** `telemetry.enabled` (default **false**); `viz.show_comm_range` (view-only, no physics effect).

> **Note on the failure hazard rate.** With `λ = 0`, the `S_FAIL` state is never entered. Set `λ > 0` to study resilience and to give the failure-loop closure something to act on.
>
> **Note on `config_hash`.** The hash is computed from the **merged config after applying CLI overrides** but before unit/enum transformation. Enabling telemetry via override is a genuine config change and is reflected in the hash; absent optional blocks do not change it.

---

## 7. Decomposition algorithms (the comparison axis)

`DecompositionAlgo` has four **first-class, paired-seed** peers — three position-based baselines plus the contribution:

| Algorithm | Kind | Module |
|---|---|---|
| `classic_voronoi` | Euclidean Voronoi (position-based) | `planning/classic_voronoi.py` |
| `kmeans` | position k-means (position-based) | `planning/kmeans_heuristic.py` (`weighted=False`) |
| `tgc_basic` | unweighted topological (position-based) | `planning/weighted_decomposition.py` (ablation twin) |
| `weighted_voronoi` | **battery-weighted TGC — the central contribution** | `planning/weighted_decomposition.py` |

All four run through the *identical* pipeline on the *same* per-replication seeds, so any metric difference is attributable to the algorithm, not the noise. (`kmeans` with `weighted=True` is also the heuristic-tier realization used by the small-swarm tier selector — same code, two roles, distinguished by its reported identity.)

---

## 8. Observability and telemetry

When enabled, the engine emits a structured event log of every mission **discontinuity** — semantically classified transitions with per-phase deltas and read-only agent snapshots. The telemetry layer is architecturally **decoupled** from the simulation core (it shares the agent's `Recorder.open/close` protocol via a `FanoutRecorder`); when disabled (the default), no telemetry objects are constructed and the run is **byte-identical** to the pre-telemetry baseline.

Two output formats:

- **GPX 1.1 tracks** — one `<trk>` per drone, for QGIS / Google Earth / gpx.studio. Planar simulation metres are projected to WGS-84 via an equirectangular approximation with a configurable false origin (default Vilnius); altitude `z` maps to `<ele>` exactly. One file per drone (`tracks_drone_<id>.gpx`).
- **JSONL event log** — a compact causal trace (run header → events → run summary) small enough for an LLM context window. Each event carries a semantic verb, the state transition (`from`/`to`/`reason`), battery fraction and zone, position, and per-phase deltas. Sparse by design.

Enable by adding `telemetry: {enabled: true}` to the scenario YAML (all sub-fields have defaults). `run_single_mission` enables telemetry automatically so the demo always produces GPX.

---

## 9. Structured run output — where results live

Every experiment writes into one self-describing, comparable **run folder**. A run holds one or more **simulations** (e.g. one per decomposition algorithm); each simulation owns its artifacts and two JSON logs:

```
runs/run-2026-06-28-11-59-35/          ← a RUN  (name = the dated folder, id = a GUID)
  run.json                             ← manifest: identity, software/git commit, timing, sims
  simulation-weighted_voronoi/         ← a SIMULATION within the run
    plan.json                          ← the launch PLAN (every input/setup)
    results.json                       ← the OUTCOME (success rate, SMDP, MC logic, timing)
    environment.png partition.png paths.png replay.gif state_gantt.png
    battery.png pi_bars.png tracks_drone_*.gpx
  simulation-kmeans/ ...
```

Runs are identified and compared by `run_id` (GUID), `run_name` (date), and a per-simulation `config_hash` (exact-input match).

- **`plan.json`** — a curated `setup` highlight block (drones, drone type, battery capacity, the battery-zone thresholds 0.75/0.40/0.20, obstacle setup + the obstacle-to-survey-area ratio, dynamic-obstacles flag, decomposition algorithm, flight planner, the energy-weighting flag, mission type, seed), plus engine-derived spatial quantities (survey/free/obstacle areas, launch pose, candidate count), plus the **full config dump**, plus a software/git block for reproducibility.
- **`results.json`** — for a Monte-Carlo batch: how many replications ran and **why it stopped** (`ci_converged` vs `reached_n_max`) with the full convergence trace; mission outcome counts and success fraction (distinct from the SMDP non-ergodic share); the SMDP stationary distribution + efficiency (with 95% CIs); per-metric mean/std/CI (energy, duration, workload std, planning time); and total + per-replication wall time. For a single mission: the terminal outcome, its SMDP, the single-run metrics, and timing.

All JSON is strict-valid (non-finite values become `null`), so it loads cleanly in `jq`, pandas, or any GIS/analysis tool.

---

## 10. Installation

Requires **Python 3.12+**.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: source .venv/Scripts/activate
pip install -r requirements.txt        # networkx, shapely, scipy, numpy, matplotlib, pulp, pyyaml, pytest
pip install -e .                        # editable install of the uav_swarm_sim package
```

The `anthropic` package is required **only** for live LLM diagnosis (§12). Everything else runs without it.

---

## 11. Running

All entry points live in `src/uav_swarm_sim/experiments/`. Runs are **deterministic** given `(config, master_seed, replication, algorithm, planner)`.

```bash
# One mission, full visual dump + GPX, into a structured run folder (the defense demo)
python -m uav_swarm_sim.experiments.run_single_mission --config config/default.yaml --name demo --base runs

# Headline study: classic_voronoi vs kmeans vs tgc_basic vs weighted_voronoi
#   -> one run, one simulation per algorithm, each with plan.json + Monte-Carlo results.json
python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/default.yaml --base runs

# Fine-grained fleet-size sweep with the empirical break-even (weighted-TGC vs k-means)
python -m uav_swarm_sim.experiments.run_scale_tiers --n-range 2 100 4 --out runs/sweep

# Optimal fleet size for THIS area/battery (analytical Pareto knee, no simulation)
python -m uav_swarm_sim.experiments.run_fleet_sizing_analyzer --config config/default.yaml --n-max 20 --plot runs/fleet.png

# Dubins vs discretized grid (FIXED_WING / VTOL only)
python -m uav_swarm_sim.experiments.run_kinematics_comparison --config config/default.yaml --out runs/kinematics

# Launch-site optimization study + suitability heatmap
python -m uav_swarm_sim.experiments.run_launch_site_study --config config/default.yaml --out runs/launch

# Reproduce one replication as a replay GIF
python -m uav_swarm_sim.experiments.run_replay --config config/scenarios/smoke.yaml --replication 0 --out runs/replay

# LLM-as-a-judge mission diagnosis over a telemetry log (offline; --dry-run is free, see §12)
python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --dry-run

# Shape study preflight (see "Shape study preflight" below):
#   (A3) generate the equal-area shape family + print the shape-descriptor table
python -m uav_swarm_sim.experiments.generate_shapes --out-dir data/areas/shapes
#   (A2) is this (shape, fleet, battery) point battery-limited or fuel-surplus?
python -m uav_swarm_sim.experiments.run_regime_calculator \
    --geojson data/areas/shapes/square.geojson --n-drones 5 --verify
```

Smoke test: `pytest tests/test_smoke.py`. Full suite: `pytest -q`.

### How many drones is optimal?

There is **no single number** — the optimum depends on the survey area, the battery capacity, and the platform. Two tools answer it:

- **`run_fleet_sizing_analyzer`** computes the analytical **Pareto knee**: for your config it prints, per fleet size, the expected mission duration and the *marginal* time each added drone saves, and marks the **knee** — the smallest fleet beyond which each extra drone saves less than `--knee-frac` (default 5%) of the mission time. That knee is the practical "optimal" fleet size: more drones still finish faster, but with diminishing returns and a larger shared battery pool to carry. This is a fast, closed-form analysis (no simulation).
- **`run_scale_tiers`** then shows, across the fleet-size grid, the point at which the **battery-weighted TGC overtakes the position k-means baseline** — the empirical break-even that motivates the three-tier policy.

The default config ships `n_drones: 5`, which sits in the **heuristic tier** (≤15). The `tier_thresholds: [15, 50]` partition the operating regime: **≤15** drones use the coupled k-means heuristic, **16–49** compare both methods (this is where the contribution earns its keep), and **≥50** use the TGC decomposition. So for a small reconnaissance team the heuristic is fine; the battery-weighted TGC contribution matters most at medium-to-large fleet sizes. Run the analyzer on your real area to get the concrete knee for the defense.

### A note on running at defense scale

Each Monte-Carlo batch runs up to `n_max` missions (adaptive CI stopping usually stops far earlier). The decomposition comparison is four batches; the scale sweep is two methods per fleet size. This can take a while on real configs — start coarse (a few fleet sizes, or a lower `n_max`) to sanity-check, then run the full grid for the final figures.

### Shape study preflight (A2 + A3): is the baseline even in the regime where shape matters?

The next planned study asks a focused question: **does the shape of the survey area change how much the battery-weighted decomposition wins?** Before spending hours on shape sweeps, two standalone tools settle a prerequisite that, if ignored, invalidates the whole study. They do **not** simulate anything expensive — they are fast, analytical, and each answers one question.

#### Why this preflight exists

The weighted decomposition only has anything to *do* when drones are under **energy pressure** — i.e. when a drone cannot cover its assigned zone on a single battery and must land, swap, and continue. That regime is called **battery-limited**. If instead every drone can comfortably finish its zone on one charge (the **fuel-surplus** regime), there are no swaps to rebalance, the weighted method and the position-based baselines behave almost identically, and any shape sweep comes out **flat** — proving nothing. So the first thing to establish, for a given (shape, fleet size, battery), is *which regime you are in*. That is the go/no-go these tools produce.

#### A3 — the equal-area shape family (`generate_shapes.py`)

To attribute an effect to **shape**, every test shape must have the **same area** — otherwise a bigger area needs more energy and you cannot tell shape from size apart. `generate_shapes.py` builds a family of shapes all normalized to **exactly** the same area (default 1 km², to within `1e-6` relative error): a `square`, three rectangles (`2:1`, `4:1`, `8:1`), an `l_shape`, a `disk`, a 5-point `star`, and a `pinwheel`. Each is written as a metric-metre GeoJSON that the normal parser (`planning/geojson_parser.load_area`) reads unchanged, so the shapes drop straight into the rest of the pipeline.

For each shape it also prints a **descriptor row** — two numbers that quantify "how awkward is this shape", both intuitive once unpacked:

- **Solidity = area ÷ convex-hull area.** The convex hull is the shape you'd get by stretching a rubber band around the outline. A convex shape (square, disk, rectangle) fills its own hull completely, so solidity **= 1.0**. A shape with dents or arms (L, star, pinwheel) leaves gaps between itself and the rubber band, so solidity **< 1.0** — the lower the number, the more concave. This is the study's headline concavity variable (hypothesis **H5**): `l_shape ≈ 0.86` > `star ≈ 0.49` > `pinwheel ≈ 0.40`.
- **Isoperimetric ratio = perimeter² ÷ (4π·area).** A pure circle is the most compact possible shape and scores **1.0**; everything more elongated or more jagged scores higher (more perimeter for the same area). It rises with both elongation (`8:1` rectangle ≈ 3.2) and raggedness (pinwheel ≈ 4.4).

A third figure, **`strips≈`**, estimates how many parallel back-and-forth coverage lanes (the boustrophedon sweep) the shape needs = short side of its tightest bounding rectangle ÷ the effective swath width. It is a quick proxy for coverage effort.

```bash
python -m uav_swarm_sim.experiments.generate_shapes --out-dir data/areas/shapes
# prints the descriptor table and writes 8 equal-area GeoJSONs
```

#### A2 — the regime calculator (`run_regime_calculator.py`)

This is the go/no-go. It compares the **energy needed to cover the area once** against the **energy the fleet actually has**, and reports which regime you are in. It runs no long simulation; its one optional `--verify` step runs a single short mission to prove its arithmetic matches the engine.

**1. `E_cover` — the energy to photograph the whole area once.** This is not a hand-tuned formula. The tool builds the *same* boustrophedon coverage plan the simulator flies, rebuilds the drone's legs exactly as the execution engine does (even leg = a `COVERAGE` strip with the camera on, odd leg = a `TURN` connector between strips), and integrates each leg with the identical `power × dt` energy model. Because the analytical construction mirrors the executed physics leg-for-leg, its total is the exact value the engine's tick-by-tick sum converges to — verified below. `E_cover` is reported as a breakdown:

| Term | What it is |
|---|---|
| **Coverage strips** | Propulsion along the survey lanes (the dominant, shape-driven cost) |
| **Inter-strip connectors** | The `TURN` hops linking one lane to the next |
| **Camera payload** | Sensor/gimbal energy, charged **only** while photographing (`COVERAGE` segments); zero unless `--sensor-power-w` > 0 |
| **Round-trip transit** | Cruising from the launch site to the area and back |
| **Vertical takeoff+landing** | The climb/descent budget. **Flagged:** this is a *reserve/budget* term — in the single-layer engine it is **not** actually drained, so it is reported but sits slightly apart from the flown energy. |

**2. `B_usable` — the energy one drone may actually spend.** A battery is never drained to zero; a reserve floor is kept. Three floors are printed (all read from config, none hard-coded), and the one used is chosen with `--usable-floor`:

- `terminal` *(default)* = down to the terminal reserve (`1 − critical`, i.e. 0.80 of capacity);
- `return` = down to the level that forces the drone home in the executed sim (`1 − nominal`, 0.60);
- `rth` = the hard epsilon floor only (`1 − reserve_frac`, 0.95).

> **Thesis-affecting choice.** The floor changes `B_usable`, hence the regime and the crossover fleet size. It is flagged in the output; pick one deliberately and state it in the thesis. The default `terminal` is the conservative battery-zone reserve.

**3. Two tests, because batteries are not shared.** The tool reports two ratios, and this distinction is the important subtlety:

- **Pooled ratio** = `E_cover ÷ (n · B_usable)`. This treats the whole fleet's energy as one shared tank. If it is **> 1**, the total job exceeds the total fuel, so *by counting alone* at least one drone must swap → **definitely battery-limited**. But if it is **< 1**, that does **not** prove fuel-surplus: it only means the fleet has enough energy *on average*. Averages hide imbalance.
- **Per-drone (assignment-aware) check.** In reality each drone is handed **its own zone** and can only spend **its own** battery on it. So the tool builds the real weighted-Voronoi partition for that fleet, computes each zone's true `E_cover`, and checks the **busiest** drone against one battery. If that single worst zone exceeds `B_usable`, that drone must swap — and the mission is battery-limited **even if the pooled ratio said surplus**.

The final regime is **battery-limited if *either* test trips** (`pooled > 1` **or** `max-zone > 1`). Worked example on the `pinwheel` at `n = 4`: the pooled ratio is `0.92` (which the naive one-tank view would wave through as "surplus"), but the most awkward arm of the pinwheel gives the busiest drone a zone costing `1.31 × B_usable` — so it is correctly flagged **battery-limited**. That gap is precisely the shape-and-imbalance effect the whole study is about, and why the pooled ratio alone is not trusted for the go/no-go.

The tool also prints the per-drone zone table and the **zone imbalance** (busiest ÷ least-busy zone energy) — a direct readout of how unevenly a given shape forces the work to be split.

**4. `--verify` — proof the arithmetic is real.** With `--verify` the tool runs one actual short mission (obstacles and random failures off, multirotor) and compares the energy the engine *actually drained* during the coverage phase against its own analytical number. They agree to under ~1% at the default timestep (the residual is just the engine charging a full `power × dt` on each leg's last partial tick, and it shrinks as the timestep shrinks). The check only counts as clean when the run had **zero battery swaps** — a mid-mission swap makes the engine re-fly the interrupted lane and double-count, so the verifier automatically picks the smallest swap-free fleet, and a run that still swaps is reported as **not clean** rather than a false pass.

```bash
# default 1 km² square, fleet of 5, with the engine cross-check
python -m uav_swarm_sim.experiments.run_regime_calculator \
    --geojson data/areas/shapes/square.geojson --n-drones 5 --verify
```

**How to read the result.** The report ends with a plain **Go / No-Go**:

- **GO (battery-limited)** → swaps are needed, shape and weighting both matter, run the sweep. Sweep fleet sizes that straddle the printed crossover `n*` so the sweep actually crosses the regime boundary.
- **NO-GO (fuel-surplus)** → the sweep would be flat. The tool lists how to cross into the interesting regime: use a smaller fleet, a smaller battery, a bigger/thinner area, or turn the camera payload on.

For reference, the shipped 1 km² `square` with the default 100 Wh battery is **fuel-surplus at `n = 5`** (pooled `0.53`, busiest drone `0.76`), with crossover `n* = 3` — so a meaningful sweep on that baseline must include the smaller, battery-limited fleet sizes (roughly `n ≤ 2`), or raise `E_cover` some other way. Establishing that number *is* the reason this tool runs first.

---

## 12. Automated mission diagnosis (LLM as a judge)

An LLM-as-a-judge reads the telemetry JSONL and produces a structured, **grounded** mission diagnosis (outcome classification, a single root cause from a fixed taxonomy, a causal narrative, contributing factors, a critical-event timeline, recommendations). The logic lives in `metrics/llm_judge.py`; the CLI is `experiments/run_llm_diagnosis.py`.

Three commitments make it an evaluation device, not a chatbot:

- **Deterministic grounding.** Ground-truth facts (outcome, terminal trigger, which drone died, per-drone minimum battery, swap/obstacle counts) are extracted from the log **before** any model call, handed to the model, and used afterward to **verify its claims** — mismatches are flagged in a grounding audit rather than trusted.
- **Fixed taxonomy.** The root cause must come from a closed set aligned to the simulation's terminal physics, so verdicts are machine-comparable across runs.
- **Injectable, model-agnostic.** The judge takes a `model` callable; an Anthropic adapter is provided (lazily imported). The module imports with no third-party dependency, and the full pipeline (facts → prompt → parse → ground → render) is unit-testable offline.

```bash
# FREE: deterministic facts + the exact prompt, no API call
python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --dry-run
# LIVE: writes a Markdown report (needs ANTHROPIC_API_KEY + the anthropic package)
python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --out runs/demo/diagnosis.md
```

For human-readable interpretation with the **figures** in hand, the repo ships a **mission-analyst system prompt** (`mission_analyst_prompt.md`): paste it as the system prompt, attach the run's `events.jsonl` + config + figures, and ask "analyze this run." It reads plots as well as logs and is grounding-disciplined (cites the figure/drone/timestamp behind each claim).

---

## 13. Mapping thesis claims to code, and reconciliation

### Thesis-claim → module (the explainability invariant)

Every quantitative claim maps to exactly one place, and the state/maneuver/algorithm names are identical across the simulation, metrics, plots, and thesis text (single home: `infrastructure/enums.py`).

| Thesis claim | Module(s) |
|---|---|
| Continuous component (grey-box) energy, `E = Σ P(maneuver)·dt`; no distance-averages | `physical_model/energy_model.py` |
| Kinematically realistic trajectories (true flyable length) | `physical_model/dubins.py`, `physical_model/motion_model.py` |
| Grid comparison planner (planning-speed vs kinematic-accuracy) | `planning/grid_planner.py` + `experiments/run_kinematics_comparison.py` |
| 15.14% formation drag benefit, scope-limited to cruise on FW/VTOL | `physical_model/aero_correction.py` + `execution/formation_manager.py` |
| 2.5D vertical climb/descent; mass couples in only via `m·g·dz` on climb | `physical_model/vertical_segments.py` + `physical_model/energy_model.py` |
| GeoJSON ingest; synthetic full-height obstacle field; safe corridor / clearance | `planning/geojson_parser.py`, `planning/obstacle_generator.py`, `planning/environment_map.py`, `planning/gvg_builder.py` |
| Topological free-space regions (atomic decomposition units) | `planning/tgc.py` |
| **Area ∝ momentary battery — the central contribution** | `planning/weighted_decomposition.py` (with `tgc_basic` as its switched-off ablation twin) |
| Position-based baselines: Euclidean Voronoi, position k-means | `planning/classic_voronoi.py`, `planning/kmeans_heuristic.py` |
| Launch site as an optimization variable (3 criteria, staging ring outside the polygon) | `planning/launch_site_optimizer.py` |
| Boustrophedon coverage paths; multi-layer planning; leg repair; target tours | `planning/coverage_path.py`, `planning/layer_planner.py`, `planning/trajectory_validation.py`, `planning/target_mission.py` |
| Seven-state automaton; agent/fleet; dynamic RTH (route-vs-return, pre-empts) | `execution/state_machine.py`, `execution/agent.py`, `execution/fleet.py`, `execution/rth_calculator.py` |
| Event-driven redistribution; swap ≠ failure; three-tier selection; proactive avoidance; finite swap pool; hazard failure | `execution/redistribution.py`, `execution/events.py`, `execution/algorithm_selector.py`, `execution/safety_monitor.py`, `execution/swap_station.py`, `execution/failure_model.py` |
| Semi-Markov (battery = hidden memory); stationary `π` with embedded→time-weighted correction; efficiency = `π(S2)/(π(S3)+π(S_OBS)+π(S_SWAP))` | `metrics/smdp_estimator.py`, `metrics/stationary_distribution.py`, `metrics/efficiency_score.py` |
| Monte Carlo with CI-based convergence; empirical break-even; internal validation | `metrics/monte_carlo.py`, `metrics/convergence.py`, `metrics/comparison.py`, `metrics/validation.py` |
| Reproducible content-addressed RNG (paired-seed Monte Carlo); config + `config_hash` | `infrastructure/rng.py`, `infrastructure/config.py` |
| Structured run output (plan/results/manifest); GPX + JSONL; grounded LLM judge | `metrics/run_output.py`, `metrics/gpx_exporter.py`, `metrics/llm_log_exporter.py`, `metrics/llm_judge.py` |
| Shape study preflight: equal-area shape family (solidity / isoperimetric descriptors) | `experiments/generate_shapes.py` |
| Shape study preflight: battery-limited vs fuel-surplus regime gate (`E_cover` vs pooled `n·B_usable` **and** the per-drone assignment-aware check), analytical `E_cover` verified against the engine | `experiments/run_regime_calculator.py` |

### Status and where the spec is ahead of the written text

The implementation and its hardening are complete and the regression suite is green. A few items remain for the **written Part II text** (the code is settled):

1. **The SMDP / stochastic-analysis layer is not yet described in Part II** (guideline 1.4 currently promises only the three deterministic metrics). Recommended additions: the semi-Markov justification (battery as hidden memory → `(state, sojourn time)`); `π` as the time-fraction metric valid only under the closed loop; the embedded→time-weighted correction; the efficiency score and its throughput-oriented denominator.
2. **The physical-vs-analytical failure treatment** (§5) deserves an explicit sentence distinguishing the irreversible physical failure from the analytical slot-replacement closure.
3. **The battery-zone thresholds** (HIGH ≥75 / NOMINAL ≥40 / CRITICAL ≥20 / TERMINAL <20) and the **launch-site three-criterion objective** are modeling parameters that should be introduced in the text if cited as results.

Design decisions taken during the build, now reflected in the code: external-literature reproduction was **dropped** (no access to the anchor papers' datasets) in favour of **internal validation** and paired-seed comparative claims (`metrics/validation.py`); **k-means was promoted to a first-class comparison peer** (§7); a **fine-grained adaptive scale sweep** with CI-based stopping locates the empirical break-even, while the three-tier policy remains the *operational* selector informed by it. The **shape-study preflight tooling** (equal-area shape generator + regime calculator, §11) is in place ahead of the written text; the shape sweep itself, and hypothesis **H5** (shape effect governed by solidity, not raw concavity), remain to be run and written up.

> **Not a study: coverage altitude.** Because obstacles are full-height prisms (always taller than the mission altitude) and the mission is flown at constant altitude, flight altitude does not change which obstacles must be avoided — it only changes one-time climb energy. There is therefore no interior altitude optimum to study; the 2.5D framing here is about modeling the vertical takeoff/RTH segments separately from horizontal coverage, not about optimizing the coverage altitude.

---

## 14. Scope boundaries

Explicitly **out of scope** (named in the thesis as context, not required by the guidelines): wind-field modeling; communication/network *modeling* (`viz.show_comm_range` draws a circle for readability only — it does **not** model link budgets); the 3-D Dubins-airplane extension (excluded by the constant-altitude assumption); real flight-data regression (no physical experiments); and learning-based planners (MARL/DRL). The architecture isolates each so that adding one later touches a single module.

---

## 15. The thesis goal

**Title.** *Optimising flight missions between identical reconnaissance drones* (MSc, Vilnius Gediminas Technical University, Antanas Gustaitis Aviation Institute).

**Problem.** A homogeneous fleet of reconnaissance UAVs must cover a bounded survey area, in the presence of obstacles, under hard energy limits. Identical drones do **not** stay identical in operation: their **momentary battery levels diverge** as the mission unfolds (different transit distances, different obstacle detours, swaps at different times). Classical area-partitioning methods — equal-area splits, Euclidean Voronoi, position-based k-means — ignore this and hand a depleted drone the same workload as a full one, which forces premature returns, extra battery swaps, and longer makespans.

**Aim.** To design and computationally validate a mission-optimization method that allocates coverage work **in proportion to each drone's momentary battery level**, on a topological representation of the free space, so that the fleet's energy is spent more evenly and the mission completes more efficiently than under position-based baselines.

**Central contribution.** An **energy-weighted spatial decomposition**: the survey area is reduced to a **Generalized Voronoi Graph (GVG)** and a **Topological Graph Construction (TGC)** of free-space regions and safe corridors, and those regions are assigned to drones with each drone's share **weighted by its current battery state** rather than by position alone. This is the `weighted_voronoi` algorithm; its position-only twin (`tgc_basic`) and the position-based baselines (`classic_voronoi`, `kmeans`) are its controls.

**Method of validation.** Because no physical flight experiments were performed, the claim is established **comparatively and stochastically**:
- a **grey-box component energy model** (per-maneuver power × time) and **kinematically realistic** Dubins/holonomic trajectories give physically meaningful relative energies;
- every method is run on **paired identical random seeds** through one pipeline, so systematic modeling error cancels in the differences;
- the swarm's long-run behavior is analyzed as a **Semi-Markov Decision Process** — the battery acts as hidden memory, so the state is `(behavioral state, sojourn time)` — yielding a **stationary distribution** over states and an **efficiency score** (the time-fraction productively surveying versus lost to return, avoidance, and swapping);
- results are taken to **statistical convergence** by Monte Carlo with a confidence-interval stopping rule, and the **scale at which the battery-weighted method overtakes** the position-based heuristic is located empirically.

**Hypothesis under test.** Weighting the decomposition by momentary battery level produces a **more balanced workload, fewer battery swaps, and a higher surveying-efficiency** than position-based decomposition for identical drones — with the advantage growing as the fleet and the area scale up.

**What the artifact delivers for the defense.** A reproducible simulator that, for any survey polygon and fleet, produces the comparison figures and numbers (workload balance, energy, swap counts, SMDP efficiency, the empirical break-even fleet size), each traceable to a single module and emitted into a self-describing, comparable run folder — so every claim in the written thesis can be regenerated and audited on demand.
