# UAV Swarm Reconnaissance Simulation (2.5D Layered Architecture)

A modular, explainable simulation of **coordinated mission optimization for a homogeneous
reconnaissance UAV swarm**. It is the computational artifact for the master's thesis
*Optimising flight missions between identical reconnaissance drones* (Konstantin Belena,
Vilnius Gediminas Technical University, Antanas Gustaitis Aviation Institute).

The simulation operationalizes the thesis's central methodological contribution —
**energy-weighted spatial decomposition by momentary battery level on a topological-graph
(TGC/GVG) framework for identical drones in obstacle environments**.

The system operates in a **2.5D layered environment**: coverage planning stays strictly 2D
at a fixed mission altitude, connected by physics-accurate vertical climb and descent
segments. The mission itself is flown at constant altitude — the drones do not dive between
layers during coverage.

> **Where things live.** This README is the **front page**: the map and the invariants.
> **`docs/cli_map.md`** is the **reference**: every script, every flag, every config key.
> **`docs/archive/PROJECT_HISTORY.md`** is the **record**: every measured result, falsified
> hypothesis, bug fix and methodological rule from the work that preceded the current
> experiment spec. A fact lives in exactly one of the three.

---

## 1. What this project is

A discrete-time, Monte-Carlo simulation that, for a fleet of identical reconnaissance drones:

1. ingests an exploration-area boundary (GeoJSON) and a synthetic obstacle field (full-height
   prisms — fixed, immovable, taller than the mission altitude, so they block at every
   altitude);
2. chooses an energy-aware launch site strictly from a safe staging ring **outside** the
   survey polygon (a decision variable, not an assumption);
3. partitions the area among the drones so each drone's region is proportional to its
   **momentary battery level**;
4. enforces fleet logistics: a finite **shared battery pool** and exact battery-swap cycle
   counting;
5. flies kinematically realistic coverage paths under an **eight-state behavioural
   automaton** (§4) with a dynamic return-to-home rule, event-driven redistribution,
   proactive obstacle avoidance, and rigorous **terminal-state** evaluation;
6. measures the result deterministically (energy, duration, workload balance, swap metrics)
   and stochastically (stationary distribution over states, efficiency score), with
   statistical convergence;
7. writes every run into a **structured, self-describing run folder** (§9) so any run is
   reproducible, comparable and analyzable after the fact.

**Energy is always `E = Σ P(maneuver)·dt`** — never a per-distance shortcut. Many design
choices exist only to protect that invariant.

---

## 2. Architecture: layers mirroring the methodology

| Package | Thesis layer | Responsibility |
|---|---|---|
| `physical_model/` | **Physical model** | Grey-box component energy model; aerodynamic formation correction; Dubins kinematics; the `MotionModel` platform abstraction; 1-D vertical takeoff/landing and inter-layer climb segments. |
| `planning/` | **Planning** | GeoJSON parsing; Poisson prism obstacle generation; layered environment mapping; GVG + TGC construction; the weighted decomposition (central contribution) and its three position-based baselines; the energy cost-to-go map; Dijkstra cost database; launch-site optimizer; boustrophedon coverage paths; the grid comparative planner; pre-flight trajectory validation. |
| `execution/` | **Execution** | The eight-state automaton; the agent and fleet; the event bus; dynamic RTH calculator; event-driven redistribution; proactive safety monitor; formation manager; battery-swap station with finite shared pool; hazard-rate failure model; three-tier scale-dependent selection. |
| `metrics/` | **Analysis (SMDP)** | State-history recording; deterministic mission metrics; SMDP estimation; the stationary distribution with the embedded→time-weighted correction; the efficiency score; the Monte-Carlo runner with CI-based convergence; the structured run-output writer; telemetry with GPX + JSONL exporters. |
| `infrastructure/` | *(supporting)* | Typed configuration and `config_hash`; reproducible content-addressed RNG; the `SimulationEngine` orchestrator; all visualization. |
| `experiments/` | *(supporting)* | Thin CLI entry points that compose the layers into experiments. |

Lower layers never import higher ones. The single conductor is
`infrastructure/simulation_engine.py`.

### How a single mission flows through the layers

```
config/*.yaml
   │  (infrastructure.config: typed, validated, unit-normalized; config_hash)
   ▼
EnvironmentMap (full-height prisms) → LayerStack (mission-altitude plane)   [planning]
   ▼
GVG  →  TGC (corridors + free-space regions + adjacency)                    [planning]
   ▼
LaunchSiteOptimizer  →  launch site (staging ring outside the polygon)      [planning]
   ▼
Decomposer (tier-selected): area ∝ momentary battery level                  [planning]
   ▼
CoveragePath per zone (boustrophedon + smoothing + leg validation)          [planning]
   ▼
SimulationEngine dt-loop (fail-fast):                                 [infrastructure]
   fleet init (circular deployment) → failure → safety → agents step
   (motion + energy + state + RTH) → swap station (shared pool)
   → drain events → redistribution → terminal evaluation
   ▼
StateHistory + MissionMetrics                                              [metrics]
   ▼
SMDP estimate → stationary π (embedded → time-weighted) → efficiency score  [metrics]
   ▼
Structured run folder: plan.json + results.json + figures + GIF + GPX       [metrics]
```

The single most important computational subtlety is in
`metrics/stationary_distribution.py`: the left-eigenvector of the embedded transition matrix
yields **visit frequencies**, not time fractions. The code multiplies each embedded `π[i]` by
the mean sojourn time `m_i` and renormalizes to obtain the **time-weighted** `π` the
efficiency score requires. Both are always plotted side by side so the correction is visible
rather than implicit.

---

## 3. Platform support and the energy-coefficient caveat

`platform_type` is a configuration enum; each run simulates **one** homogeneous platform:

- **`FIXED_WING`** — `DubinsModel` kinematics (minimum turn radius); formation flight yields
  a **drag/energy benefit** (≈15.14 %, Guo et al. 2025), applied only during launch,
  transit-to-zone, and episodic RTH — never during dispersed coverage.
- **`MULTIROTOR`** — `HolonomicModel` kinematics (in-place turns, Euclidean legs); formation
  downwash is a **safety constraint** (wake zones as invisible obstacles), **not** an energy
  benefit.
- **`VTOL`** — `DubinsModel` for the horizontal coverage phase, bracketed by 1-D vertical
  climb/descent segments. Cruise gets the formation benefit; vertical phases treat downwash
  as a constraint.

> **⚠ Energy-coefficient caveat (read before trusting absolute numbers).**
> No physical flight experiments were conducted. Motor power coefficients are **theoretical
> approximations from typical platform specifications** following Steup et al.'s component
> method. The **quadrotor** (`MULTIROTOR`) coefficients are closest to the validated Steup
> baseline. The **`FIXED_WING` and `VTOL`** tables are **coarser extrapolations** and should
> be read as *relative*, structurally-consistent estimates — not calibrated absolute
> energies. All comparative results are computed on **paired identical seeds**, so
> conclusions about *differences* are robust even where absolute energies are approximate.

---

## 4. Terminal states and the behavioural automaton

The simulation enforces strict, mutually-exclusive terminal outcomes, evaluated once per tick
inside `SimulationEngine.run()`, preventing unrealistic "zombie" computations. The verdict is
carried on `MissionResult` as an `Outcome`:

- **`MISSION_SUCCESS`** — the partitioned area is covered AND every surviving drone has
  returned to `S0_IDLE`.
- **`MISSION_FAILED`** — a physics-dictated halt, evaluated **fail-fast, before every other
  outcome**: an **airborne drone's battery reaches 0**, or the **shared battery pool is
  exhausted** before coverage completes. Hazard-induced `S_FAIL` does **not** trigger this —
  those failures populate `S_FAIL` for the elevated-hazard Monte-Carlo statistics, and the
  run continues via redistribution.
- **`MISSION_PARTIAL`** — every surviving drone finished every leg it did not forfeit and is
  idle, but ≥ 1 coverage strip was skipped as unreachable (`safety.stall_skip`). Checked
  **before** success, so a forfeited strip can never be classified `MISSION_SUCCESS`.
- **`MISSION_INCOMPLETE`** — neither terminal condition fired before the run ended (e.g. the
  `sim.max_timesteps` ceiling). The default outcome.

**Circular deployment footprint:** drones are instantiated in a computed ring around the
launch pose at `t = 0`, preventing artificial collisions and immediate `S_OBS` deadlocks.

### The eight-state automaton

Each drone runs an eight-state FSM (`infrastructure/enums.py`):

| State | Meaning | Airborne? | Energy |
|---|---|---|---|
| `S0_IDLE` | On the ground at the launch site | no | idle draw only |
| `S1_TRANSIT` | Flying to/from the assigned zone | yes | flight |
| `S2_MISSION` | Sweeping a coverage strip, **camera ON — the only productive state** | yes | flight + camera |
| `S3_RTH` | Returning to home | yes | flight |
| `S_SWAP` | Battery swap at the ground station | no (landed) | **zero energy**, costs time |
| `S_OBS` | Obstacle-avoidance maneuver | yes | flight |
| `S_FAIL` | Lost — removed from the active fleet | no | none |
| `S_FERRY` | Repositioning between coverage strips, **camera OFF** | yes | flight |

**On `S_FERRY`:** it is the camera-off repositioning leg between coverage strips —
non-productive, but **airborne**, so it consumes flight energy and carries failure-hazard
exposure (`AgentState.is_airborne`). Structurally it is the odd-parity connector: coverage
legs are boustrophedon, even `_cov_idx` = COVERAGE (`S2_MISSION`), odd = TURN connector
(`S_FERRY`). Note that `S_FERRY` is at the centre of a documented open discrepancy — see
`docs/archive/PROJECT_HISTORY.md` §7.1–§7.2.

**Coverage area is not the flyable area** — a drone may fly outside the survey polygon
whenever it is not in `S2_MISSION`.

### Return-to-home priority

Inside `S2_MISSION` the return triggers are evaluated in a deliberate priority order (first
match wins):

```
obstacle threat  >  dynamic RTH (rth_energy)  >  CRITICAL battery  >  TERMINAL battery
                 >  coverage complete
```

> **The static RTH threshold is 0.40, not 0.20.** `Battery.zone` classifies
> `critical ≤ f < nominal` as `CRITICAL`, i.e. `[0.20, 0.40)`, so the `critical_battery`
> guard fires at **nominal = 0.40**; `0.20` is merely where `TERMINAL` begins. Under
> `rth.energy_map.zone_demotion` the `CRITICAL` branch is removed entirely and the dynamic
> cost-to-go map governs, with `TERMINAL` as the sole failsafe.

---

## 5. The S_FAIL dual view (physical layer vs. analysis layer)

Failure is modeled **differently and deliberately** in two separate layers. This is a
conscious modeling decision, documented here and in `metrics/smdp_estimator.py`:

- **Physical simulation layer (thesis-faithful, irreversible).** When a drone fails,
  `execution/fleet.py::kill` removes it permanently and `execution/redistribution.py`
  immediately re-partitions the **uncovered** work among the **surviving active** agents. The
  failed drone does not return. A battery-depletion failure of an airborne drone halts the
  run with `MISSION_FAILED`; a hazard-induced failure (λ > 0) removes the agent and triggers
  redistribution without halting.

- **SMDP analysis layer (ergodicity device).** The stationary distribution `π = πP` exists
  only if the embedded Markov chain is **ergodic**. A terminal `S_FAIL` is absorbing and
  would make `π` undefined. The estimator therefore models a **generic agent-slot**: a failed
  slot is closed at a configurable mean repair/replacement time and given a synthetic
  transition `S_FAIL → S0`. This closes the loop alongside the genuine
  `S3 → S_SWAP → S0` swap loop.

This is **not** a claim that the physical swarm self-heals. It is the standard
renewal-theoretic treatment that lets us speak of long-run time-fractions per state for a
continuously-operated slot. The estimator exposes `close_failure_loop` (default `True`); with
`False` it refuses to compute `π` and reports `ergodic=False` — the correct behavior for a
literal absorbing-failure reading.

Contrast with **battery swap**, which is reversible and does *not* trigger redistribution:
the swapped drone resumes its own remaining plan. The swap/failure asymmetry is enforced by
wiring (only `FAILURE` and `NEW_TASK` reach the redistributor), not by convention.

---

## 6. Configuration

`config/default.yaml` mirrors the typed schema in `infrastructure/config.py` one-to-one.
**`load_config` reads exactly ONE YAML — there is no merge or inheritance between files.**
Optional blocks (`telemetry`, `coverage`, `rth.energy_map`, `safety.obstacle_recovery`, …)
fall back to dataclass defaults when absent. Units in the YAML are Wh and degrees where
noted; the loader converts to SI (Wh→J, deg→rad) exactly once.

| File | Purpose |
|---|---|
| `config/default.yaml` | The reference config; mirrors the schema 1:1 |
| `config/djimatrice4e.yaml` | DJI Matrice 4E platform (99.5 Wh, swath 132 m, square obstacles) |
| `config/scenarios/smoke.yaml` | Minimal smoke config (n=3) — fast tests only |
| `config/study01_demand.yaml` | **Frozen test fixture** — see its header; not experiment evidence |

> **`config_hash`** is computed from the merged config *after* CLI overrides but *before*
> unit/enum transformation, so it is a genuine provenance fingerprint. Comments do not
> affect it; absent optional blocks do not change it.

**Every config key, its default and its meaning: `docs/cli_map.md` §2.**

---

## 7. Decomposition algorithms (the comparison axis)

`DecompositionAlgo` has four **first-class, paired-seed** peers — three position-based
baselines plus the contribution:

| Algorithm | Kind | Module |
|---|---|---|
| `classic_voronoi` | Plain nearest-seed Euclidean Voronoi (position-based) | `planning/classic_voronoi.py` |
| `kmeans` | Position k-means (position-based) | `planning/kmeans_heuristic.py` |
| `tgc_basic` | Unweighted topological (position-based) | `planning/weighted_decomposition.py` (ablation twin) |
| `weighted_voronoi` | **Battery-weighted TGC — the central contribution** | `planning/weighted_decomposition.py` |

All four run through the *identical* pipeline on the *same* per-replication seeds, so any
metric difference is attributable to the algorithm, not the noise.

> **THE CRITICAL NULL.** For a homogeneous fleet (identical drones, `battery_frac = 1.0`)
> with λ = 0, **`weighted_voronoi ≡ tgc_basic` byte-identically**. Equal battery fractions
> produce an identical partition *by construction*. The weighting differentiates **only**
> when batteries have diverged — a heterogeneous fleet, or a post-failure redistribution at
> λ > 0. A clean full-battery run reproducing this exact zero is a **correctness check, not a
> failure**.

---

## 8. Structured run output — where results live

Every experiment writes into one self-describing, comparable **run folder**. A run holds one
or more **simulations** (e.g. one per decomposition algorithm); each owns its artifacts and
two JSON logs:

```
runs/run-2026-06-28-11-59-35/          ← a RUN  (name = the dated folder, id = a GUID)
  run.json                             ← manifest: identity, software/git commit, timing
  simulation-weighted_voronoi/         ← a SIMULATION within the run
    plan.json                          ← the launch PLAN (every input/setup)
    results.json                       ← the OUTCOME (success rate, SMDP, MC logic, timing)
    environment.png partition.png paths.png replay.gif state_gantt.png
    battery.png pi_bars.png smdp_convergence.png tracks_drone_*.gpx
  simulation-kmeans/ ...
```

Runs are identified and compared by `run_id` (GUID), `run_name` (date), and a per-simulation
`config_hash` (exact-input match). All JSON is strict-valid (non-finite values become
`null`), so it loads cleanly in `jq`, pandas, or any analysis tool.

`run_single_mission` additionally reports **SMDP convergence diagnostics** per state
(`metrics/smdp_convergence.py`): the raw visit count, a **Wilson 95 % CI** for every observed
transition probability, and a **weakest-state summary**. If the weakest state's CI is wide,
everything derived from the stationary distribution is anecdotal — run more replications
before quoting it.

---

## 9. Installation

Requires **Python 3.12+**.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e ".[dev]"                              # editable install + pytest-xdist
```

Run everything from the repository root — config files reference the map via relative paths.

```bash
pytest tests/integration/test_smoke.py               # fast smoke check
pytest -n logical --dist loadscope -q                # the full suite, exactly as CI runs it
pytest -m "not slow"                                 # fast local dev loop
```

`--dist loadscope` keeps a module's tests on one worker, so module-scoped fixtures are built
only once.

---

## 10. Running

Runs are **deterministic** given `(config, master_seed, replication, algorithm, planner)`.
Five representative entry points:

```bash
# One mission, full visual dump + GPX, into a structured run folder (the defense demo)
python -m uav_swarm_sim.experiments.run_single_mission --config config/default.yaml --name demo --base runs

# Headline comparison: classic_voronoi vs kmeans vs tgc_basic vs weighted_voronoi
python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/default.yaml --base runs

# Shape sweep: survey shape x fleet size x decomposition variant, paired seeds
python -m uav_swarm_sim.experiments.run_shape_sweep --config config/default.yaml --mode clean --budget full --jobs 4

# Scale experiment: area x obstacle count x fleet size
python -m uav_swarm_sim.experiments.run_area_obstacle_sweep \
    --areas 1,2,4,8,16 --densities 0,8 --n 2,4,6 --reps 20 --out runs/scale

# Analytical, no simulation: is this (shape, fleet, battery) battery-limited or fuel-surplus?
python -m uav_swarm_sim.experiments.run_regime_calculator \
    --geojson data/areas/shapes/square.geojson --n-drones 5 --verify
```

**Every script, every flag, its default and how to read its output: `docs/cli_map.md` §4.**

Useful universal flags: `--config`, `--out`/`--base`, `--run-name`, `--jobs` (byte-identical
to serial), `--resume`, `--profile`. There is **no** universal `--set key=value` — physics
parameters change only via YAML; CLI flags expose study axes only.

---

## 11. Mapping thesis claims to code (the explainability invariant)

Every quantitative claim maps to exactly one place, and the state/maneuver/algorithm names
are identical across the simulation, metrics, plots and thesis text (single home:
`infrastructure/enums.py`).

| Thesis claim | Module(s) |
|---|---|
| Continuous component (grey-box) energy, `E = Σ P(maneuver)·dt`; no distance-averages | `physical_model/energy_model.py` |
| Kinematically realistic trajectories (true flyable length) | `physical_model/dubins.py`, `physical_model/motion_model.py` |
| Grid comparison planner (planning-speed vs kinematic-accuracy) | `planning/grid_planner.py` |
| 15.14 % formation drag benefit, scope-limited to cruise on FW/VTOL | `physical_model/aero_correction.py` + `execution/formation_manager.py` |
| 2.5D vertical climb/descent; mass couples in only via `m·g·dz` on climb | `physical_model/vertical_segments.py` + `physical_model/energy_model.py` |
| GeoJSON ingest; synthetic full-height obstacle field; clearance | `planning/geojson_parser.py`, `planning/obstacle_generator.py`, `planning/environment_map.py`, `planning/gvg_builder.py` |
| Topological free-space regions (atomic decomposition units) | `planning/tgc.py` |
| **Area ∝ momentary battery — the central contribution** | `planning/weighted_decomposition.py` (with `tgc_basic` as its switched-off ablation twin) |
| Position-based baselines: Euclidean Voronoi, position k-means | `planning/classic_voronoi.py`, `planning/kmeans_heuristic.py` |
| Launch site as an optimization variable (3 criteria, staging ring) | `planning/launch_site_optimizer.py` |
| Boustrophedon coverage paths; multi-layer planning; leg repair | `planning/coverage_path.py`, `planning/layer_planner.py`, `planning/trajectory_validation.py` |
| Eight-state automaton; agent/fleet; dynamic RTH (pre-empts the static net) | `execution/state_machine.py`, `execution/agent.py`, `execution/fleet.py`, `execution/rth_calculator.py` |
| **Dynamic RTH energy reserve** — per-replication Dijkstra cost-to-go map (`E_home` + parent pointer) driving the return decision, the return route and the resume transit | `planning/energy_map.py`, `execution/rth_calculator.py` |
| Event-driven redistribution; swap ≠ failure; three-tier selection; proactive avoidance; finite swap pool; hazard failure | `execution/redistribution.py`, `execution/events.py`, `execution/algorithm_selector.py`, `execution/safety_monitor.py`, `execution/swap_station.py`, `execution/failure_model.py` |
| Semi-Markov (battery = hidden memory); stationary `π` with embedded→time-weighted correction; the efficiency score | `metrics/smdp_estimator.py`, `metrics/stationary_distribution.py`, `metrics/efficiency_score.py` |
| Monte Carlo with CI-based convergence; empirical break-even; internal validation | `metrics/monte_carlo.py`, `metrics/convergence.py`, `metrics/comparison.py`, `metrics/validation.py` |
| Reproducible content-addressed RNG (paired-seed Monte Carlo); config + `config_hash` | `infrastructure/rng.py`, `infrastructure/config.py` |
| Structured run output (plan/results/manifest); GPX + JSONL | `metrics/run_output.py`, `metrics/gpx_exporter.py`, `metrics/llm_log_exporter.py` |

> **`efficiency` is the SMDP *throughput* ratio, NOT energy efficiency.** As implemented
> (`metrics/efficiency_score.py`) the denominator is
> `π(S1_TRANSIT) + π(S_FERRY) + π(S3_RTH) + π(S_OBS) + π(S_SWAP)` — every airborne or
> in-service state that is not productive coverage. Three retired documents specified a
> narrower three-state denominator; that discrepancy is unresolved and recorded in
> `docs/archive/PROJECT_HISTORY.md` §7.1.

---

## 12. The current experiment specification

*Supervisor-fixed. This supersedes every earlier scope statement in this repository; all
prior experimental work is method-development history and is recorded in
`docs/archive/PROJECT_HISTORY.md`.*

**Main experiment.** One identical multirotor type; baseline **5 UAVs**, with **3** and **8**
as additional checks; **~1000 × 750 m** area (larger variants: 2 × 2 km square, 3 × 1.5 km
elongated, 27 × 0.15 km strip); a **single photogrammetric coverage mission** at **100 m**
altitude, **10 m/s**, **80/70 % image overlap**; **~10 static obstacles covering ~5 % of the
area**; **no unlimited batteries and no battery swaps**; **classic Voronoi/Lloyd**
partitioning compared against **battery-remaining-weighted** partitioning; **≥ 30
replications** per scenario with different random initial conditions.

**Metrics, evaluated separately:** coverage · total energy · mission duration · minimum final
battery · workload balance · safety violations.

**Core contribution:** battery-state-aware work allocation + dynamic RTH energy reserve.
Launch-site optimisation is a separate optional comparison.

**Out of scope for the main experiment** until the baseline completes missions with no
collisions, no speed violations, no trajectory jumps and safe return of all UAVs: Dubins
curves, aerodynamic wakes, dynamic obstacles, Markov stationary distribution.

### Spec-vs-code gaps — as of commit `493e8d2`

The code does not yet meet the target above in **five** places. Full descriptions and
ownership live on the GitHub Project board **"mag"**; they are listed here only so the gap is
visible on the front page.

**OPEN DECISIONS — author only (CLAUDE.md working rule 7):**

- **Outcome semantics under "no battery swaps".** `total_reserve_batteries: 0` does not
  disable swaps; the first swap request trips `pool_exhausted` → `MISSION_FAILED`
  (`infrastructure/simulation_engine.py:660-663`), fail-fast before `MISSION_PARTIAL`.
- **Definition of "safety violations"** — separation breach, obstacle penetration, speed-cap
  violation? `execution/safety_monitor.py` reads `min_separation_m` to *trigger* avoidance
  but never counts breaches.

**WORK ITEMS — no semantics decision needed:**

- **Lloyd/centroidal relaxation is not implemented.** `planning/classic_voronoi.py` is plain
  nearest-seed Euclidean Voronoi.
- **Forward overlap is not modelled.** `physical_model/drone_specs.py:59` uses
  `effective_swath = swath_width_m * (1 - overlap_frac)` — a single scalar, side overlap only.
- **`MissionMetrics` has no minimum-final-battery field**
  (`metrics/mission_metrics.py:14-28`). The other four metrics exist.

---

## 13. Scope boundaries

Explicitly **out of scope** for the model itself: wind-field modeling; communication/network
modeling (`viz.show_comm_range` draws a circle for readability only — it does **not** model
link budgets); the 3-D Dubins-airplane extension (excluded by the constant-altitude
assumption); real flight-data regression (no physical experiments); and learning-based
planners (MARL/DRL). The architecture isolates each so that adding one later touches a single
module.

**Not a study: coverage altitude.** Because obstacles are full-height prisms and the mission
is flown at constant altitude, flight altitude does not change which obstacles must be
avoided — it only changes one-time climb energy. There is therefore no interior altitude
optimum to study; the 2.5D framing is about modeling the vertical segments separately from
horizontal coverage, not about optimizing the coverage altitude.

**One known model limitation with a hard consequence:** the energy model has **no turn/bank
aero-penalty term**. This is acceptable for the multirotor thesis platform, where turn energy
is small. It is **the trigger for any fixed-wing claim** — and because adding it re-baselines
energy, it must be done *before* any final re-runs or not at all. See
`docs/archive/PROJECT_HISTORY.md` §7.5.

---

## 14. The thesis goal

**Title.** *Optimising flight missions between identical reconnaissance drones* (MSc, Vilnius
Gediminas Technical University, Antanas Gustaitis Aviation Institute).

**Problem.** A homogeneous fleet of reconnaissance UAVs must cover a bounded survey area, in
the presence of obstacles, under hard energy limits. Identical drones do **not** stay
identical in operation: their **momentary battery levels diverge** as the mission unfolds
(different transit distances, different obstacle detours). Classical area-partitioning
methods — equal-area splits, Euclidean Voronoi, position-based k-means — ignore this and hand
a depleted drone the same workload as a full one, which forces premature returns and longer
makespans.

**Aim.** To design and computationally validate a mission-optimization method that allocates
coverage work **in proportion to each drone's momentary battery level**, on a topological
representation of the free space, so that the fleet's energy is spent more evenly and the
mission completes more efficiently than under position-based baselines.

**Central contribution.** An **energy-weighted spatial decomposition**: the survey area is
reduced to a **Generalized Voronoi Graph (GVG)** and a **Topological Graph Construction
(TGC)** of free-space regions and safe corridors, and those regions are assigned to drones
with each drone's share **weighted by its current battery state** rather than by position
alone — together with a **dynamic return-to-home energy reserve** that replaces a static
battery-fraction threshold with a per-replication, obstacle-aware cost-to-go estimate.

**Method of validation.** Because no physical flight experiments were performed, the claim is
established **comparatively**:

- a **grey-box component energy model** (per-maneuver power × time) and kinematically
  realistic trajectories give physically meaningful relative energies;
- every method is run on **paired identical random seeds** through one pipeline, so
  systematic modeling error cancels in the differences;
- results are taken to statistical stability by Monte Carlo over ≥ 30 replications per
  scenario.

**Hypothesis under test.** Weighting the decomposition by momentary battery level produces a
**more balanced workload, a higher minimum final battery, and lower total energy** than
position-based decomposition for identical drones.
