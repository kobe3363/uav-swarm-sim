# UAV Swarm Reconnaissance Simulation

A modular, explainable simulation of **coordinated mission optimization for a homogeneous reconnaissance UAV swarm**. It is the Part III computational artifact for the master's thesis *Optimising flight missions between identical reconnaissance drones* (Konstantin Belena, Vilnius Gediminas Technical University, Antanas Gustaitis Aviation Institute).

The simulation operationalizes the thesis's central methodological contribution — **energy-weighted spatial decomposition by momentary battery level on a topological-graph (TGC/GVG) framework for identical drones in obstacle environments** — and proves its properties through a **Semi-Markov Decision Process (SMDP)** analysis layer.

---

## 1. What this project is

A discrete-time, Monte-Carlo simulation that, for a fleet of identical reconnaissance drones:

1. ingests an exploration-area boundary (GeoJSON) and a synthetic obstacle field,
2. chooses an optimal launch site (treated as a decision variable, not an assumption),
3. partitions the area among the drones so each drone's region is proportional to its **momentary battery level**,
4. flies kinematically realistic coverage paths under a seven-state behavioral automaton with a dynamic return-to-home rule, event-driven redistribution, proactive obstacle avoidance, and battery-swap "reincarnation",
5. and measures the result both deterministically (energy, duration, workload balance) and stochastically (stationary distribution over states, efficiency score), with statistical convergence.

It is written for **clarity at a thesis defense**: one module per idea, strictly typed interfaces, and a direct mapping from every quantitative claim in the thesis to exactly one place in the code.

---

## 2. Architecture: five layers mirroring the methodology

The package layout deliberately mirrors the three methodology layers of the thesis, plus an analysis layer and an infrastructure layer.

| Package | Thesis layer | Responsibility |
|---|---|---|
| `physical_model/` | **Physical model** (guidelines 1.1–1.4) | Grey-box component energy model; aerodynamic formation correction; Dubins kinematics; the `MotionModel` platform abstraction; 1-D vertical takeoff/landing segments. |
| `planning/` | **Planning** (guidelines 2.1–2.2) | GeoJSON parsing; Poisson obstacle generation; GVG + TGC construction; the weighted decomposition (central contribution) and its baselines; the k-means heuristic; Dijkstra cost database; launch-site optimizer; boustrophedon coverage paths; the grid comparative planner; target-visit tour planning (`target_mission.py`); drone-sized moving obstacles (`dynamic_obstacles.py`). |
| `execution/` | **Execution** (guidelines 3.1–3.4) | The seven-state automaton; the agent and fleet; the event bus; dynamic RTH calculator; event-driven redistribution; proactive safety monitor; formation manager; battery-swap station; hazard-rate failure model; three-tier scale-dependent selection; swarm passive/active dynamic-obstacle sensing (`sensing.py`). |
| `metrics/` | **Analysis (SMDP)** | State-history recording; deterministic mission metrics; SMDP estimation (embedded chain + mean sojourns); the stationary distribution with the embedded→time-weighted correction; the efficiency score; the Monte-Carlo runner with CI-based convergence; algorithm comparison; validation. |
| `infrastructure/` | **(supporting)** | Typed configuration; reproducible RNG; logging; the simulation engine/orchestrator; all visualization, including the optional view-only comm-range overlay (`VizConfig`). |
| `experiments/` | **(supporting)** | Thin CLI entry points that compose the layers into the thesis experiments. |

### How a single mission flows through the layers

```
config/default.yaml
   │  (infrastructure.config: typed, validated, unit-normalized)
   ▼
EnvironmentMap  ←  GeoJSON boundary + Poisson obstacles        [planning]
   ▼
GVG  →  TGC (corridors + free-space regions + adjacency)        [planning]
   ▼
LaunchSiteOptimizer  →  launch site (3 criteria: distance,      [planning]
   │                     formation-corrected energy, expected swaps)
   ▼
Decomposer (tier-selected): area ∝ momentary battery level      [planning]
   ▼
CoveragePath per zone (boustrophedon + Dubins smoothing)        [planning]
   ▼
SimulationEngine dt-loop:                                       [infrastructure]
   failure → safety → agents step (motion+energy+state+RTH)     [execution + physical]
   → swap station → drain events → redistribution
   ▼
StateHistory + MissionMetrics                                   [metrics]
   ▼
SMDP estimate → stationary π (embedded → time-weighted)         [metrics]
   → efficiency score
   ▼
MonteCarlo / Comparison / Validation  →  CSV + Markdown + plots [metrics + infrastructure]
```

The single most important computational subtlety is in `metrics/stationary_distribution.py`: the left-eigenvector of the embedded transition matrix yields **visit frequencies**, not time fractions. The code multiplies each embedded `π[i]` by the mean sojourn time `m_i` and renormalizes to obtain the **time-weighted** `π` that the efficiency score requires. The two distributions are always plotted side by side so the correction is visible rather than implicit.

---

## 3. Platform support and the energy-coefficient caveat

`platform_type` is a configuration enum; each run simulates **one** homogeneous platform:

- **`FIXED_WING`** — `DubinsModel` kinematics (minimum turn radius); formation flight yields a **drag/energy benefit** (≈15.14% in homogeneous formation, Guo et al. 2025), applied only during launch, transit-to-zone, and episodic RTH — never during dispersed coverage.
- **`MULTIROTOR`** — `HolonomicModel` kinematics (in-place turns, Euclidean legs); formation downwash is treated as a **safety constraint** (wake zones as "invisible obstacles"), **not** an energy benefit.
- **`VTOL`** — `DubinsModel` for the horizontal coverage phase, bracketed by 1-D vertical climb/descent segments; the three-phase mission of §2.5 verbatim. Cruise gets the formation energy benefit; vertical phases treat downwash as a constraint.

> **⚠ Energy-coefficient caveat (read before trusting absolute numbers).**
> No physical flight experiments were conducted (thesis guideline 1.1). Motor power coefficients are **theoretical approximations from typical platform specifications** following Steup et al.'s component method. The **quadrotor** (`MULTIROTOR`) coefficients are closest to the validated Steup baseline. The **`FIXED_WING` and `VTOL`** coefficient tables are **coarser extrapolations** and should be read as *relative*, structurally-consistent estimates — not calibrated absolute energies. All comparative results (algorithm-vs-algorithm, site-vs-site, Dubins-vs-grid) are computed on **paired identical seeds**, so conclusions about *differences* are robust even where absolute energies are approximate.

---

## 4. The S_FAIL dual view (physical layer vs. analysis layer)

Failure is modeled **differently and deliberately** in two separate layers. This is a conscious modeling decision, documented here and in `metrics/smdp_estimator.py`:

- **Physical simulation layer (thesis-faithful, irreversible).** When a drone fails, `execution/fleet.py::kill` removes it permanently and `execution/redistribution.py` immediately re-partitions the **uncovered** work among the **surviving active** agents via the weighted TGC decomposer. The failed drone does not return. This matches the thesis: failure is irreversible; the zone is redistributed.

- **SMDP analysis layer (ergodicity device).** The stationary distribution `π = πP` exists only if the embedded Markov chain is **ergodic** (strongly connected). A terminal `S_FAIL` is an absorbing state and would make `π` undefined. The estimator therefore models a **generic agent-slot**: a failed slot is closed at a configurable mean repair/replacement time and given a synthetic transition `S_FAIL → S0` (a replacement drone occupies the slot). This closes the loop alongside the genuine `S3 → S_SWAP → S0` swap loop.

This is **not** a claim that the physical swarm self-heals. It is the standard renewal-theoretic treatment that lets us speak of long-run time-fractions per state for a continuously-operated slot. The estimator exposes `close_failure_loop` (default `True`); with `False` it refuses to compute `π` and reports `ergodic=False`, which is the correct behavior for a literal absorbing-failure reading.

Contrast with **battery swap**, which is reversible and does *not* trigger redistribution: the swapped drone resumes its own remaining plan. The swap/failure asymmetry is enforced by wiring (only `FAILURE` and `NEW_TASK` reach the redistributor), not by convention.

---

## 5. Configuration — all global inputs in one place

`config/default.yaml` mirrors the typed schema in `infrastructure/config.py` one-to-one. The complete set of global inputs that propagate through the system:

- **Fleet (homogeneous):** `n_drones`, `battery_capacity_wh`, `drone_dims_m`.
- **Platform:** `type`, cruise/coverage/climb/descent speeds, `r_min_m`, `omega_max`, and the per-maneuver grey-box **`power_w` table** (one table per platform type).
- **Sensor:** `swath_width_m`, `overlap_frac` (drives boustrophedon strip spacing).
- **Aerodynamics:** `formation_drag_reduction` (0.1514), downwash geometry, formation spacing, RTH rendezvous window.
- **Environment:** `geojson_path`, obstacle density / size range / shapes / class count, clearance buffer.
- **Launch:** candidate sites (explicit list or sample count) and the three objective weights `w_distance`, `w_energy`, `w_swaps`.
- **Battery zones:** `high: 0.75`, `nominal: 0.40`, `critical: 0.20` (TERMINAL < 0.20).
- **Swap:** service time, number of bays. **Failure:** hazard rate `λ` per flight-hour. **Safety:** separation, obstacle buffer, prediction horizon. **RTH:** check interval, small `reserve_frac` ε (≈0.05 — explicitly *not* a static 30%).
- **Simulation:** `dt_s`, `max_timesteps`, `master_seed`.
- **Monte Carlo:** `n_max: 1000`, `n_min`, `ci_tolerance`.
- **Tier thresholds:** `[15, 50]`.
- **Mission type:** `mission.type` = `coverage` (default) or `target_visit`; for target missions, `n_targets` or explicit `target_coordinates`, and `weight_targets_by_battery` (per-drone target count ∝ battery).
- **Dynamic obstacles:** `dynamic_obstacles.enabled` (default **false**), `count`, `speed_m_s`, `size_m`, `passive_sense_range_m`, `active_sense_range_m`, `active_scan_power_w`, `dynamic_hold_s` — drone-sized moving obstacles with swarm-triggered passive/active LIDAR sensing.
- **Visualization (view-only, no physics effect):** `viz.show_comm_range` (default **false**), `comm_range_m`, `comm_range_alpha`, `comm_range_dashed` — draws a communication-range circle around each drone in the replay GIF and the state-colored path PNG.

> **Note on the failure hazard rate.** With `λ = 0`, the `S_FAIL` state is never entered and never populates in Monte Carlo; set `λ > 0` to study resilience and to give the failure-loop closure something to act on.

---

## 6. Installation

Requires **Python 3.12+**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # networkx, shapely, scipy, numpy, matplotlib, pulp, pyyaml, pytest
```

---

## 7. Running

All entry points live in `src/uav_swarm_sim/experiments/` and take a `--config` path plus an output directory.

```bash
# One mission with full visual output (the defense demo)
python -m uav_swarm_sim.experiments.run_single_mission --config config/default.yaml --out runs/demo

# Headline experiment: classic_voronoi vs tgc_basic vs weighted_voronoi
python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/scenarios/tier_mid_comparison.yaml --out runs/decomp

# Guideline 1.2: Dubins vs discretized grid (FIXED_WING / VTOL only)
python -m uav_swarm_sim.experiments.run_kinematics_comparison --config config/default.yaml --out runs/kinematics

# Guideline 3.3: scale-tier sweep, locate the empirical break-even in 16–49
python -m uav_swarm_sim.experiments.run_scale_tiers --config config/default.yaml --n 8 12 24 36 48 64 80 --out runs/tiers

# §2.4: launch-site optimization study
python -m uav_swarm_sim.experiments.run_launch_site_study --config config/default.yaml --out runs/launch


python -m uav_swarm_sim.experiments.plot_launch_suitability
```

Each run writes figures (environment, partition, trajectories, per-agent state Gantt, battery traces, embedded-vs-time-weighted `π` bars, comparison box plots, Monte-Carlo convergence), a `result.json`/CSV, and a structured log. Runs are **deterministic** given `(config, master_seed, replication, algorithm, planner)`.

Smoke test:

```bash
pytest tests/test_smoke.py
```

---

## 8. Mapping thesis claims to code (explainability invariant)

| Thesis claim | Where it lives |
|---|---|
| Continuous component energy (no static averages) | `physical_model/energy_model.py` |
| 15.14% formation drag benefit, scope-limited | `physical_model/aero_correction.py` + `execution/formation_manager.py` |
| Kinematically realistic (true flyable length) | `physical_model/dubins.py`, `physical_model/motion_model.py` |
| **Area ∝ momentary battery (central contribution)** | `planning/weighted_decomposition.py` (with `tgc_basic.py` as its switched-off twin) |
| Works in obstacle environments (clearance between obstacle classes) | `planning/gvg_builder.py`, `planning/tgc.py` |
| Launch site as an optimization variable (3 criteria) | `planning/launch_site_optimizer.py` |
| Dynamic RTH (route-vs-return, not a static reserve) | `execution/rth_calculator.py` |
| Event-driven redistribution; swap ≠ failure | `execution/redistribution.py`, `execution/events.py` |
| Three-tier scale selection (~15 threshold) | `execution/algorithm_selector.py` |
| Semi-Markov, not Markov (battery = hidden memory) | `metrics/smdp_estimator.py` |
| Stationary `π` with embedded→time-weighted correction | `metrics/stationary_distribution.py` |
| Efficiency = π(S2)/(π(S3)+π(S_OBS)+π(S_SWAP)) | `metrics/efficiency_score.py` |
| Monte Carlo N≈1000 with convergence | `metrics/monte_carlo.py` |
| Speed claims (>60× at scale; ~1e-4 s/task replanning) | `planning/tgc.py` timing, `metrics/validation.py` |

---

## 9. Extension points for thesis text

These are places where the **simulation specification is ahead of the current Part II thesis text**. They do not affect the code, but the author may wish to reconcile the written methodology before the defense.

1. **The entire SMDP / stochastic-analysis layer is not yet described in Part II.** Methodological guideline 1.4 currently promises only **three deterministic metrics** (total mission energy, mission duration, workload-distribution standard deviation). The simulation additionally produces the primary stochastic metric (the stationary distribution `π`) and the **efficiency score** `π(S2) / (π(S3) + π(S_OBS) + π(S_SWAP))`, together with the Monte-Carlo convergence procedure. **Recommended addition to Part II:** a subsection introducing (a) the Semi-Markov justification — battery level as hidden memory violating the memoryless property, making the state two-dimensional `(state, sojourn time)`; (b) the stationary distribution as the time-fraction-per-state metric, valid only under the closed loop; (c) the embedded→time-weighted correction; and (d) the efficiency score and its throughput-oriented denominator choice.

2. **The closed-loop ergodicity requirement deserves an explicit sentence.** Part II describes the swap "reincarnation" loop and irreversible failure redistribution, but does not state that the stationary-distribution analysis requires a closed, ergodic chain — hence the generic-agent-slot replacement device used only in the analysis layer (§4 above). One paragraph distinguishing the *physical* irreversibility of failure from the *analytical* slot-replacement closure would pre-empt an examiner's question.

3. **The efficiency-score denominator choice (including `π(S_SWAP)`) is a defensible but unstated modeling decision.** The rationale — swap time is mission overhead under a throughput-oriented metric, consistent with the thesis leitmotif that minimizing swaps saves both energy and mission duration — should be stated explicitly so it reads as a choice, not an accident.

4. **The Dubins-vs-grid comparison and the launch-site three-criterion objective are described in Part II prose but not yet as formal experiment specifications.** The blueprint formalizes both (`run_kinematics_comparison`, `run_launch_site_study`); a one-line forward reference in §2.7.1's experiment plan would tie the text to the artifact.

5. **The battery-zone thresholds (HIGH ≥75 / NOMINAL ≥40 / CRITICAL ≥20 / TERMINAL <20) appear only in the simulation.** They serve as coarse transition guards and reporting bins (the fine instrument is the dynamic RTH calculator). If they are to be cited as results, they should be introduced in Part II as a modeling parameter with a brief justification.

---

## 10. Scope boundaries

Explicitly **out of scope** (named in the thesis as context, not required by the guidelines): wind-field modeling, communication/network *modeling* (note: `viz.show_comm_range` draws a comm-range circle for readability only — it does **not** model link budgets or connectivity), the 3-D Dubins airplane extension (excluded by §2.5's constant-altitude assumption), real flight-data regression (no physical experiments), and learning-based planners (MARL/DRL). The architecture isolates each so that adding one later touches a single module.
