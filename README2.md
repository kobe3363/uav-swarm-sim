# UAV Swarm Reconnaissance Simulation (2.5D Layered Architecture)

A modular, explainable simulation of **coordinated mission optimization for a homogeneous reconnaissance UAV swarm**. It is the Part III computational artifact for the master's thesis *Optimising flight missions between identical reconnaissance drones* (Konstantin Belena, Vilnius Gediminas Technical University, Antanas Gustaitis Aviation Institute).

The simulation operationalizes the thesis's central methodological contribution — **energy-weighted spatial decomposition by momentary battery level on a topological-graph (TGC/GVG) framework for identical drones in obstacle environments** — and proves its properties through a **Semi-Markov Decision Process (SMDP)** analysis layer. 

The system operates in a **2.5D Layered Environment**, where coverage planning remains strictly 2D within configurable horizontal altitude bands, connected by physics-accurate vertical climb and descent segments.

---

## 1. What this project is

A discrete-time, Monte-Carlo simulation that, for a fleet of identical reconnaissance drones:

1. Ingests an exploration-area boundary (GeoJSON) and a synthetic obstacle field (extruded as 3D prisms over `[floor, ceil]` ranges).
2. Chooses an optimal, energy-aware launch site strictly from a safe staging ring.
3. Partitions the area among the drones across multiple altitude layers, so each drone's region is proportional to its **momentary battery level**.
4. Enforces strict fleet logistics: finite **Shared Battery Pool** and exact battery swap cycle counting.
5. Flies kinematically realistic coverage paths under a seven-state behavioral automaton. Features include: dynamic return-to-home, proactive obstacle avoidance, event-driven redistribution, and rigorous **Terminal State** evaluation (Mission Success vs. Physics-Dictated Failure).
6. Measures the result both deterministically (energy, duration, swap metrics) and stochastically (stationary distribution over states, efficiency score), with statistical convergence.

It is written for **clarity at a thesis defense**: one module per idea, strictly typed interfaces, and a direct mapping from every quantitative claim in the thesis to exactly one place in the code.

---

## 2. Architecture: Five layers mirroring the methodology

The package layout deliberately mirrors the three methodology layers of the thesis, plus an analysis layer and an infrastructure layer.

| Package | Responsibility |
|---|---|
| `physical_model/` | Grey-box component energy model; aerodynamic formation correction; Dubins kinematics; the `MotionModel` abstraction; **1-D vertical takeoff/landing and inter-layer climbs**. Realistic Turn aerodynamic penalties. |
| `planning/` | GeoJSON parsing; 3D Prism Poisson obstacles; Layered Environment mapping (`LayerStack`); GVG + TGC construction; the weighted decomposition (central contribution); optimal launch-site optimization with exact swap math; target-visit and dynamic obstacle routines. |
| `execution/` | The seven-state automaton; the agent and fleet; dynamic RTH calculator; event-driven redistribution; proactive safety monitor; battery-swap station with **finite Shared Battery Pool**; strictly decoupled hazard-failure vs. battery-depletion failure (`MISSION_FAILED`). |
| `metrics/` | State-history recording; deterministic mission metrics; SMDP estimation (embedded chain + mean sojourns); the stationary distribution with the embedded→time-weighted correction; Monte-Carlo runner. |
| `infrastructure/` | Typed configuration; reproducible RNG; logging; the `SimulationEngine` orchestrator; visualization. |
| `experiments/` | CLI entry points composing the layers into thesis experiments (e.g., Fleet Sizing Pareto Analyzer, Decomposition Comparisons). |

### The 2.5D Execution Flow

    config/default.yaml
       │
    EnvironmentMap (3D Prisms) → LayerStack (sliced 2D horizontal planes)
       ▼
    LaunchSiteOptimizer (Exact B6.2 Swap Math, Staging Ring)
       ▼
    Decomposer: area ∝ momentary battery level (assigned per layer)
       ▼
    CoveragePath per zone (boustrophedon + Dubins smoothing)
       ▼
    SimulationEngine dt-loop (Fail-Fast execution):
       fleet init (Circular Deployment) → safety → agents step (incl. inter-layer transit)
       → swap station (Shared Pool) → Terminal Evaluation (SUCCESS / FAILED / INCOMPLETE)
       ▼
    Metrics: SMDP estimate → stationary π → efficiency score

---

## 3. Terminal States and Realistic Dynamics (Phase 2 Updates)

The simulation enforces strict, mutually exclusive terminal outcomes evaluated at the end of each simulation tick, preventing unrealistic "zombie" computations:
- **`MISSION_SUCCESS`**: 100% of the area is covered AND every surviving drone has returned to the launch site (`S0_IDLE`).
- **`MISSION_FAILED`**: A physics-dictated halt. Triggers immediately if an **airborne drone's battery reaches 0** (`S_FAIL` mid-flight), or if the **Shared Battery Pool is exhausted** while area coverage is incomplete. 

**Circular Deployment Footprint:** Drones are instantiated in a mathematically calculated ring around the launch pose at `t=0`, preventing artificial collisions and immediate `S_OBS` deadlocks.

---

## 4. The S_FAIL Dual View (Physical vs. Analysis Layer)

Failure is modeled **differently and deliberately** in two separate layers:

- **Physical simulation layer (Irreversible & Terminal).** A drone falling out of the sky due to `battery == 0` halts the simulation with `MISSION_FAILED`. 
- **SMDP analysis layer (Hazard Rates & Ergodicity).** For Monte-Carlo resilience statistics, random hazard-induced failures (`λ > 0`) permanently remove the agent and trigger an event-driven **Redistribution** of uncovered work among survivors. This does NOT trigger a mission halt. A generic agent-slot replacement device creates a synthetic `S_FAIL → S0` transition to close the loop, guaranteeing an ergodic Markov chain for the stationary distribution analysis.

---

## 5. Platform support and the energy-coefficient caveat

Runs simulate **one** homogeneous platform (`FIXED_WING`, `MULTIROTOR`, or `VTOL`).

> **⚠ Energy-coefficient caveat:** No physical flight experiments were conducted. Motor power coefficients are **theoretical approximations** via Steup et al.'s component method. The **quadrotor** (`MULTIROTOR`) coefficients are closest to the validated Steup baseline. All comparative results are computed on **paired identical seeds**, ensuring conclusions about *differences* are robust even where absolute energies are approximate.

---

## 6. Installation

Requires **Python 3.12+**.

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt        

---

## 7. Running Experiments

Entry points live in `src/uav_swarm_sim/experiments/`.

    # Headline experiment: classic vs tgc_basic vs weighted_voronoi
    python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/scenarios/tier_mid_comparison.yaml --out runs/decomp

    # B6.2 Fleet Sizing (Analytical Pareto Knee)
    python -m uav_swarm_sim.experiments.run_fleet_sizing_analyzer --config config/default.yaml --n-max 20

    # Launch Site Suitability Heatmap
    python -m uav_swarm_sim.experiments.plot_launch_suitability

    # Smoke test
    pytest tests/test_smoke.py

---

## 8. Development Roadmap & Horizon

The core architecture, 2.5D logic, and B6 Exact Fleet Sizing math are complete (151/151 Green Tests). The simulation is currently executing the final engineering phases:

- **Phase 2 (Active/Complete):** Strict terminal states, finite battery logistics, circular deployment, and boustrophedon turn penalties.
- **Phase 3 (Planned):** Observability and Telemetry. Exporting highly detailed CSV/GPX traces of every drone's spatial position and state for external GIS routing analysis.
- **Phase 4 (Planned):** Automated AI analysis (LLM as a judge) over simulation logs to auto-diagnose mission failures.
- **Final B6 Experiment (Planned):** The multi-layer Altitude-Tradeoff Study (Vertical energy cost vs. per-layer obstacle sparsity reduction).