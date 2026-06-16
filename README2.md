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
7. Optionally exports a structured **telemetry log**: GPX 1.1 tracks for GIS visualization and a JSONL event log for automated mission analysis.
8. Optionally runs an **LLM-as-a-judge** over that telemetry log to auto-diagnose the mission — classifying the outcome, attributing a root cause, and **verifying the diagnosis against deterministic ground truth**.

It is written for **clarity at a thesis defense**: one module per idea, strictly typed interfaces, and a direct mapping from every quantitative claim in the thesis to exactly one place in the code.

---

## 2. Architecture: Five layers mirroring the methodology

The package layout deliberately mirrors the three methodology layers of the thesis, plus an analysis layer and an infrastructure layer.

| Package | Responsibility |
|---|---|
| `physical_model/` | Grey-box component energy model; aerodynamic formation correction; Dubins kinematics; the `MotionModel` abstraction; **1-D vertical takeoff/landing and inter-layer climbs**. Realistic Turn aerodynamic penalties. |
| `planning/` | GeoJSON parsing; 3D Prism Poisson obstacles; Layered Environment mapping (`LayerStack`); GVG + TGC construction; the weighted decomposition (central contribution); optimal launch-site optimization with exact swap math; target-visit and dynamic obstacle routines. |
| `execution/` | The seven-state automaton; the agent and fleet; dynamic RTH calculator; event-driven redistribution; proactive safety monitor; battery-swap station with **finite Shared Battery Pool**; strictly decoupled hazard-failure vs. battery-depletion failure (`MISSION_FAILED`). |
| `metrics/` | State-history recording; deterministic mission metrics; SMDP estimation (embedded chain + mean sojourns); the stationary distribution with the embedded→time-weighted correction; Monte-Carlo runner; **event-driven telemetry collector with GPX track and LLM-ready JSONL exporters**; **grounded LLM-as-a-judge mission-diagnosis engine (`llm_judge.py`)**. |
| `infrastructure/` | Typed configuration; reproducible RNG; logging; the `SimulationEngine` orchestrator; visualization. |
| `experiments/` | CLI entry points composing the layers into thesis experiments (e.g., Fleet Sizing Pareto Analyzer, Decomposition Comparisons, **automated mission diagnosis `run_llm_diagnosis.py`**). |

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
       → [if telemetry enabled] periodic position fixes + event capture
       ▼
    Metrics: SMDP estimate → stationary π → efficiency score
       ▼
    [if telemetry enabled] GPX tracks + JSONL event log written to disk
       ▼
    [offline, optional] LLM-as-a-judge over the JSONL → grounded MissionDiagnosis report

---

## 3. Terminal States and Realistic Dynamics (Phase 2)

The simulation enforces strict, mutually exclusive terminal outcomes evaluated at the end of each simulation tick, preventing unrealistic "zombie" computations:
- **`MISSION_SUCCESS`**: 100% of the area is covered AND every surviving drone has returned to the launch site (`S0_IDLE`).
- **`MISSION_FAILED`**: A physics-dictated halt. Triggers immediately if an **airborne drone's battery reaches 0** (`S_FAIL` mid-flight), or if the **Shared Battery Pool is exhausted** while area coverage is incomplete. 

**Circular Deployment Footprint:** Drones are instantiated in a mathematically calculated ring around the launch pose at `t=0`, preventing artificial collisions and immediate `S_OBS` deadlocks.

---

## 4. Event-Driven Telemetry and Observability (Phase 3)

When enabled, the simulation engine emits a structured event log of every mission **discontinuity** — not raw per-tick dumps, but semantically classified transitions with per-phase deltas and instantaneous agent snapshots. This feeds two output formats:

- **GPX 1.1 tracks** — one `<trk>` per drone, for loading directly into QGIS, Google Earth, or mission planners. Planar simulation coordinates are projected to geographic WGS-84 via a local tangent-plane (equirectangular) approximation with a configurable false origin. Altitude `z` maps to `<ele>` exactly, so 3D viewers render the layer structure correctly.
- **JSONL event log** — a compact, causal trace of the mission (run header → events → run summary) small enough for an LLM context window. Each event carries a semantic verb (`OBSTACLE`, `SWAP_REQ`, `SWAP_DONE`, `FAIL`, `TERMINAL`), the state transition (`from`/`to`/`reason`), battery fraction and zone, position, and per-phase deltas (`Δt`, `ΔE`, `Δdist`). Sparse by design: fields appear only where they apply.

The telemetry layer is architecturally **decoupled** from the simulation core. It shares the agent's existing `Recorder.open/close` protocol via a `FanoutRecorder` and pulls snapshots read-only from the live fleet. No changes to `agent.py`, `state_history.py`, or `core_types.py` were required. When disabled (the default), no telemetry objects are constructed and the run is **byte-identical** to the pre-Phase-3 baseline.

### Enabling Telemetry

Add a `telemetry:` block to your scenario YAML file (e.g. `config/default.yaml` or any scenario-specific YAML you pass with `--config`). Place it at the **top level** of the file, alongside existing blocks like `fleet:`, `sensor:`, `sim:`, etc.:

```yaml
# config/default.yaml (or any scenario YAML)
# ... existing blocks: fleet, sensor, sim, env, etc. ...

telemetry:
  enabled: true
  gpx_path: "runs/my_run/tracks.gpx"
  llm_log_path: "runs/my_run/events.jsonl"
  fix_interval_s: 30.0                     # periodic GPX position-fix cadence (seconds)
  origin_lat: 54.6872                      # tangent-plane false origin latitude (default: Vilnius)
  origin_lon: 25.2797                      # tangent-plane false origin longitude
  epoch_iso: "2026-01-01T00:00:00Z"        # GPX <time> = this epoch + sim seconds
```

When the simulation run finishes, two files appear at the configured paths. All fields have defaults, so the minimal enablement is just:

```yaml
telemetry:
  enabled: true
```

which writes `telemetry_tracks.gpx` and `telemetry_events.jsonl` to the working directory with Vilnius as the projection origin.

When the `telemetry:` block is **absent** from the YAML (or `enabled: false`), the feature does not exist — no objects are constructed, no files are written, and the `config_hash` is unchanged because the hash is computed from the raw YAML before parsing.

### Telemetry Output Schema

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

## 5. Automated Mission Diagnosis (Phase 4 — LLM as a Judge)

An LLM-as-a-judge reads the Phase 3 JSONL log and produces a structured, **grounded** mission diagnosis: an outcome classification, a single root cause from a fixed taxonomy, a causal narrative, contributing factors, a critical-event timeline, and recommendations. The logic lives in `metrics/llm_judge.py`; the CLI is `experiments/run_llm_diagnosis.py`.

Three design commitments make this an evaluation device rather than a chatbot:

- **Deterministic grounding.** Before any model is called, the ground-truth facts (outcome, the terminal trigger, which drone died, per-drone minimum battery, swap / obstacle counts) are extracted directly from the log — no model involved. These facts are handed to the model AND used afterwards to **verify its claims**. If the model names a drone that never failed or contradicts the recorded outcome, the mismatch is flagged in a "grounding audit" rather than silently trusted.
- **Fixed taxonomy.** The model must attribute the root cause from a closed set aligned to the simulation's terminal physics (`BATTERY_DEPLETION_AIRBORNE`, `SHARED_POOL_EXHAUSTION`, `COVERAGE_TIMEOUT`, `HAZARD_ATTRITION`, `OBSTACLE_THRASHING`, `SWAP_LOGISTICS`, `NOMINAL_SUCCESS`, `UNDETERMINED`), so verdicts are machine-comparable across runs.
- **Injectable, model-agnostic.** The judge takes a `model` callable `(system, user) -> str`. An Anthropic adapter is provided (`anthropic_model`, lazily imported), but any other LLM is wired in with a five-line adapter. The module imports with no third-party dependency, and the whole pipeline (facts → prompt → parse → ground → render) is unit-testable offline.

### Running the diagnosis

The diagnosis runs **offline against a telemetry JSONL file** — it is independent of `--out` and of the simulation run itself.

```bash
# 1. produce a telemetry log (Section 4), e.g. runs/demo/events.jsonl

# 2. FREE: inspect the deterministic facts + the exact prompt, no API call, no cost
python -m uav_swarm_sim.experiments.run_llm_diagnosis \
    --log runs/demo/events.jsonl --dry-run

# 3. LIVE diagnosis -> writes a Markdown report (needs an API key, see below)
python -m uav_swarm_sim.experiments.run_llm_diagnosis \
    --log runs/demo/events.jsonl --out runs/demo/diagnosis.md --model claude-sonnet-4-6
```

The live call requires an Anthropic API key, read from the `ANTHROPIC_API_KEY` environment variable (or passed with `--api-key`), and the `anthropic` package installed (`pip install anthropic` — only needed for live diagnosis, not for `--dry-run`). Set `--model` to whatever model id your key has access to.

### Diagnosis output

A Markdown report whose verdict is an LLM judgment, but whose **facts and grounding audit are computed deterministically from the log**:

```
# Mission Diagnosis - MISSION_FAILED

**Root cause:** `BATTERY_DEPLETION_AIRBORNE`  |  **Confidence:** 0.92  |  **Grounded:** yes

## Summary
Drone 1 depleted its battery mid-mission at t=1180s while in S2_MISSION, halting the run at 93% coverage.

## Mission facts (deterministic)
- Failed drones: [1] (n_failed=1)
- Terminal trigger: battery_depleted
- Lowest battery observed: drone 1 at 0%
...

## Grounding audit
- [OK] outcome_matches_log - model=MISSION_FAILED log=MISSION_FAILED
- [OK] cited_drones_exist - cited=[0, 1] fleet=0..1 unknown=[]
- [OK] root_cause_consistent_with_terminal - model=BATTERY_DEPLETION_AIRBORNE deterministic_hint=BATTERY_DEPLETION_AIRBORNE
```

### Interactive analysis (the mission-analyst prompt)

The programmatic judge above returns a machine-checkable JSON verdict. For quick, **human-readable** interpretation — when you (or an examiner) want a narrative diagnosis and you also have the **figures** in hand — the repository ships a ready-to-use **mission-analyst system prompt** at `docs/mission_analyst_prompt.md`. Paste it as the system prompt in any Claude (or other LLM) chat, attach the run's artifacts, and ask *"analyze this run."* Unlike the CLI judge, it reads the plots as well as the logs and returns a structured written report. The two interfaces are complementary: the CLI judge for a scriptable, auditable verdict; the prompt for the readable story.

**What to pass to it** (any subset works; more is sharper, and it will request what is missing):

| Input | Why it helps |
|---|---|
| **`events.jsonl`** | The richest single input; enables transition-level dissection (`from`→`to` loops, the `per_drone_obstacle_events` tally). **Attach this first.** |
| **The config YAML** (`default.yaml` or your scenario) | Turns generic advice ("raise the reserve") into concrete edits ("`rth.reserve_frac` 0.05 → 0.15"). |
| **The console summary line** | `[single] aborted=… coverage=… energy=… duration=… workload_std=… efficiency=…` — the headline deterministic metrics. |
| **Figures** | `state_gantt.png` (most diagnostic), `battery.png`, `pi_bars.png`, `partition.png`, `paths.png`, `environment.png`, `replay.gif`. |
| **Context** | The decomposition algorithm, the seed, whether dynamic obstacles or a nonzero `failure.hazard_rate_per_hour` were active, and **what outcome you expected** (so it can judge *why* a result is unsatisfying, not just *what* it is). |

**What to expect back** — a written report with eight sections:

1. **Verdict** — outcome + single most likely root cause + confidence.
2. **Mission setup** — platform, fleet, area, layers, logistics, decomposition, seed.
3. **What happened** — a chronological narrative grounded in events and figures.
4. **Per-drone notes** — a table of each drone's behavior and fate.
5. **Problems identified (ranked)** — each as *Observation (with evidence) → Hypothesis → Confidence*.
6. **Recommendations (ranked)** — concrete named config knobs with expected effect.
7. **Anomalies & caveats** — contradictions between sources, suspicious values.
8. **What would sharpen the analysis** — the specific follow-up artifact to provide next.

The analyst is **grounding-disciplined**: it cites the figure / drone / timestamp behind each claim, separates observation from hypothesis, and *flags contradictions between sources rather than papering over them* — for example, a `coverage_frac` of 0 reported alongside heavy `S2_MISSION` time is surfaced as a likely accounting issue to investigate, not silently accepted. It typically ends by requesting one specific follow-up artifact, so treat it as a conversation: answer its question and it will go deeper.

**Tips:**
- Lead with `events.jsonl`. With figures alone the analyst reasons well but cannot quote exact transitions or per-drone counts.
- Provide the config every run — it is what makes the recommendations actionable rather than generic.
- State your intent ("I expected full coverage in under 1500 s") to give it a target to diagnose against.
- If you run both the CLI judge and the prompt, reconcile their verdicts; agreement raises confidence, disagreement is itself a useful signal.

---

## 6. The S_FAIL Dual View (Physical vs. Analysis Layer)

Failure is modeled **differently and deliberately** in two separate layers:

- **Physical simulation layer (Irreversible & Terminal).** A drone falling out of the sky due to `battery == 0` halts the simulation with `MISSION_FAILED`. 
- **SMDP analysis layer (Hazard Rates & Ergodicity).** For Monte-Carlo resilience statistics, random hazard-induced failures (`λ > 0`) permanently remove the agent and trigger an event-driven **Redistribution** of uncovered work among survivors. This does NOT trigger a mission halt. A generic agent-slot replacement device creates a synthetic `S_FAIL → S0` transition to close the loop, guaranteeing an ergodic Markov chain for the stationary distribution analysis.

---

## 7. Platform support and the energy-coefficient caveat

Runs simulate **one** homogeneous platform (`FIXED_WING`, `MULTIROTOR`, or `VTOL`).

> **⚠ Energy-coefficient caveat:** No physical flight experiments were conducted. Motor power coefficients are **theoretical approximations** via Steup et al.'s component method. The **quadrotor** (`MULTIROTOR`) coefficients are closest to the validated Steup baseline. All comparative results are computed on **paired identical seeds**, ensuring conclusions about *differences* are robust even where absolute energies are approximate.

---

## 8. Installation

Requires **Python 3.12+**.

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt        

The `anthropic` package is required **only** for live LLM diagnosis (Section 5). Everything else — the simulation, telemetry export, and the diagnosis `--dry-run` — runs without it.

---

## 9. Running Experiments

Entry points live in `src/uav_swarm_sim/experiments/`.

    # Headline experiment: classic vs tgc_basic vs weighted_voronoi
    python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/scenarios/tier_mid_comparison.yaml --out runs/decomp

    # B6.2 Fleet Sizing (Analytical Pareto Knee)
    python -m uav_swarm_sim.experiments.run_fleet_sizing_analyzer --config config/default.yaml --n-max 20

    # Launch Site Suitability Heatmap
    python -m uav_swarm_sim.experiments.plot_launch_suitability

    # LLM-as-a-judge mission diagnosis over a telemetry log (see Section 5)
    python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --out runs/demo/diagnosis.md
    python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --dry-run   # facts + prompt only, no API call

    # Smoke test
    pytest tests/test_smoke.py

To run any experiment **with telemetry enabled**, add the `telemetry:` block to the YAML you pass via `--config` (see Section 4 above). The GPX and JSONL files will be written to the configured paths when the run completes.

---

## 10. Development Roadmap & Horizon

The core architecture, 2.5D logic, and B6 Exact Fleet Sizing math are complete (151/151 Green Tests). The simulation has completed its observability and AI-analysis phases:

- **Phase 2 (Complete):** Strict terminal states, finite battery logistics, circular deployment, boustrophedon turn penalties, and pre-flight trajectory validation mechanism (Task 2.5 Q1).
- **Phase 3 (Complete):** Observability and Telemetry. An event-driven telemetry collector logs mission discontinuities (state transitions, obstacle encounters, swaps, terminal verdicts) alongside periodic position fixes. Two exporters serialize the single in-memory log: **GPX 1.1** tracks (one `<trk>` per drone, local tangent-plane projection, exact `<ele>` altitudes) for QGIS / Google Earth, and **JSONL** (sparse, semantic, causal event rows with per-phase energy/time/distance deltas) for the Phase 4 judge. The collector is a read-only probe via a fan-out recorder — it never feeds back into physics, energy, or the SMDP. Disabled by default; enable via a `telemetry:` config block.
- **Phase 4 (Complete):** Automated AI analysis (LLM as a judge) over the JSONL telemetry logs to auto-diagnose missions. The judge extracts deterministic ground truth from the log, constrains the model to a fixed root-cause taxonomy, and **verifies the model's verdict against that ground truth** (a grounding audit that flags hallucinated drones or contradicted outcomes). The model is injectable and model-agnostic; the pipeline is unit-testable offline.
- **Task 2.5 Q2 (Planned):** Stateful S_OBS recovery sub-FSM (EVADE/HOLD/REJOIN) wired into the trajectory validation mechanism.
- **Final B6 Experiment (Planned):** The multi-layer Altitude-Tradeoff Study (Vertical energy cost vs. per-layer obstacle sparsity reduction).
