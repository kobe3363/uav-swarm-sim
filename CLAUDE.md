# CLAUDE.md — uav-swarm-sim (MSc thesis simulator)

## What this project is
A 2.5D UAV swarm reconnaissance simulator for the MSc thesis **"Optimising flight
missions between identical reconnaissance drones"** (Vilnius Tech / VGTU, Antanas
Gustaitis Aviation Institute). Pure Python: NumPy, SciPy, Shapely 2.x, NetworkX, PuLP,
Matplotlib, PyYAML, pytest. **No MATLAB.**

Central contribution: spatial decomposition **GVG→TGC** + optimal launch siting +
SMDP-analyzed dynamic RTH, validated by Semi-Markov (SMDP) analysis and Monte Carlo.

Communicate with the user in **Lithuanian**; keep technical terms in English.

## NON-NEGOTIABLE working rules
1. **The regression net is SACRED.** Run `python -m pytest -q` after every change and
   report the green count. Nothing existing may go red. New behavior gets new tests.
2. **File-first, no hallucination.** Read the real source before editing. Verify exact
   signatures, enum members, config field names. `python -m py_compile` before delivering.
3. **Byte-identity discipline.** New behavior is gated default-OFF behind a config flag;
   flag-off runs must be byte-identical to pre-change.
4. **Physical truth over approximations.** Analytical formulas (E_cover, RTH lookahead)
   must EXACTLY mirror execution-time physics (discrete P·dt integration, camera term).
   Single source of truth: analytical == execution by construction where possible.
5. **Propose plan BEFORE code** for anything touching the decomposition core, energy
   model, FSM, connector routing, or anything thesis-affecting. Wait for the user's go.
6. **Flag every thesis-affecting decision** explicitly (metric definitions, thresholds,
   re-baselines, config defaults).
7. **Honest reporting.** If a result falsifies a hypothesis, report it as it falls.
   Never force-confirm. A clean falsification is a result.
8. **Deliverable = downloadable git format-patch.** Do NOT push, do NOT open PRs, do NOT
   watch CI — the user applies patches (`git am`), creates branches/PRs, and watches CI
   himself. End your task at the patch + Lithuanian apply instructions.
9. Exclude auto-generated artifacts (`*.egg-info/SOURCES.txt`) from patches/commits.

## Architecture facts (verified; do not re-derive wrongly)
- **FSM: 8 states** — S0_IDLE, S1_TRANSIT, S2_MISSION, S_FERRY, S3_RTH, S_SWAP, S_OBS,
  S_FAIL. `S_FERRY` = camera-off repositioning between coverage strips (airborne).
- **Coverage legs are boustrophedon:** even global leg index = COVERAGE strip (camera ON,
  S2_MISSION); odd = TURN connector (camera OFF, S_FERRY). Connector detection is
  STRUCTURAL (`_cov_idx % 2 == 1`), NOT per-segment maneuver — holonomic strip legs carry
  in-place-yaw TURN segments.
- **Camera energy:** `sensor.sensor_power_w`, charged ONLY over COVERAGE segments;
  default 0 (opt-in). The RTH lookahead includes it (mirrors execution).
- **Connector routing (Step 2, merged):** plan-time obstacle-aware via
  `planning/visibility_router.py` (visibility graph + Dijkstra over buffered obstacles),
  gated behind `coverage.ferry_free_space` (default OFF), `operating_area=convex_hull`,
  `operating_margin_m=50`. Option P single source: `boustrophedon` precomputes
  `CoveragePlan.connectors`; both `agent._build_coverage_legs` and
  `run_regime_calculator._rebuild_coverage_legs` replay them.
- **4 decomposition peers:** `weighted_voronoi` (battery-weighted TGC), `tgc_basic`
  (unweighted TGC twin), `classic_voronoi`, `kmeans`.
- **CRITICAL KNOWN NULL:** for a homogeneous fleet (identical drones, all starting at
  battery_frac=1.0) with λ=0, `weighted_voronoi ≡ tgc_basic` BYTE-IDENTICALLY (equal
  fractions → equal weights → identical partition). Battery-weighting differentiates only
  for diverged batteries. Redistribution fires only on FAILURE (λ>0) and currently always
  uses WeightedTgcDecomposer regardless of mission algo (known ablation-fidelity issue,
  task ADV-03, currently gated/rejected).
- **Regime classification:** per-drone **max-zone ratio** (busiest zone vs one usable
  battery) is PRIMARY; pooled E_cover/(n·B_usable) is only a lower bound (batteries are
  not pooled). See `run_regime_calculator.py`.
- **S_SWAP:** ground queue costs TIME but ZERO energy (drones land).
- Obstacles are full-height prisms by default. Multirotor (holonomic) is the study
  platform. Coverage area (camera must photograph the survey polygon) ≠ flyable area
  (drone may fly anywhere obstacles permit, incl. outside the polygon, when not in
  S_MISSION).
- **Paired seeds:** `RngFactory.stream(name, replication)` is a pure function of
  (master_seed, name, replication) — sharing one factory across variants gives exact
  paired seeds. Fixed-N MC (not CI-convergence stopping) preserves exact pairing.
- Key experiment scripts: `run_single_mission`, `run_decomposition_comparison`,
  `run_regime_calculator`, `run_shape_regime_table`, `run_shape_sweep` (S5),
  `run_fleet_sizing_analyzer` (B6.2 Pareto), `run_launch_site_study`, `run_spare_sizing`.
- Shapes: `data/areas/shapes/` — 9 equal-area (1 km²) GeoJSON incl. `c_shape` (primary
  concave test vehicle); donut deferred (interior-ring decomposition untested).

## Rejected — do NOT revisit or schedule
- 3D Dubins with coupled pitch/yaw (breaks byte-identity).
- CGAL rewrite of the GVG core (highest churn/risk).
- Constant-altitude / full-height-prism altitude study (no interior optimum; altitude
  only adds climb energy). An altitude study is meaningful ONLY with finite-height
  obstacles (ENG-07).
- Heterogeneous initial batteries as an S5 lever (out-of-scope stochasticity) and λ>0 as
  the primary shape-study driver — both rejected by the author.

## Roadmap
The canonical outstanding-task list lives in `docs/thesis_roadmap.md`. Read it before
picking up any task. Task IDs (NOW-*, STUDY-*, ADV-*, ENG-*, FIX-*) refer to that file.

## Environment (the user's machine)
Windows, MINGW64 git, venv (`source .venv/Scripts/activate`), Python 3.13.14. Repo path
contains a space — always quote it. The user applies patches at Desktop via `git am`,
then pushes and opens PRs himself. Copilot + CodeRabbit review runs on PRs — their
comments are often real but must each be verified against the actual code.
