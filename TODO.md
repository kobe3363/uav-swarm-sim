# TODO.md — Canonical Outstanding-Task List

**Last updated:** 2026-08-01 · **Test baseline:** 465 green · **Branch:** main

*This file is the single source of truth for WHAT is left to do. For architecture facts, environment rules, working rules, and document hierarchy, read `CLAUDE.md`.*

## 1. Where the project stands
EM-01 (energy-map dynamic RTH) Stages 1–4 are merged. A per-replication Dijkstra cost-to-go grid drives the RTH decision, the return route, and the resume transit. The obstacle-boxing livelock is resolved; the residual is accounted for via `MISSION_PARTIAL` + `skipped_legs`. Sample ceiling moved 92% → 94%.

The technical debt that blocked the thesis result (A1–A3) and the last physics change (B1) are all merged. **The remaining thesis result is C1**: a four-arm A/B of static-threshold RTH vs the dynamic energy map on identical paired seeds.

## 2. Critical path to the thesis result

### A. Technical debt — DONE
- [x] **A1:** `MISSION_PARTIAL` added to `run_output` outcome bucketing. **MERGED (`fd61438`).**
- [x] **A2:** Stage-4 flag-off byte-identity test (`safety.stall_skip`), cross-commit golden. **MERGED (`0409186`).**
- [x] **A3:** Docs sync — static RTH threshold corrected to 0.40, merged B1 + B2 floor propagated. **MERGED (PR #39).**

### B. Last physics change — DONE
- [x] **B1:** §9 battery-zone demotion — under `energy_map.zone_demotion` the CRITICAL guard branch is removed (no new constant, no `battery_zones` value change); TERMINAL is the failsafe. B2 floor sweep chose **0.10** for the Stage-5 map arm. **MERGED (PR #38).**

### C. The thesis result — C1 is the only open critical-path item
- [ ] **C1:** **Stage 5 A/B experiment** (Agent: Fable/Opus High, Plan → author GO; new `run_rth_ab.py`; the long four-arm run is the author's, on Azure). **Four arms on identical paired seeds**, each adding exactly one factor:

  | Arm | slug | EnergyMapConfig | battery_zones | isolates |
  |---|---|---|---|---|
  | 1 | `static40` | absent (all False) | unchanged (guard fires at nominal 0.40) | literature baseline |
  | 2 | `route-only` | `enabled + route` | unchanged | obstacle-aware routing alone |
  | 3 | `decide-route` | `enabled + decide + route` | unchanged | map deciding, static 0.40 still pre-empts |
  | 4 | `full-map` | `enabled + decide + route + zone_demotion` | `critical=0.10` (in-config only) | map as sole normal decider |

  Constants pinned identically across all four arms: `total_reserve_batteries=None`, `stall_skip=true` (orthogonal to the RTH axis — held constant, not an arm difference), `stall_detector=true`, `transit_free_space=true`, `reserve_frac`, `master_seed`. Headline observable: transition-reason inversion (`critical_battery` → `rth_energy`). Success predicate strict (`is MISSION_SUCCESS`); SUCCESS/PARTIAL/INCOMPLETE are three separate rows, never folded into "fail" (the INCOMPLETE→PARTIAL shift is itself a thesis effect). Contrast reading: arm3−arm1 isolates routing, arm4−arm3 isolates decide+demotion. Config `study01_demand.yaml`, 1 km², 100 reps/arm. Caveat: with `stall_skip` ON, arm 1 is "static-40% net + EM-01 skip", not a pristine literature baseline — record in the read-out.

  **Harness delivered** (`run_rth_ab.py` + `tests/integration/test_rth_ab.py`, 15 tests, suite 450 → 465; PR #41 bot review addressed: preset-energy_map fail-fast, resume `--arms` warning, closure binding, gate wording, malformed-record test): four-arm construction via `dataclasses.replace`, shared-`RngFactory` pairing, crash-safe per-rep jsonl partials + `--resume`, strict predicate, three-way outcome tabulation, R1 reason-exclusivity assertion; arm-1 gate vs `run_demand`: record-level equality on all shared fields (real `run_demand` invocation) + byte-identical run signature under `run_demand`-mirrored engine construction. **Remaining: the author's Azure run** — measured serial estimate 32–37 h (arm-1 5-rep probe: median 380.8 s, mean 552.3 s/rep; arms 2–4 ≈ 250–260 s/rep), or run E2 `--jobs` first (~12–16 h). Tiny-world note for the read-out: the arm-1 `critical_battery` dominance is a 1 km²-regime behavior — at smoke-test scale the analytic reserve fires above 0.40 and even arm 1 attributes returns to `rth_energy`; the harness smoke therefore asserts the structural facts and the inversion is measured by the run itself.
- [ ] **C2:** STUDY-01 re-run with the new RTH. Gated by E2 (`--jobs`: 30 h serial → ~8 h).
- [ ] **C3:** Shipped S5 re-run with the new RTH. Gated by E3 (visibility-graph caching).

## 3. Supervisor track
- [ ] **D1:** Fix and send supervisor package (confirm the 0.40 static return threshold everywhere; add EM-01 results, incl. the B2 "willing-map over conservative floor" finding and the "static fractional floor breaks on the scale axis" limitation).
- [ ] **D2:** Obtain K1–K4 decisions (scale experiment scope: area tiers, n-grid ceiling, reps/CI target, clean vs shipped proportion).
- [ ] **D3:** Rewrite `scale_sweep_v2.md` (L-shape only, growing area at fixed edge proportions & the count of fixed-size static obstacles). After D2.

## 4. Old debts (some gate the critical path — see below)
- [ ] **E1:** NOW-03 read-out docs merge (recover from `docs/now-03-s5-readout` branch).
- [ ] **E2:** `--jobs` for the demand path (ENG-09 joblib pattern). **Gates C2** (STUDY-01 re-run: 30 h serial → ~8 h) and can also shorten C1 (measured 32–37 h serial → ~12–16 h) if run before it.
- [ ] **E3:** Visibility-graph caching (perf fix for obstacle runs). **Gates C3** (shipped re-run pays `route_transit` 347–791 s/rep otherwise).

## 5. Engineering backlog
- Variantas C (deterministic candidate grid for launch optimizer).
- ENG-01 turn aero penalty (must be done BEFORE any final re-runs or not at all — it re-baselines energy).
- ENG-13 `run_scale_tiers` retrofit into `RunContext` (structural; does not consume the superseded `runs/run_scale_tiers` data).
- ADV-03 redistribution ablation-fidelity fix.
- B7 PyVRP baseline.
- Test-suite restructuring stages 2–3 (stage 1 already merged).
- `docs/architecture.md` (execution order, CLI map, config surface).
- Launch story Phase 2 (per-algo launch co-optimization); footprint-aware launch clearance.
