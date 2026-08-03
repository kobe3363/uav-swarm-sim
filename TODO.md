# TODO.md — Canonical Outstanding-Task List

**Last updated:** 2026-08-03 · **Test baseline:** 500 green (+1 documented skip) · **Branch:** main

*This file is the single source of truth for WHAT is left to do. For architecture facts, environment rules, working rules, and document hierarchy, read `CLAUDE.md`.*

## 1. Where the project stands
EM-01 (energy-map dynamic RTH) Stages 1–4 are merged. A per-replication Dijkstra cost-to-go grid drives the RTH decision, the return route, and the resume transit. The obstacle-boxing livelock is resolved; the residual is accounted for via `MISSION_PARTIAL` + `skipped_legs`. Sample ceiling moved 92% → 94%.

The technical debt that blocked the thesis result (A1–A3) and the last physics change (B1) are all merged. **C1 and C2 are measured**; **C3** (the shipped decomposition re-run under the new RTH) is the last thesis result, run in progress on Azure.

## 2. Critical path to the thesis result

### A. Technical debt — DONE
- [x] **A1:** `MISSION_PARTIAL` added to `run_output` outcome bucketing. **MERGED (`fd61438`).**
- [x] **A2:** Stage-4 flag-off byte-identity test (`safety.stall_skip`), cross-commit golden. **MERGED (`0409186`).**
- [x] **A3:** Docs sync — static RTH threshold corrected to 0.40, merged B1 + B2 floor propagated. **MERGED (PR #39).**

### B. Last physics change — DONE
- [x] **B1:** §9 battery-zone demotion — under `energy_map.zone_demotion` the CRITICAL guard branch is removed (no new constant, no `battery_zones` value change); TERMINAL is the failsafe. B2 floor sweep chose **0.10** for the Stage-5 map arm. **MERGED (PR #38).**

### C. The thesis result — C1 is the only open critical-path item
- [x] **C1:** **Stage 5 A/B experiment** (Agent: Fable/Opus High, Plan → author GO; new `run_rth_ab.py`; the long four-arm run is the author's, on Azure). **Four arms on identical paired seeds**, each adding exactly one factor:

  | Arm | slug | EnergyMapConfig | battery_zones | isolates |
  |---|---|---|---|---|
  | 1 | `static40` | absent (all False) | unchanged (guard fires at nominal 0.40) | literature baseline |
  | 2 | `route-only` | `enabled + route` | unchanged | obstacle-aware routing alone |
  | 3 | `decide-route` | `enabled + decide + route` | unchanged | map deciding, static 0.40 still pre-empts |
  | 4 | `full-map` | `enabled + decide + route + zone_demotion` | `critical=0.10` (in-config only) | map as sole normal decider |

  Constants pinned identically across all four arms: `total_reserve_batteries=None`, `stall_skip=true` (orthogonal to the RTH axis — held constant, not an arm difference), `stall_detector=true`, `transit_free_space=true`, `reserve_frac`, `master_seed`. Headline observable: transition-reason inversion (`critical_battery` → `rth_energy`). Success predicate strict (`is MISSION_SUCCESS`); SUCCESS/PARTIAL/INCOMPLETE are three separate rows, never folded into "fail" (the INCOMPLETE→PARTIAL shift is itself a thesis effect). Contrast reading: arm3−arm1 isolates routing, arm4−arm3 isolates decide+demotion. Config `study01_demand.yaml`, 1 km², 100 reps/arm. Caveat: with `stall_skip` ON, arm 1 is "static-40% net + EM-01 skip", not a pristine literature baseline — record in the read-out.

  **Harness delivered** (`run_rth_ab.py` + `tests/integration/test_rth_ab.py`, 15 tests, suite 450 → 465; PR #41 bot review addressed: preset-energy_map fail-fast, resume `--arms` warning, closure binding, gate wording, malformed-record test): four-arm construction via `dataclasses.replace`, shared-`RngFactory` pairing, crash-safe per-rep jsonl partials + `--resume`, strict predicate, three-way outcome tabulation, R1 reason-exclusivity assertion; arm-1 gate vs `run_demand`: record-level equality on all shared fields (real `run_demand` invocation) + byte-identical run signature under `run_demand`-mirrored engine construction. DONE — four-arm run complete, read-out `docs/reports/c1_stage5_ab_readout.md`. One-line result: reason-attribution inversion (critical_battery→rth_energy); energy −5.5% / makespan −12.9% / swaps −3 significant; success 95→98% NOT significant (equivalent success at lower cost); zone_demotion regresses 2 reps (real trade-off).
- [x] **C2:** STUDY-01 re-run under the new (arm-4) RTH — **DONE** (`spares_c2_newrth`, 500 reps, `master_seed=42`; read-out `docs/reports/c2_study01_readout.md`). Config-hash + command verified arm-4 "full-map"; supersedes the pre-EM-01 STUDY-01 numbers (`runs/spares_final_demand`, `30f4209`). **95% knee = 6 (Wilson-certified); 99% is structurally unreachable** — a 1.6% residual `MISSION_INCOMPLETE` (8/500) caps `success_frac ≤ 0.984` for any spare count, so 99% is an INCOMPLETE-cause problem, not a spare problem. Demand tightly concentrated at B=5 (467/492). C1↔C2 cross-note (`docs/reports/c1_c2_cross_note.md`): reps 76 & 100 fail in BOTH experiments (same seed) — the deeper-sortie cost of `zone_demotion` is a systematic, quantified trade-off (~1.6–2% of worlds), not noise.
- [ ] **C3:** Shipped S5 re-run with the new RTH. Config frozen `config/shape_sweep_newrth.yaml` (#49); full shipped run in progress on Azure (probe 5:47 → ~8h40 est).

## 3. Supervisor track
- [ ] **D1:** Fix and send supervisor package (confirm the 0.40 static return threshold everywhere; add EM-01 results, incl. the B2 "willing-map over conservative floor" finding and the "static fractional floor breaks on the scale axis" limitation).
- [ ] **D2:** Obtain K1–K4 decisions (scale experiment scope: area tiers, n-grid ceiling, reps/CI target, clean vs shipped proportion).
- [ ] **D3:** Rewrite `scale_sweep_v2.md` (L-shape only, growing area at fixed edge proportions & the count of fixed-size static obstacles). After D2.
  - **D3 machinery — DELIVERED (not a D2/D3 check-off):** `experiments/run_area_obstacle_sweep.py` (branch `area-obstacle-sweep`) sweeps AREA (L-shape family regenerated per `--areas`) × obstacle COUNT (`--densities`, fixed size via `--obstacle-size-m` ⇒ `size_range=[S,S]`) × `n`, composing the S5 primitives (metric/contrast/regime semantics frozen by import). Every K1–K4 quantity is a CLI flag. Byte-identity gates green: equivalence cell reproduces `run_shape_sweep` **shipped** l_shape bitwise (incl. fixture-parity HARD STOP against the on-disk fixture), serial↔spawn identical, rep-prefix identical. This is the **runner D3 will drive** — **D2 and D3 stay open**: D2 = the K1–K4 supervisor decisions, D3 = the `scale_sweep_v2.md` rewrite (still gated on D2).

## 4. Old debts (some gate the critical path — see below)
- [x] **E1:** NOW-03 read-out docs merge — MERGED (#48); reports `docs/reports/s5_shipped_readout.md` + `docs/reports/s5_clean_readout.md` (delivered WITHOUT a thesis verdict — preserve that boundary; both carry a provenance reconstruction note).
- [x] **E2:** `--jobs` for the demand paths (`run_rth_ab` + `run_spare_sizing --demand-mode`), ENG-09 spawn-pool pattern via a shared `experiments/_parallel.py`. **DELIVERED (patch, branch `e2-jobs-demand-paths`).** Byte-identity gate green for BOTH scripts — serial `--jobs 1` vs spawn `--jobs 2/3` bitwise-identical on every physics field (wall-clock excluded); single-writer-in-parent keeps the crash-safe partials + `--resume`; BLAS/OpenMP thread pin added to both entrypoints (the load-bearing cause of ENG-09 determinism). Suite 465 → 473 green (+1 documented skip: the pin-load-bearing guard is inert on the smoke sim). Realised speedup is core-bound (`--jobs auto` = physical cores − 1); the tiny-world probe cannot exhibit it (per-rep ≈0.17 s → spawn overhead dominates, ~1.1×), but the real per-rep cost (250–380 s) makes it near-linear in workers — the h estimates below stay ESTIMATES until a real-scale run measures them. **Gates C2**; can also shorten C1.
- [x] **E3:** Visibility-graph caching for `route_transit`. **DELIVERED (patch, branch `e3-visibility-graph-cache`; post-merge contract correction on `codex/e3-cache-contract`).** The endpoint-independent O(V²) obstacle-vertex visibility result is memoised on a per-replication engine dict, keyed by `(sha1(obs.wkb), sha1(region.wkb))`, and spliced into each `(a,b)` query byte-identically (cache consulted purely by coordinate-key pair, never by node index → the a/b-on-vertex dedup shift is irrelevant; forced-collision seam test proves it). Only the True (visible) edge set is stored → O(visible), not O(V²). The dense in-process cached-vs-uncached gate avoids a platform-specific pinned GEOS hash; direct get-or-build and serial/spawn cache-exercised gates prove the cache is populated and reused. Measured on that rep: **V=1349, 168,540 visible edges cached (≈ tens of MB/worker), 5.88× speedup (898→153 s)**; also shortens the `test_transit_livelock` B1 leg the same way. Per-worker under `--jobs` (spawn), E2 determinism gate still green. Suite 474 → 482 green (+1 skip; note: the previously documented 473 baseline measured 474 pre-E3 on this box). **Unblocks C3.**

## 5. Engineering backlog
- Variantas C (deterministic candidate grid for launch optimizer).
- ENG-01 turn aero penalty (must be done BEFORE any final re-runs or not at all — it re-baselines energy).
- ENG-13 `run_scale_tiers` retrofit into `RunContext` (structural; does not consume the superseded `runs/run_scale_tiers` data).
- ADV-03 redistribution ablation-fidelity fix.
- B7 PyVRP baseline.
- Test-suite restructuring stages 2–3 (stage 1 already merged).
- `docs/architecture.md` (execution order, CLI map, config surface).
- Launch story Phase 2 (per-algo launch co-optimization); footprint-aware launch clearance.
