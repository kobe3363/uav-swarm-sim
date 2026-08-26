# PROJECT_HISTORY — the complete record of pre-reset work

**Status:** ARCHIVE. This document is closed. It records the state of the project as of
commit `493e8d2`, immediately before the supervisor fixed a new main-experiment spec that
supersedes the regime every experiment below was run in.

---

## 1. Scope note

### 1.1 What this document replaces

This file is the single surviving record of the documents retired in the same branch
(`docs/archive-compact`, commit 2). It replaces:

`PROJECT_GUIDE.md` · `ADVERSARIAL_TESTER_PROMPT.md` · `mission_analyst_prompt.md` ·
`docs/archive/TODO_legacy.md` · `docs/reports/c1_stage5_ab_readout.md` ·
`docs/reports/c2_study01_readout.md` · `docs/reports/c1_c2_cross_note.md` ·
`docs/reports/c3_shipped_newrth_readout.md` · `docs/reports/s5_shipped_readout.md` ·
`docs/reports/s5_clean_readout.md` · `docs/reports/d1_supervisor_package.md` ·
`docs/proposals/energy_map_rth.md` · `docs/proposals/scale_sweep_v2.md` ·
`config/study01_demand_newrth.yaml` · `config/shape_sweep_newrth.yaml`

The full text of every one of them is preserved in git and is retrievable (§8).

### 1.2 Retrieval anchor

```text
493e8d29b090785ad073c5939aa473c209747b88
493e8d2 Add comprehensive CLI and config reference map
```

To retrieve any retired file in full, use the **commit SHA** — it is immutable and always resolvable:

```bash
git show 493e8d2:docs/reports/c1_stage5_ab_readout.md
```

Optionally the author may create a readable alias for it (this tag is **not** created by
the archive commits, so verify it exists before relying on it):

```bash
git tag -a pre-reset-archive 493e8d29b090785ad073c5939aa473c209747b88 -m "Full documentation set before the ARCHIVE-COMPACT-01 retirement"
git show pre-reset-archive:docs/reports/c1_stage5_ab_readout.md   # once the tag exists
```

Substitute any path from the index in §8. Nothing in this archive is the only copy of
anything — it is a distillation, and the originals remain one command away.

### 1.3 What is gone and is not coming back

The raw run data (`runs/**` — per-replication records, `results.json`, `contrasts.csv`,
`shape_sweep.csv`, `plan.json`) was left on Azure VMs or deleted. It is **not** recoverable
and the author does not want it recovered. Consequently **every number in this document is
quoted from a committed read-out, not recomputed from data.** Each carries its `file:line`
and the commit SHA of the report it came from.

### 1.4 The rule this document was written under

> Measured numbers and results are quoted **verbatim** from their source and never
> corrected, rounded, re-derived, recomputed or "improved". Factual **descriptions of the
> system** are written from the **code**; any document that contradicted the code has been
> corrected here, with the drift recorded in §7.

Where a source's number is ambiguous, it is quoted as written and marked **[AMBIGUOUS]**.
Where a figure was the author's own diagnosis rather than a repo artifact, its source
labelled it so, and that label is carried through as **[AUTHOR DIAGNOSIS]**.

---

## 2. System summary

Enough to read §3 without ever having seen `PROJECT_GUIDE.md`.

### 2.1 What the simulator is

A discrete-time, Monte-Carlo simulation of a **homogeneous** fleet of identical
reconnaissance UAVs covering a bounded survey area in the presence of obstacles, under hard
energy limits. It exists to test one idea — **energy-weighted spatial decomposition**:
divide the survey area among drones so each drone's patch is sized in proportion to its
*momentary* battery level, on a topological-graph representation of the free space — and to
characterise the result both deterministically and as a Semi-Markov process.

One replication: ingest a GeoJSON boundary, scatter synthetic obstacles, build the GVG
skeleton and condense it into TGC free-space regions, choose a launch site outside the
survey polygon, partition the regions among drones, plan a boustrophedon coverage path per
zone, fly every drone forward in `dt` steps under the behavioural automaton draining energy
as `power × time`, record every state interval, then compute deterministic metrics and the
stationary distribution.

**Energy is always `E = Σ P(maneuver)·dt`.** There is no per-distance shortcut anywhere in
the model. Many design choices exist only to protect that invariant.

### 2.2 The five architectural tiers

Lower tiers never import higher ones.

| Package | Responsibility |
|---|---|
| `infrastructure/` | Typed `Config` + YAML loader + `config_hash`; the seeded `RngFactory`; `core_types` (`Pose`, `Path`, `Region`, `Zone`, `Partition`, `MissionResult`); the `SimulationEngine` orchestrator; all visualization |
| `physical_model/` | Grey-box component energy model; Dubins kinematics; the `MotionModel` platform abstraction (`DubinsModel` for FW/VTOL, `HolonomicModel` for multirotor); formation aero correction; `Battery`; 1-D vertical takeoff/landing segments |
| `planning/` | GeoJSON parsing; Poisson obstacle generation; `EnvironmentMap`; GVG construction; TGC regions; the weighted decomposition (the contribution) and its three peers; the energy cost-to-go map; launch-site optimizer; boustrophedon coverage paths; the visibility router |
| `execution/` | The behavioural automaton; agent and fleet; the event bus; the dynamic RTH calculator; event-driven redistribution; proactive safety monitor; formation manager; battery-swap station with a finite shared pool; hazard-rate failure model |
| `metrics/` | State-history recording; deterministic mission metrics; SMDP estimation; the stationary distribution with the embedded-to-time-weighted correction; the efficiency score; Monte-Carlo with CI convergence; structured run output; telemetry |

The single conductor is `infrastructure/simulation_engine.py`. The `experiments/run_*.py`
scripts are thin CLI front doors; `docs/cli_map.md` is their reference.

### 2.3 The behavioural automaton — EIGHT states

`AgentState` (`src/uav_swarm_sim/infrastructure/enums.py:19-45`) has **eight** members:

| State | Meaning | Airborne? |
|---|---|---|
| `S0_IDLE` | On the ground at the launch site | no |
| `S1_TRANSIT` | Flying to/from the assigned zone | yes |
| `S2_MISSION` | Sweeping a coverage strip, **camera ON — the only productive state** | yes |
| `S3_RTH` | Returning to home | yes |
| `S_SWAP` | Battery swap at the ground station — costs TIME, **zero energy** | no |
| `S_OBS` | Obstacle-avoidance maneuver | yes |
| `S_FAIL` | Lost, removed from the active fleet | no |
| `S_FERRY` | Repositioning between coverage strips, **camera OFF** — non-productive but airborne | yes |

Coverage legs are boustrophedon and **structurally** parity-indexed: even `_cov_idx` =
COVERAGE (`S2_MISSION`), odd = TURN connector (`S_FERRY`). The connector is a structural
consequence of odd parity, not a per-segment maneuver decision.

`S_FERRY` is inside `AgentState.is_airborne` (`enums.py:38-45`), so ferry legs consume
flight energy and carry failure-hazard exposure. This is load-bearing — see §7.2.

**Coverage area is not the flyable area.** A drone may fly outside the survey polygon
whenever it is not in `S2_MISSION`.

### 2.4 The four decomposition peers

| Algorithm | Kind |
|---|---|
| `classic_voronoi` | Plain nearest-seed Euclidean Voronoi; ignores battery and obstacle topology |
| `kmeans` | Position k-means plus greedy drone-to-cluster assignment on a flight-cost matrix |
| `tgc_basic` | Unweighted topological decomposition — the ablation twin |
| `weighted_voronoi` | **Battery-weighted TGC — the central contribution** |

`tgc_basic` is a class (`TgcBasicDecomposer`) inside `planning/weighted_decomposition.py`,
deliberately next to its weighted twin so the two cannot drift apart.

### 2.5 THE CRITICAL NULL

For a homogeneous fleet (identical drones, `battery_frac = 1.0`) with hazard rate λ = 0,

> **`weighted_voronoi ≡ tgc_basic` BYTE-IDENTICALLY.**

Equal battery fractions produce an identical partition *by construction*. The weighting
differentiates **only** when batteries have diverged — a heterogeneous fleet, or a
post-failure redistribution at λ > 0. Every clean full-battery experiment therefore
reproduces this null exactly, and does so as a *correctness check*, not as a failure.
§4.7 records the measurements.

Redistribution today always routes through `WeightedTgcDecomposer` — an ablation-fidelity
issue, tracked historically as ADV-03.

### 2.6 The paired-seed contract — the methodological cornerstone

`RngFactory.stream(name, replication)` is a pure function of
`(master_seed, name, replication)`. The same `master_seed` yields bitwise-identical results.
A shared `RngFactory` across compared arms guarantees each arm sees the *same* environment
and the *same* failure draws at the same replication index — so any metric difference is
attributable to the algorithm, never to the noise. Every comparison below is a **paired
contrast between two named arms**.

### 2.7 The RTH threshold — read this before §3

`Battery.zone` classifies `critical ≤ f < nominal` as `CRITICAL`, i.e. `[0.20, 0.40)`.
Therefore the `critical_battery` return guard fires at **nominal = 0.40**, not at 0.20;
0.20 is merely where `TERMINAL` begins. **The static RTH threshold is 0.40.** Under
`rth.energy_map.zone_demotion` the `CRITICAL` branch is removed entirely and the dynamic
cost-to-go map governs, with `TERMINAL` as the sole failsafe.

Every arm-1 baseline in §3 is a "static 40 % net", never a "static 20 % net".

### 2.8 Other load-bearing facts

- **Camera energy** is `sensor.sensor_power_w`, charged **only** over COVERAGE segments.
  The RTH lookahead mirrors execution exactly.
- **Energy-unit discipline:** `E_home` uses CRUISE; the next-bundle term uses
  COVERAGE + camera. The two are never mixed.
- **Drone deployment:** drones ring at `deploy_poses[i]` at radius R for takeoff only; all
  return to a single `launch_pose`, so one `E_home` network serves the whole swarm.
- **Regime classification:** the per-drone max-zone ratio is PRIMARY. The pooled
  `E_cover/(n·B_usable)` is only a lower bound — averages hide imbalance.
- **`efficiency`** is the SMDP **throughput** ratio, **NOT** energy efficiency. It is
  orthogonal to energy and balance, and that orthogonality explains most of §3's
  "efficiency favours the peer" rows.
- **kmeans init variance** (`STREAM_KMEANS_INIT`, replication-keyed) is a legitimate
  algorithm characteristic, reported as a finding, never pinned away.
- **`S_SWAP`** costs TIME but ZERO energy; the ground queue wait counts as swap-state time.

---

## 3. Experiment record

Eight subsections, one per experiment. Each gives: the question → the setup → the headline
numbers **copied verbatim with their CIs** → the verdict → the status under the new
supervisor spec.

**Status vocabulary.** *HISTORY ONLY* = the numbers describe a regime the new spec does not
reproduce; keep for the method chapter, never cite as main-experiment evidence.
*SUPERSEDED* = a later experiment in this same archive replaced it. *STILL VALID* = the
design, mechanism or invariant carries forward unchanged.

**Source SHAs.** `c1_stage5_ab_readout.md` = `d54f652`; `c2_study01_readout.md` and
`c1_c2_cross_note.md` = `81a9226`; `c3_shipped_newrth_readout.md` = `f97de54`;
`s5_shipped_readout.md` and `s5_clean_readout.md` = `2fcd473`;
`d1_supervisor_package.md` = `3add399`; `energy_map_rth.md` and `scale_sweep_v2.md` =
`e025088`; `TODO_legacy.md` = `9159ca6`.

---

### 3.1 C1 — Stage-5 four-arm RTH A/B

*Source: `docs/reports/c1_stage5_ab_readout.md` @ `d54f652`.*

**Question.** Does a dynamic, distance-aware energy-map RTH beat a static battery-fraction
threshold, on identical paired seeds?

**Setup.** Run `runs/c1_rth_ab/rth_ab_2026-08-02-08-55-47_7a99a3`; four arms × **100 paired
replications**; `master_seed=42`; `config=study01_demand.yaml`; 1 km²; obstacles present;
harness `run_rth_ab.py`; `--jobs auto`, byte-identical to serial (c1:3-6).

| Arm | slug | energy_map flags | terminal_floor |
|---|---|---|---|
| 1 | static40 | all off | 0.20 |
| 2 | route-only | enabled + route | 0.20 |
| 3 | decide-route | enabled + decide + route | 0.20 |
| 4 | full-map | enabled + decide + route + zone_demotion | **0.10** |

Constants pinned identically across all four arms (verified from the four `plan.json`):
`stall_skip=true`, `stall_detector=true`, `transit_free_space=true`, `reserve_frac=0.05`,
unbounded pool, `n_drones=5`. `config_hash` is byte-identical across all four arms despite
different flags, so the environment is shared per replication and **paired seeds hold**
(c1:18-21).

#### Headline — the reason-attribution inversion

Transition reasons summed over 100 reps per arm (c1:29-34):

| Arm | rth_energy | critical_battery | terminal_battery |
|---|---|---|---|
| 1 static40 | 0 | 746 | 0 |
| 2 route-only | 0 | 767 | 0 |
| 3 decide-route | 0 | 767 | 0 |
| 4 full-map | **496** | **0** | 0 |

Arm 1 returns are governed **100 % by the static net**; arm 4 returns are governed **100 %
by the dynamic map**, with `critical_battery=0` and `terminal_battery=0` (c1:36-38).

Arms 2 and 3 are **byte-identical to each other** on every per-replication field, even
though arm 3 actively consults the map (`n_map_hits` mean **321.71/rep** vs 0 in arm 2).
**[AMBIGUOUS]** — the source asserts both halves of that sentence without reconciling
them, and never enumerates the compared field set: `n_map_hits` demonstrably differs
between the two arms, so “every per-replication field” cannot be literal. The natural
reading is that map/route telemetry counters are excluded and the claim covers the
physics and outcome fields, but the source does not say so. Quoted as written (c1:40-45,
restated at c1:104-107); deciding the field set would be a metric-semantics call (§6.8).
The static 0.40 net fully pre-empts the map while `zone_demotion=false`, so the contrast
decomposes as **arm3 − arm1 = routing**, **arm4 − arm3 = decide+demotion** (c1:40-45).

#### The result sentence, quoted verbatim (c1:55-58)

> The dynamic energy-map RTH (arm 4) achieves **equivalent mission success** to the static
> 40 % threshold (98 % vs 95 %, difference not significant), while cutting **total energy by
> 5.5 %**, **makespan by 12.9 %**, and **battery swaps by ~3 per mission** — all strongly
> significant on paired contrasts.

"Equivalent safety at lower cost" is the honest and stronger claim; "higher success" is not
supported at this sample size (c1:60-61).

#### Outcome breakdown — three rows, never folded (c1:67-72)

| Arm | SUCCESS | PARTIAL | INCOMPLETE | FAILED | success_frac | Wilson 95 % CI |
|---|---|---|---|---|---|---|
| 1 static40 | 95 | 5 | 0 | 0 | 0.950 | [0.888, 0.978] |
| 2 route-only | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |
| 3 decide-route | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |
| 4 full-map | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |

Aggregate over all 400 replications: SUCCESS 389, PARTIAL 11, INCOMPLETE 0, FAILED 0
(c1:75-76).

#### Paired contrasts

Method: paired bootstrap, 10 000 resamples, paired by replication index; report-level seed
20260802, unrelated to `master_seed`. "n" for demand = complete-case, both arms strict
SUCCESS (c1:83-87).

**arm2 − arm1 — ROUTING added** (c1:91-97)

| Metric | arm1 | arm2 | Δ | 95 % CI |
|---|---|---|---|---|
| success_frac | 0.950 | 0.980 | +0.030 | [0.000, +0.070] |
| coverage_frac | 0.99964 | 0.99987 | +0.00022 | [0, +0.00050] |
| total_energy_j | 2,534,971 | 2,573,563 | **+38,592** | [+26,759, +50,592] |
| duration_s | 3047.5 | 3086.1 | +38.6 | [−10.8, +90.1] (ns) |
| demand (n=95) | — | — | +0.12 (med 0) | [−0.05, +0.29] (ns) |

**arm3 − arm2 — DECIDE added.** Every per-replication field byte-identical; Δ = 0, CI [0,0]
on all metrics (c1:104-107).

**arm4 − arm3 — ZONE_DEMOTION added** (c1:111-117)

| Metric | arm3 | arm4 | Δ | 95 % CI |
|---|---|---|---|---|
| success_frac | 0.980 | 0.980 | 0.000 | [−0.040, +0.040] |
| coverage_frac | 0.99987 | 0.99973 | −0.00014 | [−0.00060, +0.00023] (ns) |
| total_energy_j | 2,573,563 | 2,396,136 | **−177,427 (−6.9 %)** | [−198,411, −156,518] |
| duration_s | 3086.1 | 2655.0 | **−431.2 (−14.0 %)** | [−505.6, −353.6] |
| demand (n=96) | — | — | **−2.78 (med −3)** | [−3.02, −2.56] |

**arm4 − arm1 — FULL effect** (c1:125-131)

| Metric | arm1 | arm4 | Δ | 95 % CI |
|---|---|---|---|---|
| success_frac | 0.950 | 0.980 | +0.030 | [−0.010, +0.080] (**ns — crosses zero**) |
| coverage_frac | 0.99964 | 0.99973 | +0.00008 | [−0.00031, +0.00048] (ns) |
| total_energy_j | 2,534,971 | 2,396,136 | **−138,835 (−5.5 %)** | [−160,094, −117,899] |
| duration_s | 3047.5 | 2655.0 | **−392.5 (−12.9 %)** | [−463.3, −320.7] |
| demand (n=94) | — | — | **−2.66 (med −3)** | [−2.90, −2.41] |

#### Secondary metrics

**Sortie depth** — `return_depths`, battery fraction at the return decision, pooled
(c1:139-144):

| Arm | n | min | median | mean |
|---|---|---|---|---|
| 1 | 746 | 0.3977 | 0.3999 | 0.3999 |
| 2 | 767 | 0.3977 | 0.3999 | 0.3998 |
| 3 | 767 | 0.3977 | 0.3999 | 0.3998 |
| 4 | 496 | 0.1077 | 0.1964 | 0.1947 |

Arms 1–3 return exactly at the 0.40 static net. Arm 4 flies to ~0.195 mean; **the 0.10
floor is never reached** (min 0.1077), so `terminal_battery = 0` (c1:146-149).

**Demand (swaps, strict SUCCESS)** (c1:151-152): arm1 min 5 / med 8 / mean 7.61 / max 12;
arms 2 and 3 min 5 / med 8 / mean 7.74 / max 13; arm4 min 4 / med 5 / mean 4.97 / max 6.

**Map/route counters, sum / mean-per-rep** (c1:154-158): arm1 0/0/0; arm2
`n_route_fallbacks` 145/1.45; arm3 `n_map_hits` 32171/321.71, `n_map_fallbacks` 2862/28.62,
`n_route_fallbacks` 145/1.45; arm4 `n_map_hits` 30411/304.11, `n_map_fallbacks` 2513/25.13,
`n_route_fallbacks` 67/0.67. All map fallbacks are arming-bound, not decide.

#### The quantified cost — `zone_demotion` regresses 2 replications

Non-monotonic across the pipeline on the 6 reps that are ever PARTIAL (c1:173-180):

| rep | arm1 | arm2 | arm3 | arm4 |
|---|---|---|---|---|
| 71 | PARTIAL | SUCCESS | SUCCESS | SUCCESS |
| 76 | SUCCESS | SUCCESS | SUCCESS | **PARTIAL** |
| 79 | PARTIAL | PARTIAL | PARTIAL | SUCCESS |
| 88 | PARTIAL | PARTIAL | PARTIAL | SUCCESS |
| 98 | PARTIAL | SUCCESS | SUCCESS | SUCCESS |
| 100 | PARTIAL | SUCCESS | SUCCESS | **PARTIAL** |

Reps 76 and 100: the deeper sorties over-commit and miss coverage — `coverage_frac`
**0.9887** and **0.9838**. The source calls this "a real trade-off, not noise" (c1:182-186).

**Verdict.** The inversion is complete and the efficiency win is strongly significant; the
success improvement is not. The win is entirely attributable to `zone_demotion`, not to
routing — routing alone *raises* energy.

**Limitations, quoted (c1:188-193).** Arm 1 is "static-40 % net + `stall_skip`", not a
pristine literature baseline. 1 km² only; the 4–16 km² scale axis is unvalidated and is
where the fractional floor and the deeper-sortie margin would be stressed first. This is a
paired **contrast**, not a Wilson-certified proportion. Difference CIs are paired bootstrap,
a disclosed methodological choice over a closed-form Wilson-difference.

**Status under the new spec: HISTORY ONLY (numbers) + STILL VALID (design).**
The headline bundles a −3 swaps/mission term, and the new spec has no swaps at all; the run
also used an unbounded pool at 1 km². The *numbers* therefore do not transfer. The **arm
ladder** and the **`critical_battery` → `rth_energy` reason-inversion observable** are the
direct blueprint for the new spec's dynamic-RTH-reserve contribution and carry forward
unchanged.

---

### 3.2 C2 — STUDY-01 spare-sizing re-run under the new RTH

*Source: `docs/reports/c2_study01_readout.md` @ `81a9226`.*

**Question.** Under the arm-4 RTH, how many spare battery packs are needed to hit a 95 % and
a 99 % mission-success target?

**Setup.** Run `spares_c2_newrth/run-2026-08-02-19-13-12/`; **500 replications**;
demand-mode; `master_seed=42`; `config=study01_demand_newrth.yaml` (arm-4 "full-map");
1 km²; `--jobs auto` (c2:3-6).

**Run validity (c2:12-21).** 500/500 complete, indices 1..500, no gaps or duplicates,
`master_seed=42` on every record. New RTH confirmed: `config_hash = 2d9f954a4b…964606`
matches a fresh `load_config("config/study01_demand_newrth.yaml")`; `run.json.command`
records the config; it resolves to arm-4. **Honest limitation:** telemetry was OFF, so the
records carry no per-return `rth_reason`/`n_map_hits`. Map-governance is established by
**config identity, not decision-level attribution**.

#### Headline — spare-count knees (c2:27-30)

| Target | empirical knee | Wilson-certified knee | status |
|---|---|---|---|
| 0.95 | 5 | **6** | Wilson-certified (500 ≥ 73-rep floor) |
| 0.99 | — | — | **structurally unreachable** |

Recomputed `P(success | B)` directly from the 500 records (demand ≤ B):
`n_le = [0,0,0,0,16,483,491,492]` at `B=0..7` — matches `results.json.cdf`/`knees` exactly.
500 reps clears both Wilson floors (381 for 99 %, 73 for 95 %) (c2:32-36).

#### The load-bearing finding (c2:42-48)

**8/500 replications (1.6 %) never succeed at any B** — demand = ∞, all
`MISSION_INCOMPLETE`. This sets a hard ceiling `success_frac ≤ 0.984` for **every** battery
count. Since 0.984 < 0.99, **no finite (or infinite) spare pool reaches the 99 % target.**
99 % requires removing the INCOMPLETE cause, which is a different problem from spare-sizing.

#### Demand distribution — 492 successful reps (c2:53-58)

min 4, median 5, mean ≈ 4.99, max (finite) 7. Highly concentrated:
demand=4 → 16, **demand=5 → 467 (dominant)**, demand=6 → 8, demand=7 → 1.
Never-succeed fraction 8/500 = 1.6 %. The CDF rises near-vertically to 0.966 at B=5, then
approaches the ~0.984 ceiling flatly at B=6–7.

#### Outcome mix, 500 reps (c2:68-75)

| Outcome | count | frac |
|---|---|---|
| SUCCESS | 492 | 98.4 % |
| PARTIAL | 0 | 0.0 % |
| INCOMPLETE | 8 | 1.6 % |
| FAILED | 0 | 0.0 % |

`success_frac = 0.984`, Wilson 95 % CI ≈ **[0.9687, 0.9919]**.

**Supersession, quoted (c2:81-84).** These knees and CDF **replace** the pre-EM-01 STUDY-01
numbers (`runs/spares_final_demand`, commit `30f4209`, old static 0.40 RTH). Any comparison
must label the old numbers "superseded (pre-EM-01)" and frame the result as "the new-RTH
re-run gives X", never "X beats the old Y".

**Notable (c2:90-94).** `analytical_prior_spares = 1` (formula `E_cover/B_usable − n +
margin`) is far below the empirical knee of 5–6. `formula_validation.verdict =
"inconclusive"`, computed only for the 0.99 target whose empirical knee is null. Whether the
gap is a formula artifact or an RTH-model artifact is **an open question — no verdict was
formed**. See §7.4.

**Verdict.** 95 % has a clean certified knee of 6. 99 % is not a spare-sizing question at
all under this RTH.

**Status under the new spec: SUPERSEDED / OUT OF SCOPE.** The entire experiment sizes a
finite spare-battery pool. The new spec has neither swaps nor a pool, so no knee transfers.
What survives is *method*: the demand-mode equivalence `success(k, B) ⇔ D_k ≤ B` (one
unbounded batch reconstructs the whole success-vs-B curve in `O(reps)` instead of
`O(|grid| × reps)`), and the reasoning pattern that a residual-INCOMPLETE ceiling makes a
target structurally unreachable regardless of the resource being sized.

---

### 3.3 C1 ↔ C2 cross-check — the same residual-failure seeds

*Source: `docs/reports/c1_c2_cross_note.md` @ `81a9226`. No new run; all numbers trace to
the C1 and C2 read-outs.*

**Question.** Is the arm-4 residual failure a systematic mechanism or sampling noise?

**Setup.** Both experiments ran arm-4 under `master_seed=42`, so their low replication
indices share the same worlds (x-note:3-5).

**The overlap (x-note:9-12).**

- **C1 arm-4** failed, as `MISSION_PARTIAL`, on replications **76 and 100**.
- **C2** failed, as `MISSION_INCOMPLETE` with demand = ∞, on replications **76, 100**, 109,
  151, 162, 191, 313, 396 — 8/500 = 1.6 %.

Replications **76 and 100 fail in both experiments**: same seed, same world, same underlying
cause. The outcome *label* differs by run mode (PARTIAL in the A/B harness vs INCOMPLETE in
the demand runner) but the physical mechanism is one (x-note:14-17).

**Verdict, quoted (x-note:22-24).** "This is not sampling noise — it recurs on the *same
seeds* across two independent experiments at two sample sizes." The map trades ~1.6–2 % of
worlds into residual failure for the efficiency gain on the rest.

A concrete follow-up was recorded and explicitly deferred (x-note:35-39): inspect what makes
seeds 76/100 over-commit, and whether a small margin adjustment on the map's decide
threshold recovers them without sacrificing the energy win. Never executed.

**Status under the new spec: HISTORY ONLY.** It rests on both C1 and C2. What survives is
the **methodological lesson**: recurrence on the *same seeds* across two independent
experiments at two sample sizes is what distinguishes a systematic mechanism from noise —
and paired-seed determinism is what makes that test possible at all.

---

### 3.4 C3 — shipped shape sweep under the new (arm-4) RTH

*Source: `docs/reports/c3_shipped_newrth_readout.md` @ `f97de54`.*

**Question.** Under the new RTH, does the decomposition choice separate TGC from its peers,
and does the shape/fleet-size axis matter?

**Setup.** Run folder `runs/shape_sweep_shipped_newrth/` (`run_id=231d6da1ed67`), run at git
commit **`81a9226`**, `config/shape_sweep_newrth.yaml` (arm-4 "full-map"),
`--mode shipped --budget full --jobs auto`, `master_seed=42`, Linux/py3.12, **wall time
29,737 s (≈8.26 h)**. Grid: **9 shapes × n ∈ {2, 4} = 18 cells**, 7 variants,
**N = 100 paired replications per cell**, fixed-N, no early stop. 126 variant-rows;
882 contrast-rows (c3:11-36).

Arm-4 config identity confirmed at `config/shape_sweep_newrth.yaml`: `rth.energy_map.{enabled,
decide, route, zone_demotion} = true` (lines 276–279); `battery_zones.critical = 0.10`
(line 240); `coverage.transit_free_space = true` (line 157) (c3:40-46).

**Gates.** Invariant gate `weighted_voronoi − tgc_basic`: **PASS** — all **126** contrast
rows `exact_zero = True`, `diff_mean = 0`, `diff_ci = 0`, 0 violations; corroborated by
`run.json:summary.readout` (`null_all_exact = true`, `null_max_abs = 0.0`) (c3:53-59).
Structural gates G-B / G-C / G-P / G-E all PASS: 0 missing cells, `n_runs` all = 100,
882/882 contrast rows fully paired with `dropped_pairs = 0`, all cells carry non-zero CI
(c3:63-68).

**Sign convention (c3:70-75).** All contrasts `diff = tgc − peer`. `total_energy`,
`makespan`, `energy_imbalance`, `length_imbalance`, `swaps` — lower is better, so a negative
diff favours TGC. `efficiency` (SMDP throughput ratio) and `success` — higher is better.

**CI definitions (c3:76-85).** *Per-cell paired CI* = the `diff_ci` column of
`contrasts.csv`, a 95 % normal-approx half-width `Z·√(var/n)`, Z = 1.959963985, ddof = 1, on
the per-replication paired difference. *Pooled aggregate* = mean of the per-cell `diff_mean`
values ± a **between-cell dispersion** half-width `Z·sd/√n_cells` — this measures
between-shape dispersion, **NOT** a within-cell paired CI.

#### A.1 — `tgc_basic − classic_voronoi`, pooled over 18 cells (c3:96-104)

| metric | pooled Δ (± btwn-cell CI) | pooled sig | cells CI-excl-0 | favour TGC / classic | direction |
|---|---|---|---|---|---|
| total_energy (J) | **−39,093 ± 24,100** | EXCL-0 | 11/18 | 10 / 1 | TGC lower |
| makespan (s) | **−1,156 ± 233** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| energy_imbalance | **−0.652 ± 0.171** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| length_imbalance | **−0.754 ± 0.202** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| swaps | **−0.659 ± 0.358** | EXCL-0 | 13/18 | 11 / 2 | TGC fewer |
| efficiency | **−0.520 ± 0.273** | EXCL-0 | 14/18 | 2 / 12 | **classic higher** |
| success | +0.0017 ± 0.0029 | straddles-0 | 0/18 | — | wash |

Pooled `total_energy` is **not single-cell-carried**: leave-one-out over the 18 cells keeps
the pooled value in **[−42.5 kJ, −29.7 kJ]**, all 18 still excluding 0 (c3:115-117).

#### A.2 — `tgc_basic − kmeans`, pooled over 18 cells (c3:120-128)

| metric | pooled Δ (± btwn-cell CI) | pooled sig | cells CI-excl-0 | favour TGC / kmeans | direction |
|---|---|---|---|---|---|
| total_energy (J) | **−12,948 ± 8,639** | EXCL-0 | 7/18 | 6 / 1 | TGC lower |
| makespan (s) | **−244.1 ± 70.2** | EXCL-0 | 17/18 | 17 / 0 | TGC lower |
| energy_imbalance | **−0.124 ± 0.028** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| length_imbalance | **−0.122 ± 0.032** | EXCL-0 | 17/18 | 17 / 0 | TGC lower |
| swaps | −0.111 ± 0.140 | straddles-0 | 12/18 | 10 / 2 | (pooled wash) |
| efficiency | **−0.320 ± 0.144** | EXCL-0 | 14/18 | 2 / 12 | **kmeans higher** |
| success | +0.0006 ± 0.0019 | straddles-0 | 0/18 | — | wash |

Leave-one-out on `total_energy` keeps the pooled value in **[−15.0 kJ, −10.4 kJ]**, every
one still excluding 0; biggest mover is dropping star_5 n=4 → −10.4 kJ. The separation is
**distributed across cells**, not an artefact of one. Magnitude ≈ **0.5 % of a ~2.4 MJ
mission**, and only 7/18 cells resolve individually, so the source states it as a **modest
but robust** TGC energy edge, not a decisive one (c3:130-140).

#### A.4 — Regime overlay on `total_energy` (c3:153-167)

Regime tags are analytical and **obstacle-free**; in shipped a BATTERY-LIMITED tag is a
**lower bound** on true battery pressure. Tag distribution: **BATTERY-LIMITED 10,
FUEL-SURPLUS 8, no BORDERLINE** — it differs from NOW-03 shipped (10 / 6 / 2) because the
arm-4 `critical=0.10` floor enlarges `B_usable`, pushing l_shape n=4 and star_5 n=4 from
BORDERLINE to FUEL-SURPLUS. **Caveat: regime is confounded with n** — every FUEL-SURPLUS
cell is n=4, and 7 of 10 BATTERY-LIMITED cells are n=2.

| contrast | BATTERY-LIMITED (10) | FUEL-SURPLUS (8) |
|---|---|---|
| tgc − classic | −31,192 ± 39,390 (straddles-0) | **−48,970 ± 24,157 (EXCL-0)** |
| tgc − kmeans | −9,918 ± 12,304 (straddles-0) | **−16,736 ± 12,260 (EXCL-0)** |

#### B — n-dependence (c3:177-196)

Pooled `total_energy` (tgc − classic), split by n:

| n | pooled Δ (± btwn-cell CI) | significance |
|---|---|---|
| n = 2 | **−12,632 ± 16,893** | **straddles-0** |
| n = 4 | **−65,555 ± 38,865** | **EXCL-0** |

Per-shape `total_energy` (tgc − classic); `*` = per-cell CI excludes 0:

| shape | n=2 diff ± CI | n=4 diff ± CI | note |
|---|---|---|---|
| square | +6,556 ± 14,056 | −77,672 ± 14,780 * | sign flip (n=2 ns) |
| rect_2_1 | +1,259 ± 11,907 | −31,996 ± 11,146 * | sign flip (n=2 ns) |
| rect_4_1 | −10,215 ± 12,611 | −23,070 ± 11,608 * | same sign |
| rect_8_1 | −51,256 ± 41,221 * | −20,461 ± 43,224 | n=2 stronger |
| disk | **+19,325 ± 16,508 \*** | −75,900 ± 14,354 * | **genuine reversal at n=2 (classic better)** |
| l_shape | +6,575 ± 12,755 | −50,920 ± 22,680 * | sign flip (n=2 ns) |
| star_5 | −4,542 ± 23,585 | −106,305 ± 23,407 * | n=4 far stronger |
| pinwheel | −52,086 ± 34,430 * | −198,232 ± 32,574 * | both TGC, n=4 far stronger |
| c_shape | −29,303 ± 16,229 * | −5,437 ± 14,069 | n=2 stronger |

Four shapes show a raw sign flip between n=2 and n=4 (square, rect_2_1, disk, l_shape), but
**only disk n=2 is a resolved reversal**: **+19,325 ± 16,508**, CI excludes 0, corroborated
in the tgc−kmeans contrast (disk n=2 = **+22,432 ± 14,162 \***, kmeans also lower). At n=4
every shape is TGC-favouring and 7/9 resolve (c3:198-208).

The n-split is **not** echoed on the balance metrics: pooled tgc−classic makespan
(−1,090 vs −1,221 s) and energy_imbalance (−0.43 vs −0.87) both exclude 0 at **both** n.
Only `efficiency` shifts: n=2 −0.643 EXCL-0 favouring classic; n=4 −0.398 straddles-0. So
the n-dependence is specific to **total_energy** and throughput, not to workload balance
(c3:210-214).

For **tgc − kmeans** the n-pattern is **mixed, not cleanly monotone** — no clean n-law
(c3:216-218).

#### C — Launch axis (c3:227-231)

`diff = optimized-launch − naive-centroid-launch`; negative = the optimizer saves energy.
Pooled over 18 cells:

| algo | pooled Δ total_energy (J) | significance | cells optimizer-saves (neg) | cells CI-excl-0 |
|---|---|---|---|---|
| tgc | **−4,419 ± 7,408** | straddles-0 | 9/18 | 6/18 |
| classic | **−1,424 ± 11,286** | straddles-0 | 8/18 | 9/18 |
| kmeans | **−2,501 ± 6,029** | straddles-0 | 9/18 | 8/18 |

**Honest null:** launch-axis optimization is a total-energy wash for all three
decompositions in the new-RTH shipped grid (c3:239-243).

#### D — Success (c3:251-259)

Per-variant `success_mean` across the 18 cells:

| variant | mean | min cell | range |
|---|---|---|---|
| tgc_basic | 0.9989 | 0.990 (rect_8_1 n=2) | 0.990–1.000 |
| weighted_voronoi | 0.9989 | 0.990 (rect_8_1 n=2) | 0.990–1.000 |
| kmeans | 0.9983 | 0.990 (rect_8_1 n=2) | 0.990–1.000 |
| tgc_naive_launch | 0.9983 | 0.990 (disk n=2) | 0.990–1.000 |
| classic_voronoi | 0.9972 | 0.980 (rect_8_1 n=2) | 0.980–1.000 |
| kmeans_naive_launch | 0.9972 | 0.980 (disk n=2) | 0.980–1.000 |
| classic_naive_launch | 0.9967 | 0.970 (star_5 n=2) | 0.970–1.000 |

No variant collapses; the decomposition contrast on success is a wash — 0/18 cells resolve
for either peer (c3:261-265).

**Data limitation (c3:267-272).** `shape_sweep.csv` surfaces only `success_mean`, not a
SUCCESS/PARTIAL/INCOMPLETE breakdown, so the ~0.3–3 % non-success in the worst cells
(rect_8_1 n=2, star_5 n=2, disk n=2) **cannot be attributed to a cause** from these
aggregates.

#### E — Qualitative comparison to NOW-03 shipped (c3:283-290)

**Not a paired delta.** NOW-03 shipped used the old static-0.40 RTH with
`transit_free_space` OFF, so it is not byte-comparable; only ordering/direction is compared.

| finding | NOW-03 shipped | C3 new-RTH shipped | survives? |
|---|---|---|---|
| TGC < classic on total energy | −54 kJ pooled (CI excl 0) | −39 kJ pooled (CI excl 0) | **Yes** (direction holds) |
| TGC balance edge vs classic | 18/18-ish, CI excl 0 | 18/18, CI excl 0 | **Yes**, unchanged |
| TGC ≈ kmeans on total energy | wash (−8 kJ, straddles 0) | **modest separation** (−13 kJ, CI excl 0, distributed) | **Shifts**: wash → modest TGC edge |
| TGC balance edge vs kmeans | small, mostly excl 0 | 17–18/18, CI excl 0 | **Yes**, unchanged |
| kmeans/classic higher `efficiency` than TGC | kmeans +0.15 excl 0; classic wash | kmeans +0.32, classic +0.52, both excl 0 | **Yes, strengthened** |
| Launch-axis optimization pays off on energy | falsified (wash) | falsified (wash) | **Yes** (null reproduced) |

**Winner's-curse ban (c3:316-321).** `run.json:summary.readout` / `summary.md` carry
best-of-baseline H1/H2 statistics (`tgc_adv_vs_best_baseline_by_n`,
`h2_corr_isoperimetric = +0.868`); these are the banned figures and were **not** used. See
§7.3.

**Verdict.** TGC beats classic_voronoi decisively on energy and unanimously on balance;
against kmeans it holds a modest but LOO-robust energy edge and a near-unanimous balance
edge, while losing on throughput. The TGC−classic energy advantage is an **n=4 phenomenon**.

**Status under the new spec: HISTORY ONLY.** Different area (1 km² equal-area shape family
vs one ~0.75 km² rectangle), different obstacle regime (8/km² Poisson, random size vs ~10
fixed obstacles at ≈5 % areal coverage), different fleet grid (n ∈ {2,4} vs {3,5,8}), and a
different peer set — the new spec drops kmeans and the three naive-launch twins. **STILL
VALID within it:** the `weighted_voronoi ≡ tgc_basic` λ=0 null (§2.5), which predicts the
new experiment reproduces the same exact zero unless batteries diverge.

---

### 3.5 S5 / NOW-03 — shipped shape sweep (old RTH)

*Source: `docs/reports/s5_shipped_readout.md` @ `2fcd473`.*

**Question.** The same decomposition question as C3, one RTH generation earlier.

**Setup (s5s:18-25).** Run `shape_sweep_shipped_2026-07-09-18-46-09_a5fd4f`, git commit
`e9c40e2` (post launch-RNG-fix `a0871b6` / #18, verified ancestor); mode **shipped**
(obstacle_density = config default 8/km², λ = 0, obstacles paired per seed); grid **9 shapes
× n ∈ {2, 4} = 18 cells**; 7 variants; **N = 100** paired replications per cell; 126
variant-rows.

A brief-vs-reality discrepancy was flagged at the time and is preserved (s5s:26-49): the
task brief described both folders as "45 cells (9 shapes × 5 n) × 7 variants, N=100, same
grid", which does not match the files on disk. From `run_shape_sweep.py`:
`BUDGETS = {"quick": {"clean": 5, "shipped": 10}, "full": {"clean": 20, "shipped": 100}}`
and `SHIPPED_DEFAULT_NS = [2, 4]`. The mismatch was **by design, not a data fault**.
(Both constants still exist in the code today, at `run_shape_sweep.py:742` and `:744`.)

All sanity gates PASS (s5s:55-60): G-A `weighted_voronoi − tgc_basic = 0` exactly, 0
contrast violations and 0 cell-mean violations; G-B 0 missing rows; G-C `n_runs` all = 100;
G-E all 126 rows stochastic.

#### 1a. Total energy (J) — per cell, paired diff ± 95 % CI (s5s:84-103)

| shape | n | regime | tgc − classic | tgc − kmeans |
|---|---|---|---|---|
| square | 2 | BATTERY-LIMITED | +810 ± 74921 | −26792 ± 71100 |
| square | 4 | FUEL-SURPLUS | +3016 ± 76003 | +54461 ± 77952 |
| rect_2_1 | 2 | BATTERY-LIMITED | −29534 ± 53936 | +710 ± 43941 |
| rect_2_1 | 4 | FUEL-SURPLUS | −13702 ± 70151 | +16689 ± 68584 |
| rect_4_1 | 2 | BATTERY-LIMITED | −50863 ± 46725 | −23443 ± 54576 |
| rect_4_1 | 4 | FUEL-SURPLUS | −39974 ± 55968 | −48189 ± 43223 |
| rect_8_1 | 2 | BATTERY-LIMITED | −91645 ± 113434 | +14073 ± 94750 |
| rect_8_1 | 4 | FUEL-SURPLUS | −95992 ± 132649 | +11863 ± 77863 |
| disk | 2 | BATTERY-LIMITED | +10135 ± 66776 | −1546 ± 73908 |
| disk | 4 | FUEL-SURPLUS | −110939 ± 64675 | −70495 ± 81597 |
| l_shape | 2 | BATTERY-LIMITED | −16299 ± 45417 | +1913 ± 42762 |
| l_shape | 4 | BORDERLINE | −9084 ± 67833 | +25657 ± 66163 |
| star_5 | 2 | BATTERY-LIMITED | +43409 ± 86307 | +32641 ± 62538 |
| star_5 | 4 | BORDERLINE | −158479 ± 69144 | −46360 ± 51594 |
| pinwheel | 2 | BATTERY-LIMITED | −108763 ± 93058 | −66181 ± 93809 |
| pinwheel | 4 | BATTERY-LIMITED | −158624 ± 97076 | −16232 ± 98509 |
| c_shape | 2 | BATTERY-LIMITED | −63591 ± 51190 | +13416 ± 26773 |
| c_shape | 4 | FUEL-SURPLUS | −83933 ± 60183 | −19056 ± 32451 |

#### 1b. Total energy — aggregates (s5s:107-116)

| aggregate | tgc − classic (J) | tgc − kmeans (J) |
|---|---|---|
| n = 2 | **−34038 ± 32303** | −6134 ± 18989 |
| n = 4 | **−74190 ± 40789** | −10185 ± 26515 |
| **ALL** | **−54114 ± 26983** | **−8159 ± 15849** |

tgc − classic: btwn-cell CI excludes 0, 13/18 cells negative. tgc − kmeans: **straddles 0**
— indistinguishable on total energy.

#### 1c. Secondary metrics — aggregates ALL (s5s:120-127)

| metric | tgc − classic | tgc − kmeans | direction |
|---|---|---|---|
| makespan (s) | **−1286.7 ± 252.8** | −225.2 ± 110.2 | lower=better |
| energy_imbalance | **−0.614 ± 0.163** | −0.115 ± 0.026 | lower=better |
| length_imbalance | **−0.707 ± 0.193** | −0.113 ± 0.029 | lower=better |
| swaps | −1.281 ± 0.798 | −0.022 ± 0.509 | lower=better |
| efficiency (throughput) | −0.143 ± 0.210 | **−0.152 ± 0.085** | higher=better |
| success | +0.013 ± 0.016 | −0.001 ± 0.010 | higher=better |

#### 2a. Problem B — does the TGC edge survive the neutral pad? (s5s:152-163)

`tgc_naive − classic_naive` is **NOT** precomputed in `contrasts.csv`; only an unpaired
point-diff of means is available, flagged `*`.

| metric | OPT tgc−kmeans | NAIVE tgc−kmeans | NAIVE tgc−classic* |
|---|---|---|---|
| total_energy (J) | −8159 ± 15849 | −7162 ± 17901 | −78431 ± 50161* |
| efficiency | −0.152 ± 0.085 | −0.154 ± 0.093 | −0.120 ± 0.209* |

The tgc−kmeans gap is essentially identical on both pads, so it is a **decomposition-level
property, not a home-field artefact of the optimizer launch**.

#### 3. Launch axis (s5s:178-181)

| algo | mean diff (J) | btwn-cell CI | cells where optimizer saves (neg) |
|---|---|---|---|
| tgc | **+2406** | 22081 | 9/18 |
| classic | **−21911** | 30358 | 10/18 |
| kmeans | **+3403** | 13435 | 7/18 |

For TGC and kmeans the aggregate is slightly *positive* — a wash, CI straddles 0; only for
classic is it net-negative (−22 kJ). Flagged at the time as a candidate falsification.

#### 4. Regime overlay — total-energy paired diff, mean (btwn-cell CI) (s5s:203-208)

Cell counts: BATTERY-LIMITED 10, BORDERLINE 2, FUEL-SURPLUS 6.

| contrast | BATTERY-LIMITED | BORDERLINE | FUEL-SURPLUS |
|---|---|---|---|
| tgc − classic | −46497 (37829) | −83781 (146405) | −56921 (37397) |
| tgc − kmeans | −7144 (17100) | −10352 (70575) | −9121 (36739) |
| tgc_naive − kmeans_naive | −16668 (23483) | −17545 (119511) | +12142 (15240) |
| tgc − tgc_naive (launch) | +4624 (36141) | +3856 (22230) | −1774 (31644) |

#### 5. H2 — shape descriptors (s5s:224-231)

"Advantage" = per-shape peak over n of `tgc_naive − kmeans_naive`, oriented so > 0 means TGC
better.

| advantage from | corr(solidity) | corr(isoperimetric) | LOO iso range |
|---|---|---|---|
| total_energy | −0.280 | **+0.436** | +0.308 (drop star_5) … +0.690 (drop square) |
| efficiency | +0.011 | +0.369 | +0.178 (drop disk) … +0.682 (drop pinwheel) |

Shape descriptors, solidity/isoperimetric (s5s:229-231): square 1.00/1.27, rect_2_1
1.00/1.43, rect_4_1 1.00/1.99, rect_8_1 1.00/3.22, disk 1.00/1.00, l_shape 0.86/1.70,
star_5 0.49/3.47, pinwheel 0.40/4.39, c_shape 0.80/2.49.

A weak positive isoperimetric correlation that is **not robust** — LOO moves it by up to
±0.35. This is a *different statistic* from `summary.md`'s H2 `corr_isoperimetric = +0.787`,
which uses the banned optimizer-pad best-of-baseline efficiency peak; on the clean naive-pad
contrast the strong +0.787 does **not** reproduce (s5s:233-240). See §7.3.

#### 6. Variance (s5s:251-257)

In shipped, every variant's per-cell CI is dominated by the obstacle draw, so kmeans init
variance is **not separable** here:

| variant | mean efficiency CI | cells with CI > 0 | mean total_energy CI (J) |
|---|---|---|---|
| tgc_basic | 0.1689 | 18/18 | 54245 |
| classic_voronoi | 0.1631 | 18/18 | 55181 |
| kmeans | 0.1559 | 18/18 | 51796 |
| tgc_naive_launch | 0.1657 | 18/18 | 51996 |
| kmeans_naive_launch | 0.1573 | 18/18 | 47607 |

#### 7. Shipped vs clean (s5s:274-289)

Restricted to the n ∈ {2,4} overlap; CI widths **not** comparable (N = 100 vs 20). Compared
on the **sign** of each paired contrast:

| contrast | sign flips / 18 overlap cells | reading |
|---|---|---|
| tgc − classic | 8/18 | every flip is at a cell where at least one side is statistically indistinguishable from 0; the TGC<classic ranking holds in both modes, stronger in clean |
| tgc − kmeans | 8/18 | all flips are near-zero-magnitude — coin-toss around 0 |
| tgc_naive − kmeans_naive | 12/18 | same: differences are ~0, signs flip on noise |
| tgc − tgc_naive (launch) | 9/18 | launch benefit is ~0 in both modes |

Detail of the tgc − classic overlap (SHIPPED | CLEAN, J): square 2 (+810 | −988), square 4
(+3016 | −141254), rect_2_1 2 (−29534 | 0), rect_4_1 2 (−50863 | 0), rect_8_1 2 (−91645 |
0), disk 2 (+10135 | −182463), l_shape 2 (−16299 | +864), c_shape 2 (−63591 | +925).
**Numbers-only conclusion: obstacles change the level, not the ranking.**

**Verdict.** TGC < classic on energy and balance; TGC ≈ kmeans on energy, with kmeans ahead
on throughput. Launch axis a wash for TGC/kmeans.

**Status under the new spec: HISTORY ONLY, and within this archive SUPERSEDED by C3.**
Old static-0.40 RTH with `transit_free_space` OFF. C3 re-ran the same grid under the new
stack and its own §E documents which directions survived.

---

### 3.6 S5 / NOW-03 — clean shape sweep (old RTH, obstacle-free)

*Source: `docs/reports/s5_clean_readout.md` @ `2fcd473`.*

**Question.** The pure shape effect, with obstacles removed — the script's PRIMARY mode.

**Setup (s5c:17-25).** Run `shape_sweep_clean_postfix`, git commit `1214148` (post
launch-RNG-fix `a0871b6` / #18); mode **clean** (obstacle_density = 0, λ = 0); grid
**9 shapes × n ∈ {2,3,4,5,6} = 45 cells**; 7 variants; **N = 20** paired replications per
cell; 315 variant-rows. This folder is the **post-launch-RNG-fix re-run** that was required
before NOW-03; the pre-fix clean grid was stale (s5c:27-28).

All gates PASS (s5c:48-53), including the clean-only G-D: all non-kmeans variants
deterministic with CI = 0; kmeans/kmeans_naive CI > 0 allowed and confined to a small
subset.

#### 1a. Total energy — aggregates (s5c:74-81)

| aggregate | tgc − classic (J) | tgc − kmeans (J) |
|---|---|---|
| n = 2 | −42627 ± 62385 | −17875 ± 34983 |
| n = 3 | −74913 ± 78328 | −131 ± 433 |
| n = 4 | −133783 ± 60537 | +462 ± 415 |
| n = 5 | −110019 ± 69780 | +654 ± 735 |
| n = 6 | −136144 ± 71413 | +1038 ± 575 |
| **ALL** | **−99497 ± 31198** | **−3170 ± 7020** |

tgc − classic: −99.5 kJ, btwn-cell CI excludes 0, negative at every n; **the lead grows with
n** (−43 kJ at n=2 → −136 kJ at n=6). tgc − kmeans: −3.2 kJ, but **driven entirely by one
outlier cell** — for n ≥ 3 the sign turns slightly positive, kmeans marginally lower energy
(s5c:83-90).

**Outlier flag — c_shape n = 2 (s5c:92-98).** The single cell where kmeans has init variance
(CI > 0). Per-variant total energy there: tgc **714962 J**, classic **714037 J**, **kmeans
875624 ± 135 J**, kmeans_naive **874859 ± 192 J**. kmeans lands on a different partition
costing **+160 kJ** but scoring **higher SMDP efficiency** (kmeans **4.497** vs tgc
**3.920**) — an energy-vs-throughput trade, not a plain kmeans failure. This one cell
dominates the n=2 and ALL tgc−kmeans energy aggregates.

#### 1b. Secondary metrics — aggregates ALL (s5c:104-107)

| metric | tgc − classic | tgc − kmeans | direction |
|---|---|---|---|
| makespan (s) | **−1292.7 ± 335.4** | −16.6 ± 31.8 | lower=better |
| efficiency (throughput) | **+0.481 ± 0.243** | −0.034 ± 0.026 | higher=better |
| swaps | −1.289 ± 0.296 | +0.000 ± 0.000 | lower=better |
| success | +0.000 ± 0.000 | +0.000 ± 0.000 | higher=better |

In the obstacle-free grid TGC beats classic on energy, makespan, swaps **and** SMDP
efficiency. `success = 1.0` for all variants in every clean cell (s5c:109-113).

#### 2a. Problem B (s5c:128-131)

| metric | OPT tgc−kmeans | NAIVE tgc−kmeans | NAIVE tgc−classic* |
|---|---|---|---|
| total_energy (J) | −3170 ± 7020 | −3655 ± 7041 | −57750 ± 24892* |
| efficiency | −0.034 ± 0.026 | −0.007 ± 0.025 | +0.220 ± 0.270* |

The near-tie is a **decomposition property, not manufactured by the optimizer launch**.

#### 3. Launch axis (s5c:151-154)

| algo | mean diff (J) | btwn-cell CI | cells where optimizer saves (neg) |
|---|---|---|---|
| tgc | **+716** | 5643 | 21/45 |
| classic | **+42463** | 29542 | 12/45 |
| kmeans | **+232** | 5740 | 21/45 |

A wash for TGC and kmeans, and **net-harmful for classic** (+42.5 kJ; the optimizer pad is
worse than the naive centroid in 33/45 cells) (s5c:156-161).

#### 4. Regime overlay (s5c:174-179)

Regime tags here are **exact** (clean = obstacle-free matches the analytical A2 rule; no
lower-bound caveat). Cell counts: BATTERY-LIMITED 20, BORDERLINE 4, FUEL-SURPLUS 21.

| contrast | BATTERY-LIMITED | BORDERLINE | FUEL-SURPLUS |
|---|---|---|---|
| tgc − classic | −97514 (57911) | −82696 (108539) | −104586 (34665) |
| tgc − kmeans | −8102 (15739) | +409 (787) | +845 (392) |
| tgc_naive − kmeans_naive | −8095 (15841) | −239 (1084) | −76 (401) |
| tgc − tgc_naive (launch) | +5095 (8525) | +8616 (9275) | −4960 (8362) |

#### 5. H2 — shape descriptors (s5c:196-199)

| advantage from | corr(solidity) | corr(isoperimetric) | LOO iso range |
|---|---|---|---|
| total_energy | −0.054 | +0.042 | −0.611 (drop c_shape) … +0.175 (drop pinwheel) |
| efficiency | +0.731 | −0.644 | −0.831 (drop rect_8_1) … −0.503 (drop pinwheel) |

On total energy the correlation is ≈ 0 and **entirely an artefact of one shape**: dropping
c_shape (the kmeans-init outlier, adv = +161.6 kJ) moves it to **−0.611**. On efficiency the
per-shape advantages are **negligible** (all |adv| ≤ 0.06) and the correlations are computed
on noise-level magnitudes, LOO-unstable, and point the **opposite** way to the shipped
naive-pad result (s5c:201-209). The pipeline `summary.md` figure for clean is
`corr_isoperimetric = −0.066`, itself computed on the banned best-of-baseline optimizer-pad
advantage (s5c:210-213). See §7.3.

#### 6. Variance — the clean isolation the shipped grid cannot give (s5c:227-233)

| variant | mean efficiency CI | cells with CI > 0 | mean total_energy CI (J) |
|---|---|---|---|
| tgc_basic | 0.0000 | **0/45** | 0 |
| classic_voronoi | 0.0000 | **0/45** | 0 |
| tgc_naive_launch | 0.0000 | **0/45** | 0 |
| kmeans | 0.0023 | **1/45** | 3 |
| kmeans_naive_launch | 0.0025 | **5/45** | 10 |

tgc/classic/tgc_naive are fully deterministic — 0/45 cells with any variance; paired-seed
determinism holds exactly. kmeans init instability is **rare, not pervasive**: only 1/45
kmeans cells (c_shape n=2, efficiency CI = **0.104**) and 5/45 kmeans_naive cells carry any
init variance (s5c:235-244).

**Verdict.** TGC ≫ classic on every metric including throughput; TGC ≈ kmeans to a near-exact
tie once the single c_shape n=2 outlier is set aside. No stable shape-descriptor law.

**Status under the new spec: HISTORY ONLY, and within this archive SUPERSEDED by C3.**
Old RTH, N = 20, and an obstacle-free grid that the new spec (~10 obstacles ≈5 % of area)
does not correspond to.

---

### 3.7 The launch-site optimiser — a null reproduced three times

*Sources: `c3_shipped_newrth_readout.md` §C @ `f97de54`; `s5_shipped_readout.md` §3 @
`2fcd473`; `s5_clean_readout.md` §3 @ `2fcd473`; `d1_supervisor_package.md` §7a @ `3add399`.*

**Question.** Does siting the launch pad with the three-criterion optimizer, rather than at
a naive centroid pad, save total energy?

**Setup.** Not a separate run. Every shape-sweep grid carries three `*_naive_launch` twins
alongside the optimizer-sited variants, so the launch axis is measured *in passing* as the
paired contrast `algo − algo_naive` on identical seeds. Measured on three independent grids:
NOW-03 clean (45 cells, N=20), NOW-03 shipped (18 cells, N=100), C3 new-RTH shipped
(18 cells, N=100).

**Headline — all three pooled aggregates, `diff = optimized − naive`; negative = optimizer
saves energy:**

| grid | tgc | classic | kmeans |
|---|---|---|---|
| NOW-03 clean (s5c:151-154) | **+716** ± 5643 | **+42463** ± 29542 | **+232** ± 5740 |
| NOW-03 shipped (s5s:178-181) | **+2406** ± 22081 | **−21911** ± 30358 | **+3403** ± 13435 |
| C3 new-RTH shipped (c3:227-231) | **−4,419 ± 7,408** | **−1,424 ± 11,286** | **−2,501 ± 6,029** |

In C3 **every** pooled aggregate straddles 0, for all three algorithms, with the optimizer
saving energy in only about half the cells each way (c3:239-243). In NOW-03 clean it is a
wash for tgc/kmeans and **net-harmful for classic** (+42.5 kJ, worse in 33/45 cells)
(s5c:156-161). In NOW-03 shipped it is a wash for tgc/kmeans and net-negative only for
classic (−22 kJ) (s5s:183-187).

One quantitative shift is recorded between grids: classic's launch term was net-negative
(−21.9 kJ) under NOW-03 shipped and becomes a wash (−1.4 kJ, CI straddles 0) under the C3
new-RTH stack (c3:241-243).

**Verdict, quoted (c3:239-243).** "**Honest null: launch-axis optimization is a total-energy
wash for all three decompositions in the new-RTH shipped grid.**" The D1 synthesis states it
across all three runs (d1:265-270): "**Honest null: launch-axis optimization is a
total-energy wash (harmful for classic in the clean grid), reproduced across three runs.**"

**Why the null is not a surprise, and a known simplification behind it.** The optimizer's
stated objective is swap-waste/energy in battery-limited cells, and that benefit does not
surface on the total-energy aggregate (c3:241-243, s5s:186-187). Independently, the
`launch_site_optimizer` `expected_swaps` criterion uses only transit + return energy, **not**
the full coverage energy of a zone, so in practice it often evaluates to 0 and contributes
little — a documented simplification, not a bug.

**Status under the new spec: HISTORY ONLY, null retained.** The supervisor makes
launch-site optimisation a *separate optional comparison*, not a main-experiment axis. The
null is retained precisely because it is the evidence for not spending the new
experiment's budget on that axis.

---

### 3.8 EM-01 — the energy-map dynamic RTH

*Source: `docs/proposals/energy_map_rth.md` @ `e025088` (Lithuanian design proposal, base
commit `30f4209`); delivery record in `docs/archive/TODO_legacy.md` @ `9159ca6`.*

**Question.** The dynamic RTH the thesis claimed was effectively inoperative at 1 km² — it
was pre-empted by the static 40 % threshold. Can a per-replication energy cost-to-go map
make the RTH decision positionally exact and obstacle-aware at every scale?

#### The three motivating problems (em:16-20, 36-111)

1. **The dynamic RTH was invisible.** `should_return` is tested *first* in the guard order
   (`state_machine.py:111-112`), but at 1 km² the return cost is small, so its threshold is
   only reached at ~20 % battery — far below the static 0.40 net, which therefore fires
   first every time. Numerical basis (em:72-79): `reserve = reserve_frac·cap = 0.05 ·
   360 000 = 18 000 J`; return from the deepest point ≈ 1 km × 18.333 J/m ≈ 18 kJ plus
   landing, against a 360 kJ capacity.
2. **Straight-chord returns caused obstacle-boxing.** `S3_RTH` was planned as a straight
   chord and avoidance was a reactive 15 m sidestep; after `_OBS_REENTRY_BUDGET = 6` failed
   re-entries the drone was declared boxed in and sent home — a livelock mechanism.
3. **`path_clear` every 5 s was the #1 CPU hot spot** — its own docstring names it "the
   RTH-lookahead hot path (~63 % of mission runtime)". *(Still present in the code today at
   `planning/environment_map.py:156-157`.)*

#### The design

One Dijkstra per replication from the base over an occupancy costmap, storing per cell
`E_home` (J, cheapest energy home) plus a parent pointer (em:115-130).

- **Grid resolution — strictly battery-tied (em:136-153).** Cell edge = the distance costing
  exactly 1/1000 of battery capacity at reference cruise. Arithmetic: capacity
  `100.0 Wh × 3600 = 360 000 J`; MULTIROTOR `CRUISE = 220.0 W`, `v_cruise = 12.0 m/s` →
  **18.333 J/m**; cell edge = `(360 000 / 1000) / 18.333 = 360 J / 18.333 J/m = 19.64 m ≈
  20 m`. Explicitly **not** obstacle-tied: real obstacles are of arbitrary size, so the
  apparatus must not depend on the `obstacle_size_range_m = [20, 80]` convention.
- **Grid dimensions for the L-shape family** (em:159-165), bbox scale = √A, cell 20 m,
  8-connected (~4 edges/cell):

  | A (km²) | bbox (m) | grid | cells | ~edges |
  |---:|---:|---:|---:|---:|
  | 1 | 1155×1155 | 58×58 | 3 364 | 13 456 |
  | 2 | 1633×1633 | 82×82 | 6 724 | 26 896 |
  | 4 | 2309×2309 | 116×116 | 13 456 | 53 824 |
  | 8 | 3266×3266 | 164×164 | 26 896 | 107 584 |
  | 16 | 4619×4619 | 231×231 | 53 361 | 213 444 |

- **Occupancy costmap — the colours ARE the weights (em:189-194).** Green (`f < yellow_thr`)
  ×1.0; Yellow (`yellow_thr ≤ f < 0.5`) ×1.5; Red (`f ≥ 0.5`) ∞, blocked. The yellow penalty
  of 1.5 is derived as the expected sub-cell sidestep multiplier averaged over occupancy
  fractions φ ∈ (0, 0.5), with 2.0 as the conservative near-blocked bound (em:199-211).
  Red threshold ≥ 0.5 balances conservatism against corridor loss: at 20 m resolution it
  keeps ≈ ≥10 m of free corridor in yellow cells, far more than the ~1.2 m drone bbox
  (em:235-246).
- **Storage:** only `E_home: float` plus `parent: int` per cell — two arrays,
  `float64[nx·ny]` + `int32[nx·ny]`; at 16 km² ≈ 53k cells ≈ **0.6 MB** (em:256-261).
- **Build cost ESTIMATE (em:287-301):** 1 km² ~0.1–0.3 s/rep; 4 km² ~0.3–1 s; 16 km²
  **~1–5 s/rep**, against **[AUTHOR DIAGNOSIS]** FIX-B1 visibility routing at
  **347–791 s/rep** — about two orders cheaper, and once per replication rather than
  per tick.
- **Cadence — battery-quantized (em:306-316).** Below the per-sortie arming threshold the
  energy decision is evaluated every **1 % of battery drop**; one cell hop = 360 J = 0.1 %
  of capacity, so 1 % ≈ 10 hops. Obstacle threats stay asynchronous — the cadence governs
  only the energy decision.
- **Arming threshold, with a proof (em:341-358).**
  `arm = (max E_home over remaining-plan cells + max leg bundle + reserve) / capacity +
  delta`. Above `arm`, `should_return` is *provably* False, so skipping the check loses no
  decision; `delta` absorbs the ≤1-cell quantization error.
- **The ×1.5 obstacle fudge is REMOVED** when the map decides — the map is obstacle-aware by
  construction, so the fudge would be double-counting (em:335-337).

#### Staged rollout, all flag-gated default-OFF (em:463-469)

S1 map builder → S2 decide + arming → S3 return routing → S4 resume routing → S5
skip-on-stall. Each stage required a full green suite, new tests for new behaviour, and a
**flag-off byte-identity fixture with an unchanged config hash**.

#### B1 — battery-zone demotion, and the B2 floor sweep (em:406-446)

Under `rth.energy_map.zone_demotion` the static `critical_battery` guard branch is
**removed** — no new constant, no `battery_zones` value change — leaving `terminal_battery`
as the only failsafe. The proposal is explicit that this removes the **0.40** net, not a
0.20 one, and records why a "~0.15 terminal" variant was rejected as a no-op: the guard
fires at 0.40 regardless, so lowering `critical` to 0.15 merely widens the CRITICAL band.

**B2 TERMINAL-floor sweep result at 1 km² (em:431-438), verbatim:**

- floor **0.20**: the static fractional net **takes over ~47 % of returns** from the
  distance-aware map — **14 `terminal_battery` vs 16 `rth_energy`**;
- floor **0.10**: the override disappears — `terminal_battery → 0`, the map alone governs
  every return, **arrival-reserve min 0.087**;
- **0.10 ≡ 0.05 byte-identically at 1 km²** (below ~0.12 the map always returns first), so
  the floor value is a **scale-axis question, not a 1 km² question**;
- swaps are quantized at 5 and the floor **does not move them** — reported as a negative.

Chosen Stage-5 arm-4 floor: **0.10**, applied **in-config** for that experiment only, never
by editing `default.yaml`.

#### Delivery record (TODO_legacy @ `9159ca6`)

EM-01 Stages 1–4 merged; a per-replication Dijkstra cost-to-go grid drives the RTH decision,
the return route and the resume transit; the obstacle-boxing livelock resolved with the
residual accounted for via `MISSION_PARTIAL` + `skipped_legs`; **sample ceiling moved 92 % →
94 %** (legacy:14). A1 merged `fd61438`; A2 merged `0409186`; A3 merged PR #39; B1 merged
PR #38 (legacy:21-26).

**Verdict.** The design shipped, the A/B validated it (§3.1), and the reason-attribution
inversion is the observable that proves the map governs.

**Status under the new spec: STILL VALID (design and implementation).** The energy-map RTH
*is* the new spec's second named core contribution ("dynamic RTH energy reserve").

**But its constants are regime-specific and must be re-derived.** Both anchors above were
computed for the old platform and area:

- the **20 m cell** follows from `100 Wh` capacity and `18.333 J/m` at MULTIROTOR
  `CRUISE = 220 W` / `v_cruise = 12 m/s`. The new spec flies at **10 m/s**, and
  `config/djimatrice4e.yaml` carries a different capacity (99.5 Wh) and power table — the
  cell edge must be recomputed from the same rule, not copied.
- the **TERMINAL floor 0.10** was chosen in a regime where `0.10 ≡ 0.05` byte-identically,
  and the source itself says the floor is a scale-axis question. It does not transfer to
  ~0.75 km² unexamined.

**Honest limits the proposal recorded and that still apply (em:520-535):** coverage strips
still lie over obstacles — the map handles return/transit/resume routing, not strip
geometry; no wind or voltage sag is modelled, and the cell-size rule is robust only to
battery-capacity changes; the 8-connected square grid has a directional anisotropy artefact
(hex was left as a future option, explicitly not implemented); not-in-area cells are treated
conservatively as red, so the map does not exploit flyable space outside the survey polygon.
And the literature note the proposal repeats twice (em:124-130, em:533-535): **cost-to-go
grids and occupancy costmaps are classical robotics** (the direct analogue is ROS
`costmap_2d`); the contribution is **not the grid** but the battery-normalized,
per-replication energy map replacing a static RTH threshold, validated by paired-seed A/B.

---

## 4. Falsified hypotheses and confirmed nulls

Fourteen entries. Each is a result — several were *predictions the project made and then
disproved on its own data*, which is why they are recorded as prominently as the wins.

**4.1 — The INCOMPLETE → PARTIAL shift does not exist. FALSIFIED.**
The anticipated shift does not appear: `MISSION_INCOMPLETE = 0` in **all 400** C1
replications (aggregate SUCCESS 389, PARTIAL 11, INCOMPLETE 0, FAILED 0). The shift that
actually occurs is **PARTIAL → SUCCESS**. The source instructs that the INCOMPLETE→PARTIAL
narrative "should be dropped for this sample" (c1:74-77, c1:164-165).

**4.2 — The map does not significantly improve success. NOT SUPPORTED at n=100.**
arm4 − arm1 `success_frac` = **+0.030, 95 % CI [−0.010, +0.080]** — crosses zero (c1:127).
The defensible claim is equivalent success at lower energy, time and swaps (c1:168-169).

**4.3 — Routing alone is not a pure win. FALSIFIED.**
arm2 − arm1 `total_energy_j` = **+38,592 J, CI [+26,759, +50,592]** — routing alone
*raises* energy while improving completion. The entire energy/makespan/demand win comes from
`zone_demotion`, not from routing (c1:95, c1:99-100, c1:119-121, c1:166-167).

**4.4 — Launch-axis optimization does not pay off on total energy. CONFIRMED NULL,
reproduced three times.** See §3.7 for all nine pooled figures (c3:227-243, s5s:178-187,
s5c:151-161, d1:265-270).

**4.5 — The isoperimetric shape-law does not survive a bias-free comparison. FALSIFIED.**
The pipeline `summary.md` H2 figure is computed on the **best-of-baseline (winner's-curse)**
optimizer-pad advantage, which the analysis bans. Recomputed free of that bias: the clean
naive-pad energy correlation is **+0.042** and is itself an artefact of one shape — dropping
c_shape moves it to **−0.611**; on the shipped naive pad it is only a weak, LOO-unstable
**+0.436** (LOO range +0.308 … +0.690). "A simple solidity/isoperimetric shape-law for the
decomposition advantage is **not supported**" (s5c:196-217, s5s:224-240, d1:272-281).

**4.6 — H5 as a convex-hull/solidity framing is retired. FALSIFIED, with a replacement.**
Zone imbalance correlates with the **isoperimetric/elongation** measure, **not** with
solidity: the thin **convex** `rect_8_1` partitions as badly as, or worse than, the concave
star/pinwheel. Because connectors are already free-space chords, there is no
concavity-vs-hull routing lever to pull; what hurts is *elongation of the partitioned
pieces*, which both thin rectangles and concave arms produce. Across shapes,
`corr(peak weighting benefit, solidity) ≈ −0.505`. Also noted: solidity is 1.0 for 5 of the
8 shapes, so this family has little solidity variance (README @ `9c139ec`:406, :416).

**4.7 — `weighted_voronoi ≡ tgc_basic` at λ=0 with full batteries. CONFIRMED NULL, exact.**
C3: **126/126** contrast rows `exact_zero = True`, `diff_mean = 0`, `diff_ci = 0`, 0
violations; `null_all_exact = true`, `null_max_abs = 0.0` (c3:53-59). NOW-03 shipped: G-A
PASS, 0 contrast violations and 0 cell-mean violations (s5s:57). NOW-03 clean: G-A PASS, 0
violations (s5c:50). This is the invariant of §2.5, and it held exactly at every scale
tested.

**4.8 — `decide` without `zone_demotion` is a result-level no-op. CONFIRMED.**
Arms 2 and 3 are byte-identical on every per-replication field (Δ = 0, CI [0,0] on all
metrics) despite arm 3 running the map decision at **321.71 hits/rep**. The static 0.40 net
fully pre-empts the map's decision until `zone_demotion` removes it (c1:40-45, c1:104-107).

**4.9 — Plan-time `E_home = ∞` skip. FALSIFIED by measurement, never shipped.**
`E_home = ∞` is **not** a reachability test. The M0 measurement over 50 replications found
**432/5520 ∞-entry strips executed successfully** through sub-cell S_OBS corridors — a 20 m
red cell does not imply an unreachable strip. A plan-time skip would have converted 46
successes into PARTIAL and driven the sample ceiling **92 % → 0 %**. The shipped mechanism is
instead **runtime skip-on-stall**, triggered by the FIX-B4 `StallDetector` budget (5
no-progress swaps) (em:381-402).

**4.10 — The B2 "willing-map over conservative floor" risk did not materialise at 1 km².**
Arm-4 pooled return depth: min **0.1077**, median **0.1964**, mean **0.1947** over 496
returns — the 0.10 floor is **never reached**, so `terminal_battery = 0`. The source scopes
this explicitly: good, "but scoped to this 100-rep, 1 km² sample; **not extrapolable**"
(c1:144-149).

**4.11 — The 99 % spare-sizing target is structurally unreachable. CONFIRMED.**
8/500 replications (1.6 %) never succeed at any B, capping `success_frac ≤ 0.984 < 0.99` for
every spare count. 99 % is an INCOMPLETE-cause problem, not a spare problem (c2:42-48).

**4.12 — The analytical spare-count prior does not match the empirical knee. UNRESOLVED.**
`analytical_prior_spares = 1` (formula `E_cover/B_usable − n + margin`) against an empirical
knee of 5–6. `formula_validation.verdict = "inconclusive"`, and it was computed only for the
0.99 target whose empirical knee is null. Whether the gap is a formula artifact or an
RTH-model artifact **was left open — no verdict formed** (c2:90-94). Carried to §7.4.

**4.13 — The TERMINAL floor value is not a 1 km² question. CONFIRMED NEGATIVE.**
**0.10 ≡ 0.05 byte-identically** at 1 km² (below ~0.12 the map always returns first), and
swaps are quantized at 5 with the floor not moving them — "reported as a negative"
(em:436-438).

**4.14 — kmeans init variance is rare, not pervasive. CONFIRMED, and concentrated.**
In the obstacle-free clean grid, where deterministic variants have exactly zero variance,
only **1/45** kmeans cells (c_shape n=2, efficiency CI **0.104**) and **5/45**
kmeans_naive cells carry any init variance. Elsewhere k-means++ lands on the same partition
across all 20 seeds. "TGC deterministic vs kmeans init-unstable" is **true but concentrated**
(s5c:227-244). In the shipped grid the obstacle draw swamps init variance entirely, so the
two are not separable there (s5s:251-261).

---

## 5. Bugs found and fixed

Ten. Two of them invalidated results that had already been quoted, which is why the
superseded-data rule in §6.3 exists.

**5.1 — The launch-site RNG defect.** *Mechanism:* every Monte-Carlo replication re-picked
the launch pad, which contaminated the variance of **every** optimizer-sited variant.
*Found and fixed* 2026-07-05, commit **`a0871b6`**, PR #18. *Blast radius:* **all numbers
from the pre-fix `shape_sweep_clean` run were declared invalid**, including three figures
that had already been cited in drafts — "TGC vs classic +0.31", "optimizer wastes +13.7 kJ",
"H2 corr −0.137". The clean grid had to be re-run from scratch as
`runs/shape_sweep_clean_postfix` (`scale_sweep_v2.md:21-31`, `s5_clean_readout.md:27-28`).

**5.2 — The `path_clear` CPU hot spot.** *Mechanism:* the RTH lookahead called
`env.path_clear(route)` every 5 s per drone; the function's own docstring names it "the
RTH-lookahead hot path (~63 % of mission runtime)". *Fixed* by vectorising it into batch
Shapely 2.x predicates over the whole sample array in a single C call — PR #26, commit
**`1214148`**, measured **≈2.3× per mission**, and byte-identical by construction (the same
GEOS predicates, `argmax` returning the first violating index exactly as the sequential loop
returned). *(The docstring and the vectorised implementation are both still in the code at
`planning/environment_map.py:150-190`.)* (`scale_sweep_v2.md:42-51, :294`.)

**5.3 — The swap livelock (FIX-B1).** *Mechanism:* a blocked straight resume chord looped
`S_OBS → boxed-in → RTH → swap` forever. *Reference case:* replication 1, drone #3, **151
swaps**. *Fixed* by `coverage.transit_free_space`, which routes S1 transit chords (initial
assign, post-swap resume, redistribution re-transit) around obstacles at plan time; an
unobstructed chord still takes the straight chord, preserving byte-identity where nothing is
blocked (`config/study01_demand.yaml:140-147`).

**5.4 — The residual livelock burn (FIX-B4).** *Mechanism:* a residual livelock consumed the
full 25 ks `max_timesteps` budget. *Fixed* by `safety.stall_detector`, which halts early —
`MISSION_INCOMPLETE` plus a `stalled_agents` diagnostic — when a drone requests **5
consecutive swaps with zero coverage progress**, cutting the cost to **~2 ks sim-time**
instead of the full 25 ks (`config/study01_demand.yaml:241-246`).

**5.5 — Obstacle-boxed coverage strips (EM-01 Stage 4).** *Mechanism:* a strip the drone
could not enter stalled the mission with no accounting. *Fixed* by `safety.stall_skip`
(requires `stall_detector`): on a stall-budget hit the engine calls `agent.skip_stuck_leg()`,
the strip is recorded in `MissionResult.skipped_legs` as an `(agent_id, leg)` pair, and the
terminal outcome becomes `MISSION_PARTIAL`. **Never a silent drop** — the coverage gap is
logged and reported. Flag-off, `skip_stuck_leg` is never called, `skipped_legs` is always
empty, and the run is byte-identical (`energy_map_rth.md:391-402`). Note this is the *second*
design: the plan-time variant was falsified first (§4.9).

**5.6 — `MISSION_PARTIAL` missing from outcome bucketing (A1).** *Mechanism:* `run_output`
did not bucket the new outcome, so partial missions were mis-tabulated. *Merged* as
**`fd61438`** (`TODO_legacy.md:21`). Its companion A2 — the Stage-4 flag-off byte-identity
test with a cross-commit golden — merged as **`0409186`** (`TODO_legacy.md:22`).

**5.7 — `route_transit` recomputing the visibility graph per query (E3).** *Mechanism:* the
endpoint-independent O(V²) obstacle-vertex visibility result was rebuilt on every call.
*Fixed* by memoising it on a per-replication engine dict keyed by
`(sha1(obs.wkb), sha1(region.wkb))` and splicing it into each `(a,b)` query byte-identically
— the cache is consulted purely by coordinate-key pair, never by node index, and only the
True (visible) edge set is stored, so it is O(visible) not O(V²). *Measured on the dense
replication:* **V = 1349, 168,540 visible edges cached (≈ tens of MB/worker), 5.88×
speedup (898 → 153 s)**. Suite **474 → 482** green (+1 skip)
(`TODO_legacy.md:53`).

**5.8 — Non-deterministic parallel execution (E2 / ENG-09).** *Mechanism:* BLAS/OpenMP
thread counts varied across workers. *Fixed* by pinning BLAS/OpenMP threads at both demand
entrypoints — recorded as **"the load-bearing cause of ENG-09 determinism"**. Delivered with
`--jobs` on both demand paths via a shared `experiments/_parallel.py`; the byte-identity gate
is green for both scripts, serial `--jobs 1` vs spawn `--jobs 2/3` bitwise-identical on every
physics field. Suite **465 → 473** green (+1 documented skip) (`TODO_legacy.md:52`).

**5.9 — Run folders overwriting each other (ENG-13).** *Mechanism:* repeated runs wrote into
the same directory. *Fixed* by unique non-overwriting run directories — merged PR #23, commit
**`261e21f`** (`TODO_legacy.md:58`). *(A label collision is recorded there: a separate,
deeper per-sim-schema task also carried the "ENG-13" name and was never done.)*

**5.10 — SafetyMonitor O(n²) pairwise separation scan.** *Mechanism:* every drone pair was
checked each tick. *Replaced* by `SafetyMonitor._separation_yielders`, a KDTree query;
`experiments/bench_separation.py` measures old vs new across fleet sizes and **asserts both
produce identical results**, with the speedup growing with n (`docs/cli_map.md:483-492`).

**Test-baseline trajectory across these fixes,** as recorded at the time: 450 → 465 (C1
harness), 465 → 473 (E2), 474 → 482 (E3), reaching **500 green (+1 documented skip)** at
`TODO_legacy.md:9` on 2026-08-03. *(The suite has grown further since; the current count is
whatever `python -m pytest -n logical --dist loadscope -q` reports today.)*

---

## 6. Methodological rules that survive the reset

These are not results. They are the working discipline that produced the results, and every
one of them applies unchanged to the new main experiment.

**6.1 — Paired-seed determinism is the cornerstone.** `RngFactory.stream(name, replication)`
is a pure function of `(master_seed, name, replication)`. A shared factory across arms
guarantees identical environment and failure draws at the same replication index. This is
what makes a *paired contrast* meaningful, what let C1 confirm `config_hash` identity across
four differently-flagged arms, what let the C1↔C2 cross-check identify the same failing
worlds in two separate experiments, and what makes every byte-identity gate possible at all.
Protect it before anything else.

**6.2 — Timing-probe before every long run.** The project burned **53 hours reaching 0 of 50
completed tiers** on Azure before the design was recognised as the problem rather than the
implementation (`scale_sweep_v2.md:37-53`). CLAUDE.md records this together with a second,
30-hour lesson. The rule that came out of it: always probe with a small `--reps` and
extrapolate, and **one replication is not a probe when that replication is a known outlier**.
Corollary from E2: a tiny-world probe *cannot* exhibit a parallel speedup — at ≈0.17 s/rep
spawn overhead dominates (~1.1×) while the real per-rep cost of 250–380 s makes it near-linear
in workers, so probe estimates stay ESTIMATES until a real-scale run measures them
(`TODO_legacy.md:52`).

**6.3 — Superseded-data discipline.** Every read-out in §3 names **one** run folder as its
single source of truth and computes every number only from it. Superseded numbers are cited
*as superseded*, never for a value, and a re-run is framed as "the new run gives X", never
"X beats the old Y" (`c2_study01_readout.md:81-84`, `c3_shipped_newrth_readout.md:3-6`). §5.1
is why: a bug can retroactively invalidate an entire grid, and mixed provenance makes that
unrecoverable.

**6.4 — Flag-off byte-identity is a gate, not a nicety.** New behaviour ships default-OFF
behind a config flag, and the flag-off run must be byte-identical to pre-change — including
an unchanged `config_hash`, via the optional-key provenance-hash pattern. A missing
byte-identity test is **debt, not a detail**. Deliberate bug fixes are *not* flag-gated
(`energy_map_rth.md:452-473`).

**6.5 — The winner's-curse ban.** Never compare an arm against the best-of-others per cell.
Peers are always reported **separately**, each as its own paired contrast with a CI on the
difference. Where the analysis pipeline itself emitted best-of-baseline statistics, the
read-outs recomputed the bias-free version and flagged the divergence — which is exactly how
the isoperimetric shape-law was falsified (§4.5) (`c3:316-321`, `s5s:311-314`, `s5c:273-275`).

**6.6 — SUCCESS / PARTIAL / INCOMPLETE are three rows, never folded into "fail".** The
distinction between them is itself a thesis effect: C1's real finding was a PARTIAL→SUCCESS
shift, invisible if the two had been pooled, and C2's whole load-bearing result is that the
INCOMPLETE class — not the spare count — sets the ceiling.

**6.7 — A read-out is not a verdict.** Every §3 report was delivered as a *data-analyst
read-out*: exact numbers, tables, methodology notes, intermediate conclusions, and
**explicitly no thesis verdict** — interpretation was a separate, later step
(`c3:6-9`, `s5s:8-11`, `s5c:7-8`). The boundary was preserved even in the D1 synthesis, which
states it decides no new thesis-metric semantics (`d1:3-6`, `d1:337-347`).

**6.8 — Never decide thesis-metric semantics silently.** Outcome classification, success
predicates, what counts as coverage — if a task appears to require changing any of these,
stop and ask the author. This is CLAUDE.md working rule 7 and it is the reason several
questions in this archive are recorded as open rather than answered.

**6.9 — The per-drone check is PRIMARY; pooled ratios are a lower bound.** Batteries are not
shared, so `E_cover/(n·B_usable)` only says the fleet has enough energy *on average* —
averages hide imbalance. The regime is battery-limited if **either** the pooled ratio > 1
**or** the busiest drone's own zone exceeds one battery. The worked pinwheel case at n = 4
makes the point: pooled ratio **0.92** (which a one-tank view waves through as surplus) but
the busiest drone's zone costs **1.31 × B_usable** — correctly battery-limited
(README @ `9c139ec`:374-379).

**6.10 — Fixed-N beats adaptive stopping when pairing matters.** CI-based adaptive stopping
would break the paired-seed protocol, so every sweep cell uses a fixed N and reports its CI
rather than stopping on it (`scale_sweep_v2.md:251-254`).

**6.11 — Analytical formulas must mirror execution physics exactly.** `E_cover` and the RTH
lookahead rebuild the drone's legs exactly as the execution engine does — even leg a
`COVERAGE` strip with the camera on, odd leg a `TURN` connector — and integrate with the
identical `power × dt` model, including the discrete integration and the camera term. Energy
units are kept separate: `E_home` uses CRUISE, the next-bundle term uses COVERAGE + camera,
and the two are never mixed.

---

## 7. Known open discrepancies carried into the defence

Genuine, unresolved, and citable. Each is a place where two authoritative statements in this
repository disagree, or where a question was explicitly left unanswered.

### 7.1 — `efficiency_score._DENOM_STATES` vs the oracle spec

Two committed definitions of the headline metric's denominator disagree:

| Source | Denominator |
|---|---|
| **Code** — `src/uav_swarm_sim/metrics/efficiency_score.py:35-41` | `S1_TRANSIT` + **`S_FERRY`** + `S3_RTH` + `S_OBS` + `S_SWAP` (**5 states**) |
| **Oracle spec** — `README.md:464` @ `9c139ec`, `ADVERSARIAL_TESTER_PROMPT.md:56` (oracle item 13) @ `a6c30d8`, `PROJECT_GUIDE.md:215` @ `91e7109` | `S3_RTH` + `S_OBS` + `S_SWAP` (**3 states**) |

The code's own module docstring argues the 5-state form explicitly — only `S2_MISSION` has
the camera on and is productive, so outbound transit and camera-off ferrying are overhead
like any other. The three prose sources state the 3-state form. **Not adjudicated here.**
Both are quoted verbatim; deciding which is the thesis metric is a metric-semantics call
under §6.8, and it changes every published `efficiency` number.

### 7.2 — Two source files still say "seven-state", and both omit `S_FERRY` from the taxonomy

`AgentState` has **eight** members (`enums.py:19-45`), but:

| Site | Defect |
|---|---|
| `src/uav_swarm_sim/infrastructure/enums.py:20` | docstring says *"Seven-state behavioral automaton"*; line 23 lists *"Author's extensions: S3_RTH, S_OBS, S_SWAP"* — **`S_FERRY` appears in neither the base set nor the extensions**, though it is declared at line 32 |
| `src/uav_swarm_sim/execution/state_machine.py:1` | the same docstring, with the same omission |

**Exact required fix** (debt for a separate dispatch; **src/ was not touched by this
cleanup**): `S_FERRY` belongs in the author's extensions, and the count becomes **eight**.

**Why this is not cosmetic.** `S_FERRY` is inside `AgentState.is_airborne`
(`enums.py:38-45`), so ferry legs **consume flight energy and carry failure-hazard
exposure**. That bears directly on two of the supervisor's named metrics — total energy and
workload balance — and it is the same state whose membership is disputed in §7.1. The two
discrepancies are one family: **the project has never fully settled whether camera-off ferry
flight is overhead, and the code and the prose answer differently.**

### 7.3 — Three different values for "the H2 isoperimetric correlation"

All three are the **banned best-of-baseline** statistic, from three different pipelines. They
are recorded separately, with provenance, and are **never merged into one number**:

| Value | Source | Grid |
|---|---|---|
| `corr_isoperimetric = +0.787` | `s5_shipped_readout.md:240` @ `2fcd473` | `shape_sweep_shipped` `summary.md` (old RTH) |
| `corr_isoperimetric = −0.066` | `s5_clean_readout.md:210-213` @ `2fcd473` | `shape_sweep_clean_postfix` `run.json` readout |
| `h2_corr_isoperimetric = +0.868` | `c3_shipped_newrth_readout.md:321` @ `f97de54` | `shape_sweep_shipped_newrth` `run.json:summary.readout` |

The bias-free recomputations (§4.5) are +0.042 (clean naive-pad) and +0.436 (shipped
naive-pad). **None of the three table rows above should be cited.** They are listed so that
if one surfaces in an old draft it can be identified as the banned figure rather than
mistaken for a result.

### 7.4 — The analytical spare-count prior vs the empirical knee

`analytical_prior_spares = 1` from `E_cover/B_usable − n + margin`, against an empirical knee
of **5–6**. `formula_validation.verdict = "inconclusive"`, and it was computed only for the
0.99 target whose empirical knee is null. Whether the gap is a **formula artifact or an
RTH-model artifact is an open question — no verdict was formed**
(`c2_study01_readout.md:90-94`). Under the new spec the spare-sizing use is moot, but the
underlying question — does the analytical `E_cover`/`B_usable` model predict executed
sortie demand — is not.

### 7.5 — ENG-01 turn aerodynamics is not implemented

The energy model has **no turn/bank aero-penalty term**. This was assessed as acceptable for
a **multirotor** platform, where turn energy is small, and it is the thesis platform. It is
**the trigger for any fixed-wing claim**: a fixed-wing energy result would require ENG-01
first, and **ENG-01 re-baselines energy**, so it must be done *before* any final re-runs or
not at all. Recorded as a scope boundary, not a bug
(`d1_supervisor_package.md:287-292`, `TODO_legacy.md:57`).

### 7.6 — Gaps between the code and the new supervisor spec

Five open items were identified while writing this archive — two requiring an author
semantics decision under §6.8, three ordinary work items. **They are tracked on the GitHub
Project board "mag" and are deliberately not narrated here**: this document is a record of
the pre-reset regime, not a to-do list. `README.md` §14 carries the pointer and the
commit stamp.

---

## 8. Retired-artifact index

Every path below was deleted in `docs: retire superseded documentation (see
PROJECT_HISTORY)`. All are retrievable in full by commit SHA (or by the `pre-reset-archive`
tag, if the author has created it — see §1.2):

```bash
git show 493e8d2:<path>
```

Anchor SHA for every row: **`493e8d29b090785ad073c5939aa473c209747b88`** (`493e8d2`).

| Deleted path | What it contained | Distilled into |
|---|---|---|
| `PROJECT_GUIDE.md` | 402-line return-in-a-year manual: file-by-file dictionary of ~60 modules, directory tree, setup instructions incl. Colab/Docker paths, two Mermaid dependency diagrams, five flagged simplifications | §2 |
| `ADVERSARIAL_TESTER_PROMPT.md` | Adversarial-QA system prompt: an 18-item invariant "oracle", a known-simplifications list, 10 test dimensions, severity rubric | §7.1 (oracle item 13); the rest is a reusable prompt, not evidence |
| `mission_analyst_prompt.md` | Mission-analyst system prompt for reading a run's figures + `events.jsonl`: artifact-by-artifact reading guide, symptom→cause→knob playbook, grounding rules | not distilled — a working prompt, superseded with the telemetry workflow |
| `docs/archive/TODO_legacy.md` | The retired task tracker: A1–A3, B1, C1–C3, D1–D3, E1–E3 delivery records and the engineering backlog | §5, §6, §3.8 |
| `docs/reports/c1_stage5_ab_readout.md` | C1 four-arm RTH A/B read-out, 100 paired reps/arm | §3.1 |
| `docs/reports/c2_study01_readout.md` | C2 STUDY-01 spare-sizing re-run, 500 reps, demand-mode | §3.2 |
| `docs/reports/c1_c2_cross_note.md` | The C1↔C2 seed-overlap bridge | §3.3 |
| `docs/reports/c3_shipped_newrth_readout.md` | C3 shipped shape sweep under arm-4 RTH, 18 cells × 7 variants × N=100 | §3.4 |
| `docs/reports/s5_shipped_readout.md` | NOW-03 shipped shape sweep, old RTH, 18 cells × N=100 | §3.5 |
| `docs/reports/s5_clean_readout.md` | NOW-03 clean shape sweep, old RTH, 45 cells × N=20 | §3.6 |
| `docs/reports/d1_supervisor_package.md` | Supervisor-facing synthesis of C1+C2+C3+NOW-03 (DRAFT); model limitations; open K1–K4 scope decisions | §3.7, §7.5 — its numbers are all quoted from the six read-outs above |
| `docs/proposals/energy_map_rth.md` | EM-01 design proposal (Lithuanian): cost-to-go grid, resolution derivation, occupancy costmap, five integration seams, staged rollout, B1/B2 | §3.8 |
| `docs/proposals/scale_sweep_v2.md` | Scale-sweep v2 design proposal (Lithuanian): the 53 h incident, the 1 km² caveat, H-A/H-B hypotheses, cost model, three scope variants, K1–K4 decision points | §5.1, §5.2, §6.2, §6.10 |
| `config/study01_demand_newrth.yaml` | The **C2 frozen config**: an exact copy of `study01_demand.yaml` with only the RTH changed to arm-4 (`rth.energy_map.{enabled,decide,route,zone_demotion}=true`, `battery_zones.critical=0.10`) | §3.2 |
| `config/shape_sweep_newrth.yaml` | The **C3 frozen config**: an exact copy of `config/default.yaml` with exactly three fields changed — `coverage.transit_free_space: true`, `battery_zones.critical: 0.10`, and the arm-4 `rth.energy_map` block | §3.4 |

### Deliberately NOT deleted

`config/study01_demand.yaml` was on the retirement list and was **kept**. It is a frozen test
fixture: six integration modules load it
(`test_energy_map_stage3`, `test_energy_map_stage4`, `test_energy_map_zone_demotion`,
`test_stall_skip`, `test_transit_cache_identity`, `test_transit_livelock`), it carries the
`test_energy_map_stage4` cross-commit golden, and `run_rth_ab.py:529` uses it as its
`--config` default. Deleting it would void four bug-fix regression tests. It is **not**
experiment evidence and **not** a template for the new spec; its own header now says so.

### Dangling references left in place

Two source comments cite a file that now exists only at the anchor tag. `src/` was out of
scope for this cleanup and was not edited:

- `src/uav_swarm_sim/infrastructure/simulation_engine.py:219` → `docs/proposals/energy_map_rth.md`
- `src/uav_swarm_sim/planning/energy_map.py:7` → `docs/proposals/energy_map_rth.md`

Both resolve via `git show 493e8d2:docs/proposals/energy_map_rth.md`.
