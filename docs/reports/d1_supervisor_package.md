# D1 — Supervisor package (DRAFT for author review)

**Status:** DRAFT. This is a supervisor-facing synthesis of the merged, verified
experiment read-outs. The author edits and owns the final framing. It **decides no new
thesis-metric semantics** — outcome classes, success predicates, and metric definitions are
reported as fixed by the prior experiments.

**Sources (the only sources; every number below traces to one of these).** Nothing here is
recomputed from raw CSVs; all figures are quoted from the read-outs:

- **[C1]** `docs/reports/c1_stage5_ab_readout.md` — four-arm RTH A/B (100 paired reps/arm).
- **[C2]** `docs/reports/c2_study01_readout.md` — STUDY-01 spare-sizing under the new RTH (500 reps).
- **[X-note]** `docs/reports/c1_c2_cross_note.md` — the C1↔C2 seed-overlap bridge.
- **[C3]** `docs/reports/c3_shipped_newrth_readout.md` — decomposition under the new RTH (shipped grid).
- **[NOW-03 shipped]** `docs/reports/s5_shipped_readout.md` — decomposition, old RTH, obstacle grid.
- **[NOW-03 clean]** `docs/reports/s5_clean_readout.md` — decomposition, old RTH, obstacle-free grid.

**Provenance caveat (E1).** The two NOW-03 read-outs ([NOW-03 shipped], [NOW-03 clean]) each
carry a provenance-reconstruction note in their source folders (per `TODO.md` E1). Wherever
NOW-03 numbers appear below, read them as reconstructed-provenance figures, not fresh runs.

**Superseded-data rule.** Only the six read-outs above are cited. The pre-EM-01 STUDY-01
numbers (`runs/spares_final_demand`, `30f4209`) are superseded by [C2] and are not used.

---

## 0. One-paragraph summary for the supervisor

Two orthogonal stories. **(A) The energy-map RTH** (EM-01, "new RTH") delivers *equivalent
mission success at materially lower cost* — energy −5.5%, makespan −12.9%, ~3 fewer swaps per
mission — by replacing the static 40% return threshold with a dynamic cost-to-go map that
lets drones fly deeper before returning [C1]. This is honest, significant, and carries a
quantified cost: on ~1.6–2% of worlds the deeper sortie over-commits and fails to complete
coverage, and the *same seeds* fail in two independent experiments [X-note]. That residual
also makes the 99% spare-sizing target structurally unreachable — an INCOMPLETE-cause
problem, not a spare-count problem [C2]. **(B) The decomposition choice** (TGC vs classic
Voronoi vs k-means) is *orthogonal to the RTH axis*: TGC beats classic_voronoi decisively on
energy and workload balance, ties k-means to a modest edge, and its advantage over classic is
concentrated at n=4 — which is the empirical motivation for the scale experiment [C3]. Two
findings are honestly falsified: launch-axis optimization does not pay off on total energy,
and the shape-descriptor "isoperimetric law" does not survive a bias-free comparison [C3],
[NOW-03 clean]. The open decisions for the supervisor are the scale-experiment scope (shape
narrowing, obstacle axis, sizing) — all cheap, re-runnable CLI dials on a runner that is
already built.

---

## 1. The RTH baseline: the 0.40 effective static return threshold

Every experiment in this package is anchored to one baseline number: **the static RTH returns
a drone at 40% battery** (nominal). This is the literature-style conservative baseline that
the energy-map RTH is measured against.

**The config names are misleading — one line of explanation.** The field
`battery_zones.critical` reads as though the static return happens at 0.10 or 0.20, but it does
not. `Battery.zone` classifies `critical ≤ f < nominal` as CRITICAL, so the `critical_battery`
return guard fires at **nominal = 0.40**, and 0.20 is merely where the TERMINAL failsafe begins
(CLAUDE.md, Architecture Facts). Empirically confirmed in [C1]: arms 1–3 return at a median
battery fraction of **0.3999** — i.e. exactly the 0.40 static net. When the dynamic map
governs (arm 4), the same drones fly to a mean of **0.1947** before returning, and the 0.10
terminal floor is never reached (min 0.1077) [C1].

---

## 2. C1 — the new-RTH A/B (four arms, 100 paired reps): equivalent success at lower cost

**Design.** Four arms on identical paired seeds (`master_seed=42`, `config=study01_demand.yaml`,
1 km², obstacles present), each adding exactly one factor. The `config_hash` is byte-identical
across all four arms, so the environment is shared per replication and paired seeds hold [C1].

| Arm | slug | adds | terminal floor |
|---|---|---|---|
| 1 | static40 | (literature baseline) | 0.20 |
| 2 | route-only | obstacle-aware routing | 0.20 |
| 3 | decide-route | map deciding (static 0.40 still pre-empts) | 0.20 |
| 4 | full-map | + zone_demotion (map is sole normal decider) | 0.10 |

**Headline — the reason-attribution inversion.** Summed over 100 reps, arm-1 returns are
**100% `critical_battery`** (746 returns, 0 `rth_energy`); arm-4 returns are **100%
`rth_energy`** (496 returns, `critical_battery = 0`, `terminal_battery = 0`) [C1]. The static
net is fully replaced by the dynamic map — the inversion the thesis predicted.

**A clean isolation fact.** Arms 2 and 3 are **byte-identical** to each other on every
per-replication field, even though arm 3 actively consults the map (321.71 hits/rep) [C1]. The
static 0.40 net pre-empts the map's decision until `zone_demotion` removes it. So `decide`
without `zone_demotion` is a result-level no-op, and the contrast decomposes cleanly as
**arm3 − arm1 = routing**, **arm4 − arm3 = decide+demotion** [C1].

**The result sentence (what is actually significant).** At n=100 the map does **not**
significantly improve success (95% → 98%, difference CI crosses zero). What it delivers,
significantly, is equivalent success at lower cost [C1]:

> The dynamic energy-map RTH (arm 4) achieves **equivalent mission success** to the static 40%
> threshold (98% vs 95%, difference not significant), while cutting **total energy by 5.5%**
> (−138.8 kJ, CI [−160.1, −117.9]), **makespan by 12.9%** (−392.5 s, CI [−463.3, −320.7]), and
> **battery swaps by ~3 per mission** (median −3) — all strongly significant on paired
> contrasts. [C1]

"Equivalent safety at lower cost" is the honest and stronger claim; "higher success" is not
supported at this sample size [C1]. The entire energy/makespan/swap win comes from
`zone_demotion` (the arm4−arm3 step: −6.9% energy, −14.0% makespan, −3 swaps), not from
routing; routing alone (arm2−arm1) *raises* energy +38.6 kJ while improving completion [C1].

**Outcome breakdown (three rows, never folded).**

| Arm | SUCCESS | PARTIAL | INCOMPLETE | FAILED | success_frac | Wilson 95% CI |
|---|---|---|---|---|---|---|
| 1 static40 | 95 | 5 | 0 | 0 | 0.950 | [0.888, 0.978] |
| 4 full-map | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |

*(Source [C1]; arms 2 and 3 also 98/2/0/0.)* A predicted narrative is **falsified honestly**:
the anticipated INCOMPLETE → PARTIAL shift does not appear — `MISSION_INCOMPLETE = 0` in all
400 replications. The real shift is PARTIAL → SUCCESS [C1].

**The quantified cost — `zone_demotion` regresses 2 reps (a real trade-off, not noise).** On
the 6 reps that are ever PARTIAL across the pipeline, `zone_demotion` turns two
previously-successful worlds (reps 76 and 100) into PARTIAL: the deeper sorties over-commit and
miss coverage (coverage_frac 0.9887 / 0.9838) [C1]. This is the map saving energy by flying
deeper at the cost of less margin — ~2/100 reps. It belongs in the package as an honest cost of
the energy win, and §4 shows it recurs on the *same seeds* in a second experiment.

**Baseline caveat (from [C1]).** Arm 1 is "static-40% net **plus** the EM-01 stall-skip", not a
pristine literature baseline — `stall_skip` is held constant across arms as an orthogonal
factor, so arm 1 is the fairest *available* baseline but not a textbook one. Record this.

---

## 3. C2 — spare-sizing under the new RTH: 95% knee = 6, 99% structurally unreachable

**Run.** `spares_c2_newrth`, 500 replications, demand-mode, `master_seed=42`,
`config=study01_demand_newrth.yaml` (the arm-4 "full-map" RTH), 1 km². Config identity to arm-4
is verified by hash + command + file content; this **supersedes** the pre-EM-01 STUDY-01
numbers [C2].

**Headline knees.**

| Target | empirical knee | Wilson-certified knee | status |
|---|---|---|---|
| 0.95 | 5 | **6** | Wilson-certified (500 ≥ 73-rep floor) [C2] |
| 0.99 | — | — | **structurally unreachable** (see below) [C2] |

**Why 99% has no answer.** **8/500 replications (1.6%) never succeed at any B** (demand = ∞,
all `MISSION_INCOMPLETE`), which sets a hard ceiling of `success_frac ≤ 0.984` for *every*
spare count [C2]. Since 0.984 < 0.99, no finite (or infinite) spare pool reaches 99%. Under
this RTH, "how many spares for 99%?" is the wrong question — 99% requires removing the
INCOMPLETE cause, a different problem from spare-sizing. 95% is a clean, certified knee of 6
[C2].

**Demand is tightly concentrated.** Of 492 successful reps: demand=4 → 16, **demand=5 → 467
(dominant, 95% of mass)**, demand=6 → 8, demand=7 → 1 [C2]. The dynamic map produces a
predictable, low-variance sortie count — no long tail between 5 and 7 that a noisier static net
might have produced [C2].

**Honest validity limit (from [C2]).** Telemetry was OFF for this run, so map-governance is
established by config identity (hash + command + file), not by per-return `rth_reason`
telemetry. This is a strong config-level chain, but it is not decision-level attribution.

---

## 4. C1 ↔ C2 cross-note: the residual-failure cost is systematic, not noise

Both C1 (arm-4, 100 reps) and C2 (arm-4, 500 reps) ran under `master_seed=42`, so their low
replication indices share the same worlds. The residual failures line up [X-note]:

- **C1 arm-4** failed (as `MISSION_PARTIAL`) on reps **76 and 100** [X-note].
- **C2** failed (as `MISSION_INCOMPLETE`, demand = ∞) on reps **76, 100**, 109, 151, 162, 191,
  313, 396 — 8/500 [X-note].

Reps **76 and 100 fail in both experiments** — same seed, same world, same underlying
mechanism (the map flies deeper, leaves less margin, and in those worlds the deep sortie does
not complete coverage). The outcome *label* differs by run mode (PARTIAL vs INCOMPLETE) but the
physical cause is one [X-note].

**What this establishes for the supervisor.** The energy-map win is real, but it trades ~1.6–2%
of worlds into residual failure for the efficiency gain on the rest — and this is a
**systematic mechanism, not sampling noise**, because it recurs on the same seeds across two
independent experiments at two sample sizes [X-note]. A reviewer will ask about the 99% ceiling
and the regressed reps; the seed overlap is the evidence that answers them. A concrete (not-now)
follow-up: a small margin adjustment on the map's decide threshold might recover seeds 76/100
without sacrificing the energy win — a question for the scale axis / a margin sweep, not a
C1/C2 re-run [X-note].

---

## 5. Decomposition (ORTHOGONAL to the RTH story): TGC vs classic vs k-means

**This is a separate axis from §§2–4.** The decomposition experiments (NOW-03 and C3) vary
*how the survey area is partitioned among drones*, holding the algorithm family fixed; they are
orthogonal to the RTH change. Stated plainly so the two stories are not conflated.

**The invariant that must hold (and does).** For a homogeneous λ=0 fleet, `weighted_voronoi ≡
tgc_basic` byte-identically. C3 confirms this on **126/126** contrast rows (`exact_zero`, 0
violations) [C3]; NOW-03 confirms it too (G-A pass) [NOW-03 shipped], [NOW-03 clean]. So
`weighted_voronoi` carries no independent information and the peer analysis is TGC vs
{classic_voronoi, k-means}.

### 5a. TGC ≫ classic_voronoi — decisive, and it survives the new RTH

| source | TGC − classic, total energy (pooled) | balance metrics |
|---|---|---|
| [NOW-03 shipped] (old RTH) | −54 kJ (btwn-cell CI excl 0) | makespan/imbalance CI excl 0 |
| [NOW-03 clean] (old RTH) | −99.5 kJ (CI excl 0; grows with n) | makespan/efficiency/swaps CI excl 0 |
| [C3] (new RTH, shipped) | **−39.1 kJ ± 24.1 (CI excl 0)** | **makespan, energy- & length-imbalance unanimous 18/18** |

TGC uses less total energy than classic_voronoi and holds a strong, **unanimous** workload-
balance advantage (lower makespan and lower energy/length imbalance in all 18 cells) under the
new RTH [C3]. The pooled energy figure is not single-cell-carried: leave-one-out over the 18
cells keeps it in [−42.5 kJ, −29.7 kJ], all excluding 0 [C3]. The direction is unchanged from
NOW-03 [C3]. The one metric that favours classic is `efficiency` — the SMDP **throughput** ratio
(π(S2)/π(overhead)), *not* energy efficiency (CLAUDE.md) — an orthogonal throughput/energy
tension, present in NOW-03 and strengthened in C3 [C3].

### 5b. TGC ≈ k-means — an *evolution/tension* from wash to a modest edge (qualitative only)

This is the one place where the new RTH shifts a ranking, and it must be stated carefully:

- **NOW-03 (old RTH): a wash.** TGC ≈ k-means on total energy — [NOW-03 shipped] −8.2 kJ ± 15.8
  (CI straddles 0); [NOW-03 clean] −3.2 kJ ± 7.0, and that clean aggregate is **carried
  entirely by one outlier cell** (c_shape n=2, −160.7 kJ, the single cell where k-means has
  init variance); excluding it, TGC−kmeans is ≈0-to-slightly-positive everywhere [NOW-03 clean].
- **C3 (new RTH): a modest, distributed edge.** TGC−kmeans total energy resolves to **−12.9 kJ
  ± 8.6 (CI excludes 0)**, and — unlike the NOW-03 clean outlier — it is **distributed**, not
  single-cell: leave-one-out keeps it in [−15.0 kJ, −10.4 kJ], every one still excluding 0 [C3].

**How to state it (do not overclaim).** The magnitude is small (≈0.5% of a ~2.4 MJ mission) and
only **7/18** cells resolve individually; it is also **n=4-concentrated** (the n=2 aggregate
straddles 0) [C3]. So the defensible phrasing is a **modest but robust** TGC energy edge over
k-means — **not** "TGC beats k-means." Frame it as an **evolution/tension**: NOW-03 old-RTH wash
→ C3 new-RTH modest edge. This is **qualitative only** — the two runs use a different RTH and a
different transit regime (`transit_free_space` OFF in NOW-03, ON in C3), so they are **not
byte-comparable**; only the direction/ordering is compared [C3]. On makespan and both imbalance
metrics TGC beats k-means nearly unanimously (17–18/18) in both eras [C3], [NOW-03].

---

## 6. The n-dependence (C3): TGC's edge over classic is an n=4 phenomenon

The TGC total-energy advantage over classic_voronoi is **systematically stronger at n=4 than
n=2**, and at n=2 it is not resolvable in aggregate [C3]:

| n | pooled TGC − classic, total energy | significance |
|---|---|---|
| n = 2 | −12,632 ± 16,893 | **straddles 0** |
| n = 4 | −65,555 ± 38,865 | **CI excludes 0** |

Per-shape, four shapes show a raw sign flip between n=2 and n=4 (square, rect_2_1, disk,
l_shape), but of these **only disk n=2 is a *resolved* reversal** (+19,325 ± 16,508, i.e.
classic significantly lower energy at disk n=2, corroborated in the tgc−kmeans contrast); the
other three are positive-mean but CI-straddles-0 [C3]. The honest statement: **TGC's energy
advantage over classic is an n=4 phenomenon; at n=2 it collapses to a wash in aggregate, with
one shape (disk) genuinely reversing** [C3]. The balance metrics do *not* show this n-split
(makespan and imbalance exclude 0 at both n) — the n-dependence is specific to total energy and
throughput [C3].

**Why this matters here.** This is a new, RTH-independent structural feature that NOW-03 did not
foreground [C3], and it is the **empirical motivation for the scale experiment**: it directly
tees up the open H1/H3 questions — mode-dependence and n*(shape, B) — that D2/D3 are meant to
answer.

---

## 7. Two honest falsifications (kept in the package, not buried)

**(a) Launch-axis optimization does not pay off on total energy.** Optimizer-sited launch vs
naive centroid pad: in [C3] every pooled aggregate straddles 0 for all three algorithms (tgc
−4.4 kJ, classic −1.4 kJ, kmeans −2.5 kJ, all CI-straddle-0), reproducing the NOW-03 null [C3].
In [NOW-03 clean] it is a wash for tgc/kmeans and **net-harmful for classic (+42.5 kJ**, worse
in 33/45 cells) [NOW-03 clean]; [NOW-03 shipped] agrees (tgc/kmeans wash, classic −22 kJ) [NOW-03
shipped]. **Honest null: launch-axis optimization is a total-energy wash (harmful for classic in
the clean grid), reproduced across three runs** [C3], [NOW-03 clean], [NOW-03 shipped].

**(b) The isoperimetric shape-law relied on a banned comparison and does not reproduce.** The
pipeline `summary.md` reported a strong H2 correlation `corr_isoperimetric = +0.787`, but that
figure is computed on the **best-of-baseline** (winner's-curse) optimizer-pad advantage, which
the analysis explicitly bans [NOW-03 shipped]. On the bias-free clean naive-pad decomposition
contrast it does **not** reproduce: the energy correlation is ≈0 (+0.042) and is *itself* an
artefact of one shape (dropping c_shape moves it to −0.611); on the shipped naive pad it is only
a weak, LOO-unstable +0.44 [NOW-03 clean], [NOW-03 shipped]. **A simple solidity/isoperimetric
shape-law for the decomposition advantage is not supported** [NOW-03 clean]. C3 does not
resurrect it — it too excludes the banned best-of-baseline H2 figure [C3]. *(This falsification
is load-bearing for the §9 shape-narrowing decision.)*

---

## 8. Model limitations (stated up front for the supervisor)

1. **ENG-01 turn-aerodynamics is NOT implemented.** The energy model has no turn/bank
   aero-penalty term (TODO.md, backlog). This is acceptable for a **multirotor** platform (the
   thesis platform), where turn energy is small. It is **the trigger** for any **fixed-wing**
   claim: a fixed-wing energy result would require ENG-01 first, and ENG-01 re-baselines energy,
   so it must be done *before* any final re-runs or not at all (TODO.md). Flag this as a scope
   boundary, not a bug.

2. **The static fractional battery floor breaks on the scale axis.** The energy-map arm keeps a
   static *fractional* terminal floor (`critical=0.10`); a fixed fraction of a fixed battery is
   an absolute energy margin that does **not** scale with area. This limitation was identified in
   EM-01/B2 and is re-validated as a concern on the scale axis: at 1 km² the 0.10 floor was never
   reached (min 0.1077) and the "willing-map over conservative floor" risk did not materialise
   [C1], but [C1] explicitly scopes that to the 100-rep, 1 km² sample and names the 4–16 km²
   scale axis as where the fractional floor and the deeper-sortie margin would be stressed first.

3. **The C3 success CSV carries no PARTIAL/INCOMPLETE split.** `shape_sweep.csv` surfaces only
   `success_mean`, not a SUCCESS/PARTIAL/INCOMPLETE breakdown, so the ~0.3–3% non-success in the
   worst decomposition cells (rect_8_1 n=2, star_5 n=2, disk n=2) cannot be attributed to a cause
   from C3 alone — contrast C1/C2, which carry per-outcome rows [C3]. Recorded as a data
   limitation, not a semantics decision.

---

## 9. Open decisions for the supervisor (this tees up D2)

The scale experiment (D2 = scope decisions, D3 = `scale_sweep_v2.md` rewrite) needs the
supervisor's call on the following. **Crucially: the runner (`run_area_obstacle_sweep`) is
already BUILT and byte-identity-gated against the shipped shape-sweep primitives, and every
K1–K4 quantity is a CLI flag** (TODO.md, D3 machinery). So these are **cheap, re-runnable dials,
not irreversible commitments** — the decisions set defaults, not architecture.

1. **Shape narrowing: 9 shapes → 1 (L-shape).** Defensible, because the shape-descriptor law is
   falsified (§7b) — there is no supported shape-law to sweep 9 shapes for, and the L-shape is a
   representative non-convex family. **But this is the supervisor's call**, since it narrows the
   external validity of the shape story. (The n-dependence in §6 is the axis that *does* carry a
   real effect and is worth preserving.)

2. **Obstacle count: primary axis vs spot-check.** The runner sweeps obstacle COUNT as a first-
   class axis (fixed obstacle size, `--densities`), but whether obstacle-count is a *primary*
   experimental axis or a *spot-check* around one nominal density is a scope decision. Obstacles
   demonstrably shift contrast *levels* without overturning rankings in NOW-03 [NOW-03 shipped],
   which argues a spot-check may suffice — supervisor decides.

3. **K1–K4 sizing.** Area tiers (K1), n-grid ceiling (K2 — §6 argues for pushing past n=4),
   reps/CI target per cell (K3), and the clean/shipped proportion (K4). All four are CLI flags on
   the built runner; the §6 n-dependence and the §8.2 scale-floor concern are the two effects the
   sizing should be powered to resolve.

---

## Appendix — boundaries honoured in this draft

- The read-outs were delivered **without thesis verdicts**; this draft synthesizes and
  interprets for a supervisor audience but does not overclaim beyond what they support.
- **No success predicate, outcome class, or metric definition is re-decided** here — all are
  reported as fixed by the prior experiments (CLAUDE.md working rule 7).
- **Nothing was recomputed from raw run data.** Every number cites one of the six read-outs.
- The E1 provenance-reconstruction note is flagged wherever NOW-03 shipped/clean numbers are
  cited (preamble + §5, §7).
- Superseded-data rule honoured: pre-EM-01 STUDY-01 is cited only as "superseded", never for a
  number.
