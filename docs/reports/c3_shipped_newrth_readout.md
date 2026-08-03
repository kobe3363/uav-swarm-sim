# C3 — Shipped shape-sweep re-run under the new (arm-4) RTH (read-out)

**Data source (single source of truth):** `runs/shape_sweep_shipped_newrth/`
(`run.json`, `summary.md`, `shape_sweep.csv`, `contrasts.csv`). Every number in
this document is computed **only** from that folder (superseded-data rule).

**Role:** *data analyst read-out* — exact numbers, tables, methodology notes.
**No final thesis verdict**; intermediate conclusions are for the coordinator.
Structured to slot into the later C1+C2+C3 unified package (D1).

**Provenance line.** Run folder `runs/shape_sweep_shipped_newrth/`
(`run_id=231d6da1ed67`), git commit **`81a9226`**, config
**`config/shape_sweep_newrth.yaml`** (arm-4 "full-map" RTH), `--mode shipped
--budget full --jobs auto`, `master_seed=42`, Linux/py3.12, wall time 29,737 s
(≈8.26 h). This is a **fresh shipped result under the new stack** — **not** a
paired delta against the superseded old-RTH NOW-03 shipped run
(`runs/shape_sweep_shipped`, `docs/reports/s5_shipped_readout.md`). Any
comparison to NOW-03 in §E is **qualitative only**.

---

## Phase 0 — Run identity & STOP checks

### 0.0 — Run identity (from `run.json` + `config/shape_sweep_newrth.yaml`)

| field | value | source |
|---|---|---|
| run_name | `shape_sweep_shipped_newrth` | `run.json:run_name` |
| git commit | `81a9226` | `run.json:software.git_commit` |
| config | `config/shape_sweep_newrth.yaml` | `run.json:command` |
| mode | **shipped** (obstacle_density = 8/km², λ=0, obstacles paired per seed) | `run.json:summary.mode`, `summary.md:3` |
| grid | **9 shapes × n ∈ {2, 4}** = 18 cells | `run.json:summary.{shapes,ns}` |
| variants | 7 (weighted_voronoi, tgc_basic, classic_voronoi, kmeans + tgc/classic/kmeans `_naive_launch`) | `shape_sweep.csv` |
| N per cell | **100** paired replications (fixed-N, no early stop) | `shape_sweep.csv:n_runs` (all = 100) |
| total rows | 126 variant-rows (18 × 7); 882 contrast-rows (18 × 7 metrics × 7 contrasts) | file line counts |
| master_seed | 42 | `config…:sim.master_seed` |

**Arm-4 "full-map" config identity — CONFIRMED** (file:line, `config/shape_sweep_newrth.yaml`):

- `rth.energy_map.{enabled, decide, route, zone_demotion} = true` (lines 276–279) —
  the dynamic cost-to-go map is the **sole normal RTH decider**.
- `battery_zones.critical = 0.10` (line 240) — the arm-4 TERMINAL failsafe floor
  (`ARM4_TERMINAL_FLOOR`); under `zone_demotion` it no longer gates a CRITICAL guard.
- `coverage.transit_free_space = true` (line 157) — FIX-B1 obstacle-aware S1 transit,
  matching the C1 base (`study01_demand.yaml:140`) and the C2 freeze
  (`study01_demand_newrth.yaml:152`).

This **matches the arm-4 winner** frozen in `docs/reports/c1_stage5_ab_readout.md`.
→ **STOP condition (0) NOT triggered** — this is the arm-4 new-RTH shipped config.

### 0.1 — INVARIANT GATE: `weighted_voronoi − tgc_basic`

**PASS.** All **126** `weighted_voronoi − tgc_basic` contrast rows (18 cells × 7
metrics) are `exact_zero = True` with `diff_mean = 0` and `diff_ci = 0`; 0
violations. Independently corroborated by `run.json:summary.readout`
(`null_all_exact = true`, `null_max_abs = 0.0`). The homogeneous-fleet λ=0 null
(weighted ≡ tgc byte-identically at any scale) **holds exactly**. → **STOP
condition (1) NOT triggered.** weighted_voronoi carries no independent
information and is omitted from the peer analysis below.

### 0.2 — Structural sanity gates

| gate | check | result |
|---|---|---|
| G-B | full 9 × 2 × 7 = 126 variant-rows, no missing/crashed cell | **PASS** — 0 missing |
| G-C | every variant × cell has N = 100 | **PASS** — `n_runs` all = 100 |
| G-P | every contrast row `n_pairs = 100`, `dropped_pairs = 0` | **PASS** — 882/882 fully paired, zero drops |
| G-E | shipped-only: obstacle draws make every variant stochastic (CI > 0) | **PASS** — all cells carry non-zero CI (obstacle field) |

**Sign convention** (all contrasts `diff = tgc − peer`, i.e. left − right of the
contrast string): `total_energy`, `makespan`, `energy_imbalance`,
`length_imbalance`, `swaps` — **lower is better ⇒ negative diff favours TGC**.
`efficiency` (SMDP throughput ratio π(S2)/π(overhead), *not* energy efficiency —
CLAUDE.md) and `success` — **higher is better ⇒ positive diff favours TGC**.

**CI definitions.** *Per-cell paired CI* = the `diff_ci` column of
`contrasts.csv` (95 % normal-approx half-width `Z·√(var/n)`, Z = 1.959963985,
ddof = 1, on the per-replication paired difference — mirrors
`metrics/convergence.ci_half_width`). "CI-excludes-0" means `|diff_mean| >
diff_ci`. *Pooled aggregate* = mean of the per-cell `diff_mean` values ± a
**between-cell dispersion** half-width (Z·sd/√n_cells). This pooled CI measures
**between-shape dispersion**, NOT a within-cell paired CI — it is the same
aggregate the NOW-03 read-outs used, reported with that caveat throughout.
**Winner's-curse ban honoured:** the two peers are always reported separately,
each as a paired contrast; no best-of-baseline is used.

---

## A — Decomposition headline (new-RTH shipped)

Pooled over all 18 cells (mean per-cell paired diff ± between-cell CI), plus the
per-cell CI-excludes-0 tally and direction.

### A.1 — `tgc_basic − classic_voronoi`

| metric | pooled Δ (± btwn-cell CI) | pooled sig | cells CI-excl-0 | favour TGC / classic | direction |
|---|---|---|---|---|---|
| total_energy (J) | **−39,093 ± 24,100** | EXCL-0 | 11/18 | 10 / 1 | TGC lower |
| makespan (s) | **−1,156 ± 233** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| energy_imbalance | **−0.652 ± 0.171** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| length_imbalance | **−0.754 ± 0.202** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| swaps | **−0.659 ± 0.358** | EXCL-0 | 13/18 | 11 / 2 | TGC fewer |
| efficiency | **−0.520 ± 0.273** | EXCL-0 | 14/18 | 2 / 12 | **classic higher** |
| success | +0.0017 ± 0.0029 | straddles-0 | 0/18 | — | wash |

**Reading (numbers only).** TGC uses **less total energy** than classic_voronoi
(−39 kJ pooled, CI excludes 0; 10/18 cells resolve to TGC, only disk n=2 resolves
to classic — see §B), and holds a **strong, unanimous balance/time advantage**:
lower makespan, lower energy- and length-imbalance in **all 18 cells** (CI
excludes 0 every cell), and fewer swaps in 11/18. The **only** metric that favours
classic is `efficiency` (SMDP throughput ratio) — classic is higher in 12/18
(pooled −0.52, CI excludes 0). `efficiency` is a throughput ratio orthogonal to
energy/balance (CLAUDE.md), and is the same throughput/energy tension NOW-03
recorded. Success is a wash (0/18 resolve).
Pooled `total_energy` is **not single-cell-carried**: leave-one-out over the 18
cells keeps the pooled value in **[−42.5 kJ, −29.7 kJ]**, all 18 still exclude 0.

### A.2 — `tgc_basic − kmeans`

| metric | pooled Δ (± btwn-cell CI) | pooled sig | cells CI-excl-0 | favour TGC / kmeans | direction |
|---|---|---|---|---|---|
| total_energy (J) | **−12,948 ± 8,639** | EXCL-0 | 7/18 | 6 / 1 | TGC lower |
| makespan (s) | **−244.1 ± 70.2** | EXCL-0 | 17/18 | 17 / 0 | TGC lower |
| energy_imbalance | **−0.124 ± 0.028** | EXCL-0 | 18/18 | **18 / 0** | TGC lower |
| length_imbalance | **−0.122 ± 0.032** | EXCL-0 | 17/18 | 17 / 0 | TGC lower |
| swaps | −0.111 ± 0.140 | straddles-0 | 12/18 | 10 / 2 | (pooled wash) |
| efficiency | **−0.320 ± 0.144** | EXCL-0 | 14/18 | 2 / 12 | **kmeans higher** |
| success | +0.0006 ± 0.0019 | straddles-0 | 0/18 | — | wash |

**Does the new-RTH shipped stack separate TGC from kmeans?** — **Yes, modestly,
on total energy.** Where NOW-03 shipped found TGC ≈ kmeans on total_energy (a
wash, −8.2 kJ ± 15.8, CI straddled 0), the arm-4 new-RTH shipped run resolves a
**small TGC-favouring separation of −12.9 kJ ± 8.6 (CI excludes 0)**. Critically,
this is **NOT carried by a single cell**: leave-one-out over all 18 cells keeps
the pooled value in **[−15.0 kJ, −10.4 kJ]**, every one still excluding 0 (biggest
mover: drop star_5 n=4 → −10.4 kJ). So — unlike the NOW-03 *clean* c_shape-n=2
outlier that carried its aggregate — here the separation is **distributed across
cells**, not an artefact of one. The magnitude is small (≈0.5 % of a ~2.4 MJ
mission) and only 7/18 cells resolve individually, so this is best stated as a
**modest but robust** TGC energy edge over kmeans, not a decisive one.
On makespan and both imbalance metrics TGC beats kmeans nearly unanimously
(17–18/18). On `efficiency` kmeans is higher (12/18, pooled −0.32, CI excludes
0) — the same throughput/energy tension as against classic. swaps and success
are washes.

### A.3 — `weighted_voronoi − tgc_basic`

**≡ 0 exactly** on all 18 cells × 7 metrics (invariant gate §0.1). No independent
information.

### A.4 — Regime overlay (`total_energy`, pooled by analytical regime tag)

Regime tags are the analytical, **obstacle-free** A2 tags (same convention as
NOW-03); in shipped a "BATTERY-LIMITED" tag is a **lower bound** on battery
pressure (obstacles add unmodelled detours). This run's tag distribution is
**BATTERY-LIMITED 10, FUEL-SURPLUS 8, no BORDERLINE** — it differs from NOW-03
shipped (10 / 6 / 2 BORDERLINE) because the arm-4 `critical=0.10` floor enlarges
usable battery `B_usable`, pushing l_shape n=4 and star_5 n=4 from BORDERLINE to
FUEL-SURPLUS. **Caveat:** regime is confounded with n here — every FUEL-SURPLUS
cell is an n=4 cell, and 7 of 10 BATTERY-LIMITED cells are n=2 (pinwheel n=4 is
the lone battery-limited n=4). So the regime split below is largely the §B
n-split re-labelled.

| contrast | BATTERY-LIMITED (10) | FUEL-SURPLUS (8) |
|---|---|---|
| tgc − classic | −31,192 ± 39,390 (straddles-0) | **−48,970 ± 24,157 (EXCL-0)** |
| tgc − kmeans | −9,918 ± 12,304 (straddles-0) | **−16,736 ± 12,260 (EXCL-0)** |

---

## B — n-dependence (candidate finding, tested with CIs)

**The dispatch hypothesis is CONFIRMED for TGC − classic:** the TGC total-energy
advantage over classic_voronoi is **systematically stronger at n=4 than n=2**,
and at n=2 it is not resolvable in aggregate.

Pooled `total_energy` (tgc − classic), split by n:

| n | pooled Δ (± btwn-cell CI) | significance |
|---|---|---|
| n = 2 | **−12,632 ± 16,893** | **straddles-0** |
| n = 4 | **−65,555 ± 38,865** | **EXCL-0** |

Per-shape `total_energy` (tgc − classic), n=2 vs n=4 (`*` = per-cell CI excludes 0):

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

**Reading (numbers only).** Four shapes show a raw **sign flip** between n=2 and
n=4 (square, rect_2_1, disk, l_shape — exactly the set the dispatch flagged), but
of these **only disk n=2 is a resolved reversal**: +19,325 ± 16,508 (CI excludes
0 ⇒ classic significantly lower energy at disk n=2), corroborated in the
tgc−kmeans contrast (disk n=2 = +22,432 ± 14,162 *, kmeans also lower). The other
three n=2 "flips" are positive-mean but **CI-straddles-0** — within noise, not
resolved reversals. At n=4 every shape is TGC-favouring and 7/9 resolve. So the
honest statement is: **TGC's energy advantage over classic is an n=4 phenomenon;
at n=2 it collapses to a wash in aggregate, with one shape (disk) genuinely
reversing.** This connects to the open **H1/H3** mode-dependence / n*(shape,B)
questions — it is a **reportable finding**, not noise.

The n-split is *not* echoed on the balance metrics: pooled tgc−classic makespan
(−1,090 vs −1,221 s) and energy_imbalance (−0.43 vs −0.87) both exclude 0 at
**both** n. Only `efficiency` shifts (n=2 −0.643 EXCL-0 favouring classic; n=4
−0.398 straddles-0). So the n-dependence is specific to **total_energy** (and
throughput), not to workload balance.

For **tgc − kmeans** the n-pattern is **mixed, not cleanly monotone**: n=4
stronger for square/rect_4_1/rect_8_1, but n=2 stronger (and kmeans-favouring at
disk n=2 +22,432 *) elsewhere. No clean n-law for the kmeans contrast.

---

## C — Launch axis (does optimizer-sited launch pay off on energy?)

`diff = optimized-launch − naive-centroid-launch`; **negative = the optimizer
saves energy**. Pooled over 18 cells.

| algo | pooled Δ total_energy (J) | significance | cells optimizer-saves (neg) | cells CI-excl-0 |
|---|---|---|---|---|
| tgc | **−4,419 ± 7,408** | straddles-0 | 9/18 | 6/18 |
| classic | **−1,424 ± 11,286** | straddles-0 | 8/18 | 9/18 |
| kmeans | **−2,501 ± 6,029** | straddles-0 | 9/18 | 8/18 |

**Reading (numbers only).** On total energy the optimizer-sited launch **does not
reliably beat the naive centroid pad** for **any** of the three algorithms — every
pooled aggregate straddles 0, with the optimizer saving energy in only ~half the
cells each way. This **reproduces NOW-03's falsification** that launch-axis
optimization does not pay off on total energy (NOW-03 shipped §3). One quantitative
shift vs NOW-03: there, classic's launch term was net-negative (−21.9 kJ); here
under the new stack it too is a wash (−1.4 kJ, CI straddles 0). **Honest null:
launch-axis optimization is a total-energy wash for all three decompositions in the
new-RTH shipped grid.** (The optimizer's stated objective is swap-waste/energy in
battery-limited cells; that benefit does not surface on the total-energy aggregate
here — flagged, consistent with NOW-03.)

---

## D — Success / outcome

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

**No variant collapses** — every variant's per-cell success stays ≥ 0.97, mean
≥ 0.9967. The decomposition contrast on success is a **wash**: 0/18 cells resolve
CI-excludes-0 for either tgc−classic or tgc−kmeans. Success is equivalent across
decompositions at this sample size, consistent with the "equivalent success"
reading of C1.

**Limitation.** `shape_sweep.csv` surfaces only `success_mean`, not a
SUCCESS/PARTIAL/INCOMPLETE breakdown, so PARTIAL/INCOMPLETE counts **cannot be
separated here** (contrast C1/C2, which carry per-outcome rows). The ~0.3–3 %
non-success in the worst cells (rect_8_1 n=2, star_5 n=2, disk n=2) cannot be
attributed to a cause from these aggregates. Recorded as an honest data limitation;
not a semantics decision.

---

## E — Qualitative comparison to NOW-03 shipped

**This is qualitative, NOT a paired delta** — the NOW-03 shipped run
(`runs/shape_sweep_shipped`, `docs/reports/s5_shipped_readout.md`) used the old
static-0.40 RTH with `transit_free_space` OFF, so it is **not byte-comparable** to
this new-RTH + transit-on stack. Only the **ordering/direction** is compared.

| finding | NOW-03 shipped | C3 new-RTH shipped | survives? |
|---|---|---|---|
| TGC < classic on total energy | −54 kJ pooled (CI excl 0) | −39 kJ pooled (CI excl 0) | **Yes** (direction holds) |
| TGC balance edge vs classic (makespan, imbalance) | 18/18-ish, CI excl 0 | 18/18, CI excl 0 | **Yes**, unchanged |
| TGC ≈ kmeans on total energy | wash (−8 kJ, straddles 0) | **modest separation** (−13 kJ, CI excl 0, distributed) | **Shifts**: wash → modest TGC edge |
| TGC balance edge vs kmeans | small, mostly excl 0 | 17–18/18, CI excl 0 | **Yes**, unchanged |
| kmeans/classic higher `efficiency` (throughput) than TGC | kmeans +0.15 excl 0; classic wash | kmeans +0.32, classic +0.52, both excl 0 | **Yes, strengthened** |
| Launch-axis optimization pays off on energy | falsified (wash) | falsified (wash) | **Yes** (null reproduced) |

**Numbers-only summary.** The two NOW-03 headline directions — **TGC ≫
classic** (energy + balance) and the **kmeans/classic throughput edge** — **survive
the new-RTH stack**. The one qualitative change is the **TGC ≈ kmeans energy
wash tightening into a modest, distributed TGC-favouring separation** (−13 kJ, CI
excludes 0, LOO-robust) under arm-4. A **new, RTH-independent** structural feature
this run surfaces is the **n-dependence of the TGC−classic energy advantage** (§B:
n=4 strong, n=2 a wash with a genuine disk-n=2 reversal), which the NOW-03 shipped
grid — same n∈{2,4} subset — did not foreground.

---

## Methodology notes

1. **CI definition:** per-cell paired CIs are the `diff_ci` column of
   `contrasts.csv` (95 % normal-approx half-width `Z·√(var/n)`, Z = 1.959963985,
   ddof = 1, on the per-replication paired difference). Pooled aggregates use a
   **between-cell dispersion** half-width (Z·sd/√n_cells) over the per-cell
   `diff_mean` values, labelled as such — this is a between-shape dispersion, NOT
   a within-cell paired CI.
2. **All 882 contrast rows are fully paired** (`n_pairs = 100`, `dropped_pairs =
   0`) — no complete-case dropping was needed for any contrast/metric/cell.
3. **No per-replication raw data** ships in the folder (only cell means/CIs and
   precomputed paired contrasts). Contrasts not in `contrasts.csv` (notably
   `tgc_naive − classic_naive`) are unavailable here and are **not** reported.
4. **Winner's-curse ban honoured:** the two peers (classic, kmeans) are reported
   separately, each as a paired contrast with a CI on the difference; no
   best-of-baseline is used. Where `run.json:summary.readout` / `summary.md` carry
   best-of-baseline H1/H2 statistics (`tgc_adv_vs_best_baseline_by_n`,
   `h2_corr_isoperimetric = +0.868`), those are the banned winner's-curse figures
   and are **not** used in this read-out.
5. **weighted_voronoi is omitted** from the peer analysis — byte-identical to
   tgc_basic for this homogeneous full-battery λ=0 fleet (invariant gate §0.1
   exact).
6. **Regime tags are analytical & obstacle-free** for shipped; BATTERY-LIMITED is
   a lower bound on true battery pressure, and regime is confounded with n in this
   grid (§A.4).
7. **Provenance boundary:** this is a data-analyst read-out with intermediate
   conclusions only. No thesis verdict is issued; interpretation and the unified
   C1+C2+C3 narrative are the coordinator's (D1).
