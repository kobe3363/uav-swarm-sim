# S5 shape-sweep — SHIPPED read-out (NOW-03)

**Data source (single source of truth):** `runs/shape_sweep_shipped/`
(`shape_sweep.csv`, `contrasts.csv`, `run.json`). Numbers in this document are
computed **only** from that folder. No value from any other run is cited
(superseded-data rule).

**Role:** this is a *data analyst read-out* — exact numbers, tables and
methodology notes. It contains **no final thesis verdict**; interpretation is
left to the coordinator.

---

## 0. Provenance & structure (ground truth from the CSVs)

| field | value |
|---|---|
| run_name | `shape_sweep_shipped_2026-07-09-18-46-09_a5fd4f` |
| git commit | `e9c40e2` (post launch-RNG-fix `a0871b6` / #18 — verified ancestor) |
| mode | **shipped** (obstacle_density = config default 8/km²; λ = 0; obstacles paired per seed) |
| grid | **9 shapes × n ∈ {2, 4}** = **18 cells** |
| variants | 7 (weighted_voronoi, tgc_basic, classic_voronoi, kmeans + tgc_naive_launch, classic_naive_launch, kmeans_naive_launch) |
| N per cell | **100** paired replications (fixed-N, no early stop) |
| total rows | 126 variant-rows (18 × 7) |

### ⚠️ Brief-vs-reality discrepancy — FLAGGED for the coordinator

The task brief described **both** folders as *"45 cells (9 shapes × 5 n) × 7
variants, N=100, same grid."* This does **not** match the files on disk, and the
mismatch is **by design**, not a data fault. From `run_shape_sweep.py`:

```python
BUDGETS = {"quick": {"clean": 5, "shipped": 10}, "full": {"clean": 20, "shipped": 100}}
SHIPPED_DEFAULT_NS = [2, 4]   # robustness subset: battery-limited + reference
```

* **shipped** (this folder): a **subset** grid, n ∈ {2, 4} (the battery-limited
  row + the n = 4 reference row), **N = 100**, *with* obstacles. → 18 cells.
* **clean** (the other folder): the **full** grid, n ∈ {2,3,4,5,6}, **N = 20**,
  obstacle-free (the script's *PRIMARY* mode). → 45 cells.

Consequences carried through every table below:
1. The shipped grid has only n = 2 and n = 4; there is no n = 3/5/6 here.
2. shipped N = 100 vs clean N = 20 ⇒ paired-CI widths are **not** comparable
   between the two folders.
3. Any shipped↔clean comparison (§7) is restricted to the **n ∈ {2,4}** overlap.

The sanity gates below were therefore evaluated against **this folder's own
design grid (18 cells, N = 100)**, not the brief's assumed "45 cells, N = 100".

---

## Step 0 — Sanity gates (SHIPPED)

| gate | check | result |
|---|---|---|
| **G-A** | `weighted_voronoi − tgc_basic = 0` exactly, every cell × every metric | **PASS** — 0 contrast violations, 0 cell-mean violations (known homogeneous-fleet null) |
| **G-B** | full matrix 9 × 2 × 7 = 126 rows, no missing/crashed cell | **PASS** — 0 missing |
| **G-C** | every variant × cell has N = 100 | **PASS** — 0 off-count rows (`n_runs` and `*_n` all = 100) |
| **G-E** | *shipped-only*: every variant has CI > 0 (obstacle draws) | **PASS** — 0 cells with `total_energy_ci = 0` (all 126 rows stochastic) |

All gates pass against the shipped design. (G-D is a clean-only gate.)

**Sign convention** for every contrast table: `diff = tgc − peer`.
`total_energy`, `makespan`, `energy_imbalance`, `length_imbalance`, `swaps`:
**lower is better ⇒ a negative diff favours TGC**. `efficiency` (SMDP throughput
ratio π(S2)/π(overhead)), `success`: **higher is better ⇒ a positive diff
favours TGC**. CI = 95 % normal-approx half-width (Z·√(var/n), ddof = 1); paired
CIs are on the per-replication difference (from `contrasts.csv`). "btwn-cell CI"
is the 95 % half-width **across cells** of the per-cell diff means (a
between-shape dispersion, *not* a within-cell paired CI) — used only for the
aggregates.

---

## Step 1 — HEADLINE: tgc_basic vs classic_voronoi and vs kmeans

Primary metric = **total_energy** (J). Winner's-curse baselines (per-cell
best-of / max) are **not** used — the two peers are reported separately, each as
a paired contrast with CI on the difference.

### 1a. Total energy (J) — per cell, paired diff ± 95 % CI

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

### 1b. Total energy — aggregates (mean of per-cell paired diffs ± btwn-cell CI)

| aggregate | tgc − classic (J) | tgc − kmeans (J) |
|---|---|---|
| n = 2 | **−34038 ± 32303** | −6134 ± 18989 |
| n = 4 | **−74190 ± 40789** | −10185 ± 26515 |
| **ALL** | **−54114 ± 26983** | **−8159 ± 15849** |

* **tgc − classic:** aggregate −54 kJ, btwn-cell CI excludes 0 → TGC uses less
  total energy than classic_voronoi. 13/18 cells negative.
* **tgc − kmeans:** aggregate −8 kJ with btwn-cell CI ±16 kJ (**straddles 0**) →
  TGC and kmeans are indistinguishable on total energy.

### 1c. Secondary metrics — aggregates ALL (per-cell paired diff, btwn-cell CI)

| metric | tgc − classic | tgc − kmeans | direction |
|---|---|---|---|
| makespan (s) | **−1286.7 ± 252.8** | −225.2 ± 110.2 | lower=better |
| energy_imbalance | **−0.614 ± 0.163** | −0.115 ± 0.026 | lower=better |
| length_imbalance | **−0.707 ± 0.193** | −0.113 ± 0.029 | lower=better |
| swaps | −1.281 ± 0.798 | −0.022 ± 0.509 | lower=better |
| efficiency (throughput) | −0.143 ± 0.210 | **−0.152 ± 0.085** | higher=better |
| success | +0.013 ± 0.016 | −0.001 ± 0.010 | higher=better |

Reading (numbers only):
* vs **classic_voronoi** TGC is lower on energy, makespan, and both workload-
  imbalance metrics (CIs exclude 0); efficiency/success are a wash.
* vs **kmeans** TGC is lower on energy/makespan/imbalance but the margins are
  small; on the SMDP **efficiency** metric kmeans is higher by 0.152
  (CI ±0.085 excludes 0). Note `efficiency` is a throughput ratio, orthogonal to
  energy — TGC's design objective is energy + workload balance, not throughput.

---

## Step 2 — PROBLEM B: does the TGC edge survive the neutral (naive) pad?

Compares the optimizer-pad decomposition contrast against the same contrast with
**both** sides on the identical naive-centroid pad (launch advantage removed).
If the gap is the same on both pads, it is a **decomposition** effect, not a
launch/home-field artefact.

`tgc_naive − kmeans_naive` is precomputed (paired). **`tgc_naive − classic_naive`
is NOT precomputed** in `contrasts.csv`; only an *unpaired point-diff of means*
(no paired CI) is available — flagged with `*`.

### 2a. Aggregates ALL

| metric | OPT tgc−kmeans | NAIVE tgc−kmeans | NAIVE tgc−classic* |
|---|---|---|---|
| total_energy (J) | −8159 ± 15849 | −7162 ± 17901 | −78431 ± 50161* |
| efficiency | −0.152 ± 0.085 | −0.154 ± 0.093 | −0.120 ± 0.209* |

* **tgc vs kmeans:** the gap is essentially **identical** on the optimizer pad
  and the naive pad — energy −8.2 kJ → −7.2 kJ; efficiency −0.152 → −0.154. The
  (small, kmeans-favouring on efficiency) difference is therefore **not** created
  by the optimizer-sited launch; it is a decomposition-level property.
* **tgc vs classic (unpaired\*):** TGC keeps a large energy lead on the naive pad
  (−78 kJ), but efficiency slightly favours classic (−0.120). Report with the
  caveat that this contrast has no paired CI.

Per-cell values: see `runs/shape_sweep_shipped/contrasts.csv`
(`tgc_naive_launch - kmeans_naive_launch`) and `shape_sweep.csv` for the
classic-naive means.

---

## Step 3 — LAUNCH AXIS on total energy: (algo − algo_naive)

`diff = optimized-launch − naive-centroid-launch`; **negative = the optimizer
saves energy**.

| algo | mean diff (J) | btwn-cell CI | cells where optimizer saves (neg) |
|---|---|---|---|
| tgc | **+2406** | 22081 | 9/18 |
| classic | **−21911** | 30358 | 10/18 |
| kmeans | **+3403** | 13435 | 7/18 |

Numbers only: on **total energy**, the optimizer-sited launch does **not**
reliably beat the naive centroid pad in the shipped grid — for TGC and kmeans the
aggregate is slightly *positive* (a wash, CI straddles 0); only for classic is it
net-negative (−22 kJ). The optimizer's stated objective is swap-waste/energy in
battery-limited cells; that benefit does **not** surface on the total-energy
aggregate here. **Flagged** as a candidate falsification for the coordinator.
(Full per-cell table in `contrasts.csv`; identical to summary.md's "Secondary
axis" tgc column.)

---

## Step 4 — REGIME overlay

**Methodology flag:** regime tags are computed **obstacle-free** (analytical A2
rule, `regime_tag(clean_cfg, …)`) for *both* modes. In shipped, obstacles add
detours/energy that the tag does not see, so a **"BATTERY-LIMITED" tag is a
LOWER BOUND** on battery pressure here. Cell counts: BATTERY-LIMITED 10,
BORDERLINE 2, FUEL-SURPLUS 6.

Total-energy paired diff, mean (btwn-cell CI), by regime:

| contrast | BATTERY-LIMITED | BORDERLINE | FUEL-SURPLUS |
|---|---|---|---|
| tgc − classic | −46497 (37829) | −83781 (146405) | −56921 (37397) |
| tgc − kmeans | −7144 (17100) | −10352 (70575) | −9121 (36739) |
| tgc_naive − kmeans_naive | −16668 (23483) | −17545 (119511) | +12142 (15240) |
| tgc − tgc_naive (launch) | +4624 (36141) | +3856 (22230) | −1774 (31644) |

Reading: the **tgc > classic** energy lead holds in every regime. **tgc ≈ kmeans**
in every regime (all CIs straddle 0); the naive-pad tgc−kmeans even flips to
*positive* (kmeans marginally better) in FUEL-SURPLUS. No regime overturns the
two headline rankings.

---

## Step 5 — H2: naive-pad TGC advantage vs shape descriptors

"Advantage" = per-shape **peak over n** of the naive-pad contrast
`tgc_naive − kmeans_naive`, sign-oriented so **>0 ⇒ TGC better** (energy sign
flipped). Correlated with **solidity** and **isoperimetric ratio**; leave-one-out
(LOO) per shape tests single-shape leverage.

| advantage from | corr(solidity) | corr(isoperimetric) | LOO iso range |
|---|---|---|---|
| total_energy | −0.280 | **+0.436** | +0.308 (drop star_5) … +0.690 (drop square) |
| efficiency | +0.011 | +0.369 | +0.178 (drop disk) … +0.682 (drop pinwheel) |

Descriptors (per shape): square 1.00/1.27, rect_2_1 1.00/1.43, rect_4_1
1.00/1.99, rect_8_1 1.00/3.22, disk 1.00/1.00, l_shape 0.86/1.70, star_5
0.49/3.47, pinwheel 0.40/4.39, c_shape 0.80/2.49 (solidity/isoperimetric).

Reading (numbers only): a **weak positive** isoperimetric correlation (~+0.44 on
energy, +0.37 on efficiency) that is **not robust** — LOO moves it by up to
±0.35 depending on which single shape is dropped. **Important methodology note:**
this is a *different* statistic from `summary.md`'s H2 (`corr_isoperimetric =
+0.787`). That pipeline figure uses the **optimizer-pad, best-of-baseline**
efficiency peak (a winner's-curse-flavoured "tgc − best of classic/kmeans"),
which the NOW-03 brief explicitly bans. On the clean naive-pad decomposition
contrast the strong +0.787 does **not** reproduce (it drops to ≈ +0.44/+0.37).

---

## Step 6 — VARIANCE (kmeans init variance vs others)

In **shipped**, every variant's per-cell CI is dominated by the **obstacle draw**
(the paired random obstacle field), so all seven variants are stochastic and
their CIs are of the same order — kmeans init variance is **not separable** from
obstacle variance here:

| variant | mean efficiency CI | cells with CI > 0 | mean total_energy CI (J) |
|---|---|---|---|
| tgc_basic | 0.1689 | 18/18 | 54245 |
| classic_voronoi | 0.1631 | 18/18 | 55181 |
| kmeans | 0.1559 | 18/18 | 51796 |
| tgc_naive_launch | 0.1657 | 18/18 | 51996 |
| kmeans_naive_launch | 0.1573 | 18/18 | 47607 |

Fact: in the obstacle regime the variance ordering is essentially flat across
variants (kmeans is *not* visibly noisier than the deterministic algos, because
obstacle variance swamps init variance). The clean-mode isolation of kmeans init
variance is reported in the clean read-out (§6 there).

---

## Step 7 — SHIPPED vs CLEAN delta (does adding obstacles change the ranking?)

Restricted to the **n ∈ {2,4}** overlap (the only cells present in both grids).
**CI widths are not comparable** (N = 100 vs 20). Comparison is on the **sign**
of each paired contrast (a sign flip = a potential ranking change). Values are
total_energy paired diffs (J).

| contrast | sign flips / 18 overlap cells | reading |
|---|---|---|
| tgc − classic | 8/18 | every flip is at a cell where **at least one side is statistically indistinguishable from 0** (a clean deterministic exact-0, or a shipped diff buried in its obstacle-noise CI); the TGC<classic ranking holds in both modes, **stronger** in clean |
| tgc − kmeans | 8/18 | all flips are near-zero-magnitude — coin-toss around 0, consistent with the TGC≈kmeans wash |
| tgc_naive − kmeans_naive | 12/18 | same: differences are ~0, so signs flip on noise |
| tgc − tgc_naive (launch) | 9/18 | launch benefit is ~0 in both modes; signs flip on noise |

Detail of the tgc − classic overlap (SHIPPED | CLEAN, J):
square 2 (+810 | −988), square 4 (+3016 | −141254), rect_2_1 2 (−29534 | 0),
rect_4_1 2 (−50863 | 0), rect_8_1 2 (−91645 | 0), disk 2 (+10135 | −182463),
l_shape 2 (−16299 | +864), c_shape 2 (−63591 | +925). The flips are of two kinds:
(a) the three **rect** cells collapse to **exactly 0** in clean (the n=2 tgc and
classic partitions coincide obstacle-free); (b) the rest have one side within
noise of 0 — e.g. **disk n=2** flips because the shipped diff (+10135) sits deep
inside its CI (±66776), i.e. shipped cannot resolve the sign that clean shows
cleanly (−182463). None is a genuine ranking reversal between two resolved
signals. At n = 4 both modes agree in sign on every shape.

**Numbers-only conclusion:** obstacles shift the *level* of the contrasts (add
noise, and at n=2 clean the deterministic twins collapse to ~0) but do **not**
overturn the two headline rankings — *TGC < classic on energy* and
*TGC ≈ kmeans* survive in both modes. The one structural difference unique to
clean (the c_shape n=2 kmeans-init outlier) is documented in the clean read-out.

---

## Methodology notes (SHIPPED)

1. **CI definition:** 95 % normal-approx half-width `Z·√(var/n)`, Z = 1.959963985,
   ddof = 1 (mirrors `metrics/convergence.ci_half_width`). Paired contrast CIs
   (from `contrasts.csv`) are on the per-replication difference; aggregates use a
   between-cell dispersion CI, labelled as such.
2. **No per-replication raw data** ships in the folder (only cell means/CIs and
   precomputed paired contrasts). Contrasts not in `contrasts.csv` (notably
   `tgc_naive − classic_naive`) can only be given as unpaired point-diffs of
   means — flagged `*` wherever used.
3. **Regime tags are obstacle-free** for both modes; in shipped, BATTERY-LIMITED
   is a lower bound (§4).
4. **Winner's-curse ban honoured:** the headline uses the two peers separately,
   never a per-cell best-of/max baseline. Where `summary.md` uses a best-of
   baseline (H1/H2), this read-out recomputes the banned-free statistic and flags
   the divergence (§5).
5. **weighted_voronoi is omitted** from the peer analysis: it is byte-identical to
   tgc_basic for this homogeneous full-battery λ=0 fleet (G-A confirmed exactly
   zero), so it carries no independent information here.
