# S5 shape-sweep — CLEAN read-out (NOW-03)

**Data source (single source of truth):** `runs/shape_sweep_clean_postfix/`
(`shape_sweep.csv`, `contrasts.csv`, `run.json`). Numbers in this document are
computed **only** from that folder (superseded-data rule).

**Role:** *data analyst read-out* — exact numbers, tables, methodology notes.
**No final thesis verdict**; interpretation is the coordinator's.

**CLEAN is the script's PRIMARY mode** (obstacle_density = 0; isolates the pure
shape effect and matches the analytical A2 regime tags exactly).

---

## 0. Provenance & structure (ground truth from the CSVs)

| field | value |
|---|---|
| run_name | `shape_sweep_clean_postfix` |
| git commit | `1214148` (post launch-RNG-fix `a0871b6` / #18) |
| mode | **clean** (obstacle_density = 0; λ = 0) |
| grid | **9 shapes × n ∈ {2,3,4,5,6}** = **45 cells** |
| variants | 7 (as shipped) |
| N per cell | **20** paired replications (fixed-N, no early stop) |
| total rows | 315 variant-rows (45 × 7) |

This folder is the **post-launch-RNG-fix re-run** flagged as required before
NOW-03 (the pre-fix clean grid was stale).

### ⚠️ Brief-vs-reality discrepancy — FLAGGED for the coordinator

The task brief described both folders as *"45 cells (9×5) × 7 variants, N=100,
same grid."* The grid count (45) is right for **clean**, but **N = 20 here, not
100** — this is by design (`BUDGETS["full"] = {clean: 20, shipped: 100}`). The
*shipped* folder is a different, smaller grid (18 cells, n ∈ {2,4}, N = 100).
Consequences:
* clean CIs are built on 20 paired reps (wider than shipped's 100-rep CIs);
* shipped↔clean comparison is limited to the n ∈ {2,4} overlap (see the shipped
  read-out §7).

Sanity gates below are evaluated against **this folder's design grid (45 cells,
N = 20)**, not the brief's "45 × N=100".

---

## Step 0 — Sanity gates (CLEAN)

| gate | check | result |
|---|---|---|
| **G-A** | `weighted_voronoi − tgc_basic = 0` exactly, every cell × metric | **PASS** — 0 violations (homogeneous-fleet null holds) |
| **G-B** | full matrix 9 × 5 × 7 = 315 rows, no missing cell | **PASS** — 0 missing |
| **G-C** | every variant × cell has N = 20 | **PASS** — 0 off-count rows |
| **G-D** | *clean-only*: all non-kmeans variants deterministic (CI = 0); kmeans/kmeans_naive CI > 0 allowed | **PASS** — 0 non-kmeans variance violations; kmeans variance appears in a small subset (see §6) |

All gates pass against the clean design.

**Sign convention** (as shipped): `diff = tgc − peer`; total_energy / makespan /
imbalance / swaps lower=better (**neg favours TGC**); efficiency (SMDP throughput
ratio) / success higher=better (**pos favours TGC**). Because clean is
obstacle-free, tgc_basic, classic_voronoi and the naive twins are **deterministic**
→ their per-cell CIs and most paired-contrast CIs are **exactly 0** (a diff of two
deterministic quantities). This is expected, not a defect. Aggregates still carry
a between-cell dispersion CI (labelled "btwn-cell CI").

---

## Step 1 — HEADLINE: tgc_basic vs classic_voronoi and vs kmeans

Primary metric = **total_energy** (J). Peers reported separately (no winner's-curse
best-of baseline).

### 1a. Total energy — aggregates (mean per-cell paired diff ± btwn-cell CI)

| aggregate | tgc − classic (J) | tgc − kmeans (J) |
|---|---|---|
| n = 2 | −42627 ± 62385 | −17875 ± 34983 |
| n = 3 | −74913 ± 78328 | −131 ± 433 |
| n = 4 | −133783 ± 60537 | +462 ± 415 |
| n = 5 | −110019 ± 69780 | +654 ± 735 |
| n = 6 | −136144 ± 71413 | +1038 ± 575 |
| **ALL** | **−99497 ± 31198** | **−3170 ± 7020** |

* **tgc − classic:** aggregate **−99.5 kJ**, btwn-cell CI excludes 0, negative at
  every n; the lead **grows with n** (−43 kJ at n=2 → −136 kJ at n=6). Robust
  TGC energy advantage over classic_voronoi.
* **tgc − kmeans:** aggregate **−3.2 kJ**, but this is **driven entirely by one
  outlier cell** (c_shape n=2, −160.7 kJ; see below). For n ≥ 3 the sign turns
  slightly **positive** (+0.5…+1.0 kJ, i.e. kmeans marginally lower energy) with
  CIs that barely exclude 0. Reading: **TGC ≈ kmeans** on total energy, kmeans
  fractionally better at higher n.

**Outlier flag — c_shape n = 2:** the single cell where kmeans has init variance
(CI > 0). Per-variant total energy there: tgc 714962 J, classic 714037 J,
**kmeans 875624 ± 135 J**, kmeans_naive 874859 ± 192 J. kmeans lands on a
different partition that costs **+160 kJ energy** but scores **higher SMDP
efficiency** (kmeans 4.497 vs tgc 3.920) — an energy↔throughput trade, not a
plain "kmeans failure". This one cell dominates the n=2 and ALL tgc−kmeans energy
aggregates; exclude it and tgc−kmeans is ≈ 0-to-slightly-positive everywhere.

### 1b. Secondary metrics — aggregates ALL

| metric | tgc − classic | tgc − kmeans | direction |
|---|---|---|---|
| makespan (s) | **−1292.7 ± 335.4** | −16.6 ± 31.8 | lower=better |
| efficiency (throughput) | **+0.481 ± 0.243** | −0.034 ± 0.026 | higher=better |
| swaps | −1.289 ± 0.296 | +0.000 ± 0.000 | lower=better |
| success | +0.000 ± 0.000 | +0.000 ± 0.000 | higher=better |

Reading: vs **classic** TGC is lower energy, lower makespan, fewer swaps *and*
higher SMDP efficiency (+0.48, CI excludes 0) in the obstacle-free grid. vs
**kmeans** every metric is a near-exact tie (makespan −16.6 s of a ~thousands-s
mission; efficiency −0.034; swaps and success identical). success = 1.0 for all
variants in every clean cell.

*(energy_imbalance / length_imbalance per-cell diffs are in `contrasts.csv`; they
mirror the shipped pattern — strongly negative vs classic, mildly negative vs
kmeans.)*

---

## Step 2 — PROBLEM B: does the TGC edge survive the neutral (naive) pad?

`tgc_naive − kmeans_naive` is precomputed (paired). **`tgc_naive − classic_naive`
is NOT precomputed** → unpaired point-diff of means only (no CI), flagged `*`.

### 2a. Aggregates ALL

| metric | OPT tgc−kmeans | NAIVE tgc−kmeans | NAIVE tgc−classic* |
|---|---|---|---|
| total_energy (J) | −3170 ± 7020 | −3655 ± 7041 | −57750 ± 24892* |
| efficiency | −0.034 ± 0.026 | −0.007 ± 0.025 | +0.220 ± 0.270* |

* **tgc vs kmeans:** optimizer-pad and naive-pad gaps are **the same** (energy
  −3.2 → −3.7 kJ, both dominated by the c_shape n=2 outlier; efficiency −0.034 →
  −0.007, i.e. even closer to 0 on the naive pad). The near-tie is a
  **decomposition** property, **not** manufactured by the optimizer launch — a
  home-field artefact is **not** present.
* **tgc vs classic (unpaired\*):** TGC keeps a large energy lead on the naive pad
  (−58 kJ) and here also a positive efficiency edge (+0.220), but with no paired
  CI (caveat).

Per-cell values in `contrasts.csv` / `shape_sweep.csv`.

---

## Step 3 — LAUNCH AXIS on total energy: (algo − algo_naive)

`diff = optimized − naive`; **negative = optimizer saves energy**.

| algo | mean diff (J) | btwn-cell CI | cells where optimizer saves (neg) |
|---|---|---|---|
| tgc | **+716** | 5643 | 21/45 |
| classic | **+42463** | 29542 | 12/45 |
| kmeans | **+232** | 5740 | 21/45 |

Numbers only: on total energy the optimizer-sited launch is a **wash for TGC and
kmeans** (aggregate ≈ +0.2…+0.7 kJ, CI straddles 0, ~half the cells each way) and
**net-harmful for classic** (+42.5 kJ; the optimizer pad is worse than the naive
centroid for classic_voronoi in 33/45 cells, with a few very large positive
outliers e.g. pinwheel, star_5, c_shape n=3). The optimizer's energy value does
**not** appear on the total-energy aggregate in the obstacle-free grid.
**Flagged** for the coordinator (consistent with the shipped finding §3).

---

## Step 4 — REGIME overlay

Regime tags here are the **exact analytical A2 tags** (clean = obstacle-free, so
the tag matches execution physics; no lower-bound caveat needed, unlike shipped).
Cell counts: BATTERY-LIMITED 20, BORDERLINE 4, FUEL-SURPLUS 21.

Total-energy paired diff, mean (btwn-cell CI), by regime:

| contrast | BATTERY-LIMITED | BORDERLINE | FUEL-SURPLUS |
|---|---|---|---|
| tgc − classic | −97514 (57911) | −82696 (108539) | −104586 (34665) |
| tgc − kmeans | −8102 (15739) | +409 (787) | +845 (392) |
| tgc_naive − kmeans_naive | −8095 (15841) | −239 (1084) | −76 (401) |
| tgc − tgc_naive (launch) | +5095 (8525) | +8616 (9275) | −4960 (8362) |

Reading: **tgc > classic** on energy in every regime (−83…−105 kJ). **tgc ≈
kmeans**: the BATTERY-LIMITED −8.1 kJ is again the c_shape n=2 outlier; in
BORDERLINE and FUEL-SURPLUS kmeans is marginally lower-energy (+0.4…+0.8 kJ, CIs
just exclude 0). The launch benefit for TGC flips sign by regime (mildly positive
= optimizer costs more in battery-limited/borderline; mildly negative = saves in
surplus) — all within ±9 kJ.

---

## Step 5 — H2: naive-pad TGC advantage vs shape descriptors

"Advantage" = per-shape **peak over n** of `tgc_naive − kmeans_naive`, oriented so
**>0 ⇒ TGC better**. Correlated with solidity and isoperimetric ratio; LOO tests
single-shape leverage.

| advantage from | corr(solidity) | corr(isoperimetric) | LOO iso range |
|---|---|---|---|
| total_energy | −0.054 | +0.042 | −0.611 (drop c_shape) … +0.175 (drop pinwheel) |
| efficiency | +0.731 | −0.644 | −0.831 (drop rect_8_1) … −0.503 (drop pinwheel) |

Reading (numbers only, honest falsification):
* **On total energy** the correlation is ≈ 0 (+0.042) and **entirely an artefact
  of one shape**: dropping c_shape (the kmeans-init outlier, adv = +161.6 kJ)
  moves it to **−0.611**. There is no stable descriptor law.
* **On efficiency** the per-shape advantages are **negligible** (all |adv| ≤ 0.06;
  star_5 and pinwheel ≈ 0). The apparent solidity +0.731 / isoperimetric −0.644
  correlations are computed on **noise-level magnitudes** and are LOO-unstable
  (isoperimetric ranges −0.83…−0.50 depending on the shape dropped) — and they
  point the **opposite** way to the shipped naive-pad result (+0.44).
* **Contrast with the pipeline `summary.md` figure** `corr_isoperimetric =
  −0.066` (clean readout in `run.json`) — itself computed on the banned
  best-of-baseline optimizer-pad advantage, which is why it differs again from
  both the shipped +0.787 and the naive-pad numbers here.

**Numbers-only conclusion:** a simple solidity/isoperimetric law for the
decomposition (TGC-vs-kmeans, naive-pad) advantage is **not supported** in the
clean grid — the signal is single-shape-driven and/or of negligible magnitude.

---

## Step 6 — VARIANCE (kmeans init variance vs others) — clean isolation

Because clean is obstacle-free, the deterministic algos have **exactly zero**
variance, so any non-zero kmeans CI is **pure `STREAM_KMEANS_INIT` init
variance** (clean isolation the shipped grid cannot give):

| variant | mean efficiency CI | cells with CI > 0 | mean total_energy CI (J) |
|---|---|---|---|
| tgc_basic | 0.0000 | **0/45** | 0 |
| classic_voronoi | 0.0000 | **0/45** | 0 |
| tgc_naive_launch | 0.0000 | **0/45** | 0 |
| kmeans | 0.0023 | **1/45** | 3 |
| kmeans_naive_launch | 0.0025 | **5/45** | 10 |

Facts:
* tgc/classic/tgc_naive are **fully deterministic** (0/45 cells with any
  variance) — the paired-seed determinism holds exactly.
* kmeans init instability is **rare, not pervasive**: only **1/45** kmeans cells
  (**c_shape n=2**, efficiency CI = 0.104) and **5/45** kmeans_naive cells carry
  any init variance. Elsewhere the k-means++ init lands on the same partition
  across all 20 seeds.
* The single unstable cell (c_shape n=2) is large (the §1 outlier). So "TGC
  deterministic vs kmeans init-unstable" is **true but concentrated** — reported
  as a fact, not attributed to any other cause.

---

## Step 7 — SHIPPED vs CLEAN delta

The joint (shipped↔clean) delta table lives in the **shipped read-out §7**
(restricted to the n ∈ {2,4} overlap; CIs not comparable, N = 100 vs 20).
Summary of that analysis: the two headline rankings — **TGC < classic on energy**
and **TGC ≈ kmeans** — hold in **both** modes. Adding obstacles shifts contrast
*levels* and adds noise. The n=2 "sign flips" are not genuine reversals: some are
clean exact-0 cells (the three rectangles, where the obstacle-free tgc/classic
partitions coincide), and the rest have one side within noise of 0 (e.g. disk n=2
is unresolved in shipped, +10 kJ ± 67 kJ, but cleanly −182 kJ in clean). The one
structural feature unique to clean is the c_shape n=2 kmeans-init outlier (§1,
§6). Numbers-only conclusion: **obstacles change the level, not the ranking.**

---

## Methodology notes (CLEAN)

1. **CI definition:** 95 % normal-approx half-width `Z·√(var/n)`, Z = 1.959963985,
   ddof = 1 (`metrics/convergence.ci_half_width`). In clean, deterministic-variant
   CIs are exactly 0 by construction (obstacle-free, paired seeds).
2. **No per-replication raw data** in the folder; contrasts absent from
   `contrasts.csv` (e.g. `tgc_naive − classic_naive`) are unpaired point-diffs of
   means only (flagged `*`).
3. **N = 20** here (vs shipped 100) — clean btwn-cell CIs are correspondingly
   wider; do not compare CI widths across folders.
4. **Winner's-curse ban honoured** — peers reported separately; the banned
   best-of-baseline H1/H2 statistics from `summary.md`/`run.json` are recomputed
   free of that bias and the divergence is flagged (§5).
5. **weighted_voronoi omitted** from the peer analysis (byte-identical to
   tgc_basic for this homogeneous λ=0 fleet — G-A exact).
6. **Regime tags are exact** in clean (obstacle-free matches A2 analytics); no
   lower-bound caveat (contrast shipped §4).
