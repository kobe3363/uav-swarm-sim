# C1 — Stage 5 four-arm RTH A/B read-out (the thesis result)

**Run:** `runs/c1_rth_ab/rth_ab_2026-08-02-08-55-47_7a99a3`, four arms × 100 paired
replications, `master_seed=42`, `config=study01_demand.yaml`, 1 km², obstacles present.
**Harness:** `run_rth_ab.py` (task C1). **Parallelism:** `--jobs auto` (E2), byte-identical
to serial. **Analysis:** read-only, from the per-replication records; no repo code changed.

The four arms each add exactly one factor on identical seeds, so every comparison is a
**paired contrast between two named arms** (winner's-curse ban: never arm-vs-best-of-others).

| Arm | slug | energy_map flags | terminal_floor |
|---|---|---|---|
| 1 | static40 | all off | 0.20 |
| 2 | route-only | enabled + route | 0.20 |
| 3 | decide-route | enabled + decide + route | 0.20 |
| 4 | full-map | enabled + decide + route + zone_demotion | **0.10** |

Constants pinned identically across all arms (verified from the four `plan.json`):
`stall_skip=true`, `stall_detector=true`, `transit_free_space=true`, `reserve_frac=0.05`,
unbounded pool, `n_drones=5`. `config_hash` is byte-identical across all four arms despite
different flags → the environment is shared per replication; **paired seeds hold**.

---

## Headline — reason-attribution inversion

Transition reasons summed over 100 reps per arm:

| Arm | rth_energy | critical_battery | terminal_battery |
|---|---|---|---|
| 1 static40 | 0 | 746 | 0 |
| 2 route-only | 0 | 767 | 0 |
| 3 decide-route | 0 | 767 | 0 |
| 4 full-map | **496** | **0** | 0 |

Arm 1 returns are governed **100% by the static net** (`critical_battery`); arm 4 returns are
governed **100% by the dynamic map** (`rth_energy`), with `critical_battery=0` and
`terminal_battery=0`. This is the full inversion the thesis predicted.

Arm 2 and arm 3 are **byte-identical to each other** (767 = 767, and identical on every
per-replication field): the static 0.40 net fully pre-empts the map's decision while
`zone_demotion=false`, even though arm 3 actively consults the map (`n_map_hits` mean
321.71/rep vs 0 in arm 2). **`decide` without `zone_demotion` is a no-op at the result
level** — this confirms the contrast frame and means the isolation must be read as:
**arm3 − arm1 = routing**, **arm4 − arm3 = decide+demotion**.

---

## The result sentence (what is actually significant)

At n=100 the map does **not** significantly improve success (95% → 98%, CI on the difference
crosses zero). What it does, significantly, is deliver **equivalent success at materially
lower cost**:

> The dynamic energy-map RTH (arm 4) achieves **equivalent mission success** to the static
> 40% threshold (98% vs 95%, difference not significant), while cutting **total energy by
> 5.5%**, **makespan by 12.9%**, and **battery swaps by ~3 per mission** — all strongly
> significant on paired contrasts.

"Equivalent safety at lower cost" is the honest and stronger claim; "higher success" is not
supported at this sample size.

---

## Outcome breakdown (three rows, never folded)

| Arm | SUCCESS | PARTIAL | INCOMPLETE | FAILED | success_frac | Wilson 95% CI |
|---|---|---|---|---|---|---|
| 1 static40 | 95 | 5 | 0 | 0 | 0.950 | [0.888, 0.978] |
| 2 route-only | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |
| 3 decide-route | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |
| 4 full-map | 98 | 2 | 0 | 0 | 0.980 | [0.930, 0.994] |

**Falsified expectation (honest finding):** the anticipated INCOMPLETE → PARTIAL shift does
**not** appear — `MISSION_INCOMPLETE = 0` in all 400 replications (aggregate: SUCCESS 389,
PARTIAL 11, INCOMPLETE 0, FAILED 0). The shift that occurs is PARTIAL → SUCCESS. The
INCOMPLETE→PARTIAL narrative should be dropped for this sample.

---

## Paired contrasts

Method: paired bootstrap (10,000 resamples, paired by replication index; report-level
seed 20260802 for reproducibility, unrelated to `master_seed`), applied uniformly to
proportions and continuous metrics. "n" for demand = complete-case (both arms strict
SUCCESS); one-sided-success reps are reported separately as asymmetry counts, never hidden
in a mean.

**arm2 − arm1 — ROUTING added**

| Metric | arm1 | arm2 | Δ | 95% CI |
|---|---|---|---|---|
| success_frac | 0.950 | 0.980 | +0.030 | [0.000, +0.070] |
| coverage_frac | 0.99964 | 0.99987 | +0.00022 | [0, +0.00050] |
| total_energy_j | 2,534,971 | 2,573,563 | **+38,592** | [+26,759, +50,592] |
| duration_s | 3047.5 | 3086.1 | +38.6 | [−10.8, +90.1] (ns) |
| demand (n=95) | — | — | +0.12 (med 0) | [−0.05, +0.29] (ns) |

Routing alone improves completion (3 reps flip to SUCCESS, none the other way) **but costs
energy** (+38.6 kJ). Mixed effect, not a pure win.

**arm3 − arm2 — DECIDE added**

Every per-replication field is **byte-identical** between arm 2 and arm 3 (Δ = 0, CI [0,0]
on all metrics), despite arm 3 running the map decision (321.71 hits/rep). `decide` without
`zone_demotion` never overrides the static 0.40 threshold → a result-level no-op. Confirms
the contrast frame more strongly than expected.

**arm4 − arm3 — ZONE_DEMOTION added (the map-governs step)**

| Metric | arm3 | arm4 | Δ | 95% CI |
|---|---|---|---|---|
| success_frac | 0.980 | 0.980 | 0.000 | [−0.040, +0.040] |
| coverage_frac | 0.99987 | 0.99973 | −0.00014 | [−0.00060, +0.00023] (ns) |
| total_energy_j | 2,573,563 | 2,396,136 | **−177,427 (−6.9%)** | [−198,411, −156,518] |
| duration_s | 3086.1 | 2655.0 | **−431.2 (−14.0%)** | [−505.6, −353.6] |
| demand (n=96) | — | — | **−2.78 (med −3)** | [−3.02, −2.56] |

**The entire energy/makespan/demand win comes from `zone_demotion`, not from routing.**
The map governing deeper sorties (return at ~0.195 vs 0.40) means fewer sorties → less
ferrying → less energy and time.

**arm4 − arm1 — FULL effect**

| Metric | arm1 | arm4 | Δ | 95% CI |
|---|---|---|---|---|
| success_frac | 0.950 | 0.980 | +0.030 | [−0.010, +0.080] (**ns — crosses zero**) |
| coverage_frac | 0.99964 | 0.99973 | +0.00008 | [−0.00031, +0.00048] (ns) |
| total_energy_j | 2,534,971 | 2,396,136 | **−138,835 (−5.5%)** | [−160,094, −117,899] |
| duration_s | 3047.5 | 2655.0 | **−392.5 (−12.9%)** | [−463.3, −320.7] |
| demand (n=94) | — | — | **−2.66 (med −3)** | [−2.90, −2.41] |

---

## Secondary metrics

**Sortie depth** (`return_depths`, battery fraction at the return decision, pooled):

| Arm | n | min | median | mean |
|---|---|---|---|---|
| 1 | 746 | 0.3977 | 0.3999 | 0.3999 |
| 2 | 767 | 0.3977 | 0.3999 | 0.3998 |
| 3 | 767 | 0.3977 | 0.3999 | 0.3998 |
| 4 | 496 | 0.1077 | 0.1964 | 0.1947 |

Arms 1–3 return exactly at the 0.40 static net. Arm 4 lets the drone fly to ~0.195 mean;
**the 0.10 floor is never reached** (min 0.1077) → `terminal_battery = 0`. The B2
"willing-map over conservative floor" risk did **not** materialise at this scale — good,
but scoped to this 100-rep, 1 km² sample; not extrapolable.

**Demand (swaps, strict SUCCESS):** arm1 min5/med8/mean7.61/max12; arm2 & arm3
min5/med8/mean7.74/max13; arm4 min4/med5/mean4.97/max6.

**Map/route counters (sum / mean-per-rep):** arm1 0/0/0; arm2 `n_route_fallbacks` 145/1.45;
arm3 `n_map_hits` 32171/321.71, `n_map_fallbacks` 2862/28.62, `n_route_fallbacks` 145/1.45;
arm4 `n_map_hits` 30411/304.11, `n_map_fallbacks` 2513/25.13, `n_route_fallbacks` 67/0.67.
(All map fallbacks are arming-bound, not decide — see B2 finding.) `zone_demotion` also
lowers route fallbacks (1.45 → 0.67).

---

## Honest findings & limitations

1. **No INCOMPLETE→PARTIAL shift** — the expected narrative is falsified in this sample
   (INCOMPLETE = 0 everywhere). The real shift is PARTIAL→SUCCESS.
2. **Routing alone (arm2−arm1) raises energy** (+38.6 kJ) while improving completion — the
   energy win is entirely from `zone_demotion`, not routing.
3. **Success improvement is not significant at n=100** (95%→98%, CI crosses zero). The
   defensible claim is equivalent success at lower energy/time/swaps.
4. **`zone_demotion` regresses 2 previously-successful replications.** Non-monotonic across
   the pipeline on the 6 reps that are ever PARTIAL:

   | rep | arm1 | arm2 | arm3 | arm4 |
   |---|---|---|---|---|
   | 71 | PARTIAL | SUCCESS | SUCCESS | SUCCESS |
   | 76 | SUCCESS | SUCCESS | SUCCESS | **PARTIAL** |
   | 79 | PARTIAL | PARTIAL | PARTIAL | SUCCESS |
   | 88 | PARTIAL | PARTIAL | PARTIAL | SUCCESS |
   | 98 | PARTIAL | SUCCESS | SUCCESS | SUCCESS |
   | 100 | PARTIAL | SUCCESS | SUCCESS | **PARTIAL** |

   Rep 76 & 100: the deeper sorties enabled by `zone_demotion` occasionally over-commit and
   miss coverage (coverage_frac 0.9887 / 0.9838). **This is a real trade-off, not noise:**
   the map saves energy by flying deeper but has less margin, so in ~2/100 reps a deep
   sortie causes a PARTIAL. A reviewer will ask about this — it belongs in the supervisor
   package as an honest cost of the energy win.

**Limitations:** arm 1 is "static-40% net + `stall_skip`", not a pristine literature
baseline (per `plan.json`). 1 km² only — the scale axis (4–16 km²) is unvalidated and is
where the fractional floor and the deeper-sortie margin would be stressed first
(`scale_sweep_v2`, supervisor-gated). This is a paired **contrast**, not a Wilson-certified
proportion — certification is C2/C3. The difference CIs are paired bootstrap, disclosed as a
methodological choice over a closed-form Wilson-difference.

---

## Reproduction

```bash
# Azure, one box, tmux
python -m uav_swarm_sim.experiments.run_rth_ab \
  --config config/study01_demand.yaml --reps 100 \
  --out runs/c1_rth_ab --jobs auto
# Resume after interruption:
#   ... --resume runs/c1_rth_ab/rth_ab_<...>
```

Paired seeds (`master_seed=42`), `--jobs auto` byte-identical to serial (E2 determinism
gate). Raw per-replication records live under each arm's `results.json` /
`results_partial.jsonl` in the run folder.
