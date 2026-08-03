# C2 — STUDY-01 spare-sizing re-run under the new RTH (read-out)

**Run:** `spares_c2_newrth/run-2026-08-02-19-13-12/`, 500 replications, demand-mode,
`master_seed=42`, `config=study01_demand_newrth.yaml` (arm-4 "full-map" RTH), 1 km².
**Parallelism:** `--jobs auto` (E2). **Analysis:** read-only, from the run outputs; no repo
code changed. This run **supersedes** the pre-EM-01 STUDY-01 numbers.

---

## Run validity

- **500/500 complete**, indices 1..500, no gaps/dupes, `master_seed=42` on every record.
- **New RTH confirmed** — `config_hash = 2d9f954a4b…964606` matches a fresh
  `load_config("config/study01_demand_newrth.yaml")`; `run.json.command` records
  `--config config/study01_demand_newrth.yaml`; the config resolves to arm-4
  (`energy_map.{enabled,decide,route,zone_demotion}=true`, `battery_zones.critical=0.10`).
- **Honest limitation:** telemetry was OFF, so the demand-mode records do not carry
  per-return `rth_reason`/`n_map_hits` fields. Map-governance is established by config
  identity (hash + command + file content), **not** by per-return attribution telemetry.
  This is a strong-enough validity chain (the frozen config unambiguously configures arm-4),
  but it is config-level, not decision-level.

---

## Headline — spare-count knees

| Target | empirical knee | Wilson-certified knee | status |
|---|---|---|---|
| 0.95 | 5 | **6** | Wilson-certified (500 ≥ 73-rep floor) |
| 0.99 | — | — | **structurally unreachable — see below** |

Recomputed P(success \| B) directly from the 500 records (demand ≤ B): n_le =
[0,0,0,0,16,483,491,492] at B=0..7 — **matches** `results.json.cdf`/`knees` exactly. 500
reps clears both Wilson floors (381 for 99%, 73 for 95%), so **if** a knee existed it would
certify.

---

## The load-bearing finding — 0.99 is not a spare-sizing question

**8/500 replications (1.6%) never succeed at any B** (demand = ∞, all
`MISSION_INCOMPLETE`). This sets a hard ceiling `success_frac ≤ 0.984` for **every** battery
count. Since 0.984 < 0.99, **no finite (or infinite) spare pool reaches the 99% target** —
the barrier is the INCOMPLETE cause, not the number of spares. For the thesis / supervisor
package: under this RTH configuration, "how many spares for 99%?" has no answer; 99% requires
removing the INCOMPLETE cause, which is a different problem from spare-sizing. 95% is a clean,
certified knee of 6.

---

## Demand distribution (492 successful reps)

- min 4, median 5, mean ≈ 4.99, max (finite) 7.
- Highly concentrated: demand=4 → 16, **demand=5 → 467 (dominant)**, demand=6 → 8,
  demand=7 → 1.
- Never-succeed fraction: **8/500 = 1.6%** (all `MISSION_INCOMPLETE`).
- CDF: near-vertical rise to 0.966 at B=5 (most mass), then a flat approach to the ~0.984
  ceiling at B=6–7 — the tail is set not by spares but by the 8 reps that never complete.

The tight distribution (95%+ of mass at exactly B=5) indicates the dynamic energy-map RTH
produces a **predictable, low-variance sortie count** — no long tail between 5 and 7 that a
noisier/more-conservative static net might have produced.

---

## Outcome mix (500 reps)

| Outcome | count | frac |
|---|---|---|
| SUCCESS | 492 | 98.4% |
| PARTIAL | 0 | 0.0% |
| INCOMPLETE | 8 | 1.6% |
| FAILED | 0 | 0.0% |

`success_frac = 0.984`, Wilson 95% CI ≈ [0.9687, 0.9919].

---

## Supersession

These knees/CDF **replace** the pre-EM-01 STUDY-01 numbers (`runs/spares_final_demand`,
commit `30f4209`, which ran the old static 0.40 RTH). Any old-vs-new comparison must label
the old numbers "superseded (pre-EM-01)" and frame as "the new-RTH re-run gives X", not
"X beats the old Y".

---

## Notable / limitations

- **Analytical prior mismatch:** `analytical_prior_spares = 1` (formula
  `E_cover/B_usable − n + margin`) is far below the empirical knee (5–6).
  `formula_validation.verdict = "inconclusive"` (computed only for the 0.99 target, whose
  empirical knee is null). Whether the gap is a formula artifact or an RTH-model artifact is
  an open question — no verdict formed here.
- 1 km² only; strict success predicate (`is MISSION_SUCCESS`); feeds the D1 supervisor
  package spare-sizing section and the thesis STUDY-01 chapter.
- Map-governance validity is config-level (telemetry off) — see Run validity.

---

## Reproduction

```bash
python -m uav_swarm_sim.experiments.run_spare_sizing --demand-mode --reps 500 \
  --config config/study01_demand_newrth.yaml --out runs/spares_c2_newrth --jobs auto
# Resume (takes the FILE path, not the dir):
#   ... --resume runs/spares_c2_newrth/<run_dir>/.../results_partial.jsonl
```
