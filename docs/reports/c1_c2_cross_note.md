# Note — C1 ↔ C2 cross-experiment: the same residual-failure seeds

Both C1 (four-arm A/B, 100 reps) and C2 (STUDY-01 demand re-run, 500 reps) ran the arm-4
"full-map" RTH under `master_seed=42`, so their low replication indices share the same
worlds. The residual-failure replications line up.

## The overlap

- **C1 arm-4** failed (as `MISSION_PARTIAL`) on replications **76 and 100** — the deeper
  sorties enabled by `zone_demotion` over-commit and miss coverage in those two worlds.
- **C2** failed (as `MISSION_INCOMPLETE`, demand = ∞) on replications **76, 100**, 109,
  151, 162, 191, 313, 396 — 8/500 = 1.6%.

Replications 76 and 100 fail in **both** experiments. Same seed, same world, same
underlying cause: the map flying deeper leaves less margin, and in those worlds the deep
sortie does not complete coverage. The outcome label differs by run mode (PARTIAL in the
A/B harness vs INCOMPLETE in the demand runner), but the physical mechanism is one.

## Why this matters for the thesis

The energy-map RTH's win is real and significant (C1: energy −5.5%, makespan −12.9%, swaps
−3). But it carries a **reproducible, seed-specific cost**: on a small fraction of worlds
(~1.6–2%), the deeper sortie the map permits fails to complete. This is not sampling noise —
it recurs on the *same seeds* across two independent experiments at two sample sizes.

Consequences to carry into D1 (supervisor package):

1. **The 99% spare-sizing target is unreachable** under this RTH (C2): the ~1.6% residual
   INCOMPLETE sets a `success_frac ≤ 0.984` ceiling that no spare count can overcome. 99%
   is an INCOMPLETE-cause problem, not a spare problem.
2. **The trade-off is honest and quantified:** the map trades ~1.6–2% of worlds into
   residual failure for a 5.5%/12.9%/3-swap efficiency gain on the rest. A reviewer will ask
   about this; the C1↔C2 seed overlap is the evidence that it is a systematic mechanism, not
   a fluke.
3. **A concrete follow-up exists** (not for now): inspect what makes seeds 76/100 (and the
   other C2 INCOMPLETE seeds) over-commit — is it a geometry the map mis-estimates, and
   would a small margin adjustment on the map's decide threshold recover them without
   sacrificing the energy win? This is the kind of question the scale axis (`scale_sweep_v2`)
   and a margin sweep could answer; it is NOT a C1/C2 re-run.

All numbers here trace to the C1 and C2 read-out docs and the raw records; no new run.
