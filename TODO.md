# TODO.md — canonical outstanding-task list

**Last updated:** 2026-07-31 · **Test baseline:** 438 green · **Branch:** main

This file is the single source of truth for *what is left to do*. It does not repeat
architecture facts (see `CLAUDE.md`) or project history (see the coordinator's memory).

---

## 0. Document hierarchy — read this first

| Layer | Lives in | Contains | Authority |
|---|---|---|---|
| **1. Code** | the repo | what the system actually does | **highest** — always wins |
| **2. CLAUDE.md** | repo root | non-negotiable working rules, verified architecture facts, environment | binding for every agent |
| **3. TODO.md** | repo root (this file) | outstanding work, priority, owner, agent tier, DoD | binding for task selection |
| **4. docs/proposals/** | repo | design documents for in-flight work (e.g. `energy_map_rth.md`) | binding *within* their stage |
| **5. Project instructions** | Claude UI project settings | coordinator role, dispatch policy, language | coordination window only |
| **6. Coordinator memory** | Claude UI | durable state, superseded-data rule, learnings | may lag; never overrides 1–4 |

**Contradiction rule.** If any two layers disagree, the lower number wins. An agent that
finds a contradiction must **STOP and report it** — never silently pick a side, never
"fix" the higher layer to match the lower one. Report format: what contradicts what,
`file:line` for both, and which layer you believe is stale.

Known-stale-by-design: the coordinator memory is regenerated periodically and may
describe a task as open after it was merged. Verify against the repo before acting.

---

## 1. Where the project stands

EM-01 (energy-map dynamic RTH) Stages 1–4 are merged. A per-replication Dijkstra
cost-to-go grid now drives the RTH decision, the return route, and the resume transit.
The obstacle-boxing livelock is resolved; the residual (genuinely unreachable legs) is
accounted for via `MISSION_PARTIAL` + `skipped_legs`. Sample ceiling moved 92% → 94%
(50 reps — a sample statement, not certified).

**The thesis result is Stage 5**: an A/B of static-threshold RTH vs the dynamic energy
map on identical seeds. Everything in §2 exists to unblock it.

---

## 2. Critical path to the thesis result

### A. Technical debt (do first — A1 blocks C1)

| ID | Task | Why | Agent |
|---|---|---|---|
| **A1** | Add `MISSION_PARTIAL` to `run_output._OUTCOMES` (~:317) | Without it PARTIAL gets no row in `outcome_counts` → Stage 5 A/B cannot report it. **Blocks C1.** | Sonnet High, Accept edits, standard thinking |
| **A2** | Dedicated flag-off byte-identity test for Stage 4 (`safety.stall_skip`) | Stage 4 shipped with only a structural argument; every other stage has the test. This is debt, not a detail. | Sonnet High, Accept edits, standard thinking |
| **A3** | Docs sync | `docs/proposals/energy_map_rth.md` §7e still describes the **falsified** plan-time feasibility criterion — replace with the runtime skip-on-stall criterion + the M0 numbers (432/5520 ∞-entry strips execute successfully; 20 m red cell is not a reachability test). Delete `docs/thesis_roadmap.md` and repoint `CLAUDE.md` §Roadmap at this file. Drop the obsolete FIX-01 entry. | Sonnet Medium, Accept edits, standard thinking |

**A1 DoD:** PARTIAL appears in `outcome_counts` for a run that produces one; full pytest
green with exact count reported; flag-off byte-identity unaffected.

### B. Last physics change (needs author sign-off before dispatch)

| ID | Task | Why | Agent |
|---|---|---|---|
| **B1** | §9 battery-zone demotion: when the energy map is ON, demote the static `critical` 0.20 guard to a TERMINAL-only failsafe (~0.15) | The dynamic RTH must govern; the static net should only catch true emergencies. This is the change that makes `rth_energy` the dominant transition reason in the A/B. | **Fable/Opus High, Plan → author GO → Accept edits, extended thinking** |

**Author decision required before dispatch:** the exact terminal value (0.15 proposed).
Thesis-affecting — it shifts every energy number.
**B1 DoD:** flag-off byte-identity; transition-reason traces show `rth_energy` governing
with the map ON; full pytest green.

### C. The thesis result

| ID | Task | Why | Agent |
|---|---|---|---|
| **C1** | **Stage 5 A/B experiment.** Three arms on identical seeds: (1) static-20% baseline, (2) route-only map, (3) full map (decide + route). Flags already permit all three without code changes. Observables: sortie depth, demand median, success ceiling, PARTIAL rate, `rth_energy` vs `critical_battery` transition counts, energy/makespan. | This *is* the thesis contribution, measured. | **Fable/Opus High, Plan → author GO, extended thinking** (design + harness); execution by author on Azure |
| **C2** | STUDY-01 re-run with the new RTH | Every current STUDY-01 number is superseded by the new physics (see §4). | Author runs; read-out by Opus High, read-only |
| **C3** | Shipped S5 re-run with the new RTH | `runs/shape_sweep_shipped` is livelock-contaminated and pre-dates EM-01. | Author runs; read-out by Opus High, read-only |

**Sequencing:** A1 → B1 → C1. C2/C3 after C1 fixes the final configuration — running them
earlier wastes compute, because the physics is still moving.

**Before any long run:** timing probe with small `--reps` and extrapolate (53 h and 30 h
lessons). Check whether the script has `--jobs` and say so *before* handing over a command.

---

## 3. Supervisor track (parallel, not blocking)

| ID | Task | Notes |
|---|---|---|
| **D1** | Fix and send the supervisor package | Requires: (a) the 40% → 20% correction everywhere (the static pre-emption threshold is CRITICAL 0.20, `state_machine.py:97-98`; 40% is only a reporting bin, `battery.py:46-47`); (b) recompute the analytical-prior gap with the 20% floor (effective sortie window ~80%, not ~60% — the "≈8× underestimate" story may soften); (c) add the EM-01 results. Coordinator writes it; no agent. |
| **D2** | Obtain supervisor decisions K1–K4 | Scope of the scale experiment: area tiers, n-grid ceiling, reps/CI target, clean vs shipped proportion. See `docs/proposals/scale_sweep_v2.md`. |
| **D3** | Rewrite `scale_sweep_v2.md` after the author's narrowing | New axis (author's decision): **L-shape only**, growing (a) area at fixed edge proportions and (b) the *count* of static obstacles at *fixed* obstacle size (circles = trees). This separates "number of obstacles" from "fraction of area occupied". Dynamic obstacles out of scope. This replaces the 9-shape design. Do after D2. | Opus High, Plan mode, docs only |

---

## 4. Superseded data — never cite

- The first clean-grid night run and every number derived from it (bugged launch path).
- `runs/shape_sweep_shipped` — livelock-contaminated.
- `runs/run_scale_tiers` — empty/incomplete (53 h Azure run, 0/50 tiers).
- **All STUDY-01 numbers** (`runs/spares_final_demand`, 500 reps) — pre-EM-01 physics.
  Superseded once B1 + C1 land. Re-run (C2) before citing anything in the thesis.

Still valid: `runs/shape_sweep_clean_postfix` (clean primary grid, post-fix, obstacle-free
→ immune to the livelock).

---

## 5. Old debts (small, unblock nothing, but real)

| ID | Task | Agent |
|---|---|---|
| **E1** | NOW-03 read-out docs were never merged — `docs/reports/` is empty although the S5 read-out was produced. Recover from the `docs/now-03-s5-readout` branch or regenerate. | Sonnet Medium, Accept edits |
| **E2** | `--jobs` for `run_spare_sizing` (ENG-09 joblib pattern; serial 500-rep = 30 h 44 min → ~8 h). Requires bitwise serial == parallel verification. | Sonnet High, Plan → Accept |
| **E3** | Visibility-graph caching — `route_transit` costs 347–791 s/rep at build in obstacle-dense layouts. Any future obstacles run pays it. | Opus Medium/High, Plan → GO, byte-identity mandatory |

---

## 6. Engineering backlog (optional; schedule only after §2 is done)

- **Variantas C** — deterministic candidate grid for the launch optimizer. The current
  optimizer is "best of 48 random samples", not a true optimum. Required before any
  positive launch-axis claim in the thesis; otherwise the honest null stands.
- **ENG-01 turn aero penalty** — do it *before* any final re-runs, or not at all: it
  re-baselines energy.
- **ENG-13** — `run_scale_tiers` retrofit into a structured `RunContext`.
- **ADV-03** — redistribution always routes through `WeightedTgcDecomposer` regardless of
  the mission algorithm (ablation-fidelity issue).
- **B7** — PyVRP baseline; strengthens the thesis comparison.
- **Test-suite restructuring** — stage 1 (pytest-xdist + slow markers) is safe anytime;
  stages 2–3 (conftest fixture rework, directory reorg) are high churn on the sacred net.
- **`docs/architecture.md`** — execution order, CLI map, config surface.
- Launch story Phase 2 (per-algo launch co-optimization); footprint-aware launch clearance
  (only matters for large swarms).

Rejected work is listed in `CLAUDE.md` §Rejected — do not re-open it from here.

---

## 7. Agent dispatch quick reference

Full policy lives in the project instructions; this is the tier mapping used by the
tasks above.

| Task shape | Model | Mode | Thinking |
|---|---|---|---|
| Read-only diagnosis, audit, grep | Sonnet High | Plan / read-only | standard |
| Pre-run audit before expensive compute | Opus/Fable High | read-only | extended |
| Mechanical, test-guarded implementation | Sonnet Medium/High | Accept edits | standard |
| Core-touching or thesis-affecting seam | Fable/Opus High | **Plan → author GO → Accept** | extended |
| Open architectural / mathematical design | Fable/Opus High | Plan | extended/maximum |

Never use Auto mode on a core seam. Auto is acceptable only for mechanical/accounting
work *and* only with absolute STOP conditions written into the prompt.

**Every dispatch prompt must state:** model tier + mode; "read CLAUDE.md first"; a
Phase 0 audit/measurement with STOP conditions *before* the plan; acceptance criteria;
deliverable = `git format-patch` + Lithuanian apply instructions + a report to the
coordinator; and "NEVER: push, open PRs, watch CI, invent facts".

**Mandatory gates for every code stage:** (a) Phase 0 measurement with `file:line`
citations before planning; (b) new behavior behind a flag, default OFF; (c) **flag-off
byte-identity test** — the gate that matters most, its absence is debt; (d) full pytest
with the exact green count; (e) a STOP condition for thesis-metric semantics (outcome
classification, success predicates) — an agent never decides those silently.

**Measure first.** Two regressions were avoided by a Phase 0 measurement that showed the
planned work was wrong (the Stage 4 plan-time criterion would have dropped the ceiling
92% → 0%; a pre-run audit stopped a 5–8 h empty Azure run). Every substantial dispatch
carries a Phase 0 with the STOP condition "if the measurement shows this stage is
unnecessary, stop and say so".
