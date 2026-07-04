# Thesis Development Roadmap — canonical outstanding-task list

> Single source of truth for remaining work. Task IDs are referenced by CLAUDE.md, agent
> prompts, and the chat-side project memory. Update this file when a task completes.
> Tool column: **Code** = Claude Code (local CLI, delivers format-patches, never pushes,
> never watches CI); **Chat** = Claude.ai chat; **User** = run locally by the author.
> Mode column applies to Claude Code: **Plan** = plan mode (propose → approve → edit);
> **Accept** = accept-edits mode (mechanical, test-guarded). Auto mode is never used.

## Status snapshot
- master green at 279 tests (S5 harness merged); STUDY-01 adds +19 (=298) once its patch
  lands. Applied-patch state may differ — run `python -m pytest -q` to confirm.
- S5 validation grid PASSED (4 shapes × n{2,4,6}, regime tags correct).
- Scoped null CONFIRMED: weighted_voronoi − tgc_basic ≡ 0 for the homogeneous fleet.
- S5 headline: TGC vs classic_voronoi vs kmeans + optimal-vs-centroid launch axis.

## NOW — critical path
| ID | Task | Tool/Mode | Depends | Definition of Done |
|---|---|---|---|---|
| NOW-02 | Run the FULL clean grid (`run_shape_sweep --mode clean --budget full`, ~10.6 h overnight) + later the shipped robustness subset (`--mode shipped --budget full`, ~21 h) | **User** (terminal); Chat for log triage if cells crash | S5 merged | `runs/shape_sweep_clean/shape_sweep.csv` + `contrasts.csv` + `summary.md` complete; 0 crashed cells (crashes diagnosed, never skipped) |
| NOW-03 | S5 results read-out & honest hypothesis analysis (TGC-vs-naive verdict, launch-axis magnitude, weighted-null confirmation, H1/H3 regime-conditional read, correlations vs solidity AND isoperimetric, go/no-go for STUDY-03) | **Chat** | NOW-02 | Written analysis merged (README/report); no force-confirmation |
| NOW-04 | Thesis framing decision: battery-weighting scope. Draft two candidate framings: (1) integrated system (optimal launch + GVG→TGC + SMDP-analyzed dynamic RTH); (2) battery-weighting scoped to diverged-battery/future-work regime. Author decides with supervisor. | **Chat** + supervisor | informed by NOW-03 | Supervisor-agreed framing; README + thesis text updated |
| FIX-01 | `run_fleet_sizing_analyzer.py` is broken on main: imports `TURN_FACTOR_DEFAULT`, removed by commit 8bed800. Small import fix + test. (Found by the STUDY-01 agent; STUDY-01 worked around it by inlining planning-layer helpers.) | **Code / Accept** | none | Import fixed; analyzer runs; suite green |

## STUDY — thesis studies
| ID | Task | Tool/Mode | Depends | Definition of Done |
|---|---|---|---|---|
| STUDY-01 | Pareto spare-battery sizing (H4 knee). **CODE COMPLETE** (spare_sizing.py + run_spare_sizing.py + 19 tests; Wilson CI; knee at 99%/95%; analytical-prior validation; reps floor: 99% certification needs ≥381 reps → use `--reps 500` for the final run). Remaining: apply the patch, PR, merge; then run the study. | **User** (apply patch, merge) → **User** (run) | S5 merged | Patch merged (298 green); knee plot + validated/refuted verdict from a real run |
| STUDY-02 | Single-run test scripts (no MC): scenario list (shape × algo × launch variant), outputs PNG + GPX + energy breakdown, S_FERRY visible (#17becf), structured runs/ dirs | **Code / Accept** | none | Scripts merged; suite green |
| STUDY-03 | Strip-ordering / decomposition mechanism study. GATED on NOW-03 explicit GO. Non-circular hypothesis frozen BEFORE runs (isolate ordering from partition: same partition, permuted strip orders vs shipped serpentine). | **Chat (Step 1: freeze hypothesis)** → **Code / Plan (Step 2)** | NOW-03 go | Study script + honest analysis merged |

## ADV — advanced / later
| ID | Task | Tool/Mode | Depends | Definition of Done |
|---|---|---|---|---|
| ADV-01 | NFZs outside the survey plot at scale (border deconfliction). Seed exists: Step 2 routing + notch-NFZ test. NFZ schema; interplay with operating_area=convex_hull + margin; multi-NFZ scenarios; analytical==execution with NFZs; flag-off byte-identity. | **Code / Plan** | none hard | NFZ support + scale study merged |
| ADV-02 | Donut/annulus multi-drone decomposition (interior rings). DEFERRED — confirm prioritization first. Coverage strips already work on holed polygons; multi-drone decomposition untested. Degenerate-ring policy = thesis-affecting → propose-before-code. | **Code / Plan** | user go | Donut in A3 family, decomposition clean n=2..6 all 4 algos |
| ADV-03 | Diverged-battery regime study + redistribution ablation-fidelity fix (redistribution should use the mission's own decomposer; today always WeightedTgcDecomposer). CURRENTLY REJECTED — only if NOW-04 re-scopes battery-weighting as a demonstrated contribution. | **Code / Plan** | NOW-04 gate | Fix + failure-driven study merged (only if gated open) |

## ENG — engineering backlog (parallel track)
| ID | Task | Tool/Mode | Depends | Definition of Done |
|---|---|---|---|---|
| ENG-01 | Turn aerodynamics energy penalty (turns cost energy, not only time). Analytical E_cover must mirror it. Deliberate re-baseline, flagged. | **Code / Plan** | before any final re-runs | Merged; fixtures re-baselined deliberately |
| ENG-02 | Explicit Mission Success/Fail hard terminal states; SMDP ergodicity impact analyzed (close-loop policy stated) | **Code / Plan** | none | Merged; suite green |
| ENG-03 | Shared Battery Pool logistics (finite pool, MC-derived size from STUDY-01 knee); zero-energy ground queue preserved | **Code / Plan** | STUDY-01 | Merged; pool is a binding resource |
| ENG-04 | Circular deployment footprints; launch-optimizer respects footprint; flag-off byte-identity | **Code / Accept** | none | Merged; suite green |
| ENG-05 | Full CSV/GPX telemetry traces (t,x,y,z,state,battery per drone; configurable stride; no slowdown when disabled) | **Code / Accept** | none | Merged; exporter documented |
| ENG-06 | LLM-as-judge automated run analysis (rubric; ingest plan.json/results.json/traces; citations to run data) | **Code / Accept** | ENG-05 | Merged pipeline |
| ENG-07 | Altitude-tradeoff study — FINITE-HEIGHT obstacles ONLY (full-height-prism variant rejected: no interior optimum) | **Chat (Step 1)** → **Code / Plan** | finite-ceiling scenarios designed | Study merged; interior optimum demonstrably exists in setup |
| ENG-08 | B4: dacite declarative YAML loader; byte-identical parsed configs on all fixtures | **Code / Accept** | none | Merged; boilerplate removed |
| ENG-09 | B5: parallel MC via joblib. RNG determinism under parallelism: bitwise-identical serial vs parallel on same seeds; pairing preserved; ≥4× on 8 cores. Needed before any full shipped-grid (N≈100) rerun. | **Code / Plan** | none | Merged; speedup measured |
| ENG-10 | B6: Hypothesis property tests (≥10 meaningful properties: geometry invariants, energy non-negativity, parity/state invariants; no flaky tests) | **Code / Accept** | none | Merged; in CI |
| ENG-11 | B7: PyVRP baseline peer (zone-assignment as VRP; same paired-seed protocol and metrics schema as S5) | **Code / Plan** | none | Merged; honest comparison |
| ENG-12 | Mark S_FERRY in PNG figures (color exists: #17becf) | **Code / Accept** | none | Merged; cosmetic only |
| ENG-13 | Retrofit `run_scale_tiers` to structured RunContext/`--base` (same numbers as legacy path via regression fixture) | **Code / Accept** | none | Merged; uniform output |

## Rejected — do NOT schedule
3D Dubins (coupled pitch/yaw); CGAL GVG rewrite; constant-altitude/full-prism altitude
study; heterogeneous initial batteries as an S5 lever; λ>0 as the primary shape-study
driver.
