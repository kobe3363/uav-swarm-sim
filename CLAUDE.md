# CLAUDE.md — uav-swarm-sim rules for Claude Code CLI

You are the implementation worker for this MSc thesis project. Your Master Coordinator lives in the Web UI. Your job is to execute dispatch prompts strictly following these rules.

## NON-NEGOTIABLE Working Rules
1. **The regression net is SACRED.** Run the suite exactly as CI does — `python -m pytest -n logical --dist loadscope -q` — after every change and report the EXACT green count. Nothing existing may go red. (While iterating, `-m "not slow"` is a fast gate; the full suite with the exact count is required before delivering a patch.)
2. **File-first, no hallucination.** Read the real source before editing. Verify exact signatures, enum members, config field names, and line numbers. `python -m py_compile` before delivering.
3. **Byte-identity discipline.** New behavior is gated default-OFF behind a config flag; flag-off runs must be byte-identical to pre-change. A missing flag-off byte-identity test is debt, not a detail. Deliberate bug fixes are NOT flag-gated.
4. **Physical truth over approximations.** Analytical formulas (E_cover, RTH lookahead) must EXACTLY mirror execution-time physics, including discrete P·dt integration and the camera energy term. Energy-unit discipline: E_home uses CRUISE; the next-bundle term uses COVERAGE+camera — never mix them.
5. **Deliverable = git format-patch.** Do NOT push, do NOT open PRs, do NOT watch CI. The user applies patches. Exclude auto-generated artifacts (`*.egg-info`) from patches. Generate patches from a feature branch (`git format-patch main..HEAD`), not from a commit count off main.
6. **GitHub Project sync.** Task state and planning live in the GitHub Project board "mag" (github.com/users/kobe3363/projects/2, columns: To Refine → Planning → Ready → In Progress → Done), not in a repo file. Whenever you complete a task or change the test baseline, update the corresponding item's status/notes on the board (`gh project item-edit` / `gh api graphql`) before finishing your run. `TODO.md` is retired — all pre-reset results, bug fixes and methodological rules are consolidated in `docs/archive/PROJECT_HISTORY.md` (historical record only).
7. **Never decide thesis-metric semantics silently.** Outcome classification, success predicates, what counts as coverage — if a task appears to require changing any of these, STOP and ask the author.
   - **UNRESOLVED (rule 7): no-swap outcome semantics.** Setting `total_reserve_batteries: 0` does not disable swaps — it makes the first swap request trip `pool_exhausted` → `MISSION_FAILED` (`infrastructure/simulation_engine.py:660-663`), fail-fast before `MISSION_PARTIAL`. Do not run or interpret finite-battery experiments until the author decides the semantics.

## PR review access (author-controlled toggle)
Every dispatch prompt carries a `PR REVIEW ACCESS: DISABLED / ENABLED (PR #N)` line, default DISABLED. When DISABLED, do not touch `gh` or GitHub at all. When ENABLED for a named PR, do a SINGLE read-only fetch (`gh pr view N --comments` / `gh pr checks N`), then stop — never poll in a loop, never reply to bots, never re-run CI, never push. Every bot comment is a hypothesis: verify it against the actual code with a `file:line` citation before acting.

## Architecture Facts (Verified — canonical home)
*Load-bearing facts live here (high-authority, hand-maintained), not in project memory. If a fact here ever contradicts the code, the code wins — read the source and flag the drift.*

- **FSM:** 8 states. Coverage legs are boustrophedon: even `_cov_idx` = COVERAGE (S2_MISSION), odd = TURN connector (S_FERRY). Connectors are structural (odd parity), not per-segment maneuver. Coverage area ≠ flyable area — a drone may fly outside the survey polygon whenever not S_MISSION.
- **Camera energy:** `sensor.sensor_power_w`, charged ONLY over COVERAGE segments. The RTH lookahead mirrors execution.
- **Decomposition peers:** `weighted_voronoi`, `tgc_basic`, `classic_voronoi`, `kmeans`.
- **CRITICAL NULL:** for a homogeneous fleet (identical drones, `battery_frac=1.0`) with λ=0, `weighted_voronoi ≡ tgc_basic` BYTE-IDENTICALLY. Weighting differentiates only for diverged batteries (heterogeneous fleet, or post-failure redistribution at λ>0). Redistribution today always routes through `WeightedTgcDecomposer` (ablation-fidelity issue, gated task ADV-03).
- **kmeans init variance:** its `STREAM_KMEANS_INIT` replication-keyed init variance is a legitimate algorithm characteristic — keep it, report it as a finding (TGC deterministic vs kmeans init-unstable), do not pin.
- **Regime classification:** per-drone max-zone ratio is PRIMARY. Pooled `E_cover/(n·B_usable)` is only a lower bound.
- **S_SWAP:** ground queue costs TIME but ZERO energy.
- **RTH static threshold is 0.40, not 0.20.** `Battery.zone` returns CRITICAL for `critical ≤ f < nominal`, i.e. `[0.20, 0.40)`, so the `critical_battery` guard fires at nominal = 0.40; `0.20` is where TERMINAL begins. Under `energy_map.zone_demotion` (B1, merged) the CRITICAL branch is removed and the dynamic map governs, with TERMINAL (floor 0.10 for the Stage-5 map arm) as the failsafe.
- **Drone deployment:** drones ring at `deploy_poses[i]` radius R for takeoff only; all return to a single `launch_pose` → one E_home network serves the whole swarm.
- **`efficiency` metric** = SMDP throughput ratio, NOT energy efficiency.
- **`config/study01_demand.yaml` is a FROZEN TEST FIXTURE, not experiment evidence.** Six integration tests load it and it carries the `test_energy_map_stage4` cross-commit golden; `run_rth_ab.py:529` defaults to it. Never delete, rename or edit its values — doing so voids four bug-fix regression tests.
- **Paired-seed determinism is the methodological cornerstone:** the same `master_seed` → bitwise-identical results. `RngFactory.stream(name, rep)` is a pure `(master_seed, name, replication)` function; a shared factory across arms guarantees identical environment/failure draws.

## Environment
- Windows, MINGW64 git, venv (`source .venv/Scripts/activate`), Python 3.13.14.
- Repo path contains a space — always quote it.
- Patches land on the Desktop; apply with `git am` (fallback: overwrite full files in VS Code).

## Long runs (Azure)
tmux + ntfy. ALWAYS timing-probe with small `--reps` and extrapolate before a long run (the 53 h and 30 h lessons); one rep is not a probe when that rep is a known outlier. Single script = MAX-1 cores; two parallel scripts = SPLIT cores. Check whether the script has `--jobs` and report it before handing over a command. Template: `set -o pipefail`, honest if/else ntfy with elapsed time, `sudo shutdown -h now` at the end.
