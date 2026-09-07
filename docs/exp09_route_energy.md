# EXP-09: routed return and executed energy

Implementation base: `7eca307971ca123e638e947ce52548f9806b5ad6`.
This is an implementation/acceptance note, not experiment results.

## Approved configuration and API

`rth.execution_coherent` defaults to `false`. When enabled it requires the
existing `coverage.transit_free_space` and `coverage.ferry_free_space` flags.
It does **not** require an energy map. The supported scope is a static coverage
mission with a single layer, MULTIROTOR, and ground-based unbounded obstacles.
Finite-height prisms need vertical collision validation and are not silently
accepted by this separate 1-D takeoff/landing model.

Test configuration: homogeneous M4E, 100 m AGL, `v_cruise=v_coverage=10 m/s`,
platform ascent/descent speeds unchanged, EXP-01 photogrammetry and EXP-04
no-swap enabled. The 10 m/s command is a **test override**; the untouched DJI
preset still has 21/15 m/s and 91.5 m AGL. The test camera draw of 20 W is an
**ASSUMPTION**, not a DJI specification. Production camera draw stays 0 W.

Normal dynamic return uses `rth.energy_map.zone_demotion=true` (valid with
`execution_coherent=true`, even when the map is disabled). `rth.emergency_frac`
defaults to `null`, preserving the `battery_zones.critical` fallback. EXP-09
tests explicitly select 0.05. Reporting bins are unchanged. Reserve is separately
`rth.reserve_frac=0.05`: energy intended to remain after touchdown, charged once
in the budget, never added to the emergency threshold.

| Guard | Coherent behavior |
|---|---|
| Physical loss | Existing airborne depletion/hazard lifecycle; survivors continue |
| Energy admission | Before launch, each productive bundle and each movement step |
| Static CRITICAL | Retained unless existing `zone_demotion` is enabled |
| Emergency | Independent fraction; null retains the legacy TERMINAL-zone guard |
| Completion | RTH followed by paid LAND; existing no-swap outcome classification |

Legacy 5 s / 1%-capacity periodic checks remain in the disabled executor. They
cannot postpone coherent admission: even a 1000 s interval and a 30 s outer
step must check the next productive bundle before moving. Remaining time in an
outer step is carried across leg boundaries, with a new admission check there.

`RthCalculator.return_plan(pose, *, base=..., altitude_m=...)` returns a
`ReturnPlan`: validated `path` including LAND, `energy_j`, `horizontal_s`, and
`altitude_m`. The per-drone base is a method argument; the constructor order in
SimulationEngine is unchanged. Execution, live decisions, and the EXP-06
callback use the same home. `return_energy` retains its old positional arguments
and adds keyword-only `base`. EXP-06 `estimate_fast` is unchanged. In coherent
mode its context accepts the actual routed RTH callable and does not let an
infinite heuristic grid cell override a valid visibility route. Demand and
budget meanings remain unchanged; callers must handle `RouteUnavailable` rather
than turn an unknown cost into a finite balancing weight. The engine logs and
omits an unavailable observational method at t=0, retaining any valid method.
An unreachable proxy anchor must not abort an already-validated fleet flight.

The optional energy map selects geometry only. Geometry fingerprints invalidate
stale maps; an absent/unknown fingerprint uses visibility routing. Every map
attachment, final hop, fallback, strip, connector, and transit is validated
against the same buffered obstacle union. Ground/coverage profiles use the
existing separate 1-D altitude state, not an invented 3-D aerodynamic model.
Camera-off paths may leave the survey inside the configured operating region;
productive paths must remain in the survey. Touching a buffer boundary attains
the configured clearance; 1e-8 m is the coordinate-rounding tolerance.
SafetyMonitor's raw-obstacle predicate and the EXP-10 counters are untouched.

An enabled visibility router now raises `RouteUnavailable` instead of returning
an obstructed chord. This is the approved intentional bug fix, not a second
opt-in flag. Legacy routing-disabled chords and the legacy RTH 1.5 factor remain.

## Failure handling

A rejected preflight assignment remains unassigned S0. `_fleet_settled` can
settle it and coverage below the gate produces normal MISSION_PARTIAL.

Airborne RTH infeasibility emits `rth_infeasible` with time, agent ID, reason,
and deficit joules. Unknown route cost has `deficit_j=null`, never fabricated
joules. Diagnostics are published on the event bus, logged, retained on each
agent, and exposed in `MissionResult.rth_infeasible_events`. No new outcome is
introduced and no global halt is requested.

A geometrically valid but unaffordable return is attempted and physical drain
can falsify the no-depletion hypothesis. A drone without any validated route
holds at its actual position, consumes HOVER, and can deplete; no empty path is
interpreted as touchdown. A reserve shortfall alone triggers immediate return
and diagnostics. The existing engine performs `fleet.kill` for airborne
depletion and settles the surviving fleet before reporting MISSION_FAILED.
Invalid candidate replans are rejected without replacing the decomposition
algorithm or implementing EXP-08 reallocation policy.

## Numerical energy note

The disabled executor samples the maneuver at the step's start and charges a
whole dt. At maneuver boundaries this can bias either direction; at an isolated
constant-power leg's final partial step it overcharges.

Concrete independently calculated example: LAND from **101 m** at 8 m/s and
106.6 W takes 12.625 s. With dt=0.5 s the old rule bills 13 s:

* old: 1385.800 J;
* duration integral: 1345.825 J;
* error, old minus integral: **+39.975 J** (about +2.97%).

At exactly 100 m and dt=0.5 s, 12.5 s aligns with the step, so this particular
tail error is **zero**, and LAND costs 1332.5 J. Separately, the old Agent did
not execute LAND at all, despite reserving it; that omission is distinct from
the integration bias. Coherent execution now pays both takeoff and landing
using the existing profiles, without an extra m*g*h charge or descent credit.

`EnergyModel.path_interval_energy` integrates each actual maneuver overlap;
camera power is added only to productive COVERAGE overlaps. TURN has zero
horizontal speed and its own table power. Prediction/execution tolerance is
`max(1e-6 J, 1e-10 * E)`; no tolerance relaxation was used.

Approved budget example (10 m/s test override): capacity = 99.5 Wh = 358200 J,
reserve = 17910 J. A 100 m strip with the assumed 20 W test payload costs
`(149.2+20)*10 = 1692 J`. A 1000 m return with total yaw pi radians and LAND
from 100 m costs `156.3*100 + 149.2*pi + 106.6*12.5 = 17431.23 J`.
Admission requires `1692 + 17431.23 + 17910 = 37033.23 J`, reserve once.

## Acceptance scope and regression expectations

New tests cover clear and detoured transit/connector/RTH, invalid candidates,
buffer-only intrusion, unreachable/insufficient-energy returns, per-drone home,
stale maps, partial strips and camera power, reserve boundaries, interval/dt
effects, real LAND, preflight settlement, and a faulted drone with a surviving
drone that continues and lands. Complete-flight cached/uncached traces are
compared exactly. These are executable scenarios, not universal guarantees.

EXP-07 Lloyd algorithms are absent at the pinned base. Shared route/energy APIs
are tested independently; no substitute partitioner is introduced. No EXP-08,
EXP-10, EXP-13, power-table, reporting-bin, or protected-preset changes.

## Interaction with EXP-08 re-partitioning (rebase onto EXP-08)

`rth.execution_coherent: true` together with `mission.repartition_enabled:
true` is **rejected at configuration load**. Coherent execution replaces
`Agent.step`, and that method is where EXP-08 announces `ZONE_COMPLETE` and
sets the one-tick re-task hold. Without it a drone that finishes its zone
goes straight to `S3_RTH`, which `eligible_executors` excludes -- so
**zone-completion** re-partitioning would never fire and the work it was
meant to hand over would be left unassigned. The other EXP-08 triggers
(interval, failure, retirement) are untouched by this flag; only the
zone-completion path depends on the bypassed announcement. The combination
is refused rather than run as a silent no-op. This constraint did not exist
at the pinned base -- EXP-08 postdates it -- and lifting it means routing
the announcement and hold through `CoherentFlight`, which is a separate
task, not a rebase fix.

The historical study01 replication 1 has a staging point
`(955.5694218007066, 7.237592922942396)` inside the clearance buffer, 0.52080258 m
from free space, while outside the raw obstacle. Therefore:

* `test_dense_mission_cache_byte_identical`: successful unsafe flights -> equal
  explicit rejection in both cache branches. A separate complete valid flight
  retains exact trace/energy/photo cache identity.
* `test_boxing_map_routing_unblocks_the_livelocked_replication` becomes
  `test_boxing_map_routing_rejects_buffered_resume`: successful buffered resume
  -> rejection of the invalid map attachment and chord fallback.
* `test_zone_demotion_shifts_return_attribution_to_rth_energy`: attribution on
  the invalid dense start -> attribution with the same workload/capacity in a
  common clear world and the approved independent 5% emergency floor. The
  original input is retained as rejection under both demotion settings.
* `test_fix_b1_routed_transit_unblocks_the_livelocked_replication` becomes
  `test_fix_b1_rejects_the_buffered_launch_in_livelocked_replication`: SUCCESS
  -> explicit preflight rejection, because the launch is inside clearance.
* `test_stall_skip_turns_the_boxed_replication_partial` becomes
  `test_stall_skip_cannot_make_an_unreachable_return_safe`: PARTIAL -> FAILED
  under the existing depletion classifier. The blocked drone's actual home
  chord is independently checked as obstructed; its holding battery reaches
  zero. Strip skipping cannot make the rejected map/chord return safe.

The new flag-OFF physical golden was generated from the pinned 7eca307 source,
not from the modified executor. It checks complete serialized physical output
against `tests/fixtures/exp09_legacy_7eca307.json`; an additional comparison
checks omitted versus explicit false/null options. Existing historical disabled
goldens are not repinned. No baseline suite or full CI suite is run locally.

## Local validation commands

Run from `.worktrees/exp-09`, using its `.venv/Scripts/python.exe` (Python
3.13.14, isolated source imports and shared read-only project dependencies).
CI currently uses Ubuntu/Python 3.12 and the complete `tests/` selection; that
full selection is deferred to the author's PR by the explicit task override.
The commands below select affected modules only.

All three final selections completed without failures. The existing
`test_blas_pin_is_load_bearing_or_skip` skipped because this smoke simulation
showed no observable difference between serial and multithreaded BLAS; its
original skip condition and assertions are unchanged. Flag-OFF physical golden,
historical map-disabled goldens, and cache identity checks passed.

```text
python -m pytest tests/unit/execution/test_exp09_coherent.py tests/integration/test_exp09_routes_energy.py tests/integration/test_transit_livelock.py tests/integration/test_stall_skip.py tests/integration/test_exp06_energy_balance.py -n 2 --dist loadscope -q --tb=short

python -m pytest tests/unit/execution/test_exp09_coherent.py tests/unit/planning/test_transit_routing.py tests/unit/planning/test_trajectory_validation.py tests/unit/planning/test_coverage_path_multipolygon.py tests/unit/planning/test_energy_balance_path.py tests/unit/planning/test_energy_map.py tests/unit/planning/test_visibility_router_cache.py tests/unit/execution/test_rth_preemption.py tests/unit/execution/test_rth_map_decide.py tests/unit/execution/test_energy_map_routing.py tests/unit/execution/test_agent_photo_events.py tests/unit/physical_model/test_energy_model.py tests/unit/physical_model/test_motion_model.py tests/unit/physical_model/test_vertical_segments.py tests/unit/infrastructure/test_config.py tests/unit/infrastructure/test_config_energy_balance.py tests/unit/infrastructure/test_djimatrice4e_config.py -n 2 --dist loadscope -q --tb=short

python -m pytest tests/integration/test_connector_routing.py tests/integration/test_energy_map_stage1.py tests/integration/test_energy_map_stage2.py tests/integration/test_energy_map_stage3.py tests/integration/test_energy_map_stage4.py tests/integration/test_energy_map_zone_demotion.py tests/integration/test_transit_cache_identity.py tests/integration/test_exp04_no_swap.py tests/integration/test_exp01_photogrammetry.py tests/integration/test_rth_ab.py tests/unit/execution/test_execution_layer.py tests/unit/execution/test_ferry.py tests/unit/execution/test_obstacle_recovery.py tests/unit/execution/test_exp04_no_swap_state_machine.py tests/unit/planning/test_energy_balance_fast.py tests/unit/planning/test_planning_layer.py -n 2 --dist loadscope -q --tb=short
```

Syntax was checked for every modified or newly added Python file:

```text
python -m py_compile src/uav_swarm_sim/execution/agent.py src/uav_swarm_sim/execution/coherent_flight.py src/uav_swarm_sim/execution/rth_calculator.py src/uav_swarm_sim/execution/state_machine.py src/uav_swarm_sim/infrastructure/config.py src/uav_swarm_sim/infrastructure/core_types.py src/uav_swarm_sim/infrastructure/enums.py src/uav_swarm_sim/infrastructure/simulation_engine.py src/uav_swarm_sim/physical_model/energy_model.py src/uav_swarm_sim/planning/energy_balance.py src/uav_swarm_sim/planning/energy_map.py src/uav_swarm_sim/planning/visibility_router.py tests/integration/test_energy_map_stage3.py tests/integration/test_energy_map_zone_demotion.py tests/integration/test_exp09_routes_energy.py tests/integration/test_stall_skip.py tests/integration/test_transit_cache_identity.py tests/integration/test_transit_livelock.py tests/unit/execution/test_exp09_coherent.py
git diff --check
```

Protected files were compared to 7eca307, and AST equality was checked for
`estimate_fast`, `_make_decomposer`, `_evaluate_terminal_no_swap`, and
`_fleet_settled`.

The intentional historical expectation failures are named above. Transient new
test-fixture failures during implementation were corrected, without relaxing
the energy tolerance: `test_straight_and_detour_use_buffered_geometry`,
`test_return_prediction_equals_execution_yaw_and_100m_landing`,
`test_reporting_bins_and_emergency_guard_are_independent`,
`test_no_route_and_invalid_fallback_are_rejected`,
`test_stale_map_uses_current_visibility_geometry`,
`test_unreachable_return_hovers_and_remains_falsifiable`, and
`test_rejected_prelaunch_plan_settles_as_normal_partial`. These involved the
Obstacle constructor order, a missing test work leg, and the engine's lazy
build/launch feasibility boundary. The two renamed legacy tests also initially
used the wrong fixture accessors (`deploy_poses` is a list; diagnostics contain
Event objects); the corrected tests assert independent geometry and depletion.
