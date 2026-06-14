"""B6.2 fleet-sizing analytical core (Dorling-style). PURE -- no I/O, no side
effects, never imports or runs the SimulationEngine / Agent / dt loop.

Given the real launch geometry (base pose + free-space extent) and the platform
energy model, it estimates, for each fleet size N, the mission duration, the
number of battery swaps, and the marginal time saved by the N-th drone -- the
diminishing-returns Pareto curve an operator reads to choose N.

It reuses the launch optimizer's energy gate (``furthest_point_feasible`` /
``InfeasibleMissionError``) so "Mission Impossible" means exactly the same thing
here as in the simulation.

Model (first-order, deliberately analytical)
-------------------------------------------
* Coverage length = (free area / effective swath) * turn_factor -- the
  boustrophedon strip length inflated for U-turn connectors (a FLOOR; real
  obstacle detours add more). Coverage energy is charged at the COVERAGE rate,
  transit at the CRUISE rate -- never a single blended constant.
* Per sortie a drone spends takeoff + round-trip transit + landing as overhead
  and the rest of one *usable* battery on coverage. Total coverage work divided
  by that per-sortie coverage budget = total required sorties (battery cycles).
* For N drones the sorties split as evenly as possible; the BOTTLENECK drone
  (most sorties) sets the makespan: B full sorties of flight time plus ground
  swap-queue delay.
* Swap queue: drones LAND to wait, so queueing costs TIME ONLY (zero energy);
  only ``n_bays`` swap at once, so each swap round costs
  ceil(contending / n_bays) * service_time. The contending count per round comes
  from the sortie distribution.
* Duration-via-(/N) is an OPTIMISTIC bound (perfectly divisible coverage, one
  representative transit distance). It is deliberately NOT the feasibility gate
  -- Step 1's furthest-point check is, and the two are kept separate so the
  average can never mask the worst case.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..infrastructure.enums import ManeuverType
from ..physical_model.drone_specs import PlatformSpec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.vertical_segments import landing_profile, takeoff_profile
from ..planning.launch_site_optimizer import InfeasibleMissionError, furthest_point_feasible

TURN_FACTOR_DEFAULT = 1.15
RESERVE_FRAC_DEFAULT = 0.05


@dataclass(frozen=True)
class FleetSizingInputs:
    """Planning-layer geometry the calculator needs (built by the runner from the
    EnvironmentMap and the real base pose)."""
    area_m2: float              # navigable (free-space) area to be covered
    furthest_dist_m: float      # base -> furthest free-space vertex (Step 1 gate)
    transit_dist_m: float       # base -> work-area centroid (representative one-way)
    altitude_m: float = 100.0


@dataclass(frozen=True)
class SortieBudget:
    usable_j: float
    takeoff_j: float
    landing_j: float
    transit_round_j: float
    overhead_j: float                 # takeoff + transit_round + landing
    coverage_budget_j: float          # usable - overhead (coverage energy per sortie)
    t_takeoff_s: float
    t_landing_s: float
    t_transit_round_s: float
    t_coverage_per_sortie_s: float    # coverage_budget_j / P_coverage
    coverage_length_per_sortie_m: float

    @property
    def t_sortie_s(self) -> float:
        """Wall-clock flight time of one full sortie (no swap)."""
        return self.t_takeoff_s + self.t_transit_round_s + self.t_coverage_per_sortie_s + self.t_landing_s


@dataclass(frozen=True)
class FleetSizingRow:
    n: int
    est_duration_s: float
    est_total_swaps: int
    bottleneck_sorties: int
    marginal_time_saved_s: float      # duration(n-1) - duration(n); 0.0 at n_min
    feasible: bool = True             # True once the Step-1 gate passes (global)


@dataclass(frozen=True)
class FleetSizingReport:
    total_coverage_length_m: float
    total_coverage_j: float
    total_sorties: float              # real-valued energy ratio
    total_sorties_int: int            # ceil -> integer battery cycles
    budget: SortieBudget
    rows: list[FleetSizingRow]


# --------------------------------------------------------------------------- #
# Step 2 helpers                                                              #
# --------------------------------------------------------------------------- #
def total_coverage_length(
    area_m2: float, effective_swath_m: float, turn_factor: float = TURN_FACTOR_DEFAULT
) -> float:
    """Boustrophedon strip length to sweep ``area_m2``, inflated by
    ``turn_factor`` for the U-turn connectors. A floor: obstacle detours add
    more."""
    if effective_swath_m <= 0:
        raise ValueError("effective_swath_m must be > 0")
    if area_m2 < 0:
        raise ValueError("area_m2 must be >= 0")
    if turn_factor < 1.0:
        raise ValueError("turn_factor must be >= 1.0")
    return (area_m2 / effective_swath_m) * turn_factor


def sortie_budget(
    em: EnergyModel,
    spec: PlatformSpec,
    transit_dist_m: float,
    altitude_m: float = 100.0,
    reserve_frac: float = RESERVE_FRAC_DEFAULT,
) -> SortieBudget:
    """Per-sortie energy/time split: takeoff + round-trip transit + landing as
    overhead, the remaining usable battery as coverage budget. Transit at the
    CRUISE rate; coverage time = coverage budget / COVERAGE power."""
    if transit_dist_m < 0:
        raise ValueError("transit_dist_m must be >= 0")
    take = takeoff_profile(spec, em, altitude_m)
    land = landing_profile(spec, em, altitude_m)
    transit_round_j = 2.0 * em.distance_energy(transit_dist_m, ManeuverType.CRUISE, spec.v_cruise)
    overhead_j = take.energy_j + land.energy_j + transit_round_j
    usable_j = spec.battery_capacity_j * (1.0 - reserve_frac)
    coverage_budget_j = usable_j - overhead_j
    if coverage_budget_j <= 0.0:
        raise InfeasibleMissionError(
            f"Per-sortie transit + vertical overhead ({overhead_j:.0f} J) meets "
            f"or exceeds one usable battery ({usable_j:.0f} J): a drone cannot "
            "make any coverage progress from this launch site. Move the site "
            "closer, lower the altitude, or add battery capacity."
        )
    p_cov = spec.power_w[ManeuverType.COVERAGE]
    t_cov = coverage_budget_j / p_cov                        # energy / power = time
    cov_len = coverage_budget_j * spec.v_coverage / p_cov    # distance covered per sortie
    t_transit_round = (2.0 * transit_dist_m) / spec.v_cruise
    return SortieBudget(
        usable_j=usable_j,
        takeoff_j=take.energy_j,
        landing_j=land.energy_j,
        transit_round_j=transit_round_j,
        overhead_j=overhead_j,
        coverage_budget_j=coverage_budget_j,
        t_takeoff_s=take.duration_s,
        t_landing_s=land.duration_s,
        t_transit_round_s=t_transit_round,
        t_coverage_per_sortie_s=t_cov,
        coverage_length_per_sortie_m=cov_len,
    )


def total_sorties(total_coverage_j: float, coverage_budget_per_sortie_j: float) -> float:
    """Real-valued battery cycles of coverage work the whole map requires."""
    if coverage_budget_per_sortie_j <= 0:
        raise ValueError("coverage_budget_per_sortie_j must be > 0")
    return total_coverage_j / coverage_budget_per_sortie_j


# --------------------------------------------------------------------------- #
# Step 3: the sweep                                                          #
# --------------------------------------------------------------------------- #
def _bottleneck_and_swap_delay(
    n: int, total_sorties_int: int, n_bays: int, service_time_s: float
) -> tuple[int, float, int]:
    """For ``n`` drones doing ``total_sorties_int`` sorties split as evenly as
    possible, return (bottleneck drone's sortie count, its total ground
    swap-queue delay, fleet-wide swap count).

    Sortie counts are q or q+1 (q = total // n). The bottleneck drone swaps once
    before each sortie after its first. In a given swap round only the drones
    that still have a later sortie contend, and only ``n_bays`` swap at a time,
    so that round costs ceil(contending / n_bays) * service_time.
    """
    q, r = divmod(total_sorties_int, n)
    # rounds before sorties 2..q involve every active drone (all do >= q);
    # the final round before sortie q+1 (only when r > 0) involves the r long-haul drones.
    full_rounds = max(0, q - 1)
    swap_delay = full_rounds * math.ceil(n / n_bays) * service_time_s
    if r == 0:
        bottleneck = q
    else:
        bottleneck = q + 1
        if q >= 1:
            swap_delay += math.ceil(r / n_bays) * service_time_s
    n_active = min(n, total_sorties_int)
    fleet_swaps = total_sorties_int - n_active
    return bottleneck, swap_delay, fleet_swaps


def sweep(
    inputs: FleetSizingInputs,
    em: EnergyModel,
    spec: PlatformSpec,
    effective_swath_m: float,
    service_time_s: float,
    n_bays: int,
    n_min: int = 1,
    n_max: int = 20,
    turn_factor: float = TURN_FACTOR_DEFAULT,
    reserve_frac: float = RESERVE_FRAC_DEFAULT,
) -> FleetSizingReport:
    """Full analysis: Step-1 feasibility gate, Step-2 total workload, Step-3
    per-N sweep with bottleneck duration + swap queue.

    Raises InfeasibleMissionError if the furthest navigable point is unreachable
    on one battery, or if transit overhead leaves no coverage budget.
    """
    if n_min < 1 or n_max < n_min:
        raise ValueError("require 1 <= n_min <= n_max")
    if n_bays < 1:
        raise ValueError("n_bays must be >= 1")

    # Step 1 -- hard survival gate (separate from the optimistic /N duration)
    if not furthest_point_feasible(em, spec, inputs.furthest_dist_m, inputs.altitude_m, reserve_frac):
        raise InfeasibleMissionError(
            "Mission Impossible from this launch site: the furthest navigable "
            f"point ({inputs.furthest_dist_m:.0f} m away) cannot be reached and "
            "returned from on a single battery."
        )

    # Step 2 -- total workload and required sorties
    budget = sortie_budget(em, spec, inputs.transit_dist_m, inputs.altitude_m, reserve_frac)
    cov_len = total_coverage_length(inputs.area_m2, effective_swath_m, turn_factor)
    cov_j = em.distance_energy(cov_len, ManeuverType.COVERAGE, spec.v_coverage)
    sorties = total_sorties(cov_j, budget.coverage_budget_j)
    sorties_int = max(1, math.ceil(sorties))

    # Step 3 -- sweep N
    rows: list[FleetSizingRow] = []
    prev_duration: float | None = None
    for n in range(n_min, n_max + 1):
        bottleneck, swap_delay, fleet_swaps = _bottleneck_and_swap_delay(
            n, sorties_int, n_bays, service_time_s
        )
        duration = bottleneck * budget.t_sortie_s + swap_delay
        marginal = 0.0 if prev_duration is None else (prev_duration - duration)
        rows.append(FleetSizingRow(
            n=n,
            est_duration_s=duration,
            est_total_swaps=fleet_swaps,
            bottleneck_sorties=bottleneck,
            marginal_time_saved_s=marginal,
            feasible=True,
        ))
        prev_duration = duration

    return FleetSizingReport(
        total_coverage_length_m=cov_len,
        total_coverage_j=cov_j,
        total_sorties=sorties,
        total_sorties_int=sorties_int,
        budget=budget,
        rows=rows,
    )
