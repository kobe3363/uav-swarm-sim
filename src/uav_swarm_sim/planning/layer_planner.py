"""Batch 2: drone<->layer assignment + per-layer decomposition orchestration.

The 2.5D contribution is realized as two levels of decomposition:

  * Level 1 (here): partition the FLEET across the coverage layers.
  * Level 2 (reused): on each layer, the existing weighted TGC decomposer runs
    UNCHANGED over that layer's sliced 2D EnvironmentMap, so the central
    contribution (zone area proportional to momentary battery) is realized per
    layer with no edits to the decomposers themselves.

Single-layer-z0 invariant
-------------------------
With one layer -- or the ``single`` policy -- every drone lands on layer 0 and
the orchestration reduces to exactly one ``decomposer.decompose(tgc_0, env_0,
all_drones)`` call, byte-identical to the 2D pipeline. Every assignment policy in
this batch is DETERMINISTIC (no RNG), so assignment perturbs nothing downstream.

Net-energy coupling (deferred to Batch 3)
----------------------------------------
A drone assigned to layer ``z`` will spend climb/descent energy reaching it, so
its battery available for *coverage* is ``B_i - E_climb(z) - E_descent(z)``. That
discount needs the vertical energy term, which Batch 3 adds; until then
assignment uses raw battery and raw layer area. ``assign_to_layers`` is shaped so
the discount can slot in later without changing its callers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..infrastructure.core_types import DroneStateView, Partition, Zone
from .decomposition_base import Decomposer
from .environment_map import LayerStack
from .gvg_builder import build_gvg
from .tgc import TGCGraph, build_tgc

# Policy names -- mirror infrastructure.config._LAYER_POLICIES (validated there).
POLICY_SINGLE = "single"
POLICY_AREA_BALANCED = "area_balanced"
POLICY_BATTERY_TIERED = "battery_tiered"


@dataclass
class LayerGraphs:
    """Per-layer ``(EnvironmentMap, TGCGraph)``, built once and reused for both
    the decomposition and (Batch 4) per-layer redistribution.

    ``planning_time_s`` is the summed TGC build time across layers; for a single
    layer it equals the lone TGC's build time (the 2D value).
    """
    by_layer: dict[int, tuple[object, TGCGraph]]
    planning_time_s: float


def build_layer_graphs(
    layer_stack: LayerStack,
    *,
    gvg_sample_step_m: float,
    gvg_spur_min_m: float,
) -> LayerGraphs:
    """Build a GVG + TGC on each layer's sliced map. Layer order is ascending
    altitude (index 0..k-1), so any stateful per-layer RNG is consumed in a
    fixed, deterministic order.

    (Layers that end up with no assigned drones still get a graph here; for the
    single-layer regression case there are none, so this is exact. Making it lazy
    is a possible later optimization.)
    """
    by_layer: dict[int, tuple[object, TGCGraph]] = {}
    planning_time = 0.0
    for idx in range(layer_stack.n_layers):
        env = layer_stack.layer(idx)
        gvg = build_gvg(env, sample_step_m=gvg_sample_step_m, spur_min_m=gvg_spur_min_m)
        tgc = build_tgc(env, gvg)
        by_layer[idx] = (env, tgc)
        planning_time += tgc.planning_time_s
    return LayerGraphs(by_layer, planning_time)


def assign_to_layers(
    drones: list[DroneStateView],
    layer_stack: LayerStack,
    policy: str,
) -> dict[int, list[DroneStateView]]:
    """Partition drones across layers -> ``{layer_idx: [DroneStateView, ...]}``.

    Deterministic for every policy. With one layer (or ``single``) all drones go
    to layer 0 -- the 2D regression case. Every drone is assigned to exactly one
    layer, so downstream zone ids never collide.
    """
    k = layer_stack.n_layers
    if k <= 1 or policy == POLICY_SINGLE:
        return {0: list(drones)}
    if policy == POLICY_AREA_BALANCED:
        return _assign_area_balanced(drones, layer_stack)
    if policy == POLICY_BATTERY_TIERED:
        return _assign_battery_tiered(drones, layer_stack)
    raise ValueError(f"unknown layer assignment policy: {policy!r}")


def decompose_layers(
    layer_graphs: LayerGraphs,
    assignment: dict[int, list[DroneStateView]],
    decomposer: Decomposer,
) -> Partition:
    """Run the (reused) decomposer on each populated layer over that layer's
    sliced map, then merge into one ``Partition`` whose zones are stamped with
    their layer index.

    Single layer + all drones on layer 0 => exactly one decomposer call whose
    zones are returned with ``layer == 0`` (already their default) => the 2D
    partition unchanged.
    """
    zones: dict[int, Zone] = {}
    plan_time = 0.0
    for idx, (env, tgc) in layer_graphs.by_layer.items():
        layer_drones = assignment.get(idx, [])
        if not layer_drones:
            continue
        part = decomposer.decompose(tgc, env, layer_drones)
        plan_time += part.planning_time_s
        for did, zone in part.zones.items():
            zone.layer = idx
            zones[did] = zone
    # borrow the decomposer's algo label (identical across layers)
    return Partition(decomposer.name, zones, plan_time)


# --------------------------------------------------------------------------- #
# deterministic assignment policies                                           #
# --------------------------------------------------------------------------- #
def _layer_areas(layer_stack: LayerStack) -> list[float]:
    return [float(layer_stack.layer(i).free_space.area) for i in range(layer_stack.n_layers)]


def _largest_remainder(n: int, weights: list[float]) -> list[int]:
    """Apportion ``n`` integer seats across bins proportional to ``weights``.

    Leftover seats go to the largest fractional parts; ties break to the lower
    index. Degenerate inputs (all-zero weights) put everyone on bin 0.
    """
    k = len(weights)
    if n <= 0:
        return [0] * k
    total = sum(weights)
    if total <= 0:
        counts = [0] * k
        counts[0] = n
        return counts
    quotas = [n * w / total for w in weights]
    floors = [int(math.floor(q)) for q in quotas]
    rem = n - sum(floors)
    order = sorted(range(k), key=lambda i: (-(quotas[i] - floors[i]), i))
    for i in range(rem):
        floors[order[i]] += 1
    return floors


def _assign_area_balanced(
    drones: list[DroneStateView], layer_stack: LayerStack
) -> dict[int, list[DroneStateView]]:
    """Drone COUNT per layer proportional to that layer's coverable (free-space)
    area; sparser higher layers carry proportionally more drones. Drones fill
    layers in index order, taken in ascending id order (deterministic)."""
    k = layer_stack.n_layers
    counts = _largest_remainder(len(drones), _layer_areas(layer_stack))
    out: dict[int, list[DroneStateView]] = {i: [] for i in range(k)}
    ordered = sorted(drones, key=lambda d: d.id)
    it = iter(ordered)
    for i in range(k):
        for _ in range(counts[i]):
            out[i].append(next(it))
    return out


def _assign_battery_tiered(
    drones: list[DroneStateView], layer_stack: LayerStack
) -> dict[int, list[DroneStateView]]:
    """Even split across layers (remainder to the lower, cheaper-to-reach
    layers); within that, the highest-battery drones go to the highest layers,
    since reaching a higher layer costs more climb energy. Sort key (battery, id)
    is fully deterministic -- with equal initial battery it falls back to id."""
    k = layer_stack.n_layers
    n = len(drones)
    base, rem = divmod(n, k)
    counts = [base + (1 if i < rem else 0) for i in range(k)]
    ordered = sorted(drones, key=lambda d: (d.battery_frac, d.id))  # ascending battery
    out: dict[int, list[DroneStateView]] = {i: [] for i in range(k)}
    pos = 0
    for i in range(k):  # fill lowest layer first with the lowest-battery drones
        for _ in range(counts[i]):
            out[i].append(ordered[pos])
            pos += 1
    return out
