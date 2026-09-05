"""Unit tests for drone<->layer assignment and per-layer decomposition merge."""
from __future__ import annotations

import pytest
from shapely.geometry import box

from uav_swarm_sim.infrastructure.core_types import (
    DroneStateView,
    Partition,
    Pose,
    Zone,
)
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo
from uav_swarm_sim.planning.layer_planner import (
    POLICY_AREA_BALANCED,
    POLICY_BATTERY_TIERED,
    POLICY_SINGLE,
    LayerGraphs,
    _assign_area_balanced,
    _assign_battery_tiered,
    _largest_remainder,
    _layer_areas,
    assign_to_layers,
    decompose_layers,
)

_ORIGIN = Pose(0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# fakes (duck-typed collaborators; no mock, no patching)                       #
# --------------------------------------------------------------------------- #
class _FakeFreeSpace:
    def __init__(self, area: float) -> None:
        self.area = area


class _FakeLayer:
    def __init__(self, area: float) -> None:
        self.free_space = _FakeFreeSpace(area)


class FakeLayerStack:
    """Minimal LayerStack surface used by the planner: ``n_layers`` + ``layer(i)``."""

    def __init__(self, areas: list[float]) -> None:
        self._layers = [_FakeLayer(a) for a in areas]
        self.layer_calls: list[int] = []

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def layer(self, idx: int) -> _FakeLayer:
        self.layer_calls.append(idx)
        return self._layers[idx]


class FakeDecomposer:
    """Returns one Zone per drone, with a per-call planning time."""

    def __init__(self, algo: DecompositionAlgo, times: list[float]) -> None:
        self.name = algo
        self._times = list(times)
        self.calls: list[tuple[object, object, list[DroneStateView]]] = []

    def decompose(self, tgc, env, drones, target_area=None) -> Partition:
        self.calls.append((tgc, env, list(drones)))
        t = self._times[len(self.calls) - 1] if len(self.calls) <= len(self._times) else 0.0
        zones = {
            d.id: Zone(
                drone_id=d.id,
                regions=[],
                polygon=box(0.0, 0.0, 1.0, 1.0),
                entry_pose=_ORIGIN,
            )
            for d in drones
        }
        return Partition(self.name, zones, t)


def _drones(spec: list[tuple[int, float]]) -> list[DroneStateView]:
    """Build drones from an explicit ``[(id, battery_frac), ...]`` list."""
    return [DroneStateView(id=i, battery_frac=b, pose=_ORIGIN) for i, b in spec]


def _ids(assignment: dict[int, list[DroneStateView]]) -> dict[int, list[int]]:
    return {k: [d.id for d in v] for k, v in assignment.items()}


@pytest.fixture
def four_drones() -> list[DroneStateView]:
    return _drones([(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)])


# --------------------------------------------------------------------------- #
# _largest_remainder                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n, weights",
    [
        (4, [1.0, 3.0]),
        (7, [2.0, 2.0, 1.0]),
        (1, [5.0, 5.0]),
        (10, [0.5, 0.25, 0.25]),
    ],
)
def test_largest_remainder_conserves_seats(n, weights):
    counts = _largest_remainder(n, weights)
    assert len(counts) == len(weights)
    assert sum(counts) == n
    assert all(c >= 0 for c in counts)


@pytest.mark.parametrize("n", [0, -1, -5])
def test_largest_remainder_non_positive_n_is_all_zero(n):
    assert _largest_remainder(n, [1.0, 2.0, 3.0]) == [0, 0, 0]


def test_largest_remainder_zero_weights_dump_on_bin_zero():
    counts = _largest_remainder(5, [0.0, 0.0, 0.0])
    assert counts == [5, 0, 0]
    assert sum(counts) == 5


def test_largest_remainder_leftover_goes_to_largest_fraction():
    # quotas = [2.25, 0.75] -> floors [2, 0], one seat left; the LARGER
    # fractional part (bin 1) must take it even though it is the higher index.
    assert _largest_remainder(3, [3.0, 1.0]) == [2, 1]


def test_largest_remainder_ties_break_to_lower_index():
    # equal weights -> identical fractional parts; the single leftover seat
    # must land on the lowest index.
    assert _largest_remainder(4, [1.0, 1.0, 1.0]) == [2, 1, 1]


# --------------------------------------------------------------------------- #
# _layer_areas                                                                #
# --------------------------------------------------------------------------- #
def test_layer_areas_in_ascending_layer_order():
    stack = FakeLayerStack([100.0, 250.0, 25.0])
    areas = _layer_areas(stack)
    assert areas == [pytest.approx(100.0), pytest.approx(250.0), pytest.approx(25.0)]
    assert all(isinstance(a, float) for a in areas)
    assert stack.layer_calls == [0, 1, 2]


# --------------------------------------------------------------------------- #
# assign_to_layers -- dispatch                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "policy", [POLICY_SINGLE, POLICY_AREA_BALANCED, POLICY_BATTERY_TIERED, "bogus"]
)
def test_single_layer_short_circuits_every_policy(policy, four_drones):
    # k <= 1 returns before the policy is ever inspected -- even an unknown one.
    out = assign_to_layers(four_drones, FakeLayerStack([500.0]), policy)
    assert _ids(out) == {0: [0, 1, 2, 3]}


def test_single_policy_puts_everyone_on_layer_zero(four_drones):
    out = assign_to_layers(four_drones, FakeLayerStack([100.0, 100.0, 100.0]), POLICY_SINGLE)
    assert _ids(out) == {0: [0, 1, 2, 3]}


def test_unknown_policy_raises_value_error(four_drones):
    with pytest.raises(ValueError) as exc:
        assign_to_layers(four_drones, FakeLayerStack([100.0, 100.0]), "bogus")
    assert "bogus" in str(exc.value)


@pytest.mark.parametrize("policy", [POLICY_AREA_BALANCED, POLICY_BATTERY_TIERED])
def test_every_drone_assigned_exactly_once(policy):
    drones = _drones([(0, 0.9), (1, 0.4), (2, 0.7), (3, 0.4), (4, 0.1)])
    out = assign_to_layers(drones, FakeLayerStack([300.0, 100.0, 100.0]), policy)
    placed = [d.id for layer in out.values() for d in layer]
    assert sorted(placed) == [0, 1, 2, 3, 4]
    assert len(placed) == len(set(placed))


# --------------------------------------------------------------------------- #
# _assign_area_balanced                                                       #
# --------------------------------------------------------------------------- #
def test_area_balanced_counts_are_proportional_to_free_space(four_drones):
    # areas 100 : 300 with 4 drones -> exact quotas [1.0, 3.0], no remainder.
    out = _assign_area_balanced(four_drones, FakeLayerStack([100.0, 300.0]))
    assert [len(out[i]) for i in (0, 1)] == [1, 3]


def test_area_balanced_fills_layers_in_ascending_id_order():
    # ids deliberately out of order on input; the policy must sort by id.
    drones = _drones([(3, 1.0), (1, 1.0), (2, 1.0), (0, 1.0)])
    out = _assign_area_balanced(drones, FakeLayerStack([100.0, 100.0]))
    assert _ids(out) == {0: [0, 1], 1: [2, 3]}


def test_area_balanced_keeps_empty_layers_in_the_mapping():
    drones = _drones([(0, 1.0), (1, 1.0)])
    out = _assign_area_balanced(drones, FakeLayerStack([100.0, 0.0, 0.0]))
    assert set(out) == {0, 1, 2}
    assert _ids(out) == {0: [0, 1], 1: [], 2: []}


# --------------------------------------------------------------------------- #
# _assign_battery_tiered                                                      #
# --------------------------------------------------------------------------- #
def test_battery_tiered_remainder_goes_to_lower_layers():
    drones = _drones([(i, 0.1 * (i + 1)) for i in range(5)])
    out = _assign_battery_tiered(drones, FakeLayerStack([1.0, 1.0]))
    assert [len(out[i]) for i in (0, 1)] == [3, 2]


def test_battery_tiered_sends_highest_battery_to_highest_layer():
    drones = _drones([(0, 0.10), (1, 0.90), (2, 0.50), (3, 0.70)])
    out = _assign_battery_tiered(drones, FakeLayerStack([1.0, 1.0]))
    low = [d.battery_frac for d in out[0]]
    high = [d.battery_frac for d in out[1]]
    assert max(low) < min(high)
    assert _ids(out) == {0: [0, 2], 1: [3, 1]}


def test_battery_tiered_equal_battery_falls_back_to_id():
    drones = _drones([(3, 0.5), (0, 0.5), (2, 0.5), (1, 0.5)])
    out = _assign_battery_tiered(drones, FakeLayerStack([1.0, 1.0]))
    assert _ids(out) == {0: [0, 1], 1: [2, 3]}


# --------------------------------------------------------------------------- #
# decompose_layers                                                            #
# --------------------------------------------------------------------------- #
def test_decompose_layers_merges_and_stamps_layer_index():
    graphs = LayerGraphs({0: ("env0", "tgc0"), 1: ("env1", "tgc1")}, planning_time_s=9.0)
    assignment = {0: _drones([(0, 1.0)]), 1: _drones([(1, 1.0), (2, 1.0)])}
    dec = FakeDecomposer(DecompositionAlgo.WEIGHTED_VORONOI, [0.25, 0.75])

    part = decompose_layers(graphs, assignment, dec)

    assert set(part.zones) == {0, 1, 2}
    assert part.zones[0].layer == 0
    assert part.zones[1].layer == 1
    assert part.zones[2].layer == 1
    # planning time is the SUM of the per-layer partitions, not the LayerGraphs value
    assert part.planning_time_s == pytest.approx(1.0)


def test_decompose_layers_passes_tgc_and_env_in_order():
    graphs = LayerGraphs({0: ("env0", "tgc0")}, planning_time_s=0.0)
    dec = FakeDecomposer(DecompositionAlgo.TGC_BASIC, [0.1])

    decompose_layers(graphs, {0: _drones([(0, 1.0)])}, dec)

    tgc, env, drones = dec.calls[0]
    assert (tgc, env) == ("tgc0", "env0")
    assert [d.id for d in drones] == [0]


def test_decompose_layers_skips_layers_with_no_drones():
    graphs = LayerGraphs(
        {0: ("env0", "tgc0"), 1: ("env1", "tgc1"), 2: ("env2", "tgc2")},
        planning_time_s=0.0,
    )
    assignment = {0: _drones([(0, 1.0)]), 1: []}  # layer 1 empty, layer 2 absent
    dec = FakeDecomposer(DecompositionAlgo.KMEANS, [0.5])

    part = decompose_layers(graphs, assignment, dec)

    assert len(dec.calls) == 1
    assert set(part.zones) == {0}


def test_decompose_layers_borrows_the_decomposer_algo_label():
    graphs = LayerGraphs({0: ("env0", "tgc0")}, planning_time_s=0.0)
    dec = FakeDecomposer(DecompositionAlgo.CLASSIC_VORONOI, [0.0])

    part = decompose_layers(graphs, {0: _drones([(0, 1.0)])}, dec)

    assert part.algo is DecompositionAlgo.CLASSIC_VORONOI


def test_decompose_layers_empty_assignment_yields_empty_partition():
    graphs = LayerGraphs({0: ("env0", "tgc0"), 1: ("env1", "tgc1")}, planning_time_s=4.0)
    dec = FakeDecomposer(DecompositionAlgo.WEIGHTED_VORONOI, [])

    part = decompose_layers(graphs, {}, dec)

    assert part.zones == {}
    assert part.planning_time_s == pytest.approx(0.0)
    assert dec.calls == []
