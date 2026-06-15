"""B6.3 standalone launch-suitability heat map (staging-periphery edition).

Renders, over the flyable STAGING ring that surrounds the survey area, how good
each candidate launch point is in terms of the EXACT fleet fatigue (battery-swap
count) the mission would incur if the fleet launched from there -- using the very
same canonical workload math the ``launch_site_optimizer`` scores with, so the
picture and the optimizer's choice are consistent by construction.

Staging constraint
------------------
A Ground Control Station cannot sit inside the survey polygon (swamp / field /
hazard area). Valid launch points lie in the staging ring just OUTSIDE
``env.area`` (where, since obstacles live only inside ``area``, the ground is
clear). The heat map therefore forms a ring AROUND an uncoloured target polygon:
no suitability is rendered inside ``env.area``.

For each grid cell (cell size = the platform's effective swath) the script:
  1. masks the cell if its centroid is inside ``env.area`` (target polygon) or
     outside the staging ring;
  2. otherwise runs the energy feasibility gate (reach the furthest navigable
     point and return on one usable battery, incl. vertical takeoff/landing);
  3. if feasible, computes the EXACT fleet swaps for N = fleet.n_drones via
     ``required_sorties`` + ``fleet_swaps`` (site-specific transit overhead);
  4. colours the cell by swap zone.

Zones (N = fleet.n_drones):
  A green   : swaps == 0
  B yellow  : 0 < swaps < N/2
  C orange  : N/2 <= swaps <= N
  D red     : N  < swaps <= 2N
  masked    : swaps > 2N  OR infeasible  OR inside area  OR outside the staging ring

Grid extent comes from the staging ring's bounds (``area.buffer(standoff)``), so
the whole peripheral band is visible -- NOT ``free_space.bounds`` (which is a
subset of ``area`` and would hide the periphery entirely).

Overlay: wireframe of the survey-area boundary, the free-space boundary, every
obstacle, plus a large star at the base pose the (updated) optimizer selects on
the same seeded streams the simulation uses.

Standalone on purpose: matplotlib lives here, never in the core engine. Run from
the repo root, e.g.::

    python -m uav_swarm_sim.experiments.plot_launch_suitability --out launch_suitability_map.png
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from shapely.geometry import Point

from ..infrastructure.config import load_config
from ..infrastructure.rng import STREAM_LAUNCH_SAMPLING, STREAM_OBSTACLES, RngFactory
from ..physical_model.aero_correction import AeroCorrection
from ..physical_model.drone_specs import build_spec
from ..physical_model.energy_model import EnergyModel
from ..physical_model.motion_model import make_motion_model
from ..planning.environment_map import EnvironmentMap
from ..planning.geojson_parser import load_area
from ..planning.gvg_builder import build_gvg
from ..planning.launch_site_optimizer import (
    InfeasibleMissionError,
    _RESERVE_FRAC,
    _STAGING_STANDOFF_M,
    _furthest_free_vertex_dist,
    _staging_region,
    fleet_swaps,
    furthest_point_feasible,
    optimize,
    required_sorties,
)
from ..planning.obstacle_generator import generate as generate_obstacles
from ..planning.tgc import build_tgc

# zone codes -> colours (1..4); 0/NaN == masked (uncoloured)
_ZONE_COLORS = ["#2ca02c", "#f4d03f", "#e67e22", "#c0392b"]  # A green, B yellow, C orange, D red


# --------------------------------------------------------------------------- #
# planning-layer setup (identical to the fleet-sizing runner's layer-0 build)  #
# --------------------------------------------------------------------------- #
def _build_planning_layer(cfg):
    """Reproduce the engine's layer-0 world + real launch site, deterministically.
    Returns (env, spec, em, base_pose)."""
    spec = build_spec(cfg)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    aero = AeroCorrection(cfg.aero, spec.platform)

    rngf = RngFactory(cfg.sim.master_seed)
    obs_rng = rngf.stream(STREAM_OBSTACLES, 0)
    launch_rng = rngf.stream(STREAM_LAUNCH_SAMPLING, 0)

    area = load_area(cfg.env.geojson_path)
    obstacles = generate_obstacles(area, cfg.env, obs_rng)
    env = EnvironmentMap(area, obstacles, cfg.env.clearance_buffer_m)
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    tgc = build_tgc(env, gvg)

    # the energy-aware optimizer raises InfeasibleMissionError if no site works
    base_pose, _scores = optimize(
        cfg.launch, tgc, env, motion, em, aero, spec,
        cfg.fleet.n_drones, launch_rng, cfg.env.coverage_altitude_m,
    )
    return env, spec, em, base_pose


# --------------------------------------------------------------------------- #
# zoning                                                                      #
# --------------------------------------------------------------------------- #
def _zone(swaps: int, n: int) -> int:
    """Zone code for a swap count: 1=A, 2=B, 3=C, 4=D, 0=masked (>2N)."""
    if swaps == 0:
        return 1
    if swaps < n / 2.0:
        return 2
    if swaps <= n:
        return 3
    if swaps <= 2 * n:
        return 4
    return 0


def _suitability_grid(env, spec, em, n_drones, altitude_m, step_m):
    """Build the zone grid (NaN where masked) over the STAGING ring.

    A cell is colourable only if its centroid is OUTSIDE the survey area and
    INSIDE the staging ring; cells inside ``env.area`` are explicitly masked so
    the target polygon renders hollow. Extent is the staging ring's bounds.
    """
    region = _staging_region(env)            # flyable band outside the survey area
    area = env.area
    minx, miny, maxx, maxy = region.bounds   # ring bbox -> captures the whole periphery
    xs = np.arange(minx + step_m / 2.0, maxx, step_m)
    ys = np.arange(miny + step_m / 2.0, maxy, step_m)
    area_m2 = float(env.free_space.area)
    centroid = env.free_space.centroid
    cx, cy = centroid.x, centroid.y

    zone = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            p = Point(x, y)
            if area.contains(p):
                continue  # inside the survey polygon -> never a staging site (masked)
            if not region.covers(p):
                continue  # beyond the staging band -> masked
            furthest = _furthest_free_vertex_dist(env, (x, y))
            if not furthest_point_feasible(em, spec, furthest, altitude_m):
                continue  # cannot reach-and-return -> masked
            transit = float(np.hypot(x - cx, y - cy))
            try:
                sorties = required_sorties(
                    em, spec, area_m2, transit, altitude_m,
                    spec.swath_width_m, _RESERVE_FRAC,
                )
            except InfeasibleMissionError:
                continue  # no coverage budget from here -> masked
            z = _zone(fleet_swaps(sorties, n_drones), n_drones)
            zone[j, i] = z if z > 0 else np.nan
    extent = (minx, maxx, miny, maxy)
    return zone, extent


# --------------------------------------------------------------------------- #
# rendering                                                                   #
# --------------------------------------------------------------------------- #
def _plot_boundary(ax, geom, **kw):
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, **kw)
        for ring in poly.interiors:
            xs, ys = ring.xy
            ax.plot(xs, ys, **kw)


def _render(env, base_pose, zone, extent, n_drones, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(_ZONE_COLORS)
    cmap.set_bad(color=(0, 0, 0, 0))  # transparent where masked (NaN)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        np.ma.masked_invalid(zone), extent=extent, origin="lower",
        cmap=cmap, norm=norm, alpha=0.75, aspect="equal", interpolation="nearest",
    )

    # wireframe overlays: survey-area outline (the hollow target), free space, obstacles
    _plot_boundary(ax, env.area, color="#000000", linewidth=1.6, zorder=4)
    _plot_boundary(ax, env.free_space, color="#555555", linewidth=1.0, zorder=3)
    for ob in env.obstacles:
        _plot_boundary(ax, ob.polygon, color="#777777", linewidth=0.8, zorder=3)

    # selected base pose (in the staging ring)
    ax.scatter(
        [base_pose.x], [base_pose.y], marker="*", s=620,
        facecolor="#ffffff", edgecolor="#000000", linewidth=1.6, zorder=6,
        label="selected base",
    )

    half = n_drones / 2.0
    legend = [
        Patch(facecolor=_ZONE_COLORS[0], edgecolor="none", label="A  0 swaps"),
        Patch(facecolor=_ZONE_COLORS[1], edgecolor="none", label=f"B  0 < swaps < {half:g}"),
        Patch(facecolor=_ZONE_COLORS[2], edgecolor="none", label=f"C  {half:g} ≤ swaps ≤ {n_drones}"),
        Patch(facecolor=_ZONE_COLORS[3], edgecolor="none", label=f"D  {n_drones} < swaps ≤ {2 * n_drones}"),
        Patch(facecolor="none", edgecolor="#999999", label="masked  (target / >2N / infeasible)"),
    ]
    ax.legend(handles=legend, loc="upper right", framealpha=0.9, fontsize=9, title=f"swap zones (N={n_drones})")

    ax.set_title("Launch-site suitability — staging periphery, exact fleet swaps")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Launch-site suitability map (staging periphery, exact fleet swaps).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="launch_suitability_map.png")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    try:
        env, spec, em, base_pose = _build_planning_layer(cfg)
    except InfeasibleMissionError as exc:
        print("MISSION IMPOSSIBLE")
        print(str(exc))
        return 2

    n_drones = cfg.fleet.n_drones
    altitude_m = cfg.env.coverage_altitude_m
    step_m = spec.swath_width_m

    zone, extent = _suitability_grid(env, spec, em, n_drones, altitude_m, step_m)
    _render(env, base_pose, zone, extent, n_drones, args.out)

    colored = int(np.count_nonzero(~np.isnan(zone)))
    print(f"[launch suitability map saved to {args.out}]")
    print(f"  grid {zone.shape[1]}x{zone.shape[0]} cells @ {step_m:.0f} m  |  "
          f"staging standoff {_STAGING_STANDOFF_M:.0f} m  |  "
          f"{colored} staging/feasible cells  |  base ({base_pose.x:,.1f}, {base_pose.y:,.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
