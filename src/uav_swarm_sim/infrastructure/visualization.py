"""All Matplotlib outputs, from one module. Pure functions: take computed data,
emit a PNG, return its path. No business logic.

A fixed state color map is used everywhere so every figure in the thesis uses
identical state colors.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

from shapely.geometry import MultiPolygon, GeometryCollection

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from ..infrastructure.enums import AgentState  # noqa: E402

STATE_COLORS = {
    AgentState.S0_IDLE: "#9e9e9e",
    AgentState.S1_TRANSIT: "#1f77b4",
    AgentState.S2_MISSION: "#2ca02c",
    AgentState.S3_RTH: "#ff7f0e",
    AgentState.S_SWAP: "#9467bd",
    AgentState.S_OBS: "#d62728",
    AgentState.S_FAIL: "#000000",
}
_ZONE_CMAP = plt.get_cmap("tab20")


def _poly_patches(ax, geom, **kw):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.is_empty:
            continue
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, **kw)


def plot_environment(env, gvg, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = env.area.exterior.xy
    ax.plot(xs, ys, color="black", lw=1.5)
    for o in env.obstacles:
        _poly_patches(ax, o.polygon, color="#888888", alpha=0.7)
    if gvg is not None and gvg.number_of_edges() > 0:
        for u, v in gvg.edges():
            ax.plot([u[0], v[0]], [u[1], v[1]], color="#1f77b4", lw=0.6, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title("Environment + GVG")
    return _save(fig, out)


def plot_partition(env, partition, launch, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = env.area.exterior.xy
    ax.plot(xs, ys, color="black", lw=1.0)
    
    for i, (did, zone) in enumerate(sorted(partition.zones.items())):
        color = _ZONE_CMAP(i % 20)
        geom = zone.polygon
        
        # --- FIX: Handle geometry fractured by obstacle subtraction ---
        if getattr(geom, "geom_type", "") in ("MultiPolygon", "GeometryCollection"):
            # Unpack the fractured pieces and plot them individually
            for poly in geom.geoms:
                if poly.geom_type == "Polygon":
                    _poly_patches(ax, poly, color=color, alpha=0.55)
        else:
            # Standard single Polygon handling
            _poly_patches(ax, geom, color=color, alpha=0.55)
        # --------------------------------------------------------------
        
        ax.plot(zone.entry_pose.x, zone.entry_pose.y, "o", color="black", ms=3)
        
    for o in env.obstacles:
        _poly_patches(ax, o.polygon, color="#444444", alpha=0.9)
        
    if launch is not None:
        ax.plot(launch.x, launch.y, "*", color="red", ms=16, label="launch")
        ax.legend(loc="upper right")
        
    ax.set_aspect("equal")
    ax.set_title(f"Partition ({partition.algo.value})")
    return _save(fig, out)


def plot_trajectories(result, env, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = env.area.exterior.xy
    ax.plot(xs, ys, color="black", lw=1.0)
    for o in env.obstacles:
        _poly_patches(ax, o.polygon, color="#888888", alpha=0.7)
    if result.partition is not None:
        for i, (did, zone) in enumerate(sorted(result.partition.zones.items())):
            _poly_patches(ax, zone.polygon, color=_ZONE_CMAP(i % 20), alpha=0.15)
    ax.set_aspect("equal")
    ax.set_title("Zones / trajectories")
    return _save(fig, out)


def plot_state_gantt(history, out: Path) -> Path:
    sojourns = history.sojourns()
    agents = sorted({s.agent_id for s in sojourns})
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(agents) + 1.5))
    row = {a: i for i, a in enumerate(agents)}
    for s in sojourns:
        ax.barh(row[s.agent_id], s.duration, left=s.t_in, height=0.6,
                color=STATE_COLORS.get(s.state, "#cccccc"))
    ax.set_yticks(list(row.values()))
    ax.set_yticklabels([f"drone {a}" for a in agents])
    ax.set_xlabel("time (s)")
    ax.set_title("Per-agent state timeline")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS.values()]
    ax.legend(handles, [s.name for s in STATE_COLORS], ncol=4, fontsize=7, loc="upper right")
    return _save(fig, out)


def plot_battery_traces(history, zones_cfg, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    aids = sorted({s.agent_id for s in history.sojourns()})
    for a in aids:
        trace = history.battery_trace(a)
        if trace:
            ts, fr = zip(*trace)
            ax.plot(ts, fr, lw=1.0, label=f"drone {a}")
    for y, name in [(zones_cfg.high, "HIGH"), (zones_cfg.nominal, "NOMINAL"), (zones_cfg.critical, "CRITICAL")]:
        ax.axhline(y, ls="--", color="#999999", lw=0.7)
        ax.text(0, y, name, fontsize=6, color="#666666", va="bottom")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("battery fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("Battery traces")
    ax.legend(fontsize=7)
    return _save(fig, out)


def plot_pi_bars(pi_embedded: dict, pi_time: dict, out: Path) -> Path:
    states = [s for s in pi_embedded.keys()]
    labels = [s.name for s in states]
    emb = [pi_embedded[s] for s in states]
    tim = [pi_time[s] for s in states]
    x = np.arange(len(states))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, emb, width=0.4, label="embedded (visit freq.)", color="#1f77b4")
    ax.bar(x + 0.2, tim, width=0.4, label="time-weighted", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("stationary probability")
    ax.set_title("Embedded vs time-weighted stationary distribution")
    ax.legend()
    return _save(fig, out)


def plot_mc_convergence(trace: list[tuple[int, float, float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ks = [t[0] for t in trace]
    means = [t[1] for t in trace]
    hws = [t[2] if np.isfinite(t[2]) else np.nan for t in trace]
    ax.plot(ks, means, color="#2ca02c", label="running mean pi_time(S2)")
    ax.fill_between(ks, [m - h for m, h in zip(means, hws)], [m + h for m, h in zip(means, hws)],
                    color="#2ca02c", alpha=0.2, label="95% CI")
    ax.set_xlabel("replication")
    ax.set_ylabel("pi_time(S2)")
    ax.set_title("Monte-Carlo convergence")
    ax.legend()
    return _save(fig, out)


def plot_comparison_box(data: dict[str, list[float]], metric: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(data.keys())
    ax.boxplot([data[k] for k in labels], labels=labels)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 2D positional replay (needs position traces logged by the engine)           #
# --------------------------------------------------------------------------- #
def _state_legend_handles():
    return [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=8, label=s.name)
        for s, c in STATE_COLORS.items()
    ]


def _make_comm_circles(ax, aids, viz):
    """Create one low-zorder comm-range Circle patch per drone (returned as a
    dict keyed by agent id). Returns {} when viz is None or the overlay is off.
    Dashed unfilled outline by default; translucent fill if comm_range_dashed
    is False. Circles start at the origin; callers reposition them."""
    if viz is None or not viz.show_comm_range:
        return {}
    circles = {}
    for a in aids:
        col = STATE_COLORS.get(AgentState.S2_MISSION, "#2ca02c")
        if viz.comm_range_dashed:
            c = Circle((0.0, 0.0), viz.comm_range_m, fill=False, ls="--", lw=0.8,
                       edgecolor=col, alpha=viz.comm_range_alpha, zorder=1)
        else:
            c = Circle((0.0, 0.0), viz.comm_range_m, fill=True, lw=0.0,
                       facecolor=col, alpha=viz.comm_range_alpha, zorder=1)
        ax.add_patch(c)
        circles[a] = c
    return circles


def _comm_legend_handle(viz):
    if viz is None or not viz.show_comm_range:
        return []
    return [plt.Line2D([0], [0], marker="o", markerfacecolor="none", color="#2ca02c",
                       ls="--" if viz.comm_range_dashed else "-", markersize=10,
                       label=f"comm range ({viz.comm_range_m:.0f} m)")]


def _draw_static_background(ax, env, partition):
    xs, ys = env.area.exterior.xy
    ax.plot(xs, ys, color="black", lw=1.0)
    for o in env.obstacles:
        _poly_patches(ax, o.polygon, color="#888888", alpha=0.6)
    if partition is not None:
        for i, (did, zone) in enumerate(sorted(partition.zones.items())):
            _poly_patches(ax, zone.polygon, color=_ZONE_CMAP(i % 20), alpha=0.12)
    ax.set_aspect("equal")


def plot_state_colored_paths(history, env, out: Path, partition=None, viz=None) -> Path:
    """Static PNG: each drone's flown path, every segment colored by the state
    the drone was in at that moment. Dependency-free (no Pillow/ffmpeg) and a
    good still for a thesis figure. Requires the engine to have logged positions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_static_background(ax, env, partition)
    aids = sorted({s.agent_id for s in history.sojourns()})
    for aid in aids:
        tr = history.position_trace(aid)
        for (t0, x0, y0, st0), (t1, x1, y1, st1) in zip(tr, tr[1:]):
            ax.plot([x0, x1], [y0, y1], color=STATE_COLORS.get(st0, "#cccccc"), lw=1.0, alpha=0.85)
    # dynamic-obstacle trajectories (dashed red), if any were recorded
    dyn_frames = history.dynamic_obstacle_frames()
    if dyn_frames:
        paths: dict[int, list[tuple[float, float]]] = {}
        for (_t, _mode, obs) in dyn_frames:
            for (oid, ox, oy) in obs:
                paths.setdefault(oid, []).append((ox, oy))
        for oid, pts in paths.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#e6194b", lw=1.2,
                    ls="--", alpha=0.8, label="dynamic obstacle" if oid == 0 else None)
    circles = _make_comm_circles(ax, aids, viz)
    for aid, c in circles.items():
        tr = history.position_trace(aid)
        if tr:
            c.center = (tr[-1][1], tr[-1][2])   # last known position
    ax.legend(handles=_state_legend_handles() + _comm_legend_handle(viz),
              fontsize=7, loc="upper right", ncol=2)
    ax.set_title("Flight paths colored by mission state")
    return _save(fig, out)


def animate_mission(
    history,
    env,
    out: Path,
    fps: int = 12,
    max_frames: int = 200,
    trail: int = 12,
    partition=None,
    viz=None,
) -> Path:
    """Render a 2D replay GIF: drones move over time, each dot colored by its
    current mission state (transit / mission / RTH / swap / obstacle / fail),
    with a fading trail. ``max_frames`` subsamples long missions so the GIF stays
    small; ``trail`` is how many past samples to draw behind each drone.

    Requires position traces (engine logs them every tick) and Pillow (bundled
    with Matplotlib via PillowWriter -- no external programs needed).

    -------------------------------------------------------------------------
    SWITCHING OUTPUT FORMAT (read this if you want MP4, frames, or HTML):

      * MP4 (smaller, scrubbable, but needs the external `ffmpeg` program on
        PATH -- e.g. `brew install ffmpeg` / `choco install ffmpeg`):
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(out_with_mp4_suffix, writer=writer)
        i.e. replace the two PillowWriter lines below with the two above and
        pass an `.mp4` path. Everything else is identical.

      * Individual PNG frames (no animation deps at all; assemble later in any
        video tool): in `update`, after drawing, call
            fig.savefig(out_dir / f"frame_{fi:04d}.png", dpi=120)
        and skip `anim.save`.

      * Interactive HTML5 (for a webpage / notebook):
            html = anim.to_jshtml(fps=fps); Path(out).with_suffix(".html").write_text(html)

      * Higher quality GIF: raise `dpi` in the savefig call inside _save-style
        logic, or raise `fps`/`max_frames`. Larger files, slower to write.
    -------------------------------------------------------------------------
    """
    import matplotlib.animation as animation

    aids = sorted({s.agent_id for s in history.sojourns()})
    traces = {a: history.position_trace(a) for a in aids}
    aids = [a for a in aids if traces[a]]
    if not aids:
        raise ValueError(
            "no position traces found -- run a mission with the engine that logs "
            "positions (SimulationEngine records them every tick)"
        )

    n = min(len(traces[a]) for a in aids)
    stride = max(1, n // max(1, max_frames))
    frame_idx = list(range(0, n, stride))

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_static_background(ax, env, partition)
    handles = _state_legend_handles()
    dyn_frames = history.dynamic_obstacle_frames()
    if dyn_frames:
        handles.append(plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#e6194b",
                                  markeredgecolor="black", markersize=9, label="dynamic obstacle"))
    handles += _comm_legend_handle(viz)
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    scat = ax.scatter([], [], s=70, zorder=5, edgecolors="black", linewidths=0.5)
    obst = ax.scatter([], [], s=90, marker="X", zorder=6, c="#e6194b", edgecolors="black", linewidths=0.6)
    comm_circles = _make_comm_circles(ax, aids, viz)   # {} when overlay is off
    trails = {a: ax.plot([], [], lw=1.2, alpha=0.5)[0] for a in aids}
    title = ax.set_title("")

    def update(fi):
        idx = frame_idx[fi]
        xs, ys, cols = [], [], []
        for a in aids:
            t, x, y, st = traces[a][idx]
            xs.append(x); ys.append(y); cols.append(STATE_COLORS.get(st, "#cccccc"))
            if a in comm_circles:
                comm_circles[a].center = (x, y)
            lo = max(0, idx - trail * stride)
            seg = traces[a][lo: idx + 1: stride]
            trails[a].set_data([p[1] for p in seg], [p[2] for p in seg])
            trails[a].set_color(STATE_COLORS.get(st, "#cccccc"))
        scat.set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))
        scat.set_color(cols)
        mode_txt = ""
        if dyn_frames and idx < len(dyn_frames):
            _t, mode, obs = dyn_frames[idx]
            if obs:
                obst.set_offsets(np.array([[ox, oy] for (_oid, ox, oy) in obs]))
                # blink: pulse size/alpha while the swarm is actively scanning (alarm)
                active = (mode == "ACTIVE")
                pulse = (fi % 2 == 0)
                obst.set_sizes([(150 if pulse else 60) if active else 90] * len(obs))
                obst.set_alpha((1.0 if pulse else 0.35) if active else 0.85)
            else:
                obst.set_offsets(np.empty((0, 2)))
            mode_txt = f"  |  swarm: {mode}"
        title.set_text(f"t = {traces[aids[0]][idx][0]:.0f} s{mode_txt}")
        return [scat, obst, title, *trails.values(), *comm_circles.values()]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), blit=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)   # <-- swap this line for MP4 (see docstring)
    anim.save(out, writer=writer)              # <-- and this one
    plt.close(fig)
    return out


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
