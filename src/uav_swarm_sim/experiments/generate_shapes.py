"""A3 -- equal-area survey-shape generator for the shape study.

Emits a family of survey polygons that all have the SAME area (default
1 km^2 = 1_000_000 m^2) but very different *shape*, written as GeoJSON in exactly
the format the simulator's area loader (``planning/geojson_parser.load_area``)
expects: a ``FeatureCollection`` whose first ``Feature`` is a metric-metre
``Polygon`` (CCW exterior). Because every coordinate magnitude is well beyond the
lon/lat band, the loader treats them as metres (never geographic) -- matching the
existing ``data/areas/*.geojson`` fixtures.

Why equal area matters
----------------------
The shape study isolates the effect of *shape* on the weighted-decomposition
advantage. Holding area constant means the coverage-strip work (~ area / swath)
is shape-invariant; what changes between shapes is the strip count (turns), the
transit geometry, and -- the H5 variable -- the **solidity** (area / hull_area),
which measures concavity. So each shape is normalised to EXACTLY the target area
(relative area error < 1e-6, asserted) before it is written.

Descriptor table
-----------------
For each shape we print: area (== target), perimeter ``P``, the **isoperimetric
ratio** ``P^2 / (4*pi*A)`` (1.0 for a disk, growing with elongation/concavity),
the **convex-hull area**, the **solidity** ``area / hull_area`` (1.0 = convex;
< 1 for L / star / pinwheel -- the H5 concavity variable), the axis-aligned
bounding box, and the **estimated boustrophedon strip count**. The strip estimate
mirrors the engine's coverage planner (``planning/coverage_path``), which sweeps
across the *minimum-rotated-rectangle* short axis at spacing ``effective_swath``:
``strips ~= short_side(min_rotated_rect) / effective_swath`` where
``effective_swath = swath_width_m * (1 - overlap_frac)`` read from config.

Usage
-----
    python -m uav_swarm_sim.experiments.generate_shapes \
        [--config config/default.yaml] [--target-area-m2 1000000] \
        [--out-dir data/areas/shapes] [--disk-sides 128]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from ..infrastructure.config import load_config


# --------------------------------------------------------------------------- #
# Raw shape builders (arbitrary size; normalised to the target area afterwards) #
# Each returns a convex/concave Shapely Polygon centred roughly on the origin.  #
# --------------------------------------------------------------------------- #
def _square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _rectangle(ratio: float) -> Polygon:
    """Axis-aligned rectangle with long:short side ratio ``ratio`` (>= 1)."""
    w, h = math.sqrt(ratio), 1.0 / math.sqrt(ratio)  # w*h == 1, w/h == ratio
    return Polygon([(0, 0), (w, 0), (w, h), (0, h)])


def _l_shape() -> Polygon:
    """An 'L': a 2x2 square with the top-right 1x1 quadrant removed. Solidity
    = 3/4 = 0.75 (the removed quadrant is exactly the hull deficit)."""
    return Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])


def _disk(n_sides: int) -> Polygon:
    """Regular ``n_sides``-gon approximating a circle (convex; solidity ~= 1,
    isoperimetric ratio -> 1 as n grows)."""
    pts = [
        (math.cos(2 * math.pi * k / n_sides), math.sin(2 * math.pi * k / n_sides))
        for k in range(n_sides)
    ]
    return Polygon(pts)


def _star(points: int = 5, inner_ratio: float = 0.4) -> Polygon:
    """Regular ``points``-pointed star. ``inner_ratio`` = inner/outer radius;
    smaller => spikier => lower solidity. Concave (solidity < 1)."""
    verts = []
    for k in range(2 * points):
        r = 1.0 if k % 2 == 0 else inner_ratio
        ang = math.pi / 2 + math.pi * k / points  # first spike points up
        verts.append((r * math.cos(ang), r * math.sin(ang)))
    return Polygon(verts)


def _pinwheel(blades: int = 4, inner_ratio: float = 0.28) -> Polygon:
    """A rotationally-symmetric pinwheel: ``blades`` swept arms. Concave, with a
    solidity between the star and the L (the arms sweep tangentially, so the hull
    deficit is the gaps between consecutive blades)."""
    verts = []
    step = 2 * math.pi / blades
    for b in range(blades):
        base = b * step
        # tip (outer), then a trailing inner notch that creates the swept 'blade'
        verts.append((math.cos(base), math.sin(base)))                      # outer tip
        verts.append((inner_ratio * math.cos(base + step * 0.5),
                      inner_ratio * math.sin(base + step * 0.5)))           # inner notch
    return Polygon(verts)


def shape_builders(disk_sides: int) -> dict[str, Polygon]:
    """The full equal-area family, keyed by the filename stem used on disk."""
    return {
        "square": _square(),
        "rect_2_1": _rectangle(2.0),
        "rect_4_1": _rectangle(4.0),
        "rect_8_1": _rectangle(8.0),
        "l_shape": _l_shape(),
        "disk": _disk(disk_sides),
        "star_5": _star(5, 0.4),
        "pinwheel": _pinwheel(4, 0.28),
    }


# --------------------------------------------------------------------------- #
# Normalisation + descriptors                                                 #
# --------------------------------------------------------------------------- #
def normalize_to_area(poly: Polygon, target_area_m2: float) -> Polygon:
    """Uniformly scale ``poly`` to EXACTLY ``target_area_m2`` and translate its
    bounding box minimum to the origin (positive metric quadrant, so the loader's
    geographic heuristic never misfires). Winding is normalised to CCW."""
    if poly.area <= 0:
        raise ValueError("cannot normalise a degenerate polygon")
    k = math.sqrt(target_area_m2 / poly.area)      # area scales as k^2
    poly = scale(poly, xfact=k, yfact=k, origin=(0, 0))
    minx, miny, _, _ = poly.bounds
    poly = translate(poly, xoff=-minx, yoff=-miny)
    return orient(poly, sign=1.0)                  # CCW exterior (matches loader)


def _min_rot_rect_sides(poly: Polygon) -> tuple[float, float]:
    """(long_side, short_side) lengths of the minimum rotated rectangle -- the
    frame the boustrophedon planner sweeps in."""
    mrr = poly.minimum_rotated_rectangle
    xy = list(mrr.exterior.coords)[:4]
    e = [math.dist(xy[i], xy[(i + 1) % 4]) for i in range(4)]
    a, b = e[0], e[1]
    return (max(a, b), min(a, b))


def describe(name: str, poly: Polygon, effective_swath_m: float) -> dict:
    """Compute the shape-study descriptor row for one polygon."""
    area = poly.area
    perim = poly.length
    hull_area = poly.convex_hull.area
    minx, miny, maxx, maxy = poly.bounds
    long_side, short_side = _min_rot_rect_sides(poly)
    # engine sweeps strips across the SHORT axis of the min-rotated rectangle
    strips_est = short_side / effective_swath_m if effective_swath_m > 0 else float("nan")
    return {
        "name": name,
        "area_m2": area,
        "perimeter_m": perim,
        "isoperimetric": perim * perim / (4.0 * math.pi * area),
        "hull_area_m2": hull_area,
        "solidity": area / hull_area,
        "bbox_w_m": maxx - minx,
        "bbox_h_m": maxy - miny,
        "mrr_long_m": long_side,
        "mrr_short_m": short_side,
        "strips_est": strips_est,
    }


# --------------------------------------------------------------------------- #
# GeoJSON I/O (loader-compatible)                                             #
# --------------------------------------------------------------------------- #
def to_geojson(name: str, poly: Polygon, descriptor: dict) -> dict:
    """FeatureCollection with one metric-metre Polygon Feature, exactly the shape
    of the existing data/areas fixtures (so ``load_area`` reads it unchanged)."""
    ring = [[float(x), float(y)] for x, y in poly.exterior.coords]
    props = {
        "name": f"shape study: {name} (metric, meters; equal-area)",
        "shape": name,
        # descriptors embedded for provenance (ignored by the loader)
        "area_m2": round(descriptor["area_m2"], 3),
        "solidity": round(descriptor["solidity"], 6),
        "isoperimetric": round(descriptor["isoperimetric"], 6),
    }
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def write_shape(out_dir: Path, name: str, poly: Polygon, descriptor: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.geojson"
    path.write_text(json.dumps(to_geojson(name, poly, descriptor), indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _render_table(rows: list[dict], target_area_m2: float, effective_swath_m: float) -> str:
    head = (
        "| shape | area (m²) | perimeter (m) | isoperim P²/4πA | "
        "hull area (m²) | solidity | bbox w×h (m) | strips≈ |"
    )
    sep = "|:------|----------:|--------------:|----------------:|---------------:|---------:|:------------|--------:|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['area_m2']:,.1f} | {r['perimeter_m']:,.1f} | "
            f"{r['isoperimetric']:.4f} | {r['hull_area_m2']:,.1f} | {r['solidity']:.4f} | "
            f"{r['bbox_w_m']:,.0f}×{r['bbox_h_m']:,.0f} | {r['strips_est']:.1f} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public build helper (used by the runner AND the tests)                       #
# --------------------------------------------------------------------------- #
def build_all(
    target_area_m2: float, effective_swath_m: float, disk_sides: int
) -> list[tuple[str, Polygon, dict]]:
    """Return [(name, normalised polygon, descriptor), ...] for the whole family.
    Asserts every shape hits the target area within 1e-6 relative error."""
    out: list[tuple[str, Polygon, dict]] = []
    for name, raw in shape_builders(disk_sides).items():
        poly = normalize_to_area(raw, target_area_m2)
        rel_err = abs(poly.area - target_area_m2) / target_area_m2
        if rel_err >= 1e-6:
            raise AssertionError(
                f"shape {name!r} area error {rel_err:.2e} >= 1e-6 "
                f"(got {poly.area:.6f}, target {target_area_m2})"
            )
        out.append((name, poly, describe(name, poly, effective_swath_m)))
    return out


# --------------------------------------------------------------------------- #
# entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate equal-area survey shapes as loader-compatible GeoJSON."
    )
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--target-area-m2", type=float, default=1_000_000.0)
    ap.add_argument("--out-dir", default="data/areas/shapes")
    ap.add_argument("--disk-sides", type=int, default=128)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    effective_swath_m = cfg.sensor.swath_width_m * (1.0 - cfg.sensor.overlap_frac)

    rows: list[dict] = []
    out_dir = Path(args.out_dir)
    written: list[Path] = []
    for name, poly, descriptor in build_all(
        args.target_area_m2, effective_swath_m, args.disk_sides
    ):
        written.append(write_shape(out_dir, name, poly, descriptor))
        rows.append(descriptor)

    print("# Equal-area survey shapes (A3)\n")
    print(f"- Target area: {args.target_area_m2:,.0f} m²  "
          f"(= {args.target_area_m2 / 1e6:.3f} km²); all shapes normalised to this "
          "within < 1e-6 relative error.")
    print(f"- Effective swath (config): {effective_swath_m:,.1f} m  "
          f"(swath_width_m {cfg.sensor.swath_width_m:.0f} × (1 − overlap {cfg.sensor.overlap_frac:.2f})).")
    print(f"- Output: {out_dir}/  ({len(written)} files)\n")
    print(_render_table(rows, args.target_area_m2, effective_swath_m))
    print("\n_Solidity = area / convex-hull area (1.0 = convex; the H5 concavity "
          "variable). Isoperimetric ratio P²/(4πA) = 1.0 for a disk, grows with "
          "elongation/concavity. strips≈ = short side of the minimum rotated "
          "rectangle ÷ effective swath (the engine's boustrophedon sweep axis)._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
