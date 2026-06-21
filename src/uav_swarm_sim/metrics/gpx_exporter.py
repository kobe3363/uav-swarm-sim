"""GPX 1.1 track exporter (Phase 3, Consumer 1: GIS).

One ``<trk>`` per drone, sampled at the telemetry breakpoints + periodic position
fixes, in standard GPX so the exact 2.5D flight paths load straight into QGIS /
Google Earth / mission planners. Built with the stdlib ``xml.etree.ElementTree``
-- no gpxpy, no heavy dependency -- which gives correct escaping and
well-formedness for free.

Projection: the simulation plane is metres on a LOCAL ENU tangent plane; GPX is
geographic WGS-84 lat/lon. We attach a configurable false origin (``lat0``,
``lon0``) and convert with the equirectangular (flat-earth) approximation -- exact
enough at survey scale. Altitude ``z -> <ele>`` is exact (the 2.5D layer
altitudes are real metres).
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import xml.etree.ElementTree as ET

_GPX_NS = "http://www.topografix.com/GPX/1/1"
_M_PER_DEG_LAT = 111_320.0   # mean metres per degree of latitude (WGS-84)


def _project(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Local ENU metres (x = East, y = North) -> (lat, lon), equirectangular."""
    lat = lat0 + y / _M_PER_DEG_LAT
    lon = lon0 + x / (_M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon


def build_gpx(
    telemetry,
    drone_id: int | None = None,
    lat0: float = 54.6872,
    lon0: float = 25.2797,
    epoch_iso: str = "2026-01-01T00:00:00Z",
    creator: str = "uav-swarm-sim",
) -> str:
    """Serialize track(s) in ``telemetry`` to a GPX 1.1 XML string.
    If ``drone_id`` is provided, only that specific drone's track is serialized.
    """
    epoch = _dt.datetime.fromisoformat(epoch_iso.replace("Z", "+00:00"))
    gpx = ET.Element("gpx", {"version": "1.1", "creator": creator, "xmlns": _GPX_NS})
    
    for did in telemetry.drone_ids():
        # If a specific drone ID was requested, skip all others
        if drone_id is not None and did != drone_id:
            continue
            
        track = telemetry.gpx_track(did)
        if not track:
            continue
            
        trk = ET.SubElement(gpx, "trk")
        ET.SubElement(trk, "name").text = f"drone_{did}"
        seg = ET.SubElement(trk, "trkseg")
        
        for (t, x, y, z) in track:
            lat, lon = _project(x, y, lat0, lon0)
            pt = ET.SubElement(seg, "trkpt", {"lat": f"{lat:.8f}", "lon": f"{lon:.8f}"})
            ET.SubElement(pt, "ele").text = f"{z:.2f}"
            ts = epoch + _dt.timedelta(seconds=float(t))
            ET.SubElement(pt, "time").text = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            
    ET.indent(gpx)  # pretty-print (Python 3.9+)
    body = ET.tostring(gpx, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def write_gpx(telemetry, path, **kwargs) -> None:
    """Write separate GPX files for each drone in the swarm."""
    base_dir = os.path.dirname(path) or "."
    os.makedirs(base_dir, exist_ok=True)
    
    # Extract filename base to append the drone ID (e.g., "tracks" from "tracks.gpx")
    base_name, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() != '.gpx':
        ext = '.gpx'
        
    # Generate one file per drone
    for did in telemetry.drone_ids():
        drone_path = os.path.join(base_dir, f"{base_name}_drone_{did}{ext}")
        gpx_data = build_gpx(telemetry, drone_id=did, **kwargs)
        
        with open(drone_path, "w", encoding="utf-8") as f:
            f.write(gpx_data)