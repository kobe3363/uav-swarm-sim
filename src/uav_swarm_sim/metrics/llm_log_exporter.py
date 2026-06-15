"""LLM-ready event-log exporter (Phase 3, Consumer 2: the Phase-4 judge).

Emits JSON Lines: a one-line ``run_header``, one line per mission discontinuity
(the TelemetryLog events), then a one-line ``run_summary``. The result is a
dense, semantic, causal trace small enough to fit an LLM context window yet
sufficient to diagnose the mission ("why did it fail?"). Each row is
self-contained, and sparse event-specific fields (``obstacle_id``, ``outcome``,
...) appear only where they apply -- exactly what JSONL handles cleanly and a
wide, mostly-empty CSV does not.

Read the file top-to-bottom: the header gives the setup once, the summary gives
the verdict, and the events in between explain how the verdict was reached.
"""
from __future__ import annotations

import json

_COMPACT = (",", ":")  # no spaces -> fewer tokens


def build_jsonl(telemetry) -> str:
    """Serialize ``telemetry`` to a JSONL string: header, events, summary."""
    lines = [json.dumps({"kind": "run_header", **telemetry.header}, separators=_COMPACT)]
    for ev in telemetry.events():
        lines.append(json.dumps(ev.to_record(), separators=_COMPACT))
    lines.append(json.dumps({"kind": "run_summary", **telemetry.summary}, separators=_COMPACT))
    return "\n".join(lines) + "\n"


def write_jsonl(telemetry, path) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_jsonl(telemetry))
