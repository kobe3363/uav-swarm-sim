"""CLI: run the Phase 4 LLM-as-a-judge diagnosis over a telemetry JSONL log.

    # full diagnosis (needs ANTHROPIC_API_KEY in the environment or --api-key):
    python -m uav_swarm_sim.experiments.run_llm_diagnosis \
        --log runs/demo/events.jsonl --out runs/demo/diagnosis.md

    # inspect the deterministic facts + the EXACT prompt, no API call, no cost:
    python -m uav_swarm_sim.experiments.run_llm_diagnosis \
        --log runs/demo/events.jsonl --dry-run

The diagnosis report is always backed by a deterministic facts block and a
grounding audit, so an LLM hallucination (wrong outcome, non-existent drone) is
flagged rather than silently trusted.
"""
from __future__ import annotations

import argparse
import json
import os

from ..metrics.llm_judge import (
    anthropic_model,
    build_prompt,
    diagnose,
    extract_facts,
    facts_to_dict,
    load_jsonl,
    render_markdown,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LLM-as-a-judge mission diagnosis over a Phase 3 telemetry JSONL log.")
    ap.add_argument("--log", required=True, help="path to the telemetry events JSONL")
    ap.add_argument("--out", default=None, help="path to write the Markdown diagnosis report")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model id")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--max-events", type=int, default=400,
                    help="cap on log events sent to the model (STATE rows downsampled)")
    ap.add_argument("--api-key", default=None, help="overrides the ANTHROPIC_API_KEY env var")
    ap.add_argument("--dry-run", action="store_true",
                    help="print deterministic facts + the prompt; do NOT call the API")
    args = ap.parse_args()

    records = load_jsonl(args.log)

    if args.dry_run:
        facts = extract_facts(records)
        system, user = build_prompt(records, facts, max_events=args.max_events)
        print("=== DETERMINISTIC FACTS ===")
        print(json.dumps(facts_to_dict(facts), indent=2))
        print("\n=== SYSTEM PROMPT ===\n" + system)
        print("\n=== USER PROMPT ===\n" + user)
        return 0

    model = anthropic_model(args.model, max_tokens=args.max_tokens, api_key=args.api_key)
    diag = diagnose(records, model, max_events=args.max_events)
    report = render_markdown(diag)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)

    g = diag.grounding
    print(f"[diagnosis] outcome={diag.outcome} root_cause={diag.root_cause} "
          f"confidence={diag.confidence:.2f} grounded={diag.grounded} "
          f"checks={g.get('passed')}/{g.get('total')}"
          + (f" report -> {args.out}" if args.out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
