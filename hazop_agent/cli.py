"""Command line entry point: python -m hazop_agent --pid ... --node ... [--runs N]"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .agent import DEFAULT_MAX_STEPS, run_agent, run_single, save_run
from .llm import get_llm


def _next_run_dirs(out: Path, n: int) -> list[Path]:
    """run_01, run_02, ... continuing after any runs already in `out` (never overwrites)."""
    taken = [int(m.group(1)) for p in out.glob("run_*") if (m := re.fullmatch(r"run_(\d+)", p.name))]
    start = max(taken, default=0) + 1
    return [out / f"run_{i:02d}" for i in range(start, start + n)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hazop_agent", description=__doc__)
    ap.add_argument("--pid", required=True, help="P&ID image (png/jpg/...) or PDF (first page is used)")
    ap.add_argument("--node", required=True, help="node description file (first line '# <node name>')")
    ap.add_argument("--out", default="results/agent", help="output folder; each run goes to <out>/run_NN")
    ap.add_argument("--runs", type=int, default=1, help="number of independent runs")
    ap.add_argument("--mode", choices=["agent", "single"], default="agent",
                    help="agent = tool-calling loop (default); single = one prompt, one call, for comparison")
    ap.add_argument("--provider", help="anthropic | openai | claude_cli | fake (default: $HAZOP_LLM_PROVIDER or anthropic)")
    ap.add_argument("--model", help="model name (default: $HAZOP_LLM_MODEL or the provider default)")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="agent step limit")
    ap.add_argument("--criteria", help="risk criteria JSON (default: the packaged 5x5 table)")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    llm = get_llm(args.provider, args.model)
    out = Path(args.out)
    for i, run_dir in enumerate(_next_run_dirs(out, args.runs), 1):
        print(f"[{i}/{args.runs}] {args.mode} run with {llm.label} -> {run_dir}", flush=True)
        if args.mode == "agent":
            result = run_agent(llm, args.pid, args.node, args.criteria, max_steps=args.max_steps)
        else:
            result = run_single(llm, args.pid, args.node, args.criteria)
        save_run(result, run_dir)
        m = result["meta"]
        print(f"    status={m['status']} rows={m['rows']} rejected={m['rejected_rows']} steps={m['steps']} "
              f"tokens in/out={m['usage']['input_tokens']}/{m['usage']['output_tokens']}", flush=True)
        if m["status"] not in ("finished",):
            print(f"    warning: run ended with status '{m['status']}'", file=sys.stderr)


if __name__ == "__main__":
    main()
