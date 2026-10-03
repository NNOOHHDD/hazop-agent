"""Command line entry point: python -m hazop_eval --reference ref.csv --runs results/agent/run_* --out results/eval"""
from __future__ import annotations

import argparse
from pathlib import Path

from hazop_agent.agent import load_node
from hazop_agent.llm import get_llm

from .evaluate import evaluate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hazop_eval", description=__doc__)
    ap.add_argument("--reference", required=True, help="reference worksheet CSV (see examples/mcfc/reference_worksheet.csv)")
    ap.add_argument("--runs", required=True, nargs="+", help="run folders (with worksheet.json) or worksheet.json files")
    ap.add_argument("--out", default="results/eval", help="output folder for metrics.json, matches.json, report.html")
    ap.add_argument("--node", help="node description file, given to the judge as process context")
    ap.add_argument("--matcher", choices=["llm", "embedding"], default="llm",
                    help="primary matcher (default llm; the embedding matcher is always computed for comparison)")
    ap.add_argument("--embedding", choices=["tfidf", "openai"], default="tfidf",
                    help="embedding backend: tfidf (offline, lexical) or openai (text-embedding-3-small)")
    ap.add_argument("--threshold", type=float, help="cosine threshold for the embedding matcher")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge score")
    ap.add_argument("--frequent", type=float, default=0.6,
                    help="share of runs at or above which a reference row counts as 'frequently reproduced' (default 0.6)")
    ap.add_argument("--provider", help="LLM provider for matching and judging (default: $HAZOP_EVAL_PROVIDER, then $HAZOP_LLM_PROVIDER)")
    ap.add_argument("--model", help="model for matching and judging (default: $HAZOP_EVAL_MODEL, then $HAZOP_LLM_MODEL)")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import os

    need_llm = args.matcher == "llm" or not args.no_judge
    llm = None
    if need_llm:
        provider = args.provider or os.getenv("HAZOP_EVAL_PROVIDER") or None
        model = args.model or os.getenv("HAZOP_EVAL_MODEL") or None
        llm = get_llm(provider, model)
    context = ""
    if args.node:
        name, desc = load_node(args.node)
        context = f"{name}: {desc}"
    runs = [p for p in args.runs if Path(p).exists()]
    if not runs:
        raise SystemExit("no run folders found")
    m = evaluate(args.reference, runs, args.out, llm, matcher=args.matcher, embedding=args.embedding,
                 threshold=args.threshold, judge=not args.no_judge, frequent=args.frequent, context=context)
    c = m["reproducibility"]["counts"]
    print(f"recall {m['summary']['recall'].get('mean')}, precision {m['summary']['precision'].get('mean')}; "
          f"reference rows never/occasionally/frequently reproduced: {c['never']}/{c['occasional']}/{c['frequent']}; "
          f"agent-only scenarios: {len(m['ai_only']['clusters'])}")


if __name__ == "__main__":
    main()
