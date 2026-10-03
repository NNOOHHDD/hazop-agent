"""MCP server (stdio) exposing the agent and the evaluator as two tools.

    pip install -r requirements-mcp.txt
    python -m hazop_agent.mcp_server

Built on the `mcp` Python SDK 2.x, where the former FastMCP class is called MCPServer.
The LLM is chosen with the same environment variables as the command line tools.
"""
from __future__ import annotations

import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .agent import DEFAULT_MAX_STEPS, run_agent, run_single, save_run
from .llm import get_llm

server = MCPServer("hazop-agent", instructions="Draft HAZOP worksheets from a P&ID and evaluate them against a reference worksheet.")


def _next_dir(out: Path) -> Path:
    taken = [int(m.group(1)) for p in out.glob("run_*") if (m := re.fullmatch(r"run_(\d+)", p.name))]
    return out / f"run_{max(taken, default=0) + 1:02d}"


@server.tool()
def generate_worksheet(pid_path: str, node_path: str, out_dir: str = "results/agent", mode: str = "agent",
                       max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Run the HAZOP agent once on a P&ID (image or PDF) and a node description file.
    Writes worksheet.json, worksheet.csv and log.json to <out_dir>/run_NN and returns the run summary and rows."""
    llm = get_llm()
    result = run_agent(llm, pid_path, node_path, max_steps=max_steps) if mode == "agent" else run_single(llm, pid_path, node_path)
    run_dir = save_run(result, _next_dir(Path(out_dir)))
    return {"run_dir": str(run_dir), "meta": result["meta"], "rows": result["rows"]}


@server.tool()
def evaluate_worksheet(reference_csv: str, run_dirs: list[str], out_dir: str = "results/eval",
                       matcher: str = "llm", judge: bool = True) -> dict:
    """Compare one or more agent runs with a reference worksheet (CSV).
    Writes metrics.json, matches.json and report.html to out_dir and returns the summary and the triage counts."""
    from hazop_eval.evaluate import evaluate

    llm = get_llm() if (matcher == "llm" or judge) else None
    m = evaluate(reference_csv, run_dirs, out_dir, llm, matcher=matcher, judge=judge, log=lambda *_: None)
    never = [r["ref_id"] for r in m["reproducibility"]["reference_rows"] if r["class"] == "never"]
    return {"report": str(Path(out_dir) / "report.html"), "summary": m["summary"],
            "triage_counts": m["reproducibility"]["counts"], "never_reproduced": never,
            "agent_only_scenarios": len(m["ai_only"]["clusters"])}


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    server.run("stdio")


if __name__ == "__main__":
    main()
