"""Single-agent tool loop and the single-call comparison mode."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from .criteria import criteria_text, guidewords_text, load_criteria
from .llm import LLM, extract_json, text_block
from .pid import load_pid
from .prompts import AGENT_KICKOFF, AGENT_SYSTEM, SINGLE_PROMPT
from .schema import FIELDS, ROW_JSON_SCHEMA, validate_row
from .tools import TOOL_SPECS, Toolbox

DEFAULT_MAX_STEPS = 60
MAX_NUDGES = 2
LOG_TEXT_LIMIT = 4000


def load_node(path: str | Path) -> tuple[str, str]:
    """Node file: first line '# <node name>', the rest is the description."""
    text = Path(path).read_text(encoding="utf-8").strip()
    first, _, rest = text.partition("\n")
    if first.startswith("#"):
        return first.lstrip("# ").strip(), rest.strip()
    return Path(path).stem, text


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _clip(s: str, n: int = LOG_TEXT_LIMIT) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


def _blocks_for_log(blocks: list[dict]) -> str:
    return " ".join(_clip(b["text"]) if b["type"] == "text" else f"[image {b.get('media_type')}]" for b in blocks)


def _add_usage(total: dict, usage: dict) -> None:
    for k in ("input_tokens", "output_tokens"):
        total[k] = total.get(k, 0) + int(usage.get(k) or 0)


def run_agent(llm: LLM, pid_path: str | Path, node_path: str | Path, criteria_path: str | Path | None = None,
              max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Run the agent once. Returns {"meta", "rows", "log"}."""
    node, description = load_node(node_path)
    criteria = load_criteria(criteria_path)
    box = Toolbox(pid_image=load_pid(pid_path), node_name=node, node_description=description, criteria=criteria)
    system = AGENT_SYSTEM.format(criteria=criteria_text(criteria))
    messages: list[dict] = [{"role": "user", "content": [text_block(AGENT_KICKOFF.format(node=node))]}]
    steps, usage, nudges, status = [], {"input_tokens": 0, "output_tokens": 0}, 0, "max_steps"
    started, t0 = _now(), time.time()

    for step in range(1, max_steps + 1):
        resp = llm.chat(system, messages, tools=TOOL_SPECS, max_tokens=16000, task="agent")
        _add_usage(usage, resp.usage)
        messages.append(resp.as_message())
        entry = {"step": step, "stop_reason": resp.stop_reason, "text": _clip(resp.text), "usage": resp.usage, "tool_calls": []}
        steps.append(entry)

        if resp.stop_reason == "refusal":
            status = "refusal"
            break
        if not resp.tool_calls:
            if nudges >= MAX_NUDGES:
                status = "stopped_without_finish"
                break
            nudges += 1
            messages.append({"role": "user", "content": [text_block(
                "Continue the study using the tools. Record rows with add_row and call finish when the node is covered.")]})
            continue

        results = []
        for call in resp.tool_calls:
            if resp.stop_reason == "max_tokens":
                content, is_error = [text_block("ERROR: your reply was cut off at the output limit, so this call may be "
                                                "incomplete and was not run. Send it again, one row at a time.")], True
            else:
                content, is_error = box.call(call["name"], call["input"])
            results.append({"id": call["id"], "name": call["name"], "content": content, "is_error": is_error})
            entry["tool_calls"].append({"name": call["name"], "input": call["input"],
                                        "result": _blocks_for_log(content), "is_error": is_error})
        messages.append({"role": "tool", "results": results})
        if box.finished:
            status = "finished"
            break

    meta = {"mode": "agent", "node": node, "pid": str(pid_path), "model": llm.label, "status": status,
            "steps": len(steps), "max_steps": max_steps, "rows": len(box.rows), "rejected_rows": box.rejected,
            "summary": box.summary, "usage": usage, "started": started, "elapsed_s": round(time.time() - t0, 1),
            "criteria": criteria.get("name")}
    return {"meta": meta, "rows": box.rows, "log": {"meta": meta, "system": system, "steps": steps}}


def run_single(llm: LLM, pid_path: str | Path, node_path: str | Path, criteria_path: str | Path | None = None) -> dict:
    """Comparison mode: one prompt, one call, the whole worksheet in one answer (no tools)."""
    node, description = load_node(node_path)
    criteria = load_criteria(criteria_path)
    schema = json.dumps({k: v.get("description", v.get("type")) for k, v in ROW_JSON_SCHEMA["properties"].items()}, indent=1)
    prompt = SINGLE_PROMPT.format(node=node, description=description, guidewords=guidewords_text(),
                                  criteria=criteria_text(criteria), schema=schema)
    started, t0 = _now(), time.time()
    resp = llm.complete(prompt, image=load_pid(pid_path), max_tokens=32000, task="single")
    parsed = extract_json(resp.text)
    items = parsed.get("worksheet") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
    rows, problems = [], []
    for i, item in enumerate(items or [], 1):
        if isinstance(item, dict) and not item.get("node"):
            item = {**item, "node": node}
        row, errors, warnings = validate_row(item, criteria)
        if row:
            rows.append(row)
        if errors or warnings:
            problems.append({"item": i, "errors": errors, "warnings": warnings})
    status = "finished" if rows else ("unparseable" if items is None else "no_valid_rows")
    meta = {"mode": "single", "node": node, "pid": str(pid_path), "model": llm.label, "status": status, "steps": 1,
            "rows": len(rows), "rejected_rows": sum(1 for p in problems if p["errors"]), "usage": resp.usage,
            "started": started, "elapsed_s": round(time.time() - t0, 1), "criteria": criteria.get("name")}
    log = {"meta": meta, "prompt": prompt, "reply": _clip(resp.text, 200000), "validation": problems}
    return {"meta": meta, "rows": rows, "log": log}


def save_run(result: dict, out_dir: str | Path) -> Path:
    """Write worksheet.json, worksheet.csv and log.json into out_dir."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "worksheet.json").write_text(json.dumps({"meta": result["meta"], "rows": result["rows"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "worksheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["no"] + FIELDS)
        w.writeheader()
        for i, r in enumerate(result["rows"], 1):
            w.writerow({"no": i, **{k: "; ".join(v) if isinstance(v, list) else v for k, v in r.items()}})
    (out / "log.json").write_text(json.dumps(result["log"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
