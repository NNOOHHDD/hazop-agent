"""LLM judge: logic of each row's cause -> consequence chain, scored 1-5."""
from __future__ import annotations

import json

from hazop_agent.llm import LLM, extract_json

from .cache import LLMCache

JUDGE_SYSTEM = "You review HAZOP worksheets for a process safety team. Answer with JSON only."

JUDGE_PROMPT = """Rate how sound the cause -> consequence logic of each HAZOP row is, for the process described.

5 = correct and specific; the consequence follows directly from the cause
4 = correct, with minor gaps or vagueness
3 = plausible but vague, or an intermediate step is missing
2 = doubtful; the link is weak or partly wrong
1 = wrong or self-contradictory
Judge only the cause -> consequence logic, not the safeguards, ratings or recommendations.

## Process context
{context}

## Rows
```json
{rows}
```

Return JSON only: {{"scores": [{{"row_id": "<id>", "score": <1-5>, "reason": "<one sentence>"}}]}} with one entry per row."""

BATCH = 10


def judge_rows(llm: LLM, rows: list[dict], context: str, cache: LLMCache | None = None) -> dict:
    """Return {row_id: {"score": int, "reason": str}} (rows the judge skipped are absent)."""
    out: dict = {}
    for i in range(0, len(rows), BATCH):
        chunk = [{"row_id": r["row_id"], "deviation": r.get("deviation", ""), "causes": r.get("causes", ""),
                  "consequences": r.get("consequences", "")} for r in rows[i:i + BATCH]]
        prompt = JUDGE_PROMPT.format(context=context, rows=json.dumps(chunk, ensure_ascii=False, indent=1))
        key = LLMCache.key(llm.label, "judge", JUDGE_SYSTEM, prompt)
        got = cache.get(key) if cache else None
        if got is None:
            resp = llm.complete(prompt, system=JUDGE_SYSTEM, max_tokens=4000, task="judge")
            parsed = extract_json(resp.text)
            got = parsed.get("scores", []) if isinstance(parsed, dict) else []
            got = [s for s in got if isinstance(s, dict)]
            if cache:
                cache.put(key, got)
        ids = {c["row_id"] for c in chunk}
        for s in got:
            try:
                score = int(s.get("score"))
            except (TypeError, ValueError):
                continue
            if s.get("row_id") in ids and 1 <= score <= 5:
                out[s["row_id"]] = {"score": score, "reason": str(s.get("reason", ""))}
    return out
