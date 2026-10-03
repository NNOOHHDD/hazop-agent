"""Decide which AI rows and reference rows describe the same scenario.

llm_match       - an LLM reads one AI row against all reference rows and names the matches,
                  with a one-sentence reason for each (stored as evidence).
embedding_match - cosine similarity of 'deviation + cause + consequence' above a threshold.
Both are many-to-many: one AI row may cover several reference rows and vice versa.
"""
from __future__ import annotations

import json

from hazop_agent.llm import LLM, extract_json

from .cache import LLMCache
from .textsim import DEFAULT_THRESHOLDS, similarity_matrix

MATCH_SYSTEM = "You compare HAZOP scenarios for a process safety review. Answer with JSON only."

MATCH_PROMPT = """Decide which rows of a reference HAZOP worksheet describe the SAME scenario as the scenario under review.

Same scenario = the same initiating cause (same failure mechanism; where equipment is named, the same item or an obviously equivalent one) leading to the same kind of consequence.
- Wording, level of detail and the exact deviation label may differ, but the deviation direction must be compatible ('No Flow' is compatible with 'Less Flow', not with 'More Flow').
- Ignore safeguards, ratings and recommendations.
- A reference row with no consequence recorded can match on the cause alone.
- If the scenario under review names several causes or consequences, it matches each reference row whose cause and consequence it fully contains.

## Scenario under review
```json
{ai}
```

## Reference rows
```json
{refs}
```

Return JSON only: {{"matches": [{{"ref_id": "<id>", "reason": "<one sentence>"}}]}}
Use an empty list when no reference row is the same scenario."""


def _scenario(r: dict, id_key: str) -> dict:
    return {id_key: r[id_key], "deviation": r.get("deviation", ""), "causes": r.get("causes", ""),
            "consequences": r.get("consequences", "")}


def llm_match(llm: LLM, ai_rows: list[dict], refs: list[dict], cache: LLMCache | None = None,
              progress=None) -> list[dict]:
    """Return [{"ai_id", "ref_id", "reason"}]."""
    ref_json = json.dumps([_scenario(r, "ref_id") for r in refs], ensure_ascii=False, indent=1)
    valid = {r["ref_id"] for r in refs}
    pairs = []
    for n, row in enumerate(ai_rows, 1):
        ai = {k: v for k, v in _scenario(row, "row_id").items() if k != "row_id"}
        prompt = MATCH_PROMPT.format(ai=json.dumps(ai, ensure_ascii=False, indent=1), refs=ref_json)
        key = LLMCache.key(llm.label, "match", MATCH_SYSTEM, prompt)
        out = cache.get(key) if cache else None
        if out is None:
            resp = llm.complete(prompt, system=MATCH_SYSTEM, max_tokens=4000, task="match")
            parsed = extract_json(resp.text)
            out = parsed.get("matches", []) if isinstance(parsed, dict) else []
            out = [m for m in out if isinstance(m, dict)]
            if cache:
                cache.put(key, out)
        seen = set()
        for m in out:
            rid = str(m.get("ref_id", "")).strip()
            if rid in valid and rid not in seen:
                seen.add(rid)
                pairs.append({"ai_id": row["row_id"], "ref_id": rid, "reason": str(m.get("reason", ""))})
        if progress:
            progress(n, len(ai_rows))
    return pairs


def _text(r: dict) -> str:
    return f"{r.get('deviation', '')}. {r.get('causes', '')}. {r.get('consequences', '')}"


def embedding_match(ai_rows: list[dict], refs: list[dict], backend: str = "tfidf",
                    threshold: float | None = None) -> list[dict]:
    """Return [{"ai_id", "ref_id", "score"}] for pairs with cosine >= threshold."""
    if not ai_rows or not refs:
        return []
    thr = DEFAULT_THRESHOLDS.get(backend, 0.5) if threshold is None else threshold
    sims = similarity_matrix([_text(r) for r in ai_rows], [_text(r) for r in refs], backend)
    return [{"ai_id": a["row_id"], "ref_id": r["ref_id"], "score": round(sims[i][j], 4)}
            for i, a in enumerate(ai_rows) for j, r in enumerate(refs) if sims[i][j] >= thr]
