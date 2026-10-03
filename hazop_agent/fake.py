"""Scripted stand-in for an LLM: exercises the whole agent / evaluation flow without any API call.

Its output is placeholder text, not a HAZOP result. Use it for tests and smoke runs only
(``--provider fake``). Canned rows can be replaced with HAZOP_FAKE_ROWS=<rows.json>.
"""
from __future__ import annotations

import json
import os
import random
import re

from .llm import LLM, LLMResponse, extract_json

PLACEHOLDER_ROWS = [
    {"deviation": "Less Flow | main feed line", "causes": "Feed control valve fails closed",
     "consequences": "Loss of feed to the unit and reduced output", "safeguards": ["Low flow alarm"], "FI": 4, "SI": 2,
     "recommendations": []},
    {"deviation": "More Flow | main feed line", "causes": "Feed control valve fails open",
     "consequences": "Overheating of the unit", "safeguards": ["High temperature alarm"], "FI": 3, "SI": 3,
     "recommendations": ["Add an independent high temperature trip"]},
    {"deviation": "High Pressure | outlet line", "causes": "Outlet isolation valve closed in error",
     "consequences": "Overpressure and possible loss of containment, fire", "safeguards": ["Pressure relief valve"],
     "FI": 3, "SI": 5, "recommendations": ["Review relief valve sizing"]},
    {"deviation": "Low Temperature | unit", "causes": "Heater failure",
     "consequences": "Reduced output", "safeguards": ["Low temperature alarm"], "FI": 4, "SI": 2,
     "recommendations": []},
    {"deviation": "No Flow | air supply", "causes": "Air blower trip",
     "consequences": "Loss of cooling and overheating", "safeguards": ["Low flow alarm"], "FI": 3, "SI": 3,
     "recommendations": []},
]


def _rows_pool() -> list[dict]:
    path = os.getenv("HAZOP_FAKE_ROWS")
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return PLACEHOLDER_ROWS


def _complete_row(r: dict, node: str) -> dict:
    row = {"node": node, "safeguards": [], "recommendations": [], **r}
    row["RI"] = row["FI"] * row["SI"]
    row.setdefault("SIL", None)
    if row["SIL"] is None:
        row.pop("SIL")             # let the schema fill it from the matrix
    return row


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


class FakeLLM(LLM):
    provider = "fake"

    def __init__(self, seed: int = 0, keep: float = 0.7):
        super().__init__("fake-scripted")
        self.seed = seed - 1
        self.keep = keep
        self._plan: list = []

    def _usage(self, prompt_len: int = 0) -> dict:
        return {"input_tokens": prompt_len // 4, "output_tokens": 50}

    # ---- agent: a fixed tool-call script --------------------------------------------------
    def _new_plan(self, node: str) -> list:
        self.seed += 1
        rng = random.Random(self.seed)
        pool = _rows_pool()
        chosen = [r for r in pool if rng.random() < self.keep] or pool[:1]
        plan: list = [[("view_pid", {}), ("get_node_description", {})], [("list_guidewords", {})]]
        bad = dict(_complete_row(chosen[0], node), RI=99)           # one invalid row, to exercise the error path
        plan.append([("add_row", bad)])
        plan += [[("add_row", _complete_row(r, node))] for r in chosen]
        plan.append([("finish", {"summary": f"Placeholder study with {len(chosen)} rows."})])
        return plan

    def chat(self, system, messages, tools=None, max_tokens=16000, task=None):
        if tools:
            return self._agent(messages)
        prompt = "\n".join(b.get("text", "") for m in messages if m["role"] == "user" for b in m["content"])
        if task == "single":
            return self._single(prompt)
        if task == "match":
            return self._match(prompt)
        if task == "judge":
            return self._judge(prompt)
        return LLMResponse(text="{}", usage=self._usage(len(prompt)), model=self.model)

    def _agent(self, messages):
        if len(messages) == 1:
            m = re.search(r"node '([^']+)'", messages[0]["content"][0]["text"])
            self._plan = self._new_plan(m.group(1) if m else "Node")
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if n_assistant >= len(self._plan):
            return LLMResponse(text="Done.", model=self.model, usage=self._usage())
        calls = [{"id": f"fake_{n_assistant}_{i}", "name": n, "input": a} for i, (n, a) in enumerate(self._plan[n_assistant])]
        return LLMResponse(text="", tool_calls=calls, stop_reason="tool_use", model=self.model, usage=self._usage())

    # ---- single call -----------------------------------------------------------------------
    def _single(self, prompt):
        m = re.search(r"## Node: (.+)", prompt)
        node = m.group(1).strip() if m else "Node"
        self.seed += 1
        rng = random.Random(self.seed)
        pool = _rows_pool()
        rows = [_complete_row(r, node) for r in pool if rng.random() < self.keep] or [_complete_row(pool[0], node)]
        return LLMResponse(text=json.dumps({"worksheet": rows}), model=self.model, usage=self._usage(len(prompt)))

    # ---- evaluation helpers: crude word overlap, NOT a real judgement -----------------------
    def _match(self, prompt):
        blocks = re.findall(r"```json\s*(.*?)```", prompt, flags=re.S)
        ai = extract_json(blocks[0]) if blocks else {}
        refs = extract_json(blocks[1]) if len(blocks) > 1 else []
        a = _tokens(f"{ai.get('causes', '')} {ai.get('consequences', '')}")
        matches = []
        for r in refs or []:
            b = _tokens(f"{r.get('causes', '')} {r.get('consequences', '')}")
            j = len(a & b) / max(1, len(a | b))
            if j >= 0.3:
                matches.append({"ref_id": r["ref_id"], "reason": f"fake matcher: word overlap {j:.2f}"})
        return LLMResponse(text=json.dumps({"matches": matches}), model=self.model, usage=self._usage(len(prompt)))

    def _judge(self, prompt):
        blocks = re.findall(r"```json\s*(.*?)```", prompt, flags=re.S)
        rows = extract_json(blocks[0]) if blocks else []
        scores = [{"row_id": r["row_id"], "score": 4 if len(str(r.get("consequences", "")).split()) > 3 else 3,
                   "reason": "fake judge: placeholder score"} for r in rows or []]
        return LLMResponse(text=json.dumps({"scores": scores}), model=self.model, usage=self._usage(len(prompt)))
