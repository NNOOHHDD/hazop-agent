"""The agent's tools and their executor."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .criteria import guidewords
from .llm import image_block, text_block
from .schema import ROW_JSON_SCHEMA, validate_row

TOOL_SPECS = [
    {"name": "view_pid",
     "description": "Show the P&ID drawing of the plant section being studied (image).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_node_description",
     "description": "Return the node name, its scope and the design intent written by the study team.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_guidewords",
     "description": "Return the standard HAZOP guidewords, the process parameters and the deviation format.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "add_row",
     "description": "Add one scenario (one cause -> consequence pair) to the worksheet. The row is checked against "
                    "the schema; errors are returned and the row is not recorded until they are fixed.",
     "input_schema": ROW_JSON_SCHEMA},
    {"name": "finish",
     "description": "End the study of this node. Give a one-paragraph summary of what was covered.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                      "required": ["summary"], "additionalProperties": False}},
]


@dataclass
class Toolbox:
    pid_image: tuple[bytes, str]
    node_name: str
    node_description: str
    criteria: dict
    rows: list[dict] = field(default_factory=list)
    rejected: int = 0
    finished: bool = False
    summary: str = ""

    def call(self, name: str, args: dict) -> tuple[list[dict], bool]:
        """Run one tool. Returns (content blocks, is_error)."""
        args = args if isinstance(args, dict) else {}
        if name == "view_pid":
            return [text_block(f"P&ID for node '{self.node_name}':"), image_block(*self.pid_image)], False
        if name == "get_node_description":
            return [text_block(json.dumps({"node": self.node_name, "description": self.node_description}, ensure_ascii=False))], False
        if name == "list_guidewords":
            g = guidewords()
            out = {"guidewords": [{"name": x["name"], "meaning": x["meaning"]} for x in g["guidewords"]],
                   "parameters": [p["name"] for p in g["parameters"]],
                   "deviation_format": g["deviation_format"], "source": g["source"]}
            return [text_block(json.dumps(out, ensure_ascii=False))], False
        if name == "add_row":
            if "_unparsed_input" in args:
                self.rejected += 1
                return [text_block("ERROR: the tool input was not valid JSON. Send the row again.")], True
            row, errors, warnings = validate_row(args, self.criteria)
            if errors:
                self.rejected += 1
                return [text_block("Row NOT recorded. Fix: " + "; ".join(errors)
                                   + (" | Notes: " + "; ".join(warnings) if warnings else ""))], True
            self.rows.append(row)
            msg = f"Row {len(self.rows)} recorded."
            if warnings:
                msg += " Notes: " + "; ".join(warnings)
            return [text_block(msg)], False
        if name == "finish":
            self.finished = True
            self.summary = str(args.get("summary") or "")
            return [text_block(f"Finished with {len(self.rows)} rows.")], False
        return [text_block(f"ERROR: unknown tool '{name}'.")], True
