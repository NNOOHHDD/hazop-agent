"""Reference data: guidewords / parameters and the FI / SI / RI / SIL ranking scales.

Both live in ``hazop_agent/data/`` as JSON so that a project can swap in its own
tables (``--criteria my_matrix.json``) without touching code.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path


def _load_packaged(name: str) -> dict:
    return json.loads(resources.files("hazop_agent.data").joinpath(name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def guidewords() -> dict:
    return _load_packaged("guidewords.json")


def load_criteria(path: str | Path | None = None) -> dict:
    if path is None:
        return _load_packaged("risk_criteria.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def suggested_sil(criteria: dict, fi: int, si: int) -> int:
    """Target SIL from the criteria's SI x FI matrix."""
    return int(criteria["sil_matrix"]["SI"][str(si)][fi - 1])


def criteria_text(criteria: dict) -> str:
    """Plain-text rendering of the scales, used in prompts."""
    lines = ["Frequency index (FI):"]
    lines += [f"  {r['value']} {r['label']}: {r['description']}" for r in criteria["frequency_index"]]
    lines.append("Severity index (SI):")
    lines += [f"  {r['value']} {r['label']}: {r['description']}" for r in criteria["severity_index"]]
    lines.append("Risk index: RI = FI x SI.")
    lines.append("Target SIL (rows SI, columns FI 1..5; 0 = no SIL requirement):")
    for si in ("5", "4", "3", "2", "1"):
        lines.append(f"  SI {si}: " + " ".join(str(v) for v in criteria["sil_matrix"]["SI"][si]))
    lines.append("Convention: " + criteria.get("convention", ""))
    return "\n".join(lines)


def guidewords_text() -> str:
    g = guidewords()
    lines = ["Guidewords (" + g["source"].split(".")[0] + "):"]
    lines += [f"  {x['name']}: {x['meaning']}" for x in g["guidewords"]]
    lines.append("Parameters: " + ", ".join(p["name"] for p in g["parameters"]))
    lines.append("Deviation format: " + g["deviation_format"])
    return "\n".join(lines)


# --- deviation parsing (used by the evaluator for coverage) -------------------------------

def _synonym_table(kind: str) -> list[tuple[str, str]]:
    pairs = []
    for item in guidewords()[kind]:
        for s in item["synonyms"] + [item["name"].lower()]:
            pairs.append((s.lower(), item["name"]))
    return sorted(set(pairs), key=lambda p: -len(p[0]))   # longest phrase first


def parse_deviation(text: str) -> tuple[str | None, str | None]:
    """'Low Temperature | MCFC stack' -> ('Less', 'Temperature'). Either part may be None."""
    head = str(text or "").split("|")[0].lower()
    head = re.sub(r"[^a-z0-9 \-]+", " ", head)
    padded = f" {' '.join(head.split())} "
    gw = next((name for syn, name in _synonym_table("guidewords") if f" {syn} " in padded), None)
    par = next((name for syn, name in _synonym_table("parameters") if f" {syn} " in padded), None)
    return gw, par


def coverage_key(text: str) -> str | None:
    """Guideword-parameter key with coverage families applied ('No Flow' -> 'Less Flow')."""
    gw, par = parse_deviation(text)
    if not gw or not par:
        return None
    gw = guidewords().get("coverage_families", {}).get(gw, gw)
    return f"{gw} {par}"
