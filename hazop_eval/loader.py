"""Read the reference worksheet (CSV) and agent runs (worksheet.json)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _split(v: str) -> list[str]:
    return [s.strip() for s in str(v or "").split(";") if s.strip()]


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_reference(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            rows.append({
                "ref_id": (r.get("ref_id") or f"R{i:02d}").strip(),
                "node": r.get("node", ""), "deviation": r.get("deviation", ""),
                "causes": r.get("causes", ""), "consequences": r.get("consequences", ""),
                "safeguards": _split(r.get("safeguards")), "recommendations": _split(r.get("recommendations")),
                "FI": _int(r.get("FI")), "SI": _int(r.get("SI")), "RI": _int(r.get("RI")), "SIL": _int(r.get("SIL")),
            })
    ids = [r["ref_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("reference ref_id values must be unique")
    return rows


def load_run(path: str | Path) -> dict:
    """Accepts a run folder (containing worksheet.json) or the worksheet.json file itself."""
    p = Path(path)
    if p.is_dir():
        p = p / "worksheet.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    name = p.parent.name if p.name == "worksheet.json" else p.stem
    rows = []
    for i, r in enumerate(data.get("rows", []), 1):
        rows.append({**r, "row_id": f"{name}#{i}", "run": name})
    return {"name": name, "meta": data.get("meta", {}), "rows": rows}


def load_runs(paths: list[str | Path]) -> list[dict]:
    runs = [load_run(p) for p in paths]
    names = [r["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError(f"run names must be unique: {names}")
    return runs
