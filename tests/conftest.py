import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "mcfc"


@pytest.fixture
def pid_path():
    return EXAMPLE / "pid.jpg"


@pytest.fixture
def node_path():
    return EXAMPLE / "node.md"


@pytest.fixture
def reference_path():
    return EXAMPLE / "reference_worksheet.csv"


@pytest.fixture
def fake_rows_from_reference(tmp_path, reference_path, monkeypatch):
    """Canned agent rows: every other reference row (same wording) plus one agent-only row."""
    rows = []
    for r in list(csv.DictReader(open(reference_path, encoding="utf-8")))[::2]:
        rows.append({"deviation": r["deviation"] + " | stack", "causes": r["causes"],
                     "consequences": r["consequences"] or "No hazardous consequence",
                     "safeguards": r["safeguards"].split("; "), "FI": int(r["FI"]), "SI": int(r["SI"]),
                     "recommendations": []})
    rows.append({"deviation": "Reverse Flow | anode outlet", "causes": "Back flow from the vent header",
                 "consequences": "Air ingress to the anode and local hot spots", "safeguards": [],
                 "FI": 2, "SI": 4, "recommendations": ["Add a check valve"]})
    p = tmp_path / "fake_rows.json"
    p.write_text(json.dumps(rows))
    monkeypatch.setenv("HAZOP_FAKE_ROWS", str(p))
    return rows
