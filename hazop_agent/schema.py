"""Worksheet row schema and validation.

A row is one cause -> consequence scenario under one deviation. ``validate_row`` returns
the normalised row plus a list of errors (row rejected) and warnings (row accepted).
"""
from __future__ import annotations

from typing import Any

from .criteria import suggested_sil

FIELDS = ["node", "deviation", "causes", "consequences", "safeguards",
          "FI", "SI", "RI", "SIL", "recommendations"]
TEXT_FIELDS = ["node", "deviation", "causes", "consequences"]
LIST_FIELDS = ["safeguards", "recommendations"]
MAX_TEXT = 1500

# JSON schema given to the LLM for the add_row tool (and used to describe the single-call output).
ROW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "node": {"type": "string", "description": "Node name"},
        "deviation": {"type": "string", "description": "'<Guideword> <Parameter> | <stream or equipment>', e.g. 'Less Flow | cooling water supply'"},
        "causes": {"type": "string", "description": "Initiating cause of this scenario"},
        "consequences": {"type": "string", "description": "Consequence if the scenario develops"},
        "safeguards": {"type": "array", "items": {"type": "string"}, "description": "Existing safeguards shown on the P&ID or stated in the node description; empty if none"},
        "FI": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Frequency index (before safeguards)"},
        "SI": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Severity index (before safeguards)"},
        "RI": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Risk index = FI x SI"},
        "SIL": {"type": "integer", "minimum": 0, "maximum": 4, "description": "Target SIL from the matrix (0 = no SIL requirement)"},
        "recommendations": {"type": "array", "items": {"type": "string"}, "description": "New measures proposed; empty if none"},
    },
    "required": ["node", "deviation", "causes", "consequences", "safeguards", "FI", "SI", "RI", "SIL", "recommendations"],
    "additionalProperties": False,
}


def _as_list(v: Any) -> list[str] | None:
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip(" -•\t") for s in v.replace(";", "\n").splitlines() if s.strip(" -•\t")]
    if isinstance(v, (list, tuple)):
        return [str(s).strip() for s in v if str(s).strip()]
    return None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return None


def validate_row(raw: Any, criteria: dict) -> tuple[dict | None, list[str], list[str]]:
    """Return (row or None, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return None, ["row must be a JSON object"], []
    unknown = sorted(set(raw) - set(FIELDS))
    if unknown:
        warnings.append(f"ignored unknown field(s): {', '.join(unknown)}")
    row: dict[str, Any] = {}

    for f in TEXT_FIELDS:
        v = raw.get(f)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"'{f}' must be a non-empty string")
            continue
        v = " ".join(v.split())
        if len(v) > MAX_TEXT:
            errors.append(f"'{f}' is longer than {MAX_TEXT} characters")
        row[f] = v

    for f in LIST_FIELDS:
        v = _as_list(raw.get(f))
        if v is None:
            errors.append(f"'{f}' must be a list of strings")
        else:
            row[f] = v

    for f in ("FI", "SI"):
        v = _as_int(raw.get(f))
        if v is None or not 1 <= v <= 5:
            errors.append(f"'{f}' must be an integer from 1 to 5")
        else:
            row[f] = v

    if "FI" in row and "SI" in row:
        expected = row["FI"] * row["SI"]
        ri = _as_int(raw.get("RI"))
        if raw.get("RI") is None:
            row["RI"] = expected
            warnings.append(f"RI was missing and was set to FI x SI = {expected}")
        elif ri != expected:
            errors.append(f"'RI' must equal FI x SI = {expected} (got {raw.get('RI')!r})")
        else:
            row["RI"] = ri

        target = suggested_sil(criteria, row["FI"], row["SI"])
        sil = _as_int(raw.get("SIL"))
        if raw.get("SIL") is None:
            row["SIL"] = target
            warnings.append(f"SIL was missing and was set from the matrix to {target}")
        elif sil is None or not 0 <= sil <= 4:
            errors.append("'SIL' must be an integer from 0 to 4 (0 = no SIL requirement)")
        else:
            row["SIL"] = sil
            if sil != target:
                warnings.append(f"SIL {sil} differs from the matrix value {target} for FI={row['FI']}, SI={row['SI']}")

    if errors:
        return None, errors, warnings
    return {f: row[f] for f in FIELDS}, errors, warnings
