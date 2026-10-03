"""Metrics computed from match pairs."""
from __future__ import annotations

import math
import statistics
from collections import Counter

from hazop_agent.criteria import coverage_key


def match_scores(ai_rows: list[dict], refs: list[dict], pairs: list[dict]) -> dict:
    """Recall = reference rows covered by at least one AI row; precision = AI rows that match at least one reference row."""
    matched_ref = sorted({p["ref_id"] for p in pairs})
    matched_ai = {p["ai_id"] for p in pairs}
    recall = len(matched_ref) / len(refs) if refs else 0.0
    precision = len(matched_ai) / len(ai_rows) if ai_rows else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"n_ai_rows": len(ai_rows), "n_reference_rows": len(refs), "n_pairs": len(pairs),
            "reference_rows_matched": len(matched_ref), "ai_rows_matched": len(matched_ai),
            "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4),
            "matched_reference_ids": matched_ref}


def _diffs(pairs, ai_by_id, ref_by_id, field):
    out = []
    for p in pairs:
        a, r = ai_by_id[p["ai_id"]].get(field), ref_by_id[p["ref_id"]].get(field)
        if isinstance(a, int) and isinstance(r, int):
            out.append(a - r)
    return out


def risk_agreement(pairs: list[dict], ai_by_id: dict, ref_by_id: dict) -> dict:
    """FI / SI / RI agreement over matched pairs (AI minus reference)."""
    res = {"n_pairs": len(pairs)}
    for f in ("FI", "SI", "RI"):
        d = _diffs(pairs, ai_by_id, ref_by_id, f)
        if not d:
            res[f] = None
            continue
        entry = {"n": len(d), "exact": round(sum(x == 0 for x in d) / len(d), 4),
                 "mean_abs_diff": round(sum(abs(x) for x in d) / len(d), 3),
                 "mean_signed_diff": round(sum(d) / len(d), 3)}
        if f != "RI":
            entry["within_1"] = round(sum(abs(x) <= 1 for x in d) / len(d), 4)
        res[f] = entry
    return res


def deviation_coverage(ai_rows: list[dict], refs: list[dict]) -> dict:
    """Guideword-parameter coverage ('Low Temperature' and 'Less Temperature' are the same key; 'No' counts as 'Less')."""
    ref_keys = Counter(k for r in refs if (k := coverage_key(r["deviation"])))
    ai_keys = Counter(k for r in ai_rows if (k := coverage_key(r["deviation"])))
    unparsed = sorted({r["deviation"] for r in ai_rows if not coverage_key(r["deviation"])})
    covered = sorted(set(ref_keys) & set(ai_keys))
    return {"reference_keys": sorted(ref_keys), "ai_keys": dict(sorted(ai_keys.items())),
            "covered": covered, "missing": sorted(set(ref_keys) - set(ai_keys)),
            "ai_only": sorted(set(ai_keys) - set(ref_keys)), "unparsed_ai_deviations": unparsed,
            "coverage": round(len(covered) / len(ref_keys), 4) if ref_keys else None}


def pair_agreement(a_pairs: list[dict], b_pairs: list[dict], ai_rows: list[dict], refs: list[dict]) -> dict:
    """Agreement of two matchers over the full AI x reference grid (accuracy and Cohen's kappa)."""
    n = len(ai_rows) * len(refs)
    if not n:
        return {"n_cells": 0}
    A = {(p["ai_id"], p["ref_id"]) for p in a_pairs}
    B = {(p["ai_id"], p["ref_id"]) for p in b_pairs}
    both, only_a, only_b = len(A & B), len(A - B), len(B - A)
    neither = n - both - only_a - only_b
    po = (both + neither) / n
    pa, pb = len(A) / n, len(B) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n_cells": n, "both": both, "only_first": only_a, "only_second": only_b,
            "accuracy": round(po, 4), "cohen_kappa": round(kappa, 4)}


def describe(values: list[float]) -> dict:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
            "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4), "max": round(max(vals), 4)}
