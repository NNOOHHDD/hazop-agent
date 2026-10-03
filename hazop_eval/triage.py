"""Sort reference rows by how often the agent reproduced them, and group the agent-only rows.

frequent   - reproduced in >= `frequent` of the runs: generic items the agent handles reliably
occasional - reproduced in some runs
never      - never reproduced: candidates for project-specific hazards that need expert time
             (or plain misses; the report cannot tell these apart, an expert must)
"""
from __future__ import annotations

from hazop_agent.criteria import coverage_key

from .textsim import cosine_sparse, tfidf_vectors

CLASSES = ("never", "occasional", "frequent")


def classify(k: int, n: int, frequent: float) -> str:
    if k == 0:
        return "never"
    return "frequent" if n and k / n >= frequent - 1e-9 else "occasional"


def reproducibility(refs: list[dict], matched_by_run: dict[str, set], frequent: float) -> dict:
    n = len(matched_by_run)
    rows = []
    for r in refs:
        runs = [name for name, ids in matched_by_run.items() if r["ref_id"] in ids]
        rows.append({"ref_id": r["ref_id"], "deviation": r["deviation"], "causes": r["causes"],
                     "consequences": r["consequences"], "safeguards": r["safeguards"], "FI": r["FI"], "SI": r["SI"],
                     "runs_matched": len(runs), "fraction": round(len(runs) / n, 4) if n else 0.0,
                     "matched_runs": runs, "class": classify(len(runs), n, frequent)})
    counts = {c: sum(1 for x in rows if x["class"] == c) for c in CLASSES}
    hist = {str(k): sum(1 for x in rows if x["runs_matched"] == k) for k in range(n + 1)}
    return {"n_runs": n, "frequent_threshold": frequent, "counts": counts, "histogram_runs_matched": hist,
            "reference_rows": rows}


def ai_only_clusters(rows: list[dict], n_runs: int, judge: dict, threshold: float = 0.5) -> list[dict]:
    """Group unmatched AI rows from all runs into near-duplicate clusters (lexical similarity, same deviation key)."""
    if not rows:
        return []
    vecs = tfidf_vectors([f"{r['causes']}. {r['consequences']}" for r in rows])
    keys = [coverage_key(r["deviation"]) for r in rows]
    clusters: list[dict] = []
    for i, r in enumerate(rows):
        for c in clusters:
            j = c["_rep"]
            if keys[i] == keys[j] and cosine_sparse(vecs[i], vecs[j]) >= threshold:
                c["_members"].append(i)
                break
        else:
            clusters.append({"_rep": i, "_members": [i]})
    out = []
    for c in clusters:
        rep = rows[c["_rep"]]
        members = [rows[i] for i in c["_members"]]
        runs = sorted({m["run"] for m in members})
        scores = [judge[m["row_id"]]["score"] for m in members if m["row_id"] in judge]
        out.append({"deviation": rep["deviation"], "causes": rep["causes"], "consequences": rep["consequences"],
                    "safeguards": rep.get("safeguards", []), "recommendations": rep.get("recommendations", []),
                    "FI": rep.get("FI"), "SI": rep.get("SI"), "run_count": len(runs), "n_runs": n_runs, "runs": runs,
                    "row_ids": [m["row_id"] for m in members],
                    "judge_mean": round(sum(scores) / len(scores), 2) if scores else None})
    out.sort(key=lambda x: (-x["run_count"], x["deviation"]))
    return out
