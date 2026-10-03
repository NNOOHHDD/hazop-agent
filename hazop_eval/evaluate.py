"""Run the whole evaluation and write metrics.json, matches.json and report.html."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from hazop_agent.llm import LLM

from . import __version__
from .cache import LLMCache
from .judge import judge_rows
from .loader import load_reference, load_runs
from .matching import embedding_match, llm_match
from .metrics import deviation_coverage, describe, match_scores, pair_agreement, risk_agreement
from .report import render_report
from .textsim import DEFAULT_THRESHOLDS
from .triage import ai_only_clusters, reproducibility


def evaluate(reference: str | Path, run_paths: list[str | Path], out_dir: str | Path, llm: LLM | None,
             matcher: str = "llm", embedding: str = "tfidf", threshold: float | None = None,
             judge: bool = True, frequent: float = 0.6, context: str = "", log=print) -> dict:
    if matcher == "llm" and llm is None:
        raise ValueError("matcher 'llm' needs an LLM")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = LLMCache(out / "llm_cache.json")
    refs = load_reference(reference)
    ref_by_id = {r["ref_id"]: r for r in refs}
    runs = load_runs(run_paths)
    thr = DEFAULT_THRESHOLDS.get(embedding, 0.5) if threshold is None else threshold
    context = context or (refs[0]["node"] if refs else "")

    per_run, matches, matched_by_run, judged, unmatched = [], {}, {}, {}, []
    for run in runs:
        rows = run["rows"]
        ai_by_id = {r["row_id"]: r for r in rows}
        log(f"[{run['name']}] {len(rows)} rows")
        emb_pairs = embedding_match(rows, refs, embedding, thr)
        if matcher == "llm":
            llm_pairs = llm_match(llm, rows, refs, cache)
            primary, secondary = llm_pairs, emb_pairs
        else:
            primary, secondary = emb_pairs, []
        scores = match_scores(rows, refs, primary)
        matched_by_run[run["name"]] = set(scores["matched_reference_ids"])
        matched_ai = {p["ai_id"] for p in primary}
        unmatched += [r for r in rows if r["row_id"] not in matched_ai]

        entry = {"run": run["name"], "model": run["meta"].get("model"), "mode": run["meta"].get("mode"),
                 "status": run["meta"].get("status"), "match": scores,
                 "risk_agreement": risk_agreement(primary, ai_by_id, ref_by_id),
                 "coverage": deviation_coverage(rows, refs)}
        if matcher == "llm":
            emb = match_scores(rows, refs, emb_pairs)
            entry["embedding_match"] = {k: v for k, v in emb.items() if k != "matched_reference_ids"}
            entry["matcher_agreement"] = pair_agreement(llm_pairs, emb_pairs, rows, refs)
        if judge and llm is not None:
            js = judge_rows(llm, rows, context, cache)
            judged.update(js)
            vals = [v["score"] for v in js.values()]
            entry["judge"] = {**describe(vals), "distribution": {str(s): vals.count(s) for s in range(1, 6)},
                              "unscored_rows": len(rows) - len(vals)}
        per_run.append(entry)
        matches[run["name"]] = {"primary": primary, "embedding": emb_pairs if matcher == "llm" else None}

    repro = reproducibility(refs, matched_by_run, frequent)
    clusters = ai_only_clusters(unmatched, len(runs), judged)
    summary = {k: describe([r["match"][k] for r in per_run]) for k in ("recall", "precision", "f1")}
    summary["coverage"] = describe([r["coverage"]["coverage"] for r in per_run])
    if judge and llm is not None:
        summary["judge_mean"] = describe([r["judge"].get("mean") for r in per_run])
    for f in ("FI", "SI"):
        summary[f"{f}_exact"] = describe([(r["risk_agreement"][f] or {}).get("exact") for r in per_run])
    if matcher == "llm":
        summary["embedding_recall"] = describe([r["embedding_match"]["recall"] for r in per_run])
        summary["matcher_kappa"] = describe([r["matcher_agreement"].get("cohen_kappa") for r in per_run])

    metrics = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"), "hazop_eval_version": __version__,
        "inputs": {"reference": str(reference), "n_reference_rows": len(refs), "runs": [r["name"] for r in runs],
                   "matcher": matcher, "matcher_model": llm.label if (llm and matcher == "llm") else None,
                   "embedding_backend": embedding, "embedding_threshold": thr,
                   "judge_model": llm.label if (judge and llm) else None, "frequent_threshold": frequent,
                   "llm_cache": {"hits": cache.hits, "misses": cache.misses}},
        "summary": summary, "runs": per_run, "reproducibility": repro,
        "ai_only": {"n_rows": len(unmatched), "clusters": clusters},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "matches.json").write_text(json.dumps(matches, indent=2, ensure_ascii=False), encoding="utf-8")
    all_rows = {r["row_id"]: r for run in runs for r in run["rows"]}
    (out / "report.html").write_text(render_report(metrics, matches, all_rows, judged), encoding="utf-8")
    log(f"wrote {out / 'metrics.json'}, {out / 'matches.json'}, {out / 'report.html'}")
    return metrics
