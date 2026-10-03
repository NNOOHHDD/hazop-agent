import json

import pytest

from hazop_agent.agent import run_agent, save_run
from hazop_agent.fake import FakeLLM
from hazop_eval.cache import LLMCache
from hazop_eval.evaluate import evaluate
from hazop_eval.loader import load_reference
from hazop_eval.matching import embedding_match, llm_match
from hazop_eval.metrics import match_scores, pair_agreement, risk_agreement
from hazop_eval.triage import ai_only_clusters, classify, reproducibility


def _ai(i, **kw):
    r = {"row_id": f"run#{i}", "run": "run", "deviation": "Less Flow | air", "causes": "c", "consequences": "q",
         "FI": 4, "SI": 2, "RI": 8, "SIL": 0, "safeguards": [], "recommendations": []}
    r.update(kw)
    return r


def test_reference_loads(reference_path):
    refs = load_reference(reference_path)
    assert len(refs) == 29 and refs[0]["ref_id"] == "R01" and refs[7]["consequences"] == ""
    assert all(r["RI"] == r["FI"] * r["SI"] for r in refs)


def test_match_scores_and_risk_agreement():
    refs = [{"ref_id": "R1", "FI": 4, "SI": 2}, {"ref_id": "R2", "FI": 3, "SI": 5}]
    ai = [_ai(1), _ai(2, FI=3, SI=3, RI=9), _ai(3)]
    pairs = [{"ai_id": "run#1", "ref_id": "R1"}, {"ai_id": "run#2", "ref_id": "R2"}]
    s = match_scores(ai, refs, pairs)
    assert s["recall"] == 1.0 and s["precision"] == pytest.approx(2 / 3, abs=1e-3)
    ra = risk_agreement(pairs, {a["row_id"]: a for a in ai}, {r["ref_id"]: {**r, "RI": r["FI"] * r["SI"]} for r in refs})
    assert ra["FI"]["exact"] == 1.0 and ra["SI"]["exact"] == 0.5 and ra["SI"]["mean_signed_diff"] == -1.0


def test_pair_agreement_kappa():
    ai, refs = [_ai(1), _ai(2)], [{"ref_id": "R1"}, {"ref_id": "R2"}]
    p = [{"ai_id": "run#1", "ref_id": "R1"}]
    assert pair_agreement(p, p, ai, refs)["cohen_kappa"] == 1.0
    assert pair_agreement(p, [], ai, refs)["only_first"] == 1


def test_classification_thresholds():
    assert classify(0, 3, 0.6) == "never"
    assert classify(1, 3, 0.6) == "occasional"
    assert classify(2, 3, 0.6) == "frequent"
    assert classify(3, 5, 0.6) == "frequent" and classify(2, 5, 0.6) == "occasional"


def test_reproducibility_counts():
    refs = [{"ref_id": x, "deviation": "", "causes": "", "consequences": "", "safeguards": [], "FI": 1, "SI": 1}
            for x in ("A", "B", "C")]
    rep = reproducibility(refs, {"r1": {"A", "B"}, "r2": {"A"}, "r3": {"A"}}, 0.6)
    assert rep["counts"] == {"never": 1, "occasional": 1, "frequent": 1}
    assert rep["histogram_runs_matched"] == {"0": 1, "1": 1, "2": 0, "3": 1}


def test_ai_only_clusters_merge_near_duplicates():
    rows = [_ai(1, run="a", causes="Blower trips", consequences="Loss of cooling air"),
            _ai(2, run="b", causes="Blower trip", consequences="Loss of cooling air"),
            _ai(3, run="b", deviation="More Pressure | x", causes="Blower trips", consequences="Loss of cooling air")]
    cl = ai_only_clusters(rows, 2, {})
    assert [c["run_count"] for c in cl] == [2, 1]


def test_matchers_with_fake_llm(reference_path):
    refs = load_reference(reference_path)
    ai = [_ai(1, causes=refs[0]["causes"], consequences=refs[0]["consequences"])]
    cache = LLMCache(None)
    pairs = llm_match(FakeLLM(), ai, refs, cache)
    assert {"ai_id": "run#1", "ref_id": "R01"} in [{k: p[k] for k in ("ai_id", "ref_id")} for p in pairs]
    assert all(p["reason"] for p in pairs)
    emb = embedding_match(ai, refs, "tfidf", 0.3)
    assert any(p["ref_id"] == "R01" for p in emb)


def test_full_evaluation_and_report(pid_path, node_path, reference_path, fake_rows_from_reference, tmp_path):
    llm = FakeLLM()
    runs = []
    for i in range(1, 4):
        runs.append(save_run(run_agent(llm, pid_path, node_path), tmp_path / f"run_{i:02d}"))
    out = tmp_path / "eval"
    m = evaluate(reference_path, runs, out, FakeLLM(), context="MCFC stack", log=lambda *_: None)
    assert {p.name for p in out.iterdir()} >= {"metrics.json", "matches.json", "report.html", "llm_cache.json"}
    assert m["reproducibility"]["n_runs"] == 3 and sum(m["reproducibility"]["counts"].values()) == 29
    assert 0 < m["summary"]["recall"]["mean"] <= 1 and m["runs"][0]["judge"]["n"] > 0
    assert any("Reverse Flow" in c["deviation"] for c in m["ai_only"]["clusters"])
    html = (out / "report.html").read_text()
    order = [html.index(s) for s in ("Where expert time is needed", "never reproduced", "only the agent proposed",
                                     "Occasionally reproduced", "Frequently reproduced", "Metrics", "Match evidence")]
    assert order == sorted(order)
    json.loads((out / "metrics.json").read_text())
    # second evaluation is served from the cache
    m2 = evaluate(reference_path, runs, out, FakeLLM(), context="MCFC stack", log=lambda *_: None)
    assert m2["inputs"]["llm_cache"]["misses"] == 0 and m2["summary"] == m["summary"]


def test_embedding_only_evaluation_needs_no_llm(pid_path, node_path, reference_path, tmp_path):
    run = save_run(run_agent(FakeLLM(), pid_path, node_path), tmp_path / "run_01")
    m = evaluate(reference_path, [run], tmp_path / "eval", None, matcher="embedding", judge=False, log=lambda *_: None)
    assert m["inputs"]["matcher"] == "embedding" and "judge" not in m["runs"][0]
