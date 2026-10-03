"""Static, self-contained HTML report. The list that needs expert time comes first."""
from __future__ import annotations

from html import escape

CSS = """
:root { --bg:#ffffff; --fg:#1d232b; --muted:#5b6673; --line:#d9dee4; --head:#f3f5f7; --accent:#9a3412;
        --focus:#fff4e5; --ok:#e9f6ee; --mid:#f4f6fb; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14181d; --fg:#e4e8ec; --muted:#9aa5b1; --line:#323a44; --head:#1d232b; --accent:#fb923c;
          --focus:#2b2117; --ok:#16261d; --mid:#1a1f27; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 36px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
h3 { font-size: 15px; margin: 20px 0 6px; }
p.muted, .muted { color: var(--muted); }
.note { border-left: 3px solid var(--accent); padding: 8px 12px; background: var(--head); margin: 12px 0; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
.card { border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; min-width: 150px; }
.card b { display: block; font-size: 20px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px; font-size: 13px; }
th, td { border: 1px solid var(--line); padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: var(--head); }
tr.focus td { background: var(--focus); } tr.freq td { background: var(--ok); } tr.occ td { background: var(--mid); }
td.num { text-align: right; white-space: nowrap; }
details { margin: 4px 0; } summary { cursor: pointer; }
code { font-size: 12px; overflow-wrap: anywhere; }
p, summary, li { overflow-wrap: anywhere; }
"""


def _e(x) -> str:
    if x is None or x == "":
        return '<span class="muted">-</span>'
    if isinstance(x, (list, tuple)):
        return "<br>".join(escape(str(i)) for i in x) if x else '<span class="muted">-</span>'
    return escape(str(x))


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _stat(d: dict, pct: bool = True) -> str:
    if not d or not d.get("n"):
        return "-"
    f = _pct if pct else (lambda v: f"{v:.2f}")
    return f"{f(d['mean'])} (sd {f(d['sd'])})" if d["n"] > 1 else f(d["mean"])


def _table(head: list[str], rows: list[list[str]], classes: list[str] | None = None) -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join(f'<tr class="{classes[i] if classes else ""}">' + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for i, r in enumerate(rows))
    return f'<div class="wrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _ref_rows(rows: list[dict], n: int, cls: str) -> str:
    if not rows:
        return '<p class="muted">None.</p>'
    return _table(["Ref", "Deviation", "Cause", "Consequence", "Safeguards (reference)", "FI/SI", "Reproduced"],
                  [[_e(r["ref_id"]), _e(r["deviation"]), _e(r["causes"]), _e(r["consequences"]), _e(r["safeguards"]),
                    f"{_e(r['FI'])}/{_e(r['SI'])}", f'<span class="num">{r["runs_matched"]} of {n}</span>'] for r in rows],
                  [cls] * len(rows))


def render_report(metrics: dict, matches: dict, ai_rows: dict, judged: dict) -> str:
    inp, rep, summ = metrics["inputs"], metrics["reproducibility"], metrics["summary"]
    n = rep["n_runs"]
    by_class = {c: [r for r in rep["reference_rows"] if r["class"] == c] for c in ("never", "occasional", "frequent")}
    clusters = metrics["ai_only"]["clusters"]
    thr_txt = f"{100 * rep['frequent_threshold']:.0f}%"
    s = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
         f"<title>HAZOP Evaluation Report</title><style>{CSS}</style></head><body><main>",
         "<h1>HAZOP evaluation report</h1>",
         f"<p class='muted'>{len(inp['runs'])} run(s) compared with {inp['n_reference_rows']} reference rows "
         f"(<code>{_e(inp['reference'])}</code>). Matching: {_e(inp['matcher'])}"
         f"{' (' + _e(inp['matcher_model']) + ')' if inp.get('matcher_model') else ''}; "
         f"embedding comparison: {_e(inp['embedding_backend'])} &ge; {inp['embedding_threshold']}. Generated {_e(metrics['generated'])}.</p>",
         "<div class='note'>This report is a screening aid. 'Never reproduced' means the agent did not find the item in any run; "
         "it may be specific to this project, or the agent may simply have missed it. Agent-only rows may be valid additions "
         "or errors. Matches and judge scores come from an LLM and should be spot-checked (see section 5).</div>",
         "<div class='cards'>",
         f"<div class='card'><span class='muted'>Never reproduced</span><b>{len(by_class['never'])}</b></div>",
         f"<div class='card'><span class='muted'>Agent-only scenarios</span><b>{len(clusters)}</b></div>",
         f"<div class='card'><span class='muted'>Occasionally</span><b>{len(by_class['occasional'])}</b></div>",
         f"<div class='card'><span class='muted'>Frequently (&ge;{thr_txt} of runs)</span><b>{len(by_class['frequent'])}</b></div>",
         f"<div class='card'><span class='muted'>Recall / precision</span><b>{_pct(summ['recall'].get('mean'))} / {_pct(summ['precision'].get('mean'))}</b></div>",
         "</div>",
         "<h2>1. Where expert time is needed</h2>",
         f"<h3>1a. Reference rows the agent never reproduced (0 of {n} runs) &mdash; possible project-specific hazards</h3>",
         _ref_rows(by_class["never"], n, "focus"),
         "<h3>1b. Scenarios only the agent proposed (no reference counterpart) &mdash; review for validity</h3>"]
    if clusters:
        s.append(_table(["Deviation", "Cause", "Consequence", "Safeguards", "Recommendations", "FI/SI", "Seen in", "Judge"],
                        [[_e(c["deviation"]), _e(c["causes"]), _e(c["consequences"]), _e(c["safeguards"]),
                          _e(c["recommendations"]), f"{_e(c['FI'])}/{_e(c['SI'])}",
                          f"{c['run_count']} of {c['n_runs']} runs", _e(c["judge_mean"])] for c in clusters],
                        ["focus"] * len(clusters)))
        s.append("<p class='muted'>Near-duplicates across runs are grouped by word overlap; one representative row is shown.</p>")
    else:
        s.append('<p class="muted">None.</p>')
    s += [f"<h2>2. Occasionally reproduced (1 to below {thr_txt} of runs)</h2>", _ref_rows(by_class["occasional"], n, "occ"),
          f"<h2>3. Frequently reproduced (&ge;{thr_txt} of runs) &mdash; generic items the agent handles</h2>",
          _ref_rows(by_class["frequent"], n, "freq")]

    # ---- metrics
    s.append("<h2>4. Metrics</h2>")
    s.append(_table(["Metric", "Across runs"], [
        ["Recall (reference rows matched)", _stat(summ["recall"])],
        ["Precision (agent rows with a reference match)", _stat(summ["precision"])],
        ["F1", _stat(summ["f1"])],
        ["Guideword-parameter coverage", _stat(summ["coverage"])],
        ["FI exact agreement (matched pairs)", _stat(summ["FI_exact"])],
        ["SI exact agreement (matched pairs)", _stat(summ["SI_exact"])],
    ] + ([["Judge score, cause &rarr; consequence (1-5)", _stat(summ["judge_mean"], pct=False)]] if "judge_mean" in summ else [])
      + ([["Recall with embedding matching", _stat(summ["embedding_recall"])],
          ["LLM vs embedding matching, Cohen's kappa", _stat(summ["matcher_kappa"], pct=False)]] if "embedding_recall" in summ else [])))
    head = ["Run", "Model", "Status", "Rows", "Recall", "Precision", "F1", "Coverage", "FI exact / &plusmn;1", "SI exact / &plusmn;1",
            "Mean |&Delta;RI|", "Judge"]
    body = []
    for r in metrics["runs"]:
        ra = r["risk_agreement"]
        fi, si, ri = ra.get("FI") or {}, ra.get("SI") or {}, ra.get("RI") or {}
        body.append([_e(r["run"]), _e(r["model"]), _e(r["status"]), str(r["match"]["n_ai_rows"]), _pct(r["match"]["recall"]),
                     _pct(r["match"]["precision"]), _pct(r["match"]["f1"]), _pct(r["coverage"]["coverage"]),
                     f"{_pct(fi.get('exact'))} / {_pct(fi.get('within_1'))}", f"{_pct(si.get('exact'))} / {_pct(si.get('within_1'))}",
                     _e(ri.get("mean_abs_diff")), _e((r.get("judge") or {}).get("mean"))])
    s.append("<h3>Per run</h3>" + _table(head, body))
    cov_rows = [[_e(r["run"]), _e(r["coverage"]["missing"]), _e(r["coverage"]["ai_only"]), _e(r["coverage"]["unparsed_ai_deviations"])]
                for r in metrics["runs"]]
    s.append("<h3>Guideword-parameter coverage</h3>" + _table(["Run", "Reference deviations not addressed", "Deviations only the agent used", "Unparsed agent deviations"], cov_rows))
    if any("matcher_agreement" in r for r in metrics["runs"]):
        s.append("<h3>LLM matching vs embedding matching</h3>" + _table(
            ["Run", "Recall (LLM)", "Recall (embedding)", "Pairs both", "LLM only", "Embedding only", "Kappa"],
            [[_e(r["run"]), _pct(r["match"]["recall"]), _pct(r["embedding_match"]["recall"]), str(r["matcher_agreement"].get("both")),
              str(r["matcher_agreement"].get("only_first")), str(r["matcher_agreement"].get("only_second")),
              _e(r["matcher_agreement"].get("cohen_kappa"))] for r in metrics["runs"]]))
    hist = rep["histogram_runs_matched"]
    s.append("<h3>Reproduction across runs</h3>" + _table(["Runs in which a reference row was matched"] + list(hist),
                                                          [["Reference rows"] + [str(v) for v in hist.values()]]))

    # ---- evidence
    s.append("<h2>5. Match evidence</h2><p class='muted'>For each reference row: the agent rows matched to it and the matcher's reason.</p>")
    for r in rep["reference_rows"]:
        items = []
        for run, m in matches.items():
            for p in m["primary"]:
                if p["ref_id"] == r["ref_id"]:
                    a = ai_rows.get(p["ai_id"], {})
                    why = p.get("reason") or (f"cosine {p['score']}" if "score" in p else "")
                    j = judged.get(p["ai_id"])
                    items.append(f"<li><code>{_e(p['ai_id'])}</code> {_e(a.get('deviation'))}: {_e(a.get('causes'))} &rarr; "
                                 f"{_e(a.get('consequences'))} <span class='muted'>(FI/SI {_e(a.get('FI'))}/{_e(a.get('SI'))}"
                                 f"{'; judge ' + str(j['score']) if j else ''})</span><br><span class='muted'>Reason: {_e(why)}</span></li>")
        s.append(f"<details><summary><b>{_e(r['ref_id'])}</b> {_e(r['deviation'])}: {_e(r['causes'])} &rarr; {_e(r['consequences'])} "
                 f"<span class='muted'>({r['runs_matched']} of {n} runs)</span></summary>"
                 f"<ul>{''.join(items) or '<li class=muted>No agent row matched.</li>'}</ul></details>")
    s.append("</main></body></html>")
    return "\n".join(s)
