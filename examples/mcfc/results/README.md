# Demo results

Outputs of the demo in the main README: three agent runs on the MCFC stack node and one evaluation report against the 29-row reference worksheet.

**This is a small-sample example** (one node, three runs, one model, evaluated by the same model). It shows what the tools produce; it is not a benchmark, and the numbers will move between runs and models.

## Run conditions

| | |
|---|---|
| Date | 2026-10-03 |
| Provider | `claude_cli` (Claude Code CLI 2.1.288, no API key) |
| Model | `opus` (provider default; recorded as `claude_cli:opus` in every `worksheet.json`) |
| Agent | `--mode agent`, `--runs 3`, default step limit (60), default risk criteria |
| Evaluation | LLM matcher and judge with the same provider and model; TF-IDF embedding comparison at threshold 0.3; frequent threshold 0.6 |
| Runs | 3 (all finished; 20, 27 and 25 rows; 0 rows rejected by the schema) |
| Time | agent runs 153 s, 206 s and 126 s; evaluation about 11 min (80 LLM calls) |

```bash
python -m hazop_agent --pid examples/mcfc/pid.jpg --node examples/mcfc/node.md \
    --runs 3 --out examples/mcfc/results/agent
python -m hazop_eval --reference examples/mcfc/reference_worksheet.csv \
    --runs examples/mcfc/results/agent/run_* --node examples/mcfc/node.md \
    --out examples/mcfc/results/eval
```

with `HAZOP_LLM_PROVIDER=claude_cli` in `.env`. `eval/llm_cache.json` holds the matcher and judge answers, so re-running the second command reproduces the report without LLM calls.

## Key numbers

Matching and risk agreement (mean over the 3 runs, range in brackets):

| Metric | Value |
|---|---|
| Recall (reference rows matched) | 0.47 (0.45-0.48) |
| Precision (agent rows matched) | 0.46 (0.44-0.50) |
| F1 | 0.47 (0.46-0.47) |
| FI agreement, exact / within 1 | 0.31 / 0.86 |
| SI agreement, exact / within 1 | 0.25 / 0.86 |
| RI agreement, exact | 0.10 |
| RI difference, agent minus reference (mean) | -1.5 (agent ranks lower on average) |

Risk agreement is computed over matched pairs (17, 28 and 19 pairs in the three runs).

Triage report:

| Class | Count |
|---|---|
| Frequently reproduced (in at least 2 of 3 runs): generic items | 13 of 29 reference rows |
| Occasionally reproduced (1 of 3 runs) | 4 of 29 reference rows |
| Never reproduced: candidates for project-specific hazards, or misses | 12 of 29 reference rows |
| Agent-only scenarios (near-duplicates grouped) | 35 groups from 39 rows (32 seen in 1 run, 2 in 2 runs, 1 in all 3) |

## Reading these numbers

- The reference worksheet is one team's study. Agent-only scenarios are not errors by definition; many address deviations the reference does not list (composition, utilities, start-up). An expert has to accept or reject them.
- The matcher and judge are the same model as the agent, so the evaluation is not independent. The TF-IDF embedding matcher agrees poorly with the LLM matcher here (mean Cohen's kappa 0.07, embedding recall 0.16): the agent words scenarios differently from the reference, which a lexical measure cannot follow. Read the match reasons in `eval/matches.json` or in section 5 of the report before relying on the LLM numbers.
- The agent's FI / SI are usually within one step of the reference, but rarely exact. RI exact agreement is low because a one-step difference in FI or SI changes the product.
- **The agent underestimates risk.** Its RI is on average 1.5 lower than the reference's for the same scenario (mean signed difference -0.2 to -2.4 across the three runs). A worksheet drafted this way would rank hazards as less severe than the human team did, so every FI / SI / RI must be reviewed before use.
- These results come from the deliberately simplified single-agent version in this repository. The metric definitions, model and settings differ from those in the manuscript in preparation, so these numbers cannot be compared directly with the manuscript's.

## Files

| Path | Content |
|---|---|
| `agent/run_NN/worksheet.json`, `worksheet.csv` | the agent's worksheet with run metadata |
| `agent/run_NN/log.json` | system prompt and every tool call with its result |
| `eval/metrics.json` | all numbers, per run and summarised |
| `eval/matches.json` | every matched pair with the matcher's reason |
| `eval/report.html` | the self-contained report (first screen in [docs/report_example.png](../../../docs/report_example.png)) |
| `eval/llm_cache.json` | cached matcher and judge answers |
