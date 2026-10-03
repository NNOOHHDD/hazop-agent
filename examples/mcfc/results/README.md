# Demo results

This folder is for the outputs of the demo in the main README: three agent runs and one evaluation report.

```bash
python -m hazop_agent --pid examples/mcfc/pid.jpg --node examples/mcfc/node.md \
    --runs 3 --out examples/mcfc/results/agent
python -m hazop_eval --reference examples/mcfc/reference_worksheet.csv \
    --runs examples/mcfc/results/agent/run_* --node examples/mcfc/node.md \
    --out examples/mcfc/results/eval
```

No real-model results are committed yet. The whole flow has been checked end to end with the scripted fake LLM (`--provider fake`), whose output is placeholder text and therefore not kept here.
