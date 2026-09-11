# Spotify support agent

Submission report: [report](docs/report.md). Implementation choices:
[decision log](docs/decision_log.md). See [submission checklist](docs/submission_checklist.md).

Offline prototype: classify requests, retrieve historical cases, draft replies and
recommend automatic handling or human review. It cannot access accounts or send messages.

## Setup

Install and start Ollama, then run from this project directory:

```powershell
py -m pip install -r requirements.txt
ollama pull qwen3:4b
$env:OLLAMA_MODEL = "qwen3:4b"
py run.py agent "I was charged twice for my subscription"
```

The agent uses Ollama at localhost:11434, with no API key. Groq is used separately by
the reply judge. Keep `.env` private; `.env.example` describes optional API settings.
Initial model download and full evaluation can take longer than 15 minutes. Prepared
data is used directly; do not rerun preparation for the demo.

## Offline commands

```powershell
py run.py retrieve "I was charged twice for my subscription"
py run.py baselines "I was charged twice for my subscription"
py run.py eval-baselines
py run.py eval-baselines-test
py run.py verify-results
py -m unittest discover -s tests -v
```

These need no LLM calls. Development baselines use leave-one-conversation-out evaluation.
`verify-results` re-scores saved predictions; it does not rerun the agent.
Test baselines fit all 50 development examples, then score the same 150 test examples
used for the agent. Test labels never enter baseline training.

## Agent evaluation

```powershell
py run.py eval-agent --split dev --limit 50
py run.py eval-agent --split test --limit 150
```

These generate local responses and can be slow. Code changes create new result folders.
Resume with the same code, model name and data; do not mix different agent versions.

Completed frozen run:
`results/agent_test/e14347f8f9f185d629c453a8c973ff168d6183da3eb2af7605772c106b96e4d8/`.
Use `recovery_predictions.json` and `recovery_metrics.json` for the complete 150 examples.
The original `predictions.json` contains only 89. `py resume_test.py` recovers this
particular frozen run and records invalid outputs as failures. The ordinary evaluator
still stops on errors.

## Structure

- `run.py`: CLI entry point.
- `resume_test.py`: recovery for the recorded test run.
- `hiver/`: Python package: agent, retrieval, baselines, evaluation and judge.
- `hiver/tools/`: data preparation and inspection.
- `data/prepared/`: annotations and historical retrieval corpus.
- `results/`: saved experiments and metrics.
- `docs/`: rubric, status and next steps.
- `tests/`: offline checks.

The root project `Hiver` and Python package `hiver` are both needed. Backups, raw data
and API caches are ignored by Git. Old results are retained as experiment evidence.

## Current evidence

Frozen test: intent accuracy 58%, macro F1 0.5796, action accuracy 66.67%, escalation
recall 38/48 (79.17%). The agent auto-handled 72/150 examples, including 10 labelled
for escalation. One invalid generation counts as incorrect. Labels are AI-assisted
annotations, manually reviewed and corrected by the project author (author-confirmed
for all 200 examples). This is not an independent second review. Metrics measure agreement
with labels, not real-world safety or reply quality.

User-observed runtime was about 55 minutes for the first 89 examples; this is not an
instrumented latency benchmark. The 20-reply human/judge study matched 54/80 dimension
scores (67.5%); usefulness agreement was only 45%. Both flagged zero critical errors,
so critical-error sensitivity remains unknown. The report explains the disagreements.

## Reply review

```powershell
py run.py judge-baselines --prepare-only
py run.py judge-baselines --limit 5
```

This prepares a blind 20-reply development-baseline review, then uses Groq to judge.
Follow `docs/judge_rubric.md`: enter your own scores before opening judge results.
This sample does not establish judge agreement for agent test replies.
See `docs/next_steps.md` for remaining work.

The agent review is complete. Saved evidence is under
`results/agent_review/a3b4be5445390607/judge/b0bd4a688c636ff2/`.
`py run.py verify-results` also verifies this saved agreement without API calls.
`py run.py judge-agent --limit 20` uses Groq only when scores for the current review
and judge configuration have not already been saved. Do not change frozen ratings
to improve agreement.

## Submission archive

`py build_submission.py` writes `dist/Hiver-submission.zip` from an explicit file list.
It excludes `.env`, API caches, raw CSV, editor settings and old backups. The archive
contains code, prepared data, frozen test results and review evidence. It is a local
archive, not a published repository. Follow the submission checklist before uploading.

Source: Customer Support on Twitter by thoughtvector on Kaggle, CC BY-NC-SA 4.0.
Historical replies can be outdated and do not prove resolution. AI assistance was
used for implementation and annotations.
