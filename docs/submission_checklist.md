# Submission checklist

## Completed evidence

- Runnable local agent, retrieval, and two baselines.
- 200 annotated examples: 50 development and 150 test, with sampling notes.
- Frozen 150-example test for all three systems, including one generation failure.
- 20 human-scored agent replies, frozen before judge execution; judge agreement and disagreements.
- Report with five observed failure modes, misleading-headline discussion and next-week plan.
- Fifteen-item decision log and offline metric verification.

## Required checks before sending

1. The author confirms manual review and correction of all 200 annotations, in addition
   to the separate 20 reply-quality ratings. Describe the dataset as AI-assisted,
   manually reviewed and corrected. Keep the original drafting provenance; confirm
   that this method meets the reported revised assignment allowance.
2. Saved-results verification passed in 13.77 seconds with dependencies installed.
   Fresh full agent inference exceeds 15 minutes. The README distinguishes these paths;
   acceptance of saved-score reproduction is not guaranteed. No clean-install timing is claimed.
3. The repository is published at https://github.com/Aawaizp/hiver-support-agent.
   The author should confirm reviewer access and submit the link. Never publish `.env`.
4. The full report is now included as a README section, as permitted by the assignment.
   Read it and be ready to explain/modify the code live. No PDF export is required for
   this format; any optional PDF must respect the six-page limit.
5. Submit the repo link and report through the assignment form; do not email it.
   Verify the form and submission deadline with the original invitation.

## Reviewer commands

```powershell
py -m pip install -r requirements.txt
py run.py verify-results
py -m unittest discover -s tests -v
py run.py eval-baselines-test
```

These use local files and do not call a model. For live inference, install/start Ollama,
pull `qwen3:4b`, then use the README demo command. No clean-machine installation timing
is claimed. Keep the final scores frozen; further changes require new evaluation.
