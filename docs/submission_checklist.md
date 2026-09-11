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
2. Fresh full agent inference exceeded 15 minutes on this computer. Saved-results
   scoring is quick, but must not be described as fresh inference. Confirm the expected
   reproduction scope or plan a genuinely faster independently measured setup.
3. Publish the curated contents to a repository and confirm the reviewer can access it.
   No repository or remote has been created by this workflow. Do not publish `.env`.
4. Read the report yourself and be ready to explain/modify the code live. If exporting
   the report, keep it within six pages; alternatively include it as a README section,
   as permitted by the assignment. Current Markdown has not been paginated as a PDF.
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
