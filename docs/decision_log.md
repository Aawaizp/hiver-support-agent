# Decision log

1. Use SpotifyCares only: keep the support domain coherent while retaining varied requests.
2. Use nine intents: cover recurring issue types without a large sparse taxonomy.
3. Label one message per conversation: reduce repeated-thread weighting in evaluation.
4. Separate 50 development and 150 test conversations: support tuning without training on test labels.
5. Exclude exploration conversations: avoid evaluating the initial familiarization examples.
6. Use a random split: practical for the prototype; explicitly forgo claims of temporal generalization.
7. Remove substantive retrieval overlaps but retain generic acknowledgements: reduce leakage without deleting ordinary support language.
8. Use TF-IDF retrieval: low cost, offline operation and traceable case IDs; semantic recall is limited.
9. Switch from quota-limited Gemini experiments to local Qwen3:4b: allow completion without paid inference; accept slow hardware-dependent runtime.
10. Limit generation to one historical case and four context turns: reduce prompt size, accepting possible context loss.
11. Combine structured decisions with Python escalation checks: make some routing explicit; broad checks can over-escalate.
12. Compare with majority/escalate and nearest-neighbor baselines: measure gains over trivial and inexpensive alternatives.
13. Fingerprint code/data/model name and preserve old results: prevent silent mixing of different experiments. This does not pin the installed model digest or all dependencies.
14. Record invalid generations rather than silently discarding them: preserve the 150-example denominator and disclose operational failure.
15. Freeze the agent before reporting test results: use later inspection for failure analysis, not further tuning; keep human/judge agreement pending until actually measured.
