# Spotify support agent: evaluation report

**Status: evaluation and reviewer/judge comparison complete; submission checks remain.**

## 1. Problem framing

The prototype classifies SpotifyCares requests, drafts historically grounded replies,
and recommends automatic handling or human review. Good behavior means understanding
the current issue, providing a useful supported response, and identifying cases needing
human access or investigation. Automatic handling means a response can be drafted;
it does not mean the customer's issue was resolved.

The system cannot access Spotify accounts, send DMs, issue refunds, correct catalog
entries, inspect linked images, or verify current service status. It is an offline
research prototype, not ready for unattended customer-facing deployment.

## 2. Data and system

Source: [Customer Support on Twitter, thoughtvector](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
SpotifyCares was selected as one brand with varied support intents. Sampling used seed
42: choose conversations uniformly, then one eligible customer message per conversation.
Of 28,171 eligible conversations, 50 exploration groups were excluded; the evaluation
split contains 50 development and 150 test examples. These are conversation-disjoint
random splits, not chronological splits. Only messages with a direct brand response
were eligible. No language filter was applied.

The historical retrieval pool began with 3,000 conversations. Lexical duplicate review
flagged 52 pairs; 50 generic pairs were retained and two substantive-overlap retrieval
conversations removed. The remaining 2,998 conversations yield 4,469 historical cases.
Evaluation conversations are excluded from retrieval. This does not rule out semantic
duplicates or model pretraining exposure to public data.

The 200 annotations are AI-assisted, manually reviewed and corrected by the project
author, as confirmed by the author after the evaluation. Labels cover intent,
handling, rationale, required/forbidden reply points
and uncertainty; the annotation record marks 16 examples uncertain. Routing metrics
use intent/action labels. This is author review, not independent second-reviewer
validation. Initial annotation provenance is retained alongside the later review record.

Nine intents cover account access/security, billing, subscription plans, playback/offline,
app/device issues, music/library/catalog, product feedback, praise/thanks and other/unclear.

The pipeline retrieves a TF-IDF word-ngram match using the current message and recent
customer context. Qwen3:4b runs locally through Ollama with one retrieved case, its first
support reply, and the last four context turns. Thinking is disabled; temperature is 0,
context limit 4,096 and output limit 300 tokens. Pydantic validates structured output.
Python applies escalation checks to model flags and keyword patterns. Citations outside
retrieved case IDs are removed. Valid IDs alone do not establish factual grounding.

## 3. Baselines and final test results

Both baselines fit only the 50 development examples. The trivial baseline predicts the
most common development intent and always escalates. The simple baseline uses character
3-5-gram TF-IDF nearest-neighbor labels and fixed reply templates. Neither uses an LLM.
Development comparisons used leave-one-conversation-out fitting. The table below uses
the same 150 held-out test inputs for all three systems.

| Metric | Trivial | Simple | Agent |
|---|---:|---:|---:|
| Intent accuracy | 14.00% | 32.00% | 58.00% |
| Intent macro F1 | 0.0273 | 0.2794 | 0.5796 |
| Action accuracy | 32.00% | 64.00% | 66.67% |
| Escalation recall | 100.00% | 62.50% | 79.17% |
| Auto-handle coverage | 0.00% | 56.00% | 48.00% |
| Unsafe auto-handle rate | Undefined | 21.43% | 13.89% |
| Required escalations caught | 48/48 | 30/48 | 38/48 |
| Incorrectly auto-handled | 0 | 18 | 10 |

Accuracy is correct predictions divided by all 150 inputs. Macro F1 averages across
the nine intents. Escalation recall is required escalations caught / 48. Coverage is
auto-handled / 150. Unsafe auto-handle rate is labelled escalations among auto-handled
examples; it is undefined when nothing is auto-handled.

The agent has 87 correct intent predictions and 100 correct actions. It auto-handled
72 examples, escalated 77, and produced one invalid output. Of its escalations, 39 were
labelled auto-handle. The simple baseline has 96 correct actions: the agent improves
action accuracy by just four examples, while catching eight more required escalations.
This is a safety/coverage trade-off, not a large across-the-board improvement.

One output failed JSON validation with a truncated string. The first 89 successful
outputs were preserved; recovery recorded generation failures and continued. The final
failure is counted as an incorrect intent and action, neither successful escalation
nor auto-handling. Failure rate is 1/150 (0.67%). Earlier failed attempts are not counted
as additional examples. No successful-only accuracy is substituted for full-set scores.

## 4. Top five observed failure modes

These are qualitative diagnostic examples, not independent human rubric ratings. IDs
locate the original rows in the frozen test predictions. Hypotheses need development
experiments; the test outputs were not used for further agent tuning.

| Failure and real example | Observation | Hypothesis / next experiment |
|---|---|---|
| Resolved issue still escalated: 2210891 | Customer says earlier steps failed but it was fixed that morning. Router still cites failed troubleshooting. | Keyword matches ignore resolution and time; evaluate unresolved-state checks on development examples. |
| Historical case contaminates current request: 1715187 | Customer asks about missing content; reply refers to payment updates. | Weak lexical retrieval distracts the small model; compare relevance filtering on development data. |
| Unsupported capability/private-data request: 1639591 | French paid-but-still-free complaint receives an English request for account email/username via DM to check status. | Old support practices override prototype limits; separately evaluate reply constraints and language handling. |
| Missed catalog correction and repeated question: 223010 | Customer names two swapped songs; agent asks which songs and auto-handles. | Handling classification and context comprehension fail despite a catalog intent; evaluate correction requests versus availability questions. |
| Invalid structured output: 1163234 | Link-only input with no context yields a truncated JSON string containing repeated zeros. | Output degeneration/limit interaction; retain failure accounting and test bounded repair or explicit abstention on development data. |

Escalation is not enough to make a reply safe: a model can escalate while requesting
unnecessary account identifiers or claiming capabilities it does not have.

## 5. What is misleading about my headline number?

"58% intent accuracy" measures agreement with author-reviewed, AI-assisted labels on
150 historical examples, not independently validated correctness or resolution. Label ambiguity, languages,
random rather than temporal splitting and public-data exposure limit interpretation.
No confidence interval or statistical significance claim is made.

"79% escalation recall" still misses ten of 48 labelled escalations and sends 39
labelled auto-handle cases to human review. An always-escalate system achieves 100%
recall with zero automation. The 13.89% unsafe auto-handle rate measures routing only:
it does not count every unsupported claim or harmful reply in other routes.

Speed is another limitation. The user observed approximately 55 minutes for the first
89 examples, about 37 seconds per example including workflow overhead. This is not an
instrumented benchmark. Recalculating saved scores is quick; generating every reply
again is not. Fresh full inference has not met the assignment's 15-minute target.

## 6. Reply judge and human agreement

The existing rubric scores relevance, grounding, usefulness and safety from 0 to 2,
plus a critical-error flag and explanation. A Groq-backed judge receives message,
context, historical evidence, reply and action, without intent labels or model reasoning.

A fixed seed-42 sample of 20 successful frozen agent replies was scored by the project
reviewer before the Groq judge run. It excludes the single generation failure and
retains the generator's four-turn context and first historical-reply limits. This
sample is small and does not estimate end-to-end success. Earlier baseline/demo reviews
are separate evidence and must not be substituted for agent agreement.

The judge model was `qwen/qwen3.8-27b`. Ratings matched exactly on 54/80 dimension
scores (67.5%). The 80 scores are grouped within 20 replies, not 80 independent samples.

| Dimension | Human mean / 2 | Judge mean / 2 | Exact agreement | Linear weighted kappa |
|---|---:|---:|---:|---:|
| Relevance | 1.55 | 1.65 | 60% | 0.231 |
| Grounding | 1.35 | 1.65 | 70% | 0.592 |
| Usefulness | 1.35 | 1.30 | 45% | 0.052 |
| Safety | 1.90 | 1.95 | 95% | 0.643 |

Neither rated any sampled reply as a critical error. Exact agreement on that flag
is 100%, but kappa is undefined and stored as null. All-negative agreement provides
no evidence of sensitivity to critical errors. High safety agreement is not proof of safety.

Usefulness agreement is weak. In review-17 the human accepts referral for disputed
charges, while the judge penalizes not requesting account details as old cases did.
Review-09 similarly treats account identifiers as required. These rationales blur
historical precedent with the offline prototype's actual capabilities. In review-13
the judge calls an invented email address a material unsupported detail but gives
grounding=1, although the rubric's description points to 0. Structured JSON validation
does not detect that semantic inconsistency.

Human scores also need scrutiny: review-02 says the agent activated access, although
the reply only says access is active and the customer reports gaining access. The
assistant raised this interpretation before the final ratings were frozen; the
reviewer retained the explanation. Original human/judge scores are preserved, not
edited to improve agreement. The reviewer is part of the project, had prior exposure
to some outputs, and received that feedback. This is not independent blind validation.
Generator and judge also share the Qwen family. The judge is useful diagnostic evidence,
but these results do not justify using it as the sole quality gate.

Evidence: `results/agent_review/a3b4be5445390607/judge/b0bd4a688c636ff2/` contains the
frozen human review, judge outputs and agreement report. No human scores were sent to
the judge. Future rubric calibration needs a separate sample, not rescoring this one
until agreement increases.

## 7. One more week

Prioritize independent label review and a new evaluation set before further tuning.
Then measure context-aware routing against broad keyword rules, validate historical
evidence relevance, and test bounded invalid-output handling. Instrument model load,
retrieval, prompt processing and generation time before changing performance settings.
Evaluate faster configurations on development data and preserve failure/coverage metrics.
Complete a blinded reply-quality study and assess whether any slice is safe to automate.

## 8. Reproduction and remaining submission work

`py run.py verify-results` recalculates all three score tables from saved predictions,
checks test input/label alignment and compares against saved metrics. It performs no
fresh LLM inference. `py run.py eval-baselines-test` freshly evaluates both baselines.
See the README for full inference and recovery commands.

Evidence: `results/agent_test/e14347f8f9f185d629c453a8c973ff168d6183da3eb2af7605772c106b96e4d8/`
and `results/baselines_test/7b2d1d92e828737cea676cb483158c2898620841601e337f303b73ab2d46531c/`.
Use the agent's `recovery_*` files, not its partial original predictions.

Before submission: resolve the fresh-inference runtime requirement, confirm the
reported revised annotation allowance, and check repo access.
The supplied PDF requests hand-labelled examples; the reported revised AI allowance
has not been independently verified. Submit the repo and report through the assignment
form, not email. AI assistance was used for code and annotations. See `decision_log.md`.
