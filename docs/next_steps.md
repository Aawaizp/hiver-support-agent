# Submission next steps

The test comparison, 20-reply human/judge study and report are complete.

1. Read `docs/submission_checklist.md` and resolve the annotation/runtime requirements.
2. Verify local evidence with `py run.py verify-results`; read the report yourself.
3. Use the curated submission archive for a repository; confirm reviewer access.
4. Submit the repo link and report through the assignment form.

## Development-only improvement candidates

The keyword router scans customer and support context without distinguishing resolved
issues. Broad patterns such as `password reset`, `already tried` and `doesn't work`
can unnecessarily escalate safe questions. A future development experiment should
distinguish current unresolved customer issues from old/support text, then measure
both escalation recall and false escalation rate.

The generator sees only the last four context turns and one historical case. This can
lose issue context. Investigate development errors before changing those limits, since
more context increases latency. Historical cases must not become facts about the
current customer. Invalid output handling and private-information requests also need
reply-quality evaluation.

No accuracy improvement is claimed by this setup/evaluation cleanup. Agent behavior
is frozen so saved test metrics describe the same version. Later behavior changes
are a new experiment; the inspected test set is not fresh unbiased validation for tuning.
