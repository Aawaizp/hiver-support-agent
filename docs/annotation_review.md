# Annotation review update

The project author confirms manually reviewing and correcting all 200 intent/action
annotations, beyond the separate 20 reply-quality ratings. This confirmation was given
after the saved test evaluation and judge comparison.

Current description: **AI-assisted annotations, manually reviewed and corrected.**

This records the author's confirmation; it is not an independent audit of how the
review was performed or an assertion that labels were originally written without AI.
The initial annotation-method JSON remains as historical drafting provenance.
Run `py run.py verify-results` to detect differences between current test labels and
the labels used for saved scores. If labels differ, report a separate rescoring with
a correction log rather than overwriting the original evaluation history.
