from hiver.paths import ROOT
from collections import Counter
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from hiver.retrieval import search_text

DATA = ROOT / "data/prepared"

INTENTS = {
    "account_access_security",
    "billing_payments",
    "subscription_plans",
    "playback_offline",
    "app_device_issues",
    "music_library_catalog",
    "product_feedback",
    "praise_thanks",
    "other_unclear",
}

REPLIES = {
    "account_access_security":
        "What happens when you try to access your account? "
        "Please describe the error without sharing your password.",
    "billing_payments":
        "Could you describe the payment problem? "
        "Please don't share card details.",
    "subscription_plans":
        "Which plan are you using, and what happens when you try?",
    "playback_offline":
        "What device are you using, and what happens during playback "
        "or downloading?",
    "app_device_issues":
        "What device and app version are you using, and what goes wrong?",
    "music_library_catalog":
        "Which song, album, or playlist do you mean, "
        "and what problem are you seeing?",
    "product_feedback":
        "Thanks for sharing your suggestion.",
    "praise_thanks":
        "Thanks for the kind words. Glad to hear from you!",
    "other_unclear":
        "Could you explain what you need help with?",
}

ESCALATION_REPLY = (
    "Please contact official Spotify support for further review. "
    "I can't investigate your account here."
)


def load_development():
    examples = []

    for path in sorted(DATA.glob("dev_batch_*.json")):
        examples.extend(json.loads(path.read_text(encoding="utf-8")))

    if len(examples) != 50:
        raise ValueError(f"Expected 50 development examples, got {len(examples)}.")

    return examples


class Baselines:
    def __init__(self, training_examples):
        if not training_examples:
            raise ValueError("Training examples cannot be empty.")

        self.examples = list(training_examples)
        seen = set()

        for example in self.examples:
            if example["split"] != "dev":
                raise ValueError("Only development examples can train baselines.")
            if example["intent"] not in INTENTS:
                raise ValueError("Missing or invalid intent.")
            if example["expected_action"] not in {"auto_handle", "escalate"}:
                raise ValueError("Missing or invalid action.")
            if example["customer_tweet_id"] in seen:
                raise ValueError("Duplicate training example.")
            seen.add(example["customer_tweet_id"])

        counts = Counter(item["intent"] for item in self.examples)

        # Alphabetical tie-breaking makes results reproducible.
        self.majority = min(counts, key=lambda label: (-counts[label], label))

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=50_000,
            dtype=np.float32,
        )

        documents = [
            search_text(item["customer_text"], item["prior_context"])
            for item in self.examples
        ]
        self.matrix = self.vectorizer.fit_transform(documents)

    def trivial(self, message, context=None):
        return {
            "status": "ok",
            "system": "trivial",
            "intent": self.majority,
            "action": "escalate",
            "reason": "Always use the most common training intent and escalate.",
            "reply": ESCALATION_REPLY,
            "evidence_ids": [],
        }

    def simple(self, message, context=None):
        query = search_text(message, context or [])
        vector = self.vectorizer.transform([query])

        if vector.nnz == 0:
            result = self.trivial(message, context)
            result["system"] = "simple"
            result["reason"] = "No matching vocabulary; use the trivial fallback."
            result["similarity"] = 0.0
            result["label_source_id"] = None
            return result

        scores = (self.matrix @ vector.T).toarray().ravel()
        best_index = int(np.argmax(scores))
        source = self.examples[best_index]

        intent = source["intent"]
        action = source["expected_action"]

        return {
            "status": "ok",
            "system": "simple",
            "intent": intent,
            "action": action,
            "reason": "Use the intent and action of the nearest training example.",
            "reply": (
                ESCALATION_REPLY
                if action == "escalate"
                else REPLIES[intent]
            ),
            # This ID explains the label choice, not historical reply evidence.
            "label_source_id": source["customer_tweet_id"],
            "similarity": round(float(scores[best_index]), 4),
            "evidence_ids": [],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()

    models = Baselines(load_development())

    for predict in (models.trivial, models.simple):
        print(json.dumps(
            predict(args.message),
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()