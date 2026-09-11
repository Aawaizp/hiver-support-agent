from hiver.paths import ROOT
from collections import defaultdict
from pathlib import Path
import argparse
import html
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = ROOT
DATA = BASE / "data/prepared"
CORPUS = DATA / "retrieval_conversations_clean.jsonl"


def clean(text):
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return " ".join(text.split())


def search_text(message, context):
    # Earlier customer messages help explain replies like "still broken".
    earlier = [
        clean(turn["text"])
        for turn in context
        if turn["role"] == "customer"
    ]
    # Repeat the current message to give it more weight.
    current = clean(message)
    return " ".join(earlier[-3:] + [current, current])


def get_context(tweet, tweets):
    context = []
    visited = {tweet["tweet_id"]}
    parent = tweet.get("parent_id")

    while parent is not None:
        parent = str(parent)
        if parent in visited or parent not in tweets:
            break

        visited.add(parent)
        previous = tweets[parent]
        context.append({
            "role": "customer" if previous["inbound"] else "support",
            "text": previous["text"],
        })
        parent = previous.get("parent_id")

    return list(reversed(context))


class Retriever:
    def __init__(self):
        if not CORPUS.is_file():
            raise FileNotFoundError(f"Missing cleaned corpus: {CORPUS}")

        # Read IDs only. Annotation labels must not enter retrieval.
        evaluation_ids = set()
        for pattern in ("dev_batch_*.json", "test_batch_*.json"):
            for path in DATA.glob(pattern):
                items = json.loads(path.read_text(encoding="utf-8"))
                evaluation_ids.update(
                    str(item["conversation_id"]) for item in items
                )

        self.cases = []

        with CORPUS.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                conversation = json.loads(line)
                group = str(conversation["conversation_id"])

                if group in evaluation_ids:
                    raise ValueError(
                        f"Data leakage: evaluation conversation {group}"
                    )

                tweets = {
                    str(tweet["tweet_id"]): tweet
                    for tweet in conversation["tweets"]
                }

                replies = defaultdict(list)

                for tweet in tweets.values():
                    parent = tweet.get("parent_id")
                    if (
                        not tweet["inbound"]
                        and tweet["author_id"] == "SpotifyCares"
                        and parent is not None
                    ):
                        replies[str(parent)].append(tweet)

                for customer_id, responses in replies.items():
                    customer = tweets.get(customer_id)
                    if not customer or not customer["inbound"]:
                        continue

                    context = get_context(customer, tweets)
                    document = search_text(customer["text"], context)
                    if not document.strip():
                        continue

                    def reply_order(reply):
                        timestamp = pd.to_datetime(
                            reply["created_at"], utc=True, errors="coerce"
                        )
                        return (
                            pd.isna(timestamp),
                            0 if pd.isna(timestamp) else timestamp.value,
                            int(reply["tweet_id"]),
                        )

                    responses.sort(key=reply_order)

                    self.cases.append({
                        "case_id": f"{group}:{customer_id}",
                        "conversation_id": group,
                        "customer_text": customer["text"],
                        "prior_context": context,
                        "historical_replies": [
                            {
                                "tweet_id": str(reply["tweet_id"]),
                                "text": reply["text"],
                            }
                            for reply in responses
                        ],
                        "_document": document,
                    })

        if not self.cases:
            raise ValueError("No usable customer/support pairs found.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50_000,
            dtype=np.float32,
        )

        self.matrix = self.vectorizer.fit_transform(
            case["_document"] for case in self.cases
        )

    def search(self, message, context=None, k=3):
        if k < 1:
            raise ValueError("k must be at least 1.")

        query = search_text(message, context or [])
        if not query.strip():
            return []

        vector = self.vectorizer.transform([query])
        if vector.nnz == 0:
            return []

        scores = (self.matrix @ vector.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")

        results = []
        selected_groups = set()

        for index in order:
            score = float(scores[index])
            if score <= 0:
                break

            case = self.cases[int(index)]
            group = case["conversation_id"]

            # Avoid using several hits from the same conversation.
            if group in selected_groups:
                continue
            selected_groups.add(group)

            result = {
                key: value
                for key, value in case.items()
                if key != "_document"
            }
            result["similarity"] = round(score, 4)
            results.append(result)

            if len(results) == k:
                break

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()

    retriever = Retriever()
    print(f"Indexed {len(retriever.cases):,} historical cases.")

    results = retriever.search(args.message)

    if not results:
        print("No matching cases found.")
        return

    for number, case in enumerate(results, start=1):
        print("\n" + "=" * 65)
        print(
            f"MATCH {number} | {case['case_id']} "
            f"| similarity: {case['similarity']}"
        )
        print("\nPAST CUSTOMER:")
        print(html.unescape(case["customer_text"]))

        print("\nPAST SPOTIFY REPLIES:")
        for reply in case["historical_replies"]:
            print(html.unescape(reply["text"]))
            print()


if __name__ == "__main__":
    main()