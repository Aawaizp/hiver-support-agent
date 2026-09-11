from hiver.paths import ROOT
from pathlib import Path
import html
import json
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = ROOT
DATA = BASE / "data/prepared"
EXPLORATION = BASE / "data/exploration/spotify_examples.jsonl"
REPORT = DATA / "duplicate_report.json"
THRESHOLD = 0.90


def normalize(text):
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return " ".join(re.findall(r"\w+", text))


def main():
    records = []
    seen = set()

    def add(split, group, tweet_id, text):
        normalized = normalize(text)
        if not normalized:
            return

        key = (split, str(group), normalized)
        if key in seen:
            return
        seen.add(key)

        records.append({
            "split": split,
            "conversation_id": str(group),
            "tweet_id": str(tweet_id),
            "text": text,
            "normalized": normalized,
        })

    example_count = {"dev": 0, "test": 0}

    for split in ("dev", "test"):
        for path in sorted(DATA.glob(f"{split}_batch_*.json")):
            for item in json.loads(path.read_text(encoding="utf-8")):
                example_count[split] += 1
                group = item["conversation_id"]

                add(
                    split, group,
                    item["customer_tweet_id"], item["customer_text"],
                )

                # Earlier customer messages can also reveal overlap.
                for turn in item["prior_context"]:
                    if turn["role"] == "customer":
                        add(split, group, turn["tweet_id"], turn["text"])

    if example_count != {"dev": 50, "test": 150}:
        raise ValueError(f"Unexpected example counts: {example_count}")

    with (DATA / "retrieval_conversations.jsonl").open(
        encoding="utf-8"
    ) as file:
        for line in file:
            item = json.loads(line)
            for turn in item["tweets"]:
                if turn["inbound"]:
                    add(
                        "retrieval", item["conversation_id"],
                        turn["tweet_id"], turn["text"],
                    )

    with EXPLORATION.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            add(
                "exploration", item["customer_tweet_id"],
                item["customer_tweet_id"], item["customer_text"],
            )

    print(f"Checking {len(records):,} customer texts...")

    # Character patterns catch minor spelling and wording changes.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(
        item["normalized"] for item in records
    )

    evaluation_indices = [
        index for index, item in enumerate(records)
        if item["split"] in {"dev", "test"}
    ]

    matches = []
    seen_pairs = set()

    for index in evaluation_indices:
        source = records[index]
        similarities = (matrix[index] @ matrix.T).tocsr()

        for other_index, score in zip(
            similarities.indices, similarities.data
        ):
            other_index = int(other_index)
            target = records[other_index]

            if source["split"] == target["split"]:
                continue

            pair = tuple(sorted((index, other_index)))
            if pair in seen_pairs:
                continue

            exact = source["normalized"] == target["normalized"]

            # Short generic messages are compared exactly only.
            near = (
                float(score) >= THRESHOLD
                and min(
                    len(source["normalized"]),
                    len(target["normalized"]),
                ) >= 30
            )
            if not (exact or near):
                continue

            seen_pairs.add(pair)
            matches.append({
                "match_type": "normalized_exact" if exact else "near",
                "similarity": round(float(score), 4),
                "left": {
                    key: value for key, value in source.items()
                    if key != "normalized"
                },
                "right": {
                    key: value for key, value in target.items()
                    if key != "normalized"
                },
            })

    matches.sort(key=lambda item: -item["similarity"])

    comparison_counts = {}
    affected_evaluation = set()
    affected_retrieval = set()

    for match in matches:
        sides = [match["left"], match["right"]]
        comparison = " / ".join(sorted(side["split"] for side in sides))
        comparison_counts[comparison] = (
            comparison_counts.get(comparison, 0) + 1
        )

        for side in sides:
            if side["split"] in {"dev", "test"}:
                affected_evaluation.add(
                    (side["split"], side["conversation_id"])
                )
            elif side["split"] == "retrieval":
                affected_retrieval.add(side["conversation_id"])

    summary = {
        "examples": example_count,
        "customer_texts_checked": len(records),
        "similarity_threshold": THRESHOLD,
        "flagged_pairs": len(matches),
        "pairs_by_split": comparison_counts,
        "affected_evaluation_conversations": len(affected_evaluation),
        "affected_retrieval_conversations": len(affected_retrieval),
    }

    REPORT.write_text(
        json.dumps(
            {"summary": summary, "matches": matches},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nFull report: {REPORT}")
    print("Original examples and retrieval data were not changed.")


if __name__ == "__main__":
    main()
