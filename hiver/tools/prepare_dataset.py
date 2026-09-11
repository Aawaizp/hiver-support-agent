from hiver.paths import ROOT
from collections import defaultdict
from pathlib import Path
import json
import random

import pandas as pd

CSV_PATH = ROOT / "data/raw/twcs.csv"
BASE = ROOT
EXPLORATION = BASE / "data/exploration/spotify_examples.jsonl"
OUTPUT = BASE / "data/prepared"

BRAND = "SpotifyCares"
SEED = 42
COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at",
    "text", "in_response_to_tweet_id", "response_tweet_id",
]


def chunks():
    return pd.read_csv(
        CSV_PATH,
        usecols=COLUMNS,
        dtype="string",
        keep_default_na=False,
        chunksize=100_000,
    )


def tweet_id(value):
    value = str(value).strip()
    return int(value) if value else None


# Union-find: tweets joined by a reply link belong to one group.
parents = {}


def find(item):
    parents.setdefault(item, item)
    while parents[item] != item:
        parents[item] = parents[parents[item]]
        item = parents[item]
    return item


def union(left, right):
    left, right = find(left), find(right)
    if left != right:
        parents[max(left, right)] = min(left, right)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Update CSV_PATH: {CSV_PATH}")
    if not EXPLORATION.is_file():
        raise FileNotFoundError(f"Missing exploration file: {EXPLORATION}")
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Stop to protect existing annotations."
        )

    with EXPLORATION.open(encoding="utf-8") as file:
        exploration_ids = {
            int(json.loads(line)["customer_tweet_id"])
            for line in file if line.strip()
        }

    print("Pass 1/2: Grouping linked tweets. This may take a few minutes.")
    spotify_ids = set()

    for chunk in chunks():
        for row in chunk.to_dict("records"):
            current = tweet_id(row["tweet_id"])
            if current is None:
                raise ValueError("Found a tweet without an ID.")

            find(current)
            previous = tweet_id(row["in_response_to_tweet_id"])
            if previous is not None:
                union(current, previous)

            # Include reverse links too, including links to missing tweets.
            for reply in row["response_tweet_id"].split(","):
                linked = tweet_id(reply)
                if linked is not None:
                    union(current, linked)

            if row["author_id"] == BRAND:
                spotify_ids.add(current)

    excluded = {find(item) for item in exploration_ids}
    spotify_groups = {find(item) for item in spotify_ids} - excluded

    print("Pass 2/2: Collecting Spotify conversations...")
    conversations = defaultdict(dict)

    for chunk in chunks():
        for row in chunk.to_dict("records"):
            current = tweet_id(row["tweet_id"])
            group = find(current)
            if group not in spotify_groups:
                continue

            inbound = row["inbound"].strip().lower()
            if inbound not in {"true", "false"}:
                raise ValueError(f"Invalid inbound value for tweet {current}")

            conversations[group][current] = {
                "tweet_id": str(current),
                "author_id": row["author_id"],
                "inbound": inbound == "true",
                "created_at": row["created_at"],
                "text": row["text"],
                "parent_id": tweet_id(row["in_response_to_tweet_id"]),
            }

    # The graph is no longer needed; release its memory.
    parents.clear()

    # Keep conversations involving Spotify as the only support brand.
    # Choose customer messages with a direct Spotify response.
    candidates = {}
    for group, tweets in conversations.items():
        brands = {
            item["author_id"]
            for item in tweets.values() if not item["inbound"]
        }
        if brands != {BRAND}:
            continue

        customer_ids = set()
        for reply in tweets.values():
            parent = reply["parent_id"]
            if (
                reply["author_id"] == BRAND
                and not reply["inbound"]
                and parent in tweets
                and tweets[parent]["inbound"]
                and tweets[parent]["text"].strip()
            ):
                customer_ids.add(parent)

        if customer_ids:
            candidates[group] = sorted(customer_ids)

    if len(candidates) < 200:
        raise ValueError(
            f"Only {len(candidates)} eligible conversations; need 200."
        )

    rng = random.Random(SEED)
    selected_groups = rng.sample(sorted(candidates), 200)
    selected_set = set(selected_groups)

    def make_example(group, split):
        tweets = conversations[group]
        current = rng.choice(candidates[group])
        customer = tweets[current]

        # Follow ancestors only: sibling branches and future replies
        # must never become context for this customer message.
        context = []
        visited = {current}
        previous = customer["parent_id"]
        incomplete = False

        while previous is not None:
            if previous in visited or previous not in tweets:
                incomplete = True
                break
            visited.add(previous)
            ancestor = tweets[previous]
            context.append({
                "tweet_id": ancestor["tweet_id"],
                "role": "customer" if ancestor["inbound"] else "support",
                "text": ancestor["text"],
            })
            previous = ancestor["parent_id"]

        context.reverse()

        return {
            "conversation_id": str(group),
            "customer_tweet_id": str(current),
            "split": split,
            "prior_context": context,
            "linked_context_incomplete": incomplete,
            "customer_text": customer["text"],
            "intent": "",
            "expected_action": "",
            "action_reason": "",
            "required_reply_points": [],
            "forbidden_claims": [],
            "uncertain": False,
            "annotation_note": "",
        }

    examples = [
        make_example(group, "dev" if index < 50 else "test")
        for index, group in enumerate(selected_groups)
    ]

    retrieval_groups = sorted(set(candidates) - selected_set)
    retrieval_groups = rng.sample(
        retrieval_groups, min(3000, len(retrieval_groups))
    )

    assert not selected_set.intersection(retrieval_groups)
    assert not selected_set.intersection(excluded)
    assert len({item["conversation_id"] for item in examples}) == 200

    OUTPUT.mkdir(parents=True)

    # Keep development and test examples in separate annotation files.
    for split in ("dev", "test"):
        subset = [item for item in examples if item["split"] == split]
        for start in range(0, len(subset), 20):
            batch_number = start // 20 + 1
            write_json(
                OUTPUT / f"{split}_batch_{batch_number:02d}.json",
                subset[start:start + 20],
            )

    with (OUTPUT / "retrieval_conversations.jsonl").open(
        "x", encoding="utf-8"
    ) as file:
        for group in retrieval_groups:
            record = {
                "conversation_id": str(group),
                # Parent links preserve the actual conversation structure.
                "tweets": sorted(
                    conversations[group].values(),
                    key=lambda item: int(item["tweet_id"]),
                ),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "seed": SEED,
        "brand": BRAND,
        "eligible_conversations": len(candidates),
        "exploration_groups_excluded": len(excluded),
        "development_examples": 50,
        "test_examples": 150,
        "retrieval_conversations": len(retrieval_groups),
        "sampling": (
            "Uniform conversation sampling, then one random eligible "
            "customer message per selected conversation."
        ),
        "split_method": "Random conversation split; not a temporal split.",
        "language_filter": "None; record unsupported languages during annotation.",
        "limitations": [
            "Only conversations with a direct Spotify response are eligible.",
            "Private messages and linked images are unavailable.",
            "Historical replies are not proof of successful resolution.",
            "Near-duplicate text across splits still needs checking.",
        ],
    }
    write_json(OUTPUT / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2))
    print(f"\nFiles saved in: {OUTPUT}")
    print("Do not annotate yet: send manifest.json for the next check.")


if __name__ == "__main__":
    main()