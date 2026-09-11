from hiver.paths import ROOT
from pathlib import Path
import json

import pandas as pd

# Change this to your extracted CSV location.
CSV_PATH = ROOT / "data/raw/twcs.csv"

BRAND = "SpotifyCares"
SAMPLE_SIZE = 50
SEED = 42

OUTPUT_DIR = ROOT / "data" / "exploration"

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]


def read_chunks():
    return pd.read_csv(
        CSV_PATH,
        usecols=COLUMNS,
        dtype="string",
        keep_default_na=False,
        chunksize=100_000,
    )


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Update CSV_PATH at the top of this script."
        )

    # First pass: collect replies written by Spotify.
    print("Pass 1/2: Finding Spotify support replies...")
    support_parts = []

    for chunk in read_chunks():
        inbound = chunk["inbound"].str.strip().str.lower()
        mask = (
            chunk["author_id"].eq(BRAND)
            & inbound.eq("false")
            & chunk["in_response_to_tweet_id"].ne("")
        )
        support_parts.append(chunk.loc[mask].copy())

    support = pd.concat(support_parts, ignore_index=True)
    support = support.drop_duplicates("tweet_id")

    if support.empty:
        raise ValueError("No Spotify replies found. Check the dataset.")

    parent_ids = set(support["in_response_to_tweet_id"])

    # Second pass: find the customer messages those replies answer.
    print("Pass 2/2: Finding matching customer messages...")
    customer_parts = []

    for chunk in read_chunks():
        inbound = chunk["inbound"].str.strip().str.lower()
        mask = (
            chunk["tweet_id"].isin(parent_ids)
            & inbound.eq("true")
            & chunk["text"].str.strip().ne("")
        )
        customer_parts.append(chunk.loc[mask].copy())

    customers = pd.concat(customer_parts, ignore_index=True)
    customers = customers.drop_duplicates("tweet_id")

    # For this first inspection, use conversation-opening messages.
    # Follow-up messages may need earlier context to make sense.
    customers = customers.loc[
        customers["in_response_to_tweet_id"].eq("")
    ]

    if customers.empty:
        raise ValueError("No matching conversation-opening messages found.")

    sample = customers.sample(
        n=min(SAMPLE_SIZE, len(customers)),
        random_state=SEED,
    )

    support["_time"] = pd.to_datetime(
        support["created_at"], errors="coerce", utc=True
    )
    support = support.sort_values(
        ["_time", "tweet_id"], na_position="last"
    )

    replies_by_parent = {
        parent_id: group
        for parent_id, group in support.groupby(
            "in_response_to_tweet_id", sort=False
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "spotify_examples.jsonl"
    text_path = OUTPUT_DIR / "spotify_examples.txt"

    with (
        json_path.open("w", encoding="utf-8") as json_file,
        text_path.open("w", encoding="utf-8") as text_file,
    ):
        for number, (_, customer) in enumerate(sample.iterrows(), start=1):
            replies = replies_by_parent[customer["tweet_id"]]

            record = {
                "customer_tweet_id": customer["tweet_id"],
                "customer_text": customer["text"],
                "support_replies": [
                    {
                        "tweet_id": reply["tweet_id"],
                        "text": reply["text"],
                    }
                    for _, reply in replies.iterrows()
                ],
                "purpose": "exploration_only",
            }

            json_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            text_file.write(
                f"{'=' * 60}\n"
                f"EXAMPLE {number} | ID: {customer['tweet_id']}\n\n"
                f"CUSTOMER:\n{customer['text']}\n\n"
            )

            for reply in record["support_replies"]:
                text_file.write(f"SPOTIFY:\n{reply['text']}\n\n")

    print(f"\nFound {len(customers):,} matching opening messages.")
    print(f"Saved {len(sample)} exploration examples.")
    print(f"\nRead this file:\n{text_path}")
    print(f"\nStructured copy:\n{json_path}")


if __name__ == "__main__":
    main()