from hiver.paths import ROOT
import html
import json
from pathlib import Path

DATA = ROOT / "data/prepared"


def clean(text):
    return html.unescape(text).strip()


def main():
    examples = {}

    for pattern in ("dev_batch_*.json", "test_batch_*.json"):
        for path in sorted(DATA.glob(pattern)):
            for item in json.loads(path.read_text(encoding="utf-8")):
                examples[str(item["customer_tweet_id"])] = (path.name, item)

    if not examples:
        print(f"No annotation files found in:\n{DATA}")
        return

    print(f"Loaded {len(examples)} examples.")
    print("Enter a customer_tweet_id. Type q to exit.")

    while True:
        entered = input("\nTweet ID: ").strip()

        if entered.lower() == "q":
            break

        if entered not in examples:
            print("ID not found. Use customer_tweet_id from your batch file.")
            continue

        filename, item = examples[entered]

        print("\n" + "=" * 65)
        print(f"FILE: {filename}")
        print(f"TWEET ID: {entered}")
        print("=" * 65)

        print("\nEARLIER CONVERSATION\n")

        if not item["prior_context"]:
            print("(No earlier messages available.)")
        else:
            for turn in item["prior_context"]:
                print(f"{turn['role'].upper()}:")
                print(clean(turn["text"]))
                print()

        if item.get("linked_context_incomplete"):
            print("[Some linked earlier messages are missing.]")

        print("\n" + "-" * 65)
        print("CUSTOMER MESSAGE TO LABEL:\n")
        print(clean(item["customer_text"]))
        print("-" * 65)


if __name__ == "__main__":
    main()