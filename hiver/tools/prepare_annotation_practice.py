from hiver.paths import ROOT
from pathlib import Path
import json

BASE_DIR = ROOT
SOURCE = BASE_DIR / "data" / "exploration" / "spotify_examples.jsonl"
OUTPUT = BASE_DIR / "data" / "exploration" / "annotation_practice.json"

# A mix of straightforward and ambiguous exploration examples.
SELECTED_IDS = [
    "262672",
    "453442",
    "1944119",
    "1151239",
    "1790161",
    "1973063",
    "1995442",
    "1989324",
    "1772026",
    "2368012",
]


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Run inspect_spotify.py first: {SOURCE}")

    # Protect any annotation work from an accidental rerun.
    if OUTPUT.exists():
        raise FileExistsError(
            f"Already exists: {OUTPUT}\n"
            "Open that file to continue your annotations."
        )

    examples = {}
    with SOURCE.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                examples[item["customer_tweet_id"]] = item

    missing = [item_id for item_id in SELECTED_IDS if item_id not in examples]
    if missing:
        raise ValueError(f"Expected exploration examples missing: {missing}")

    practice = []
    for item_id in SELECTED_IDS:
        example = examples[item_id]
        practice.append({
            "customer_tweet_id": item_id,
            "customer_text": example["customer_text"],
            "intent": "",
            "expected_action": "",
            "action_reason": "",
            "required_reply_points": [],
            "forbidden_claims": [],
            "uncertain": False,
            "annotation_note": "",
        })

    # Historical replies are deliberately omitted so you make your
    # own decision from the information available to the agent.
    with OUTPUT.open("x", encoding="utf-8") as destination:
        json.dump(practice, destination, ensure_ascii=False, indent=2)

    print(f"Created {len(practice)} practice examples.")
    print(f"Open and edit:\n{OUTPUT}")


if __name__ == "__main__":
    main()