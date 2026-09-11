from hiver.paths import ROOT
from pathlib import Path
import json

DATA = ROOT / "data/prepared"
SOURCE = DATA / "retrieval_conversations.jsonl"
OUTPUT = DATA / "retrieval_conversations_clean.jsonl"
AUDIT = DATA / "duplicate_review.json"

# Decisions made by inspecting duplicate_report.json.
EXCLUDE = {
    "2863": "Duplicate promotional text also appears in test context.",
    "2598638": "Near-duplicate album availability question appears in test.",
}


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError(
            "Cleaned output or audit already exists. Existing files protected."
        )

    kept = []
    removed = []

    with SOURCE.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            group = record["conversation_id"]

            if group in EXCLUDE:
                removed.append(group)
            else:
                kept.append(record)

    if set(removed) != set(EXCLUDE):
        raise ValueError(
            f"Expected to remove {sorted(EXCLUDE)}, found {sorted(removed)}."
        )

    with OUTPUT.open("x", encoding="utf-8") as file:
        for record in kept:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    audit = {
        "flagged_pairs_reviewed": 52,
        "generic_message_pairs_retained": 50,
        "excluded_retrieval_conversations": EXCLUDE,
        "retrieval_conversations_remaining": len(kept),
        "evaluation_examples_changed": 0,
        "decision": (
            "Retain common acknowledgements. Conservatively exclude "
            "retrieval conversations with substantive text overlap."
        ),
        "limitation": (
            "This lexical similarity check cannot rule out every "
            "semantic duplicate or other source of leakage."
        ),
    }

    with AUDIT.open("x", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)

    print(f"Removed: {len(removed)} conversations")
    print(f"Remaining: {len(kept)} conversations")
    print(f"Use this retrieval file from now on:\n{OUTPUT}")


if __name__ == "__main__":
    main()