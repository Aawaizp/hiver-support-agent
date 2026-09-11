from hiver.paths import ROOT
import argparse
import hashlib
import json
from pathlib import Path

from hiver.agent import SupportAgent
from hiver.baselines import load_development
from hiver.evaluate_baselines import calculate_metrics

BASE = ROOT
RESULTS = BASE / "results"


def fingerprint(model):
    """Prevent mixing results from different code, data, or models."""
    digest = hashlib.sha256(model.encode("utf-8"))

    paths = [
        BASE / "hiver/evaluate_agent.py",
        BASE / "hiver/evaluate_baselines.py",
        BASE / "hiver/baselines.py",
        BASE / "hiver/paths.py",
        BASE / "hiver/agent.py",
        BASE / "hiver/retrieval.py",
        BASE / "hiver/llm_requests.py",
        BASE / "data/prepared/retrieval_conversations_clean.jsonl",
        *sorted((BASE / "data/prepared").glob("dev_batch_*.json")),
    ]

    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())

    return digest.hexdigest()


def save_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum new examples to process in this run.",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    if args.split == "dev":
        examples = load_development()
    else:
        examples = []
        for path in sorted((BASE / "data/prepared").glob("test_batch_*.json")):
            examples.extend(json.loads(path.read_text(encoding="utf-8")))

        if len(examples) != 150:
            raise ValueError(f"Expected 150 test examples, found {len(examples)}")
    agent = SupportAgent()

    try:

        data_hash = hashlib.sha256(
            json.dumps(examples, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        run_id = hashlib.sha256(
            f"{fingerprint(agent.model)}:{args.split}:{data_hash}".encode("utf-8")
        ).hexdigest()
        folder = RESULTS / f"agent_{args.split}" / run_id

        folder.mkdir(parents=True, exist_ok=True)

        prediction_path = folder / "predictions.json"
        if prediction_path.exists():
            rows = json.loads(prediction_path.read_text(encoding="utf-8"))
        else:
            rows = []

        completed = {row["customer_tweet_id"] for row in rows}
        print(f"Already saved: {len(completed)}/{len(examples)}")

        processed = 0
        stopped_error = None

        for example in examples:
            tweet_id = example["customer_tweet_id"]

            if tweet_id in completed:
                continue
            if processed >= args.limit:
                break

            print(f"Processing tweet {tweet_id}...")

            try:
                prediction = agent.run(
                    example["customer_text"],
                    example["prior_context"],
                )
                if prediction.get("status") != "ok":
                    raise RuntimeError("Agent returned an unsuccessful result.")

            except Exception as error:
                # Store the error type, not potentially sensitive error text.
                stopped_error = type(error).__name__
                print(
                    f"Stopped after {stopped_error}. "
                    "Saved predictions are safe. Check quota or run this "
                    "example separately to diagnose the failure."
                )
                break

            rows.append({
                "customer_tweet_id": tweet_id,
                "conversation_id": example["conversation_id"],
                "customer_text": example["customer_text"],
                "expected_intent": example["intent"],
                "expected_action": example["expected_action"],
                "label_uncertain": example["uncertain"],
                "prediction": prediction,
            })

            save_atomic(prediction_path, rows)
            completed.add(tweet_id)
            processed += 1
            print(f"Saved: {len(rows)}/{len(examples)}")

        report = {
            "run_id": run_id,
            "model": agent.model,
            "evaluation": f"{args.split} evaluation",
            "label_basis": "AI-drafted labels; human review not verified",
            "completed": len(rows),
            "total": len(examples),
            "complete": len(rows) == len(examples),
            "stopped_error_type": stopped_error,
            "metrics": calculate_metrics(rows) if rows else None,
            "limitations": [
                "Partial results are not comparable with full baseline results.",
                "Metrics measure agreement with supplied labels.",
                "Reply quality and human judge agreement are not measured here.",
                "Failed calls are recorded as incomplete, not successful predictions.",
            ],
        }

        save_atomic(folder / "metrics.json", report)

        print(f"\nCompleted: {len(rows)}/{len(examples)}")
        if report["complete"]:
            print(json.dumps(report["metrics"], indent=2))
        else:
            print("Evaluation is incomplete; do not report headline scores yet.")

        print(f"Results folder:\n{folder}")

    finally:
        agent.close()


if __name__ == "__main__":
    main()