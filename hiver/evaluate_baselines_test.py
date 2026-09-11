"""Score fixed development-trained baselines on the held-out test set."""
import hashlib
import json
import time

from hiver.paths import ROOT
from hiver.baselines import Baselines, INTENTS, load_development
from hiver.evaluate_baselines import calculate_metrics


def validate_splits(development, test):
    for split, rows, expected in (("dev", development, 50), ("test", test, 150)):
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} {split} examples.")
        for key in ("customer_tweet_id", "conversation_id"):
            if len({str(row[key]) for row in rows}) != len(rows):
                raise ValueError(f"Duplicate {key} in {split}.")
        for row in rows:
            if row["split"] != split or row["intent"] not in INTENTS:
                raise ValueError(f"Invalid split or intent in {split}.")
            if row["expected_action"] not in {"auto_handle", "escalate"}:
                raise ValueError(f"Invalid action in {split}.")
    for key in ("customer_tweet_id", "conversation_id"):
        if {str(x[key]) for x in development} & {str(x[key]) for x in test}:
            raise ValueError(f"Development/test overlap: {key}.")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    development = load_development()
    test = []
    for path in sorted((ROOT / "data/prepared").glob("test_batch_*.json")):
        test.extend(json.loads(path.read_text(encoding="utf-8")))
    validate_splits(development, test)
    started = time.perf_counter()
    # Test labels are used only for scoring, never for fitting or prediction.
    models = Baselines(development)
    predictions = {"trivial": [], "simple": []}
    for example in test:
        for name in predictions:
            prediction = getattr(models, name)(example["customer_text"], example["prior_context"])
            predictions[name].append({
                "customer_tweet_id": example["customer_tweet_id"],
                "conversation_id": example["conversation_id"],
                "customer_text": example["customer_text"],
                "expected_intent": example["intent"],
                "expected_action": example["expected_action"],
                "label_uncertain": example["uncertain"],
                "prediction": prediction,
            })
    digest = hashlib.sha256(json.dumps(
        {"development": development, "test": test}, sort_keys=True,
    ).encode("utf-8"))
    for name in ("baselines.py", "retrieval.py", "evaluate_baselines.py", "evaluate_baselines_test.py"):
        digest.update((ROOT / "hiver" / name).read_bytes())
    folder = ROOT / "results/baselines_test" / digest.hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": True,
        "evaluation": "Held-out test; both baselines trained on 50 development examples",
        "examples": len(test),
        "label_basis": "AI-assisted annotations",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metrics": {name: calculate_metrics(rows) for name, rows in predictions.items()},
        "limitations": ["Agreement with supplied labels, not verified real-world accuracy.",
                        "Routing metrics do not evaluate reply quality."],
    }
    for name, rows in predictions.items():
        write_json(folder / f"{name}_predictions.json", rows)
    write_json(folder / "metrics.json", report)
    print(json.dumps(report, indent=2))
    print(f"Saved: {folder}")


if __name__ == "__main__":
    main()
