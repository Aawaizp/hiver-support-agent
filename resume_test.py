import hashlib
import json

from pydantic import ValidationError
from hiver.agent import SupportAgent
from hiver.evaluate_agent import BASE, fingerprint, save_atomic
from hiver.evaluate_baselines import calculate_metrics

RUN_ID = "e14347f8f9f185d629c453a8c973ff168d6183da3eb2af7605772c106b96e4d8"


def main():
    examples = []
    for path in sorted((BASE / "data/prepared").glob("test_batch_*.json")):
        examples.extend(json.loads(path.read_text(encoding="utf-8")))

    if len(examples) != 150:
        raise ValueError("Expected 150 test examples.")

    agent = SupportAgent()
    try:
        data_hash = hashlib.sha256(
            json.dumps(examples, sort_keys=True, ensure_ascii=False)
            .encode("utf-8")
        ).hexdigest()
        current_id = hashlib.sha256(
            f"{fingerprint(agent.model)}:test:{data_hash}".encode("utf-8")
        ).hexdigest()

        if current_id != RUN_ID:
            raise RuntimeError(
                "Code, model name or data changed. Stop: do not mix runs."
            )

        folder = BASE / "results/agent_test" / RUN_ID
        output = folder / "recovery_predictions.json"
        source = output if output.exists() else folder / "predictions.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
        completed = {str(row["customer_tweet_id"]) for row in rows}
        print(f"Already recorded: {len(rows)}/150")

        for example in examples:
            tweet_id = str(example["customer_tweet_id"])
            if tweet_id in completed:
                continue

            print(f"Processing {tweet_id}...", flush=True)
            try:
                prediction = agent.run(
                    example["customer_text"], example["prior_context"]
                )
                if prediction.get("status") != "ok":
                    raise RuntimeError("Unexpected unsuccessful response.")
            except ValidationError:
                prediction = {
                    "status": "generation_failure",
                    "intent": "__generation_failure__",
                    "action": "__generation_failure__",
                    "reply": "",
                    "evidence_ids": [],
                    "error_type": "ValidationError",
                }
                print("Invalid output recorded as failure.", flush=True)

            rows.append({
                "customer_tweet_id": example["customer_tweet_id"],
                "conversation_id": example["conversation_id"],
                "customer_text": example["customer_text"],
                "expected_intent": example["intent"],
                "expected_action": example["expected_action"],
                "label_uncertain": example["uncertain"],
                "prediction": prediction,
            })
            save_atomic(output, rows)
            completed.add(tweet_id)
            print(f"Recorded: {len(rows)}/150", flush=True)

        failures = sum(
            row["prediction"]["status"] != "ok" for row in rows
        )
        report = {
            "run_id": RUN_ID,
            "complete": len(rows) == 150,
            "examples": len(rows),
            "generation_failures": failures,
            "generation_failure_rate": failures / len(rows),
            "metrics": calculate_metrics(rows),
            "label_basis": "AI-assisted annotations, manually reviewed and corrected (author-confirmed)",
            "failure_policy": (
                "Invalid outputs count as incorrect intent and action; "
                "they are neither auto-handled nor successfully escalated."
            ),
            "limitations": [
                "Routing metrics do not measure reply safety.",
                "Previous failed attempts are not included in the failure count.",
            ],
        }
        save_atomic(folder / "recovery_metrics.json", report)
        print(json.dumps(report, indent=2))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
    
