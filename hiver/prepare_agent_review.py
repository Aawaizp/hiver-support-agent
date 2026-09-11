"""Prepare a fixed human review sample without generating or judging replies."""
import hashlib
import json
import random

from hiver.paths import ROOT
from hiver.retrieval import Retriever

RUN_ID = "e14347f8f9f185d629c453a8c973ff168d6183da3eb2af7605772c106b96e4d8"


def main():
    source = ROOT / "results/agent_test" / RUN_ID / "recovery_predictions.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len(rows) != 150 or len({str(x["customer_tweet_id"]) for x in rows}) != 150:
        raise ValueError("Expected 150 distinct recorded test examples.")
    inputs = {}
    for path in sorted((ROOT / "data/prepared").glob("test_batch_*.json")):
        for example in json.loads(path.read_text(encoding="utf-8")):
            inputs[str(example["customer_tweet_id"])] = example
    if set(inputs) != {str(x["customer_tweet_id"]) for x in rows}:
        raise ValueError("Test inputs and predictions do not match.")
    eligible = sorted([r for r in rows if r["prediction"]["status"] == "ok"],
                      key=lambda r: str(r["customer_tweet_id"]))
    selected = random.Random(42).sample(eligible, 20)
    cases = {c["case_id"]: c for c in Retriever().cases}
    candidates = []
    for number, row in enumerate(selected, 1):
        original = inputs[str(row["customer_tweet_id"])]
        if original["customer_text"] != row["customer_text"]:
            raise ValueError("Customer text changed after prediction.")
        prediction = row["prediction"]
        historical = []
        for ref in prediction["retrieved_cases"]:
            case = cases[ref["case_id"]]
            historical.append({"case_id": case["case_id"],
                               "customer_text": case["customer_text"],
                               "historical_replies": case["historical_replies"][:1]})
        candidates.append({
            "sample_id": f"agent-review-{number:02d}",
            "customer_message": row["customer_text"],
            "prior_context": original["prior_context"][-4:],
            "historical_cases": historical,
            "reply": prediction["reply"],
            "action": prediction["action"],
            "evidence_ids": prediction.get("evidence_ids", []),
        })
    signature = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT / "results/agent_review" / signature
    folder.mkdir(parents=True, exist_ok=True)
    review = folder / "human_review.json"
    if not review.exists():
        value = {
            "dataset_type": "frozen_agent_test_replies",
            "run_id": RUN_ID,
            "reviewer": "",
            "sampling": "Seed 42; uniform sample of 20 successful replies; failed generation excluded and reported separately.",
            "instructions": "Use docs/judge_rubric.md. Fill only reviewer and human_scores yourself before viewing judge scores. Historical cases concern other customers. Context and first historical reply match the generator input limits.",
            "examples": [dict(c, human_scores={
                "relevance": None, "grounding": None, "usefulness": None,
                "safety": None, "critical_error": None, "explanation": "",
            }) for c in candidates],
        }
        with review.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        with (folder / "candidates.json").open("x", encoding="utf-8") as stream:
            json.dump(candidates, stream, ensure_ascii=False, indent=2)
    print(f"Review file: {review}")
    print("20 replies prepared. No model calls; existing ratings preserved.")


if __name__ == "__main__":
    main()
