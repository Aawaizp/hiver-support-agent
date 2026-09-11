"""Recalculate saved test metrics without model calls; this is not fresh inference."""
import json
from pathlib import Path

from hiver.paths import ROOT
from hiver.evaluate_baselines import calculate_metrics

AGENT_RUN = "e14347f8f9f185d629c453a8c973ff168d6183da3eb2af7605772c106b96e4d8"
BASELINE_RUN = "7b2d1d92e828737cea676cb483158c2898620841601e337f303b73ab2d46531c"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_rows(rows, examples):
    expected = {str(x["customer_tweet_id"]): x for x in examples}
    ids = [str(r["customer_tweet_id"]) for r in rows]
    if len(expected) != 150 or len(ids) != 150 or len(set(ids)) != 150 or set(ids) != set(expected):
        raise ValueError("Results must cover exactly the same 150 distinct test inputs.")
    for row in rows:
        original = expected[str(row["customer_tweet_id"])]
        for result_key, input_key in (("conversation_id", "conversation_id"),
                                      ("customer_text", "customer_text"),
                                      ("expected_intent", "intent"),
                                      ("expected_action", "expected_action")):
            if row[result_key] != original[input_key]:
                raise ValueError(f"Input/label mismatch for {row['customer_tweet_id']}: {result_key}")


def main():
    examples = []
    for path in sorted((ROOT / "data/prepared").glob("test_batch_*.json")):
        examples.extend(read(path))
    baseline_folder = ROOT / "results/baselines_test" / BASELINE_RUN
    agent_folder = ROOT / "results/agent_test" / AGENT_RUN
    baseline_report = read(baseline_folder / "metrics.json")
    agent_report = read(agent_folder / "recovery_metrics.json")
    summary = {}
    for name, path in (("trivial", baseline_folder / "trivial_predictions.json"),
                       ("simple", baseline_folder / "simple_predictions.json"),
                       ("agent", agent_folder / "recovery_predictions.json")):
        rows = read(path)
        validate_rows(rows, examples)
        metrics = calculate_metrics(rows)
        recorded = agent_report["metrics"] if name == "agent" else baseline_report["metrics"][name]
        if metrics != recorded:
            raise ValueError(f"Saved and recalculated metrics disagree for {name}.")
        summary[name] = {"metrics": metrics, "generation_failures": sum(
            row["prediction"].get("status") != "ok" for row in rows)}
    if summary["agent"]["generation_failures"] != agent_report["generation_failures"]:
        raise ValueError("Recorded generation failure count disagrees.")
    from hiver.judge_agent import agreement
    review_folder = ROOT / "results/agent_review/a3b4be5445390607"
    judge_folder = review_folder / "judge/b0bd4a688c636ff2"
    human = read(judge_folder / "human_review_snapshot.json")
    judged = read(judge_folder / "judge_results.json")
    expected_agreement = read(judge_folder / "agreement.json")
    recomputed = agreement(human, judged["judgments"])
    recomputed["judge_model"] = judged["model"]
    if recomputed != expected_agreement:
        raise ValueError("Saved judge agreement differs from recalculation.")
    print(json.dumps({"verified": True, "mode": "Saved prediction scoring; no fresh model inference",
                      "examples": 150, "systems": summary,
                      "judge_exact_agreement": recomputed["exact_dimension_agreement"]}, indent=2))


if __name__ == "__main__":
    main()
