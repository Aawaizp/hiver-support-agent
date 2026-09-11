from hiver.paths import ROOT
from collections import Counter
from pathlib import Path
import json

from sklearn.metrics import accuracy_score, f1_score

from hiver.baselines import Baselines, INTENTS, load_development

OUTPUT = ROOT / "results"


def divide(numerator, denominator):
    # None means the metric is undefined, not zero.
    return numerator / denominator if denominator else None


def calculate_metrics(rows):
    actual_intents = [row["expected_intent"] for row in rows]
    predicted_intents = [row["prediction"]["intent"] for row in rows]

    expected_escalations = sum(
        row["expected_action"] == "escalate" for row in rows
    )
    caught_escalations = sum(
        row["expected_action"] == "escalate"
        and row["prediction"]["action"] == "escalate"
        for row in rows
    )
    auto_handled = sum(
        row["prediction"]["action"] == "auto_handle" for row in rows
    )
    unsafe_auto_handled = sum(
        row["expected_action"] == "escalate"
        and row["prediction"]["action"] == "auto_handle"
        for row in rows
    )
    correct_actions = sum(
        row["expected_action"] == row["prediction"]["action"]
        for row in rows
    )

    return {
        "examples": len(rows),
        "intent_accuracy": accuracy_score(
            actual_intents, predicted_intents
        ),
        "intent_macro_f1": f1_score(
            actual_intents,
            predicted_intents,
            labels=sorted(INTENTS),
            average="macro",
            zero_division=0,
        ),
        "action_accuracy": divide(correct_actions, len(rows)),
        "escalation_recall": divide(
            caught_escalations, expected_escalations
        ),
        "auto_handle_coverage": divide(auto_handled, len(rows)),
        "unsafe_auto_handle_rate": divide(
            unsafe_auto_handled, auto_handled
        ),
        "counts": {
            "expected_escalations": expected_escalations,
            "caught_escalations": caught_escalations,
            "auto_handled": auto_handled,
            "unsafe_auto_handled": unsafe_auto_handled,
        },
    }


def main():
    examples = load_development()
    predictions = {"trivial": [], "simple": []}

    for number, example in enumerate(examples, start=1):
        training = [
            item for item in examples
            if item["conversation_id"] != example["conversation_id"]
        ]

        models = Baselines(training)

        for name in predictions:
            predict = getattr(models, name)
            result = predict(
                example["customer_text"],
                example["prior_context"],
            )

            predictions[name].append({
                "customer_tweet_id": example["customer_tweet_id"],
                "conversation_id": example["conversation_id"],
                "customer_text": example["customer_text"],
                "expected_intent": example["intent"],
                "expected_action": example["expected_action"],
                "label_uncertain": example["uncertain"],
                "prediction": result,
            })

        if number % 10 == 0:
            print(f"Evaluated {number}/{len(examples)} examples.")

    report = {
        "evaluation": "Leave-one-conversation-out development evaluation",
        "label_basis": "AI-drafted labels; independent human review not verified",
        "limitations": [
            "These are development results, not final test results.",
            "Scores measure agreement with the supplied labels.",
            "Unsafe auto-handle rate measures routing errors, not reply safety.",
            "Reply quality is not evaluated by this script.",
        ],
        "intent_counts": dict(Counter(
            item["intent"] for item in examples
        )),
        "uncertain_labels_included": sum(
            item["uncertain"] for item in examples
        ),
        "metrics": {
            name: calculate_metrics(rows)
            for name, rows in predictions.items()
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, rows in predictions.items():
        path = OUTPUT / f"{name}_dev_predictions.json"
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report_path = OUTPUT / "baselines_dev_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["metrics"], indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()