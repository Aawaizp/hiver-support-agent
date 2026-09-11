import unittest
from hiver.evaluate_baselines_test import validate_splits
from hiver.evaluate_baselines import calculate_metrics


def examples(split, count):
    return [{"split": split, "customer_tweet_id": f"{split}-{n}",
             "conversation_id": f"{split}-{n}", "intent": "other_unclear",
             "expected_action": "escalate"} for n in range(count)]


class TestEvaluationTests(unittest.TestCase):
    def test_rejects_conversation_leakage(self):
        dev, test = examples("dev", 50), examples("test", 150)
        validate_splits(dev, test)
        test[0]["conversation_id"] = dev[0]["conversation_id"]
        with self.assertRaises(ValueError):
            validate_splits(dev, test)

    def test_failed_generation_counts_as_incorrect(self):
        rows = [{"expected_intent": "other_unclear", "expected_action": "escalate",
                 "prediction": {"intent": "__generation_failure__",
                                "action": "__generation_failure__"}}]
        metrics = calculate_metrics(rows)
        self.assertEqual(metrics["intent_accuracy"], 0)
        self.assertEqual(metrics["action_accuracy"], 0)
        self.assertEqual(metrics["escalation_recall"], 0)
        self.assertEqual(metrics["auto_handle_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
